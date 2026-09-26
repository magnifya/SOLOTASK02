"""Tests for the 1:1 inbox redelivery-job event chain endpoint.

GET /v1/devices/{device_id}/inbox-job-events pages a device's append-only
job lifecycle chain. A first queue/dispatch/recover/cancel of a job each
records one event, as does the first lease completion that moves a running
job to succeeded/failed; replays, failed requests and ordinary leases
record nothing; batch operations record their events in input order. Each
device's seq runs from 1 continuously. The GET takes no body (non-empty ->
400/request_body) and only single-valued after/limit query parameters
(defaults 0/100, ASCII decimal, ranges 0..2^63-1 / 1..100; invalid or
repeated -> 400 with that parameter, anything else -> 400/query). An
unknown device is 404/device_id (revoked stays readable). The 200 body keys
are device_id, events, next_after, has_more; each event is seq, job_id,
type, state; an empty page leaves next_after at after.
"""
import json
import os
import tempfile
import threading
import unittest
from http.client import HTTPConnection

from e2ee_backend.models import Device, SignedPreKey
from e2ee_backend.persistence import (
    StateFileError, attach_persistence)
from e2ee_backend.service import DeviceService, ServiceError
from e2ee_backend.http_app import create_server


_PAST = "2000-01-01T00:00:00.000000+00:00"


class EventMixin:
    def _build(self) -> None:
        self.service = DeviceService()
        self.service.store.add_device(Device("u", "alice", "ik"))
        self.service.store.add_device(Device(
            "u", "bob", "ik",
            prekeys=[SignedPreKey("pk1", "pubk1")]))
        self.service.store.add_device(Device("u", "carol", "ik"))
        self.sid1 = self.service.store.create_session(
            "alice", "bob", "pk1", "ek1").session_id

    def _post_message(self, message_id="m1", sequence=1) -> None:
        self.service.post_message({
            "session_id": self.sid1, "sender_device_id": "alice",
            "message_id": message_id, "sequence": sequence,
            "nonce": f"n{message_id}", "ciphertext": "ct"})

    def _job(self, job_id, op="queue", device="bob", **extra):
        payload = {"device_id": device, "job_id": job_id, "op": op}
        payload.update(extra)
        return self.service.inbox_job(payload)

    def _queue(self, job_id, device="bob"):
        body, status = self._job(job_id, device=device)
        self.assertEqual(status, 201)
        return body

    def _dispatch(self, job_id):
        body, status = self._job(job_id, "dispatch")
        self.assertEqual(status, 201)
        return body

    def _expire(self, lease_id) -> None:
        with self.service.store._lock:
            for state in self.service.store._delivery.values():
                for lease in state.leases:
                    if lease.lease_id == lease_id:
                        lease.leased_until = _PAST

    def _events(self, after=0, limit=100, device="bob"):
        return self.service.inbox_job_events_page(device, after, limit)

    def _types(self, **kwargs):
        return [(e["seq"], e["job_id"], e["type"], e["state"])
                for e in self._events(**kwargs)["events"]]


class EventServiceTest(EventMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def test_empty_chain_shape(self) -> None:
        body = self._events()
        self.assertEqual(list(body),
                         ["device_id", "events", "next_after", "has_more"])
        self.assertEqual(body["device_id"], "bob")
        self.assertEqual(body["events"], [])
        self.assertEqual(body["next_after"], 0)
        self.assertFalse(body["has_more"])

    def test_queue_records_one_event(self) -> None:
        self._queue("J1")
        self.assertEqual(self._types(),
                         [(1, "J1", "queue", "pending")])
        # A replay records nothing and leaves the chain untouched.
        self._job("J1", "queue")
        self.assertEqual(self._types(),
                         [(1, "J1", "queue", "pending")])

    def test_dispatch_empty_running_vs_succeeded(self) -> None:
        # No messages: an empty dispatch ends the job succeeded.
        self._queue("J0")
        self._dispatch("J0")
        self.assertEqual(self._types(),
                         [(1, "J0", "queue", "pending"),
                          (2, "J0", "dispatch", "succeeded")])
        # With messages the dispatch keeps the job running.
        self._post_message()
        self._queue("J1")
        self._dispatch("J1")
        self.assertEqual(self._types()[-1],
                         (4, "J1", "dispatch", "running"))

    def test_complete_records_terminal_event(self) -> None:
        self._post_message()
        self._queue("J1")
        self._dispatch("J1")
        _, status = self.service.inbox_lease_complete(
            "bob", "J1", {"completion_id": "c1", "outcome": "delivered"})
        self.assertEqual(status, 201)
        self.assertEqual(self._types(),
                         [(1, "J1", "queue", "pending"),
                          (2, "J1", "dispatch", "running"),
                          (3, "J1", "complete", "succeeded")])
        # Replaying the completion appends no event.
        _, status = self.service.inbox_lease_complete(
            "bob", "J1", {"completion_id": "c1", "outcome": "delivered"})
        self.assertEqual(status, 200)
        self.assertEqual(len(self._events()["events"]), 3)

    def test_failed_completion_records_failed_state(self) -> None:
        self._post_message()
        self._queue("J1")
        self._dispatch("J1")
        self.service.inbox_lease_complete(
            "bob", "J1", {"completion_id": "c1", "outcome": "failed"})
        self.assertEqual(self._types()[-1],
                         (3, "J1", "complete", "failed"))

    def test_ordinary_lease_records_no_event(self) -> None:
        self._post_message()
        # A plain inbox claim (not a job) records no job event, even when
        # that lease is later completed.
        self.service.inbox_claim("bob", {"lease_id": "L1", "limit": 10})
        self.assertEqual(self._events()["events"], [])
        self.service.inbox_lease_complete(
            "bob", "L1", {"completion_id": "c", "outcome": "delivered"})
        self.assertEqual(self._events()["events"], [])

    def test_recover_records_event(self) -> None:
        self._post_message()
        self._queue("J1")
        self._dispatch("J1")
        self._expire("J1")
        body, status = self._job("J1", "recover", recovery_id="r1")
        self.assertEqual(status, 201)
        self.assertEqual(body["state"], "running")
        self.assertEqual(self._types()[-1],
                         (3, "J1", "recover", "running"))
        # Replaying the recovery appends nothing.
        self._job("J1", "recover", recovery_id="r1")
        self.assertEqual(len(self._events()["events"]), 3)

    def test_empty_recover_ends_succeeded(self) -> None:
        self._post_message()
        self._queue("J2")
        self._dispatch("J2")
        self._expire("J2")
        # Ack the single leased message so the recovery selection is empty.
        self.service.ack_message(self.sid1, {
            "device_id": "bob", "message_id": "m1", "sequence": 1})
        body, status = self._job("J2", "recover", recovery_id="r2")
        self.assertEqual(status, 201)
        self.assertEqual(body["state"], "succeeded")
        self.assertEqual(self._types()[-1],
                         (3, "J2", "recover", "succeeded"))

    def test_cancel_records_event(self) -> None:
        self._queue("J1")
        body, status = self._job("J1", "cancel", cancellation_id="x")
        self.assertEqual(status, 201)
        self.assertEqual(body["state"], "cancelled")
        self.assertEqual(self._types(),
                         [(1, "J1", "queue", "pending"),
                          (2, "J1", "cancel", "cancelled")])
        # A replay with the same cancellation_id appends nothing.
        self._job("J1", "cancel", cancellation_id="x")
        self.assertEqual(len(self._events()["events"]), 2)

    def test_status_is_read_only(self) -> None:
        self._queue("J1")
        self._job("J1", "status")
        self.assertEqual(self._types(),
                         [(1, "J1", "queue", "pending")])

    def test_chains_are_per_device(self) -> None:
        # carol has a 1:1 session only if we give her a message addressed to
        # her; queue alone needs no session.
        self._queue("Jb", device="bob")
        self._queue("Jc", device="carol")
        self.assertEqual(
            [(e["seq"], e["job_id"]) for e in self._events(device="bob")[
                "events"]],
            [(1, "Jb")])
        self.assertEqual(
            [(e["seq"], e["job_id"]) for e in self._events(device="carol")[
                "events"]],
            [(1, "Jc")])

    def test_paging_after_and_limit(self) -> None:
        for index in range(5):
            self._queue(f"J{index}")
        page = self._events(after=0, limit=2)
        self.assertEqual([e["seq"] for e in page["events"]], [1, 2])
        self.assertEqual(page["next_after"], 2)
        self.assertTrue(page["has_more"])
        page = self._events(after=2, limit=2)
        self.assertEqual([e["seq"] for e in page["events"]], [3, 4])
        self.assertEqual(page["next_after"], 4)
        self.assertTrue(page["has_more"])
        page = self._events(after=4, limit=2)
        self.assertEqual([e["seq"] for e in page["events"]], [5])
        self.assertEqual(page["next_after"], 5)
        self.assertFalse(page["has_more"])
        # An empty page leaves next_after at after.
        page = self._events(after=5, limit=2)
        self.assertEqual(page["events"], [])
        self.assertEqual(page["next_after"], 5)
        self.assertFalse(page["has_more"])

    def test_event_key_order(self) -> None:
        self._queue("J1")
        event = self._events()["events"][0]
        self.assertEqual(list(event), ["seq", "job_id", "type", "state"])

    def test_unknown_device_is_404(self) -> None:
        with self.assertRaises(ServiceError) as caught:
            self.service.inbox_job_events_page("ghost", 0, 100)
        self.assertEqual((caught.exception.status_code,
                          caught.exception.field), (404, "device_id"))

    def test_revoked_device_stays_readable(self) -> None:
        self._queue("J1")
        self.service.revoke_device("bob")
        body = self._events()
        self.assertEqual(body["events"][0]["job_id"], "J1")

    def test_service_rejects_bad_paging_values(self) -> None:
        for after, limit, field in (
                (-1, 100, "after"),
                (2**63, 100, "after"),
                (0, 0, "limit"),
                (0, 101, "limit"),
                (True, 100, "after"),
                (0, True, "limit")):
            with self.assertRaises(ServiceError) as caught:
                self.service.inbox_job_events_page("bob", after, limit)
            self.assertEqual(caught.exception.field, field, (after, limit))

    def test_batch_operations_record_in_input_order(self) -> None:
        self._post_message("m1", 1)
        self._post_message("m2", 2)
        self._queue("A")
        self._queue("B")
        _, status = self.service.inbox_job_dispatch_batch(
            {"device_id": "bob",
             "items": [{"job_id": "A"}, {"job_id": "B"}]})
        self.assertEqual(status, 201)
        # Both dispatch events land in input order (A takes the messages,
        # B finds an empty inbox and ends succeeded).
        self.assertEqual(self._types()[-2:],
                         [(3, "A", "dispatch", "running"),
                          (4, "B", "dispatch", "succeeded")])
        self._queue("C")
        _, status = self.service.inbox_job_cancel_batch(
            {"device_id": "bob",
             "items": [{"job_id": "C", "cancellation_id": "cC"}]})
        self.assertEqual(status, 201)
        self.assertEqual(self._types()[-1],
                         (6, "C", "cancel", "cancelled"))


class EventPersistenceTest(EventMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()
        self.directory = tempfile.mkdtemp()
        self.path = os.path.join(self.directory, "state.json")
        self.state_store = attach_persistence(self.service, self.path)

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self.directory, ignore_errors=True)

    def _restart(self):
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        self.service = restarted
        return restarted

    def test_events_persist_and_restore(self) -> None:
        self._post_message()
        self._queue("J1")
        self._dispatch("J1")
        self.service.inbox_lease_complete(
            "bob", "J1", {"completion_id": "c1", "outcome": "delivered"})
        self._queue("J2")
        expected = self._types()
        restarted = self._restart()
        body = restarted.inbox_job_events_page("bob", 0, 100)
        restored = [(e["seq"], e["job_id"], e["type"], e["state"])
                    for e in body["events"]]
        self.assertEqual(restored, expected)

    def test_section_present_with_device_id_prefix(self) -> None:
        self._queue("J1")
        with open(self.path, encoding="utf-8") as handle:
            document = json.load(handle)
        self.assertIn("redelivery_job_events", document)
        self.assertEqual(document["redelivery_job_events"], [{
            "device_id": "bob", "seq": 1, "job_id": "J1",
            "type": "queue", "state": "pending"}])
        # It follows redelivery_jobs in the document key order.
        names = list(document)
        self.assertLess(names.index("redelivery_jobs"),
                        names.index("redelivery_job_events"))

    def test_legacy_document_without_section_loads(self) -> None:
        self._queue("J1")
        with open(self.path, encoding="utf-8") as handle:
            document = json.load(handle)
        document.pop("redelivery_jobs")
        document.pop("redelivery_job_events", None)
        document.pop("integrity_log_version", None)
        legacy_path = os.path.join(self.directory, "legacy.json")
        with open(legacy_path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        restarted = DeviceService()
        attach_persistence(restarted, legacy_path)  # must not raise
        body = restarted.inbox_job_events_page("bob", 0, 100)
        self.assertEqual(body["events"], [])

    def test_restore_rejects_malformed_events(self) -> None:
        import copy
        self._queue("J1")
        with open(self.path, encoding="utf-8") as handle:
            base = json.load(handle)
        for index, mutate in enumerate((
                lambda d: d.__setitem__("redelivery_job_events", {}),
                lambda d: d["redelivery_job_events"][0].pop("state"),
                lambda d: d["redelivery_job_events"][0].__setitem__(
                    "extra", 1),
                lambda d: d["redelivery_job_events"][0].__setitem__(
                    "seq", 0),
                lambda d: d["redelivery_job_events"][0].__setitem__(
                    "type", "nope"),
                lambda d: d["redelivery_job_events"][0].__setitem__(
                    "state", "running"),
                lambda d: d["redelivery_job_events"][0].__setitem__(
                    "device_id", "ghost"),
                lambda d: d["redelivery_job_events"][0].__setitem__(
                    "job_id", "ghost"),
                lambda d: d["redelivery_job_events"][0].__setitem__(
                    "seq", 2),
        )):
            with self.subTest(index=index):
                document = copy.deepcopy(base)
                mutate(document)
                document.pop("integrity_log_version", None)
                bad_path = os.path.join(self.directory, f"bad{index}.json")
                with open(bad_path, "w", encoding="utf-8") as handle:
                    json.dump(document, handle)
                with open(bad_path, "rb") as handle:
                    original = handle.read()
                restarted = DeviceService()
                with self.assertRaises(StateFileError):
                    attach_persistence(restarted, bad_path)
                with open(bad_path, "rb") as handle:
                    self.assertEqual(handle.read(), original)


class EventHTTPTest(EventMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()
        self.server, self.service = create_server(
            "127.0.0.1", 0, self.service)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _request(self, path, raw=None, method="GET"):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        headers = {"Content-Type": "application/json"} if raw else {}
        conn.request(method, path, body=raw, headers=headers)
        response = conn.getresponse()
        data = response.read().decode("utf-8")
        conn.close()
        return response.status, (json.loads(data) if data else None), data

    PATH = "/v1/devices/bob/inbox-job-events"

    def test_200_key_order_and_byte_stable(self) -> None:
        self._queue("J1")
        status, body, raw = self._request(self.PATH)
        self.assertEqual(status, 200)
        self.assertEqual(list(body),
                         ["device_id", "events", "next_after", "has_more"])
        self.assertEqual(list(body["events"][0]),
                         ["seq", "job_id", "type", "state"])
        names = ['"device_id"', '"events"', '"next_after"', '"has_more"']
        positions = [raw.index(name) for name in names]
        self.assertEqual(positions, sorted(positions))
        _, _, second = self._request(self.PATH)
        self.assertEqual(raw, second)

    def test_empty_page_next_after_echoes_after(self) -> None:
        status, body, _ = self._request(self.PATH + "?after=7&limit=10")
        self.assertEqual(status, 200)
        self.assertEqual(body["events"], [])
        self.assertEqual(body["next_after"], 7)
        self.assertFalse(body["has_more"])

    def test_non_empty_body_is_400_request_body(self) -> None:
        status, body, _ = self._request(self.PATH, raw=b'{}')
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "request_body")

    def test_unknown_query_param_is_400_query(self) -> None:
        for suffix in ("?state=all", "?foo", "?after=1&foo"):
            status, body, _ = self._request(self.PATH + suffix)
            self.assertEqual(status, 400, suffix)
            self.assertEqual(body["field"], "query", suffix)

    def test_after_validation(self) -> None:
        for suffix, ok in (
                ("", True),
                ("?after=0", True),
                ("?after=007", True),
                ("?after=9223372036854775807", True),
                ("?after=9223372036854775808", False),
                ("?after=-1", False),
                ("?after=1.0", False),
                ("?after=0x1", False),
                ("?after=", False),
                ("?after=1%20", False),
                ("?after=1&after=2", False)):
            status, body, _ = self._request(self.PATH + suffix)
            if ok:
                self.assertEqual(status, 200, suffix)
            else:
                self.assertEqual((status, body["field"]),
                                 (400, "after"), suffix)

    def test_limit_validation(self) -> None:
        for suffix, ok in (
                ("", True),
                ("?limit=1", True),
                ("?limit=100", True),
                ("?limit=010", True),
                ("?limit=0", False),
                ("?limit=101", False),
                ("?limit=-1", False),
                ("?limit=1.5", False),
                ("?limit=", False),
                ("?limit=1&limit=2", False)):
            status, body, _ = self._request(self.PATH + suffix)
            if ok:
                self.assertEqual(status, 200, suffix)
            else:
                self.assertEqual((status, body["field"]),
                                 (400, "limit"), suffix)

    def test_unknown_device_is_404(self) -> None:
        status, body, _ = self._request(
            "/v1/devices/ghost/inbox-job-events")
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "device_id")

    def test_revoked_device_still_readable(self) -> None:
        self._queue("J1")
        self.service.revoke_device("bob")
        status, body, _ = self._request(self.PATH)
        self.assertEqual(status, 200)
        self.assertEqual(body["events"][0]["job_id"], "J1")

    def test_trailing_empty_question_mark_accepted(self) -> None:
        status, _, _ = self._request(self.PATH + "?")
        self.assertEqual(status, 200)


if __name__ == "__main__":
    unittest.main()
