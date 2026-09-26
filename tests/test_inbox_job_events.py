"""Tests for the paginated 1:1 inbox redelivery-job event chain endpoint.

GET /v1/devices/{device_id}/inbox-job-events returns one page of the
device's redelivery-job lifecycle events. It takes no request body
(non-empty -> 400/request_body) and only single-valued ``after``
(default 0; unsigned decimal in 0..2**63-1) and ``limit`` (default 100;
1..100) query parameters; anything else is 400/query and a malformed
value is 400 with the parameter name. An unknown device is
404/device_id; a revoked device's chain stays readable. The first
successful queue/dispatch/recover/cancel of a job and the first
completion of the lease a running job currently holds each append one
event (replays, failures and ordinary leases append nothing; batches
append in input order); the per-device ``seq`` runs consecutively from
1. The page is the events with ``seq > after``, at most ``limit`` of
them. The query is purely read-only (no write, no commit_seq change).
"""
import json
import os
import shutil
import tempfile
import threading
import unittest
from http.client import HTTPConnection

from e2ee_backend.models import Device, SignedPreKey
from e2ee_backend.service import DeviceService, ServiceError
from e2ee_backend.http_app import create_server
from e2ee_backend.persistence import (
    PersistenceUnavailable,
    StateFileError,
    attach_persistence,
)


class EventsMixin:
    def _build(self) -> None:
        self.service = DeviceService()
        self.service.store.add_device(Device("u", "alice", "ik"))
        self.service.store.add_device(Device(
            "u", "bob", "ik",
            prekeys=[SignedPreKey("pk1", "pubk1")]))
        self.sid = self.service.store.create_session(
            "alice", "bob", "pk1", "ek1").session_id

    def _post_message(self, message_id="m1", sequence=1) -> None:
        self.service.post_message({
            "session_id": self.sid, "sender_device_id": "alice",
            "message_id": message_id, "sequence": sequence,
            "nonce": f"n{message_id}", "ciphertext": "ct"})

    def _op(self, job_id, op, device="bob", **extra):
        payload = {"device_id": device, "job_id": job_id, "op": op}
        payload.update(extra)
        return self.service.inbox_job(payload)

    def _events(self, device="bob", after=0, limit=100):
        return self.service.inbox_job_events_page(device, after, limit)

    def _chain(self, device="bob"):
        return [(event["seq"], event["job_id"], event["type"],
                 event["state"])
                for event in self._events(device)["events"]]


class EventsServiceTest(EventsMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def test_empty_chain_shape(self) -> None:
        body = self._events()
        self.assertEqual(list(body),
                         ["device_id", "events", "next_after", "has_more"])
        self.assertEqual(body, {"device_id": "bob", "events": [],
                                "next_after": 0, "has_more": False})

    def test_queue_dispatch_complete_chain_with_item_key_order(self) -> None:
        self._post_message()
        self._op("J1", "queue")
        self._op("J1", "dispatch")
        self.service.inbox_lease_complete(
            "bob", "J1", {"completion_id": "C1", "outcome": "delivered"})
        body = self._events()
        self.assertEqual(
            self._chain(),
            [(1, "J1", "queue", "pending"),
             (2, "J1", "dispatch", "running"),
             (3, "J1", "complete", "succeeded")])
        for item in body["events"]:
            self.assertEqual(list(item), ["seq", "job_id", "type", "state"])
        self.assertEqual(body["next_after"], 3)
        self.assertFalse(body["has_more"])

    def test_failed_completion_records_failed_state(self) -> None:
        self._post_message()
        self._op("J1", "queue")
        self._op("J1", "dispatch")
        self.service.inbox_lease_complete(
            "bob", "J1", {"completion_id": "C1", "outcome": "failed"})
        self.assertEqual(self._chain()[-1], (3, "J1", "complete", "failed"))

    def test_empty_dispatch_and_empty_recover_states(self) -> None:
        # An empty selection dispatches straight to succeeded.
        self._op("JE", "queue")
        self._op("JE", "dispatch")
        # A running job whose recovery finds nothing left ends succeeded.
        self._post_message()
        self._op("JR", "queue")
        self._op("JR", "dispatch")
        self.service.inbox_release("bob", "JR")
        self.service.inbox_claim("bob", {"lease_id": "ord", "limit": 10})
        self.service.inbox_lease_complete(
            "bob", "ord", {"completion_id": "Cx", "outcome": "delivered"})
        self.service.ack_message(
            self.sid, {"device_id": "bob", "message_id": "m1",
                       "sequence": 1})
        self._op("JR", "recover", recovery_id="R1")
        self.assertEqual(
            self._chain(),
            [(1, "JE", "queue", "pending"),
             (2, "JE", "dispatch", "succeeded"),
             (3, "JR", "queue", "pending"),
             (4, "JR", "dispatch", "running"),
             (5, "JR", "recover", "succeeded")])

    def test_recover_and_cancel_events(self) -> None:
        self._post_message()
        self._op("J1", "queue")
        self._op("J1", "dispatch")
        self.service.inbox_release("bob", "J1")
        self._op("J1", "recover", recovery_id="R1")
        self._op("J2", "queue")
        self._op("J2", "cancel", cancellation_id="X1")
        self.assertEqual(
            self._chain(),
            [(1, "J1", "queue", "pending"),
             (2, "J1", "dispatch", "running"),
             (3, "J1", "recover", "running"),
             (4, "J2", "queue", "pending"),
             (5, "J2", "cancel", "cancelled")])

    def test_replays_and_failures_record_nothing(self) -> None:
        self._post_message()
        self._op("J1", "queue")
        self._op("J1", "queue")                    # queue replay
        self._op("J1", "dispatch")
        self._op("J1", "dispatch")                 # dispatch replay
        self._op("J1", "status")                   # read-only
        # A recover on a still-valid lease is 409; a cancel of a
        # never-queued job is 404. Neither records an event.
        for op, extra in (("recover", {"recovery_id": "R1"}),):
            with self.assertRaises(ServiceError):
                self._op("J1", op, **extra)
        with self.assertRaises(ServiceError):
            self._op("J2", "cancel", cancellation_id="X1")
        self.service.inbox_lease_complete(
            "bob", "J1", {"completion_id": "C1", "outcome": "delivered"})
        # A replayed completion freezes the first response: no new event.
        self.service.inbox_lease_complete(
            "bob", "J1", {"completion_id": "C1", "outcome": "delivered"})
        self.assertEqual(
            self._chain(),
            [(1, "J1", "queue", "pending"),
             (2, "J1", "dispatch", "running"),
             (3, "J1", "complete", "succeeded")])

    def test_ordinary_lease_lifecycle_records_nothing(self) -> None:
        self._post_message()
        self.service.inbox_claim("bob", {"lease_id": "ord", "limit": 10})
        self.service.inbox_lease_renew("bob", "ord", {"renewal_id": "RN"})
        self.service.inbox_lease_complete(
            "bob", "ord", {"completion_id": "C1", "outcome": "delivered"})
        self.assertEqual(self._chain(), [])

    def test_lease_renew_release_record_nothing(self) -> None:
        self._post_message()
        self._op("J1", "queue")
        self._op("J1", "dispatch")
        self.service.inbox_lease_renew("bob", "J1", {"renewal_id": "RN"})
        self.service.inbox_release("bob", "J1")
        self.assertEqual(
            self._chain(),
            [(1, "J1", "queue", "pending"),
             (2, "J1", "dispatch", "running")])

    def test_batches_record_in_input_order(self) -> None:
        for job_id in ("J1", "J2", "J3", "J4"):
            self._op(job_id, "queue")
        self.service.inbox_job_dispatch_batch(
            {"device_id": "bob", "items": [{"job_id": "J2"},
                                           {"job_id": "J1"}]})
        self.service.inbox_job_cancel_batch(
            {"device_id": "bob", "items": [
                {"job_id": "J4", "cancellation_id": "X4"},
                {"job_id": "J3", "cancellation_id": "X3"}]})
        self.assertEqual(
            self._chain(),
            [(1, "J1", "queue", "pending"),
             (2, "J2", "queue", "pending"),
             (3, "J3", "queue", "pending"),
             (4, "J4", "queue", "pending"),
             (5, "J2", "dispatch", "succeeded"),
             (6, "J1", "dispatch", "succeeded"),
             (7, "J4", "cancel", "cancelled"),
             (8, "J3", "cancel", "cancelled")])

    def test_complete_batch_records_in_input_order(self) -> None:
        self._post_message()
        self._op("J1", "queue")
        self._op("J1", "dispatch")
        self._post_message("m2", 2)
        self._op("J2", "queue")
        self._op("J2", "dispatch")
        self.service.inbox_job_complete_batch(
            {"device_id": "bob", "items": [
                {"lease_id": "J2", "completion_id": "C2",
                 "outcome": "failed"},
                {"lease_id": "J1", "completion_id": "C1",
                 "outcome": "delivered"}]})
        self.assertEqual(
            self._chain()[-2:],
            [(5, "J2", "complete", "failed"),
             (6, "J1", "complete", "succeeded")])

    def test_chains_are_per_device(self) -> None:
        self.service.store.add_device(Device("u", "carol", "ik"))
        self._op("J1", "queue")
        self._op("J2", "queue", device="carol")
        self._op("J1", "cancel", cancellation_id="X1")
        self.assertEqual(self._chain("bob"),
                         [(1, "J1", "queue", "pending"),
                          (2, "J1", "cancel", "cancelled")])
        self.assertEqual(self._chain("carol"),
                         [(1, "J2", "queue", "pending")])

    def test_pagination_after_limit_next_after_and_has_more(self) -> None:
        for job_id in ("J1", "J2", "J3", "J4"):
            self._op(job_id, "queue")
        first = self._events(after=0, limit=2)
        self.assertEqual([e["seq"] for e in first["events"]], [1, 2])
        self.assertEqual(first["next_after"], 2)
        self.assertTrue(first["has_more"])
        second = self._events(after=2, limit=2)
        self.assertEqual([e["seq"] for e in second["events"]], [3, 4])
        self.assertEqual(second["next_after"], 4)
        self.assertFalse(second["has_more"])
        third = self._events(after=4, limit=2)
        self.assertEqual(third["events"], [])
        # Empty page: next_after echoes after.
        self.assertEqual(third["next_after"], 4)
        self.assertFalse(third["has_more"])

    def test_unknown_device_is_404_device_id(self) -> None:
        with self.assertRaises(ServiceError) as caught:
            self._events(device="ghost")
        self.assertEqual(caught.exception.status_code, 404)
        self.assertEqual(caught.exception.field, "device_id")

    def test_revoked_device_chain_stays_readable(self) -> None:
        self._op("J1", "queue")
        self.service.revoke_device("bob")
        self.assertEqual(self._chain(), [(1, "J1", "queue", "pending")])

    def test_invalid_after_limit_at_service_layer(self) -> None:
        for after, limit, field in (
                (-1, 100, "after"),
                (2**63, 100, "after"),
                (1.0, 100, "after"),
                (True, 100, "after"),
                (0, 0, "limit"),
                (0, 101, "limit"),
                (0, True, "limit")):
            with self.subTest(field=field):
                with self.assertRaises(ServiceError) as caught:
                    self.service.inbox_job_events_page("bob", after, limit)
                self.assertEqual(caught.exception.field, field)
                self.assertEqual(caught.exception.status_code, 400)
        # The extremes of the after range are accepted.
        self.service.inbox_job_events_page("bob", 2**63 - 1, 100)

    def test_query_is_read_only(self) -> None:
        self._op("J1", "queue")
        store = self.service.store
        with store._lock:
            before = {device_id: [(e.seq, e.job_id, e.type, e.state)
                                  for e in chain]
                      for device_id, chain
                      in store._redelivery_job_events.items()}
        first = self._events()
        second = self._events()
        self.assertEqual(first, second)
        with store._lock:
            after = {device_id: [(e.seq, e.job_id, e.type, e.state)
                                 for e in chain]
                     for device_id, chain
                     in store._redelivery_job_events.items()}
        self.assertEqual(before, after)


class EventsHTTPTest(EventsMixin, unittest.TestCase):
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

    def _queue_http(self, job_id):
        status, _, _ = self._request(
            "/v1/inbox-jobs",
            json.dumps({"device_id": "bob", "job_id": job_id,
                        "op": "queue"}),
            method="POST")
        self.assertEqual(status, 201)

    def test_default_query_returns_200_with_key_order(self) -> None:
        self._queue_http("J1")
        status, body, raw = self._request("/v1/devices/bob/inbox-job-events")
        self.assertEqual(status, 200)
        self.assertEqual(list(body),
                         ["device_id", "events", "next_after", "has_more"])
        self.assertEqual(list(body["events"][0]),
                         ["seq", "job_id", "type", "state"])
        self.assertEqual(body["events"][0],
                         {"seq": 1, "job_id": "J1", "type": "queue",
                          "state": "pending"})
        self.assertLess(raw.index('"device_id"'), raw.index('"events"'))
        self.assertLess(raw.index('"events"'), raw.index('"next_after"'))
        self.assertLess(raw.index('"next_after"'),
                        raw.index('"has_more"'))

    def test_repeated_gets_are_byte_identical(self) -> None:
        self._queue_http("J1")
        _, _, first = self._request("/v1/devices/bob/inbox-job-events")
        _, _, second = self._request("/v1/devices/bob/inbox-job-events")
        self.assertEqual(first, second)

    def test_error_body_key_order(self) -> None:
        status, body, _ = self._request(
            "/v1/devices/bob/inbox-job-events?bogus=1")
        self.assertEqual(status, 400)
        self.assertEqual(list(body), ["message", "field"])

    def test_unknown_query_parameter_is_400_query(self) -> None:
        for path in ("/v1/devices/bob/inbox-job-events?x=1",
                     "/v1/devices/bob/inbox-job-events?foo",
                     "/v1/devices/bob/inbox-job-events?state=all",
                     "/v1/devices/bob/inbox-job-events?after=1&x="):
            with self.subTest(path=path):
                status, body, _ = self._request(path)
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], "query")

    def test_after_parameter_validation(self) -> None:
        for value in ("-1", "1.5", "%2B1", "x", "1a", "%201", "1%20",
                      "", "9223372036854775808"):
            with self.subTest(value=value):
                status, body, _ = self._request(
                    f"/v1/devices/bob/inbox-job-events?after={value}")
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], "after")
        status, body, _ = self._request(
            "/v1/devices/bob/inbox-job-events?after=1&after=2")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "after")
        # A leading-zero decimal is still an unsigned decimal integer,
        # and 2**63-1 is the largest accepted value.
        status, _, _ = self._request(
            "/v1/devices/bob/inbox-job-events?after=01")
        self.assertEqual(status, 200)
        status, _, _ = self._request(
            "/v1/devices/bob/inbox-job-events?after=9223372036854775807")
        self.assertEqual(status, 200)

    def test_limit_parameter_validation(self) -> None:
        for value in ("0", "101", "-1", "x", "1.0", "%2B1", ""):
            with self.subTest(value=value):
                status, body, _ = self._request(
                    f"/v1/devices/bob/inbox-job-events?limit={value}")
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], "limit")
        status, body, _ = self._request(
            "/v1/devices/bob/inbox-job-events?limit=1&limit=2")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "limit")
        status, _, _ = self._request(
            "/v1/devices/bob/inbox-job-events?limit=100")
        self.assertEqual(status, 200)

    def test_pagination_over_http(self) -> None:
        for job_id in ("J1", "J2", "J3"):
            self._queue_http(job_id)
        status, body, _ = self._request(
            "/v1/devices/bob/inbox-job-events?after=0&limit=2")
        self.assertEqual(status, 200)
        self.assertEqual([e["seq"] for e in body["events"]], [1, 2])
        self.assertEqual(body["next_after"], 2)
        self.assertTrue(body["has_more"])
        status, body, _ = self._request(
            "/v1/devices/bob/inbox-job-events?after=2&limit=2")
        self.assertEqual([e["seq"] for e in body["events"]], [3])
        self.assertEqual(body["next_after"], 3)
        self.assertFalse(body["has_more"])

    def test_nonempty_body_is_400_request_body(self) -> None:
        status, body, _ = self._request(
            "/v1/devices/bob/inbox-job-events", raw="{}")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "request_body")

    def test_empty_query_string_is_allowed(self) -> None:
        status, _, _ = self._request("/v1/devices/bob/inbox-job-events?")
        self.assertEqual(status, 200)

    def test_unknown_device_is_404(self) -> None:
        status, body, _ = self._request(
            "/v1/devices/ghost/inbox-job-events")
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "device_id")


class EventsPersistenceTest(EventsMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self._build()
        self.path = os.path.join(self.directory, "state.json")
        self.state_store = attach_persistence(self.service, self.path)

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def _document(self):
        with open(self.path, encoding="utf-8") as handle:
            return json.load(handle)

    def test_section_follows_redelivery_jobs_with_prefixed_items(self) -> None:
        self._op("J1", "queue")
        document = self._document()
        keys = list(document)
        self.assertEqual(keys.index("redelivery_job_events"),
                         keys.index("redelivery_jobs") + 1)
        self.assertEqual(document["redelivery_job_events"], [
            {"device_id": "bob", "seq": 1, "job_id": "J1",
             "type": "queue", "state": "pending"}])

    def test_restart_restores_chain_and_integrity_passes(self) -> None:
        self._post_message()
        self._op("J1", "queue")
        self._op("J1", "dispatch")
        self.service.inbox_lease_complete(
            "bob", "J1", {"completion_id": "C1", "outcome": "delivered"})
        before = self._events()
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        page = restarted.inbox_job_events_page("bob", 0, 100)
        self.assertEqual(page, before)
        # The chain continues where the restored seq left off.
        restarted.inbox_job({"device_id": "bob", "job_id": "J2",
                             "op": "queue"})
        page = restarted.inbox_job_events_page("bob", 0, 100)
        self.assertEqual(page["events"][-1],
                         {"seq": 4, "job_id": "J2", "type": "queue",
                          "state": "pending"})
        self.assertTrue(restarted.persistence_integrity()["consistent"])

    def test_save_failure_rolls_events_back(self) -> None:
        import e2ee_backend.persistence as persistence_mod

        real_fsync = persistence_mod.os.fsync
        calls = {"n": 0}

        def fail_first_directory_fsync(fd: int) -> None:  # noqa: ANN001
            if os.fstat(fd).st_mode & 0o170000 == 0o040000:
                calls["n"] += 1
                if calls["n"] == 1:
                    raise OSError("transient directory fsync failure")
            real_fsync(fd)

        generation = self.state_store.commit_seq
        persistence_mod.os.fsync = fail_first_directory_fsync
        try:
            with self.assertRaises(PersistenceUnavailable):
                self._op("J1", "queue")
        finally:
            persistence_mod.os.fsync = real_fsync
        # Rolled back: no generation consumed and no event recorded.
        self.assertEqual(self.state_store.commit_seq, generation)
        self.assertEqual(self._events()["events"], [])
        # The queue can be retried and now commits with seq 1.
        _, status = self._op("J1", "queue")
        self.assertEqual(status, 201)
        self.assertEqual(self._chain(), [(1, "J1", "queue", "pending")])

    def test_query_consumes_no_generation(self) -> None:
        self._op("J1", "queue")
        generation = self.state_store.commit_seq
        self._events()
        self._events(after=0, limit=1)
        self.assertEqual(self.state_store.commit_seq, generation)

    def test_legacy_document_without_section_loads(self) -> None:
        self._op("J1", "queue")
        document = self._document()
        document.pop("redelivery_job_events")
        document.pop("integrity_log_version", None)
        legacy_path = os.path.join(self.directory, "legacy.json")
        with open(legacy_path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        restarted = DeviceService()
        attach_persistence(restarted, legacy_path)  # must not raise
        # The dropped chain is simply gone: the job has no events and a
        # fresh operation starts a new chain at seq 1.
        page = restarted.inbox_job_events_page("bob", 0, 100)
        self.assertEqual(page["events"], [])
        restarted.inbox_job({"device_id": "bob", "job_id": "J2",
                             "op": "queue"})
        page = restarted.inbox_job_events_page("bob", 0, 100)
        self.assertEqual(page["events"], [
            {"seq": 1, "job_id": "J2", "type": "queue",
             "state": "pending"}])

    def _document_with_events(self, mutate):
        # A fresh message per invocation (posting the same id twice is a
        # 409), so J1's first dispatch always leases something and runs.
        self._message_seq = getattr(self, "_message_seq", 0) + 1
        self._post_message(f"m{self._message_seq}", self._message_seq)
        self._op("J1", "queue")
        self._op("J1", "dispatch")
        self._op("J2", "queue")
        document = self._document()
        mutate(document)
        # Write to a fresh path as a marker-less/sidecar-less document, so
        # rejection comes from the payload's own semantic validation rather
        # than the integrity hash gate; the original state file is
        # untouched.
        document.pop("integrity_log_version", None)
        bad_path = os.path.join(self.directory, "bad.json")
        with open(bad_path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        return bad_path

    def _assert_refuses_startup(self, bad_path) -> None:
        with open(bad_path, "rb") as handle:
            original = handle.read()
        restarted = DeviceService()
        with self.assertRaises(StateFileError):
            attach_persistence(restarted, bad_path)
        # The rejected file is never overwritten.
        with open(bad_path, "rb") as handle:
            self.assertEqual(handle.read(), original)

    def test_restore_rejects_malformed_section(self) -> None:
        def mutate(document):
            document["redelivery_job_events"] = {}
        self._assert_refuses_startup(self._document_with_events(mutate))

    def test_restore_rejects_bad_item_shape(self) -> None:
        def append(item):
            return lambda d: d["redelivery_job_events"].append(item)
        for mutate in (
                append("x"),
                append({"device_id": "bob", "seq": 4, "job_id": "J1",
                        "type": "queue"}),
                append({"device_id": "bob", "seq": 4, "job_id": "J1",
                        "type": "queue", "state": "pending", "x": 1}),
                append({"device_id": "", "seq": 4, "job_id": "J1",
                        "type": "queue", "state": "pending"}),
                append({"device_id": "bob", "seq": 0, "job_id": "J1",
                        "type": "queue", "state": "pending"}),
                append({"device_id": "bob", "seq": True, "job_id": "J1",
                        "type": "queue", "state": "pending"}),
                append({"device_id": "bob", "seq": 4, "job_id": "",
                        "type": "queue", "state": "pending"}),
                append({"device_id": "bob", "seq": 4, "job_id": "J1",
                        "type": "bogus", "state": "pending"}),
                append({"device_id": "bob", "seq": 4, "job_id": "J1",
                        "type": "queue", "state": "bogus"}),
                append({"device_id": "bob", "seq": 4, "job_id": "J1",
                        "type": "queue", "state": "running"}),
                append({"device_id": "ghost", "seq": 4, "job_id": "J1",
                        "type": "queue", "state": "pending"}),
                append({"device_id": "bob", "seq": 4, "job_id": "nope",
                        "type": "queue", "state": "pending"}),
                append({"device_id": "alice", "seq": 1, "job_id": "J1",
                        "type": "queue", "state": "pending"})):
            with self.subTest(mutate=mutate):
                self._assert_refuses_startup(
                    self._document_with_events(mutate))

    def test_restore_rejects_broken_seq_chain(self) -> None:
        def drop_first(document):
            document["redelivery_job_events"].pop(0)

        def drop_middle(document):
            document["redelivery_job_events"].pop(1)

        def duplicate(document):
            event = dict(document["redelivery_job_events"][1])
            document["redelivery_job_events"].append(event)

        for mutate in (drop_first, drop_middle, duplicate):
            with self.subTest(mutate=mutate):
                self._assert_refuses_startup(
                    self._document_with_events(mutate))

    def test_restore_rejects_last_event_state_mismatch(self) -> None:
        # A valid complete/succeeded event appended for J1 while the job
        # itself is still persisted running: the type/state pair is legal,
        # but the job's last event must match its current state.
        def mutate(document):
            document["redelivery_job_events"].append(
                {"device_id": "bob", "seq": 4, "job_id": "J1",
                 "type": "complete", "state": "succeeded"})
        self._assert_refuses_startup(self._document_with_events(mutate))


if __name__ == "__main__":
    unittest.main()
