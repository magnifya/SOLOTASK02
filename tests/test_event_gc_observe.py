"""Tests for the redelivery-event retention observation and the expired
registration cleanup.

``GET /v1/event-gc/{device_id}`` is a read-only paged observation of a
device's retention watermark, surviving event-chain bounds and every
checkpoint registration (sorted by ``consumer_id`` code point): a
non-empty body is 400/request_body, ``after``/``limit`` follow the events
pagination contract, any other parameter is 400/query, an unknown device
is 404/device_id and revoked registrations stay visible.

``POST /v1/event-gc/{device_id}/cleanup-expired`` takes no query and no
body and deletes every registration whose non-null ``expires`` has
passed (revoke records and lease-less registrations are kept): nothing
expired is 200/removed 0 and writes nothing, otherwise one commit makes
it 201; a deleted registration reads again as seq 0. Both share the
store lock and the persistence transaction of the existing event GC.
"""
import json
import os
import shutil
import tempfile
import threading
import unittest
from datetime import datetime, timedelta, timezone
from http.client import HTTPConnection

from e2ee_backend.models import Device, SignedPreKey
from e2ee_backend.service import DeviceService, ServiceError
from e2ee_backend.http_app import create_server
from e2ee_backend.persistence import PersistenceUnavailable, attach_persistence


class EventGcObserveMixin:
    def _build(self) -> None:
        self.service = DeviceService()
        self.service.store.add_device(Device("u", "alice", "ik"))
        self.service.store.add_device(Device(
            "u", "bob", "ik",
            prekeys=[SignedPreKey("pk1", "pubk1")]))
        self.sid = self.service.store.create_session(
            "alice", "bob", "pk1", "ek1").session_id

    def _op(self, job_id, op="queue", device="bob", **extra):
        payload = {"device_id": device, "job_id": job_id, "op": op}
        payload.update(extra)
        return self.service.inbox_job(payload)

    def _touch(self, consumer, device="bob"):
        return self.service.event_gc(
            device, {"consumer": consumer, "op": "touch", "seq": None})

    def _revoke(self, consumer, device="bob"):
        return self.service.event_gc(
            device, {"consumer": consumer, "op": "revoke", "seq": None})

    def _checkpoint(self, consumer, seq, device="bob"):
        return self.service.inbox_job_event_checkpoint(
            device, {"consumer_id": consumer, "seq": seq})

    def _observe(self, device="bob", after=0, limit=100):
        return self.service.event_gc_observe(device, after, limit)

    def _expire(self, consumer, device="bob"):
        record = self.service.store._redelivery_job_event_checkpoints[
            (device, consumer)]
        record.expires = (
            datetime.now(timezone.utc) - timedelta(days=1)
        ).isoformat(timespec="microseconds")

    def _cleanup(self, device="bob"):
        return self.service.event_gc_cleanup_expired(device)


class EventGcObserveTest(EventGcObserveMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def test_unknown_device_is_404(self) -> None:
        with self.assertRaises(ServiceError) as caught:
            self._observe(device="ghost")
        self.assertEqual(caught.exception.status_code, 404)
        self.assertEqual(caught.exception.field, "device_id")

    def test_revoked_device_stays_observable(self) -> None:
        self._touch("c1")
        self.service.revoke_device("bob")
        body = self._observe()
        self.assertEqual(body["device_id"], "bob")
        self.assertEqual([c["consumer_id"] for c in body["consumers"]], ["c1"])

    def test_empty_device_body_and_key_order(self) -> None:
        body = self._observe()
        self.assertEqual(list(body),
                         ["device_id", "watermark", "first_seq", "last_seq",
                          "consumers", "next_after", "has_more"])
        self.assertEqual(body, {
            "device_id": "bob", "watermark": 0,
            "first_seq": None, "last_seq": None,
            "consumers": [], "next_after": 0, "has_more": False})

    def test_watermark_and_chain_bounds(self) -> None:
        self._op("J1")
        self._op("J2")
        self._op("J3")
        self._touch("c1")
        self._checkpoint("c1", 3)
        self.service.event_gc(
            "bob", {"consumer": None, "op": "prune", "seq": 2})
        body = self._observe()
        self.assertEqual(body["watermark"], 2)
        self.assertEqual(body["first_seq"], 3)
        self.assertEqual(body["last_seq"], 3)

    def test_consumers_sorted_with_item_key_order(self) -> None:
        self._op("J1")
        self._touch("c2")
        self._touch("c10")
        self._touch("c1")  # code-point order: c1, c10, c2
        body = self._observe()
        self.assertEqual([c["consumer_id"] for c in body["consumers"]],
                         ["c1", "c10", "c2"])
        for consumer in body["consumers"]:
            self.assertEqual(list(consumer),
                             ["consumer_id", "seq", "updated_at", "active",
                              "expires"])

    def test_active_reflects_revoke_expiry_watermark(self) -> None:
        self._op("J1")
        self._op("J2")
        self._touch("active-c")          # seq 0, fresh lease -> active
        self._touch("revoked-c")
        self._revoke("revoked-c")        # active false, expires null
        self._touch("expired-c")
        self._expire("expired-c")        # active false
        self._touch("legacy-c")
        # A lease-less record (legacy four-key shape) never expires and so
        # stays active once its seq is at least the watermark.
        legacy = self.service.store._redelivery_job_event_checkpoints[
            ("bob", "legacy-c")]
        legacy.expires = None
        by_id = {c["consumer_id"]: c for c in self._observe()["consumers"]}
        self.assertTrue(by_id["active-c"]["active"])
        self.assertFalse(by_id["revoked-c"]["active"])
        self.assertIsNone(by_id["revoked-c"]["expires"])
        self.assertFalse(by_id["expired-c"]["active"])
        self.assertTrue(by_id["legacy-c"]["active"])
        self.assertIsNone(by_id["legacy-c"]["expires"])
        # Advance the watermark past active-c's checkpoint (seq 0): it is
        # retained in the listing but reports inactive.
        self._checkpoint("active-c", 2)
        self._checkpoint("legacy-c", 2)
        self.service.event_gc(
            "bob", {"consumer": None, "op": "prune", "seq": 1})
        by_id = {c["consumer_id"]: c for c in self._observe()["consumers"]}
        self.assertFalse(by_id["expired-c"]["active"])
        self.assertTrue(by_id["active-c"]["active"])
        self.assertTrue(by_id["legacy-c"]["active"])

    def test_pagination_offset_and_has_more(self) -> None:
        for cid in ("a", "b", "c", "d"):
            self._touch(cid)
        page = self._observe(after=1, limit=2)
        self.assertEqual([c["consumer_id"] for c in page["consumers"]],
                         ["b", "c"])
        self.assertEqual(page["next_after"], 3)
        self.assertTrue(page["has_more"])
        last = self._observe(after=3, limit=2)
        self.assertEqual([c["consumer_id"] for c in last["consumers"]], ["d"])
        self.assertEqual(last["next_after"], 4)
        self.assertFalse(last["has_more"])
        empty = self._observe(after=4, limit=2)
        self.assertEqual(empty["consumers"], [])
        self.assertEqual(empty["next_after"], 4)
        self.assertFalse(empty["has_more"])

    def test_observe_is_read_only(self) -> None:
        self._op("J1")
        self._touch("c1")
        first = self._observe()
        second = self._observe(after=0, limit=50)
        self.assertEqual(first, second)
        self.assertEqual(
            self.service.store._redelivery_job_event_checkpoints[
                ("bob", "c1")].expires,
            first["consumers"][0]["expires"])


class EventGcObserveHTTPTest(EventGcObserveMixin, unittest.TestCase):
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

    def _request(self, path, raw=None, method=None):
        use_method = method or ("POST" if raw is not None else "GET")
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        headers = {"Content-Type": "application/json"} if raw else {}
        conn.request(use_method, path, body=raw, headers=headers)
        response = conn.getresponse()
        data = response.read().decode("utf-8")
        conn.close()
        return response.status, (json.loads(data) if data else None), data

    def test_observe_ok_with_key_order(self) -> None:
        self._op("J1")
        self._touch("c1")
        status, body, raw = self._request("/v1/event-gc/bob")
        self.assertEqual(status, 200)
        self.assertEqual(list(body),
                         ["device_id", "watermark", "first_seq", "last_seq",
                          "consumers", "next_after", "has_more"])
        self.assertEqual(body["first_seq"], 1)
        self.assertEqual(body["last_seq"], 1)
        self.assertLess(raw.index('"device_id"'), raw.index('"watermark"'))
        self.assertLess(raw.index('"watermark"'), raw.index('"first_seq"'))

    def test_nonempty_body_is_400_request_body(self) -> None:
        status, body, _ = self._request("/v1/event-gc/bob", raw="{}",
                                        method="GET")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "request_body")
        self.assertEqual(list(body), ["message", "field"])

    def test_query_validation(self) -> None:
        for suffix, field in (
                ("?foo=1", "query"),
                ("?after", "after"),
                ("?after=-1", "after"),
                ("?after=1.0", "after"),
                ("?after=x", "after"),
                ("?after=1&after=2", "after"),
                ("?limit=0", "limit"),
                ("?limit=101", "limit"),
                ("?limit=x", "limit"),
                ("?limit=1&limit=2", "limit"),
                ("?state=all", "query")):
            with self.subTest(suffix=suffix):
                status, body, _ = self._request(f"/v1/event-gc/bob{suffix}")
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], field)

    def test_trailing_empty_question_mark_is_ok(self) -> None:
        status, _, _ = self._request("/v1/event-gc/bob?")
        self.assertEqual(status, 200)

    def test_unknown_and_misrouted_paths(self) -> None:
        status, body, _ = self._request("/v1/event-gc/ghost")
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "device_id")
        status, body, _ = self._request("/v1/event-gc/bob/extra")
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "device_id")

    def test_cleanup_flow(self) -> None:
        self._op("J1")
        self._touch("kept-revoked")
        self._revoke("kept-revoked")
        self._touch("gone-expired")
        self._expire("gone-expired")
        status, body, raw = self._request(
            "/v1/event-gc/bob/cleanup-expired", raw="")
        self.assertEqual(status, 201)
        self.assertEqual(list(body), ["device_id", "removed"])
        self.assertEqual(body, {"device_id": "bob", "removed": 1})
        records = self.service.store._redelivery_job_event_checkpoints
        self.assertNotIn(("bob", "gone-expired"), records)
        self.assertIn(("bob", "kept-revoked"), records)
        # A deleted registration reads as never registered.
        read, read_status = self.service.inbox_job_event_checkpoint(
            "bob", {"consumer_id": "gone-expired", "seq": None})
        self.assertEqual(read_status, 200)
        self.assertEqual(read["seq"], 0)
        # Nothing left to remove: 200 and no write.
        status, body, _ = self._request(
            "/v1/event-gc/bob/cleanup-expired", raw="")
        self.assertEqual(status, 200)
        self.assertEqual(body, {"device_id": "bob", "removed": 0})

    def test_cleanup_validation(self) -> None:
        status, body, _ = self._request(
            "/v1/event-gc/bob/cleanup-expired?x=1", raw="")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "query")
        status, body, _ = self._request(
            "/v1/event-gc/bob/cleanup-expired", raw="{}")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "request_body")
        status, body, _ = self._request(
            "/v1/event-gc/ghost/cleanup-expired", raw="")
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "device_id")
        # A GET on the POST-only cleanup path is not a route: 404.
        status, _, _ = self._request(
            "/v1/event-gc/bob/cleanup-expired")
        self.assertEqual(status, 404)
        # A trailing empty '?' is accepted on the POST cleanup route.
        status, body, _ = self._request(
            "/v1/event-gc/bob/cleanup-expired?", raw="")
        self.assertEqual(status, 200)
        self.assertEqual(body["removed"], 0)


class EventGcCleanupPersistenceTest(EventGcObserveMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self._build()
        self.path = os.path.join(self.directory, "state.json")
        self.state_store = attach_persistence(self.service, self.path)

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_cleanup_consumes_one_generation_and_persists(self) -> None:
        self._op("J1")
        self._touch("c1")
        self._expire("c1")
        generation = self.state_store.commit_seq
        body, status = self._cleanup()
        self.assertEqual(status, 201)
        self.assertEqual(body["removed"], 1)
        self.assertEqual(self.state_store.commit_seq, generation + 1)
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        self.assertNotIn(
            ("bob", "c1"),
            restarted.store._redelivery_job_event_checkpoints)

    def test_noop_cleanup_consumes_no_generation(self) -> None:
        self._op("J1")
        self._touch("c1")
        generation = self.state_store.commit_seq
        body, status = self._cleanup()
        self.assertEqual(status, 200)
        self.assertEqual(body["removed"], 0)
        self.assertEqual(self.state_store.commit_seq, generation)

    def test_save_failure_rolls_cleanup_back(self) -> None:
        import e2ee_backend.persistence as persistence_mod

        self._op("J1")
        self._touch("c1")
        self._expire("c1")
        # Persist the expired state as the last committed generation (via
        # an unrelated consumer) so a rollback restores c1 still expired.
        self._touch("c2")
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
                self._cleanup()
        finally:
            persistence_mod.os.fsync = real_fsync
        self.assertEqual(self.state_store.commit_seq, generation)
        # The expired registration is still present after the rollback.
        self.assertIn(
            ("bob", "c1"),
            self.service.store._redelivery_job_event_checkpoints)
        body, status = self._cleanup()
        self.assertEqual(status, 201)
        self.assertEqual(body["removed"], 1)


if __name__ == "__main__":
    unittest.main()
