"""Tests for the cross-device expired-registration cleanup batch.

``POST /v1/event-gc-batch/cleanup-expired`` takes no query (400/query)
and a body of exactly ``mode``, ``device_ids``, ``after``, ``limit``
(bad JSON/non-object 400/request_body; missing/invalid 400/field; extra
key 400/key; invalid or repeated item 400/device_ids[i]). The page is
the device ids at the zero-based offset ``after``, at most ``limit`` of
them, processed in input order at one instant under the store lock;
revoked devices stay cleanable. The body keys are ``mode``, ``results``,
``next_after``, ``has_more``; each item is ``device_id``, ``watermark``,
``expired``, ``removed``, ``error`` — non-negative integers and a null
error for a known device, nulls and ``{"status": 404, "field":
"device_id"}`` for an unknown one. ``preview`` only counts (200, no
write); ``commit`` deletes across the page in one commit (201 when
anything was removed, else 200) and a data-file failure rolls
everything back with 503/data_file.
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
    attach_persistence,
)


class EventGcBatchMixin:
    def _build(self) -> None:
        self.service = DeviceService()
        self.service.store.add_device(Device("u", "alice", "ik"))
        self.service.store.add_device(Device(
            "u", "bob", "ik",
            prekeys=[SignedPreKey("pk1", "pubk1")]))
        self.service.store.add_device(Device(
            "u", "dave", "ik",
            prekeys=[SignedPreKey("pk2", "pubk2")]))
        self.service.store.create_session("alice", "bob", "pk1", "ek1")
        self.service.store.create_session("alice", "dave", "pk2", "ek2")

    def _op(self, job_id, device="bob"):
        return self.service.inbox_job(
            {"device_id": device, "job_id": job_id, "op": "queue"})

    def _touch(self, consumer="c1", device="bob"):
        return self.service.event_gc(
            device, {"consumer": consumer, "op": "touch", "seq": None})

    def _revoke(self, consumer="c1", device="bob"):
        return self.service.event_gc(
            device, {"consumer": consumer, "op": "revoke", "seq": None})

    def _expire_lease(self, consumer="c1", device="bob"):
        record = self.service.store._redelivery_job_event_checkpoints[
            (device, consumer)]
        record.expires = "2020-01-01T00:00:00.000000+00:00"

    def _batch(self, mode="preview", device_ids=("bob",), after=0,
               limit=100):
        return self.service.event_gc_batch_cleanup_expired(
            {"mode": mode, "device_ids": list(device_ids),
             "after": after, "limit": limit})


class EventGcBatchValidationTest(EventGcBatchMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def _error(self, payload):
        with self.assertRaises(ServiceError) as caught:
            self.service.event_gc_batch_cleanup_expired(payload)
        return caught.exception

    def test_non_object_body_is_400_request_body(self) -> None:
        for payload in (None, "junk", 1, [1], True):
            with self.subTest(payload=payload):
                error = self._error(payload)
                self.assertEqual(error.status_code, 400)
                self.assertEqual(error.field, "request_body")

    def test_missing_fields(self) -> None:
        base = {"mode": "preview", "device_ids": ["bob"], "after": 0,
                "limit": 10}
        for name in ("mode", "device_ids", "after", "limit"):
            with self.subTest(name=name):
                payload = {key: value for key, value in base.items()
                           if key != name}
                error = self._error(payload)
                self.assertEqual(error.status_code, 400)
                self.assertEqual(error.field, name)

    def test_extra_key_is_400_with_that_key(self) -> None:
        error = self._error({"mode": "preview", "device_ids": ["bob"],
                             "after": 0, "limit": 10, "bogus": 1})
        self.assertEqual(error.status_code, 400)
        self.assertEqual(error.field, "bogus")

    def test_mode_validation(self) -> None:
        for mode in (None, "", "PREVIEW", "purge", 1, True, ["preview"]):
            with self.subTest(mode=mode):
                error = self._error({"mode": mode, "device_ids": ["bob"],
                                     "after": 0, "limit": 10})
                self.assertEqual(error.status_code, 400)
                self.assertEqual(error.field, "mode")

    def test_device_ids_array_validation(self) -> None:
        for ids in (None, "bob", {}, 1, True, []):
            with self.subTest(ids=ids):
                error = self._error({"mode": "preview", "device_ids": ids,
                                     "after": 0, "limit": 10})
                self.assertEqual(error.status_code, 400)
                self.assertEqual(error.field, "device_ids")

    def test_device_ids_item_validation(self) -> None:
        for ids, field in (
                ([""], "device_ids[0]"),
                ([None], "device_ids[0]"),
                ([1], "device_ids[0]"),
                (["bob", ""], "device_ids[1]"),
                (["bob", "alice", "bob"], "device_ids[2]")):
            with self.subTest(ids=ids):
                error = self._error({"mode": "preview", "device_ids": ids,
                                     "after": 0, "limit": 10})
                self.assertEqual(error.status_code, 400)
                self.assertEqual(error.field, field)

    def test_after_and_limit_validation(self) -> None:
        for after in (-1, True, 1.5, "0", None):
            with self.subTest(after=after):
                error = self._error({"mode": "preview",
                                     "device_ids": ["bob"],
                                     "after": after, "limit": 10})
                self.assertEqual(error.status_code, 400)
                self.assertEqual(error.field, "after")
        for limit in (0, 101, -1, True, 1.5, "10", None):
            with self.subTest(limit=limit):
                error = self._error({"mode": "preview",
                                     "device_ids": ["bob"],
                                     "after": 0, "limit": limit})
                self.assertEqual(error.status_code, 400)
                self.assertEqual(error.field, "limit")


class EventGcBatchCleanupTest(EventGcBatchMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def test_preview_counts_without_removing(self) -> None:
        self._op("J1")
        self._touch("c1")
        self._touch("c2")
        self._expire_lease("c1")
        body, status = self._batch("preview", ["bob"])
        self.assertEqual(status, 200)
        self.assertEqual(list(body),
                         ["mode", "results", "next_after", "has_more"])
        self.assertEqual(body["mode"], "preview")
        self.assertEqual(list(body["results"][0]),
                         ["device_id", "watermark", "expired", "removed",
                          "error"])
        self.assertEqual(body["results"], [{
            "device_id": "bob", "watermark": 0, "expired": 1,
            "removed": 0, "error": None}])
        self.assertEqual(body["next_after"], 1)
        self.assertFalse(body["has_more"])
        # Nothing was deleted.
        self.assertEqual(
            set(self.service.store._redelivery_job_event_checkpoints),
            {("bob", "c1"), ("bob", "c2")})

    def test_commit_deletes_and_reports(self) -> None:
        self._op("J1")
        self._touch("c1")
        self._touch("c2")
        self._expire_lease("c1")
        body, status = self._batch("commit", ["bob"])
        self.assertEqual(status, 201)
        self.assertEqual(body["results"][0]["expired"], 1)
        self.assertEqual(body["results"][0]["removed"], 1)
        self.assertEqual(
            set(self.service.store._redelivery_job_event_checkpoints),
            {("bob", "c2")})
        # A replay is a no-op 200.
        body, status = self._batch("commit", ["bob"])
        self.assertEqual(status, 200)
        self.assertEqual(body["results"][0]["removed"], 0)

    def test_unknown_device_item(self) -> None:
        body, status = self._batch("preview", ["ghost"])
        self.assertEqual(status, 200)
        self.assertEqual(body["results"], [{
            "device_id": "ghost", "watermark": None, "expired": None,
            "removed": None,
            "error": {"status": 404, "field": "device_id"}}])
        self.assertEqual(list(body["results"][0]["error"]),
                         ["status", "field"])

    def test_mixed_known_unknown_and_revoked(self) -> None:
        self._op("J1", device="bob")
        self._touch("c1", device="bob")
        self._expire_lease("c1", device="bob")
        self._op("J2", device="dave")
        self._touch("c9", device="dave")
        self._expire_lease("c9", device="dave")
        self.service.revoke_device("dave")
        body, status = self._batch(
            "commit", ["bob", "ghost", "dave"])
        self.assertEqual(status, 201)
        results = {item["device_id"]: item for item in body["results"]}
        self.assertEqual(results["bob"]["removed"], 1)
        self.assertIsNone(results["ghost"]["watermark"])
        self.assertEqual(results["ghost"]["error"],
                         {"status": 404, "field": "device_id"})
        # The revoked device is still cleaned.
        self.assertEqual(results["dave"]["removed"], 1)
        self.assertEqual(
            set(self.service.store._redelivery_job_event_checkpoints),
            set())

    def test_revoke_records_and_leaseless_checkpoints_survive(self) -> None:
        self._op("J1")
        self._touch("c1")
        self._revoke("c1")
        self.service.inbox_job_event_checkpoint(
            "bob", {"consumer_id": "c2", "seq": 1})
        body, status = self._batch("commit", ["bob"])
        self.assertEqual(status, 200)
        self.assertEqual(body["results"][0]["expired"], 0)
        self.assertEqual(
            set(self.service.store._redelivery_job_event_checkpoints),
            {("bob", "c1"), ("bob", "c2")})

    def test_watermark_reflects_prune(self) -> None:
        self._op("J1")
        self._op("J2")
        self._touch("c1")
        self.service.inbox_job_event_checkpoint(
            "bob", {"consumer_id": "c1", "seq": 2})
        self.service.event_gc("bob", {"consumer": None, "op": "prune",
                                      "seq": 1})
        body, _ = self._batch("preview", ["bob"])
        self.assertEqual(body["results"][0]["watermark"], 1)

    def test_pagination(self) -> None:
        ids = ["bob", "dave", "alice"]
        body, status = self._batch("preview", ids, after=0, limit=2)
        self.assertEqual(status, 200)
        self.assertEqual([i["device_id"] for i in body["results"]],
                         ["bob", "dave"])
        self.assertEqual(body["next_after"], 2)
        self.assertTrue(body["has_more"])
        body, _ = self._batch("preview", ids, after=2, limit=2)
        self.assertEqual([i["device_id"] for i in body["results"]],
                         ["alice"])
        self.assertEqual(body["next_after"], 3)
        self.assertFalse(body["has_more"])
        # An empty page echoes the offset.
        body, _ = self._batch("preview", ids, after=9, limit=2)
        self.assertEqual(body["results"], [])
        self.assertEqual(body["next_after"], 9)
        self.assertFalse(body["has_more"])

    def test_commit_spans_page_devices_in_one_commit(self) -> None:
        self._op("J1", device="bob")
        self._touch("c1", device="bob")
        self._expire_lease("c1", device="bob")
        self._op("J2", device="dave")
        self._touch("c9", device="dave")
        self._expire_lease("c9", device="dave")
        body, status = self._batch("commit", ["bob", "dave"])
        self.assertEqual(status, 201)
        self.assertEqual([i["removed"] for i in body["results"]], [1, 1])
        self.assertEqual(
            set(self.service.store._redelivery_job_event_checkpoints),
            set())


class EventGcBatchHTTPTest(EventGcBatchMixin, unittest.TestCase):
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

    def _request(self, path, raw=None, method="POST"):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        headers = {"Content-Type": "application/json"} if raw is not None \
            else {}
        conn.request(method, path, body=raw, headers=headers)
        response = conn.getresponse()
        data = response.read().decode("utf-8")
        conn.close()
        return response.status, (json.loads(data) if data else None), data

    def _batch_http(self, payload, query=""):
        return self._request(
            f"/v1/event-gc-batch/cleanup-expired{query}",
            raw=json.dumps(payload))

    def test_flow_with_key_order(self) -> None:
        status, body, raw = self._batch_http(
            {"mode": "preview", "device_ids": ["bob", "ghost"],
             "after": 0, "limit": 100})
        self.assertEqual(status, 200)
        self.assertEqual(list(body),
                         ["mode", "results", "next_after", "has_more"])
        self.assertEqual(list(body["results"][0]),
                         ["device_id", "watermark", "expired", "removed",
                          "error"])
        self.assertEqual(list(body["results"][1]["error"]),
                         ["status", "field"])
        for earlier, later in (
                ('"mode"', '"results"'),
                ('"results"', '"next_after"'),
                ('"next_after"', '"has_more"'),
                ('"device_id"', '"watermark"'),
                ('"watermark"', '"expired"'),
                ('"expired"', '"removed"'),
                ('"removed"', '"error"'),
                ('"status"', '"field"')):
            self.assertLess(raw.index(earlier), raw.index(later))

    def test_query_is_400_query(self) -> None:
        for query in ("?x=1", "?after=0"):
            with self.subTest(query=query):
                status, body, _ = self._batch_http(
                    {"mode": "preview", "device_ids": ["bob"],
                     "after": 0, "limit": 1}, query=query)
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], "query")
        # A bare trailing ? is accepted.
        status, _, _ = self._batch_http(
            {"mode": "preview", "device_ids": ["bob"],
             "after": 0, "limit": 1}, query="?")
        self.assertEqual(status, 200)

    def test_bad_json_is_400_request_body(self) -> None:
        status, body, _ = self._request(
            "/v1/event-gc-batch/cleanup-expired", raw="junk")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "request_body")
        self.assertEqual(list(body), ["message", "field"])

    def test_commit_over_http(self) -> None:
        self._op("J1")
        self._touch("c1")
        self._expire_lease("c1")
        status, body, _ = self._batch_http(
            {"mode": "commit", "device_ids": ["bob"],
             "after": 0, "limit": 100})
        self.assertEqual(status, 201)
        self.assertEqual(body["results"][0]["removed"], 1)

    def test_get_is_404(self) -> None:
        status, _, _ = self._request(
            "/v1/event-gc-batch/cleanup-expired", method="GET")
        self.assertEqual(status, 404)


class EventGcBatchPersistenceTest(EventGcBatchMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self._build()
        self.path = os.path.join(self.directory, "state.json")
        self.state_store = attach_persistence(self.service, self.path)

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_preview_consumes_no_generation(self) -> None:
        self._op("J1")
        self._touch("c1")
        self._expire_lease("c1")
        generation = self.state_store.commit_seq
        body, status = self._batch("preview", ["bob"])
        self.assertEqual(status, 200)
        self.assertEqual(body["results"][0]["expired"], 1)
        self.assertEqual(self.state_store.commit_seq, generation)

    def test_commit_is_one_generation_and_survives_restart(self) -> None:
        self._op("J1", device="bob")
        self._touch("c1", device="bob")
        self._expire_lease("c1", device="bob")
        self._op("J2", device="dave")
        self._touch("c9", device="dave")
        self._expire_lease("c9", device="dave")
        generation = self.state_store.commit_seq
        body, status = self._batch("commit", ["bob", "dave"])
        self.assertEqual(status, 201)
        self.assertEqual([i["removed"] for i in body["results"]], [1, 1])
        self.assertEqual(self.state_store.commit_seq, generation + 1)
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        self.assertEqual(
            set(restarted.store._redelivery_job_event_checkpoints), set())
        self.assertTrue(restarted.persistence_integrity()["consistent"])

    def test_save_failure_rolls_whole_batch_back(self) -> None:
        import e2ee_backend.persistence as persistence_mod

        self._op("J1", device="bob")
        self._touch("c1", device="bob")
        self._expire_lease("c1", device="bob")
        self._op("J2", device="dave")
        self._touch("c9", device="dave")
        self._expire_lease("c9", device="dave")
        # Anchor the expired leases durably via another commit.
        self._touch("c2", device="bob")
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
                self._batch("commit", ["bob", "dave"])
        finally:
            persistence_mod.os.fsync = real_fsync
        # Nothing advanced: no generation consumed, every expired record
        # is back in memory.
        self.assertEqual(self.state_store.commit_seq, generation)
        self.assertEqual(
            set(self.service.store._redelivery_job_event_checkpoints),
            {("bob", "c1"), ("bob", "c2"), ("dave", "c9")})
        # The batch can be retried and now commits once.
        body, status = self._batch("commit", ["bob", "dave"])
        self.assertEqual(status, 201)
        self.assertEqual(self.state_store.commit_seq, generation + 1)
        self.assertEqual(
            set(self.service.store._redelivery_job_event_checkpoints),
            {("bob", "c2")})


if __name__ == "__main__":
    unittest.main()
