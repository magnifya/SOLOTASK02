"""Tests for the batch-cleanup audit checkpoint page.

``GET /v1/event-gc-batch/checkpoints`` takes no request body (non-empty
-> 400/request_body) and only the single-valued query parameters
``after``/``limit`` (defaults 0/100): strict ASCII decimal in 0..2**63-1
and 1..100 respectively; a repeated, empty or malformed parameter is 400
with that parameter name as field, and any other parameter is 400/query.
Validation order is request body, other parameters, ``after``, ``limit``.

The body keys are ``audit_count``, ``checkpoints``, ``next_after`` and
``has_more`` in that order; checkpoints are paged in creation order
(consumers that never advanced are not listed) and each item's keys are
``consumer_id``, ``after``, ``updated_at`` and ``pending`` in that
order, with ``pending`` the count of audit records the consumer has not
yet consumed. The lookup is read-only: no persistence, no ``commit_seq``
advance, byte-identical while state is unchanged and stable across a
restart.
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
from e2ee_backend.persistence import attach_persistence

PATH = "/v1/event-gc-batch/checkpoints"


class CheckpointsMixin:
    def _build(self) -> None:
        self.service = DeviceService()
        self.service.store.add_device(Device("u", "alice", "ik"))
        self.service.store.add_device(Device(
            "u", "bob", "ik",
            prekeys=[SignedPreKey("pk1", "pubk1")]))

    def _op(self, job_id, device="bob"):
        return self.service.inbox_job(
            {"device_id": device, "job_id": job_id, "op": "queue"})

    def _touch(self, consumer, device="bob"):
        return self.service.event_gc(
            device, {"consumer": consumer, "op": "touch", "seq": None})

    def _expire_lease(self, consumer, device="bob"):
        record = self.service.store._redelivery_job_event_checkpoints[
            (device, consumer)]
        record.expires = "2020-01-01T00:00:00.000000+00:00"

    def _commit(self, request_id, device_ids, after=0, limit=100):
        return self.service.event_gc_cleanup_expired_batch({
            "mode": "commit", "device_ids": device_ids,
            "after": after, "limit": limit, "request_id": request_id})

    def _advance(self, consumer_id, expected, after):
        return self.service.event_gc_batch_checkpoint({
            "consumer_id": consumer_id,
            "expected": expected, "after": after})

    def _seed(self):
        # Two audit records (commit order r1, r2), then checkpoints in
        # creation order c1 (after 1), c2 (after 2). c3 never advances.
        self._op("J1")
        self._touch("c1", device="bob")
        self._expire_lease("c1", device="bob")
        self._commit("r1", ["bob"])
        self._touch("c2", device="bob")
        self._expire_lease("c2", device="bob")
        self._commit("r2", ["bob"])
        self._advance("c1", 0, 1)
        self._advance("c2", 0, 2)


class CheckpointsServiceTest(CheckpointsMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def test_shape_key_order_and_pending(self) -> None:
        self._seed()
        body = self.service.event_gc_batch_checkpoint_list(0, 100)
        self.assertEqual(list(body),
                         ["audit_count", "checkpoints", "next_after",
                          "has_more"])
        self.assertEqual(body["audit_count"], 2)
        self.assertEqual([item["consumer_id"]
                          for item in body["checkpoints"]], ["c1", "c2"])
        first, second = body["checkpoints"]
        for item in body["checkpoints"]:
            self.assertEqual(list(item),
                             ["consumer_id", "after", "updated_at",
                              "pending"])
        self.assertEqual(first["after"], 1)
        self.assertIsInstance(first["updated_at"], str)
        self.assertEqual(first["pending"], 1)
        self.assertEqual(second["after"], 2)
        self.assertEqual(second["pending"], 0)
        self.assertEqual(body["next_after"], 2)
        self.assertFalse(body["has_more"])

    def test_consumers_without_records_are_not_listed(self) -> None:
        self._seed()
        # A read-only checkpoint query creates no record.
        self.service.event_gc_batch_checkpoint({
            "consumer_id": "ghost", "expected": None, "after": None})
        body = self.service.event_gc_batch_checkpoint_list(0, 100)
        self.assertEqual([item["consumer_id"]
                          for item in body["checkpoints"]], ["c1", "c2"])

    def test_empty(self) -> None:
        body = self.service.event_gc_batch_checkpoint_list(0, 100)
        self.assertEqual(body, {"audit_count": 0, "checkpoints": [],
                                "next_after": 0, "has_more": False})

    def test_audit_count_tracks_commits(self) -> None:
        self._seed()
        self._touch("c3", device="bob")
        self._expire_lease("c3", device="bob")
        self._commit("r3", ["bob"])
        body = self.service.event_gc_batch_checkpoint_list(0, 100)
        self.assertEqual(body["audit_count"], 3)
        self.assertEqual([item["pending"] for item in body["checkpoints"]],
                         [2, 1])

    def test_paging(self) -> None:
        self._seed()
        first = self.service.event_gc_batch_checkpoint_list(0, 1)
        self.assertEqual([i["consumer_id"] for i in first["checkpoints"]],
                         ["c1"])
        self.assertEqual(first["next_after"], 1)
        self.assertTrue(first["has_more"])
        second = self.service.event_gc_batch_checkpoint_list(1, 1)
        self.assertEqual([i["consumer_id"] for i in second["checkpoints"]],
                         ["c2"])
        self.assertEqual(second["next_after"], 2)
        self.assertFalse(second["has_more"])
        empty = self.service.event_gc_batch_checkpoint_list(2, 1)
        self.assertEqual(empty["checkpoints"], [])
        self.assertEqual(empty["next_after"], 2)
        self.assertFalse(empty["has_more"])

    def test_validation_in_service(self) -> None:
        for after in (-1, 2**63, 0.0, "0", True, False):
            with self.subTest(after=after):
                with self.assertRaises(ServiceError) as caught:
                    self.service.event_gc_batch_checkpoint_list(after, 100)
                self.assertEqual(caught.exception.field, "after")
        for limit in (0, 101, -1, 1.0, "1", True):
            with self.subTest(limit=limit):
                with self.assertRaises(ServiceError) as caught:
                    self.service.event_gc_batch_checkpoint_list(0, limit)
                self.assertEqual(caught.exception.field, "limit")


class CheckpointsHTTPTest(CheckpointsMixin, unittest.TestCase):
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

    def _request(self, path=PATH, raw=None, method="GET"):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        headers = {"Content-Type": "application/json"} if raw is not None \
            else {}
        conn.request(method, path, body=raw, headers=headers)
        response = conn.getresponse()
        data = response.read().decode("utf-8")
        conn.close()
        return response.status, (json.loads(data) if data else None), data

    def test_page_over_http(self) -> None:
        self._seed()
        status, body, raw = self._request()
        self.assertEqual(status, 200)
        self.assertEqual(list(body),
                         ["audit_count", "checkpoints", "next_after",
                          "has_more"])
        self.assertEqual([i["consumer_id"] for i in body["checkpoints"]],
                         ["c1", "c2"])
        for earlier, later in (
                ('"audit_count"', '"checkpoints"'),
                ('"checkpoints"', '"next_after"'),
                ('"next_after"', '"has_more"'),
                ('"consumer_id"', '"after"'),
                ('"after"', '"updated_at"'),
                ('"updated_at"', '"pending"')):
            self.assertLess(raw.index(earlier), raw.index(later))

    def test_empty_collection_over_http(self) -> None:
        status, body, _ = self._request()
        self.assertEqual(status, 200)
        self.assertEqual(body, {"audit_count": 0, "checkpoints": [],
                                "next_after": 0, "has_more": False})

    def test_paging_query(self) -> None:
        self._seed()
        status, body, _ = self._request(path=PATH + "?after=1&limit=1")
        self.assertEqual(status, 200)
        self.assertEqual([i["consumer_id"] for i in body["checkpoints"]],
                         ["c2"])
        self.assertEqual(body["next_after"], 2)
        self.assertFalse(body["has_more"])
        status, body, _ = self._request(path=PATH + "?after=9&limit=2")
        self.assertEqual(status, 200)
        self.assertEqual(body["checkpoints"], [])
        self.assertEqual(body["next_after"], 9)
        self.assertFalse(body["has_more"])

    def test_query_validation(self) -> None:
        good_after = str(2**63 - 1)
        cases = [
            ("after=-1", "after"),
            ("after=01%20", "after"),
            ("after=%201", "after"),
            ("after=+1", "after"),
            ("after=1.0", "after"),
            ("after=0x1", "after"),
            ("after=abc", "after"),
            ("after=", "after"),
            ("after=" + str(2**63), "after"),
            ("after=" + good_after + "0", "after"),
            ("after=1&after=2", "after"),
            ("limit=0", "limit"),
            ("limit=101", "limit"),
            ("limit=-1", "limit"),
            ("limit=", "limit"),
            ("limit=x", "limit"),
            ("limit=1&limit=2", "limit"),
            ("foo=1", "query"),
            ("after=0&mode=preview", "query"),
        ]
        for query, field in cases:
            with self.subTest(query=query):
                status, body, _ = self._request(path=PATH + "?" + query)
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], field)
                self.assertEqual(list(body), ["message", "field"])

    def test_boundary_values_accepted(self) -> None:
        self._seed()
        status, body, _ = self._request(
            path=PATH + "?after=9223372036854775807&limit=1")
        self.assertEqual(status, 200)
        self.assertEqual(body["checkpoints"], [])
        self.assertEqual(body["next_after"], 2**63 - 1)
        status, body, _ = self._request(path=PATH + "?limit=100")
        self.assertEqual(status, 200)
        self.assertEqual(len(body["checkpoints"]), 2)

    def test_trailing_question_mark_uses_defaults(self) -> None:
        self._seed()
        status, body, _ = self._request(path=PATH + "?")
        self.assertEqual(status, 200)
        self.assertEqual([i["consumer_id"] for i in body["checkpoints"]],
                         ["c1", "c2"])

    def test_nonempty_body_rejected(self) -> None:
        status, body, _ = self._request(raw="{}")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "request_body")
        self.assertEqual(list(body), ["message", "field"])

    def test_validation_order_body_before_query(self) -> None:
        # A non-empty body is rejected before any query parameter is
        # looked at.
        status, body, _ = self._request(path=PATH + "?foo=1", raw="{}")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "request_body")

    def test_validation_order_query_before_after(self) -> None:
        status, body, _ = self._request(path=PATH + "?after=x&foo=1")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "query")

    def test_validation_order_after_before_limit(self) -> None:
        status, body, _ = self._request(path=PATH + "?after=x&limit=0")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "after")

    def test_post_is_404(self) -> None:
        status, _, _ = self._request(method="POST", raw="{}")
        self.assertEqual(status, 404)


class CheckpointsPersistenceTest(CheckpointsMixin, unittest.TestCase):
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

    def test_read_writes_nothing_and_advances_no_generation(self) -> None:
        self._seed()
        generation = self.state_store.commit_seq
        before = self._document()
        first = self.service.event_gc_batch_checkpoint_list(0, 100)
        first_raw = json.dumps(first, separators=(",", ":"),
                               ensure_ascii=False)
        # Repeated reads are byte-identical while nothing changes.
        second = self.service.event_gc_batch_checkpoint_list(0, 100)
        self.assertEqual(json.dumps(second, separators=(",", ":"),
                                    ensure_ascii=False), first_raw)
        self.assertEqual(self.state_store.commit_seq, generation)
        self.assertEqual(self._document(), before)

    def test_page_survives_restart_in_creation_order(self) -> None:
        self._seed()
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        body = restarted.event_gc_batch_checkpoint_list(0, 100)
        self.assertEqual(body["audit_count"], 2)
        self.assertEqual([item["consumer_id"]
                          for item in body["checkpoints"]], ["c1", "c2"])
        self.assertEqual([item["after"] for item in body["checkpoints"]],
                         [1, 2])
        self.assertEqual([item["pending"] for item in body["checkpoints"]],
                         [1, 0])
        for item in body["checkpoints"]:
            self.assertEqual(list(item),
                             ["consumer_id", "after", "updated_at",
                              "pending"])


if __name__ == "__main__":
    unittest.main()
