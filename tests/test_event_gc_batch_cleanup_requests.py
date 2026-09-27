"""Tests for the batch cleanup idempotency-record audit.

``GET /v1/event-gc-batch/cleanup-expired/requests/{request_id}`` takes no
request body (non-empty -> 400/request_body) and no query parameters
(any -> 400/query); the path id is strictly percent-decoded as UTF-8 (a
bad escape or invalid UTF-8 is 400/request_id) and an unknown id is
404/request_id.

``GET /v1/event-gc-batch/cleanup-expired/requests`` takes no request body
and single-valued ``after``/``limit`` query parameters (defaults 0/100):
strict ASCII decimal in 0..2**63-1 and 1..100 respectively; a repeated,
empty or malformed parameter is 400 with that parameter name as field,
and any other parameter is 400/query.

The list body keys are ``requests``, ``next_after`` and ``has_more`` in
that order; records are paged in commit order and each item's keys are
``request_id``, ``device_ids``, ``after``, ``limit``, ``status`` and
``response`` in that order, with the frozen response's nested key order
and values untouched. Both lookups are read-only: no persistence, no
``commit_seq`` advance, byte-identical while state is unchanged and
stable across a restart.
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

PATH = "/v1/event-gc-batch/cleanup-expired/requests"


class RequestsMixin:
    def _build(self) -> None:
        self.service = DeviceService()
        self.service.store.add_device(Device("u", "alice", "ik"))
        self.service.store.add_device(Device(
            "u", "bob", "ik",
            prekeys=[SignedPreKey("pk1", "pubk1")]))
        self.service.store.add_device(Device("u", "dave", "ik"))

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

    def _seed(self):
        # r1: a real deletion (201); r2: deletion-less (200); r3: unknown
        # devices only (200). Commit order is r1, r2, r3.
        self._op("J1")
        self._touch("c1", device="bob")
        self._touch("c2", device="bob")
        self._expire_lease("c1", device="bob")
        self._expire_lease("c2", device="bob")
        r1_body, r1_status = self._commit("r1", ["bob"])
        self._touch("c3", device="dave")
        r2_body, r2_status = self._commit(
            "r2", ["alice", "dave", "ghost"], after=1, limit=1)
        r3_body, r3_status = self._commit("r3", ["ghost"])
        return [("r1", r1_body, r1_status),
                ("r2", r2_body, r2_status),
                ("r3", r3_body, r3_status)]


class RequestsServiceTest(RequestsMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def test_detail_shape_key_order_and_frozen_values(self) -> None:
        seeded = dict((rid, (body, status))
                      for rid, body, status in self._seed())
        body = self.service.event_gc_batch_cleanup_request_get("r1")
        self.assertEqual(list(body),
                         ["request_id", "device_ids", "after", "limit",
                          "status", "response"])
        self.assertEqual(body["request_id"], "r1")
        self.assertEqual(body["device_ids"], ["bob"])
        self.assertEqual(body["after"], 0)
        self.assertEqual(body["limit"], 100)
        self.assertEqual(body["status"], seeded["r1"][1])
        self.assertEqual(body["response"], seeded["r1"][0])
        self.assertEqual(list(body["response"]),
                         ["mode", "results", "next_after", "has_more"])
        self.assertEqual(list(body["response"]["results"][0]),
                         ["device_id", "watermark", "expired", "removed",
                          "error"])

    def test_detail_keeps_frozen_response_after_state_changes(self) -> None:
        self._seed()
        body = self.service.event_gc_batch_cleanup_request_get("r2")
        self.assertEqual(body["status"], 200)
        self.assertEqual(body["device_ids"], ["alice", "dave", "ghost"])
        self.assertEqual(body["after"], 1)
        self.assertEqual(body["limit"], 1)
        frozen = body["response"]
        # A later mutation cannot change the frozen response.
        self._touch("later", device="alice")
        again = self.service.event_gc_batch_cleanup_request_get("r2")
        self.assertEqual(again["response"], frozen)

    def test_detail_device_ids_are_copies(self) -> None:
        self._seed()
        body = self.service.event_gc_batch_cleanup_request_get("r1")
        body["device_ids"].append("tampered")
        body["response"]["results"].append("tampered")
        again = self.service.event_gc_batch_cleanup_request_get("r1")
        self.assertEqual(again["device_ids"], ["bob"])
        self.assertEqual(
            [item["device_id"] for item in again["response"]["results"]],
            ["bob"])

    def test_unknown_request_is_404(self) -> None:
        self._seed()
        with self.assertRaises(ServiceError) as caught:
            self.service.event_gc_batch_cleanup_request_get("nope")
        self.assertEqual(caught.exception.status_code, 404)
        self.assertEqual(caught.exception.field, "request_id")
        self.assertEqual(list(caught.exception.to_body()),
                         ["message", "field"])

    def test_list_commit_order_and_key_order(self) -> None:
        self._seed()
        body = self.service.event_gc_batch_cleanup_request_list(0, 100)
        self.assertEqual(list(body), ["requests", "next_after", "has_more"])
        self.assertEqual([item["request_id"] for item in body["requests"]],
                         ["r1", "r2", "r3"])
        for item in body["requests"]:
            self.assertEqual(list(item),
                             ["request_id", "device_ids", "after", "limit",
                              "status", "response"])
        self.assertEqual(body["next_after"], 3)
        self.assertFalse(body["has_more"])

    def test_list_empty(self) -> None:
        body = self.service.event_gc_batch_cleanup_request_list(0, 100)
        self.assertEqual(body, {"requests": [], "next_after": 0,
                                "has_more": False})

    def test_list_paging(self) -> None:
        self._seed()
        first = self.service.event_gc_batch_cleanup_request_list(0, 2)
        self.assertEqual([i["request_id"] for i in first["requests"]],
                         ["r1", "r2"])
        self.assertEqual(first["next_after"], 2)
        self.assertTrue(first["has_more"])
        second = self.service.event_gc_batch_cleanup_request_list(2, 2)
        self.assertEqual([i["request_id"] for i in second["requests"]],
                         ["r3"])
        self.assertEqual(second["next_after"], 3)
        self.assertFalse(second["has_more"])
        empty = self.service.event_gc_batch_cleanup_request_list(3, 2)
        self.assertEqual(empty["requests"], [])
        self.assertEqual(empty["next_after"], 3)
        self.assertFalse(empty["has_more"])

    def test_validation_in_service(self) -> None:
        for after in (-1, 2**63, 0.0, "0", True, False):
            with self.subTest(after=after):
                with self.assertRaises(ServiceError) as caught:
                    self.service.event_gc_batch_cleanup_request_list(
                        after, 100)
                self.assertEqual(caught.exception.field, "after")
        for limit in (0, 101, -1, 1.0, "1", True):
            with self.subTest(limit=limit):
                with self.assertRaises(ServiceError) as caught:
                    self.service.event_gc_batch_cleanup_request_list(
                        0, limit)
                self.assertEqual(caught.exception.field, "limit")


class RequestsHTTPTest(RequestsMixin, unittest.TestCase):
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

    # -- detail -----------------------------------------------------------

    def test_detail_over_http(self) -> None:
        self._seed()
        status, body, raw = self._request(path=PATH + "/r2")
        self.assertEqual(status, 200)
        self.assertEqual(list(body),
                         ["request_id", "device_ids", "after", "limit",
                          "status", "response"])
        self.assertEqual(body["request_id"], "r2")
        self.assertEqual(list(body["response"]),
                         ["mode", "results", "next_after", "has_more"])
        for earlier, later in (
                ('"request_id"', '"device_ids"'),
                ('"device_ids"', '"after"'),
                ('"after"', '"limit"'),
                ('"limit"', '"status"'),
                ('"status"', '"response"')):
            self.assertLess(raw.index(earlier), raw.index(later))

    def test_detail_unknown_is_404(self) -> None:
        status, body, _ = self._request(path=PATH + "/nope")
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "request_id")
        self.assertEqual(list(body), ["message", "field"])

    def test_detail_empty_id_is_404(self) -> None:
        status, body, _ = self._request(path=PATH + "/")
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "request_id")

    def test_detail_deeper_path_is_404(self) -> None:
        status, body, _ = self._request(path=PATH + "/r1/extra")
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "request_id")

    def test_detail_strict_percent_decode(self) -> None:
        self._commit("a/b", ["bob"])
        # A percent-encoded slash is part of the id and decodes to a/b.
        status, body, _ = self._request(path=PATH + "/a%2Fb")
        self.assertEqual(status, 200)
        self.assertEqual(body["request_id"], "a/b")
        for bad in ("/a%zz", "/a%2", "/a%", "/a%ff%ff"):
            with self.subTest(bad=bad):
                status, body, _ = self._request(path=PATH + bad)
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], "request_id")
                self.assertEqual(list(body), ["message", "field"])

    def test_detail_unicode_id_is_utf8_percent_decoded(self) -> None:
        request_id = "req-é-☃"
        self._commit(request_id, ["bob"])
        encoded = "".join(
            f"%{byte:02X}" for byte in request_id.encode("utf-8"))
        status, body, _ = self._request(path=PATH + "/" + encoded)
        self.assertEqual(status, 200)
        self.assertEqual(body["request_id"], request_id)

    def test_detail_query_rejected(self) -> None:
        self._seed()
        for query in ("?after=0", "?limit=1", "?foo", "?x=1"):
            with self.subTest(query=query):
                status, body, _ = self._request(path=PATH + "/r1" + query)
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], "query")
        # A bare trailing '?' carries no parameter and is accepted.
        status, _, _ = self._request(path=PATH + "/r1?")
        self.assertEqual(status, 200)

    def test_detail_nonempty_body_rejected(self) -> None:
        self._seed()
        status, body, raw = self._request(path=PATH + "/r1", raw="{}")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "request_body")
        self.assertEqual(list(body), ["message", "field"])

    # -- list -------------------------------------------------------------

    def test_list_over_http(self) -> None:
        self._seed()
        status, body, raw = self._request()
        self.assertEqual(status, 200)
        self.assertEqual(list(body), ["requests", "next_after", "has_more"])
        self.assertEqual([i["request_id"] for i in body["requests"]],
                         ["r1", "r2", "r3"])
        self.assertLess(raw.index('"requests"'), raw.index('"next_after"'))
        self.assertLess(raw.index('"next_after"'), raw.index('"has_more"'))

    def test_list_empty_collection(self) -> None:
        status, body, _ = self._request()
        self.assertEqual(status, 200)
        self.assertEqual(body, {"requests": [], "next_after": 0,
                                "has_more": False})

    def test_list_paging_query(self) -> None:
        self._seed()
        status, body, _ = self._request(path=PATH + "?after=1&limit=1")
        self.assertEqual(status, 200)
        self.assertEqual([i["request_id"] for i in body["requests"]],
                         ["r2"])
        self.assertEqual(body["next_after"], 2)
        self.assertTrue(body["has_more"])
        status, body, _ = self._request(path=PATH + "?after=9&limit=2")
        self.assertEqual(status, 200)
        self.assertEqual(body["requests"], [])
        self.assertEqual(body["next_after"], 9)
        self.assertFalse(body["has_more"])

    def test_list_query_validation(self) -> None:
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

    def test_list_boundary_values_accepted(self) -> None:
        self._seed()
        status, body, _ = self._request(
            path=PATH + "?after=9223372036854775807&limit=1")
        self.assertEqual(status, 200)
        self.assertEqual(body["requests"], [])
        self.assertEqual(body["next_after"], 2**63 - 1)
        status, body, _ = self._request(path=PATH + "?limit=100")
        self.assertEqual(status, 200)
        self.assertEqual(len(body["requests"]), 3)

    def test_list_trailing_question_mark_uses_defaults(self) -> None:
        self._seed()
        status, body, _ = self._request(path=PATH + "?")
        self.assertEqual(status, 200)
        self.assertEqual([i["request_id"] for i in body["requests"]],
                         ["r1", "r2", "r3"])

    def test_list_nonempty_body_rejected(self) -> None:
        status, body, _ = self._request(raw="{}")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "request_body")

    def test_list_post_is_404(self) -> None:
        status, _, _ = self._request(method="POST", raw="{}")
        self.assertEqual(status, 404)


class RequestsPersistenceTest(RequestsMixin, unittest.TestCase):
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

    def _anchor_durable(self):
        self._touch("anchor-lease", device="alice")

    def test_reads_write_nothing_and_advance_no_generation(self) -> None:
        self._op("J1")
        self._touch("c1", device="bob")
        self._expire_lease("c1", device="bob")
        self._anchor_durable()
        self._commit("r1", ["bob"])
        generation = self.state_store.commit_seq
        before = self._document()
        detail1 = self.service.event_gc_batch_cleanup_request_get("r1")
        list1 = self.service.event_gc_batch_cleanup_request_list(0, 100)
        detail_raw = json.dumps(detail1, separators=(",", ":"),
                                ensure_ascii=False)
        list_raw = json.dumps(list1, separators=(",", ":"),
                              ensure_ascii=False)
        # Repeated reads are byte-identical while nothing changes.
        detail2 = self.service.event_gc_batch_cleanup_request_get("r1")
        list2 = self.service.event_gc_batch_cleanup_request_list(0, 100)
        self.assertEqual(json.dumps(detail2, separators=(",", ":"),
                                    ensure_ascii=False), detail_raw)
        self.assertEqual(json.dumps(list2, separators=(",", ":"),
                                    ensure_ascii=False), list_raw)
        self.assertEqual(self.state_store.commit_seq, generation)
        self.assertEqual(self._document(), before)

    def test_records_survive_restart_in_commit_order(self) -> None:
        self._seed()
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        body = restarted.event_gc_batch_cleanup_request_list(0, 100)
        self.assertEqual([item["request_id"] for item in body["requests"]],
                         ["r1", "r2", "r3"])
        detail = restarted.event_gc_batch_cleanup_request_get("r2")
        self.assertEqual(detail["device_ids"], ["alice", "dave", "ghost"])
        self.assertEqual(detail["after"], 1)
        self.assertEqual(detail["limit"], 1)
        self.assertEqual(detail["status"], 200)
        self.assertEqual(list(detail["response"]),
                         ["mode", "results", "next_after", "has_more"])

    def test_preview_creates_no_audit_record(self) -> None:
        self._op("J1")
        self._touch("c1", device="bob")
        self._expire_lease("c1", device="bob")
        self.service.event_gc_cleanup_expired_batch({
            "mode": "preview", "device_ids": ["bob"],
            "after": 0, "limit": 100})
        body = self.service.event_gc_batch_cleanup_request_list(0, 100)
        self.assertEqual(body["requests"], [])
        self.assertEqual(
            self._document()["event_gc_batch_cleanup_requests"], [])

    def test_commit_without_request_id_creates_no_audit_record(self) -> None:
        self._op("J1")
        self._touch("c1", device="bob")
        self._expire_lease("c1", device="bob")
        self._anchor_durable()
        self.service.event_gc_cleanup_expired_batch({
            "mode": "commit", "device_ids": ["bob"],
            "after": 0, "limit": 100})
        body = self.service.event_gc_batch_cleanup_request_list(0, 100)
        self.assertEqual(body["requests"], [])


if __name__ == "__main__":
    unittest.main()
