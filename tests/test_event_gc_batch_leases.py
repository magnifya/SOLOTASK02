"""Tests for the batch-cleanup audit lease page.

``GET /v1/event-gc-batch/leases`` pages the cleanup audit claim leases
in creation order. The GET takes no request body (non-empty ->
400/request_body) and only single-valued ``consumer_id``/``after``/
``limit`` query parameters: ``consumer_id`` omitted lists every
consumer's leases and must otherwise be a non-empty string; ``after``
defaults to 0 (strict ASCII decimal in 0..2**63-1) and ``limit``
defaults to 100 (1..100); a repeated, empty or malformed parameter is
400 with that parameter name as field, and any other parameter is
400/query.

The body keys are ``leases``, ``next_after`` and ``has_more`` in that
order; each item's keys are ``lease_id``, ``consumer_id``, ``expected``,
``next_after``, ``expires``, ``effective_expires``, ``renewal_count``
and ``state`` in that order. ``state`` is decided at the lock-held
instant: released (terminal release), confirmed (terminal confirm or
the consumer's checkpoint reached ``next_after``), expired (the
effective deadline — the last renewal's value, the claim ``expires``
when never renewed — is at or before now), otherwise active. The lookup
is read-only: no persistence, no ``commit_seq`` advance, byte-identical
while state is unchanged.
"""
import json
import os
import shutil
import tempfile
import threading
import unittest
from http.client import HTTPConnection

from e2ee_backend.models import Device
from e2ee_backend.service import DeviceService, ServiceError
from e2ee_backend.http_app import create_server
from e2ee_backend.persistence import attach_persistence

PATH = "/v1/event-gc-batch/leases"
PAST = "2000-01-01T00:00:00.000000+00:00"


class LeasesMixin:
    def _build(self) -> None:
        self.service = DeviceService()
        self.service.store.add_device(Device("u", "alice", "ik"))
        self.service.store.add_device(Device("u", "bob", "ik"))

    def _commit(self, request_id, device_ids=None):
        # Every idempotent commit appends exactly one audit record.
        return self.service.event_gc_cleanup_expired_batch({
            "mode": "commit", "device_ids": device_ids or ["bob"],
            "after": 0, "limit": 100, "request_id": request_id})

    def _claim(self, consumer, lease_id, expected=0, limit=100):
        return self.service.event_gc_batch_claim({
            "consumer_id": consumer, "lease_id": lease_id,
            "expected": expected, "limit": limit})

    def _renew(self, consumer, lease_id, renewal_id):
        return self.service.event_gc_batch_lease_renew({
            "consumer_id": consumer, "lease_id": lease_id,
            "renewal_id": renewal_id})

    def _op(self, consumer, lease_id, expected, op):
        return self.service.event_gc_batch_lease_op({
            "consumer_id": consumer, "lease_id": lease_id,
            "expected": expected, "op": op})

    def _checkpoint(self, consumer, expected, after):
        return self.service.event_gc_batch_checkpoint({
            "consumer_id": consumer, "expected": expected, "after": after})

    def _page(self, consumer_id=None, after=0, limit=100):
        return self.service.event_gc_batch_leases(consumer_id, after,
                                                  limit)

    def _seed(self):
        # Three audit records; leases created in order L1, L2, L3.
        self._commit("r1")
        self._commit("r2")
        self._commit("r3")
        _, status = self._claim("c1", "L1", expected=0, limit=2)
        self.assertEqual(status, 201)
        _, status = self._claim("c2", "L2", expected=0, limit=1)
        self.assertEqual(status, 201)
        _, status = self._claim("c3", "L3", expected=0, limit=3)
        self.assertEqual(status, 201)


class LeasesServiceTest(LeasesMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def test_empty_page_shape(self) -> None:
        body = self._page()
        self.assertEqual(list(body), ["leases", "next_after", "has_more"])
        self.assertEqual(body, {"leases": [], "next_after": 0,
                                "has_more": False})

    def test_creation_order_and_item_keys(self) -> None:
        self._seed()
        body = self._page()
        self.assertEqual([i["lease_id"] for i in body["leases"]],
                         ["L1", "L2", "L3"])
        for item in body["leases"]:
            self.assertEqual(list(item),
                             ["lease_id", "consumer_id", "expected",
                              "next_after", "expires", "effective_expires",
                              "renewal_count", "state"])
            self.assertTrue(item["expires"].endswith("+00:00"))
            self.assertTrue(item["effective_expires"].endswith("+00:00"))
            self.assertEqual(item["state"], "active")
            self.assertEqual(item["renewal_count"], 0)
            self.assertEqual(item["effective_expires"], item["expires"])
        self.assertEqual([i["next_after"] for i in body["leases"]],
                         [2, 1, 3])
        self.assertEqual(body["next_after"], 3)
        self.assertFalse(body["has_more"])

    def test_consumer_filter(self) -> None:
        self._seed()
        body = self._page(consumer_id="c2")
        self.assertEqual([i["lease_id"] for i in body["leases"]], ["L2"])
        self.assertEqual(body["next_after"], 1)
        self.assertFalse(body["has_more"])
        body = self._page(consumer_id="nobody")
        self.assertEqual(body, {"leases": [], "next_after": 0,
                                "has_more": False})

    def test_paging(self) -> None:
        self._seed()
        first = self._page(after=0, limit=2)
        self.assertEqual([i["lease_id"] for i in first["leases"]],
                         ["L1", "L2"])
        self.assertEqual(first["next_after"], 2)
        self.assertTrue(first["has_more"])
        second = self._page(after=2, limit=2)
        self.assertEqual([i["lease_id"] for i in second["leases"]],
                         ["L3"])
        self.assertEqual(second["next_after"], 3)
        self.assertFalse(second["has_more"])
        empty = self._page(after=3, limit=2)
        self.assertEqual(empty["leases"], [])
        self.assertEqual(empty["next_after"], 3)
        self.assertFalse(empty["has_more"])

    def test_state_confirmed_by_terminal_and_checkpoint(self) -> None:
        self._seed()
        # Explicit confirm: terminal confirm wins over everything.
        _, status = self._op("c1", "L1", 0, "confirm")
        self.assertEqual(status, 201)
        # Implicit confirm: advancing c2's checkpoint to L2's next_after.
        _, status = self._checkpoint("c2", 0, 1)
        self.assertEqual(status, 201)
        body = self._page()
        states = {i["lease_id"]: i["state"] for i in body["leases"]}
        self.assertEqual(states, {"L1": "confirmed", "L2": "confirmed",
                                  "L3": "active"})

    def test_state_released_beats_checkpoint(self) -> None:
        self._seed()
        _, status = self._op("c3", "L3", 0, "release")
        self.assertEqual(status, 201)
        # Even after the checkpoint passes next_after, release stays
        # released (terminal release is decided first).
        _, status = self._checkpoint("c3", 0, 3)
        self.assertEqual(status, 201)
        body = self._page(consumer_id="c3")
        self.assertEqual(body["leases"][0]["state"], "released")

    def test_state_expired_uses_effective_deadline(self) -> None:
        self._seed()
        # Expire the claim deadline but renew past it: still active.
        self.service.store._cleanup_leases["L1"].expires = PAST
        with self.assertRaises(ServiceError) as caught:
            self._renew("c1", "L1", "rn1")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")
        # Renew first, then age out only the claim deadline.
        _, status = self._renew("c2", "L2", "rn1")
        self.assertEqual(status, 201)
        self.service.store._cleanup_leases["L2"].expires = PAST
        body = self._page()
        states = {i["lease_id"]: i["state"] for i in body["leases"]}
        self.assertEqual(states["L1"], "expired")
        self.assertEqual(states["L2"], "active")
        item = next(i for i in body["leases"] if i["lease_id"] == "L2")
        self.assertEqual(item["renewal_count"], 1)
        self.assertEqual(item["expires"], PAST)
        self.assertNotEqual(item["effective_expires"], PAST)
        # Aging the effective (renewal) deadline too flips it to expired.
        self.service.store._cleanup_leases["L2"].renewals[-1] \
            .expires = PAST
        body = self._page(consumer_id="c2")
        self.assertEqual(body["leases"][0]["state"], "expired")

    def test_renewal_count_and_effective_expires(self) -> None:
        self._seed()
        _, status = self._renew("c2", "L2", "rn1")
        self.assertEqual(status, 201)
        _, status = self._renew("c2", "L2", "rn2")
        self.assertEqual(status, 201)
        body = self._page(consumer_id="c2")
        item = body["leases"][0]
        self.assertEqual(item["renewal_count"], 2)
        self.assertEqual(item["effective_expires"],
                         self.service.store._cleanup_leases["L2"]
                         .renewals[-1].expires)

    def test_read_only(self) -> None:
        self._seed()
        first = self._page()
        second = self._page()
        self.assertEqual(first, second)

    def test_validation_in_service(self) -> None:
        for consumer_id in ("", 1, True, [], {}):
            with self.subTest(consumer_id=consumer_id):
                with self.assertRaises(ServiceError) as caught:
                    self.service.event_gc_batch_leases(consumer_id, 0, 100)
                self.assertEqual(caught.exception.status_code, 400)
                self.assertEqual(caught.exception.field, "consumer_id")
        for after in (-1, 2**63, 0.0, "0", True, False):
            with self.subTest(after=after):
                with self.assertRaises(ServiceError) as caught:
                    self.service.event_gc_batch_leases(None, after, 100)
                self.assertEqual(caught.exception.status_code, 400)
                self.assertEqual(caught.exception.field, "after")
        for limit in (0, 101, -1, 1.0, "1", True):
            with self.subTest(limit=limit):
                with self.assertRaises(ServiceError) as caught:
                    self.service.event_gc_batch_leases(None, 0, limit)
                self.assertEqual(caught.exception.status_code, 400)
                self.assertEqual(caught.exception.field, "limit")


class LeasesHTTPTest(LeasesMixin, unittest.TestCase):
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

    def test_page_over_http_with_key_order(self) -> None:
        self._seed()
        status, body, raw = self._request()
        self.assertEqual(status, 200)
        self.assertEqual(list(body), ["leases", "next_after", "has_more"])
        self.assertEqual([i["lease_id"] for i in body["leases"]],
                         ["L1", "L2", "L3"])
        for earlier, later in (
                ('"leases"', '"next_after"'),
                ('"next_after"', '"has_more"'),
                ('"lease_id"', '"consumer_id"'),
                ('"consumer_id"', '"expected"'),
                ('"expected"', '"next_after"'),
                ('"next_after"', '"expires"'),
                ('"expires"', '"effective_expires"'),
                ('"effective_expires"', '"renewal_count"'),
                ('"renewal_count"', '"state"')):
            self.assertLess(raw.index(earlier), raw.index(later))

    def test_empty_page_over_http(self) -> None:
        status, body, _ = self._request()
        self.assertEqual(status, 200)
        self.assertEqual(body, {"leases": [], "next_after": 0,
                                "has_more": False})

    def test_consumer_filter_and_paging_over_http(self) -> None:
        self._seed()
        status, body, _ = self._request(path=PATH + "?consumer_id=c1")
        self.assertEqual(status, 200)
        self.assertEqual([i["lease_id"] for i in body["leases"]], ["L1"])
        status, body, _ = self._request(path=PATH + "?after=1&limit=1")
        self.assertEqual(status, 200)
        self.assertEqual([i["lease_id"] for i in body["leases"]], ["L2"])
        self.assertEqual(body["next_after"], 2)
        self.assertTrue(body["has_more"])
        status, body, _ = self._request(
            path=PATH + "?consumer_id=c1&after=5")
        self.assertEqual(status, 200)
        self.assertEqual(body["leases"], [])
        self.assertEqual(body["next_after"], 5)
        self.assertFalse(body["has_more"])

    def test_query_validation(self) -> None:
        good_after = str(2**63 - 1)
        cases = [
            ("consumer_id=", "consumer_id"),
            ("consumer_id=a&consumer_id=b", "consumer_id"),
            ("after=-1", "after"),
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
            ("foo", "query"),
            ("consumer_id=c1&mode=preview", "query"),
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
        self.assertEqual(body["leases"], [])
        self.assertEqual(body["next_after"], 2**63 - 1)
        status, body, _ = self._request(path=PATH + "?limit=100")
        self.assertEqual(status, 200)
        self.assertEqual(len(body["leases"]), 3)

    def test_trailing_question_mark_uses_defaults(self) -> None:
        self._seed()
        status, body, _ = self._request(path=PATH + "?")
        self.assertEqual(status, 200)
        self.assertEqual(len(body["leases"]), 3)

    def test_request_body_rejected(self) -> None:
        for raw in ("{}", "null", "[]", "x"):
            with self.subTest(raw=raw):
                status, body, _ = self._request(raw=raw)
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], "request_body")
                self.assertEqual(list(body), ["message", "field"])

    def test_body_check_precedes_query_check(self) -> None:
        status, body, _ = self._request(path=PATH + "?foo=1", raw="{}")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "request_body")

    def test_post_is_404(self) -> None:
        status, _, _ = self._request(method="POST", raw="{}")
        self.assertEqual(status, 404)


class LeasesPersistenceTest(LeasesMixin, unittest.TestCase):
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

    def test_reads_write_nothing_and_advance_no_generation(self) -> None:
        self._seed()
        generation = self.state_store.commit_seq
        before = self._document()
        first = self._page()
        first_raw = json.dumps(first, separators=(",", ":"),
                               ensure_ascii=False)
        # Repeated reads are byte-identical while nothing changes.
        second = self._page()
        self.assertEqual(json.dumps(second, separators=(",", ":"),
                                    ensure_ascii=False), first_raw)
        self.assertEqual(self.state_store.commit_seq, generation)
        self.assertEqual(self._document(), before)

    def test_page_stable_across_restart_in_creation_order(self) -> None:
        self._seed()
        before = self._page()
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        after = restarted.event_gc_batch_leases(None, 0, 100)
        self.assertEqual(after, before)
        self.assertEqual([i["lease_id"] for i in after["leases"]],
                         ["L1", "L2", "L3"])

    def test_no_new_document_section(self) -> None:
        self._seed()
        keys_before = list(self._document())
        self._page()
        self.assertEqual(list(self._document()), keys_before)


if __name__ == "__main__":
    unittest.main()
