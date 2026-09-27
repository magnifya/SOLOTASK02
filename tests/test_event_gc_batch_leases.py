"""Tests for the batch-cleanup audit lease page.

``GET /v1/event-gc-batch/leases`` pages the cleanup audit claim leases.
The GET takes no request body (a non-empty one, however framed, is
400/request_body) and only single-valued ``consumer_id``/``after``/
``limit`` query parameters (defaults omitted/0/100): ``consumer_id``
omitted lists every consumer, else it must be a non-empty string
(repeated or empty -> 400/consumer_id); ``after`` is a strict ASCII
decimal integer in 0..2**63-1 and ``limit`` in 1..100 (a repeated,
empty or malformed value is 400 with that parameter name), and any
other parameter is 400/query. Validation order is request body, other
parameters, consumer_id, after, limit.

Leases are filtered by consumer in creation order and then paged by
the zero-based ``after`` offset; the state is decided at one query
instant in the order released (terminal == release), confirmed
(terminal == confirm or the consumer checkpoint reached next_after),
expired (effective deadline at or before now), otherwise active. The
body keys are ``leases``, ``next_after`` and ``has_more`` in that
order; each item's keys are ``lease_id``, ``consumer_id``,
``expected``, ``next_after``, ``expires``, ``effective_expires``,
``renewal_count`` and ``state`` in that order, with every timestamp a
UTC string carrying six microsecond digits and ``+00:00``. The lookup
is read-only: no persistence, no ``commit_seq`` advance,
byte-identical while state is unchanged and stable across a restart.
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
PAST = "2020-01-01T00:00:00.000000+00:00"


class LeasesMixin:
    def _build(self) -> None:
        self.service = DeviceService()
        self.service.store.add_device(Device("u", "alice", "ik"))
        self.service.store.add_device(Device("u", "bob", "ik"))

    def _commit(self, request_id):
        # Each idempotent commit appends exactly one audit record.
        return self.service.event_gc_cleanup_expired_batch({
            "mode": "commit", "device_ids": ["bob"],
            "after": 0, "limit": 100, "request_id": request_id})

    def _claim(self, consumer, lease_id, expected=0, limit=1):
        return self.service.event_gc_batch_claim({
            "consumer_id": consumer, "lease_id": lease_id,
            "expected": expected, "limit": limit})

    def _op(self, consumer, lease_id, op, expected=0):
        return self.service.event_gc_batch_lease_op({
            "consumer_id": consumer, "lease_id": lease_id,
            "expected": expected, "op": op})

    def _renew(self, consumer, lease_id, renewal_id):
        return self.service.event_gc_batch_lease_renew({
            "consumer_id": consumer, "lease_id": lease_id,
            "renewal_id": renewal_id})

    def _advance(self, consumer, expected, after):
        return self.service.event_gc_batch_checkpoint({
            "consumer_id": consumer, "expected": expected,
            "after": after})

    def _page(self, consumer_id=None, after=0, limit=100):
        return self.service.event_gc_batch_leases(
            consumer_id, after, limit)

    def _seed(self):
        # Three audit records; five leases committed in the order
        # L1..L5 across three consumers:
        #   L1 (c1) released, L2 (c2) active with one renewal,
        #   L3 (c1) expired, L4 (c3) confirmed via explicit confirm,
        #   L5 (c1) confirmed implicitly once c1's checkpoint reaches
        #   its next_after (no terminal marker).
        self._commit("r1")
        self._commit("r2")
        self._commit("r3")
        self._claim("c1", "L1")
        self._op("c1", "L1", "release")
        self._claim("c2", "L2", limit=2)
        self._renew("c2", "L2", "renew-1")
        self._claim("c1", "L3", limit=100)
        self.service.store._cleanup_leases["L3"].expires = PAST
        self._claim("c3", "L4")
        self._op("c3", "L4", "confirm")
        self._claim("c1", "L5")
        # Moving c1's checkpoint to 1 implicitly acknowledges L5 (and
        # L1, whose release must still win the state classification).
        self._advance("c1", 0, 1)


class LeasesServiceTest(LeasesMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def test_empty_page_shape(self) -> None:
        body = self._page()
        self.assertEqual(list(body), ["leases", "next_after", "has_more"])
        self.assertEqual(body, {"leases": [], "next_after": 0,
                                "has_more": False})

    def test_creation_order_item_keys_and_states(self) -> None:
        self._seed()
        body = self._page()
        self.assertEqual([i["lease_id"] for i in body["leases"]],
                         ["L1", "L2", "L3", "L4", "L5"])
        self.assertEqual([i["consumer_id"] for i in body["leases"]],
                         ["c1", "c2", "c1", "c3", "c1"])
        self.assertEqual([i["state"] for i in body["leases"]],
                         ["released", "active", "expired", "confirmed",
                          "confirmed"])
        for item in body["leases"]:
            self.assertEqual(list(item), [
                "lease_id", "consumer_id", "expected", "next_after",
                "expires", "effective_expires", "renewal_count", "state"])
            self.assertEqual(item["expected"], 0)
            for name in ("expires", "effective_expires"):
                text = item[name]
                self.assertTrue(text.endswith("+00:00"), text)
                fractional = text.split(".", 1)[1]
                self.assertEqual(len(fractional), 12)  # 6 digits + +00:00
                self.assertTrue(fractional[:6].isdigit())
                self.assertEqual(fractional[6:], "+00:00")

    def test_page_fields(self) -> None:
        self._seed()
        items = {item["lease_id"]: item for item in self._page()["leases"]}
        l2 = items["L2"]
        self.assertEqual(l2["next_after"], 2)
        self.assertEqual(l2["renewal_count"], 1)
        # The claim deadline stays frozen; the effective deadline is
        # the last renewal's value.
        self.assertNotEqual(l2["effective_expires"], l2["expires"])
        self.assertEqual(l2["effective_expires"],
                         self.service.store._cleanup_leases["L2"]
                         .renewals[-1].expires)
        l1 = items["L1"]
        self.assertEqual(l1["next_after"], 1)
        self.assertEqual(l1["renewal_count"], 0)
        self.assertEqual(l1["effective_expires"], l1["expires"])
        l3 = items["L3"]
        self.assertEqual(l3["expires"], PAST)
        self.assertEqual(l3["effective_expires"], PAST)

    def test_released_wins_over_acknowledgement(self) -> None:
        self._seed()
        # c1's checkpoint (1) reaches L1's next_after too, but the
        # explicit release takes precedence in the state order.
        l1 = next(item for item in self._page()["leases"]
                  if item["lease_id"] == "L1")
        self.assertEqual(l1["state"], "released")

    def test_implicit_confirm_has_no_terminal_marker(self) -> None:
        self._seed()
        self.assertIsNone(
            self.service.store._cleanup_leases["L5"].terminal)

    def test_consumer_filter_in_creation_order(self) -> None:
        self._seed()
        body = self._page(consumer_id="c1")
        self.assertEqual([i["lease_id"] for i in body["leases"]],
                         ["L1", "L3", "L5"])
        self.assertTrue(all(i["consumer_id"] == "c1"
                            for i in body["leases"]))
        self.assertEqual(body["next_after"], 3)
        self.assertFalse(body["has_more"])
        self.assertEqual(
            [i["lease_id"] for i in
             self._page(consumer_id="c2")["leases"]], ["L2"])
        self.assertEqual(self._page(consumer_id="nobody")["leases"], [])

    def test_paging(self) -> None:
        self._seed()
        first = self._page(after=0, limit=2)
        self.assertEqual([i["lease_id"] for i in first["leases"]],
                         ["L1", "L2"])
        self.assertEqual(first["next_after"], 2)
        self.assertTrue(first["has_more"])
        second = self._page(after=2, limit=2)
        self.assertEqual([i["lease_id"] for i in second["leases"]],
                         ["L3", "L4"])
        self.assertEqual(second["next_after"], 4)
        self.assertTrue(second["has_more"])
        third = self._page(after=4, limit=2)
        self.assertEqual([i["lease_id"] for i in third["leases"]], ["L5"])
        self.assertEqual(third["next_after"], 5)
        self.assertFalse(third["has_more"])
        empty = self._page(after=5, limit=2)
        self.assertEqual(empty["leases"], [])
        self.assertEqual(empty["next_after"], 5)
        self.assertFalse(empty["has_more"])

    def test_paging_after_filter(self) -> None:
        self._seed()
        body = self._page(consumer_id="c1", after=1, limit=1)
        self.assertEqual([i["lease_id"] for i in body["leases"]], ["L3"])
        self.assertEqual(body["next_after"], 2)
        self.assertTrue(body["has_more"])

    def test_active_state_for_fresh_lease(self) -> None:
        self._commit("r1")
        self._claim("c1", "L1")
        item = self._page()["leases"][0]
        self.assertEqual(item["state"], "active")
        self.assertEqual(item["renewal_count"], 0)

    def test_validation_in_service(self) -> None:
        for consumer_id in ("", 1, True, [], {}):
            with self.subTest(consumer_id=consumer_id):
                with self.assertRaises(ServiceError) as caught:
                    self.service.event_gc_batch_leases(
                        consumer_id, 0, 100)
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
        # None consumer_id is the "list every consumer" query, not 400.
        self.assertEqual(self.service.event_gc_batch_leases(None, 0, 100),
                         {"leases": [], "next_after": 0, "has_more": False})


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
                         ["L1", "L2", "L3", "L4", "L5"])
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

    def test_consumer_filter_query(self) -> None:
        self._seed()
        status, body, _ = self._request(path=PATH + "?consumer_id=c1")
        self.assertEqual(status, 200)
        self.assertEqual([i["lease_id"] for i in body["leases"]],
                         ["L1", "L3", "L5"])
        self.assertEqual(body["next_after"], 3)
        self.assertFalse(body["has_more"])
        status, body, _ = self._request(
            path=PATH + "?consumer_id=c1&after=1&limit=1")
        self.assertEqual(status, 200)
        self.assertEqual([i["lease_id"] for i in body["leases"]], ["L3"])
        self.assertEqual(body["next_after"], 2)
        self.assertTrue(body["has_more"])

    def test_paging_query(self) -> None:
        self._seed()
        status, body, _ = self._request(path=PATH + "?after=4&limit=2")
        self.assertEqual(status, 200)
        self.assertEqual([i["lease_id"] for i in body["leases"]], ["L5"])
        self.assertEqual(body["next_after"], 5)
        status, body, _ = self._request(path=PATH + "?after=9&limit=2")
        self.assertEqual(status, 200)
        self.assertEqual(body["leases"], [])
        self.assertEqual(body["next_after"], 9)
        self.assertFalse(body["has_more"])

    def test_query_validation(self) -> None:
        good_after = str(2**63 - 1)
        cases = [
            ("consumer_id=", "consumer_id"),
            ("consumer_id=a&consumer_id=b", "consumer_id"),
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
            ("foo", "query"),
            ("after=0&state=active", "query"),
        ]
        for query, field in cases:
            with self.subTest(query=query):
                status, body, _ = self._request(path=PATH + "?" + query)
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], field)
                self.assertEqual(list(body), ["message", "field"])

    def test_validation_order(self) -> None:
        # Unknown parameter beats consumer_id/after/limit problems.
        status, body, _ = self._request(
            path=PATH + "?consumer_id=&foo=1")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "query")
        status, body, _ = self._request(path=PATH + "?after=x&foo=1")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "query")
        # consumer_id is checked before after and limit.
        status, body, _ = self._request(
            path=PATH + "?consumer_id=&after=x")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "consumer_id")
        # after before limit.
        status, body, _ = self._request(path=PATH + "?after=x&limit=0")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "after")

    def test_boundary_values_accepted(self) -> None:
        self._seed()
        status, body, _ = self._request(
            path=PATH + "?after=9223372036854775807&limit=1")
        self.assertEqual(status, 200)
        self.assertEqual(body["leases"], [])
        self.assertEqual(body["next_after"], 2**63 - 1)
        status, body, _ = self._request(path=PATH + "?limit=100")
        self.assertEqual(status, 200)
        self.assertEqual(len(body["leases"]), 5)

    def test_trailing_question_mark_uses_defaults(self) -> None:
        self._seed()
        status, body, _ = self._request(path=PATH + "?")
        self.assertEqual(status, 200)
        self.assertEqual(len(body["leases"]), 5)

    def test_nonempty_body_rejected(self) -> None:
        status, body, _ = self._request(raw="{}")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "request_body")
        self.assertEqual(list(body), ["message", "field"])

    def _raw_request(self, request: bytes):
        import socket
        sock = socket.create_connection(("127.0.0.1", self.port), timeout=5)
        try:
            sock.sendall(request)
            chunks = []
            while True:
                data = sock.recv(4096)
                if not data:
                    break
                chunks.append(data)
        finally:
            sock.close()
        head, _, raw = b"".join(chunks).partition(b"\r\n\r\n")
        status = int(head.split(b" ", 2)[1])
        return status, (json.loads(raw.decode("utf-8")) if raw else None)

    def test_chunked_nonempty_body_rejected(self) -> None:
        # A legally framed (chunked) non-empty body is 400/request_body
        # just like a Content-Length framed one.
        status, body = self._raw_request(
            b"GET " + PATH.encode() + b" HTTP/1.1\r\n"
            b"Host: 127.0.0.1\r\n"
            b"Transfer-Encoding: chunked\r\n"
            b"\r\n"
            b"2\r\n{}\r\n0\r\n\r\n")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "request_body")
        self.assertEqual(list(body), ["message", "field"])

    def test_chunked_empty_body_accepted(self) -> None:
        status, body = self._raw_request(
            b"GET " + PATH.encode() + b" HTTP/1.1\r\n"
            b"Host: 127.0.0.1\r\n"
            b"Transfer-Encoding: chunked\r\n"
            b"\r\n"
            b"0\r\n\r\n")
        self.assertEqual(status, 200)
        self.assertEqual(body["leases"], [])

    def test_body_checked_before_query(self) -> None:
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
                         ["L1", "L2", "L3", "L4", "L5"])
        filtered = restarted.event_gc_batch_leases("c1", 0, 100)
        self.assertEqual([i["lease_id"] for i in filtered["leases"]],
                         ["L1", "L3", "L5"])

    def test_no_new_document_section(self) -> None:
        self._seed()
        keys_before = list(self._document())
        self._page()
        self.assertEqual(list(self._document()), keys_before)


if __name__ == "__main__":
    unittest.main()
