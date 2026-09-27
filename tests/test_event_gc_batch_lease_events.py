"""Tests for the batch-cleanup audit lease lifecycle event stream.

``GET /v1/event-gc-batch/lease-events`` pages the cleanup audit claim
lease lifecycle events. The GET takes no request body (a non-empty
one, however framed, is 400/request_body) and only single-valued
``consumer_id``/``lease_id``/``after``/``limit`` query parameters
(defaults omitted/omitted/0/100): both ids are optional filters that
must be non-empty strings when given (repeated or empty -> 400 with
that field); ``after`` is a strict ASCII decimal integer in
0..2**63-1 and ``limit`` in 1..100; any other parameter is
400/query. Validation order is request body, other parameters,
consumer_id, lease_id, after, limit.

The stream is a single global chain across every lease (seq runs
consecutively from 1, not per consumer). A non-empty claim records
``claim``, each first renewal ``renew``, the first explicit
confirm/release the matching event, and a checkpoint/consume advance
that crosses still-unterminated leases appends one
``implicit_confirm`` per lease in lease creation order; replays,
failures, empty no-ops and mere expiry record nothing. The body keys
are ``events``, ``next_after`` and ``has_more`` in that order; each
event is ``seq``, ``lease_id``, ``consumer_id``, ``type`` in that
order. An empty page echoes ``after`` as ``next_after``; otherwise it
is the last event's seq. The lookup is read-only and stable across a
restart.
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
from e2ee_backend.persistence import attach_persistence, StateFileError

PATH = "/v1/event-gc-batch/lease-events"
PAST = "2020-01-01T00:00:00.000000+00:00"


class LeaseEventsMixin:
    def _build(self) -> None:
        self.service = DeviceService()
        self.service.store.add_device(Device("u", "bob", "ik"))

    def _commit(self, request_id):
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

    def _consume(self, consumer, expected, limit):
        return self.service.event_gc_batch_consume({
            "consumer_id": consumer, "expected": expected,
            "limit": limit})

    def _page(self, consumer_id=None, lease_id=None, after=0, limit=100):
        return self.service.event_gc_batch_lease_events(
            consumer_id, lease_id, after, limit)

    def _types(self, **kwargs):
        return [event["type"]
                for event in self._page(**kwargs)["events"]]

    def _tuples(self, **kwargs):
        return [(event["seq"], event["lease_id"], event["type"])
                for event in self._page(**kwargs)["events"]]


class LeaseEventsServiceTest(LeaseEventsMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def test_empty_page_shape(self) -> None:
        self.assertEqual(self._page(), {
            "events": [], "next_after": 0, "has_more": False})

    def test_claim_renew_release_chain(self) -> None:
        self._commit("r1")
        self.assertEqual(self._claim("c1", "L1")[1], 201)
        self.assertEqual(self._renew("c1", "L1", "rn1")[1], 201)
        self.assertEqual(self._renew("c1", "L1", "rn1")[1], 200)
        self.assertEqual(self._op("c1", "L1", "release")[1], 201)
        self.assertEqual(self._op("c1", "L1", "release")[1], 200)
        body = self._page()
        self.assertEqual(self._types(), ["claim", "renew", "release"])
        self.assertEqual([event["seq"] for event in body["events"]],
                         [1, 2, 3])
        for event in body["events"]:
            self.assertEqual(
                list(event), ["seq", "lease_id", "consumer_id", "type"])
            self.assertEqual(event["lease_id"], "L1")
            self.assertEqual(event["consumer_id"], "c1")
        self.assertEqual(list(body), ["events", "next_after", "has_more"])

    def test_explicit_confirm_records_confirm(self) -> None:
        self._commit("r1")
        self._claim("c1", "L1")
        self.assertEqual(self._op("c1", "L1", "confirm")[1], 201)
        # The replay writes nothing.
        self.assertEqual(self._op("c1", "L1", "confirm")[1], 200)
        self.assertEqual(self._types(), ["claim", "confirm"])

    def test_empty_claim_records_nothing_and_id_stays_free(self) -> None:
        self._commit("r1")
        self._advance("c9", 0, 1)
        body, status = self._claim("c9", "LE", expected=1, limit=10)
        self.assertEqual(status, 200)
        self.assertEqual(body["records"], [])
        self.assertEqual(self._page(), {
            "events": [], "next_after": 0, "has_more": False})
        # The id was never occupied: the same id then claims normally.
        self._advance("c9", 1, 1)  # equal no-op
        self._commit("r2")
        claim, status = self._claim("c9", "LE", expected=1, limit=1)
        self.assertEqual(status, 201)
        self.assertEqual(self._types(lease_id="LE"), ["claim"])

    def test_checkpoint_implicit_confirms_in_creation_order(self) -> None:
        for rid in ("r1", "r2", "r3"):
            self._commit(rid)
        # L1 released (explicit release event, never implicit).
        self._claim("c1", "L1", limit=1)
        self._op("c1", "L1", "release")
        # L2 spans 0..3 for the same consumer; L3 spans 0..1 for c2.
        self._claim("c1", "L2", limit=100)
        self._claim("c2", "L3", limit=1)
        # An expired already-acknowledged-again lease for c1 is also
        # crossed below but its expiry records nothing.
        self._advance("c1", 0, 3)
        # L2 is crossed while still unterminated -> implicit_confirm;
        # L1 was released and records no second terminal event.
        self.assertEqual(
            self._types(consumer_id="c1"),
            ["claim", "release", "claim", "implicit_confirm"])
        # c2's lease is untouched by c1's advance.
        self.assertEqual(self._types(consumer_id="c2"), ["claim"])
        self._advance("c2", 0, 1)
        self.assertEqual(
            self._types(consumer_id="c2"),
            ["claim", "implicit_confirm"])

    def test_consume_implicit_confirms_in_creation_order(self) -> None:
        self._commit("r1")
        self._claim("c1", "L1", limit=1)
        # Consume 0..1 crosses L1.
        body, status = self._consume("c1", 0, 1)
        self.assertEqual(status, 201)
        self.assertEqual(self._types(), ["claim", "implicit_confirm"])
        # An empty consume is a 200 no-op recording nothing.
        body, status = self._consume("c1", 1, 10)
        self.assertEqual(status, 200)
        self.assertEqual(self._types(), ["claim", "implicit_confirm"])

    def test_explicit_confirm_mixes_with_other_implicit_confirms(self):
        for rid in ("r1", "r2", "r3"):
            self._commit(rid)
        # Two leases for c1 in creation order: A 0..1, B 0..3. A must
        # be out of the way before B can be claimed; release A so the
        # later advance leaves it alone.
        self._claim("c1", "A", limit=1)
        self._op("c1", "A", "release")
        self._claim("c1", "B", limit=100)
        # Confirm B explicitly: it advances 0->3 and records confirm
        # for B; A stays at its release.
        self.assertEqual(self._op("c1", "B", "confirm")[1], 201)
        self.assertEqual(self._types(consumer_id="c1"),
                         ["claim", "release", "claim", "confirm"])

    def test_expiry_records_nothing(self) -> None:
        self._commit("r1")
        self._claim("c1", "L1")
        self.service.store._cleanup_leases["L1"].expires = PAST
        # Either op on an expired lease conflicts and records nothing.
        with self.assertRaises(ServiceError):
            self._op("c1", "L1", "confirm")
        with self.assertRaises(ServiceError):
            self._op("c1", "L1", "release")
        with self.assertRaises(ServiceError):
            self._renew("c1", "L1", "rn1")
        self.assertEqual(self._types(), ["claim"])

    def test_failed_claim_records_nothing(self) -> None:
        self._commit("r1")
        # Wrong expected conflicts and records nothing.
        with self.assertRaises(ServiceError):
            self._claim("c1", "L1", expected=5)
        self.assertEqual(self._page()["events"], [])
        self._claim("c1", "L1", limit=1)
        # A busy second claim conflicts and records nothing.
        with self.assertRaises(ServiceError):
            self._claim("c1", "L2", expected=0)
        self.assertEqual(self._types(), ["claim"])

    def test_global_chain_is_consecutive_across_consumers(self) -> None:
        for rid in ("r1", "r2"):
            self._commit(rid)
        self._claim("c1", "L1", limit=1)
        self._claim("c2", "L2", limit=1)
        self._renew("c1", "L1", "rn")
        self._op("c2", "L2", "release")
        body = self._page()
        self.assertEqual([event["seq"] for event in body["events"]],
                         [1, 2, 3, 4])
        self.assertEqual(
            [(event["lease_id"], event["type"])
             for event in body["events"]],
            [("L1", "claim"), ("L2", "claim"), ("L1", "renew"),
             ("L2", "release")])

    def test_filters(self) -> None:
        for rid in ("r1", "r2"):
            self._commit(rid)
        self._claim("c1", "L1", limit=1)
        self._claim("c2", "L2", limit=1)
        self._op("c1", "L1", "release")
        self.assertEqual(
            [event["lease_id"] for event in
             self._page(consumer_id="c1")["events"]], ["L1", "L1"])
        self.assertEqual(self._types(lease_id="L1"),
                         ["claim", "release"])
        self.assertEqual(self._types(lease_id="nope"), [])
        self.assertEqual(self._types(consumer_id="nobody"), [])
        # Both filters together.
        self.assertEqual(
            self._types(consumer_id="c1", lease_id="L2"), [])
        self.assertEqual(
            self._types(consumer_id="c1", lease_id="L1"),
            ["claim", "release"])

    def test_paging_by_seq_with_and_without_filters(self) -> None:
        for rid in ("r1", "r2", "r3"):
            self._commit(rid)
        self._claim("c1", "L1", limit=1)
        self._op("c1", "L1", "release")
        self._claim("c2", "L2", limit=1)
        self._claim("c1", "L3", limit=100)
        self._advance("c1", 0, 3)
        # Full stream: claim L1, release L1, claim L2, claim L3,
        # implicit_confirm L3.
        self.assertEqual(
            [event["seq"] for event in self._page(limit=2)["events"]],
            [1, 2])
        page2 = self._page(after=2, limit=2)
        self.assertEqual(
            [event["seq"] for event in page2["events"]], [3, 4])
        self.assertEqual(page2["next_after"], 4)
        self.assertTrue(page2["has_more"])
        page3 = self._page(after=4, limit=2)
        self.assertEqual(
            [event["seq"] for event in page3["events"]], [5])
        self.assertEqual(page3["next_after"], 5)
        self.assertFalse(page3["has_more"])
        empty = self._page(after=5, limit=2)
        self.assertEqual(empty, {
            "events": [], "next_after": 5, "has_more": False})
        # c1's filtered stream is seqs 1,2,4,5: the cursor stays on
        # global seq values.
        first = self._page(consumer_id="c1", limit=2)
        self.assertEqual(
            [event["seq"] for event in first["events"]], [1, 2])
        self.assertEqual(first["next_after"], 2)
        self.assertTrue(first["has_more"])
        second = self._page(consumer_id="c1", after=2, limit=2)
        self.assertEqual(
            [event["seq"] for event in second["events"]], [4, 5])
        self.assertEqual(second["next_after"], 5)
        self.assertFalse(second["has_more"])

    def test_service_validation(self) -> None:
        for value in ("", 1, True, [], {}):
            with self.subTest(consumer_id=value):
                with self.assertRaises(ServiceError) as caught:
                    self._page(consumer_id=value)
                self.assertEqual(caught.exception.status_code, 400)
                self.assertEqual(caught.exception.field, "consumer_id")
        for value in ("", 1, True, [], {}):
            with self.subTest(lease_id=value):
                with self.assertRaises(ServiceError) as caught:
                    self._page(lease_id=value)
                self.assertEqual(caught.exception.status_code, 400)
                self.assertEqual(caught.exception.field, "lease_id")
        for after in (-1, 2**63, 0.0, "0", True, False):
            with self.subTest(after=after):
                with self.assertRaises(ServiceError) as caught:
                    self._page(after=after)
                self.assertEqual(caught.exception.status_code, 400)
                self.assertEqual(caught.exception.field, "after")
        for limit in (0, 101, -1, 1.0, "1", True):
            with self.subTest(limit=limit):
                with self.assertRaises(ServiceError) as caught:
                    self._page(limit=limit)
                self.assertEqual(caught.exception.status_code, 400)
                self.assertEqual(caught.exception.field, "limit")


class LeaseEventsHTTPTest(LeaseEventsMixin, unittest.TestCase):
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

    def _seed(self):
        for rid in ("r1", "r2"):
            self._commit(rid)
        self._claim("c1", "L1", limit=1)
        self._renew("c1", "L1", "rn")
        self._claim("c2", "L2", limit=1)
        self._op("c1", "L1", "release")
        self._advance("c2", 0, 1)

    def test_empty_page_over_http(self) -> None:
        status, body, _ = self._request()
        self.assertEqual(status, 200)
        self.assertEqual(body, {"events": [], "next_after": 0,
                                "has_more": False})

    def test_page_over_http_with_key_order(self) -> None:
        self._seed()
        status, body, raw = self._request()
        self.assertEqual(status, 200)
        self.assertEqual(list(body), ["events", "next_after", "has_more"])
        self.assertEqual(
            [(event["lease_id"], event["type"])
             for event in body["events"]],
            [("L1", "claim"), ("L1", "renew"), ("L2", "claim"),
             ("L1", "release"), ("L2", "implicit_confirm")])
        for event in body["events"]:
            self.assertEqual(
                list(event),
                ["seq", "lease_id", "consumer_id", "type"])
        for earlier, later in (
                ('"events"', '"next_after"'),
                ('"next_after"', '"has_more"'),
                ('"seq"', '"lease_id"'),
                ('"lease_id"', '"consumer_id"'),
                ('"consumer_id"', '"type"')):
            self.assertLess(raw.index(earlier), raw.index(later))

    def test_filters_and_paging_query(self) -> None:
        self._seed()
        status, body, _ = self._request(path=PATH + "?consumer_id=c1")
        self.assertEqual(status, 200)
        self.assertEqual(
            [event["lease_id"] for event in body["events"]],
            ["L1", "L1", "L1"])
        status, body, _ = self._request(path=PATH + "?lease_id=L2")
        self.assertEqual(status, 200)
        self.assertEqual(
            [event["type"] for event in body["events"]],
            ["claim", "implicit_confirm"])
        status, body, _ = self._request(
            path=PATH + "?consumer_id=c1&lease_id=L1&after=1&limit=1")
        self.assertEqual(status, 200)
        self.assertEqual(
            [event["seq"] for event in body["events"]], [2])
        self.assertEqual(body["next_after"], 2)
        self.assertTrue(body["has_more"])

    def test_query_validation(self) -> None:
        good_after = str(2**63 - 1)
        cases = [
            ("consumer_id=", "consumer_id"),
            ("consumer_id=a&consumer_id=b", "consumer_id"),
            ("lease_id=", "lease_id"),
            ("lease_id=a&lease_id=b", "lease_id"),
            ("after=-1", "after"),
            ("after=01%20", "after"),
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
        # Unknown parameter beats the id/after/limit problems.
        status, body, _ = self._request(
            path=PATH + "?consumer_id=&foo=1")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "query")
        # consumer_id before lease_id before after before limit.
        status, body, _ = self._request(
            path=PATH + "?consumer_id=&lease_id=")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "consumer_id")
        status, body, _ = self._request(path=PATH + "?lease_id=&after=x")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "lease_id")
        status, body, _ = self._request(path=PATH + "?after=x&limit=0")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "after")

    def test_boundary_values_accepted(self) -> None:
        self._seed()
        status, body, _ = self._request(
            path=PATH + "?after=9223372036854775807&limit=1")
        self.assertEqual(status, 200)
        self.assertEqual(body["events"], [])
        self.assertEqual(body["next_after"], 2**63 - 1)
        status, body, _ = self._request(path=PATH + "?limit=100")
        self.assertEqual(status, 200)
        self.assertEqual(len(body["events"]), 5)

    def test_trailing_question_mark_uses_defaults(self) -> None:
        self._seed()
        status, body, _ = self._request(path=PATH + "?")
        self.assertEqual(status, 200)
        self.assertEqual(len(body["events"]), 5)

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
        status, body = self._raw_request(
            b"GET " + PATH.encode() + b" HTTP/1.1\r\n"
            b"Host: 127.0.0.1\r\n"
            b"Transfer-Encoding: chunked\r\n"
            b"\r\n"
            b"2\r\n{}\r\n0\r\n\r\n")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "request_body")

    def test_chunked_empty_body_accepted(self) -> None:
        status, body = self._raw_request(
            b"GET " + PATH.encode() + b" HTTP/1.1\r\n"
            b"Host: 127.0.0.1\r\n"
            b"Transfer-Encoding: chunked\r\n"
            b"\r\n"
            b"0\r\n\r\n")
        self.assertEqual(status, 200)
        self.assertEqual(body["events"], [])

    def test_body_checked_before_query(self) -> None:
        status, body, _ = self._request(path=PATH + "?foo=1", raw="{}")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "request_body")

    def test_post_is_404(self) -> None:
        status, _, _ = self._request(method="POST", raw="{}")
        self.assertEqual(status, 404)


class LeaseEventsPersistenceTest(LeaseEventsMixin, unittest.TestCase):
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

    def _seed(self):
        for rid in ("r1", "r2", "r3"):
            self._commit(rid)
        self._claim("c1", "L1", limit=1)
        self._renew("c1", "L1", "rn")
        self._op("c1", "L1", "confirm")
        self._claim("c2", "L2", limit=1)
        self._advance("c2", 0, 1)

    def test_section_follows_cleanup_leases(self) -> None:
        self._seed()
        keys = list(self._document())
        self.assertIn("cleanup_lease_events", keys)
        self.assertEqual(
            keys.index("cleanup_lease_events"),
            keys.index("cleanup_leases") + 1)
        events = self._document()["cleanup_lease_events"]
        self.assertEqual(
            [event["type"] for event in events],
            ["claim", "renew", "confirm", "claim", "implicit_confirm"])
        for event in events:
            self.assertEqual(
                list(event),
                ["seq", "lease_id", "consumer_id", "type"])

    def test_reads_write_nothing_and_advance_no_generation(self) -> None:
        self._seed()
        generation = self.state_store.commit_seq
        before = self._document()
        first = self._page()
        self._page(consumer_id="c1")
        self._page(lease_id="L1", after=2, limit=1)
        self.assertEqual(self.state_store.commit_seq, generation)
        self.assertEqual(self._document(), before)
        self.assertTrue(
            self.service.persistence_integrity()["consistent"])

    def test_page_stable_across_restart(self) -> None:
        self._seed()
        before = self._page()
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        after = restarted.event_gc_batch_lease_events(
            None, None, 0, 100)
        self.assertEqual(after, before)
        self.assertEqual(
            restarted.event_gc_batch_lease_events(
                "c1", None, 0, 100),
            self._page(consumer_id="c1"))
        self.assertTrue(
            restarted.persistence_integrity()["consistent"])

    def test_events_commit_and_roll_back_with_their_mutation(self) -> None:
        self._commit("r1")
        generation = self.state_store.commit_seq
        # A non-empty claim appends its claim event in the same
        # generation: exactly one commit_seq step.
        self._claim("c1", "L1", limit=1)
        self.assertEqual(self.state_store.commit_seq, generation + 1)
        self.assertEqual(self._types(), ["claim"])
        # A claim replay consumes no generation and adds no event.
        self._claim("c1", "L1", limit=1)
        self.assertEqual(self.state_store.commit_seq, generation + 1)
        self.assertEqual(self._types(), ["claim"])

    def test_legacy_document_without_section_loads_empty(self) -> None:
        self._seed()
        document = self._document()
        document.pop("cleanup_lease_events")
        document.pop("integrity_log_version", None)
        legacy_path = os.path.join(self.directory, "legacy.json")
        with open(legacy_path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        restarted = DeviceService()
        attach_persistence(restarted, legacy_path)  # must not raise
        self.assertEqual(
            restarted.event_gc_batch_lease_events(None, None, 0, 100),
            {"events": [], "next_after": 0, "has_more": False})
        self.assertTrue(
            restarted.persistence_integrity()["consistent"])

    def _assert_refuses_startup(self, mutate):
        document = self._document()
        mutate(document)
        document.pop("integrity_log_version", None)
        bad_path = os.path.join(
            self.directory,
            f"bad-{threading.get_ident()}-{len(os.listdir(self.directory))}.json")
        with open(bad_path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        with open(bad_path, "rb") as handle:
            original = handle.read()
        service = DeviceService()
        with self.assertRaises(StateFileError):
            attach_persistence(service, bad_path)
        with open(bad_path, "rb") as handle:
            self.assertEqual(handle.read(), original)

    def test_non_consecutive_seq_refuses_startup(self) -> None:
        self._seed()

        def mutate(document):
            document["cleanup_lease_events"][2]["seq"] = 99
        self._assert_refuses_startup(mutate)

    def test_bad_type_refuses_startup(self) -> None:
        self._seed()

        def mutate(document):
            document["cleanup_lease_events"][0]["type"] = "expire"
        self._assert_refuses_startup(mutate)

    def test_bad_key_order_refuses_startup(self) -> None:
        self._seed()

        def mutate(document):
            raw = document["cleanup_lease_events"][0]
            document["cleanup_lease_events"][0] = {
                "lease_id": raw["lease_id"], "seq": raw["seq"],
                "consumer_id": raw["consumer_id"], "type": raw["type"]}
        self._assert_refuses_startup(mutate)

    def test_unknown_lease_reference_refuses_startup(self) -> None:
        self._seed()

        def mutate(document):
            document["cleanup_lease_events"][0]["lease_id"] = "ghost"
        self._assert_refuses_startup(mutate)

    def test_consumer_mismatch_refuses_startup(self) -> None:
        self._seed()

        def mutate(document):
            document["cleanup_lease_events"][0]["consumer_id"] = "other"
        self._assert_refuses_startup(mutate)

    def test_terminal_event_without_marker_refuses_startup(self) -> None:
        self._seed()

        def mutate(document):
            # L1's confirm event requires terminal == "confirm".
            document["cleanup_leases"][0]["terminal"] = None
        self._assert_refuses_startup(mutate)

    def test_implicit_confirm_with_marker_refuses_startup(self) -> None:
        self._seed()

        def mutate(document):
            # L2 (index 1) carries implicit_confirm and must stay
            # marker-less.
            document["cleanup_leases"][1]["terminal"] = "confirm"
        self._assert_refuses_startup(mutate)

    def test_implicit_confirm_without_checkpoint_refuses_startup(self):
        self._seed()

        def mutate(document):
            # Drop c2's checkpoint below L2.next_after while the event
            # says it was implicitly confirmed.
            document["cleanup_checkpoints"] = []
        self._assert_refuses_startup(mutate)


if __name__ == "__main__":
    unittest.main()
