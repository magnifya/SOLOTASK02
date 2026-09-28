"""Tests for the batch-cleanup audit lease lifecycle event stream.

``GET /v1/event-gc-batch/lease-events`` pages the lease lifecycle
events. The GET takes no request body (a non-empty one, however
framed, is 400/request_body) and only single-valued
``consumer_id``/``lease_id``/``after``/``limit`` query parameters
(defaults omitted/omitted/0/100): the two ids are optional and, when
present, must be non-empty strings (repeated or empty -> 400 with that
field); ``after`` is a strict ASCII decimal integer in 0..2**63-1 and
``limit`` in 1..100 (a repeated, empty or malformed value is 400 with
that parameter name); any other parameter is 400/query. Validation
order is request body, other parameters, consumer_id, lease_id,
after, limit.

The events are globally ordered by ``seq`` (consecutive from 1) and
paged with the seq contract — the page is the matches with
``seq > after``, at most ``limit``; an empty page echoes
``next_after=after``, otherwise it is the last event's seq. The body
keys are ``events``, ``next_after`` and ``has_more``; each item's
keys are ``seq``, ``lease_id``, ``consumer_id`` and ``type``.

Recording rules: a non-empty claim appends ``claim``; a first renewal
appends ``renew``; a first explicit confirm/release appends that
type; a checkpoint/consume advance crossing an unterminated lease
appends ``implicit_confirm`` per crossed lease in lease creation
order. Replays, conflicts, empty-page claims, equal-value no-ops and
expirations append nothing. Events commit in the same locked
transaction (and commit_seq) as the change they record and roll back
with it; the query is read-only. The stream persists in the new
version=1 ``cleanup_lease_events`` section (missing loads as empty and
joins the integrity snapshot), with a contradicting document refused
at startup.
"""
import json
import os
import tempfile
import threading
import unittest
from http.client import HTTPConnection

from e2ee_backend.models import Device
from e2ee_backend.service import DeviceService, ServiceError
from e2ee_backend.http_app import create_server
from e2ee_backend.persistence import attach_persistence, PersistenceUnavailable

PATH = "/v1/event-gc-batch/lease-events"
PAST = "2020-01-01T00:00:00.000000+00:00"


class EventsMixin:
    def _build(self) -> None:
        self.service = DeviceService()
        self.service.store.add_device(Device("u", "alice", "ik"))

    def _commit(self, request_id):
        # Each idempotent commit appends exactly one audit record.
        return self.service.event_gc_cleanup_expired_batch({
            "mode": "commit", "device_ids": ["alice"],
            "after": 0, "limit": 100, "request_id": request_id})

    def _claim(self, consumer, lease_id, expected=0, limit=100):
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

    def _checkpoint(self, consumer, expected, after):
        return self.service.event_gc_batch_checkpoint({
            "consumer_id": consumer, "expected": expected,
            "after": after})

    def _consume(self, consumer, expected, limit=100):
        return self.service.event_gc_batch_consume({
            "consumer_id": consumer, "expected": expected,
            "limit": limit})

    def _events(self, consumer_id=None, lease_id=None, after=0,
                limit=100):
        return self.service.event_gc_batch_lease_events(
            consumer_id, lease_id, after, limit)

    def _types(self, **kwargs):
        return [(event["lease_id"], event["type"])
                for event in self._events(**kwargs)["events"]]


class EventsServiceTest(EventsMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def test_empty_stream_shape(self) -> None:
        body = self._events()
        self.assertEqual(list(body), ["events", "next_after", "has_more"])
        self.assertEqual(body, {"events": [], "next_after": 0,
                                "has_more": False})

    def test_claim_renew_release_stream(self) -> None:
        self._commit("r1")
        self._claim("c1", "L1", limit=1)
        self._renew("c1", "L1", "rn1")
        self._renew("c1", "L1", "rn2")
        self._op("c1", "L1", "release")
        body = self._events()
        self.assertEqual([e["seq"] for e in body["events"]], [1, 2, 3, 4])
        self.assertEqual(
            [(e["lease_id"], e["type"]) for e in body["events"]],
            [("L1", "claim"), ("L1", "renew"), ("L1", "renew"),
             ("L1", "release")])
        self.assertEqual(body["next_after"], 4)
        self.assertFalse(body["has_more"])
        for event in body["events"]:
            self.assertEqual(list(event),
                             ["seq", "lease_id", "consumer_id", "type"])
            self.assertEqual(event["consumer_id"], "c1")

    def test_explicit_confirm_event(self) -> None:
        self._commit("r1")
        self._claim("c1", "L1", limit=1)
        self._op("c1", "L1", "confirm")
        self.assertEqual(self._types(),
                         [("L1", "claim"), ("L1", "confirm")])

    def test_implicit_confirm_via_checkpoint_in_creation_order(self) -> None:
        # Three audit records; two unterminated leases of c1. Advancing
        # the checkpoint across both appends one implicit_confirm per
        # lease in lease creation order.
        self._commit("r1")
        self._commit("r2")
        self._commit("r3")
        self._claim("c1", "L1", limit=1)
        # L2 starts at 0 as well only after L1 is out of the way; an
        # expired L1 does not block, so expire it directly.
        self.service.store._cleanup_leases["L1"].expires = PAST
        self._claim("c1", "L2", expected=0, limit=2)
        self._checkpoint("c1", 0, 2)
        self.assertEqual(self._types(),
                         [("L1", "claim"), ("L2", "claim"),
                          ("L1", "implicit_confirm"),
                          ("L2", "implicit_confirm")])
        self.assertEqual([e["seq"] for e in self._events()["events"]],
                         [1, 2, 3, 4])

    def test_implicit_confirm_via_consume(self) -> None:
        self._commit("r1")
        self._claim("c1", "L1", limit=1)
        self._consume("c1", 0, limit=1)
        self.assertEqual(self._types(),
                         [("L1", "claim"), ("L1", "implicit_confirm")])

    def test_explicit_confirm_crossing_other_expired_lease(self) -> None:
        # An explicit confirm of L3 moves the checkpoint over the
        # already-expired L1; L1 is implicitly confirmed (creation
        # order first), the target L3 explicitly.
        self._commit("r1")
        self._commit("r2")
        self._claim("c1", "L1", limit=1)
        self.service.store._cleanup_leases["L1"].expires = PAST
        self._claim("c1", "L3", expected=0, limit=2)
        self._op("c1", "L3", "confirm")
        self.assertEqual(self._types(),
                         [("L1", "claim"), ("L3", "claim"),
                          ("L1", "implicit_confirm"),
                          ("L3", "confirm")])

    def test_replays_conflicts_noops_and_expiry_record_nothing(self) -> None:
        self._commit("r1")
        self._commit("r2")
        self._claim("c1", "L1", limit=1)
        # Exact claim replay.
        self._claim("c1", "L1", limit=1)
        self._renew("c1", "L1", "rn1")
        # Exact renewal replay.
        self._renew("c1", "L1", "rn1")
        # Empty-page claim (checkpoint stays at 0, audit tail not
        # reached here; instead exercise the equal-value checkpoint
        # no-op and a released-lease confirm read-only case).
        self._checkpoint("c1", 0, 0)
        self._op("c1", "L1", "release")
        # Replaying the same release answers 200 and records nothing.
        self._op("c1", "L1", "release")
        # A released lease cannot then be confirmed (409, no event).
        with self.assertRaises(ServiceError) as caught:
            self._op("c1", "L1", "confirm")
        self.assertEqual(caught.exception.status_code, 409)
        types = self._types()
        self.assertEqual(types,
                         [("L1", "claim"), ("L1", "renew"),
                          ("L1", "release")])
        self.assertEqual([e["seq"] for e in self._events()["events"]],
                         [1, 2, 3])

    def test_empty_page_claim_records_nothing(self) -> None:
        self._commit("r1")
        # Move the consumer to the audit tail then claim an empty page.
        self._checkpoint("c1", 0, 1)
        _body, status = self._claim("c1", "EMPTY", expected=1, limit=1)
        self.assertEqual(status, 200)
        self.assertEqual(self._events()["events"], [])
        # The id stays free (a later real claim with the same id is not
        # a replay conflict).
        self._commit("r2")
        _body, status = self._claim("c1", "EMPTY", expected=1, limit=1)
        self.assertEqual(status, 201)
        self.assertEqual(self._types(lease_id="EMPTY"),
                         [("EMPTY", "claim")])

    def test_monotonic_advance_does_not_reconfirm(self) -> None:
        self._commit("r1")
        self._commit("r2")
        self._claim("c1", "L1", limit=1)
        self._checkpoint("c1", 0, 1)
        # A later advance past the already-crossed lease adds no event.
        self._checkpoint("c1", 1, 2)
        self.assertEqual(self._types(),
                         [("L1", "claim"), ("L1", "implicit_confirm")])

    def test_consumer_and_lease_filters(self) -> None:
        self._commit("r1")
        self._commit("r2")
        self._claim("c1", "L1", limit=1)
        self._op("c1", "L1", "confirm")
        self._claim("c2", "L2", limit=1)
        self.assertEqual(self._types(consumer_id="c1"),
                         [("L1", "claim"), ("L1", "confirm")])
        self.assertEqual(self._types(lease_id="L2"),
                         [("L2", "claim")])
        self.assertEqual(self._types(consumer_id="c1", lease_id="L2"),
                         [])
        # Unknown filter values are not errors: just an empty page.
        self.assertEqual(self._events(consumer_id="nope")["events"], [])
        self.assertEqual(self._events(lease_id="nope")["events"], [])

    def test_seq_paging_contract(self) -> None:
        self._commit("r1")
        self._claim("c1", "L1", limit=1)
        self._renew("c1", "L1", "rn1")
        self._op("c1", "L1", "confirm")
        first = self._events(limit=2)
        self.assertEqual([e["seq"] for e in first["events"]], [1, 2])
        self.assertEqual(first["next_after"], 2)
        self.assertTrue(first["has_more"])
        second = self._events(after=2, limit=2)
        self.assertEqual([e["seq"] for e in second["events"]], [3])
        self.assertEqual(second["next_after"], 3)
        self.assertFalse(second["has_more"])
        # An empty page echoes the after cursor.
        empty = self._events(after=3, limit=2)
        self.assertEqual(empty["events"], [])
        self.assertEqual(empty["next_after"], 3)
        self.assertFalse(empty["has_more"])
        # The cursor applies before the filter: paging c1 after seq 2
        # only sees the later c1 event.
        filtered = self._events(consumer_id="c1", after=2)
        self.assertEqual([e["seq"] for e in filtered["events"]], [3])

    def test_validation_in_service(self) -> None:
        for kwargs in (
                {"consumer_id": ""},
                {"lease_id": ""},
                {"consumer_id": 1},
                {"lease_id": 1},
                {"after": -1},
                {"after": True},
                {"after": 2**63},
                {"limit": 0},
                {"limit": 101},
                {"limit": True},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ServiceError) as caught:
                    self._events(**kwargs)
                field = next(iter(kwargs))
                self.assertEqual(caught.exception.field, field)
                self.assertEqual(caught.exception.status_code, 400)


class EventsPersistenceTest(EventsMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.path = os.path.join(self.directory, "state.json")
        self.service = DeviceService()
        self.state_store = attach_persistence(self.service, self.path)
        self.service.store.add_device(Device("u", "alice", "ik"))

    def _document(self):
        with open(self.path, encoding="utf-8") as handle:
            return json.load(handle)

    def test_stream_persists_and_reloads(self) -> None:
        self._commit("r1")
        self._claim("c1", "L1", limit=1)
        self._renew("c1", "L1", "rn1")
        self._consume("c1", 0, limit=1)
        document = self._document()
        self.assertIn("cleanup_lease_events", document)
        self.assertEqual(document["cleanup_lease_events"], [
            {"seq": 1, "lease_id": "L1", "consumer_id": "c1",
             "type": "claim"},
            {"seq": 2, "lease_id": "L1", "consumer_id": "c1",
             "type": "renew"},
            {"seq": 3, "lease_id": "L1", "consumer_id": "c1",
             "type": "implicit_confirm"},
        ])
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        self.assertEqual(
            restarted.event_gc_batch_lease_events(None, None, 0, 100),
            self.service.event_gc_batch_lease_events(None, None, 0, 100))
        self.assertTrue(restarted.persistence_integrity()["consistent"])

    def test_read_only_query_consumes_no_generation(self) -> None:
        self._commit("r1")
        self._claim("c1", "L1", limit=1)
        generation = self.state_store.commit_seq
        body_a = self._events()
        body_b = self._events()
        self.assertEqual(body_a, body_b)
        self.assertEqual(self.state_store.commit_seq, generation)

    def test_event_rolls_back_with_failed_transaction(self) -> None:
        import e2ee_backend.persistence as persistence_mod

        self._commit("r1")
        self._claim("c1", "L1", limit=1)
        generation = self.state_store.commit_seq
        real_fsync = persistence_mod.os.fsync
        calls = {"n": 0}

        def fail_first_directory_fsync(fd) -> None:  # noqa: ANN001
            if os.fstat(fd).st_mode & 0o170000 == 0o040000:
                calls["n"] += 1
                if calls["n"] == 1:
                    raise OSError("transient directory fsync failure")
            real_fsync(fd)

        persistence_mod.os.fsync = fail_first_directory_fsync
        try:
            with self.assertRaises(PersistenceUnavailable):
                self._op("c1", "L1", "confirm")
        finally:
            persistence_mod.os.fsync = real_fsync
        self.assertEqual(self.state_store.commit_seq, generation)
        # The confirm event and the checkpoint advance both rolled back:
        # only the claim event survives.
        self.assertEqual(self._types(), [("L1", "claim")])
        body, _status = self._checkpoint("c1", None, None)
        self.assertEqual(body["after"], 0)
        # The retried confirm commits and resumes the seq at 2.
        _body, status = self._op("c1", "L1", "confirm")
        self.assertEqual(status, 201)
        self.assertEqual(self.state_store.commit_seq, generation + 1)
        self.assertEqual(
            [(e["seq"], e["type"]) for e in self._events()["events"]],
            [(1, "claim"), (2, "confirm")])

    def _restart_with_mutated_document(self, mutate):
        document = self._document()
        mutate(document)
        document.pop("integrity_log_version", None)
        bad_path = os.path.join(self.directory, "bad.json")
        with open(bad_path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        service = DeviceService()
        return service, bad_path

    def test_missing_section_loads_empty(self) -> None:
        self._commit("r1")
        service, bad_path = self._restart_with_mutated_document(
            lambda d: d.pop("cleanup_lease_events"))
        attach_persistence(service, bad_path)  # must not raise
        self.assertEqual(service.store._cleanup_lease_events, [])
        self.assertTrue(service.persistence_integrity()["consistent"])

    def test_bad_documents_refuse_startup(self) -> None:
        from e2ee_backend.persistence import StateFileError

        self._commit("r1")
        self._claim("c1", "L1", limit=1)

        def must_refuse(mutate, needle):
            service, bad_path = self._restart_with_mutated_document(mutate)
            with self.assertRaises(StateFileError) as caught:
                attach_persistence(service, bad_path)
            self.assertIn(needle, str(caught.exception))

        good = {"seq": 1, "lease_id": "L1", "consumer_id": "c1",
                "type": "claim"}

        def with_events(events):
            return lambda d: d.__setitem__("cleanup_lease_events", events)

        must_refuse(with_events([{}]), "must have exactly the keys")
        must_refuse(with_events("x"), "malformed top-level")
        must_refuse(with_events([{
            "seq": 2, "lease_id": "L1", "consumer_id": "c1",
            "type": "claim"}]), "consecutively")
        must_refuse(with_events([{
            "seq": 0, "lease_id": "L1", "consumer_id": "c1",
            "type": "claim"}]), "positive integer")
        must_refuse(with_events([{
            "seq": 1, "lease_id": "", "consumer_id": "c1",
            "type": "claim"}]), "lease_id")
        must_refuse(with_events([{
            "seq": 1, "lease_id": "L1", "consumer_id": "",
            "type": "claim"}]), "consumer_id")
        must_refuse(with_events([{
            "seq": 1, "lease_id": "L1", "consumer_id": "c1",
            "type": "bogus"}]), "type")
        must_refuse(with_events([{
            "seq": 1, "lease_id": "NOPE", "consumer_id": "c1",
            "type": "claim"}]), "unknown cleanup lease")
        must_refuse(with_events([{
            "seq": 1, "lease_id": "L1", "consumer_id": "c2",
            "type": "claim"}]), "does not match the lease")
        # A release event contradicting an unmarked lease.
        release = dict(good, type="release")
        must_refuse(with_events([release]), "not marked release")
        # Two terminating events on one lease.
        must_refuse(with_events([
            dict(good, type="confirm"),
            {"seq": 2, "lease_id": "L1", "consumer_id": "c1",
             "type": "implicit_confirm"}]),
            "more than one terminating event")


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

    def _request(self, path=PATH, raw=None, method="GET",
                 extra_headers=None):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        headers = {"Content-Type": "application/json"} if raw is not None \
            else {}
        if extra_headers:
            headers.update(extra_headers)
        conn.request(method, path, body=raw, headers=headers)
        response = conn.getresponse()
        data = response.read().decode("utf-8")
        conn.close()
        return response.status, (json.loads(data) if data else None), data

    def test_page_over_http_with_key_order(self) -> None:
        self._commit("r1")
        self._claim("c1", "L1", limit=1)
        self._op("c1", "L1", "confirm")
        status, body, raw = self._request()
        self.assertEqual(status, 200)
        self.assertEqual(list(body), ["events", "next_after", "has_more"])
        self.assertEqual([i["type"] for i in body["events"]],
                         ["claim", "confirm"])
        for earlier, later in (
                ('"events"', '"next_after"'),
                ('"next_after"', '"has_more"'),
                ('"seq"', '"lease_id"'),
                ('"lease_id"', '"consumer_id"'),
                ('"consumer_id"', '"type"')):
            self.assertLess(raw.index(earlier), raw.index(later))

    def test_empty_page_over_http(self) -> None:
        status, body, _ = self._request()
        self.assertEqual(status, 200)
        self.assertEqual(body, {"events": [], "next_after": 0,
                                "has_more": False})

    def test_filters_and_paging_over_http(self) -> None:
        self._commit("r1")
        self._commit("r2")
        self._claim("c1", "L1", limit=1)
        self._op("c1", "L1", "confirm")
        self._claim("c2", "L2", limit=1)
        status, body, _ = self._request(path=PATH + "?consumer_id=c1")
        self.assertEqual(status, 200)
        self.assertEqual([i["lease_id"] for i in body["events"]],
                         ["L1", "L1"])
        status, body, _ = self._request(
            path=PATH + "?lease_id=L2&after=0&limit=10")
        self.assertEqual(status, 200)
        self.assertEqual([i["seq"] for i in body["events"]], [3])
        self.assertEqual(body["next_after"], 3)
        status, body, _ = self._request(path=PATH + "?after=9")
        self.assertEqual(status, 200)
        self.assertEqual(body["events"], [])
        self.assertEqual(body["next_after"], 9)

    def test_query_validation(self) -> None:
        good_after = str(2**63 - 1)
        cases = [
            ("consumer_id=", "consumer_id"),
            ("consumer_id=a&consumer_id=b", "consumer_id"),
            ("lease_id=", "lease_id"),
            ("lease_id=a&lease_id=b", "lease_id"),
            ("after=-1", "after"),
            ("after=01%20", "after"),
            ("after=%201", "after"),
            ("after=+1", "after"),
            ("after=1.0", "after"),
            ("after=0x1", "after"),
            ("after=" + str(2**63), "after"),
            ("after=" + good_after + "0", "after"),
            ("limit=", "limit"),
            ("limit=0", "limit"),
            ("limit=101", "limit"),
            ("limit=1&limit=2", "limit"),
            ("limit=1.0", "limit"),
            ("limit=true", "limit"),
            ("foo=bar", "query"),
            ("foo", "query"),
            ("after=1&foo=bar", "query"),
        ]
        for query, field in cases:
            with self.subTest(query=query):
                status, body, _ = self._request(path=PATH + "?" + query)
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], field)

    def test_validation_order_body_before_query(self) -> None:
        # A non-empty body is rejected as request_body even when the
        # query string is also bad.
        status, body, _ = self._request(path=PATH + "?foo=bar",
                                        raw="{}")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "request_body")

    def test_nonempty_body_rejected(self) -> None:
        status, body, _ = self._request(raw="{}")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "request_body")

    def test_chunked_nonempty_body_rejected(self) -> None:
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.connect()
        conn.send(b"GET " + PATH.encode() + b" HTTP/1.1\r\n"
                  b"Host: 127.0.0.1\r\n"
                  b"Transfer-Encoding: chunked\r\n"
                  b"Content-Type: application/json\r\n"
                  b"Connection: close\r\n\r\n"
                  b"2\r\n{}\r\n0\r\n\r\n")
        raw = b""
        while True:
            chunk = conn.sock.recv(4096)
            if not chunk:
                break
            raw += chunk
        conn.close()
        self.assertIn(b"400", raw.split(b"\r\n", 1)[0])
        self.assertIn(b'"request_body"', raw)

    def test_trailing_question_mark_accepted(self) -> None:
        status, _body, _ = self._request(path=PATH + "?")
        self.assertEqual(status, 200)

    def test_method_routing(self) -> None:
        # The route is GET-only; a POST does not match any handler.
        status, _body, _ = self._request(method="POST", raw="{}")
        self.assertEqual(status, 404)


if __name__ == "__main__":
    unittest.main()
