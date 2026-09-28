"""Tests for named lease-event subscriptions and the consume fix.

``POST /v1/lease-subs`` registers a named binding
(subscriber_id/consumer_id/lease_id filters, each filter a non-empty
string or null); the first registration is 201, an identical replay
200 and a changed binding 409/subscriber_id. ``GET
/v1/lease-subs/{subscriber_id}`` pages the filtered event stream
without advancing the position (only ``limit``; unknown id
404/subscriber_id). ``POST /v1/lease-subs/{subscriber_id}/ack``
moves the position to a matching event seq (equal 200, forward 201;
backwards/out-of-bounds/non-matching 409/after; unknown id
404/subscriber_id and never creates one).

The sibling ``POST /v1/event-gc-batch/lease-events/consume`` fix: a
brand-new subscriber reading an empty stream with expected 0 also
gets a ``lease_event_cursors`` record at after 0 and a 201; later
empty pages are 200 and write nothing. The two sections are
independent and never migrated. The new ``lease_subscriptions``
section serializes right after ``lease_event_cursors`` (the 27th
canonical snapshot section): a missing section loads empty, while
duplicates, format errors or cursor contradictions refuse startup
without overwriting the file, and the probe/sidecar state_hash cover
it.
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

SUBS_PATH = "/v1/lease-subs"
CONSUME_PATH = "/v1/event-gc-batch/lease-events/consume"


class LeaseSubsMixin:
    def _build(self) -> None:
        self.service = DeviceService()
        self.service.store.add_device(Device("u", "bob", "ik"))

    def _commit(self, request_id):
        return self.service.event_gc_cleanup_expired_batch({
            "mode": "commit", "device_ids": ["bob"],
            "after": 0, "limit": 100, "request_id": request_id})

    def _claim(self, consumer, lease_id, expected=0, limit=100):
        return self.service.event_gc_batch_claim({
            "consumer_id": consumer, "lease_id": lease_id,
            "expected": expected, "limit": limit})

    def _advance(self, consumer, expected, after):
        return self.service.event_gc_batch_checkpoint({
            "consumer_id": consumer, "expected": expected,
            "after": after})

    def _register(self, subscriber_id, consumer_id=None, lease_id=None):
        return self.service.lease_subscription_register({
            "subscriber_id": subscriber_id, "consumer_id": consumer_id,
            "lease_id": lease_id})

    def _page(self, subscriber_id, limit=100):
        return self.service.lease_subscription_page(subscriber_id, limit)

    def _ack(self, subscriber_id, expected, after):
        return self.service.lease_subscription_ack(
            subscriber_id, {"expected": expected, "after": after})

    def _consume(self, subscriber_id, expected, limit):
        return self.service.event_gc_batch_lease_event_consume({
            "subscriber_id": subscriber_id, "expected": expected,
            "limit": limit})

    def _seed_two_consumers(self):
        # Events: r1, claim c1/L1 (seq 1), advance c1 0->1
        # (implicit_confirm seq 2); r2, claim c2/L2 (seq 3).
        self._commit("r1")
        self._claim("c1", "L1")
        self._advance("c1", 0, 1)
        self._commit("r2")
        self._claim("c2", "L2")


class LeaseSubsServiceTest(LeaseSubsMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()
        self._seed_two_consumers()

    def test_first_registration_201_at_zero_and_replay_200(self):
        view, status = self._register("s1", "c1")
        self.assertEqual(status, 201)
        self.assertEqual(view, {
            "subscriber_id": "s1", "consumer_id": "c1",
            "lease_id": None, "after": 0})
        self.assertEqual(list(view),
                         ["subscriber_id", "consumer_id", "lease_id",
                          "after"])
        view2, status2 = self._register("s1", "c1")
        self.assertEqual(status2, 200)
        self.assertEqual(view2, view)

    def test_replay_after_ack_still_reports_after_zero(self):
        self._register("s1")
        self._ack("s1", 0, 2)
        view, status = self._register("s1")
        self.assertEqual(status, 200)
        # The registration response is the frozen first view: the
        # binding moved to 2 via ack, but the replay still reports 0.
        self.assertEqual(view["after"], 0)
        self.assertEqual(
            self.service.store._lease_subscriptions["s1"].after, 2)

    def test_null_filters_register(self):
        view, status = self._register("s1")
        self.assertEqual(status, 201)
        self.assertIsNone(view["consumer_id"])
        self.assertIsNone(view["lease_id"])

    def test_different_filters_conflict(self):
        self._register("s1", "c1")
        for consumer_id, lease_id in (("c2", None), ("c1", "L1"),
                                      (None, None)):
            with self.subTest(consumer_id=consumer_id, lease_id=lease_id):
                with self.assertRaises(ServiceError) as caught:
                    self._register("s1", consumer_id, lease_id)
                self.assertEqual(caught.exception.status_code, 409)
                self.assertEqual(caught.exception.field, "subscriber_id")
        # The stored binding is untouched.
        self.assertEqual(
            self.service.store._lease_subscriptions["s1"].consumer_id,
            "c1")

    def test_register_validation(self):
        cases = [
            ("not object", [], "request_body"),
            ("missing field",
             {"subscriber_id": "s", "consumer_id": None}, "lease_id"),
            ("empty subscriber",
             {"subscriber_id": "", "consumer_id": None,
              "lease_id": None}, "subscriber_id"),
            ("int subscriber",
             {"subscriber_id": 1, "consumer_id": None,
              "lease_id": None}, "subscriber_id"),
            ("bool subscriber",
             {"subscriber_id": True, "consumer_id": None,
              "lease_id": None}, "subscriber_id"),
            ("empty consumer",
             {"subscriber_id": "s", "consumer_id": "",
              "lease_id": None}, "consumer_id"),
            ("int consumer",
             {"subscriber_id": "s", "consumer_id": 1,
              "lease_id": None}, "consumer_id"),
            ("empty lease",
             {"subscriber_id": "s", "consumer_id": None,
              "lease_id": ""}, "lease_id"),
            ("extra key",
             {"subscriber_id": "s", "consumer_id": None,
              "lease_id": None, "x": 1}, "x"),
        ]
        for label, payload, field in cases:
            with self.subTest(label):
                with self.assertRaises(ServiceError) as caught:
                    self.service.lease_subscription_register(payload)
                self.assertEqual(caught.exception.status_code, 400)
                self.assertEqual(caught.exception.field, field)
        # Nothing was registered by the failed calls.
        self.assertNotIn("s",
                         self.service.store._lease_subscriptions)

    def test_page_filters_by_binding_and_does_not_advance(self):
        self._register("s1", "c1")
        page = self._page("s1", limit=1)
        self.assertEqual(list(page),
                         ["subscriber_id", "events", "next_after",
                          "has_more"])
        self.assertEqual([event["seq"] for event in page["events"]], [1])
        self.assertEqual(page["next_after"], 1)
        self.assertTrue(page["has_more"])
        for event in page["events"]:
            self.assertEqual(
                list(event),
                ["seq", "lease_id", "consumer_id", "type"])
        # c1's events are seqs 1 (claim) and 2 (implicit_confirm);
        # seq 3 belongs to c2 and must not show.
        page_all = self._page("s1")
        self.assertEqual([event["seq"] for event in page_all["events"]],
                         [1, 2])
        self.assertFalse(page_all["has_more"])
        # The read advanced nothing.
        self.assertEqual(
            self.service.store._lease_subscriptions["s1"].after, 0)

    def test_page_filters_by_lease_id_and_both(self):
        self._register("bylease", None, "L2")
        self.assertEqual(
            [event["seq"] for event in self._page("bylease")["events"]],
            [3])
        self._register("both", "c1", "L2")
        self.assertEqual(self._page("both")["events"], [])

    def test_page_empty_echoes_position(self):
        self._register("s1", "nobody")
        page = self._page("s1")
        self.assertEqual(page, {
            "subscriber_id": "s1", "events": [],
            "next_after": 0, "has_more": False})
        self._ack("s1", 0, 0)
        # Position stays 0; paging remains stable.
        self.assertEqual(self._page("s1"), page)

    def test_page_unknown_subscriber_is_404_and_creates_nothing(self):
        with self.assertRaises(ServiceError) as caught:
            self._page("ghost")
        self.assertEqual(caught.exception.status_code, 404)
        self.assertEqual(caught.exception.field, "subscriber_id")
        self.assertNotIn("ghost",
                         self.service.store._lease_subscriptions)

    def test_ack_equal_is_200_noop(self):
        self._register("s1")
        view, status = self._ack("s1", 0, 0)
        self.assertEqual(status, 200)
        self.assertEqual(view, {"subscriber_id": "s1", "after": 0})
        self.assertEqual(list(view), ["subscriber_id", "after"])

    def test_ack_forward_is_201_and_page_then_follows(self):
        self._register("s1")
        view, status = self._ack("s1", 0, 3)
        self.assertEqual(status, 201)
        self.assertEqual(view, {"subscriber_id": "s1", "after": 3})
        self.assertEqual(self._page("s1")["events"], [])
        # Equal replay is 200.
        self.assertEqual(self._ack("s1", 3, 3)[1], 200)

    def test_ack_backwards_out_of_bounds_and_non_matching(self):
        self._register("s1", "c1")
        self._ack("s1", 0, 2)
        with self.assertRaises(ServiceError) as caught:
            self._ack("s1", 2, 1)
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "after")
        # Above the global event stream's last seq.
        with self.assertRaises(ServiceError) as caught:
            self._ack("s1", 2, 4)
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "after")
        # seq 3 exists but belongs to c2, which the binding excludes.
        with self.assertRaises(ServiceError) as caught:
            self._ack("s1", 2, 3)
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "after")
        # The failed calls moved nothing.
        self.assertEqual(
            self.service.store._lease_subscriptions["s1"].after, 2)

    def test_ack_expected_conflict(self):
        self._register("s1")
        with self.assertRaises(ServiceError) as caught:
            self._ack("s1", 1, 1)
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "expected")

    def test_ack_large_integers_are_conflicts_not_400(self):
        # The wire contract puts no upper bound on the non-negative
        # integers: a huge position is a state 409, not a field 400.
        self._register("s1")
        huge = 2**63
        with self.assertRaises(ServiceError) as caught:
            self._ack("s1", huge, 0)
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "expected")
        with self.assertRaises(ServiceError) as caught:
            self._ack("s1", 0, huge)
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "after")

    def test_ack_unknown_is_404_and_creates_nothing(self):
        with self.assertRaises(ServiceError) as caught:
            self._ack("ghost", 0, 0)
        self.assertEqual(caught.exception.status_code, 404)
        self.assertEqual(caught.exception.field, "subscriber_id")
        self.assertNotIn("ghost",
                         self.service.store._lease_subscriptions)

    def test_ack_validation(self):
        self._register("s1")
        for payload, field in (
                ({}, "expected"),
                ({"expected": 0}, "after"),
                ({"expected": -1, "after": 0}, "expected"),
                ({"expected": True, "after": 0}, "expected"),
                ({"expected": 0, "after": 1.0}, "after"),
                ({"expected": 0, "after": None}, "after"),
                ({"expected": "0", "after": 0}, "expected"),
                ({"expected": 0, "after": 0, "z": 1}, "z")):
            with self.subTest(payload=payload):
                with self.assertRaises(ServiceError) as caught:
                    self.service.lease_subscription_ack("s1", payload)
                self.assertEqual(caught.exception.status_code, 400)
                self.assertEqual(caught.exception.field, field)
        with self.assertRaises(ServiceError) as caught:
            self.service.lease_subscription_ack("s1", [])
        self.assertEqual(caught.exception.status_code, 400)
        self.assertEqual(caught.exception.field, "request_body")

    def test_subscriptions_and_cursors_are_independent(self):
        # A consume cursor subscriber and a subscription subscriber
        # with the same id never interact: separate stores and
        # positions.
        self._consume("dup", 0, 100)
        self._register("dup", "c1")
        cursor = self.service.store._lease_event_cursors["dup"]
        subscription = self.service.store._lease_subscriptions["dup"]
        self.assertEqual(cursor.after, 3)
        self.assertEqual(subscription.after, 0)
        # Acking the subscription leaves the cursor untouched.
        self._ack("dup", 0, 2)
        self.assertEqual(cursor.after, 3)
        self.assertEqual(subscription.after, 2)
        # Consuming again leaves the subscription untouched.
        self._consume("dup", 3, 100)
        self.assertEqual(subscription.after, 2)


class ConsumeEmptyStreamServiceTest(LeaseSubsMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def test_new_subscriber_empty_stream_registers_cursor_201(self):
        view, status = self._consume("s0", 0, 10)
        self.assertEqual(status, 201)
        self.assertEqual(view, {
            "subscriber_id": "s0", "events": [],
            "next_after": 0, "has_more": False})
        record = self.service.store._lease_event_cursors["s0"]
        self.assertEqual(record.after, 0)

    def test_later_empty_page_is_200_and_writes_nothing(self):
        self.assertEqual(self._consume("s0", 0, 10)[1], 201)
        snapshot = self.service.store.snapshot_state()
        view, status = self._consume("s0", 0, 10)
        self.assertEqual(status, 200)
        self.assertEqual(view["next_after"], 0)
        self.assertEqual(self.service.store.snapshot_state(), snapshot)

    def test_registered_at_zero_rejects_nonzero_expected(self):
        self.assertEqual(self._consume("s0", 0, 10)[1], 201)
        with self.assertRaises(ServiceError) as caught:
            self._consume("s0", 1, 10)
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "expected")

    def test_events_after_empty_registration_advance_normally(self):
        self.assertEqual(self._consume("s0", 0, 10)[1], 201)
        self._commit("r1")
        self._claim("c1", "L1")
        view, status = self._consume("s0", 0, 10)
        self.assertEqual(status, 201)
        self.assertEqual([event["seq"] for event in view["events"]], [1])
        self.assertEqual(view["next_after"], 1)
        self.assertEqual(
            self.service.store._lease_event_cursors["s0"].after, 1)


class LeaseSubsHTTPTest(LeaseSubsMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()
        self._seed_two_consumers()
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

    def _request(self, method, path, raw=None):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        headers = {"Content-Type": "application/json"} \
            if raw is not None else {}
        conn.request(method, path, body=raw, headers=headers)
        response = conn.getresponse()
        data = response.read().decode("utf-8")
        conn.close()
        return response.status, (json.loads(data) if data else None), data

    def test_register_get_ack_flow(self):
        status, body, raw = self._request(
            "POST", SUBS_PATH,
            json.dumps({"subscriber_id": "s1", "consumer_id": "c1",
                        "lease_id": None}))
        self.assertEqual(status, 201)
        self.assertEqual(body, {
            "subscriber_id": "s1", "consumer_id": "c1",
            "lease_id": None, "after": 0})
        for earlier, later in (
                ('"subscriber_id"', '"consumer_id"'),
                ('"consumer_id"', '"lease_id"'),
                ('"lease_id"', '"after"')):
            self.assertLess(raw.index(earlier), raw.index(later))
        # Replay.
        status, _, _ = self._request(
            "POST", SUBS_PATH,
            json.dumps({"subscriber_id": "s1", "consumer_id": "c1",
                        "lease_id": None}))
        self.assertEqual(status, 200)
        # Filtered page.
        status, body, _ = self._request("GET", SUBS_PATH + "/s1?limit=1")
        self.assertEqual(status, 200)
        self.assertEqual(list(body),
                         ["subscriber_id", "events", "next_after",
                          "has_more"])
        self.assertEqual([event["seq"] for event in body["events"]], [1])
        # Ack forward.
        status, body, raw = self._request(
            "POST", SUBS_PATH + "/s1/ack",
            json.dumps({"expected": 0, "after": 2}))
        self.assertEqual(status, 201)
        self.assertEqual(body, {"subscriber_id": "s1", "after": 2})
        self.assertLess(raw.index('"subscriber_id"'),
                        raw.index('"after"'))

    def test_register_conflict_and_bad_bodies(self):
        self._request("POST", SUBS_PATH, json.dumps(
            {"subscriber_id": "s1", "consumer_id": None,
             "lease_id": None}))
        status, body, _ = self._request(
            "POST", SUBS_PATH,
            json.dumps({"subscriber_id": "s1", "consumer_id": "c",
                        "lease_id": None}))
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "subscriber_id")
        for raw_body, field in (
                ("[]", "request_body"),
                ("not json", "request_body"),
                (json.dumps({"subscriber_id": "s", "consumer_id": None}),
                 "lease_id"),
                (json.dumps({"subscriber_id": "", "consumer_id": None,
                             "lease_id": None}), "subscriber_id"),
                (json.dumps({"subscriber_id": "s", "consumer_id": 1,
                             "lease_id": None}), "consumer_id"),
                (json.dumps({"subscriber_id": "s", "consumer_id": None,
                             "lease_id": 1}), "lease_id"),
                (json.dumps({"subscriber_id": "s", "consumer_id": None,
                             "lease_id": None, "x": 1}), "x")):
            status, body, _ = self._request("POST", SUBS_PATH, raw_body)
            self.assertEqual(status, 400, raw_body)
            self.assertEqual(body["field"], field, raw_body)
            self.assertEqual(list(body), ["message", "field"])

    def test_register_query_rejected(self):
        status, body, _ = self._request(
            "POST", SUBS_PATH + "?foo=1",
            json.dumps({"subscriber_id": "q", "consumer_id": None,
                        "lease_id": None}))
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "query")

    def test_get_unknown_and_query_body_validation(self):
        status, body, _ = self._request("GET", SUBS_PATH + "/ghost")
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "subscriber_id")
        self._register("s1")
        for query, field in (
                ("?limit=0", "limit"),
                ("?limit=101", "limit"),
                ("?limit=x", "limit"),
                ("?limit=", "limit"),
                ("?limit=1&limit=2", "limit"),
                ("?after=0", "query"),
                ("?foo", "query")):
            status, body, _ = self._request("GET", SUBS_PATH + "/s1"
                                            + query)
            self.assertEqual(status, 400, query)
            self.assertEqual(body["field"], field, query)
        # Non-empty body and validation order (body before query).
        status, body, _ = self._request("GET", SUBS_PATH + "/s1", "{}")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "request_body")
        status, body, _ = self._request(
            "GET", SUBS_PATH + "/s1?foo=1", "{}")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "request_body")
        # Trailing bare question mark is accepted.
        self.assertEqual(self._request("GET", SUBS_PATH + "/s1?")[0], 200)

    def test_get_path_decoding(self):
        status, body, _ = self._request("GET", SUBS_PATH + "/a%2Fb")
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "subscriber_id")
        status, body, _ = self._request("GET", SUBS_PATH + "/%zz")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "subscriber_id")
        status, body, _ = self._request(
            "GET", SUBS_PATH + "/a/b/ack",
            json.dumps({"expected": 0, "after": 0}))
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "subscriber_id")

    def test_unknown_sub_paths_are_404_subscriber(self):
        self._register("s1")
        for method, path, raw in (
                ("POST", SUBS_PATH + "/s1/other",
                 json.dumps({"expected": 0, "after": 0})),
                ("POST", SUBS_PATH + "/s1/ack/extra",
                 json.dumps({"expected": 0, "after": 0})),
                ("GET", SUBS_PATH + "/s1/ack", None),
                ("POST", SUBS_PATH + "/",
                 json.dumps({"subscriber_id": "x", "consumer_id": None,
                             "lease_id": None}))):
            status, body, _ = self._request(method, path, raw)
            self.assertEqual(status, 404, (method, path))
            self.assertEqual(body["field"], "subscriber_id",
                             (method, path))

    def test_ack_unknown_and_bad_bodies(self):
        status, body, _ = self._request(
            "POST", SUBS_PATH + "/ghost/ack",
            json.dumps({"expected": 0, "after": 0}))
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "subscriber_id")
        self._register("s1")
        for raw_body, field in (
                ("{}", "expected"),
                (json.dumps({"expected": 0}), "after"),
                (json.dumps({"expected": -1, "after": 0}), "expected"),
                (json.dumps({"expected": True, "after": 0}), "expected"),
                (json.dumps({"expected": 0, "after": 1.5}), "after"),
                (json.dumps({"expected": 0, "after": None}), "after"),
                (json.dumps({"expected": 0, "after": 0, "z": 1}), "z"),
                ("{bad", "request_body")):
            status, body, _ = self._request(
                "POST", SUBS_PATH + "/s1/ack", raw_body)
            self.assertEqual(status, 400, raw_body)
            self.assertEqual(body["field"], field, raw_body)
        status, body, _ = self._request(
            "POST", SUBS_PATH + "/s1/ack?foo",
            json.dumps({"expected": 0, "after": 0}))
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "query")

    def test_consume_empty_stream_over_http(self):
        # The shared server is seeded, so run this against a fresh
        # server whose event stream is empty.
        empty_service = DeviceService()
        server, _ = create_server("127.0.0.1", 0, empty_service)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever,
                                  daemon=True)
        thread.start()
        try:
            def post(body):
                conn = HTTPConnection("127.0.0.1", port, timeout=5)
                conn.request("POST", CONSUME_PATH, body=json.dumps(body),
                             headers={"Content-Type": "application/json"})
                response = conn.getresponse()
                data = response.read().decode("utf-8")
                conn.close()
                return response.status, json.loads(data)

            status, body = post({"subscriber_id": "new", "expected": 0,
                                 "limit": 10})
            self.assertEqual(status, 201)
            self.assertEqual(body, {
                "subscriber_id": "new", "events": [],
                "next_after": 0, "has_more": False})
            status, body = post({"subscriber_id": "new", "expected": 0,
                                 "limit": 10})
            self.assertEqual(status, 200)
            status, body = post({"subscriber_id": "new", "expected": 1,
                                 "limit": 10})
            self.assertEqual(status, 409)
            self.assertEqual(body["field"], "expected")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


class LeaseSubsPersistenceTest(LeaseSubsMixin, unittest.TestCase):
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

    def test_section_order_and_item_shape(self):
        self._seed_two_consumers()
        self._consume("cursor-sub", 0, 1)
        self._register("sub1", "c1", "L1")
        self._ack("sub1", 0, 1)
        keys = list(self._document())
        self.assertLess(keys.index("cleanup_lease_events"),
                        keys.index("lease_event_cursors"))
        self.assertEqual(
            keys.index("lease_subscriptions"),
            keys.index("lease_event_cursors") + 1)
        subscriptions = self._document()["lease_subscriptions"]
        self.assertEqual(subscriptions, [{
            "subscriber_id": "sub1", "consumer_id": "c1",
            "lease_id": "L1", "after": 1}])
        for item in subscriptions:
            self.assertEqual(list(item),
                             ["subscriber_id", "consumer_id", "lease_id",
                              "after"])

    def test_empty_stream_cursor_persists_at_zero(self):
        generation = self.state_store.commit_seq
        self.assertEqual(self._consume("s0", 0, 10)[1], 201)
        self.assertEqual(self.state_store.commit_seq, generation + 1)
        self.assertEqual(self._document()["lease_event_cursors"],
                         [{"subscriber_id": "s0", "after": 0}])
        # The second empty page consumes no generation.
        self.assertEqual(self._consume("s0", 0, 10)[1], 200)
        self.assertEqual(self.state_store.commit_seq, generation + 1)

    def test_equal_ack_writes_nothing(self):
        self._seed_two_consumers()
        self._register("sub1")
        generation = self.state_store.commit_seq
        before = self._document()
        self.assertEqual(self._ack("sub1", 0, 0)[1], 200)
        self.assertEqual(self.state_store.commit_seq, generation)
        self.assertEqual(self._document(), before)

    def test_restart_restores_both_sections(self):
        self._seed_two_consumers()
        self._consume("s0", 0, 10)
        self._register("sub1", "c1")
        self._ack("sub1", 0, 2)
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        self.assertEqual(
            restarted.store._lease_event_cursors["s0"].after, 3)
        subscription = restarted.store._lease_subscriptions["sub1"]
        self.assertEqual(subscription.consumer_id, "c1")
        self.assertEqual(subscription.lease_id, None)
        self.assertEqual(subscription.after, 2)
        # c1's events stop at seq 2: seq 3 belongs to c2 and is
        # filtered out, so the page is empty from position 2.
        self.assertEqual(
            restarted.lease_subscription_page("sub1", 100)["events"],
            [])
        self.assertTrue(
            restarted.persistence_integrity()["consistent"])

    def test_legacy_document_without_section_loads_empty(self):
        self._seed_two_consumers()
        self._register("sub1", "c1")
        document = self._document()
        document.pop("lease_subscriptions")
        document.pop("integrity_log_version", None)
        legacy_path = os.path.join(self.directory, "legacy.json")
        with open(legacy_path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        restarted = DeviceService()
        attach_persistence(restarted, legacy_path)  # must not raise
        self.assertEqual(restored_state(restarted, "sub1"), None)
        with self.assertRaises(ServiceError) as caught:
            restarted.lease_subscription_page("sub1", 100)
        self.assertEqual(caught.exception.status_code, 404)
        self.assertTrue(
            restarted.persistence_integrity()["consistent"])

    def test_legacy_document_without_cursor_section_still_loads(self):
        self._seed_two_consumers()
        document = self._document()
        document.pop("lease_event_cursors")
        document.pop("lease_subscriptions")
        document.pop("integrity_log_version", None)
        legacy_path = os.path.join(self.directory, "legacy2.json")
        with open(legacy_path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        restarted = DeviceService()
        attach_persistence(restarted, legacy_path)
        # A new subscriber starts at 0 against the restored stream.
        self.assertEqual(
            restarted.event_gc_batch_lease_event_consume(
                {"subscriber_id": "fresh", "expected": 0,
                 "limit": 100})[0]["next_after"], 3)
        self.assertTrue(
            restarted.persistence_integrity()["consistent"])

    def _assert_refuses_startup(self, mutate):
        document = self._document()
        mutate(document)
        document.pop("integrity_log_version", None)
        bad_path = os.path.join(
            self.directory,
            f"bad-{threading.get_ident()}-"
            f"{len(os.listdir(self.directory))}.json")
        with open(bad_path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        with open(bad_path, "rb") as handle:
            original = handle.read()
        service = DeviceService()
        with self.assertRaises(StateFileError):
            attach_persistence(service, bad_path)
        with open(bad_path, "rb") as handle:
            self.assertEqual(handle.read(), original)

    def test_duplicate_subscriber_refuses_startup(self):
        self._seed_two_consumers()
        self._register("sub1")

        def mutate(document):
            document["lease_subscriptions"].append(
                dict(document["lease_subscriptions"][0]))
        self._assert_refuses_startup(mutate)

    def test_bad_key_order_refuses_startup(self):
        self._seed_two_consumers()
        self._register("sub1", "c1", "L1")
        self._ack("sub1", 0, 1)

        def mutate(document):
            raw = document["lease_subscriptions"][0]
            document["lease_subscriptions"][0] = {
                "subscriber_id": raw["subscriber_id"],
                "lease_id": raw["lease_id"],
                "consumer_id": raw["consumer_id"],
                "after": raw["after"]}
        self._assert_refuses_startup(mutate)

    def test_bad_filter_types_refuse_startup(self):
        self._seed_two_consumers()
        self._register("sub1")

        def mutate_consumer(document):
            document["lease_subscriptions"][0]["consumer_id"] = ""

        def mutate_lease(document):
            document["lease_subscriptions"][0]["lease_id"] = 3

        def mutate_subscriber(document):
            document["lease_subscriptions"][0]["subscriber_id"] = 7
        self._assert_refuses_startup(mutate_consumer)
        self._assert_refuses_startup(mutate_lease)
        self._assert_refuses_startup(mutate_subscriber)

    def test_after_on_non_matching_event_refuses_startup(self):
        self._seed_two_consumers()
        self._register("sub1", "c2")

        def mutate(document):
            # seq 1 belongs to c1, which the c2 binding excludes.
            document["lease_subscriptions"][0]["after"] = 1
        self._assert_refuses_startup(mutate)

    def test_after_out_of_bounds_refuses_startup(self):
        self._seed_two_consumers()
        self._register("sub1")

        def mutate(document):
            document["lease_subscriptions"][0]["after"] = 99
        self._assert_refuses_startup(mutate)

    def test_malformed_section_refuses_startup(self):
        self._seed_two_consumers()
        self._register("sub1")

        def mutate(document):
            document["lease_subscriptions"] = {}
        self._assert_refuses_startup(mutate)


def restored_state(service, subscriber_id):
    return service.store._lease_subscriptions.get(subscriber_id)


if __name__ == "__main__":
    unittest.main()
