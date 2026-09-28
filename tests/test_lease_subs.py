"""Tests for the filter-bound lease-event subscriptions.

``POST /v1/lease-subs`` freezes a (consumer_id, lease_id) filter for a
named subscriber; ``GET /v1/lease-subs/{subscriber_id}`` pages the
matching lifecycle events without advancing; ``POST
/v1/lease-subs/{subscriber_id}/ack`` advances the frozen cursor. The
section (``lease_subscriptions``, canonical section 27, serialized
right after ``lease_event_cursors``) is independent of the plain
``lease_event_cursors`` and nothing migrates between them.

The same file covers the adjusted ``POST
/v1/event-gc-batch/lease-events/consume`` contract: a brand-new
subscriber polling an empty stream still anchors an ``after=0`` record
and answers 201; later empty pages answer 200 and write nothing.
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

CONSUME_PATH = "/v1/event-gc-batch/lease-events/consume"
SUBS_PATH = "/v1/lease-subs"


class LeaseSubsMixin:
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

    def _renew(self, consumer, lease_id, renewal_id):
        return self.service.event_gc_batch_lease_renew({
            "consumer_id": consumer, "lease_id": lease_id,
            "renewal_id": renewal_id})

    def _op(self, consumer, lease_id, op, expected=0):
        return self.service.event_gc_batch_lease_op({
            "consumer_id": consumer, "lease_id": lease_id,
            "expected": expected, "op": op})

    def _seed(self):
        # seq 1: c1/L1 claim, seq 2: c1/L1 renew, seq 3: c2/L2 claim,
        # seq 4: c1/L1 release.
        self._commit("r1")
        self._claim("c1", "L1", limit=1)
        self._renew("c1", "L1", "rn")
        self._claim("c2", "L2", limit=1)
        self._op("c1", "L1", "release")

    # -- the adjusted consume contract ---------------------------------

    def _consume(self, subscriber_id="sub", expected=0, limit=100):
        return self.service.event_gc_batch_lease_event_consume({
            "subscriber_id": subscriber_id, "expected": expected,
            "limit": limit})

    def test_consume_empty_stream_anchors_new_subscriber(self) -> None:
        body, status = self._consume("sub1")
        self.assertEqual(status, 201)
        self.assertEqual(body, {
            "subscriber_id": "sub1", "events": [],
            "next_after": 0, "has_more": False})
        self.assertEqual(list(body),
                         ["subscriber_id", "events", "next_after",
                          "has_more"])
        self.assertIn("sub1",
                      self.service.store._lease_event_cursors)
        self.assertEqual(
            self.service.store._lease_event_cursors["sub1"].after, 0)

    def test_consume_later_empty_page_is_200_and_write_free(self) -> None:
        self._consume("sub1")
        body, status = self._consume("sub1")
        self.assertEqual(status, 200)
        self.assertEqual(body["events"], [])
        self.assertEqual(body["next_after"], 0)

    def test_consume_anchor_does_not_touch_subscriptions(self) -> None:
        self._consume("sub1")
        self.assertEqual(self.service.store._lease_subscriptions, {})
        self.service.lease_subscribe({
            "subscriber_id": "sub1", "consumer_id": None,
            "lease_id": None})
        # The plain consume cursor survives alongside the subscription.
        body, status = self._consume("sub1")
        self.assertEqual(status, 200)
        self.assertEqual(
            self.service.store._lease_event_cursors["sub1"].after, 0)

    def test_consume_nonempty_page_still_201(self) -> None:
        self._seed()
        body, status = self._consume("sub1", limit=2)
        self.assertEqual(status, 201)
        self.assertEqual([event["seq"] for event in body["events"]],
                         [1, 2])
        self.assertEqual(body["next_after"], 2)
        self.assertTrue(body["has_more"])

    def test_consume_expected_conflict(self) -> None:
        self._consume("sub1")
        with self.assertRaises(ServiceError) as caught:
            self._consume("sub1", expected=1)
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "expected")

    # -- POST /v1/lease-subs --------------------------------------------

    def test_subscribe_first_is_201_with_key_order(self) -> None:
        body, status = self.service.lease_subscribe({
            "subscriber_id": "I1", "consumer_id": "c1",
            "lease_id": None})
        self.assertEqual(status, 201)
        self.assertEqual(list(body),
                         ["subscriber_id", "consumer_id", "lease_id",
                          "after"])
        self.assertEqual(body, {
            "subscriber_id": "I1", "consumer_id": "c1",
            "lease_id": None, "after": 0})

    def test_subscribe_all_filters_null(self) -> None:
        body, status = self.service.lease_subscribe({
            "subscriber_id": "I1", "consumer_id": None,
            "lease_id": None})
        self.assertEqual(status, 201)
        self.assertIsNone(body["consumer_id"])
        self.assertIsNone(body["lease_id"])

    def test_subscribe_same_filter_replay_is_200(self) -> None:
        payload = {"subscriber_id": "I1", "consumer_id": "c1",
                   "lease_id": "L1"}
        first, status = self.service.lease_subscribe(payload)
        self.assertEqual(status, 201)
        replay, status = self.service.lease_subscribe(dict(payload))
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)

    def test_subscribe_replay_echoes_frozen_after_zero(self) -> None:
        # Even after an ack advanced the live cursor, a same-filter
        # POST replay returns the frozen creation view with after=0.
        self._seed()
        payload = {"subscriber_id": "I1", "consumer_id": "c1",
                   "lease_id": None}
        first, _ = self.service.lease_subscribe(payload)
        self.assertEqual(first["after"], 0)
        self.service.lease_subscription_ack(
            "I1", {"expected": 0, "after": 1})
        replay, status = self.service.lease_subscribe(dict(payload))
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)
        self.assertEqual(replay["after"], 0)
        # The live cursor is untouched by the replay.
        self.assertEqual(
            self.service.store._lease_subscriptions["I1"].after, 1)
        _, status = self.service.lease_subscription_ack(
            "I1", {"expected": 1, "after": 1})
        self.assertEqual(status, 200)

    def test_subscribe_different_filter_is_409_subscriber(self) -> None:
        self.service.lease_subscribe({
            "subscriber_id": "I1", "consumer_id": "c1",
            "lease_id": None})
        for changed in (
                {"subscriber_id": "I1", "consumer_id": "c2",
                 "lease_id": None},
                {"subscriber_id": "I1", "consumer_id": "c1",
                 "lease_id": "L1"},
                {"subscriber_id": "I1", "consumer_id": None,
                 "lease_id": None}):
            with self.subTest(changed=changed):
                with self.assertRaises(ServiceError) as caught:
                    self.service.lease_subscribe(changed)
                self.assertEqual(caught.exception.status_code, 409)
                self.assertEqual(caught.exception.field,
                                 "subscriber_id")

    def test_subscribe_validation(self) -> None:
        cases = [
            ({"subscriber_id": "", "consumer_id": None,
              "lease_id": None}, "subscriber_id"),
            ({"subscriber_id": 5, "consumer_id": None,
              "lease_id": None}, "subscriber_id"),
            ({"subscriber_id": "I1", "consumer_id": "",
              "lease_id": None}, "consumer_id"),
            ({"subscriber_id": "I1", "consumer_id": 5,
              "lease_id": None}, "consumer_id"),
            ({"subscriber_id": "I1", "consumer_id": True,
              "lease_id": None}, "consumer_id"),
            ({"subscriber_id": "I1", "consumer_id": None,
              "lease_id": ""}, "lease_id"),
            ({"subscriber_id": "I1", "consumer_id": None,
              "lease_id": []}, "lease_id"),
            ({"subscriber_id": "I1", "consumer_id": None}, "lease_id"),
            ({"consumer_id": None, "lease_id": None},
             "subscriber_id"),
            ({"subscriber_id": "I1", "consumer_id": None,
              "lease_id": None, "extra": 1}, "extra"),
            ([], "request_body"),
            (None, "request_body"),
        ]
        for payload, field in cases:
            with self.subTest(payload=payload):
                with self.assertRaises(ServiceError) as caught:
                    self.service.lease_subscribe(payload)
                self.assertEqual(caught.exception.status_code, 400)
                self.assertEqual(caught.exception.field, field)

    # -- GET page -------------------------------------------------------

    def test_page_filters_and_does_not_advance(self) -> None:
        self._seed()
        self.service.lease_subscribe({
            "subscriber_id": "I1", "consumer_id": "c1",
            "lease_id": None})
        body = self.service.lease_subscription_get("I1", 100)
        self.assertEqual(list(body),
                         ["subscriber_id", "events", "next_after",
                          "has_more"])
        self.assertEqual([event["seq"] for event in body["events"]],
                         [1, 2, 4])
        self.assertEqual(body["next_after"], 4)
        self.assertFalse(body["has_more"])
        for event in body["events"]:
            self.assertEqual(
                list(event),
                ["seq", "lease_id", "consumer_id", "type"])
        # Read-only: the cursor is still 0 and a repeat is identical.
        self.assertEqual(
            self.service.store._lease_subscriptions["I1"].after, 0)
        self.assertEqual(
            body, self.service.lease_subscription_get("I1", 100))

    def test_page_filters_by_lease(self) -> None:
        self._seed()
        self.service.lease_subscribe({
            "subscriber_id": "I2", "consumer_id": None,
            "lease_id": "L2"})
        body = self.service.lease_subscription_get("I2", 100)
        self.assertEqual([event["seq"] for event in body["events"]],
                         [3])

    def test_page_respects_limit_and_has_more(self) -> None:
        self._seed()
        self.service.lease_subscribe({
            "subscriber_id": "I1", "consumer_id": None,
            "lease_id": None})
        body = self.service.lease_subscription_get("I1", 2)
        self.assertEqual([event["seq"] for event in body["events"]],
                         [1, 2])
        self.assertEqual(body["next_after"], 2)
        self.assertTrue(body["has_more"])

    def test_page_unknown_subscription_is_404(self) -> None:
        with self.assertRaises(ServiceError) as caught:
            self.service.lease_subscription_get("ghost", 100)
        self.assertEqual(caught.exception.status_code, 404)
        self.assertEqual(caught.exception.field, "subscriber_id")

    # -- POST ack -------------------------------------------------------

    def test_ack_unknown_is_404_and_creates_nothing(self) -> None:
        with self.assertRaises(ServiceError) as caught:
            self.service.lease_subscription_ack(
                "ghost", {"expected": 0, "after": 0})
        self.assertEqual(caught.exception.status_code, 404)
        self.assertEqual(caught.exception.field, "subscriber_id")
        self.assertEqual(self.service.store._lease_subscriptions, {})

    def test_ack_equal_is_200_and_write_free(self) -> None:
        self.service.lease_subscribe({
            "subscriber_id": "I1", "consumer_id": None,
            "lease_id": None})
        body, status = self.service.lease_subscription_ack(
            "I1", {"expected": 0, "after": 0})
        self.assertEqual(status, 200)
        self.assertEqual(body, {"subscriber_id": "I1", "after": 0})
        self.assertEqual(list(body), ["subscriber_id", "after"])

    def test_ack_expected_conflict(self) -> None:
        self.service.lease_subscribe({
            "subscriber_id": "I1", "consumer_id": None,
            "lease_id": None})
        with self.assertRaises(ServiceError) as caught:
            self.service.lease_subscription_ack(
                "I1", {"expected": 1, "after": 1})
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "expected")

    def test_ack_forward_to_matching_event_is_201(self) -> None:
        self._seed()
        self.service.lease_subscribe({
            "subscriber_id": "I1", "consumer_id": "c1",
            "lease_id": None})
        body, status = self.service.lease_subscription_ack(
            "I1", {"expected": 0, "after": 1})
        self.assertEqual(status, 201)
        self.assertEqual(body, {"subscriber_id": "I1", "after": 1})
        # The next page starts past the acked seq, still filtered.
        page = self.service.lease_subscription_get("I1", 100)
        self.assertEqual([event["seq"] for event in page["events"]],
                         [2, 4])

    def test_ack_backwards_is_409_after(self) -> None:
        self._seed()
        self.service.lease_subscribe({
            "subscriber_id": "I1", "consumer_id": None,
            "lease_id": None})
        self.service.lease_subscription_ack(
            "I1", {"expected": 0, "after": 2})
        with self.assertRaises(ServiceError) as caught:
            self.service.lease_subscription_ack(
                "I1", {"expected": 2, "after": 1})
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "after")

    def test_ack_past_stream_is_409_after(self) -> None:
        self._seed()
        self.service.lease_subscribe({
            "subscriber_id": "I1", "consumer_id": None,
            "lease_id": None})
        with self.assertRaises(ServiceError) as caught:
            self.service.lease_subscription_ack(
                "I1", {"expected": 0, "after": 99})
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "after")

    def test_ack_non_matching_event_is_409_after(self) -> None:
        self._seed()
        # seq 3 belongs to c2/L2; a c1-bound subscription cannot ack it.
        self.service.lease_subscribe({
            "subscriber_id": "I1", "consumer_id": "c1",
            "lease_id": None})
        with self.assertRaises(ServiceError) as caught:
            self.service.lease_subscription_ack(
                "I1", {"expected": 0, "after": 3})
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "after")

        self.service.lease_subscribe({
            "subscriber_id": "I2", "consumer_id": None,
            "lease_id": "L1"})
        with self.assertRaises(ServiceError) as caught:
            self.service.lease_subscription_ack(
                "I2", {"expected": 0, "after": 3})
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "after")

    def test_ack_validation(self) -> None:
        self.service.lease_subscribe({
            "subscriber_id": "I1", "consumer_id": None,
            "lease_id": None})
        cases = [
            ({"expected": 0}, "after"),
            ({"after": 0}, "expected"),
            ({"expected": -1, "after": 0}, "expected"),
            ({"expected": 0, "after": -1}, "after"),
            ({"expected": True, "after": 0}, "expected"),
            ({"expected": 0, "after": 1.0}, "after"),
            ({"expected": "0", "after": 0}, "expected"),
            ({"expected": 0, "after": None}, "after"),
            ({"expected": 0, "after": 0, "x": 1}, "x"),
            ("nope", "request_body"),
        ]
        for payload, field in cases:
            with self.subTest(payload=payload):
                with self.assertRaises(ServiceError) as caught:
                    self.service.lease_subscription_ack("I1", payload)
                self.assertEqual(caught.exception.status_code, 400)
                self.assertEqual(caught.exception.field, field)


class LeaseSubsHTTPTest(LeaseSubsMixin, unittest.TestCase):
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

    def _request(self, method, path, raw=None):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        headers = {"Content-Type": "application/json"} if raw is not None \
            else {}
        conn.request(method, path, body=raw, headers=headers)
        response = conn.getresponse()
        data = response.read().decode("utf-8")
        conn.close()
        return response.status, (json.loads(data) if data else None), data

    def _post(self, path, payload):
        return self._request("POST", path, json.dumps(payload))

    def test_consume_empty_stream_anchors_over_http(self) -> None:
        status, body, raw = self._post(
            CONSUME_PATH,
            {"subscriber_id": "sub1", "expected": 0, "limit": 10})
        self.assertEqual(status, 201)
        self.assertEqual(body["events"], [])
        self.assertEqual(body["next_after"], 0)
        for earlier, later in (
                ('"subscriber_id"', '"events"'),
                ('"events"', '"next_after"'),
                ('"next_after"', '"has_more"')):
            self.assertLess(raw.index(earlier), raw.index(later))
        # The anchor is durable in memory and a repeat empty page is 200.
        status, body, _ = self._post(
            CONSUME_PATH,
            {"subscriber_id": "sub1", "expected": 0, "limit": 10})
        self.assertEqual(status, 200)

    def test_subscribe_and_page_and_ack_over_http(self) -> None:
        self._seed()
        status, body, raw = self._post(SUBS_PATH, {
            "subscriber_id": "I1", "consumer_id": "c1",
            "lease_id": None})
        self.assertEqual(status, 201)
        for earlier, later in (
                ('"subscriber_id"', '"consumer_id"'),
                ('"consumer_id"', '"lease_id"'),
                ('"lease_id"', '"after"')):
            self.assertLess(raw.index(earlier), raw.index(later))
        # Same filter replay 200.
        status, _, _ = self._post(SUBS_PATH, {
            "subscriber_id": "I1", "consumer_id": "c1",
            "lease_id": None})
        self.assertEqual(status, 200)
        # Filtered page.
        status, body, _ = self._request(
            "GET", SUBS_PATH + "/I1?limit=100")
        self.assertEqual(status, 200)
        self.assertEqual([event["seq"] for event in body["events"]],
                         [1, 2, 4])
        # Ack forward 201.
        status, body, _ = self._post(SUBS_PATH + "/I1/ack",
                                     {"expected": 0, "after": 1})
        self.assertEqual(status, 201)
        self.assertEqual(body, {"subscriber_id": "I1", "after": 1})
        # Equal ack 200.
        status, body, _ = self._post(SUBS_PATH + "/I1/ack",
                                     {"expected": 1, "after": 1})
        self.assertEqual(status, 200)

    def test_subscribe_conflict_over_http(self) -> None:
        self._post(SUBS_PATH, {"subscriber_id": "I1",
                               "consumer_id": "c1", "lease_id": None})
        status, body, _ = self._post(SUBS_PATH, {
            "subscriber_id": "I1", "consumer_id": None,
            "lease_id": None})
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "subscriber_id")

    def test_get_and_ack_unknown_404(self) -> None:
        status, body, _ = self._request("GET", SUBS_PATH + "/ghost")
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "subscriber_id")
        status, body, _ = self._post(SUBS_PATH + "/ghost/ack",
                                     {"expected": 0, "after": 0})
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "subscriber_id")

    def test_get_deeper_path_404(self) -> None:
        status, body, _ = self._request("GET", SUBS_PATH + "/a/b")
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "subscriber_id")
        status, body, _ = self._request(
            "POST", SUBS_PATH + "/a/b/ack", json.dumps(
                {"expected": 0, "after": 0}))
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "subscriber_id")

    def test_ack_deeper_path_with_bad_body_still_404(self) -> None:
        # Routing the unknown id happens before reading the JSON body.
        status, body, _ = self._request(
            "POST", SUBS_PATH + "/a/b/ack", "not-json")
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "subscriber_id")

    def test_percent_encoded_path(self) -> None:
        encoded_id = "a%2Fb"  # decodes to 'a/b' as one segment
        self._post(SUBS_PATH, {"subscriber_id": "a/b",
                               "consumer_id": None, "lease_id": None})
        status, body, _ = self._request(
            "GET", SUBS_PATH + "/" + encoded_id)
        self.assertEqual(status, 200)
        self.assertEqual(body["subscriber_id"], "a/b")
        status, _, _ = self._request("GET", SUBS_PATH + "/a%ZZ")
        self.assertEqual(status, 400)

    def test_subscribe_query_rejected(self) -> None:
        status, body, _ = self._request(
            "POST", SUBS_PATH + "?foo", json.dumps(
                {"subscriber_id": "I1", "consumer_id": None,
                 "lease_id": None}))
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "query")
        # A trailing bare '?' carries no parameter and is accepted.
        status, _, _ = self._request(
            "POST", SUBS_PATH + "?", json.dumps(
                {"subscriber_id": "I2", "consumer_id": None,
                 "lease_id": None}))
        self.assertEqual(status, 201)

    def test_get_query_validation(self) -> None:
        self._post(SUBS_PATH, {"subscriber_id": "I1",
                               "consumer_id": None, "lease_id": None})
        for query, field in (
                ("limit=0", "limit"),
                ("limit=101", "limit"),
                ("limit=abc", "limit"),
                ("limit=1&limit=2", "limit"),
                ("after=1", "query"),
                ("foo", "query")):
            with self.subTest(query=query):
                status, body, _ = self._request(
                    "GET", f"{SUBS_PATH}/I1?{query}")
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], field)

    def test_get_nonempty_body_rejected_before_query(self) -> None:
        self._post(SUBS_PATH, {"subscriber_id": "I1",
                               "consumer_id": None, "lease_id": None})
        status, body, _ = self._request(
            "GET", SUBS_PATH + "/I1?foo", "x")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "request_body")

    def test_ack_query_rejected(self) -> None:
        self._post(SUBS_PATH, {"subscriber_id": "I1",
                               "consumer_id": None, "lease_id": None})
        status, body, _ = self._request(
            "POST", SUBS_PATH + "/I1/ack?foo", json.dumps(
                {"expected": 0, "after": 0}))
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "query")

    def test_subscribe_bad_body(self) -> None:
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("POST", SUBS_PATH, body="nope",
                     headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        body = json.loads(response.read().decode("utf-8"))
        conn.close()
        self.assertEqual(response.status, 400)
        self.assertEqual(body["field"], "request_body")


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

    def test_empty_consume_anchor_persists_section_26(self) -> None:
        body, status = self._consume("sub1")
        self.assertEqual(status, 201)
        document = self._document()
        self.assertEqual(document["lease_event_cursors"],
                         [{"subscriber_id": "sub1", "after": 0}])
        # Restart: the anchor is read back and still answers 200 on a
        # later empty page.
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        body, status = restarted.event_gc_batch_lease_event_consume({
            "subscriber_id": "sub1", "expected": 0, "limit": 10})
        self.assertEqual(status, 200)
        self.assertEqual(body["events"], [])

    def test_section_27_follows_section_26(self) -> None:
        self._seed()
        self._consume("sub1", limit=10)
        self.service.lease_subscribe({
            "subscriber_id": "I1", "consumer_id": "c1",
            "lease_id": None})
        self.service.lease_subscription_ack(
            "I1", {"expected": 0, "after": 2})
        document = self._document()
        keys = list(document)
        self.assertEqual(
            keys.index("lease_subscriptions"),
            keys.index("lease_event_cursors") + 1)
        self.assertEqual(len(keys) - 3, 27)  # minus envelope keys
        self.assertEqual(document["lease_subscriptions"], [{
            "subscriber_id": "I1", "consumer_id": "c1",
            "lease_id": None, "after": 2}])
        for record in document["lease_subscriptions"]:
            self.assertEqual(
                list(record),
                ["subscriber_id", "consumer_id", "lease_id", "after"])

    def test_subscription_restarts_and_stays_consistent(self) -> None:
        self._seed()
        self.service.lease_subscribe({
            "subscriber_id": "I1", "consumer_id": None,
            "lease_id": "L2"})
        self.service.lease_subscription_ack(
            "I1", {"expected": 0, "after": 3})
        page = self.service.lease_subscription_get("I1", 100)
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        self.assertEqual(
            restarted.lease_subscription_get("I1", 100), page)
        self.assertEqual(
            restarted.lease_subscription_get("I1", 100)["events"], [])
        self.assertTrue(
            restarted.persistence_integrity()["consistent"])
        # Same filter replay still 200 after restart.
        _, status = restarted.lease_subscribe({
            "subscriber_id": "I1", "consumer_id": None,
            "lease_id": "L2"})
        self.assertEqual(status, 200)

    def test_equal_ack_consumes_no_generation(self) -> None:
        self.service.lease_subscribe({
            "subscriber_id": "I1", "consumer_id": None,
            "lease_id": None})
        generation = self.state_store.commit_seq
        before = self._document()
        self.service.lease_subscription_ack(
            "I1", {"expected": 0, "after": 0})
        self.service.lease_subscribe({
            "subscriber_id": "I1", "consumer_id": None,
            "lease_id": None})
        self.assertEqual(self.state_store.commit_seq, generation)
        self.assertEqual(self._document(), before)

    def test_create_and_forward_each_step_one_generation(self) -> None:
        self._seed()
        generation = self.state_store.commit_seq
        _, status = self.service.lease_subscribe({
            "subscriber_id": "I1", "consumer_id": "c1",
            "lease_id": None})
        self.assertEqual(status, 201)
        self.assertEqual(self.state_store.commit_seq,
                         generation + 1)
        _, status = self.service.lease_subscription_ack(
            "I1", {"expected": 0, "after": 1})
        self.assertEqual(status, 201)
        self.assertEqual(self.state_store.commit_seq,
                         generation + 2)

    def test_legacy_document_without_section_loads_empty(self) -> None:
        self._seed()
        document = self._document()
        document.pop("lease_subscriptions")
        document.pop("lease_event_cursors", None)
        document.pop("integrity_log_version", None)
        legacy_path = os.path.join(self.directory, "legacy.json")
        with open(legacy_path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        restarted = DeviceService()
        attach_persistence(restarted, legacy_path)  # must not raise
        with self.assertRaises(ServiceError) as caught:
            restarted.lease_subscription_get("I1", 100)
        self.assertEqual(caught.exception.status_code, 404)
        self.assertTrue(
            restarted.persistence_integrity()["consistent"])

    def _assert_refuses_startup(self, mutate):
        document = self._document()
        mutate(document)
        document.pop("integrity_log_version", None)
        bad_path = os.path.join(self.directory, "bad.json")
        with open(bad_path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        with self.assertRaises(StateFileError):
            attach_persistence(DeviceService(), bad_path)

    def test_malformed_subscription_refuses_startup(self) -> None:
        self._seed()
        self.service.lease_subscribe({
            "subscriber_id": "I1", "consumer_id": "c1",
            "lease_id": None})
        good_record = {
            "subscriber_id": "I1", "consumer_id": "c1",
            "lease_id": None, "after": 0}

        self._assert_refuses_startup(
            lambda doc: doc.__setitem__(
                "lease_subscriptions", "not-a-list"))

        def bad_key_order(doc):
            doc["lease_subscriptions"][0] = {
                "after": 0, "subscriber_id": "I1",
                "consumer_id": "c1", "lease_id": None}
        self._assert_refuses_startup(bad_key_order)

        def extra_key(doc):
            record = dict(good_record)
            record["extra"] = 1
            doc["lease_subscriptions"][0] = record
        self._assert_refuses_startup(extra_key)

        def empty_consumer(doc):
            doc["lease_subscriptions"][0]["consumer_id"] = ""
        self._assert_refuses_startup(empty_consumer)

        def bool_after(doc):
            doc["lease_subscriptions"][0]["after"] = True
        self._assert_refuses_startup(bool_after)

        def duplicate(doc):
            doc["lease_subscriptions"].append(dict(good_record))
        self._assert_refuses_startup(duplicate)

        # after=2 is c1/L1 (matches c1 filter): accepted at startup.
        document = self._document()
        document["lease_subscriptions"][0]["after"] = 2
        document.pop("integrity_log_version", None)
        good_path = os.path.join(self.directory, "good.json")
        with open(good_path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        attach_persistence(DeviceService(), good_path)  # no raise

    def test_non_matching_cursor_refuses_startup(self) -> None:
        self._seed()
        self.service.lease_subscribe({
            "subscriber_id": "I1", "consumer_id": "c1",
            "lease_id": None})
        # seq 3 belongs to c2/L2: contradicts the c1 frozen filter.
        self._assert_refuses_startup(
            lambda doc: doc["lease_subscriptions"][0].__setitem__(
                "after", 3))
        self._assert_refuses_startup(
            lambda doc: doc["lease_subscriptions"][0].__setitem__(
                "after", 99))
        # A lease filter contradicted by the target event.
        def lease_mismatch(doc):
            doc["lease_subscriptions"][0]["consumer_id"] = None
            doc["lease_subscriptions"][0]["lease_id"] = "L2"
            doc["lease_subscriptions"][0]["after"] = 1
        self._assert_refuses_startup(lease_mismatch)


if __name__ == "__main__":
    unittest.main()
