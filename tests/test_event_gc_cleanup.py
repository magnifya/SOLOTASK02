"""Tests for event-retention observation and expired-registration cleanup.

``GET /v1/event-gc/{device_id}`` pages a device's retention registrations
(revoke records included). The non-empty body is 400/request_body and the
only accepted query parameters are ``after``/``limit`` under the existing
event-paging contract (anything else is 400/query); an unknown device is
404/device_id. Records are ordered by ``consumer_id`` code point and paged
by a zero-based offset; the 200 body is ``device_id``, ``watermark``,
``first_seq``, ``last_seq``, ``consumers``, ``next_after``, ``has_more``
(the two seqs null when the event chain is empty), each item
``consumer_id``, ``seq``, ``updated_at``, ``active``, ``expires``, with
``active`` only for an unrevoked, unexpired lease whose seq is at the
watermark or above. The GET writes nothing.

``POST /v1/event-gc/{device_id}/cleanup-expired`` takes no query and no
body (400/query or 400/request_body), applies the same device rule, and
deletes the registrations whose ``expires`` is non-null and due (revoke
records are kept); the consumer is afterwards unregistered. Nothing
deleted answers 200, otherwise 201, body ``device_id``, ``removed``. The
cleanup shares the store lock with the event GC, checkpoints and event
appends; one commit adds one commit_seq and a data-file failure rolls
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


class EventGcHousekeepingMixin:
    def _build(self) -> None:
        self.service = DeviceService()
        self.service.store.add_device(Device("u", "alice", "ik"))
        self.service.store.add_device(Device(
            "u", "bob", "ik",
            prekeys=[SignedPreKey("pk1", "pubk1")]))
        self.sid = self.service.store.create_session(
            "alice", "bob", "pk1", "ek1").session_id

    def _op(self, job_id, op="queue", device="bob"):
        return self.service.inbox_job(
            {"device_id": device, "job_id": job_id, "op": op})

    def _touch(self, consumer="c1", device="bob"):
        return self.service.event_gc(
            device, {"consumer": consumer, "op": "touch", "seq": None})

    def _revoke(self, consumer="c1", device="bob"):
        return self.service.event_gc(
            device, {"consumer": consumer, "op": "revoke", "seq": None})

    def _prune(self, seq, device="bob"):
        return self.service.event_gc(
            device, {"consumer": None, "op": "prune", "seq": seq})

    def _checkpoint(self, consumer="c1", seq=None, device="bob"):
        return self.service.inbox_job_event_checkpoint(
            device, {"consumer_id": consumer, "seq": seq})

    def _observe(self, after=0, limit=100, device="bob"):
        return self.service.event_gc_observe(device, after, limit)

    def _cleanup(self, device="bob"):
        return self.service.event_gc_cleanup_expired(device)

    def _expire_lease(self, consumer="c1", device="bob"):
        record = self.service.store._redelivery_job_event_checkpoints[
            (device, consumer)]
        record.expires = "2020-01-01T00:00:00.000000+00:00"


class EventGcObserveTest(EventGcHousekeepingMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def test_empty_device_has_null_seqs_and_empty_page(self) -> None:
        self.service.store.add_device(Device("u", "dave", "ik"))
        body = self._observe(device="dave")
        self.assertEqual(list(body),
                         ["device_id", "watermark", "first_seq", "last_seq",
                          "consumers", "next_after", "has_more"])
        self.assertEqual(body, {
            "device_id": "dave", "watermark": 0, "first_seq": None,
            "last_seq": None, "consumers": [], "next_after": 0,
            "has_more": False})

    def test_top_level_and_item_key_order(self) -> None:
        self._op("J1")
        self._touch("c1")
        body = self._observe()
        self.assertEqual(list(body),
                         ["device_id", "watermark", "first_seq", "last_seq",
                          "consumers", "next_after", "has_more"])
        self.assertEqual(list(body["consumers"][0]),
                         ["consumer_id", "seq", "updated_at", "active",
                          "expires"])

    def test_watermark_and_chain_bounds(self) -> None:
        for job_id in ("J1", "J2", "J3"):
            self._op(job_id)
        self._touch("c1")
        self._checkpoint("c1", 3)
        body = self._observe()
        self.assertEqual(body["watermark"], 0)
        self.assertEqual(body["first_seq"], 1)
        self.assertEqual(body["last_seq"], 3)
        self._prune(2)
        body = self._observe()
        self.assertEqual(body["watermark"], 2)
        self.assertEqual(body["first_seq"], 3)
        self.assertEqual(body["last_seq"], 3)
        # Pruning the whole chain leaves both bounds null.
        self._prune(3)
        body = self._observe()
        self.assertEqual(body["watermark"], 3)
        self.assertIsNone(body["first_seq"])
        self.assertIsNone(body["last_seq"])

    def test_records_ordered_by_consumer_id_code_point(self) -> None:
        for consumer in ("b", "a", "_", "中"):
            self._touch(consumer)
        body = self._observe()
        self.assertEqual([item["consumer_id"] for item in body["consumers"]],
                         ["_", "a", "b", "中"])

    def test_offset_pagination(self) -> None:
        for consumer in ("c1", "c2", "c3"):
            self._touch(consumer)
        page = self._observe(after=0, limit=2)
        self.assertEqual([i["consumer_id"] for i in page["consumers"]],
                         ["c1", "c2"])
        self.assertEqual(page["next_after"], 2)
        self.assertTrue(page["has_more"])
        page = self._observe(after=2, limit=2)
        self.assertEqual([i["consumer_id"] for i in page["consumers"]],
                         ["c3"])
        self.assertEqual(page["next_after"], 3)
        self.assertFalse(page["has_more"])
        # An empty page echoes the offset and has no tail.
        page = self._observe(after=3, limit=2)
        self.assertEqual(page["consumers"], [])
        self.assertEqual(page["next_after"], 3)
        self.assertFalse(page["has_more"])
        # An offset beyond the end is just an empty page there.
        page = self._observe(after=9, limit=2)
        self.assertEqual(page["consumers"], [])
        self.assertEqual(page["next_after"], 9)
        self.assertFalse(page["has_more"])

    def test_limit_one_walks_every_record(self) -> None:
        for consumer in ("c1", "c2", "c3", "c4"):
            self._touch(consumer)
        seen = []
        after = 0
        while True:
            page = self._observe(after=after, limit=1)
            if not page["consumers"]:
                break
            seen.append(page["consumers"][0]["consumer_id"])
            after = page["next_after"]
        self.assertEqual(seen, ["c1", "c2", "c3", "c4"])
        self.assertEqual(after, 4)

    def test_active_only_for_unrevoked_unexpired_current_lease(self) -> None:
        self._op("J1")
        self._op("J2")
        self._touch("fresh")           # active, future lease
        self._touch("revoked")
        self._revoke("revoked")        # active False, expires null, listed
        self._touch("expired")
        self._expire_lease("expired")  # active False, expires past, listed
        # A lease-less checkpoint (never touched) stays active by itself.
        self._checkpoint("leaseless", 1)
        body = self._observe()
        by_id = {item["consumer_id"]: item for item in body["consumers"]}
        self.assertEqual(set(by_id), {"fresh", "revoked", "expired",
                                      "leaseless"})
        self.assertTrue(by_id["fresh"]["active"])
        self.assertIsNotNone(by_id["fresh"]["expires"])
        self.assertFalse(by_id["revoked"]["active"])
        self.assertIsNone(by_id["revoked"]["expires"])
        self.assertFalse(by_id["expired"]["active"])
        self.assertEqual(by_id["expired"]["expires"],
                         "2020-01-01T00:00:00.000000+00:00")
        self.assertTrue(by_id["leaseless"]["active"])
        self.assertIsNone(by_id["leaseless"]["expires"])
        self.assertEqual(by_id["leaseless"]["seq"], 1)

    def test_active_false_once_seq_falls_below_watermark(self) -> None:
        self._op("J1")
        self._op("J2")
        self._touch("leader")
        self._touch("laggard")
        self._checkpoint("leader", 2)
        # The laggard's lease expires; the leader then lets pruning pass
        # the laggard's registered seq (0), so it is inactive on two
        # grounds while its record remains observable.
        self._expire_lease("laggard")
        self._prune(2)
        by_id = {item["consumer_id"]: item
                 for item in self._observe()["consumers"]}
        self.assertTrue(by_id["leader"]["active"])
        self.assertFalse(by_id["laggard"]["active"])

    def test_touch_reactivates_a_revoked_consumer(self) -> None:
        self._op("J1")
        self._touch("c1")
        self._revoke("c1")
        self.assertFalse(self._observe()["consumers"][0]["active"])
        self._touch("c1")
        self.assertTrue(self._observe()["consumers"][0]["active"])

    def test_unknown_device_is_404(self) -> None:
        with self.assertRaises(ServiceError) as caught:
            self._observe(device="ghost")
        self.assertEqual(caught.exception.status_code, 404)
        self.assertEqual(caught.exception.field, "device_id")

    def test_revoked_device_stays_observable(self) -> None:
        self._op("J1")
        self._touch()
        self.service.revoke_device("bob")
        body = self._observe()
        self.assertEqual(body["device_id"], "bob")
        self.assertEqual([i["consumer_id"] for i in body["consumers"]],
                         ["c1"])

    def test_pagination_parameter_validation(self) -> None:
        for kwargs, field in (
                ({"after": -1}, "after"),
                ({"after": True}, "after"),
                ({"after": 1.0}, "after"),
                ({"after": "1"}, "after"),
                ({"after": 2**63}, "after"),
                ({"limit": 0}, "limit"),
                ({"limit": 101}, "limit"),
                ({"limit": True}, "limit"),
                ({"limit": 1.5}, "limit"),
                ({"limit": "2"}, "limit")):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ServiceError) as caught:
                    self._observe(**kwargs)
                self.assertEqual(caught.exception.status_code, 400)
                self.assertEqual(caught.exception.field, field)


class EventGcCleanupTest(EventGcHousekeepingMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def test_no_records_is_200_noop(self) -> None:
        self.service.store.add_device(Device("u", "dave", "ik"))
        body, status = self._cleanup(device="dave")
        self.assertEqual(status, 200)
        self.assertEqual(list(body), ["device_id", "removed"])
        self.assertEqual(body, {"device_id": "dave", "removed": 0})

    def test_nothing_expired_is_200_and_keeps_records(self) -> None:
        self._op("J1")
        self._touch("c1")
        self._checkpoint("c2", 1)  # lease-less, never expires
        body, status = self._cleanup()
        self.assertEqual(status, 200)
        self.assertEqual(body, {"device_id": "bob", "removed": 0})
        self.assertEqual(
            set(self.service.store._redelivery_job_event_checkpoints),
            {("bob", "c1"), ("bob", "c2")})

    def test_expired_registrations_are_deleted(self) -> None:
        self._op("J1")
        self._touch("c1")
        self._touch("c2")
        self._touch("c3")
        self._expire_lease("c1")
        self._expire_lease("c3")
        body, status = self._cleanup()
        self.assertEqual(status, 201)
        self.assertEqual(body, {"device_id": "bob", "removed": 2})
        self.assertEqual(
            set(self.service.store._redelivery_job_event_checkpoints),
            {("bob", "c2")})

    def test_revoke_records_are_never_deleted(self) -> None:
        self._op("J1")
        self._touch("c1")
        self._touch("c2")
        self._revoke("c1")
        self._expire_lease("c2")
        body, status = self._cleanup()
        self.assertEqual(status, 201)
        self.assertEqual(body["removed"], 1)
        # The revoke record (active false, expires null) survives and is
        # still observable; only the expired lease was removed.
        self.assertIn(("bob", "c1"),
                      self.service.store._redelivery_job_event_checkpoints)
        self.assertNotIn(("bob", "c2"),
                         self.service.store._redelivery_job_event_checkpoints)
        observed = {i["consumer_id"]: i for i in self._observe()["consumers"]}
        self.assertIn("c1", observed)
        self.assertFalse(observed["c1"]["active"])
        self.assertIsNone(observed["c1"]["expires"])

    def test_deleted_consumer_is_unregistered(self) -> None:
        self._op("J1")
        self._touch("c1")
        self._expire_lease("c1")
        self._cleanup()
        # No record: a revoke is 404/consumer and a fresh touch 201.
        with self.assertRaises(ServiceError) as caught:
            self._revoke("c1")
        self.assertEqual(caught.exception.status_code, 404)
        self.assertEqual(caught.exception.field, "consumer")
        body, status = self._touch("c1")
        self.assertEqual(status, 201)
        self.assertEqual(body["seq"], 0)

    def test_replay_is_200_noop(self) -> None:
        self._op("J1")
        self._touch("c1")
        self._expire_lease("c1")
        _, status = self._cleanup()
        self.assertEqual(status, 201)
        body, status = self._cleanup()
        self.assertEqual(status, 200)
        self.assertEqual(body["removed"], 0)

    def test_other_devices_are_untouched(self) -> None:
        self._op("J1", device="bob")
        self._touch("c1", device="bob")
        self._expire_lease("c1", device="bob")
        # A second registered device with its own expired lease.
        self.service.store.add_device(Device(
            "u", "dave", "ik",
            prekeys=[SignedPreKey("pk2", "pubk2")]))
        self.service.store.create_session("alice", "dave", "pk2", "ek2")
        self._op("J9", device="dave")
        self._touch("c9", device="dave")
        self._expire_lease("c9", device="dave")
        body, status = self._cleanup(device="bob")
        self.assertEqual(status, 201)
        self.assertEqual(body, {"device_id": "bob", "removed": 1})
        self.assertIn(("dave", "c9"),
                      self.service.store._redelivery_job_event_checkpoints)

    def test_unknown_device_is_404(self) -> None:
        with self.assertRaises(ServiceError) as caught:
            self._cleanup(device="ghost")
        self.assertEqual(caught.exception.status_code, 404)
        self.assertEqual(caught.exception.field, "device_id")

    def test_revoked_device_stays_cleanable(self) -> None:
        self._op("J1")
        self._touch("c1")
        self._expire_lease("c1")
        self.service.revoke_device("bob")
        body, status = self._cleanup()
        self.assertEqual(status, 201)
        self.assertEqual(body["removed"], 1)


class EventGcHousekeepingHTTPTest(EventGcHousekeepingMixin,
                                  unittest.TestCase):
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

    def _request(self, path, raw=None, method="GET"):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        headers = {"Content-Type": "application/json"} if raw is not None \
            else {}
        conn.request(method, path, body=raw, headers=headers)
        response = conn.getresponse()
        data = response.read().decode("utf-8")
        conn.close()
        return response.status, (json.loads(data) if data else None), data

    def _observe_http(self, query="", device="bob", raw=None):
        return self._request(f"/v1/event-gc/{device}{query}", raw=raw)

    def _cleanup_http(self, device="bob", query="", raw=None):
        return self._request(
            f"/v1/event-gc/{device}/cleanup-expired{query}",
            raw=raw, method="POST")

    def _queue_http(self, job_id):
        status, _, _ = self._request(
            "/v1/inbox-jobs",
            json.dumps({"device_id": "bob", "job_id": job_id,
                        "op": "queue"}), method="POST")
        self.assertEqual(status, 201)

    def _touch_http(self, consumer="c1"):
        status, _, _ = self._request(
            "/v1/event-gc/bob",
            json.dumps({"consumer": consumer, "op": "touch", "seq": None}),
            method="POST")
        self.assertEqual(status, 201)

    # -- observation ------------------------------------------------------

    def test_observe_flow_with_key_order(self) -> None:
        self._queue_http("J1")
        self._queue_http("J2")
        self._touch_http("c1")
        status, body, raw = self._observe_http()
        self.assertEqual(status, 200)
        self.assertEqual(list(body),
                         ["device_id", "watermark", "first_seq", "last_seq",
                          "consumers", "next_after", "has_more"])
        self.assertEqual(body["watermark"], 0)
        self.assertEqual(body["first_seq"], 1)
        self.assertEqual(body["last_seq"], 2)
        self.assertEqual(list(body["consumers"][0]),
                         ["consumer_id", "seq", "updated_at", "active",
                          "expires"])
        for earlier, later in (
                ('"device_id"', '"watermark"'),
                ('"watermark"', '"first_seq"'),
                ('"first_seq"', '"last_seq"'),
                ('"last_seq"', '"consumers"'),
                ('"consumers"', '"next_after"'),
                ('"next_after"', '"has_more"'),
                ('"consumer_id"', '"seq"'),
                ('"seq"', '"updated_at"'),
                ('"updated_at"', '"active"'),
                ('"active"', '"expires"')):
            self.assertLess(raw.index(earlier), raw.index(later))

    def test_observe_repeated_gets_are_byte_identical(self) -> None:
        self._queue_http("J1")
        self._touch_http("c1")
        _, _, first = self._observe_http()
        _, _, second = self._observe_http()
        self.assertEqual(first, second)

    def test_observe_nonempty_body_is_400_request_body(self) -> None:
        for raw in ("{}", "junk", "null"):
            with self.subTest(raw=raw):
                status, body, _ = self._observe_http(raw=raw)
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], "request_body")
                self.assertEqual(list(body), ["message", "field"])

    def test_observe_unknown_query_is_400_query(self) -> None:
        for query in ("?x=1", "?foo", "?after=1&x=", "?limit=5&state=all"):
            with self.subTest(query=query):
                status, body, _ = self._observe_http(query=query)
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], "query")

    def test_observe_empty_query_string_is_allowed(self) -> None:
        status, _, _ = self._observe_http(query="?")
        self.assertEqual(status, 200)

    def test_observe_after_parameter_validation(self) -> None:
        for value in ("-1", "1.5", "%2B1", "x", "1a", "%201", "1%20",
                      "", "9223372036854775808"):
            with self.subTest(value=value):
                status, body, _ = self._observe_http(query=f"?after={value}")
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], "after")
        status, body, _ = self._observe_http(query="?after=1&after=2")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "after")
        status, _, _ = self._observe_http(query="?after=01")
        self.assertEqual(status, 200)
        status, _, _ = self._observe_http(
            query="?after=9223372036854775807")
        self.assertEqual(status, 200)

    def test_observe_limit_parameter_validation(self) -> None:
        for value in ("0", "101", "-1", "x", "1.0", "%2B1", ""):
            with self.subTest(value=value):
                status, body, _ = self._observe_http(query=f"?limit={value}")
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], "limit")
        status, body, _ = self._observe_http(query="?limit=1&limit=2")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "limit")
        status, _, _ = self._observe_http(query="?limit=100")
        self.assertEqual(status, 200)

    def test_observe_pagination_over_http(self) -> None:
        for consumer in ("c1", "c2", "c3"):
            self._touch_http(consumer)
        status, body, _ = self._observe_http("?after=0&limit=2")
        self.assertEqual(status, 200)
        self.assertEqual([i["consumer_id"] for i in body["consumers"]],
                         ["c1", "c2"])
        self.assertEqual(body["next_after"], 2)
        self.assertTrue(body["has_more"])
        status, body, _ = self._observe_http("?after=2&limit=2")
        self.assertEqual([i["consumer_id"] for i in body["consumers"]],
                         ["c3"])
        self.assertEqual(body["next_after"], 3)
        self.assertFalse(body["has_more"])

    def test_observe_unknown_device_is_404(self) -> None:
        status, body, _ = self._observe_http(device="ghost")
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "device_id")

    def test_get_on_cleanup_path_is_404(self) -> None:
        status, body, _ = self._request(
            "/v1/event-gc/bob/cleanup-expired")
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "device_id")

    # -- cleanup ----------------------------------------------------------

    def test_cleanup_nothing_is_200_with_key_order(self) -> None:
        self._queue_http("J1")
        self._touch_http("c1")
        status, body, raw = self._cleanup_http()
        self.assertEqual(status, 200)
        self.assertEqual(list(body), ["device_id", "removed"])
        self.assertEqual(body, {"device_id": "bob", "removed": 0})
        self.assertLess(raw.index('"device_id"'), raw.index('"removed"'))

    def test_cleanup_expired_is_201(self) -> None:
        self._queue_http("J1")
        self._touch_http("c1")
        self._touch_http("c2")
        self._expire_lease("c1")
        status, body, _ = self._cleanup_http()
        self.assertEqual(status, 201)
        self.assertEqual(body, {"device_id": "bob", "removed": 1})

    def test_cleanup_query_is_400_query(self) -> None:
        for query in ("?x=1", "?after=0", "?"):
            with self.subTest(query=query):
                status, body, _ = self._cleanup_http(query=query)
                if query == "?":
                    self.assertEqual(status, 200)
                else:
                    self.assertEqual(status, 400)
                    self.assertEqual(body["field"], "query")

    def test_cleanup_body_is_400_request_body(self) -> None:
        for raw in ("{}", "null", "junk"):
            with self.subTest(raw=raw):
                status, body, _ = self._cleanup_http(raw=raw)
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], "request_body")

    def test_cleanup_unknown_device_is_404(self) -> None:
        status, body, _ = self._cleanup_http(device="ghost")
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "device_id")

    def test_cleanup_deeper_path_is_404(self) -> None:
        status, body, _ = self._request(
            "/v1/event-gc/bob/cleanup-expired/extra", method="POST")
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "device_id")


class EventGcHousekeepingPersistenceTest(EventGcHousekeepingMixin,
                                         unittest.TestCase):
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

    def test_observe_consumes_no_generation(self) -> None:
        self._op("J1")
        self._touch("c1")
        generation = self.state_store.commit_seq
        for _ in range(3):
            self._observe()
        self.assertEqual(self.state_store.commit_seq, generation)
        self.assertTrue(self.service.persistence_integrity()["consistent"])

    def test_noop_cleanup_consumes_no_generation(self) -> None:
        self._op("J1")
        self._touch("c1")
        generation = self.state_store.commit_seq
        body, status = self._cleanup()
        self.assertEqual(status, 200)
        self.assertEqual(body["removed"], 0)
        self.assertEqual(self.state_store.commit_seq, generation)

    def test_cleanup_commits_one_generation_and_only_checkpoints(self) -> None:
        self._op("J1")
        self._op("J2")
        self._touch("c1")
        self._checkpoint("c1", 2)
        self._touch("c2")
        before = self._document()
        self._expire_lease("c2")
        # Anchor the expired lease durably via another commit.
        self._touch("c1")
        generation = self.state_store.commit_seq
        body, status = self._cleanup()
        self.assertEqual(status, 201)
        self.assertEqual(body["removed"], 1)
        self.assertEqual(self.state_store.commit_seq, generation + 1)
        after = self._document()
        self.assertEqual(
            [item["consumer_id"]
             for item in after["redelivery_job_event_checkpoints"]],
            ["c1"])
        # The event chain and the watermark section are untouched.
        self.assertEqual(after["redelivery_job_events"],
                         before["redelivery_job_events"])
        self.assertEqual(after["event_gc"], before["event_gc"])
        self.assertTrue(self.service.persistence_integrity()["consistent"])

    def test_revoke_record_persists_through_cleanup_and_restart(self) -> None:
        self._op("J1")
        self._touch("c1")
        self._touch("c2")
        self._revoke("c1")
        self._expire_lease("c2")
        body, status = self._cleanup()
        self.assertEqual((status, body["removed"]), (201, 1))
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        page = restarted.event_gc_observe("bob", 0, 100)
        by_id = {i["consumer_id"]: i for i in page["consumers"]}
        self.assertEqual(set(by_id), {"c1"})
        self.assertFalse(by_id["c1"]["active"])
        self.assertIsNone(by_id["c1"]["expires"])
        # The surviving chain and watermark survive the restart unchanged.
        self.assertEqual(page["first_seq"], 1)
        self.assertEqual(page["last_seq"], 1)
        self.assertTrue(restarted.persistence_integrity()["consistent"])

    def test_save_failure_rolls_cleanup_back(self) -> None:
        import e2ee_backend.persistence as persistence_mod

        self._op("J1")
        self._touch("c1")
        self._touch("c2")
        self._expire_lease("c1")
        # Make the expired lease durable via a second successful commit
        # (renewing c2), so last-good already contains the expired record.
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
        # Nothing advanced: no generation consumed, the expired record is
        # back in memory and still present durably on disk.
        self.assertEqual(self.state_store.commit_seq, generation)
        self.assertIn(("bob", "c1"),
                      self.service.store._redelivery_job_event_checkpoints)
        document = self._document()
        self.assertIn("c1", [
            item["consumer_id"]
            for item in document["redelivery_job_event_checkpoints"]])
        # The cleanup can be retried and now commits.
        body, status = self._cleanup()
        self.assertEqual(status, 201)
        self.assertEqual(body["removed"], 1)
        self.assertEqual(self.state_store.commit_seq, generation + 1)


if __name__ == "__main__":
    unittest.main()
