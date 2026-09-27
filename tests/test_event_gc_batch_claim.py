"""Tests for the batch-cleanup audit claim endpoint.

``POST /v1/event-gc-batch/claim`` atomically reads one page of the
batch-cleanup audit chain (the committed
``event_gc_batch_cleanup_requests`` records, in commit order) for one
consumer under a 30-second lease without advancing the consumer's
checkpoint. The call takes no query parameters (any -> 400/query); the
body carries exactly ``consumer_id`` and ``lease_id`` (non-empty
strings), ``expected`` (a non-boolean integer in 0..2^63-1) and
``limit`` (a non-boolean integer in 1..100); a bad/non-object body is
400/request_body, a missing/wrongly typed/extra field is 400 with that
field. The ``lease_id`` is judged first: an exact replay (same consumer,
expected and limit) returns the first response byte-identically with
200 whether the lease is active, acknowledged or expired; a different
consumer or payload is 409/lease_id. For a fresh id, an ``expected``
that is not the consumer's current checkpoint is 409/expected and a
consumer that already holds another unexpired, unacknowledged lease is
409/consumer_id. An empty page answers 200 and occupies no lease id; a
non-empty page answers 201, reads but never advances the checkpoint, and
records the lease. A data-file failure is 503/data_file with the lease,
the commit generation and both files rolled back. Success keys are
``consumer_id``, ``lease_id``, ``records``, ``next_after`` and
``expires`` in that order; each record keeps the six-key audit wire
view. Non-empty leases persist in the version-1 ``cleanup_leases``
section immediately after ``cleanup_checkpoints``.
"""
import copy
import json
import os
import shutil
import tempfile
import threading
import unittest
from datetime import datetime, timedelta, timezone
from http.client import HTTPConnection

from e2ee_backend.models import Device
from e2ee_backend.service import DeviceService, ServiceError
from e2ee_backend.http_app import create_server
from e2ee_backend.persistence import (
    PersistenceUnavailable,
    attach_persistence,
)

PATH = "/v1/event-gc-batch/claim"


class ClaimMixin:
    def _build(self) -> None:
        self.service = DeviceService()
        self.service.store.add_device(Device("u", "alice", "ik"))
        self.service.store.add_device(Device("u", "bob", "ik"))

    def _commit(self, request_id, device_ids=None):
        # Every idempotent commit appends exactly one audit record.
        return self.service.event_gc_cleanup_expired_batch({
            "mode": "commit", "device_ids": device_ids or ["bob"],
            "after": 0, "limit": 100, "request_id": request_id})

    def _claim(self, consumer="c1", lease="L1", expected=0, limit=100):
        return self.service.event_gc_batch_claim({
            "consumer_id": consumer, "lease_id": lease,
            "expected": expected, "limit": limit})

    def _consume(self, consumer="c1", expected=0, limit=100):
        return self.service.event_gc_batch_consume({
            "consumer_id": consumer, "expected": expected, "limit": limit})

    def _checkpoint_read(self, consumer="c1"):
        body, _ = self.service.event_gc_batch_checkpoint(
            {"consumer_id": consumer, "expected": None, "after": None})
        return body


class ClaimServiceTest(ClaimMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def test_empty_chain_empty_page_does_not_occupy_id(self) -> None:
        body, status = self._claim(lease="L1")
        self.assertEqual(status, 200)
        self.assertEqual(list(body),
                         ["consumer_id", "lease_id", "records",
                          "next_after", "expires"])
        self.assertEqual(body["consumer_id"], "c1")
        self.assertEqual(body["lease_id"], "L1")
        self.assertEqual(body["records"], [])
        self.assertEqual(body["next_after"], 0)
        self.assertTrue(body["expires"].endswith("+00:00"))
        # The id stays free: with records later the same id claims 201.
        self._commit("r1")
        body, status = self._claim(lease="L1")
        self.assertEqual(status, 201)
        self.assertEqual([r["request_id"] for r in body["records"]],
                         ["r1"])

    def test_nonempty_page_claims_without_advancing_checkpoint(self) -> None:
        self._commit("r1")
        self._commit("r2")
        body, status = self._claim(expected=0, limit=1)
        self.assertEqual(status, 201)
        self.assertEqual(list(body),
                         ["consumer_id", "lease_id", "records",
                          "next_after", "expires"])
        self.assertEqual([r["request_id"] for r in body["records"]],
                         ["r1"])
        for record in body["records"]:
            self.assertEqual(list(record),
                             ["request_id", "device_ids", "after",
                              "limit", "status", "response"])
        self.assertEqual(body["next_after"], 1)
        self.assertTrue(body["expires"].endswith("+00:00"))
        # The claim reads but never advances the checkpoint.
        self.assertEqual(self._checkpoint_read()["after"], 0)

    def test_expires_is_about_30_seconds_ahead(self) -> None:
        before = datetime.now(timezone.utc)
        self._commit("r1")
        body, status = self._claim(limit=100)
        after = datetime.now(timezone.utc)
        self.assertEqual(status, 201)
        expires = datetime.fromisoformat(body["expires"])
        self.assertGreaterEqual(
            expires, before + timedelta(seconds=30) - timedelta(seconds=1))
        self.assertLessEqual(
            expires, after + timedelta(seconds=30) + timedelta(seconds=1))

    def test_paging_with_limit(self) -> None:
        for index in range(3):
            self._commit(f"r{index}")
        first, status = self._claim(lease="L1", expected=0, limit=2)
        self.assertEqual(status, 201)
        self.assertEqual([r["request_id"] for r in first["records"]],
                         ["r0", "r1"])
        self.assertEqual(first["next_after"], 2)

    def test_exact_replay_is_200_byte_identical_and_write_free(self) -> None:
        self._commit("r1")
        first, status = self._claim(lease="L1", expected=0, limit=1)
        self.assertEqual(status, 201)
        replay, status = self._claim(lease="L1", expected=0, limit=1)
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)

    def test_replay_stays_frozen_after_checkpoint_advanced(self) -> None:
        self._commit("r1")
        first, status = self._claim(lease="L1", expected=0, limit=100)
        self.assertEqual(status, 201)
        self._consume(expected=0, limit=100)
        replay, status = self._claim(lease="L1", expected=0, limit=100)
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)

    def test_same_lease_id_other_consumer_is_409(self) -> None:
        self._commit("r1")
        self._claim(consumer="c1", lease="shared", expected=0, limit=1)
        with self.assertRaises(ServiceError) as caught:
            self._claim(consumer="c2", lease="shared", expected=0, limit=1)
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_same_lease_id_changed_payload_is_409(self) -> None:
        self._commit("r1")
        self._commit("r2")
        self._claim(lease="L1", expected=0, limit=1)
        for payload in (
                {"consumer_id": "c1", "lease_id": "L1",
                 "expected": 0, "limit": 2},
                {"consumer_id": "c1", "lease_id": "L1",
                 "expected": 1, "limit": 1}):
            with self.subTest(payload=payload):
                with self.assertRaises(ServiceError) as caught:
                    self.service.event_gc_batch_claim(payload)
                self.assertEqual(caught.exception.status_code, 409)
                self.assertEqual(caught.exception.field, "lease_id")

    def test_lease_id_conflict_precedes_other_conflicts(self) -> None:
        self._commit("r1")
        # L1 is live for c1; reusing it from another consumer with a bad
        # expected still reports lease_id (it is judged first).
        self._claim(consumer="c1", lease="L1", expected=0, limit=1)
        with self.assertRaises(ServiceError) as caught:
            self._claim(consumer="c2", lease="L1", expected=5, limit=1)
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_expected_mismatch_is_409(self) -> None:
        self._commit("r1")
        with self.assertRaises(ServiceError) as caught:
            self._claim(lease="L1", expected=1)
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "expected")
        # After consuming one record the checkpoint is 1.
        self._consume(expected=0, limit=1)
        with self.assertRaises(ServiceError) as caught:
            self._claim(lease="L1", expected=0)
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "expected")

    def test_one_live_lease_per_consumer(self) -> None:
        self._commit("r1")
        _, status = self._claim(consumer="c1", lease="L1", limit=1)
        self.assertEqual(status, 201)
        with self.assertRaises(ServiceError) as caught:
            self._claim(consumer="c1", lease="L2", expected=0, limit=1)
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "consumer_id")

    def test_different_consumers_lease_independently(self) -> None:
        self._commit("r1")
        _, status = self._claim(consumer="c1", lease="L1", limit=1)
        self.assertEqual(status, 201)
        body, status = self._claim(consumer="c2", lease="LX", limit=100)
        self.assertEqual(status, 201)
        self.assertEqual([r["request_id"] for r in body["records"]],
                         ["r1"])
        # Neither claim moved a checkpoint.
        self.assertEqual(self._checkpoint_read("c1")["after"], 0)
        self.assertEqual(self._checkpoint_read("c2")["after"], 0)

    def test_acknowledged_lease_allows_reclaim_with_new_id(self) -> None:
        self._commit("r1")
        self._commit("r2")
        first, status = self._claim(consumer="c1", lease="L1",
                                    expected=0, limit=1)
        self.assertEqual(status, 201)
        # A concurrent different id is blocked while the lease is live.
        with self.assertRaises(ServiceError) as caught:
            self._claim(consumer="c1", lease="L2", expected=0, limit=1)
        self.assertEqual(caught.exception.field, "consumer_id")
        # consume reaches next_after: the lease is acknowledged.
        self._consume(expected=0, limit=1)
        second, status = self._claim(consumer="c1", lease="L2",
                                     expected=1, limit=100)
        self.assertEqual(status, 201)
        self.assertEqual([r["request_id"] for r in second["records"]],
                         ["r2"])
        self.assertEqual(second["next_after"], 2)
        # The old id keeps replaying its frozen first response.
        replay, status = self._claim(consumer="c1", lease="L1",
                                     expected=0, limit=1)
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)

    def test_checkpoint_advance_acknowledges_lease(self) -> None:
        self._commit("r1")
        self._claim(consumer="c1", lease="L1", expected=0, limit=1)
        # checkpoint advance to exactly next_after acknowledges.
        _, status = self.service.event_gc_batch_checkpoint({
            "consumer_id": "c1", "expected": 0, "after": 1})
        self.assertEqual(status, 201)
        _, status = self._claim(consumer="c1", lease="L2",
                                expected=1, limit=100)
        self.assertEqual(status, 200)  # empty page, id free afterwards

    def test_payload_validation(self) -> None:
        for payload, field in (
                (None, "request_body"),
                ([], "request_body"),
                ("x", "request_body"),
                ({"lease_id": "L1", "expected": 0, "limit": 1},
                 "consumer_id"),
                ({"consumer_id": "", "lease_id": "L1", "expected": 0,
                  "limit": 1}, "consumer_id"),
                ({"consumer_id": 1, "lease_id": "L1", "expected": 0,
                  "limit": 1}, "consumer_id"),
                ({"consumer_id": "c1", "expected": 0, "limit": 1},
                 "lease_id"),
                ({"consumer_id": "c1", "lease_id": "", "expected": 0,
                  "limit": 1}, "lease_id"),
                ({"consumer_id": "c1", "lease_id": 0, "expected": 0,
                  "limit": 1}, "lease_id"),
                ({"consumer_id": "c1", "lease_id": "L1", "limit": 1},
                 "expected"),
                ({"consumer_id": "c1", "lease_id": "L1", "expected": None,
                  "limit": 1}, "expected"),
                ({"consumer_id": "c1", "lease_id": "L1", "expected": True,
                  "limit": 1}, "expected"),
                ({"consumer_id": "c1", "lease_id": "L1", "expected": -1,
                  "limit": 1}, "expected"),
                ({"consumer_id": "c1", "lease_id": "L1", "expected": 1.0,
                  "limit": 1}, "expected"),
                ({"consumer_id": "c1", "lease_id": "L1", "expected": 2**63,
                  "limit": 1}, "expected"),
                ({"consumer_id": "c1", "lease_id": "L1", "expected": 0},
                 "limit"),
                ({"consumer_id": "c1", "lease_id": "L1", "expected": 0,
                  "limit": None}, "limit"),
                ({"consumer_id": "c1", "lease_id": "L1", "expected": 0,
                  "limit": True}, "limit"),
                ({"consumer_id": "c1", "lease_id": "L1", "expected": 0,
                  "limit": 0}, "limit"),
                ({"consumer_id": "c1", "lease_id": "L1", "expected": 0,
                  "limit": 101}, "limit"),
                ({"consumer_id": "c1", "lease_id": "L1", "expected": 0,
                  "limit": -1}, "limit"),
                ({"consumer_id": "c1", "lease_id": "L1", "expected": 0,
                  "limit": 1.0}, "limit"),
                ({"consumer_id": "c1", "lease_id": "L1", "expected": 0,
                  "limit": "1"}, "limit"),
                ({"consumer_id": "c1", "lease_id": "L1", "expected": 0,
                  "limit": 1, "x": 1}, "x"),
                ({"consumer_id": "c1", "lease_id": "L1", "expected": 0,
                  "limit": 1, "y": 1, "z": 2}, "y")):
            with self.subTest(payload=payload):
                with self.assertRaises(ServiceError) as caught:
                    self.service.event_gc_batch_claim(payload)
                self.assertEqual(caught.exception.status_code, 400)
                self.assertEqual(caught.exception.field, field)

    def test_boundary_integers_pass_shape_validation(self) -> None:
        self._commit("r1")
        # 2^63-1 is a valid shape; it fails only the checkpoint match.
        with self.assertRaises(ServiceError) as caught:
            self._claim(lease="L1", expected=2**63 - 1)
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "expected")
        _, status = self._claim(lease="L2", expected=0, limit=1)
        self.assertEqual(status, 201)


class ClaimConcurrencyTest(ClaimMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()
        self._commit("r1")

    def test_concurrent_fresh_ids_one_201_rest_409_consumer(self) -> None:
        results = []
        barrier = threading.Barrier(8)

        def worker(index: int) -> None:
            barrier.wait()
            try:
                _body, status = self._claim(
                    consumer="c1", lease=f"L{index}", expected=0, limit=1)
            except ServiceError as error:
                status = (error.status_code, error.field)
            results.append(status)

        threads = [threading.Thread(target=worker, args=(i,))
                   for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
        self.assertEqual(results.count(201), 1)
        self.assertEqual(results.count((409, "consumer_id")), 7)


class ClaimHTTPTest(ClaimMixin, unittest.TestCase):
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

    def _request(self, path=PATH, raw=None, method="POST"):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        headers = {"Content-Type": "application/json"} if raw else {}
        conn.request(method, path, body=raw, headers=headers)
        response = conn.getresponse()
        data = response.read().decode("utf-8")
        conn.close()
        return response.status, (json.loads(data) if data else None), data

    def _claim_http(self, consumer="c1", lease="L1", expected=0,
                    limit=100, path=PATH):
        return self._request(
            path, json.dumps({"consumer_id": consumer, "lease_id": lease,
                              "expected": expected, "limit": limit}))

    def test_claim_over_http_with_key_order(self) -> None:
        self._commit("r1")
        self._commit("r2")
        status, body, raw = self._claim_http(limit=1)
        self.assertEqual(status, 201)
        self.assertEqual(list(body),
                         ["consumer_id", "lease_id", "records",
                          "next_after", "expires"])
        for earlier, later in (
                ('"consumer_id"', '"lease_id"'),
                ('"lease_id"', '"records"'),
                ('"records"', '"next_after"'),
                ('"next_after"', '"expires"'),
                ('"request_id"', '"device_ids"'),
                ('"device_ids"', '"after"'),
                ('"after"', '"limit"'),
                ('"limit"', '"status"'),
                ('"status"', '"response"')):
            self.assertLess(raw.index(earlier), raw.index(later))
        self.assertEqual(body["next_after"], 1)
        # Byte-identical replay with 200.
        status, body2, raw2 = self._claim_http(limit=1)
        self.assertEqual(status, 200)
        self.assertEqual(raw2, raw)
        self.assertEqual(body2, body)

    def test_query_rejected(self) -> None:
        for query in ("?expected=0", "?foo", "?x=1"):
            with self.subTest(query=query):
                status, body, _ = self._claim_http(path=PATH + query)
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], "query")
                self.assertEqual(list(body), ["message", "field"])
        # A bare trailing '?' carries no parameter and is accepted.
        status, _, _ = self._claim_http(path=PATH + "?")
        self.assertEqual(status, 200)

    def test_bad_json_and_non_object_are_400_request_body(self) -> None:
        for raw in ("{", "[1]", "null", '"s"'):
            with self.subTest(raw=raw):
                status, body, _ = self._request(raw=raw)
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], "request_body")
                self.assertEqual(list(body), ["message", "field"])

    def test_field_errors_over_http(self) -> None:
        for payload, field in (
                ({"lease_id": "L1", "expected": 0, "limit": 1},
                 "consumer_id"),
                ({"consumer_id": "c1", "lease_id": "", "expected": 0,
                  "limit": 1}, "lease_id"),
                ({"consumer_id": "c1", "expected": 0, "limit": 1},
                 "lease_id"),
                ({"consumer_id": "c1", "lease_id": "L1", "limit": 1},
                 "expected"),
                ({"consumer_id": "c1", "lease_id": "L1", "expected": True,
                  "limit": 1}, "expected"),
                ({"consumer_id": "c1", "lease_id": "L1", "expected": 0,
                  "limit": 0}, "limit"),
                ({"consumer_id": "c1", "lease_id": "L1", "expected": 0,
                  "limit": 101}, "limit"),
                ({"consumer_id": "c1", "lease_id": "L1", "expected": 0,
                  "limit": 1, "extra": 1}, "extra")):
            with self.subTest(payload=payload):
                status, body, _ = self._request(raw=json.dumps(payload))
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], field)
                self.assertEqual(list(body), ["message", "field"])

    def test_conflicts_over_http(self) -> None:
        self._commit("r1")
        self._claim_http(consumer="c1", lease="L1", limit=1)
        status, body, _ = self._claim_http(consumer="c2", lease="L1",
                                           expected=0, limit=1)
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "lease_id")
        status, body, _ = self._claim_http(consumer="c1", lease="L2",
                                           expected=1, limit=1)
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "expected")
        status, body, _ = self._claim_http(consumer="c1", lease="L2",
                                           expected=0, limit=1)
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "consumer_id")

    def test_get_is_404(self) -> None:
        status, _, _ = self._request(method="GET")
        self.assertEqual(status, 404)


class ClaimPersistenceTest(ClaimMixin, unittest.TestCase):
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

    def test_nonempty_claim_consumes_a_generation_empty_does_not(self) -> None:
        self._commit("r1")
        generation = self.state_store.commit_seq
        _, status = self._claim(lease="L1", expected=0, limit=100)
        self.assertEqual(status, 201)
        self.assertEqual(self.state_store.commit_seq, generation + 1)
        before = self._document()
        _, status = self._claim(lease="L1", expected=0, limit=100)
        self.assertEqual(status, 200)
        self.assertEqual(self.state_store.commit_seq, generation + 1)
        self.assertEqual(self._document(), before)

    def test_section_order_and_item_key_order(self) -> None:
        self._commit("r1")
        self._claim(lease="L1", expected=0, limit=1)
        keys = list(self._document())
        index = keys.index("cleanup_checkpoints")
        self.assertEqual(keys[index:index + 2],
                         ["cleanup_checkpoints", "cleanup_leases"])
        leases = self._document()["cleanup_leases"]
        self.assertEqual(len(leases), 1)
        self.assertEqual(list(leases[0]),
                         ["lease_id", "consumer_id", "expected",
                          "next_after", "limit", "expires"])

    def test_restart_replays_and_still_enforces_live_lease(self) -> None:
        self._commit("r1")
        body, status = self._claim(lease="L1", expected=0, limit=1)
        self.assertEqual(status, 201)
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        replay, status = restarted.event_gc_batch_claim({
            "consumer_id": "c1", "lease_id": "L1",
            "expected": 0, "limit": 1})
        self.assertEqual(status, 200)
        self.assertEqual(replay, body)
        with self.assertRaises(ServiceError) as caught:
            restarted.event_gc_batch_claim({
                "consumer_id": "c1", "lease_id": "L9",
                "expected": 0, "limit": 1})
        self.assertEqual((caught.exception.status_code,
                          caught.exception.field), (409, "consumer_id"))
        self.assertTrue(restarted.persistence_integrity()["consistent"])

    def _reload(self, mutate):
        payload = copy.deepcopy(self.service.store.snapshot_state())
        mutate(payload)
        restarted = DeviceService()
        restarted.store.restore_state(payload)
        return restarted

    def test_expired_lease_can_be_replaced_after_restart(self) -> None:
        self._commit("r1")
        self._claim(lease="L1", expected=0, limit=1)
        past = (datetime.now(timezone.utc) - timedelta(seconds=1)
                ).isoformat(timespec="microseconds")

        def expire(payload) -> None:
            payload["cleanup_leases"][0]["expires"] = past

        restarted = self._reload(expire)
        body, status = restarted.event_gc_batch_claim({
            "consumer_id": "c1", "lease_id": "L2",
            "expected": 0, "limit": 1})
        self.assertEqual(status, 201)
        self.assertEqual([r["request_id"] for r in body["records"]],
                         ["r1"])
        # The expired id remains occupied and replays 200.
        replay, status = restarted.event_gc_batch_claim({
            "consumer_id": "c1", "lease_id": "L1",
            "expected": 0, "limit": 1})
        self.assertEqual(status, 200)
        self.assertEqual(replay["records"], body["records"])
        self.assertEqual(replay["expires"], past)

    def test_save_failure_rolls_lease_back(self) -> None:
        import e2ee_backend.persistence as persistence_mod

        self._commit("r1")
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
                self._claim(lease="L1", expected=0, limit=1)
        finally:
            persistence_mod.os.fsync = real_fsync
        self.assertEqual(self.state_store.commit_seq, generation)
        self.assertEqual(self._document()["cleanup_leases"], [])
        self.assertFalse(self.service.store._cleanup_leases)
        body, status = self._claim(lease="L1", expected=0, limit=1)
        self.assertEqual(status, 201)
        self.assertEqual([r["request_id"] for r in body["records"]],
                         ["r1"])
        self.assertEqual(self.state_store.commit_seq, generation + 1)

    def test_malformed_section_refuses_restore(self) -> None:
        from e2ee_backend.storage import DeviceStore

        self._commit("r1")
        self._commit("r2")
        self._claim(lease="L1", expected=0, limit=1)
        base = copy.deepcopy(self.service.store.snapshot_state())
        future = (datetime.now(timezone.utc) + timedelta(seconds=100)
                  ).isoformat(timespec="microseconds")

        def duplicate_id(payload) -> None:
            payload["cleanup_leases"].append(
                copy.deepcopy(payload["cleanup_leases"][0]))

        def wrong_key_order(payload) -> None:
            payload["cleanup_leases"][0] = {
                "consumer_id": "c1", "lease_id": "L1", "expected": 0,
                "next_after": 1, "limit": 1,
                "expires": payload["cleanup_leases"][0]["expires"]}

        def bool_limit(payload) -> None:
            payload["cleanup_leases"][0]["limit"] = True

        def equal_window(payload) -> None:
            payload["cleanup_leases"][0]["expected"] = 1

        def zero_limit(payload) -> None:
            payload["cleanup_leases"][0]["limit"] = 0

        def beyond_audit(payload) -> None:
            payload["cleanup_leases"][0]["next_after"] = 3

        def window_exceeds_limit(payload) -> None:
            payload["cleanup_leases"][0]["next_after"] = 2

        def two_live_same_consumer(payload) -> None:
            payload["cleanup_leases"][0]["expires"] = future
            payload["cleanup_leases"].append({
                "lease_id": "L2", "consumer_id": "c1", "expected": 0,
                "next_after": 1, "limit": 1, "expires": future})

        def bad_expires(payload) -> None:
            payload["cleanup_leases"][0]["expires"] = "not-a-time"

        def empty_lease_id(payload) -> None:
            payload["cleanup_leases"][0]["lease_id"] = ""

        def section_object(payload) -> None:
            payload["cleanup_leases"] = {}

        for name, mutate in (
                ("duplicate id", duplicate_id),
                ("wrong key order", wrong_key_order),
                ("bool limit", bool_limit),
                ("expected == next_after", equal_window),
                ("limit 0", zero_limit),
                ("next_after beyond audit", beyond_audit),
                ("window exceeds limit", window_exceeds_limit),
                ("two live leases same consumer", two_live_same_consumer),
                ("bad expires", bad_expires),
                ("empty lease_id", empty_lease_id),
                ("section not a list", section_object)):
            with self.subTest(name=name):
                payload = copy.deepcopy(base)
                mutate(payload)
                with self.assertRaises((ValueError, TypeError)):
                    DeviceStore().restore_state(payload)

    def test_valid_sections_load(self) -> None:
        from e2ee_backend.storage import DeviceStore

        self._commit("r1")
        self._claim(lease="L1", expected=0, limit=1)
        base = copy.deepcopy(self.service.store.snapshot_state())
        future = (datetime.now(timezone.utc) + timedelta(seconds=100)
                  ).isoformat(timespec="microseconds")

        # Two live leases for different consumers are fine.
        payload = copy.deepcopy(base)
        payload["cleanup_leases"][0]["expires"] = future
        payload["cleanup_leases"].append({
            "lease_id": "L2", "consumer_id": "c2", "expected": 0,
            "next_after": 1, "limit": 1, "expires": future})
        DeviceStore().restore_state(payload)

        # A live plus an expired lease for the same consumer is fine.
        payload = copy.deepcopy(base)
        payload["cleanup_leases"][0]["expires"] = future
        payload["cleanup_leases"].append({
            "lease_id": "L2", "consumer_id": "c1", "expected": 0,
            "next_after": 1, "limit": 1,
            "expires": (datetime.now(timezone.utc) - timedelta(seconds=1)
                        ).isoformat(timespec="microseconds")})
        DeviceStore().restore_state(payload)


if __name__ == "__main__":
    unittest.main()
