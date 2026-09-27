"""Tests for the batch-cleanup audit lease resolution endpoint.

``POST /v1/event-gc-batch/lease`` confirms or releases a lease claimed
by ``POST /v1/event-gc-batch/claim``. The call takes no query
parameters (any -> 400/query); the body carries exactly
``consumer_id`` and ``lease_id`` (non-empty strings), ``expected`` (a
non-boolean integer in 0..2^63-1) and ``op`` (``"confirm"`` or
``"release"``); a bad/non-object body is 400/request_body and a
missing/wrongly typed/extra field is 400 with that field.

Under the one store lock the ``lease_id`` resolves first: an unknown id
is 404/lease_id and an id owned by another consumer is
409/consumer_id. A lease already resolved with the same op replays the
first response as 200; the other op is 409/lease_id. A first
resolution on an expired unconfirmed lease is 409/lease_id (checked
before ``expected``); when the checkpoint already reached
``next_after`` a confirm is a read-only 200 and a release is
409/lease_id; otherwise ``expected`` must equal the lease's frozen
origin (409/expected). A first confirm answers 201 and advances the
checkpoint to ``next_after``; a first release answers 201 without
advancing and unblocks the consumer's next claim. Success keys are
``consumer_id``, ``lease_id``, ``op`` and ``after``. Resolutions
persist in the seventh ``cleanup_leases`` key ``terminal``
(null/"confirm"/"release"); legacy six-key items load as null.
"""
import json
import os
import shutil
import tempfile
import threading
import unittest
from contextlib import contextmanager
from datetime import datetime, timedelta
from http.client import HTTPConnection
from unittest import mock

from e2ee_backend.models import Device
from e2ee_backend.service import DeviceService, ServiceError
from e2ee_backend.http_app import create_server
from e2ee_backend.persistence import (
    PersistenceUnavailable,
    StateFileError,
    attach_persistence,
)

PATH = "/v1/event-gc-batch/lease"
PAST = "2020-01-01T00:00:00.000000+00:00"
FUTURE = "2099-01-01T00:00:00.000000+00:00"


class LeaseMixin:
    def _build(self) -> None:
        self.service = DeviceService()
        self.service.store.add_device(Device("u", "alice", "ik"))
        self.service.store.add_device(Device("u", "bob", "ik"))

    def _commit(self, request_id, device_ids=("bob",)):
        return self.service.event_gc_cleanup_expired_batch({
            "mode": "commit", "device_ids": list(device_ids),
            "after": 0, "limit": 100, "request_id": request_id})

    def _claim(self, consumer="c1", lease_id="L1", expected=0, limit=100):
        return self.service.event_gc_batch_claim({
            "consumer_id": consumer, "lease_id": lease_id,
            "expected": expected, "limit": limit})

    def _lease(self, consumer="c1", lease_id="L1", expected=0,
               op="confirm"):
        return self.service.event_gc_batch_lease({
            "consumer_id": consumer, "lease_id": lease_id,
            "expected": expected, "op": op})

    def _checkpoint(self, consumer="c1"):
        body, _ = self.service.event_gc_batch_checkpoint(
            {"consumer_id": consumer, "expected": None, "after": None})
        return body

    def _expire(self, lease_id):
        self.service.store._cleanup_leases[lease_id].expires = PAST

    @contextmanager
    def _shift_now(self, seconds):
        # Move only the store's "now" (and deadline comparisons) into
        # the future; fromisoformat keeps parsing stored timestamps.
        import e2ee_backend.storage as storage_mod

        class ShiftedDatetime(datetime):
            @classmethod
            def now(cls, tz=None):  # noqa: ANN001
                return datetime.now(tz) + timedelta(seconds=seconds)

        with mock.patch.object(storage_mod, "datetime", ShiftedDatetime):
            yield


class LeaseServiceTest(LeaseMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def test_confirm_first_201_advances_checkpoint(self) -> None:
        self._commit("r1")
        self._commit("r2")
        first, claim_status = self._claim(limit=1)
        self.assertEqual(claim_status, 201)
        body, status = self._lease(op="confirm")
        self.assertEqual(status, 201)
        self.assertEqual(list(body),
                         ["consumer_id", "lease_id", "op", "after"])
        self.assertEqual(body, {"consumer_id": "c1", "lease_id": "L1",
                                "op": "confirm", "after": 1})
        checkpoint = self._checkpoint()
        self.assertEqual(checkpoint["after"], 1)
        self.assertIsNotNone(checkpoint["updated_at"])
        self.assertEqual(
            self.service.store._cleanup_leases["L1"].terminal,
            "confirm")

    def test_release_first_201_keeps_checkpoint_and_unblocks(self) -> None:
        self._commit("r1")
        self._commit("r2")
        self.assertEqual(self._claim(limit=1)[1], 201)
        body, status = self._lease(op="release")
        self.assertEqual(status, 201)
        self.assertEqual(body, {"consumer_id": "c1", "lease_id": "L1",
                                "op": "release", "after": 0})
        # The checkpoint is untouched.
        self.assertEqual(self._checkpoint()["after"], 0)
        self.assertEqual(
            self.service.store._cleanup_leases["L1"].terminal,
            "release")
        # The consumer can immediately claim again from the same origin.
        second, claim_status = self._claim(lease_id="L2", expected=0,
                                           limit=10)
        self.assertEqual(claim_status, 201)
        self.assertEqual(second["next_after"], 2)

    def test_confirm_does_not_block_follow_up_claim(self) -> None:
        self._commit("r1")
        self._commit("r2")
        self.assertEqual(self._claim(limit=1)[1], 201)
        self.assertEqual(self._lease(op="confirm")[1], 201)
        body, status = self._claim(lease_id="L2", expected=1, limit=10)
        self.assertEqual(status, 201)
        self.assertEqual(body["next_after"], 2)

    def test_same_op_replay_is_200_byte_identical(self) -> None:
        self._commit("r1")
        self.assertEqual(self._claim(limit=1)[1], 201)
        first, status = self._lease(op="confirm")
        self.assertEqual(status, 201)
        replay, replay_status = self._lease(op="confirm")
        self.assertEqual(replay_status, 200)
        self.assertEqual(replay, first)
        # Still the original response after the lease deadline passes.
        with self._shift_now(31):
            later, later_status = self._lease(op="confirm")
        self.assertEqual(later_status, 200)
        self.assertEqual(later, first)

    def test_release_replay_is_200_byte_identical(self) -> None:
        self._commit("r1")
        self.assertEqual(self._claim(limit=1)[1], 201)
        first, status = self._lease(op="release")
        self.assertEqual(status, 201)
        replay, replay_status = self._lease(op="release")
        self.assertEqual(replay_status, 200)
        self.assertEqual(replay, first)
        with self._shift_now(31):
            later, later_status = self._lease(op="release")
        self.assertEqual(later_status, 200)
        self.assertEqual(later, first)

    def test_same_op_replay_ignores_stale_expected(self) -> None:
        # The expected-value check belongs to the first resolution only;
        # a same-op replay answers 200 with the original response even
        # when its expected is stale.
        self._commit("r1")
        self.assertEqual(self._claim(limit=1)[1], 201)
        first, status = self._lease(op="confirm")
        self.assertEqual(status, 201)
        replay, replay_status = self._lease(expected=99, op="confirm")
        self.assertEqual(replay_status, 200)
        self.assertEqual(replay, first)

        self._commit("r2")
        self.assertEqual(self._claim(lease_id="L2", expected=1,
                                     limit=1)[1], 201)
        first, status = self._lease(lease_id="L2", expected=1,
                                    op="release")
        self.assertEqual(status, 201)
        replay, replay_status = self._lease(lease_id="L2", expected=42,
                                            op="release")
        self.assertEqual(replay_status, 200)
        self.assertEqual(replay, first)

    def test_other_op_after_terminal_is_409_lease_id(self) -> None:
        self._commit("r1")
        self._commit("r2")
        self.assertEqual(self._claim(limit=1)[1], 201)
        self.assertEqual(self._lease(op="confirm")[1], 201)
        with self.assertRaises(ServiceError) as caught:
            self._lease(op="release")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")

        self.assertEqual(self._claim(lease_id="L2", expected=1,
                                     limit=1)[1], 201)
        self.assertEqual(self._lease(lease_id="L2", expected=1,
                                     op="release")[1], 201)
        with self.assertRaises(ServiceError) as caught:
            self._lease(lease_id="L2", expected=1, op="confirm")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_unknown_lease_is_404(self) -> None:
        with self.assertRaises(ServiceError) as caught:
            self._lease(lease_id="nope", op="confirm")
        self.assertEqual(caught.exception.status_code, 404)
        self.assertEqual(caught.exception.field, "lease_id")
        with self.assertRaises(ServiceError) as caught:
            self._lease(lease_id="nope", op="release")
        self.assertEqual(caught.exception.status_code, 404)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_consumer_mismatch_is_409_consumer_id(self) -> None:
        self._commit("r1")
        self.assertEqual(self._claim(limit=1)[1], 201)
        with self.assertRaises(ServiceError) as caught:
            self._lease(consumer="intruder", op="confirm")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "consumer_id")
        # Mismatch precedes the expected-value check.
        with self.assertRaises(ServiceError) as caught:
            self._lease(consumer="intruder", expected=99,
                        op="release")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "consumer_id")

    def test_expected_must_equal_lease_origin(self) -> None:
        self._commit("r1")
        self.assertEqual(self._claim(limit=1)[1], 201)
        for op in ("confirm", "release"):
            with self.subTest(op=op):
                with self.assertRaises(ServiceError) as caught:
                    self._lease(expected=1, op=op)
                self.assertEqual(caught.exception.status_code, 409)
                self.assertEqual(caught.exception.field, "expected")
        # The origin value still resolves.
        self.assertEqual(self._lease(op="release")[1], 201)

    def test_expired_unconfirmed_lease_is_409_for_both_ops(self) -> None:
        self._commit("r1")
        self.assertEqual(self._claim(limit=1)[1], 201)
        self._expire("L1")
        for op in ("confirm", "release"):
            with self.subTest(op=op):
                with self.assertRaises(ServiceError) as caught:
                    self._lease(op=op)
                self.assertEqual(caught.exception.status_code, 409)
                self.assertEqual(caught.exception.field, "lease_id")
        # Expiry is checked before expected: a wrong expected still
        # reports the expired lease as 409/lease_id.
        with self.assertRaises(ServiceError) as caught:
            self._lease(expected=42, op="confirm")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")
        # Nothing was written.
        self.assertIsNone(
            self.service.store._cleanup_leases["L1"].terminal)
        self.assertEqual(self._checkpoint()["after"], 0)

    def test_acknowledged_confirm_is_read_only_200(self) -> None:
        self._commit("r1")
        self.assertEqual(self._claim(limit=1)[1], 201)
        # The checkpoint endpoint acknowledges the lease first.
        advanced, status = self.service.event_gc_batch_checkpoint({
            "consumer_id": "c1", "expected": 0, "after": 1})
        self.assertEqual(status, 201)
        body, status = self._lease(op="confirm")
        self.assertEqual(status, 200)
        self.assertEqual(body, {"consumer_id": "c1", "lease_id": "L1",
                                "op": "confirm", "after": 1})
        # Read-only: the lease stays unterminal and a replay is still a
        # plain 200.
        self.assertIsNone(
            self.service.store._cleanup_leases["L1"].terminal)
        replay, replay_status = self._lease(op="confirm")
        self.assertEqual(replay_status, 200)
        self.assertEqual(replay, body)
        # A wrong expected is ignored once the lease was acknowledged:
        # the state decision precedes the expected-value check.
        stale, stale_status = self._lease(expected=99, op="confirm")
        self.assertEqual(stale_status, 200)
        self.assertEqual(stale, body)

    def test_acknowledged_release_is_409(self) -> None:
        self._commit("r1")
        self.assertEqual(self._claim(limit=1)[1], 201)
        self.service.event_gc_batch_checkpoint({
            "consumer_id": "c1", "expected": 0, "after": 1})
        with self.assertRaises(ServiceError) as caught:
            self._lease(op="release")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")
        # Even a fresh expiry does not change the answer: acknowledged
        # takes precedence; no terminal is written either.
        self._expire("L1")
        with self.assertRaises(ServiceError):
            self._lease(op="release")
        self.assertIsNone(
            self.service.store._cleanup_leases["L1"].terminal)

    def test_consume_acknowledgement_uses_same_rules(self) -> None:
        self._commit("r1")
        self.assertEqual(self._claim(limit=1)[1], 201)
        consumed, status = self.service.event_gc_batch_consume({
            "consumer_id": "c1", "expected": 0, "limit": 100})
        self.assertEqual(status, 201)
        self.assertEqual(consumed["next_after"], 1)
        self.assertEqual(self._lease(op="confirm")[1], 200)
        with self.assertRaises(ServiceError) as caught:
            self._lease(op="release")
        self.assertEqual(caught.exception.field, "lease_id")

    def test_partial_advance_before_resolution_is_409_expected(self) -> None:
        self._commit("r1")
        self._commit("r2")
        self.assertEqual(self._claim(limit=2)[1], 201)
        # The lease spans [0, 2); advance the checkpoint only to 1, so
        # the lease origin 0 is no longer the current checkpoint.
        self.service.event_gc_batch_checkpoint({
            "consumer_id": "c1", "expected": 0, "after": 1})
        for op in ("confirm", "release"):
            for expected in (0, 1):
                with self.subTest(op=op, expected=expected):
                    with self.assertRaises(ServiceError) as caught:
                        self._lease(expected=expected, op=op)
                    self.assertEqual(caught.exception.status_code, 409)
                    self.assertEqual(caught.exception.field, "expected")
        # Nothing was terminalized; reaching next_after makes a confirm
        # a read-only 200 (implicit acknowledgement).
        self.service.event_gc_batch_checkpoint({
            "consumer_id": "c1", "expected": 1, "after": 2})
        body, status = self._lease(expected=0, op="confirm")
        self.assertEqual(status, 200)
        self.assertEqual(body["after"], 2)
        self.assertIsNone(
            self.service.store._cleanup_leases["L1"].terminal)

    def test_release_replay_stays_frozen_after_checkpoint_moves(self) -> None:
        self._commit("r1")
        self._commit("r2")
        self.assertEqual(self._claim(limit=1)[1], 201)
        first, status = self._lease(op="release")
        self.assertEqual(status, 201)
        self.assertEqual(first["after"], 0)
        # A later checkpoint advance past next_after does not change the
        # frozen release response.
        self.service.event_gc_batch_checkpoint({
            "consumer_id": "c1", "expected": 0, "after": 2})
        replay, replay_status = self._lease(op="release")
        self.assertEqual(replay_status, 200)
        self.assertEqual(replay, first)


class LeaseValidationTest(LeaseMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def _expect_400(self, payload, field):
        with self.assertRaises(ServiceError) as caught:
            self.service.event_gc_batch_lease(payload)
        self.assertEqual(caught.exception.status_code, 400)
        self.assertEqual(caught.exception.field, field)
        self.assertEqual(list(caught.exception.to_body()),
                         ["message", "field"])

    def test_body_must_be_object(self) -> None:
        for payload in (None, [], "s", 1, True):
            with self.subTest(payload=payload):
                self._expect_400(payload, "request_body")

    def test_required_fields(self) -> None:
        self._expect_400({"lease_id": "L", "expected": 0,
                          "op": "confirm"}, "consumer_id")
        self._expect_400({"consumer_id": "c", "expected": 0,
                          "op": "confirm"}, "lease_id")
        self._expect_400({"consumer_id": "c", "lease_id": "L",
                          "op": "confirm"}, "expected")
        self._expect_400({"consumer_id": "c", "lease_id": "L",
                          "expected": 0}, "op")

    def test_field_shapes(self) -> None:
        base = {"consumer_id": "c", "lease_id": "L", "expected": 0,
                "op": "confirm"}
        for bad in ("", None, 1, True, []):
            with self.subTest(bad=bad):
                self._expect_400(dict(base, consumer_id=bad),
                                 "consumer_id")
        for bad in ("", None, 1, True, []):
            with self.subTest(bad=bad):
                self._expect_400(dict(base, lease_id=bad), "lease_id")
        for bad in (True, False, -1, 2**63, 1.0, "0", None):
            with self.subTest(bad=bad):
                self._expect_400(dict(base, expected=bad), "expected")
        for bad in ("", "CONFIRM", "Confirm", "ack", None, 1, True, []):
            with self.subTest(bad=bad):
                self._expect_400(dict(base, op=bad), "op")

    def test_extra_field_is_400_with_that_key(self) -> None:
        self._expect_400({"consumer_id": "c", "lease_id": "L",
                          "expected": 0, "op": "confirm",
                          "extra": 1}, "extra")

    def test_2_63_minus_1_accepted_as_shape(self) -> None:
        # The shape is valid (not 400); an unknown id only fails with
        # 404/lease_id.
        with self.assertRaises(ServiceError) as caught:
            self._lease(expected=2**63 - 1)
        self.assertEqual(caught.exception.status_code, 404)
        self.assertEqual(caught.exception.field, "lease_id")


class LeaseConcurrencyTest(LeaseMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()
        self._commit("r1")
        self._commit("r2")
        self.assertEqual(self._claim(limit=1)[1], 201)

    def test_parallel_confirms_linearize_to_one_201(self) -> None:
        barrier = threading.Barrier(8)
        results = []

        def worker() -> None:
            barrier.wait()
            body, status = self._lease(op="confirm")
            results.append((status, body))

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
        self.assertEqual(len(results), 8)
        self.assertEqual(sum(1 for status, _ in results if status == 201),
                         1)
        first = next(body for status, body in results if status == 201)
        for status, body in results:
            self.assertIn(status, (200, 201))
            self.assertEqual(body, first)
        self.assertEqual(self._checkpoint()["after"], 1)

    def test_parallel_mixed_ops_have_one_201_and_no_double_terminal(
            self) -> None:
        barrier = threading.Barrier(8)
        outcomes = []

        def worker(op) -> None:
            barrier.wait()
            try:
                body, status = self._lease(op=op)
            except ServiceError as error:
                outcomes.append((error.status_code, error.field, None))
                return
            outcomes.append((status, None, body))

        ops = ("confirm",) * 4 + ("release",) * 4
        threads = [threading.Thread(target=worker, args=(op,))
                   for op in ops]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
        self.assertEqual(len(outcomes), 8)
        self.assertEqual(sum(1 for code, _, _ in outcomes if code == 201),
                         1)
        for code, field, _ in outcomes:
            self.assertIn(code, (200, 201, 409))
            if code == 409:
                self.assertEqual(field, "lease_id")
        # The lease ended in exactly one durable terminal state.
        terminal = self.service.store._cleanup_leases["L1"].terminal
        self.assertIn(terminal, ("confirm", "release"))
        if terminal == "confirm":
            self.assertEqual(self._checkpoint()["after"], 1)
        else:
            self.assertEqual(self._checkpoint()["after"], 0)


class LeaseHTTPTest(LeaseMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()
        self.server, _ = create_server("127.0.0.1", 0, self.service)
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

    def _lease_http(self, consumer="c1", lease_id="L1", expected=0,
                    op="confirm", path=PATH):
        return self._request(path, json.dumps({
            "consumer_id": consumer, "lease_id": lease_id,
            "expected": expected, "op": op}))

    def test_confirm_over_http_with_key_order(self) -> None:
        self._commit("r1")
        self._claim(limit=1)
        status, body, raw = self._lease_http(op="confirm")
        self.assertEqual(status, 201)
        self.assertEqual(list(body),
                         ["consumer_id", "lease_id", "op", "after"])
        for earlier, later in (
                ('"consumer_id"', '"lease_id"'),
                ('"lease_id"', '"op"'),
                ('"op"', '"after"')):
            self.assertLess(raw.index(earlier), raw.index(later))
        self.assertEqual(body["after"], 1)

    def test_release_over_http_with_key_order(self) -> None:
        self._commit("r1")
        self._claim(limit=1)
        status, body, raw = self._lease_http(op="release")
        self.assertEqual(status, 201)
        self.assertEqual(list(body),
                         ["consumer_id", "lease_id", "op", "after"])
        self.assertEqual(body["after"], 0)
        self.assertLess(raw.index('"release"'), raw.index('"after"'))

    def test_query_rejected(self) -> None:
        for query in ("?op=confirm", "?foo", "?x=1"):
            with self.subTest(query=query):
                status, body, _ = self._lease_http(path=PATH + query)
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], "query")
        self._commit("r1")
        self._claim(limit=1)
        status, _, _ = self._lease_http(path=PATH + "?")
        self.assertEqual(status, 201)

    def test_bad_json_and_non_object_are_400_request_body(self) -> None:
        for raw in ("{", "[1]", "null", '"s"'):
            with self.subTest(raw=raw):
                status, body, _ = self._request(raw=raw)
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], "request_body")
                self.assertEqual(list(body), ["message", "field"])

    def test_field_errors_over_http(self) -> None:
        for payload, field in (
                ({"lease_id": "L", "expected": 0, "op": "confirm"},
                 "consumer_id"),
                ({"consumer_id": "c", "expected": 0, "op": "confirm"},
                 "lease_id"),
                ({"consumer_id": "c", "lease_id": "", "expected": 0,
                  "op": "confirm"}, "lease_id"),
                ({"consumer_id": "c", "lease_id": "L",
                  "op": "confirm"}, "expected"),
                ({"consumer_id": "c", "lease_id": "L", "expected": 0},
                 "op"),
                ({"consumer_id": "c", "lease_id": "L",
                  "expected": True, "op": "confirm"}, "expected"),
                ({"consumer_id": "c", "lease_id": "L", "expected": -1,
                  "op": "release"}, "expected"),
                ({"consumer_id": "c", "lease_id": "L", "expected": 0,
                  "op": "ack"}, "op"),
                ({"consumer_id": "c", "lease_id": "L", "expected": 0,
                  "op": "confirm", "extra": 1}, "extra")):
            with self.subTest(payload=payload):
                status, body, _ = self._request(raw=json.dumps(payload))
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], field)
                self.assertEqual(list(body), ["message", "field"])

    def test_conflicts_and_unknown_over_http(self) -> None:
        self._commit("r1")
        self._claim(limit=1)
        status, body, _ = self._lease_http(lease_id="ghost")
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "lease_id")
        status, body, _ = self._lease_http(consumer="intruder")
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "consumer_id")
        status, body, _ = self._lease_http(expected=5)
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "expected")
        self.assertEqual(self._lease_http(op="confirm")[0], 201)
        status, body, _ = self._lease_http(op="release")
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "lease_id")
        status, body, _ = self._lease_http(op="confirm")
        self.assertEqual(status, 200)

    def test_get_is_404(self) -> None:
        status, _, _ = self._request(method="GET")
        self.assertEqual(status, 404)


class LeasePersistenceTest(LeaseMixin, unittest.TestCase):
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

    def _lease_record(self, lease_id="L1"):
        return next(item for item in self._document()["cleanup_leases"]
                    if item["lease_id"] == lease_id)

    def test_first_resolution_consumes_one_generation_replays_none(
            self) -> None:
        self._commit("r1")
        self.assertEqual(self._claim(limit=1)[1], 201)
        generation = self.state_store.commit_seq
        self.assertEqual(self._lease(op="confirm")[1], 201)
        self.assertEqual(self.state_store.commit_seq, generation + 1)
        self.assertEqual(self._lease(op="confirm")[1], 200)
        self.assertEqual(self.state_store.commit_seq, generation + 1)

    def test_acknowledged_confirm_200_consumes_no_generation(self) -> None:
        self._commit("r1")
        self.assertEqual(self._claim(limit=1)[1], 201)
        self.service.event_gc_batch_checkpoint({
            "consumer_id": "c1", "expected": 0, "after": 1})
        generation = self.state_store.commit_seq
        document_before = self._document()
        self.assertEqual(self._lease(op="confirm")[1], 200)
        self.assertEqual(self.state_store.commit_seq, generation)
        self.assertEqual(self._document(), document_before)
        self.assertIsNone(self._lease_record()["terminal"])

    def test_terminal_is_serialized_as_seventh_key(self) -> None:
        self._commit("r1")
        self._commit("r2")
        self.assertEqual(self._claim(limit=1)[1], 201)
        record = self._lease_record()
        self.assertEqual(list(record),
                         ["lease_id", "consumer_id", "expected",
                          "next_after", "limit", "expires", "terminal"])
        self.assertIsNone(record["terminal"])
        self.assertEqual(self._lease(op="confirm")[1], 201)
        record = self._lease_record()
        self.assertEqual(list(record)[-1], "terminal")
        self.assertEqual(record["terminal"], "confirm")

        self.assertEqual(self._claim(lease_id="L2", expected=1,
                                     limit=1)[1], 201)
        self.assertEqual(self._lease(lease_id="L2", expected=1,
                                     op="release")[1], 201)
        record = self._lease_record("L2")
        self.assertEqual(list(record)[-1], "terminal")
        self.assertEqual(record["terminal"], "release")

    def test_save_failure_rolls_confirm_back(self) -> None:
        import e2ee_backend.persistence as persistence_mod

        self._commit("r1")
        self.assertEqual(self._claim(limit=1)[1], 201)
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
                self._lease(op="confirm")
        finally:
            persistence_mod.os.fsync = real_fsync
        self.assertEqual(self.state_store.commit_seq, generation)
        self.assertIsNone(
            self.service.store._cleanup_leases["L1"].terminal)
        self.assertEqual(self._checkpoint()["after"], 0)
        self.assertIsNone(self._lease_record()["terminal"])
        self.assertEqual(self._document()["cleanup_checkpoints"], [])
        # The confirm can be retried and now commits.
        body, status = self._lease(op="confirm")
        self.assertEqual(status, 201)
        self.assertEqual(body["after"], 1)
        self.assertEqual(self.state_store.commit_seq, generation + 1)

    def test_save_failure_rolls_release_back(self) -> None:
        import e2ee_backend.persistence as persistence_mod

        self._commit("r1")
        self.assertEqual(self._claim(limit=1)[1], 201)
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
                self._lease(op="release")
        finally:
            persistence_mod.os.fsync = real_fsync
        self.assertEqual(self.state_store.commit_seq, generation)
        self.assertIsNone(
            self.service.store._cleanup_leases["L1"].terminal)
        self.assertIsNone(self._lease_record()["terminal"])
        # The failed release kept the lease blocking: a same-consumer
        # claim is still 409/consumer_id until the release commits.
        with self.assertRaises(ServiceError) as caught:
            self._claim(lease_id="L2", expected=0, limit=10)
        self.assertEqual(caught.exception.field, "consumer_id")
        body, status = self._lease(op="release")
        self.assertEqual(status, 201)
        self.assertEqual(body["after"], 0)
        self.assertEqual(self.state_store.commit_seq, generation + 1)

    def test_restart_preserves_terminal_and_replay(self) -> None:
        self._commit("r1")
        self._commit("r2")
        first_claim, _ = self._claim(limit=1)
        first_confirm, _ = self._lease(op="confirm")
        self.assertEqual(self._claim(lease_id="L2", expected=1,
                                     limit=1)[1], 201)
        first_release, _ = self._lease(lease_id="L2", expected=1,
                                       op="release")
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        replay, status = restarted.event_gc_batch_lease({
            "consumer_id": "c1", "lease_id": "L1", "expected": 0,
            "op": "confirm"})
        self.assertEqual(status, 200)
        self.assertEqual(replay, first_confirm)
        replay, status = restarted.event_gc_batch_lease({
            "consumer_id": "c1", "lease_id": "L2", "expected": 1,
            "op": "release"})
        self.assertEqual(status, 200)
        self.assertEqual(replay, first_release)
        # The confirmed lease advanced the checkpoint; the released one
        # did not, and the claim view still replays byte-identically.
        self.assertEqual(
            restarted.event_gc_batch_checkpoint({
                "consumer_id": "c1", "expected": None,
                "after": None})[0]["after"], 1)
        claim_replay, claim_status = restarted.event_gc_batch_claim({
            "consumer_id": "c1", "lease_id": "L1", "expected": 0,
            "limit": 1})
        self.assertEqual(claim_status, 200)
        self.assertEqual(claim_replay, first_claim)
        self.assertTrue(
            restarted.persistence_integrity()["consistent"])


class LeaseRestoreValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self._seed_n = 0

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def _document_with_lease(self):
        # A fresh service per call: one committed audit record and one
        # committed lease, written to a fresh
        # marker-less/sidecar-less path, so rejection comes from the
        # payload's own semantic validation rather than the hash gate.
        self._seed_n += 1
        service = DeviceService()
        service.store.add_device(Device("u", "bob", "ik"))
        seed_path = os.path.join(self.directory,
                                 f"seed{self._seed_n}.json")
        attach_persistence(service, seed_path)
        service.event_gc_cleanup_expired_batch({
            "mode": "commit", "device_ids": ["bob"], "after": 0,
            "limit": 100, "request_id": "r1"})
        service.event_gc_batch_claim({"consumer_id": "c1",
                                      "lease_id": "L1",
                                      "expected": 0, "limit": 1})
        with open(seed_path, encoding="utf-8") as handle:
            document = json.load(handle)
        document.pop("integrity_log_version", None)
        bad_path = os.path.join(self.directory, f"bad{self._seed_n}.json")
        with open(bad_path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        return bad_path, document

    def _write(self, document, name=None):
        self._seed_n += 1
        path = os.path.join(self.directory,
                            name or f"doc{self._seed_n}.json")
        document.pop("integrity_log_version", None)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        return path

    def _assert_refuses_startup(self, path) -> str:
        with open(path, "rb") as handle:
            original = handle.read()
        service = DeviceService()
        with self.assertRaises(StateFileError) as caught:
            attach_persistence(service, path)
        with open(path, "rb") as handle:
            self.assertEqual(handle.read(), original)
        return str(caught.exception)

    @staticmethod
    def _lease(lease_id="L1", consumer="c1", expected=0, next_after=1,
               limit=1, expires=FUTURE, terminal=None):
        return {"lease_id": lease_id, "consumer_id": consumer,
                "expected": expected, "next_after": next_after,
                "limit": limit, "expires": expires,
                "terminal": terminal}

    def test_legacy_six_key_item_loads_terminal_null(self) -> None:
        path, document = self._document_with_lease()
        for lease in document["cleanup_leases"]:
            del lease["terminal"]
        self.assertEqual(list(document["cleanup_leases"][0]),
                         ["lease_id", "consumer_id", "expected",
                          "next_after", "limit", "expires"])
        legacy_path = self._write(document)
        service = DeviceService()
        attach_persistence(service, legacy_path)
        self.assertIsNone(
            service.store._cleanup_leases["L1"].terminal)
        # The legacy lease still blocks a same-consumer claim.
        with self.assertRaises(ServiceError) as caught:
            service.event_gc_batch_claim({"consumer_id": "c1",
                                          "lease_id": "L2",
                                          "expected": 0, "limit": 1})
        self.assertEqual(caught.exception.field, "consumer_id")
        # And it resolves normally: the first resolution writes seven
        # keys.
        body, status = service.event_gc_batch_lease({
            "consumer_id": "c1", "lease_id": "L1", "expected": 0,
            "op": "release"})
        self.assertEqual(status, 201)
        self.assertEqual(body["after"], 0)
        with open(legacy_path, encoding="utf-8") as handle:
            rewritten = json.load(handle)
        self.assertEqual(
            list(rewritten["cleanup_leases"][0]),
            ["lease_id", "consumer_id", "expected", "next_after",
             "limit", "expires", "terminal"])
        # The migrated file passes the read-only integrity probe.
        self.assertTrue(service.persistence_integrity()["consistent"])

    def test_illegal_terminal_refuses(self) -> None:
        path, document = self._document_with_lease()
        for value in ("done", 0, True, [], {"x": 1}):
            document["cleanup_leases"][0]["terminal"] = value
            bad_path = self._write(dict(
                document, cleanup_leases=[dict(
                    document["cleanup_leases"][0])]))
            with self.subTest(value=value):
                self.assertIn("terminal",
                              self._assert_refuses_startup(bad_path))

    def test_terminal_key_in_wrong_position_refuses(self) -> None:
        path, document = self._document_with_lease()
        lease = document["cleanup_leases"][0]
        reordered = {"terminal": None,
                     "lease_id": lease["lease_id"],
                     "consumer_id": lease["consumer_id"],
                     "expected": lease["expected"],
                     "next_after": lease["next_after"],
                     "limit": lease["limit"],
                     "expires": lease["expires"]}
        document["cleanup_leases"] = [reordered]
        self.assertIn("must have exactly the keys",
                      self._assert_refuses_startup(
                          self._write(document)))

    def test_confirm_without_checkpoint_at_next_after_refuses(self) -> None:
        path, document = self._document_with_lease()
        document["cleanup_leases"][0]["terminal"] = "confirm"
        # No checkpoint section: the consumer's checkpoint reads 0,
        # below next_after 1.
        self.assertIn("checkpoint",
                      self._assert_refuses_startup(
                          self._write(document)))

    def test_confirm_with_checkpoint_at_next_after_loads(self) -> None:
        path, document = self._document_with_lease()
        document["cleanup_leases"][0]["terminal"] = "confirm"
        document["cleanup_checkpoints"].append({
            "consumer_id": "c1", "after": 1,
            "updated_at": "2026-01-01T00:00:00.000000+00:00"})
        service = DeviceService()
        attach_persistence(service, self._write(document))
        self.assertEqual(
            service.store._cleanup_leases["L1"].terminal, "confirm")
        # A terminal confirm never blocks a later claim.
        body, status = service.event_gc_batch_claim({
            "consumer_id": "c1", "lease_id": "L2", "expected": 1,
            "limit": 1})
        self.assertEqual(status, 200)
        self.assertEqual(body["records"], [])

    def test_release_without_checkpoint_loads_and_unblocks(self) -> None:
        path, document = self._document_with_lease()
        # A released lease plus a second still-open lease for the same
        # consumer, both future and unacknowledged: only the
        # unterminal one counts, so startup accepts.
        document["cleanup_leases"][0]["terminal"] = "release"
        document["cleanup_leases"].append(self._lease(
            lease_id="L2", expires=FUTURE))
        service = DeviceService()
        attach_persistence(service, self._write(document))
        self.assertEqual(
            service.store._cleanup_leases["L1"].terminal, "release")
        self.assertIsNone(
            service.store._cleanup_leases["L2"].terminal)
        with self.assertRaises(ServiceError) as caught:
            service.event_gc_batch_claim({"consumer_id": "c1",
                                          "lease_id": "L3",
                                          "expected": 0, "limit": 1})
        self.assertEqual(caught.exception.field, "consumer_id")
        # The released lease still replays its resolution.
        body, status = service.event_gc_batch_lease({
            "consumer_id": "c1", "lease_id": "L1", "expected": 0,
            "op": "release"})
        self.assertEqual(status, 200)
        self.assertEqual(body["after"], 0)

    def test_two_unterminal_open_leases_still_refuse(self) -> None:
        path, document = self._document_with_lease()
        document["cleanup_leases"].append(self._lease(
            lease_id="L2", expires=FUTURE))
        self.assertIn("more than one unacknowledged, unexpired",
                      self._assert_refuses_startup(
                          self._write(document)))


if __name__ == "__main__":
    unittest.main()
