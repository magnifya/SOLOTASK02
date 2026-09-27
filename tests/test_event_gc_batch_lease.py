"""Tests for the batch-cleanup audit lease confirm/release endpoint.

``POST /v1/event-gc-batch/lease`` resolves a page lease previously
granted by ``POST /v1/event-gc-batch/claim``. The call takes no query
parameters (any -> 400/query); the body carries exactly
``consumer_id`` and ``lease_id`` (non-empty strings), ``expected`` (a
non-boolean integer in 0..2^63-1) and ``op`` (``confirm`` or
``release``); a bad/non-object body is 400/request_body and a
missing/wrongly typed/extra field is 400 with that field.

Resolution order under the one store lock: an unknown lease id is
404/lease_id and a lease owned by another consumer is
409/consumer_id; a lease already resolved with the same op replays its
frozen first response (200) while the other op is 409/lease_id; an
unacknowledged lease whose deadline has passed is 409/lease_id for
either op (before the compare-and-set); once the checkpoint reached
``next_after`` a confirm is a read-only 200 (regardless of expected)
and a release is 409/lease_id; a first resolution on an active
unacknowledged lease needs ``expected`` to equal both the lease start
and the current checkpoint (else 409/expected). A first confirm
advances the checkpoint to ``next_after`` (201); a first release moves
no checkpoint and unblocks a later claim (201). Success keys are
``consumer_id``, ``lease_id``, ``op`` and ``after`` in that order
(``after`` is ``next_after`` for confirm, ``expected`` for release).

The lease's ``cleanup_leases`` record gains a seventh key
``terminal`` (null/"confirm"/"release"); old six-key records load with
null, malformed values or a confirmed lease whose checkpoint is short
of ``next_after`` refuse startup.
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

    def _lease_op(self, consumer="c1", lease_id="L1", expected=0,
                  op="confirm"):
        return self.service.event_gc_batch_lease_op({
            "consumer_id": consumer, "lease_id": lease_id,
            "expected": expected, "op": op})

    def _checkpoint(self, consumer="c1"):
        body, _ = self.service.event_gc_batch_checkpoint(
            {"consumer_id": consumer, "expected": None, "after": None})
        return body

    def _expire(self, lease_id):
        self.service.store._cleanup_leases[lease_id].expires = PAST


class LeaseServiceTest(LeaseMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def test_first_confirm_201_advances_checkpoint(self) -> None:
        self._commit("r1")
        self._commit("r2")
        self._claim(limit=2)
        body, status = self._lease_op(op="confirm")
        self.assertEqual(status, 201)
        self.assertEqual(list(body),
                         ["consumer_id", "lease_id", "op", "after"])
        self.assertEqual(body, {
            "consumer_id": "c1", "lease_id": "L1",
            "op": "confirm", "after": 2})
        self.assertEqual(self._checkpoint()["after"], 2)
        lease = self.service.store._cleanup_leases["L1"]
        self.assertEqual(lease.terminal, "confirm")

    def test_first_release_201_does_not_advance_and_unblocks(self) -> None:
        self._commit("r1")
        self._claim(limit=1)
        body, status = self._lease_op(op="release")
        self.assertEqual(status, 201)
        self.assertEqual(body, {
            "consumer_id": "c1", "lease_id": "L1",
            "op": "release", "after": 0})
        # The checkpoint never moves.
        self.assertEqual(self._checkpoint()["after"], 0)
        lease = self.service.store._cleanup_leases["L1"]
        self.assertEqual(lease.terminal, "release")
        # The released lease no longer blocks a new claim.
        again, claim_status = self._claim(lease_id="L2", limit=1)
        self.assertEqual(claim_status, 201)
        self.assertEqual(again["next_after"], 1)

    def test_same_op_replay_is_200_byte_identical(self) -> None:
        self._commit("r1")
        self._claim(limit=1)
        first, status = self._lease_op(op="confirm")
        self.assertEqual(status, 201)
        # Same op replays with the first body and writes nothing.
        replay, status = self._lease_op(op="confirm")
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)
        # The replay is robust to a stale ``expected`` too.
        replay, status = self._lease_op(expected=9, op="confirm")
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)

        # Release flow on a second lease behaves the same. The first
        # confirm advanced c1's checkpoint to 1; commit another record
        # and claim the next page from offset 1.
        self._commit("r2")
        self._claim(lease_id="L2", expected=1, limit=1)
        first_rel, status = self._lease_op(lease_id="L2", expected=1,
                                           op="release")
        self.assertEqual(status, 201)
        replay, status = self._lease_op(lease_id="L2", expected=1,
                                        op="release")
        self.assertEqual(status, 200)
        self.assertEqual(replay, first_rel)

    def test_different_op_after_terminal_is_409(self) -> None:
        self._commit("r1")
        self._claim(limit=1)
        self.assertEqual(self._lease_op(op="confirm")[1], 201)
        with self.assertRaises(ServiceError) as caught:
            self._lease_op(op="release")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")

        # Release then confirm on another lease.
        self._commit("r2")
        self._claim(lease_id="L2", expected=1, limit=1)
        self.assertEqual(
            self._lease_op(lease_id="L2", expected=1,
                           op="release")[1], 201)
        with self.assertRaises(ServiceError) as caught:
            self._lease_op(lease_id="L2", expected=1, op="confirm")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_unknown_lease_404(self) -> None:
        with self.assertRaises(ServiceError) as caught:
            self._lease_op(lease_id="nope")
        self.assertEqual(caught.exception.status_code, 404)
        self.assertEqual(caught.exception.field, "lease_id")
        self.assertEqual(list(caught.exception.to_body()),
                         ["message", "field"])

    def test_consumer_mismatch_409_consumer_id(self) -> None:
        self._commit("r1")
        self._claim(limit=1)
        with self.assertRaises(ServiceError) as caught:
            self._lease_op(consumer="other")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "consumer_id")
        # The conflict does not resolve the lease.
        self.assertIsNone(
            self.service.store._cleanup_leases["L1"].terminal)

    def test_first_op_expected_must_match_lease_start_and_checkpoint(
            self) -> None:
        self._commit("r1")
        self._commit("r2")
        self._claim(limit=2)
        # Move the checkpoint forward but keep it below next_after.
        body, status = self.service.event_gc_batch_checkpoint({
            "consumer_id": "c1", "expected": 0, "after": 1})
        self.assertEqual(status, 201)
        self.assertEqual(body["after"], 1)
        # Lease start 0 no longer equals the current checkpoint 1.
        with self.assertRaises(ServiceError) as caught:
            self._lease_op(expected=0, op="confirm")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "expected")
        with self.assertRaises(ServiceError) as caught:
            self._lease_op(expected=0, op="release")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "expected")
        # A value matching neither is reported as expected too.
        with self.assertRaises(ServiceError) as caught:
            self._lease_op(expected=2, op="confirm")
        self.assertEqual(caught.exception.field, "expected")

    def test_expired_unacknowledged_lease_409_before_expected(self) -> None:
        self._commit("r1")
        self._claim(limit=1)
        self._expire("L1")
        # Even a stale expected that would fail the compare-and-set is
        # reported as lease_id: the expiry state wins.
        with self.assertRaises(ServiceError) as caught:
            self._lease_op(expected=9, op="confirm")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")
        with self.assertRaises(ServiceError) as caught:
            self._lease_op(expected=9, op="release")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")
        self.assertIsNone(
            self.service.store._cleanup_leases["L1"].terminal)

    def test_checkpoint_at_next_after_confirm_readonly_release_409(
            self) -> None:
        self._commit("r1")
        self._claim(limit=1)
        # Acknowledge through consume rather than the lease endpoint.
        consumed, status = self.service.event_gc_batch_consume({
            "consumer_id": "c1", "expected": 0, "limit": 100})
        self.assertEqual(status, 201)
        self.assertEqual(consumed["next_after"], 1)
        # Confirm is a read-only 200 even with a stale ``expected`` and
        # records no terminal marker.
        body, status = self._lease_op(expected=0, op="confirm")
        self.assertEqual(status, 200)
        self.assertEqual(body["after"], 1)
        self.assertIsNone(
            self.service.store._cleanup_leases["L1"].terminal)
        # Repeated confirms stay read-only.
        _, status = self._lease_op(expected=1, op="confirm")
        self.assertEqual(status, 200)
        # A release is rejected once acknowledged.
        with self.assertRaises(ServiceError) as caught:
            self._lease_op(expected=1, op="release")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_consumers_are_independent(self) -> None:
        self._commit("r1")
        self._claim(consumer="c1", lease_id="L1", limit=1)
        self._claim(consumer="c2", lease_id="M1", limit=1)
        body, status = self._lease_op(consumer="c2", lease_id="M1",
                                      expected=0, op="release")
        self.assertEqual(status, 201)
        self.assertEqual(body["consumer_id"], "c2")
        # c1's lease is untouched and still open.
        lease = self.service.store._cleanup_leases["L1"]
        self.assertIsNone(lease.terminal)
        with self.assertRaises(ServiceError) as caught:
            self._claim(consumer="c1", lease_id="L2", limit=1)
        self.assertEqual(caught.exception.field, "consumer_id")


class LeaseValidationTest(LeaseMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def _expect_400(self, payload, field):
        with self.assertRaises(ServiceError) as caught:
            self.service.event_gc_batch_lease_op(payload)
        self.assertEqual(caught.exception.status_code, 400)
        self.assertEqual(caught.exception.field, field)
        self.assertEqual(list(caught.exception.to_body()),
                         ["message", "field"])

    def test_body_must_be_object(self) -> None:
        for payload in (None, [], "s", 1, True):
            with self.subTest(payload=payload):
                self._expect_400(payload, "request_body")

    def test_required_fields(self) -> None:
        self._expect_400(
            {"lease_id": "L", "expected": 0, "op": "confirm"},
            "consumer_id")
        self._expect_400(
            {"consumer_id": "c", "expected": 0, "op": "confirm"},
            "lease_id")
        self._expect_400(
            {"consumer_id": "c", "lease_id": "L", "op": "confirm"},
            "expected")
        self._expect_400(
            {"consumer_id": "c", "lease_id": "L", "expected": 0},
            "op")

    def test_field_shapes(self) -> None:
        base = {"consumer_id": "c", "lease_id": "L", "expected": 0,
                "op": "confirm"}
        for bad in ("", None, 1, True, []):
            with self.subTest(bad=("consumer", bad)):
                self._expect_400(dict(base, consumer_id=bad),
                                 "consumer_id")
        for bad in ("", None, 1, True, []):
            with self.subTest(bad=("lease", bad)):
                self._expect_400(dict(base, lease_id=bad), "lease_id")
        for bad in (True, False, -1, 2**63, 1.0, "0", None):
            with self.subTest(bad=("expected", bad)):
                self._expect_400(dict(base, expected=bad), "expected")
        for bad in (None, "", "CONFIRM", "ack", 1, True, []):
            with self.subTest(bad=("op", bad)):
                self._expect_400(dict(base, op=bad), "op")

    def test_extra_field_is_400_with_that_key(self) -> None:
        self._expect_400({"consumer_id": "c", "lease_id": "L",
                          "expected": 0, "op": "confirm", "extra": 1},
                         "extra")

    def test_2_63_minus_1_accepted_as_shape(self) -> None:
        # The shape is valid (not 400); with no lease it fails as 404.
        with self.assertRaises(ServiceError) as caught:
            self._lease_op(expected=2**63 - 1)
        self.assertEqual(caught.exception.status_code, 404)
        self.assertEqual(caught.exception.field, "lease_id")


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

    def _op_http(self, consumer="c1", lease_id="L1", expected=0,
                 op="confirm", path=PATH):
        return self._request(path, json.dumps({
            "consumer_id": consumer, "lease_id": lease_id,
            "expected": expected, "op": op}))

    def test_confirm_over_http_with_key_order(self) -> None:
        self._commit("r1")
        self._claim(limit=1)
        status, body, raw = self._op_http(op="confirm")
        self.assertEqual(status, 201)
        self.assertEqual(list(body),
                         ["consumer_id", "lease_id", "op", "after"])
        for earlier, later in (
                ('"consumer_id"', '"lease_id"'),
                ('"lease_id"', '"op"'),
                ('"op"', '"after"')):
            self.assertLess(raw.index(earlier), raw.index(later))
        self.assertEqual(body["after"], 1)

    def test_query_rejected(self) -> None:
        for query in ("?after=0", "?foo", "?x=1"):
            with self.subTest(query=query):
                status, body, _ = self._op_http(path=PATH + query)
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], "query")
        status, _, _ = self._op_http(path=PATH + "?")
        self.assertEqual(status, 404)

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
                ({"consumer_id": "c", "lease_id": "L", "expected": 0,
                  "op": "ack"}, "op"),
                ({"consumer_id": "c", "lease_id": "L", "expected": 0,
                  "op": "confirm", "extra": 1}, "extra")):
            with self.subTest(payload=payload):
                status, body, _ = self._request(raw=json.dumps(payload))
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], field)
                self.assertEqual(list(body), ["message", "field"])

    def test_conflicts_over_http(self) -> None:
        self._commit("r1")
        self._claim(limit=1)
        # Unknown lease.
        status, body, _ = self._op_http(lease_id="nope")
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "lease_id")
        # Other consumer.
        status, body, _ = self._op_http(consumer="other")
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "consumer_id")
        # Stale expected.
        status, body, _ = self._op_http(expected=5)
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "expected")
        # Confirm succeeds, release then conflicts.
        self.assertEqual(self._op_http()[0], 201)
        status, body, _ = self._op_http(op="release")
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "lease_id")

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

    def test_first_ops_consume_one_generation_each(self) -> None:
        self._commit("r1")
        self._commit("r2")
        self._claim(limit=2)
        generation = self.state_store.commit_seq
        self.assertEqual(self._lease_op(op="confirm")[1], 201)
        self.assertEqual(self.state_store.commit_seq, generation + 1)
        # Same-op replay consumes no generation.
        self.assertEqual(self._lease_op(op="confirm")[1], 200)
        self.assertEqual(self.state_store.commit_seq, generation + 1)

    def test_readonly_confirm_consumes_no_generation(self) -> None:
        self._commit("r1")
        self._claim(limit=1)
        self.service.event_gc_batch_consume({
            "consumer_id": "c1", "expected": 0, "limit": 100})
        generation = self.state_store.commit_seq
        before = self._document()
        _, status = self._lease_op(op="confirm")
        self.assertEqual(status, 200)
        self.assertEqual(self.state_store.commit_seq, generation)
        self.assertEqual(self._document(), before)

    def test_confirm_writes_terminal_and_checkpoint(self) -> None:
        self._commit("r1")
        self._claim(limit=1)
        self.assertEqual(self._lease_op(op="confirm")[1], 201)
        section = self._document()["cleanup_leases"]
        self.assertEqual(len(section), 1)
        self.assertEqual(list(section[0]),
                         ["lease_id", "consumer_id", "expected",
                          "next_after", "limit", "expires", "terminal",
                          "renewals"])
        self.assertEqual(section[0]["terminal"], "confirm")
        self.assertEqual(section[0]["renewals"], [])
        checkpoints = self._document()["cleanup_checkpoints"]
        self.assertEqual(len(checkpoints), 1)
        self.assertEqual(checkpoints[0]["after"], 1)

    def test_release_writes_terminal_without_checkpoint(self) -> None:
        self._commit("r1")
        self._claim(limit=1)
        self.assertEqual(self._lease_op(op="release")[1], 201)
        section = self._document()["cleanup_leases"]
        self.assertEqual(section[0]["terminal"], "release")
        self.assertEqual(self._document()["cleanup_checkpoints"], [])

    def test_restart_preserves_terminal_and_replay(self) -> None:
        self._commit("r1")
        self._claim(limit=1)
        first, status = self._lease_op(op="confirm")
        self.assertEqual(status, 201)
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        lease = restarted.store._cleanup_leases["L1"]
        self.assertEqual(lease.terminal, "confirm")
        replay, status = restarted.event_gc_batch_lease_op({
            "consumer_id": "c1", "lease_id": "L1", "expected": 0,
            "op": "confirm"})
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)
        self.assertTrue(
            restarted.persistence_integrity()["consistent"])

    def test_restart_released_lease_does_not_block(self) -> None:
        self._commit("r1")
        self._claim(limit=1)
        self.assertEqual(self._lease_op(op="release")[1], 201)
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        body, status = restarted.event_gc_batch_claim({
            "consumer_id": "c1", "lease_id": "L2", "expected": 0,
            "limit": 1})
        self.assertEqual(status, 201)
        self.assertEqual(body["next_after"], 1)

    def test_legacy_six_key_record_loads_with_terminal_null(self) -> None:
        self._commit("r1")
        self._claim(limit=1)
        document = self._document()
        for lease in document["cleanup_leases"]:
            lease.pop("terminal")
            lease.pop("renewals")
        document.pop("integrity_log_version", None)
        legacy_path = os.path.join(self.directory, "legacy.json")
        with open(legacy_path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        restarted = DeviceService()
        attach_persistence(restarted, legacy_path)
        lease = restarted.store._cleanup_leases["L1"]
        self.assertIsNone(lease.terminal)
        self.assertTrue(
            restarted.persistence_integrity()["consistent"])

    def test_save_failure_rolls_resolution_back(self) -> None:
        import e2ee_backend.persistence as persistence_mod

        self._commit("r1")
        self._claim(limit=1)
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
                self._lease_op(op="confirm")
        finally:
            persistence_mod.os.fsync = real_fsync
        self.assertEqual(self.state_store.commit_seq, generation)
        lease = self.service.store._cleanup_leases["L1"]
        self.assertIsNone(lease.terminal)
        self.assertEqual(self._checkpoint()["after"], 0)
        # The resolution can be retried and now commits.
        body, status = self._lease_op(op="confirm")
        self.assertEqual(status, 201)
        self.assertEqual(body["after"], 1)
        self.assertEqual(self.state_store.commit_seq, generation + 1)


class LeaseConcurrencyTest(LeaseMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()
        self._commit("r1")
        self._claim(limit=1)

    def test_concurrent_confirms_linearize_to_one_201(self) -> None:
        results = []
        barrier = threading.Barrier(8)

        def worker() -> None:
            barrier.wait()
            try:
                _body, status = self._lease_op(op="confirm")
                results.append(status)
            except ServiceError as error:
                results.append(error.status_code)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
        self.assertEqual(sorted(results).count(201), 1)
        self.assertEqual(sorted(results).count(200), 7)


class LeaseRestoreValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self._seed_n = 0

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def _document_with_lease(self, mutate, terminal=None, commits=1):
        self._seed_n += 1
        service = DeviceService()
        service.store.add_device(Device("u", "bob", "ik"))
        seed_path = os.path.join(self.directory,
                                 f"seed{self._seed_n}.json")
        attach_persistence(service, seed_path)
        for number in range(1, commits + 1):
            service.event_gc_cleanup_expired_batch({
                "mode": "commit", "device_ids": ["bob"], "after": 0,
                "limit": 100, "request_id": f"r{number}"})
        service.event_gc_batch_claim({"consumer_id": "c1",
                                      "lease_id": "L1",
                                      "expected": 0, "limit": 1})
        if terminal is not None:
            service.event_gc_batch_lease_op({
                "consumer_id": "c1", "lease_id": "L1",
                "expected": 0, "op": terminal})
        with open(seed_path, encoding="utf-8") as handle:
            document = json.load(handle)
        mutate(document)
        document.pop("integrity_log_version", None)
        bad_path = os.path.join(self.directory, f"bad{self._seed_n}.json")
        with open(bad_path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        return bad_path

    def _assert_refuses_startup(self, bad_path) -> str:
        with open(bad_path, "rb") as handle:
            original = handle.read()
        service = DeviceService()
        with self.assertRaises(StateFileError) as caught:
            attach_persistence(service, bad_path)
        with open(bad_path, "rb") as handle:
            self.assertEqual(handle.read(), original)
        return str(caught.exception)

    @staticmethod
    def _lease(lease_id="L1", consumer="c1", expected=0, next_after=1,
               limit=1, expires=FUTURE, terminal=None):
        record = {"lease_id": lease_id, "consumer_id": consumer,
                  "expected": expected, "next_after": next_after,
                  "limit": limit, "expires": expires}
        if terminal is not None:
            record["terminal"] = terminal
        return record

    def test_bad_terminal_value_refuses(self) -> None:
        def mutate(document):
            document["cleanup_leases"][0]["terminal"] = "ack"
        message = self._assert_refuses_startup(
            self._document_with_lease(mutate))
        self.assertIn("terminal", message)

    def test_terminal_wrong_type_refuses(self) -> None:
        def mutate(document):
            document["cleanup_leases"][0]["terminal"] = True
        message = self._assert_refuses_startup(
            self._document_with_lease(mutate))
        self.assertIn("malformed payload", message)

    def test_confirmed_lease_with_short_checkpoint_refuses(self) -> None:
        # A confirmed lease requires the consumer's stored checkpoint to
        # have reached next_after.
        def mutate(document):
            document["cleanup_leases"][0]["terminal"] = "confirm"
        message = self._assert_refuses_startup(
            self._document_with_lease(mutate))
        self.assertIn("confirmed", message)

    def test_released_open_lease_is_not_counted_as_open(self) -> None:
        # One released unacknowledged, unexpired lease plus one ordinary
        # open lease for the same consumer must load: the released lease
        # no longer counts toward the one-open-lease invariant.
        def mutate(document):
            document["cleanup_leases"] = [
                self._lease("L1", terminal="release"),
                self._lease("L2", expected=1, next_after=2)]
            # The hand-built seven-key leases stand for a legacy file
            # that predates the lifecycle event stream section.
            document.pop("cleanup_lease_events", None)
        path = self._document_with_lease(mutate, commits=2)
        service = DeviceService()
        attach_persistence(service, path)
        self.assertEqual(
            service.store._cleanup_leases["L1"].terminal, "release")

    def test_confirmed_lease_with_checkpoint_at_next_after_loads(
            self) -> None:
        path = self._document_with_lease(lambda d: None,
                                         terminal="confirm")
        service = DeviceService()
        attach_persistence(service, path)
        self.assertEqual(
            service.store._cleanup_leases["L1"].terminal, "confirm")
        self.assertTrue(service.persistence_integrity()["consistent"])


if __name__ == "__main__":
    unittest.main()
