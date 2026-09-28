"""Tests for the batch-cleanup audit claim endpoint.

``POST /v1/event-gc-batch/claim`` returns one page of the batch-cleanup
audit chain (the committed ``event_gc_batch_cleanup_requests`` records,
in commit order) starting at the consumer's checkpoint without
advancing it, and reserves the page under a 30-second lease keyed by a
client-chosen ``lease_id``. The call takes no query parameters (any ->
400/query); the body carries exactly ``consumer_id`` and ``lease_id``
(non-empty strings), ``expected`` (a non-boolean integer in
0..2^63-1) and ``limit`` (a non-boolean integer in 1..100); a
bad/non-object body is 400/request_body and a missing/wrongly
typed/extra field is 400 with that field.

The ``lease_id`` is resolved first: an exact-payload replay answers
200 with the byte-identical first response while the lease is active,
after it was acknowledged (the checkpoint reached ``next_after``) and
after it expired; the id committed with another payload is
409/lease_id. A new id then needs ``expected`` to equal the current
checkpoint (else 409/expected) and a consumer with another
unacknowledged, unexpired lease gets 409/consumer_id. An empty page
answers 200 and occupies neither the id nor a commit generation; a
non-empty page answers 201 and persists once, so a data-file failure
is 503/data_file with the lease rolled back. Success keys are
``consumer_id``, ``lease_id``, ``records``, ``next_after`` and
``expires`` in that order. Leases persist in the ``cleanup_leases``
section right after ``cleanup_checkpoints``.
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

PATH = "/v1/event-gc-batch/claim"
PAST = "2020-01-01T00:00:00.000000+00:00"
FUTURE = "2099-01-01T00:00:00.000000+00:00"


class ClaimMixin:
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

    def _checkpoint(self, consumer="c1"):
        body, _ = self.service.event_gc_batch_checkpoint(
            {"consumer_id": consumer, "expected": None, "after": None})
        return body

    def _expire(self, lease_id):
        self.service.store._cleanup_leases[lease_id].expires = PAST

    @contextmanager
    def _shift_now(self, seconds):
        # Move only the store's "now" (and the 30-second deadline) into
        # the future; fromisoformat keeps parsing stored timestamps.
        import e2ee_backend.storage as storage_mod

        class ShiftedDatetime(datetime):
            @classmethod
            def now(cls, tz=None):  # noqa: ANN001
                return datetime.now(tz) + timedelta(seconds=seconds)

        with mock.patch.object(storage_mod, "datetime", ShiftedDatetime):
            yield


class ClaimServiceTest(ClaimMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def test_empty_chain_200_does_not_occupy_id_or_write(self) -> None:
        body, status = self._claim(lease_id="L1")
        self.assertEqual(status, 200)
        self.assertEqual(list(body),
                         ["consumer_id", "lease_id", "records",
                          "next_after", "expires"])
        self.assertEqual(body["consumer_id"], "c1")
        self.assertEqual(body["lease_id"], "L1")
        self.assertEqual(body["records"], [])
        self.assertEqual(body["next_after"], 0)
        self.assertTrue(body["expires"].endswith("+00:00"))
        # No lease record was created ...
        self.assertEqual(self.service.store._cleanup_leases, {})
        # ... so the id stays free, even for a different payload.
        body2, status2 = self._claim(consumer="other", lease_id="L1",
                                     expected=0, limit=5)
        self.assertEqual(status2, 200)
        self.assertEqual(body2["records"], [])

    def test_nonempty_page_201_without_advancing_checkpoint(self) -> None:
        self._commit("r1")
        self._commit("r2")
        body, status = self._claim(expected=0, limit=1)
        self.assertEqual(status, 201)
        self.assertEqual(list(body),
                         ["consumer_id", "lease_id", "records",
                          "next_after", "expires"])
        self.assertEqual(len(body["records"]), 1)
        self.assertEqual(list(body["records"][0]),
                         ["request_id", "device_ids", "after", "limit",
                          "status", "response"])
        self.assertEqual(body["records"][0]["request_id"], "r1")
        self.assertEqual(body["next_after"], 1)
        expires = datetime.fromisoformat(body["expires"])
        self.assertEqual(
            body["expires"],
            expires.isoformat(timespec="microseconds"))
        self.assertTrue(body["expires"].endswith("+00:00"))
        now = datetime.now(expires.tzinfo)
        self.assertGreaterEqual(
            expires, now + timedelta(seconds=30) - timedelta(seconds=1))
        self.assertLessEqual(expires, now + timedelta(seconds=31))
        # The claim never moves the checkpoint.
        self.assertEqual(self._checkpoint()["after"], 0)

    def test_replay_is_200_byte_identical_in_every_state(self) -> None:
        self._commit("r1")
        self._commit("r2")
        first, status = self._claim(expected=0, limit=1)
        self.assertEqual(status, 201)
        # Active: exact replay.
        replay, status = self._claim(expected=0, limit=1)
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)
        # Acknowledged via consume: still the original response.
        consumed, consume_status = self.service.event_gc_batch_consume(
            {"consumer_id": "c1", "expected": 0, "limit": 100})
        self.assertEqual(consume_status, 201)
        self.assertEqual(consumed["next_after"], 2)
        replay, status = self._claim(expected=0, limit=1)
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)
        # Expired: still the original response (the frozen expires is
        # untouched; only "now" moves past it).
        with self._shift_now(31):
            replay, status = self._claim(expected=0, limit=1)
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)

    def test_replay_page_stays_frozen_as_chain_grows(self) -> None:
        self._commit("r1")
        first, status = self._claim(expected=0, limit=1)
        self.assertEqual(status, 201)
        self._commit("r2")
        self._commit("r3")
        replay, status = self._claim(expected=0, limit=1)
        self.assertEqual(status, 200)
        self.assertEqual([r["request_id"] for r in replay["records"]],
                         ["r1"])
        self.assertEqual(replay, first)

    def test_lease_id_different_payload_is_409(self) -> None:
        self._commit("r1")
        self._commit("r2")
        self.assertEqual(self._claim(expected=0, limit=1)[1], 201)
        for payload in (
                {"consumer_id": "c1", "lease_id": "L1",
                 "expected": 0, "limit": 2},
                {"consumer_id": "c1", "lease_id": "L1",
                 "expected": 1, "limit": 1},
                {"consumer_id": "other", "lease_id": "L1",
                 "expected": 0, "limit": 1}):
            with self.subTest(payload=payload):
                with self.assertRaises(ServiceError) as caught:
                    self.service.event_gc_batch_claim(payload)
                self.assertEqual(caught.exception.status_code, 409)
                self.assertEqual(caught.exception.field, "lease_id")

    def test_lease_id_check_precedes_expected_and_consumer(self) -> None:
        self._commit("r1")
        self.assertEqual(self._claim(expected=0, limit=1)[1], 201)
        # A stale expected that would also be a busy-consumer case is
        # still reported as lease_id because the id resolves first.
        with self.assertRaises(ServiceError) as caught:
            self._claim(lease_id="L1", expected=9, limit=2)
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_expected_mismatch_409(self) -> None:
        self._commit("r1")
        with self.assertRaises(ServiceError) as caught:
            self._claim(lease_id="A", expected=1, limit=100)
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "expected")
        self.assertEqual(self._claim(lease_id="A", expected=0,
                                     limit=100)[1], 201)

    def test_open_lease_blocks_same_consumer_409_consumer_id(self) -> None:
        self._commit("r1")
        self.assertEqual(self._claim(lease_id="L1", limit=1)[1], 201)
        with self.assertRaises(ServiceError) as caught:
            self._claim(lease_id="L2", expected=0, limit=10)
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "consumer_id")

    def test_consumers_are_independent(self) -> None:
        self._commit("r1")
        self.assertEqual(self._claim(consumer="c1", lease_id="L1",
                                     limit=1)[1], 201)
        body2, status2 = self._claim(consumer="c2", lease_id="M1",
                                     expected=0, limit=1)
        self.assertEqual(status2, 201)
        self.assertEqual(body2["consumer_id"], "c2")

    def test_acknowledged_lease_allows_reclaim(self) -> None:
        self._commit("r1")
        self._commit("r2")
        self.assertEqual(self._claim(lease_id="L1", limit=1)[1], 201)
        # Advancing the checkpoint past next_after acknowledges the lease
        # even though its 30-second deadline has not passed.
        body, status = self.service.event_gc_batch_checkpoint({
            "consumer_id": "c1", "expected": 0, "after": 1})
        self.assertEqual(status, 201)
        self.assertEqual(body["after"], 1)
        body2, status2 = self._claim(consumer="c1", lease_id="L2",
                                     expected=1, limit=10)
        self.assertEqual(status2, 201)
        self.assertEqual([r["request_id"] for r in body2["records"]],
                         ["r2"])
        self.assertEqual(body2["next_after"], 2)

    def test_expired_lease_allows_reclaim(self) -> None:
        self._commit("r1")
        self.assertEqual(self._claim(lease_id="L1", limit=1)[1], 201)
        self._expire("L1")
        body2, status2 = self._claim(lease_id="L2", expected=0, limit=10)
        self.assertEqual(status2, 201)
        self.assertEqual(body2["next_after"], 1)

    def test_nonempty_claim_is_persisted_ready_for_restart(self) -> None:
        # Sanity: a lease is reachable through the store index.
        self._commit("r1")
        self.assertEqual(self._claim()[1], 201)
        lease = self.service.store._cleanup_leases["L1"]
        self.assertEqual(
            (lease.lease_id, lease.consumer_id, lease.expected,
             lease.next_after, lease.limit),
            ("L1", "c1", 0, 1, 100))


class ClaimValidationTest(ClaimMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def _expect_400(self, payload, field):
        with self.assertRaises(ServiceError) as caught:
            self.service.event_gc_batch_claim(payload)
        self.assertEqual(caught.exception.status_code, 400)
        self.assertEqual(caught.exception.field, field)
        self.assertEqual(list(caught.exception.to_body()),
                         ["message", "field"])

    def test_body_must_be_object(self) -> None:
        for payload in (None, [], "s", 1, True):
            with self.subTest(payload=payload):
                self._expect_400(payload, "request_body")

    def test_required_fields(self) -> None:
        self._expect_400({"lease_id": "L", "expected": 0, "limit": 1},
                         "consumer_id")
        self._expect_400({"consumer_id": "c", "expected": 0, "limit": 1},
                         "lease_id")
        self._expect_400({"consumer_id": "c", "lease_id": "L",
                          "limit": 1}, "expected")
        self._expect_400({"consumer_id": "c", "lease_id": "L",
                          "expected": 0}, "limit")

    def test_field_shapes(self) -> None:
        base = {"consumer_id": "c", "lease_id": "L", "expected": 0,
                "limit": 1}
        for bad in ("", None, 1, True, []):
            payload = dict(base, consumer_id=bad)
            with self.subTest(bad=bad):
                self._expect_400(payload, "consumer_id")
        for bad in ("", None, 1, True, []):
            payload = dict(base, lease_id=bad)
            with self.subTest(bad=bad):
                self._expect_400(payload, "lease_id")
        for bad in (True, False, -1, 2**63, 1.0, "0", None):
            payload = dict(base, expected=bad)
            with self.subTest(bad=bad):
                self._expect_400(payload, "expected")
        for bad in (True, False, 0, 101, -1, 1.5, "1", None):
            payload = dict(base, limit=bad)
            with self.subTest(bad=bad):
                self._expect_400(payload, "limit")

    def test_extra_field_is_400_with_that_key(self) -> None:
        self._expect_400({"consumer_id": "c", "lease_id": "L",
                          "expected": 0, "limit": 1, "extra": 1},
                         "extra")

    def test_2_63_minus_1_accepted_as_shape(self) -> None:
        # The shape is valid (not 400); on an empty chain it only fails
        # the checkpoint value check, i.e. 409/expected.
        with self.assertRaises(ServiceError) as caught:
            self._claim(lease_id="L", expected=2**63 - 1)
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "expected")


class ClaimHTTPTest(ClaimMixin, unittest.TestCase):
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

    def _claim_http(self, consumer="c1", lease_id="L1", expected=0,
                    limit=100, path=PATH):
        return self._request(path, json.dumps({
            "consumer_id": consumer, "lease_id": lease_id,
            "expected": expected, "limit": limit}))

    def test_claim_over_http_with_key_order(self) -> None:
        self._commit("r1")
        status, body, raw = self._claim_http(limit=1)
        self.assertEqual(status, 201)
        self.assertEqual(list(body),
                         ["consumer_id", "lease_id", "records",
                          "next_after", "expires"])
        for earlier, later in (
                ('"consumer_id"', '"lease_id"'),
                ('"lease_id"', '"records"'),
                ('"records"', '"next_after"'),
                ('"next_after"', '"expires"')):
            self.assertLess(raw.index(earlier), raw.index(later))
        self.assertEqual(body["next_after"], 1)

    def test_query_rejected(self) -> None:
        for query in ("?after=0", "?foo", "?x=1"):
            with self.subTest(query=query):
                status, body, _ = self._claim_http(path=PATH + query)
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], "query")
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
                ({"lease_id": "L", "expected": 0, "limit": 1},
                 "consumer_id"),
                ({"consumer_id": "c", "expected": 0, "limit": 1},
                 "lease_id"),
                ({"consumer_id": "c", "lease_id": "",
                  "expected": 0, "limit": 1}, "lease_id"),
                ({"consumer_id": "c", "lease_id": "L", "limit": 1},
                 "expected"),
                ({"consumer_id": "c", "lease_id": "L", "expected": 0},
                 "limit"),
                ({"consumer_id": "c", "lease_id": "L",
                  "expected": True, "limit": 1}, "expected"),
                ({"consumer_id": "c", "lease_id": "L",
                  "expected": 0, "limit": 0}, "limit"),
                ({"consumer_id": "c", "lease_id": "L",
                  "expected": 0, "limit": 101}, "limit"),
                ({"consumer_id": "c", "lease_id": "L", "expected": 0,
                  "limit": 1, "extra": 1}, "extra")):
            with self.subTest(payload=payload):
                status, body, _ = self._request(raw=json.dumps(payload))
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], field)
                self.assertEqual(list(body), ["message", "field"])

    def test_conflicts_over_http(self) -> None:
        self._commit("r1")
        self.assertEqual(self._claim_http(lease_id="L1", limit=1)[0], 201)
        status, body, _ = self._claim_http(lease_id="L1", limit=2)
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "lease_id")
        status, body, _ = self._claim_http(lease_id="L2", expected=0,
                                           limit=10)
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "consumer_id")
        status, body, _ = self._claim_http(
            consumer="c2", lease_id="M1", expected=5, limit=1)
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "expected")

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

    def test_nonempty_claim_consumes_a_generation_empty_does_not(
            self) -> None:
        self._commit("r1")
        generation = self.state_store.commit_seq
        _body, status = self._claim(lease_id="L1", limit=1)
        self.assertEqual(status, 201)
        self.assertEqual(self.state_store.commit_seq, generation + 1)
        # A consumer at the chain tail (checkpoint 1 == audit count) gets
        # an empty page: 200, no generation, byte-identical document.
        self.service.event_gc_batch_checkpoint({
            "consumer_id": "tail", "expected": 0, "after": 1})
        self.assertEqual(self.state_store.commit_seq, generation + 2)
        before = self._document()
        _body, status = self._claim(consumer="tail", lease_id="T1",
                                    expected=1, limit=100)
        self.assertEqual(status, 200)
        self.assertEqual(self.state_store.commit_seq, generation + 2)
        self.assertEqual(self._document(), before)

    def test_section_order_and_record_shape(self) -> None:
        self._commit("r1")
        self._claim(lease_id="L1", limit=1)
        document = self._document()
        keys = list(document)
        self.assertEqual(
            keys.index("cleanup_leases"),
            keys.index("cleanup_checkpoints") + 1)
        section = document["cleanup_leases"]
        self.assertEqual(len(section), 1)
        self.assertEqual(list(section[0]),
                         ["lease_id", "consumer_id", "expected",
                          "next_after", "limit", "expires", "terminal",
                          "renewals"])
        self.assertEqual(
            {k: section[0][k] for k in
             ("lease_id", "consumer_id", "expected", "next_after",
              "limit")},
            {"lease_id": "L1", "consumer_id": "c1", "expected": 0,
             "next_after": 1, "limit": 1})
        # A freshly claimed lease carries no explicit resolution and no
        # renewals yet (new writes always carry all eight keys).
        self.assertIsNone(section[0]["terminal"])
        self.assertEqual(section[0]["renewals"], [])

    def test_empty_page_writes_no_section_entries(self) -> None:
        _body, status = self._claim()
        self.assertEqual(status, 200)
        self.assertEqual(self._document()["cleanup_leases"], [])

    def test_restart_replays_and_still_blocks(self) -> None:
        self._commit("r1")
        first, status = self._claim(lease_id="L1", limit=1)
        self.assertEqual(status, 201)
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        replay, status = restarted.event_gc_batch_claim({
            "consumer_id": "c1", "lease_id": "L1",
            "expected": 0, "limit": 1})
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)
        with self.assertRaises(ServiceError) as caught:
            restarted.event_gc_batch_claim({
                "consumer_id": "c1", "lease_id": "L2",
                "expected": 0, "limit": 10})
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "consumer_id")
        self.assertTrue(
            restarted.persistence_integrity()["consistent"])

    def test_restart_acknowledged_or_expired_lease_does_not_block(self) -> None:
        self._commit("r1")
        self._commit("r2")
        self.assertEqual(self._claim(consumer="c1", lease_id="L1",
                                     limit=1)[1], 201)
        self.assertEqual(self._claim(consumer="c2", lease_id="M1",
                                     limit=1)[1], 201)
        # Acknowledge c1's lease durably; then build a fresh
        # marker-less/sidecar-less document in which c2's lease is
        # expired (stripping the integrity fields bypasses the hash
        # gate so the payload's own validation is what matters).
        self.service.event_gc_batch_checkpoint({
            "consumer_id": "c1", "expected": 0, "after": 1})
        document = self._document()
        for lease in document["cleanup_leases"]:
            if lease["lease_id"] == "M1":
                lease["expires"] = PAST
        document.pop("integrity_log_version", None)
        legacy_path = os.path.join(self.directory, "restart.json")
        with open(legacy_path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        restarted = DeviceService()
        attach_persistence(restarted, legacy_path)
        body1, status1 = restarted.event_gc_batch_claim({
            "consumer_id": "c1", "lease_id": "L2",
            "expected": 1, "limit": 10})
        self.assertEqual(status1, 201)
        self.assertEqual(body1["next_after"], 2)
        body2, status2 = restarted.event_gc_batch_claim({
            "consumer_id": "c2", "lease_id": "M2",
            "expected": 0, "limit": 10})
        self.assertEqual(status2, 201)
        self.assertEqual(body2["next_after"], 2)

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
                self._claim(lease_id="L1", limit=1)
        finally:
            persistence_mod.os.fsync = real_fsync
        self.assertEqual(self.state_store.commit_seq, generation)
        self.assertEqual(self.service.store._cleanup_leases, {})
        self.assertEqual(self._document()["cleanup_leases"], [])
        # The claim can be retried and now commits.
        body, status = self._claim(lease_id="L1", limit=1)
        self.assertEqual(status, 201)
        self.assertEqual(body["next_after"], 1)
        self.assertEqual(self.state_store.commit_seq, generation + 1)


class ClaimRestoreValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self._seed_n = 0

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def _document_with_lease(self, mutate):
        # A fresh service per call: two committed audit records and one
        # committed lease, so every mutation starts from the same
        # document regardless of subTest order. The document is written
        # to a fresh marker-less/sidecar-less path, so rejection comes
        # from the payload's own semantic validation rather than the
        # integrity hash gate.
        self._seed_n += 1
        service = DeviceService()
        service.store.add_device(Device("u", "bob", "ik"))
        seed_path = os.path.join(self.directory,
                                 f"seed{self._seed_n}.json")
        attach_persistence(service, seed_path)
        service.event_gc_cleanup_expired_batch({
            "mode": "commit", "device_ids": ["bob"], "after": 0,
            "limit": 100, "request_id": "r1"})
        service.event_gc_cleanup_expired_batch({
            "mode": "commit", "device_ids": ["bob"], "after": 0,
            "limit": 100, "request_id": "r2"})
        service.event_gc_batch_claim({"consumer_id": "c1",
                                      "lease_id": "L1",
                                      "expected": 0, "limit": 1})
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
        # The rejected file is never overwritten.
        with open(bad_path, "rb") as handle:
            self.assertEqual(handle.read(), original)
        return str(caught.exception)

    @staticmethod
    def _lease(lease_id="L1", consumer="c1", expected=0, next_after=1,
               limit=1, expires=FUTURE):
        return {"lease_id": lease_id, "consumer_id": consumer,
                "expected": expected, "next_after": next_after,
                "limit": limit, "expires": expires}

    def test_legacy_document_without_section_loads(self) -> None:
        def mutate(document):
            # A file predating the cleanup lease feature carries
            # neither the leases section nor the later lease-event
            # stream; both missing sections load as empty.
            document.pop("cleanup_leases")
            document.pop("cleanup_lease_events")
        path = self._document_with_lease(mutate)
        service = DeviceService()
        attach_persistence(service, path)  # must not raise
        self.assertEqual(service.store._cleanup_leases, {})
        self.assertTrue(service.persistence_integrity()["consistent"])

    def test_malformed_section_refuses(self) -> None:
        message = self._assert_refuses_startup(
            self._document_with_lease(
                lambda d: d.update(cleanup_leases={})))
        self.assertIn("malformed", message)

    def test_duplicate_lease_id_refuses(self) -> None:
        def mutate(document):
            document["cleanup_leases"] = [
                self._lease("L1"), self._lease("L1", consumer="c2",
                                               expected=1, next_after=2)]
        message = self._assert_refuses_startup(
            self._document_with_lease(mutate))
        self.assertIn("duplicate cleanup lease", message)

    def test_bad_key_order_refuses(self) -> None:
        def mutate(document):
            document["cleanup_leases"] = [{
                "lease_id": "L1", "expected": 0, "consumer_id": "c1",
                "next_after": 1, "limit": 1, "expires": FUTURE}]
        message = self._assert_refuses_startup(
            self._document_with_lease(mutate))
        self.assertIn("must have exactly the keys", message)

    def test_bad_value_shapes_refuse(self) -> None:
        def set_leases(lease):
            def mutate(document):
                document["cleanup_leases"] = [lease]
            return mutate

        for mutate in (
                set_leases(self._lease(lease_id="")),
                set_leases(self._lease(consumer="")),
                set_leases(self._lease(expected=True)),
                set_leases(self._lease(expected=-1)),
                set_leases(self._lease(next_after=99)),
                set_leases(self._lease(limit=0)),
                set_leases(self._lease(limit=101)),
                set_leases(self._lease(next_after=0)),
                set_leases(self._lease(expected=1, next_after=1)),
                set_leases(self._lease(expected=0, next_after=2,
                                       limit=1)),
                set_leases(self._lease(
                    expires="2020-01-01T00:00:00+00:00")),
                set_leases("x")):
            with self.subTest(mutate=mutate):
                self.assertIn("malformed payload",
                              self._assert_refuses_startup(
                                  self._document_with_lease(mutate)))

    def test_two_open_leases_per_consumer_refuse(self) -> None:
        def mutate(document):
            document["cleanup_leases"] = [
                self._lease("L1", expected=0, next_after=1,
                            expires=FUTURE),
                self._lease("L2", expected=1, next_after=2,
                            expires=FUTURE)]
        message = self._assert_refuses_startup(
            self._document_with_lease(mutate))
        self.assertIn("more than one unacknowledged, unexpired", message)

    def test_acknowledged_or_expired_second_leases_accepted(self) -> None:
        # c1's first lease is acknowledged (checkpoint at 1), so a later
        # still-future lease for offsets 1..2 is the only open one.
        def mutate_ack(document):
            document["cleanup_leases"] = [
                self._lease("L1", expected=0, next_after=1,
                            expires=FUTURE),
                self._lease("L2", expected=1, next_after=2,
                            expires=FUTURE)]
            document["cleanup_checkpoints"].append({
                "consumer_id": "c1", "after": 1,
                "updated_at": "2026-01-01T00:00:00.000000+00:00"})
        path = self._document_with_lease(mutate_ack)
        service = DeviceService()
        attach_persistence(service, path)
        self.assertEqual(set(service.store._cleanup_leases),
                         {"L1", "L2"})
        self.assertTrue(service.persistence_integrity()["consistent"])

        # Two past (expired) leases for one consumer are also fine.
        def mutate_expired(document):
            document["cleanup_leases"] = [
                self._lease("L1", expires=PAST),
                self._lease("L2", expected=1, next_after=2,
                            expires=PAST)]
        path2 = self._document_with_lease(mutate_expired)
        service2 = DeviceService()
        attach_persistence(service2, path2)
        self.assertEqual(set(service2.store._cleanup_leases),
                         {"L1", "L2"})


if __name__ == "__main__":
    unittest.main()
