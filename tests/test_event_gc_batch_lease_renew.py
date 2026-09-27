"""Tests for the batch-cleanup audit lease renewal endpoint.

``POST /v1/event-gc-batch/lease/renew`` extends a page lease previously
granted by ``POST /v1/event-gc-batch/claim`` by 30 seconds. The call
takes no query parameters (any -> 400/query); the body carries exactly
``consumer_id``, ``lease_id`` and ``renewal_id``, all non-empty strings;
a bad/non-object body is 400/request_body and a missing/wrongly typed/
extra field is 400 with that field.

Resolution order under the one store lock: an unknown lease id is
404/lease_id and a lease owned by another consumer is
409/consumer_id; the same renewal_id on the same lease replays its
frozen first response (200) even after the lease was resolved, expired
or the checkpoint moved. A first renewal requires the lease to be
unterminated, unexpired (against its effective deadline) and the
consumer's checkpoint to still equal the lease start while being short
of next_after (otherwise 409/lease_id); the eleventh renewal is
409/lease_id. A first renewal answers 201 and extends the effective
``expires`` by exactly 30 seconds; the claim deadline itself stays
frozen. Success keys are ``consumer_id``, ``lease_id``,
``renewal_id`` and ``expires`` in that order (UTC, six microsecond
digits, +00:00).

The lease's ``cleanup_leases`` record gains an eighth key ``renewals``
(elements ``renewal_id``/``expires``); old six-/seven-key records load
with an empty chain. A duplicate renewal id, a deadline that does not
extend the previous effective deadline by exactly 30 seconds, or more
than ten entries refuse startup without overwriting the file.
"""
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
    StateFileError,
    attach_persistence,
)

PATH = "/v1/event-gc-batch/lease/renew"
PAST = "2020-01-01T00:00:00.000000+00:00"


def _parse(stamp):
    return datetime.fromisoformat(stamp)


class RenewMixin:
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

    def _renew(self, consumer="c1", lease_id="L1", renewal_id="R1",
               payload=None):
        body = payload if payload is not None else {
            "consumer_id": consumer, "lease_id": lease_id,
            "renewal_id": renewal_id}
        return self.service.event_gc_batch_lease_renew(body)

    def _lease(self, lease_id="L1"):
        return self.service.store._cleanup_leases[lease_id]

    def _set_claim_expires(self, lease_id, stamp):
        lease = self._lease(lease_id)
        lease.expires = stamp

    def _set_effective_expires(self, lease_id, stamp):
        lease = self._lease(lease_id)
        if lease.renewals:
            lease.renewals[-1].expires = stamp
        else:
            lease.expires = stamp


class RenewServiceTest(RenewMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def test_first_renew_201_extends_thirty_seconds(self) -> None:
        self._commit("r1")
        claim, status = self._claim(limit=1)
        self.assertEqual(status, 201)
        claimed_expires = claim["expires"]
        body, status = self._renew()
        self.assertEqual(status, 201)
        self.assertEqual(list(body),
                         ["consumer_id", "lease_id", "renewal_id",
                          "expires"])
        self.assertEqual(body["consumer_id"], "c1")
        self.assertEqual(body["lease_id"], "L1")
        self.assertEqual(body["renewal_id"], "R1")
        self.assertEqual(
            _parse(body["expires"]),
            _parse(claimed_expires) + timedelta(seconds=30))
        self.assertTrue(body["expires"].endswith("+00:00"))
        self.assertEqual(len(self._lease().renewals), 1)

    def test_ten_renewals_chain_exactly_thirty_seconds(self) -> None:
        self._commit("r1")
        claim, _ = self._claim(limit=1)
        previous = claim["expires"]
        for number in range(1, 11):
            body, status = self._renew(renewal_id=f"R{number}")
            self.assertEqual(status, 201)
            self.assertEqual(
                _parse(body["expires"]),
                _parse(previous) + timedelta(seconds=30))
            previous = body["expires"]
        self.assertEqual(len(self._lease().renewals), 10)
        # The eleventh renewal is 409/lease_id.
        with self.assertRaises(ServiceError) as caught:
            self._renew(renewal_id="R11")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")
        self.assertEqual(len(self._lease().renewals), 10)
        # Replays keep answering 200 past the cap (replay precedes the
        # first-time checks and writes nothing).
        replay, status = self._renew(renewal_id="R1")
        self.assertEqual(status, 200)
        self.assertEqual(replay["renewal_id"], "R1")
        self.assertEqual(len(self._lease().renewals), 10)

    def test_same_renewal_id_replays_200_byte_identical(self) -> None:
        self._commit("r1")
        self._claim(limit=1)
        first, status = self._renew(renewal_id="R1")
        self.assertEqual(status, 201)
        # A second renewal advances the effective deadline further.
        second, status = self._renew(renewal_id="R2")
        self.assertEqual(status, 201)
        # Replaying R1 rebuilds its frozen response, not the current one.
        replay, status = self._renew(renewal_id="R1")
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)
        self.assertNotEqual(replay["expires"], second["expires"])
        self.assertEqual(len(self._lease().renewals), 2)

    def test_renewal_id_scoped_to_one_lease(self) -> None:
        for number in (1, 2):
            self._commit(f"r{number}")
        self._claim(lease_id="L1", limit=1)
        first, status = self._renew(lease_id="L1", renewal_id="DUP")
        self.assertEqual(status, 201)
        # A second lease for another consumer may reuse the same id.
        claim2, status = self._claim(consumer="c2", lease_id="M1",
                                     expected=0, limit=1)
        self.assertEqual(status, 201)
        other, status = self._renew(consumer="c2", lease_id="M1",
                                    renewal_id="DUP")
        self.assertEqual(status, 201)
        self.assertEqual(other["renewal_id"], "DUP")
        self.assertEqual(
            _parse(other["expires"]),
            _parse(claim2["expires"]) + timedelta(seconds=30))
        self.assertNotEqual(first["expires"], other["expires"])

    def test_unknown_lease_404_lease_id(self) -> None:
        with self.assertRaises(ServiceError) as caught:
            self._renew(lease_id="nope")
        self.assertEqual(caught.exception.status_code, 404)
        self.assertEqual(caught.exception.field, "lease_id")
        self.assertEqual(list(caught.exception.to_body()),
                         ["message", "field"])

    def test_consumer_mismatch_409_consumer_id(self) -> None:
        self._commit("r1")
        self._claim(limit=1)
        with self.assertRaises(ServiceError) as caught:
            self._renew(consumer="other")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "consumer_id")
        self.assertEqual(self._lease().renewals, [])

    def test_replay_precedence_after_terminal(self) -> None:
        self._commit("r1")
        self._claim(limit=1)
        first, status = self._renew(renewal_id="R1")
        self.assertEqual(status, 201)
        # Confirming the lease ends it; the R1 renewal nevertheless
        # keeps replaying its frozen first response.
        self.assertEqual(
            self.service.event_gc_batch_lease_op({
                "consumer_id": "c1", "lease_id": "L1", "expected": 0,
                "op": "confirm"})[1], 201)
        replay, status = self._renew(renewal_id="R1")
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)

    def test_replay_precedence_after_expiry(self) -> None:
        import e2ee_backend.storage as storage_mod

        self._commit("r1")
        # Claim and renew against a fixed clock ten minutes in the past
        # so both frozen deadlines are already expired at the real
        # current time, while the durable records keep their original
        # frozen values.
        real_datetime = storage_mod.datetime
        fixed_now = real_datetime.now(timezone.utc) - timedelta(minutes=10)

        class FixedDateTime(real_datetime):
            @classmethod
            def now(cls, tz=None):
                return fixed_now if tz is not None else fixed_now.replace(
                    tzinfo=None)

        storage_mod.datetime = FixedDateTime
        try:
            self._claim(limit=1)
            first, status = self._renew(renewal_id="R1")
            self.assertEqual(status, 201)
        finally:
            storage_mod.datetime = real_datetime
        # Real wall time: the lease is expired. The committed renewal
        # nevertheless keeps replaying its frozen first response...
        replay, status = self._renew(renewal_id="R1")
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)
        # ...while a genuinely new renewal is rejected as expired.
        with self.assertRaises(ServiceError) as caught:
            self._renew(renewal_id="R2")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_terminal_lease_first_renew_409(self) -> None:
        self._commit("r1")
        self._claim(limit=1)
        self.assertEqual(
            self.service.event_gc_batch_lease_op({
                "consumer_id": "c1", "lease_id": "L1", "expected": 0,
                "op": "release"})[1], 201)
        with self.assertRaises(ServiceError) as caught:
            self._renew(renewal_id="R1")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")
        self.assertEqual(self._lease().renewals, [])

    def test_expired_lease_first_renew_409(self) -> None:
        self._commit("r1")
        self._claim(limit=1)
        self._set_claim_expires("L1", PAST)
        with self.assertRaises(ServiceError) as caught:
            self._renew(renewal_id="R1")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_checkpoint_off_start_409(self) -> None:
        self._commit("r1")
        self._commit("r2")
        self._claim(limit=2)
        # Advance the checkpoint inside the claimed page but keep it
        # below next_after.
        _, status = self.service.event_gc_batch_checkpoint({
            "consumer_id": "c1", "expected": 0, "after": 1})
        self.assertEqual(status, 201)
        with self.assertRaises(ServiceError) as caught:
            self._renew(renewal_id="R1")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_acknowledged_lease_first_renew_409(self) -> None:
        self._commit("r1")
        self._claim(limit=1)
        self.service.event_gc_batch_consume({
            "consumer_id": "c1", "expected": 0, "limit": 100})
        with self.assertRaises(ServiceError) as caught:
            self._renew(renewal_id="R1")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_renewal_keeps_lease_blocking_after_claim_deadline(self) -> None:
        self._commit("r1")
        self._claim(limit=1)
        self.assertEqual(self._renew(renewal_id="R1")[1], 201)
        # Even with the frozen claim deadline in the past, the renewed
        # (effective) deadline still blocks another claim.
        self._set_claim_expires("L1", PAST)
        with self.assertRaises(ServiceError) as caught:
            self._claim(lease_id="L2", limit=1)
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "consumer_id")
        # Once the effective deadline passes too, a new claim succeeds.
        self._set_effective_expires("L1", PAST)
        self.assertEqual(self._claim(lease_id="L2", limit=1)[1], 201)

    def test_claim_replay_returns_frozen_initial_expires(self) -> None:
        self._commit("r1")
        first, status = self._claim(limit=1)
        self.assertEqual(status, 201)
        self.assertEqual(self._renew(renewal_id="R1")[1], 201)
        self.assertEqual(self._renew(renewal_id="R2")[1], 201)
        replay, status = self._claim(limit=1)
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)
        self.assertEqual(replay["expires"], first["expires"])

    def test_renewal_extends_window_for_lease_op(self) -> None:
        self._commit("r1")
        self._claim(limit=1)
        self.assertEqual(self._renew(renewal_id="R1")[1], 201)
        # The claim deadline passes but the renewal keeps the lease
        # resolvable; an expired-only lease would be 409.
        self._set_claim_expires("L1", PAST)
        body, status = self.service.event_gc_batch_lease_op({
            "consumer_id": "c1", "lease_id": "L1", "expected": 0,
            "op": "confirm"})
        self.assertEqual(status, 201)
        self.assertEqual(body["after"], 1)


class RenewValidationTest(RenewMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def _expect_400(self, payload, field):
        with self.assertRaises(ServiceError) as caught:
            self.service.event_gc_batch_lease_renew(payload)
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
            {"lease_id": "L", "renewal_id": "R"}, "consumer_id")
        self._expect_400(
            {"consumer_id": "c", "renewal_id": "R"}, "lease_id")
        self._expect_400(
            {"consumer_id": "c", "lease_id": "L"}, "renewal_id")

    def test_fields_must_be_nonempty_strings(self) -> None:
        base = {"consumer_id": "c", "lease_id": "L", "renewal_id": "R"}
        for name in ("consumer_id", "lease_id", "renewal_id"):
            for bad in ("", None, 1, True, [], {}):
                with self.subTest(name=name, bad=bad):
                    self._expect_400(dict(base, **{name: bad}), name)

    def test_extra_field_is_400_with_that_key(self) -> None:
        self._expect_400({"consumer_id": "c", "lease_id": "L",
                          "renewal_id": "R", "extra": 1}, "extra")


class RenewHTTPTest(RenewMixin, unittest.TestCase):
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

    def _renew_http(self, consumer="c1", lease_id="L1", renewal_id="R1",
                    path=PATH):
        return self._request(path, json.dumps({
            "consumer_id": consumer, "lease_id": lease_id,
            "renewal_id": renewal_id}))

    def test_renew_over_http_with_key_order(self) -> None:
        self._commit("r1")
        self._claim(limit=1)
        status, body, raw = self._renew_http()
        self.assertEqual(status, 201)
        self.assertEqual(list(body),
                         ["consumer_id", "lease_id", "renewal_id",
                          "expires"])
        for earlier, later in (
                ('"consumer_id"', '"lease_id"'),
                ('"lease_id"', '"renewal_id"'),
                ('"renewal_id"', '"expires"')):
            self.assertLess(raw.index(earlier), raw.index(later))

    def test_query_rejected(self) -> None:
        for query in ("?foo=1", "?x", "?renewal_id=R"):
            with self.subTest(query=query):
                status, body, _ = self._renew_http(path=PATH + query)
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], "query")
        # A trailing empty ? carries no parameter and routes normally
        # (here to the unknown-lease 404, not a 400/query).
        status, body, _ = self._renew_http(path=PATH + "?")
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "lease_id")

    def test_bad_json_and_non_object_are_400_request_body(self) -> None:
        for raw in ("{", "[1]", "null", '"s"'):
            with self.subTest(raw=raw):
                status, body, _ = self._request(raw=raw)
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], "request_body")
                self.assertEqual(list(body), ["message", "field"])

    def test_field_errors_over_http(self) -> None:
        base = {"consumer_id": "c1", "lease_id": "L1",
                "renewal_id": "R1"}
        for payload, field in (
                ({"lease_id": "L1", "renewal_id": "R1"}, "consumer_id"),
                ({"consumer_id": "c1", "renewal_id": "R1"}, "lease_id"),
                ({"consumer_id": "c1", "lease_id": "L1"}, "renewal_id"),
                (dict(base, consumer_id=""), "consumer_id"),
                (dict(base, lease_id=3), "lease_id"),
                (dict(base, renewal_id=None), "renewal_id"),
                (dict(base, extra=1), "extra")):
            with self.subTest(payload=payload):
                status, body, _ = self._request(raw=json.dumps(payload))
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], field)
                self.assertEqual(list(body), ["message", "field"])

    def test_conflicts_over_http(self) -> None:
        self._commit("r1")
        self._claim(limit=1)
        status, body, _ = self._renew_http(lease_id="nope")
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "lease_id")
        status, body, _ = self._renew_http(consumer="other")
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "consumer_id")
        self.assertEqual(self._renew_http()[0], 201)
        # Replay is 200 and byte-identical.
        first_status = 201
        status_a, body_a, _ = self._renew_http()
        self.assertEqual(status_a, 200)
        status_b, body_b, _ = self._renew_http()
        self.assertEqual(status_b, 200)
        self.assertEqual(body_a, body_b)
        self.assertEqual(first_status, 201)

    def test_get_is_404(self) -> None:
        status, _, _ = self._request(method="GET")
        self.assertEqual(status, 404)


class RenewPersistenceTest(RenewMixin, unittest.TestCase):
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

    def test_first_renew_consumes_one_generation_replay_none(self) -> None:
        self._commit("r1")
        self._claim(limit=1)
        generation = self.state_store.commit_seq
        self.assertEqual(self._renew(renewal_id="R1")[1], 201)
        self.assertEqual(self.state_store.commit_seq, generation + 1)
        self.assertEqual(self._renew(renewal_id="R1")[1], 200)
        self.assertEqual(self.state_store.commit_seq, generation + 1)
        # A distinct renewal consumes one more generation.
        self.assertEqual(self._renew(renewal_id="R2")[1], 201)
        self.assertEqual(self.state_store.commit_seq, generation + 2)

    def test_replay_does_not_write(self) -> None:
        self._commit("r1")
        self._claim(limit=1)
        self._renew(renewal_id="R1")
        before = self._document()
        _, status = self._renew(renewal_id="R1")
        self.assertEqual(status, 200)
        self.assertEqual(self._document(), before)

    def test_record_shape_is_eight_keys(self) -> None:
        self._commit("r1")
        self._claim(limit=1)
        self.assertEqual(self._renew(renewal_id="R1")[1], 201)
        section = self._document()["cleanup_leases"]
        self.assertEqual(len(section), 1)
        self.assertEqual(list(section[0]),
                         ["lease_id", "consumer_id", "expected",
                          "next_after", "limit", "expires", "terminal",
                          "renewals"])
        renewals = section[0]["renewals"]
        self.assertEqual(len(renewals), 1)
        self.assertEqual(list(renewals[0]), ["renewal_id", "expires"])
        self.assertEqual(renewals[0]["renewal_id"], "R1")
        self.assertEqual(
            _parse(renewals[0]["expires"]),
            _parse(section[0]["expires"]) + timedelta(seconds=30))

    def test_unrenewed_claim_writes_empty_renewals(self) -> None:
        self._commit("r1")
        self._claim(limit=1)
        section = self._document()["cleanup_leases"]
        self.assertEqual(list(section[0])[-1], "renewals")
        self.assertEqual(section[0]["renewals"], [])

    def test_restart_preserves_chain_and_replays(self) -> None:
        self._commit("r1")
        claim, _ = self._claim(limit=1)
        first, status = self._renew(renewal_id="R1")
        self.assertEqual(status, 201)
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        lease = restarted.store._cleanup_leases["L1"]
        self.assertEqual(len(lease.renewals), 1)
        self.assertEqual(lease.renewals[0].renewal_id, "R1")
        self.assertEqual(lease.renewals[0].expires, first["expires"])
        replay, status = restarted.event_gc_batch_lease_renew({
            "consumer_id": "c1", "lease_id": "L1",
            "renewal_id": "R1"})
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)
        # The claim replay stays frozen at the original deadline.
        claim_replay, status = restarted.event_gc_batch_claim({
            "consumer_id": "c1", "lease_id": "L1", "expected": 0,
            "limit": 1})
        self.assertEqual(status, 200)
        self.assertEqual(claim_replay["expires"], claim["expires"])
        self.assertTrue(
            restarted.persistence_integrity()["consistent"])

    def test_save_failure_rolls_renewal_back(self) -> None:
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
                self._renew(renewal_id="R1")
        finally:
            persistence_mod.os.fsync = real_fsync
        self.assertEqual(self.state_store.commit_seq, generation)
        self.assertEqual(self._lease().renewals, [])
        # The same renewal can be retried and now commits.
        body, status = self._renew(renewal_id="R1")
        self.assertEqual(status, 201)
        self.assertEqual(body["renewal_id"], "R1")
        self.assertEqual(self.state_store.commit_seq, generation + 1)


class RestoreValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.service = DeviceService()
        self.service.store.add_device(Device("u", "bob", "ik"))
        self.path = os.path.join(self.directory, "state.json")
        attach_persistence(self.service, self.path)
        self.service.event_gc_cleanup_expired_batch({
            "mode": "commit", "device_ids": ["bob"], "after": 0,
            "limit": 100, "request_id": "r1"})
        self.service.event_gc_batch_claim({"consumer_id": "c1",
                                          "lease_id": "L1",
                                          "expected": 0, "limit": 1})
        self.service.event_gc_batch_lease_renew({
            "consumer_id": "c1", "lease_id": "L1",
            "renewal_id": "R1"})

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def _bad_file(self, mutate):
        with open(self.path, encoding="utf-8") as handle:
            document = json.load(handle)
        mutate(document)
        document.pop("integrity_log_version", None)
        bad_path = os.path.join(
            self.directory, f"bad-{len(os.listdir(self.directory))}.json")
        with open(bad_path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        return bad_path

    def _assert_refuses(self, bad_path):
        with open(bad_path, "rb") as handle:
            original = handle.read()
        service = DeviceService()
        with self.assertRaises(StateFileError):
            attach_persistence(service, bad_path)
        with open(bad_path, "rb") as handle:
            self.assertEqual(handle.read(), original)

    def test_legacy_seven_key_record_loads_with_empty_chain(self) -> None:
        path = self._bad_file(lambda document: [
            lease.pop("renewals")
            for lease in document["cleanup_leases"]])
        service = DeviceService()
        attach_persistence(service, path)
        self.assertEqual(service.store._cleanup_leases["L1"].renewals,
                         [])
        self.assertTrue(service.persistence_integrity()["consistent"])

    def test_legacy_six_key_record_loads(self) -> None:
        def mutate(document):
            for lease in document["cleanup_leases"]:
                lease.pop("renewals")
                lease.pop("terminal")
        path = self._bad_file(mutate)
        service = DeviceService()
        attach_persistence(service, path)
        lease = service.store._cleanup_leases["L1"]
        self.assertIsNone(lease.terminal)
        self.assertEqual(lease.renewals, [])

    def test_duplicate_renewal_id_refuses(self) -> None:
        def mutate(document):
            renewals = document["cleanup_leases"][0]["renewals"]
            renewals.append(dict(renewals[0]))
        self._assert_refuses(self._bad_file(mutate))

    def test_wrong_increment_refuses(self) -> None:
        def mutate(document):
            item = document["cleanup_leases"][0]["renewals"][0]
            item["expires"] = PAST
        self._assert_refuses(self._bad_file(mutate))

    def test_bad_timestamp_shape_refuses(self) -> None:
        def mutate(document):
            document["cleanup_leases"][0]["renewals"][0][
                "expires"] = "2099-01-01T00:00:00+00:00"
        self._assert_refuses(self._bad_file(mutate))

    def test_non_string_renewal_id_refuses(self) -> None:
        def mutate(document):
            document["cleanup_leases"][0]["renewals"][0][
                "renewal_id"] = 5
        self._assert_refuses(self._bad_file(mutate))

    def test_wrong_element_key_order_refuses(self) -> None:
        def mutate(document):
            old = document["cleanup_leases"][0]["renewals"][0]
            document["cleanup_leases"][0]["renewals"][0] = {
                "expires": old["expires"], "renewal_id": old["renewal_id"]}
        self._assert_refuses(self._bad_file(mutate))

    def test_extra_element_key_refuses(self) -> None:
        def mutate(document):
            document["cleanup_leases"][0]["renewals"][0]["extra"] = 1
        self._assert_refuses(self._bad_file(mutate))

    def test_renewals_not_a_list_refuses(self) -> None:
        def mutate(document):
            document["cleanup_leases"][0]["renewals"] = {}
        self._assert_refuses(self._bad_file(mutate))

    def test_more_than_ten_renewals_refuses(self) -> None:
        def mutate(document):
            lease = document["cleanup_leases"][0]
            # Replace the whole chain with eleven fabricated,
            # uniquely-id'd, correctly-incrementing entries; only the
            # count is wrong.
            chain_from = lease["expires"]
            generated = []
            for _ in range(11):
                chain_from = (_parse(chain_from) + timedelta(seconds=30)) \
                    .isoformat(timespec="microseconds")
                generated.append(chain_from)
            lease["renewals"] = [{
                "renewal_id": f"F{i}", "expires": stamp}
                for i, stamp in enumerate(generated)]
        self._assert_refuses(self._bad_file(mutate))


if __name__ == "__main__":
    unittest.main()
