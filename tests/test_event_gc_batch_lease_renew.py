"""Tests for the batch-cleanup audit lease renewal endpoint.

``POST /v1/event-gc-batch/lease/renew`` extends a page lease granted by
``POST /v1/event-gc-batch/claim`` for another 30 seconds. The call takes
no query parameters (any -> 400/query); the body carries exactly
``consumer_id``, ``lease_id`` and ``renewal_id``, all non-empty
strings; a bad/non-object body is 400/request_body and a
missing/wrongly typed/extra field is 400 with that field.

Resolution order under the one store lock: an unknown lease id is
404/lease_id and a lease owned by another consumer is
409/consumer_id; a replay of the same ``renewal_id`` on the same lease
(ids are only unique within one lease) always answers 200 with the
byte-identical first response, even after the lease expired or was
confirmed/released. A first renewal requires the lease to be not
terminal, not expired against its current effective deadline, the
consumer's checkpoint to equal the lease's ``expected`` while still
below ``next_after``, and fewer than ten renewals to exist — otherwise
409/lease_id. A first renewal answers 201 and extends the effective
deadline (claim value initially, the last renewal's afterwards) by
exactly 30 seconds; the eleventh renewal is 409. Success keys are
``consumer_id``, ``lease_id``, ``renewal_id`` and ``expires`` in that
order. The claim replay keeps answering with its original frozen
``expires`` while all liveness decisions use the effective deadline.

The lease's ``cleanup_leases`` record always serializes eight keys,
``renewals`` appended after ``terminal``; each renewal item carries
exactly ``renewal_id``/``expires``. Old six/seven-key records load with
an empty renewal list; duplicate ids, non-+30s chaining, more than ten
items, a bad type or a wrong key order refuse startup without
overwriting the file.
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

import e2ee_backend.storage as storage_mod
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
FUTURE = "2099-01-01T00:00:00.000000+00:00"


class RenewMixin:
    def _build(self) -> None:
        self.service = DeviceService()
        self.service.store.add_device(Device("u", "alice", "ik"))
        self.service.store.add_device(Device("u", "bob", "ik"))

    def _commit(self, request_id, device_ids=("bob",)):
        return self.service.event_gc_cleanup_expired_batch({
            "mode": "commit", "device_ids": list(device_ids),
            "after": 0, "limit": 100, "request_id": request_id})

    def _claim(self, consumer="c1", lease_id="L1", expected=0,
               limit=100):
        return self.service.event_gc_batch_claim({
            "consumer_id": consumer, "lease_id": lease_id,
            "expected": expected, "limit": limit})

    def _renew(self, consumer="c1", lease_id="L1", renewal_id="n1"):
        return self.service.event_gc_batch_lease_renew({
            "consumer_id": consumer, "lease_id": lease_id,
            "renewal_id": renewal_id})

    def _lease_op(self, consumer="c1", lease_id="L1", expected=0,
                  op="confirm"):
        return self.service.event_gc_batch_lease_op({
            "consumer_id": consumer, "lease_id": lease_id,
            "expected": expected, "op": op})

    def _checkpoint_advance(self, after, consumer="c1", expected=0):
        return self.service.event_gc_batch_checkpoint({
            "consumer_id": consumer, "expected": expected,
            "after": after})

    def _checkpoint(self, consumer="c1"):
        body, _ = self.service.event_gc_batch_checkpoint(
            {"consumer_id": consumer, "expected": None, "after": None})
        return body

    def _set_expires(self, lease_id, value, renewal_index=None):
        lease = self.service.store._cleanup_leases[lease_id]
        if renewal_index is None:
            lease.expires = value
        else:
            lease.renewals[renewal_index].expires = value

    @contextmanager
    def _advance_clock(self, seconds):
        # Shift the storage layer's notion of now without rewriting any
        # frozen deadline, so replay answers keep their first values
        # while liveness decisions observe the passage of time.
        real_datetime = storage_mod.datetime
        shift = timedelta(seconds=seconds)

        class ShiftedDatetime(real_datetime):
            @classmethod
            def now(cls, tz=None):
                return real_datetime.now(tz) + shift

        storage_mod.datetime = ShiftedDatetime
        try:
            yield
        finally:
            storage_mod.datetime = real_datetime


class RenewServiceTest(RenewMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def test_first_renew_201_extends_by_exactly_30_seconds(self) -> None:
        self._commit("r1")
        claim, claim_status = self._claim(limit=1)
        self.assertEqual(claim_status, 201)
        body, status = self._renew()
        self.assertEqual(status, 201)
        self.assertEqual(list(body),
                         ["consumer_id", "lease_id", "renewal_id",
                          "expires"])
        self.assertEqual(body["consumer_id"], "c1")
        self.assertEqual(body["lease_id"], "L1")
        self.assertEqual(body["renewal_id"], "n1")
        claimed = datetime.fromisoformat(claim["expires"])
        renewed = datetime.fromisoformat(body["expires"])
        self.assertEqual(renewed - claimed, timedelta(seconds=30))
        self.assertTrue(body["expires"].endswith("+00:00"))
        self.assertEqual(
            len(body["expires"].split("+")[0].split(".")[1]), 6)
        lease = self.service.store._cleanup_leases["L1"]
        self.assertEqual(len(lease.renewals), 1)
        self.assertEqual(lease.renewals[0].renewal_id, "n1")
        self.assertEqual(lease.renewals[0].expires, body["expires"])
        # Renewal never moves the checkpoint.
        self.assertEqual(self._checkpoint()["after"], 0)

    def test_each_renewal_chains_off_the_last_one(self) -> None:
        self._commit("r1")
        claim, _ = self._claim(limit=1)
        previous = claim["expires"]
        for index in range(1, 11):
            body, status = self._renew(renewal_id=f"n{index}")
            self.assertEqual(status, 201)
            self.assertEqual(
                datetime.fromisoformat(body["expires"])
                - datetime.fromisoformat(previous),
                timedelta(seconds=30))
            previous = body["expires"]
        self.assertEqual(
            len(self.service.store._cleanup_leases["L1"].renewals), 10)

    def test_eleventh_renewal_is_409(self) -> None:
        self._commit("r1")
        self._claim(limit=1)
        for index in range(1, 11):
            self.assertEqual(self._renew(renewal_id=f"n{index}")[1],
                             201)
        with self.assertRaises(ServiceError) as caught:
            self._renew(renewal_id="n11")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")
        # The rejected renewal leaves no record behind.
        self.assertEqual(
            len(self.service.store._cleanup_leases["L1"].renewals), 10)

    def test_same_renewal_id_replays_200_byte_identical(self) -> None:
        self._commit("r1")
        self._claim(limit=1)
        first, status = self._renew(renewal_id="n1")
        self.assertEqual(status, 201)
        replay, status = self._renew(renewal_id="n1")
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)

    def test_replay_after_expiry_still_200(self) -> None:
        self._commit("r1")
        self._claim(limit=1)
        first, _ = self._renew(renewal_id="n1")
        # Let the renewed effective deadline (claim + 60s) pass without
        # rewriting any frozen value; the replay answers from the frozen
        # renewal anyway.
        with self._advance_clock(61):
            replay, status = self._renew(renewal_id="n1")
            self.assertEqual(status, 200)
            self.assertEqual(replay, first)
            # A new renewal on the expired lease conflicts.
            with self.assertRaises(ServiceError) as caught:
                self._renew(renewal_id="n2")
            self.assertEqual(caught.exception.status_code, 409)
            self.assertEqual(caught.exception.field, "lease_id")
        # The stored renewal is untouched by the read-only replay.
        self.assertEqual(
            self.service.store._cleanup_leases["L1"].renewals[0].expires,
            first["expires"])

    def test_replay_after_terminal_still_200_new_id_conflicts(
            self) -> None:
        self._commit("r1")
        self._claim(limit=1)
        first, _ = self._renew(renewal_id="n1")
        self.assertEqual(self._lease_op(op="confirm")[1], 201)
        # The replay of n1 is unaffected by the terminal marker.
        replay, status = self._renew(renewal_id="n1")
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)
        # A fresh renewal on the confirmed lease conflicts.
        with self.assertRaises(ServiceError) as caught:
            self._renew(renewal_id="n2")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_replay_after_release_still_200(self) -> None:
        self._commit("r1")
        self._claim(limit=1)
        first, _ = self._renew(renewal_id="n1")
        self.assertEqual(self._lease_op(op="release")[1], 201)
        replay, status = self._renew(renewal_id="n1")
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)
        with self.assertRaises(ServiceError) as caught:
            self._renew(renewal_id="n2")
        self.assertEqual(caught.exception.field, "lease_id")

    def test_renewal_id_reusable_across_leases(self) -> None:
        self._commit("r1")
        self._commit("r2")
        # c1 claims page [0,1), confirms it, then claims [1,2): two
        # distinct leases may both use the same renewal id.
        self._claim(lease_id="L1", limit=1)
        body1, status1 = self._renew(lease_id="L1", renewal_id="same")
        self.assertEqual(status1, 201)
        self._lease_op(lease_id="L1", op="confirm")
        self._claim(lease_id="L2", expected=1, limit=1)
        body2, status2 = self._renew(lease_id="L2", renewal_id="same")
        self.assertEqual(status2, 201)
        self.assertNotEqual(body1["expires"], body2["expires"])
        # Each lease keeps its own renewal.
        self.assertEqual(
            [r.renewal_id
             for r in self.service.store._cleanup_leases["L1"].renewals],
            ["same"])
        self.assertEqual(
            [r.renewal_id
             for r in self.service.store._cleanup_leases["L2"].renewals],
            ["same"])

    def test_unknown_lease_404(self) -> None:
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
            self._renew(consumer="other", renewal_id="x")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "consumer_id")
        # The rejected renewal writes nothing.
        self.assertEqual(
            self.service.store._cleanup_leases["L1"].renewals, [])

    def test_checkpoint_moved_off_expected_is_409(self) -> None:
        self._commit("r1")
        self._commit("r2")
        self._claim(limit=2)
        self._checkpoint_advance(after=1, expected=0)
        with self.assertRaises(ServiceError) as caught:
            self._renew(renewal_id="n1")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")
        # Reaching next_after conflicts as well.
        self._checkpoint_advance(after=2, expected=1)
        with self.assertRaises(ServiceError) as caught:
            self._renew(renewal_id="n1")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_expired_claim_deadline_409_without_renewals(self) -> None:
        self._commit("r1")
        self._claim(limit=1)
        self._set_expires("L1", PAST)
        with self.assertRaises(ServiceError) as caught:
            self._renew(renewal_id="n1")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_claim_replay_keeps_original_expires_liveness_uses_effective(
            self) -> None:
        self._commit("r1")
        claim, _ = self._claim(limit=1)
        first, _ = self._renew(renewal_id="n1")
        # Advance past the frozen claim deadline (claim+30s) but not
        # past the renewed effective deadline (claim+60s).
        with self._advance_clock(31):
            replay, status = self._claim(limit=1)
            self.assertEqual(status, 200)
            self.assertEqual(replay["expires"], claim["expires"])
            # The renewed lease still blocks a second claim by this
            # consumer: liveness uses the effective (last renewal)
            # deadline.
            with self.assertRaises(ServiceError) as caught:
                self._claim(lease_id="L2", limit=1)
            self.assertEqual(caught.exception.status_code, 409)
            self.assertEqual(caught.exception.field, "consumer_id")
        # The frozen claim value and the renewal chain are untouched.
        lease = self.service.store._cleanup_leases["L1"]
        self.assertEqual(lease.expires, claim["expires"])
        self.assertEqual(lease.renewals[0].expires, first["expires"])


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
            {"lease_id": "L", "renewal_id": "n"}, "consumer_id")
        self._expect_400(
            {"consumer_id": "c", "renewal_id": "n"}, "lease_id")
        self._expect_400(
            {"consumer_id": "c", "lease_id": "L"}, "renewal_id")

    def test_field_shapes(self) -> None:
        base = {"consumer_id": "c", "lease_id": "L",
                "renewal_id": "n"}
        for bad in ("", None, 1, True, []):
            with self.subTest(bad=("consumer", bad)):
                self._expect_400(dict(base, consumer_id=bad),
                                 "consumer_id")
        for bad in ("", None, 1, True, []):
            with self.subTest(bad=("lease", bad)):
                self._expect_400(dict(base, lease_id=bad), "lease_id")
        for bad in ("", None, 1, True, []):
            with self.subTest(bad=("renewal", bad)):
                self._expect_400(dict(base, renewal_id=bad),
                                 "renewal_id")

    def test_extra_field_is_400_with_that_key(self) -> None:
        self._expect_400({"consumer_id": "c", "lease_id": "L",
                          "renewal_id": "n", "extra": 1}, "extra")

    def test_missing_field_before_lease_lookup(self) -> None:
        # A malformed body is 400 even though the lease does not exist.
        self._expect_400({"consumer_id": "c", "lease_id": "ghost"},
                         "renewal_id")


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

    def _renew_http(self, consumer="c1", lease_id="L1",
                    renewal_id="n1", path=PATH):
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
        self.assertTrue(body["expires"].endswith("+00:00"))

    def test_query_rejected(self) -> None:
        for query in ("?foo=1", "?foo", "?x="):
            with self.subTest(query=query):
                status, body, _ = self._renew_http(path=PATH + query)
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], "query")

    def test_bad_json_and_non_object_are_400_request_body(self) -> None:
        for raw in ("{", "[1]", "null", '"s"'):
            with self.subTest(raw=raw):
                status, body, _ = self._request(raw=raw)
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], "request_body")
                self.assertEqual(list(body), ["message", "field"])

    def test_field_errors_over_http(self) -> None:
        for payload, field in (
                ({"lease_id": "L", "renewal_id": "n"}, "consumer_id"),
                ({"consumer_id": "c", "renewal_id": "n"}, "lease_id"),
                ({"consumer_id": "c", "lease_id": "L"}, "renewal_id"),
                ({"consumer_id": "", "lease_id": "L",
                  "renewal_id": "n"}, "consumer_id"),
                ({"consumer_id": 1, "lease_id": "L",
                  "renewal_id": "n"}, "consumer_id"),
                ({"consumer_id": "c", "lease_id": "L",
                  "renewal_id": None}, "renewal_id"),
                ({"consumer_id": "c", "lease_id": "L",
                  "renewal_id": "n", "extra": 1}, "extra")):
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
        status, body, _ = self._renew_http(renewal_id="n1")
        self.assertEqual(status, 200)
        # Let the renewed deadline pass: a new renewal conflicts.
        with self._advance_clock(61):
            status, body, _ = self._renew_http(renewal_id="n2")
            self.assertEqual(status, 409)
            self.assertEqual(body["field"], "lease_id")
            # The frozen replay still answers 200.
            status, _body, _ = self._renew_http(renewal_id="n1")
            self.assertEqual(status, 200)

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
        self.assertEqual(self._renew()[1], 201)
        self.assertEqual(self.state_store.commit_seq, generation + 1)
        self.assertEqual(self._renew()[1], 200)
        self.assertEqual(self.state_store.commit_seq, generation + 1)

    def test_writes_eight_keys_and_renewal_item_shape(self) -> None:
        self._commit("r1")
        self._claim(limit=1)
        self.assertEqual(self._renew()[1], 201)
        section = self._document()["cleanup_leases"]
        self.assertEqual(list(section[0]),
                         ["lease_id", "consumer_id", "expected",
                          "next_after", "limit", "expires", "terminal",
                          "renewals"])
        self.assertEqual(list(section[0]["renewals"][0]),
                         ["renewal_id", "expires"])
        self.assertEqual(section[0]["renewals"][0]["renewal_id"], "n1")

    def test_fresh_claim_writes_empty_renewals_list(self) -> None:
        self._commit("r1")
        self._claim(limit=1)
        section = self._document()["cleanup_leases"]
        self.assertEqual(list(section[0])[-1], "renewals")
        self.assertEqual(section[0]["renewals"], [])

    def test_restart_preserves_renewals_and_replay(self) -> None:
        self._commit("r1")
        claim, _ = self._claim(limit=1)
        first, status = self._renew()
        self.assertEqual(status, 201)
        second, status = self._renew(renewal_id="n2")
        self.assertEqual(status, 201)
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        lease = restarted.store._cleanup_leases["L1"]
        self.assertEqual([r.renewal_id for r in lease.renewals],
                         ["n1", "n2"])
        replay, http_status = restarted.event_gc_batch_lease_renew({
            "consumer_id": "c1", "lease_id": "L1",
            "renewal_id": "n1"})
        self.assertEqual(http_status, 200)
        self.assertEqual(replay, first)
        # The claim replay keeps the original frozen deadline.
        claim_replay, claim_status = restarted.event_gc_batch_claim({
            "consumer_id": "c1", "lease_id": "L1", "expected": 0,
            "limit": 1})
        self.assertEqual(claim_status, 200)
        self.assertEqual(claim_replay["expires"], claim["expires"])
        # A next renewal chains off n2's deadline after restart.
        third, status = restarted.event_gc_batch_lease_renew({
            "consumer_id": "c1", "lease_id": "L1", "renewal_id": "n3"})
        self.assertEqual(status, 201)
        self.assertEqual(
            datetime.fromisoformat(third["expires"])
            - datetime.fromisoformat(second["expires"]),
            timedelta(seconds=30))
        self.assertTrue(
            restarted.persistence_integrity()["consistent"])

    def test_legacy_seven_key_record_loads_with_empty_renewals(
            self) -> None:
        self._commit("r1")
        self._claim(limit=1)
        document = self._document()
        for lease in document["cleanup_leases"]:
            lease.pop("renewals")
        document.pop("integrity_log_version", None)
        legacy_path = os.path.join(self.directory, "legacy7.json")
        with open(legacy_path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        restarted = DeviceService()
        attach_persistence(restarted, legacy_path)
        lease = restarted.store._cleanup_leases["L1"]
        self.assertEqual(lease.renewals, [])
        body, status = restarted.event_gc_batch_lease_renew({
            "consumer_id": "c1", "lease_id": "L1", "renewal_id": "n1"})
        self.assertEqual(status, 201)
        self.assertEqual(
            datetime.fromisoformat(body["expires"])
            - datetime.fromisoformat(lease.expires),
            timedelta(seconds=30))

    def test_legacy_six_key_record_loads_with_empty_renewals(
            self) -> None:
        self._commit("r1")
        self._claim(limit=1)
        document = self._document()
        for lease in document["cleanup_leases"]:
            lease.pop("terminal")
            lease.pop("renewals")
        document.pop("integrity_log_version", None)
        legacy_path = os.path.join(self.directory, "legacy6.json")
        with open(legacy_path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        restarted = DeviceService()
        attach_persistence(restarted, legacy_path)
        lease = restarted.store._cleanup_leases["L1"]
        self.assertIsNone(lease.terminal)
        self.assertEqual(lease.renewals, [])
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
                self._renew()
        finally:
            persistence_mod.os.fsync = real_fsync
        self.assertEqual(self.state_store.commit_seq, generation)
        lease = self.service.store._cleanup_leases["L1"]
        self.assertEqual(lease.renewals, [])
        # The renewal can be retried and now commits.
        body, status = self._renew()
        self.assertEqual(status, 201)
        self.assertEqual(body["renewal_id"], "n1")
        self.assertEqual(self.state_store.commit_seq, generation + 1)


class RenewConcurrencyTest(RenewMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()
        self._commit("r1")
        self._claim(limit=1)

    def test_concurrent_same_renewal_id_linearizes_to_one_201(
            self) -> None:
        results = []
        barrier = threading.Barrier(8)

        def worker() -> None:
            barrier.wait()
            try:
                _body, status = self._renew(renewal_id="n1")
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


class RenewRestoreValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self._seed_n = 0

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def _document_with_renewals(self, mutate, renewals=("n1",)):
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
        for renewal_id in renewals:
            status = service.event_gc_batch_lease_renew({
                "consumer_id": "c1", "lease_id": "L1",
                "renewal_id": renewal_id})[1]
            assert status == 201
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

    def test_duplicate_renewal_id_refuses(self) -> None:
        def mutate(document):
            renewals = document["cleanup_leases"][0]["renewals"]
            renewals.append(dict(renewals[0]))
        message = self._assert_refuses_startup(
            self._document_with_renewals(mutate))
        self.assertIn("renewal_id", message)

    def test_wrong_increment_refuses(self) -> None:
        def mutate(document):
            item = document["cleanup_leases"][0]["renewals"][0]
            item["expires"] = FUTURE
        message = self._assert_refuses_startup(
            self._document_with_renewals(mutate))
        self.assertIn("30 seconds", message)

    def test_more_than_ten_renewals_refuses(self) -> None:
        ids = [f"n{i}" for i in range(1, 11)]

        def mutate(document):
            renewals = document["cleanup_leases"][0]["renewals"]
            base = datetime.fromisoformat(renewals[-1]["expires"])
            renewals.append({
                "renewal_id": "n11",
                "expires": (base + timedelta(seconds=30)).isoformat(
                    timespec="microseconds"),
            })
        message = self._assert_refuses_startup(
            self._document_with_renewals(mutate, renewals=ids))
        self.assertIn("renewals", message)

    def test_renewal_item_wrong_key_order_refuses(self) -> None:
        def mutate(document):
            item = document["cleanup_leases"][0]["renewals"][0]
            document["cleanup_leases"][0]["renewals"][0] = {
                "expires": item["expires"],
                "renewal_id": item["renewal_id"]}
        message = self._assert_refuses_startup(
            self._document_with_renewals(mutate))
        self.assertIn("renewals[0]", message)

    def test_renewal_item_extra_key_refuses(self) -> None:
        def mutate(document):
            document["cleanup_leases"][0]["renewals"][0]["extra"] = 1
        message = self._assert_refuses_startup(
            self._document_with_renewals(mutate))
        self.assertIn("renewals[0]", message)

    def test_renewal_empty_id_refuses(self) -> None:
        def mutate(document):
            document["cleanup_leases"][0]["renewals"][0][
                "renewal_id"] = ""
        message = self._assert_refuses_startup(
            self._document_with_renewals(mutate))
        self.assertIn("renewal_id", message)

    def test_renewal_bad_timestamp_refuses(self) -> None:
        def mutate(document):
            document["cleanup_leases"][0]["renewals"][0][
                "expires"] = "2020-01-01T00:00:00+00:00"
        message = self._assert_refuses_startup(
            self._document_with_renewals(mutate))
        self.assertIn("expires", message)

    def test_renewals_not_a_list_refuses(self) -> None:
        def mutate(document):
            document["cleanup_leases"][0]["renewals"] = {}
        message = self._assert_refuses_startup(
            self._document_with_renewals(mutate))
        self.assertIn("renewals must be a list", message)

    def test_renewal_item_not_object_refuses(self) -> None:
        def mutate(document):
            document["cleanup_leases"][0]["renewals"][0] = "n1"
        message = self._assert_refuses_startup(
            self._document_with_renewals(mutate))
        self.assertIn("must be an object", message)

    def test_ten_chained_renewals_load_and_still_block(self) -> None:
        # Ten renewals with the final deadline in the future load
        # normally (the eleventh live request then conflicts; that is
        # covered at the service layer). Here the stored deadlines are
        # necessarily in the past because they chain from a claim made
        # seconds ago plus at most 300 seconds, so rewrite them to chain
        # from FUTURE to exercise the open-lease path too.
        ids = [f"n{i}" for i in range(1, 11)]
        path = self._document_with_renewals(lambda d: None,
                                            renewals=ids)
        with open(path, encoding="utf-8") as handle:
            document = json.load(handle)
        document.pop("integrity_log_version", None)
        lease = document["cleanup_leases"][0]
        base = datetime.fromisoformat(FUTURE)
        lease["expires"] = (base - timedelta(seconds=30)) \
            .isoformat(timespec="microseconds")
        for index, item in enumerate(lease["renewals"]):
            item["expires"] = (base + timedelta(
                seconds=30 * index)).isoformat(timespec="microseconds")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        service = DeviceService()
        attach_persistence(service, path)
        stored = service.store._cleanup_leases["L1"]
        self.assertEqual(len(stored.renewals), 10)
        with self.assertRaises(ServiceError) as caught:
            service.event_gc_batch_claim({"consumer_id": "c1",
                                          "lease_id": "L2",
                                          "expected": 0, "limit": 1})
        self.assertEqual(caught.exception.field, "consumer_id")
        self.assertTrue(service.persistence_integrity()["consistent"])


if __name__ == "__main__":
    unittest.main()
