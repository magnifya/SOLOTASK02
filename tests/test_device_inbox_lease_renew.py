"""Tests for the 1:1 inbox lease renewal endpoint.

POST /v1/devices/{device_id}/inbox/leases/{lease_id}/renew extends an
occupied, still-active inbox lease's effective deadline by exactly 30
seconds. The JSON body must be an object carrying a non-empty string
``renewal_id``: bad JSON / a non-object body is 400/request_body, a
missing, empty or wrongly typed field is 400/renewal_id. An unknown
lease is 404/lease_id, one owned by another device is 409/lease_id;
both precede the path device's state. Replaying the same renewal_id on
the same lease returns the first response byte-identically with 200
(even after expiry, release or device revocation), while the id is free
to reuse on another lease. A first renewal on a revoked device is
409/device_id; a released or expired lease is 409/lease_id. A first
renewal returns 201 with device_id/lease_id/renewal_id/leased_until and
persists one generation; the effective deadline is the claim value
initially and the last renewal's value afterwards.
"""
import json
import os
import shutil
import tempfile
import threading
import unittest
from datetime import datetime, timedelta, timezone
from http.client import HTTPConnection

from e2ee_backend.models import (
    Device, MessageLease, MessageLeaseRenewal, SignedPreKey)
from e2ee_backend.persistence import (
    PersistenceUnavailable, StateFileError, attach_persistence)
from e2ee_backend.service import DeviceService, ServiceError
from e2ee_backend.http_app import create_server

LEASE_SECONDS = 30


def _parse(stamp: str) -> datetime:
    return datetime.fromisoformat(stamp)


class RenewMixin:
    def _build(self) -> None:
        self.service = DeviceService()
        self.service.store.add_device(Device("u", "alice", "ik"))
        self.service.store.add_device(Device(
            "u", "bob", "ik",
            prekeys=[SignedPreKey("pk1", "pubk1"),
                     SignedPreKey("pk2", "pubk2")]))
        self.service.store.add_device(
            Device("u", "carol", "ik",
                   prekeys=[SignedPreKey("pkC", "pubkC")]))
        # Session alice -> bob with three messages; another carol -> bob.
        self.sid1 = self.service.store.create_session(
            "alice", "bob", "pk1", "ek1").session_id
        self.sid2 = self.service.store.create_session(
            "carol", "bob", "pk2", "ek2").session_id
        for sequence in range(1, 4):
            self.service.post_message({
                "session_id": self.sid1, "sender_device_id": "alice",
                "message_id": f"a{sequence}", "sequence": sequence,
                "nonce": f"na{sequence}", "ciphertext": "ct"})
        for sequence in range(1, 3):
            self.service.post_message({
                "session_id": self.sid2, "sender_device_id": "carol",
                "message_id": f"b{sequence}", "sequence": sequence,
                "nonce": f"nb{sequence}", "ciphertext": "ct"})

    def _claim(self, lease_id="L1", limit=2):
        body, status = self.service.inbox_claim(
            "bob", {"lease_id": lease_id, "limit": limit})
        self.assertEqual(status, 201)
        return body

    def _renew(self, device_id, lease_id, renewal_id):
        return self.service.inbox_lease_renew(
            device_id, lease_id, {"renewal_id": renewal_id})

    def _expire_lease(self, lease_id, renewals=False,
                      seconds_ago=5) -> None:
        """Rewrite the lease's stored deadlines to the past in memory."""
        # With a renewal, the renewal deadline is claim+30s, so the claim
        # has to be more than 30 seconds back for it to also be in the past.
        if renewals:
            seconds_ago = max(seconds_ago, LEASE_SECONDS + 5)
        claim = datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)
        claim_s = claim.isoformat(timespec="microseconds")
        renewal_s = (claim + timedelta(seconds=LEASE_SECONDS)) \
            .isoformat(timespec="microseconds")
        for state in self.service.store._delivery.values():
            for lease in state.leases:
                if lease.lease_id != lease_id:
                    continue
                lease.leased_until = claim_s
                if renewals:
                    lease.renewals = [
                        MessageLeaseRenewal("R1", renewal_s)]


class RenewServiceTest(RenewMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def test_first_renew_201_extends_claim_deadline_by_30s(self) -> None:
        claim = self._claim("L1", 2)
        body, status = self._renew("bob", "L1", "R1")
        self.assertEqual(status, 201)
        self.assertEqual(list(body),
                         ["device_id", "lease_id", "renewal_id",
                          "leased_until"])
        self.assertEqual(body["device_id"], "bob")
        self.assertEqual(body["lease_id"], "L1")
        self.assertEqual(body["renewal_id"], "R1")
        self.assertRegex(body["leased_until"], r"\.\d{6}\+00:00$")
        self.assertEqual(
            _parse(body["leased_until"]) - _parse(claim["leased_until"]),
            timedelta(seconds=LEASE_SECONDS))

    def test_second_renew_chains_from_last_renewal(self) -> None:
        claim = self._claim("L1", 2)
        first, _ = self._renew("bob", "L1", "R1")
        second, status = self._renew("bob", "L1", "R2")
        self.assertEqual(status, 201)
        self.assertEqual(
            _parse(second["leased_until"]) - _parse(first["leased_until"]),
            timedelta(seconds=LEASE_SECONDS))
        self.assertEqual(
            _parse(second["leased_until"]) - _parse(claim["leased_until"]),
            timedelta(seconds=2 * LEASE_SECONDS))

    def test_renew_extends_every_copy_of_the_lease(self) -> None:
        self._claim("L1", 2)
        self._renew("bob", "L1", "R1")
        copies = [lease for state in self.service.store._delivery.values()
                  for lease in state.leases if lease.lease_id == "L1"]
        self.assertEqual(len(copies), 2)
        deadlines = {lease.renewals[-1].leased_until for lease in copies}
        self.assertEqual(len(deadlines), 1)
        for lease in copies:
            self.assertEqual([(r.renewal_id, r.leased_until)
                              for r in lease.renewals],
                             [(r.renewal_id, r.leased_until)
                              for r in copies[0].renewals])

    def test_replay_same_renewal_id_returns_first_response_200(self) -> None:
        self._claim("L1", 2)
        first, first_status = self._renew("bob", "L1", "R1")
        self.assertEqual(first_status, 201)
        replay, status = self._renew("bob", "L1", "R1")
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)

    def test_replay_wins_after_expiry(self) -> None:
        # Fabricate a claim whose renewal deadline is already in the past;
        # the replay still answers with that frozen committed deadline.
        self._claim("L1", 2)
        claim_at = datetime.now(timezone.utc) - timedelta(
            seconds=LEASE_SECONDS + 5)
        renewal_at = claim_at + timedelta(seconds=LEASE_SECONDS)
        renewal_s = renewal_at.isoformat(timespec="microseconds")
        for state in self.service.store._delivery.values():
            for lease in state.leases:
                if lease.lease_id == "L1":
                    lease.leased_until = claim_at.isoformat(
                        timespec="microseconds")
                    lease.renewals = [MessageLeaseRenewal("R1", renewal_s)]
        replay, status = self._renew("bob", "L1", "R1")
        self.assertEqual(status, 200)
        self.assertEqual(replay, {
            "device_id": "bob", "lease_id": "L1", "renewal_id": "R1",
            "leased_until": renewal_s})

    def test_replay_wins_after_release(self) -> None:
        self._claim("L1", 2)
        first, _ = self._renew("bob", "L1", "R1")
        self.service.inbox_release("bob", "L1")
        replay, status = self._renew("bob", "L1", "R1")
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)

    def test_replay_wins_after_device_revoked(self) -> None:
        self._claim("L1", 2)
        first, _ = self._renew("bob", "L1", "R1")
        self.service.revoke_device("bob")
        replay, status = self._renew("bob", "L1", "R1")
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)

    def test_renewal_id_is_reusable_across_leases(self) -> None:
        self._claim("L1", 2)
        first, _ = self._renew("bob", "L1", "SHARED")
        self.service.inbox_release("bob", "L1")
        second_claim = self._claim("L2", 10)
        again, status = self._renew("bob", "L2", "SHARED")
        self.assertEqual(status, 201)
        self.assertEqual(again["renewal_id"], "SHARED")
        self.assertEqual(
            _parse(again["leased_until"])
            - _parse(second_claim["leased_until"]),
            timedelta(seconds=LEASE_SECONDS))
        self.assertNotEqual(again["leased_until"], first["leased_until"])

    def test_unknown_lease_is_404_lease_id(self) -> None:
        with self.assertRaises(ServiceError) as caught:
            self._renew("bob", "NOPE", "R1")
        self.assertEqual(caught.exception.status_code, 404)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_unknown_lease_404_precedes_unknown_device(self) -> None:
        with self.assertRaises(ServiceError) as caught:
            self._renew("ghost", "NOPE", "R1")
        self.assertEqual(caught.exception.status_code, 404)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_cross_device_renew_is_409_lease_id(self) -> None:
        self._claim("L1", 2)
        with self.assertRaises(ServiceError) as caught:
            self._renew("carol", "L1", "R1")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_cross_device_conflict_precedes_path_device_state(self) -> None:
        self._claim("L1", 2)
        self.service.revoke_device("carol")
        with self.assertRaises(ServiceError) as caught:
            self._renew("carol", "L1", "R1")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_first_renew_on_revoked_device_is_409_device_id(self) -> None:
        self._claim("L1", 2)
        self.service.revoke_device("bob")
        with self.assertRaises(ServiceError) as caught:
            self._renew("bob", "L1", "R1")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "device_id")

    def test_renew_released_lease_is_409_lease_id(self) -> None:
        self._claim("L1", 2)
        self.service.inbox_release("bob", "L1")
        with self.assertRaises(ServiceError) as caught:
            self._renew("bob", "L1", "R1")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_renew_expired_lease_is_409_lease_id(self) -> None:
        self._claim("L1", 2)
        self._expire_lease("L1")
        with self.assertRaises(ServiceError) as caught:
            self._renew("bob", "L1", "R1")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_bad_body_shapes_are_400(self) -> None:
        self._claim("L1", 2)
        for payload in (None, "x", 7, ["R1"], True):
            with self.subTest(payload=payload):
                with self.assertRaises(ServiceError) as caught:
                    self.service.inbox_lease_renew("bob", "L1", payload)
                self.assertEqual(caught.exception.status_code, 400)
                self.assertEqual(caught.exception.field, "request_body")

    def test_bad_renewal_id_is_400(self) -> None:
        self._claim("L1", 2)
        for payload in ({}, {"renewal_id": ""}, {"renewal_id": 7},
                        {"renewal_id": None}, {"renewal_id": True},
                        {"renewal_id": ["R"]}):
            with self.subTest(payload=payload):
                with self.assertRaises(ServiceError) as caught:
                    self.service.inbox_lease_renew("bob", "L1", payload)
                self.assertEqual(caught.exception.status_code, 400)
                self.assertEqual(caught.exception.field, "renewal_id")

    def test_renewed_deadline_keeps_message_withheld_from_new_claim(
            self) -> None:
        # The claim deadline has passed (29 seconds ago), but one renewal
        # moved the effective deadline to one second from now, so a new
        # claim still cannot take the message.
        claim = self._claim("L1", 1)
        self.assertEqual([m["message_id"] for m in claim["messages"]],
                         ["a1"])
        renewed_at = datetime.now(timezone.utc) + timedelta(seconds=1)
        claim_at = renewed_at - timedelta(seconds=LEASE_SECONDS)
        for state in self.service.store._delivery.values():
            for lease in state.leases:
                if lease.lease_id == "L1":
                    lease.leased_until = claim_at.isoformat(
                        timespec="microseconds")
                    lease.renewals = [MessageLeaseRenewal(
                        "R1", renewed_at.isoformat(timespec="microseconds"))]
        body, status = self.service.inbox_claim(
            "bob", {"lease_id": "L2", "limit": 10})
        self.assertEqual(status, 201)
        self.assertNotIn("a1", [m["message_id"] for m in body["messages"]])


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

    def test_first_renew_persists_and_advances_one_generation(self) -> None:
        self._claim("L1", 2)
        before = self.state_store.commit_seq
        body, status = self._renew("bob", "L1", "R1")
        self.assertEqual(status, 201)
        self.assertEqual(self.state_store.commit_seq, before + 1)

        leased = [record for record in self._document()["delivery"]
                  if record.get("leases")]
        self.assertEqual(len(leased), 2)
        for record in leased:
            lease = record["leases"][0]
            self.assertEqual(list(lease),
                             ["lease_id", "limit", "leased_until",
                              "released_at", "renewals", "completion", "ack_id"])
            self.assertIsNone(lease["completion"])
            self.assertEqual(len(lease["renewals"]), 1)
            renewal = lease["renewals"][0]
            self.assertEqual(list(renewal), ["renewal_id", "leased_until"])
            self.assertEqual(renewal["renewal_id"], "R1")
            self.assertEqual(renewal["leased_until"], body["leased_until"])

    def test_replay_persists_nothing(self) -> None:
        self._claim("L1", 2)
        first, _ = self._renew("bob", "L1", "R1")
        generation = self.state_store.commit_seq
        replay, status = self._renew("bob", "L1", "R1")
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)
        self.assertEqual(self.state_store.commit_seq, generation)

    def test_restart_restores_renewals_and_replays(self) -> None:
        claim = self._claim("L1", 2)
        first, _ = self._renew("bob", "L1", "R1")
        second, _ = self._renew("bob", "L1", "R2")
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        replay1, status = restarted.inbox_lease_renew(
            "bob", "L1", {"renewal_id": "R1"})
        self.assertEqual(status, 200)
        self.assertEqual(replay1, first)
        replay2, status = restarted.inbox_lease_renew(
            "bob", "L1", {"renewal_id": "R2"})
        self.assertEqual(status, 200)
        self.assertEqual(replay2, second)
        # The chain survived: a third renewal extends the second one.
        third, status = restarted.inbox_lease_renew(
            "bob", "L1", {"renewal_id": "R3"})
        self.assertEqual(status, 201)
        self.assertEqual(
            _parse(third["leased_until"])
            - _parse(second["leased_until"]),
            timedelta(seconds=LEASE_SECONDS))
        # And the renewed deadline still protects the claimed messages.
        body, status = restarted.inbox_claim(
            "bob", {"lease_id": "L2", "limit": 10})
        self.assertEqual(status, 201)
        self.assertEqual([m["message_id"] for m in body["messages"]],
                         ["a3", "b1", "b2"])

    def test_save_failure_rolls_back_and_surfaces(self) -> None:
        import e2ee_backend.persistence as persistence_mod

        self._claim("L1", 2)
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
                self._renew("bob", "L1", "R1")
        finally:
            persistence_mod.os.fsync = real_fsync
        self.assertEqual(self.state_store.commit_seq, generation)
        # The renewal did not land in memory: R1 is a fresh 201.
        body, status = self._renew("bob", "L1", "R1")
        self.assertEqual(status, 201)
        self.assertEqual(self.state_store.commit_seq, generation + 1)

    def test_legacy_lease_without_renewals_loads(self) -> None:
        self._claim("L1", 2)
        document = self._document()
        for record in document["delivery"]:
            for lease in record.get("leases", []):
                lease.pop("renewals", None)
        document.pop("integrity_log_version", None)
        legacy_path = os.path.join(self.directory, "legacy.json")
        with open(legacy_path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        restarted = DeviceService()
        attach_persistence(restarted, legacy_path)  # must not raise
        body, status = restarted.inbox_lease_renew(
            "bob", "L1", {"renewal_id": "R1"})
        self.assertEqual(status, 201)
        self.assertRegex(body["leased_until"], r"\.\d{6}\+00:00$")

    def _malformed_document(self, mutate):
        self._claim_counter = getattr(self, "_claim_counter", 0) + 1
        # Release every earlier probe lease so its messages become
        # claimable again (there are only five messages in the fixture).
        for earlier in range(1, self._claim_counter):
            self.service.inbox_release("bob", f"L1-{earlier}")
        lease_id = f"L1-{self._claim_counter}"
        self._claim(lease_id, 2)
        self.service.inbox_lease_renew(
            "bob", lease_id, {"renewal_id": "R1"})
        document = self._document()
        mutate(document)
        document.pop("integrity_log_version", None)
        bad_path = os.path.join(
            self.directory, f"bad-{self._claim_counter}.json")
        with open(bad_path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        return bad_path

    def _assert_refuses_startup(self, bad_path) -> None:
        with open(bad_path, "rb") as handle:
            original = handle.read()
        restarted = DeviceService()
        with self.assertRaises(StateFileError):
            attach_persistence(restarted, bad_path)
        with open(bad_path, "rb") as handle:
            self.assertEqual(handle.read(), original)

    def test_restore_rejects_renewals_not_a_list(self) -> None:
        def mutate(document):
            for lease in (l for r in document["delivery"]
                         for l in r.get("leases", [])):
                lease["renewals"] = {"renewal_id": "R1"}
        self._assert_refuses_startup(self._malformed_document(mutate))

    def test_restore_rejects_bad_renewal_items(self) -> None:
        def as_list(not_a_dict):
            def mutate(document):
                for lease in (l for r in document["delivery"]
                             for l in r.get("leases", [])):
                    lease["renewals"] = [not_a_dict]
            return mutate
        for item in (7, "R1", None):
            with self.subTest(item=item):
                self._assert_refuses_startup(
                    self._malformed_document(as_list(item)))

    def test_restore_rejects_missing_or_empty_renewal_id(self) -> None:
        def missing(document):
            for lease in (l for r in document["delivery"]
                         for l in r.get("leases", [])):
                del lease["renewals"][0]["renewal_id"]

        def empty(document):
            for lease in (l for r in document["delivery"]
                         for l in r.get("leases", [])):
                lease["renewals"][0]["renewal_id"] = ""
        self._assert_refuses_startup(self._malformed_document(missing))
        self._assert_refuses_startup(self._malformed_document(empty))

    def test_restore_rejects_malformed_renewal_deadline(self) -> None:
        for bad in ("not-a-timestamp",
                    "2026-09-25T14:00:00+00:00",        # no microseconds
                    "2026-09-25T14:00:00.000000",       # no offset
                    "2026-09-25T14:00:00.000000+01:00"):
            with self.subTest(bad=bad):
                def mutate(document):
                    for lease in (l for r in document["delivery"]
                                 for l in r.get("leases", [])):
                        lease["renewals"][0]["leased_until"] = bad
                self._assert_refuses_startup(
                    self._malformed_document(mutate))

    def test_restore_rejects_duplicate_renewal_id(self) -> None:
        def mutate(document):
            for lease in (l for r in document["delivery"]
                         for l in r.get("leases", [])):
                first = dict(lease["renewals"][0])
                # Keep the +30 chain valid; only the id repeats.
                claim = _parse(lease["leased_until"])
                second_deadline = (claim + timedelta(
                    seconds=2 * LEASE_SECONDS)).isoformat(
                    timespec="microseconds")
                second = {"renewal_id": "R1",
                          "leased_until": second_deadline}
                lease["renewals"] = [first, second]
        self._assert_refuses_startup(self._malformed_document(mutate))

    def test_restore_rejects_deadline_not_exactly_30_seconds(self) -> None:
        def mutate(document):
            for lease in (l for r in document["delivery"]
                         for l in r.get("leases", [])):
                claim = _parse(lease["leased_until"])
                lease["renewals"][0]["leased_until"] = (
                    claim + timedelta(seconds=29)).isoformat(
                    timespec="microseconds")
        self._assert_refuses_startup(self._malformed_document(mutate))

    def test_restore_rejects_inconsistent_renewals_across_records(
            self) -> None:
        def mutate(document):
            leased = [r for r in document["delivery"] if r.get("leases")]
            self.assertEqual(len(leased), 2)
            leased[1]["leases"][0]["renewals"][0]["renewal_id"] = "OTHER"
        self._assert_refuses_startup(self._malformed_document(mutate))


class RenewHTTPTest(RenewMixin, unittest.TestCase):
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

    def _request(self, device_id, lease_id, raw):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request(
            "POST",
            f"/v1/devices/{device_id}/inbox/leases/{lease_id}/renew",
            body=raw, headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        data = response.read().decode("utf-8")
        conn.close()
        return response.status, (json.loads(data) if data else None), data

    def _claim(self, lease_id="L1", limit=2):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("POST", "/v1/devices/bob/inbox/claim",
                     body=json.dumps({"lease_id": lease_id, "limit": limit}),
                     headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        response.read()
        conn.close()
        self.assertEqual(response.status, 201)

    def test_renew_and_replay_over_http(self) -> None:
        self._claim("L1", 2)
        status, body, raw = self._request(
            "bob", "L1", json.dumps({"renewal_id": "R1"}))
        self.assertEqual(status, 201)
        self.assertEqual(list(body),
                         ["device_id", "lease_id", "renewal_id",
                          "leased_until"])
        self.assertLess(raw.index('"device_id"'), raw.index('"lease_id"'))
        self.assertLess(raw.index('"lease_id"'),
                        raw.index('"renewal_id"'))
        self.assertLess(raw.index('"renewal_id"'),
                        raw.index('"leased_until"'))
        status, replay, replay_raw = self._request(
            "bob", "L1", json.dumps({"renewal_id": "R1"}))
        self.assertEqual(status, 200)
        self.assertEqual(replay_raw, raw)

    def test_bodies_over_http(self) -> None:
        self._claim("L1", 2)
        for raw in ("not json{", "[]", "7"):
            with self.subTest(raw=raw):
                status, body, _ = self._request("bob", "L1", raw)
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], "request_body")
        for raw in (json.dumps({}), json.dumps({"renewal_id": ""}),
                    json.dumps({"renewal_id": 5})):
            with self.subTest(raw=raw):
                status, body, _ = self._request("bob", "L1", raw)
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], "renewal_id")

    def test_errors_over_http(self) -> None:
        self._claim("L1", 2)
        status, body, _ = self._request(
            "bob", "NOPE", json.dumps({"renewal_id": "R1"}))
        self.assertEqual((status, body["field"]), (404, "lease_id"))
        status, body, _ = self._request(
            "carol", "L1", json.dumps({"renewal_id": "R1"}))
        self.assertEqual((status, body["field"]), (409, "lease_id"))
        self.service.store.revoke_device("bob")
        status, body, _ = self._request(
            "bob", "L1", json.dumps({"renewal_id": "R1"}))
        self.assertEqual((status, body["field"]), (409, "device_id"))

    def test_percent_encoded_lease_id_over_http(self) -> None:
        self._claim("a/b", 1)
        status, body, _ = self._request(
            "bob", "a%2Fb", json.dumps({"renewal_id": "R1"}))
        self.assertEqual(status, 201)
        self.assertEqual(body["lease_id"], "a/b")


if __name__ == "__main__":
    unittest.main()
