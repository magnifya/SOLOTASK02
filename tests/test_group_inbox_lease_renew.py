"""Tests for the group-inbox lease renewal endpoint.

POST /v1/devices/{device_id}/group-inbox/leases/{lease_id}/renew
extends an occupied, still-active group lease's effective deadline by
exactly 30 seconds. The JSON body must be an object carrying only a
non-empty string ``renewal_id``: bad JSON / a non-object body is
400/request_body, a missing, empty or wrongly typed field is
400/renewal_id, an unexpected top-level key is 400/that key, and any
query parameter is 400/query. An unknown group lease is 404/lease_id,
one owned by another device (or committed in the 1:1 namespace) is
409/lease_id; both precede the path device's state. Replaying the same
renewal_id on the same lease returns the first response
byte-identically with 200 (even after expiry, release or device
revocation), while the id is free to reuse on other leases. A first
renewal on a revoked device is 409/device_id; a released or expired
lease is 409/lease_id. A first renewal returns 201 with
device_id/lease_id/renewal_id/leased_until and persists one generation;
the effective deadline is the claim value initially and the last
renewal's value afterwards.
"""
import json
import os
import shutil
import tempfile
import threading
import unittest
from datetime import datetime, timedelta, timezone
from http.client import HTTPConnection

from e2ee_backend.models import Device, MessageLeaseRenewal, SignedPreKey
from e2ee_backend.persistence import (
    PersistenceUnavailable, StateFileError, attach_persistence)
from e2ee_backend.service import DeviceService, ServiceError
from e2ee_backend.http_app import create_server

LEASE_SECONDS = 30


def _parse(stamp: str) -> datetime:
    return datetime.fromisoformat(stamp)


def _message(session_id, message_id, sequence, sender):
    return {
        "session_id": session_id,
        "sender_device_id": sender,
        "message_id": message_id,
        "sequence": sequence,
        "nonce": f"nonce-{message_id}",
        "ciphertext": "ciphertext",
    }


class GroupRenewMixin:
    def _build(self) -> None:
        self.service = DeviceService()
        self.store = self.service.store
        self.store.add_device(Device("u", "d1", "ik"))
        self.store.add_device(Device("u", "d3", "ik"))
        self.store.add_device(Device("u", "d4", "ik"))
        self.store.add_device(Device(
            "u", "d2", "ik",
            prekeys=[SignedPreKey("pk1", "pubk1")]))
        # d1 creates g1(d1,d2,d3); gs1 freezes that roster.
        self.service.create_group({
            "group_id": "g1", "creator_device_id": "d1",
            "member_device_ids": ["d2", "d3"]})
        self.gs1 = self.service.create_group_session({
            "group_id": "g1", "initiator_device_id": "d1",
            "ephemeral_key": "epk1"})["session_id"]
        # m1/m2 from d1, m3 from d2: d2 may claim m1/m2 only, d3 all 3.
        for message_id, sequence, sender in (
                ("m1", 1, "d1"), ("m2", 2, "d1"), ("m3", 3, "d2")):
            self.service.post_message(
                _message(self.gs1, message_id, sequence, sender))
        # A 1:1 session d1 -> d2 never contributes to a group claim.
        self.one2one = self.store.create_session(
            "d1", "d2", "pk1", "ek").session_id
        self.service.post_message(
            _message(self.one2one, "p1", 1, "d1"))

    def _claim(self, lease_id="L1", limit=2, device_id="d3"):
        body, status = self.service.group_inbox_claim(
            device_id, {"lease_id": lease_id, "limit": limit})
        self.assertEqual(status, 201)
        return body

    def _renew(self, device_id, lease_id, renewal_id):
        return self.service.group_inbox_lease_renew(
            device_id, lease_id, {"renewal_id": renewal_id})

    def _expire_group_lease(self, lease_id, *, with_renewal=False,
                            seconds_ago=5) -> None:
        """Rewrite the lease's stored deadlines to the past in memory."""
        if with_renewal:
            seconds_ago = max(seconds_ago, LEASE_SECONDS + 5)
        claim = datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)
        claim_s = claim.isoformat(timespec="microseconds")
        renewal_s = (claim + timedelta(seconds=LEASE_SECONDS)) \
            .isoformat(timespec="microseconds")
        for state in self.store._group_delivery.values():
            for lease in state.leases:
                if lease.lease_id != lease_id:
                    continue
                lease.leased_until = claim_s
                if with_renewal:
                    lease.renewals = [
                        MessageLeaseRenewal("R1", renewal_s)]


class GroupRenewServiceTest(GroupRenewMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def test_first_renew_201_extends_claim_deadline_by_30s(self) -> None:
        claim = self._claim("L1", 2)
        body, status = self._renew("d3", "L1", "R1")
        self.assertEqual(status, 201)
        self.assertEqual(list(body),
                         ["device_id", "lease_id", "renewal_id",
                          "leased_until"])
        self.assertEqual(body["device_id"], "d3")
        self.assertEqual(body["lease_id"], "L1")
        self.assertEqual(body["renewal_id"], "R1")
        self.assertRegex(body["leased_until"], r"\.\d{6}\+00:00$")
        self.assertEqual(
            _parse(body["leased_until"]) - _parse(claim["leased_until"]),
            timedelta(seconds=LEASE_SECONDS))

    def test_second_renew_chains_from_last_renewal(self) -> None:
        claim = self._claim("L1", 2)
        first, _ = self._renew("d3", "L1", "R1")
        second, status = self._renew("d3", "L1", "R2")
        self.assertEqual(status, 201)
        self.assertEqual(
            _parse(second["leased_until"])
            - _parse(first["leased_until"]),
            timedelta(seconds=LEASE_SECONDS))
        self.assertEqual(
            _parse(second["leased_until"])
            - _parse(claim["leased_until"]),
            timedelta(seconds=2 * LEASE_SECONDS))

    def test_renew_extends_every_copy_of_the_lease(self) -> None:
        self._claim("L1", 2)
        self._renew("d3", "L1", "R1")
        copies = [lease
                  for state in self.store._group_delivery.values()
                  for lease in state.leases if lease.lease_id == "L1"]
        self.assertEqual(len(copies), 2)
        deadlines = {lease.renewals[-1].leased_until for lease in copies}
        self.assertEqual(len(deadlines), 1)
        for lease in copies:
            self.assertEqual([(r.renewal_id, r.leased_until)
                              for r in lease.renewals],
                             [(r.renewal_id, r.leased_until)
                              for r in copies[0].renewals])

    def test_replay_same_id_returns_first_response_200(self) -> None:
        self._claim("L1", 2)
        first, first_status = self._renew("d3", "L1", "R1")
        self.assertEqual(first_status, 201)
        replay, status = self._renew("d3", "L1", "R1")
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)

    def test_replay_wins_after_expiry(self) -> None:
        claim_at = datetime.now(timezone.utc) - timedelta(
            seconds=LEASE_SECONDS + 5)
        renewal_s = (claim_at + timedelta(seconds=LEASE_SECONDS)) \
            .isoformat(timespec="microseconds")
        self._claim("L1", 2)
        for state in self.store._group_delivery.values():
            for lease in state.leases:
                if lease.lease_id == "L1":
                    lease.leased_until = claim_at.isoformat(
                        timespec="microseconds")
                    lease.renewals = [MessageLeaseRenewal("R1", renewal_s)]
        replay, status = self._renew("d3", "L1", "R1")
        self.assertEqual(status, 200)
        self.assertEqual(replay, {
            "device_id": "d3", "lease_id": "L1", "renewal_id": "R1",
            "leased_until": renewal_s})

    def test_replay_wins_after_release(self) -> None:
        self._claim("L1", 2)
        first, _ = self._renew("d3", "L1", "R1")
        self.service.group_inbox_release("d3", "L1")
        replay, status = self._renew("d3", "L1", "R1")
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)

    def test_replay_wins_after_device_revoked(self) -> None:
        self._claim("L1", 2)
        first, _ = self._renew("d3", "L1", "R1")
        self.service.revoke_device("d3")
        replay, status = self._renew("d3", "L1", "R1")
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)

    def test_renewal_id_is_reusable_across_leases(self) -> None:
        self._claim("L1", 2)
        first, _ = self._renew("d3", "L1", "SHARED")
        self.service.group_inbox_release("d3", "L1")
        second_claim = self._claim("L2", 10)
        again, status = self._renew("d3", "L2", "SHARED")
        self.assertEqual(status, 201)
        self.assertEqual(again["renewal_id"], "SHARED")
        self.assertEqual(
            _parse(again["leased_until"])
            - _parse(second_claim["leased_until"]),
            timedelta(seconds=LEASE_SECONDS))
        self.assertNotEqual(again["leased_until"], first["leased_until"])

    def test_renewal_id_is_reusable_across_namespaces(self) -> None:
        self._claim("L1", 1)
        group_body, status = self._renew("d3", "L1", "SHARED")
        self.assertEqual(status, 201)
        # d2 holds a 1:1 lease on p1 and may reuse the same renewal id.
        one_body, one_status = self.service.inbox_claim(
            "d2", {"lease_id": "O1", "limit": 5})
        self.assertEqual(one_status, 201)
        renewed, status = self.service.inbox_lease_renew(
            "d2", "O1", {"renewal_id": "SHARED"})
        self.assertEqual(status, 201)
        self.assertEqual(
            _parse(renewed["leased_until"])
            - _parse(one_body["leased_until"]),
            timedelta(seconds=LEASE_SECONDS))
        self.assertNotEqual(renewed["leased_until"],
                            group_body["leased_until"])

    def test_unknown_lease_is_404_lease_id(self) -> None:
        with self.assertRaises(ServiceError) as caught:
            self._renew("d3", "NOPE", "R1")
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
            self._renew("d2", "L1", "R1")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_cross_device_conflict_precedes_path_device_state(self) -> None:
        self._claim("L1", 2)
        self.service.revoke_device("d2")
        with self.assertRaises(ServiceError) as caught:
            self._renew("d2", "L1", "R1")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_one_to_one_lease_id_is_409_lease_id(self) -> None:
        # An id committed in the 1:1 namespace conflicts, regardless of
        # the path device's state.
        self.service.inbox_claim("d2", {"lease_id": "X1", "limit": 5})
        with self.assertRaises(ServiceError) as caught:
            self._renew("d2", "X1", "R1")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_first_renew_on_revoked_device_is_409_device_id(self) -> None:
        self._claim("L1", 2)
        self.service.revoke_device("d3")
        with self.assertRaises(ServiceError) as caught:
            self._renew("d3", "L1", "R1")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "device_id")

    def test_renew_released_lease_is_409_lease_id(self) -> None:
        self._claim("L1", 2)
        self.service.group_inbox_release("d3", "L1")
        with self.assertRaises(ServiceError) as caught:
            self._renew("d3", "L1", "R1")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_renew_expired_lease_is_409_lease_id(self) -> None:
        self._claim("L1", 2)
        self._expire_group_lease("L1")
        with self.assertRaises(ServiceError) as caught:
            self._renew("d3", "L1", "R1")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_bad_body_shapes_are_400(self) -> None:
        self._claim("L1", 2)
        for payload in (None, "x", 7, ["R1"], True):
            with self.subTest(payload=payload):
                with self.assertRaises(ServiceError) as caught:
                    self.service.group_inbox_lease_renew(
                        "d3", "L1", payload)
                self.assertEqual(caught.exception.status_code, 400)
                self.assertEqual(caught.exception.field, "request_body")

    def test_bad_renewal_id_is_400(self) -> None:
        self._claim("L1", 2)
        for payload in ({}, {"renewal_id": ""}, {"renewal_id": 7},
                        {"renewal_id": None}, {"renewal_id": True},
                        {"renewal_id": ["R"]}):
            with self.subTest(payload=payload):
                with self.assertRaises(ServiceError) as caught:
                    self.service.group_inbox_lease_renew(
                        "d3", "L1", payload)
                self.assertEqual(caught.exception.status_code, 400)
                self.assertEqual(caught.exception.field, "renewal_id")

    def test_extra_field_is_400_with_that_field_name(self) -> None:
        self._claim("L1", 2)
        with self.assertRaises(ServiceError) as caught:
            self.service.group_inbox_lease_renew(
                "d3", "L1", {"renewal_id": "R1", "extra": 1})
        self.assertEqual(caught.exception.status_code, 400)
        self.assertEqual(caught.exception.field, "extra")

    def test_renewed_deadline_keeps_message_withheld_from_new_claim(
            self) -> None:
        # The claim deadline has passed, but one renewal moved the
        # effective deadline into the future, so a new claim still
        # cannot take the message.
        self._claim("L1", 1)
        renewed_at = datetime.now(timezone.utc) + timedelta(seconds=1)
        claim_at = renewed_at - timedelta(seconds=LEASE_SECONDS)
        for state in self.store._group_delivery.values():
            for lease in state.leases:
                if lease.lease_id == "L1":
                    lease.leased_until = claim_at.isoformat(
                        timespec="microseconds")
                    lease.renewals = [MessageLeaseRenewal(
                        "R1", renewed_at.isoformat(timespec="microseconds"))]
        body, status = self.service.group_inbox_claim(
            "d3", {"lease_id": "L2", "limit": 10})
        self.assertEqual(status, 201)
        self.assertNotIn("m1", [m["message_id"] for m in body["messages"]])


class GroupRenewPersistenceTest(GroupRenewMixin, unittest.TestCase):
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
        body, status = self._renew("d3", "L1", "R1")
        self.assertEqual(status, 201)
        self.assertEqual(self.state_store.commit_seq, before + 1)

        leased = [record for record in self._document()["group_delivery"]
                  if record.get("leases")]
        self.assertEqual(len(leased), 2)
        for record in leased:
            lease = record["leases"][0]
            self.assertEqual(list(lease),
                             ["lease_id", "limit", "leased_until",
                              "released_at", "renewals"])
            self.assertIsNone(lease["released_at"])
            self.assertEqual(len(lease["renewals"]), 1)
            renewal = lease["renewals"][0]
            self.assertEqual(list(renewal),
                             ["renewal_id", "leased_until"])
            self.assertEqual(renewal["renewal_id"], "R1")
            self.assertEqual(renewal["leased_until"],
                             body["leased_until"])

    def test_replay_persists_nothing(self) -> None:
        self._claim("L1", 2)
        first, _ = self._renew("d3", "L1", "R1")
        generation = self.state_store.commit_seq
        replay, status = self._renew("d3", "L1", "R1")
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)
        self.assertEqual(self.state_store.commit_seq, generation)

    def test_restart_restores_renewals_and_replays(self) -> None:
        self._claim("L1", 2)
        first, _ = self._renew("d3", "L1", "R1")
        second, _ = self._renew("d3", "L1", "R2")
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        replay1, status = restarted.group_inbox_lease_renew(
            "d3", "L1", {"renewal_id": "R1"})
        self.assertEqual(status, 200)
        self.assertEqual(replay1, first)
        replay2, status = restarted.group_inbox_lease_renew(
            "d3", "L1", {"renewal_id": "R2"})
        self.assertEqual(status, 200)
        self.assertEqual(replay2, second)
        # The chain survived: a third renewal extends the second one.
        third, status = restarted.group_inbox_lease_renew(
            "d3", "L1", {"renewal_id": "R3"})
        self.assertEqual(status, 201)
        self.assertEqual(
            _parse(third["leased_until"])
            - _parse(second["leased_until"]),
            timedelta(seconds=LEASE_SECONDS))
        # And the renewed deadline still protects the claimed messages.
        body, status = restarted.group_inbox_claim(
            "d3", {"lease_id": "L2", "limit": 10})
        self.assertEqual(status, 201)
        self.assertEqual([m["message_id"] for m in body["messages"]],
                         ["m3"])

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
                self._renew("d3", "L1", "R1")
        finally:
            persistence_mod.os.fsync = real_fsync
        self.assertEqual(self.state_store.commit_seq, generation)
        # The renewal did not land in memory: R1 is a fresh 201.
        body, status = self._renew("d3", "L1", "R1")
        self.assertEqual(status, 201)
        self.assertEqual(self.state_store.commit_seq, generation + 1)

    def test_legacy_lease_without_renewals_loads(self) -> None:
        self._claim("L1", 2)
        document = self._document()
        for record in document["group_delivery"]:
            for lease in record.get("leases", []):
                lease.pop("renewals", None)
        document.pop("integrity_log_version", None)
        legacy_path = os.path.join(self.directory, "legacy.json")
        with open(legacy_path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        restarted = DeviceService()
        attach_persistence(restarted, legacy_path)  # must not raise
        body, status = restarted.group_inbox_lease_renew(
            "d3", "L1", {"renewal_id": "R1"})
        self.assertEqual(status, 201)
        self.assertRegex(body["leased_until"], r"\.\d{6}\+00:00$")

    def _malformed_document(self, mutate):
        self._probe_counter = getattr(self, "_probe_counter", 0) + 1
        # Release every earlier probe lease so its messages become
        # claimable again (the fixture leases two messages per probe).
        for earlier in range(1, self._probe_counter):
            self.service.group_inbox_release("d3", f"L{earlier}")
        lease_id = f"L{self._probe_counter}"
        self._claim(lease_id, 2)
        self._renew("d3", lease_id, "R1")
        document = self._document()
        mutate(document)
        document.pop("integrity_log_version", None)
        bad_path = os.path.join(
            self.directory, f"bad-{self._probe_counter}.json")
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
            for lease in (l for r in document["group_delivery"]
                         for l in r.get("leases", [])):
                lease["renewals"] = {"renewal_id": "R1"}
        self._assert_refuses_startup(self._malformed_document(mutate))

    def test_restore_rejects_bad_renewal_items(self) -> None:
        for item in (7, "R1", None):
            with self.subTest(item=item):
                def mutate(document, item=item):
                    for lease in (l for r in document["group_delivery"]
                                 for l in r.get("leases", [])):
                        lease["renewals"] = [item]
                self._assert_refuses_startup(
                    self._malformed_document(mutate))

    def test_restore_rejects_missing_or_empty_renewal_id(self) -> None:
        def missing(document):
            for lease in (l for r in document["group_delivery"]
                         for l in r.get("leases", [])):
                del lease["renewals"][0]["renewal_id"]

        def empty(document):
            for lease in (l for r in document["group_delivery"]
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
                def mutate(document, bad=bad):
                    for lease in (l for r in document["group_delivery"]
                                 for l in r.get("leases", [])):
                        lease["renewals"][0]["leased_until"] = bad
                self._assert_refuses_startup(
                    self._malformed_document(mutate))

    def test_restore_rejects_duplicate_renewal_id(self) -> None:
        def mutate(document):
            for lease in (l for r in document["group_delivery"]
                         for l in r.get("leases", [])):
                first = dict(lease["renewals"][0])
                # Keep the +30 chain valid; only the id repeats.
                claim = _parse(lease["leased_until"])
                second_deadline = (claim + timedelta(
                    seconds=2 * LEASE_SECONDS)).isoformat(
                    timespec="microseconds")
                lease["renewals"] = [
                    first,
                    {"renewal_id": "R1",
                     "leased_until": second_deadline}]
        self._assert_refuses_startup(self._malformed_document(mutate))

    def test_restore_rejects_deadline_not_exactly_30_seconds(self) -> None:
        def mutate(document):
            for lease in (l for r in document["group_delivery"]
                         for l in r.get("leases", [])):
                claim = _parse(lease["leased_until"])
                lease["renewals"][0]["leased_until"] = (
                    claim + timedelta(seconds=29)).isoformat(
                    timespec="microseconds")
        self._assert_refuses_startup(self._malformed_document(mutate))

    def test_restore_rejects_inconsistent_renewals_across_records(
            self) -> None:
        def mutate(document):
            leased = [r for r in document["group_delivery"]
                      if r.get("leases")]
            self.assertEqual(len(leased), 2)
            leased[1]["leases"][0]["renewals"][0]["renewal_id"] = "OTHER"
        self._assert_refuses_startup(self._malformed_document(mutate))

    def test_restore_rejects_group_terminal_keys(self) -> None:
        # Group leases never carry the 1:1 bulk-ack id field (their
        # optional completion is a legal key since completions were
        # added).
        for key in ("ack_id",):
            with self.subTest(key=key):
                def mutate(document, key=key):
                    for lease in (l for r in document["group_delivery"]
                                 for l in r.get("leases", [])):
                        lease[key] = None
                self._assert_refuses_startup(
                    self._malformed_document(mutate))


class GroupRenewHTTPTest(GroupRenewMixin, unittest.TestCase):
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

    def _request(self, device_id, lease_id, raw, query=""):
        target = f"/v1/devices/{device_id}/group-inbox/leases/" \
                 f"{lease_id}/renew{query}"
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("POST", target, body=raw,
                     headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        data = response.read().decode("utf-8")
        conn.close()
        return response.status, (json.loads(data) if data else None), data

    def _claim(self, lease_id="L1", limit=2, device_id="d3"):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request(
            "POST", f"/v1/devices/{device_id}/group-inbox/claim",
            body=json.dumps({"lease_id": lease_id, "limit": limit}),
            headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        response.read()
        conn.close()
        self.assertEqual(response.status, 201)

    def test_renew_and_replay_over_http(self) -> None:
        self._claim("L1", 2)
        status, body, raw = self._request(
            "d3", "L1", json.dumps({"renewal_id": "R1"}))
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
            "d3", "L1", json.dumps({"renewal_id": "R1"}))
        self.assertEqual(status, 200)
        self.assertEqual(replay_raw, raw)

    def test_bodies_over_http(self) -> None:
        self._claim("L1", 2)
        for raw in ("not json{", "[]", "7"):
            with self.subTest(raw=raw):
                status, body, _ = self._request("d3", "L1", raw)
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], "request_body")
        for raw in (json.dumps({}), json.dumps({"renewal_id": ""}),
                    json.dumps({"renewal_id": 5})):
            with self.subTest(raw=raw):
                status, body, _ = self._request("d3", "L1", raw)
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], "renewal_id")

    def test_extra_field_over_http(self) -> None:
        self._claim("L1", 2)
        status, body, _ = self._request(
            "d3", "L1",
            json.dumps({"renewal_id": "R1", "nope": 1}))
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "nope")

    def test_query_parameters_are_400_query(self) -> None:
        self._claim("L1", 2)
        raw = json.dumps({"renewal_id": "R1"})
        for query in ("?foo", "?renewal_id=R1", "?x="):
            with self.subTest(query=query):
                status, body, _ = self._request("d3", "L1", raw, query)
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], "query")
        # A bare trailing '?' carries no parameter and is accepted.
        status, _, _ = self._request("d3", "L1", raw, "?")
        self.assertEqual(status, 201)

    def test_errors_over_http(self) -> None:
        self._claim("L1", 2)
        status, body, _ = self._request(
            "d3", "NOPE", json.dumps({"renewal_id": "R1"}))
        self.assertEqual((status, body["field"]), (404, "lease_id"))
        status, body, _ = self._request(
            "d2", "L1", json.dumps({"renewal_id": "R1"}))
        self.assertEqual((status, body["field"]), (409, "lease_id"))
        self.service.store.revoke_device("d3")
        status, body, _ = self._request(
            "d3", "L1", json.dumps({"renewal_id": "R1"}))
        self.assertEqual((status, body["field"]), (409, "device_id"))

    def test_bad_path_escapes_are_400_by_segment(self) -> None:
        self._claim("L1", 2)
        raw = json.dumps({"renewal_id": "R1"})
        # Bad device_id escape is reported first.
        status, body, _ = self._request("d%zz", "L1", raw)
        self.assertEqual((status, body["field"]), (400, "device_id"))
        # Invalid UTF-8 in the device segment.
        status, body, _ = self._request("d%ff", "L1", raw)
        self.assertEqual((status, body["field"]), (400, "device_id"))
        # A valid device segment reaches the lease-segment check.
        status, body, _ = self._request("d3", "L%zz", raw)
        self.assertEqual((status, body["field"]), (400, "lease_id"))
        status, body, _ = self._request("d3", "L%ff", raw)
        self.assertEqual((status, body["field"]), (400, "lease_id"))

    def test_percent_encoded_segments_over_http(self) -> None:
        self._claim("a/b", 1, device_id="d3")
        self._claim("L1", 1, device_id="d3")
        status, body, _ = self._request(
            "d3", "a%2Fb", json.dumps({"renewal_id": "R1"}))
        self.assertEqual(status, 201)
        self.assertEqual(body["lease_id"], "a/b")
        # A percent-encoded slash in the device segment decodes to an
        # ordinary '/', so it names an unknown (single-segment) device —
        # routing never splits on the encoded slash (not a 404 route).
        status, body, _ = self._request(
            "d%33", "L1", json.dumps({"renewal_id": "R1"}))
        self.assertEqual(status, 201)
        self.assertEqual(body["device_id"], "d3")

    def test_concurrent_same_renewal_linearizes_to_one_201(self) -> None:
        self._claim("L1", 2)
        results = []
        barrier = threading.Barrier(8)

        def worker() -> None:
            barrier.wait()
            status, body, _ = self._request(
                "d3", "L1", json.dumps({"renewal_id": "RACE"}))
            results.append((status, body["leased_until"]))

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
        self.assertEqual(len(results), 8)
        self.assertEqual(sorted(status for status, _ in results),
                         [200] * 7 + [201])
        deadlines = {deadline for _status, deadline in results}
        self.assertEqual(len(deadlines), 1)


if __name__ == "__main__":
    unittest.main()
