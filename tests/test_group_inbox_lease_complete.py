"""Tests for the group-inbox lease completion endpoint.

POST /v1/devices/{device_id}/group-inbox/leases/{lease_id}/complete
records the terminal outcome (``delivered`` or ``failed``) of one
occupied, still-active group lease. The JSON body must be an object
carrying only a non-empty string ``completion_id`` and an ``outcome``
of exactly ``delivered``/``failed``: bad JSON / a non-object body is
400/request_body, an unexpected top-level key is 400/that key, a
missing, empty or wrongly typed ``completion_id`` is 400/completion_id,
a missing or non-allowed ``outcome`` is 400/outcome, and any query
parameter is 400/query. An unknown group lease is 404/lease_id, one
owned by another device (or committed in the 1:1 namespace) is
409/lease_id; both precede the path device's state. Replaying the same
completion_id on the same lease returns the first response
byte-identically with 200 (even after expiry or device revocation),
while the id is free to reuse on other leases; completing the lease
again under another id is 409/completion_id. A first completion on an
unknown/revoked device is 409/device_id; a released or expired lease is
409/lease_id. A first completion returns 201 with
device_id/lease_id/completion_id/outcome/completed_at and persists one
generation. Completion ends the lease: no renewal or first release
follows, and a ``delivered`` outcome is not an acknowledgement — the
still-unacked messages may be claimed again.
"""
import json
import os
import shutil
import tempfile
import threading
import unittest
from datetime import datetime, timedelta, timezone
from http.client import HTTPConnection

from e2ee_backend.models import Device, SignedPreKey
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


class GroupCompleteMixin:
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

    def _complete(self, device_id="d3", lease_id="L1", completion_id="C1",
                  outcome="delivered"):
        return self.service.group_inbox_lease_complete(
            device_id, lease_id,
            {"completion_id": completion_id, "outcome": outcome})

    def _expire_group_lease(self, lease_id, seconds_ago=5) -> None:
        """Rewrite the lease's stored deadline to the past in memory."""
        claim = datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)
        claim_s = claim.isoformat(timespec="microseconds")
        for state in self.store._group_delivery.values():
            for lease in state.leases:
                if lease.lease_id == lease_id:
                    lease.leased_until = claim_s


class GroupCompleteServiceTest(GroupCompleteMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def test_first_complete_201_five_fields_in_order(self) -> None:
        self._claim("L1", 2)
        body, status = self._complete()
        self.assertEqual(status, 201)
        self.assertEqual(list(body),
                         ["device_id", "lease_id", "completion_id",
                          "outcome", "completed_at"])
        self.assertEqual(body["device_id"], "d3")
        self.assertEqual(body["lease_id"], "L1")
        self.assertEqual(body["completion_id"], "C1")
        self.assertEqual(body["outcome"], "delivered")
        self.assertRegex(body["completed_at"], r"\.\d{6}\+00:00$")

    def test_failed_outcome_is_accepted(self) -> None:
        self._claim("L1", 2)
        body, status = self._complete(outcome="failed")
        self.assertEqual(status, 201)
        self.assertEqual(body["outcome"], "failed")

    def test_completion_stamps_every_copy_of_the_lease(self) -> None:
        self._claim("L1", 2)
        body, _ = self._complete()
        copies = [lease
                  for state in self.store._group_delivery.values()
                  for lease in state.leases if lease.lease_id == "L1"]
        self.assertEqual(len(copies), 2)
        for lease in copies:
            self.assertIsNotNone(lease.completion)
            self.assertEqual(lease.completion.completion_id, "C1")
            self.assertEqual(lease.completion.outcome, "delivered")
            self.assertEqual(lease.completion.completed_at,
                             body["completed_at"])

    def test_replay_same_id_returns_first_response_200(self) -> None:
        self._claim("L1", 2)
        first, first_status = self._complete()
        self.assertEqual(first_status, 201)
        replay, status = self._complete()
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)

    def test_replay_same_id_other_outcome_returns_frozen_first(self) -> None:
        self._claim("L1", 2)
        first, _ = self._complete(outcome="delivered")
        replay, status = self._complete(outcome="failed")
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)

    def test_replay_wins_after_expiry(self) -> None:
        self._claim("L1", 2)
        first, _ = self._complete()
        self._expire_group_lease("L1")
        replay, status = self._complete()
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)

    def test_replay_wins_after_device_revoked(self) -> None:
        self._claim("L1", 2)
        first, _ = self._complete()
        self.service.revoke_device("d3")
        replay, status = self._complete()
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)

    def test_second_completion_under_another_id_is_409(self) -> None:
        self._claim("L1", 2)
        self._complete()
        with self.assertRaises(ServiceError) as caught:
            self._complete(completion_id="C2")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "completion_id")

    def test_completion_id_is_reusable_across_leases(self) -> None:
        self._claim("L1", 2)
        self._complete("d3", "L1", "SHARED")
        self._claim("L2", 2)
        body, status = self._complete("d3", "L2", "SHARED")
        self.assertEqual(status, 201)
        self.assertEqual(body["completion_id"], "SHARED")

    def test_completion_id_is_reusable_across_namespaces(self) -> None:
        self._claim("L1", 1)
        self._complete("d3", "L1", "SHARED")
        # d2 holds a 1:1 lease on p1 and may reuse the same completion id.
        self.service.inbox_claim("d2", {"lease_id": "O1", "limit": 5})
        body, status = self.service.inbox_lease_complete(
            "d2", "O1", {"completion_id": "SHARED", "outcome": "failed"})
        self.assertEqual(status, 201)
        self.assertEqual(body["outcome"], "failed")

    def test_unknown_lease_is_404_lease_id(self) -> None:
        with self.assertRaises(ServiceError) as caught:
            self._complete(lease_id="NOPE")
        self.assertEqual(caught.exception.status_code, 404)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_unknown_lease_404_precedes_unknown_device(self) -> None:
        with self.assertRaises(ServiceError) as caught:
            self._complete(device_id="ghost", lease_id="NOPE")
        self.assertEqual(caught.exception.status_code, 404)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_cross_device_complete_is_409_lease_id(self) -> None:
        self._claim("L1", 2)
        with self.assertRaises(ServiceError) as caught:
            self._complete(device_id="d2")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_cross_device_conflict_precedes_path_device_state(self) -> None:
        self._claim("L1", 2)
        self.service.revoke_device("d2")
        with self.assertRaises(ServiceError) as caught:
            self._complete(device_id="d2")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_one_to_one_lease_id_is_409_lease_id(self) -> None:
        # An id committed in the 1:1 namespace conflicts, regardless of
        # the path device's state.
        self.service.inbox_claim("d2", {"lease_id": "X1", "limit": 5})
        with self.assertRaises(ServiceError) as caught:
            self._complete(device_id="d2", lease_id="X1")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_first_complete_on_revoked_device_is_409_device_id(self) -> None:
        self._claim("L1", 2)
        self.service.revoke_device("d3")
        with self.assertRaises(ServiceError) as caught:
            self._complete()
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "device_id")

    def test_first_complete_on_unknown_device_is_409_device_id(self) -> None:
        # A lease owner that vanishes from the device index cannot happen
        # through the API, but the store reason still maps cleanly.
        self._claim("L1", 2)
        index = self.store._device_index
        user_key = index.pop("d3")
        self.store._devices.pop(user_key)
        with self.assertRaises(ServiceError) as caught:
            self._complete()
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "device_id")

    def test_complete_released_lease_is_409_lease_id(self) -> None:
        self._claim("L1", 2)
        self.service.group_inbox_release("d3", "L1")
        with self.assertRaises(ServiceError) as caught:
            self._complete()
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_complete_expired_lease_is_409_lease_id(self) -> None:
        self._claim("L1", 2)
        self._expire_group_lease("L1")
        with self.assertRaises(ServiceError) as caught:
            self._complete()
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_bad_body_shapes_are_400(self) -> None:
        self._claim("L1", 2)
        for payload in (None, "x", 7, ["C1"], True):
            with self.subTest(payload=payload):
                with self.assertRaises(ServiceError) as caught:
                    self.service.group_inbox_lease_complete(
                        "d3", "L1", payload)
                self.assertEqual(caught.exception.status_code, 400)
                self.assertEqual(caught.exception.field, "request_body")

    def test_bad_completion_id_is_400(self) -> None:
        self._claim("L1", 2)
        for payload in ({"outcome": "delivered"},
                        {"completion_id": "", "outcome": "delivered"},
                        {"completion_id": 7, "outcome": "delivered"},
                        {"completion_id": None, "outcome": "delivered"},
                        {"completion_id": True, "outcome": "delivered"},
                        {"completion_id": ["C"], "outcome": "delivered"}):
            with self.subTest(payload=payload):
                with self.assertRaises(ServiceError) as caught:
                    self.service.group_inbox_lease_complete(
                        "d3", "L1", payload)
                self.assertEqual(caught.exception.status_code, 400)
                self.assertEqual(caught.exception.field, "completion_id")

    def test_bad_outcome_is_400(self) -> None:
        self._claim("L1", 2)
        for payload in ({"completion_id": "C1"},
                        {"completion_id": "C1", "outcome": ""},
                        {"completion_id": "C1", "outcome": "acked"},
                        {"completion_id": "C1", "outcome": "DELIVERED"},
                        {"completion_id": "C1", "outcome": 7},
                        {"completion_id": "C1", "outcome": None},
                        {"completion_id": "C1", "outcome": True}):
            with self.subTest(payload=payload):
                with self.assertRaises(ServiceError) as caught:
                    self.service.group_inbox_lease_complete(
                        "d3", "L1", payload)
                self.assertEqual(caught.exception.status_code, 400)
                self.assertEqual(caught.exception.field, "outcome")

    def test_extra_field_is_400_with_that_field_name(self) -> None:
        self._claim("L1", 2)
        with self.assertRaises(ServiceError) as caught:
            self.service.group_inbox_lease_complete(
                "d3", "L1",
                {"completion_id": "C1", "outcome": "delivered",
                 "extra": 1})
        self.assertEqual(caught.exception.status_code, 400)
        self.assertEqual(caught.exception.field, "extra")

    def test_completed_lease_cannot_be_renewed(self) -> None:
        self._claim("L1", 2)
        self._complete()
        with self.assertRaises(ServiceError) as caught:
            self.service.group_inbox_lease_renew(
                "d3", "L1", {"renewal_id": "R1"})
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_completed_lease_cannot_be_released(self) -> None:
        self._claim("L1", 2)
        self._complete()
        with self.assertRaises(ServiceError) as caught:
            self.service.group_inbox_release("d3", "L1")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_delivered_completion_is_not_an_acknowledgement(self) -> None:
        self._claim("L1", 2)
        self._complete(outcome="delivered")
        # The messages stay unacked on their group delivery records...
        for state in self.store._group_delivery.values():
            self.assertFalse(state.acked)
        # ... and still show up in the group inbox.
        inbox = self.service.device_group_inbox("d3", 10)
        self.assertEqual(
            sorted(m["message_id"] for m in inbox["messages"]),
            ["m1", "m2", "m3"])

    def test_completed_lease_stops_withholding_messages(self) -> None:
        self._claim("L1", 2)
        self._complete()
        # A new lease may claim the still-unacked messages again.
        body, status = self.service.group_inbox_claim(
            "d3", {"lease_id": "L2", "limit": 10})
        self.assertEqual(status, 201)
        self.assertEqual([m["message_id"] for m in body["messages"]],
                         ["m1", "m2", "m3"])

    def test_lease_get_reports_completed_state_and_completion(self) -> None:
        self._claim("L1", 2)
        completed, _ = self._complete(outcome="failed")
        body = self.service.group_inbox_lease_get("d3", "L1")
        self.assertEqual(body["state"], "completed")
        self.assertEqual(body["completion"], {
            "completion_id": "C1",
            "outcome": "failed",
            "completed_at": completed["completed_at"]})
        self.assertEqual(list(body["completion"]),
                         ["completion_id", "outcome", "completed_at"])


class GroupCompletePersistenceTest(GroupCompleteMixin, unittest.TestCase):
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

    def test_first_complete_persists_and_advances_one_generation(
            self) -> None:
        self._claim("L1", 2)
        before = self.state_store.commit_seq
        body, status = self._complete()
        self.assertEqual(status, 201)
        self.assertEqual(self.state_store.commit_seq, before + 1)

        leased = [record for record in self._document()["group_delivery"]
                  if record.get("leases")]
        self.assertEqual(len(leased), 2)
        for record in leased:
            lease = record["leases"][0]
            self.assertEqual(list(lease),
                             ["lease_id", "limit", "leased_until",
                              "released_at", "renewals", "completion"])
            completion = lease["completion"]
            self.assertEqual(list(completion),
                             ["completion_id", "outcome", "completed_at"])
            self.assertEqual(completion["completion_id"], "C1")
            self.assertEqual(completion["outcome"], "delivered")
            self.assertEqual(completion["completed_at"],
                             body["completed_at"])

    def test_replay_persists_nothing(self) -> None:
        self._claim("L1", 2)
        first, _ = self._complete()
        generation = self.state_store.commit_seq
        replay, status = self._complete()
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)
        self.assertEqual(self.state_store.commit_seq, generation)

    def test_restart_restores_completion_and_replays(self) -> None:
        self._claim("L1", 2)
        first, _ = self._complete(outcome="failed")
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        replay, status = restarted.group_inbox_lease_complete(
            "d3", "L1", {"completion_id": "C1", "outcome": "failed"})
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)
        # The completed lease stays on the records as history.
        body = restarted.group_inbox_lease_get("d3", "L1")
        self.assertEqual(body["state"], "completed")
        self.assertEqual(body["completion"]["outcome"], "failed")
        # And it still cannot be renewed, released or completed again.
        with self.assertRaises(ServiceError) as caught:
            restarted.group_inbox_lease_renew(
                "d3", "L1", {"renewal_id": "R1"})
        self.assertEqual(caught.exception.status_code, 409)
        with self.assertRaises(ServiceError) as caught:
            restarted.group_inbox_release("d3", "L1")
        self.assertEqual(caught.exception.status_code, 409)
        with self.assertRaises(ServiceError) as caught:
            restarted.group_inbox_lease_complete(
                "d3", "L1", {"completion_id": "C2", "outcome": "failed"})
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "completion_id")

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
                self._complete()
        finally:
            persistence_mod.os.fsync = real_fsync
        self.assertEqual(self.state_store.commit_seq, generation)
        # The completion did not land in memory: C1 is a fresh 201.
        body, status = self._complete()
        self.assertEqual(status, 201)
        self.assertEqual(self.state_store.commit_seq, generation + 1)

    def test_legacy_lease_without_completion_loads(self) -> None:
        self._claim("L1", 2)
        document = self._document()
        for record in document["group_delivery"]:
            for lease in record.get("leases", []):
                lease.pop("completion", None)
        document.pop("integrity_log_version", None)
        legacy_path = os.path.join(self.directory, "legacy.json")
        with open(legacy_path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        restarted = DeviceService()
        attach_persistence(restarted, legacy_path)  # must not raise
        body = restarted.group_inbox_lease_get("d3", "L1")
        self.assertIsNone(body["completion"])
        self.assertEqual(body["state"], "active")
        # The lease completes normally afterwards.
        completed, status = restarted.group_inbox_lease_complete(
            "d3", "L1", {"completion_id": "C1", "outcome": "delivered"})
        self.assertEqual(status, 201)
        self.assertRegex(completed["completed_at"], r"\.\d{6}\+00:00$")

    def _malformed_document(self, mutate):
        self._probe_counter = getattr(self, "_probe_counter", 0) + 1
        # Complete every earlier probe lease so its messages become
        # claimable again (the fixture leases two messages per probe).
        for earlier in range(1, self._probe_counter):
            self._complete(lease_id=f"L{earlier}",
                           completion_id=f"C{earlier}")
        lease_id = f"L{self._probe_counter}"
        self._claim(lease_id, 2)
        self._complete(lease_id=lease_id, completion_id=f"C{lease_id[1:]}")
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

    def test_restore_rejects_completion_not_null_or_object(self) -> None:
        for bad in (7, "C1", [], True):
            with self.subTest(bad=bad):
                def mutate(document, bad=bad):
                    for lease in (l for r in document["group_delivery"]
                                 for l in r.get("leases", [])):
                        lease["completion"] = bad
                self._assert_refuses_startup(
                    self._malformed_document(mutate))

    def test_restore_rejects_completion_key_order_or_extra_key(self) -> None:
        def reordered(document):
            for lease in (l for r in document["group_delivery"]
                         for l in r.get("leases", [])):
                completion = lease["completion"]
                lease["completion"] = {
                    "outcome": completion["outcome"],
                    "completion_id": completion["completion_id"],
                    "completed_at": completion["completed_at"]}

        def extra(document):
            for lease in (l for r in document["group_delivery"]
                         for l in r.get("leases", [])):
                lease["completion"]["extra"] = 1

        def missing(document):
            for lease in (l for r in document["group_delivery"]
                         for l in r.get("leases", [])):
                del lease["completion"]["completed_at"]
        self._assert_refuses_startup(self._malformed_document(reordered))
        self._assert_refuses_startup(self._malformed_document(extra))
        self._assert_refuses_startup(self._malformed_document(missing))

    def test_restore_rejects_bad_completion_fields(self) -> None:
        def empty_id(document):
            for lease in (l for r in document["group_delivery"]
                         for l in r.get("leases", [])):
                lease["completion"]["completion_id"] = ""

        def bad_outcome(document):
            for lease in (l for r in document["group_delivery"]
                         for l in r.get("leases", [])):
                lease["completion"]["outcome"] = "acked"

        def bad_stamp(document):
            for lease in (l for r in document["group_delivery"]
                         for l in r.get("leases", [])):
                lease["completion"]["completed_at"] = \
                    "2026-09-25T14:00:00+00:00"  # no microseconds
        self._assert_refuses_startup(self._malformed_document(empty_id))
        self._assert_refuses_startup(self._malformed_document(bad_outcome))
        self._assert_refuses_startup(self._malformed_document(bad_stamp))

    def test_restore_rejects_release_and_completion_together(self) -> None:
        def mutate(document):
            for lease in (l for r in document["group_delivery"]
                         for l in r.get("leases", [])):
                lease["released_at"] = \
                    "2026-09-25T14:00:00.000000+00:00"
        self._assert_refuses_startup(self._malformed_document(mutate))

    def test_restore_rejects_inconsistent_completion_across_records(
            self) -> None:
        def mutate(document):
            leased = [r for r in document["group_delivery"]
                      if r.get("leases")]
            self.assertEqual(len(leased), 2)
            leased[1]["leases"][0]["completion"]["outcome"] = "failed"
        self._assert_refuses_startup(self._malformed_document(mutate))


class GroupCompleteHTTPTest(GroupCompleteMixin, unittest.TestCase):
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
                 f"{lease_id}/complete{query}"
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

    def test_complete_and_replay_over_http(self) -> None:
        self._claim("L1", 2)
        payload = json.dumps(
            {"completion_id": "C1", "outcome": "delivered"})
        status, body, raw = self._request("d3", "L1", payload)
        self.assertEqual(status, 201)
        self.assertEqual(list(body),
                         ["device_id", "lease_id", "completion_id",
                          "outcome", "completed_at"])
        self.assertLess(raw.index('"device_id"'), raw.index('"lease_id"'))
        self.assertLess(raw.index('"lease_id"'),
                        raw.index('"completion_id"'))
        self.assertLess(raw.index('"completion_id"'),
                        raw.index('"outcome"'))
        self.assertLess(raw.index('"outcome"'),
                        raw.index('"completed_at"'))
        status, replay, replay_raw = self._request("d3", "L1", payload)
        self.assertEqual(status, 200)
        self.assertEqual(replay_raw, raw)

    def test_bodies_over_http(self) -> None:
        self._claim("L1", 2)
        for raw in ("not json{", "[]", "7"):
            with self.subTest(raw=raw):
                status, body, _ = self._request("d3", "L1", raw)
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], "request_body")
        for raw in (json.dumps({"outcome": "delivered"}),
                    json.dumps({"completion_id": "",
                                "outcome": "delivered"}),
                    json.dumps({"completion_id": 5,
                                "outcome": "delivered"})):
            with self.subTest(raw=raw):
                status, body, _ = self._request("d3", "L1", raw)
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], "completion_id")
        for raw in (json.dumps({"completion_id": "C1"}),
                    json.dumps({"completion_id": "C1", "outcome": "x"}),
                    json.dumps({"completion_id": "C1", "outcome": 5})):
            with self.subTest(raw=raw):
                status, body, _ = self._request("d3", "L1", raw)
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], "outcome")

    def test_extra_field_over_http(self) -> None:
        self._claim("L1", 2)
        status, body, _ = self._request(
            "d3", "L1",
            json.dumps({"completion_id": "C1", "outcome": "delivered",
                        "nope": 1}))
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "nope")

    def test_query_parameters_are_400_query(self) -> None:
        self._claim("L1", 2)
        raw = json.dumps({"completion_id": "C1", "outcome": "delivered"})
        for query in ("?foo", "?outcome=delivered", "?x="):
            with self.subTest(query=query):
                status, body, _ = self._request("d3", "L1", raw, query)
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], "query")
        # A bare trailing '?' carries no parameter and is accepted.
        status, _, _ = self._request("d3", "L1", raw, "?")
        self.assertEqual(status, 201)

    def test_errors_over_http(self) -> None:
        self._claim("L1", 2)
        payload = json.dumps({"completion_id": "C1", "outcome": "failed"})
        status, body, _ = self._request("d3", "NOPE", payload)
        self.assertEqual((status, body["field"]), (404, "lease_id"))
        status, body, _ = self._request("d2", "L1", payload)
        self.assertEqual((status, body["field"]), (409, "lease_id"))
        self.service.store.revoke_device("d3")
        status, body, _ = self._request("d3", "L1", payload)
        self.assertEqual((status, body["field"]), (409, "device_id"))

    def test_bad_path_escapes_are_400_by_segment(self) -> None:
        self._claim("L1", 2)
        raw = json.dumps({"completion_id": "C1", "outcome": "delivered"})
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
        raw = json.dumps({"completion_id": "C1", "outcome": "delivered"})
        status, body, _ = self._request("d3", "a%2Fb", raw)
        self.assertEqual(status, 201)
        self.assertEqual(body["lease_id"], "a/b")
        # A percent-encoded slash in the device segment decodes to an
        # ordinary '/', so it names an unknown (single-segment) device —
        # routing never splits on the encoded slash (not a 404 route).
        self._claim("L1", 1, device_id="d3")
        status, body, _ = self._request("d%33", "L1", raw)
        self.assertEqual(status, 201)
        self.assertEqual(body["device_id"], "d3")

    def test_concurrent_same_completion_linearizes_to_one_201(self) -> None:
        self._claim("L1", 2)
        results = []
        barrier = threading.Barrier(8)
        payload = json.dumps(
            {"completion_id": "RACE", "outcome": "delivered"})

        def worker() -> None:
            barrier.wait()
            status, body, _ = self._request("d3", "L1", payload)
            results.append((status, body["completed_at"]))

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
        self.assertEqual(len(results), 8)
        self.assertEqual(sorted(status for status, _ in results),
                         [200] * 7 + [201])
        stamps = {stamp for _status, stamp in results}
        self.assertEqual(len(stamps), 1)


if __name__ == "__main__":
    unittest.main()
