"""Tests for the 1:1 inbox lease completion endpoint.

POST /v1/devices/{device_id}/inbox/leases/{lease_id}/complete settles an
occupied, still-active inbox lease exactly once. The JSON body must be an
object carrying a non-empty string ``completion_id`` and an ``outcome``
of ``delivered`` or ``failed``: bad JSON / a non-object body is
400/request_body, a missing, empty or wrongly typed field is
400/completion_id resp. 400/outcome. An unknown lease is 404/lease_id,
one owned by another device is 409/lease_id; both precede the path
device's state. Replaying the same completion_id on the same lease
returns the first response byte-identically with 200 (even after expiry
or device revocation), a different completion_id is 409/completion_id,
and the id is free to reuse on another lease. A first completion on a
revoked device is 409/device_id; a released or expired lease is
409/lease_id. A first completion returns 201 with
device_id/lease_id/completion_id/outcome/completed_at and persists one
generation. Completion is terminal: the lease can neither be renewed nor
released afterwards, and its still-unacked messages become claimable
again — ``delivered`` is a delivery report, not an acknowledgement.
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


class CompleteMixin:
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

    def _complete(self, device_id, lease_id, completion_id,
                  outcome="delivered"):
        return self.service.inbox_lease_complete(
            device_id, lease_id,
            {"completion_id": completion_id, "outcome": outcome})

    def _expire_lease(self, lease_id, seconds_ago=5) -> None:
        """Rewrite the lease's stored deadline to the past in memory."""
        claim = datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)
        claim_s = claim.isoformat(timespec="microseconds")
        for state in self.service.store._delivery.values():
            for lease in state.leases:
                if lease.lease_id == lease_id:
                    lease.leased_until = claim_s


class CompleteServiceTest(CompleteMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def test_first_complete_201_key_order_and_timestamp(self) -> None:
        self._claim("L1", 2)
        body, status = self._complete("bob", "L1", "C1")
        self.assertEqual(status, 201)
        self.assertEqual(list(body),
                         ["device_id", "lease_id", "completion_id",
                          "outcome", "completed_at"])
        self.assertEqual(body["device_id"], "bob")
        self.assertEqual(body["lease_id"], "L1")
        self.assertEqual(body["completion_id"], "C1")
        self.assertEqual(body["outcome"], "delivered")
        self.assertRegex(body["completed_at"], r"\.\d{6}\+00:00$")

    def test_failed_outcome_is_accepted(self) -> None:
        self._claim("L1", 2)
        body, status = self._complete("bob", "L1", "C1", outcome="failed")
        self.assertEqual(status, 201)
        self.assertEqual(body["outcome"], "failed")

    def test_complete_stamps_every_copy_of_the_lease(self) -> None:
        self._claim("L1", 2)
        body, _ = self._complete("bob", "L1", "C1")
        copies = [lease for state in self.service.store._delivery.values()
                  for lease in state.leases if lease.lease_id == "L1"]
        self.assertEqual(len(copies), 2)
        for lease in copies:
            self.assertIsNotNone(lease.completion)
            self.assertEqual(lease.completion.completion_id, "C1")
            self.assertEqual(lease.completion.outcome, "delivered")
            self.assertEqual(lease.completion.completed_at,
                             body["completed_at"])

    def test_replay_same_completion_id_returns_first_response_200(
            self) -> None:
        self._claim("L1", 2)
        first, first_status = self._complete("bob", "L1", "C1")
        self.assertEqual(first_status, 201)
        replay, status = self._complete("bob", "L1", "C1")
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)

    def test_replay_wins_after_expiry(self) -> None:
        self._claim("L1", 2)
        first, _ = self._complete("bob", "L1", "C1")
        self._expire_lease("L1")
        replay, status = self._complete("bob", "L1", "C1")
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)

    def test_replay_wins_after_device_revoked(self) -> None:
        self._claim("L1", 2)
        first, _ = self._complete("bob", "L1", "C1")
        self.service.revoke_device("bob")
        replay, status = self._complete("bob", "L1", "C1")
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)

    def test_different_completion_id_is_409_completion_id(self) -> None:
        self._claim("L1", 2)
        self._complete("bob", "L1", "C1")
        with self.assertRaises(ServiceError) as caught:
            self._complete("bob", "L1", "C2")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "completion_id")

    def test_completion_id_is_reusable_across_leases(self) -> None:
        self._claim("L1", 2)
        self._complete("bob", "L1", "SHARED")
        self._claim("L2", 2)
        body, status = self._complete("bob", "L2", "SHARED")
        self.assertEqual(status, 201)
        self.assertEqual(body["completion_id"], "SHARED")

    def test_unknown_lease_is_404_lease_id(self) -> None:
        with self.assertRaises(ServiceError) as caught:
            self._complete("bob", "NOPE", "C1")
        self.assertEqual(caught.exception.status_code, 404)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_unknown_lease_404_precedes_unknown_device(self) -> None:
        with self.assertRaises(ServiceError) as caught:
            self._complete("ghost", "NOPE", "C1")
        self.assertEqual(caught.exception.status_code, 404)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_cross_device_complete_is_409_lease_id(self) -> None:
        self._claim("L1", 2)
        with self.assertRaises(ServiceError) as caught:
            self._complete("carol", "L1", "C1")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_cross_device_conflict_precedes_path_device_state(self) -> None:
        self._claim("L1", 2)
        self.service.revoke_device("carol")
        with self.assertRaises(ServiceError) as caught:
            self._complete("carol", "L1", "C1")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_first_complete_on_revoked_device_is_409_device_id(self) -> None:
        self._claim("L1", 2)
        self.service.revoke_device("bob")
        with self.assertRaises(ServiceError) as caught:
            self._complete("bob", "L1", "C1")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "device_id")

    def test_complete_released_lease_is_409_lease_id(self) -> None:
        self._claim("L1", 2)
        self.service.inbox_release("bob", "L1")
        with self.assertRaises(ServiceError) as caught:
            self._complete("bob", "L1", "C1")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_complete_expired_lease_is_409_lease_id(self) -> None:
        self._claim("L1", 2)
        self._expire_lease("L1")
        with self.assertRaises(ServiceError) as caught:
            self._complete("bob", "L1", "C1")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_bad_body_shapes_are_400(self) -> None:
        self._claim("L1", 2)
        for payload in (None, "x", 7, ["C1"], True):
            with self.subTest(payload=payload):
                with self.assertRaises(ServiceError) as caught:
                    self.service.inbox_lease_complete("bob", "L1", payload)
                self.assertEqual(caught.exception.status_code, 400)
                self.assertEqual(caught.exception.field, "request_body")

    def test_bad_completion_id_is_400(self) -> None:
        self._claim("L1", 2)
        for payload in ({}, {"completion_id": ""}, {"completion_id": 7},
                        {"completion_id": None}, {"completion_id": True},
                        {"completion_id": ["C"]}):
            with self.subTest(payload=payload):
                with self.assertRaises(ServiceError) as caught:
                    self.service.inbox_lease_complete("bob", "L1", payload)
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
                    self.service.inbox_lease_complete("bob", "L1", payload)
                self.assertEqual(caught.exception.status_code, 400)
                self.assertEqual(caught.exception.field, "outcome")

    def test_completed_lease_cannot_be_renewed(self) -> None:
        self._claim("L1", 2)
        self._complete("bob", "L1", "C1")
        with self.assertRaises(ServiceError) as caught:
            self.service.inbox_lease_renew("bob", "L1", {"renewal_id": "R1"})
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_completed_lease_cannot_be_released(self) -> None:
        self._claim("L1", 2)
        self._complete("bob", "L1", "C1")
        with self.assertRaises(ServiceError) as caught:
            self.service.inbox_release("bob", "L1")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_completion_frees_unacked_messages_for_new_claims(self) -> None:
        # delivered is a delivery report, not an acknowledgement: the
        # completed lease stops withholding its still-unacked messages.
        claim = self._claim("L1", 2)
        self.assertEqual([m["message_id"] for m in claim["messages"]],
                         ["a1", "a2"])
        self._complete("bob", "L1", "C1")
        body, status = self.service.inbox_claim(
            "bob", {"lease_id": "L2", "limit": 10})
        self.assertEqual(status, 201)
        self.assertEqual([m["message_id"] for m in body["messages"]],
                         ["a1", "a2", "a3", "b1", "b2"])

    def test_replayed_claim_after_completion_reactivates_nothing(
            self) -> None:
        # Replaying the original claim still returns its frozen first
        # response; the completion stays settled regardless.
        self._claim("L1", 2)
        self._complete("bob", "L1", "C1")
        replay, status = self.service.inbox_claim(
            "bob", {"lease_id": "L1", "limit": 2})
        self.assertEqual(status, 200)
        body, _ = self.service.inbox_claim(
            "bob", {"lease_id": "L2", "limit": 10})
        self.assertIn("a1", [m["message_id"] for m in body["messages"]])


class CompletePersistenceTest(CompleteMixin, unittest.TestCase):
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
        body, status = self._complete("bob", "L1", "C1")
        self.assertEqual(status, 201)
        self.assertEqual(self.state_store.commit_seq, before + 1)

        leased = [record for record in self._document()["delivery"]
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
        first, _ = self._complete("bob", "L1", "C1")
        generation = self.state_store.commit_seq
        replay, status = self._complete("bob", "L1", "C1")
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)
        self.assertEqual(self.state_store.commit_seq, generation)

    def test_restart_restores_completion_and_replays(self) -> None:
        self._claim("L1", 2)
        first, _ = self._complete("bob", "L1", "C1")
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        replay, status = restarted.inbox_lease_complete(
            "bob", "L1", {"completion_id": "C1", "outcome": "delivered"})
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)
        # The lease stayed terminal across the restart.
        with self.assertRaises(ServiceError) as caught:
            restarted.inbox_lease_renew("bob", "L1", {"renewal_id": "R1"})
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")
        with self.assertRaises(ServiceError) as caught:
            restarted.inbox_release("bob", "L1")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")
        # And its unacked messages are claimable again.
        body, status = restarted.inbox_claim(
            "bob", {"lease_id": "L2", "limit": 10})
        self.assertEqual(status, 201)
        self.assertEqual([m["message_id"] for m in body["messages"]],
                         ["a1", "a2", "a3", "b1", "b2"])

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
                self._complete("bob", "L1", "C1")
        finally:
            persistence_mod.os.fsync = real_fsync
        self.assertEqual(self.state_store.commit_seq, generation)
        # The completion did not land in memory: C1 is a fresh 201.
        body, status = self._complete("bob", "L1", "C1")
        self.assertEqual(status, 201)
        self.assertEqual(self.state_store.commit_seq, generation + 1)

    def test_legacy_lease_without_completion_loads(self) -> None:
        self._claim("L1", 2)
        document = self._document()
        for record in document["delivery"]:
            for lease in record.get("leases", []):
                lease.pop("completion", None)
        document.pop("integrity_log_version", None)
        legacy_path = os.path.join(self.directory, "legacy.json")
        with open(legacy_path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        restarted = DeviceService()
        attach_persistence(restarted, legacy_path)  # must not raise
        body, status = restarted.inbox_lease_complete(
            "bob", "L1", {"completion_id": "C1", "outcome": "delivered"})
        self.assertEqual(status, 201)
        self.assertRegex(body["completed_at"], r"\.\d{6}\+00:00$")

    def _malformed_document(self, mutate):
        self._claim_counter = getattr(self, "_claim_counter", 0) + 1
        # Completed leases free their messages, so each probe can claim
        # afresh under a new lease id.
        lease_id = f"L1-{self._claim_counter}"
        self._claim(lease_id, 2)
        self._complete("bob", lease_id, "C1")
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

    def test_restore_rejects_completion_not_an_object(self) -> None:
        for bad in (7, "C1", ["C1"], True):
            with self.subTest(bad=bad):
                def mutate(document):
                    for lease in (l for r in document["delivery"]
                                  for l in r.get("leases", [])):
                        lease["completion"] = bad
                self._assert_refuses_startup(
                    self._malformed_document(mutate))

    def test_restore_rejects_missing_or_empty_completion_id(self) -> None:
        def missing(document):
            for lease in (l for r in document["delivery"]
                          for l in r.get("leases", [])):
                del lease["completion"]["completion_id"]

        def empty(document):
            for lease in (l for r in document["delivery"]
                          for l in r.get("leases", [])):
                lease["completion"]["completion_id"] = ""
        self._assert_refuses_startup(self._malformed_document(missing))
        self._assert_refuses_startup(self._malformed_document(empty))

    def test_restore_rejects_bad_outcome(self) -> None:
        for bad in ("acked", "DELIVERED", "", 7, None):
            with self.subTest(bad=bad):
                def mutate(document):
                    for lease in (l for r in document["delivery"]
                                  for l in r.get("leases", [])):
                        lease["completion"]["outcome"] = bad
                self._assert_refuses_startup(
                    self._malformed_document(mutate))

    def test_restore_rejects_malformed_completed_at(self) -> None:
        for bad in ("not-a-timestamp",
                    "2026-09-25T14:00:00+00:00",        # no microseconds
                    "2026-09-25T14:00:00.000000",       # no offset
                    "2026-09-25T14:00:00.000000+01:00"):
            with self.subTest(bad=bad):
                def mutate(document):
                    for lease in (l for r in document["delivery"]
                                  for l in r.get("leases", [])):
                        lease["completion"]["completed_at"] = bad
                self._assert_refuses_startup(
                    self._malformed_document(mutate))

    def test_restore_rejects_inconsistent_completion_across_records(
            self) -> None:
        def mutate(document):
            leased = [r for r in document["delivery"] if r.get("leases")]
            self.assertEqual(len(leased), 2)
            leased[1]["leases"][0]["completion"]["completion_id"] = "OTHER"
        self._assert_refuses_startup(self._malformed_document(mutate))


class CompleteHTTPTest(CompleteMixin, unittest.TestCase):
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
            f"/v1/devices/{device_id}/inbox/leases/{lease_id}/complete",
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

    def test_complete_and_replay_over_http(self) -> None:
        self._claim("L1", 2)
        status, body, raw = self._request(
            "bob", "L1", json.dumps(
                {"completion_id": "C1", "outcome": "delivered"}))
        self.assertEqual(status, 201)
        self.assertEqual(list(body),
                         ["device_id", "lease_id", "completion_id",
                          "outcome", "completed_at"])
        self.assertLess(raw.index('"device_id"'), raw.index('"lease_id"'))
        self.assertLess(raw.index('"lease_id"'),
                        raw.index('"completion_id"'))
        self.assertLess(raw.index('"completion_id"'), raw.index('"outcome"'))
        self.assertLess(raw.index('"outcome"'), raw.index('"completed_at"'))
        status, replay, replay_raw = self._request(
            "bob", "L1", json.dumps(
                {"completion_id": "C1", "outcome": "delivered"}))
        self.assertEqual(status, 200)
        self.assertEqual(replay_raw, raw)

    def test_bodies_over_http(self) -> None:
        self._claim("L1", 2)
        for raw in ("not json{", "[]", "7"):
            with self.subTest(raw=raw):
                status, body, _ = self._request("bob", "L1", raw)
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], "request_body")
        for raw in (json.dumps({}), json.dumps({"completion_id": ""}),
                    json.dumps({"completion_id": 5})):
            with self.subTest(raw=raw):
                status, body, _ = self._request("bob", "L1", raw)
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], "completion_id")
        for raw in (json.dumps({"completion_id": "C1"}),
                    json.dumps({"completion_id": "C1", "outcome": "acked"})):
            with self.subTest(raw=raw):
                status, body, _ = self._request("bob", "L1", raw)
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], "outcome")

    def test_errors_over_http(self) -> None:
        self._claim("L1", 2)
        payload = json.dumps({"completion_id": "C1", "outcome": "delivered"})
        status, body, _ = self._request("bob", "NOPE", payload)
        self.assertEqual((status, body["field"]), (404, "lease_id"))
        status, body, _ = self._request("carol", "L1", payload)
        self.assertEqual((status, body["field"]), (409, "lease_id"))
        self.service.store.revoke_device("bob")
        status, body, _ = self._request("bob", "L1", payload)
        self.assertEqual((status, body["field"]), (409, "device_id"))

    def test_percent_encoded_lease_id_over_http(self) -> None:
        self._claim("a/b", 1)
        status, body, _ = self._request(
            "bob", "a%2Fb", json.dumps(
                {"completion_id": "C1", "outcome": "failed"}))
        self.assertEqual(status, 201)
        self.assertEqual(body["lease_id"], "a/b")
        self.assertEqual(body["outcome"], "failed")


if __name__ == "__main__":
    unittest.main()
