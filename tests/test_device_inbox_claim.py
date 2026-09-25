"""Tests for the 1:1 inbox redelivery lease endpoint.

POST /v1/devices/{device_id}/inbox/claim leases up to ``limit`` unacked
1:1-inbox messages (those with no lease or an expired one) for 30 seconds,
under the store lock shared with submission/ack/retry/revocation. A non-empty
claim returns 201 with a UTC ISO-8601 deadline; an empty selection returns
200 with ``leased_until`` null and writes nothing. The client-chosen
``lease_id`` is idempotent only for the same device and limit.
"""
import json
import os
import shutil
import tempfile
import threading
import unittest
from datetime import datetime
from http.client import HTTPConnection

from e2ee_backend.models import Device, SignedPreKey
from e2ee_backend.persistence import StateFileError, attach_persistence
from e2ee_backend.service import DeviceService, ServiceError
from e2ee_backend.http_app import create_server


class LeaseMixin:
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
        # A group session including bob never contributes to the claim.
        self.service.create_group({
            "group_id": "g1", "creator_device_id": "alice",
            "member_device_ids": ["bob"]})
        self.group_sid = self.service.create_group_session({
            "group_id": "g1", "initiator_device_id": "alice",
            "ephemeral_key": "epk"})["session_id"]
        self.service.post_message({
            "session_id": self.group_sid, "sender_device_id": "alice",
            "message_id": "g1", "sequence": 1,
            "nonce": "ng1", "ciphertext": "ct"})

    def _expire_all_leases(self, service=None) -> None:
        store = (service or self.service).store
        for state in store._delivery.values():
            for lease in state.leases:
                lease.leased_until = "2000-01-01T00:00:00.000000+00:00"


class LeaseServiceTest(LeaseMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def test_claims_in_inbox_order_up_to_limit(self) -> None:
        body, status = self.service.inbox_claim(
            "bob", {"lease_id": "L1", "limit": 2})
        self.assertEqual(status, 201)
        self.assertEqual(list(body),
                         ["device_id", "lease_id", "leased_until",
                          "messages"])
        self.assertEqual(body["device_id"], "bob")
        self.assertEqual(body["lease_id"], "L1")
        self.assertEqual([m["message_id"] for m in body["messages"]],
                         ["a1", "a2"])
        for message in body["messages"]:
            self.assertEqual(list(message), [
                "session_id", "sender_device_id", "message_id", "sequence",
                "nonce", "ciphertext", "created_at"])
        # The deadline parses, is UTC, carries six microsecond digits and is
        # roughly 30 seconds in the future.
        deadline = body["leased_until"]
        self.assertRegex(deadline, r"\.\d{6}\+00:00$")
        parsed = datetime.fromisoformat(deadline)
        self.assertEqual(parsed.utcoffset().total_seconds(), 0)
        delta = parsed - datetime.now(parsed.tzinfo)
        self.assertAlmostEqual(delta.total_seconds(), 30, delta=2)

    def test_second_claim_skips_active_leased_messages(self) -> None:
        first, _ = self.service.inbox_claim(
            "bob", {"lease_id": "L1", "limit": 2})
        second, status = self.service.inbox_claim(
            "bob", {"lease_id": "L2", "limit": 10})
        self.assertEqual(status, 201)
        # a1/a2 held by L1, so the next in order are a3, b1, b2.
        self.assertEqual([m["message_id"] for m in second["messages"]],
                         ["a3", "b1", "b2"])
        # All five are now leased, so another claim returns the empty 200.
        third, status = self.service.inbox_claim(
            "bob", {"lease_id": "L3", "limit": 10})
        self.assertEqual(status, 200)
        self.assertEqual(third["messages"], [])
        self.assertIsNone(third["leased_until"])
        self.assertEqual(third["lease_id"], "L3")

    def test_limit_caps_even_with_free_messages(self) -> None:
        body, _ = self.service.inbox_claim(
            "bob", {"lease_id": "L1", "limit": 1})
        self.assertEqual([m["message_id"] for m in body["messages"]], ["a1"])

    def test_exact_replay_returns_first_response_200(self) -> None:
        first, first_status = self.service.inbox_claim(
            "bob", {"lease_id": "L1", "limit": 2})
        replay, replay_status = self.service.inbox_claim(
            "bob", {"lease_id": "L1", "limit": 2})
        self.assertEqual(first_status, 201)
        self.assertEqual(replay_status, 200)
        self.assertEqual(replay, first)

    def test_replay_identical_after_messages_acked(self) -> None:
        first, _ = self.service.inbox_claim(
            "bob", {"lease_id": "L1", "limit": 2})
        # Acknowledging the leased messages does not alter the frozen replay.
        self.service.sync_device_ack_batch("bob", {
            "items": [{"session_id": self.sid1, "cursor": 2}]})
        replay, status = self.service.inbox_claim(
            "bob", {"lease_id": "L1", "limit": 2})
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)

    def test_replay_succeeds_after_device_revoked(self) -> None:
        first, _ = self.service.inbox_claim(
            "bob", {"lease_id": "L1", "limit": 2})
        self.service.revoke_device("bob")
        replay, status = self.service.inbox_claim(
            "bob", {"lease_id": "L1", "limit": 2})
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)

    def test_changed_limit_is_409_lease_id(self) -> None:
        self.service.inbox_claim("bob", {"lease_id": "L1", "limit": 2})
        with self.assertRaises(ServiceError) as caught:
            self.service.inbox_claim("bob", {"lease_id": "L1", "limit": 3})
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_cross_device_reuse_is_409_lease_id(self) -> None:
        self.service.inbox_claim("bob", {"lease_id": "L1", "limit": 2})
        with self.assertRaises(ServiceError) as caught:
            self.service.inbox_claim("carol", {"lease_id": "L1", "limit": 2})
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_cross_device_conflict_takes_precedence_over_unknown_device(
            self) -> None:
        self.service.inbox_claim("bob", {"lease_id": "L1", "limit": 2})
        with self.assertRaises(ServiceError) as caught:
            self.service.inbox_claim("ghost", {"lease_id": "L1", "limit": 2})
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_expired_leases_free_messages_for_new_id(self) -> None:
        first, _ = self.service.inbox_claim(
            "bob", {"lease_id": "L1", "limit": 10})
        self.assertEqual(len(first["messages"]), 5)
        # While active nothing is available.
        empty, status = self.service.inbox_claim(
            "bob", {"lease_id": "L2", "limit": 10})
        self.assertEqual(status, 200)
        self.assertEqual(empty["messages"], [])
        # Once every L1 lease expires, a new id claims them all again.
        self._expire_all_leases()
        again, status = self.service.inbox_claim(
            "bob", {"lease_id": "L3", "limit": 10})
        self.assertEqual(status, 201)
        self.assertEqual([m["message_id"] for m in again["messages"]],
                         ["a1", "a2", "a3", "b1", "b2"])
        # A fresh deadline was assigned.
        self.assertNotEqual(again["leased_until"], first["leased_until"])

    def test_expired_lease_does_not_block_but_other_active_lease_does(self) -> None:
        # Lease a1/a2 under L1, then a3/b1/b2 under L2.
        self.service.inbox_claim("bob", {"lease_id": "L1", "limit": 2})
        self.service.inbox_claim("bob", {"lease_id": "L2", "limit": 10})
        # Expire only L1's leases; a1/a2 are claimable, L2's stay held.
        for state in self.service.store._delivery.values():
            for lease in state.leases:
                if lease.lease_id == "L1":
                    lease.leased_until = "2000-01-01T00:00:00.000000+00:00"
        body, status = self.service.inbox_claim(
            "bob", {"lease_id": "L3", "limit": 10})
        self.assertEqual(status, 201)
        self.assertEqual([m["message_id"] for m in body["messages"]],
                         ["a1", "a2"])

    def test_empty_claim_does_not_occupy_lease_id(self) -> None:
        # Lease everything first.
        self.service.inbox_claim("bob", {"lease_id": "L1", "limit": 10})
        empty, status = self.service.inbox_claim(
            "bob", {"lease_id": "FREE", "limit": 10})
        self.assertEqual(status, 200)
        self.assertEqual(empty["messages"], [])
        # The id was not occupied: it can later be used by another device
        # without a conflict.
        body, status = self.service.inbox_claim(
            "carol", {"lease_id": "FREE", "limit": 10})
        self.assertEqual(status, 200)  # carol also has an empty inbox

    def test_device_unknown_or_revoked(self) -> None:
        with self.assertRaises(ServiceError) as caught:
            self.service.inbox_claim("ghost", {"lease_id": "Z", "limit": 2})
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "device_id")
        self.service.revoke_device("bob")
        with self.assertRaises(ServiceError) as caught:
            self.service.inbox_claim("bob", {"lease_id": "Z", "limit": 2})
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "device_id")

    def test_body_validation(self) -> None:
        cases = [
            (None, "request_body"),
            ("x", "request_body"),
            ([1, 2], "request_body"),
            (42, "request_body"),
            ({}, "lease_id"),
            ({"limit": 2}, "lease_id"),
            ({"lease_id": "", "limit": 2}, "lease_id"),
            ({"lease_id": 5, "limit": 2}, "lease_id"),
            ({"lease_id": None, "limit": 2}, "lease_id"),
            ({"lease_id": "L"}, "limit"),
            ({"lease_id": "L", "limit": True}, "limit"),
            ({"lease_id": "L", "limit": False}, "limit"),
            ({"lease_id": "L", "limit": 0}, "limit"),
            ({"lease_id": "L", "limit": 101}, "limit"),
            ({"lease_id": "L", "limit": -1}, "limit"),
            ({"lease_id": "L", "limit": 1.5}, "limit"),
            ({"lease_id": "L", "limit": "2"}, "limit"),
            ({"lease_id": "L", "limit": None}, "limit"),
        ]
        for payload, field in cases:
            with self.subTest(payload=payload):
                with self.assertRaises(ServiceError) as caught:
                    self.service.inbox_claim("bob", payload)
                self.assertEqual(caught.exception.status_code, 400)
                self.assertEqual(caught.exception.field, field)

    def test_claim_linearizes_with_ack(self) -> None:
        # Ack a1/a2 first; the claim then starts at a3.
        self.service.sync_device_ack_batch("bob", {
            "items": [{"session_id": self.sid1, "cursor": 2}]})
        body, status = self.service.inbox_claim(
            "bob", {"lease_id": "L1", "limit": 10})
        self.assertEqual(status, 201)
        self.assertEqual([m["message_id"] for m in body["messages"]],
                         ["a3", "b1", "b2"])


class LeasePersistenceTest(LeaseMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self._build()
        self.path = os.path.join(self.directory, "state.json")
        self.state_store = attach_persistence(self.service, self.path)

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_nonempty_claim_persists_and_advances_one_generation(self) -> None:
        before = self.state_store.commit_seq
        _, status = self.service.inbox_claim(
            "bob", {"lease_id": "L1", "limit": 2})
        self.assertEqual(status, 201)
        self.assertEqual(self.state_store.commit_seq, before + 1)

        with open(self.path, encoding="utf-8") as handle:
            document = json.load(handle)
        leased = [record for record in document["delivery"]
                  if record.get("leases")]
        self.assertEqual(len(leased), 2)
        for record in leased:
            self.assertEqual(list(record), [
                "session_id", "message_id", "attempts", "attempt_ids",
                "acked", "ack_sequence", "leases"])
            self.assertEqual(len(record["leases"]), 1)
            self.assertEqual(list(record["leases"][0]),
                             ["lease_id", "limit", "leased_until",
                              "released_at"])
            self.assertEqual(record["leases"][0]["lease_id"], "L1")
            self.assertEqual(record["leases"][0]["limit"], 2)
            self.assertIsNone(record["leases"][0]["released_at"])

    def test_empty_claim_persists_nothing_and_advances_no_generation(self) -> None:
        self.service.inbox_claim("bob", {"lease_id": "L1", "limit": 10})
        generation = self.state_store.commit_seq
        _, status = self.service.inbox_claim(
            "bob", {"lease_id": "L2", "limit": 10})
        self.assertEqual(status, 200)
        self.assertEqual(self.state_store.commit_seq, generation)

    def test_exact_replay_persists_nothing(self) -> None:
        first, _ = self.service.inbox_claim(
            "bob", {"lease_id": "L1", "limit": 2})
        generation = self.state_store.commit_seq
        replay, status = self.service.inbox_claim(
            "bob", {"lease_id": "L1", "limit": 2})
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)
        self.assertEqual(self.state_store.commit_seq, generation)

    def test_restart_restores_leases_and_replays(self) -> None:
        first, _ = self.service.inbox_claim(
            "bob", {"lease_id": "L1", "limit": 2})
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        # The still-active L1 leases withhold a1/a2 from a fresh claim.
        second, status = restarted.inbox_claim(
            "bob", {"lease_id": "L2", "limit": 10})
        self.assertEqual(status, 201)
        self.assertEqual([m["message_id"] for m in second["messages"]],
                         ["a3", "b1", "b2"])
        # The original id replays byte-identically after the restart.
        replay, status = restarted.inbox_claim(
            "bob", {"lease_id": "L1", "limit": 2})
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)
        # A changed limit / another device still conflict.
        with self.assertRaises(ServiceError):
            restarted.inbox_claim("bob", {"lease_id": "L1", "limit": 3})
        with self.assertRaises(ServiceError):
            restarted.inbox_claim("carol", {"lease_id": "L1", "limit": 2})

    def test_legacy_delivery_record_without_leases_loads(self) -> None:
        # Strip the leases field from a persisted record: an older
        # version-1 writer would never have emitted it, and it must load.
        # Written to a fresh path with the integrity marker removed (so it
        # is a genuine marker-less, sidecar-less legacy document).
        self.service.inbox_claim("bob", {"lease_id": "L1", "limit": 1})
        with open(self.path, encoding="utf-8") as handle:
            document = json.load(handle)
        for record in document["delivery"]:
            record.pop("leases", None)
        document.pop("integrity_log_version", None)
        legacy_path = os.path.join(self.directory, "legacy.json")
        with open(legacy_path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        restarted = DeviceService()
        attach_persistence(restarted, legacy_path)  # must not raise
        body, _ = restarted.inbox_claim(
            "bob", {"lease_id": "L2", "limit": 1})
        # With no restored lease on a1, it is claimable again.
        self.assertEqual([m["message_id"] for m in body["messages"]], ["a1"])

    def _malformed_document(self, mutate):
        self.service.inbox_claim("bob", {"lease_id": "L1", "limit": 2})
        with open(self.path, encoding="utf-8") as handle:
            document = json.load(handle)
        mutate(document)
        # Write to a fresh path as a marker-less/sidecar-less document, so
        # rejection comes from the payload's own semantic validation rather
        # than the integrity hash gate; the original state file is untouched.
        document.pop("integrity_log_version", None)
        bad_path = os.path.join(self.directory, "bad.json")
        with open(bad_path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        return bad_path

    def _assert_refuses_startup(self, bad_path) -> None:
        with open(bad_path, "rb") as handle:
            original = handle.read()
        restarted = DeviceService()
        with self.assertRaises(StateFileError):
            attach_persistence(restarted, bad_path)
        # The rejected file is never overwritten.
        with open(bad_path, "rb") as handle:
            self.assertEqual(handle.read(), original)

    def test_restore_rejects_bad_lease_types(self) -> None:
        def mutate(document):
            document["delivery"][0]["leases"][0]["limit"] = True
        bad_path = self._malformed_document(mutate)
        self._assert_refuses_startup(bad_path)

    def test_restore_rejects_non_string_lease_id(self) -> None:
        def mutate(document):
            document["delivery"][0]["leases"][0]["lease_id"] = 7
        bad_path = self._malformed_document(mutate)
        self._assert_refuses_startup(bad_path)

    def test_restore_rejects_out_of_range_limit(self) -> None:
        def mutate(document):
            document["delivery"][0]["leases"][0]["limit"] = 0
        bad_path = self._malformed_document(mutate)
        self._assert_refuses_startup(bad_path)

    def test_restore_rejects_missing_deadline(self) -> None:
        def mutate(document):
            del document["delivery"][0]["leases"][0]["leased_until"]
        bad_path = self._malformed_document(mutate)
        self._assert_refuses_startup(bad_path)

    def test_restore_rejects_leases_not_a_list(self) -> None:
        def mutate(document):
            document["delivery"][0]["leases"] = {}
        bad_path = self._malformed_document(mutate)
        self._assert_refuses_startup(bad_path)

    def test_restore_rejects_same_id_with_changed_limit(self) -> None:
        # L1 is recorded on two records (limit 2); change one copy so the
        # same id binds two different limits.
        def mutate(document):
            leased = [r for r in document["delivery"] if r.get("leases")]
            self.assertEqual(len(leased), 2)
            leased[1]["leases"][0]["limit"] = 3
        bad_path = self._malformed_document(mutate)
        self._assert_refuses_startup(bad_path)

    def test_restore_rejects_same_id_across_devices(self) -> None:
        # Build a second device lease with the same lease_id L1 by hand:
        # carol has a 1:1 session addressed to her (bob -> carol).
        sid_to_carol = self.service.store.create_session(
            "bob", "carol", "pkC", "ekC").session_id
        self.service.post_message({
            "session_id": sid_to_carol, "sender_device_id": "bob",
            "message_id": "c1", "sequence": 1,
            "nonce": "nc1", "ciphertext": "ct"})
        self.service.inbox_claim("carol", {"lease_id": "L9", "limit": 1})

        def mutate(document):
            record = next(
                r for r in document["delivery"]
                if r["session_id"] == sid_to_carol)
            record["leases"][0]["lease_id"] = "L1"
        bad_path = self._malformed_document(mutate)
        self._assert_refuses_startup(bad_path)


class LeaseHTTPTest(LeaseMixin, unittest.TestCase):
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

    def _request(self, device_id, raw):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("POST", f"/v1/devices/{device_id}/inbox/claim",
                     body=raw,
                     headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        data = response.read().decode("utf-8")
        return response.status, (json.loads(data) if data else None), data

    def test_claim_and_replay_over_http(self) -> None:
        status, body, raw = self._request(
            "bob", json.dumps({"lease_id": "L1", "limit": 2}))
        self.assertEqual(status, 201)
        self.assertEqual(list(body),
                         ["device_id", "lease_id", "leased_until",
                          "messages"])
        # Key order is also correct in the serialized bytes.
        self.assertLess(raw.index('"device_id"'), raw.index('"lease_id"'))
        self.assertLess(raw.index('"lease_id"'),
                        raw.index('"leased_until"'))
        self.assertLess(raw.index('"leased_until"'),
                        raw.index('"messages"'))
        status, replay, _ = self._request(
            "bob", json.dumps({"lease_id": "L1", "limit": 2}))
        self.assertEqual(status, 200)
        self.assertEqual(replay, body)

    def test_empty_response_serializes_null(self) -> None:
        self._request("bob", json.dumps({"lease_id": "L1", "limit": 10}))
        status, body, raw = self._request(
            "bob", json.dumps({"lease_id": "L2", "limit": 10}))
        self.assertEqual(status, 200)
        self.assertEqual(body["messages"], [])
        self.assertIsNone(body["leased_until"])
        self.assertIn('"leased_until":null', raw.replace(" ", ""))

    def test_bad_json_and_non_object_body(self) -> None:
        for raw in ("{not json", json.dumps([1, 2]), ""):
            status, body, _ = self._request("bob", raw)
            self.assertEqual(status, 400, raw)
            self.assertEqual(list(body), ["message", "field"])
            self.assertEqual(body["field"], "request_body")

    def test_field_errors_over_http(self) -> None:
        cases = [
            (json.dumps({"limit": 2}), "lease_id"),
            (json.dumps({"lease_id": "L", "limit": True}), "limit"),
            (json.dumps({"lease_id": "L", "limit": 101}), "limit"),
        ]
        for raw, field in cases:
            status, body, _ = self._request("bob", raw)
            self.assertEqual(status, 400, raw)
            self.assertEqual(body["field"], field)

    def test_device_errors_over_http(self) -> None:
        status, body, _ = self._request(
            "ghost", json.dumps({"lease_id": "Z", "limit": 2}))
        self.assertEqual(status, 409)
        self.assertEqual(list(body), ["message", "field"])
        self.assertEqual(body["field"], "device_id")
        self.service.store.revoke_device("bob")
        status, body, _ = self._request(
            "bob", json.dumps({"lease_id": "Z", "limit": 2}))
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "device_id")

    def test_conflict_over_http(self) -> None:
        self._request("bob", json.dumps({"lease_id": "L1", "limit": 2}))
        status, body, _ = self._request(
            "bob", json.dumps({"lease_id": "L1", "limit": 3}))
        self.assertEqual(status, 409)
        self.assertEqual(list(body), ["message", "field"])
        self.assertEqual(body["field"], "lease_id")
        status, body, _ = self._request(
            "carol", json.dumps({"lease_id": "L1", "limit": 2}))
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "lease_id")


if __name__ == "__main__":
    unittest.main()
