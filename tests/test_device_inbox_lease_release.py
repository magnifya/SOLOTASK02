"""Tests for the 1:1 inbox lease release endpoint.

POST /v1/devices/{device_id}/inbox/leases/{lease_id}/release releases an
occupied inbox lease before its deadline so its messages can be claimed
again. The endpoint takes no request body. An unknown lease id is
404/lease_id, a lease owned by another device is 409/lease_id, and a first
release on a revoked device is 409/device_id. A first release returns 201
with ``released_at``/``released_count`` and persists one generation; a
repeat returns the first response byte-identically with 200, even after
the device is revoked.
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
from e2ee_backend.persistence import (
    PersistenceUnavailable, StateFileError, attach_persistence)
from e2ee_backend.service import DeviceService, ServiceError
from e2ee_backend.http_app import create_server


class ReleaseMixin:
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


class ReleaseServiceTest(ReleaseMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def test_first_release_201_response_shape(self) -> None:
        self._claim("L1", 2)
        body, status = self.service.inbox_release("bob", "L1")
        self.assertEqual(status, 201)
        self.assertEqual(list(body),
                         ["device_id", "lease_id", "released_at",
                          "released_count"])
        self.assertEqual(body["device_id"], "bob")
        self.assertEqual(body["lease_id"], "L1")
        self.assertEqual(body["released_count"], 2)
        self.assertIsInstance(body["released_count"], int)
        stamp = body["released_at"]
        self.assertRegex(stamp, r"\.\d{6}\+00:00$")
        parsed = datetime.fromisoformat(stamp)
        self.assertEqual(parsed.utcoffset().total_seconds(), 0)

    def test_released_count_matches_claimed_messages(self) -> None:
        self._claim("L1", 10)
        body, _ = self.service.inbox_release("bob", "L1")
        self.assertEqual(body["released_count"], 5)

    def test_repeat_release_replays_first_response_200(self) -> None:
        self._claim("L1", 2)
        first, first_status = self.service.inbox_release("bob", "L1")
        self.assertEqual(first_status, 201)
        replay, status = self.service.inbox_release("bob", "L1")
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)

    def test_repeat_release_succeeds_after_device_revoked(self) -> None:
        self._claim("L1", 2)
        first, _ = self.service.inbox_release("bob", "L1")
        self.service.revoke_device("bob")
        replay, status = self.service.inbox_release("bob", "L1")
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)

    def test_unknown_lease_is_404_lease_id(self) -> None:
        with self.assertRaises(ServiceError) as caught:
            self.service.inbox_release("bob", "NOPE")
        self.assertEqual(caught.exception.status_code, 404)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_unknown_lease_404_precedes_unknown_device(self) -> None:
        with self.assertRaises(ServiceError) as caught:
            self.service.inbox_release("ghost", "NOPE")
        self.assertEqual(caught.exception.status_code, 404)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_cross_device_release_is_409_lease_id(self) -> None:
        self._claim("L1", 2)
        with self.assertRaises(ServiceError) as caught:
            self.service.inbox_release("carol", "L1")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_cross_device_conflict_precedes_path_device_state(self) -> None:
        self._claim("L1", 2)
        self.service.revoke_device("carol")
        with self.assertRaises(ServiceError) as caught:
            self.service.inbox_release("carol", "L1")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_first_release_on_revoked_device_is_409_device_id(self) -> None:
        self._claim("L1", 2)
        self.service.revoke_device("bob")
        with self.assertRaises(ServiceError) as caught:
            self.service.inbox_release("bob", "L1")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "device_id")

    def test_release_frees_messages_before_deadline(self) -> None:
        self._claim("L1", 2)
        # While L1 is held, nothing else is claimable for a fresh id.
        empty, status = self.service.inbox_claim(
            "bob", {"lease_id": "L2", "limit": 1})
        # a1/a2 held; the next free message is a3.
        self.assertEqual([m["message_id"] for m in empty["messages"]],
                         ["a3"])
        self.service.inbox_release("bob", "L1")
        again, status = self.service.inbox_claim(
            "bob", {"lease_id": "L3", "limit": 10})
        self.assertEqual(status, 201)
        self.assertEqual([m["message_id"] for m in again["messages"]],
                         ["a1", "a2", "b1", "b2"])

    def test_claim_replay_after_release_stays_frozen_and_inactive(self) -> None:
        first_claim = self._claim("L1", 2)
        self.service.inbox_release("bob", "L1")
        # The original claim id replays its frozen first response...
        replay, status = self.service.inbox_claim(
            "bob", {"lease_id": "L1", "limit": 2})
        self.assertEqual(status, 200)
        self.assertEqual(replay, first_claim)
        # ...but does not reactivate the lease: a new id claims a1/a2.
        again, status = self.service.inbox_claim(
            "bob", {"lease_id": "L2", "limit": 10})
        self.assertEqual(status, 201)
        self.assertEqual([m["message_id"] for m in again["messages"]],
                         ["a1", "a2", "a3", "b1", "b2"])

    def test_release_after_expiry_still_works(self) -> None:
        self._claim("L1", 2)
        for state in self.service.store._delivery.values():
            for lease in state.leases:
                lease.leased_until = "2000-01-01T00:00:00.000000+00:00"
        body, status = self.service.inbox_release("bob", "L1")
        self.assertEqual(status, 201)
        self.assertEqual(body["released_count"], 2)

    def test_release_after_ack_still_works(self) -> None:
        self._claim("L1", 2)
        self.service.sync_device_ack_batch("bob", {
            "items": [{"session_id": self.sid1, "cursor": 2}]})
        body, status = self.service.inbox_release("bob", "L1")
        self.assertEqual(status, 201)
        self.assertEqual(body["released_count"], 2)


class ReleasePersistenceTest(ReleaseMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self._build()
        self.path = os.path.join(self.directory, "state.json")
        self.state_store = attach_persistence(self.service, self.path)

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_first_release_persists_and_advances_one_generation(self) -> None:
        self._claim("L1", 2)
        before = self.state_store.commit_seq
        body, status = self.service.inbox_release("bob", "L1")
        self.assertEqual(status, 201)
        self.assertEqual(self.state_store.commit_seq, before + 1)

        with open(self.path, encoding="utf-8") as handle:
            document = json.load(handle)
        leased = [record for record in document["delivery"]
                  if record.get("leases")]
        self.assertEqual(len(leased), 2)
        for record in leased:
            self.assertEqual(list(record["leases"][0]),
                             ["lease_id", "limit", "leased_until",
                              "released_at", "renewals"])
            self.assertEqual(record["leases"][0]["released_at"],
                             body["released_at"])
            self.assertEqual(record["leases"][0]["renewals"], [])

    def test_repeat_release_persists_nothing(self) -> None:
        self._claim("L1", 2)
        first, _ = self.service.inbox_release("bob", "L1")
        generation = self.state_store.commit_seq
        replay, status = self.service.inbox_release("bob", "L1")
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)
        self.assertEqual(self.state_store.commit_seq, generation)

    def test_restart_restores_release(self) -> None:
        self._claim("L1", 2)
        first, _ = self.service.inbox_release("bob", "L1")
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        # The release replays byte-identically after the restart.
        replay, status = restarted.inbox_release("bob", "L1")
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)
        # And the freed messages are claimable under a new id.
        again, status = restarted.inbox_claim(
            "bob", {"lease_id": "L2", "limit": 10})
        self.assertEqual(status, 201)
        self.assertEqual([m["message_id"] for m in again["messages"]],
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
                self.service.inbox_release("bob", "L1")
        finally:
            persistence_mod.os.fsync = real_fsync
        # Rolled back: no generation consumed and the lease is still held,
        # so a fresh claim cannot take a1/a2 and the release can be retried.
        self.assertEqual(self.state_store.commit_seq, generation)
        empty, _ = self.service.inbox_claim(
            "bob", {"lease_id": "L2", "limit": 2})
        self.assertEqual([m["message_id"] for m in empty["messages"]],
                         ["a3", "b1"])
        generation = self.state_store.commit_seq
        body, status = self.service.inbox_release("bob", "L1")
        self.assertEqual(status, 201)
        self.assertEqual(self.state_store.commit_seq, generation + 1)
        self.assertEqual(body["released_count"], 2)

    def test_legacy_lease_without_released_at_loads(self) -> None:
        # A version-1 lease written before the release feature carries no
        # released_at key; it must load as null (never released).
        self._claim("L1", 1)
        with open(self.path, encoding="utf-8") as handle:
            document = json.load(handle)
        for record in document["delivery"]:
            for lease in record.get("leases", []):
                lease.pop("released_at", None)
        document.pop("integrity_log_version", None)
        legacy_path = os.path.join(self.directory, "legacy.json")
        with open(legacy_path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        restarted = DeviceService()
        attach_persistence(restarted, legacy_path)  # must not raise
        # The restored lease is still held (a1 withheld) and releasable.
        body, _ = restarted.inbox_claim(
            "bob", {"lease_id": "L2", "limit": 1})
        self.assertEqual([m["message_id"] for m in body["messages"]], ["a2"])
        released, status = restarted.inbox_release("bob", "L1")
        self.assertEqual(status, 201)
        self.assertEqual(released["released_count"], 1)

    def _malformed_document(self, mutate):
        self._claim_counter = getattr(self, "_claim_counter", 0) + 1
        lease_id = f"L1-{self._claim_counter}"
        self._claim(lease_id, 2)
        self.service.inbox_release("bob", lease_id)
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

    def test_restore_rejects_non_string_released_at(self) -> None:
        def mutate(document):
            leased = [r for r in document["delivery"] if r.get("leases")]
            for record in leased:
                record["leases"][0]["released_at"] = 7
        self._assert_refuses_startup(self._malformed_document(mutate))

    def test_restore_rejects_malformed_released_at(self) -> None:
        for bad in ("not-a-timestamp",
                    "2026-09-25T14:00:00+00:00",        # no microseconds
                    "2026-09-25T14:00:00.000000",       # no offset
                    "2026-09-25T14:00:00.000000+01:00",  # not UTC
                    "2026-13-25T14:00:00.000000+00:00"):  # invalid month
            with self.subTest(bad=bad):
                def mutate(document):
                    leased = [r for r in document["delivery"]
                              if r.get("leases")]
                    for record in leased:
                        record["leases"][0]["released_at"] = bad
                self._assert_refuses_startup(self._malformed_document(mutate))

    def test_restore_rejects_inconsistent_released_at_across_records(
            self) -> None:
        # L1 is recorded on two records; release stamps one shared
        # released_at. Differing values across records must refuse startup.
        def mutate(document):
            leased = [r for r in document["delivery"] if r.get("leases")]
            self.assertEqual(len(leased), 2)
            leased[1]["leases"][0]["released_at"] = \
                "2000-01-01T00:00:00.000000+00:00"
        self._assert_refuses_startup(self._malformed_document(mutate))

    def test_restore_rejects_released_at_on_one_record_only(self) -> None:
        def mutate(document):
            leased = [r for r in document["delivery"] if r.get("leases")]
            self.assertEqual(len(leased), 2)
            leased[1]["leases"][0]["released_at"] = None
        self._assert_refuses_startup(self._malformed_document(mutate))


class ReleaseHTTPTest(ReleaseMixin, unittest.TestCase):
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

    def _request(self, device_id, lease_id, raw=None):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request(
            "POST", f"/v1/devices/{device_id}/inbox/leases/{lease_id}/release",
            body=raw)
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

    def test_release_and_replay_over_http(self) -> None:
        self._claim("L1", 2)
        status, body, raw = self._request("bob", "L1")
        self.assertEqual(status, 201)
        self.assertEqual(list(body),
                         ["device_id", "lease_id", "released_at",
                          "released_count"])
        # Key order is also correct in the serialized bytes.
        self.assertLess(raw.index('"device_id"'), raw.index('"lease_id"'))
        self.assertLess(raw.index('"lease_id"'),
                        raw.index('"released_at"'))
        self.assertLess(raw.index('"released_at"'),
                        raw.index('"released_count"'))
        status, replay, replay_raw = self._request("bob", "L1")
        self.assertEqual(status, 200)
        self.assertEqual(replay_raw, raw)

    def test_non_empty_body_is_400_request_body(self) -> None:
        self._claim("L1", 2)
        for raw in ("{}", "x", json.dumps({"lease_id": "L1"})):
            status, body, _ = self._request("bob", "L1", raw=raw)
            self.assertEqual(status, 400, raw)
            self.assertEqual(list(body), ["message", "field"])
            self.assertEqual(body["field"], "request_body")
        # The rejections consumed nothing: the release still works.
        status, body, _ = self._request("bob", "L1")
        self.assertEqual(status, 201)

    def test_unknown_lease_over_http(self) -> None:
        status, body, _ = self._request("bob", "NOPE")
        self.assertEqual(status, 404)
        self.assertEqual(list(body), ["message", "field"])
        self.assertEqual(body["field"], "lease_id")

    def test_cross_device_over_http(self) -> None:
        self._claim("L1", 2)
        status, body, _ = self._request("carol", "L1")
        self.assertEqual(status, 409)
        self.assertEqual(list(body), ["message", "field"])
        self.assertEqual(body["field"], "lease_id")

    def test_revoked_device_over_http(self) -> None:
        self._claim("L1", 2)
        self.service.store.revoke_device("bob")
        status, body, _ = self._request("bob", "L1")
        self.assertEqual(status, 409)
        self.assertEqual(list(body), ["message", "field"])
        self.assertEqual(body["field"], "device_id")

    def test_percent_encoded_ids_over_http(self) -> None:
        # A lease id containing a slash reaches the store percent-decoded.
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("POST", "/v1/devices/bob/inbox/claim",
                     body=json.dumps({"lease_id": "a/b", "limit": 1}),
                     headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        response.read()
        conn.close()
        self.assertEqual(response.status, 201)
        status, body, _ = self._request("bob", "a%2Fb")
        self.assertEqual(status, 201)
        self.assertEqual(body["lease_id"], "a/b")


if __name__ == "__main__":
    unittest.main()
