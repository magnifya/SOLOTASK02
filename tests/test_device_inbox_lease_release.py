"""Tests for the 1:1 inbox lease release endpoint.

POST /v1/devices/{device_id}/inbox/leases/{lease_id}/release releases an
occupied inbox lease ahead of its deadline so its messages can be claimed
again. The request carries no body (a non-empty one is 400/request_body).
An unknown lease id is 404/lease_id, an id owned by another device is
409/lease_id, and a first release on a revoked device is 409/device_id. A
first release returns 201 with ``released_at``/``released_count``; a repeat
returns the first response byte-identically with 200, even after the device
is revoked. A released lease no longer blocks new claims, while the
original claim id still replays its frozen first response.
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

    def _claim(self, lease_id="L1", limit=2, device_id="bob"):
        return self.service.inbox_claim(
            device_id, {"lease_id": lease_id, "limit": limit})


class ReleaseServiceTest(ReleaseMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def test_first_release_201_shape(self) -> None:
        self._claim("L1", 2)
        body, status = self.service.inbox_release_lease("bob", "L1")
        self.assertEqual(status, 201)
        self.assertEqual(list(body),
                         ["device_id", "lease_id", "released_at",
                          "released_count"])
        self.assertEqual(body["device_id"], "bob")
        self.assertEqual(body["lease_id"], "L1")
        self.assertEqual(body["released_count"], 2)
        # UTC ISO-8601 with six microsecond digits and the +00:00 offset.
        self.assertRegex(body["released_at"], r"\.\d{6}\+00:00$")
        parsed = datetime.fromisoformat(body["released_at"])
        self.assertEqual(parsed.utcoffset().total_seconds(), 0)

    def test_unknown_lease_is_404(self) -> None:
        with self.assertRaises(ServiceError) as caught:
            self.service.inbox_release_lease("bob", "nope")
        self.assertEqual(caught.exception.status_code, 404)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_empty_claim_occupies_no_id(self) -> None:
        # An empty claim keeps its id free, so a release of it is a 404.
        self._claim("L1", 10)  # leases everything, still active
        _, status = self.service.inbox_claim(
            "bob", {"lease_id": "L9", "limit": 10})
        self.assertEqual(status, 200)  # empty selection: id stays free
        with self.assertRaises(ServiceError) as caught:
            self.service.inbox_release_lease("bob", "L9")
        self.assertEqual(caught.exception.status_code, 404)

    def test_other_device_lease_is_409_lease_id(self) -> None:
        self._claim("L1", 2)
        with self.assertRaises(ServiceError) as caught:
            self.service.inbox_release_lease("carol", "L1")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_first_release_on_revoked_device_is_409_device_id(self) -> None:
        self._claim("L1", 2)
        self.service.store.revoke_device("bob")
        with self.assertRaises(ServiceError) as caught:
            self.service.inbox_release_lease("bob", "L1")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "device_id")

    def test_repeat_release_replays_first_response(self) -> None:
        self._claim("L1", 2)
        first, status = self.service.inbox_release_lease("bob", "L1")
        self.assertEqual(status, 201)
        again, status = self.service.inbox_release_lease("bob", "L1")
        self.assertEqual(status, 200)
        self.assertEqual(again, first)

    def test_repeat_release_allowed_after_revocation(self) -> None:
        self._claim("L1", 2)
        first, _ = self.service.inbox_release_lease("bob", "L1")
        self.service.store.revoke_device("bob")
        again, status = self.service.inbox_release_lease("bob", "L1")
        self.assertEqual(status, 200)
        self.assertEqual(again, first)

    def test_released_messages_are_claimable_before_deadline(self) -> None:
        self._claim("L1", 2)
        self.service.inbox_release_lease("bob", "L1")
        # Without waiting for the 30s deadline, a new id claims a1/a2 again.
        body, status = self.service.inbox_claim(
            "bob", {"lease_id": "L2", "limit": 10})
        self.assertEqual(status, 201)
        self.assertEqual([m["message_id"] for m in body["messages"]],
                         ["a1", "a2", "a3", "b1", "b2"])

    def test_original_claim_replay_does_not_reactivate(self) -> None:
        first_claim, _ = self._claim("L1", 2)
        self.service.inbox_release_lease("bob", "L1")
        # The original claim id replays its frozen first response...
        replay, status = self._claim("L1", 2)
        self.assertEqual(status, 200)
        self.assertEqual(replay, first_claim)
        # ...but the lease stays released: a new id still claims a1/a2.
        body, status = self.service.inbox_claim(
            "bob", {"lease_id": "L2", "limit": 10})
        self.assertEqual(status, 201)
        self.assertIn("a1", [m["message_id"] for m in body["messages"]])

    def test_released_count_tracks_claimed_messages(self) -> None:
        self._claim("L1", 1)
        body, _ = self.service.inbox_release_lease("bob", "L1")
        self.assertEqual(body["released_count"], 1)
        # a1 was released, so L2 claims all five messages.
        self._claim("L2", 10)
        body, _ = self.service.inbox_release_lease("bob", "L2")
        self.assertEqual(body["released_count"], 5)


class ReleasePersistenceTest(ReleaseMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self._build()
        self.path = os.path.join(self.directory, "state.json")
        self.state_store = attach_persistence(self.service, self.path)

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_first_release_advances_one_generation(self) -> None:
        self._claim("L1", 2)
        generation = self.state_store.commit_seq
        _, status = self.service.inbox_release_lease("bob", "L1")
        self.assertEqual(status, 201)
        self.assertEqual(self.state_store.commit_seq, generation + 1)

    def test_repeat_release_consumes_no_generation(self) -> None:
        self._claim("L1", 2)
        self.service.inbox_release_lease("bob", "L1")
        generation = self.state_store.commit_seq
        _, status = self.service.inbox_release_lease("bob", "L1")
        self.assertEqual(status, 200)
        self.assertEqual(self.state_store.commit_seq, generation)

    def test_release_persists_released_at_in_key_order(self) -> None:
        self._claim("L1", 2)
        self.service.inbox_release_lease("bob", "L1")
        with open(self.path, encoding="utf-8") as handle:
            document = json.load(handle)
        leased = [record for record in document["delivery"]
                  if record.get("leases")]
        self.assertEqual(len(leased), 2)
        for record in leased:
            lease = record["leases"][0]
            self.assertEqual(list(lease),
                             ["lease_id", "limit", "leased_until",
                              "released_at"])
            self.assertRegex(lease["released_at"], r"\.\d{6}\+00:00$")

    def test_restart_restores_release_and_replays(self) -> None:
        self._claim("L1", 2)
        first, _ = self.service.inbox_release_lease("bob", "L1")
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        # The released lease no longer withholds a1/a2 from a fresh claim.
        body, status = restarted.inbox_claim(
            "bob", {"lease_id": "L2", "limit": 10})
        self.assertEqual(status, 201)
        self.assertIn("a1", [m["message_id"] for m in body["messages"]])
        # The release replays byte-identically after the restart.
        replay, status = restarted.inbox_release_lease("bob", "L1")
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)

    def test_failed_write_rolls_back_the_release(self) -> None:
        def raise_oserror(state):
            raise OSError("simulated disk failure")

        self._claim("L1", 2)
        generation = self.state_store.commit_seq
        self.state_store.save = raise_oserror
        with self.assertRaises(PersistenceUnavailable):
            self.service.inbox_release_lease("bob", "L1")
        del self.state_store.save  # un-patch the simulated failure
        self.assertEqual(self.state_store.commit_seq, generation)
        # Rolled back: the lease is still held (blocks a fresh claim) and
        # carries no release timestamp.
        for state in self.service.store._delivery.values():
            for lease in state.leases:
                self.assertIsNone(lease.released_at)
        body, status = self.service.inbox_claim(
            "bob", {"lease_id": "L2", "limit": 10})
        self.assertEqual(status, 201)
        self.assertEqual([m["message_id"] for m in body["messages"]],
                         ["a3", "b1", "b2"])

    def test_legacy_lease_without_released_at_loads(self) -> None:
        # Strip the released_at key: an older version-1 writer would never
        # have emitted it, and it must load as null.
        self._claim("L1", 2)
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
        body, status = restarted.inbox_release_lease("bob", "L1")
        self.assertEqual(status, 201)
        self.assertEqual(body["released_count"], 2)

    def _malformed_document(self, mutate):
        self._claim("L1", 2)
        self.service.inbox_release_lease("bob", "L1")
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
            leased[0]["leases"][0]["released_at"] = 7
        self._assert_refuses_startup(self._malformed_document(mutate))

    def test_restore_rejects_malformed_released_at(self) -> None:
        def mutate(document):
            leased = [r for r in document["delivery"] if r.get("leases")]
            leased[0]["leases"][0]["released_at"] = "2024-01-01 00:00:00"
        self._assert_refuses_startup(self._malformed_document(mutate))

    def test_restore_rejects_naive_released_at(self) -> None:
        def mutate(document):
            leased = [r for r in document["delivery"] if r.get("leases")]
            leased[0]["leases"][0]["released_at"] = \
                "2024-01-01T00:00:00.000000"
        self._assert_refuses_startup(self._malformed_document(mutate))

    def test_restore_rejects_inconsistent_released_at(self) -> None:
        # L1 is recorded on two records with the same released_at; change
        # one copy so the same id binds two different release timestamps.
        def mutate(document):
            leased = [r for r in document["delivery"] if r.get("leases")]
            self.assertEqual(len(leased), 2)
            leased[1]["leases"][0]["released_at"] = \
                "2000-01-01T00:00:00.000000+00:00"
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
            "POST",
            f"/v1/devices/{device_id}/inbox/leases/{lease_id}/release",
            body=raw,
            headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        data = response.read().decode("utf-8")
        return response.status, (json.loads(data) if data else None), data

    def test_release_and_replay_over_http(self) -> None:
        self._claim("L1", 2)
        status, body, raw = self._request("bob", "L1")
        self.assertEqual(status, 201)
        self.assertEqual(list(body),
                         ["device_id", "lease_id", "released_at",
                          "released_count"])
        # Key order is also correct in the serialized bytes.
        self.assertLess(raw.index('"device_id"'), raw.index('"lease_id"'))
        self.assertLess(raw.index('"lease_id"'), raw.index('"released_at"'))
        self.assertLess(raw.index('"released_at"'),
                        raw.index('"released_count"'))
        status, again, raw_again = self._request("bob", "L1")
        self.assertEqual(status, 200)
        self.assertEqual(raw_again, raw)

    def test_non_empty_body_is_400_request_body(self) -> None:
        self._claim("L1", 2)
        status, body, _ = self._request("bob", "L1", raw="{}")
        self.assertEqual(status, 400)
        self.assertEqual(list(body), ["message", "field"])
        self.assertEqual(body["field"], "request_body")

    def test_error_statuses_and_fields(self) -> None:
        status, body, _ = self._request("bob", "nope")
        self.assertEqual((status, body["field"]), (404, "lease_id"))
        self.assertEqual(list(body), ["message", "field"])
        self._claim("L1", 2)
        status, body, _ = self._request("carol", "L1")
        self.assertEqual((status, body["field"]), (409, "lease_id"))


class ReleaseHTTPFailureTest(ReleaseMixin, unittest.TestCase):
    """The 503/data_file path needs a real data file to fail."""

    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self._build()
        self.state_store = attach_persistence(
            self.service, os.path.join(self.directory, "state.json"))
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
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_failed_write_is_503_data_file(self) -> None:
        def raise_oserror(state):
            raise OSError("simulated disk failure")

        self._claim("L1", 2)
        self.state_store.save = raise_oserror
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("POST", "/v1/devices/bob/inbox/leases/L1/release")
        response = conn.getresponse()
        self.assertEqual(response.status, 503)
        body = json.loads(response.read().decode("utf-8"))
        self.assertEqual(list(body), ["message", "field"])
        self.assertEqual(body["field"], "data_file")
        # Rolled back: no release timestamp survived.
        for state in self.service.store._delivery.values():
            for lease in state.leases:
                self.assertIsNone(lease.released_at)


if __name__ == "__main__":
    unittest.main()
