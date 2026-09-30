"""Persistence/recovery tests for the session_rotations section."""
import base64
import json
import os
import shutil
import tempfile
import unittest

from cryptography.hazmat.primitives.asymmetric import x25519
from cryptography.hazmat.primitives import serialization

from e2ee_backend.models import Device, SignedPreKey
from e2ee_backend.persistence import (
    StateFileError,
    attach_persistence,
)
from e2ee_backend.service import DeviceService


def _valid_key() -> str:
    raw = x25519.X25519PrivateKey.generate().public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return base64.b64encode(raw).decode()


class SessionRotationRecoveryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.path = os.path.join(self.directory, "state.json")
        service = DeviceService()
        attach_persistence(service, self.path)
        service.store.add_device(
            Device("u", "a", _valid_key(),
                   prekeys=[SignedPreKey("ak1", _valid_key())]))
        service.store.add_device(
            Device("u", "b", _valid_key(),
                   prekeys=[SignedPreKey("rk1", _valid_key()),
                            SignedPreKey("rk2", _valid_key())]))
        predecessor = service.create_session({
            "initiator_device_id": "a", "recipient_device_id": "b",
            "prekey_id": "rk1", "ephemeral_key": _valid_key()})
        self.predecessor_id = predecessor["session_id"]
        # One message so predecessor_last_sequence is 1, not 0.
        service.post_message({
            "session_id": self.predecessor_id,
            "sender_device_id": "a", "message_id": "m1",
            "sequence": 1, "nonce": "n1", "ciphertext": "c"})
        rotated, status = service.rotate_session(self.predecessor_id, {
            "rotation_id": "rot-1", "actor_device_id": "a",
            "prekey_id": "rk2", "ephemeral_key": _valid_key()})
        self.assertEqual(status, 201)
        self.rotated = rotated
        with open(self.path, encoding="utf-8") as handle:
            self.document = json.load(handle)

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def _load(self, document) -> DeviceService:
        path = os.path.join(self.directory, "candidate.json")
        document.pop("integrity_log_version", None)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        service = DeviceService()
        attach_persistence(service, path)
        return service

    def _assert_rejected(self, document) -> None:
        path = os.path.join(self.directory, "bad.json")
        document.pop("integrity_log_version", None)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        before = open(path, "rb").read()
        with self.assertRaises(StateFileError):
            attach_persistence(DeviceService(), path)
        self.assertEqual(open(path, "rb").read(), before)

    def test_rotation_survives_restart_and_replays_200(self) -> None:
        service = self._load(self.document)
        rotations = service.store.snapshot_state()["session_rotations"]
        self.assertEqual(len(rotations), 1)
        record = rotations[0]
        self.assertEqual(record["rotation_id"], "rot-1")
        self.assertEqual(record["predecessor_session_id"],
                         self.predecessor_id)
        self.assertEqual(record["successor_session_id"],
                         self.rotated["session_id"])
        self.assertEqual(record["predecessor_last_sequence"], 1)
        # A replay after restart returns the original response with 200.
        body, status = service.rotate_session(self.predecessor_id, {
            "rotation_id": "rot-1", "actor_device_id": "a",
            "prekey_id": "rk2",
            "ephemeral_key": self.rotated["ephemeral_key"]})
        self.assertEqual(status, 200)
        self.assertEqual(body, self.rotated)

    def test_old_session_still_rejects_writes_after_restart(self) -> None:
        service = self._load(self.document)
        with self.assertRaises(Exception) as caught:
            service.post_message({
                "session_id": self.predecessor_id,
                "sender_device_id": "a", "message_id": "m2",
                "sequence": 2, "nonce": "n2", "ciphertext": "c"})
        self.assertEqual(caught.exception.field, "session_id")
        self.assertEqual(caught.exception.status_code, 409)
        # The successor accepts a fresh sequence-1 message.
        body = service.post_message({
            "session_id": self.rotated["session_id"],
            "sender_device_id": "a", "message_id": "x1",
            "sequence": 1, "nonce": "x1", "ciphertext": "c"})
        self.assertEqual(body["sequence"], 1)

    def test_get_rotation_after_restart(self) -> None:
        service = self._load(self.document)
        by_pred = service.get_session_rotation(self.predecessor_id)
        self.assertEqual(by_pred, self.rotated)
        by_succ = service.get_session_rotation(self.rotated["session_id"])
        self.assertEqual(by_succ, self.rotated)

    # -- corruption gates --------------------------------------------------

    def test_missing_optional_section_loads_empty(self) -> None:
        document = dict(self.document)
        del document["session_rotations"]
        service = self._load(document)
        self.assertEqual(
            service.store.snapshot_state()["session_rotations"], [])

    def test_section_wrong_type_rejected(self) -> None:
        document = dict(self.document)
        document["session_rotations"] = {}
        self._assert_rejected(document)

    def test_duplicate_rotation_id_rejected(self) -> None:
        document = dict(self.document)
        records = list(document["session_rotations"])
        records.append(dict(records[0]))
        records[-1]["successor_session_id"] = self.predecessor_id
        document["session_rotations"] = records
        self._assert_rejected(document)

    def test_forked_predecessor_rejected(self) -> None:
        document = dict(self.document)
        bad = dict(document["session_rotations"][0])
        bad["rotation_id"] = "rot-2"
        bad["successor_session_id"] = self.predecessor_id
        document["session_rotations"].append(bad)
        self._assert_rejected(document)

    def test_unknown_predecessor_rejected(self) -> None:
        document = dict(self.document)
        document["session_rotations"][0]["predecessor_session_id"] = "ghost"
        self._assert_rejected(document)

    def test_unknown_successor_rejected(self) -> None:
        document = dict(self.document)
        document["session_rotations"][0]["successor_session_id"] = "ghost"
        self._assert_rejected(document)

    def test_endpoints_must_match_predecessor(self) -> None:
        document = dict(self.document)
        document["session_rotations"][0]["recipient_device_id"] = "ak1"
        self._assert_rejected(document)

    def test_prekey_must_belong_to_recipient(self) -> None:
        document = dict(self.document)
        record = document["session_rotations"][0]
        record["prekey_id"] = "ak1"  # the initiator's own prekey
        record["public_key"] = next(
            d for d in document["devices"] if d["device_id"] == "a"
        )["prekeys"][0]["public_key"]
        self._assert_rejected(document)

    def test_frozen_values_must_match_successor(self) -> None:
        document = dict(self.document)
        document["session_rotations"][0]["identity_key"] = "tampered"
        self._assert_rejected(document)

    def test_wrong_predecessor_last_sequence_rejected(self) -> None:
        document = dict(self.document)
        document["session_rotations"][0]["predecessor_last_sequence"] = 99
        self._assert_rejected(document)

    def test_negative_predecessor_last_sequence_rejected(self) -> None:
        document = dict(self.document)
        document["session_rotations"][0]["predecessor_last_sequence"] = -1
        self._assert_rejected(document)

    def test_missing_string_field_rejected(self) -> None:
        document = dict(self.document)
        del document["session_rotations"][0]["rotation_id"]
        self._assert_rejected(document)


if __name__ == "__main__":
    unittest.main()
