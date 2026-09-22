"""Persistence/recovery tests for the group_session_rotations section."""
import base64
import json
import os
import shutil
import tempfile
import unittest

from cryptography.hazmat.primitives.asymmetric import x25519
from cryptography.hazmat.primitives import serialization

from e2ee_backend.models import Device
from e2ee_backend.persistence import (
    StateFileError,
    attach_persistence,
)
from e2ee_backend.service import DeviceService


def _valid_key() -> str:
    raw = x25519.X25519PrivateKey.generate().public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return base64.b64encode(raw).decode()


class GroupRotationRecoveryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.path = os.path.join(self.directory, "state.json")
        service = DeviceService()
        attach_persistence(service, self.path)
        service.store.add_device(Device("u", "creator", "ik"))
        service.store.add_device(Device("u", "alice", "ik"))
        service.store.add_device(Device("u", "bob", "ik"))
        service.create_group({
            "group_id": "g1", "creator_device_id": "creator",
            "member_device_ids": ["alice", "bob"]})
        predecessor = service.create_group_session({
            "group_id": "g1", "initiator_device_id": "creator",
            "ephemeral_key": _valid_key()})
        self.predecessor_id = predecessor["session_id"]
        rotated, status = service.rotate_group_session(
            self.predecessor_id, {
                "rotation_id": "rot-1",
                "actor_device_id": "creator",
                "ephemeral_key": _valid_key(),
                "expected_revision": 1})
        self.assertEqual(status, 201)
        self.rotated = rotated
        with open(self.path, encoding="utf-8") as handle:
            self.document = json.load(handle)

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def _load(self, document) -> DeviceService:
        path = os.path.join(self.directory, "candidate.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        service = DeviceService()
        attach_persistence(service, path)
        return service

    def _assert_rejected(self, document) -> None:
        path = os.path.join(self.directory, "bad.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        before = open(path, "rb").read()
        with self.assertRaises(StateFileError):
            attach_persistence(DeviceService(), path)
        # A rejected file must never be overwritten at startup.
        self.assertEqual(open(path, "rb").read(), before)

    def test_rotation_survives_restart_and_replays_200(self) -> None:
        service = DeviceService()
        attach_persistence(service, self.path)
        rotations = service.store.snapshot_state()["group_session_rotations"]
        self.assertEqual(len(rotations), 1)
        record = rotations[0]
        self.assertEqual(record["rotation_id"], "rot-1")
        self.assertEqual(record["predecessor_session_id"],
                         self.predecessor_id)
        self.assertEqual(record["successor_session_id"],
                         self.rotated["session_id"])
        # Idempotent replay after restart returns the original response.
        body, status = service.rotate_group_session(
            self.predecessor_id, {
                "rotation_id": "rot-1", "actor_device_id": "creator",
                "ephemeral_key": _valid_key(), "expected_revision": 1})
        self.assertEqual(status, 200)
        self.assertEqual(body, self.rotated)
        # And the no-fork rule survived the restart.
        from e2ee_backend.service import ServiceError
        with self.assertRaises(ServiceError) as caught:
            service.rotate_group_session(
                self.predecessor_id, {
                    "rotation_id": "rot-2", "actor_device_id": "creator",
                    "ephemeral_key": _valid_key(), "expected_revision": 1})
        self.assertEqual(caught.exception.status_code, 409)

    def test_legacy_file_without_section_loads_empty(self) -> None:
        document = dict(self.document)
        del document["group_session_rotations"]
        service = self._load(document)
        self.assertEqual(
            service.store.snapshot_state()["group_session_rotations"], [])

    def test_section_must_be_a_list(self) -> None:
        document = dict(self.document)
        document["group_session_rotations"] = {"rotation_id": "rot-1"}
        self._assert_rejected(document)

    def test_well_formed_record_loads(self) -> None:
        service = self._load(self.document)
        self.assertEqual(
            len(service.store.snapshot_state()[
                "group_session_rotations"]), 1)

    def test_malformed_records_are_rejected(self) -> None:
        record = self.document["group_session_rotations"][0]

        def with_record(mutated) -> dict:
            document = dict(self.document)
            document["group_session_rotations"] = [mutated]
            return document

        bad = [
            "not-an-object",
            {k: v for k, v in record.items() if k != "rotation_id"},
            {**record, "rotation_id": ""},
            {**record, "predecessor_session_id": ""},
            {**record, "predecessor_session_id": "missing-session"},
            {**record, "successor_session_id": "missing-session"},
            {**record, "group_id": "g-other"},
            {**record, "actor_device_id": "ghost"},
            {**record, "actor_device_id": "alice"},  # not the creator
            {**record, "revision": 0},
            {**record, "revision": "1"},
            {**record, "revision": True},
            {**record, "members": []},
            {**record, "members": ["creator", "creator"]},
            {**record, "members": ["creator", 1]},
            {**record, "created_at": ""},
            # Frozen values must match the successor snapshot.
            {**record, "revision": 2},
            {**record, "members": ["creator", "alice"]},
            {**record, "actor_device_id": "creator",
             "successor_session_id": self.predecessor_id},
        ]
        for mutated in bad:
            self._assert_rejected(with_record(mutated))

    def test_duplicate_rotation_id_rejected(self) -> None:
        record = self.document["group_session_rotations"][0]
        document = dict(self.document)
        document["group_session_rotations"] = [record, dict(record)]
        self._assert_rejected(document)

    def test_forked_predecessor_rejected(self) -> None:
        # Build a second successor in the live store, then hand-craft a
        # document where both records name the same predecessor.
        service = DeviceService()
        attach_persistence(service, self.path)
        _, second = service.rotate_group_session(
            self.rotated["session_id"], {
                "rotation_id": "rot-2", "actor_device_id": "creator",
                "ephemeral_key": _valid_key(), "expected_revision": 1})
        document = service.store.snapshot_state()
        records = document["group_session_rotations"]
        # Repoint the second record's predecessor at the first one's.
        records[1]["predecessor_session_id"] = self.predecessor_id
        # Keep its frozen successor values internally consistent but the
        # predecessor is now referenced by two records (a fork).
        self._assert_rejected({"version": 1, **document})

    def test_rotation_id_conflict_across_predecessors_rejected(self) -> None:
        service = DeviceService()
        attach_persistence(service, self.path)
        _, second = service.rotate_group_session(
            self.rotated["session_id"], {
                "rotation_id": "rot-2", "actor_device_id": "creator",
                "ephemeral_key": _valid_key(), "expected_revision": 1})
        document = service.store.snapshot_state()
        # Reuse the first rotation_id for the second record.
        document["group_session_rotations"][1]["rotation_id"] = "rot-1"
        self._assert_rejected({"version": 1, **document})


if __name__ == "__main__":
    unittest.main()
