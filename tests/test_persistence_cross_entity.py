"""Startup recovery tests for cross-entity consistency in version-1 files.

A structurally valid JSON document may still be semantically contradictory:
a session pointing at an unregistered device, a group whose creator is not
its first member, a frozen group session ahead of the group's current
revision, and so on. Every such file must make startup refuse with
``StateFileError`` while leaving the file's bytes and inode untouched, and
``serve --data-file`` must emit one single-line JSON error on stderr with
``field=data_file``.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from typing import Any, Dict, Tuple

from e2ee_backend.models import Device
from e2ee_backend.persistence import StateFileError, attach_persistence
from e2ee_backend.service import DeviceService


def build_full_fixture(directory: str) -> Tuple[DeviceService, str, Dict[str, Any]]:
    """Persist devices, a 1:1 session, a group and a group session.

    Returns ``(service, path, good_document)``.
    """
    path = os.path.join(directory, "state.json")
    service = DeviceService()
    attach_persistence(service, path)
    for device_id in ("a", "b", "c"):
        service.store.add_device(Device("u", device_id, "ik"))
    service.store.add_prekey("b", "pk1", "pkpub")
    service.store.create_session("a", "b", "pk1", "epk")
    service.create_group({
        "group_id": "g1", "creator_device_id": "c",
        "member_device_ids": ["a", "b"]})
    service.create_group_session({
        "group_id": "g1", "initiator_device_id": "c",
        "ephemeral_key": "epk"})
    with open(path, encoding="utf-8") as handle:
        document = json.load(handle)
    return service, path, document


class _RejectionBase(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        _service, _path, self.good_document = build_full_fixture(self.directory)

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def assert_rejected(self, document: Dict[str, Any]) -> None:
        """Startup refuses *document* without touching its file."""
        path = os.path.join(self.directory, "bad.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        before = open(path, "rb").read()
        before_ino = os.stat(path).st_ino
        with self.assertRaises(StateFileError):
            attach_persistence(DeviceService(), path)
        self.assertEqual(open(path, "rb").read(), before)
        self.assertEqual(os.stat(path).st_ino, before_ino)

    def with_doc(self, mutator) -> Dict[str, Any]:
        document = deepcopy(self.good_document)
        mutator(document)
        return document


class SessionConsistencyStartupTest(_RejectionBase):
    def _session(self, document: Dict[str, Any]) -> Dict[str, Any]:
        return document["sessions"][0]

    def test_initiator_must_be_registered(self) -> None:
        def mutate(document: Dict[str, Any]) -> None:
            self._session(document)["initiator_device_id"] = "ghost"
        self.assert_rejected(self.with_doc(mutate))

    def test_recipient_must_be_registered(self) -> None:
        def mutate(document: Dict[str, Any]) -> None:
            self._session(document)["recipient_device_id"] = "ghost"
        self.assert_rejected(self.with_doc(mutate))

    def test_prekey_must_belong_to_recipient(self) -> None:
        def mutate(document: Dict[str, Any]) -> None:
            session = self._session(document)
            # pk1 is b's key; claiming it under a different recipient, and a
            # key id that exists on no device, are both contradictions.
            session["recipient_device_id"] = "c"
        self.assert_rejected(self.with_doc(mutate))

        def mutate2(document: Dict[str, Any]) -> None:
            self._session(document)["prekey_id"] = "nope"
        self.assert_rejected(self.with_doc(mutate2))

    def test_prekey_of_initiator_is_not_enough(self) -> None:
        # Give the initiator its own identically-named key and confirm the
        # prekey must belong to the *recipient*, not either endpoint.
        def mutate(document: Dict[str, Any]) -> None:
            session = self._session(document)
            initiator = next(d for d in document["devices"]
                             if d["device_id"] == "a")
            initiator["prekeys"].append(
                {"key_id": "other", "public_key": "x", "revoked": False})
            session["prekey_id"] = "other"
        self.assert_rejected(self.with_doc(mutate))

    def test_duplicate_session_id_is_rejected(self) -> None:
        def mutate(document: Dict[str, Any]) -> None:
            document["sessions"].append(deepcopy(document["sessions"][0]))
        self.assert_rejected(self.with_doc(mutate))

    def test_string_fields_reject_wrong_types_and_emptiness(self) -> None:
        fields = ("session_id", "initiator_device_id", "recipient_device_id",
                  "prekey_id", "ephemeral_key", "identity_key", "public_key",
                  "created_at")
        for field in fields:
            for bad in (True, 1, None, ""):
                def mutate(document: Dict[str, Any], field=field,
                           bad=bad) -> None:
                    self._session(document)[field] = bad
                self.assert_rejected(self.with_doc(mutate))

    def test_well_formed_session_loads(self) -> None:
        service = DeviceService()
        attach_persistence(service, os.path.join(self.directory, "state.json"))
        views = service.store.snapshot_state()["sessions"]
        self.assertEqual(len(views), 1)
        self.assertEqual(views[0]["recipient_device_id"], "b")


class GroupConsistencyStartupTest(_RejectionBase):
    def _group(self, document: Dict[str, Any]) -> Dict[str, Any]:
        return document["groups"][0]

    def test_creator_must_be_registered(self) -> None:
        def mutate(document: Dict[str, Any]) -> None:
            group = self._group(document)
            group["creator_device_id"] = "ghost"
            group["members"][0] = "ghost"
        self.assert_rejected(self.with_doc(mutate))

    def test_creator_must_be_first_member(self) -> None:
        def mutate(document: Dict[str, Any]) -> None:
            self._group(document)["members"] = ["a", "c", "b"]
        self.assert_rejected(self.with_doc(mutate))

    def test_members_must_be_nonempty(self) -> None:
        def mutate_empty(document: Dict[str, Any]) -> None:
            self._group(document)["members"] = []
        self.assert_rejected(self.with_doc(mutate_empty))

        def mutate_typed(document: Dict[str, Any]) -> None:
            self._group(document)["members"] = "c,a,b"
        self.assert_rejected(self.with_doc(mutate_typed))

    def test_duplicate_members_are_rejected(self) -> None:
        def mutate(document: Dict[str, Any]) -> None:
            self._group(document)["members"] = ["c", "a", "a"]
        self.assert_rejected(self.with_doc(mutate))

    def test_member_wrong_type_is_rejected(self) -> None:
        def mutate(document: Dict[str, Any]) -> None:
            self._group(document)["members"] = ["c", 3]
        self.assert_rejected(self.with_doc(mutate))

    def test_revision_must_be_a_positive_integer(self) -> None:
        for bad in (0, -1, "1", True, 1.5, None):
            def mutate(document: Dict[str, Any], bad=bad) -> None:
                self._group(document)["revision"] = bad
            self.assert_rejected(self.with_doc(mutate))

    def test_identifier_wrong_types_are_rejected(self) -> None:
        for field, bad in (("group_id", 7), ("creator_device_id", False),
                           ("created_at", ""), ("group_id", None)):
            def mutate(document: Dict[str, Any], field=field,
                       bad=bad) -> None:
                self._group(document)[field] = bad
            self.assert_rejected(self.with_doc(mutate))

    def test_well_formed_group_loads(self) -> None:
        service = DeviceService()
        attach_persistence(service, os.path.join(self.directory, "state.json"))
        groups = service.store.snapshot_state()["groups"]
        self.assertEqual(groups[0]["members"], ["c", "a", "b"])
        self.assertEqual(groups[0]["revision"], 1)


class GroupSessionConsistencyStartupTest(_RejectionBase):
    def _gs(self, document: Dict[str, Any]) -> Dict[str, Any]:
        return document["group_sessions"][0]

    def test_group_must_exist(self) -> None:
        def mutate(document: Dict[str, Any]) -> None:
            self._gs(document)["group_id"] = "ghost"
        self.assert_rejected(self.with_doc(mutate))

    def test_duplicate_session_id_is_rejected(self) -> None:
        def mutate(document: Dict[str, Any]) -> None:
            document["group_sessions"].append(
                deepcopy(document["group_sessions"][0]))
        self.assert_rejected(self.with_doc(mutate))

    def test_id_collision_with_one_to_one_session_is_rejected(self) -> None:
        def mutate(document: Dict[str, Any]) -> None:
            document["group_sessions"][0]["session_id"] = \
                document["sessions"][0]["session_id"]
        self.assert_rejected(self.with_doc(mutate))

    def test_members_must_be_nonempty_and_deduplicated(self) -> None:
        def mutate_empty(document: Dict[str, Any]) -> None:
            self._gs(document)["members"] = []
        self.assert_rejected(self.with_doc(mutate_empty))

        def mutate_dup(document: Dict[str, Any]) -> None:
            self._gs(document)["members"] = ["c", "a", "a"]
        self.assert_rejected(self.with_doc(mutate_dup))

        def mutate_typed(document: Dict[str, Any]) -> None:
            self._gs(document)["members"] = ["c", None]
        self.assert_rejected(self.with_doc(mutate_typed))

    def test_initiator_must_be_a_frozen_member(self) -> None:
        def mutate(document: Dict[str, Any]) -> None:
            self._gs(document)["initiator_device_id"] = "a"
            self._gs(document)["members"] = ["c", "b"]
        self.assert_rejected(self.with_doc(mutate))

    def test_revision_must_be_positive_and_not_ahead_of_group(self) -> None:
        for bad in (0, -1, "1", True, 1.5, None):
            def mutate(document: Dict[str, Any], bad=bad) -> None:
                self._gs(document)["revision"] = bad
            self.assert_rejected(self.with_doc(mutate))

        def mutate_ahead(document: Dict[str, Any]) -> None:
            self._gs(document)["revision"] = 2
        self.assert_rejected(self.with_doc(mutate_ahead))

    def test_older_equal_revision_snapshots_load(self) -> None:
        # The frozen snapshot at the group's current revision is valid and
        # survives a restart.
        path = os.path.join(self.directory, "state.json")
        restarted = DeviceService()
        attach_persistence(restarted, path)
        state = restarted.store.snapshot_state()
        self.assertEqual(state["groups"][0]["revision"], 1)
        self.assertEqual(state["group_sessions"][0]["revision"], 1)

    def test_snapshot_behind_current_group_revision_loads(self) -> None:
        # The group advanced after the snapshot was frozen: revision 1 < 3.
        def mutate(document: Dict[str, Any]) -> None:
            document["groups"][0]["revision"] = 3
        path = os.path.join(self.directory, "behind.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.with_doc(mutate), handle)
        service = DeviceService()
        attach_persistence(service, path)
        state = service.store.snapshot_state()
        self.assertEqual(state["groups"][0]["revision"], 3)
        self.assertEqual(state["group_sessions"][0]["revision"], 1)

    def test_string_fields_reject_wrong_types(self) -> None:
        for field in ("session_id", "group_id", "initiator_device_id",
                      "ephemeral_key", "created_at"):
            for bad in (False, 9, None, ""):
                def mutate(document: Dict[str, Any], field=field,
                           bad=bad) -> None:
                    self._gs(document)[field] = bad
                self.assert_rejected(self.with_doc(mutate))


class DeviceConsistencyStartupTest(_RejectionBase):
    def _device(self, document: Dict[str, Any],
                device_id: str = "b") -> Dict[str, Any]:
        return next(d for d in document["devices"]
                    if d["device_id"] == device_id)

    def test_duplicate_prekey_key_id_is_rejected(self) -> None:
        def mutate(document: Dict[str, Any]) -> None:
            prekeys = self._device(document)["prekeys"]
            prekeys.append({"key_id": "pk1", "public_key": "other",
                            "revoked": False})
        self.assert_rejected(self.with_doc(mutate))

    def test_prekey_fields_reject_wrong_types(self) -> None:
        for field, bad in (("key_id", 1), ("key_id", True),
                           ("public_key", 5), ("public_key", None),
                           ("revoked", "yes")):
            def mutate(document: Dict[str, Any], field=field,
                       bad=bad) -> None:
                self._device(document)["prekeys"][0][field] = bad
            self.assert_rejected(self.with_doc(mutate))

    def test_device_fields_reject_wrong_types(self) -> None:
        for field, bad in (("user_id", 1), ("device_id", None),
                           ("identity_key", True), ("registered_at", 3),
                           ("rotated_at", ""), ("revoked", 0)):
            def mutate(document: Dict[str, Any], field=field,
                       bad=bad) -> None:
                self._device(document)[field] = bad
            self.assert_rejected(self.with_doc(mutate))

    def test_legacy_v1_file_without_rotation_fields_loads(self) -> None:
        # An older version-1 writer predates rotated_at and per-record
        # revoked flags; absence keeps the legacy default behavior.
        def mutate(document: Dict[str, Any]) -> None:
            for device in document["devices"]:
                device.pop("rotated_at", None)
                device.pop("revoked", None)
                for prekey in device["prekeys"]:
                    prekey.pop("revoked", None)
        path = os.path.join(self.directory, "legacy.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.with_doc(mutate), handle)
        service = DeviceService()
        attach_persistence(service, path)
        device = service.store.find_by_device_id("b")
        self.assertIsNotNone(device)
        self.assertFalse(device.revoked)
        self.assertEqual(device.rotated_at, device.registered_at)
        self.assertEqual(
            service.store.active_prekey_ids(device), ["pk1"])


class ServeRefusesBadFileTest(unittest.TestCase):
    """End-to-end: serve --data-file on a contradictory file."""

    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        _service, _path, self.good_document = build_full_fixture(self.directory)

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_stderr_single_line_json_and_file_untouched(self) -> None:
        document = deepcopy(self.good_document)
        # A group session frozen two revisions ahead of its group.
        document["group_sessions"][0]["revision"] = 99
        path = os.path.join(self.directory, "bad.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        before = open(path, "rb").read()
        before_ino = os.stat(path).st_ino

        result = subprocess.run(
            [sys.executable, "-m", "e2ee_backend", "serve",
             "--data-file", path, "--port", "0"],
            capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        line = result.stderr.strip()
        self.assertEqual(line.count("\n"), 0)
        self.assertEqual(json.loads(line)["field"], "data_file")
        self.assertEqual(open(path, "rb").read(), before)
        self.assertEqual(os.stat(path).st_ino, before_ino)


if __name__ == "__main__":
    unittest.main()
