"""Startup cross-entity consistency checks for version=1 state files.

A structurally valid JSON document may still be semantically contradictory:
a session endpoint that names no registered device, a pre-key that does not
belong to the session recipient, a group whose creator is missing or not the
first member, a frozen group-session snapshot ahead of its group's revision,
duplicated ids or members, or fields carrying booleans/numbers/null instead of
strings. Every such contradiction must make ``--data-file`` startup refuse
with :class:`StateFileError`, leaving the rejected file's bytes and inode
untouched. Older version=1 files that merely predate later-added optional
sections still load by the original rules.
"""
import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from typing import Any, Dict

from e2ee_backend.models import Device
from e2ee_backend.persistence import StateFileError, attach_persistence
from e2ee_backend.service import DeviceService
from e2ee_backend.storage import DeviceStore


def build_good_document() -> Dict[str, Any]:
    """Build a valid persisted document with 1:1 and group entities.

    Devices: creator/alice/bob/carol (carol registered but group-less). A 1:1
    session alice -> bob uses bob's pre-key pk1; group g1 has creator first
    followed by alice/bob at revision 2 (alice was added after creation); the
    group session froze the creator/alice/bob roster at revision 1.
    """
    service = DeviceService()
    for device_id in ("creator", "alice", "bob", "carol"):
        service.store.add_device(Device("u", device_id, f"ik-{device_id}"))
    service.store.add_prekey("bob", "pk1", "pk-pub-1")
    service.store.add_prekey("bob", "pk2", "pk-pub-2")
    service.store.create_session("alice", "bob", "pk1", "eph-one")
    service.create_group({
        "group_id": "g1", "creator_device_id": "creator",
        "member_device_ids": ["alice"]})
    service.store.add_group_member("g1", "creator", "bob")
    session = service.create_group_session({
        "group_id": "g1", "initiator_device_id": "creator",
        "ephemeral_key": "eph-group"})
    assert session["revision"] == 2
    # Freeze an older-revision snapshot as well, via a synthetic record that
    # the store itself would never overwrite: build the snapshot at revision 1
    # by restoring a hand-adjusted copy, proving revision <= current loads.
    document = service.store.snapshot_state()
    document["group_sessions"][0]["revision"] = 1
    DeviceStore().restore_state(copy.deepcopy(document))
    return document


class _ConsistencyBase(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.good_document = build_good_document()

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def _write(self, document: Dict[str, Any], name: str = "bad.json") -> str:
        path = os.path.join(self.directory, name)
        stamped = {"version": 1, **document}
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(stamped, handle)
        return path

    def _assert_rejected(self, document: Dict[str, Any]) -> None:
        path = self._write(document)
        before = open(path, "rb").read()
        before_ino = os.stat(path).st_ino
        with self.assertRaises(StateFileError):
            attach_persistence(DeviceService(), path)
        # A rejected document must never be overwritten: same bytes, same
        # inode, no leftover temporary file.
        self.assertEqual(open(path, "rb").read(), before)
        self.assertEqual(os.stat(path).st_ino, before_ino)
        self.assertEqual([n for n in os.listdir(self.directory)
                          if n.endswith(".tmp")], [])

    def _mutated(self, mutate: Any) -> Dict[str, Any]:
        document = copy.deepcopy(self.good_document)
        mutate(document)
        return document


class DeviceFieldConsistencyTest(_ConsistencyBase):
    """Device identifiers, pre-key ids and string fields must be well typed."""

    def test_good_document_loads(self) -> None:
        path = self._write(self.good_document, "state.json")
        service = DeviceService()
        attach_persistence(service, path)
        bob = service.store.find_by_device_id("bob")
        self.assertEqual([pk.key_id for pk in bob.prekeys], ["pk1", "pk2"])
        self.assertEqual(
            service.store.find_by_device_id("creator").rotated_at,
            self.good_document["devices"][0]["rotated_at"])

    def test_device_field_wrong_types_rejected(self) -> None:
        def mutate(device_index: int, field: str, value: Any) -> Any:
            return lambda doc: doc["devices"][device_index].__setitem__(
                field, value)

        for field in ("user_id", "device_id", "identity_key",
                      "registered_at", "rotated_at"):
            for value in (True, 3, None, ""):
                self._assert_rejected(
                    self._mutated(mutate(0, field, value)))

    def test_prekey_field_wrong_types_rejected(self) -> None:
        for field in ("key_id", "public_key"):
            for value in (False, 7, None, ""):
                def mutate(doc: Dict[str, Any], field=field,
                           value=value) -> None:
                    doc["devices"][2]["prekeys"][0][field] = value
                self._assert_rejected(self._mutated(mutate))

        def revoked_bool_to_int(doc: Dict[str, Any]) -> None:
            doc["devices"][2]["prekeys"][0]["revoked"] = 1
        self._assert_rejected(self._mutated(revoked_bool_to_int))

        def revoked_missing_ok_is_not_case(doc: Dict[str, Any]) -> None:
            del doc["devices"][2]["prekeys"][0]["revoked"]
        # Absent revoked defaults to False, like the legacy rule.
        path = self._write(self._mutated(revoked_missing_ok_is_not_case),
                           "legacy-pk.json")
        service = DeviceService()
        attach_persistence(service, path)
        self.assertFalse(
            service.store.find_by_device_id("bob").prekeys[0].revoked)

    def test_duplicate_prekey_id_rejected(self) -> None:
        def mutate(doc: Dict[str, Any]) -> None:
            doc["devices"][2]["prekeys"][1]["key_id"] = "pk1"
        self._assert_rejected(self._mutated(mutate))

    def test_missing_required_string_field_rejected(self) -> None:
        for field in ("user_id", "device_id", "identity_key",
                      "registered_at", "prekeys"):
            def mutate(doc: Dict[str, Any], field=field) -> None:
                del doc["devices"][0][field]
            self._assert_rejected(self._mutated(mutate))


class SessionConsistencyTest(_ConsistencyBase):
    """sessions must reference registered devices and the recipient's prekey."""

    def setUp(self) -> None:
        super().setUp()
        self.one2one = self.good_document["sessions"][0]

    def test_dangling_endpoints_rejected(self) -> None:
        def bad_initiator(doc: Dict[str, Any]) -> None:
            doc["sessions"][0]["initiator_device_id"] = "ghost"
        self._assert_rejected(self._mutated(bad_initiator))

        def bad_recipient(doc: Dict[str, Any]) -> None:
            doc["sessions"][0]["recipient_device_id"] = "ghost"
        self._assert_rejected(self._mutated(bad_recipient))

    def test_prekey_must_belong_to_recipient(self) -> None:
        def unknown_prekey(doc: Dict[str, Any]) -> None:
            doc["sessions"][0]["prekey_id"] = "pk-unknown"
        self._assert_rejected(self._mutated(unknown_prekey))

        # pk-pub key id pk1 exists, but on bob: pointing the session at carol
        # as recipient makes pk1 a foreign pre-key for her.
        def foreign_prekey(doc: Dict[str, Any]) -> None:
            doc["sessions"][0]["recipient_device_id"] = "carol"
        self._assert_rejected(self._mutated(foreign_prekey))

    def test_duplicate_session_id_rejected(self) -> None:
        def mutate(doc: Dict[str, Any]) -> None:
            doc["sessions"].append(copy.deepcopy(doc["sessions"][0]))
        self._assert_rejected(self._mutated(mutate))

    def test_session_id_collision_with_group_session_rejected(self) -> None:
        def mutate(doc: Dict[str, Any]) -> None:
            doc["group_sessions"][0]["session_id"] = \
                doc["sessions"][0]["session_id"]
        self._assert_rejected(self._mutated(mutate))

    def test_string_fields_reject_wrong_types(self) -> None:
        fields = ("session_id", "initiator_device_id",
                  "recipient_device_id", "prekey_id", "ephemeral_key",
                  "identity_key", "public_key", "created_at")
        for field in fields:
            for value in (True, 4, None, ""):
                def mutate(doc: Dict[str, Any], field=field,
                           value=value) -> None:
                    doc["sessions"][0][field] = value
                self._assert_rejected(self._mutated(mutate))

    def test_missing_field_rejected(self) -> None:
        for field in ("session_id", "initiator_device_id",
                      "recipient_device_id", "prekey_id", "ephemeral_key",
                      "identity_key", "public_key", "created_at"):
            def mutate(doc: Dict[str, Any], field=field) -> None:
                del doc["sessions"][0][field]
            self._assert_rejected(self._mutated(mutate))


class GroupConsistencyTest(_ConsistencyBase):
    """groups require a registered first-member creator and positive revision."""

    def test_unknown_creator_rejected(self) -> None:
        def mutate(doc: Dict[str, Any]) -> None:
            doc["groups"][0]["creator_device_id"] = "ghost"
            doc["groups"][0]["members"][0] = "ghost"
        self._assert_rejected(self._mutated(mutate))

    def test_creator_must_be_first_member(self) -> None:
        def mutate(doc: Dict[str, Any]) -> None:
            doc["groups"][0]["members"] = ["alice", "creator", "bob"]
        self._assert_rejected(self._mutated(mutate))

    def test_members_must_be_nonempty_deduped_strings(self) -> None:
        def empty_members(doc: Dict[str, Any]) -> None:
            doc["groups"][0]["members"] = []
        self._assert_rejected(self._mutated(empty_members))

        def dup_members(doc: Dict[str, Any]) -> None:
            doc["groups"][0]["members"] = ["creator", "alice", "alice"]
        self._assert_rejected(self._mutated(dup_members))

        def typed_members(doc: Dict[str, Any]) -> None:
            doc["groups"][0]["members"] = ["creator", 3]
        self._assert_rejected(self._mutated(typed_members))

        def null_members(doc: Dict[str, Any]) -> None:
            doc["groups"][0]["members"] = ["creator", None]
        self._assert_rejected(self._mutated(null_members))

        def empty_member(doc: Dict[str, Any]) -> None:
            doc["groups"][0]["members"] = ["creator", ""]
        self._assert_rejected(self._mutated(empty_member))

        def not_a_list(doc: Dict[str, Any]) -> None:
            doc["groups"][0]["members"] = "creator"
        self._assert_rejected(self._mutated(not_a_list))

    def test_revision_must_be_positive_integer(self) -> None:
        for value in (0, -1, "2", True, 1.0, None):
            def mutate(doc: Dict[str, Any], value=value) -> None:
                doc["groups"][0]["revision"] = value
            self._assert_rejected(self._mutated(mutate))

    def test_string_fields_reject_wrong_types(self) -> None:
        for field in ("group_id", "creator_device_id", "created_at"):
            for value in (False, 9, None, ""):
                def mutate(doc: Dict[str, Any], field=field,
                           value=value) -> None:
                    doc["groups"][0][field] = value
                self._assert_rejected(self._mutated(mutate))

    def test_duplicate_group_id_rejected(self) -> None:
        def mutate(doc: Dict[str, Any]) -> None:
            doc["groups"].append(copy.deepcopy(doc["groups"][0]))
        self._assert_rejected(self._mutated(mutate))


class GroupSessionConsistencyTest(_ConsistencyBase):
    """group_sessions must freeze a consistent snapshot of a stored group."""

    def test_unknown_group_rejected(self) -> None:
        def mutate(doc: Dict[str, Any]) -> None:
            doc["group_sessions"][0]["group_id"] = "ghost-group"
        self._assert_rejected(self._mutated(mutate))

    def test_duplicate_group_session_id_rejected(self) -> None:
        def mutate(doc: Dict[str, Any]) -> None:
            doc["group_sessions"].append(
                copy.deepcopy(doc["group_sessions"][0]))
        self._assert_rejected(self._mutated(mutate))

    def test_members_must_be_nonempty_deduped_strings(self) -> None:
        def empty_members(doc: Dict[str, Any]) -> None:
            doc["group_sessions"][0]["members"] = []
        self._assert_rejected(self._mutated(empty_members))

        def dup_members(doc: Dict[str, Any]) -> None:
            doc["group_sessions"][0]["members"] = [
                "creator", "creator", "alice"]
        self._assert_rejected(self._mutated(dup_members))

        def typed_member(doc: Dict[str, Any]) -> None:
            doc["group_sessions"][0]["members"] = ["creator", True]
        self._assert_rejected(self._mutated(typed_member))

        def null_member(doc: Dict[str, Any]) -> None:
            doc["group_sessions"][0]["members"] = ["creator", None]
        self._assert_rejected(self._mutated(null_member))

        def empty_member(doc: Dict[str, Any]) -> None:
            doc["group_sessions"][0]["members"] = ["creator", ""]
        self._assert_rejected(self._mutated(empty_member))

        def not_a_list(doc: Dict[str, Any]) -> None:
            doc["group_sessions"][0]["members"] = "creator"
        self._assert_rejected(self._mutated(not_a_list))

    def test_initiator_must_be_registered_member(self) -> None:
        def unknown_initiator(doc: Dict[str, Any]) -> None:
            doc["group_sessions"][0]["initiator_device_id"] = "ghost"
            doc["group_sessions"][0]["members"] = [
                "ghost", "creator", "alice", "bob"]
        self._assert_rejected(self._mutated(unknown_initiator))

        def outsider_initiator(doc: Dict[str, Any]) -> None:
            # carol is registered but not frozen into this group session.
            doc["group_sessions"][0]["initiator_device_id"] = "carol"
        self._assert_rejected(self._mutated(outsider_initiator))

        def initiator_not_member(doc: Dict[str, Any]) -> None:
            doc["group_sessions"][0]["initiator_device_id"] = "alice"
            doc["group_sessions"][0]["members"] = ["creator", "bob"]
        self._assert_rejected(self._mutated(initiator_not_member))

    def test_revision_bounds(self) -> None:
        for value in (0, -3, "1", True, 2.0, None):
            def mutate(doc: Dict[str, Any], value=value) -> None:
                doc["group_sessions"][0]["revision"] = value
            self._assert_rejected(self._mutated(mutate))

        def beyond_current(doc: Dict[str, Any]) -> None:
            doc["group_sessions"][0]["revision"] = 3
        self._assert_rejected(self._mutated(beyond_current))

    def test_older_equal_revision_snapshots_still_load(self) -> None:
        # revision 1 snapshot while the group sits at revision 2 (fixture),
        # and a revision-equal snapshot, are both legitimate freezes.
        for revision in (1, 2):
            document = copy.deepcopy(self.good_document)
            document["group_sessions"][0]["revision"] = revision
            path = self._write(document, f"rev-{revision}.json")
            service = DeviceService()
            attach_persistence(service, path)
            stored = next(iter(service.store._group_sessions.values()))
            self.assertEqual(stored.revision, revision)

    def test_string_fields_reject_wrong_types(self) -> None:
        fields = ("session_id", "group_id", "initiator_device_id",
                  "ephemeral_key", "created_at")
        for field in fields:
            for value in (True, 5, None, ""):
                def mutate(doc: Dict[str, Any], field=field,
                           value=value) -> None:
                    doc["group_sessions"][0][field] = value
                self._assert_rejected(self._mutated(mutate))


class LegacyFilesStillLoadTest(_ConsistencyBase):
    """Old version=1 files missing later-added optional fields still load."""

    def test_without_rotated_at_loads(self) -> None:
        def mutate(doc: Dict[str, Any]) -> None:
            for device in doc["devices"]:
                device["rotated_at"] = device["registered_at"]
                del device["rotated_at"]
        document = self._mutated(mutate)
        path = self._write(document, "legacy-rotated.json")
        service = DeviceService()
        attach_persistence(service, path)
        self.assertEqual(
            service.store.find_by_device_id("alice").rotated_at,
            self.good_document["devices"][1]["registered_at"])

    def test_without_used_nonces_and_cursors_loads(self) -> None:
        def mutate(doc: Dict[str, Any]) -> None:
            del doc["used_nonces"]
            del doc["group_sync_cursors"]
        document = self._mutated(mutate)
        path = self._write(document, "legacy-sections.json")
        service = DeviceService()
        attach_persistence(service, path)
        self.assertEqual(service.store._group_sync_cursors, {})
        self.assertEqual(service.store._used_nonces, {})


class ServeRefusesSemanticallyBadFileTest(_ConsistencyBase):
    """``serve --data-file`` prints one stderr JSON line, field=data_file."""

    def test_serve_rejects_bad_file_with_data_file_field(self) -> None:
        path = self._write(
            self._mutated(lambda doc: doc["sessions"][0].__setitem__(
                "recipient_device_id", "ghost")))
        before = open(path, "rb").read()
        before_ino = os.stat(path).st_ino
        result = subprocess.run(
            [sys.executable, "-m", "e2ee_backend", "serve",
             "--host", "127.0.0.1", "--port", "0", "--data-file", path],
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
