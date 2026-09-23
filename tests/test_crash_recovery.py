"""Crash-boundary recovery for the version=1 state file.

A process can die between fsync-ing a temporary snapshot and finishing the
atomic replace (or between the replace and the directory fsync). At the next
startup the same directory may therefore hold one or more ``.state-*.tmp``
snapshots beside (or instead of) the formal file:

* a valid formal file is authoritative and every leftover is removed without
  touching the formal file's bytes or inode;
* with the formal file missing, the newest-mtime leftover that parses as
  version=1 *and* passes the full semantic restore checks is atomically
  recovered into place and the other leftovers removed;
* with no valid candidate the leftovers are removed and an empty state is
  created;
* an existing-but-corrupt formal file still makes startup refuse, with both
  the file and the leftovers left untouched.
"""
import json
import os
import shutil
import tempfile
import unittest
from typing import Any, Dict, List, Tuple

from e2ee_backend.models import Device
from e2ee_backend.persistence import StateFileError, attach_persistence
from e2ee_backend.service import DeviceService


def build_fixture(directory: str,
                  cursor: int = 0) -> Tuple[DeviceService, str, str, bytes]:
    """Persist one group session with seq 1..3; return (service, path, sid, bytes)."""
    path = os.path.join(directory, "state.json")
    service = DeviceService()
    attach_persistence(service, path)
    for device_id in ("creator", "alice", "bob"):
        service.store.add_device(Device("u", device_id, "ik"))
    service.create_group({
        "group_id": "g1", "creator_device_id": "creator",
        "member_device_ids": ["alice", "bob"]})
    session = service.create_group_session({
        "group_id": "g1", "initiator_device_id": "creator",
        "ephemeral_key": "epk"})
    sid = session["session_id"]
    for sequence in range(1, 4):
        service.post_message({
            "session_id": sid, "sender_device_id": "creator",
            "message_id": f"m{sequence}", "sequence": sequence,
            "nonce": f"n{sequence}", "ciphertext": "ct"})
    if cursor:
        service.sync_group_checkpoint(
            sid, {"device_id": "alice", "cursor": cursor})
    with open(path, "rb") as handle:
        formal_bytes = handle.read()
    return service, path, sid, formal_bytes


def write_tmp(directory: str, name: str, content: bytes,
              mtime_ns: int | None = None) -> str:
    """Write a leftover temporary snapshot with an optional fixed mtime."""
    path = os.path.join(directory, name)
    with open(path, "wb") as handle:
        handle.write(content)
    if mtime_ns is not None:
        os.utime(path, ns=(mtime_ns, mtime_ns))
    return path


def tmp_names(directory: str) -> List[str]:
    """Names of crash-leftover files (staged snapshots or inode backups)."""
    return sorted(name for name in os.listdir(directory)
                  if name.startswith(".state-")
                  and (name.endswith(".tmp") or name.endswith(".bak")))


class ValidFormalWinsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        _service, self.path, self.sid, self.formal_bytes = build_fixture(
            self.directory)
        self.formal_ino = os.stat(self.path).st_ino

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_valid_formal_survives_newer_valid_leftover(self) -> None:
        # The leftover is a complete, newer-looking snapshot; it must not
        # overwrite the formal file.
        empty = json.dumps({"version": 1}).encode("utf-8")
        leftover = write_tmp(self.directory, ".state-newer.tmp", empty)
        os.utime(leftover, ns=(10**18, 10**18))
        service = DeviceService()
        attach_persistence(service, self.path)
        self.assertEqual(open(self.path, "rb").read(), self.formal_bytes)
        self.assertEqual(os.stat(self.path).st_ino, self.formal_ino)
        self.assertEqual(tmp_names(self.directory), [])
        self.assertIsNotNone(
            service.store.find_by_device_id("creator"))

    def test_valid_formal_survives_corrupt_leftovers(self) -> None:
        write_tmp(self.directory, ".state-bad1.tmp", b"{not json")
        write_tmp(self.directory, ".state-bad2.tmp",
                  json.dumps({"version": 2}).encode("utf-8"))
        service = DeviceService()
        attach_persistence(service, self.path)
        self.assertEqual(open(self.path, "rb").read(), self.formal_bytes)
        self.assertEqual(os.stat(self.path).st_ino, self.formal_ino)
        self.assertEqual(tmp_names(self.directory), [])

    def test_normal_run_leaves_no_leftovers(self) -> None:
        service = DeviceService()
        attach_persistence(service, self.path)
        self.assertEqual(tmp_names(self.directory), [])

    def test_stale_inode_backup_is_cleaned_without_touching_formal(
            self) -> None:
        # A .bak left by a crashed previous commit is a leftover like any
        # other: the valid formal file wins and the backup is removed.
        write_tmp(self.directory, ".state-old.bak", b'{"version": 1}')
        service = DeviceService()
        attach_persistence(service, self.path)
        self.assertEqual(open(self.path, "rb").read(), self.formal_bytes)
        self.assertEqual(os.stat(self.path).st_ino, self.formal_ino)
        self.assertEqual(tmp_names(self.directory), [])


class MissingFormalRecoveryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        _service, self.path, self.sid, self.formal_bytes = build_fixture(
            self.directory, cursor=2)

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_single_valid_leftover_is_atomically_recovered(self) -> None:
        os.unlink(self.path)
        leftover = write_tmp(self.directory, ".state-crash.tmp",
                             self.formal_bytes)
        leftover_ino = os.stat(leftover).st_ino
        service = DeviceService()
        attach_persistence(service, self.path)
        # Recovery is an os.replace: the leftover becomes the formal file,
        # keeping its inode and bytes.
        self.assertTrue(os.path.exists(self.path))
        self.assertEqual(os.stat(self.path).st_ino, leftover_ino)
        self.assertEqual(open(self.path, "rb").read(), self.formal_bytes)
        self.assertEqual(tmp_names(self.directory), [])
        # The recovered state is fully usable, including the saved cursor.
        body = service.sync_group_messages(self.sid, "alice", None, 100)
        self.assertEqual([m["sequence"] for m in body["messages"]], [3])

    def test_newest_mtime_valid_snapshot_is_chosen(self) -> None:
        _service_b, path_b, sid_b, bytes_b = build_fixture(
            tempfile.mkdtemp(), cursor=0)
        os.unlink(self.path)
        # Older valid snapshot (no cursor) and newer valid snapshot (cursor 2).
        write_tmp(self.directory, ".state-old.tmp", bytes_b, mtime_ns=1000)
        write_tmp(self.directory, ".state-new.tmp", self.formal_bytes,
                  mtime_ns=2000)
        service = DeviceService()
        attach_persistence(service, self.path)
        self.assertEqual(open(self.path, "rb").read(), self.formal_bytes)
        self.assertEqual(tmp_names(self.directory), [])
        self.assertEqual(
            service.store._group_sync_cursors[(self.sid, "alice")].cursor, 2)
        shutil.rmtree(os.path.dirname(path_b), ignore_errors=True)

    def test_invalid_newer_snapshot_falls_back_to_older_valid(self) -> None:
        os.unlink(self.path)
        document = json.loads(self.formal_bytes.decode("utf-8"))
        # Semantically corrupt: cursor past the session's max sequence (3).
        document["group_sync_cursors"] = [{
            "session_id": self.sid, "device_id": "alice", "cursor": 9,
            "updated_at": "2026-01-01T00:00:00+00:00"}]
        corrupt = json.dumps(document).encode("utf-8")
        write_tmp(self.directory, ".state-valid-old.tmp", self.formal_bytes,
                  mtime_ns=1000)
        write_tmp(self.directory, ".state-corrupt-new.tmp", corrupt,
                  mtime_ns=2000)
        service = DeviceService()
        attach_persistence(service, self.path)
        self.assertEqual(open(self.path, "rb").read(), self.formal_bytes)
        self.assertEqual(tmp_names(self.directory), [])
        self.assertIsNotNone(service.store.find_by_device_id("alice"))

    def test_backup_only_leftover_is_recovered(self) -> None:
        # Formal missing; the only leftover is a .bak inode backup holding the
        # previous committed document. It is a valid recovery candidate.
        os.unlink(self.path)
        write_tmp(self.directory, ".state-old.bak", self.formal_bytes)
        service = DeviceService()
        attach_persistence(service, self.path)
        self.assertEqual(open(self.path, "rb").read(), self.formal_bytes)
        self.assertEqual(tmp_names(self.directory), [])
        body = service.sync_group_messages(self.sid, "alice", None, 100)
        self.assertEqual([m["sequence"] for m in body["messages"]], [3])

    def test_syntactically_bad_leftovers_are_skipped(self) -> None:
        os.unlink(self.path)
        write_tmp(self.directory, ".state-a.tmp", b"", mtime_ns=3000)
        write_tmp(self.directory, ".state-b.tmp", b"{truncated",
                  mtime_ns=2000)
        write_tmp(self.directory, ".state-c.tmp",
                  json.dumps([1, 2, 3]).encode("utf-8"), mtime_ns=1500)
        write_tmp(self.directory, ".state-d.bak", self.formal_bytes,
                  mtime_ns=1000)
        service = DeviceService()
        attach_persistence(service, self.path)
        self.assertEqual(open(self.path, "rb").read(), self.formal_bytes)
        self.assertEqual(tmp_names(self.directory), [])

    def test_quarantined_bad_snapshot_never_outranks_good_backup(self) -> None:
        # Formal missing. A .bak holds the last committed document; a .bad
        # quarantine file holds a *failed* transaction stamped one generation
        # newer (and newer mtime). Recovery must promote the .bak and delete
        # the .bad — the aborted snapshot is never authoritative.
        os.unlink(self.path)
        good_doc = json.loads(self.formal_bytes.decode("utf-8"))
        bad_doc = dict(good_doc)
        bad_doc["commit_seq"] = good_doc["commit_seq"] + 1
        write_tmp(self.directory, ".state-failed.bad",
                  json.dumps(bad_doc).encode("utf-8"), mtime_ns=10**18)
        write_tmp(self.directory, ".state-good.bak", self.formal_bytes,
                  mtime_ns=1000)
        service = DeviceService()
        attach_persistence(service, self.path)
        self.assertEqual(open(self.path, "rb").read(), self.formal_bytes)
        self.assertEqual(sorted(os.listdir(self.directory)),
                         [os.path.basename(self.path)])
        # Resumed generations continue from the committed document, never from
        # the quarantined one.
        self.assertEqual(
            json.loads(open(self.path, encoding="utf-8").read())[
                "commit_seq"],
            good_doc["commit_seq"])


class DirectoryFsyncFailureTest(unittest.TestCase):
    """A directory-fsync failure after the rename is the same transaction."""

    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.service, self.path, self.sid, _bytes = build_fixture(
            self.directory)
        with open(self.path, "rb") as handle:
            self.good_bytes = handle.read()
        self.good_ino = os.stat(self.path).st_ino
        self.good_seq = json.loads(self.good_bytes.decode("utf-8"))[
            "commit_seq"]

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def _all_state_names(self) -> List[str]:
        """Every ``.state-*`` leftover of any suffix."""
        return sorted(name for name in os.listdir(self.directory)
                      if name.startswith(".state-"))

    def test_directory_fsync_once_rolls_back_cleanly(self) -> None:
        # The directory fsync fails on the commit but succeeds when flushing
        # the rollback: the old inode is renamed back into place, and nothing
        # is quarantined.
        import e2ee_backend.persistence as persistence_mod
        from e2ee_backend.persistence import PersistenceUnavailable

        real_fsync = persistence_mod.os.fsync
        failed_once = False

        def fail_first_directory_fsync(fd: int) -> None:  # noqa: ANN001
            nonlocal failed_once
            if os.fstat(fd).st_mode & 0o170000 == 0o040000:
                if not failed_once:
                    failed_once = True
                    raise OSError("simulated directory fsync failure")
            real_fsync(fd)

        persistence_mod.os.fsync = fail_first_directory_fsync
        try:
            with self.assertRaises(PersistenceUnavailable):
                self.service.revoke_device("alice")
        finally:
            persistence_mod.os.fsync = real_fsync
        self.assertFalse(
            self.service.store.find_by_device_id("alice").revoked)
        self.assertEqual(open(self.path, "rb").read(), self.good_bytes)
        self.assertEqual(os.stat(self.path).st_ino, self.good_ino)
        self.assertEqual(self._all_state_names(), [])

    def test_persistent_fsync_failure_preserves_bak_and_vacates_formal(
            self) -> None:
        import e2ee_backend.persistence as persistence_mod
        from e2ee_backend.persistence import PersistenceUnavailable

        real_fsync = persistence_mod.os.fsync

        def fail_on_directory_fd(fd: int) -> None:  # noqa: ANN001
            # Fail every directory fsync: neither the commit nor the rollback
            # flush can be made durable.
            if os.fstat(fd).st_mode & 0o170000 == 0o040000:
                raise OSError("simulated directory fsync failure")
            real_fsync(fd)

        persistence_mod.os.fsync = fail_on_directory_fd
        try:
            with self.assertRaises(PersistenceUnavailable):
                self.service.revoke_device("alice")
        finally:
            persistence_mod.os.fsync = real_fsync
        # Memory rolled back to the last committed state.
        self.assertFalse(
            self.service.store.find_by_device_id("alice").revoked)
        # The formal path is absent: an uncommitted snapshot can never remain
        # authoritative. The old inode survives as exactly one .bak ...
        self.assertFalse(os.path.exists(self.path))
        leftovers = self._all_state_names()
        self.assertEqual(len(leftovers), 1)
        self.assertTrue(leftovers[0].endswith(".bak"))
        backup = os.path.join(self.directory, leftovers[0])
        self.assertEqual(open(backup, "rb").read(), self.good_bytes)
        self.assertEqual(os.stat(backup).st_ino, self.good_ino)
        # ... and no quarantine snapshot (.bad) is left behind.
        self.assertFalse(any(name.endswith(".bad")
                             for name in leftovers))

        # Restart: the missing formal file is recovered from the preserved
        # backup (the failed newer-generation snapshot is gone), and the
        # retried revocation commits exactly one continuous new generation.
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        self.assertEqual(open(self.path, "rb").read(), self.good_bytes)
        self.assertEqual(os.stat(self.path).st_ino, self.good_ino)
        self.assertEqual(self._all_state_names(), [])
        restarted.revoke_device("alice")
        self.assertTrue(
            restarted.store.find_by_device_id("alice").revoked)
        with open(self.path, encoding="utf-8") as handle:
            self.assertEqual(json.load(handle)["commit_seq"],
                             self.good_seq + 1)

    def test_rollback_rename_failure_preserves_bak_and_vacates_formal(
            self) -> None:
        import e2ee_backend.persistence as persistence_mod
        from e2ee_backend.persistence import PersistenceUnavailable

        real_fsync = persistence_mod.os.fsync
        real_replace = persistence_mod.os.replace
        calls = {"replace": 0}

        def fail_on_directory_fd(fd: int) -> None:  # noqa: ANN001
            # The post-replace directory fsync cannot be made durable, so the
            # transaction aborts and the rollback rename is attempted.
            if os.fstat(fd).st_mode & 0o170000 == 0o040000:
                raise OSError("simulated directory fsync failure")
            real_fsync(fd)

        def fail_only_rollback_rename(src, dst):  # noqa: ANN001
            # 1st os.replace = the commit (allow); 2nd = the rollback rename
            # of the .bak back over the target (fail). Any later rename
            # (quarantine) succeeds.
            calls["replace"] += 1
            if calls["replace"] == 2:
                raise OSError("simulated rollback rename failure")
            return real_replace(src, dst)

        persistence_mod.os.fsync = fail_on_directory_fd
        persistence_mod.os.replace = fail_only_rollback_rename
        try:
            with self.assertRaises(PersistenceUnavailable):
                self.service.revoke_device("alice")
        finally:
            persistence_mod.os.replace = real_replace
            persistence_mod.os.fsync = real_fsync
        self.assertFalse(
            self.service.store.find_by_device_id("alice").revoked)
        # Formal path vacated; the original .bak (old inode) is preserved and
        # the failed snapshot was isolated and dropped, not promoted.
        self.assertFalse(os.path.exists(self.path))
        leftovers = self._all_state_names()
        self.assertEqual(len(leftovers), 1)
        self.assertTrue(leftovers[0].endswith(".bak"))
        backup = os.path.join(self.directory, leftovers[0])
        self.assertEqual(open(backup, "rb").read(), self.good_bytes)
        self.assertEqual(os.stat(backup).st_ino, self.good_ino)

        # Restart recovery promotes the preserved last-known-good backup; the
        # retried revocation then lands as a single new generation.
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        self.assertEqual(open(self.path, "rb").read(), self.good_bytes)
        self.assertEqual(self._all_state_names(), [])
        restarted.revoke_device("alice")
        self.assertTrue(
            restarted.store.find_by_device_id("alice").revoked)
        with open(self.path, encoding="utf-8") as handle:
            self.assertEqual(json.load(handle)["commit_seq"],
                             self.good_seq + 1)

    def test_failed_commit_consumes_no_generation_within_one_process(
            self) -> None:
        # After a definitive abort vacates the formal file, a repaired disk in
        # the SAME process recreates it at the identical generation: the two
        # failed attempts consumed no generation, so the retry writes exactly
        # one continuous new generation.
        import e2ee_backend.persistence as persistence_mod
        from e2ee_backend.persistence import PersistenceUnavailable

        real_fsync = persistence_mod.os.fsync
        failing = {"on": True}

        def fail_directory_while_on(fd: int) -> None:  # noqa: ANN001
            if failing["on"] and os.fstat(fd).st_mode & 0o170000 == 0o040000:
                raise OSError("simulated directory fsync failure")
            real_fsync(fd)

        persistence_mod.os.fsync = fail_directory_while_on
        try:
            with self.assertRaises(PersistenceUnavailable):
                self.service.revoke_device("alice")
            with self.assertRaises(PersistenceUnavailable):
                self.service.revoke_device("alice")
        finally:
            persistence_mod.os.fsync = real_fsync
        failing["on"] = False
        # Repaired retry in the same process recreates the missing file and
        # commits once; the stale pre-failure backup is only swept at the next
        # startup (a now-valid formal file wins and removes leftovers).
        body = self.service.revoke_device("alice")
        self.assertEqual(body["revoked"], True)
        with open(self.path, encoding="utf-8") as handle:
            document = json.load(handle)
        self.assertEqual(document["commit_seq"], self.good_seq + 1)
        self.assertTrue(
            self.service.store.find_by_device_id("alice").revoked)

        # A restart sees the valid formal file (strictly newer than the stale
        # backup), keeps it authoritative and sweeps the leftover.
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        self.assertEqual(self._all_state_names(), [])
        self.assertTrue(
            restarted.store.find_by_device_id("alice").revoked)
        with open(self.path, encoding="utf-8") as handle:
            self.assertEqual(json.load(handle)["commit_seq"],
                             self.good_seq + 1)


class NoValidSnapshotFallbackTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        _service, self.path, _sid, _bytes = build_fixture(self.directory)
        os.unlink(self.path)

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_empty_state_is_created_and_leftovers_removed(self) -> None:
        write_tmp(self.directory, ".state-a.tmp", b"{nope")
        write_tmp(self.directory, ".state-b.tmp",
                  json.dumps({"version": 1, "group_sync_cursors": "bad"})
                  .encode("utf-8"))
        service = DeviceService()
        attach_persistence(service, self.path)
        self.assertTrue(os.path.exists(self.path))
        self.assertEqual(tmp_names(self.directory), [])
        snapshot = service.store.snapshot_state()
        self.assertEqual(snapshot["devices"], [])
        self.assertEqual(snapshot["group_sync_cursors"], [])
        # The freshly created file is itself a valid version-1 document.
        with open(self.path, encoding="utf-8") as handle:
            document = json.load(handle)
        self.assertEqual(document["version"], 1)

    def test_missing_file_without_leftovers_creates_empty_state(self) -> None:
        service = DeviceService()
        attach_persistence(service, self.path)
        self.assertTrue(os.path.exists(self.path))
        self.assertEqual(service.store.snapshot_state()["devices"], [])


class CorruptFormalRefusesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        _service, self.path, _sid, self.formal_bytes = build_fixture(
            self.directory)

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_corrupt_formal_with_valid_leftover_refuses_and_keeps_both(
            self) -> None:
        with open(self.path, "wb") as handle:
            handle.write(b"{corrupt")
        before = open(self.path, "rb").read()
        before_ino = os.stat(self.path).st_ino
        leftover = write_tmp(self.directory, ".state-orphan.tmp",
                             self.formal_bytes)
        with self.assertRaises(StateFileError):
            attach_persistence(DeviceService(), self.path)
        # Present state is never discarded: the corrupt formal is untouched,
        # and the leftover is not moved or deleted.
        self.assertEqual(open(self.path, "rb").read(), before)
        self.assertEqual(os.stat(self.path).st_ino, before_ino)
        self.assertTrue(os.path.exists(leftover))
        self.assertEqual(open(leftover, "rb").read(), self.formal_bytes)

    def test_semantically_bad_formal_refuses_despite_leftover(self) -> None:
        document = json.loads(self.formal_bytes.decode("utf-8"))
        document["messages"][next(iter(document["messages"]))][0][
            "sequence"] = 5
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        before = open(self.path, "rb").read()
        write_tmp(self.directory, ".state-orphan.tmp", self.formal_bytes)
        with self.assertRaises(StateFileError):
            attach_persistence(DeviceService(), self.path)
        self.assertEqual(open(self.path, "rb").read(), before)
        self.assertEqual(tmp_names(self.directory), [".state-orphan.tmp"])


if __name__ == "__main__":
    unittest.main()
