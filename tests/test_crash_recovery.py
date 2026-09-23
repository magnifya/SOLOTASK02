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


def bak_names(directory: str) -> List[str]:
    """Names of pinned-inode backups only."""
    return sorted(name for name in os.listdir(directory)
                  if name.startswith(".state-") and name.endswith(".bak"))


def quarantine_names(directory: str) -> List[str]:
    """Names of new snapshots demoted off the formal path."""
    return sorted(name for name in os.listdir(directory)
                  if name.startswith(".state-")
                  and name.endswith(".quarantine"))


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

    def test_directory_fsync_failure_rolls_back_and_reports_503_field(
            self) -> None:
        # The directory fsync stays broken, so the rollback rename can land
        # but the second directory fsync (flushing the rollback) fails too.
        # That is a failed rollback step: the store must keep the old inode
        # pinned in a .bak and vacate the formal path, rather than leave the
        # un-committed new snapshot authoritative.
        import e2ee_backend.persistence as persistence_mod
        from e2ee_backend.persistence import PersistenceUnavailable

        real_fsync = persistence_mod.os.fsync

        def fail_on_directory_fd(fd: int) -> None:  # noqa: ANN001
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
        # The failed transaction is decidable: the formal path is missing,
        # the old inode survives in exactly one .bak, and no new snapshot is
        # parked where recovery could promote it.
        self.assertFalse(os.path.exists(self.path))
        baks = bak_names(self.directory)
        self.assertEqual(len(baks), 1)
        with open(os.path.join(self.directory, baks[0]), "rb") as handle:
            self.assertEqual(handle.read(), self.good_bytes)
        self.assertEqual(
            os.stat(os.path.join(self.directory, baks[0])).st_ino,
            self.good_ino)
        self.assertEqual(quarantine_names(self.directory), [])

    def test_transient_directory_fsync_failure_rolls_back_in_place(self) -> None:
        import e2ee_backend.persistence as persistence_mod
        from e2ee_backend.persistence import PersistenceUnavailable

        real_fsync = persistence_mod.os.fsync
        calls = {"n": 0}

        def fail_first_directory_fsync(fd: int) -> None:  # noqa: ANN001
            if os.fstat(fd).st_mode & 0o170000 == 0o040000:
                calls["n"] += 1
                if calls["n"] == 1:
                    raise OSError("transient directory fsync failure")
            real_fsync(fd)

        persistence_mod.os.fsync = fail_first_directory_fsync
        try:
            with self.assertRaises(PersistenceUnavailable):
                self.service.revoke_device("alice")
        finally:
            persistence_mod.os.fsync = real_fsync
        # The rollback rename and its follow-up fsync both landed: the old
        # inode is back in place and nothing is left behind.
        self.assertEqual(open(self.path, "rb").read(), self.good_bytes)
        self.assertEqual(os.stat(self.path).st_ino, self.good_ino)
        self.assertEqual(tmp_names(self.directory), [])
        # The store is not degraded: an immediate same-process retry commits
        # normally, advancing exactly one generation.
        self.service.revoke_device("alice")
        self.assertEqual(
            json.loads(open(self.path, encoding="utf-8").read())[
                "commit_seq"],
            self.good_seq + 1)
        self.assertTrue(
            self.service.store.find_by_device_id("alice").revoked)

    def test_same_process_retry_is_refused_while_degraded(self) -> None:
        import e2ee_backend.persistence as persistence_mod
        from e2ee_backend.persistence import PersistenceUnavailable

        real_fsync = persistence_mod.os.fsync

        def fail_on_directory_fd(fd: int) -> None:  # noqa: ANN001
            if os.fstat(fd).st_mode & 0o170000 == 0o040000:
                raise OSError("simulated directory fsync failure")
            real_fsync(fd)

        persistence_mod.os.fsync = fail_on_directory_fd
        try:
            with self.assertRaises(PersistenceUnavailable):
                self.service.revoke_device("alice")
        finally:
            persistence_mod.os.fsync = real_fsync

        baks_before = bak_names(self.directory)
        self.assertEqual(len(baks_before), 1)
        # Even with storage healthy again, the degraded process refuses the
        # write (a 503 at the HTTP boundary) rather than risk an un-backed
        # replace over the missing formal path; nothing on disk changes.
        with self.assertRaises(PersistenceUnavailable):
            self.service.revoke_device("alice")
        self.assertFalse(os.path.exists(self.path))
        self.assertEqual(bak_names(self.directory), baks_before)
        self.assertEqual(quarantine_names(self.directory), [])
        self.assertFalse(
            self.service.store.find_by_device_id("alice").revoked)

    def test_restart_recovers_pinned_backup_and_retry_commits_once(
            self) -> None:
        import e2ee_backend.persistence as persistence_mod
        from e2ee_backend.persistence import PersistenceUnavailable

        real_fsync = persistence_mod.os.fsync

        def fail_on_directory_fd(fd: int) -> None:  # noqa: ANN001
            if os.fstat(fd).st_mode & 0o170000 == 0o040000:
                raise OSError("simulated directory fsync failure")
            real_fsync(fd)

        persistence_mod.os.fsync = fail_on_directory_fd
        try:
            with self.assertRaises(PersistenceUnavailable):
                self.service.revoke_device("alice")
            # A retry while the failure persists still consumes no generation.
            with self.assertRaises(PersistenceUnavailable):
                self.service.revoke_device("alice")
        finally:
            persistence_mod.os.fsync = real_fsync

        # Restart with storage healthy again: the pinned old inode is the
        # highest/only candidate and is recovered into the formal path.
        service = DeviceService()
        attach_persistence(service, self.path)
        self.assertEqual(open(self.path, "rb").read(), self.good_bytes)
        self.assertEqual(os.stat(self.path).st_ino, self.good_ino)
        self.assertEqual(tmp_names(self.directory), [])
        self.assertEqual(
            json.loads(open(self.path, encoding="utf-8").read())[
                "commit_seq"],
            self.good_seq)
        self.assertFalse(service.store.find_by_device_id("alice").revoked)

        # The recovered retry commits exactly one consecutive new generation.
        service.revoke_device("alice")
        self.assertEqual(
            json.loads(open(self.path, encoding="utf-8").read())[
                "commit_seq"],
            self.good_seq + 1)
        self.assertTrue(service.store.find_by_device_id("alice").revoked)
        self.assertEqual(tmp_names(self.directory), [])


class RollbackRenameFailureTest(unittest.TestCase):
    """The rollback rename itself failing must stay decidable."""

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

    def test_rename_failure_keeps_backup_and_vacates_new_snapshot(self) -> None:
        import e2ee_backend.persistence as persistence_mod
        from e2ee_backend.persistence import PersistenceUnavailable

        real_fsync = persistence_mod.os.fsync
        real_replace = persistence_mod.os.replace
        target = os.path.abspath(self.path)

        def fail_on_directory_fd(fd: int) -> None:  # noqa: ANN001
            if os.fstat(fd).st_mode & 0o170000 == 0o040000:
                raise OSError("simulated post-replace fsync failure")
            real_fsync(fd)

        def fail_backup_rename(src: str, dst: str) -> None:
            # Fail only the rollback rename (pinned .bak back over the
            # target); the initial tmp->target replace and the later rename
            # parking the new snapshot must go through.
            if os.path.abspath(dst) == target and src.endswith(".bak"):
                raise OSError("simulated rollback rename failure")
            return real_replace(src, dst)

        persistence_mod.os.fsync = fail_on_directory_fd
        persistence_mod.os.replace = fail_backup_rename
        try:
            with self.assertRaises(PersistenceUnavailable):
                self.service.revoke_device("alice")
        finally:
            persistence_mod.os.fsync = real_fsync
            persistence_mod.os.replace = real_replace

        # The old inode is pinned in the .bak; the formal path no longer
        # holds the un-committed new snapshot (it is quarantined or gone).
        self.assertFalse(os.path.exists(self.path))
        self.assertFalse(
            self.service.store.find_by_device_id("alice").revoked)
        baks = bak_names(self.directory)
        self.assertEqual(len(baks), 1)
        backup_path = os.path.join(self.directory, baks[0])
        self.assertEqual(open(backup_path, "rb").read(), self.good_bytes)
        self.assertEqual(os.stat(backup_path).st_ino, self.good_ino)

        # Restart restores the pinned inode; the quarantined new snapshot is
        # removed, never promoted, and the retry is one fresh generation.
        service = DeviceService()
        attach_persistence(service, self.path)
        self.assertEqual(open(self.path, "rb").read(), self.good_bytes)
        self.assertEqual(os.stat(self.path).st_ino, self.good_ino)
        self.assertEqual(tmp_names(self.directory), [])
        self.assertEqual(quarantine_names(self.directory), [])
        service.revoke_device("alice")
        self.assertEqual(
            json.loads(open(self.path, encoding="utf-8").read())[
                "commit_seq"],
            self.good_seq + 1)
        self.assertTrue(service.store.find_by_device_id("alice").revoked)


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
