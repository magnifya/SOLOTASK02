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


def block_names(directory: str) -> List[str]:
    """Names of durable blocking-state markers."""
    return sorted(name for name in os.listdir(directory)
                  if name.startswith(".state-")
                  and name.endswith(".block"))


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

    def test_same_process_next_write_self_heals_and_commits_once(self) -> None:
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
                # This is the request that gets 503 (revoke "alice"); it must
                # never be replayed by the self-heal below.
                self.service.revoke_device("alice")
        finally:
            persistence_mod.os.fsync = real_fsync

        self.assertEqual(len(bak_names(self.directory)), 1)
        self.assertFalse(os.path.exists(self.path))

        # Storage is healthy again. The next persist-able write is a
        # *different*, current request (revoke "bob"), not a replay of the
        # 503ed revoke("alice"). It heals the pinned backup inside the store
        # lock and then commits exactly one new generation.
        self.service.revoke_device("bob")
        self.assertTrue(os.path.exists(self.path))
        self.assertEqual(tmp_names(self.directory), [])
        self.assertEqual(quarantine_names(self.directory), [])
        # Exactly one consecutive generation was added.
        self.assertEqual(
            json.loads(open(self.path, encoding="utf-8").read())[
                "commit_seq"],
            self.good_seq + 1)
        # The old request was not replayed: alice is still active, while the
        # current request (bob) is revoked and durable.
        self.assertFalse(
            self.service.store.find_by_device_id("alice").revoked)
        self.assertTrue(
            self.service.store.find_by_device_id("bob").revoked)
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        self.assertFalse(restarted.store.find_by_device_id("alice").revoked)
        self.assertTrue(restarted.store.find_by_device_id("bob").revoked)
        # A further write after the heal commits normally, another single step.
        restarted.revoke_device("alice")
        self.assertEqual(
            json.loads(open(self.path, encoding="utf-8").read())[
                "commit_seq"],
            self.good_seq + 2)

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


class BlockingStateTest(unittest.TestCase):
    """The undecidable failure that cannot even vacate the formal path.

    When the post-replace fsync fails, the rollback rename fails,
    and the un-committed new snapshot can be neither quarantined nor
    deleted, the store cannot guarantee a missing formal path with a unique
    backup. It must then enter the *blocking* state: leave a durable
    ``.block`` marker, never serve the residual formal file as
    authoritative, and answer 503/field=data_file for every later write
    in this process and after a restart — until the path is vacated and a
    unique verifiable backup can be promoted.
    """

    def setUp(self) -> None:
        import e2ee_backend.persistence as persistence_mod
        self.persistence_mod = persistence_mod
        self._real_fsync = persistence_mod.os.fsync
        self._real_replace = persistence_mod.os.replace
        self._real_unlink = persistence_mod.os.unlink
        self.directory = tempfile.mkdtemp()
        self.path = os.path.join(self.directory, "state.json")
        self.service = DeviceService()
        self.state_store = attach_persistence(self.service, self.path)
        for device_id in ("creator", "alice", "bob"):
            self.service.store.add_device(Device("u", device_id, "ik"))
        with open(self.path, "rb") as handle:
            self.good_bytes = handle.read()
        self.good_ino = os.stat(self.path).st_ino
        self.good_seq = json.loads(self.good_bytes.decode("utf-8"))[
            "commit_seq"]

    def tearDown(self) -> None:
        self._restore_io()
        shutil.rmtree(self.directory, ignore_errors=True)

    def _restore_io(self) -> None:
        self.persistence_mod.os.fsync = self._real_fsync
        self.persistence_mod.os.replace = self._real_replace
        self.persistence_mod.os.unlink = self._real_unlink

    def _induce_blocked(self, device_id: str = "alice") -> None:
        """Force the formal path to keep the un-committed residual file."""
        from e2ee_backend.persistence import PersistenceUnavailable
        real_fsync = self._real_fsync
        real_replace = self._real_replace
        real_unlink = self._real_unlink
        target = os.path.abspath(self.path)

        def fail_on_directory_fd(fd: int) -> None:  # noqa: ANN001
            if os.fstat(fd).st_mode & 0o170000 == 0o040000:
                raise OSError("simulated post-replace fsync failure")
            real_fsync(fd)

        def replace_patch(src: str, dst: str) -> None:
            s = os.path.abspath(src)
            d = os.path.abspath(dst)
            rollback = s.endswith(".bak") and d == target
            quarantine = s == target and d.endswith(".quarantine")
            if rollback or quarantine:
                raise OSError("simulated blocked rollback/quarantine rename")
            return real_replace(src, dst)

        def unlink_patch(path: str) -> None:
            if os.path.abspath(path) == target:
                raise OSError("simulated blocked formal unlink")
            return real_unlink(path)

        self.persistence_mod.os.fsync = fail_on_directory_fd
        self.persistence_mod.os.replace = replace_patch
        self.persistence_mod.os.unlink = unlink_patch
        try:
            with self.assertRaises(PersistenceUnavailable):
                self.service.revoke_device(device_id)
        finally:
            self._restore_io()

    def test_blocked_residual_is_never_authoritative_and_writes_503(
            self) -> None:
        from e2ee_backend.persistence import PersistenceUnavailable
        self._induce_blocked("alice")
        # The formal path survives, holding the un-committed snapshot...
        self.assertTrue(os.path.exists(self.path))
        residual = json.loads(open(self.path, encoding="utf-8").read())
        self.assertTrue(any(
            d["device_id"] == "alice" and d.get("revoked")
            for d in residual["devices"]))
        self.assertNotEqual(open(self.path, "rb").read(),
                            self.good_bytes)
        # ...but memory rolled back, so the residual is not authoritative...
        self.assertFalse(
            self.service.store.find_by_device_id("alice").revoked)
        # ...a durable marker and a pinned committed backup are present...
        self.assertEqual(len(block_names(self.directory)), 1)
        baks = bak_names(self.directory)
        self.assertEqual(len(baks), 1)
        self.assertEqual(
            open(os.path.join(self.directory, baks[0]), "rb").read(),
            self.good_bytes)
        # ...and the store is blocked (not merely degraded).
        self.assertTrue(self.state_store.blocked)
        self.assertFalse(self.state_store.degraded)

        # Every later write is 503 and moves nothing: the residual formal
        # bytes/inode are untouched and never become authoritative.
        before = open(self.path, "rb").read()
        before_ino = os.stat(self.path).st_ino
        for _ in range(2):
            with self.assertRaises(PersistenceUnavailable):
                self.service.revoke_device("bob")
        self.assertEqual(open(self.path, "rb").read(), before)
        self.assertEqual(os.stat(self.path).st_ino, before_ino)
        self.assertTrue(self.state_store.blocked)
        self.assertEqual(len(block_names(self.directory)), 1)
        self.assertFalse(
            self.service.store.find_by_device_id("alice").revoked)
        self.assertFalse(
            self.service.store.find_by_device_id("bob").revoked)

    def test_block_heals_once_path_vacated_unique_backup_promoted(
            self) -> None:
        from e2ee_backend.persistence import PersistenceUnavailable
        self._induce_blocked("alice")
        residual_ino = os.stat(self.path).st_ino
        # A write while still blocked is refused and leaves the residual in place.
        with self.assertRaises(PersistenceUnavailable):
            self.service.revoke_device("bob")
        self.assertEqual(os.stat(self.path).st_ino, residual_ino)
        # Operator resolves the on-disk state by vacating the un-decidable
        # residual; the unique pinned committed backup remains.
        os.unlink(self.path)
        # The next write heals inside the lock (promotes the unique pin
        # via hard link, clears marker/backup) and then commits the
        # current request once; the 503ed request is not replayed.
        self.service.revoke_device("bob")
        self.assertTrue(os.path.exists(self.path))
        self.assertEqual(block_names(self.directory), [])
        self.assertEqual(tmp_names(self.directory), [])
        self.assertEqual(self._formal()["commit_seq"], self.good_seq + 1)
        self.assertTrue(
            self.service.store.find_by_device_id("bob").revoked)
        self.assertFalse(
            self.service.store.find_by_device_id("alice").revoked)
        self.assertFalse(self.state_store.blocked)
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        self.assertTrue(restarted.store.find_by_device_id("bob").revoked)
        self.assertFalse(
            restarted.store.find_by_device_id("alice").revoked)

    def test_restart_refuses_while_block_marker_and_residual_present(self) -> None:
        from e2ee_backend.persistence import StateFileError
        self._induce_blocked("alice")
        marker = os.path.join(self.directory,
                               block_names(self.directory)[0])
        residual_before = open(self.path, "rb").read()
        # A restart keeps refusing: the residual formal is never accepted.
        with self.assertRaises(StateFileError):
            attach_persistence(DeviceService(), self.path)
        self.assertEqual(open(self.path, "rb").read(), residual_before)
        self.assertTrue(os.path.exists(marker))
        self.assertEqual(len(bak_names(self.directory)), 1)

    def test_restart_unblocks_when_formal_matches_unique_backup(self) -> None:
        # A valid formal file always takes precedence, even across a restart
        # with a .block marker present: restore the committed bytes onto the
        # formal path; it semantically equals the unique pin, so startup
        # accepts it, sweeps marker/pin and serves normally.
        from e2ee_backend.persistence import StateFileError
        self._induce_blocked("alice")
        with open(self.path, "wb") as handle:
            handle.write(self.good_bytes)
        service = DeviceService()
        attach_persistence(service, self.path)
        self.assertEqual(open(self.path, "rb").read(), self.good_bytes)
        self.assertEqual(block_names(self.directory), [])
        self.assertEqual(tmp_names(self.directory), [])
        self.assertFalse(
            service.store.find_by_device_id("alice").revoked)
        service.revoke_device("alice")
        self.assertEqual(
            json.loads(open(self.path, encoding="utf-8").read())[
                "commit_seq"],
            self.good_seq + 1)

    def test_restart_unblocks_from_unique_backup_after_path_vacated(self) -> None:
        from e2ee_backend.persistence import StateFileError
        self._induce_blocked("alice")
        # Merely deleting the residual is not enough while the marker sees a
        # formal file present elsewhere; here the operator vacates it and the
        # unique committed backup is recoverable on restart.
        os.unlink(self.path)
        service = DeviceService()
        attach_persistence(service, self.path)
        self.assertEqual(open(self.path, "rb").read(), self.good_bytes)
        self.assertEqual(block_names(self.directory), [])
        self.assertEqual(tmp_names(self.directory), [])
        self.assertFalse(
            service.store.find_by_device_id("alice").revoked)
        service.revoke_device("alice")
        self.assertEqual(
            json.loads(open(self.path, encoding="utf-8").read())[
                "commit_seq"],
            self.good_seq + 1)

    def test_restart_blocked_with_two_backups_still_refuses(self) -> None:
        from e2ee_backend.persistence import StateFileError
        self._induce_blocked("alice")
        os.unlink(self.path)
        # Two verifiable backups of the committed state make promotion
        # ambiguous; the block must stay and startup must keep refusing.
        twin = os.path.join(self.directory, ".state-twin.bak")
        with open(twin, "wb") as handle:
            handle.write(self.good_bytes)
        with self.assertRaises(StateFileError):
            attach_persistence(DeviceService(), self.path)
        self.assertFalse(os.path.exists(self.path))
        self.assertEqual(len(block_names(self.directory)), 1)
        self.assertEqual(len(bak_names(self.directory)), 2)

    def test_restart_blocked_committed_formal_wins_with_two_backups(self) -> None:
        # A valid formal always takes precedence even with two backups present:
        # restore the committed bytes onto the formal path; it matches the
        # backups' generation and payload, so startup accepts it and sweeps
        # marker and both backups, despite the ambiguous-backup situation.
        self._induce_blocked("alice")
        with open(os.path.join(self.directory, ".state-twin.bak"), "wb") as h:
            h.write(self.good_bytes)
        with open(self.path, "wb") as handle:
            handle.write(self.good_bytes)
        service = DeviceService()
        attach_persistence(service, self.path)
        self.assertEqual(open(self.path, "rb").read(), self.good_bytes)
        self.assertEqual(block_names(self.directory), [])
        self.assertEqual(bak_names(self.directory), [])
        self.assertFalse(
            service.store.find_by_device_id("alice").revoked)
        service.revoke_device("alice")
        self.assertEqual(
            json.loads(open(self.path, encoding="utf-8").read())[
                "commit_seq"],
            self.good_seq + 1)

    def test_committed_formal_wins_even_while_blocked(self) -> None:
        # A valid formal file always takes precedence. While blocked, restore the
        # last committed bytes onto the formal path (an operator resolving the
        # incident): the un-committed residual (next generation) is gone,
        # so the next write must accept the committed formal, clear marker and
        # backup, unblock and commit the current request once.
        self._induce_blocked("alice")
        with open(self.path, "wb") as handle:
            handle.write(self.good_bytes)
        self.service.revoke_device("bob")
        self.assertTrue(os.path.exists(self.path))
        self.assertEqual(block_names(self.directory), [])
        self.assertEqual(tmp_names(self.directory), [])
        self.assertEqual(self._formal()["commit_seq"], self.good_seq + 1)
        self.assertTrue(
            self.service.store.find_by_device_id("bob").revoked)
        self.assertFalse(
            self.service.store.find_by_device_id("alice").revoked)
        self.assertFalse(self.state_store.blocked)
        self.assertFalse(self.state_store.degraded)

    def _formal(self) -> Dict[str, Any]:
        with open(self.path, encoding="utf-8") as handle:
            return json.load(handle)


class SameProcessSelfHealTest(unittest.TestCase):
    """Same-process self-heal after an undecidable write failure.

    A transaction that leaves the formal path missing, the last committed
    inode pinned as the sole ``.bak`` and the un-committed snapshot in
    ``.quarantine`` is reported 503. The *next* persist-able write in the
    same process must, inside the storage lock, fully verify and atomically
    promote the backup, sweep the same-transaction leftovers, and then commit
    only that current request — one consecutive ``commit_seq`` step, never a
    replay of the request that already got 503. A failed heal or a failed
    current write keeps 503 and moves nothing, yet stays retryable.
    """

    def setUp(self) -> None:
        import e2ee_backend.persistence as persistence_mod
        self.persistence_mod = persistence_mod
        self._real_fsync = persistence_mod.os.fsync
        self._real_replace = persistence_mod.os.replace
        self.directory = tempfile.mkdtemp()
        self.path = os.path.join(self.directory, "state.json")
        self.service = DeviceService()
        self.state_store = attach_persistence(self.service, self.path)
        for device_id in ("creator", "alice", "bob"):
            self.service.store.add_device(Device("u", device_id, "ik"))
        self.service.create_group({
            "group_id": "g1", "creator_device_id": "creator",
            "member_device_ids": ["alice", "bob"]})
        session = self.service.create_group_session({
            "group_id": "g1", "initiator_device_id": "creator",
            "ephemeral_key": "epk"})
        self.sid = session["session_id"]
        for sequence in range(1, 4):
            self.service.post_message({
                "session_id": self.sid, "sender_device_id": "creator",
                "message_id": f"m{sequence}", "sequence": sequence,
                "nonce": f"n{sequence}", "ciphertext": "ct"})
        self.service.sync_group_checkpoint(
            self.sid, {"device_id": "alice", "cursor": 2})
        with open(self.path, "rb") as handle:
            self.good_bytes = handle.read()
        self.good_seq = json.loads(self.good_bytes.decode("utf-8"))[
            "commit_seq"]

    def tearDown(self) -> None:
        self.persistence_mod.os.fsync = self._real_fsync
        self.persistence_mod.os.replace = self._real_replace
        shutil.rmtree(self.directory, ignore_errors=True)

    def _fail_directory_fsync(self, *, fail_rollback_rename: bool = False,
                              ) -> None:
        real_fsync = self._real_fsync
        real_replace = self._real_replace
        target = os.path.abspath(self.path)

        def fail_on_directory_fd(fd: int) -> None:  # noqa: ANN001
            if os.fstat(fd).st_mode & 0o170000 == 0o040000:
                raise OSError("simulated directory fsync failure")
            real_fsync(fd)

        self.persistence_mod.os.fsync = fail_on_directory_fd
        if fail_rollback_rename:
            def fail_backup_rename(src: str, dst: str) -> None:
                if os.path.abspath(dst) == target and src.endswith(".bak"):
                    raise OSError("simulated rollback rename failure")
                return real_replace(src, dst)

            self.persistence_mod.os.replace = fail_backup_rename

    def _restore_io(self) -> None:
        self.persistence_mod.os.fsync = self._real_fsync
        self.persistence_mod.os.replace = self._real_replace

    def _induce_degraded(self, device_id: str = "alice", *,
                         with_quarantine: bool = False) -> str:
        """Drive one write into the undecidable failure; return the .bak.

        With *with_quarantine* the failure is the rollback-rename variant,
        which also parks the un-committed new snapshot in a ``.quarantine``
        file; otherwise it is the pure post-replace fsync variant, which
        leaves the formal path missing with exactly one ``.bak`` and no
        quarantine file.
        """
        from e2ee_backend.persistence import PersistenceUnavailable
        self.assertFalse(self.state_store.degraded)
        self._fail_directory_fsync(
            fail_rollback_rename=with_quarantine)
        try:
            with self.assertRaises(PersistenceUnavailable):
                self.service.revoke_device(device_id)
        finally:
            self._restore_io()
        baks = bak_names(self.directory)
        self.assertEqual(len(baks), 1)
        self.assertEqual(len(quarantine_names(self.directory)),
                         1 if with_quarantine else 0)
        self.assertFalse(os.path.exists(self.path))
        self.assertTrue(self.state_store.degraded)
        return os.path.join(self.directory, baks[0])

    def _formal_doc(self) -> Dict[str, Any]:
        with open(self.path, encoding="utf-8") as handle:
            return json.load(handle)

    def test_heal_promotes_backup_sweeps_leftovers_commits_current(
            self) -> None:
        backup = self._induce_degraded("alice", with_quarantine=True)
        backup_ino = os.stat(backup).st_ino
        self.service.revoke_device("bob")
        self.assertTrue(os.path.exists(self.path))
        self.assertEqual(tmp_names(self.directory), [])
        self.assertEqual(quarantine_names(self.directory), [])
        self.assertEqual(self._formal_doc()["commit_seq"], self.good_seq + 1)
        # Current request applied; the 503 request was never replayed.
        self.assertTrue(self.service.store.find_by_device_id("bob").revoked)
        self.assertFalse(
            self.service.store.find_by_device_id("alice").revoked)

    def test_heal_alone_hard_links_backup_without_advancing_generation(
            self) -> None:
        # Drive heal() directly to observe the promotion instant: the formal
        # path must name the pinned inode via a hard link (same inode and
        # bytes), no leftover may remain and no generation may be consumed —
        # the following request's save() is what advances commit_seq.
        import copy
        backup = self._induce_degraded("alice", with_quarantine=True)
        backup_ino = os.stat(backup).st_ino
        baseline = copy.deepcopy(self.service.store.snapshot_state())
        self.state_store.heal(baseline)
        self.assertFalse(self.state_store.degraded)
        self.assertTrue(os.path.exists(self.path))
        self.assertEqual(os.stat(self.path).st_ino, backup_ino)
        self.assertEqual(open(self.path, "rb").read(), self.good_bytes)
        self.assertEqual(tmp_names(self.directory), [])
        self.assertEqual(quarantine_names(self.directory), [])
        self.assertEqual(self.state_store.commit_seq, self.good_seq + 1)
        self.assertEqual(self._formal_doc()["commit_seq"], self.good_seq)

    def test_heal_is_redone_by_restart_and_state_is_consistent(self) -> None:
        self._induce_degraded("alice")
        self.service.revoke_device("bob")
        # Restart recovery accepts exactly what same-process heal produced,
        # including the saved cursor, the messages and the audit chains.
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        self.assertTrue(restarted.store.find_by_device_id("bob").revoked)
        self.assertFalse(restarted.store.find_by_device_id("alice").revoked)
        self.assertEqual(
            restarted.store._group_sync_cursors[(self.sid, "alice")].cursor,
            2)
        body = restarted.sync_group_messages(self.sid, "alice", None, 100)
        self.assertEqual([m["sequence"] for m in body["messages"]], [3])
        events = restarted.store.key_events_page("bob", 0, 100)[0]
        self.assertTrue(any(e["type"] == "device_revoked" for e in events))

    def test_corrupt_backup_is_refused_and_the_heal_stays_retryable(
            self) -> None:
        from e2ee_backend.persistence import PersistenceUnavailable
        backup = self._induce_degraded("alice", with_quarantine=True)
        with open(backup, "wb") as handle:
            handle.write(b"{not json")
        with self.assertRaises(PersistenceUnavailable):
            self.service.revoke_device("bob")
        # Nothing advanced: memory rolled back, formal still missing, the pin
        # and the quarantine leftover remain, and no generation was used.
        self.assertFalse(os.path.exists(self.path))
        self.assertFalse(
            self.service.store.find_by_device_id("alice").revoked)
        self.assertEqual(len(bak_names(self.directory)), 1)
        self.assertEqual(len(quarantine_names(self.directory)), 1)
        self.assertTrue(self.state_store.degraded)
        # Repair the pin with the original committed bytes and retry: heal
        # succeeds and the current request commits one generation.
        os.unlink(backup)
        write_tmp(self.directory, ".state-fixed.bak", self.good_bytes)
        self.service.revoke_device("bob")
        self.assertTrue(os.path.exists(self.path))
        self.assertEqual(tmp_names(self.directory), [])
        self.assertEqual(quarantine_names(self.directory), [])
        self.assertEqual(self._formal_doc()["commit_seq"], self.good_seq + 1)
        self.assertTrue(self.service.store.find_by_device_id("bob").revoked)
        self.assertFalse(
            self.service.store.find_by_device_id("alice").revoked)

    def test_dangling_uncommitted_snapshot_is_never_promoted(self) -> None:
        self._induce_degraded("alice", with_quarantine=True)
        quarantine = os.path.join(
            self.directory, quarantine_names(self.directory)[0])
        with open(quarantine, "rb") as handle:
            newer_bytes = handle.read()
        # Stage the un-committed snapshot as a .tmp leftover too; it parses
        # and restores but is not the committed baseline, so heal must skip
        # it and promote the backup instead.
        write_tmp(self.directory, ".state-dangling.tmp", newer_bytes)
        self.service.revoke_device("bob")
        self.assertEqual(tmp_names(self.directory), [])
        self.assertEqual(quarantine_names(self.directory), [])
        self.assertEqual(self._formal_doc()["commit_seq"], self.good_seq + 1)
        self.assertFalse(
            self.service.store.find_by_device_id("alice").revoked)
        self.assertTrue(self.service.store.find_by_device_id("bob").revoked)

    def test_promotion_failure_moves_nothing_and_next_write_heals(self) -> None:
        from e2ee_backend.persistence import PersistenceUnavailable
        self._induce_degraded("alice")
        real_link = self.persistence_mod.os.link

        def fail_link(src: str, dst: str) -> None:
            raise OSError("simulated promotion failure")

        self.persistence_mod.os.link = fail_link
        try:
            with self.assertRaises(PersistenceUnavailable):
                self.service.revoke_device("bob")
        finally:
            self.persistence_mod.os.link = real_link
        self.assertFalse(os.path.exists(self.path))
        self.assertEqual(len(bak_names(self.directory)), 1)
        self.assertTrue(self.state_store.degraded)
        self.assertFalse(
            self.service.store.find_by_device_id("alice").revoked)
        # The next write retries the heal and commits the current request.
        self.service.revoke_device("bob")
        self.assertTrue(os.path.exists(self.path))
        self.assertEqual(self._formal_doc()["commit_seq"], self.good_seq + 1)
        self.assertTrue(self.service.store.find_by_device_id("bob").revoked)

    def test_two_verifiable_candidates_are_refused_without_name_or_mtime_pick(
            self) -> None:
        from e2ee_backend.persistence import PersistenceUnavailable
        backup = self._induce_degraded("alice", with_quarantine=True)
        # A second verifiable candidate: byte-identical to the pinned backup,
        # but under a .tmp name that sorts first and with a strictly newer
        # mtime. Neither heuristic may choose it, and the two pins together
        # are ambiguous: heal must refuse promotion outright.
        extra = write_tmp(self.directory, ".state-000-extra.tmp",
                          self.good_bytes)
        os.utime(extra, ns=(10**18, 10**18))
        seq_before = self.state_store.commit_seq
        before_names = tmp_names(self.directory)
        self.assertEqual(len(before_names), 2)
        with self.assertRaises(PersistenceUnavailable):
            self.service.revoke_device("bob")
        # Nothing moved: formal stays missing, both candidates and the
        # quarantine remain, memory rolled back, no generation consumed.
        self.assertFalse(os.path.exists(self.path))
        self.assertEqual(tmp_names(self.directory), before_names)
        self.assertEqual(len(quarantine_names(self.directory)), 1)
        self.assertTrue(self.state_store.degraded)
        self.assertEqual(self.state_store.commit_seq, seq_before)
        self.assertFalse(
            self.service.store.find_by_device_id("alice").revoked)
        self.assertFalse(
            self.service.store.find_by_device_id("bob").revoked)
        # The heal stays retryable: removing the extra candidate leaves the
        # unique pin, which the next write promotes and commits once.
        os.unlink(extra)
        self.service.revoke_device("bob")
        self.assertTrue(os.path.exists(self.path))
        self.assertEqual(tmp_names(self.directory), [])
        self.assertEqual(quarantine_names(self.directory), [])
        self.assertEqual(self._formal_doc()["commit_seq"], self.good_seq + 1)
        self.assertTrue(self.service.store.find_by_device_id("bob").revoked)
        self.assertFalse(
            self.service.store.find_by_device_id("alice").revoked)

    def test_two_distinct_verifiable_candidates_are_refused(self) -> None:
        # Two candidates that both verify at the previous generation must be
        # refused even though they are not byte-identical: picking either
        # would silently commit an arbitrary one.
        from e2ee_backend.persistence import PersistenceUnavailable
        backup = self._induce_degraded("alice")
        doc = json.loads(self.good_bytes.decode("utf-8"))
        # Semantically the same committed snapshot (same round-trip), with a
        # byte-only difference, as a second .bak pin could carry after a
        # crash; it restores and equals the baseline.
        other = dict(doc)
        other_bytes = json.dumps(other, separators=(", ", ": ")).encode(
            "utf-8")
        self.assertNotEqual(other_bytes, self.good_bytes)
        extra = write_tmp(self.directory, ".state-twin.bak", other_bytes)
        with self.assertRaises(PersistenceUnavailable):
            self.service.revoke_device("bob")
        self.assertFalse(os.path.exists(self.path))
        self.assertEqual(len(bak_names(self.directory)), 2)
        self.assertTrue(self.state_store.degraded)
        os.unlink(extra)
        self.service.revoke_device("bob")
        self.assertTrue(os.path.exists(self.path))
        self.assertEqual(self._formal_doc()["commit_seq"], self.good_seq + 1)

    def test_current_write_failure_after_heal_advances_nothing_then_commits(
            self) -> None:
        from e2ee_backend.persistence import PersistenceUnavailable
        self._induce_degraded("alice")
        real_save = self.state_store.save
        failures = {"left": 1}

        def fail_once(state):  # noqa: ANN001
            if failures["left"]:
                failures["left"] -= 1
                raise OSError("simulated post-heal write failure")
            return real_save(state)

        self.state_store.save = fail_once  # type: ignore[assignment]
        with self.assertRaises(PersistenceUnavailable):
            self.service.revoke_device("bob")
        # Heal landed (no longer degraded, formal is the committed baseline)
        # but the current write failed: bytes/inode/generation unchanged.
        self.assertFalse(self.state_store.degraded)
        self.assertTrue(os.path.exists(self.path))
        self.assertEqual(open(self.path, "rb").read(), self.good_bytes)
        self.assertEqual(self._formal_doc()["commit_seq"], self.good_seq)
        self.assertFalse(
            self.service.store.find_by_device_id("alice").revoked)
        self.assertFalse(
            self.service.store.find_by_device_id("bob").revoked)
        # The retry is a normal write: one generation, current request only.
        self.service.revoke_device("bob")
        self.assertEqual(self._formal_doc()["commit_seq"], self.good_seq + 1)
        self.assertTrue(self.service.store.find_by_device_id("bob").revoked)
        self.assertFalse(
            self.service.store.find_by_device_id("alice").revoked)

    def test_valid_formal_file_takes_precedence_over_backup(self) -> None:
        backup = self._induce_degraded("alice")
        with open(backup, "rb") as handle:
            committed = handle.read()
        with open(self.path, "wb") as handle:
            handle.write(committed)
        # The same request heals (accepts the valid formal, sweeps pins) and
        # commits its own write in one generation.
        self.service.revoke_device("bob")
        self.assertEqual(tmp_names(self.directory), [])
        self.assertEqual(quarantine_names(self.directory), [])
        self.assertEqual(self._formal_doc()["commit_seq"], self.good_seq + 1)
        self.assertTrue(self.service.store.find_by_device_id("bob").revoked)

    def test_unexpected_formal_file_is_refused_and_left_untouched(self) -> None:
        from e2ee_backend.persistence import PersistenceUnavailable
        self._induce_degraded("alice", with_quarantine=True)
        quarantine = os.path.join(
            self.directory, quarantine_names(self.directory)[0])
        with open(quarantine, "rb") as handle:
            newer_bytes = handle.read()
        with open(self.path, "wb") as handle:
            handle.write(newer_bytes)
        before = open(self.path, "rb").read()
        before_ino = os.stat(self.path).st_ino
        with self.assertRaises(PersistenceUnavailable):
            self.service.revoke_device("bob")
        self.assertEqual(open(self.path, "rb").read(), before)
        self.assertEqual(os.stat(self.path).st_ino, before_ino)
        self.assertEqual(len(bak_names(self.directory)), 1)
        self.assertTrue(self.state_store.degraded)

    def test_concurrent_writes_run_one_recovery_and_serial_commits(self) -> None:
        import threading
        self._induce_degraded("alice")
        real_heal = self.state_store.heal
        calls = {"heal": 0}

        def counted_heal(baseline):  # noqa: ANN001
            calls["heal"] += 1
            return real_heal(baseline)

        self.state_store.heal = counted_heal  # type: ignore[assignment]
        errors: List[Exception] = []

        def revoke(device_id: str) -> None:
            try:
                self.service.revoke_device(device_id)
            except Exception as error:  # noqa: BLE001
                errors.append(error)

        threads = [threading.Thread(target=revoke, args=(device_id,))
                   for device_id in ("bob", "creator")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
        self.assertEqual(errors, [])
        # Exactly one self-heal, then two serialized commits.
        self.assertEqual(calls["heal"], 1)
        self.assertTrue(self.service.store.find_by_device_id("bob").revoked)
        self.assertTrue(
            self.service.store.find_by_device_id("creator").revoked)
        self.assertFalse(
            self.service.store.find_by_device_id("alice").revoked)
        self.assertEqual(self._formal_doc()["commit_seq"], self.good_seq + 2)
        self.assertEqual(tmp_names(self.directory), [])
        self.assertEqual(quarantine_names(self.directory), [])

    def test_heal_from_a_legacy_sectionless_pinned_backup(self) -> None:
        # Build a legacy version-1 document by hand: no commit_seq and no
        # key_events section (a file written before audit chains existed).
        legacy_service = DeviceService()
        for device_id in ("creator", "alice", "bob"):
            legacy_service.store.add_device(Device("u", device_id, "ik"))
        legacy_service.create_group({
            "group_id": "g1", "creator_device_id": "creator",
            "member_device_ids": ["alice", "bob"]})
        legacy_doc = dict(legacy_service.store.snapshot_state())
        legacy_doc.pop("key_events", None)  # the legacy gap
        shutil.rmtree(self.directory, ignore_errors=True)
        os.makedirs(self.directory)
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump({"version": 1, **legacy_doc}, handle)

        # Re-attach onto the legacy file; the first persist-able change
        # anchors the chains and writes generation 1. Make that write end in
        # the undecidable failure, pinning the section-less legacy bytes.
        self.service = DeviceService()
        self.state_store = attach_persistence(self.service, self.path)
        self.assertEqual(self.state_store.commit_seq, 1)
        self.good_bytes = open(self.path, "rb").read()
        self.good_seq = 0
        backup = self._induce_degraded("alice")
        self.assertNotIn(b"key_events", open(backup, "rb").read())

        # The next write heals the legacy pin (semantic, not raw-dict
        # equality), anchors the chains and commits generation 1 once.
        self.service.revoke_device("bob")
        self.assertTrue(os.path.exists(self.path))
        self.assertEqual(tmp_names(self.directory), [])
        document = self._formal_doc()
        self.assertEqual(document["commit_seq"], 1)
        self.assertIsInstance(document["key_events"], list)
        self.assertTrue(self.service.store.find_by_device_id("bob").revoked)
        self.assertFalse(
            self.service.store.find_by_device_id("alice").revoked)
        # A restart fully accepts the healed-then-committed document.
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        self.assertTrue(restarted.store.find_by_device_id("bob").revoked)
        self.assertFalse(restarted.store.find_by_device_id("alice").revoked)
        events = restarted.store.key_events_page("bob", 0, 100)[0]
        self.assertTrue(any(e["type"] == "device_revoked" for e in events))

    def test_http_heal_commits_current_request_and_failed_heal_is_503(
            self) -> None:
        import threading
        from http.client import HTTPConnection
        from e2ee_backend.http_app import create_server

        self._induce_degraded("alice")
        server, _ = create_server("127.0.0.1", 0, service=self.service)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            def revoke(device_id: str):
                conn = HTTPConnection("127.0.0.1", port, timeout=5)
                conn.request("POST", f"/v1/devices/{device_id}/revoke")
                response = conn.getresponse()
                body = json.loads(response.read().decode("utf-8"))
                conn.close()
                return response.status, body

            # A failed heal (corrupt pin) is 503/field=data_file and moves
            # nothing.
            bak = os.path.join(self.directory, bak_names(self.directory)[0])
            with open(bak, "wb") as handle:
                handle.write(b"{nope")
            status, body = revoke("bob")
            self.assertEqual(status, 503)
            self.assertEqual(body["field"], "data_file")
            self.assertFalse(os.path.exists(self.path))
            # Repair the pin; the next HTTP request self-heals and succeeds.
            with open(bak, "wb") as handle:
                handle.write(self.good_bytes)
            status, body = revoke("bob")
            self.assertEqual(status, 200, body)
            # alice still posts messages: its 503 revocation never happened.
            conn = HTTPConnection("127.0.0.1", port, timeout=5)
            conn.request("POST", "/v1/messages",
                         body=json.dumps({
                             "session_id": self.sid,
                             "sender_device_id": "alice",
                             "message_id": "after-heal", "sequence": 4,
                             "nonce": "n4", "ciphertext": "ct"}),
                         headers={"Content-Type": "application/json"})
            response = conn.getresponse()
            self.assertEqual(response.status, 201, response.read())
            conn.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
        self.assertEqual(self._formal_doc()["commit_seq"], self.good_seq + 2)
        self.assertTrue(self.service.store.find_by_device_id("bob").revoked)
        self.assertFalse(
            self.service.store.find_by_device_id("alice").revoked)


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
