"""Cross-platform lock backend and directory-fsync fallback tests.

The suite normally runs on POSIX, so the Windows byte-range backend
(``msvcrt.locking``) is exercised against an in-memory stub ``msvcrt``
module (the backend imports it lazily) driven through the real
:class:`StateFileLock`. The directory-fsync fallback is exercised with a
real atomic ``save()`` transaction whose directory fsync is rejected:

* ``EINVAL`` (the errno network/special filesystems give) must be skipped
  silently while the commit still lands;
* a genuine ``EIO`` after the replace must still propagate, trigger the
  hard-link rollback, and surface as ``PersistenceUnavailable``.
"""
from __future__ import annotations

import errno
import json
import os
import shutil
import stat
import sys
import tempfile
import types
import unittest

import e2ee_backend.locking as locking_mod
import e2ee_backend.persistence as persistence_mod
from e2ee_backend.locking import (
    StateFileLock,
    StateFileLocked,
    lock_path_for,
)
from e2ee_backend.persistence import (
    PersistenceUnavailable,
    _fsync_directory,
    _leftover_tmp_paths,
    attach_persistence,
)
from e2ee_backend.service import DeviceService


class _StubMsvcrt:
    """Minimal in-memory stand-in for the Windows ``msvcrt`` module."""

    LK_NBLCK = 1
    LK_UNLCK = 2

    def __init__(self) -> None:
        self.held = False
        self.calls = []

    def _key(self, fd: int):
        info = os.fstat(fd)
        return info.st_dev, info.st_ino

    def locking(self, fd: int, mode: int, nbytes: int) -> None:
        # The real backend always seeks to offset 0 first; record the state.
        self.calls.append((os.lseek(fd, 0, os.SEEK_CUR), mode, nbytes))
        if mode == self.LK_NBLCK:
            if self.held:
                raise OSError(errno.EACCES, "file locked")
            self.held = True
        elif mode == self.LK_UNLCK:
            self.held = False


class WindowsLockBackendTest(unittest.TestCase):
    """The msvcrt backend through the public StateFileLock API."""

    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.state_path = os.path.join(self.directory, "state.json")
        self.stub = _StubMsvcrt()
        self.fake_module = types.ModuleType("msvcrt")
        self.fake_module.LK_NBLCK = self.stub.LK_NBLCK
        self.fake_module.LK_UNLCK = self.stub.LK_UNLCK
        self.fake_module.locking = self.stub.locking
        sys.modules["msvcrt"] = self.fake_module
        self._orig_lock = locking_mod._lock_fd
        self._orig_unlock = locking_mod._unlock_fd
        locking_mod._lock_fd = locking_mod._windows_lock_fd
        locking_mod._unlock_fd = locking_mod._windows_unlock_fd

    def tearDown(self) -> None:
        locking_mod._lock_fd = self._orig_lock
        locking_mod._unlock_fd = self._orig_unlock
        sys.modules.pop("msvcrt", None)
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_acquire_locks_one_byte_at_offset_zero(self) -> None:
        with StateFileLock(self.state_path):
            self.assertTrue(self.stub.held)
            self.assertEqual(self.stub.calls[0][1:],
                             (self.stub.LK_NBLCK, 1))
        # Releasing unlocked the same range.
        self.assertFalse(self.stub.held)
        self.assertEqual(self.stub.calls[-1][1:],
                         (self.stub.LK_UNLCK, 1))
        # Every call operated at offset 0.
        self.assertTrue(all(offset == 0 for offset, _, _ in self.stub.calls))

    def test_contention_maps_to_state_file_locked_with_path(self) -> None:
        holder = StateFileLock(self.state_path)
        holder.acquire()
        try:
            with self.assertRaises(StateFileLocked) as caught:
                StateFileLock(self.state_path).acquire()
            self.assertIn(self.state_path, str(caught.exception))
        finally:
            holder.release()
        # Once released the same file is lockable again.
        with StateFileLock(self.state_path):
            pass

    def test_existing_lock_file_is_opened_not_truncated(self) -> None:
        marker = b"do-not-touch"
        lock_path = lock_path_for(self.state_path)
        with open(lock_path, "wb") as handle:
            handle.write(marker)
        with StateFileLock(self.state_path):
            self.assertEqual(open(lock_path, "rb").read(), marker)
        self.assertEqual(open(lock_path, "rb").read(), marker)

    def test_non_conflict_oserror_propagates(self) -> None:
        def fail(fd: int, mode: int, nbytes: int) -> None:
            raise OSError(errno.EIO, "io error")

        self.fake_module.locking = fail
        with self.assertRaises(OSError) as caught:
            StateFileLock(self.state_path).acquire()
        self.assertEqual(caught.exception.errno, errno.EIO)
        # The opened descriptor was closed even though locking failed.
        self.assertIsNone(StateFileLock(self.state_path)._fd)

    def test_unlock_error_is_swallowed(self) -> None:
        def fail_unlock(fd: int, mode: int, nbytes: int) -> None:
            if mode == self.stub.LK_UNLCK:
                raise OSError(errno.EIO, "io error")

        self.fake_module.locking = fail_unlock
        lock = StateFileLock(self.state_path)
        lock.acquire()
        lock.release()  # must not raise
        lock.release()  # idempotent


class DirectoryFsyncFallbackTest(unittest.TestCase):
    """Directory fsync unsupported -> skip; other errors -> fail/rollback."""

    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.path = os.path.join(self.directory, "state.json")
        self.service = DeviceService()
        self.store = attach_persistence(self.service, self.path)
        self.good_bytes = open(self.path, "rb").read()
        self.good_ino = os.stat(self.path).st_ino
        self._real_fsync = os.fsync

    def tearDown(self) -> None:
        os.fsync = self._real_fsync  # type: ignore[assignment]
        persistence_mod.sys = sys
        shutil.rmtree(self.directory, ignore_errors=True)

    def _directory_only_failure(self, err: int):
        """Return an fsync that fails with *err* on directory fds only."""
        real_fsync = self._real_fsync

        def selective_fsync(fd: int) -> None:
            if stat.S_ISDIR(os.fstat(fd).st_mode):
                raise OSError(err, "selective failure")
            real_fsync(fd)

        return selective_fsync

    def test_normal_directory_fsync_runs(self) -> None:
        self.assertTrue(_fsync_directory(self.directory))

    def test_unsupported_errnos_are_skipped(self) -> None:
        for err in (errno.EINVAL, getattr(errno, "ENOTSUP", errno.EINVAL)):
            os.fsync = self._directory_only_failure(err)  # type: ignore
            try:
                self.assertFalse(_fsync_directory(self.directory))
            finally:
                os.fsync = self._real_fsync  # type: ignore[assignment]

    def test_open_rejected_as_unsupported_is_skipped(self) -> None:
        real_open = os.open

        def fail_directory_open(path, flags, *args):
            if os.path.abspath(path) == os.path.abspath(self.directory):
                raise OSError(errno.EINVAL, "cannot open directory")
            return real_open(path, flags, *args)

        os.open = fail_directory_open  # type: ignore[assignment]
        try:
            self.assertFalse(_fsync_directory(self.directory))
        finally:
            os.open = real_open  # type: ignore[assignment]

    def test_genuine_ioerror_still_propagates(self) -> None:
        os.fsync = self._directory_only_failure(errno.EIO)  # type: ignore
        with self.assertRaises(OSError) as caught:
            _fsync_directory(self.directory)
        self.assertEqual(caught.exception.errno, errno.EIO)

    def test_save_commits_when_directory_fsync_unsupported(self) -> None:
        os.fsync = self._directory_only_failure(errno.EINVAL)  # type: ignore
        # A full atomic transaction: temp file fsync still runs, replace
        # lands, the unsupported directory flush is skipped, commit stands.
        snapshot = self.service.store.snapshot_state()
        self.store.save(snapshot)
        self.assertTrue(os.path.exists(self.path))
        leftovers = [name for name in os.listdir(self.directory)
                     if name.endswith(".tmp") or name.endswith(".bak")]
        self.assertEqual(leftovers, [])
        with open(self.path, encoding="utf-8") as handle:
            self.assertEqual(json.load(handle)["version"], 1)

    def test_directory_fsync_eio_after_replace_preserves_backup(self) -> None:
        from e2ee_backend.models import Device

        os.fsync = self._directory_only_failure(errno.EIO)  # type: ignore
        with self.assertRaises(PersistenceUnavailable):
            # Any mutating service call runs one save() transaction.
            self.service.store.add_device(Device("u", "d1", "ik"))
        # The post-replace directory fsync failing with a genuine I/O error
        # aborts the transaction; the rollback rename lands but its own
        # directory flush fails the same way. The old inode is therefore
        # preserved as a .bak and the formal path is vacated so an
        # uncommitted snapshot can never be authoritative.
        self.assertFalse(os.path.exists(self.path))
        leftovers = [name for name in os.listdir(self.directory)
                     if name.endswith(".tmp") or name.endswith(".bak")]
        self.assertEqual(len(leftovers), 1)
        self.assertTrue(leftovers[0].endswith(".bak"))
        with open(os.path.join(self.directory, leftovers[0]), "rb") as h:
            self.assertEqual(h.read(), self.good_bytes)
        # The failed registration was rolled back in memory.
        self.assertIsNone(self.service.store.find_by_device_id("d1"))
        # Startup recovery promotes the preserved backup; the retry then
        # commits the device.
        os.fsync = self._real_fsync  # type: ignore[assignment]
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        self.assertEqual(open(self.path, "rb").read(), self.good_bytes)
        restarted.store.add_device(Device("u", "d1", "ik"))
        self.assertIsNotNone(restarted.store.find_by_device_id("d1"))
        self.assertEqual(
            [n for n in os.listdir(self.directory)
             if n.endswith(".tmp") or n.endswith(".bak")], [])

    def test_windows_style_directory_open_eacces_skipped(self) -> None:
        # On win32 opening a directory for the metadata flush surfaces as
        # EACCES; the classifier only treats it as unsupported there.
        real_open = os.open

        def fail_directory_open(path, flags, *args):
            if os.path.abspath(path) == os.path.abspath(self.directory):
                raise OSError(errno.EACCES, "permission denied")
            return real_open(path, flags, *args)

        os.open = fail_directory_open  # type: ignore[assignment]
        fake_sys = types.SimpleNamespace(platform="win32")
        persistence_mod.sys = fake_sys
        try:
            self.assertFalse(_fsync_directory(self.directory))
        finally:
            os.open = real_open  # type: ignore[assignment]
            persistence_mod.sys = sys
        # The same EACCES on POSIX is a genuine error.
        os.open = fail_directory_open  # type: ignore[assignment]
        try:
            with self.assertRaises(OSError) as caught:
                _fsync_directory(self.directory)
            self.assertEqual(caught.exception.errno, errno.EACCES)
        finally:
            os.open = real_open  # type: ignore[assignment]


class LockFileExcludedFromRecoveryScanTest(unittest.TestCase):
    """The sibling .lock file never joins .state-*.tmp/.bak recovery scans."""

    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.state_path = os.path.join(self.directory, "state.json")

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_lock_path_is_not_a_leftover_candidate(self) -> None:
        lock_path = lock_path_for(self.state_path)
        with open(lock_path, "wb"):
            pass
        with open(os.path.join(self.directory, ".state-old.tmp"), "wb"):
            pass
        leftovers = _leftover_tmp_paths(self.directory, self.state_path)
        self.assertEqual(
            [os.path.basename(path) for path in leftovers],
            [".state-old.tmp"])


if __name__ == "__main__":
    unittest.main()
