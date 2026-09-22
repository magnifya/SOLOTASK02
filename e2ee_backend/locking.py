"""Process-level exclusive lock for a persistent state file.

When ``serve`` runs with ``--data-file`` (or ``$E2EE_DATA_FILE``) it acquires
an exclusive, process-owned lock on a sibling lock file in the state file's
own directory before touching the state file itself. Only one serve process
may hold the lock at a time; a second process fails to start instead of
running recovery or atomic-replace transactions against a file another
process is writing.

Two platform backends share one API:

* POSIX: a BSD-style advisory :func:`fcntl.flock` on a regular lock file.
* Windows: a non-blocking byte-range lock
  (:func:`msvcrt.locking` ``LK_NBLCK`` on one byte) on the same kind of file.

In both cases the lock is *created but never truncated* and is tied to the
process: closing the descriptor or terminating the process releases it, even
on ``SIGKILL`` (POSIX) or ``TerminateProcess`` (Windows, where the lock is
owned by the file object and dropped at process teardown). A successor can
therefore start and recover the state without any stale-lock cleanup.
"""
from __future__ import annotations

import errno
import os
import sys
from types import TracebackType
from typing import Optional, Type

#: Suffix appended to the state file name for its same-directory lock file.
_LOCK_SUFFIX = ".lock"

#: Number of bytes covered by the Windows byte-range lock (at least one byte
#: starting at offset 0).
_WINDOWS_LOCK_BYTES = 1


class StateFileLocked(Exception):
    """Another live process already holds the exclusive lock for the file."""


def lock_path_for(state_path: str) -> str:
    """Return the same-directory lock-file path for *state_path*.

    The lock sits beside the state file (never in a shared system directory)
    and is named after it, so two different state files never contend while
    every process pointed at the same file opens exactly this path. The name
    does not use the ``.state-*`` crash-leftover prefix, so recovery scans
    ignore it.
    """
    absolute = os.path.abspath(state_path)
    return absolute + _LOCK_SUFFIX


def _windows_lock_fd(fd: int) -> None:
    """Take a non-blocking exclusive Windows byte-range lock on *fd*.

    Locks one byte at offset 0 of the file via :func:`msvcrt.locking`
    (``LK_NBLCK``). Raises :class:`StateFileLocked` (empty message) when
    another live process already holds that byte range; any other
    :class:`OSError` propagates to the caller.
    """
    import msvcrt

    try:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_NBLCK, _WINDOWS_LOCK_BYTES)
    except OSError as error:
        if error.errno in (errno.EACCES, errno.EDEADLK):
            raise StateFileLocked() from None
        raise


def _windows_unlock_fd(fd: int) -> None:
    """Release the byte-range lock taken by :func:`_windows_lock_fd`."""
    import msvcrt

    try:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, _WINDOWS_LOCK_BYTES)
    except OSError:
        pass


def _posix_lock_fd(fd: int) -> None:
    """Take a non-blocking advisory ``flock`` (``LOCK_EX | LOCK_NB``).

    Raises :class:`StateFileLocked` (empty message) when another live
    process already holds the lock; any other :class:`OSError` propagates.
    """
    import fcntl

    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        if error.errno in (errno.EWOULDBLOCK, errno.EAGAIN):
            raise StateFileLocked() from None
        raise


def _posix_unlock_fd(fd: int) -> None:
    """Release the advisory ``flock`` (best effort)."""
    import fcntl

    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    except OSError:
        pass


if sys.platform == "win32":  # pragma: posix cover - exercised on Windows CI
    _lock_fd = _windows_lock_fd
    _unlock_fd = _windows_unlock_fd
else:
    _lock_fd = _posix_lock_fd
    _unlock_fd = _posix_unlock_fd


class StateFileLock:
    """An exclusive process lock held on ``<state-file>.lock``.

    The held open file descriptor is the only lock state: closing it or
    terminating the process releases the lock. The lock file itself is left
    in place on release — it is empty, never truncated, and unlinking it
    while a process is waiting could let a newcomer lock a fresh inode and
    run concurrently with the holder.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self.lock_path: str = lock_path_for(path)
        self._fd: Optional[int] = None

    def acquire(self) -> None:
        """Take the exclusive lock or raise :class:`StateFileLocked`.

        Opens (creating if needed, never truncating) the sibling lock file
        and takes a non-blocking exclusive lock through the platform
        backend. If another live process already holds it, the descriptor is
        closed and :class:`StateFileLocked` is raised without touching the
        formal state file. Any other :class:`OSError` (missing/unwritable
        directory) propagates to the caller.
        """
        directory = os.path.dirname(self.lock_path)
        os.makedirs(directory, exist_ok=True)
        # No O_TRUNC, ever: an existing lock file belongs to a possibly-live
        # holder and its bytes must be left alone.
        fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            _lock_fd(fd)
        except StateFileLocked:
            os.close(fd)
            raise StateFileLocked(
                f"state file is locked by another process: {self.path}") \
                from None
        except BaseException:
            os.close(fd)
            raise
        self._fd = fd

    def release(self) -> None:
        """Release the lock (idempotent); the kernel also releases on exit."""
        fd = self._fd
        if fd is None:
            return
        self._fd = None
        _unlock_fd(fd)
        try:
            os.close(fd)
        except OSError:
            pass

    def __enter__(self) -> "StateFileLock":
        # Idempotent: acquire_state_file_lock() returns an already-acquired
        # lock, so re-entering it as a context manager must not lock a second
        # descriptor (which would dead-lock against the first).
        if self._fd is None:
            self.acquire()
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]],
                 exc_val: Optional[BaseException],
                 exc_tb: Optional[TracebackType]) -> None:
        self.release()


def acquire_state_file_lock(state_path: str) -> StateFileLock:
    """Acquire the process-exclusive lock for *state_path* and return it.

    The caller owns the returned :class:`StateFileLock` and must keep it
    alive for the whole process lifetime, calling
    :meth:`StateFileLock.release` (or using it as a context manager) on
    shutdown. Raises :class:`StateFileLocked` when another live process
    already owns the lock.
    """
    lock = StateFileLock(state_path)
    lock.acquire()
    return lock
