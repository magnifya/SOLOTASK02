"""Process-level exclusive lock for a persistent state file.

When ``serve`` runs with ``--data-file`` (or ``$E2EE_DATA_FILE``) it acquires
an exclusive, process-owned lock on a sibling lock file in the state file's
own directory before touching the state file itself. Only one serve process
may hold the lock at a time; a second process fails to start instead of
running recovery or atomic-replace transactions against a file another
process is writing.

The lock is taken, non-blocking, on a regular lock file that is *created but
never truncated*:

* POSIX: a BSD-style advisory :func:`fcntl.flock`. ``flock`` locks are owned
  by the open file description and are released by the kernel as soon as the
  process exits normally or is killed (even by ``SIGKILL``).
* Windows: a one-byte range lock at offset 0 taken with the standard-library
  :func:`msvcrt.locking` (``LK_NBLCK``). The lock is owned by the process and
  released by the operating system when the process exits or is terminated;
  the byte range may sit beyond end of file, so the empty lock file is never
  written to or truncated.

In both cases the held open file descriptor is the only lock state, so a
successor can start and recover the state without any stale-lock cleanup.
"""
from __future__ import annotations

import errno
import os
import sys
from types import TracebackType
from typing import Optional, Type

#: Suffix appended to the state file name for its same-directory lock file.
_LOCK_SUFFIX = ".lock"

#: Number of bytes in the Windows byte-range lock (one token byte at offset 0;
#: Windows permits locking past end of file, so the lock file stays empty).
_LOCK_RANGE_BYTES = 1


class StateFileLocked(Exception):
    """Another live process already holds the exclusive lock for the file."""


class _LockContention(Exception):
    """Internal signal: the platform non-blocking lock is already held.

    Backends raise this instead of :class:`StateFileLocked` so the public
    exception can be constructed with the state-file path by the owning
    :class:`StateFileLock`.
    """


if sys.platform == "win32":  # pragma: no cover - the POSIX suite exercises
    # the public API; this import branch only exists on Windows.
    import msvcrt  # type: ignore[import-not-found]

    # The CRT reports a contended non-blocking lock as EACCES or EDEADLOCK.
    _BUSY_ERRNOS = frozenset({
        errno.EACCES,
        getattr(errno, "EDEADLOCK", errno.EDEADLK),
    })

    def _lock_nonblocking(fd: int) -> None:
        """Take the exclusive Windows byte-range lock on *fd*.

        Locks one byte at offset 0 with :func:`msvcrt.locking` in
        non-blocking mode. A lock held by another live process is reported by
        the CRT and mapped to :class:`_LockContention`; any other
        :class:`OSError` propagates to the caller.
        """
        os.lseek(fd, 0, os.SEEK_SET)
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, _LOCK_RANGE_BYTES)
        except OSError as error:
            if error.errno in _BUSY_ERRNOS:
                raise _LockContention(str(error)) from None
            raise

    def _unlock(fd: int) -> None:
        """Release the Windows byte-range lock (best effort)."""
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, _LOCK_RANGE_BYTES)
        except OSError:
            pass

else:
    import fcntl

    def _lock_nonblocking(fd: int) -> None:
        """Take a non-blocking exclusive ``flock`` on *fd*.

        ``EWOULDBLOCK``/``EAGAIN`` means another live process owns the lock
        and maps to :class:`_LockContention`; any other :class:`OSError`
        propagates.
        """
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            if error.errno in (errno.EWOULDBLOCK, errno.EAGAIN):
                raise _LockContention(str(error)) from None
            raise

    def _unlock(fd: int) -> None:
        """Release the ``flock`` (best effort)."""
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass


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
        and takes the platform's non-blocking exclusive lock. If another live
        process already holds it, the descriptor is closed and
        :class:`StateFileLocked` is raised without touching the formal state
        file. Any other :class:`OSError` (missing/unwritable directory)
        propagates to the caller.
        """
        directory = os.path.dirname(self.lock_path)
        os.makedirs(directory, exist_ok=True)
        # No O_TRUNC, ever: an existing lock file belongs to a possibly-live
        # holder and its bytes must be left alone. O_BINARY is a no-op on
        # POSIX and keeps the Windows byte-range lock position semantics.
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0)
        fd = os.open(self.lock_path, flags, 0o600)
        try:
            _lock_nonblocking(fd)
        except _LockContention:
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
        _unlock(fd)
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
