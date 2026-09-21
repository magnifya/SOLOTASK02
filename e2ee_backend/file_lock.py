"""Process-level exclusive lock for a persistent state file.

When ``serve`` runs against ``--data-file`` / ``$E2EE_DATA_FILE`` it must be
the only process managing that file. A sibling lock file (kept in the same
directory as the state file, so it is on the same file system) is held with an
advisory exclusive ``flock`` for the whole lifetime of the process. The lock
is released automatically by the operating system on normal exit *and* on
abnormal termination (the kernel drops an advisory ``flock`` when the holding
open file description closes), so a later process can start immediately and
recover the state file. In-memory mode (no data file) takes no lock.

The lock file is created (never truncated) once and left in place: removing a
lock file is unsafe because a competing process could then lock a fresh
inode. Its name (``.<state-file>.lock``) never matches the ``.state-*.tmp`` /
``.state-*.bak`` crash-leftover patterns, so crash recovery ignores it.
"""
from __future__ import annotations

import errno
import fcntl
import os
from types import TracebackType
from typing import Optional, Type


class StateFileLocked(Exception):
    """Another live process already holds the lock for the state file."""


class StateFileLock:
    """An exclusive process-scoped ``flock`` on a sibling of the state file."""

    def __init__(self, state_path: str) -> None:
        directory = os.path.dirname(os.path.abspath(state_path))
        self.path = os.path.join(directory,
                                 f".{os.path.basename(state_path)}.lock")
        self._fd: Optional[int] = None

    def acquire(self) -> None:
        """Take the lock non-blocking, creating the sibling file if needed.

        The lock file is opened without ``O_TRUNC`` so an existing one is
        never modified; the formal state file is never touched. Raises
        :class:`StateFileLocked` when another live process already holds the
        exclusive lock; other :class:`OSError` values (e.g. the directory
        cannot be created) propagate to the caller.
        """
        directory = os.path.dirname(self.path)
        os.makedirs(directory, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                if error.errno in (errno.EWOULDBLOCK, errno.EAGAIN,
                                   errno.EACCES):
                    raise StateFileLocked(
                        f"state file is already locked by another serve "
                        f"process: {self.path}") from None
                raise
        except BaseException:
            os.close(fd)
            raise
        self._fd = fd

    def release(self) -> None:
        """Drop the lock and close the descriptor (idempotent).

        Explicit release is a courtesy: if the process exits or is killed
        without calling this, the kernel releases the ``flock`` anyway.
        """
        if self._fd is None:
            return
        fd, self._fd = self._fd, None
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def __enter__(self) -> "StateFileLock":
        self.acquire()
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]],
                 exc: Optional[BaseException],
                 traceback: Optional[TracebackType]) -> None:
        self.release()
