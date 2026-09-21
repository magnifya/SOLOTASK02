"""Durable JSON-file persistence for the device store.

The whole server state is kept in one versioned JSON document. A missing file
is created on first use; a corrupted file or a document whose version is not
understood makes the server refuse to start (state is never silently
discarded). Every change is written atomically: serialize to a temporary file in the
same directory, ``fsync`` it, ``os.replace`` it over the target, and fsync
the parent directory, so a crash (or power loss) never leaves a half-written
state file nor an un-directory-entried rename. A hard link to the previous
file lets a failure of the post-rename directory fsync put the exact old
bytes and inode back.

On startup, temporary snapshots left by a crash during a previous save are
reconciled by :func:`recover_crash_leftovers`: a valid formal file is kept
and the leftovers cleaned; a missing formal file is atomically restored from
the newest verifiable ``version=1`` leftover (older or invalid ones are
discarded); with no valid snapshot an empty state is created.

Persistence is transactional with respect to the in-memory store: the change
hook fires while the store lock is held (the mutation is visible but not yet
committed to callers). When the durable write succeeds it becomes the new
last-known-good state. When writing the temporary file, ``fsync``, the
atomic replace, or the post-rename directory fsync fails (an
:class:`OSError`), the previous last-known-good state is restored into memory
under the same lock, the old state file keeps its bytes and inode, any
temporary file is cleaned up by :meth:`JsonStateStore.save`, and
:class:`PersistenceUnavailable` is raised so the HTTP layer answers
503/field=data_file. The failed mutation therefore never advances either
memory or the file.
"""
from __future__ import annotations

import copy
import json
import os
import tempfile
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .service import DeviceService

from .storage import DeviceStore

#: Persistence format version understood by this build.
STATE_VERSION = 1


class StateFileError(Exception):
    """The state file is missing fields, corrupt, or has an unknown version."""


class PersistenceUnavailable(Exception):
    """A durable write failed; the transaction was rolled back in memory.

    The in-memory store was restored to the last successfully persisted state
    under the store lock, the previous state file is intact, and the temporary
    file was removed. The HTTP layer reports this as 503/field=data_file.
    """


class JsonStateStore:
    """Versioned JSON document loaded from and atomically saved to one file."""

    def __init__(self, path: str) -> None:
        self.path = path

    def load(self) -> Optional[Dict[str, Any]]:
        """Load the document, or ``None`` when the file does not exist yet.

        Raises :class:`StateFileError` if the file cannot be decoded as JSON,
        is not a JSON object, or carries a missing/unknown ``version``.
        """
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                raw = handle.read()
        except FileNotFoundError:
            return None
        except OSError as error:
            raise StateFileError(f"cannot read state file {self.path}: {error}")

        try:
            document = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise StateFileError(
                f"state file is not valid JSON: {self.path} ({error})")

        if not isinstance(document, dict):
            raise StateFileError("state document must be a JSON object")
        version = document.get("version")
        if not isinstance(version, int) or isinstance(version, bool):
            raise StateFileError("state document is missing a numeric 'version'")
        if version != STATE_VERSION:
            raise StateFileError(
                f"unsupported state file version: {version} "
                f"(this server supports {STATE_VERSION})")
        return document

    def save(self, state: Dict[str, Any]) -> None:
        """Atomically replace the file with *state* (version stamped).

        Writes a sibling temporary file, fsyncs it, ``os.replace``\\ s it over
        the target, then fsyncs the parent directory so the rename itself is
        durable — the target is either the previous full document or the new
        full one, never a truncated mix. Before the replace a same-directory
        hard link to the existing target is taken (sharing its inode) so a
        failure of the post-rename parent-directory fsync can put the exact
        previous bytes *and inode* back. On any failure the target keeps its
        old bytes and inode, temporary files are removed, and the underlying
        :class:`OSError` propagates to the caller.
        """
        directory = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(directory, exist_ok=True)
        document = {"version": STATE_VERSION, **state}
        tmp_handle = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=directory, delete=False,
            prefix=".state-", suffix=".tmp")
        tmp_path = tmp_handle.name
        # Extra directory entry for the committed file (shares its inode),
        # used only to undo the rename when a durability step after it fails.
        # Absent on the first save, when there is nothing to restore.
        backup_path: Optional[str] = None
        replaced = False
        restored = False
        try:
            json.dump(document, tmp_handle, separators=(",", ":"),
                      ensure_ascii=False)
            tmp_handle.flush()
            os.fsync(tmp_handle.fileno())
            tmp_handle.close()
            if os.path.exists(self.path):
                backup_fd, backup_path = tempfile.mkstemp(
                    dir=directory, prefix=".state-backup-", suffix=".tmp")
                os.close(backup_fd)
                os.unlink(backup_path)
                os.link(self.path, backup_path)
            os.replace(tmp_path, self.path)
            replaced = True
            self._fsync_directory(directory)
        except BaseException:
            tmp_handle.close()
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            # The rename committed but a later durability step failed: the
            # contract is that the previous bytes AND inode survive, so swap
            # the hard-linked backup back over the target. If that undo also
            # fails, leave the backup on disk for inspection rather than
            # unlinking the last copy of the previous state.
            if replaced and backup_path is not None \
                    and os.path.exists(backup_path):
                try:
                    os.replace(backup_path, self.path)
                    self._fsync_directory(directory)
                    restored = True
                except OSError:
                    restored = False
            if backup_path is not None and (restored or not replaced):
                try:
                    os.unlink(backup_path)
                except OSError:
                    pass
            raise
        if backup_path is not None:
            try:
                os.unlink(backup_path)
            except OSError:
                pass

    @staticmethod
    def _fsync_directory(directory: str) -> None:
        """Fsync a directory fd so a rename's directory entry is durable.

        Best effort on platforms without ``O_DIRECTORY`` support; the data
        fsync and atomic replace already guarantee file-content consistency.
        """
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        dir_fd = os.open(directory, flags)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)


def _verify_snapshot_document(document: Any) -> Dict[str, Any]:
    """Validate a candidate crash-recovery snapshot structurally and semantically.

    A leftover temporary file is only accepted as a recovery snapshot when it
    is a JSON object carrying ``version == 1`` *and* its payload restores
    cleanly into a fresh :class:`DeviceStore` (which runs every cross-entity
    consistency check: references, sequence continuity, nonce sets, per-device
    cursor ranges, non-empty ``updated_at`` ...). Returns the
    version-stripped payload; raises :class:`StateFileError` otherwise.
    """
    if not isinstance(document, dict):
        raise StateFileError("recovery snapshot must be a JSON object")
    version = document.get("version")
    if not isinstance(version, int) or isinstance(version, bool):
        raise StateFileError("recovery snapshot is missing a numeric version")
    if version != STATE_VERSION:
        raise StateFileError(
            f"unsupported recovery snapshot version: {version} "
            f"(this server supports {STATE_VERSION})")
    payload = {key: value for key, value in document.items()
               if key != "version"}
    try:
        DeviceStore().restore_state(copy.deepcopy(payload))
    except (ValueError, TypeError) as error:
        raise StateFileError(f"recovery snapshot is malformed: {error}") \
            from None
    return payload


def recover_crash_leftovers(state_store: "JsonStateStore"
                            ) -> Optional[Dict[str, Any]]:
    """Reconcile temporary snapshots left by a crash during a previous save.

    A save sequence that died between writing the temporary file and
    completing the (directory-fsynced) ``os.replace`` can leave one or more
    ``.state-*.tmp`` siblings next to the state file. On the next startup:

    * when the formal state file loads, it is authoritative: every leftover
      temporary is removed and the formal file is kept byte-for-byte;
    * when the formal file is missing, the newest (by modification time)
      verifiable ``version=1`` temporary snapshot is atomically renamed into
      place and its payload returned; older or unverifiable temporaries are
      removed;
    * when no temporary verifies, an empty state is created.

    A formal file that exists but is corrupt or wrong-version is never
    touched (startup keeps refusing) and the leftovers are left alone for
    inspection. Returns the recovered version-stripped payload, or ``None``
    when the caller should load/create the formal file normally.
    """
    directory = os.path.dirname(os.path.abspath(state_store.path))
    try:
        names = os.listdir(directory)
    except FileNotFoundError:
        return None
    tmp_paths = [
        os.path.join(directory, name) for name in names
        if name.startswith(".state-") and name.endswith(".tmp")]

    def discard(tmp_path: str) -> None:
        # Removing crash debris is best effort: a file that cannot be unlinked
        # is simply reconsidered at the next startup.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    document = state_store.load()
    if document is not None:
        # The committed file is valid; an un-renamed snapshot can only be
        # older or equal, so the leftovers are stale crash debris.
        for tmp_path in tmp_paths:
            discard(tmp_path)
        return None

    if not os.path.exists(state_store.path):
        candidates: List[str] = []
        for tmp_path in tmp_paths:
            try:
                os.stat(tmp_path)
            except OSError:
                continue
            candidates.append(tmp_path)
        candidates.sort(key=lambda path: os.stat(path).st_mtime,
                        reverse=True)
        for candidate in candidates:
            try:
                with open(candidate, "r", encoding="utf-8") as handle:
                    raw = handle.read()
                candidate_document = json.loads(raw)
                payload = _verify_snapshot_document(candidate_document)
            except (OSError, StateFileError,
                    json.JSONDecodeError, UnicodeDecodeError):
                payload = None
            if payload is None:
                discard(candidate)
                continue
            # Newest verifiable snapshot wins: promote it atomically and drop
            # every other leftover.
            try:
                os.replace(candidate, state_store.path)
                state_store._fsync_directory(directory)
            except OSError:
                # Promotion failed; leave everything in place and let the
                # normal missing-file path surface the I/O error.
                return None
            for stale in tmp_paths:
                if stale != candidate:
                    discard(stale)
            return payload
    return None


def attach_persistence(service: "DeviceService", path: str) -> JsonStateStore:
    """Load *path* into *service* and persist every subsequent change.

    Before loading, temporary snapshots left by a crashed previous process
    are reconciled by :func:`recover_crash_leftovers`: a valid formal file
    wins and the leftovers are cleaned; a missing formal file is atomically
    restored from the newest verifiable ``version=1`` leftover; with no valid
    snapshot an empty file is created. A corrupt or wrong-version formal file
    raises :class:`StateFileError` before the server starts; the in-memory
    store is never touched in that case. After attachment, each committed
    mutation rewrites the file atomically (temporary write, file fsync,
    ``os.replace``, parent-directory fsync) inside the same store-lock
    transaction. A write/fsync/replace failure rolls the in-memory store back
    to the last persisted state and raises :class:`PersistenceUnavailable`;
    neither memory nor the file advances.
    """
    state_store = JsonStateStore(path)
    recovered = recover_crash_leftovers(state_store)
    if recovered is not None:
        try:
            service.store.restore_state(recovered)
        except (ValueError, TypeError) as error:
            raise StateFileError(
                f"state file has a malformed payload: {error}") from None
    else:
        document = state_store.load()
        if document is None:
            try:
                state_store.save(service.store.snapshot_state())
            except OSError as error:
                raise StateFileError(
                    f"cannot create state file {path}: {error}") from None
        else:
            try:
                service.store.restore_state(
                    {key: value for key, value in document.items()
                     if key != "version"})
            except (ValueError, TypeError) as error:
                raise StateFileError(
                    f"state file has a malformed payload: {error}") from None

    # Canonical deep copy of the last durably-committed state. It is only ever
    # replaced with a snapshot whose save() succeeded, so it stays valid input
    # for restore_state().
    last_good: Dict[str, Any] = copy.deepcopy(service.store.snapshot_state())

    def persist() -> None:
        # Called under the store lock at the end of a mutation. Snapshot the
        # post-mutation state and try to durably commit it; on an I/O failure
        # roll the in-memory store back to the last good state before
        # surfacing the error, so the failed mutation is visible nowhere.
        pending = service.store.snapshot_state()
        try:
            state_store.save(pending)
        except OSError:
            service.store.restore_state(copy.deepcopy(last_good))
            raise PersistenceUnavailable(
                f"could not persist state to {state_store.path}") from None
        last_good.clear()
        last_good.update(copy.deepcopy(pending))

    service.store.on_change = persist
    return state_store
