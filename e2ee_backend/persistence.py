"""Durable JSON-file persistence for the device store.

The whole server state is kept in one versioned JSON document. A missing file
is created on first use; a corrupted file or a document whose version is not
understood makes the server refuse to start (state is never silently
discarded). Every change is written atomically: serialize to a temporary file
in the same directory, ``fsync`` it, take a hard-link backup of the current
target, ``os.replace`` it over the target, then ``fsync`` the parent
directory, so a crash never leaves a half-written or un-renamed state file.

Persistence is transactional with respect to the in-memory store: the change
hook fires while the store lock is held (the mutation is visible but not yet
committed to callers). When the durable write succeeds it becomes the new
last-known-good state. When writing the temporary file, ``fsync`` or the
atomic replace fails (an :class:`OSError`), the previous last-known-good state
is restored into memory under the same lock, the old state file's bytes *and
inode* are restored from the backup when the replace had landed, temporary
files are cleaned up by :meth:`JsonStateStore.save`, and
:class:`PersistenceUnavailable` is raised so the HTTP layer answers
503/field=data_file. The failed mutation therefore never advances either
memory or the file.

A crash between steps can leave ``.state-*.tmp`` (staged document) or
``.state-*.bak`` (pinned previous inode) files beside the target. At the next
:func:`attach_persistence` these are resolved by
:func:`recover_crash_leftovers`: a valid formal file stays authoritative and
the leftovers are removed; with the formal file missing the newest-mtime
leftover that parses as version=1, carries the cursor and ``key_events``
sections (the footprint of one complete durable transaction), and passes
every semantic restore check is atomically recovered into place; with none
valid the leftovers are removed and an empty state is created. An
existing-but-corrupt formal file still makes startup refuse rather than
being silently overwritten.
"""
from __future__ import annotations

import copy
import errno
import json
import os
import sys
import tempfile
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .storage import DeviceStore

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .service import DeviceService

#: Persistence format version understood by this build.
STATE_VERSION = 1

#: Temporary-snapshot name parts, mirrored when scanning crash leftovers
#: (see :meth:`JsonStateStore.save`).
_TMP_PREFIX = ".state-"
_TMP_SUFFIX = ".tmp"
#: Suffix of the pre-replace hard-link backup pinning the previous inode.
_BAK_SUFFIX = ".bak"


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

        Writes a sibling temporary file, fsyncs it, then ``os.replace`` and
        fsyncs the parent directory, so the rename itself is durable across
        a crash — the target is either the previous full document or the new
        full one, never a truncated mix.

        Before the replace a hard link to the current target is taken in the
        same directory. If the replace itself succeeds but the following
        directory fsync fails, that link is renamed back over the target,
        restoring the previous document's exact bytes *and inode*; every other
        failure simply removes the temporary files, leaving the target
        untouched. Any failure propagates the underlying :class:`OSError` to
        the caller, with no temporary or backup file left behind.
        """
        directory = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(directory, exist_ok=True)
        document = {"version": STATE_VERSION, **state}
        tmp_handle = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=directory, delete=False,
            prefix=".state-", suffix=".tmp")
        tmp_path = tmp_handle.name
        backup_path: Optional[str] = None
        replaced = False
        try:
            json.dump(document, tmp_handle, separators=(",", ":"),
                      ensure_ascii=False)
            tmp_handle.flush()
            os.fsync(tmp_handle.fileno())
            tmp_handle.close()
            # Pin the old inode before renaming over it. On the first save the
            # target does not exist yet, so there is no old state to preserve.
            backup_path = _hardlink_backup(directory, self.path)
            os.replace(tmp_path, self.path)
            replaced = True
            # The rename is only durable once the directory entry is flushed;
            # without this fsync a crash can leave the directory pointing at
            # the pre-rename entry even though the replacement landed.
            _fsync_directory(directory)
        except BaseException:
            tmp_handle.close()
            _remove_quietly(tmp_path)
            if replaced and backup_path is not None:
                # The replacement landed; rename the pinned old inode back over
                # the target atomically (this also drops the new inode), then
                # flush the rollback. If even this rename fails, leave the
                # backup on disk as a crash leftover rather than deleting the
                # last good copy; startup recovery finds it.
                try:
                    os.replace(backup_path, self.path)
                    backup_path = None
                    try:
                        _fsync_directory(directory)
                    except OSError:
                        pass
                except OSError:
                    # The rollback rename failed; leave the backup on disk as a
                    # crash leftover rather than deleting the last good copy.
                    backup_path = None
            if backup_path is not None:
                _remove_quietly(backup_path)
            raise
        # Commit is durable; drop the pinned old inode (best effort — a
        # leftover backup is harmless and cleaned up at the next startup).
        if backup_path is not None:
            _remove_quietly(backup_path)
            try:
                _fsync_directory(directory)
            except OSError:
                pass


def _fsync_directory(directory: str) -> None:
    """Flush *directory*'s metadata so a recent rename survives a crash.

    Directory fsync is a POSIX durability refinement that not every platform
    or file system supports. When the capability itself is unavailable — the
    directory cannot be opened for this purpose, or ``fsync`` reports
    :data:`errno.EINVAL` (a file system/kernel that does not implement it),
    with :data:`errno.EACCES` added on Windows where flushing a directory
    handle is a privileged/unsupported operation — it is skipped silently:
    the atomic ``os.replace`` already guarantees the file is never torn, so a
    skipped directory flush only weakens rename-durability, never safety.
    Every other :class:`OSError` (a genuine I/O failure) still propagates so
    the caller can roll the transaction back and answer 503.
    """
    unsupported_open = (errno.EACCES,) if sys.platform == "win32" else ()
    try:
        dir_fd = os.open(directory, os.O_RDONLY)
    except OSError as error:
        if error.errno in unsupported_open:
            return
        raise
    try:
        unsupported_fsync = {errno.EINVAL}
        if sys.platform == "win32":  # pragma: no cover - Windows-only branch
            unsupported_fsync.add(errno.EACCES)
        try:
            os.fsync(dir_fd)
        except OSError as error:
            if error.errno not in unsupported_fsync:
                raise
    finally:
        os.close(dir_fd)


def _hardlink_backup(directory: str, target: str) -> Optional[str]:
    """Pin the current *target*'s inode via a same-directory hard link.

    Returns the backup path, or ``None`` when *target* does not exist yet
    (first save: there is no old state to preserve). The link is created in
    the same directory so it is guaranteed to be on the same file system and
    the subsequent rollback ``os.replace`` is a pure metadata rename. Raises
    :class:`OSError` if the link cannot be made; the caller treats that like
    any pre-replace failure (the target is still untouched).
    """
    if not os.path.exists(target):
        return None
    fd, backup = tempfile.mkstemp(dir=directory, prefix=_TMP_PREFIX,
                                  suffix=_BAK_SUFFIX)
    os.close(fd)
    try:
        # mkstemp created an empty file under that unique name; remove it so
        # the name is free for the hard link.
        os.unlink(backup)
        os.link(target, backup)
    except OSError:
        _remove_quietly(backup)
        raise
    return backup


def _read_version1_document(path: str) -> Optional[Dict[str, Any]]:
    """Parse *path* as a version-1 state document, or ``None`` if absent.

    Mirrors :meth:`JsonStateStore.load` for arbitrary candidate paths (crash
    leftovers rather than the formal file): JSON object carrying
    ``version == 1`` or missing/unknown, returns the raw document or ``None``
    for a missing/unreadable/undecodable/wrong-version candidate.
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = handle.read()
    except OSError:
        return None
    try:
        document = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(document, dict):
        return None
    version = document.get("version")
    if not isinstance(version, int) or isinstance(version, bool) \
            or version != STATE_VERSION:
        return None
    return document


def _candidate_is_section_complete(document: Dict[str, Any]) -> bool:
    """Require the durable-transaction sections a real snapshot always has.

    Every snapshot produced by :meth:`JsonStateStore.save` is one full
    transaction and therefore carries the per-device sync cursor sections
    (``group_sync_cursors`` and ``message_sync_cursors``) and the
    ``key_events`` audit-chain section, each as a list — an empty list when
    the section has no records, but never absent. A leftover that parses and
    restores yet omits one of these sections is a partial/legacy document,
    not a crashed atomic commit: recovering it could silently drop a device's
    resume cursor or its audit chain, so it is not a valid crash-recovery
    candidate (a section-less *formal* file is still loaded leniently on the
    normal path; this gate applies only to leftover-snapshot recovery).
    """
    return all(isinstance(document.get(name), list)
               for name in ("group_sync_cursors", "message_sync_cursors",
                            "key_events"))


def _document_restores(document: Dict[str, Any]) -> bool:
    """Full semantic verification against :meth:`DeviceStore.restore_state`.

    A structurally valid version-1 document is only a recovery candidate when
    a fresh store accepts its payload — the same group-session references,
    sequence continuity, nonce-set, per-device cursor range and ``updated_at``
    checks applied to the formal file.
    """
    try:
        DeviceStore().restore_state(
            {key: value for key, value in document.items()
             if key != "version"})
    except (ValueError, TypeError):
        return False
    return True


def _leftover_tmp_paths(directory: str, target_path: str) -> List[str]:
    """List crash-leftover snapshots in *directory*.

    These carry the same ``.state-`` prefix :meth:`JsonStateStore.save` uses,
    with either the ``.tmp`` suffix of a staged new document or the ``.bak``
    suffix of a pre-replace hard-link backup (the previous committed inode).
    The formal target itself is never considered.
    """
    try:
        names = os.listdir(directory)
    except OSError:
        return []
    leftovers: List[str] = []
    for name in names:
        if not name.startswith(_TMP_PREFIX):
            continue
        if not (name.endswith(_TMP_SUFFIX) or name.endswith(_BAK_SUFFIX)):
            continue
        full = os.path.join(directory, name)
        if os.path.abspath(full) == os.path.abspath(target_path):
            continue
        leftovers.append(full)
    return leftovers


def _remove_quietly(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def recover_crash_leftovers(state_store: "JsonStateStore") -> None:
    """Resolve temporary snapshots a crashed process left beside the file.

    * A valid formal file wins outright: it is never overwritten, and every
      leftover is removed.
    * An existing but corrupt/invalid formal file is left untouched: the
      normal load then refuses startup, so present state is never silently
      discarded (and its leftovers are kept for inspection).
    * With the formal file missing, candidates are tried newest-mtime-first;
      the first that parses as version=1, explicitly carries the
      ``group_sync_cursors``/``message_sync_cursors``/``key_events`` sections
      (each a list — the footprint of one complete durable transaction), *and*
      passes full semantic verification is atomically ``os.replace``-d into
      place (plus a parent-directory fsync) and the remaining leftovers are
      removed. A section-less legacy/partial snapshot is never promoted, so a
      resume cursor or the audit chain cannot be silently lost.
    * When no candidate is valid, all leftovers are removed; the normal
      missing-file path in :func:`attach_persistence` then creates an empty
      state.
    """
    directory = os.path.dirname(os.path.abspath(state_store.path))
    leftovers = _leftover_tmp_paths(directory, state_store.path)
    if not leftovers:
        return

    if os.path.exists(state_store.path):
        # The formal file exists. A valid one stays authoritative and the
        # leftovers are stale; an invalid one makes the normal load refuse
        # startup, so neither the file nor the leftovers are touched here.
        formal = _read_version1_document(state_store.path)
        if formal is not None and _document_restores(formal):
            for path in leftovers:
                _remove_quietly(path)
        return

    # Formal file missing: recover the newest verifiable snapshot, newest
    # modification time first (ties broken by name for deterministic order).
    def mtime_key(path: str) -> Any:
        try:
            return (os.stat(path).st_mtime_ns, path)
        except OSError:
            return (-1, path)

    candidates = sorted(leftovers, key=mtime_key, reverse=True)
    recovered = None
    for candidate in candidates:
        document = _read_version1_document(candidate)
        if (document is not None
                and _candidate_is_section_complete(document)
                and _document_restores(document)):
            recovered = candidate
            break

    if recovered is not None:
        os.replace(recovered, state_store.path)
        _fsync_directory(directory)
        leftovers = [path for path in leftovers if path != recovered]
    for path in leftovers:
        _remove_quietly(path)


def attach_persistence(service: "DeviceService", path: str) -> JsonStateStore:
    """Load *path* into *service* and persist every subsequent change.

    A missing file is created immediately (with an empty version-1 document).
    A corrupt or wrong-version file raises :class:`StateFileError` before the
    server starts; the in-memory store is never touched in that case. After
    attachment, each committed mutation rewrites the file atomically inside
    the same store-lock transaction. A write/fsync/replace failure rolls the
    in-memory store back to the last persisted state and raises
    :class:`PersistenceUnavailable`; neither memory nor the file advances.
    """
    state_store = JsonStateStore(path)
    # First resolve any temporary snapshot a crashed previous process left in
    # the same directory: a valid formal file stays and leftovers are removed;
    # a missing formal file is atomically restored from the newest verifiable
    # version-1 leftover; with no valid candidate the leftovers are removed and
    # the missing-file branch below creates an empty state.
    try:
        recover_crash_leftovers(state_store)
    except OSError as error:
        raise StateFileError(
            f"cannot recover state file {path} from a crash leftover: "
            f"{error}") from None
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
