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

A crash between steps can leave ``.state-*.tmp`` (staged document),
``.state-*.bak`` (pinned previous inode), or ``.state-*.bad`` (a snapshot
isolated after a post-replace rollback could not be completed) files beside
the target. At the next :func:`attach_persistence` these are resolved by
:func:`recover_crash_leftovers`: a valid formal file stays authoritative and
the leftovers are removed; with the formal file missing the highest
``commit_seq`` leftover that parses as version=1, carries the cursor and
``key_events`` sections (the footprint of one complete durable transaction),
and passes every semantic restore check is atomically recovered into place
(modification time newest-first breaks equal generations, and files without
the field keep the legacy newest-mtime rule); with none valid the leftovers
are removed and an empty state is created. An existing-but-corrupt formal
file still makes startup refuse rather than being silently overwritten.

Every document carries a strictly-consecutive top-level ``commit_seq``: the
first empty state is written at 0 and each successful durable transaction
advances it exactly once (a failed write consumes no generation). A legacy
version=1 file without the field loads as generation 0; a present field must
be a non-negative integer or startup refuses (bool, negative, float and
string values are rejected) with the file untouched.
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
#: Suffix of a snapshot isolated after a failed post-replace rollback: the
#: transaction was definitively aborted (503), so such a file is cleaned up
#: and is *never* a recovery candidate, even when it carries a newer
#: ``commit_seq`` than the preserved ``.bak`` of the last committed inode.
_QUARANTINE_SUFFIX = ".bad"

#: Top-level commit-generation field. Every successful durable transaction
#: writes a document whose ``commit_seq`` is exactly one higher than the
#: previous one; the first (empty) state is created at 0. A legacy version-1
#: file written before this field loads as if it carried 0.
COMMIT_SEQ_KEY = "commit_seq"


def _valid_commit_seq(value: Any) -> bool:
    """A commit generation is a non-negative JSON integer (never a bool)."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _commit_seq_of(document: Dict[str, Any]) -> int:
    """Rank a candidate by its commit generation; a missing/malformed one is 0.

    Leftover snapshots are ranked leniently here (a leftover without the field
    predates commit generations and sorts at generation 0); the *formal*
    file's field is validated strictly by :meth:`JsonStateStore.load`.
    """
    value = document.get(COMMIT_SEQ_KEY, 0)
    return value if _valid_commit_seq(value) else 0


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
        #: Commit generation the *next* save writes. The first (empty) state
        #: is created at 0; :func:`attach_persistence` re-seeds it from a
        #: loaded or recovered document (plus one) so generations stay
        #: strictly consecutive across restarts and recovery. It advances
        #: only once a save has completed durably, so a failed transaction
        #: never consumes a generation.
        self.commit_seq = 0

    def load(self) -> Optional[Dict[str, Any]]:
        """Load the document, or ``None`` when the file does not exist yet.

        Raises :class:`StateFileError` if the file cannot be decoded as JSON,
        is not a JSON object, carries a missing/unknown ``version``, or has a
        ``commit_seq`` that is present but not a non-negative integer (a bool,
        negative number, float or string). A file without ``commit_seq`` is a
        legacy version-1 document and loads as generation 0.
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
        if COMMIT_SEQ_KEY in document:
            commit_seq = document[COMMIT_SEQ_KEY]
            if not _valid_commit_seq(commit_seq):
                raise StateFileError(
                    f"state document '{COMMIT_SEQ_KEY}' must be a "
                    f"non-negative integer")
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
        restoring the previous document's exact bytes *and inode*, then the
        rollback is flushed with another directory fsync; every other failure
        simply removes the temporary files, leaving the target untouched.

        If that rollback rename itself fails, or the following rollback
        directory fsync fails, the old inode is *preserved* as the ``.bak``
        backup and the new (already replaced) snapshot is isolated under a
        ``.bad`` quarantine name and unlinked, leaving the formal target
        absent. The preserved ``.bak`` is the single last-known-good inode;
        on the next startup the missing formal file is recovered from it by
        :func:`recover_crash_leftovers`, and the quarantined snapshot (a
        failed transaction that may carry a newer ``commit_seq``) is never
        chosen. On the very first save there is no old inode to preserve, so
        the new snapshot is merely isolated the same way. Any failure
        propagates the underlying :class:`OSError` to the caller (it surfaces
        as 503/field=data_file with memory rolled back), regardless of
        whether the best-effort quarantine housekeeping succeeded.
        """
        directory = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(directory, exist_ok=True)
        document = {"version": STATE_VERSION, **state}
        # Stamp the commit generation this save is committing. The counter
        # advances only after the replace and directory fsync succeed, so a
        # failed transaction (rolled back below) consumes no generation and
        # the next retry rewrites the same one — generations on disk stay
        # strictly consecutive.
        seq = self.commit_seq
        document[COMMIT_SEQ_KEY] = seq
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
            if replaced:
                # The replacement landed and the transaction then failed
                # (only the post-replace directory fsync can fail at this
                # point). Best-effort restore the pinned old inode; if that
                # cannot be made durable the new snapshot is quarantined so
                # the formal path never ends up holding an uncommitted
                # document (see the helper for the exact on-disk outcome).
                _abort_replaced_transaction(
                    directory, self.path, backup_path)
            elif backup_path is not None:
                # The replace never ran, so the target still holds the old
                # inode and the precautionary backup is redundant.
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
        # The generation advances exactly once per durable commit.
        self.commit_seq = seq + 1


def _quarantine_path(directory: str) -> str:
    """Return a fresh, unused same-directory ``.state-*.bad`` path."""
    fd, path = tempfile.mkstemp(dir=directory, prefix=_TMP_PREFIX,
                                suffix=_QUARANTINE_SUFFIX)
    os.close(fd)
    # mkstemp created an empty file under that unique name; free the name so
    # the snapshot can be renamed onto it.
    os.unlink(path)
    return path


def _isolate_new_snapshot(directory: str, target: str) -> None:
    """Best-effort move the uncommitted snapshot at *target* off the path.

    Renames *target* to a unique ``.bad`` quarantine name and unlinks it, so
    the formal path is absent and the failed snapshot can never be taken for
    a committed generation by startup recovery. If the quarantine rename
    fails, a direct unlink of *target* is tried instead — either way the
    formal path is vacated when the directory is writable at all. If only the
    final unlink fails the snapshot remains isolated under ``.bad``, which
    startup recovery removes without ever selecting; only a completely
    unwritable directory leaves it in place (the transaction's I/O is already
    failing in that case). The directory flush afterwards is best effort.
    """
    try:
        quarantine = _quarantine_path(directory)
    except OSError:
        quarantine = None
    if quarantine is not None:
        try:
            os.replace(target, quarantine)
        except OSError:
            quarantine = None
    if quarantine is not None:
        _remove_quietly(quarantine)
    else:
        _remove_quietly(target)
    try:
        _fsync_directory(directory)
    except OSError:
        pass


def _abort_replaced_transaction(directory: str, target: str,
                                backup_path: Optional[str]) -> None:
    """Best-effort undo once the replacement has already landed.

    Called from :meth:`JsonStateStore.save`'s exception path after
    ``os.replace`` succeeded: *target* holds the new, uncommitted document
    and *backup_path* (unless this was the first save) is the hard-linked
    previous inode. The preferred outcome is a clean rollback — rename the
    backup back over the target and flush the directory — which restores the
    previous document's bytes and inode exactly.

    If the rollback rename fails, or the flush of it fails, the old inode is
    preserved as a ``.bak`` and the new snapshot is isolated/dropped so the
    formal path is missing; startup recovery then promotes the single
    last-known-good backup and never considers the quarantined generation.
    With no backup (first save, no previous document existed) the new
    snapshot is merely isolated, leaving the formal path absent. Everything
    here is best effort and must not mask the transaction's original
    :class:`OSError`, which the caller re-raises.
    """
    if backup_path is None:
        # No previous inode to preserve: just vacate the formal path.
        _isolate_new_snapshot(directory, target)
        return
    try:
        os.replace(backup_path, target)
    except OSError:
        # The old inode cannot be put back. Keep its .bak exactly where it is
        # and move the uncommitted snapshot off the formal path.
        _isolate_new_snapshot(directory, target)
        return
    # The old inode is back on the formal path. Make the rollback rename
    # durable. A platform/filesystem that cannot fsync directories returns
    # False and the rollback still stands; only a raised I/O error leaves the
    # rename of uncertain durability.
    try:
        _fsync_directory(directory)
    except OSError:
        # Re-pin the restored inode as a .bak, then vacate the formal path so
        # a crash can never reveal the (possibly still directory-linked)
        # uncommitted snapshot: startup sees the formal file missing plus one
        # good backup and recovers deterministically.
        try:
            fresh_backup = _hardlink_backup(directory, target)
        except OSError:
            fresh_backup = None
        if fresh_backup is not None:
            _remove_quietly(target)
            try:
                _fsync_directory(directory)
            except OSError:
                pass


def _directory_fsync_unsupported(error: OSError) -> bool:
    """Classify *error* as "this platform/filesystem cannot fsync a directory".

    Some POSIX network or special filesystems reject ``fsync`` on a
    directory fd with ``EINVAL`` (others use ``ENOTSUP``/``EOPNOTSUPP``);
    on Windows a directory cannot be opened for this at all and the CRT
    fails with ``EACCES`` (older runtimes raise ``EISDIR``). None of these
    means the data write itself failed — the file fsync and the atomic
    rename already landed — so the directory flush is safely skipped
    instead of failing (and rolling back) the whole transaction. Genuine
    I/O errors (``EIO``, ``EBADF``, permission problems elsewhere, ...)
    still propagate.
    """
    unsupported = {errno.EINVAL, errno.ENOTSUP,
                   getattr(errno, "EOPNOTSUPP", None)}
    if error.errno in unsupported:
        return True
    if sys.platform == "win32" and error.errno in (errno.EACCES, errno.EISDIR):
        return True
    return False


def _fsync_directory(directory: str) -> bool:
    """Flush *directory*'s metadata so a recent rename survives a crash.

    Returns ``True`` when the flush ran. When the platform or underlying
    filesystem cannot fsync directories, the capability is safely skipped
    (returns ``False``) without reporting a failure; every other
    :class:`OSError` propagates to the caller. On platforms without
    ``O_DIRECTORY`` the plain read-only open still works.
    """
    flags = os.O_RDONLY
    try:
        dir_fd = os.open(directory, flags)
    except OSError as error:
        if _directory_fsync_unsupported(error):
            return False
        raise
    try:
        os.fsync(dir_fd)
    except OSError as error:
        if _directory_fsync_unsupported(error):
            return False
        raise
    finally:
        os.close(dir_fd)
    return True


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
    with the ``.tmp`` suffix of a staged new document, the ``.bak`` suffix of
    a pre-replace hard-link backup (the previous committed inode), or the
    ``.bad`` suffix of a snapshot isolated after a failed post-replace
    rollback. The formal target itself is never considered.
    """
    try:
        names = os.listdir(directory)
    except OSError:
        return []
    leftovers: List[str] = []
    for name in names:
        if not name.startswith(_TMP_PREFIX):
            continue
        if not (name.endswith(_TMP_SUFFIX) or name.endswith(_BAK_SUFFIX)
                or name.endswith(_QUARANTINE_SUFFIX)):
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
    * With the formal file missing, only candidates that parse as
      version=1, explicitly carry the
      ``group_sync_cursors``/``message_sync_cursors``/``key_events``
      sections (each a list — the footprint of one complete durable
      transaction), *and* pass full semantic verification are considered.
      A ``.bad`` file is never a candidate: it is a snapshot of a transaction
      that was definitively aborted (the post-replace rollback could not be
      made durable and a 503 was already returned), even though it may carry
      a newer ``commit_seq``; it is only removed. Candidates rank by commit
      generation first — the highest ``commit_seq`` wins, so an older-mtime
      snapshot from a later generation is never lost to a newer-mtime stale
      one — with modification time newest-first among equal generations (a
      missing field counts as generation 0). When every candidate predates
      commit generations (none carries the field), selection stays the
      legacy newest-mtime rule. The chosen snapshot is atomically
      ``os.replace``-d into place (plus a parent-directory fsync) and the
      remaining leftovers — including every ``.bad`` quarantine file — are
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
        # leftovers (staged snapshots, old backups, quarantine files) are
        # stale and removed; an invalid one (including a present but malformed
        # commit_seq) makes the normal load refuse startup, so neither the
        # file nor the leftovers are touched here.
        formal = _read_version1_document(state_store.path)
        formal_valid = (
            formal is not None
            and (COMMIT_SEQ_KEY not in formal
                 or _valid_commit_seq(formal[COMMIT_SEQ_KEY]))
            and _document_restores(formal))
        if formal_valid:
            for path in leftovers:
                _remove_quietly(path)
        return

    # Formal file missing. Verify every leftover first (unverifiable ones are
    # skipped), then rank the survivors by commit generation, not mtime: the
    # highest commit_seq wins even when a stale older-generation snapshot has
    # a newer mtime, and equal generations order newest-mtime-first. A
    # leftover without the field is a pre-generation snapshot at seq 0; when
    # every candidate is such a snapshot the ranking collapses to the legacy
    # pure-mtime rule. Ties in mtime break by name, deterministically.
    # A .bad quarantine file is a definitively-aborted transaction (503 was
    # already returned and memory rolled back), so it never enters the
    # ranking however new its generation looks; it is swept with the rest.
    verified: List[Tuple[str, int, int]] = []
    any_with_seq = False
    for candidate in leftovers:
        if candidate.endswith(_QUARANTINE_SUFFIX):
            continue
        document = _read_version1_document(candidate)
        if document is None or not _candidate_is_section_complete(document):
            continue
        # A present-but-malformed generation (bool, negative, float, string)
        # can never come from this build's atomic writer; promoting it would
        # only fail the strict formal-file load afterwards, so it is not a
        # verifiable candidate. A missing field is the legacy case and ranks
        # at generation 0.
        if COMMIT_SEQ_KEY in document and not _valid_commit_seq(
                document[COMMIT_SEQ_KEY]):
            continue
        if COMMIT_SEQ_KEY in document:
            any_with_seq = True
        if not _document_restores(document):
            continue
        try:
            mtime_ns = os.stat(candidate).st_mtime_ns
        except OSError:
            mtime_ns = -1
        verified.append((candidate, _commit_seq_of(document), mtime_ns))

    recovered = None
    if verified:
        if any_with_seq:
            # Generation-first ordering; missing field ranks at 0, so a
            # present-generation snapshot always beats a field-less legacy
            # one regardless of mtime.
            verified.sort(key=lambda item: (item[1], item[2], item[0]))
        else:
            # Every candidate predates commit generations: keep the legacy
            # newest-mtime rule.
            verified.sort(key=lambda item: (item[2], item[0]))
        recovered = verified[-1][0]

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
        # Resume commit generations exactly where the durable document left
        # off (a legacy file without the field is generation 0): the next
        # successful transaction writes commit_seq + 1, never repeating or
        # skipping a generation after restart or crash recovery.
        state_store.commit_seq = _commit_seq_of(document) + 1
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
