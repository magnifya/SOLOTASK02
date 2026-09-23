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
is restored into memory under the same lock and
:class:`PersistenceUnavailable` is raised so the HTTP layer answers
503/field=data_file; the failed mutation therefore never advances either
memory or the file.

After the replace has landed, a failed directory fsync first rolls the rename
back: the pinned old inode is renamed over the target and a second directory
fsync flushes the rollback. When that rollback itself cannot be completed
durably — the rollback rename fails, or the second fsync fails — the failure
stays *decidable*: the old inode is kept under a ``.state-*.bak`` backup and
the new snapshot is taken off the formal path (parked under a
``.state-*.quarantine`` name the recovery scan never promotes, or deleted), so
the formal path is missing. The store is then marked degraded and reports 503
for the failed request.

The degraded store also heals itself *within the same process*: the next
persist-able write runs, inside the storage lock, :meth:`JsonStateStore.heal`
before saving — the unique pinned ``.bak`` is fully verified (version=1,
commit generation, the complete-transaction sections, every cross-entity /
cursor / nonce / audit-chain semantic check, and exact equality with the
last committed in-memory state) and atomically promoted back onto the formal
path, same-transaction leftovers (``.quarantine``/``.tmp``) are removed, and
only then is that one triggering request committed (a single consecutive
``commit_seq`` advance). The request that originally got 503 is never
replayed; the healed commit carries only the later, current request. A
corrupt, dangling or ambiguous backup is refused promotion: the request gets
503/field=data_file, memory is rolled back, the formal path stays missing,
no leftover or generation moves, and the next write retries the heal. A
valid formal file always takes precedence over every backup. Restart
recovery remains the fallback that promotes the pinned backup.

A crash between steps can leave ``.state-*.tmp`` (staged document) or
``.state-*.bak`` (pinned previous inode) files beside the target. At the next
:func:`attach_persistence` these are resolved by
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
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

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
#: Suffix of a new snapshot taken off the formal path when the rollback
#: itself cannot be completed durably. Such a file is deliberately *not* a
#: crash-recovery candidate (the scan only promotes ``.tmp``/``.bak``): the
#: transaction it staged was already reported failed and rolled back in
#: memory, so it must never be resurrected at the next startup.
_QUARANTINE_SUFFIX = ".quarantine"

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
        #: Set once a transaction ends in the undecidable-on-disk state where
        #: the formal path is missing and the last committed inode survives
        #: only as a ``.bak`` (a failed rollback rename, or a rollback rename
        #: whose follow-up fsync failed). While set, every write first runs
        #: :meth:`heal` under the store lock: the pinned backup is fully
        #: verified and atomically promoted back onto the formal path, the
        #: same-transaction leftovers are cleaned, and only then the
        #: triggering write itself is committed. A heal that cannot verify or
        #: promote the backup keeps the store degraded and the write fails
        #: with 503/field=data_file, retryable by the next write.
        self.degraded = False

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
        restoring the previous document's exact bytes *and inode*, and the
        rollback is flushed with a second directory fsync; every other
        pre-replace failure simply removes the temporary file, leaving the
        target untouched.

        When the rollback itself cannot be completed durably — the rollback
        rename fails, or the second directory fsync fails — the store falls
        back to a *decidable* crash state instead of risking an un-committed
        snapshot at the formal path: the last committed inode is kept under a
        ``.bak`` backup and the new snapshot is taken off the formal path
        (renamed aside to a ``.quarantine`` name the recovery scan never
        promotes, or deleted), so the formal path is missing and the next
        write self-heals via :meth:`heal` (or, after a restart,
        :func:`recover_crash_leftovers`). Any failure propagates the
        underlying :class:`OSError` to the caller.
        """
        directory = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(directory, exist_ok=True)
        if self.degraded:
            # The formal path is missing and the last committed inode is
            # pinned only as a .bak. A save here could not be rolled back
            # (there is no target to back up), so the persistence hook must
            # run self.heal() first and only reach this point once the formal
            # path is restored. Refuse any save that bypassed the heal.
            raise OSError(
                f"state store for {self.path} is degraded; heal() must "
                f"promote the pinned backup before another save")
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
            if replaced and backup_path is not None:
                # The replacement landed but the following directory fsync
                # failed. Restore the pinned old inode atomically first (the
                # rename itself also drops the new inode)...
                try:
                    os.replace(backup_path, self.path)
                except OSError:
                    # ...the rollback rename itself failed. The formal path
                    # still holds the un-committed new inode, which must never
                    # become authoritative: move it off the path (the old
                    # inode stays pinned in the .bak), so the next startup
                    # sees a missing formal file and recovers that backup.
                    _vacate_new_snapshot(directory, self.path)
                    # Keep the backup: it is now the only copy of the last
                    # committed inode, so it must not be cleaned up below.
                    backup_path = None
                    self.degraded = True
                else:
                    backup_path = None
                    try:
                        _fsync_directory(directory)
                    except OSError:
                        # The rollback rename restored the old inode but its
                        # durability could not be flushed: this step failed
                        # too, so the transaction is still failed, never
                        # committed. Re-pin the restored inode as a .bak and
                        # vacate the formal path again, leaving exactly the
                        # "formal missing + old-inode .bak" state startup
                        # recovery resolves. When the path could not be
                        # vacated the old inode is still authoritative at it
                        # (the safe outcome), so the store is not degraded.
                        if _repin_old_inode(directory, self.path):
                            self.degraded = True
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
        # The generation advances exactly once per durable commit.
        self.commit_seq = seq + 1

    def heal(self, baseline: Dict[str, Any]) -> None:
        """Self-heal the degraded formal path from the pinned ``.bak``.

        Called under the store lock by the persistence hook at the start of
        the next persist-able write after an undecidable write failure. The
        pinned backup is verified in full — version=1, a valid commit
        generation exactly one below the next save's generation, the whole
        cross-entity/cursor/nonce/audit-chain semantic restore, and exact
        equality with the process's last committed in-memory state — before
        it is promoted. Promotion is a same-directory *hard link*: the backup
        name keeps pinning the inode until the formal directory entry is
        fsync-durable, so a crash at this instant leaves either a valid
        formal file or the still-present backup, never neither. Only then are
        the same-transaction leftovers (redundant backups, staged ``.tmp``
        snapshots and the un-committed ``.quarantine``) removed.

        A later write commit still goes through the normal atomic
        :meth:`save`, so healing consumes no generation: the healed commit
        advances ``commit_seq`` exactly once and carries only that triggering
        request — the earlier request that got 503 is never replayed.

        Raises :class:`OSError` (mapped to 503/field=data_file by the caller)
        when no verifiable backup exists, backups disagree, or the promotion
        cannot be made durable; in every such case the formal path stays
        missing, no leftover and no generation moves, :attr:`degraded` stays
        set, and the next write retries this heal. A formal file that
        reappeared valid and matching always takes precedence over backups.
        """
        if not self.degraded:
            return
        directory = os.path.dirname(os.path.abspath(self.path))
        leftovers = _leftover_tmp_paths(directory, self.path)
        quarantined = _quarantined_paths(directory, self.path)

        if os.path.exists(self.path):
            # The formal path can only reappear out-of-band while degraded. A
            # valid one that matches the committed state always wins: keep it,
            # sweep every leftover and resume normal saves. Anything else is
            # left untouched — exactly the refusal a restart would give.
            formal = _read_version1_document(self.path)
            if (formal is not None
                    and (COMMIT_SEQ_KEY not in formal
                         or _valid_commit_seq(formal[COMMIT_SEQ_KEY]))
                    and _commit_seq_of(formal) == self.commit_seq - 1
                    and _document_restores(formal)
                    and _document_payload_equals(formal, baseline)):
                _sweep_leftovers(directory, leftovers + quarantined)
                self.degraded = False
                return
            raise OSError(
                f"cannot self-heal {self.path}: the formal file reappeared "
                f"in an unexpected state")

        # Verify every leftover against this process's last committed state.
        # Equality with *baseline* is the anti-dangling gate in addition to
        # the full semantic restore: a staged new snapshot (the transaction
        # already reported 503 and rolled back in memory) parses and may even
        # restore, but it can never equal the committed baseline and is
        # therefore never eligible here; quarantine files are never even
        # considered.
        eligible: List[str] = []
        for candidate in leftovers:
            try:
                with open(candidate, "rb") as handle:
                    raw = handle.read()
            except OSError:
                continue
            try:
                document = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if not isinstance(document, dict):
                continue
            version = document.get("version")
            if not isinstance(version, int) or isinstance(version, bool) \
                    or version != STATE_VERSION:
                continue
            if COMMIT_SEQ_KEY in document and not _valid_commit_seq(
                    document[COMMIT_SEQ_KEY]):
                continue
            # The backup must pin the immediately-previous generation (a
            # field-less legacy backup ranks at 0 with the counter at 1).
            if _commit_seq_of(document) != self.commit_seq - 1:
                continue
            if not _document_restores(document):
                continue
            if not _document_payload_equals(document, baseline):
                continue
            eligible.append(candidate)

        if not eligible:
            raise OSError(
                f"cannot self-heal {self.path}: no verifiable pinned backup "
                f"of the last committed state")
        # Every eligible pin is the same generation and equals the committed
        # state, so choosing among them cannot diverge state; prefer a .bak
        # pin over a byte-identical staged .tmp, then the stable name order.
        eligible.sort(key=lambda path: (
            not path.endswith(_BAK_SUFFIX), path))
        chosen = eligible[0]

        # Promote with a hard link, never a rename: the backup keeps pinning
        # the inode until the new formal directory entry is fsync-durable.
        linked = False
        try:
            os.link(chosen, self.path)
            linked = True
            _fsync_directory(directory)
        except OSError:
            if linked:
                # The entry may not be durable. Drop the name we just added —
                # the inode stays pinned under the backup name — and best
                # effort flush the removal, then hand the failure back.
                _remove_quietly(self.path)
                try:
                    _fsync_directory(directory)
                except OSError:
                    pass
            raise

        # The formal path durably names the committed inode; every pinned
        # copy, staged snapshot and demoted (quarantined) new snapshot is now
        # stale garbage. Sweep failures are harmless: a valid formal file
        # always wins the leftover scan at every later startup.
        _sweep_leftovers(directory, leftovers + quarantined)
        self.degraded = False


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


def _quarantine_path(directory: str) -> str:
    """Reserve a unique same-directory name for a demoted new snapshot."""
    fd, quarantine = tempfile.mkstemp(dir=directory, prefix=_TMP_PREFIX,
                                      suffix=_QUARANTINE_SUFFIX)
    os.close(fd)
    # The empty placeholder only reserves the unique name; free it for the
    # rename that moves the new snapshot aside.
    os.unlink(quarantine)
    return quarantine


def _vacate_new_snapshot(directory: str, target: str) -> None:
    """Take the un-committed new snapshot off the formal path.

    Used when a post-replace failure cannot be rolled back durably: the new
    inode must never become authoritative, so it is renamed aside to a
    ``.quarantine`` name (which the crash-leftover scan never promotes), and
    if even that rename is impossible it is deleted. Either way the formal
    path is missing afterwards; a pinned ``.bak`` of the last committed
    inode is the recovery candidate the next startup promotes. Best effort
    only — a failure here is swallowed because the caller still reports the
    transaction as failed and the old inode stays pinned elsewhere.
    """
    try:
        quarantine = _quarantine_path(directory)
        try:
            os.replace(target, quarantine)
        except OSError:
            _remove_quietly(quarantine)
            raise
    except OSError:
        _remove_quietly(target)


def _repin_old_inode(directory: str, target: str) -> bool:
    """Re-pin the restored old inode as a ``.bak`` and vacate *target*.

    The rollback rename restored the last committed inode at *target*, but
    flushing that rollback failed, so it cannot be treated as durable. Pin
    the inode under a fresh backup name, then drop *target* (a second hard
    link to the same inode, so the unlink loses nothing), recreating the
    same "formal missing + old-inode .bak" triage a plain rollback failure
    leaves. Returns ``True`` when the formal path was vacated. If the inode
    cannot be pinned a second time it simply stays on the formal path — the
    last good state — and the function returns ``False`` so the caller does
    not mark the store degraded while a good file is authoritative.
    """
    try:
        fd, backup = tempfile.mkstemp(dir=directory, prefix=_TMP_PREFIX,
                                      suffix=_BAK_SUFFIX)
        os.close(fd)
        os.unlink(backup)
        os.link(target, backup)
    except OSError:
        _remove_quietly(backup)
        # A second hard link cannot be made; instead rename the restored
        # target itself to the backup name — one atomic rename both keeps
        # the old inode (under the .bak) and vacates the formal path.
        try:
            fd, backup = tempfile.mkstemp(dir=directory, prefix=_TMP_PREFIX,
                                          suffix=_BAK_SUFFIX)
            os.close(fd)
            os.unlink(backup)
            os.replace(target, backup)
        except OSError:
            _remove_quietly(backup)
            return False
    else:
        # target and backup now name the same restored inode; dropping
        # target leaves the inode pinned exactly once, under the backup name.
        _remove_quietly(target)
    return not os.path.exists(target)


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


def _document_payload_equals(document: Dict[str, Any],
                             baseline: Dict[str, Any]) -> bool:
    """Compare a document's payload with the process's last committed state.

    The on-disk document wraps the store snapshot with ``version`` and
    ``commit_seq``; strip those envelope fields and compare the remainder,
    semantically, to *baseline* (a value straight from
    :meth:`DeviceStore.snapshot_state`). Comparison goes through a fresh
    :meth:`DeviceStore.restore_state` + :meth:`DeviceStore.snapshot_state`
    round-trip rather than a raw dict diff, so an older *committed* document
    that legitimately predates an optional section (``used_nonces``, the
    sync-cursor sections, ``key_events`` ...) still matches a baseline whose
    snapshot fills those with their defaults — while a newer, un-committed
    snapshot cannot, because its extra business state (a revoked device, an
    appended message) round-trips to a different snapshot. This is the
    anti-dangling gate for self-heal: verifiable but newer content is never
    promoted. Returns ``False`` for any document that does not restore.
    """
    payload = {key: value for key, value in document.items()
               if key not in ("version", COMMIT_SEQ_KEY)}
    try:
        candidate_store = DeviceStore()
        candidate_store.restore_state(payload)
        restored = candidate_store.snapshot_state()
    except (ValueError, TypeError):
        return False
    return restored == baseline


def _sweep_leftovers(directory: str, paths: List[str]) -> None:
    """Best-effort remove crash/transaction leftovers and flush the directory.

    Used once a valid formal file is (re-)established authoritatively — by
    self-heal or restart recovery — after which pinned backups, staged
    ``.tmp`` snapshots and demoted ``.quarantine`` files are pure garbage.
    Removal failures are swallowed: a valid formal file wins the leftover
    scan at the next startup anyway, so a stray file can never become
    authoritative.
    """
    for path in paths:
        _remove_quietly(path)
    try:
        _fsync_directory(directory)
    except OSError:
        pass


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


def _quarantined_paths(directory: str, target_path: str) -> List[str]:
    """List demoted new snapshots parked under a ``.quarantine`` name.

    These are never recovery candidates: a quarantined snapshot staged a
    transaction that was reported failed and rolled back in memory, so it is
    only garbage to remove once a valid formal state is authoritative (or a
    candidate has been recovered). A corrupt formal file keeps them on disk
    for inspection, like every other leftover.
    """
    try:
        names = os.listdir(directory)
    except OSError:
        return []
    quarantined: List[str] = []
    for name in names:
        if name.startswith(_TMP_PREFIX) and name.endswith(_QUARANTINE_SUFFIX):
            full = os.path.join(directory, name)
            if os.path.abspath(full) != os.path.abspath(target_path):
                quarantined.append(full)
    return quarantined


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
      They rank by commit generation first — the highest ``commit_seq``
      wins, so an older-mtime snapshot from a later generation is never
      lost to a newer-mtime stale one — with modification time newest-first
      among equal generations (a missing field counts as generation 0).
      When every candidate predates commit generations (none carries the
      field), selection stays the legacy newest-mtime rule. The chosen
      snapshot is atomically ``os.replace``-d into place (plus a
      parent-directory fsync) and the remaining leftovers are removed. A
      section-less legacy/partial snapshot is never promoted, so a resume
      cursor or the audit chain cannot be silently lost.
    * When no candidate is valid, all leftovers are removed; the normal
      missing-file path in :func:`attach_persistence` then creates an empty
      state.
    """
    directory = os.path.dirname(os.path.abspath(state_store.path))
    leftovers = _leftover_tmp_paths(directory, state_store.path)
    quarantined = _quarantined_paths(directory, state_store.path)
    if not leftovers and not quarantined:
        return

    if os.path.exists(state_store.path):
        # The formal file exists. A valid one stays authoritative and the
        # leftovers (including demoted snapshots parked in quarantine) are
        # stale; an invalid one (including a present but malformed
        # commit_seq) makes the normal load refuse startup, so neither the
        # file nor the leftovers are touched here.
        formal = _read_version1_document(state_store.path)
        formal_valid = (
            formal is not None
            and (COMMIT_SEQ_KEY not in formal
                 or _valid_commit_seq(formal[COMMIT_SEQ_KEY]))
            and _document_restores(formal))
        if formal_valid:
            for path in leftovers + quarantined:
                _remove_quietly(path)
        return

    # Formal file missing. Verify every leftover first (unverifiable ones are
    # skipped), then rank the survivors by commit generation, not mtime: the
    # highest commit_seq wins even when a stale older-generation snapshot has
    # a newer mtime, and equal generations order newest-mtime-first. A
    # leftover without the field is a pre-generation snapshot at seq 0; when
    # every candidate is such a snapshot the ranking collapses to the legacy
    # pure-mtime rule. Ties in mtime break by name, deterministically.
    verified: List[Tuple[str, int, int]] = []
    any_with_seq = False
    for candidate in leftovers:
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
    # A demoted snapshot is never a recovery candidate (its transaction was
    # reported failed and rolled back); once the formal path has been
    # resolved above, any quarantine file is pure garbage.
    for path in quarantined:
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
        if state_store.degraded:
            # A previous transaction ended undecidable on disk: the formal
            # path is missing and the last committed inode survives only as
            # a pinned .bak. Before this (the next) write may commit, verify
            # and atomically promote that backup inside the same storage
            # lock, cleaning the failed transaction's leftovers. Memory was
            # rolled back to last_good after the 503 and, like every later
            # failed attempt, restored to it again, so the only difference
            # between pending and last_good is *this* current request — the
            # earlier 503 request is never replayed.
            try:
                state_store.heal(copy.deepcopy(last_good))
            except OSError:
                service.store.restore_state(copy.deepcopy(last_good))
                raise PersistenceUnavailable(
                    f"could not self-heal state file {state_store.path}") \
                    from None
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
