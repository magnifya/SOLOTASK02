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

When even the decidable fallback cannot be achieved — the un-committed new
snapshot can be neither renamed aside nor deleted, so the formal path cannot be
proven missing — the store enters the *blocking state* (a sibling
``.state-*.block`` marker persists it across restarts). It never serves
that residual formal file as authoritative: every subsequent write is
503/field=data_file, in this process and after a restart, until the path is
vacated with exactly one verifiable backup (heal/restart then promote it and
clear the marker), or a file provably equal to the committed state is restored
onto the formal path (a valid formal always wins). Two or more
verifiable backups make promotion ambiguous and are likewise refused rather than picked by
name or mtime.

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

Beside the state file every normal commit maintains an append-only integrity
sidecar at ``<state>.integrity`` (a compact version=1 JSON document with one
``commit_seq``/``state_hash``/``prev_hash``/``hash`` entry per committed
generation), written in the same locked two-file transaction. The state
document records the format with ``integrity_log_version=1``; a legacy file
without that marker and without a sidecar still starts and anchors the chain
on its first commit, while any marker/sidecar disagreement or a broken /
tail-mismatched chain makes startup refuse with both files untouched. The
sidecar backs ``GET /v1/persistence/integrity/history`` and its ascending,
cursor-paged variant ``GET /v1/persistence/integrity/history/page``
(``commit_seq > after``, at most ``limit``).
"""
from __future__ import annotations

import copy
import errno
import hashlib
import json
import os
import sys
import tempfile
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from .storage import (
    DeviceStore,
    canonical_integrity_snapshot,
    integrity_state_hash,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .service import DeviceService

#: Persistence format version understood by this build.
STATE_VERSION = 1

#: Integrity sidecar log format version.
INTEGRITY_LOG_VERSION = 1
#: State-document field recording which integrity-sidecar format this file's
#: store maintains (absent on legacy documents that predate the sidecar).
INTEGRITY_LOG_VERSION_KEY = "integrity_log_version"
#: Suffix appended to the state-file path for its integrity-history sidecar.
_INTEGRITY_LOG_SUFFIX = ".integrity"

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
#: Suffix of the durable marker left when a failed transaction could not be made
#: decidable: the formal path could not be vacated, so a possibly-un-committed
#: residual formal file sits beside a ``.bak`` of the last committed
#: inode. While such a marker exists the store is in the *blocking state*:
#: writes are refused (503/field=data_file) and the residual formal file is
#: never treated as authoritative, until an operator removes the marker (after resolving
#: the on-disk state) — or until heal finds the formal path missing with
#: exactly one verifiable backup, at which point it clears the marker itself.
_BLOCK_SUFFIX = ".block"

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


class IntegrityCheckError(Exception):
    """A read-only integrity probe could not verify the state file.

    The file is missing/unreadable, cannot be parsed as a version-1 JSON
    object, carries an invalid ``commit_seq``, fails the full semantic
    restore, or its canonical snapshot differs from the in-memory state.
    Nothing is read into the live store and no file, inode, cursor or
    generation is touched; the HTTP layer reports this as
    503/field=data_file.
    """


class IntegrityLogError(Exception):
    """The integrity-history sidecar is missing or fails every check.

    Raised at startup when the ``integrity_log_version`` marker and the
    sibling ``<state>.integrity`` log disagree (one present without the
    other, or an unknown marker value), or when the log document is
    structurally invalid, its hash chain is broken, an entry hash does not
    verify, or its last entry does not match the state file's generation.
    The store refuses to start and leaves both files untouched.
    """


def _integrity_entry_hash(commit_seq: int, state_hash: str,
                          prev_hash: str) -> str:
    """Hash one integrity-log entry: the entry without ``hash``.

    Keys are sorted, separators compact and Unicode written literally; the
    UTF-8 bytes are SHA-256 digested to lowercase hex, exactly like the
    key-audit event chain.
    """
    document = {"commit_seq": commit_seq, "state_hash": state_hash,
                "prev_hash": prev_hash}
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _is_hex64(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 \
        and all(ch in "0123456789abcdef" for ch in value)


def _read_integrity_log(path: str) -> Optional[Dict[str, Any]]:
    """Parse the integrity sidecar, or ``None`` when it does not exist.

    Returns the raw parsed document (validation happens in
    :func:`_verify_integrity_log`). Unreadable/undecodable files raise
    :class:`IntegrityLogError` rather than returning ``None``: only a
    genuinely absent file means "no sidecar".
    """
    try:
        with open(path, "rb") as handle:
            raw = handle.read()
    except FileNotFoundError:
        return None
    except OSError as error:
        raise IntegrityLogError(
            f"cannot read integrity log {path}: {error}") from None
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise IntegrityLogError(
            f"integrity log is not valid JSON: {path} ({error})") from None
    if not isinstance(document, dict):
        raise IntegrityLogError("integrity log must be a JSON object")
    return document


def _verify_integrity_log(document: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Fully validate a parsed integrity sidecar; return its entries.

    Enforces top-level shape (``version == 1``, an ``entries`` list, no
    extra keys), each entry's shape and key order, strictly-consecutive
    ``commit_seq`` values in ascending order, lowercase-hex hashes, an empty
    ``prev_hash`` on the first entry chained to the previous ``hash`` on
    every later one, and that each entry's ``hash`` is the SHA-256 of its
    ``commit_seq``/``state_hash``/``prev_hash`` (sorted-key compact JSON).

    The first entry may carry any non-negative generation: a legacy file
    loaded without a sidecar anchors the chain at the generation of the
    first write that enables it; every later entry is exactly one
    generation higher. Binding of the chain *tail* to the state file's
    current generation and hash is the caller's last-entry check.
    Raises :class:`IntegrityLogError` on the first violation.
    """
    if list(document) != ["version", "entries"]:
        raise IntegrityLogError(
            "integrity log must have exactly the keys 'version' and "
            "'entries' in order")
    version = document.get("version")
    if not isinstance(version, int) or isinstance(version, bool):
        raise IntegrityLogError("integrity log is missing a numeric version")
    if version != INTEGRITY_LOG_VERSION:
        raise IntegrityLogError(
            f"unsupported integrity log version: {version} "
            f"(this server supports {INTEGRITY_LOG_VERSION})")
    entries = document.get("entries")
    if not isinstance(entries, list):
        raise IntegrityLogError("integrity log 'entries' must be a list")
    verified: List[Dict[str, Any]] = []
    prev_hash = ""
    anchor_seq: Optional[int] = None
    for index, entry in enumerate(entries):
        where = f"integrity log entries[{index}]"
        if not isinstance(entry, dict):
            raise IntegrityLogError(f"{where} must be an object")
        if list(entry) != ["commit_seq", "state_hash", "prev_hash", "hash"]:
            raise IntegrityLogError(
                f"{where} must have exactly the keys 'commit_seq', "
                f"'state_hash', 'prev_hash', 'hash' in order")
        commit_seq = entry["commit_seq"]
        state_hash = entry["state_hash"]
        entry_prev = entry["prev_hash"]
        entry_hash = entry["hash"]
        if not isinstance(commit_seq, int) or isinstance(commit_seq, bool) \
                or commit_seq < 0:
            raise IntegrityLogError(
                f"{where}.commit_seq must be a non-negative integer")
        if not _is_hex64(state_hash):
            raise IntegrityLogError(
                f"{where}.state_hash must be a 64-character lowercase hex "
                f"string")
        if not isinstance(entry_prev, str):
            raise IntegrityLogError(f"{where}.prev_hash must be a string")
        if not _is_hex64(entry_hash):
            raise IntegrityLogError(
                f"{where}.hash must be a 64-character lowercase hex string")
        if index == 0:
            anchor_seq = commit_seq
            if entry_prev != "":
                raise IntegrityLogError(
                    "integrity log first entry must have an empty prev_hash")
        else:
            expected_seq = anchor_seq + index
            if commit_seq != expected_seq:
                raise IntegrityLogError(
                    "integrity log commit_seq values must be ascending and "
                    f"consecutive: entries[{index}] carries {commit_seq}, "
                    f"expected {expected_seq}")
            if entry_prev != prev_hash:
                raise IntegrityLogError(
                    f"integrity log has a broken prev_hash link at "
                    f"entries[{index}]")
        expected_hash = _integrity_entry_hash(
            commit_seq, state_hash, entry_prev)
        if entry_hash != expected_hash:
            raise IntegrityLogError(
                f"integrity log entry hash mismatch at entries[{index}]")
        prev_hash = entry_hash
        verified.append({"commit_seq": commit_seq, "state_hash": state_hash,
                         "prev_hash": entry_prev, "hash": entry_hash})
    return verified


def _load_integrity_log(path: str) -> List[Dict[str, Any]]:
    """Read and fully validate the sidecar at *path* (``None`` if absent)."""
    document = _read_integrity_log(path)
    if document is None:
        return []
    return _verify_integrity_log(document)


def _serialize_integrity_log(entries: List[Dict[str, Any]]) -> bytes:
    """Serialize the whole sidecar as compact, key-ordered UTF-8 JSON.

    Top-level key order is ``version`` then ``entries``; each entry keeps
    ``commit_seq``/``state_hash``/``prev_hash``/``hash`` order. No
    whitespace and ``ensure_ascii=False`` match the hashed representation.
    """
    document = {"version": INTEGRITY_LOG_VERSION, "entries": entries}
    return json.dumps(document, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def _integrity_tmp_paths(directory: str) -> List[str]:
    """List staged sidecar temp files (``.integrity-*.tmp``) in *directory*.

    These are pure transactional staging files: the authoritative sidecar is
    named ``<state>.integrity`` and never matches this prefix, so every file
    listed here is leftover garbage from a crashed commit and safe to remove
    without examining it.
    """
    try:
        names = os.listdir(directory)
    except OSError:
        return []
    return [os.path.join(directory, name) for name in names
            if name.startswith(".integrity-") and name.endswith(".tmp")]


def _gate_integrity_sidecar(
        state_store: "JsonStateStore",
        document: Dict[str, Any]) -> Tuple[bool, List[Dict[str, Any]]]:
    """Validate the marker/sidecar pairing of a loaded state document.

    Returns ``(enabled, entries)``. A legacy document without the
    ``integrity_log_version`` marker and without a sidecar starts with
    ``(False, [])`` — the first write migrates it. Every other arrangement
    refuses startup with :class:`IntegrityLogError` and leaves both files
    untouched:

    * marker present but sidecar missing, or vice versa;
    * a marker whose value is not the integer ``1``;
    * a sidecar that fails the structure/chain/hash validation, is empty, or
      whose last entry does not carry the document's generation and the
      document payload's canonical state hash.
    """
    marker_present = INTEGRITY_LOG_VERSION_KEY in document
    try:
        log_document = _read_integrity_log(state_store.integrity_log_path)
    except IntegrityLogError:
        raise
    sidecar_present = log_document is not None
    if not marker_present and not sidecar_present:
        return False, []
    if marker_present and not sidecar_present:
        raise IntegrityLogError(
            "state document carries 'integrity_log_version' but the "
            f"integrity sidecar {state_store.integrity_log_path} is missing")
    if sidecar_present and not marker_present:
        raise IntegrityLogError(
            "an integrity sidecar exists beside a state document without "
            "'integrity_log_version'")
    marker = document[INTEGRITY_LOG_VERSION_KEY]
    if not isinstance(marker, int) or isinstance(marker, bool) \
            or marker != INTEGRITY_LOG_VERSION:
        raise IntegrityLogError(
            f"unsupported integrity_log_version marker: {marker!r}")
    entries = _verify_integrity_log(log_document)  # type: ignore[arg-type]
    if not entries:
        raise IntegrityLogError("integrity sidecar carries no entries")
    payload = {key: value for key, value in document.items()
               if key not in ("version", COMMIT_SEQ_KEY,
                              INTEGRITY_LOG_VERSION_KEY)}
    try:
        fresh = DeviceStore()
        fresh.restore_state(copy.deepcopy(payload))
        canonical = canonical_integrity_snapshot(fresh.snapshot_state())
    except (ValueError, TypeError) as error:
        raise IntegrityLogError(
            f"integrity sidecar state cannot be restored: {error}") from None
    expected_hash = integrity_state_hash(canonical)
    generation = _commit_seq_of(document)
    last = entries[-1]
    if last["commit_seq"] != generation:
        raise IntegrityLogError(
            "integrity sidecar last entry commit_seq "
            f"{last['commit_seq']} does not match the state document "
            f"generation {generation}")
    if last["state_hash"] != expected_hash:
        raise IntegrityLogError(
            "integrity sidecar last entry state_hash does not match the "
            "state document")
    return True, entries


class JsonStateStore:
    """Versioned JSON document loaded from and atomically saved to one file."""

    def __init__(self, path: str) -> None:
        self.path = path
        #: Path of the append-only integrity-history sidecar
        #: (``<state>.integrity``), one JSON document with one entry per
        #: committed generation. Maintained in the same locked transaction
        #: as every :meth:`save`; absent on a legacy file until its first
        #: write enables it.
        self.integrity_log_path = os.path.abspath(path) \
            + _INTEGRITY_LOG_SUFFIX
        #: Commit generation the *next* save writes. The first (empty) state
        #: is created at 0; :func:`attach_persistence` re-seeds it from a
        #: loaded or recovered document (plus one) so generations stay
        #: strictly consecutive across restarts and recovery. It advances
        #: only once a save has completed durably, so a failed transaction
        #: never consumes a generation.
        self.commit_seq = 0
        #: Whether the integrity sidecar is maintained for this store. A
        #: legacy file (no ``integrity_log_version`` marker, no sidecar)
        #: starts ``False`` and is enabled by the first successful write,
        #: which stamps the marker and creates the sidecar together; once
        #: enabled it stays enabled.
        self.integrity_log_enabled = False
        #: In-memory mirror of the sidecar entries (key order
        #: commit_seq/state_hash/prev_hash/hash), populated at startup when
        #: the sidecar is present and appended to by every durable commit.
        #: Empty (and unused) while :attr:`integrity_log_enabled` is false.
        self.integrity_entries: List[Dict[str, Any]] = []
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
        #: Set instead of :attr:`degraded` when the failed transaction cannot even be
        #: made *decidable*: the formal path could not be vacated, so a
        #: possibly-un-committed residual file still occupies it while the last
        #: committed inode survives only as a ``.bak``. This is the
        #: *blocking state*. :meth:`heal` refuses to serve that residual
        #: formal as authoritative and :meth:`save` refuses every write (the HTTP
        #: layer answers 503/field=data_file), until the on-disk state is
        #: provably decidable again (formal missing and exactly one verifiable backup),
        #: or an operator resolves it. A sibling ``.state-*.block`` marker
        #: makes the state survive a restart, which keeps refusing while the residual
        #: formal file is present or no unique backup can be promoted.
        self.blocked = False

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

    def save(self, state: Dict[str, Any], bootstrap: bool = False) -> None:
        """Atomically replace the file with *state* (version stamped).

        Writes a sibling temporary file, fsyncs it, then ``os.replace`` and
        fsyncs the parent directory, so the rename itself is durable across
        a crash — the target is either the previous full document or the new
        full one, never a truncated mix.

        Every normal commit (``bootstrap=False``) is one locked transaction
        that also appends the generation's entry to the
        ``<state>.integrity`` sidecar; on a legacy store's first commit the
        state document is stamped ``integrity_log_version`` and the sidecar
        is created in that same transaction. ``bootstrap=True`` is reserved
        for the single empty document :func:`attach_persistence` creates for
        a brand-new, never-committed store: it writes the state file alone,
        without the marker or the sidecar, so the chain starts at the first
        real commit (generation 0's entry is written then).

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
        if self.degraded or self.blocked:
            # The formal path is not safely authoritative: degraded means it is
            # missing with the last committed inode pinned only as a .bak (a
            # save here could not be rolled back — there is no target to
            # back up); blocked means an un-decidable residual may occupy it.
            # Either way the persistence hook must run self.heal() first and
            # only reach this point once the formal path is restored. Refuse
            # any save that bypassed the heal.
            state = "blocked" if self.blocked else "degraded"
            raise OSError(
                f"state store for {self.path} is {state}; heal() must "
                f"resolve the on-disk state before another save")
        document = {"version": STATE_VERSION, **state}
        # Stamp the commit generation this save is committing. The counter
        # advances only after the replace and directory fsync succeed, so a
        # failed transaction (rolled back below) consumes no generation and
        # the next retry rewrites the same one — generations on disk stay
        # strictly consecutive.
        seq = self.commit_seq
        document[COMMIT_SEQ_KEY] = seq
        # The integrity-history entry for this generation is the canonical
        # probe hash of the state being committed. A legacy store is enabled
        # on its first commit: the state document is stamped with the marker
        # immediately after commit_seq and the sidecar is created with the
        # first entry, both inside this one locked transaction. The bootstrap
        # empty document carries neither (the chain begins at the first real
        # commit).
        entry: Optional[Dict[str, Any]] = None
        if not bootstrap:
            state_hash = integrity_state_hash(
                canonical_integrity_snapshot(state))
            # Stamp the marker on every sidecar-active commit, including the
            # one that enables it: the marker is envelope metadata not part of
            # the store snapshot, so without this the next commit would drop
            # it and leave the sidecar beside an unmarked document.
            document[INTEGRITY_LOG_VERSION_KEY] = INTEGRITY_LOG_VERSION
            prev_hash = self.integrity_entries[-1]["hash"] \
                if self.integrity_entries else ""
            entry = {"commit_seq": seq, "state_hash": state_hash,
                     "prev_hash": prev_hash,
                     "hash": _integrity_entry_hash(seq, state_hash, prev_hash)}
        tmp_handle = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=directory, delete=False,
            prefix=".state-", suffix=".tmp")
        tmp_path = tmp_handle.name
        # The sidecar is rewritten whole (it stays small: one hash chain
        # entry per generation) through its own sibling temp file, staged
        # before either formal file is touched so the two commit together.
        log_tmp_handle: Any = None
        log_tmp_path: Optional[str] = None
        if not bootstrap:
            log_tmp_handle = tempfile.NamedTemporaryFile(
                mode="wb", dir=directory, delete=False,
                prefix=".integrity-", suffix=".tmp")
            log_tmp_path = log_tmp_handle.name
        backup_path: Optional[str] = None
        replaced = False
        log_replaced = False
        try:
            json.dump(document, tmp_handle, separators=(",", ":"),
                      ensure_ascii=False)
            tmp_handle.flush()
            os.fsync(tmp_handle.fileno())
            tmp_handle.close()
            if log_tmp_handle is not None and entry is not None:
                log_tmp_handle.write(
                    _serialize_integrity_log([*self.integrity_entries, entry]))
                log_tmp_handle.flush()
                os.fsync(log_tmp_handle.fileno())
                log_tmp_handle.close()
            # Pin the old inode before renaming over it. On the first save the
            # target does not exist yet, so there is no old state to preserve.
            backup_path = _hardlink_backup(directory, self.path)
            os.replace(tmp_path, self.path)
            replaced = True
            # Publish the sidecar entry in the same transaction: atomic
            # sibling rename. On a legacy-enable the sidecar did not exist;
            # afterwards it always does.
            if log_tmp_path is not None:
                os.replace(log_tmp_path, self.integrity_log_path)
                log_replaced = True
            # The renames are only durable once the directory entries are
            # flushed; without this fsync a crash can leave the directory
            # pointing at pre-rename entries even though the replacements
            # landed. One flush covers both renames.
            _fsync_directory(directory)
        except BaseException:
            tmp_handle.close()
            if log_tmp_handle is not None:
                log_tmp_handle.close()
            _remove_quietly(tmp_path)
            if log_tmp_path is not None:
                _remove_quietly(log_tmp_path)
            if replaced:
                # A state-file replacement landed (the sidecar may or may not
                # have been renamed). The new sidecar entry belongs to this
                # un-committed state, so whatever on-disk resolution follows
                # below (rollback to the pinned inode, degraded formal-missing,
                # or blocked), restore the sidecar to its pre-transaction
                # contents whenever its rename landed — absent altogether when
                # this was the enabling write. Best effort: a state/sidecar
                # mismatch after a crash makes the next startup refuse.
                if log_replaced:
                    self._rollback_integrity_log(directory)
                # Restore the pinned old inode atomically first (the rename
                # itself also drops the new inode). On the very first save
                # there was no prior target, hence no backup pin: the only
                # rollback is to vacate the new snapshot so the formal path is
                # missing again.
                state_restored = False
                if backup_path is not None:
                    try:
                        os.replace(backup_path, self.path)
                        state_restored = True
                    except OSError:
                        state_restored = False
                if state_restored:
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
                else:
                    # There was no backup (first save) or the rollback rename
                    # itself failed. The formal path still holds the
                    # un-committed new inode, which must never become
                    # authoritative: move it off the path (the old inode
                    # stays pinned in the .bak when one existed), so the next
                    # startup sees a missing formal file and recovers it.
                    _vacate_new_snapshot(directory, self.path)
                    # Keep the backup: it is now the only copy of the last
                    # committed inode, so it must not be cleaned up below.
                    backup_path = None
                    if os.path.exists(self.path):
                        # The new snapshot could not be taken off the formal
                        # path: its presence is un-decidable (it is not
                        # the committed inode, yet occupies the authoritative name).
                        # Enter the blocking state rather than ever serve it as
                        # authoritative: persist a marker so a restart keeps
                        # refusing until the path is missing with a unique backup.
                        _write_block_marker(directory)
                        self.blocked = True
                    else:
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
        # The integrity entry is now durably committed alongside the state.
        if entry is not None:
            self.integrity_entries.append(entry)
            self.integrity_log_enabled = True
        # The generation advances exactly once per durable commit.
        self.commit_seq = seq + 1

    def _rollback_integrity_log(self, directory: str) -> None:
        """Restore the pre-transaction sidecar after a state-file rollback.

        Rewrites the sidecar from the in-memory entries (which never include
        the failed commit); when the failed transaction was the legacy-enable
        one there were no entries and the sidecar did not previously exist,
        so it is removed instead. Best effort only — a mismatch between the
        state file and the sidecar after a crash makes the next startup
        refuse rather than serve either as authoritative.
        """
        if self.integrity_entries:
            tmp_handle = tempfile.NamedTemporaryFile(
                mode="wb", dir=directory, delete=False,
                prefix=".integrity-", suffix=".tmp")
            tmp_path = tmp_handle.name
            try:
                tmp_handle.write(
                    _serialize_integrity_log(self.integrity_entries))
                tmp_handle.flush()
                os.fsync(tmp_handle.fileno())
                tmp_handle.close()
                os.replace(tmp_path, self.integrity_log_path)
            except OSError:
                tmp_handle.close()
                _remove_quietly(tmp_path)
        else:
            _remove_quietly(self.integrity_log_path)

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
        when no verifiable backup exists, two or more verifiable candidates
        exist (promotion is refused outright — choosing by name or mtime is
        forbidden), or the promotion cannot be made durable; in every such
        case the formal path stays missing, no leftover and no generation
        moves, :attr:`degraded` (or :attr:`blocked`) stays set, and the next
        write retries this heal. A formal file that reappeared valid and matching
        always takes precedence over every backup — but only in the *degraded*
        state. In the *blocking* state a residual formal file is never served as
        authoritative: heal refuses while it occupies the path, and only leaves the
        blocking state once the path is missing with exactly one verifiable backup to
        promote (the durable ``.block`` marker is cleared at that point).
        """
        if not (self.degraded or self.blocked):
            return
        directory = os.path.dirname(os.path.abspath(self.path))
        leftovers = _leftover_tmp_paths(directory, self.path)
        quarantined = _quarantined_paths(directory, self.path)

        if os.path.exists(self.path):
            # The formal path can only reappear (or, while blocked, never
            # have been vacated) out of band. A *valid formal always
            # takes precedence* — and that also resolves the blocked state
            # safely: the blocked residual is the un-committed new snapshot,
            # stamped with the NEXT generation (commit_seq == self.commit_seq), so
            # it can never match the committed baseline; only a file exactly
            # equal to the last committed state (generation one below, restores
            # and equals baseline) wins. When it does, keep it, sweep
            # every leftover/marker and resume normal saves. Anything else is
            # left untouched and refused.
            formal = _read_version1_document(self.path)
            formal_is_committed = (
                formal is not None
                and (COMMIT_SEQ_KEY not in formal
                     or _valid_commit_seq(formal[COMMIT_SEQ_KEY]))
                and _commit_seq_of(formal) == self.commit_seq - 1
                and _document_restores(formal)
                and _document_payload_equals(formal, baseline))
            if formal_is_committed:
                _sweep_leftovers(
                    directory,
                    leftovers + quarantined
                    + _block_marker_paths(directory, self.path))
                self.degraded = False
                self.blocked = False
                return
            if self.blocked:
                raise OSError(
                    f"cannot self-heal {self.path}: the store is blocked by "
                    f"an un-resolved, un-committed file on the formal path")
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
        # Two or more verifiable candidates are ambiguous even when their
        # bytes are identical: the choice between pin names must never be made
        # by name or mtime (both are forgeable and neither encodes which pin
        # the crashed transaction actually committed). Refuse promotion and
        # leave everything — formal path, leftovers, memory and generation —
        # exactly as found, so an operator or a restart resolves the ambiguity;
        # the next write retries the heal.
        if len(eligible) > 1:
            raise OSError(
                f"cannot self-heal {self.path}: {len(eligible)} verifiable "
                f"pinned backups of the last committed state; refusing to "
                f"choose between candidates")
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
        if self.blocked:
            # The path was vacated out-of-band and the unique verifiable
            # backup is now durably authoritative: the undecidable condition
            # is gone, so drop the durable block marker and resume service.
            _clear_block_markers(directory, self.path)
            self.blocked = False

    def integrity_report(self, store: "DeviceStore") -> Dict[str, Any]:
        """Read-only integrity probe backing ``GET /v1/persistence/integrity``.

        The state file is read while *store*'s own lock is held (the same
        lock under which every mutation rewrites the file and appends audit
        events), so the bytes read are one complete committed document. The
        document must parse as JSON, be an object stamped ``version=1``, and
        carry either no ``commit_seq`` (a legacy file, generation 0) or a
        non-negative integer one. *store* then restores the payload into a
        fresh store and compares its canonical snapshot against the live
        in-memory snapshot — the same startup semantic validation
        (cross-entity references, sequence continuity, nonce sets, per-device
        cursor ranges, ``updated_at`` and the key-event audit chain) plus an
        exact state comparison.

        On success returns ``{"commit_seq", "state_hash", "consistent":
        True}`` in that key order, where ``state_hash`` is the lowercase
        SHA-256 hex of the canonical compact JSON snapshot (17 ordered
        sections, ``ensure_ascii=False`` UTF-8). Any failure — unreadable or
        missing file (including the degraded/blocked states where the formal
        path is absent), parse/version/generation error, semantic error, or a
        file/memory divergence — raises :class:`IntegrityCheckError`. The
        probe never writes: memory, file bytes/inode, cursors and the commit
        generation are all untouched.
        """
        def read_document() -> Tuple[Dict[str, Any], int]:
            try:
                with open(self.path, "rb") as handle:
                    raw = handle.read()
            except OSError as error:
                raise IntegrityCheckError(
                    f"cannot read state file {self.path}: {error}") from None
            try:
                document = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise IntegrityCheckError(
                    f"state file is not valid JSON: {self.path} ({error})") \
                    from None
            if not isinstance(document, dict):
                raise IntegrityCheckError(
                    "state document must be a JSON object")
            version = document.get("version")
            if not isinstance(version, int) or isinstance(version, bool):
                raise IntegrityCheckError(
                    "state document is missing a numeric 'version'")
            if version != STATE_VERSION:
                raise IntegrityCheckError(
                    f"unsupported state file version: {version} "
                    f"(this server supports {STATE_VERSION})")
            commit_seq = document.get(COMMIT_SEQ_KEY, 0)
            if not _valid_commit_seq(commit_seq):
                raise IntegrityCheckError(
                    f"state document '{COMMIT_SEQ_KEY}' must be a "
                    f"non-negative integer")
            payload = {key: value for key, value in document.items()
                       if key not in ("version", COMMIT_SEQ_KEY,
                                      INTEGRITY_LOG_VERSION_KEY)}
            return payload, commit_seq

        def expected_generation() -> int:
            # Read inside the store lock: a writer both rewrites the file
            # and advances this counter while holding that same lock, so the
            # file's generation and the expected last-committed generation
            # can never be sampled from different linearization points.
            return self.commit_seq - 1

        try:
            commit_seq, state_hash = store.integrity_evaluate(
                read_document, expected_generation)
        except IntegrityCheckError:
            raise
        except (ValueError, TypeError) as error:
            raise IntegrityCheckError(str(error)) from None
        return {"commit_seq": commit_seq,
                "state_hash": state_hash,
                "consistent": True}

    def history_report(self, store: "DeviceStore") -> Optional[Dict[str, Any]]:
        """Read-only integrity-history probe backing
        ``GET /v1/persistence/integrity/history``.

        Returns ``None`` when the state document carries no
        ``integrity_log_version`` marker (the purely in-memory mode and the
        not-yet-migrated legacy file mode both answer 409/field=data_file at
        the service layer); that decision is made from the envelope alone,
        before any semantic validation.

        With the marker, the document is fully verified (the same
        parse/version/generation/semantic/snapshot probe as
        :meth:`integrity_report`) and the sidecar is read and fully verified
        under the store lock shared with every commit; its last entry must
        carry the current committed generation and the live committed state's
        canonical hash. On success returns ``{"commit_seq", "entries"}`` in
        that key order; ``entries`` preserves each entry's
        ``commit_seq``/``state_hash``/``prev_hash``/``hash`` order. Any
        parse/chain/hash/tail failure raises :class:`IntegrityCheckError`
        (mapped to 503/field=data_file) without writing anything.
        """
        verified = self._verified_history(store)
        if verified is None:
            return None
        commit_seq, entries = verified
        return {"commit_seq": commit_seq,
                "entries": copy.deepcopy(entries)}

    def history_page_report(self, store: "DeviceStore", after: int,
                            limit: int) -> Optional[Dict[str, Any]]:
        """Read-only paged integrity-history probe backing
        ``GET /v1/persistence/integrity/history/page``.

        Marker/verification semantics are identical to
        :meth:`history_report`: ``None`` means no
        ``integrity_log_version`` marker (the service answers
        409/field=data_file) and any parse/version/generation/chain/hash/tail
        or read failure raises :class:`IntegrityCheckError` (503/data_file),
        all under the store lock shared with every commit and without
        writing anything.

        The page is the verified entries with ``commit_seq > after`` in
        ascending order, at most *limit*. On success returns
        ``{"commit_seq", "entries", "next_after", "has_more"}`` in that key
        order: ``commit_seq`` is the current committed generation (the tail),
        ``next_after`` is *after* unchanged on an empty page and otherwise
        the last returned entry's generation, and ``has_more`` says whether a
        verified entry follows that page. ``entries`` keeps each entry's
        ``commit_seq``/``state_hash``/``prev_hash``/``hash`` order.
        """
        verified = self._verified_history(store)
        if verified is None:
            return None
        commit_seq, entries = verified
        page = [entry for entry in entries
                if entry["commit_seq"] > after][:limit]
        next_after = page[-1]["commit_seq"] if page else after
        has_more = any(entry["commit_seq"] > next_after for entry in entries)
        return {"commit_seq": commit_seq,
                "entries": copy.deepcopy(page),
                "next_after": next_after,
                "has_more": has_more}

    def _verified_history(
            self, store: "DeviceStore"
    ) -> Optional[Tuple[int, List[Dict[str, Any]]]]:
        """Verify the state document and sidecar and return ``(seq, entries)``.

        Shared by :meth:`history_report` and :meth:`history_page_report`.
        Returns ``None`` when the state document carries no
        ``integrity_log_version`` marker (a decision made from the envelope
        alone). With the marker, the document is fully verified
        (parse/version/generation/semantic/snapshot against the live store)
        and the sidecar is read and fully verified under the store lock
        shared with every commit; its last entry must carry the current
        committed generation and the live committed state's canonical hash.
        Any failure raises :class:`IntegrityCheckError`; nothing is written.
        """
        def parse_state() -> Tuple[Dict[str, Any], int, Any]:
            try:
                with open(self.path, "rb") as handle:
                    raw = handle.read()
            except OSError as error:
                raise IntegrityCheckError(
                    f"cannot read state file {self.path}: {error}") from None
            try:
                document = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise IntegrityCheckError(
                    f"state file is not valid JSON: {self.path} ({error})") \
                    from None
            if not isinstance(document, dict):
                raise IntegrityCheckError(
                    "state document must be a JSON object")
            version = document.get("version")
            if not isinstance(version, int) or isinstance(version, bool):
                raise IntegrityCheckError(
                    "state document is missing a numeric 'version'")
            if version != STATE_VERSION:
                raise IntegrityCheckError(
                    f"unsupported state file version: {version}")
            commit_seq = document.get(COMMIT_SEQ_KEY, 0)
            if not _valid_commit_seq(commit_seq):
                raise IntegrityCheckError(
                    f"state document '{COMMIT_SEQ_KEY}' must be a "
                    f"non-negative integer")
            marker = document.get(INTEGRITY_LOG_VERSION_KEY)
            payload = {key: value for key, value in document.items()
                       if key not in ("version", COMMIT_SEQ_KEY,
                                      INTEGRITY_LOG_VERSION_KEY)}
            return payload, commit_seq, marker

        # Hold the same RLock the commit hook holds across the state-file
        # verification and the sidecar read, so the state generation/hash and
        # the log tail are always sampled at one committed linearization
        # point. integrity_evaluate re-enters the RLock harmlessly.
        with store._lock:
            payload, commit_seq, marker = parse_state()
            # No marker -> legacy/not-enabled: 409 regardless of whether the
            # payload would otherwise restore.
            if marker is None:
                return None
            if not isinstance(marker, int) or isinstance(marker, bool) \
                    or marker != INTEGRITY_LOG_VERSION:
                raise IntegrityCheckError(
                    f"state document has an unsupported "
                    f"'{INTEGRITY_LOG_VERSION_KEY}' marker: {marker!r}")
            if commit_seq != self.commit_seq - 1:
                raise IntegrityCheckError(
                    f"state file commit_seq {commit_seq} does not match the "
                    f"last committed generation {self.commit_seq - 1}")
            try:
                restored_store = DeviceStore()
                restored_store.restore_state(copy.deepcopy(payload))
                on_disk = canonical_integrity_snapshot(
                    restored_store.snapshot_state())
            except (ValueError, TypeError) as error:
                raise IntegrityCheckError(str(error)) from None
            live = canonical_integrity_snapshot(store.snapshot_state())
            if on_disk != live:
                raise IntegrityCheckError(
                    "state file is inconsistent with the in-memory snapshot")
            state_hash = integrity_state_hash(on_disk)
            try:
                entries = _load_integrity_log(self.integrity_log_path)
            except IntegrityLogError as error:
                raise IntegrityCheckError(str(error)) from None
            if not entries:
                raise IntegrityCheckError(
                    "integrity log is present but carries no entries")
            last = entries[-1]
            if last["commit_seq"] != commit_seq:
                raise IntegrityCheckError(
                    "integrity log last entry commit_seq "
                    f"{last['commit_seq']} does not match the state "
                    f"generation {commit_seq}")
            if last["state_hash"] != state_hash:
                raise IntegrityCheckError(
                    "integrity log last entry state_hash does not match the "
                    "current state hash")
        return commit_seq, entries


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


def _documents_payload_equal(left: Dict[str, Any],
                      right: Dict[str, Any]) -> bool:
    """Semantically compare two version-1 state documents' business payloads.

    Envelope fields (``version``, ``commit_seq``) are ignored, so two
    copies of the same committed generation compare equal regardless of byte
    formatting, while a newer un-committed snapshot round-trips to a
    different snapshot. Returns ``False`` if either document fails to restore.
    """
    def _snapshot(document: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        payload = {key: value for key, value in document.items()
                   if key not in ("version", COMMIT_SEQ_KEY)}
        try:
            store = DeviceStore()
            store.restore_state(payload)
            return store.snapshot_state()
        except (ValueError, TypeError):
            return None

    snap_left = _snapshot(left)
    snap_right = _snapshot(right)
    if snap_left is None or snap_right is None:
        return False
    return snap_left == snap_right


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


def _block_marker_paths(directory: str, target_path: str) -> List[str]:
    """List blocking-state markers left beside *target_path*."""
    try:
        names = os.listdir(directory)
    except OSError:
        return []
    markers: List[str] = []
    for name in names:
        if name.startswith(_TMP_PREFIX) and name.endswith(_BLOCK_SUFFIX):
            full = os.path.join(directory, name)
            if os.path.abspath(full) != os.path.abspath(target_path):
                markers.append(full)
    return markers


def _write_block_marker(directory: str) -> Optional[str]:
    """Persist a blocking-state marker in *directory* (best effort).

    The marker makes the "formal path may hold an un-committed residual"
    state survive a process restart, which then keeps refusing writes instead of
    treating that residual file as authoritative. Returns the marker path, or
    ``None`` if it could not be created (the in-process
    :attr:`JsonStateStore.blocked` flag still holds for the life of
    the process).
    """
    fd, marker = tempfile.mkstemp(dir=directory, prefix=_TMP_PREFIX,
                                      suffix=_BLOCK_SUFFIX)
    try:
        os.fsync(fd)
    except OSError:
        pass
    os.close(fd)
    try:
        _fsync_directory(directory)
    except OSError:
        pass
    return marker


def _clear_block_markers(directory: str, target_path: str) -> None:
    """Remove every blocking-state marker and flush the directory (best effort)."""
    for marker in _block_marker_paths(directory, target_path):
        _remove_quietly(marker)
    try:
        _fsync_directory(directory)
    except OSError:
        pass


def _remove_quietly(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def _verified_recovery_candidates(
        leftovers: List[str]) -> Tuple[List[Tuple[str, int, int]], bool]:
    """Parse and fully verify crash-leftover *leftovers*.

    Returns ``(verified, any_with_seq)`` where each verified entry is
    ``(path, commit_seq, mtime_ns)`` and *any_with_seq* says
    whether at least one candidate carried an explicit ``commit_seq`` field.

    A candidate survives only when it parses as version=1, explicitly
    carries the complete-transaction sections, carries no malformed generation field,
    and passes the full semantic restore check. A missing field ranks at
    generation 0 (the legacy case).
    """
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
    return verified, any_with_seq


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
      lost to a newer-mtime stale one. When two or more *verifiable*
      candidates tie at that highest generation the choice is ambiguous and recovery
      refuses outright: nothing is promoted or removed, the formal path stays
      missing, and :func:`attach_persistence` raises
      :class:`StateFileError` (the blocking state) rather than ever
      breaking the tie by name or mtime. When every candidate
      predates commit generations (none carries the field), selection stays the
      legacy newest-mtime rule. The chosen snapshot is atomically
      ``os.replace``-d into place (plus a parent-directory fsync) and
      the remaining leftovers are removed. A section-less legacy/partial
      snapshot is never promoted, so a resume cursor or the audit chain
      cannot be silently lost.
    * When no candidate is valid, all leftovers are removed; the normal
      missing-file path in :func:`attach_persistence` then creates an empty
      state.
    """
    directory = os.path.dirname(os.path.abspath(state_store.path))
    leftovers = _leftover_tmp_paths(directory, state_store.path)
    quarantined = _quarantined_paths(directory, state_store.path)
    markers = _block_marker_paths(directory, state_store.path)
    if not leftovers and not quarantined and not markers:
        return

    if markers:
        # A previous process ended in the blocking state: it could not vacate a
        # possibly-un-committed file from the formal path. A restart must
        # never serve that residual file as authoritative:
        #  * formal present and provably the committed state (it semantically
        #    equals the unique verifiable generation-bearing backup) -> a valid
        #    formal takes precedence: keep it, sweep leftovers/markers;
        #  * formal present in any other state (the un-committed
        #    residual, or no unique matching backup) -> refuse startup,
        #    touch nothing;
        #  * formal missing -> recover only when exactly ONE verifiable,
        #    generation-bearing candidate survives (never mtime/name picked);
        #    promote it, sweep the transaction leftovers and clear the marker;
        #  * zero or several verifiable candidates -> refuse and keep
        #    everything in place for an operator.
        verified, any_with_seq = _verified_recovery_candidates(leftovers)
        top: List[Tuple[str, int, int]] = []
        if verified and any_with_seq:
            top_seq = max(item[1] for item in verified)
            top = [item for item in verified
                     if item[1] == top_seq]
        unique_pin = top[0][0] if len(top) == 1 else None

        if os.path.exists(state_store.path):
            formal = _read_version1_document(state_store.path)
            # A valid formal always takes precedence. There is no in-memory
            # baseline at restart, so "committed" is proven relative to
            # the verified backups: the formal is the committed state iff it
            # restores, carries a valid generation, and semantically
            # equals at least one verified backup at that SAME generation.
            # The un-committed residual is stamped one generation higher than the
            # pinned commit, so it can never match and stays refused —
            # even when two or more backups exist.
            formal_is_committed = (
                formal is not None
                and (COMMIT_SEQ_KEY not in formal
                     or _valid_commit_seq(formal[COMMIT_SEQ_KEY]))
                and _document_restores(formal)
                and any(
                    _commit_seq_of(formal) == _commit_seq_of(pin_doc)
                    and _documents_payload_equal(formal, pin_doc)
                    for pin_doc in (
                        _read_version1_document(path)
                        for path, _, _ in verified)
                    if pin_doc is not None))
            if formal_is_committed:
                for path in leftovers + quarantined + markers:
                    _remove_quietly(path)
                _fsync_directory(directory)
                return
            raise OSError(
                f"state store for {state_store.path} is blocked: an "
                f"un-resolved, un-committed file occupies the formal path")
        if unique_pin is None:
            raise OSError(
                f"state store for {state_store.path} is blocked: the formal "
                f"path is missing with no unique verifiable backup")
        recovered = unique_pin
        os.replace(recovered, state_store.path)
        _fsync_directory(directory)
        leftovers = [path for path in leftovers if path != recovered]
        for path in leftovers + quarantined + markers:
            _remove_quietly(path)
        _fsync_directory(directory)
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
    # highest commit_seq wins even when a stale older-generation snapshot has a newer
    # mtime. A tie of two or more verifiable candidates at that highest
    # generation is ambiguous and aborts recovery (nothing moved, startup refuses). A
    # leftover without the field is a pre-generation snapshot at seq 0; only
    # when every candidate is such a snapshot does the ranking collapse to the
    # legacy pure-mtime rule (ties broken by name, deterministically).
    verified, any_with_seq = _verified_recovery_candidates(leftovers)

    recovered = None
    if verified:
        if any_with_seq:
            # Generation-first ordering; missing field ranks at 0, so a
            # present-generation snapshot always beats a field-less legacy one
            # regardless of mtime.
            verified.sort(key=lambda item: (item[1], item[2], item[0]))
            top_seq = verified[-1][1]
            top = [item for item in verified if item[1] == top_seq]
            # Two or more *verifiable* candidates at the same top generation are
            # ambiguous: the generation does not name which one the crashed transaction
            # committed, and neither mtime nor file name may break the tie (both
            # are forgeable and neither encodes commit identity). Refuse to promote or
            # delete anything: keep the formal path missing and every leftover in
            # place so attach_persistence refuses to start (the blocking state the
            # spec demands) until an operator resolves the duplicates. Field-less
            # legacy files never reach here: any_with_seq was False for them.
            if len(top) >= 2:
                raise OSError(
                    f"ambiguous state recovery for {state_store.path}: "
                    f"{len(top)} verifiable candidates at commit_seq "
                    f"{top_seq}; refusing to choose between them by name or "
                    f"mtime")
            recovered = top[0][0]
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
    # Staged sidecar temp files are never authoritative; drop any a crashed
    # commit left behind (the formal state file is already resolved above).
    for tmp_path in _integrity_tmp_paths(
            os.path.dirname(os.path.abspath(path))):
        _remove_quietly(tmp_path)
    document = state_store.load()
    if document is None:
        # Crash recovery could not restore a state document (the formal file
        # was missing and every leftover was absent or unverifiable), so an
        # empty state is created as before. Any sidecar left beside it refers
        # to generations that no longer exist and is swept here too, keeping
        # the pairing marker-less/sidecar-less until the first real commit
        # anchors a fresh chain.
        _remove_quietly(state_store.integrity_log_path)
        for tmp_path in _integrity_tmp_paths(
                os.path.dirname(os.path.abspath(path))):
            _remove_quietly(tmp_path)
        try:
            # The bootstrap document is the empty generation-0 state with no
            # integrity marker or sidecar; the history chain is anchored at
            # the first real commit.
            state_store.save(service.store.snapshot_state(), bootstrap=True)
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
        # Verify the integrity marker/sidecar pairing and chain before the
        # server accepts traffic. A legacy file without either is allowed and
        # migrates on its first write; any mismatch or broken chain refuses
        # startup with the on-disk files untouched.
        try:
            enabled, entries = _gate_integrity_sidecar(state_store, document)
        except IntegrityLogError as error:
            raise StateFileError(str(error)) from None
        state_store.integrity_log_enabled = enabled
        state_store.integrity_entries = entries

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
        if state_store.degraded or state_store.blocked:
            # A previous transaction ended undecidable on disk. Degraded: the
            # formal path is missing and the last committed inode survives
            # only as a pinned .bak. Blocked: an un-decidable residual
            # may occupy the path (a durable .block marker records it).
            # Before this (the next) write may commit, run heal inside
            # the same storage lock: in the degraded case it verifies and
            # atomically promotes the unique backup, cleaning the failed
            # transaction's leftovers; in the blocked case it refuses while a
            # residual file occupies the path (503) and only
            # resolves once the path is missing with exactly one
            # verifiable backup. Memory was rolled back to last_good after
            # the 503 and, like every later failed attempt, restored
            # to it again, so the only difference between pending and last_good
            # is *this* current request — the earlier 503 request is
            # never replayed.
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
    # Expose the durable store for the read-only integrity probe
    # (GET /v1/persistence/integrity); its absence is exactly how the
    # service distinguishes the in-memory mode (409/field=data_file).
    service.integrity_state_store = state_store
    return state_store
