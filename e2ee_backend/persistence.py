"""Durable two-file JSON persistence for the device store.

The whole server state is kept in one versioned JSON document, committed
together with an append-only integrity sidecar (``<state>.integrity``). A
missing state file is created on first use; a corrupted file or a document
whose version is not understood makes the server refuse to start (state is
never silently discarded). Every change is a *two-file atomic transaction*:
the new state document and the sidecar with the generation's new entry are
serialized to sibling temporary files (``.state-*.tmp`` /
``.integrity-*.tmp``) in the same directory and ``fsync``-ed, hard-link
backups of both current targets are taken when they exist
(``.state-*.bak`` / ``.integrity-*.bak``), both temporaries are
``os.replace``-d over their targets, and the parent directory is fsynced
once, so a crash always leaves a matched, hash-chain-consistent pair and
never a torn mix. The pair is only committed when both files are of the
same generation and the sidecar tail's ``state_hash`` binds that
generation's canonical state; each successful transaction raises the
generation by exactly one.

Persistence is transactional with respect to the in-memory store: the change
hook fires while the store lock is held (the mutation is visible but not yet
committed to callers). When the durable write succeeds it becomes the new
last-known-good pair. When writing either temporary file, fsync-ing either
file, or either atomic replace fails (an :class:`OSError`), the previous
last-known-good pair is restored under the same lock and
:class:`PersistenceUnavailable` is raised so the HTTP layer answers
503/field=data_file; the failed mutation therefore never advances either
memory or either file. A platform/filesystem that cannot fsync a directory
skips that one capability instead of failing the transaction (the file
fsyncs and the atomic renames already landed); genuine I/O errors still
roll the transaction back.

After both replaces have landed, a failed directory fsync first rolls the
transaction back symmetrically: each pinned old inode is renamed over its
formal target and a second directory fsync flushes the rollback. When that
rollback itself cannot be completed durably — a rollback rename fails, or
the second fsync fails — the failure stays *decidable*: both old inodes are
kept under their paired ``.state-*.bak`` / ``.integrity-*.bak`` backups and
both un-committed snapshots are taken off their formal paths (the new state
snapshot is parked under a ``.quarantine`` name the recovery scan never
promotes, or deleted; the un-committed sidecar is deleted), so *both*
formal paths are missing. The store is then marked degraded and reports 503
for the failed request.

The degraded store also heals itself *within the same process*: the next
persist-able write runs, inside the storage lock, :meth:`JsonStateStore.heal`
before saving — a unique *pair* of pinned backups is fully verified (the
state backup at the previous generation with version=1 semantics and exact
equality with the last committed in-memory state, and the sidecar backup
with a complete hash chain whose tail binds that generation and state hash),
both are hard-linked back onto their formal paths, same-transaction
leftovers (``.quarantine``/``.tmp``/redundant ``.bak`` on either file) are
removed, and only then is that one triggering request committed (a single
consecutive ``commit_seq`` advance). The request that originally got 503 is
never replayed; the healed commit carries only the later, current request.
A corrupt, dangling, unpaired or ambiguous backup is refused promotion: the
request gets 503/field=data_file, memory is rolled back, both formal paths
stay missing, no leftover and no generation moves, and the next write
retries the heal. A valid formal pair always takes precedence over every
backup. Restart recovery remains the fallback that promotes the pinned
pair.

When even the decidable fallback cannot be achieved — one formal path
(typically the state file) cannot be proven missing, so an un-committed
snapshot may still occupy it while the committed inode survives only in a
backup — the store enters the *blocking state* (a sibling
``.state-*.block`` marker persists it across restarts). It never serves
that residual formal file as authoritative: every subsequent write is
503/field=data_file, in this process and after a restart, until the path is
vacated with exactly one verifiable backup pair (heal/restart then promote
it and clear the marker), or files provably equal to the committed state
are restored onto both formal paths (a valid formal pair always wins). Two
or more verifiable state backups make a pair ambiguous even against one
sidecar backup and are refused rather than picked by name or mtime.

A crash between steps can leave ``.state-*.tmp``/``.state-*.bak`` and
``.integrity-*.tmp``/``.integrity-*.bak`` files beside the targets. At the
next :func:`attach_persistence` these are resolved by
:func:`recover_crash_leftovers`: a valid matched formal pair stays
authoritative and every leftover is removed; with the state file missing,
state leftovers and sidecar leftovers are combined into *pairs* and only a
pair that passes version=1 semantics, carries the complete-transaction
sections, has a valid sidecar chain whose tail generation and
``state_hash`` bind the state document (or, for a legacy document without
the integrity marker, has no sidecar at all) is considered; the pair at the
highest generation is recovered atomically (modification time breaks ties
only between genuinely pre-generation files). Every verifiable
``integrity_log_version=1`` state candidate must bind exactly one sidecar:
an orphaned modern snapshot, one state bound by several sidecars, or one
sidecar binding several states all make startup raise
:class:`StateFileError` — a lower-generation complete pair is never a
downgrade target. Verifiable candidates without a unique pair make startup
raise :class:`StateFileError` (exit 1,
one stderr JSON line, field=data_file) with nothing deleted or overwritten;
only when no candidate is verifiable are the leftovers removed and an empty
state created. An existing-but-corrupt formal file still makes startup
refuse rather than being silently overwritten; a legacy version=1 state
file without the marker and without a sidecar keeps loading as before.

Every document carries a strictly-consecutive top-level ``commit_seq``: the
first empty state is written at 0 and each successful durable transaction
advances it exactly once (a failed write consumes no generation). A legacy
version=1 file without the field loads as generation 0; a present field must
be a non-negative integer or startup refuses (bool, negative, float and
string values are rejected) with the file untouched.

Beside the state file every normal commit maintains the append-only
integrity sidecar at ``<state>.integrity`` (a compact version=1 JSON
document with one ``commit_seq``/``state_hash``/``prev_hash``/``hash``
entry per committed generation), written in the same locked two-file
transaction. The sidecar document's key order is ``version`` then
``entries`` and each entry keeps
``commit_seq``/``state_hash``/``prev_hash``/``hash`` order; both files are
UTF-8, ``ensure_ascii=False``, compact JSON with no whitespace or trailing
newline. The state document records the format with
``integrity_log_version=1``; a legacy file without that marker and without
a sidecar still starts and anchors the chain on its first commit, while any
marker/sidecar disagreement or a broken / tail-mismatched chain makes
startup refuse with both files untouched. The sidecar backs
``GET /v1/persistence/integrity/history`` and its ascending, cursor-paged
variant ``GET /v1/persistence/integrity/history/page``
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
#: Name parts of the integrity sidecar's own staging files and pinned-inode
#: backups: every commit stages and pins *both* formal files through the same
#: two-file transaction, so crash leftovers and backups come in pairs
#: (``.state-*`` beside ``.integrity-*``).
_INTEGRITY_TMP_PREFIX = ".integrity-"
_INTEGRITY_TMP_SUFFIX = ".tmp"
_INTEGRITY_BAK_SUFFIX = ".bak"
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


def _sidecar_leftover_paths(directory: str,
                            target_path: str) -> List[str]:
    """List crashed sidecar files: ``.integrity-*.tmp``/``.integrity-*.bak``.

    The authoritative sidecar (``<state>.integrity``) is never listed. Both
    staging snapshots (``.tmp``) and pinned inode backups (``.bak``) are
    pairing candidates; validity is decided later, never by suffix alone.
    """
    try:
        names = os.listdir(directory)
    except OSError:
        return []
    leftovers: List[str] = []
    for name in names:
        if not name.startswith(_INTEGRITY_TMP_PREFIX):
            continue
        if not (name.endswith(_INTEGRITY_TMP_SUFFIX)
                or name.endswith(_INTEGRITY_BAK_SUFFIX)):
            continue
        full = os.path.join(directory, name)
        if os.path.abspath(full) == os.path.abspath(target_path):
            continue
        leftovers.append(full)
    return leftovers


def _verify_sidecar_file(path: str) -> Optional[List[Dict[str, Any]]]:
    """Read and fully verify a sidecar candidate file.

    Returns its verified entries, or ``None`` when the file is absent,
    unreadable, not valid JSON, structurally invalid, empty, or its hash
    chain is broken (tail binding to a state document is the caller's check).
    """
    try:
        document = _read_integrity_log(path)
    except IntegrityLogError:
        return None
    if document is None:
        return None
    try:
        entries = _verify_integrity_log(document)
    except IntegrityLogError:
        return None
    return entries or None


def _state_document_state_hash(document: Dict[str, Any]) -> Optional[str]:
    """Canonical ``state_hash`` of a version-1 state document's payload.

    Strips the envelope (``version``/``commit_seq``/
    ``integrity_log_version``), restores the payload into a fresh store and
    hashes its canonical snapshot — the same round trip the startup marker
    gate uses to bind a sidecar tail. Returns ``None`` when the payload does
    not restore.
    """
    payload = {key: value for key, value in document.items()
               if key not in ("version", COMMIT_SEQ_KEY,
                              INTEGRITY_LOG_VERSION_KEY)}
    try:
        fresh = DeviceStore()
        fresh.restore_state(copy.deepcopy(payload))
        canonical = canonical_integrity_snapshot(fresh.snapshot_state())
    except (ValueError, TypeError):
        return None
    return integrity_state_hash(canonical)


def _sidecar_tail_binds(entries: List[Dict[str, Any]], seq: int,
                        state_hash: str) -> bool:
    """Whether a verified sidecar's tail binds generation *seq*/*state_hash*."""
    if not entries:
        return False
    last = entries[-1]
    return last["commit_seq"] == seq and last["state_hash"] == state_hash


class RecoveryPair:
    """One state candidate paired with the sidecar that binds its tail.

    ``sidecar_path`` is ``None`` for a legacy, marker-less state with no
    sidecar at all. The ``*_is_formal`` flags distinguish a file already on a
    formal path from a leftover pin/staged snapshot.
    """

    __slots__ = ("state_path", "sidecar_path", "state_is_formal",
                 "sidecar_is_formal", "commit_seq", "has_seq", "mtime_ns")

    def __init__(self, state_path: str, sidecar_path: Optional[str],
                 state_is_formal: bool, sidecar_is_formal: bool,
                 commit_seq: int, has_seq: bool, mtime_ns: int) -> None:
        self.state_path = state_path
        self.sidecar_path = sidecar_path
        self.state_is_formal = state_is_formal
        self.sidecar_is_formal = sidecar_is_formal
        self.commit_seq = commit_seq
        self.has_seq = has_seq
        self.mtime_ns = mtime_ns


def _pair_state_sidecar_candidates(
        state_cands: List[Tuple[str, Dict[str, Any], bool]],
        sidecar_cands: List[Tuple[str, List[Dict[str, Any]], bool]],
        formal_sidecar_present: bool
) -> List[RecoveryPair]:
    """Combine verified state and sidecar candidates into verifiable pairs.

    A marker-less *legacy* state candidate pairs (with ``sidecar_path``
    ``None``) only while no *formal* sidecar sits beside it — the existing
    missing-sidecar compatibility; any sidecar crash leftovers
    (``.integrity-*.tmp``/``.bak``) are unrelated garbage the caller sweeps
    after promoting the legacy state (a surviving formal sidecar beside a
    marker-less state is the strict startup gate's disagreement). A
    marker-stamped candidate pairs with every sidecar candidate whose
    verified tail binds the state document's generation and canonical state
    hash (more than one is an ambiguity the caller refuses to break). A
    state document carrying a bad marker value, or a marker without any
    binding sidecar, pairs with nothing.
    """
    pairs: List[RecoveryPair] = []
    for state_path, document, state_is_formal in state_cands:
        seq = _commit_seq_of(document)
        has_seq = COMMIT_SEQ_KEY in document
        try:
            mtime_ns = os.stat(state_path).st_mtime_ns
        except OSError:
            mtime_ns = -1
        marker = document.get(INTEGRITY_LOG_VERSION_KEY)
        marker_present = INTEGRITY_LOG_VERSION_KEY in document
        if not marker_present:
            if not formal_sidecar_present:
                pairs.append(RecoveryPair(
                    state_path, None, state_is_formal, False, seq,
                    has_seq, mtime_ns))
            continue
        if not (isinstance(marker, int) and not isinstance(marker, bool)
                and marker == INTEGRITY_LOG_VERSION):
            continue
        state_hash = _state_document_state_hash(document)
        if state_hash is None:
            continue
        for sidecar_path, entries, sidecar_is_formal in sidecar_cands:
            if _sidecar_tail_binds(entries, seq, state_hash):
                pairs.append(RecoveryPair(
                    state_path, sidecar_path, state_is_formal,
                    sidecar_is_formal, seq, has_seq, mtime_ns))
    return pairs


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
        #: Set once a transaction ends in the decidable-on-disk state where
        #: both formal paths are missing and the last committed inodes
        #: survive only as a verifiable pair of ``.bak`` pins (a failed
        #: rollback rename, or a rollback whose follow-up directory fsync
        #: failed). While set, every write first runs :meth:`heal` under the
        #: store lock: the pinned state/sidecar pair is fully verified and
        #: hard-linked back onto both formal paths, the same-transaction
        #: leftovers are cleaned, and only then is the triggering write
        #: committed. A heal that cannot verify or promote the pair keeps
        #: the store degraded and the write fails with 503/field=data_file,
        #: retryable by the next write.
        self.degraded = False
        #: Set instead of :attr:`degraded` when the failed transaction cannot even be
        #: made *decidable*: a formal path could not be vacated, so a
        #: possibly-un-committed residual file still occupies it while the
        #: last committed inodes survive only as ``.bak`` pins. This is the
        #: *blocking state*. :meth:`heal` refuses to serve that residual
        #: formal file as authoritative and :meth:`save` refuses every write
        #: (the HTTP layer answers 503/field=data_file), until the on-disk
        #: state is provably decidable again (both paths resolved with
        #: exactly one verifiable pair), or an operator resolves it. A
        #: sibling ``.state-*.block`` marker makes the state survive a
        #: restart, which keeps refusing while the residual formal file is
        #: present or no unique verifiable pair can be promoted.
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
        """Atomically commit *state* and its sidecar entry as one transaction.

        Both formal files are staged to sibling temporary files
        (``.state-*.tmp`` / ``.integrity-*.tmp``) and fsynced, the current
        targets are pinned with hard links (``.state-*.bak`` /
        ``.integrity-*.bak``) when they already exist, both temporaries are
        ``os.replace``-d over their targets, and the parent directory is
        fsynced once, so a crash always leaves a matched, hash-chain-bound
        pair, never a torn mix.

        Every normal commit (``bootstrap=False``) stamps ``commit_seq`` and
        ``integrity_log_version`` onto the state document and rewrites the
        sidecar with the generation's new entry; on a legacy store's first
        commit the sidecar is created in that same transaction.
        ``bootstrap=True`` writes the state file alone (the empty generation-0
        document for a brand-new store); the chain begins at the first real
        commit.

        Failure handling is symmetric for the two files. Every pre-replace
        failure removes the staged temporaries (and any pins just created),
        leaving both formal files untouched. After a replace has landed a
        failure rolls every replaced file back to its pinned inode (the first
        commit of a not-yet-existing file instead vacates the new snapshot:
        the state snapshot is quarantined, the sidecar deleted) and flushes
        the rollback with one more directory fsync. When that rollback cannot
        be completed durably — a rollback rename fails, or the second fsync
        fails — each restored inode is re-pinned as a ``.bak`` and its formal
        path is vacated, leaving exactly the "both formal paths missing + a
        verifiable pair of old-inode backups" shape :meth:`heal` and restart
        recovery resolve (:attr:`degraded`); when a formal path cannot be
        proven missing the store enters the *blocking* state
        (:attr:`blocked` plus a durable marker). A platform that cannot fsync
        directories skips that capability instead of failing. Any failure
        propagates the underlying :class:`OSError` to the caller; the
        generation advances only after the pair commits durably.
        """
        directory = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(directory, exist_ok=True)
        if self.degraded or self.blocked:
            # The formal paths are not safely authoritative: degraded means
            # they are missing with the last committed inodes pinned only as
            # paired .bak files (a save here could not be rolled back — there
            # is no target pair to back up); blocked means an un-decidable
            # residual may occupy a formal path. Either way the persistence
            # hook must run self.heal() first and only reach this point once
            # the formal pair is restored. Refuse any save that bypassed it.
            state = "blocked" if self.blocked else "degraded"
            raise OSError(
                f"state store for {self.path} is {state}; heal() must "
                f"resolve the on-disk state before another save")
        document = {"version": STATE_VERSION, **state}
        # Stamp the commit generation this save is committing. The counter
        # advances only after both replaces and the directory fsync succeed,
        # so a failed transaction (rolled back below) consumes no generation
        # and the next retry rewrites the same one — generations on disk stay
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
            document[INTEGRITY_LOG_VERSION_KEY] = INTEGRITY_LOG_VERSION
            prev_hash = self.integrity_entries[-1]["hash"] \
                if self.integrity_entries else ""
            entry = {"commit_seq": seq, "state_hash": state_hash,
                     "prev_hash": prev_hash,
                     "hash": _integrity_entry_hash(seq, state_hash, prev_hash)}
        tmp_handle = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=directory, delete=False,
            prefix=_TMP_PREFIX, suffix=_TMP_SUFFIX)
        tmp_path = tmp_handle.name
        # The sidecar is rewritten whole (it stays small: one hash chain
        # entry per generation) through its own sibling temp file, staged
        # before either formal file is touched so the two commit together.
        log_tmp_handle: Any = None
        log_tmp_path: Optional[str] = None
        if not bootstrap:
            log_tmp_handle = tempfile.NamedTemporaryFile(
                mode="wb", dir=directory, delete=False,
                prefix=_INTEGRITY_TMP_PREFIX, suffix=_INTEGRITY_TMP_SUFFIX)
            log_tmp_path = log_tmp_handle.name
        # One descriptor per participating formal file; ``replaced`` flips on
        # once its rename has landed, ``bak`` is the pinned previous inode
        # (None when that formal file did not exist yet).
        plans: List[Dict[str, Any]] = [{
            "target": self.path, "tmp": tmp_path, "handle": tmp_handle,
            "prefix": _TMP_PREFIX, "bak": None, "replaced": False,
            "quarantine": True}]
        if not bootstrap:
            plans.append({
                "target": self.integrity_log_path, "tmp": log_tmp_path,
                "handle": log_tmp_handle, "prefix": _INTEGRITY_TMP_PREFIX,
                "bak": None, "replaced": False, "quarantine": False})
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
            # Pin every existing inode before renaming over it.
            for plan in plans:
                plan["bak"] = _hardlink_backup(directory, plan["target"],
                                               plan["prefix"])
            os.replace(tmp_path, self.path)
            plans[0]["replaced"] = True
            # Publish the sidecar entry in the same transaction: a second
            # atomic sibling rename. On a legacy-enable the sidecar did not
            # exist; afterwards it always does.
            if log_tmp_path is not None:
                os.replace(log_tmp_path, self.integrity_log_path)
                plans[1]["replaced"] = True
            # The renames are only durable once the directory entries are
            # flushed; without this fsync a crash can leave the directory
            # pointing at pre-rename entries even though the replacements
            # landed. One flush covers both renames.
            _fsync_directory(directory)
        except BaseException:
            for plan in plans:
                plan["handle"].close()
                _remove_quietly(plan["tmp"])
            self._roll_back_transaction(directory, plans)
            raise
        # Commit is durable; drop every pinned old inode (best effort — a
        # leftover backup is harmless and swept at heal/startup).
        for plan in plans:
            if plan["bak"] is not None:
                _remove_quietly(plan["bak"])
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

    def _roll_back_transaction(self, directory: str,
                               plans: List[Dict[str, Any]]) -> None:
        """Undo landed renames after a failed two-file commit.

        For every file whose replace landed, rename the pinned old inode back
        over its formal target (when there was no predecessor — the first
        commit creating that file — vacate the new snapshot instead: the
        state snapshot is quarantined, the sidecar deleted), then flush all
        rollback renames with one directory fsync.

        Each plan ends with one of these outcomes:

        * ``restored`` — the committed predecessor is authoritatively back
          on the formal path (and the rollback flush landed);
        * ``pinned-vacated`` — the formal path is missing and the committed
          predecessor survives pinned as a ``.bak`` (a failed rollback
          rename, or a rollback whose directory fsync could not be flushed
          and was therefore re-pinned and vacated);
        * ``new-vacated`` — a file that did not exist before the commit was
          removed again, so the formal path is correctly absent with no
          predecessor;
        * ``residual`` — an un-committed snapshot could not be taken off the
          formal path (undecidable).

        When the rollback directory fsync fails, every ``restored`` inode is
        re-pinned under a fresh backup and its path vacated (becoming
        ``pinned-vacated``); an inode that cannot even be re-pinned simply
        stays authoritative as the committed content (the safe outcome).

        The store ends *blocked* (durable marker) if any residual remains,
        otherwise *degraded* if a committed predecessor is missing from its
        formal path (a state ``new-vacated`` counts too — the first commit
        failed and the formal state is absent; a sidecar ``new-vacated`` is
        the correct pre-enable absence and never alone degrades), otherwise
        cleanly rolled back in place.
        """
        for plan in plans:
            if not plan["replaced"]:
                plan["outcome"] = "untouched"
                continue
            target = plan["target"]
            bak = plan["bak"]
            if bak is not None:
                try:
                    os.replace(bak, target)
                    plan["outcome"] = "restored"
                except OSError:
                    # The old inode stays pinned in bak; take the
                    # un-committed new snapshot off the formal path.
                    _vacate_target(directory, target,
                                   quarantine=plan["quarantine"])
                    plan["outcome"] = "residual" \
                        if os.path.exists(target) else "pinned-vacated"
            else:
                _vacate_target(directory, target,
                               quarantine=plan["quarantine"])
                plan["outcome"] = "residual" \
                    if os.path.exists(target) else "new-vacated"
        try:
            _fsync_directory(directory)
            rollback_durable = True
        except OSError:
            rollback_durable = False
        if not rollback_durable:
            for plan in plans:
                if plan["outcome"] != "restored":
                    continue
                # The rollback rename landed but its durability was not
                # flushed: re-pin the restored inode and vacate its path so
                # heal/startup face the decidable pinned-missing shape. When
                # even that is impossible the committed inode stays
                # authoritative (the safe outcome).
                if _repin_inode(directory, plan["target"], plan["prefix"]):
                    plan["outcome"] = "pinned-vacated"

        any_residual = any(plan["outcome"] == "residual" for plan in plans)
        any_pinned_missing = any(plan["outcome"] == "pinned-vacated"
                                 for plan in plans)
        state_new_missing = any(plan["outcome"] == "new-vacated"
                                and plan["quarantine"] for plan in plans)
        if not any_residual:
            # Pins of files whose rename never landed are redundant: their
            # formal targets never stopped holding the committed inode. Drop
            # them (best effort).
            for plan in plans:
                if plan["outcome"] == "untouched" and plan["bak"] is not None:
                    _remove_quietly(plan["bak"])
                    plan["bak"] = None
            try:
                _fsync_directory(directory)
            except OSError:
                pass
        if any_residual:
            # At least one formal path may still hold an un-committed
            # snapshot: never let it become authoritative. Persist the
            # blocking marker and refuse every later write until an operator
            # (or a verified pair) resolves it.
            _write_block_marker(directory)
            self.blocked = True
        elif any_pinned_missing or state_new_missing:
            # A formal path is missing with a pinned predecessor (or the
            # very first state commit failed): decidable, self-healing on
            # the next write.
            self.degraded = True

    def heal(self, baseline: Dict[str, Any]) -> None:
        """Self-heal the missing formal pair from pinned paired backups.

        Called under the store lock by the persistence hook at the start of
        the next persist-able write after an undecidable write failure. The
        pinned *pair* is verified in full before promotion:

        * the state backup (``.state-*.bak``/``.tmp``) must parse as
          version=1 at the immediately-previous generation, pass the whole
          cross-entity/cursor/nonce/audit-chain restore, and equal the
          process's last committed in-memory *baseline* exactly (the
          anti-dangling gate — a staged new snapshot can never equal it);
        * for a modern (``integrity_log_version``-stamped) state backup the
          sidecar backup (``.integrity-*.bak``/``.tmp``) must be the unique
          one whose hash chain verifies and whose tail binds that
          generation and the state's canonical ``state_hash``;
        * a legacy, marker-less state backup (the sidecar-enabling
          transaction was the one that failed) pairs only with *no* sidecar
          at all — the triggering commit re-anchors the chain afterwards.

        Promotion is a same-directory *hard link* of each chosen backup, so
        the backup names keep pinning the inodes until the formal directory
        entries are fsync-durable. Only then are the same-transaction
        leftovers (redundant backups, staged ``.tmp`` snapshots, the
        un-committed ``.quarantine``) and any block marker removed.

        A later write commit still goes through the normal atomic
        :meth:`save`, so healing consumes no generation: the healed commit
        advances ``commit_seq`` exactly once and carries only that triggering
        request — the earlier request that got 503 is never replayed.

        Raises :class:`OSError` (mapped to 503/field=data_file by the caller)
        when no verifiable pair exists, two or more verifiable pairs exist
        (promotion is refused outright — choosing by name or mtime is
        forbidden), a formal file occupies a path without verifying, or the
        promotion cannot be made durable; in every such case the formal
        paths stay as they were, no leftover and no generation moves,
        :attr:`degraded`/`:attr:`blocked` stays set, and the next write
        retries this heal. A formal state file that verifies as the committed
        baseline always takes precedence over every backup (its missing
        sidecar may be restored from the unique matching sidecar backup); in
        the *blocking* state an un-verified residual formal file is never
        served and heal only unblocks once it is gone with one verifiable
        pair (the durable ``.block`` marker is cleared then).
        """
        if not (self.degraded or self.blocked):
            return
        directory = os.path.dirname(os.path.abspath(self.path))
        state_leftovers = _leftover_tmp_paths(directory, self.path)
        sidecar_leftovers = _sidecar_leftover_paths(
            directory, self.integrity_log_path)
        quarantined = _quarantined_paths(directory, self.path)
        state_formal_exists = os.path.exists(self.path)
        sidecar_formal_exists = os.path.exists(self.integrity_log_path)
        expected_seq = self.commit_seq - 1

        # ---- verify state candidates -------------------------------------
        # (path, document, is_formal); quarantine files are never candidates.
        state_cands: List[Tuple[str, Dict[str, Any], bool]] = []
        if state_formal_exists:
            formal_doc = self._committed_formal_document(baseline,
                                                          expected_seq)
            if formal_doc is not None:
                state_cands.append((self.path, formal_doc, True))
        else:
            for candidate in state_leftovers:
                document = _read_version1_document(candidate)
                if document is None:
                    continue
                if COMMIT_SEQ_KEY in document and not _valid_commit_seq(
                        document[COMMIT_SEQ_KEY]):
                    continue
                if _commit_seq_of(document) != expected_seq:
                    continue
                if not _document_restores(document):
                    continue
                if not _document_payload_equals(document, baseline):
                    continue
                state_cands.append((candidate, document, False))

        # A formal state file occupies the path without verifying as the
        # committed baseline: never serve it, never overwrite it. This is the
        # blocked residual (next generation) and, in the degraded state, an
        # unexpected reappearance — both refused for the operator to resolve.
        if state_formal_exists and not any(
                is_formal for _, _, is_formal in state_cands):
            if self.blocked:
                raise OSError(
                    f"cannot self-heal {self.path}: the store is blocked by "
                    f"an un-resolved, un-committed file on the formal path")
            raise OSError(
                f"cannot self-heal {self.path}: the formal file reappeared "
                f"in an unexpected state")

        # ---- verify sidecar candidates -----------------------------------
        # (path, entries, is_formal); unreadable/broken files are skipped,
        # but a present-but-broken formal sidecar is recorded as an occupant.
        sidecar_cands: List[Tuple[str, List[Dict[str, Any]], bool]] = []
        if sidecar_formal_exists:
            entries = _verify_sidecar_file(self.integrity_log_path)
            if entries is not None:
                sidecar_cands.append(
                    (self.integrity_log_path, entries, True))
        for candidate in sidecar_leftovers:
            entries = _verify_sidecar_file(candidate)
            if entries is not None:
                sidecar_cands.append((candidate, entries, False))

        # A marker-less legacy state backup pairs only while no *formal*
        # sidecar occupies the sidecar path; sidecar leftovers are swept
        # with the other same-transaction garbage after promotion.
        pairs = _pair_state_sidecar_candidates(
            state_cands, sidecar_cands, sidecar_formal_exists)

        winner_state: Optional[str] = None
        winner_sidecar: Optional[str] = None
        winner_state_is_formal = False
        winner_sidecar_is_formal = False
        formal_state_pairs = [pair for pair in pairs if pair.state_is_formal]
        if state_formal_exists:
            # A valid formal state wins outright. Its sidecar must be the
            # valid formal one (backups are then stale garbage, never an
            # ambiguity), or — when the formal sidecar is absent — the
            # unique leftover sidecar its tail binds to.
            if not formal_state_pairs:
                raise OSError(
                    f"cannot self-heal {self.path}: the committed formal "
                    f"state has no matching integrity sidecar")
            formal_sidecar_pairs = [
                pair for pair in formal_state_pairs
                if pair.sidecar_is_formal]
            if formal_sidecar_pairs:
                chosen_pair = formal_sidecar_pairs[0]
            else:
                backup_pairs = formal_state_pairs
                if len(backup_pairs) != 1:
                    raise OSError(
                        f"cannot self-heal {self.path}: the formal sidecar "
                        f"is missing and {len(backup_pairs)} sidecar backups "
                        f"match the committed state; refusing to choose")
                chosen_pair = backup_pairs[0]
            winner_state = chosen_pair.state_path
            winner_sidecar = chosen_pair.sidecar_path
            winner_state_is_formal = True
            winner_sidecar_is_formal = chosen_pair.sidecar_is_formal
        else:
            if not pairs:
                raise OSError(
                    f"cannot self-heal {self.path}: no verifiable pinned "
                    f"backup pair of the last committed state")
            # Two or more verifiable state+sidecar pairs are ambiguous even
            # when byte-identical: never choose between pin names by name or
            # mtime. Refuse and leave everything exactly as found; the next
            # write retries the heal.
            if len(pairs) > 1:
                raise OSError(
                    f"cannot self-heal {self.path}: {len(pairs)} verifiable "
                    f"pinned backup pairs of the last committed state; "
                    f"refusing to choose between candidates")
            chosen_pair = pairs[0]
            winner_state = chosen_pair.state_path
            winner_sidecar = chosen_pair.sidecar_path
            winner_state_is_formal = chosen_pair.state_is_formal
            winner_sidecar_is_formal = chosen_pair.sidecar_is_formal

        # ---- promote the pair with hard links -----------------------------
        # The sidecar directory entry is created first and the state formal
        # path last, so any observer that sees the formal state also has the
        # sidecar link staged; both backup names keep pinning their inodes
        # until the directory fsync makes the entries durable.
        linked: List[str] = []
        try:
            if winner_sidecar is not None and not winner_sidecar_is_formal:
                os.link(winner_sidecar, self.integrity_log_path)
                linked.append(self.integrity_log_path)
            if not winner_state_is_formal:
                os.link(winner_state, self.path)
                linked.append(self.path)
            _fsync_directory(directory)
        except OSError:
            for path in reversed(linked):
                _remove_quietly(path)
            try:
                _fsync_directory(directory)
            except OSError:
                pass
            raise

        # The formal paths durably name the committed inodes; every pinned
        # copy, staged snapshot and demoted (quarantined) new snapshot is now
        # stale garbage. Sweep failures are harmless: a valid formal pair
        # wins every later scan.
        sweep = state_leftovers + sidecar_leftovers + quarantined
        _sweep_leftovers(
            directory,
            sweep + _block_marker_paths(directory, self.path))
        self.degraded = False
        self.blocked = False

    def _committed_formal_document(
            self, baseline: Dict[str, Any],
            expected_seq: int) -> Optional[Dict[str, Any]]:
        """Return the formal state document when it *is* the committed state.

        Version=1, a valid generation exactly one below the next save, a full
        semantic restore, and exact equality with the in-memory last
        committed *baseline*; otherwise ``None`` (an un-committed residual is
        stamped one generation higher and can never match).
        """
        formal = _read_version1_document(self.path)
        if formal is None:
            return None
        if COMMIT_SEQ_KEY in formal and not _valid_commit_seq(
                formal[COMMIT_SEQ_KEY]):
            return None
        if _commit_seq_of(formal) != expected_seq:
            return None
        if not _document_restores(formal):
            return None
        if not _document_payload_equals(formal, baseline):
            return None
        return formal

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
        SHA-256 hex of the canonical compact JSON snapshot (19 ordered
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


def _hardlink_backup(directory: str, target: str,
                     prefix: str = _TMP_PREFIX) -> Optional[str]:
    """Pin the current *target*'s inode via a same-directory hard link.

    Returns the backup path (``<prefix>*`` + ``.bak``), or ``None`` when
    *target* does not exist yet (first commit creating this file: there is no
    old inode to preserve). The link is created in the same directory so it
    is guaranteed to be on the same file system and the subsequent rollback
    ``os.replace`` is a pure metadata rename. Raises :class:`OSError` if the
    link cannot be made; the caller treats that like any pre-replace failure
    (the target is still untouched).
    """
    if not os.path.exists(target):
        return None
    fd, backup = tempfile.mkstemp(dir=directory, prefix=prefix,
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


def _vacate_target(directory: str, target: str, *, quarantine: bool) -> None:
    """Take an un-committed new snapshot off its formal path.

    Used when a post-replace failure cannot be rolled back durably: the new
    inode must never become authoritative, so it is renamed aside to a
    ``.quarantine`` name (which the crash-leftover scan never promotes) and,
    if even that rename is impossible, deleted; or it is deleted outright
    (``quarantine=False`` — the un-committed sidecar, whose content is
    reproducible from the surviving state-side entries and must never be
    parked as a promotion candidate). Either way the formal path is missing
    afterwards; a pinned ``.bak`` of the last committed inode is the recovery
    candidate the next startup promotes. Best effort only — a failure here is
    swallowed because the caller still reports the transaction failed and
    treats an occupant the caller cannot prove gone as the blocking residual.
    """
    if not quarantine:
        _remove_quietly(target)
        return
    try:
        parked = _quarantine_path(directory)
        try:
            os.replace(target, parked)
        except OSError:
            _remove_quietly(parked)
            raise
    except OSError:
        _remove_quietly(target)


def _repin_inode(directory: str, target: str,
                 prefix: str = _TMP_PREFIX) -> bool:
    """Re-pin the restored inode at *target* as a ``.bak`` and vacate it.

    The rollback rename restored the last committed inode at *target*, but
    flushing that rollback failed, so it cannot be treated as durable. Pin
    the inode under a fresh backup name, then drop *target* (a second hard
    link to the same inode, so the unlink loses nothing), recreating the
    "formal path missing + old-inode .bak" triage a plain rollback failure
    leaves. Returns ``True`` when the formal path was vacated. If the inode
    cannot be pinned a second time it simply stays on the formal path — the
    last good state — and the function returns ``False`` so the caller keeps
    that committed file authoritative rather than marking the store
    degraded.
    """
    try:
        fd, backup = tempfile.mkstemp(dir=directory, prefix=prefix,
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
            fd, backup = tempfile.mkstemp(dir=directory, prefix=prefix,
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


def _startup_state_candidates(
        leftovers: List[str]) -> List[Tuple[str, Dict[str, Any]]]:
    """Parse and verify state crash leftovers for startup pair recovery.

    A candidate survives only when it parses as version=1, explicitly
    carries the complete-transaction sections
    (``group_sync_cursors``/``message_sync_cursors``/``key_events`` as
    lists — the footprint of one complete durable transaction), carries no
    malformed generation field, and passes the full semantic restore. A
    missing ``commit_seq`` is the legacy case (ranks at generation 0).
    """
    verified: List[Tuple[str, Dict[str, Any]]] = []
    for candidate in leftovers:
        document = _read_version1_document(candidate)
        if document is None or not _candidate_is_section_complete(document):
            continue
        # A present-but-malformed generation (bool, negative, float, string)
        # can never come from this build's atomic writer; promoting it would
        # only fail the strict formal-file load afterwards.
        if COMMIT_SEQ_KEY in document and not _valid_commit_seq(
                document[COMMIT_SEQ_KEY]):
            continue
        if not _document_restores(document):
            continue
        verified.append((candidate, document))
    return verified


def _enforce_unique_modern_pairing(
        state_cands: List[Tuple[str, Dict[str, Any], bool]],
        sidecar_cands: List[Tuple[str, List[Dict[str, Any]], bool]]
) -> None:
    """Refuse startup unless every verifiable modern state pairs uniquely.

    A verifiable state candidate stamped ``integrity_log_version=1`` is only
    recoverable as one half of a matched pair: exactly one verified sidecar
    whose complete hash chain ends in an entry carrying the state's
    generation and canonical ``state_hash``. Zero binding sidecars (an
    orphaned modern snapshot) or two or more (one state, several sidecars)
    makes the set ambiguous: startup refuses with :class:`OSError` rather
    than downgrading to some lower-generation complete pair or sweeping the
    unpaired candidate as garbage. Two or more distinct states whose tails
    are bound by the same sidecar are ambiguous for the same reason. Raises
    before anything is promoted, renamed or deleted, so every candidate
    keeps its name, bytes and inode.
    """
    bound_sidecar: Dict[str, str] = {}
    for state_path, document, _is_formal in state_cands:
        marker = document.get(INTEGRITY_LOG_VERSION_KEY)
        if not (isinstance(marker, int) and not isinstance(marker, bool)
                and marker == INTEGRITY_LOG_VERSION):
            # Legacy (marker-less) and bad-marker candidates pair by the
            # legacy rules in _pair_state_sidecar_candidates.
            continue
        state_hash = _state_document_state_hash(document)
        if state_hash is None:
            continue
        seq = _commit_seq_of(document)
        binding = [sidecar_path
                   for sidecar_path, entries, _formal in sidecar_cands
                   if _sidecar_tail_binds(entries, seq, state_hash)]
        if len(binding) != 1:
            raise OSError(
                f"ambiguous state recovery: verifiable state candidate "
                f"{state_path} (commit_seq {seq}) is bound by "
                f"{len(binding)} integrity sidecars instead of exactly "
                f"one; refusing to choose, downgrade or discard it")
        other = bound_sidecar.setdefault(binding[0], state_path)
        if other != state_path:
            raise OSError(
                f"ambiguous state recovery: integrity sidecar "
                f"{binding[0]} binds both {other} and {state_path}; "
                f"refusing to choose between them")


def _gather_startup_pairs(
        state_store: "JsonStateStore",
        include_formal_state: bool,
        include_formal_sidecar: bool
) -> Tuple[List[RecoveryPair], List[str], List[str]]:
    """Build verifiable state+sidecar pairs from leftovers (and formal files).

    Returns ``(pairs, state_leftovers, sidecar_leftovers)``. State leftovers
    pass the complete-transaction gate; sidecar leftovers and (optionally)
    the formal sidecar contribute verified chains. A marker-less legacy
    state pairs only when no sidecar file exists at all; a stamped state
    pairs with every sidecar whose verified tail binds its generation and
    state hash. Before any pair is built, every verifiable
    ``integrity_log_version=1`` state candidate must bind exactly one
    sidecar (and no sidecar may bind two states); otherwise
    :func:`_enforce_unique_modern_pairing` raises :class:`OSError` and
    startup refuses with nothing promoted, deleted or overwritten.
    """
    directory = os.path.dirname(os.path.abspath(state_store.path))
    state_leftovers = _leftover_tmp_paths(directory, state_store.path)
    sidecar_leftovers = _sidecar_leftover_paths(
        directory, state_store.integrity_log_path)
    state_cands: List[Tuple[str, Dict[str, Any], bool]] = [
        (path, document, False)
        for path, document in _startup_state_candidates(state_leftovers)]
    if include_formal_state:
        formal = _read_version1_document(state_store.path)
        if formal is not None and (
                COMMIT_SEQ_KEY not in formal
                or _valid_commit_seq(formal[COMMIT_SEQ_KEY])) \
                and _document_restores(formal):
            state_cands.append((state_store.path, formal, True))
    sidecar_cands: List[Tuple[str, List[Dict[str, Any]], bool]] = [
        (path, entries, False)
        for path in sidecar_leftovers
        for entries in [_verify_sidecar_file(path)]
        if entries is not None]
    if include_formal_sidecar and os.path.exists(
            state_store.integrity_log_path):
        entries = _verify_sidecar_file(state_store.integrity_log_path)
        if entries is not None:
            sidecar_cands.append(
                (state_store.integrity_log_path, entries, True))
    # A verifiable modern (integrity_log_version=1) state candidate without
    # exactly one binding sidecar — or two states bound by one sidecar —
    # makes the whole set ambiguous: refuse before pairing, promoting or
    # sweeping anything (a lower-generation complete pair is never a
    # downgrade target).
    _enforce_unique_modern_pairing(state_cands, sidecar_cands)
    # A legacy (marker-less) state candidate is pairable while no formal
    # sidecar occupies the sidecar path; sidecar leftovers are crash garbage
    # swept after recovery, not a disqualifier for legacy state.
    pairs = _pair_state_sidecar_candidates(
        state_cands, sidecar_cands,
        os.path.exists(state_store.integrity_log_path))
    return pairs, state_leftovers, sidecar_leftovers


def _promote_pair(directory: str, state_store: "JsonStateStore",
                  pair: RecoveryPair) -> None:
    """Atomically put one verified *pair* onto both formal paths.

    The sidecar (when the pair has one) is renamed first and the state file
    last, so any observer seeing the formal state also sees its sidecar; one
    parent-directory fsync then makes both renames durable. A legacy pair
    (``sidecar_path is None``) restores the state alone with no sidecar.
    """
    if pair.sidecar_path is not None and not pair.sidecar_is_formal:
        os.replace(pair.sidecar_path, state_store.integrity_log_path)
    if not pair.state_is_formal:
        os.replace(pair.state_path, state_store.path)
    _fsync_directory(directory)


def _choose_recovery_pair(pairs: List[RecoveryPair]) -> Optional[RecoveryPair]:
    """Pick the unique highest-generation pair, or ``None`` when ambiguous.

    Generation ranks first (highest ``commit_seq`` wins regardless of
    mtime); two or more distinct verifiable pairs tied at the highest
    generation are ambiguous and resolve to ``None`` (the caller refuses to
    choose by name or mtime). Only when every pair predates commit
    generations does the legacy newest-mtime rule apply (ties broken by
    name, deterministically).
    """
    if not pairs:
        return None
    if any(pair.has_seq for pair in pairs):
        ranked = [pair for pair in pairs if pair.has_seq]
        ranked.sort(key=lambda pair: (pair.commit_seq, pair.mtime_ns,
                                      pair.state_path,
                                      pair.sidecar_path or ""))
        top_seq = ranked[-1].commit_seq
        top = [pair for pair in ranked if pair.commit_seq == top_seq]
        if len(top) >= 2:
            return None
        return top[0]
    pairs.sort(key=lambda pair: (pair.mtime_ns, pair.state_path))
    return pairs[-1]


def recover_crash_leftovers(state_store: "JsonStateStore") -> None:
    """Resolve crashed two-file transactions left beside the formal pair.

    Recovery always works on matched *pairs* (a version=1 state snapshot and
    the sidecar whose verified chain tail binds its generation and
    ``state_hash``; a marker-less legacy state with no sidecar at all is the
    legacy pair).

    * A valid formal pair wins outright: it is never overwritten and every
      leftover is removed. When the formal state is intact but its sidecar
      formal file is missing, the unique leftover sidecar that verifies and
      binds its tail is restored onto the formal sidecar path; a broken
      formal sidecar or a missing match is left for the strict startup gate
      to refuse (nothing overwritten).
    * An existing but corrupt/invalid formal state file is left untouched:
      the normal load then refuses startup, and leftovers are kept for
      inspection.
    * With a ``.block`` marker (the previous process ended *blocked*): a
      formal state proven to be the committed one (it semantically equals a
      verifiable state backup at its generation, with a sidecar that binds
      or a unique binding sidecar backup) wins and clears the marker;
      anything else on the formal path is the un-committed residual and
      makes startup refuse, touching nothing. With the formal state
      missing, the unique verifiable generation-bearing pair is promoted
      and the marker swept; zero or several verifiable pairs refuse and
      leave everything in place.
    * With no marker and the formal state missing, every verifiable
      ``integrity_log_version=1`` state candidate must bind exactly one
      verified sidecar (and no sidecar may bind two states): an orphaned
      modern snapshot, a state with several binding sidecars or a sidecar
      shared by several states all make startup raise
      :class:`StateFileError` with nothing promoted, deleted or
      overwritten — a lower-generation complete pair is never recovered as
      a downgrade. Otherwise pairs rank by generation
      first; two or more verifiable pairs tied at the highest generation are
      ambiguous and make startup raise :class:`StateFileError` (nothing
      promoted or deleted) rather than ever breaking the tie by name or
      mtime; only fully pre-generation pairs keep the legacy newest-mtime
      rule. After promotion every remaining leftover is removed.
    * When no verifiable candidate exists at all, all leftovers (state and
      sidecar, staged ``.tmp`` and pinned ``.bak``), quarantine files and
      any dangling formal sidecar are removed; :func:`attach_persistence`
      then creates an empty state.
    """
    directory = os.path.dirname(os.path.abspath(state_store.path))
    leftovers = _leftover_tmp_paths(directory, state_store.path)
    sidecar_leftovers = _sidecar_leftover_paths(
        directory, state_store.integrity_log_path)
    quarantined = _quarantined_paths(directory, state_store.path)
    markers = _block_marker_paths(directory, state_store.path)
    if not leftovers and not sidecar_leftovers and not quarantined \
            and not markers:
        return

    state_formal_exists = os.path.exists(state_store.path)
    sidecar_formal_exists = os.path.exists(state_store.integrity_log_path)

    if markers:
        _recover_blocked(state_store, directory, leftovers,
                         sidecar_leftovers, quarantined, markers,
                         state_formal_exists, sidecar_formal_exists)
        return

    if state_formal_exists:
        _recover_with_formal_state(state_store, directory, leftovers,
                                   sidecar_leftovers, quarantined)
        return

    # Formal state missing: verify state+sidecar pairs from leftovers (a
    # surviving formal sidecar is itself a pairing candidate).
    pairs, state_leftovers, sidecar_leftovers = _gather_startup_pairs(
        state_store, False, True)
    chosen = _choose_recovery_pair(pairs)
    if chosen is None and pairs:
        # Verifiable candidates exist but no unique top pair: refuse,
        # deleting/overwriting nothing for an operator to resolve.
        top_seq = max(pair.commit_seq for pair in pairs if pair.has_seq)
        raise OSError(
            f"ambiguous state recovery for {state_store.path}: multiple "
            f"verifiable state/sidecar pairs at commit_seq {top_seq}; "
            f"refusing to choose between them by name or mtime")
    if chosen is not None:
        _promote_pair(directory, state_store, chosen)
    else:
        # No verifiable pair at all. A dangling sidecar formal file refers
        # to generations that no longer exist; drop it so the empty state
        # bootstrap starts the chain cleanly.
        _remove_quietly(state_store.integrity_log_path)
    # Whether recovered or created empty next, every leftover is stale.
    for path in state_leftovers + sidecar_leftovers + quarantined:
        _remove_quietly(path)
    _fsync_directory(directory)


def _formal_pair_acceptable(
        state_store: "JsonStateStore",
        formal: Dict[str, Any],
        require_backup_witness: bool,
        state_backup_docs: List[Dict[str, Any]]) -> Optional[RecoveryPair]:
    """Verify the formal state against its sidecar for startup recovery.

    Returns the accepted formal :class:`RecoveryPair` (its sidecar path may
    be a leftover backup that must be restored onto the formal sidecar
    path), or ``None`` when the formal state/sidecar arrangement does not
    verify. When *require_backup_witness* is set (the blocked branch) the
    formal state must additionally semantically equal one of
    *state_backup_docs* at the same generation — proof that a file on a
    possibly-residual formal path is the committed snapshot rather than the
    un-committed successor.
    """
    seq = _commit_seq_of(formal)
    if require_backup_witness and not any(
            _commit_seq_of(witness) == seq
            and _documents_payload_equal(formal, witness)
            for witness in state_backup_docs):
        return None
    marker_present = INTEGRITY_LOG_VERSION_KEY in formal
    if not marker_present:
        # Legacy formal state (missing-sidecar compatibility): acceptable
        # only while no formal sidecar sits beside it. Any sidecar leftover
        # is crash garbage the valid formal file wins over and the caller
        # sweeps; a formal sidecar beside a marker-less file is the strict
        # startup gate's marker/sidecar disagreement and is left untouched.
        if os.path.exists(state_store.integrity_log_path):
            return None
        return RecoveryPair(state_store.path, None, True, False, seq,
                            COMMIT_SEQ_KEY in formal, -1)
    marker = formal[INTEGRITY_LOG_VERSION_KEY]
    if not (isinstance(marker, int) and not isinstance(marker, bool)
            and marker == INTEGRITY_LOG_VERSION):
        return None
    state_hash = _state_document_state_hash(formal)
    if state_hash is None:
        return None
    formal_entries = _verify_sidecar_file(state_store.integrity_log_path)
    if formal_entries is not None and _sidecar_tail_binds(
            formal_entries, seq, state_hash):
        return RecoveryPair(state_store.path,
                            state_store.integrity_log_path, True, True, seq,
                            COMMIT_SEQ_KEY in formal, -1)
    # A broken formal sidecar is never overwritten; only an absent one may
    # be restored from the unique binding backup.
    if os.path.exists(state_store.integrity_log_path):
        return None
    binding: List[Tuple[str, List[Dict[str, Any]]]] = []
    for path in _sidecar_leftover_paths(
            os.path.dirname(os.path.abspath(state_store.path)),
            state_store.integrity_log_path):
        entries = _verify_sidecar_file(path)
        if entries is not None and _sidecar_tail_binds(
                entries, seq, state_hash):
            binding.append((path, entries))
    if len(binding) != 1:
        return None
    return RecoveryPair(state_store.path, binding[0][0], True, False, seq,
                        COMMIT_SEQ_KEY in formal, -1)


def _recover_with_formal_state(
        state_store: "JsonStateStore", directory: str,
        leftovers: List[str], sidecar_leftovers: List[str],
        quarantined: List[str]) -> None:
    """A formal state file exists: accept a valid pair and sweep, or leave.

    A formally valid state with a sidecar that verifies/binds (or a unique
    binding sidecar backup to restore onto a missing formal sidecar) stays
    authoritative and every leftover is removed. An invalid state file, a
    marker/sidecar disagreement or a broken chain leaves everything
    untouched for the strict startup gate to refuse with files intact.
    """
    formal = _read_version1_document(state_store.path)
    formal_valid = (
        formal is not None
        and (COMMIT_SEQ_KEY not in formal
             or _valid_commit_seq(formal[COMMIT_SEQ_KEY]))
        and _document_restores(formal))
    if not formal_valid:
        return
    accepted = _formal_pair_acceptable(
        state_store, formal, False, [])
    if accepted is None:
        return
    if accepted.sidecar_path is not None and not accepted.sidecar_is_formal:
        os.replace(accepted.sidecar_path,
                   state_store.integrity_log_path)
        _fsync_directory(directory)
    for path in leftovers + sidecar_leftovers + quarantined:
        _remove_quietly(path)
    _fsync_directory(directory)


def _recover_blocked(
        state_store: "JsonStateStore", directory: str,
        leftovers: List[str], sidecar_leftovers: List[str],
        quarantined: List[str], markers: List[str],
        state_formal_exists: bool, sidecar_formal_exists: bool) -> None:
    """Resolve a directory a previous process marked *blocked*.

    A formal state proven committed against a verifiable backup wins (its
    missing sidecar may come from the unique binding backup) and clears the
    marker; any other occupant of the formal path makes startup refuse with
    everything in place. With the formal state missing the unique
    verifiable generation-bearing pair is promoted; zero or several refuse.
    """
    state_backup_docs = [
        document for _path, document in
        _startup_state_candidates(leftovers)]
    if state_formal_exists:
        formal = _read_version1_document(state_store.path)
        if formal is not None and (
                COMMIT_SEQ_KEY not in formal
                or _valid_commit_seq(formal[COMMIT_SEQ_KEY])) \
                and _document_restores(formal):
            accepted = _formal_pair_acceptable(
                state_store, formal, True, state_backup_docs)
        else:
            accepted = None
        if accepted is None:
            raise OSError(
                f"state store for {state_store.path} is blocked: an "
                f"un-resolved, un-committed file occupies the formal path")
        if accepted.sidecar_path is not None and not accepted.sidecar_is_formal:
            os.replace(accepted.sidecar_path,
                       state_store.integrity_log_path)
            _fsync_directory(directory)
        for path in leftovers + sidecar_leftovers + quarantined + markers:
            _remove_quietly(path)
        _fsync_directory(directory)
        return

    # Formal state missing: only a unique verifiable generation-bearing
    # pair unblocks; a surviving formal sidecar is a pairing candidate but
    # a fully pre-generation legacy pair never resolves the marker.
    pairs, state_leftovers, sidecar_leftovers = _gather_startup_pairs(
        state_store, False, True)
    pairs = [pair for pair in pairs if pair.has_seq]
    if pairs:
        top_seq = max(pair.commit_seq for pair in pairs)
        top = [pair for pair in pairs if pair.commit_seq == top_seq]
    else:
        top = []
    if len(top) != 1:
        raise OSError(
            f"state store for {state_store.path} is blocked: the formal "
            f"path is missing with no unique verifiable backup pair")
    _promote_pair(directory, state_store, top[0])
    promoted_state = top[0].state_path
    promoted_sidecar = top[0].sidecar_path
    for path in state_leftovers + sidecar_leftovers + quarantined + markers:
        if path in (promoted_state, promoted_sidecar):
            continue
        _remove_quietly(path)
    _fsync_directory(directory)


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
    # First resolve crashed two-file transactions left in the same
    # directory: a valid formal pair stays and leftovers are removed; a
    # missing formal state is restored from the unique highest-generation
    # verifiable state+sidecar pair (a marker-less legacy state with no
    # sidecar is the legacy pair); verifiable candidates without a unique
    # pair make startup refuse (nothing deleted or overwritten); with no
    # verifiable candidate the leftovers are swept and the missing-file
    # branch below creates an empty state.
    try:
        recover_crash_leftovers(state_store)
    except OSError as error:
        raise StateFileError(
            f"cannot recover state file {path} from a crash leftover: "
            f"{error}") from None
    document = state_store.load()
    if document is None:
        # Crash recovery could not restore a state document (the formal file
        # was missing and every leftover was absent or unverifiable), so an
        # empty state is created. Any sidecar left beside it refers to
        # generations that no longer exist and is swept here too (recovery
        # normally already did), keeping the pairing marker-less/
        # sidecar-less until the first real commit anchors a fresh chain.
        _remove_quietly(state_store.integrity_log_path)
        for leftover in _sidecar_leftover_paths(
                os.path.dirname(os.path.abspath(path)),
                state_store.integrity_log_path):
            _remove_quietly(leftover)
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
            # A previous transaction ended undecidable on disk. Degraded:
            # both formal paths are missing and the last committed inodes
            # survive only as a verifiable pair of .bak pins. Blocked: an
            # un-decidable residual may occupy a formal path (a durable
            # .block marker records it). Before this (the next) write may
            # commit, run heal inside the same storage lock: in the degraded
            # case it verifies and hard-links the unique pair onto both
            # formal paths, cleaning the failed transaction's leftovers; in
            # the blocked case it refuses while an unverified residual
            # occupies a path (503) and resolves only once the paths are
            # missing with exactly one verifiable pair. Memory was rolled
            # back to last_good after the 503 and, like every later failed
            # attempt, restored to it again, so the only difference between
            # pending and last_good is *this* current request — the earlier
            # 503 request is never replayed.
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
