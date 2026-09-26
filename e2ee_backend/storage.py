"""Thread-safe in-memory device storage.

Devices live for the lifetime of the server process. Only public keys and
identifiers are retained. Reads return pre-key ids in the exact insertion
order, so repeated requests list them identically.
"""
from __future__ import annotations

import copy
import hashlib
import json
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .models import (
    BatchClaimDevice,
    BatchClaimSessionBinding,
    BatchClaimSessionEntry,
    ClaimSessionBinding,
    Device,
    Group,
    GroupSession,
    GroupSessionRotation,
    GroupSyncCursor,
    KeyEvent,
    Message,
    MessageDelivery,
    MessageLease,
    MessageLeaseCompletion,
    MessageLeaseRenewal,
    MessageSubmission,
    MessageSyncCursor,
    PreKeyBatchClaim,
    PreKeyClaim,
    RedeliveryJob,
    RedeliveryJobRecovery,
    Session,
    SignedPreKey,
    utc_now_iso,
)

#: Outcome codes for a failed atomic session creation.
SESSION_INITIATOR_UNKNOWN = "initiator_unknown"
SESSION_RECIPIENT_UNKNOWN = "recipient_unknown"
SESSION_PREKEY_UNKNOWN = "prekey_unknown"
SESSION_INITIATOR_REVOKED = "initiator_revoked"
SESSION_RECIPIENT_REVOKED = "recipient_revoked"
SESSION_PREKEY_REVOKED = "prekey_revoked"
SESSION_PREKEY_CONSUMED = "prekey_consumed"

#: Outcome codes for a failed session creation from a pre-key claim.
CLAIM_SESSION_CLAIM_UNKNOWN = "claim_unknown"
CLAIM_SESSION_DUPLICATE = "claim_already_used"
CLAIM_SESSION_INITIATOR_UNKNOWN = SESSION_INITIATOR_UNKNOWN
CLAIM_SESSION_INITIATOR_REVOKED = SESSION_INITIATOR_REVOKED
CLAIM_SESSION_RECIPIENT_REVOKED = SESSION_RECIPIENT_REVOKED
CLAIM_SESSION_PREKEY_REVOKED = SESSION_PREKEY_REVOKED

#: Outcome codes for an atomic pre-key claim.
CLAIM_RECIPIENT_UNKNOWN = "recipient_unknown"
CLAIM_RECIPIENT_REVOKED = "recipient_revoked"
CLAIM_NO_PREKEY = "prekey_unavailable"

#: Outcome codes for an atomic user-wide batch pre-key claim.
BATCH_CLAIM_USER_UNKNOWN = "user_unknown"
BATCH_CLAIM_NO_ACTIVE_DEVICE = "no_active_device"
BATCH_CLAIM_NO_PREKEY = "prekey_unavailable"
#: The claim_id was already committed by a single (or batch) claim of the
#: other kind; the shared idempotency namespace forbids the reuse.
CLAIM_ID_CONFLICT = "claim_id_conflict"

#: Outcome codes for an atomic batch-claim multi-device session creation.
BATCH_SESSION_CLAIM_UNKNOWN = "batch_claim_unknown"
BATCH_SESSION_CLAIM_WRONG_KIND = "batch_claim_wrong_kind"
BATCH_SESSION_DUPLICATE = "batch_claim_already_used"
BATCH_SESSION_INITIATOR_UNKNOWN = SESSION_INITIATOR_UNKNOWN
BATCH_SESSION_INITIATOR_REVOKED = SESSION_INITIATOR_REVOKED
BATCH_SESSION_INITIATOR_IN_SNAPSHOT = "initiator_in_snapshot"
BATCH_SESSION_DEVICE_SET_MISMATCH = "device_set_mismatch"
BATCH_SESSION_RECIPIENT_REVOKED = SESSION_RECIPIENT_REVOKED
BATCH_SESSION_PREKEY_REVOKED = SESSION_PREKEY_REVOKED

#: Outcome codes for a failed atomic message append.
MESSAGE_SESSION_UNKNOWN = "session_unknown"
MESSAGE_SENDER_INACTIVE = "sender_inactive"
MESSAGE_DUPLICATE_ID = "duplicate_message_id"
MESSAGE_BAD_SEQUENCE = "bad_sequence"
MESSAGE_DUPLICATE_NONCE = "duplicate_nonce"
#: A request_id already committed with different envelope fields.
MESSAGE_REQUEST_ID_CONFLICT = "request_id_conflict"

#: Outcome code for a failed message listing.
MESSAGE_DEVICE_INACTIVE = "device_inactive"

#: Outcome codes for delivery (retry/ack/status) failures.
DELIVERY_SESSION_UNKNOWN = "session_unknown"
DELIVERY_MESSAGE_UNKNOWN = "message_unknown"
DELIVERY_DEVICE_MISMATCH = "device_mismatch"
DELIVERY_DEVICE_INACTIVE = "device_inactive"
DELIVERY_BAD_SEQUENCE = "bad_sequence"

#: Outcome codes for identity-key rotation.
DEVICE_UNKNOWN = "device_unknown"
DEVICE_REVOKED = "device_revoked"
#: Outcome code for a pre-key add conflict (same id, changed key, or revoked).
PREKEY_CONFLICT = "prekey_conflict"

#: Outcome codes for group creation / membership changes.
GROUP_UNKNOWN = "group_unknown"
GROUP_DUPLICATE_ID = "group_duplicate_id"
GROUP_CREATOR_UNKNOWN = "creator_unknown"
GROUP_CREATOR_REVOKED = "creator_revoked"
GROUP_ACTOR_UNKNOWN = "actor_unknown"
GROUP_ACTOR_REVOKED = "actor_revoked"
GROUP_ACTOR_NOT_CREATOR = "actor_not_creator"
GROUP_DEVICE_UNKNOWN = "group_device_unknown"
GROUP_DEVICE_REVOKED = "group_device_revoked"

#: Outcome codes for group-session creation.
GROUP_SESSION_GROUP_UNKNOWN = "group_unknown"
GROUP_SESSION_INITIATOR_UNKNOWN = "initiator_unknown"
GROUP_SESSION_INITIATOR_INACTIVE = "initiator_inactive"
GROUP_SESSION_INITIATOR_NOT_MEMBER = "initiator_not_member"

#: Outcome codes for group-session sync (message listing / checkpoints).
SYNC_SESSION_UNKNOWN = "session_unknown"
SYNC_DEVICE_UNKNOWN = "device_unknown"
SYNC_DEVICE_INACTIVE = "device_inactive"
SYNC_DEVICE_NOT_MEMBER = "device_not_member"
SYNC_CURSOR_CONFLICT = "cursor_conflict"

#: Outcome codes for unified 1:1/group-session message sync.
MESSAGE_SYNC_SESSION_UNKNOWN = "session_unknown"
MESSAGE_SYNC_DEVICE_UNKNOWN = "device_unknown"
MESSAGE_SYNC_DEVICE_INACTIVE = "device_inactive"
MESSAGE_SYNC_DEVICE_NOT_PARTICIPANT = "device_not_participant"
MESSAGE_SYNC_CURSOR_CONFLICT = "cursor_conflict"

#: Outcome codes for a device 1:1-inbox retry batch.
INBOX_RETRY_SESSION_UNKNOWN = "session_unknown"
INBOX_RETRY_NOT_RECIPIENT = "session_not_recipient"
INBOX_RETRY_MESSAGE_UNKNOWN = "message_unknown"
INBOX_RETRY_MESSAGE_ACKED = "message_acked"

#: Outcome codes for a 1:1-inbox redelivery lease claim.
INBOX_LEASE_DEVICE_UNKNOWN = "device_unknown"
INBOX_LEASE_DEVICE_INACTIVE = "device_inactive"
#: The lease_id was already committed for another device or with another
#: limit (or a mismatched deadline); replay is only allowed for the exact
#: same device and limit.
INBOX_LEASE_CONFLICT = "lease_id_conflict"
#: No lease was ever committed under the lease_id a release names.
INBOX_LEASE_NOT_FOUND = "lease_not_found"
#: A renewal named a lease that is released or already past its current
#: effective deadline; neither can be extended.
INBOX_LEASE_UNAVAILABLE = "lease_unavailable"
#: A completion reused a completion_id already committed on the same lease
#: with a different outcome (an exact replay answers 200 instead). Ids are
#: scoped to one lease and may recur on other leases.
INBOX_LEASE_COMPLETION_CONFLICT = "completion_id_conflict"
#: A lease bulk-ack named a lease whose completion is missing or whose
#: ``outcome`` is not ``delivered``; only a delivered lease may be acked.
INBOX_LEASE_NOT_DELIVERED = "lease_not_delivered"

#: Lifetime of one inbox redelivery lease, in seconds. A message leased by a
#: claim is withheld from later claims until this deadline passes, after
#: which a fresh ``lease_id`` may claim it again.
INBOX_LEASE_SECONDS = 30

#: States of a 1:1-inbox redelivery job (``POST /v1/inbox-jobs``).
REDELIVERY_JOB_PENDING = "pending"
REDELIVERY_JOB_RUNNING = "running"
REDELIVERY_JOB_SUCCEEDED = "succeeded"
REDELIVERY_JOB_FAILED = "failed"
REDELIVERY_JOB_CANCELLED = "cancelled"
REDELIVERY_JOB_STATES = frozenset({
    REDELIVERY_JOB_PENDING,
    REDELIVERY_JOB_RUNNING,
    REDELIVERY_JOB_SUCCEEDED,
    REDELIVERY_JOB_FAILED,
    REDELIVERY_JOB_CANCELLED,
})

#: Outcome codes for a 1:1-inbox redelivery job submission.
REDELIVERY_JOB_DEVICE_UNKNOWN = "device_unknown"
REDELIVERY_JOB_DEVICE_INACTIVE = "device_inactive"
#: The job_id was already committed for another device.
REDELIVERY_JOB_CONFLICT = "job_id_conflict"
#: A dispatch/status named a job_id that was never queued.
REDELIVERY_JOB_NOT_FOUND = "job_not_found"
#: A dispatch could not take the job's lease because the job_id is already
#: occupied as an inbox lease id by an unrelated claim.
REDELIVERY_JOB_LEASE_OCCUPIED = "lease_occupied"
#: A recover named a recovery_id already committed on another job, or
#: already occupied as an unrelated inbox lease id (409/recovery_id).
REDELIVERY_JOB_RECOVERY_CONFLICT = "recovery_id_conflict"
#: A recover targeted a job whose dispatch lease is still valid, or a job
#: not in the ``running`` state (409/lease_id for a still-valid lease,
#: 409/job_id for the other states).
REDELIVERY_JOB_RECOVERY_LEASE_ACTIVE = "lease_active"
REDELIVERY_JOB_RECOVERY_STATE = "job_not_recoverable"
#: A cancel named a different cancellation_id on an already cancelled job
#: (409/cancellation_id). The same id on the same job is an idempotent
#: replay; a succeeded/failed job is a terminal-state conflict instead.
REDELIVERY_JOB_CANCELLATION_CONFLICT = "cancellation_id_conflict"
#: A cancel targeted a job already in a terminal ``succeeded``/``failed``
#: state (409/job_id). ``cancelled`` is handled by its own idempotency /
#: cancellation_id-conflict rules.
REDELIVERY_JOB_CANCEL_STATE = "job_not_cancellable"
#: A recover-batch mixed first-time items with replays of already committed
#: recoveries; the batch is atomic, so a partial replay conflicts
#: (409/items[i].recovery_id of the first replayed item).
REDELIVERY_JOB_RECOVERY_PARTIAL_REPLAY = "partial_replay"

#: Number of inbox messages one redelivery-job dispatch leases at most.
REDELIVERY_JOB_DISPATCH_LIMIT = 100


def _is_utc_microsecond_iso(value: Any) -> bool:
    """Whether *value* is a canonical UTC ISO-8601 timestamp.

    The only form this server ever writes: six microsecond digits and a
    ``+00:00`` offset, exactly as ``isoformat(timespec="microseconds")``
    emits. The round-trip comparison also rejects unparseable or
    non-canonical (but ISO-valid) renderings.
    """
    if not isinstance(value, str):
        return False
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return False
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        return False
    return parsed.isoformat(timespec="microseconds") == value

#: Outcome codes for group-session rotation.
ROTATION_SESSION_UNKNOWN = "session_unknown"
ROTATION_ACTOR_UNKNOWN = "actor_unknown"
ROTATION_ACTOR_REVOKED = "actor_revoked"
ROTATION_ACTOR_NOT_CREATOR = "actor_not_creator"
ROTATION_REVISION_MISMATCH = "revision_mismatch"
ROTATION_ID_CONFLICT = "rotation_id_conflict"
ROTATION_PREDECESSOR_ROTATED = "predecessor_rotated"

#: Key-audit event types (the ``type`` field of a :class:`KeyEvent`).
KEY_EVENT_REGISTERED = "registered"
KEY_EVENT_IDENTITY_ROTATED = "identity_rotated"
KEY_EVENT_PREKEY_ADDED = "prekey_added"
KEY_EVENT_PREKEY_REVOKED = "prekey_revoked"
KEY_EVENT_DEVICE_REVOKED = "device_revoked"

#: All five key-audit event types, for restore-time validation.
KEY_EVENT_TYPES = frozenset({
    KEY_EVENT_REGISTERED,
    KEY_EVENT_IDENTITY_ROTATED,
    KEY_EVENT_PREKEY_ADDED,
    KEY_EVENT_PREKEY_REVOKED,
    KEY_EVENT_DEVICE_REVOKED,
})


def key_event_hash(device_id: str, seq: int, event_type: str,
                   payload: Dict[str, Any], prev_hash: str,
                   created_at: str) -> str:
    """Compute the chain hash of one key-audit event.

    The hashed document is the event *without* its ``hash`` field,
    serialized as canonical JSON — keys sorted, compact separators, Unicode
    written as-is — encoded as UTF-8 and digested with SHA-256, rendered as
    lowercase hex.
    """
    document = {
        "device_id": device_id,
        "seq": seq,
        "type": event_type,
        "payload": payload,
        "prev_hash": prev_hash,
        "created_at": created_at,
    }
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class SessionCreateError(Exception):
    """An atomic session lookup/revocation check failed (nothing was written)."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class MessageCreateError(Exception):
    """An atomic message append check failed (nothing was written)."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class MessageListError(Exception):
    """An atomic message listing check failed."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class DeliveryError(Exception):
    """An atomic delivery (retry/ack/status) check failed; nothing changed."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class DeviceUpdateError(Exception):
    """An atomic identity-key rotation or pre-key add failed; nothing changed."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class PreKeyClaimError(Exception):
    """An atomic pre-key claim failed (unknown/revoked device, no key)."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class ClaimSessionError(Exception):
    """An atomic session-from-claim creation failed (nothing was written)."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class PreKeyBatchClaimError(Exception):
    """An atomic batch pre-key claim failed (unknown user / no key); nothing changed."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class BatchClaimSessionError(Exception):
    """An atomic session-set creation from a batch claim failed; nothing was written.

    Carries the offending recipient's device id (or pre-key id) for the
    reasons that name ``recipient_device_id`` / ``prekey_id``.
    """

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


class GroupError(Exception):
    """An atomic group or group-session operation failed; nothing changed."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class GroupSyncError(Exception):
    """An atomic group-session sync/checkpoint failed; the cursor is unchanged."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class MessageSyncError(Exception):
    """An atomic 1:1/group session sync/checkpoint failed; the cursor unchanged."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class MessageSyncAckBatchError(Exception):
    """A device-scoped batch sync-ack failed at one array item.

    Carries the zero-based *index* of the first (and only reported) offending
    item; validation runs in array order and the whole batch writes nothing.
    """

    def __init__(self, reason: str, index: int) -> None:
        super().__init__(reason)
        self.reason = reason
        self.index = index


class InboxRetryBatchError(Exception):
    """A device 1:1-inbox retry batch failed at one array item.

    Carries the zero-based *index* of the first (and only reported) offending
    item; items are prechecked in array order and the whole batch writes
    nothing.
    """

    def __init__(self, reason: str, index: int) -> None:
        super().__init__(reason)
        self.reason = reason
        self.index = index


class InboxLeaseError(Exception):
    """A 1:1-inbox lease claim failed; nothing was leased.

    The device-level reasons (``device_unknown`` / ``device_inactive``) map
    to 409/field=device_id like the other inbox endpoints; ``lease_id_conflict``
    maps to 409/field=lease_id when an occupied id is replayed for another
    device or with a changed limit.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class RedeliveryJobError(Exception):
    """A 1:1-inbox redelivery job operation failed; nothing was written.

    The device-level reasons (``device_unknown`` / ``device_inactive``) map
    to 409/field=device_id; ``job_not_found`` maps to 404/field=job_id and
    ``job_id_conflict`` to 409/field=job_id. A ``recover`` additionally
    raises ``recovery_id_conflict`` (409/field=recovery_id),
    ``lease_active`` (409/field=lease_id) and ``job_not_recoverable``
    (409/field=job_id). A ``cancel`` raises ``cancellation_id_conflict``
    (409/field=cancellation_id) when an already cancelled job is cancelled
    again under a different id, and ``job_not_cancellable``
    (409/field=job_id) for a ``succeeded``/``failed`` terminal job.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class RedeliveryJobRecoverBatchError(Exception):
    """A batch redelivery-job recovery failed at one array item.

    Carries the zero-based *index* of the first (and only reported)
    offending item; items are prechecked in array order and the whole batch
    writes nothing. Reasons are the per-item
    :class:`RedeliveryJobError` recover reasons plus ``partial_replay`` for
    a batch mixing first-time items with replays of committed recoveries.
    """

    def __init__(self, reason: str, index: int) -> None:
        super().__init__(reason)
        self.reason = reason
        self.index = index


class GroupSessionRotationError(Exception):
    """An atomic group-session rotation failed; nothing was written."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


#: Business-state sections covered by a persistence-integrity snapshot, in
#: the fixed key order used both for the state hash and when comparing the
#: on-disk document against the in-memory snapshot. Sections a legacy
#: version-1 file omits are filled with their empty defaults here, so the
#: hash/identity of an old document matches the store that loaded it.
INTEGRITY_SECTION_KEYS = (
    "devices",
    "sessions",
    "groups",
    "group_sessions",
    "messages",
    "delivery",
    "prekey_claims",
    "prekey_batch_claims",
    "claim_session_bindings",
    "batch_claim_session_bindings",
    "group_session_rotations",
    "group_delivery",
    "used_nonces",
    "group_sync_cursors",
    "message_sync_cursors",
    "message_submissions",
    "key_events",
    "redelivery_jobs",
)

#: Empty default for every canonical section; ``messages`` and
#: ``used_nonces`` are keyed mappings, the rest are lists. Callers get fresh
#: containers per use.
_INTEGRITY_SECTION_DEFAULTS: Dict[str, Any] = {
    name: (dict() if name in ("messages", "used_nonces") else list())
    for name in INTEGRITY_SECTION_KEYS
}


def canonical_integrity_snapshot(snapshot: Dict[str, Any]) -> "Dict[str, Any]":
    """Project a store snapshot/document payload onto the 18 canonical
    sections in :data:`INTEGRITY_SECTION_KEYS`, filling missing sections with
    their empty defaults. Unknown envelope keys (``version``,
    ``commit_seq``) are dropped and key order is normalised, so two
    semantically equal states always serialise identically regardless of the
    order of the source mapping.
    """
    canonical: Dict[str, Any] = {}
    for name in INTEGRITY_SECTION_KEYS:
        if name in snapshot:
            canonical[name] = snapshot[name]
        else:
            canonical[name] = copy.deepcopy(_INTEGRITY_SECTION_DEFAULTS[name])
    return canonical


def integrity_state_hash(canonical_snapshot: Dict[str, Any]) -> str:
    """SHA-256 (lowercase hex) of the canonical snapshot as compact UTF-8 JSON.

    Serialisation matches the durable writer: key order as built
    (``sort_keys`` off — the canonical mapping already fixes the section
    order), no whitespace, ``ensure_ascii=False`` so non-ASCII strings are
    hashed as their literal UTF-8 bytes rather than ``\\uXXXX`` escapes.
    """
    payload = json.dumps(canonical_snapshot, separators=(",", ":"),
                         ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class DeviceStore:
    """In-memory store keyed by ``(user_id, device_id)``.

    Device ids are additionally indexed globally, because the public GET route
    addresses a device by ``device_id`` alone.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        # Condition bound to the store lock. Read-only long polls (the inbox
        # wait) block on it without holding the lock between wakeups, while
        # every committed mutation notifies it once, so a wait is woken
        # promptly when a message could become deliverable or a device is
        # revoked. It carries no state of its own and is never persisted.
        self._condition = threading.Condition(self._lock)
        self._devices: Dict[Tuple[str, str], Device] = {}
        self._device_index: Dict[str, Tuple[str, str]] = {}
        self._sessions: Dict[str, Session] = {}
        # Successful one-time pre-key claims keyed by the client-chosen
        # claim_id, which is globally unique and idempotency-bearing.
        self._prekey_claims: Dict[str, PreKeyClaim] = {}
        # Successful user-wide batch claims, keyed by claim_id. The namespace
        # is shared with _prekey_claims: one claim_id can never be both.
        self._prekey_batch_claims: Dict[str, PreKeyBatchClaim] = {}
        # Sessions established via POST /v1/sessions/from-claim, keyed by the
        # claim_id they consumed. At most one binding per claim_id.
        self._claim_session_bindings: Dict[str, ClaimSessionBinding] = {}
        # Session sets established via POST /v1/sessions/from-batch-claim,
        # keyed by the batch claim_id they consumed. At most one binding per
        # batch claim_id, recording every session created from the claim.
        self._batch_claim_session_bindings: \
            Dict[str, BatchClaimSessionBinding] = {}
        self._groups: Dict[str, Group] = {}
        self._group_sessions: Dict[str, GroupSession] = {}
        # Committed group-session rotations, keyed by the client-chosen
        # rotation_id (globally unique). Two derived indexes make replay and
        # the no-fork rule O(1): by predecessor and by successor.
        self._group_session_rotations: Dict[str, GroupSessionRotation] = {}
        self._rotation_by_predecessor: Dict[str, GroupSessionRotation] = {}
        self._rotation_by_successor: Dict[str, GroupSessionRotation] = {}
        # Per-device group-session read cursors, keyed by
        # (session_id, device_id); created lazily on the first advance.
        self._group_sync_cursors: Dict[Tuple[str, str], GroupSyncCursor] = {}
        # Per-device read cursors for the unified 1:1/group-session sync API
        # (GET /v1/sessions/{id}/sync), keyed by (session_id, device_id);
        # created lazily on the first advance.
        self._message_sync_cursors: Dict[Tuple[str, str], MessageSyncCursor] = {}
        self._messages: Dict[str, List[Message]] = {}
        # Per-session set of nonces already accepted for replay protection.
        # Keyed independently of the streams so a nonce is scoped to a session.
        self._used_nonces: Dict[str, Set[str]] = {}
        # Committed idempotent message submissions, keyed by the client-chosen
        # request_id (globally unique). Each record freezes the envelope it
        # accepted so a replay returns the original response.
        self._message_submissions: Dict[str, MessageSubmission] = {}
        # Delivery state keyed by (session_id, message_id).
        self._delivery: Dict[Tuple[str, str], MessageDelivery] = {}
        # 1:1-inbox redelivery jobs, keyed by the client-chosen job_id
        # (globally unique). A dispatched job's lease lives on the leased
        # messages' delivery records under a lease_id equal to the job_id.
        self._redelivery_jobs: Dict[str, RedeliveryJob] = {}
        # Per-device delivery state for group sessions, keyed by
        # (session_id, message_id, device_id): every frozen non-sender
        # member accumulates its own dedup/ack record.
        self._group_delivery: Dict[Tuple[str, str, str], MessageDelivery] = {}
        # Append-only key-audit chains, keyed by device_id. Each committed
        # key-material change of a device appends exactly one event, inside
        # the same locked transaction as the mutation itself, so the chain
        # is persisted and rolled back together with the rest of the state.
        self._key_events: Dict[str, List[KeyEvent]] = {}
        # Devices restored from a legacy version-1 file that predates the
        # key_events section, in registration order: each still needs a
        # synthetic anchor chain, built lazily inside the same locked
        # transaction as the first change that will be persisted. ``None``
        # means no legacy file was loaded (nothing is pending).
        self._pending_anchor_devices: Optional[List[str]] = None
        # Called (under the lock) after any state mutation, for persistence.
        self.on_change: Optional[Callable[[], None]] = None

    def _notify_change(self) -> None:
        """Invoke the persistence hook after a committed mutation."""
        # Wake long-polling inbox waits first: the mutation has been applied
        # to the in-memory state under this lock, so a waiter released here
        # can immediately re-observe it. The wake is harmless if the
        # persistence hook then fails and rolls the mutation back: the waiter
        # rechecks under the lock and simply waits again.
        self._condition.notify_all()
        # Any change that will be persisted first closes the legacy gap:
        # anchor every chainless device in the same locked transaction.
        self._migrate_pending_anchors()
        if self.on_change is not None:
            self.on_change()

    def _append_key_event(self, device_id: str, event_type: str,
                          payload: Dict[str, Any]) -> KeyEvent:
        """Append one event to a device's key-audit chain. Lock required.

        The event's ``seq`` continues the chain (starting at 1); its
        ``prev_hash`` is empty for the first event and the previous event's
        ``hash`` afterwards. The caller appends only for real state
        transitions — idempotent replays and failed operations add nothing.
        """
        chain = self._key_events.setdefault(device_id, [])
        seq = len(chain) + 1
        prev_hash = chain[-1].hash if chain else ""
        created_at = utc_now_iso()
        event = KeyEvent(
            device_id=device_id, seq=seq, type=event_type, payload=payload,
            prev_hash=prev_hash,
            hash=key_event_hash(device_id, seq, event_type, payload,
                                prev_hash, created_at),
            created_at=created_at)
        chain.append(event)
        return event

    def _migrate_pending_anchors(self) -> None:
        """Synthesize anchor chains for legacy (section-less) state files.

        A version-1 file written before the key-audit feature carries no
        ``key_events`` section. Its devices are still fully described by the
        devices section, so before the first change that will be persisted the
        missing chains are rebuilt here, inside the same locked transaction:

        * per device, in registration order: a ``registered`` event seeds the
          *current* identity key and every pre-key in stored order; its
          ``prev_hash`` is empty and ``created_at`` is the fixed
          ``registered_at``;
        * then, in pre-key order, one ``prekey_revoked`` event per pre-key
          that the devices section already marks revoked;
        * then a ``device_revoked`` event for a device already revoked.

        Every synthetic event gets its ``seq``/``prev_hash``/``hash``
        recomputed by the same rules as live events, so the rebuilt chain
        replays to exactly the stored key material and revocation state — no
        revocation is duplicated or dropped. The caller must hold the store
        lock and invoke this *before* mutating a device's key material, so the
        anchor always captures the pre-change state. No-op unless a legacy
        file was restored.
        """
        pending = self._pending_anchor_devices
        if pending is None:
            return
        if not pending:
            # A legacy file with no devices has nothing to anchor; behave as
            # modern from here on so future registrations persist normally.
            self._pending_anchor_devices = None
            return
        for device_id in pending:
            if self._key_events.get(device_id):
                # A chain already exists (defensive: all pending devices are
                # chainless by construction).
                continue
            key = self._device_index.get(device_id)
            device = self._devices.get(key) if key is not None else None
            if device is None:
                continue
            # A revoked device carries the live invariant that every pre-key
            # is revoked too (revoke_device sets both atomically). A legacy
            # file may predate the audit chain and record only the device
            # flag, which the lenient loader accepts. Normalize the in-memory
            # flags here — inside the same transaction — so the rebuilt chain
            # replays to exactly the persisted devices section and no
            # revocation is omitted. A revoked device's key flags are never
            # observable differently afterwards.
            if device.revoked:
                for prekey in device.prekeys:
                    prekey.revoked = True
            anchored: List[KeyEvent] = []

            def append(event_type: str, payload: Dict[str, Any]) -> None:
                seq = len(anchored) + 1
                prev_hash = anchored[-1].hash if anchored else ""
                created_at = device.registered_at
                event_hash = key_event_hash(
                    device_id, seq, event_type, payload, prev_hash,
                    created_at)
                anchored.append(KeyEvent(
                    device_id=device_id, seq=seq, type=event_type,
                    payload=payload, prev_hash=prev_hash, hash=event_hash,
                    created_at=created_at))

            append(KEY_EVENT_REGISTERED,
                   {"identity_key": device.identity_key,
                    "signed_prekeys": [{"key_id": pk.key_id,
                                        "public_key": pk.public_key}
                                       for pk in device.prekeys]})
            # Revocation markers only: pre-key consumption (a committed claim)
            # is not part of the audit chain.
            for prekey in device.prekeys:
                if prekey.revoked:
                    append(KEY_EVENT_PREKEY_REVOKED,
                           {"key_id": prekey.key_id,
                            "public_key": prekey.public_key})
            if device.revoked:
                append(KEY_EVENT_DEVICE_REVOKED, {})
            self._key_events[device_id] = anchored
        self._pending_anchor_devices = None

    def add_device(self, device: Device) -> bool:
        """Insert a device.

        Return ``False`` (and store nothing) when the same ``device_id`` is
        already registered, whether under the same user or another one.
        """
        key = (device.user_id, device.device_id)
        with self._lock:
            if key in self._devices or device.device_id in self._device_index:
                return False
            # Anchor any legacy chainless devices in this same transaction
            # before the new device's own registered event opens its chain.
            self._migrate_pending_anchors()
            self._devices[key] = device
            self._device_index[device.device_id] = key
            self._append_key_event(
                device.device_id, KEY_EVENT_REGISTERED,
                {"identity_key": device.identity_key,
                 "signed_prekeys": [{"key_id": pk.key_id,
                                     "public_key": pk.public_key}
                                    for pk in device.prekeys]})
            self._notify_change()
            return True

    def get_device(self, user_id: str, device_id: str) -> Optional[Device]:
        """Return the device for the user, or ``None``."""
        with self._lock:
            return self._devices.get((user_id, device_id))

    def find_by_device_id(self, device_id: str) -> Optional[Device]:
        """Return the (globally unique) device with this id, or ``None``."""
        with self._lock:
            key = self._device_index.get(device_id)
            return self._devices.get(key) if key is not None else None

    def active_prekey_ids(self, device: Device) -> List[str]:
        """Return key ids of available (non-revoked, un-consumed) pre-keys.

        Order is the stable insertion order; revoked keys and keys already
        handed out by a committed claim are both excluded.
        """
        with self._lock:
            return [pk.key_id for pk in device.prekeys
                    if not pk.revoked and not pk.consumed]

    def revoke_prekey(self, device: Device, key_id: str) -> bool:
        """Mark one of the device's pre-keys revoked. Return ``False`` if absent.

        Re-revoking an already-revoked key is a state-free idempotent replay:
        it neither notifies persistence nor consumes a commit generation.
        """
        with self._lock:
            for prekey in device.prekeys:
                if prekey.key_id == key_id:
                    if not prekey.revoked:
                        prekey.revoked = True
                        self._notify_change()
                    return True
            return False

    def revoke_device(self, device_id: str) -> Optional[Device]:
        """Mark the device (and its pre-keys) revoked.

        Idempotent: a previously revoked device stays revoked. Returns the
        device or ``None`` when the id is unknown.
        """
        with self._lock:
            key = self._device_index.get(device_id)
            device = self._devices.get(key) if key is not None else None
            if device is None:
                return None
            if not device.revoked:
                # A real state transition anchors any legacy chain, appends
                # the single device-revoked event and persists exactly once.
                # A repeated revoke is a state-free idempotent replay: it
                # must not migrate anchors, append an event, refresh
                # anything, touch the file or consume a commit generation.
                self._migrate_pending_anchors()
                device.revoked = True
                for prekey in device.prekeys:
                    prekey.revoked = True
                # One event covers the whole revocation: the empty payload
                # marks the device and every pre-key of it revoked at once.
                self._append_key_event(
                    device_id, KEY_EVENT_DEVICE_REVOKED, {})
                self._notify_change()
            return device

    def revoke_prekey_by_id(self, device_id: str,
                            key_id: str) -> Tuple[Optional[Device], bool]:
        """Revoke one pre-key of a globally-addressed device.

        Returns ``(device, key_found)``: ``device`` is ``None`` for an unknown
        device; otherwise ``key_found`` says whether *key_id* existed. Revoking
        an already-revoked key is idempotent and reports it as found.
        """
        with self._lock:
            key = self._device_index.get(device_id)
            device = self._devices.get(key) if key is not None else None
            if device is None:
                return None, False
            for prekey in device.prekeys:
                if prekey.key_id == key_id:
                    if not prekey.revoked:
                        # A real revocation anchors legacy chains, appends
                        # the key event and persists in one transaction. An
                        # already-revoked key is a state-free idempotent
                        # replay: no migration, event, file write or commit
                        # generation.
                        self._migrate_pending_anchors()
                        prekey.revoked = True
                        self._append_key_event(
                            device_id, KEY_EVENT_PREKEY_REVOKED,
                            {"key_id": prekey.key_id,
                             "public_key": prekey.public_key})
                        self._notify_change()
                    return device, True
            return device, False

    def identity_view(self, device: Device) -> Dict[str, Any]:
        """Copy the device's current identity material into a three-field view."""
        return {
            "device_id": device.device_id,
            "identity_key": device.identity_key,
            "rotated_at": device.rotated_at,
        }

    def rotate_identity_key(self, device_id: str, identity_key: str
                            ) -> Tuple[Dict[str, Any], bool]:
        """Atomically rotate a device's identity key.

        Returns ``(view, changed)``: ``changed`` is True when the key was
        actually replaced (and ``rotated_at`` advanced), False when the new
        key equals the stored one (idempotent; ``rotated_at`` untouched).
        Unknown -> :class:`DeviceUpdateError` ``device_unknown`` (404);
        revoked -> ``device_revoked`` (409). All checks and the write happen
        under the store lock; a failure writes nothing.
        """
        with self._lock:
            key = self._device_index.get(device_id)
            device = self._devices.get(key) if key is not None else None
            if device is None:
                raise DeviceUpdateError(DEVICE_UNKNOWN)
            if device.revoked:
                raise DeviceUpdateError(DEVICE_REVOKED)
            changed = identity_key != device.identity_key
            if changed:
                self._migrate_pending_anchors()
                old_identity_key = device.identity_key
                device.identity_key = identity_key
                device.rotated_at = utc_now_iso()
                self._append_key_event(
                    device_id, KEY_EVENT_IDENTITY_ROTATED,
                    {"old_identity_key": old_identity_key,
                     "new_identity_key": identity_key})
                self._notify_change()
            return self.identity_view(device), changed

    def add_prekey(self, device_id: str, key_id: str, public_key: str
                   ) -> Tuple[Dict[str, Any], bool]:
        """Atomically append a pre-key (or confirm an identical existing one).

        A new ``key_id`` is appended in order and reported with ``created``
        True (201). An existing, non-revoked key with the same ``public_key``
        is idempotent (200, ``created`` False). An existing id with a changed
        key, or an id whose key was revoked, raises
        :class:`DeviceUpdateError` ``prekey_conflict`` (409/key_id). Unknown
        device -> ``device_unknown`` (404), revoked device -> ``device_revoked``
        (409). Locked; a failure writes nothing.
        """
        with self._lock:
            key = self._device_index.get(device_id)
            device = self._devices.get(key) if key is not None else None
            if device is None:
                raise DeviceUpdateError(DEVICE_UNKNOWN)
            if device.revoked:
                raise DeviceUpdateError(DEVICE_REVOKED)
            existing = next((pk for pk in device.prekeys
                             if pk.key_id == key_id), None)
            if existing is not None:
                if not existing.revoked and existing.public_key == public_key:
                    return ({"device_id": device.device_id,
                             "key_id": existing.key_id,
                             "public_key": existing.public_key}, False)
                raise DeviceUpdateError(PREKEY_CONFLICT)
            self._migrate_pending_anchors()
            prekey = SignedPreKey(key_id=key_id, public_key=public_key)
            device.prekeys.append(prekey)
            self._append_key_event(
                device_id, KEY_EVENT_PREKEY_ADDED,
                {"key_id": key_id, "public_key": public_key})
            self._notify_change()
            return ({"device_id": device.device_id,
                     "key_id": prekey.key_id,
                     "public_key": prekey.public_key}, True)

    def public_view(self, device_id: str) -> Optional[Dict[str, Any]]:
        """Atomically build the public snapshot of a device.

        The active pre-key id list is copied under the lock together with the
        rest of the record, so a concurrent revocation can never be observed
        half-applied.
        """
        with self._lock:
            key = self._device_index.get(device_id)
            device = self._devices.get(key) if key is not None else None
            if device is None:
                return None
            return {
                "identity_key": device.identity_key,
                "prekey_ids": [pk.key_id for pk in device.prekeys
                               if not pk.revoked and not pk.consumed],
                "registered_at": device.registered_at,
            }

    # -- key-audit chain ---------------------------------------------------

    @staticmethod
    def key_event_view(event: KeyEvent) -> Dict[str, Any]:
        """Copy one key-audit event into its public seven-field view."""
        return {
            "device_id": event.device_id,
            "seq": event.seq,
            "type": event.type,
            "payload": event.payload,
            "prev_hash": event.prev_hash,
            "hash": event.hash,
            "created_at": event.created_at,
        }

    def key_events_page(self, device_id: str, after: int, limit: int
                        ) -> Optional[Tuple[List[Dict[str, Any]], int, bool]]:
        """Atomically read one ascending page of a device's key-audit chain.

        Returns ``(event_views, next_after, has_more)``: the events with
        ``seq > after`` (ascending, at most *limit*), the sequence to resume
        from — the last returned ``seq``, or *after* itself when the page is
        empty — and whether further events follow. ``None`` when the device
        id is unknown. A revoked device's chain stays readable. The lookup
        and the page copy run under the store lock, so a concurrent append
        is linearized either wholly before or wholly after this read.
        """
        with self._lock:
            if device_id not in self._device_index:
                return None
            chain = self._key_events.get(device_id, [])
            page = [event for event in chain if event.seq > after][:limit]
            next_after = page[-1].seq if page else after
            has_more = any(event.seq > next_after for event in chain)
            return ([self.key_event_view(event) for event in page],
                    next_after, has_more)

    # -- pre-key claims ----------------------------------------------------

    @staticmethod
    def claim_view(claim: PreKeyClaim) -> Dict[str, Any]:
        """Copy one claim into its public six-field view."""
        return {
            "claim_id": claim.claim_id,
            "recipient_device_id": claim.recipient_device_id,
            "identity_key": claim.identity_key,
            "key_id": claim.key_id,
            "public_key": claim.public_key,
            "claimed_at": claim.claimed_at,
        }

    def claim_prekey(self, recipient_device_id: str, claim_id: str
                     ) -> Tuple[Dict[str, Any], bool]:
        """Atomically claim one of a device's one-time pre-keys.

        The whole check-and-consume runs under the store lock — the same lock
        session creation and device/key revocations take — so concurrent
        claims, session creations and revocations are linearized and exactly
        one claim wins a race for the last available key.

        A previously committed *claim_id* is idempotent: its frozen record is
        returned unchanged (``created`` False -> 200) and no further key is
        consumed, regardless of intervening rotation or revocation. A fresh
        *claim_id* takes the first pre-key (insertion order) that is neither
        revoked nor already consumed, marks it consumed, and records the
        claim. Unknown recipient -> :class:`PreKeyClaimError`
        ``recipient_unknown`` (404); revoked recipient -> ``recipient_revoked``
        (409); no available key -> ``prekey_unavailable`` (409/field
        ``prekey_id``). On failure nothing is written.
        """
        with self._lock:
            existing = self._prekey_claims.get(claim_id)
            if existing is not None:
                return self.claim_view(existing), False
            # claim_ids are a shared namespace: an id already committed by a
            # batch claim cannot also start a single-device claim.
            existing_batch = self._prekey_batch_claims.get(claim_id)
            if existing_batch is not None:
                raise PreKeyClaimError(CLAIM_ID_CONFLICT)

            recipient_key = self._device_index.get(recipient_device_id)
            recipient = (self._devices.get(recipient_key)
                         if recipient_key is not None else None)
            if recipient is None:
                raise PreKeyClaimError(CLAIM_RECIPIENT_UNKNOWN)
            if recipient.revoked:
                raise PreKeyClaimError(CLAIM_RECIPIENT_REVOKED)

            available = next((pk for pk in recipient.prekeys
                              if not pk.revoked and not pk.consumed), None)
            if available is None:
                raise PreKeyClaimError(CLAIM_NO_PREKEY)

            available.consumed = True
            claim = PreKeyClaim(
                claim_id=claim_id,
                recipient_device_id=recipient_device_id,
                key_id=available.key_id,
                identity_key=recipient.identity_key,
                public_key=available.public_key,
            )
            self._prekey_claims[claim_id] = claim
            self._notify_change()
            return self.claim_view(claim), True

    # -- user-wide batch pre-key claims -----------------------------------

    @staticmethod
    def batch_claim_view(claim: PreKeyBatchClaim) -> Dict[str, Any]:
        """Copy one batch claim into its public response view."""
        return {
            "claim_id": claim.claim_id,
            "user_id": claim.user_id,
            "claimed_at": claim.claimed_at,
            "devices": [{
                "device_id": entry.device_id,
                "identity_key": entry.identity_key,
                "key_id": entry.key_id,
                "public_key": entry.public_key,
            } for entry in claim.devices],
        }

    def claim_prekey_batch(self, user_id: str, claim_id: str
                           ) -> Tuple[Dict[str, Any], bool]:
        """Atomically claim one pre-key on every active device of a user.

        The user's devices are enumerated in registration order, skipping
        revoked devices. For each active device the first pre-key
        (insertion order) that is neither revoked nor consumed is selected.
        Selection runs first for *every* device; only when all devices have
        an available key are all of them marked consumed at once and the
        batch record written, so a device short of a key aborts the whole
        claim and no key on any device is consumed. The whole check-and-
        consume holds the store lock — the same lock single claims, session
        creation, revocations and pre-key replenishment take — so all of
        those linearize together.

        A previously committed batch *claim_id* is idempotent: its frozen
        record is returned unchanged (``created`` False -> 200) and no
        further key is consumed, regardless of later rotation/revocation.
        A user with no registered device raises
        :class:`PreKeyBatchClaimError` ``user_unknown`` (404/field
        ``user_id``); one whose devices are all revoked raises
        ``no_active_device`` (409/field ``device_id``); any active device
        without an available key raises ``prekey_unavailable`` (409/field
        ``prekey_id``). On failure nothing is written.
        """
        with self._lock:
            existing = self._prekey_batch_claims.get(claim_id)
            if existing is not None:
                return self.batch_claim_view(existing), False
            # claim_ids are one shared namespace with single claims: an id
            # already committed by a single claim cannot start a batch claim
            # (the persisted sections would otherwise collide on restart).
            if claim_id in self._prekey_claims:
                raise PreKeyBatchClaimError(CLAIM_ID_CONFLICT)

            user_devices = [device for key, device in self._devices.items()
                            if key[0] == user_id]
            if not user_devices:
                raise PreKeyBatchClaimError(BATCH_CLAIM_USER_UNKNOWN)
            active_devices = [device for device in user_devices
                              if not device.revoked]
            if not active_devices:
                raise PreKeyBatchClaimError(BATCH_CLAIM_NO_ACTIVE_DEVICE)

            # Select first, consume after: build the full (device, key) list
            # before mutating anything, so one device without a key aborts
            # the entire batch with zero consumption.
            selections: List[Tuple[Device, SignedPreKey]] = []
            for device in active_devices:
                available = next((pk for pk in device.prekeys
                                  if not pk.revoked and not pk.consumed), None)
                if available is None:
                    raise PreKeyBatchClaimError(BATCH_CLAIM_NO_PREKEY)
                selections.append((device, available))

            entries: List[BatchClaimDevice] = []
            for device, prekey in selections:
                prekey.consumed = True
                entries.append(BatchClaimDevice(
                    device_id=device.device_id,
                    identity_key=device.identity_key,
                    key_id=prekey.key_id,
                    public_key=prekey.public_key))
            batch = PreKeyBatchClaim(
                claim_id=claim_id, user_id=user_id, devices=entries)
            self._prekey_batch_claims[claim_id] = batch
            self._notify_change()
            return self.batch_claim_view(batch), True

    # -- sessions ----------------------------------------------------------

    def create_session(self, initiator_device_id: str, recipient_device_id: str,
                       prekey_id: str, ephemeral_key: str) -> Session:
        """Atomically validate and create one session.

        Lookups, revocation checks and the insert all happen while holding the
        store lock (the same lock device/pre-key revocations take), so a
        concurrent revocation is linearized either wholly before this call
        (then it fails) or wholly after it (then the session is retained).
        On any failure nothing is written and :class:`SessionCreateError`
        carries the reason.
        """
        with self._lock:
            initiator = self._device_index.get(initiator_device_id)
            initiator = self._devices.get(initiator) if initiator is not None else None
            if initiator is None:
                raise SessionCreateError(SESSION_INITIATOR_UNKNOWN)
            if initiator.revoked:
                raise SessionCreateError(SESSION_INITIATOR_REVOKED)

            recipient_key = self._device_index.get(recipient_device_id)
            recipient = (self._devices.get(recipient_key)
                         if recipient_key is not None else None)
            if recipient is None:
                raise SessionCreateError(SESSION_RECIPIENT_UNKNOWN)
            if recipient.revoked:
                raise SessionCreateError(SESSION_RECIPIENT_REVOKED)

            used_prekey = next((pk for pk in recipient.prekeys
                                if pk.key_id == prekey_id), None)
            if used_prekey is None:
                raise SessionCreateError(SESSION_PREKEY_UNKNOWN)
            if used_prekey.revoked:
                raise SessionCreateError(SESSION_PREKEY_REVOKED)
            if used_prekey.consumed:
                raise SessionCreateError(SESSION_PREKEY_CONSUMED)

            session = Session(
                session_id=uuid.uuid4().hex,
                initiator_device_id=initiator_device_id,
                recipient_device_id=recipient_device_id,
                prekey_id=prekey_id,
                ephemeral_key=ephemeral_key,
                identity_key=recipient.identity_key,
                public_key=used_prekey.public_key,
            )
            self._sessions[session.session_id] = session
            self._notify_change()
            return session

    def create_session_from_claim(self, claim_id: str,
                                  initiator_device_id: str,
                                  ephemeral_key: str) -> Session:
        """Atomically establish one session from a committed pre-key claim.

        The whole check-and-write runs under the store lock — the same lock
        taken by claims, ordinary session creation and device/pre-key
        revocations — so the binding check, every status check and the two
        inserts (session plus the claim binding) linearize together and each
        ``claim_id`` can win at most once, however many requests race.

        A ``claim_id`` without a committed claim is a 404; one whose binding
        already exists is a 409 (``claim_id``), and the repeat creates no
        session. An unknown initiator is a 404 and a revoked initiator a 409
        (``initiator_device_id``); revocation after the claim of the claim's
        recipient device or pre-key is a 409 naming ``recipient_device_id`` or
        ``prekey_id`` respectively. On success the session freezes the claim's
        ``recipient_device_id``, ``prekey_id``, ``identity_key`` and
        ``public_key`` at claim time; the initiator and ephemeral key come
        from the request. On any failure nothing is written.
        """
        with self._lock:
            if claim_id in self._claim_session_bindings:
                raise ClaimSessionError(CLAIM_SESSION_DUPLICATE)

            claim = self._prekey_claims.get(claim_id)
            if claim is None:
                raise ClaimSessionError(CLAIM_SESSION_CLAIM_UNKNOWN)

            initiator_key = self._device_index.get(initiator_device_id)
            initiator = (self._devices.get(initiator_key)
                         if initiator_key is not None else None)
            if initiator is None:
                raise ClaimSessionError(CLAIM_SESSION_INITIATOR_UNKNOWN)
            if initiator.revoked:
                raise ClaimSessionError(CLAIM_SESSION_INITIATOR_REVOKED)

            recipient_key = self._device_index.get(claim.recipient_device_id)
            recipient = (self._devices.get(recipient_key)
                         if recipient_key is not None else None)
            if recipient is None:
                # A committed claim always names a registered device
                # (restore_state enforces it and devices are never deleted);
                # reaching here means no usable recipient remains.
                raise ClaimSessionError(CLAIM_SESSION_RECIPIENT_REVOKED)
            if recipient.revoked:
                raise ClaimSessionError(CLAIM_SESSION_RECIPIENT_REVOKED)

            used_prekey = next((pk for pk in recipient.prekeys
                                if pk.key_id == claim.key_id), None)
            if used_prekey is None or used_prekey.revoked:
                raise ClaimSessionError(CLAIM_SESSION_PREKEY_REVOKED)

            session = Session(
                session_id=uuid.uuid4().hex,
                initiator_device_id=initiator_device_id,
                recipient_device_id=claim.recipient_device_id,
                prekey_id=claim.key_id,
                ephemeral_key=ephemeral_key,
                identity_key=claim.identity_key,
                public_key=claim.public_key,
            )
            self._sessions[session.session_id] = session
            self._claim_session_bindings[claim_id] = ClaimSessionBinding(
                claim_id=claim_id,
                session_id=session.session_id,
                recipient_device_id=claim.recipient_device_id,
                prekey_id=claim.key_id,
                identity_key=claim.identity_key,
                public_key=claim.public_key,
                created_at=session.created_at)
            self._notify_change()
            return session

    def get_batch_claim(self, claim_id: str) -> Optional[PreKeyBatchClaim]:
        """Return the committed batch claim with this id, or ``None``."""
        with self._lock:
            return self._prekey_batch_claims.get(claim_id)

    def create_sessions_from_batch_claim(
            self, claim_id: str, initiator_device_id: str,
            ephemeral_keys: List[Tuple[str, str]]
    ) -> BatchClaimSessionBinding:
        """Atomically establish one session per device of a batch claim.

        The whole check-and-write runs under the store lock — the same lock
        taken by both kinds of claims, ordinary session creation and
        device/pre-key revocations — so every status check, all session
        inserts and the single batch binding linearize together: either
        every session is created with the binding or nothing at all.

        *ephemeral_keys* is the request's ordered list of
        ``(device_id, ephemeral_key)`` pairs; its device set must equal the
        batch claim's frozen snapshot exactly. Validation happens in the
        batch claim's frozen device order. A ``claim_id`` without a batch
        claim is a 404 (``claim_id``; one naming a single claim is reported
        the same way at the HTTP layer); one whose batch binding already
        exists is a 409 (``claim_id``) and creates nothing. An
        unknown/revoked initiator is a 404/409 naming
        ``initiator_device_id``; an initiator that is itself one of the
        claimed devices is a 400 naming ``initiator_device_id``. A request
        device outside the snapshot is a 400 with the offending device id on
        :attr:`BatchClaimSessionError.detail`; a snapshot device missing
        from the request is a 400 with an empty detail. A claimed recipient
        device revoked after the claim is a 409 naming
        ``recipient_device_id`` (the offending device id rides on
        :attr:`BatchClaimSessionError.detail`), and a claimed pre-key
        revoked afterwards is a 409 naming ``prekey_id``. Every session
        freezes the batch claim's recipient/key/public material; only the
        ephemeral key (looked up in *ephemeral_keys* by recipient device id)
        comes from the request.
        """
        with self._lock:
            if claim_id in self._batch_claim_session_bindings:
                raise BatchClaimSessionError(BATCH_SESSION_DUPLICATE)

            batch = self._prekey_batch_claims.get(claim_id)
            if batch is None:
                # A single-claim id (bound or not) is not a batch claim; the
                # service maps both outcomes to 404/409 with field=claim_id.
                reason = (BATCH_SESSION_CLAIM_WRONG_KIND
                          if claim_id in self._prekey_claims
                          else BATCH_SESSION_CLAIM_UNKNOWN)
                raise BatchClaimSessionError(reason)

            frozen_entries = batch.devices
            initiator = self._find_device(initiator_device_id)
            if initiator is None:
                raise BatchClaimSessionError(BATCH_SESSION_INITIATOR_UNKNOWN)
            # Snapshot membership is a request-shape error (400) and wins
            # over the device's own revoked status: an initiator that is one
            # of the claimed devices is rejected as a whole batch regardless
            # of whether that recipient device was since revoked.
            if any(entry.device_id == initiator_device_id
                   for entry in frozen_entries):
                raise BatchClaimSessionError(
                    BATCH_SESSION_INITIATOR_IN_SNAPSHOT)
            if initiator.revoked:
                raise BatchClaimSessionError(BATCH_SESSION_INITIATOR_REVOKED)

            # The request's device set must equal the claim snapshot: an
            # extra request device names it in detail (the service turns it
            # into the array-item field path), a missing one leaves detail
            # empty (the service names the whole ephemeral_keys array).
            requested = [device_id for device_id, _key in ephemeral_keys]
            snapshot_ids = [entry.device_id for entry in frozen_entries]
            extra = next((device_id for device_id in requested
                          if device_id not in snapshot_ids), "")
            if extra or set(requested) != set(snapshot_ids):
                raise BatchClaimSessionError(
                    BATCH_SESSION_DEVICE_SET_MISMATCH, extra)
            ephemeral_by_device = dict(ephemeral_keys)

            # Validate every recipient and its claimed pre-key first, in the
            # claim's frozen order, before building or inserting anything, so
            # one failure leaves the whole batch unwritten.
            for entry in frozen_entries:
                recipient = self._find_device(entry.device_id)
                if recipient is None or recipient.revoked:
                    raise BatchClaimSessionError(
                        BATCH_SESSION_RECIPIENT_REVOKED, entry.device_id)
                used_prekey = next((pk for pk in recipient.prekeys
                                    if pk.key_id == entry.key_id), None)
                if used_prekey is None or used_prekey.revoked:
                    raise BatchClaimSessionError(
                        BATCH_SESSION_PREKEY_REVOKED, entry.key_id)

            sessions: List[Tuple[BatchClaimSessionEntry, Session]] = []
            for entry in frozen_entries:
                session = Session(
                    session_id=uuid.uuid4().hex,
                    initiator_device_id=initiator_device_id,
                    recipient_device_id=entry.device_id,
                    prekey_id=entry.key_id,
                    ephemeral_key=ephemeral_by_device[entry.device_id],
                    identity_key=entry.identity_key,
                    public_key=entry.public_key)
                sessions.append((entry, session))

            binding_entries: List[BatchClaimSessionEntry] = []
            for entry, session in sessions:
                self._sessions[session.session_id] = session
                binding_entries.append(BatchClaimSessionEntry(
                    recipient_device_id=entry.device_id,
                    prekey_id=entry.key_id,
                    identity_key=entry.identity_key,
                    public_key=entry.public_key,
                    session_id=session.session_id))
            binding = BatchClaimSessionBinding(
                claim_id=claim_id,
                initiator_device_id=initiator_device_id,
                entries=binding_entries)
            self._batch_claim_session_bindings[claim_id] = binding
            self._notify_change()
            return binding

    def session_view(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Return the immutable eight-field snapshot of a session, or ``None``.

        The values are copied under the lock; session records are never
        mutated after creation, so revocations cannot change this view.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            return {
                "session_id": session.session_id,
                "initiator_device_id": session.initiator_device_id,
                "recipient_device_id": session.recipient_device_id,
                "prekey_id": session.prekey_id,
                "ephemeral_key": session.ephemeral_key,
                "identity_key": session.identity_key,
                "public_key": session.public_key,
                "created_at": session.created_at,
            }

    # -- groups ------------------------------------------------------------

    @staticmethod
    def group_view(group: Group) -> Dict[str, Any]:
        """Copy one group into its public five-field view."""
        return {
            "group_id": group.group_id,
            "revision": group.revision,
            "members": list(group.members),
            "created_at": group.created_at,
        }

    def create_group(self, group_id: str, creator_device_id: str,
                     member_device_ids: List[str]) -> Group:
        """Atomically validate and create one group.

        The id must be unused and the creator an active registered device.
        Member ids are arbitrary non-empty strings already validated by the
        service; they need not resolve to registered devices and are stored
        as given. The creator is always the first member; the other ids
        follow in request order, de-duplicated (a repeated id kept once).
        The creator check and the insert happen under the store lock, the
        same lock device revocations take, so a concurrent revocation is
        linearized either wholly before this call (then it fails) or wholly
        after it (then the group is retained). On failure nothing is written.
        """
        with self._lock:
            if group_id in self._groups:
                raise GroupError(GROUP_DUPLICATE_ID)

            creator = self._active_device(creator_device_id)
            if creator is None:
                existing = self._find_device(creator_device_id)
                raise GroupError(GROUP_CREATOR_REVOKED if existing is not None
                                 else GROUP_CREATOR_UNKNOWN)

            members: List[str] = [creator_device_id]
            seen = {creator_device_id}
            for device_id in member_device_ids:
                if device_id not in seen:
                    seen.add(device_id)
                    members.append(device_id)

            group = Group(group_id=group_id,
                          creator_device_id=creator_device_id, members=members)
            self._groups[group_id] = group
            self._notify_change()
            return group

    def get_group(self, group_id: str) -> Optional[Group]:
        """Return the stored group, or ``None`` when the id is unknown."""
        with self._lock:
            return self._groups.get(group_id)

    def _find_device(self, device_id: str) -> Optional[Device]:
        """Resolve a globally-addressed device (must hold the lock)."""
        key = self._device_index.get(device_id)
        return self._devices.get(key) if key is not None else None

    def _active_device(self, device_id: str) -> Optional[Device]:
        """Resolve a device that exists and is not revoked (must hold lock)."""
        device = self._find_device(device_id)
        if device is None or device.revoked:
            return None
        return device

    def _authorize_group_actor(self, group_id: str,
                               actor_device_id: str) -> Group:
        """Resolve a group and authorize its creator (must hold the lock).

        Unknown group -> ``group_unknown``; an unknown or revoked actor ->
        ``actor_unknown`` / ``actor_revoked``; an active non-creator ->
        ``actor_not_creator``.
        """
        group = self._groups.get(group_id)
        if group is None:
            raise GroupError(GROUP_UNKNOWN)
        actor = self._find_device(actor_device_id)
        if actor is None:
            raise GroupError(GROUP_ACTOR_UNKNOWN)
        if actor.revoked:
            raise GroupError(GROUP_ACTOR_REVOKED)
        if actor_device_id != group.creator_device_id:
            raise GroupError(GROUP_ACTOR_NOT_CREATOR)
        return group

    def add_group_member(self, group_id: str, actor_device_id: str,
                         device_id: str) -> Tuple[Group, bool]:
        """Atomically authorize the creator and add one member.

        Returns ``(group, created)``: ``created`` is False (200) when the
        active device is already a member, True (201) when it was appended.
        An unknown or revoked target device raises ``group_device_unknown``
        / ``group_device_revoked`` (404 / 409, field ``device_id``). A
        successful append advances the group revision; an idempotent repeat
        does not.
        """
        with self._lock:
            group = self._authorize_group_actor(group_id, actor_device_id)
            if device_id in group.members:
                return group, False
            target = self._find_device(device_id)
            if target is None:
                raise GroupError(GROUP_DEVICE_UNKNOWN)
            if target.revoked:
                raise GroupError(GROUP_DEVICE_REVOKED)
            group.members.append(device_id)
            group.revision += 1
            self._notify_change()
            return group, True

    def remove_group_member(self, group_id: str, actor_device_id: str,
                            device_id: str) -> Group:
        """Atomically authorize the creator and remove one member.

        Always succeeds with 200 semantics: a target id that is not (or no
        longer) on the member list is an idempotent no-op leaving the
        revision untouched, and a revoked device may still be removed so no
        revoked id lingers on the roster. The one hard failure besides the
        shared authorization checks is an id that has never been a
        registered device (``group_device_unknown`` -> 404). The creator
        cannot leave their own group; that request is a no-op that keeps
        the creator as the first member.
        """
        with self._lock:
            group = self._authorize_group_actor(group_id, actor_device_id)
            if device_id == group.creator_device_id:
                return group
            if self._find_device(device_id) is None:
                raise GroupError(GROUP_DEVICE_UNKNOWN)
            if device_id in group.members:
                group.members.remove(device_id)
                group.revision += 1
                self._notify_change()
            return group

    def group_session_view(self, session: GroupSession) -> Dict[str, Any]:
        """Copy one group session into its public seven-field view."""
        return {
            "session_id": session.session_id,
            "group_id": session.group_id,
            "initiator_device_id": session.initiator_device_id,
            "ephemeral_key": session.ephemeral_key,
            "revision": session.revision,
            "members": list(session.members),
            "created_at": session.created_at,
        }

    def create_group_session(self, group_id: str, initiator_device_id: str,
                             ephemeral_key: str) -> GroupSession:
        """Atomically validate and create one group session.

        Every request creates a fresh session (a new ``session_id``);
        repeated POSTs are never deduplicated. The member list and group
        revision are frozen at creation, so later membership changes cannot
        alter the snapshot. The initiator must be an active *current* member
        of the group. On failure nothing is written.
        """
        with self._lock:
            group = self._groups.get(group_id)
            if group is None:
                raise GroupError(GROUP_SESSION_GROUP_UNKNOWN)
            initiator = self._find_device(initiator_device_id)
            if initiator is None:
                raise GroupError(GROUP_SESSION_INITIATOR_UNKNOWN)
            if initiator.revoked:
                raise GroupError(GROUP_SESSION_INITIATOR_INACTIVE)
            if initiator_device_id not in group.members:
                raise GroupError(GROUP_SESSION_INITIATOR_NOT_MEMBER)

            session = GroupSession(
                session_id=uuid.uuid4().hex,
                group_id=group_id,
                initiator_device_id=initiator_device_id,
                ephemeral_key=ephemeral_key,
                members=list(group.members),
                revision=group.revision,
            )
            self._group_sessions[session.session_id] = session
            self._notify_change()
            return session

    def get_group_session(self, session_id: str) -> Optional[GroupSession]:
        """Return the stored group session, or ``None`` when the id unknown."""
        with self._lock:
            return self._group_sessions.get(session_id)

    def group_session_member(self, session_id: str,
                             device_id: str) -> Optional[GroupSession]:
        """Resolve a group session and check frozen membership (holds lock).

        Returns the session when *device_id* is a frozen member, ``None``
        otherwise — including when the session does not exist. Membership is
        tested against the frozen snapshot, not the group's current roster.
        """
        session = self._group_sessions.get(session_id)
        if session is None or device_id not in session.members:
            return None
        return session

    # -- group-session rotation -------------------------------------------

    @staticmethod
    def rotation_view(session: GroupSession,
                      rotation: GroupSessionRotation) -> Dict[str, Any]:
        """Public view of a rotation successor: seven session fields plus
        ``rotation_id`` and ``predecessor_session_id``."""
        view = {
            "session_id": session.session_id,
            "group_id": session.group_id,
            "initiator_device_id": session.initiator_device_id,
            "ephemeral_key": session.ephemeral_key,
            "revision": session.revision,
            "members": list(session.members),
            "created_at": session.created_at,
            "rotation_id": rotation.rotation_id,
            "predecessor_session_id": rotation.predecessor_session_id,
        }
        return view

    def rotate_group_session(self, predecessor_session_id: str,
                             rotation_id: str, actor_device_id: str,
                             ephemeral_key: str, expected_revision: int
                             ) -> Tuple[GroupSession, GroupSessionRotation,
                                        bool]:
        """Atomically rotate one group session into a fresh frozen successor.

        Returns ``(successor, rotation, created)`` with ``created`` False for a
        replay of the same ``rotation_id`` on the same predecessor (200). The
        predecessor snapshot is never altered; the successor freezes the
        group's member list and revision at commit time and gets a fresh unique
        ``session_id``.

        Failure reasons (nothing written):

        * ``session_unknown`` — the predecessor is not a group session
          (404/field=session_id);
        * ``actor_unknown`` — the actor is not a registered device
          (404/field=actor_device_id);
        * ``actor_revoked`` / ``actor_not_creator`` — the actor is revoked or
          is not the group's creator (409/field=actor_device_id);
        * ``revision_mismatch`` — the group's current revision differs from
          ``expected_revision`` (409/field=expected_revision);
        * ``rotation_id_conflict`` — the id already rotated another
          predecessor (409/field=rotation_id);
        * ``predecessor_rotated`` — the predecessor already has a successor
          under a different id, so granting this would fork it
          (409/field=session_id).
        """
        with self._lock:
            predecessor = self._group_sessions.get(predecessor_session_id)
            if predecessor is None:
                raise GroupSessionRotationError(ROTATION_SESSION_UNKNOWN)

            existing = self._group_session_rotations.get(rotation_id)
            if existing is not None:
                if existing.predecessor_session_id == predecessor_session_id:
                    # Idempotent replay of the same id on the same predecessor:
                    # return the original successor even if the actor was since
                    # revoked or the group revision moved.
                    successor = self._group_sessions[
                        existing.successor_session_id]
                    return successor, existing, False
                # The id is already committed for another predecessor: the
                # rotation id namespace is global, so this is a 409 on
                # rotation_id regardless of the other request fields.
                raise GroupSessionRotationError(ROTATION_ID_CONFLICT)

            actor = self._find_device(actor_device_id)
            if actor is None:
                raise GroupSessionRotationError(ROTATION_ACTOR_UNKNOWN)
            if actor.revoked:
                raise GroupSessionRotationError(ROTATION_ACTOR_REVOKED)
            group = self._groups[predecessor.group_id]
            if actor_device_id != group.creator_device_id:
                raise GroupSessionRotationError(
                    ROTATION_ACTOR_NOT_CREATOR)

            if expected_revision != group.revision:
                raise GroupSessionRotationError(ROTATION_REVISION_MISMATCH)

            if predecessor_session_id in self._rotation_by_predecessor:
                # Another rotation id already succeeded for this predecessor:
                # never fork.
                raise GroupSessionRotationError(
                    ROTATION_PREDECESSOR_ROTATED)

            # Fresh unique session id (shared keyspace with 1:1 sessions).
            new_id = uuid.uuid4().hex
            while new_id in self._group_sessions or new_id in self._sessions:
                new_id = uuid.uuid4().hex

            successor = GroupSession(
                session_id=new_id,
                group_id=group.group_id,
                initiator_device_id=actor_device_id,
                ephemeral_key=ephemeral_key,
                members=list(group.members),
                revision=group.revision,
            )
            rotation = GroupSessionRotation(
                rotation_id=rotation_id,
                predecessor_session_id=predecessor_session_id,
                successor_session_id=new_id,
                group_id=group.group_id,
                actor_device_id=actor_device_id,
                revision=group.revision,
                members=list(group.members),
                created_at=successor.created_at,
            )
            self._group_sessions[new_id] = successor
            self._group_session_rotations[rotation_id] = rotation
            self._rotation_by_predecessor[predecessor_session_id] = rotation
            self._rotation_by_successor[new_id] = rotation
            self._notify_change()
            return successor, rotation, True

    # -- group-session sync ------------------------------------------------

    @staticmethod
    def _checkpoint_view(session_id: str, device_id: str,
                         record: GroupSyncCursor) -> Dict[str, Any]:
        """Copy one sync checkpoint into its public four-field view."""
        return {
            "session_id": session_id,
            "device_id": device_id,
            "cursor": record.cursor,
            "updated_at": record.updated_at,
        }

    def _authorize_sync_device(self, session_id: str, device_id: str
                               ) -> GroupSession:
        """Resolve a group session and authorize a frozen-member device.

        Must be called while holding the store lock. Distinguishes an unknown
        session (``session_unknown``) from device-side failures
        (``device_unknown`` / ``device_inactive`` / ``device_not_member``);
        the service maps all three device failures to 409/device_id.
        Membership is the frozen snapshot, so a removed member still syncs
        and a later-added member cannot.
        """
        session = self._group_sessions.get(session_id)
        if session is None:
            raise GroupSyncError(SYNC_SESSION_UNKNOWN)
        device = self._find_device(device_id)
        if device is None:
            raise GroupSyncError(SYNC_DEVICE_UNKNOWN)
        if device.revoked:
            raise GroupSyncError(SYNC_DEVICE_INACTIVE)
        if device_id not in session.members:
            raise GroupSyncError(SYNC_DEVICE_NOT_MEMBER)
        return session

    def group_sync_page(self, session_id: str, device_id: str,
                        after: Optional[int], limit: int) -> Dict[str, Any]:
        """Atomically read one ascending sync page, advancing the device cursor.

        When *after* is ``None`` the device's stored cursor is the starting
        point and the cursor advances to the page's last sequence (an empty
        page leaves it unchanged). An explicit *after* is a one-off query:
        the stored cursor is neither read nor moved.
        """
        with self._lock:
            self._authorize_sync_device(session_id, device_id)
            cursor_key = (session_id, device_id)
            record = self._group_sync_cursors.get(cursor_key)
            if after is None:
                start = record.cursor if record is not None else 0
                advance = True
            else:
                start = after
                advance = False
            stream = self._messages.get(session_id, [])
            page = [m for m in stream if m.sequence > start][:limit]
            next_cursor = page[-1].sequence if page else start
            has_more = any(m.sequence > next_cursor for m in stream)
            if advance and page and (record is None
                                     or next_cursor > record.cursor):
                if record is None:
                    self._group_sync_cursors[cursor_key] = \
                        GroupSyncCursor(cursor=next_cursor)
                else:
                    record.cursor = next_cursor
                    record.updated_at = utc_now_iso()
                self._notify_change()
            return {
                "messages": [self.message_view(m) for m in page],
                "next_cursor": next_cursor,
                "has_more": has_more,
            }

    def group_sync_checkpoint(self, session_id: str, device_id: str,
                              cursor: int
                              ) -> Tuple[Dict[str, Any], bool]:
        """Atomically move a device's sync cursor forward, or confirm equality.

        *cursor* must not exceed the session's largest stored sequence. A
        strictly greater cursor creates/advances the record (201, refreshed
        ``updated_at``); an equal cursor is an idempotent no-op (200, the
        timestamp is untouched); a smaller cursor conflicts (409/
        ``cursor``). The authorization checks run first: an unknown session
        raises ``session_unknown`` and an unknown/revoked/non-member device
        raises the matching device reason (all mapped to 409/device_id by the
        service). Returns ``(view, advanced)``.
        """
        with self._lock:
            session = self._authorize_sync_device(session_id, device_id)
            stream = self._messages.get(session_id, [])
            max_sequence = stream[-1].sequence if stream else 0
            if cursor > max_sequence:
                raise GroupSyncError(SYNC_CURSOR_CONFLICT)
            cursor_key = (session_id, device_id)
            record = self._group_sync_cursors.get(cursor_key)
            current = record.cursor if record is not None else 0
            if cursor < current:
                raise GroupSyncError(SYNC_CURSOR_CONFLICT)
            advanced = cursor > current
            if advanced:
                if record is None:
                    record = GroupSyncCursor(cursor=cursor)
                    self._group_sync_cursors[cursor_key] = record
                else:
                    record.cursor = cursor
                    record.updated_at = utc_now_iso()
                self._notify_change()
            if record is None:
                # No cursor has ever been advanced: the checkpoint at 0 is an
                # idempotent no-op anchored at the session's creation time.
                record = GroupSyncCursor(cursor=0,
                                         updated_at=session.created_at)
            return self._checkpoint_view(session_id, device_id, record), advanced

    # -- unified 1:1/group-session sync ------------------------------------

    @staticmethod
    def _message_sync_cursor_view(session_id: str, device_id: str,
                                  record: MessageSyncCursor) -> Dict[str, Any]:
        """Copy one unified sync checkpoint into its public four-field view."""
        return {
            "session_id": session_id,
            "device_id": device_id,
            "cursor": record.cursor,
            "updated_at": record.updated_at,
        }

    def _authorize_message_sync_device(self, session_id: str,
                                       device_id: str) -> str:
        """Resolve a 1:1/group session and authorize a reading device.

        Must be called while holding the store lock. Returns the anchor
        session's ``created_at`` (the timestamp a never-advanced zero
        checkpoint reports). An unknown session (neither a 1:1 nor a group
        session) raises ``session_unknown``; an unknown/revoked device raises
        ``device_unknown``/``device_inactive``. For a 1:1 session only its
        initiator or recipient may sync; for a group session only a device
        frozen into the member snapshot may (a removed member still syncs, a
        later-added member cannot) — both failures raise
        ``device_not_participant``.
        """
        session = self._sessions.get(session_id)
        if session is not None:
            device = self._find_device(device_id)
            if device is None:
                raise MessageSyncError(MESSAGE_SYNC_DEVICE_UNKNOWN)
            if device.revoked:
                raise MessageSyncError(MESSAGE_SYNC_DEVICE_INACTIVE)
            if device_id not in (session.initiator_device_id,
                                 session.recipient_device_id):
                raise MessageSyncError(MESSAGE_SYNC_DEVICE_NOT_PARTICIPANT)
            return session.created_at
        group_session = self._group_sessions.get(session_id)
        if group_session is None:
            raise MessageSyncError(MESSAGE_SYNC_SESSION_UNKNOWN)
        device = self._find_device(device_id)
        if device is None:
            raise MessageSyncError(MESSAGE_SYNC_DEVICE_UNKNOWN)
        if device.revoked:
            raise MessageSyncError(MESSAGE_SYNC_DEVICE_INACTIVE)
        if device_id not in group_session.members:
            raise MessageSyncError(MESSAGE_SYNC_DEVICE_NOT_PARTICIPANT)
        return group_session.created_at

    def message_sync_page(self, session_id: str, device_id: str,
                          after: Optional[int], limit: int) -> Dict[str, Any]:
        """Atomically read one ascending unified sync page, advancing the cursor.

        When *after* is ``None`` the device's stored cursor is the starting
        point and advances to the page's last sequence (an empty page leaves
        it unchanged). An explicit *after* is a one-off query: the stored
        cursor is neither read nor moved. Works for both 1:1 and group
        sessions; authorization is enforced under the same store lock.
        """
        with self._lock:
            self._authorize_message_sync_device(session_id, device_id)
            cursor_key = (session_id, device_id)
            record = self._message_sync_cursors.get(cursor_key)
            if after is None:
                start = record.cursor if record is not None else 0
                advance = True
            else:
                start = after
                advance = False
            stream = self._messages.get(session_id, [])
            page = [m for m in stream if m.sequence > start][:limit]
            next_cursor = page[-1].sequence if page else start
            has_more = any(m.sequence > next_cursor for m in stream)
            if advance and page and (record is None
                                     or next_cursor > record.cursor):
                if record is None:
                    self._message_sync_cursors[cursor_key] = \
                        MessageSyncCursor(cursor=next_cursor)
                else:
                    record.cursor = next_cursor
                    record.updated_at = utc_now_iso()
                self._notify_change()
            return {
                "messages": [self.message_view(m) for m in page],
                "next_cursor": next_cursor,
                "has_more": has_more,
            }

    def message_sync_checkpoint(self, session_id: str, device_id: str,
                                cursor: int
                                ) -> Tuple[Dict[str, Any], bool]:
        """Atomically move a device's unified sync cursor forward, or confirm it.

        *cursor* must not exceed the session's largest stored sequence. A
        strictly greater cursor creates/advances the record (201, refreshed
        ``updated_at``); an equal cursor is an idempotent no-op (200, the
        timestamp untouched); a smaller cursor conflicts (409/cursor). The
        authorization checks run first. Returns ``(view, advanced)``.
        """
        with self._lock:
            anchor_created_at = self._authorize_message_sync_device(
                session_id, device_id)
            stream = self._messages.get(session_id, [])
            max_sequence = stream[-1].sequence if stream else 0
            if cursor > max_sequence:
                raise MessageSyncError(MESSAGE_SYNC_CURSOR_CONFLICT)
            cursor_key = (session_id, device_id)
            record = self._message_sync_cursors.get(cursor_key)
            current = record.cursor if record is not None else 0
            if cursor < current:
                raise MessageSyncError(MESSAGE_SYNC_CURSOR_CONFLICT)
            advanced = cursor > current
            if advanced:
                if record is None:
                    record = MessageSyncCursor(cursor=cursor)
                    self._message_sync_cursors[cursor_key] = record
                else:
                    record.cursor = cursor
                    record.updated_at = utc_now_iso()
                self._notify_change()
            if record is None:
                # No cursor has ever been advanced: the checkpoint at 0 is an
                # idempotent no-op anchored at the session's creation time.
                record = MessageSyncCursor(cursor=0,
                                           updated_at=anchor_created_at)
            return self._message_sync_cursor_view(
                session_id, device_id, record), advanced

    def _authorize_sync_ack_device(self, session_id: str,
                                   device_id: str) -> str:
        """Resolve a session and authorize a batch sync-ack device.

        Must be called while holding the store lock. Returns the anchor
        session's ``created_at`` (the timestamp a never-advanced equal ack
        reports). An unknown session (neither a 1:1 nor a group session)
        raises ``session_unknown``; an unknown/revoked device raises
        ``device_unknown``/``device_inactive``. A 1:1 session may be acked by
        its recipient only (the initiator has no receive side there); a group
        session by a frozen member (a later-added member cannot, a removed
        member still can) — both raise ``device_not_participant``.
        """
        session = self._sessions.get(session_id)
        if session is not None:
            device = self._find_device(device_id)
            if device is None:
                raise MessageSyncError(MESSAGE_SYNC_DEVICE_UNKNOWN)
            if device.revoked:
                raise MessageSyncError(MESSAGE_SYNC_DEVICE_INACTIVE)
            if device_id != session.recipient_device_id:
                raise MessageSyncError(MESSAGE_SYNC_DEVICE_NOT_PARTICIPANT)
            return session.created_at
        group_session = self._group_sessions.get(session_id)
        if group_session is None:
            raise MessageSyncError(MESSAGE_SYNC_SESSION_UNKNOWN)
        device = self._find_device(device_id)
        if device is None:
            raise MessageSyncError(MESSAGE_SYNC_DEVICE_UNKNOWN)
        if device.revoked:
            raise MessageSyncError(MESSAGE_SYNC_DEVICE_INACTIVE)
        if device_id not in group_session.members:
            raise MessageSyncError(MESSAGE_SYNC_DEVICE_NOT_PARTICIPANT)
        return group_session.created_at

    def _plan_sync_ack_locked(self, session_id: str, device_id: str,
                               cursor: int) -> Dict[str, Any]:
        """Authorize and validate one sync-ack while holding the store lock.

        Performs every check without mutating state, so a batch can validate
        all of its items in array order before applying any of them. Returns a
        plan consumed by :meth:`_commit_sync_ack_plans_locked`. Raises
        :class:`MessageSyncError` on an unknown session, an
        unknown/revoked/unauthorized device, or a cursor below the stored
        cursor (0 when none) or above the session's largest sequence.
        """
        anchor_created_at = self._authorize_sync_ack_device(
            session_id, device_id)
        stream = self._messages.get(session_id, [])
        max_sequence = stream[-1].sequence if stream else 0
        if cursor > max_sequence:
            raise MessageSyncError(MESSAGE_SYNC_CURSOR_CONFLICT)
        cursor_key = (session_id, device_id)
        record = self._message_sync_cursors.get(cursor_key)
        current = record.cursor if record is not None else 0
        if cursor < current:
            raise MessageSyncError(MESSAGE_SYNC_CURSOR_CONFLICT)
        return {
            "session_id": session_id,
            "device_id": device_id,
            "cursor": cursor,
            "current": current,
            "record": record,
            "stream": stream,
            "is_group": self._group_sessions.get(session_id) is not None,
            "anchor_created_at": anchor_created_at,
            "advanced": cursor > current,
        }

    def _commit_sync_ack_plans_locked(self, plans: List[Dict[str, Any]],
                                      timestamp: str) -> List[MessageSyncCursor]:
        """Apply already-validated sync-ack plans in order under the lock.

        Every advancing plan shares the single *timestamp*. Each forward move
        marks the receivable messages in ``(current, cursor]`` acked one by one
        (a 1:1 session via the recipient's ``delivery`` record, a group session
        via the per-device ``group_delivery`` record while skipping the
        device's own messages), leaving ``attempts``/attempt ids untouched, and
        creates or advances the cursor record. An equal plan writes nothing and
        keeps its timestamp (a never-written cursor reports the anchor
        ``created_at``). No persistence notification is emitted; the caller
        notifies exactly once for the whole batch. Returns the stored (or
        transient) record per plan, in order.
        """
        records: List[MessageSyncCursor] = []
        for plan in plans:
            session_id = plan["session_id"]
            device_id = plan["device_id"]
            cursor = plan["cursor"]
            current = plan["current"]
            record = plan["record"]
            if plan["advanced"]:
                if plan["is_group"]:
                    for message in plan["stream"]:
                        if not (current < message.sequence <= cursor):
                            continue
                        # A device never acknowledges its own outgoing group
                        # message; every other frozen member holds a record.
                        if message.sender_device_id == device_id:
                            continue
                        key = (session_id, message.message_id, device_id)
                        state = self._group_delivery.get(key)
                        if state is None:
                            state = MessageDelivery()
                            self._group_delivery[key] = state
                        if not state.acked:
                            state.acked = True
                            state.ack_sequence = message.sequence
                else:
                    for message in plan["stream"]:
                        if not (current < message.sequence <= cursor):
                            continue
                        key = (session_id, message.message_id)
                        state = self._delivery.get(key)
                        if state is None:
                            state = MessageDelivery()
                            self._delivery[key] = state
                        if not state.acked:
                            state.acked = True
                            state.ack_sequence = message.sequence
                if record is None:
                    record = MessageSyncCursor(cursor=cursor,
                                               updated_at=timestamp)
                    self._message_sync_cursors[(session_id, device_id)] = record
                else:
                    record.cursor = cursor
                    record.updated_at = timestamp
            elif record is None:
                # An equal ack on a device that never advanced a cursor is a
                # state-free no-op anchored at the session's creation time.
                record = MessageSyncCursor(cursor=0,
                                           updated_at=plan["anchor_created_at"])
            records.append(record)
        return records

    def message_sync_ack(self, session_id: str, device_id: str,
                         cursor: int) -> Tuple[Dict[str, Any], bool]:
        """Atomically batch-acknowledge receivable messages and advance the cursor.

        The delivery acknowledgements and the ``message_sync_cursors`` advance
        commit as one locked transaction (one persistence notification).
        *cursor* must sit between the device's stored cursor (0 when none)
        and the session's largest stored sequence; a smaller or larger cursor
        raises ``cursor_conflict`` (409/cursor). On a forward move every
        receivable message with ``stored_cursor < sequence <= cursor`` is
        marked acked one by one: for a 1:1 session the recipient's delivery
        record (``delivery``) of each message is acked (created on demand,
        ``attempts``/attempt ids untouched); for a group session the per-device
        ``group_delivery`` record is acked for every such message except ones
        the acking device itself sent. The cursor then advances to *cursor*
        with a fresh ``updated_at`` (201). An equal cursor acks nothing, writes
        nothing and consumes no generation (200, timestamp unchanged); with no
        cursor record ever written its ``updated_at`` is the session's
        ``created_at``. Returns ``(view, advanced)``.
        """
        with self._lock:
            plan = self._plan_sync_ack_locked(session_id, device_id, cursor)
            record = self._commit_sync_ack_plans_locked(
                [plan], utc_now_iso())[0]
            if plan["advanced"]:
                # Acks and the cursor commit together: exactly one persistence
                # notification, so a durable write failure rolls back all the
                # delivery flags, the cursor and its timestamp as one.
                self._notify_change()
            return self._message_sync_cursor_view(
                session_id, device_id, record), plan["advanced"]

    def message_sync_ack_batch(
            self, device_id: str, items: List[Tuple[str, int]]
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """Apply many 1:1-session sync-acks for one device as one transaction.

        *items* are ``(session_id, cursor)`` pairs already validated for shape
        and unique sessions by the service. The device must exist and not be
        revoked (a batch-level ``device_unknown``/``device_inactive``
        :class:`MessageSyncError`). Items are then validated in array order;
        the first failure raises :class:`MessageSyncAckBatchError` carrying
        that item's index and the batch writes nothing:

        * a session that is neither a 1:1 nor a group session ->
          ``session_unknown`` (404);
        * a group session, or a 1:1 session whose recipient is not this device
          -> ``session_not_recipient`` (409);
        * a cursor below the stored cursor (0 when none) or above the
          session's largest sequence -> ``cursor_conflict`` (409).

        On success every advancing item shares one UTC timestamp and all acks,
        cursor advances and the single persistence notification commit under
        the one store lock (so a durable write failure rolls the whole batch
        back); equal items write nothing. Returns ``(results, any_advanced)``
        with one three-field (``session_id``/``cursor``/``updated_at``) result
        per item, in input order.
        """
        with self._lock:
            device = self._find_device(device_id)
            if device is None:
                raise MessageSyncError(MESSAGE_SYNC_DEVICE_UNKNOWN)
            if device.revoked:
                raise MessageSyncError(MESSAGE_SYNC_DEVICE_INACTIVE)
            plans: List[Dict[str, Any]] = []
            for index, (session_id, cursor) in enumerate(items):
                session = self._sessions.get(session_id)
                if session is None:
                    reason = MESSAGE_SYNC_DEVICE_NOT_PARTICIPANT \
                        if self._group_sessions.get(session_id) is not None \
                        else MESSAGE_SYNC_SESSION_UNKNOWN
                    raise MessageSyncAckBatchError(reason, index)
                if device_id != session.recipient_device_id:
                    raise MessageSyncAckBatchError(
                        MESSAGE_SYNC_DEVICE_NOT_PARTICIPANT, index)
                stream = self._messages.get(session_id, [])
                max_sequence = stream[-1].sequence if stream else 0
                if cursor > max_sequence:
                    raise MessageSyncAckBatchError(
                        MESSAGE_SYNC_CURSOR_CONFLICT, index)
                cursor_key = (session_id, device_id)
                record = self._message_sync_cursors.get(cursor_key)
                current = record.cursor if record is not None else 0
                if cursor < current:
                    raise MessageSyncAckBatchError(
                        MESSAGE_SYNC_CURSOR_CONFLICT, index)
                # Validate in array order; only after every item passes are
                # any of them committed below.
                plans.append({
                    "session_id": session_id,
                    "device_id": device_id,
                    "cursor": cursor,
                    "current": current,
                    "record": record,
                    "stream": stream,
                    "is_group": False,
                    "anchor_created_at": session.created_at,
                    "advanced": cursor > current,
                })
            any_advanced = any(plan["advanced"] for plan in plans)
            records = self._commit_sync_ack_plans_locked(
                plans, utc_now_iso())
            if any_advanced:
                # One persistence notification for the whole batch: all the
                # per-session acks and cursors commit (or roll back) together
                # and the generation advances at most once.
                self._notify_change()
            results = [{
                "session_id": plan["session_id"],
                "cursor": record.cursor,
                "updated_at": record.updated_at,
            } for plan, record in zip(plans, records)]
            return results, any_advanced

    def _inbox_entries_locked(
            self, device_id: str
    ) -> List[Tuple[str, str, Message]]:
        """Collect a device's unacked 1:1-inbox messages in inbox order.

        Mirrors :meth:`device_inbox`: every not-yet-acked message of every
        1:1 session whose ``recipient_device_id`` is *device_id*, sorted by
        ``(session.created_at, session_id, sequence)`` (session_id by code
        points). Group sessions and sessions addressed to other devices
        never contribute. The caller must hold the store lock.
        """
        entries: List[Tuple[str, str, Message]] = []
        for session_id, session in self._sessions.items():
            if session.recipient_device_id != device_id:
                continue
            for message in self._messages.get(session_id, []):
                state = self._delivery.get((session_id, message.message_id))
                if state is not None and state.acked:
                    continue
                entries.append((session.created_at, session_id, message))
        entries.sort(key=lambda entry: (entry[0], entry[1],
                                        entry[2].sequence))
        return entries

    @staticmethod
    def _lease_effective_deadline_locked(
            lease: MessageLease) -> Optional[datetime]:
        """The lease's current effective deadline as an aware ``datetime``.

        Initially the claim ``leased_until``; after one or more renewals it
        is the last renewal's ``leased_until`` (each renewal extends the
        previous deadline by exactly :data:`INBOX_LEASE_SECONDS` seconds).
        Returns ``None`` for an unparseable or naive (legacy/tampered)
        value, which the caller treats as expired.
        """
        raw = lease.renewals[-1].leased_until if lease.renewals \
            else lease.leased_until
        try:
            deadline = datetime.fromisoformat(raw)
        except (TypeError, ValueError):
            return None
        if deadline.tzinfo is None:
            return None
        return deadline

    @staticmethod
    def _lease_is_active_locked(lease: MessageLease,
                                now: datetime) -> bool:
        """Whether *lease* has not passed its deadline at *now*.

        A released lease is never active again, whatever its deadline says;
        neither is a completed one — completion ends the lease's lifecycle,
        so its still-unacked messages become claimable again and no further
        renewal or release can be applied. Deadlines written by this server
        always parse as timezone-aware ISO timestamps; an unparseable
        (legacy/tampered) value is treated as expired rather than
        withholding the message forever.
        """
        if lease.released_at is not None:
            return False
        if lease.completion is not None:
            return False
        deadline = DeviceStore._lease_effective_deadline_locked(lease)
        if deadline is None:
            return False
        return deadline > now

    def _has_active_lease_locked(self, state: MessageDelivery,
                                 now: datetime) -> bool:
        """Whether *state* still holds at least one unexpired lease."""
        return any(self._lease_is_active_locked(lease, now)
                   for lease in state.leases)

    def _find_inbox_lease_locked(
            self, lease_id: str
    ) -> Optional[Tuple[str, int, str, Optional[str], List[Tuple[str, str]]]]:
        """Look up an occupied inbox lease across the 1:1 delivery records.

        Returns ``(device_id, limit, leased_until, released_at, keys)`` where
        ``keys`` are the ``(session_id, message_id)`` pairs the lease was
        taken on, in inbox order — or ``None`` when the id has never been
        committed. Restore validation guarantees one id always binds one
        device, limit, deadline and release timestamp, so the first record
        found is authoritative.
        """
        owner = ""
        owner_limit = 0
        leased_until = ""
        released_at: Optional[str] = None
        keys: List[Tuple[str, str]] = []
        for (session_id, message_id), state in self._delivery.items():
            for lease in state.leases:
                if lease.lease_id != lease_id:
                    continue
                if not keys:
                    session = self._sessions.get(session_id)
                    owner = session.recipient_device_id if session \
                        is not None else ""
                    owner_limit = lease.limit
                    leased_until = lease.leased_until
                    released_at = lease.released_at
                keys.append((session_id, message_id))
        if not keys:
            return None
        # Re-present the leased messages in the same inbox order the original
        # claim selected them, so a replayed response keeps its item order.
        ordered = sorted(
            keys,
            key=lambda key: (self._sessions[key[0]].created_at, key[0],
                             next(m.sequence for m in self._messages[key[0]]
                                  if m.message_id == key[1])))
        return (owner, owner_limit, leased_until, released_at, ordered)

    def device_inbox(self, device_id: str, limit: int) -> Dict[str, Any]:
        """Atomically read one device's aggregated 1:1 offline inbox.

        Purely read-only: nothing is created, advanced or persisted, so an
        unchanged state answers byte-identically and consumes no commit
        generation. The whole snapshot is taken under the store lock (the
        same lock message submission, sync-ack and revocation take), so the
        page is linearized against them. The device must exist and not be
        revoked (both mapped to 409/device_id by the service).

        The inbox aggregates, across every **1:1** session whose
        ``recipient_device_id`` is this device (group sessions never
        contribute, and sessions addressed to other devices of the same user
        never leak in), every message whose ``delivery`` record is missing or
        not yet acked. Entries are ordered by ``(session.created_at,
        session_id, sequence)`` — the session_id compared by code points —
        and the page holds at most *limit* of them; ``has_more`` says whether
        the locked snapshot still had further entries beyond the page.
        """
        with self._lock:
            device = self._find_device(device_id)
            if device is None:
                raise MessageSyncError(MESSAGE_SYNC_DEVICE_UNKNOWN)
            if device.revoked:
                raise MessageSyncError(MESSAGE_SYNC_DEVICE_INACTIVE)
            entries = self._inbox_entries_locked(device_id)
            page = entries[:limit]
            return {
                "device_id": device_id,
                "messages": [self.message_view(entry[2]) for entry in page],
                "has_more": len(entries) > limit,
            }

    def device_inbox_wait(self, device_id: str, limit: int,
                          timeout_ms: int) -> Dict[str, Any]:
        """Long-polling variant of :meth:`device_inbox` (read-only).

        ``GET /v1/devices/{device_id}/inbox/wait``. Under the store lock, if
        the device already has at least one unacked 1:1 message, the first
        *limit* entries are returned immediately, exactly as
        :meth:`device_inbox` would. Otherwise the caller waits — **without**
        holding the lock, so message submission, ack and revocation stay
        unblocked — on the store condition until a committed mutation makes a
        message deliverable, the device is revoked, or *timeout_ms*
        milliseconds (measured against a monotonic clock) elapse. After every
        wakeup and once more at the deadline the state is rechecked under the
        lock, revocation taking priority over message delivery. Only a
        still-empty snapshot at the deadline answers ``200`` with
        ``messages`` empty and ``has_more`` false. Nothing is written: no
        persistence notification is produced by the wait itself, no
        ``commit_seq`` generation is consumed, and no cursor, lease, attempt
        count or sidecar is touched. The device must exist and not be revoked
        (both mapped to 409/device_id by the service).
        """
        deadline = time.monotonic() + timeout_ms / 1000.0
        with self._condition:
            while True:
                device = self._find_device(device_id)
                if device is None:
                    raise MessageSyncError(MESSAGE_SYNC_DEVICE_UNKNOWN)
                if device.revoked:
                    raise MessageSyncError(MESSAGE_SYNC_DEVICE_INACTIVE)
                entries = self._inbox_entries_locked(device_id)
                if entries:
                    page = entries[:limit]
                    return {
                        "device_id": device_id,
                        "messages": [self.message_view(entry[2])
                                     for entry in page],
                        "has_more": len(entries) > limit,
                    }
                # Nothing deliverable yet: release the lock and block until a
                # committed mutation notifies, then recheck under the lock.
                # wait_for needs a predicate; wait() with the remaining budget
                # gives the explicit deadline recheck the spec calls for. A
                # non-positive remaining budget is 0.0 (return immediately).
                remaining = max(deadline - time.monotonic(), 0.0)
                if remaining <= 0.0:
                    return {"device_id": device_id, "messages": [],
                            "has_more": False}
                self._condition.wait(remaining)

    def inbox_claim(
            self, device_id: str, lease_id: str, limit: int
    ) -> Tuple[Dict[str, Any], int, bool]:
        """Lease up to *limit* unacked 1:1-inbox messages for redelivery.

        ``POST /v1/devices/{device_id}/inbox/claim``. *lease_id* is a
        client-chosen non-empty string and *limit* a non-boolean integer in
        1..100, both already validated by the service. Under the one store
        lock (shared with message submission, ack, retry and revocation),
        the device is resolved (unknown/revoked -> :class:`InboxLeaseError`
        ``device_unknown``/``device_inactive``) and the inbox is scanned in
        its fixed ``(session.created_at, session_id, sequence)`` order: the
        first *limit* messages that carry no still-active lease are leased
        for :data:`INBOX_LEASE_SECONDS` seconds, the same lease id/deadline
        being recorded on every leased message's ``delivery`` record.

        An occupied *lease_id* is idempotent only for the same device and
        limit: that exact replay returns the first response with status 200
        and writes nothing; the id on another device, or with another limit,
        raises ``lease_id_conflict`` (409/lease_id). A claim that leases at
        least one message notifies persistence once (201, commit_seq + 1);
        an empty selection returns 200 with ``leased_until`` null and writes
        nothing, so it occupies no generation. An expired lease never blocks
        a later claim with a new id. Returns ``(body, status_code,
        any_leased)``.
        """
        with self._lock:
            now = datetime.now(timezone.utc)

            existing = self._find_inbox_lease_locked(lease_id)
            if existing is not None:
                owner, owner_limit, leased_until, _released, leased_keys = \
                    existing
                # A cross-device replay or a changed limit is a conflict
                # regardless of the path device's current state; an exact
                # replay returns the first response (200) even if the device
                # has since been revoked, mirroring the other idempotent
                # claims.
                if owner != device_id or owner_limit != limit:
                    raise InboxLeaseError(INBOX_LEASE_CONFLICT)
                # Exact replay: rebuild the first response from the messages
                # the id was originally taken on (in their original inbox
                # order), byte-identically, even if they were acked or the
                # deadline has since passed.
                messages: List[Dict[str, Any]] = []
                for replay_session_id, replay_message_id in leased_keys:
                    message = next(m for m in self._messages[replay_session_id]
                                   if m.message_id == replay_message_id)
                    messages.append(self.message_view(message))
                body = {
                    "device_id": device_id,
                    "lease_id": lease_id,
                    "leased_until": leased_until if messages else None,
                    "messages": messages,
                }
                return body, 200, False

            device = self._find_device(device_id)
            if device is None:
                raise InboxLeaseError(INBOX_LEASE_DEVICE_UNKNOWN)
            if device.revoked:
                raise InboxLeaseError(INBOX_LEASE_DEVICE_INACTIVE)

            picked: List[Tuple[Tuple[str, str], Message]] = []
            for _, session_id, message in self._inbox_entries_locked(
                    device_id):
                key = (session_id, message.message_id)
                state = self._delivery.get(key)
                if state is not None \
                        and self._has_active_lease_locked(state, now):
                    continue
                picked.append((key, message))
                if len(picked) >= limit:
                    break

            if not picked:
                # Empty set: answer 200 with a null deadline and do not
                # occupy the lease id, write a delivery record, notify
                # persistence or advance commit_seq.
                return ({"device_id": device_id, "lease_id": lease_id,
                         "leased_until": None, "messages": []}, 200, False)

            # timespec="microseconds" always emits six fractional digits;
            # isoformat() on a whole-microsecond value would otherwise drop
            # the fraction entirely.
            leased_until = (now + timedelta(seconds=INBOX_LEASE_SECONDS)) \
                .isoformat(timespec="microseconds")
            for key, _message in picked:
                state = self._delivery.get(key)
                if state is None:
                    state = MessageDelivery()
                    self._delivery[key] = state
                state.leases.append(MessageLease(
                    lease_id=lease_id, limit=limit,
                    leased_until=leased_until))
            # One persistence notification for the whole lease set: every
            # per-message lease record commits (or rolls back) together and
            # the generation advances at most once.
            self._notify_change()
            body = {
                "device_id": device_id,
                "lease_id": lease_id,
                "leased_until": leased_until,
                "messages": [self.message_view(message)
                             for _key, message in picked],
            }
            return body, 201, True

    def inbox_release(
            self, device_id: str, lease_id: str
    ) -> Tuple[Dict[str, Any], int]:
        """Release one occupied 1:1-inbox lease before its deadline.

        ``POST /v1/devices/{device_id}/inbox/leases/{lease_id}/release``.
        Under the one store lock (shared with claims, acks, retries and
        revocation), the lease is resolved first: a never-committed
        *lease_id* raises ``lease_not_found`` (404/lease_id) and one owned
        by another device raises ``lease_id_conflict`` (409/lease_id),
        regardless of the path device's state. A lease that was already
        released is an exact replay: the first response is rebuilt from the
        persisted release timestamp with status 200 and nothing is written —
        even if the device has since been revoked. Only a first release
        resolves the device (unknown/revoked -> :class:`InboxLeaseError`
        ``device_unknown``/``device_inactive``, 409/device_id).

        The release stamps the same UTC ``released_at`` (six microsecond
        digits, ``+00:00``) onto every delivery record the lease was taken
        on; the lease stays as history but no longer withholds its messages
        from new claims, and replaying the original claim still returns the
        frozen claim response without reactivating anything. The whole
        release commits as one persistence notification (201, commit_seq +
        1); a durable write failure rolls every record back. Returns
        ``(body, status_code)`` with ``released_count`` the number of
        messages the lease had claimed.
        """
        with self._lock:
            existing = self._find_inbox_lease_locked(lease_id)
            if existing is None:
                raise InboxLeaseError(INBOX_LEASE_NOT_FOUND)
            owner, _limit, _deadline, released_at, keys = existing
            # A cross-device release is a conflict regardless of the path
            # device's current state, mirroring the claim replay rules.
            if owner != device_id:
                raise InboxLeaseError(INBOX_LEASE_CONFLICT)
            if released_at is not None:
                # Exact replay: the frozen first response, byte-identical,
                # even if the device has since been revoked.
                return ({"device_id": device_id, "lease_id": lease_id,
                         "released_at": released_at,
                         "released_count": len(keys)}, 200)
            device = self._find_device(device_id)
            if device is None:
                raise InboxLeaseError(INBOX_LEASE_DEVICE_UNKNOWN)
            if device.revoked:
                raise InboxLeaseError(INBOX_LEASE_DEVICE_INACTIVE)
            # A completed lease has ended its lifecycle: its completion is
            # its terminal record and a first release cannot follow it
            # (409/lease_id). Expiry alone does not block a release.
            completed = False
            for key in keys:
                state = self._delivery.get(key)
                if state is None:
                    continue
                if any(lease.lease_id == lease_id
                       and lease.completion is not None
                       for lease in state.leases):
                    completed = True
                    break
            if completed:
                raise InboxLeaseError(INBOX_LEASE_UNAVAILABLE)
            released_at = datetime.now(timezone.utc) \
                .isoformat(timespec="microseconds")
            for key in keys:
                state = self._delivery.get(key)
                if state is None:
                    continue
                for lease in state.leases:
                    if lease.lease_id == lease_id:
                        lease.released_at = released_at
            # One persistence notification for the whole lease set: every
            # per-message release commits (or rolls back) together and the
            # generation advances at most once.
            self._notify_change()
            return ({"device_id": device_id, "lease_id": lease_id,
                     "released_at": released_at,
                     "released_count": len(keys)}, 201)

    def inbox_lease_renew(
            self, device_id: str, lease_id: str, renewal_id: str
    ) -> Tuple[Dict[str, Any], int]:
        """Renew one occupied 1:1-inbox lease for another 30 seconds.

        ``POST /v1/devices/{device_id}/inbox/leases/{lease_id}/renew``.
        Under the one store lock (shared with claims, releases, acks,
        retries and revocation), the lease is resolved first: a
        never-committed *lease_id* raises ``lease_not_found``
        (404/lease_id) and one owned by another device raises
        ``lease_id_conflict`` (409/lease_id), regardless of the path
        device's state. A replay of the same *renewal_id* on the same
        lease is decided next and returns its frozen first response with
        status 200, writing nothing — even if the lease has since expired,
        been released or the device revoked; the same *renewal_id* on a
        different lease is allowed (ids only have to be unique within one
        lease).

        Only a first renewal resolves the device (unknown ->
        ``device_unknown``, revoked -> ``device_inactive``, both
        409/device_id) and the lease state: an already-released or
        expired lease raises ``lease_unavailable`` (409/lease_id). On
        success (201) the effective deadline — the claim ``leased_until``
        initially, the last renewal's afterwards — is extended by exactly
        :data:`INBOX_LEASE_SECONDS` seconds, and the same renewal record
        (``renewal_id``/new ``leased_until``) is appended to the lease on
        every delivery record it lives on. The renewal set commits as one
        persistence notification (commit_seq + 1); a durable write
        failure rolls every record back. Returns ``(body, status_code)``
        with keys ``device_id``, ``lease_id``, ``renewal_id``,
        ``leased_until`` in that order.
        """
        with self._lock:
            now = datetime.now(timezone.utc)
            # Gather every (delivery key, lease object) carrying the id;
            # restore validation guarantees the copies stay identical, so
            # any of them is authoritative for the replay lookup.
            hits: List[Tuple[Tuple[str, str], MessageLease]] = []
            for (session_id, message_id), state in self._delivery.items():
                for lease in state.leases:
                    if lease.lease_id == lease_id:
                        hits.append(((session_id, message_id), lease))
            if not hits:
                raise InboxLeaseError(INBOX_LEASE_NOT_FOUND)
            owner_session = self._sessions.get(hits[0][0][0])
            owner = owner_session.recipient_device_id \
                if owner_session is not None else ""
            # A cross-device renewal is a conflict regardless of the path
            # device's current state, mirroring claim/release replay rules.
            if owner != device_id:
                raise InboxLeaseError(INBOX_LEASE_CONFLICT)
            # Same renewal_id on this lease is an exact replay: rebuild the
            # first response (its frozen deadline) byte-identically. The id
            # is scoped to the lease, so it may recur on other leases.
            for renewal in hits[0][1].renewals:
                if renewal.renewal_id == renewal_id:
                    return ({"device_id": device_id,
                             "lease_id": lease_id,
                             "renewal_id": renewal_id,
                             "leased_until": renewal.leased_until}, 200)
            device = self._find_device(device_id)
            if device is None:
                raise InboxLeaseError(INBOX_LEASE_DEVICE_UNKNOWN)
            if device.revoked:
                raise InboxLeaseError(INBOX_LEASE_DEVICE_INACTIVE)
            # Every copy of the lease shares one lifecycle; once released
            # or past its current effective deadline it cannot be renewed.
            if any(not self._lease_is_active_locked(lease, now)
                   for _key, lease in hits):
                raise InboxLeaseError(INBOX_LEASE_UNAVAILABLE)
            current = self._lease_effective_deadline_locked(hits[0][1])
            # Active above guarantees a parseable aware deadline on every
            # copy; the copies are identical, so they all extend the same
            # value by exactly 30 seconds.
            if current is None:  # pragma: no cover - ruled out above
                raise InboxLeaseError(INBOX_LEASE_UNAVAILABLE)
            leased_until = (current + timedelta(
                seconds=INBOX_LEASE_SECONDS)) \
                .isoformat(timespec="microseconds")
            for _key, lease in hits:
                lease.renewals.append(MessageLeaseRenewal(
                    renewal_id=renewal_id, leased_until=leased_until))
            # One persistence notification for the whole lease set: every
            # per-message renewal commits (or rolls back) together and the
            # generation advances at most once.
            self._notify_change()
            return ({"device_id": device_id,
                     "lease_id": lease_id,
                     "renewal_id": renewal_id,
                     "leased_until": leased_until}, 201)

    def inbox_lease_complete(
            self, device_id: str, lease_id: str, completion_id: str,
            outcome: str
    ) -> Tuple[Dict[str, Any], int]:
        """Record the terminal completion of one 1:1-inbox lease.

        ``POST /v1/devices/{device_id}/inbox/leases/{lease_id}/complete``.
        Under the one store lock (shared with claims, releases, renewals,
        acks, retries and revocation), the lease is resolved first: a
        never-committed *lease_id* raises ``lease_not_found``
        (404/lease_id) and one owned by another device raises
        ``lease_id_conflict`` (409/lease_id), regardless of the path
        device's state. A replay of the same *completion_id* on the same
        lease is decided next and returns its frozen first response with
        status 200, writing nothing — even if the lease has since expired,
        been released/completed or the device revoked; the replay wins
        over every later state. The same *completion_id* on the same
        lease with the request carrying a different outcome (or simply
        reusing the id after a completion with another id already
        committed) raises ``completion_id_conflict`` (409/completion_id);
        the id is scoped to the lease and may recur on other leases.

        Only a first completion resolves the device (unknown ->
        ``device_unknown``, revoked -> ``device_inactive``, both
        409/device_id) and the lease state: an already-released or
        expired lease raises ``lease_unavailable`` (409/lease_id). On
        success (201) the same completion record
        (``completion_id``/``outcome``/``completed_at``) is stamped onto
        the lease on every delivery record it lives on. Completion ends
        the lease's lifecycle: it stops withholding its still-unacked
        messages from new claims (a ``delivered`` outcome is *not* an
        acknowledgement), and afterwards neither a renewal nor a first
        release can be applied. The completion set commits as one
        persistence notification (commit_seq + 1); a durable write
        failure rolls every record back. Returns ``(body, status_code)``
        with keys ``device_id``, ``lease_id``, ``completion_id``,
        ``outcome``, ``completed_at`` in that order.
        """
        with self._lock:
            # Gather every (delivery key, lease object) carrying the id;
            # restore validation guarantees the copies stay identical, so
            # any of them is authoritative for the replay lookup.
            hits: List[Tuple[Tuple[str, str], MessageLease]] = []
            for (session_id, message_id), state in self._delivery.items():
                for lease in state.leases:
                    if lease.lease_id == lease_id:
                        hits.append(((session_id, message_id), lease))
            if not hits:
                raise InboxLeaseError(INBOX_LEASE_NOT_FOUND)
            owner_session = self._sessions.get(hits[0][0][0])
            owner = owner_session.recipient_device_id \
                if owner_session is not None else ""
            # A cross-device completion is a conflict regardless of the
            # path device's current state, mirroring claim/release/renew.
            if owner != device_id:
                raise InboxLeaseError(INBOX_LEASE_CONFLICT)
            existing = hits[0][1].completion
            # The replay decision precedes every other state check: the
            # same completion_id on this lease answers its frozen first
            # response (200) byte-identically, however stale the request.
            if existing is not None:
                if existing.completion_id == completion_id:
                    return ({"device_id": device_id,
                             "lease_id": lease_id,
                             "completion_id": existing.completion_id,
                             "outcome": existing.outcome,
                             "completed_at": existing.completed_at}, 200)
                # The id already belongs to another completion, or the
                # same lease is being completed again under a new id.
                raise InboxLeaseError(INBOX_LEASE_COMPLETION_CONFLICT)
            device = self._find_device(device_id)
            if device is None:
                raise InboxLeaseError(INBOX_LEASE_DEVICE_UNKNOWN)
            if device.revoked:
                raise InboxLeaseError(INBOX_LEASE_DEVICE_INACTIVE)
            now = datetime.now(timezone.utc)
            # Every copy of the lease shares one lifecycle; once released
            # or past its current effective deadline it cannot complete.
            if any(not self._lease_is_active_locked(lease, now)
                   for _key, lease in hits):
                raise InboxLeaseError(INBOX_LEASE_UNAVAILABLE)
            completed_at = now.isoformat(timespec="microseconds")
            completion = MessageLeaseCompletion(
                completion_id=completion_id, outcome=outcome,
                completed_at=completed_at)
            for _key, lease in hits:
                lease.completion = completion
            # A lease taken by a redelivery-job dispatch moves its job to
            # the matching terminal state in the same locked transaction.
            self._redelivery_job_completed_locked(lease_id, outcome)
            # One persistence notification for the whole lease set: every
            # per-message completion commits (or rolls back) together and
            # the generation advances at most once.
            self._notify_change()
            return ({"device_id": device_id,
                     "lease_id": lease_id,
                     "completion_id": completion_id,
                     "outcome": outcome,
                     "completed_at": completed_at}, 201)

    def inbox_lease_ack(
            self, device_id: str, lease_id: str
    ) -> Tuple[Dict[str, Any], int]:
        """Bulk-acknowledge every message one delivered 1:1-inbox lease covers.

        ``POST /v1/devices/{device_id}/inbox/leases/{lease_id}/ack`` (no
        request body). Under the one store lock (shared with claims,
        releases, renewals, completions, the per-message acks, retries and
        revocation), the lease is resolved first: a never-committed
        *lease_id* raises ``lease_not_found`` (404/lease_id) and one owned
        by another device raises ``lease_id_conflict`` (409/lease_id),
        regardless of the path device's state.

        A lease whose messages are already all acked is an idempotent
        replay: it answers 200 and writes nothing — even if the device has
        since been revoked; that decision precedes the device-state check.
        Otherwise a lease without a completion, or one completed with an
        outcome other than ``delivered``, raises ``lease_not_delivered``
        (409/lease_id); only a ``completion.outcome == "delivered"`` lease
        may be bulk-acked, so an active, expired or released lease is
        rejected. A deliverable lease on an unknown or revoked device
        raises ``device_unknown`` / ``device_inactive`` (409/device_id).

        On a first ack (201) every delivery record the lease lives on is
        set ``acked=True`` with ``ack_sequence`` equal to the message's own
        sequence, while ``attempts`` and the attempt-id dedup set are left
        untouched. The whole set commits as one persistence notification
        (commit_seq + 1); a durable write failure rolls every record back.
        Returns ``(body, status_code)`` with keys ``device_id``,
        ``lease_id``, ``acked`` (always ``True``) and ``message_count``
        (the number of messages the lease claimed) in that order.
        """
        with self._lock:
            # Gather every (delivery key, lease object) carrying the id in
            # claim (inbox) order; restore validation guarantees the copies
            # stay identical, so the first lease seen is authoritative.
            hits: List[Tuple[Tuple[str, str], MessageLease, int]] = []
            for (session_id, message_id), state in self._delivery.items():
                for lease in state.leases:
                    if lease.lease_id != lease_id:
                        continue
                    sequence = next(m.sequence
                                    for m in self._messages[session_id]
                                    if m.message_id == message_id)
                    hits.append(((session_id, message_id), lease, sequence))
            if not hits:
                raise InboxLeaseError(INBOX_LEASE_NOT_FOUND)
            owner_session = self._sessions.get(hits[0][0][0])
            owner = owner_session.recipient_device_id \
                if owner_session is not None else ""
            # A cross-device ack is a conflict regardless of the path
            # device's current state, mirroring the other lease routes.
            if owner != device_id:
                raise InboxLeaseError(INBOX_LEASE_CONFLICT)
            hits.sort(key=lambda hit: (
                self._sessions[hit[0][0]].created_at, hit[0][0],
                hit[2]))
            message_count = len(hits)

            def response() -> Dict[str, Any]:
                return {"device_id": device_id, "lease_id": lease_id,
                        "acked": True, "message_count": message_count}

            # An already-fully-acked lease is an idempotent replay: 200, no
            # write, winning over a later device revocation — decided ahead
            # of the completion and device checks.
            if all(self._delivery[key].acked for key, _lease, _seq in hits):
                return response(), 200
            completion = hits[0][1].completion
            if completion is None or completion.outcome != "delivered":
                raise InboxLeaseError(INBOX_LEASE_NOT_DELIVERED)
            device = self._find_device(device_id)
            if device is None:
                raise InboxLeaseError(INBOX_LEASE_DEVICE_UNKNOWN)
            if device.revoked:
                raise InboxLeaseError(INBOX_LEASE_DEVICE_INACTIVE)
            for key, _lease, sequence in hits:
                state = self._delivery[key]
                state.acked = True
                state.ack_sequence = sequence
            # One persistence notification for the whole lease set: every
            # per-message acknowledgement commits (or rolls back) together
            # and the generation advances at most once.
            self._notify_change()
            return response(), 201

    def inbox_lease_get(
            self, device_id: str, lease_id: str
    ) -> Dict[str, Any]:
        """Read one occupied 1:1-inbox lease's current state (read-only).

        ``GET /v1/devices/{device_id}/inbox/leases/{lease_id}``. Under the
        one store lock (shared with claims, renewals, releases, completions,
        acks, retries and revocation), the lease is resolved first: a
        never-committed *lease_id* raises ``lease_not_found``
        (404/lease_id) and one owned by another device raises
        ``lease_id_conflict`` (409/lease_id), regardless of the path
        device's state. A matching lease is returned even when its device
        has since been revoked: the lookup deliberately performs no device
        state check, so a lease stays inspectable after revocation.

        The body keys are ``device_id``, ``lease_id``, ``limit``,
        ``state``, ``leased_until``, ``released_at``, ``completion`` and
        ``messages`` in that order. ``limit`` is the claim's frozen limit;
        ``leased_until`` is the current effective deadline (the claim
        value, or the last renewal's after one or more renewals);
        ``released_at`` is the release timestamp or ``None``;
        ``completion`` is ``None`` or the completion's
        ``completion_id``/``outcome``/``completed_at``. ``state`` is one of
        ``completed`` / ``released`` / ``expired`` / ``active``: a
        completion or release wins, and an unterminated lease is active
        only while the query instant precedes its effective deadline.
        ``messages`` lists the claimed messages in claim order; each item
        is ``session_id``/``message_id``/``sequence``/``acked``/
        ``attempts`` with the delivery record's current values (an ack
        never removes an item). The query writes nothing and advances no
        commit generation, so an unchanged state answers byte-identically
        and a rebuild after restart yields the same result.
        """
        with self._lock:
            now = datetime.now(timezone.utc)
            existing = self._find_inbox_lease_locked(lease_id)
            if existing is None:
                raise InboxLeaseError(INBOX_LEASE_NOT_FOUND)
            owner, limit, _claim_deadline, _released, ordered_keys = existing
            # A cross-device lookup is a conflict regardless of the path
            # device's current state, mirroring the mutating lease routes.
            if owner != device_id:
                raise InboxLeaseError(INBOX_LEASE_CONFLICT)

            # The lease copies on every delivery record are identical by
            # restore validation; gather them in claim (inbox) order.
            leases: List[MessageLease] = []
            for key in ordered_keys:
                lease = next(item for item in self._delivery[key].leases
                             if item.lease_id == lease_id)
                leases.append(lease)
            first_lease = leases[0]
            if first_lease.completion is not None:
                state = "completed"
            elif first_lease.released_at is not None:
                state = "released"
            elif self._lease_is_active_locked(first_lease, now):
                state = "active"
            else:
                state = "expired"
            # Current effective deadline as a stored string: the claim
            # value, or the last renewal's after one or more renewals.
            leased_until = first_lease.renewals[-1].leased_until \
                if first_lease.renewals else first_lease.leased_until
            completion = None if first_lease.completion is None else {
                "completion_id": first_lease.completion.completion_id,
                "outcome": first_lease.completion.outcome,
                "completed_at": first_lease.completion.completed_at,
            }
            messages: List[Dict[str, Any]] = []
            for hit_session_id, hit_message_id in ordered_keys:
                delivery = self._delivery[(hit_session_id, hit_message_id)]
                sequence = next(m.sequence
                                for m in self._messages[hit_session_id]
                                if m.message_id == hit_message_id)
                messages.append({
                    "session_id": hit_session_id,
                    "message_id": hit_message_id,
                    "sequence": sequence,
                    "acked": delivery.acked,
                    "attempts": delivery.attempts,
                })
            return {
                "device_id": device_id,
                "lease_id": lease_id,
                "limit": limit,
                "state": state,
                "leased_until": leased_until,
                "released_at": first_lease.released_at,
                "completion": completion,
                "messages": messages,
            }

    def inbox_leases_page(self, device_id: str, state_filter: str,
                          after: int, limit: int
    ) -> Optional[Dict[str, Any]]:
        """Read one page of one device's 1:1-inbox lease history (read-only).

        ``GET /v1/devices/{device_id}/inbox/leases``. Under the one store
        lock (shared with claims, renewals, releases, completions, acks,
        retries and revocation), an unknown *device_id* returns ``None``
        (404/device_id at the service layer); a revoked device's history
        stays readable. Every lease ever committed is history: the same
        ``lease_id`` appears on one copy per leased message's delivery
        record, so the copies are deduplicated by id (restore validation
        guarantees they are identical) and ordered by the frozen initial
        claim ``leased_until`` and then the ``lease_id`` code points. Each
        lease's current state is decided exactly as in
        :meth:`inbox_lease_get` — a completion wins, then a release, then
        active while the effective deadline is still in the future, else
        expired — and ``all`` keeps every state while the other
        *state_filter* values keep only that state. The filtered list is
        then paged: the first *after* items are skipped and at most
        *limit* items are returned. The query writes nothing and advances
        no commit generation, so an unchanged state answers
        byte-identically.
        """
        with self._lock:
            now = datetime.now(timezone.utc)
            device = self._find_device(device_id)
            if device is None:
                return None

            # Deduplicate the per-message lease copies: the first delivery
            # record seen for an id is authoritative (restore validation
            # binds one id to one device, limit, deadline, release and
            # completion). Only delivery records on 1:1 sessions the path
            # device is the recipient of contribute, so leases held by
            # other devices never appear; count the claimed messages per
            # lease.
            leases_by_id: Dict[str, MessageLease] = {}
            message_counts: Dict[str, int] = {}
            for (session_id, _message_id), delivery in \
                    self._delivery.items():
                session = self._sessions.get(session_id)
                if session is None \
                        or session.recipient_device_id != device_id:
                    continue
                for lease in delivery.leases:
                    if lease.lease_id not in leases_by_id:
                        leases_by_id[lease.lease_id] = lease
                        message_counts[lease.lease_id] = 1
                    else:
                        message_counts[lease.lease_id] += 1

            ordered = sorted(
                leases_by_id.values(),
                key=lambda lease: (lease.leased_until, lease.lease_id))

            filtered: List[Dict[str, Any]] = []
            for lease in ordered:
                if lease.completion is not None:
                    current = "completed"
                elif lease.released_at is not None:
                    current = "released"
                elif self._lease_is_active_locked(lease, now):
                    current = "active"
                else:
                    current = "expired"
                if state_filter != "all" and current != state_filter:
                    continue
                leased_until = lease.renewals[-1].leased_until \
                    if lease.renewals else lease.leased_until
                filtered.append({
                    "lease_id": lease.lease_id,
                    "state": current,
                    "leased_until": leased_until,
                    "message_count": message_counts[lease.lease_id],
                })

            page = filtered[after:after + limit]
            if not page:
                next_after = after
            else:
                next_after = after + len(page)
            return {
                "device_id": device_id,
                "leases": page,
                "next_after": next_after,
                "has_more": after + limit < len(filtered),
            }

    def inbox_jobs_page(self, device_id: str, state_filter: str,
                        after: int, limit: int
                        ) -> Optional[Dict[str, Any]]:
        """Read one page of one device's redelivery-job history (read-only).

        ``GET /v1/devices/{device_id}/inbox-jobs``. Under the one store
        lock (shared with queue/dispatch/recover/cancel, lease claims,
        completions, acks and revocation), an unknown *device_id* returns
        ``None`` (404/device_id at the service layer); a revoked device's
        history stays readable. Every job ever queued is history: jobs are
        kept in first-successful-``queue`` commit order (the mapping's
        insertion order, which restore also preserves), so no sort key is
        needed. Only the path device's own jobs contribute; ``all`` keeps
        every state while the other *state_filter* values
        (``pending``/``running``/``succeeded``/``failed``/``cancelled``)
        keep only jobs currently in that state. The filtered list is then
        paged: the first *after* items are skipped and at most *limit*
        items are returned. The query writes nothing and advances no
        commit generation, so an unchanged state answers byte-identically
        and a restart (which restores the same commit order) yields the
        same pages.
        """
        with self._lock:
            device = self._find_device(device_id)
            if device is None:
                return None

            filtered: List[Dict[str, Any]] = []
            for job in self._redelivery_jobs.values():
                if job.device_id != device_id:
                    continue
                if state_filter != "all" and job.state != state_filter:
                    continue
                filtered.append({
                    "job_id": job.job_id,
                    "state": job.state,
                    "lease_id": job.lease_id,
                    "cancellation_id": job.cancellation_id,
                    "cancelled_at": job.cancelled_at,
                })

            page = filtered[after:after + limit]
            if not page:
                next_after = after
            else:
                next_after = after + len(page)
            return {
                "device_id": device_id,
                "jobs": page,
                "next_after": next_after,
                "has_more": after + limit < len(filtered),
            }

    @staticmethod
    def redelivery_job_view(job: RedeliveryJob) -> Dict[str, Any]:
        """The wire view of one redelivery job, keys in response order."""
        return {
            "job_id": job.job_id,
            "device_id": job.device_id,
            "state": job.state,
            "lease_id": job.lease_id,
        }

    def redelivery_job_submit(
            self, device_id: str, job_id: str, op: str
    ) -> Tuple[Dict[str, Any], int]:
        """Apply one 1:1-inbox redelivery job operation.

        ``POST /v1/inbox-jobs``; *op* is ``queue``, ``dispatch`` or
        ``status`` (already validated by the service, as are the non-empty
        string ids). Under the one store lock (shared with claims,
        completions, acks and revocation) the device is resolved first —
        unknown or revoked raises :class:`RedeliveryJobError`
        ``device_unknown``/``device_inactive`` (409/device_id) for every
        op. A ``job_id`` committed for another device raises
        ``job_id_conflict`` (409/job_id) whatever the op.

        ``queue`` creates the job ``pending`` (201, one persistence
        notification); replaying it on the same device returns the job's
        current view with 200 and writes nothing. ``dispatch``/``status``
        on a never-queued id raise ``job_not_found`` (404/job_id).
        ``status`` is purely read-only (200, no write). ``dispatch`` on a
        non-pending job is a replay (200, no write); on a pending job it
        leases up to :data:`REDELIVERY_JOB_DISPATCH_LIMIT` unacked inbox
        messages that carry no still-active lease, in the inbox's fixed
        ``(session.created_at, session_id, sequence)`` order, recording a
        lease whose ``lease_id`` is the *job_id* itself on every leased
        message's delivery record: a non-empty selection moves the job to
        ``running`` with ``lease_id`` set, an empty one straight to
        ``succeeded`` with ``lease_id`` null — both 201 and one
        notification. Returns ``(body, status_code)`` with the body keys
        ``job_id``, ``device_id``, ``state``, ``lease_id`` in that order.
        """
        with self._lock:
            device = self._find_device(device_id)
            if device is None:
                raise RedeliveryJobError(REDELIVERY_JOB_DEVICE_UNKNOWN)
            if device.revoked:
                raise RedeliveryJobError(REDELIVERY_JOB_DEVICE_INACTIVE)
            job = self._redelivery_jobs.get(job_id)
            if job is not None and job.device_id != device_id:
                raise RedeliveryJobError(REDELIVERY_JOB_CONFLICT)
            if op == "queue":
                if job is not None:
                    return self.redelivery_job_view(job), 200
                job = RedeliveryJob(job_id=job_id, device_id=device_id)
                self._redelivery_jobs[job_id] = job
                self._notify_change()
                return self.redelivery_job_view(job), 201
            if job is None:
                raise RedeliveryJobError(REDELIVERY_JOB_NOT_FOUND)
            if op == "status" or job.state != REDELIVERY_JOB_PENDING:
                # A status query is read-only; a dispatch replayed on a
                # non-pending job returns the current view and writes
                # nothing.
                return self.redelivery_job_view(job), 200

            now = datetime.now(timezone.utc)
            # The dispatch lease takes the job_id as its lease_id; an
            # unrelated claim that already occupies the id blocks the
            # dispatch rather than corrupting the lease namespace.
            if self._find_inbox_lease_locked(job_id) is not None:
                raise RedeliveryJobError(REDELIVERY_JOB_LEASE_OCCUPIED)
            picked: List[Tuple[Tuple[str, str], Message]] = []
            for _, session_id, message in self._inbox_entries_locked(
                    device_id):
                key = (session_id, message.message_id)
                state = self._delivery.get(key)
                if state is not None \
                        and self._has_active_lease_locked(state, now):
                    continue
                picked.append((key, message))
                if len(picked) >= REDELIVERY_JOB_DISPATCH_LIMIT:
                    break
            if picked:
                # timespec="microseconds" always emits six fractional
                # digits, mirroring the claim endpoint's deadlines.
                leased_until = (now + timedelta(
                    seconds=INBOX_LEASE_SECONDS)) \
                    .isoformat(timespec="microseconds")
                for key, _message in picked:
                    state = self._delivery.get(key)
                    if state is None:
                        state = MessageDelivery()
                        self._delivery[key] = state
                    state.leases.append(MessageLease(
                        lease_id=job_id,
                        limit=REDELIVERY_JOB_DISPATCH_LIMIT,
                        leased_until=leased_until))
                job.state = REDELIVERY_JOB_RUNNING
                job.lease_id = job_id
            else:
                # Nothing to redeliver: the job succeeds immediately and
                # never takes a lease.
                job.state = REDELIVERY_JOB_SUCCEEDED
            self._notify_change()
            return self.redelivery_job_view(job), 201

    def redelivery_job_cancel(
            self, device_id: str, job_id: str, cancellation_id: str
    ) -> Tuple[Dict[str, Any], int]:
        """Cancel a ``pending``/``running`` redelivery job.

        ``POST /v1/inbox-jobs`` with ``op=cancel`` (the non-empty
        *cancellation_id* is already validated by the service). Under the
        one store lock — shared with dispatch, leases, completions, acks and
        revocation — the device is resolved first (unknown/revoked ->
        ``device_unknown``/``device_inactive``, 409/device_id), then the
        job: a never-queued id raises ``job_not_found`` (404/job_id) and an
        id committed for another device raises ``job_id_conflict``
        (409/job_id).

        An already cancelled job replays only under the same
        *cancellation_id*: an exact replay answers its first response
        byte-identically with 200 and writes nothing (even after device
        revocation, the device gate still runs first); a different id raises
        ``cancellation_id_conflict`` (409/cancellation_id). A
        ``succeeded``/``failed`` terminal job raises
        ``job_not_cancellable`` (409/job_id).

        A first cancel of a ``pending`` job moves it to ``cancelled`` with
        ``lease_id`` null and no lease touched. A first cancel of a
        ``running`` job releases its current lease so the leased messages
        can be claimed again (the lease stays on the delivery records as
        released history; an already-released current lease keeps its
        earlier release timestamp) and the job keeps that lease's id. Both
        cases stamp ``cancellation_id`` and the UTC ``cancelled_at``
        timestamp (six microsecond digits, ``+00:00``), notify persistence
        once (201, commit_seq + 1) and return the four-key view
        ``job_id``, ``device_id``, ``state``, ``lease_id`` in that order.
        """
        with self._lock:
            device = self._find_device(device_id)
            if device is None:
                raise RedeliveryJobError(REDELIVERY_JOB_DEVICE_UNKNOWN)
            if device.revoked:
                raise RedeliveryJobError(REDELIVERY_JOB_DEVICE_INACTIVE)
            job = self._redelivery_jobs.get(job_id)
            if job is not None and job.device_id != device_id:
                raise RedeliveryJobError(REDELIVERY_JOB_CONFLICT)
            if job is None:
                raise RedeliveryJobError(REDELIVERY_JOB_NOT_FOUND)
            if job.state == REDELIVERY_JOB_CANCELLED:
                # An exact replay answers the frozen first response (200)
                # and writes nothing; a different id conflicts.
                if job.cancellation_id == cancellation_id:
                    return self.redelivery_job_view(job), 200
                raise RedeliveryJobError(
                    REDELIVERY_JOB_CANCELLATION_CONFLICT)
            if job.state in (REDELIVERY_JOB_SUCCEEDED,
                             REDELIVERY_JOB_FAILED):
                raise RedeliveryJobError(REDELIVERY_JOB_CANCEL_STATE)
            now = datetime.now(timezone.utc)
            cancelled_at = now.isoformat(timespec="microseconds")
            if job.state == REDELIVERY_JOB_RUNNING and job.lease_id:
                # Release the current lease in the same transaction so its
                # messages are claimable again. The lease stays as history;
                # a current lease already released (via the release
                # endpoint) keeps its earlier, frozen release timestamp.
                for state in self._delivery.values():
                    for lease in state.leases:
                        if lease.lease_id == job.lease_id \
                                and lease.released_at is None:
                            lease.released_at = cancelled_at
            job.state = REDELIVERY_JOB_CANCELLED
            job.cancellation_id = cancellation_id
            job.cancelled_at = cancelled_at
            self._notify_change()
            return self.redelivery_job_view(job), 201

    def redelivery_job_recover(
            self, device_id: str, job_id: str, recovery_id: str
    ) -> Tuple[Dict[str, Any], int]:
        """Recover the expired lease of a running redelivery job.

        ``POST /v1/inbox-jobs`` with ``op=recover`` (the non-empty
        *recovery_id* is already validated by the service). Under the one
        store lock — shared with leases, completions, acks and revocation —
        the device is resolved first (unknown/revoked ->
        ``device_unknown``/``device_inactive``, 409/device_id), then the
        job: a never-queued id raises ``job_not_found`` (404/job_id) and an
        id committed for another device raises ``job_id_conflict``
        (409/job_id).

        Replaying the same *recovery_id* on the same job returns the job's
        current view with 200 and writes nothing. A *recovery_id* already
        committed on another job, or already occupied as any inbox lease id
        (another recovery's new lease included), raises
        ``recovery_id_conflict`` (409/recovery_id). Only a ``running`` job
        whose current lease has expired or been released is recoverable: a
        still-valid lease raises ``lease_active`` (409/lease_id) and every
        other state raises ``job_not_recoverable`` (409/job_id).

        On a first recovery the inbox's fixed
        ``(session.created_at, session_id, sequence)`` order yields up to
        :data:`REDELIVERY_JOB_DISPATCH_LIMIT` unacked messages without a
        still-active lease. A non-empty selection is leased for the usual
        :data:`INBOX_LEASE_SECONDS` window under an *ordinary* inbox lease
        whose id is the *recovery_id*: the job stays ``running`` and its
        ``lease_id`` moves to that id (the stale dispatch/previous recovery
        lease stays on the records as history). An empty selection ends the
        job as ``succeeded`` with ``lease_id`` null. Either outcome appends
        one recovery record (``recovery_id`` plus the new lease id, or
        null), notifies persistence once (201, commit_seq + 1) and is
        terminal for this request — a later completion of the new lease
        moves the job to ``succeeded``/``failed`` exactly as a dispatch
        lease does.
        """
        with self._lock:
            now = datetime.now(timezone.utc)
            device = self._find_device(device_id)
            if device is None:
                raise RedeliveryJobError(REDELIVERY_JOB_DEVICE_UNKNOWN)
            if device.revoked:
                raise RedeliveryJobError(REDELIVERY_JOB_DEVICE_INACTIVE)
            job = self._redelivery_jobs.get(job_id)
            if job is not None and job.device_id != device_id:
                raise RedeliveryJobError(REDELIVERY_JOB_CONFLICT)
            if job is None:
                raise RedeliveryJobError(REDELIVERY_JOB_NOT_FOUND)
            # An exact replay answers the job's current view (200) and
            # writes nothing.
            if any(record.recovery_id == recovery_id
                   for record in job.recoveries):
                return self.redelivery_job_view(job), 200
            # recovery_id shares one namespace across every job's id, every
            # job's recovery history and the ordinary inbox lease ids: it
            # may not equal any (other, or this) job_id — a dispatch may
            # still take that id as its lease — may not repeat on another
            # job (its own job was handled by the replay check above) and
            # may not collide with a lease already committed, including a
            # previous recovery lease.
            if any(other.job_id == recovery_id
                   for other in self._redelivery_jobs.values()):
                raise RedeliveryJobError(
                    REDELIVERY_JOB_RECOVERY_CONFLICT)
            if any(any(record.recovery_id == recovery_id
                       for record in other.recoveries)
                   for other in self._redelivery_jobs.values()
                   if other is not job):
                raise RedeliveryJobError(
                    REDELIVERY_JOB_RECOVERY_CONFLICT)
            if self._find_inbox_lease_locked(recovery_id) is not None:
                raise RedeliveryJobError(
                    REDELIVERY_JOB_RECOVERY_CONFLICT)
            if job.state != REDELIVERY_JOB_RUNNING:
                raise RedeliveryJobError(REDELIVERY_JOB_RECOVERY_STATE)
            # Only an expired or released current lease may be recovered:
            # a still-valid one rejects 409/lease_id. The lease copies are
            # identical, so the first copy carrying the job's current id is
            # authoritative.
            current_lease: Optional[MessageLease] = None
            for state in self._delivery.values():
                for lease in state.leases:
                    if lease.lease_id == job.lease_id:
                        current_lease = lease
                        break
                if current_lease is not None:
                    break
            if current_lease is not None \
                    and self._lease_is_active_locked(current_lease, now):
                raise RedeliveryJobError(
                    REDELIVERY_JOB_RECOVERY_LEASE_ACTIVE)
            leased_until = (now + timedelta(seconds=INBOX_LEASE_SECONDS)) \
                .isoformat(timespec="microseconds")
            self._redelivery_job_apply_recovery_locked(
                job, recovery_id, now, leased_until)
            self._notify_change()
            return self.redelivery_job_view(job), 201

    def _redelivery_job_apply_recovery_locked(
            self, job: RedeliveryJob, recovery_id: str, now: datetime,
            leased_until: str) -> None:
        """Lease one recovery's messages and record it on *job*.

        Lock required; the caller guarantees *job* is recoverable (running,
        its current lease expired or released) and *recovery_id* free. The
        inbox's fixed ``(session.created_at, session_id, sequence)`` order
        yields up to :data:`REDELIVERY_JOB_DISPATCH_LIMIT` unacked messages
        without a still-active lease at *now*. A non-empty selection is
        leased for the usual :data:`INBOX_LEASE_SECONDS` window under an
        ordinary inbox lease named by the *recovery_id* (deadline
        *leased_until*): the job stays ``running`` and its ``lease_id``
        moves to that id (the stale dispatch/previous recovery lease stays
        on the records as history). An empty selection ends the job as
        ``succeeded`` with ``lease_id`` null. Either outcome appends one
        recovery record (``recovery_id`` plus the new lease id, or null).
        """
        picked: List[Tuple[str, str]] = []
        for _, session_id, message in self._inbox_entries_locked(
                job.device_id):
            key = (session_id, message.message_id)
            state = self._delivery.get(key)
            if state is not None \
                    and self._has_active_lease_locked(state, now):
                continue
            picked.append(key)
            if len(picked) >= REDELIVERY_JOB_DISPATCH_LIMIT:
                break
        if picked:
            # An ordinary inbox lease named by the recovery_id; the stale
            # dispatch lease stays on the records as history but no longer
            # withholds these messages.
            for key in picked:
                state = self._delivery.get(key)
                if state is None:
                    state = MessageDelivery()
                    self._delivery[key] = state
                state.leases.append(MessageLease(
                    lease_id=recovery_id,
                    limit=REDELIVERY_JOB_DISPATCH_LIMIT,
                    leased_until=leased_until))
            job.lease_id = recovery_id
            job.recoveries.append(RedeliveryJobRecovery(
                recovery_id=recovery_id, lease_id=recovery_id))
        else:
            # Nothing left to redeliver: the job finishes succeeded and
            # keeps no current lease.
            job.state = REDELIVERY_JOB_SUCCEEDED
            job.lease_id = None
            job.recoveries.append(RedeliveryJobRecovery(
                recovery_id=recovery_id, lease_id=None))

    def redelivery_job_recover_batch(
            self, device_id: str, items: List[Tuple[str, str]]
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Atomically recover many running redelivery jobs of one device.

        ``POST /v1/inbox-jobs/recover-batch``; *items* are ``(job_id,
        recovery_id)`` pairs already validated for shape and per-field
        uniqueness by the service. Under the one store lock — shared with
        leases, completions, acks and revocation — the device is resolved
        first (unknown/revoked -> :class:`RedeliveryJobError`
        ``device_unknown``/``device_inactive``, 409/device_id).

        Every item is then prechecked in input order with exactly the
        single-job ``op=recover`` rules, the first failure raising
        :class:`RedeliveryJobRecoverBatchError` carrying that item's index
        and writing nothing: a never-queued ``job_id`` raises
        ``job_not_found`` (404/items[i].job_id), a job of another device
        ``job_id_conflict`` (409/items[i].job_id), a ``recovery_id`` equal
        to any job id, already committed on another job or occupied as an
        inbox lease id ``recovery_id_conflict`` (409/items[i].recovery_id),
        a non-``running`` job ``job_not_recoverable`` (409/items[i].job_id)
        and a still-valid current lease ``lease_active``
        (409/items[i].lease_id). An item whose ``recovery_id`` is already
        committed on its own job is an exact replay and skips the remaining
        checks, as the single-job entry does.

        Only after every item passes does the batch resolve: when all items
        are replays the jobs' current views are returned with 200 and
        nothing is written; a mix of replays and first-time items cannot be
        applied atomically and raises ``partial_replay`` for the first
        replayed item (409/items[i].recovery_id). Otherwise each job is
        recovered in input order via
        :meth:`_redelivery_job_apply_recovery_locked` — earlier items'
        fresh leases already withhold their messages from later items — and
        the whole batch commits with one persistence notification (201,
        commit_seq + 1; a durable write failure rolls every job and lease
        back). Returns ``(results, status_code)`` with one three-key
        (``job_id``/``state``/``lease_id``) view per item, in input order.
        """
        with self._lock:
            now = datetime.now(timezone.utc)
            device = self._find_device(device_id)
            if device is None:
                raise RedeliveryJobError(REDELIVERY_JOB_DEVICE_UNKNOWN)
            if device.revoked:
                raise RedeliveryJobError(REDELIVERY_JOB_DEVICE_INACTIVE)
            jobs: List[RedeliveryJob] = []
            replays: List[bool] = []
            for index, (job_id, recovery_id) in enumerate(items):
                job = self._redelivery_jobs.get(job_id)
                if job is not None and job.device_id != device_id:
                    raise RedeliveryJobRecoverBatchError(
                        REDELIVERY_JOB_CONFLICT, index)
                if job is None:
                    raise RedeliveryJobRecoverBatchError(
                        REDELIVERY_JOB_NOT_FOUND, index)
                jobs.append(job)
                # An exact replay of a committed recovery skips the
                # remaining per-item checks (mirroring the single-job
                # entry) and is only classified here.
                if any(record.recovery_id == recovery_id
                       for record in job.recoveries):
                    replays.append(True)
                    continue
                replays.append(False)
                # The recovery_id shares one namespace across every job id,
                # every job's recovery history and the ordinary inbox lease
                # ids (a previous recovery lease included). Within one
                # batch the ids are already unique (the service rejects
                # duplicates), and no item has been applied yet, so the
                # committed state decides.
                if any(other.job_id == recovery_id
                       for other in self._redelivery_jobs.values()):
                    raise RedeliveryJobRecoverBatchError(
                        REDELIVERY_JOB_RECOVERY_CONFLICT, index)
                if any(any(record.recovery_id == recovery_id
                           for record in other.recoveries)
                       for other in self._redelivery_jobs.values()
                       if other is not job):
                    raise RedeliveryJobRecoverBatchError(
                        REDELIVERY_JOB_RECOVERY_CONFLICT, index)
                if self._find_inbox_lease_locked(recovery_id) is not None:
                    raise RedeliveryJobRecoverBatchError(
                        REDELIVERY_JOB_RECOVERY_CONFLICT, index)
                if job.state != REDELIVERY_JOB_RUNNING:
                    raise RedeliveryJobRecoverBatchError(
                        REDELIVERY_JOB_RECOVERY_STATE, index)
                # Only an expired or released current lease may be
                # recovered: a still-valid one rejects 409/lease_id. The
                # lease copies are identical, so the first copy carrying
                # the job's current id is authoritative.
                current_lease: Optional[MessageLease] = None
                for state in self._delivery.values():
                    for lease in state.leases:
                        if lease.lease_id == job.lease_id:
                            current_lease = lease
                            break
                    if current_lease is not None:
                        break
                if current_lease is not None \
                        and self._lease_is_active_locked(current_lease, now):
                    raise RedeliveryJobRecoverBatchError(
                        REDELIVERY_JOB_RECOVERY_LEASE_ACTIVE, index)
            if all(replays):
                # Every item replays its committed recovery: answer the
                # current views with 200 and write nothing (no persistence
                # notification, no commit_seq advance).
                return [self._recover_batch_item_view(job) for job in jobs], \
                    200
            if any(replays):
                # A partial replay cannot commit atomically: the batch is
                # all-or-nothing, so the first replayed item conflicts.
                raise RedeliveryJobRecoverBatchError(
                    REDELIVERY_JOB_RECOVERY_PARTIAL_REPLAY,
                    replays.index(True))
            # One deadline for the whole batch: every item commits in the
            # same locked transaction, mirroring the single recover's
            # timespec="microseconds" rendering.
            leased_until = (now + timedelta(seconds=INBOX_LEASE_SECONDS)) \
                .isoformat(timespec="microseconds")
            for job, (_job_id, recovery_id) in zip(jobs, items):
                self._redelivery_job_apply_recovery_locked(
                    job, recovery_id, now, leased_until)
            # One persistence notification for the whole batch: every job
            # and lease commits (or rolls back) together and the generation
            # advances exactly once.
            self._notify_change()
            return [self._recover_batch_item_view(job) for job in jobs], 201

    @staticmethod
    def _recover_batch_item_view(job: RedeliveryJob) -> Dict[str, Any]:
        """One recover-batch result item, keys in response order."""
        return {
            "job_id": job.job_id,
            "state": job.state,
            "lease_id": job.lease_id,
        }

    def _redelivery_job_completed_locked(self, lease_id: str,
                                         outcome: str) -> None:
        """Move a dispatched/redelivering redelivery job to a terminal state.

        Called under the store lock when an inbox lease is completed with
        *outcome*. A dispatch lease's id is the job_id itself, but a lease
        created by ``op=recover`` carries the client's ``recovery_id`` as
        its lease id, so the running job is found by its current
        ``lease_id`` rather than by job id. It becomes ``succeeded`` on
        ``delivered`` and ``failed`` on ``failed``, inside the same locked
        transaction as the completion. Leases not currently held by a
        running job change nothing (an expired/superseded lease cannot be
        completed anyway).
        """
        for job in self._redelivery_jobs.values():
            if job.state == REDELIVERY_JOB_RUNNING \
                    and job.lease_id == lease_id:
                job.state = REDELIVERY_JOB_SUCCEEDED \
                    if outcome == "delivered" else REDELIVERY_JOB_FAILED
                return

    def inbox_retry_batch(
            self, device_id: str, attempt_id: str,
            items: List[Tuple[str, str]]
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """Record one ``attempt_id`` against many unacked 1:1 inbox messages.

        *items* are ``(session_id, message_id)`` pairs already validated for
        shape and pair uniqueness by the service. The device must exist and
        not be revoked (a batch-level :class:`MessageSyncError`). Items are
        then prechecked in array order and the first failure raises
        :class:`InboxRetryBatchError` carrying that item's index; the batch
        writes nothing:

        * a session that is neither a 1:1 nor a group session ->
          ``session_unknown`` (404);
        * a group session, or a 1:1 session whose recipient is not this device
          -> ``session_not_recipient`` (409);
        * a message the session does not hold -> ``message_unknown`` (404);
        * a message already acked for the recipient -> ``message_acked``
          (409).

        Only after every item passes does the commit run, under the one store
        lock: the single *attempt_id* is added to each target message's
        ``delivery`` dedup set. A message whose set already held the id is a
        replay (its ``attempts`` is not counted); otherwise the record is
        created as needed and ``attempts`` advances by one. Every target
        shares one persistence notification, so the whole batch commits as
        one generation and a durable write failure rolls every delivery
        record back. Returns ``(results, any_new)`` with one three-field
        (``session_id``/``message_id``/``attempts``) result per item, in
        input order; when no item added the id nothing is persisted.
        """
        with self._lock:
            device = self._find_device(device_id)
            if device is None:
                raise MessageSyncError(MESSAGE_SYNC_DEVICE_UNKNOWN)
            if device.revoked:
                raise MessageSyncError(MESSAGE_SYNC_DEVICE_INACTIVE)
            targets: List[Tuple[str, Message]] = []
            for index, (session_id, message_id) in enumerate(items):
                session = self._sessions.get(session_id)
                if session is None:
                    reason = INBOX_RETRY_NOT_RECIPIENT \
                        if self._group_sessions.get(session_id) is not None \
                        else INBOX_RETRY_SESSION_UNKNOWN
                    raise InboxRetryBatchError(reason, index)
                if device_id != session.recipient_device_id:
                    raise InboxRetryBatchError(
                        INBOX_RETRY_NOT_RECIPIENT, index)
                message = next((m for m in self._messages.get(session_id, [])
                                if m.message_id == message_id), None)
                if message is None:
                    raise InboxRetryBatchError(
                        INBOX_RETRY_MESSAGE_UNKNOWN, index)
                state = self._delivery.get((session_id, message_id))
                if state is not None and state.acked:
                    raise InboxRetryBatchError(INBOX_RETRY_MESSAGE_ACKED,
                                               index)
                # Precheck in array order; only after every item passes are
                # any dedup sets touched below.
                targets.append((session_id, message))
            any_new = False
            for session_id, message in targets:
                key = (session_id, message.message_id)
                state = self._delivery.get(key)
                if state is not None and attempt_id in state.attempt_ids:
                    continue
                if state is None:
                    state = MessageDelivery()
                    self._delivery[key] = state
                state.attempt_ids.add(attempt_id)
                state.attempts += 1
                any_new = True
            if any_new:
                # One persistence notification for the whole batch: all the
                # per-message dedup sets and counters commit (or roll back)
                # together and the generation advances at most once.
                self._notify_change()
            results = [{
                "session_id": session_id,
                "message_id": message.message_id,
                "attempts": self._delivery[
                    (session_id, message.message_id)].attempts,
            } for session_id, message in targets]
            return results, any_new

    # -- messages ----------------------------------------------------------

    @staticmethod
    def message_view(message: Message) -> Dict[str, Any]:
        """Copy one message envelope into its public seven-field view."""
        return {
            "session_id": message.session_id,
            "sender_device_id": message.sender_device_id,
            "message_id": message.message_id,
            "sequence": message.sequence,
            "nonce": message.nonce,
            "ciphertext": message.ciphertext,
            "created_at": message.created_at,
        }

    def append_message(self, session_id: str, sender_device_id: str,
                       message_id: str, sequence: int, nonce: str,
                       ciphertext: str) -> Message:
        """Atomically validate and append one message to a session's stream.

        The session lookup, sender revocation check, duplicate-id check,
        sequence-continuity check and session-scoped nonce-replay check all
        happen while holding the store lock (the same lock device revocations
        take), so a concurrent revocation is linearized either wholly before
        this call (then it fails) or wholly after it (then the message is
        retained). The checks have one fixed priority: unknown session,
        inactive sender, duplicate message_id, bad sequence, duplicate nonce.
        A nonce already accepted in this same session is a replay and fails
        even when every other field is fresh; the identical nonce in a
        different session is accepted. On any failure nothing is written and
        :class:`MessageCreateError` carries the reason.
        """
        with self._lock:
            message = self._append_message_locked(
                session_id, sender_device_id, message_id, sequence, nonce,
                ciphertext)
            self._notify_change()
            return message

    def _append_message_locked(self, session_id: str, sender_device_id: str,
                               message_id: str, sequence: int, nonce: str,
                               ciphertext: str) -> Message:
        """Validate and append one message; the caller holds the store lock.

        Performs the fixed-priority checks documented on
        :meth:`append_message` and appends the envelope without notifying
        persistence, so callers can commit extra records in the same locked
        transaction (see :meth:`submit_message`).
        """
        group_session = self._group_sessions.get(session_id)
        if session_id not in self._sessions and group_session is None:
            raise MessageCreateError(MESSAGE_SESSION_UNKNOWN)

        sender_key = self._device_index.get(sender_device_id)
        sender = (self._devices.get(sender_key)
                  if sender_key is not None else None)
        if sender is None or sender.revoked:
            raise MessageCreateError(MESSAGE_SENDER_INACTIVE)
        # Only devices frozen into a group session may post into it; a
        # device removed after the freeze is rejected like an inactive one.
        if (group_session is not None
                and sender_device_id not in group_session.members):
            raise MessageCreateError(MESSAGE_SENDER_INACTIVE)

        stream = self._messages.setdefault(session_id, [])
        if any(m.message_id == message_id for m in stream):
            raise MessageCreateError(MESSAGE_DUPLICATE_ID)
        expected = stream[-1].sequence + 1 if stream else 1
        if sequence != expected:
            raise MessageCreateError(MESSAGE_BAD_SEQUENCE)
        used_nonces = self._used_nonces.setdefault(session_id, set())
        # Nonces are compared exactly as received (raw string equality);
        # a replay neither appends a message nor advances any sequence.
        if nonce in used_nonces:
            raise MessageCreateError(MESSAGE_DUPLICATE_NONCE)

        message = Message(
            session_id=session_id,
            sender_device_id=sender_device_id,
            message_id=message_id,
            sequence=sequence,
            nonce=nonce,
            ciphertext=ciphertext,
        )
        stream.append(message)
        used_nonces.add(nonce)
        return message

    @staticmethod
    def _submission_view(record: MessageSubmission) -> Dict[str, Any]:
        """Copy one submission record into its public eight-field view."""
        return {
            "request_id": record.request_id,
            "session_id": record.session_id,
            "sender_device_id": record.sender_device_id,
            "message_id": record.message_id,
            "sequence": record.sequence,
            "nonce": record.nonce,
            "ciphertext": record.ciphertext,
            "created_at": record.created_at,
        }

    def submit_message(self, request_id: str, session_id: str,
                       sender_device_id: str, message_id: str, sequence: int,
                       nonce: str, ciphertext: str
                       ) -> Tuple[Dict[str, Any], bool]:
        """Atomically commit one idempotent message submission.

        The ``request_id`` is looked up first: a record whose six envelope
        fields match the request replays the original response (``created``
        False, 200 at the HTTP layer) even if the sender has since been
        revoked; a record with any differing field conflicts
        (:class:`MessageCreateError` ``request_id_conflict``, 409). A fresh
        id runs the same fixed-priority checks as :meth:`append_message` and,
        on success, commits the message, the nonce and the idempotency record
        in this one locked transaction (a single persistence notification),
        so a concurrent identical submission linearizes to exactly one write.
        A failed validation writes nothing and does not consume the id.
        Returns ``(view, created)``.
        """
        with self._lock:
            record = self._message_submissions.get(request_id)
            if record is not None:
                if (record.session_id == session_id
                        and record.sender_device_id == sender_device_id
                        and record.message_id == message_id
                        and record.sequence == sequence
                        and record.nonce == nonce
                        and record.ciphertext == ciphertext):
                    return self._submission_view(record), False
                raise MessageCreateError(MESSAGE_REQUEST_ID_CONFLICT)
            message = self._append_message_locked(
                session_id, sender_device_id, message_id, sequence, nonce,
                ciphertext)
            record = MessageSubmission(
                request_id=request_id,
                session_id=session_id,
                sender_device_id=sender_device_id,
                message_id=message_id,
                sequence=sequence,
                nonce=nonce,
                ciphertext=ciphertext,
                created_at=message.created_at,
            )
            self._message_submissions[request_id] = record
            self._notify_change()
            return self._submission_view(record), True

    def message_page(self, session_id: str, device_id: str, after: int,
                     limit: int) -> Tuple[List[Dict[str, Any]], int]:
        """Atomically read one ascending page of a session's messages.

        Returns ``(message_views, next_after)``: the envelopes with
        ``sequence > after`` (ascending, at most *limit*) and the sequence to
        resume from — the last returned sequence, or *after* itself when the
        page is empty. The session lookup and the reader's revocation check
        run under the store lock, so a concurrent revocation is linearized
        either wholly before (then it fails) or wholly after this call.
        """
        with self._lock:
            group_session = self._group_sessions.get(session_id)
            if session_id not in self._sessions and group_session is None:
                raise MessageListError(MESSAGE_SESSION_UNKNOWN)

            device_key = self._device_index.get(device_id)
            device = (self._devices.get(device_key)
                      if device_key is not None else None)
            if device is None or device.revoked:
                raise MessageListError(MESSAGE_DEVICE_INACTIVE)
            # Group sessions are readable by the frozen member set only; a
            # device added or removed after the freeze is not a reader.
            if group_session is not None and device_id not in \
                    group_session.members:
                raise MessageListError(MESSAGE_DEVICE_INACTIVE)

            stream = self._messages.get(session_id, [])
            page = [m for m in stream if m.sequence > after][:limit]
            next_after = page[-1].sequence if page else after
            return [self.message_view(m) for m in page], next_after

    # -- delivery (reliable retry/ack/status) ------------------------------

    @staticmethod
    def _delivery_view(session_id: str, message: Message,
                       state: Optional["MessageDelivery"]) -> Dict[str, Any]:
        """Build the five-field delivery status view of one message."""
        return {
            "session_id": session_id,
            "message_id": message.message_id,
            "status": "acked" if state is not None and state.acked
            else "pending",
            "attempts": state.attempts if state is not None else 0,
            "sequence": message.sequence,
        }

    def _delivery_target(self, session_id: str, message_id: str,
                         device_id: str
                         ) -> Tuple[Optional[Session],
                                    Optional[GroupSession], Message]:
        """Resolve and authorize a (session, message, recipient) triple.

        Must be called while holding the store lock. A 1:1 session authorizes
        its active recipient only; a group session authorizes any active
        device frozen into the member snapshot except the message's sender —
        later group roster changes never widen or shrink that frozen set.
        Raises :class:`DeliveryError` with the mapped reason on any failure.
        """
        session = self._sessions.get(session_id)
        group_session = (self._group_sessions.get(session_id)
                         if session is None else None)
        if session is None and group_session is None:
            raise DeliveryError(DELIVERY_SESSION_UNKNOWN)
        message = next((m for m in self._messages.get(session_id, [])
                        if m.message_id == message_id), None)
        if message is None:
            raise DeliveryError(DELIVERY_MESSAGE_UNKNOWN)
        device_key = self._device_index.get(device_id)
        device = (self._devices.get(device_key)
                  if device_key is not None else None)
        if device is None or device.revoked:
            raise DeliveryError(DELIVERY_DEVICE_INACTIVE)
        if session is not None:
            if device_id != session.recipient_device_id:
                raise DeliveryError(DELIVERY_DEVICE_MISMATCH)
        elif device_id not in group_session.members \
                or device_id == message.sender_device_id:
            raise DeliveryError(DELIVERY_DEVICE_MISMATCH)
        return session, group_session, message

    def retry_message(self, session_id: str, message_id: str,
                      device_id: str, attempt_id: str
                      ) -> Tuple[Dict[str, Any], bool]:
        """Atomically record one delivery attempt for a message.

        A 1:1 session tracks one record for its recipient; a group session
        tracks one record per frozen non-sender member device. Each distinct
        ``attempt_id`` is counted exactly once, so retries with the same id
        are idempotent. Returns ``(view, created)`` where ``created`` says
        whether this is the first ever attempt for the message (for this
        device, in a group session) — 201 vs 200. Nothing changes on a
        failed authorization check.
        """
        with self._lock:
            _, group_session, message = self._delivery_target(
                session_id, message_id, device_id)
            if group_session is not None:
                bucket: Dict[Tuple, MessageDelivery] = self._group_delivery
                key: Tuple = (session_id, message_id, device_id)
            else:
                bucket = self._delivery
                key = (session_id, message_id)
            state = bucket.get(key)
            created = state is None
            if created:
                state = MessageDelivery()
                bucket[key] = state
            changed = created or attempt_id not in state.attempt_ids
            if changed:
                # A fresh attempt id (including the one that creates the
                # record) is the only state transition: dedup set, counter
                # and persistence advance together. A replay of an attempt
                # id already recorded is state-free: it neither persists nor
                # consumes a commit generation (and never triggers the
                # legacy-anchor migration), even after the message is acked.
                state.attempt_ids.add(attempt_id)
                state.attempts += 1
                self._notify_change()
            view = self._delivery_view(session_id, message, state)
            return view, created

    def ack_message(self, session_id: str, message_id: str, device_id: str,
                    sequence: int) -> Tuple[Dict[str, Any], bool]:
        """Atomically acknowledge a message for the active recipient.

        ``sequence`` must equal the message's stored sequence. The first ack
        marks the message acked (per device, in a group session); repeated
        acks are idempotent and one device's ack never touches another's.
        Returns ``(view, first_ack)``.
        """
        with self._lock:
            _, group_session, message = self._delivery_target(
                session_id, message_id, device_id)
            if sequence != message.sequence:
                raise DeliveryError(DELIVERY_BAD_SEQUENCE)
            if group_session is not None:
                bucket: Dict[Tuple, MessageDelivery] = self._group_delivery
                key: Tuple = (session_id, message_id, device_id)
            else:
                bucket = self._delivery
                key = (session_id, message_id)
            state = bucket.get(key)
            first_ack = state is None or not state.acked
            if first_ack:
                # The first ack (also creating the record when no retry ever
                # happened) is the only transition: the acked flag, sequence
                # and persistence commit atomically together. A repeated ack
                # by the same device is a state-free idempotent replay — no
                # file write, no commit generation, no anchor migration.
                if state is None:
                    state = MessageDelivery()
                    bucket[key] = state
                state.acked = True
                state.ack_sequence = sequence
                self._notify_change()
            view = self._delivery_view(session_id, message, state)
            return view, first_ack

    def message_delivery_status(self, session_id: str, message_id: str,
                                device_id: str) -> Dict[str, Any]:
        """Atomically read one message's delivery status for its recipient."""
        with self._lock:
            _, group_session, message = self._delivery_target(
                session_id, message_id, device_id)
            if group_session is not None:
                state = self._group_delivery.get(
                    (session_id, message_id, device_id))
            else:
                state = self._delivery.get((session_id, message_id))
            return self._delivery_view(session_id, message, state)

    # -- persistence snapshot ----------------------------------------------

    def snapshot_state(self) -> Dict[str, Any]:
        """Return a JSON-serializable deep copy of all stored state.

        Copied under the store lock, so the snapshot is a consistent
        linearization point: concurrent mutations are never seen half-applied.
        """
        with self._lock:
            devices = []
            for (user_id, _device_id), device in self._devices.items():
                devices.append({
                    "user_id": device.user_id,
                    "device_id": device.device_id,
                    "identity_key": device.identity_key,
                    "registered_at": device.registered_at,
                    "rotated_at": device.rotated_at,
                    "revoked": device.revoked,
                    "prekeys": [{"key_id": pk.key_id,
                                 "public_key": pk.public_key,
                                 "revoked": pk.revoked,
                                 "consumed": pk.consumed}
                                for pk in device.prekeys],
                })
            sessions = [{
                "session_id": s.session_id,
                "initiator_device_id": s.initiator_device_id,
                "recipient_device_id": s.recipient_device_id,
                "prekey_id": s.prekey_id,
                "ephemeral_key": s.ephemeral_key,
                "identity_key": s.identity_key,
                "public_key": s.public_key,
                "created_at": s.created_at,
            } for s in self._sessions.values()]
            prekey_claims = [{
                "claim_id": c.claim_id,
                "device_id": c.recipient_device_id,
                "key_id": c.key_id,
                "identity_key": c.identity_key,
                "public_key": c.public_key,
                "claimed_at": c.claimed_at,
            } for c in self._prekey_claims.values()]
            claim_session_bindings = [{
                "claim_id": b.claim_id,
                "session_id": b.session_id,
                "recipient_device_id": b.recipient_device_id,
                "prekey_id": b.prekey_id,
                "identity_key": b.identity_key,
                "public_key": b.public_key,
                "created_at": b.created_at,
            } for b in self._claim_session_bindings.values()]
            batch_claim_session_bindings = [{
                "claim_id": b.claim_id,
                "initiator_device_id": b.initiator_device_id,
                "created_at": b.created_at,
                "entries": [{
                    "recipient_device_id": entry.recipient_device_id,
                    "prekey_id": entry.prekey_id,
                    "identity_key": entry.identity_key,
                    "public_key": entry.public_key,
                    "session_id": entry.session_id,
                } for entry in b.entries],
            } for b in self._batch_claim_session_bindings.values()]
            prekey_batch_claims = [{
                "claim_id": b.claim_id,
                "user_id": b.user_id,
                "claimed_at": b.claimed_at,
                "devices": [{
                    "device_id": entry.device_id,
                    "identity_key": entry.identity_key,
                    "key_id": entry.key_id,
                    "public_key": entry.public_key,
                } for entry in b.devices],
            } for b in self._prekey_batch_claims.values()]
            groups = [{
                "group_id": g.group_id,
                "creator_device_id": g.creator_device_id,
                "members": list(g.members),
                "revision": g.revision,
                "created_at": g.created_at,
            } for g in self._groups.values()]
            group_sessions = [{
                "session_id": s.session_id,
                "group_id": s.group_id,
                "initiator_device_id": s.initiator_device_id,
                "ephemeral_key": s.ephemeral_key,
                "members": list(s.members),
                "revision": s.revision,
                "created_at": s.created_at,
            } for s in self._group_sessions.values()]
            group_session_rotations = [{
                "rotation_id": r.rotation_id,
                "predecessor_session_id": r.predecessor_session_id,
                "successor_session_id": r.successor_session_id,
                "group_id": r.group_id,
                "actor_device_id": r.actor_device_id,
                "revision": r.revision,
                "members": list(r.members),
                "created_at": r.created_at,
            } for r in self._group_session_rotations.values()]
            messages = {
                sid: [{
                    "session_id": m.session_id,
                    "sender_device_id": m.sender_device_id,
                    "message_id": m.message_id,
                    "sequence": m.sequence,
                    "nonce": m.nonce,
                    "ciphertext": m.ciphertext,
                    "created_at": m.created_at,
                } for m in stream]
                for sid, stream in self._messages.items()
            }
            delivery = [{
                "session_id": sid,
                "message_id": mid,
                "attempts": state.attempts,
                "attempt_ids": sorted(state.attempt_ids),
                "acked": state.acked,
                "ack_sequence": state.ack_sequence,
                "leases": [{
                    "lease_id": lease.lease_id,
                    "limit": lease.limit,
                    "leased_until": lease.leased_until,
                    "released_at": lease.released_at,
                    "renewals": [{
                        "renewal_id": renewal.renewal_id,
                        "leased_until": renewal.leased_until,
                    } for renewal in lease.renewals],
                    "completion": None
                    if lease.completion is None else {
                        "completion_id": lease.completion.completion_id,
                        "outcome": lease.completion.outcome,
                        "completed_at": lease.completion.completed_at,
                    },
                } for lease in state.leases],
            } for (sid, mid), state in self._delivery.items()]
            group_delivery = [{
                "session_id": sid,
                "message_id": mid,
                "device_id": did,
                "attempts": state.attempts,
                "attempt_ids": sorted(state.attempt_ids),
                "acked": state.acked,
                "ack_sequence": state.ack_sequence,
            } for (sid, mid, did), state in self._group_delivery.items()]
            used_nonces = {
                sid: sorted(nonces)
                for sid, nonces in self._used_nonces.items() if nonces
            }
            group_sync_cursors = [{
                "session_id": sid,
                "device_id": did,
                "cursor": record.cursor,
                "updated_at": record.updated_at,
            } for (sid, did), record in self._group_sync_cursors.items()]
            message_sync_cursors = [{
                "session_id": sid,
                "device_id": did,
                "cursor": record.cursor,
                "updated_at": record.updated_at,
            } for (sid, did), record in self._message_sync_cursors.items()]
            message_submissions = [{
                "request_id": r.request_id,
                "session_id": r.session_id,
                "sender_device_id": r.sender_device_id,
                "message_id": r.message_id,
                "sequence": r.sequence,
                "nonce": r.nonce,
                "ciphertext": r.ciphertext,
                "created_at": r.created_at,
            } for r in self._message_submissions.values()]
            redelivery_jobs = []
            for job in self._redelivery_jobs.values():
                # New writes always carry the fixed seven keys; an empty
                # recovery history serializes as []. Older four/five-key
                # items (written before the cancellation keys existed)
                # still load with recoveries=[] and the two cancellation
                # fields null.
                redelivery_jobs.append({
                    "job_id": job.job_id,
                    "device_id": job.device_id,
                    "state": job.state,
                    "lease_id": job.lease_id,
                    "recoveries": [{
                        "recovery_id": record.recovery_id,
                        "lease_id": record.lease_id,
                    } for record in job.recoveries],
                    "cancellation_id": job.cancellation_id,
                    "cancelled_at": job.cancelled_at,
                })
            key_events = [
                self.key_event_view(event)
                for chain in self._key_events.values() for event in chain
            ]
            document = {"devices": devices, "sessions": sessions,
                        "prekey_claims": prekey_claims,
                        "prekey_batch_claims": prekey_batch_claims,
                        "claim_session_bindings": claim_session_bindings,
                        "batch_claim_session_bindings":
                            batch_claim_session_bindings,
                        "groups": groups, "group_sessions": group_sessions,
                        "group_session_rotations": group_session_rotations,
                        "messages": messages, "delivery": delivery,
                        "group_delivery": group_delivery,
                        "used_nonces": used_nonces,
                        "group_sync_cursors": group_sync_cursors,
                        "message_sync_cursors": message_sync_cursors,
                        "message_submissions": message_submissions,
                        "redelivery_jobs": redelivery_jobs}
            # While a legacy (section-less) file is only loaded and no change
            # has anchored its chains yet, keep the section absent — never
            # persist a present-but-empty chain section, and keep the snapshot
            # restorable as the same legacy state (e.g. as a rollback base).
            if self._pending_anchor_devices is None:
                document["key_events"] = key_events
            return document

    @staticmethod
    def _replay_key_events(device: Device, chain: List[KeyEvent]) -> None:
        """Replay one device's key-audit chain against its stored record.

        Rebuilds the device's key material from an empty state by applying
        each event in sequence — ``registered`` seeds the identity key and
        the ordered pre-key list, ``identity_rotated`` replaces the identity
        key (its ``old_identity_key`` must match the replayed current one),
        ``prekey_added`` appends, ``prekey_revoked`` marks one key, and
        ``device_revoked`` (empty payload) revokes the device and every key.
        The replay applies the same rules the live mutations enforce: no
        event may follow a device revocation, a revoked key is never
        re-revoked, and a pre-key id is never added twice. The replayed
        final state must equal *device*'s stored identity key, pre-key
        order/public keys/revocation flags and revocation flag; any
        contradiction raises :class:`ValueError`.
        """
        where = f"key_events chain of device {device.device_id}"
        identity_key: Optional[str] = None
        prekeys: List[Dict[str, Any]] = []
        device_revoked = False
        for position, event in enumerate(chain):
            payload = event.payload
            if device_revoked:
                raise ValueError(
                    f"{where} has an event after the device revocation")
            if event.type == KEY_EVENT_REGISTERED:
                if position != 0:
                    raise ValueError(
                        f"{where} must start with a registered event")
                new_identity = payload.get("identity_key")
                if not isinstance(new_identity, str) or not new_identity:
                    raise ValueError(
                        f"{where} registered payload identity_key must be a "
                        f"non-empty string")
                raw_prekeys = payload.get("signed_prekeys")
                if not isinstance(raw_prekeys, list):
                    raise ValueError(
                        f"{where} registered payload signed_prekeys must be "
                        f"a list")
                seen_key_ids: Set[str] = set()
                for pk_index, element in enumerate(raw_prekeys):
                    if not isinstance(element, dict):
                        raise ValueError(
                            f"{where} registered signed_prekeys[{pk_index}] "
                            f"must be an object")
                    key_id = element.get("key_id")
                    public_key = element.get("public_key")
                    if not isinstance(key_id, str) or not key_id:
                        raise ValueError(
                            f"{where} registered signed_prekeys[{pk_index}]"
                            f".key_id must be a non-empty string")
                    if not isinstance(public_key, str) or not public_key:
                        raise ValueError(
                            f"{where} registered signed_prekeys[{pk_index}]"
                            f".public_key must be a non-empty string")
                    if key_id in seen_key_ids:
                        raise ValueError(
                            f"{where} registered payload repeats key_id "
                            f"{key_id}")
                    seen_key_ids.add(key_id)
                    prekeys.append({"key_id": key_id,
                                    "public_key": public_key,
                                    "revoked": False})
                identity_key = new_identity
            elif event.type == KEY_EVENT_IDENTITY_ROTATED:
                old_identity = payload.get("old_identity_key")
                new_identity = payload.get("new_identity_key")
                if not isinstance(old_identity, str) or not old_identity \
                        or not isinstance(new_identity, str) \
                        or not new_identity:
                    raise ValueError(
                        f"{where} identity_rotated payload keys must be "
                        f"non-empty strings")
                if old_identity != identity_key:
                    raise ValueError(
                        f"{where} identity_rotated old_identity_key does "
                        f"not match the replayed identity key")
                identity_key = new_identity
            elif event.type == KEY_EVENT_PREKEY_ADDED:
                key_id = payload.get("key_id")
                public_key = payload.get("public_key")
                if not isinstance(key_id, str) or not key_id \
                        or not isinstance(public_key, str) or not public_key:
                    raise ValueError(
                        f"{where} prekey_added payload key_id/public_key "
                        f"must be non-empty strings")
                if any(pk["key_id"] == key_id for pk in prekeys):
                    raise ValueError(
                        f"{where} prekey_added repeats key_id {key_id}")
                prekeys.append({"key_id": key_id, "public_key": public_key,
                                "revoked": False})
            elif event.type == KEY_EVENT_PREKEY_REVOKED:
                key_id = payload.get("key_id")
                public_key = payload.get("public_key")
                if not isinstance(key_id, str) or not key_id \
                        or not isinstance(public_key, str) or not public_key:
                    raise ValueError(
                        f"{where} prekey_revoked payload key_id/public_key "
                        f"must be non-empty strings")
                target = next((pk for pk in prekeys
                               if pk["key_id"] == key_id), None)
                if target is None:
                    raise ValueError(
                        f"{where} prekey_revoked names an unknown pre-key "
                        f"{key_id}")
                if target["public_key"] != public_key:
                    raise ValueError(
                        f"{where} prekey_revoked public_key does not match "
                        f"the added pre-key {key_id}")
                if target["revoked"]:
                    raise ValueError(
                        f"{where} prekey_revoked repeats the revocation of "
                        f"{key_id}")
                target["revoked"] = True
            else:  # KEY_EVENT_DEVICE_REVOKED
                if payload != {}:
                    raise ValueError(
                        f"{where} device_revoked payload must be empty")
                device_revoked = True
                for pk in prekeys:
                    pk["revoked"] = True
        if chain and chain[0].type != KEY_EVENT_REGISTERED:
            raise ValueError(f"{where} must start with a registered event")
        if identity_key != device.identity_key:
            raise ValueError(
                f"{where} replays to a different identity key than the "
                f"devices section")
        if len(prekeys) != len(device.prekeys) or any(
                replayed["key_id"] != stored.key_id
                or replayed["public_key"] != stored.public_key
                or replayed["revoked"] != stored.revoked
                for replayed, stored in zip(prekeys, device.prekeys)):
            raise ValueError(
                f"{where} replays to different pre-keys than the devices "
                f"section")
        if device_revoked != device.revoked:
            raise ValueError(
                f"{where} replays to a different revocation state than the "
                f"devices section")

    def restore_state(self, state: Dict[str, Any]) -> None:
        """Replace all in-memory state from a persisted (version-stripped) doc.

        Raises :class:`ValueError` when the document is malformed; the current
        in-memory state is only replaced after the whole document parses.
        """
        if not isinstance(state, dict):
            raise ValueError("state document must be a JSON object")

        raw_devices = state.get("devices", [])
        raw_sessions = state.get("sessions", [])
        raw_prekey_claims = state.get("prekey_claims", [])
        raw_prekey_batch_claims = state.get("prekey_batch_claims", [])
        raw_claim_session_bindings = state.get("claim_session_bindings", [])
        raw_batch_claim_session_bindings = state.get(
            "batch_claim_session_bindings", [])
        raw_groups = state.get("groups", [])
        raw_group_sessions = state.get("group_sessions", [])
        raw_group_session_rotations = state.get(
            "group_session_rotations", [])
        raw_messages = state.get("messages", {})
        raw_delivery = state.get("delivery", [])
        raw_group_delivery = state.get("group_delivery", [])
        raw_used_nonces = state.get("used_nonces")
        raw_group_sync_cursors = state.get("group_sync_cursors", [])
        raw_message_sync_cursors = state.get("message_sync_cursors", [])
        raw_message_submissions = state.get("message_submissions", [])
        raw_redelivery_jobs = state.get("redelivery_jobs", [])
        raw_key_events = state.get("key_events")
        if not (isinstance(raw_devices, list) and isinstance(raw_sessions, list)
                and isinstance(raw_prekey_claims, list)
                and isinstance(raw_prekey_batch_claims, list)
                and isinstance(raw_claim_session_bindings, list)
                and isinstance(raw_batch_claim_session_bindings, list)
                and isinstance(raw_groups, list)
                and isinstance(raw_group_sessions, list)
                and isinstance(raw_group_session_rotations, list)
                and isinstance(raw_messages, dict)
                and isinstance(raw_delivery, list)
                and isinstance(raw_group_delivery, list)
                and isinstance(raw_group_sync_cursors, list)
                and isinstance(raw_message_sync_cursors, list)
                and isinstance(raw_message_submissions, list)
                and isinstance(raw_redelivery_jobs, list)):
            raise ValueError("state document has a malformed top-level section")

        devices: Dict[Tuple[str, str], Device] = {}
        device_index: Dict[str, Tuple[str, str]] = {}
        # Registration order (the snapshot serializes devices in store
        # insertion order, which is registration order). Batch claims freeze
        # their devices in this same relative order; a file whose batch
        # device list contradicts it is internally inconsistent.
        registration_order: Dict[str, int] = {}
        for index, raw in enumerate(raw_devices):
            where = f"devices[{index}]"
            if not isinstance(raw, dict):
                raise ValueError(f"{where} must be an object")

            def req_str(name: str) -> str:
                value = raw.get(name)
                if not isinstance(value, str) or not value:
                    raise ValueError(
                        f"{where}.{name} must be a non-empty string")
                return value

            user_id = req_str("user_id")
            device_id = req_str("device_id")
            identity_key = req_str("identity_key")
            registered_at = req_str("registered_at")
            # Older version-1 files predate rotated_at; then it equals
            # registered_at (Device.__post_init__ fills in the default).
            rotated_at = None
            if "rotated_at" in raw:
                rotated_value = raw["rotated_at"]
                if not isinstance(rotated_value, str) or not rotated_value:
                    raise ValueError(
                        f"{where}.rotated_at must be a non-empty string")
                rotated_at = rotated_value
            revoked_value = raw.get("revoked", False)
            if not isinstance(revoked_value, bool):
                raise ValueError(f"{where}.revoked must be a boolean")
            raw_prekeys = raw.get("prekeys")
            if not isinstance(raw_prekeys, list):
                raise ValueError(f"{where}.prekeys must be a list")
            prekeys: List[SignedPreKey] = []
            seen_key_ids: Set[str] = set()
            for pk_index, pk in enumerate(raw_prekeys):
                pk_where = f"{where}.prekeys[{pk_index}]"
                if not isinstance(pk, dict):
                    raise ValueError(f"{pk_where} must be an object")
                key_id = pk.get("key_id")
                public_key = pk.get("public_key")
                if not isinstance(key_id, str) or not key_id:
                    raise ValueError(
                        f"{pk_where}.key_id must be a non-empty string")
                if not isinstance(public_key, str) or not public_key:
                    raise ValueError(
                        f"{pk_where}.public_key must be a non-empty string")
                pk_revoked = pk.get("revoked", False)
                if not isinstance(pk_revoked, bool):
                    raise ValueError(
                        f"{pk_where}.revoked must be a boolean")
                # A missing consumed flag means the key was never claimed
                # (older files); a present flag must be a real boolean.
                pk_consumed = pk.get("consumed", False)
                if not isinstance(pk_consumed, bool):
                    raise ValueError(
                        f"{pk_where}.consumed must be a boolean")
                if key_id in seen_key_ids:
                    raise ValueError(
                        f"duplicate prekey key_id in device "
                        f"{device_id}: {key_id}")
                seen_key_ids.add(key_id)
                prekeys.append(SignedPreKey(
                    key_id=key_id, public_key=public_key, revoked=pk_revoked,
                    consumed=pk_consumed))
            device = Device(
                user_id=user_id, device_id=device_id,
                identity_key=identity_key, registered_at=registered_at,
                rotated_at=rotated_at, prekeys=prekeys, revoked=revoked_value)
            key = (device.user_id, device.device_id)
            if key in devices or device.device_id in device_index:
                raise ValueError(
                    f"duplicate device in state: {device.device_id}")
            devices[key] = device
            device_index[device.device_id] = key
            registration_order[device.device_id] = index

        sessions: Dict[str, Session] = {}
        for index, raw in enumerate(raw_sessions):
            where = f"sessions[{index}]"
            if not isinstance(raw, dict):
                raise ValueError(f"{where} must be an object")
            session_fields = ("session_id", "initiator_device_id",
                              "recipient_device_id", "prekey_id",
                              "ephemeral_key", "identity_key", "public_key",
                              "created_at")
            values: Dict[str, str] = {}
            for name in session_fields:
                value = raw.get(name)
                if not isinstance(value, str) or not value:
                    raise ValueError(
                        f"{where}.{name} must be a non-empty string")
                values[name] = value
            # Both endpoints must name registered devices. A session snapshot
            # is frozen at creation, so revocation is not a bar; the devices
            # themselves must still exist in the document.
            initiator = devices.get(device_index.get(
                values["initiator_device_id"]))
            if initiator is None:
                raise ValueError(
                    f"{where} references an unknown initiator device: "
                    f"{values['initiator_device_id']}")
            recipient_key = device_index.get(values["recipient_device_id"])
            recipient = devices.get(recipient_key)
            if recipient is None:
                raise ValueError(
                    f"{where} references an unknown recipient device: "
                    f"{values['recipient_device_id']}")
            # The negotiated pre-key must be one of the recipient's own
            # pre-keys (it may since have been revoked; the snapshot is
            # frozen, but it cannot belong to another device).
            if not any(pk.key_id == values["prekey_id"]
                       for pk in recipient.prekeys):
                raise ValueError(
                    f"{where} prekey_id does not belong to recipient "
                    f"{values['recipient_device_id']}: "
                    f"{values['prekey_id']}")
            session_id = values["session_id"]
            if session_id in sessions:
                raise ValueError(
                    f"duplicate session in state: {session_id}")
            sessions[session_id] = Session(
                session_id=session_id,
                initiator_device_id=values["initiator_device_id"],
                recipient_device_id=values["recipient_device_id"],
                prekey_id=values["prekey_id"],
                ephemeral_key=values["ephemeral_key"],
                identity_key=values["identity_key"],
                public_key=values["public_key"],
                created_at=values["created_at"])

        # One-time pre-key claims. Older files predate the section: it is
        # absent and treated as empty (and absent flags on prekeys already
        # mark every key un-consumed). A present section is fully validated —
        # uniqueness of claim_id and of the claimed (device, key) pair, field
        # types, and references to a registered device and one of its own
        # pre-keys, with the frozen public key agreeing. A claim may name a
        # device/key that was revoked afterwards (records are immutable) but
        # the key it names must carry the consumed flag, and every consumed
        # key must be backed by exactly one claim.
        prekey_claims: Dict[str, PreKeyClaim] = {}
        claimed_pairs: Set[Tuple[str, str]] = set()
        for index, raw in enumerate(raw_prekey_claims):
            where = f"prekey_claims[{index}]"
            if not isinstance(raw, dict):
                raise ValueError(f"{where} must be an object")
            claim_id = raw.get("claim_id")
            device_id = raw.get("device_id")
            key_id = raw.get("key_id")
            identity_key = raw.get("identity_key")
            public_key = raw.get("public_key")
            claimed_at = raw.get("claimed_at")
            for name, value in (("claim_id", claim_id),
                                ("device_id", device_id),
                                ("key_id", key_id),
                                ("identity_key", identity_key),
                                ("public_key", public_key),
                                ("claimed_at", claimed_at)):
                if not isinstance(value, str) or not value:
                    raise ValueError(
                        f"{where}.{name} must be a non-empty string")
            if claim_id in prekey_claims:
                raise ValueError(
                    f"duplicate prekey claim in state: {claim_id}")
            device_key = device_index.get(device_id)
            target_device = devices.get(device_key) if device_key else None
            if target_device is None:
                raise ValueError(
                    f"{where} references an unknown recipient device: "
                    f"{device_id}")
            target_prekey = next((pk for pk in target_device.prekeys
                                  if pk.key_id == key_id), None)
            if target_prekey is None:
                raise ValueError(
                    f"{where} key_id does not belong to device "
                    f"{device_id}: {key_id}")
            if public_key != target_prekey.public_key:
                raise ValueError(
                    f"{where} public_key does not match the stored pre-key")
            if not target_prekey.consumed:
                raise ValueError(
                    f"{where} names a pre-key that is not marked consumed: "
                    f"{key_id}")
            pair = (device_id, key_id)
            if pair in claimed_pairs:
                raise ValueError(
                    f"{where} duplicates an existing claim for device "
                    f"{device_id} key {key_id}")
            claimed_pairs.add(pair)
            prekey_claims[claim_id] = PreKeyClaim(
                claim_id=claim_id, recipient_device_id=device_id,
                key_id=key_id, identity_key=identity_key,
                public_key=public_key, claimed_at=claimed_at)

        # The two halves of consumption are reconciled after both the single
        # and batch claim sections below (claimed_pairs accumulates both).

        # User-wide batch claims (version 1; older files predate the section,
        # which is then absent and loaded as empty). A present section is fully
        # validated: claim_id unique within the section and disjoint from the
        # single-claim namespace, every entry's device registered *to the
        # batch's user*, each key owned by that device with its frozen public
        # key matching, and the key carrying the consumed flag — with each
        # (device, key) pair backed by exactly one claim across both sections.
        prekey_batch_claims: Dict[str, PreKeyBatchClaim] = {}
        for index, raw in enumerate(raw_prekey_batch_claims):
            where = f"prekey_batch_claims[{index}]"
            if not isinstance(raw, dict):
                raise ValueError(f"{where} must be an object")
            claim_id = raw.get("claim_id")
            user_id = raw.get("user_id")
            claimed_at = raw.get("claimed_at")
            for name, value in (("claim_id", claim_id),
                                ("user_id", user_id),
                                ("claimed_at", claimed_at)):
                if not isinstance(value, str) or not value:
                    raise ValueError(
                        f"{where}.{name} must be a non-empty string")
            if claim_id in prekey_batch_claims:
                raise ValueError(
                    f"duplicate prekey batch claim in state: {claim_id}")
            # claim_ids are one shared namespace with single claims.
            if claim_id in prekey_claims:
                raise ValueError(
                    f"{where} claim_id is already used by a single claim: "
                    f"{claim_id}")
            raw_entries = raw.get("devices")
            if not isinstance(raw_entries, list) or not raw_entries:
                raise ValueError(
                    f"{where}.devices must be a non-empty list")
            entries: List[BatchClaimDevice] = []
            entry_devices: Set[str] = set()
            for e_index, raw_entry in enumerate(raw_entries):
                e_where = f"{where}.devices[{e_index}]"
                if not isinstance(raw_entry, dict):
                    raise ValueError(f"{e_where} must be an object")
                device_id = raw_entry.get("device_id")
                key_id = raw_entry.get("key_id")
                identity_key = raw_entry.get("identity_key")
                public_key = raw_entry.get("public_key")
                for name, value in (("device_id", device_id),
                                    ("key_id", key_id),
                                    ("identity_key", identity_key),
                                    ("public_key", public_key)):
                    if not isinstance(value, str) or not value:
                        raise ValueError(
                            f"{e_where}.{name} must be a non-empty string")
                if device_id in entry_devices:
                    raise ValueError(
                        f"{where} lists device {device_id} more than once")
                device_key = device_index.get(device_id)
                target_device = devices.get(device_key) if device_key else None
                if target_device is None:
                    raise ValueError(
                        f"{e_where} references an unknown device: {device_id}")
                # The device must belong to the batch's own user.
                if target_device.user_id != user_id:
                    raise ValueError(
                        f"{e_where} device {device_id} does not belong to "
                        f"user {user_id}")
                target_prekey = next((pk for pk in target_device.prekeys
                                      if pk.key_id == key_id), None)
                if target_prekey is None:
                    raise ValueError(
                        f"{e_where} key_id does not belong to device "
                        f"{device_id}: {key_id}")
                if public_key != target_prekey.public_key:
                    raise ValueError(
                        f"{e_where} public_key does not match the stored "
                        f"pre-key")
                if not target_prekey.consumed:
                    raise ValueError(
                        f"{e_where} names a pre-key that is not marked "
                        f"consumed: {key_id}")
                pair = (device_id, key_id)
                if pair in claimed_pairs:
                    raise ValueError(
                        f"{e_where} duplicates an existing claim for device "
                        f"{device_id} key {key_id}")
                claimed_pairs.add(pair)
                entry_devices.add(device_id)
                entries.append(BatchClaimDevice(
                    device_id=device_id, identity_key=identity_key,
                    key_id=key_id, public_key=public_key))
            # The frozen device list must keep the devices' relative
            # registration order (active-device enumeration filters, never
            # reorders). A contradicting file is refused at startup rather
            # than silently re-ordered.
            positions = [registration_order[entry.device_id]
                         for entry in entries]
            if positions != sorted(positions):
                raise ValueError(
                    f"{where}.devices must keep the devices' registration "
                    f"order")
            prekey_batch_claims[claim_id] = PreKeyBatchClaim(
                claim_id=claim_id, user_id=user_id, devices=entries,
                claimed_at=claimed_at)

        # The two halves of consumption must agree across both claim
        # sections: no consumed key may lack a backing claim (otherwise it
        # would be wrongly withheld after restart).
        for (_user, _did), device in devices.items():
            for pk in device.prekeys:
                if pk.consumed and (device.device_id, pk.key_id) \
                        not in claimed_pairs:
                    raise ValueError(
                        f"pre-key {pk.key_id} of device {device.device_id} is "
                        f"consumed but has no prekey_claims record")

        # Claim-to-session bindings written by POST /v1/sessions/from-claim.
        # Older files predate the section: it is absent and treated as empty
        # (then every committed claim may establish one session again). A
        # present section is fully validated — one record per claim_id and per
        # session, referencing a committed claim whose frozen recipient, key
        # and public material it repeats, and the 1:1 session created from it.
        claim_session_bindings: Dict[str, ClaimSessionBinding] = {}
        bound_sessions: Set[str] = set()
        for index, raw in enumerate(raw_claim_session_bindings):
            where = f"claim_session_bindings[{index}]"
            if not isinstance(raw, dict):
                raise ValueError(f"{where} must be an object")
            claim_id = raw.get("claim_id")
            session_id = raw.get("session_id")
            recipient_device_id = raw.get("recipient_device_id")
            prekey_id = raw.get("prekey_id")
            identity_key = raw.get("identity_key")
            public_key = raw.get("public_key")
            created_at = raw.get("created_at")
            for name, value in (("claim_id", claim_id),
                                ("session_id", session_id),
                                ("recipient_device_id", recipient_device_id),
                                ("prekey_id", prekey_id),
                                ("identity_key", identity_key),
                                ("public_key", public_key),
                                ("created_at", created_at)):
                if not isinstance(value, str) or not value:
                    raise ValueError(
                        f"{where}.{name} must be a non-empty string")
            claim = prekey_claims.get(claim_id)
            if claim is None:
                raise ValueError(
                    f"{where} references an unknown claim: {claim_id}")
            if claim_id in claim_session_bindings:
                raise ValueError(
                    f"duplicate claim session binding in state: {claim_id}")
            bound_session = sessions.get(session_id)
            if bound_session is None:
                raise ValueError(
                    f"{where} references an unknown session: {session_id}")
            if session_id in bound_sessions:
                raise ValueError(
                    f"{where} session is already bound to another claim: "
                    f"{session_id}")
            # The binding repeats the claim's frozen material; it must agree
            # with both the claim and the session created from it.
            if (recipient_device_id != claim.recipient_device_id
                    or prekey_id != claim.key_id
                    or identity_key != claim.identity_key
                    or public_key != claim.public_key):
                raise ValueError(
                    f"{where} frozen material does not match the claim record")
            if (bound_session.recipient_device_id != recipient_device_id
                    or bound_session.prekey_id != prekey_id
                    or bound_session.identity_key != identity_key
                    or bound_session.public_key != public_key):
                raise ValueError(
                    f"{where} session does not match the frozen claim material")
            bound_sessions.add(session_id)
            claim_session_bindings[claim_id] = ClaimSessionBinding(
                claim_id=claim_id, session_id=session_id,
                recipient_device_id=recipient_device_id, prekey_id=prekey_id,
                identity_key=identity_key, public_key=public_key,
                created_at=created_at)

        # Batch-claim-to-session-set bindings written by
        # POST /v1/sessions/from-batch-claim. Older files predate the
        # section: it is absent and loaded as empty. A present section is
        # fully validated — one record per batch claim_id, its entries
        # referencing the batch claim's frozen devices in exactly that
        # registration order, each with a distinct session that exists as a
        # 1:1 session and is not bound by any other claim, and the frozen
        # recipient/key/public material agreeing with both the batch claim
        # record and the session snapshot. Frozen identity keys are never
        # compared with the device's current (possibly rotated) key.
        batch_claim_session_bindings: Dict[str, BatchClaimSessionBinding] = {}
        batch_bound_sessions: Set[str] = set()
        for index, raw in enumerate(raw_batch_claim_session_bindings):
            where = f"batch_claim_session_bindings[{index}]"
            if not isinstance(raw, dict):
                raise ValueError(f"{where} must be an object")
            claim_id = raw.get("claim_id")
            initiator_device_id = raw.get("initiator_device_id")
            created_at = raw.get("created_at")
            for name, value in (("claim_id", claim_id),
                                ("initiator_device_id", initiator_device_id),
                                ("created_at", created_at)):
                if not isinstance(value, str) or not value:
                    raise ValueError(
                        f"{where}.{name} must be a non-empty string")
            batch = prekey_batch_claims.get(claim_id)
            if batch is None:
                raise ValueError(
                    f"{where} references an unknown batch claim: {claim_id}")
            if claim_id in batch_claim_session_bindings:
                raise ValueError(
                    f"duplicate batch claim session binding in state: "
                    f"{claim_id}")
            raw_entries = raw.get("entries")
            if not isinstance(raw_entries, list) or not raw_entries:
                raise ValueError(
                    f"{where}.entries must be a non-empty list")
            # The entry set must equal the batch claim snapshot, listed in
            # the same frozen registration order.
            if len(raw_entries) != len(batch.devices):
                raise ValueError(
                    f"{where} entry count does not match the batch claim "
                    f"snapshot")
            entries: List[BatchClaimSessionEntry] = []
            binding_session_ids: Set[str] = set()
            for e_index, (raw_entry, frozen) in enumerate(
                    zip(raw_entries, batch.devices)):
                e_where = f"{where}.entries[{e_index}]"
                if not isinstance(raw_entry, dict):
                    raise ValueError(f"{e_where} must be an object")
                recipient_device_id = raw_entry.get("recipient_device_id")
                prekey_id = raw_entry.get("prekey_id")
                identity_key = raw_entry.get("identity_key")
                public_key = raw_entry.get("public_key")
                session_id = raw_entry.get("session_id")
                for name, value in (
                        ("recipient_device_id", recipient_device_id),
                        ("prekey_id", prekey_id),
                        ("identity_key", identity_key),
                        ("public_key", public_key),
                        ("session_id", session_id)):
                    if not isinstance(value, str) or not value:
                        raise ValueError(
                            f"{e_where}.{name} must be a non-empty string")
                if recipient_device_id != frozen.device_id:
                    raise ValueError(
                        f"{e_where} recipient does not match the batch claim "
                        f"order: expected {frozen.device_id}, got "
                        f"{recipient_device_id}")
                # Frozen material is checked against the claim record and
                # the session snapshot only — never against a device's
                # post-rotation current key.
                if (prekey_id != frozen.key_id
                        or identity_key != frozen.identity_key
                        or public_key != frozen.public_key):
                    raise ValueError(
                        f"{e_where} frozen material does not match the "
                        f"batch claim record")
                bound_session = sessions.get(session_id)
                if bound_session is None:
                    raise ValueError(
                        f"{e_where} references an unknown session: "
                        f"{session_id}")
                if (session_id in binding_session_ids
                        or session_id in batch_bound_sessions
                        or session_id in bound_sessions):
                    raise ValueError(
                        f"{e_where} session is already bound to another "
                        f"claim: {session_id}")
                if (bound_session.recipient_device_id != recipient_device_id
                        or bound_session.prekey_id != prekey_id
                        or bound_session.identity_key != identity_key
                        or bound_session.public_key != public_key):
                    raise ValueError(
                        f"{e_where} session does not match the frozen batch "
                        f"claim material")
                if bound_session.initiator_device_id != initiator_device_id:
                    raise ValueError(
                        f"{e_where} session initiator does not match the "
                        f"binding initiator")
                binding_session_ids.add(session_id)
                entries.append(BatchClaimSessionEntry(
                    recipient_device_id=recipient_device_id,
                    prekey_id=prekey_id, identity_key=identity_key,
                    public_key=public_key, session_id=session_id))
            # Equal-length position-by-position matching enforces both the
            # exact device set and the frozen order.
            if initiator_device_id not in device_index:
                raise ValueError(
                    f"{where} references an unknown initiator device: "
                    f"{initiator_device_id}")
            if any(entry.recipient_device_id == initiator_device_id
                   for entry in entries):
                raise ValueError(
                    f"{where} initiator is one of the claimed devices")
            batch_bound_sessions.update(binding_session_ids)
            batch_claim_session_bindings[claim_id] = BatchClaimSessionBinding(
                claim_id=claim_id, initiator_device_id=initiator_device_id,
                entries=entries, created_at=created_at)

        groups: Dict[str, Group] = {}
        for index, raw in enumerate(raw_groups):
            where = f"groups[{index}]"
            if not isinstance(raw, dict):
                raise ValueError(f"{where} must be an object")
            group_id = raw.get("group_id")
            creator_device_id = raw.get("creator_device_id")
            created_at = raw.get("created_at")
            if not isinstance(group_id, str) or not group_id:
                raise ValueError(f"{where}.group_id must be a non-empty string")
            if not (isinstance(creator_device_id, str)
                    and creator_device_id):
                raise ValueError(
                    f"{where}.creator_device_id must be a non-empty string")
            if not isinstance(created_at, str) or not created_at:
                raise ValueError(
                    f"{where}.created_at must be a non-empty string")
            revision = raw.get("revision")
            if not isinstance(revision, int) or isinstance(revision, bool) \
                    or revision <= 0:
                raise ValueError(
                    f"{where}.revision must be a positive integer")
            members = raw.get("members")
            if not isinstance(members, list) or not members:
                raise ValueError(
                    f"{where}.members must be a non-empty list")
            if not all(isinstance(value, str) and value
                       for value in members):
                raise ValueError(
                    f"{where}.members must be a list of non-empty strings")
            # The stored roster is always de-duplicated; a document carrying
            # a repeated member is internally inconsistent.
            if len(set(members)) != len(members):
                raise ValueError(
                    f"{where}.members must not contain duplicate members")
            # The creator is fixed for the group's lifetime and always sits
            # at the head of the stored roster.
            if members[0] != creator_device_id:
                raise ValueError(
                    f"{where} creator must be the first member")
            # The creator must be a registered device (a revoked creator's
            # historical groups stay recoverable, so revocation is allowed).
            if creator_device_id not in device_index:
                raise ValueError(
                    f"{where} references an unknown creator device: "
                    f"{creator_device_id}")
            if group_id in groups:
                raise ValueError(
                    f"duplicate group in state: {group_id}")
            groups[group_id] = Group(
                group_id=group_id, creator_device_id=creator_device_id,
                members=list(members), revision=revision,
                created_at=created_at)

        group_sessions: Dict[str, GroupSession] = {}
        for index, raw in enumerate(raw_group_sessions):
            where = f"group_sessions[{index}]"
            if not isinstance(raw, dict):
                raise ValueError(f"{where} must be an object")
            session_id = raw.get("session_id")
            group_id = raw.get("group_id")
            initiator_device_id = raw.get("initiator_device_id")
            ephemeral_key = raw.get("ephemeral_key")
            created_at = raw.get("created_at")
            for name, value in (("session_id", session_id),
                                ("group_id", group_id),
                                ("initiator_device_id", initiator_device_id),
                                ("ephemeral_key", ephemeral_key),
                                ("created_at", created_at)):
                if not isinstance(value, str) or not value:
                    raise ValueError(
                        f"{where}.{name} must be a non-empty string")
            revision = raw.get("revision")
            if not isinstance(revision, int) or isinstance(revision, bool) \
                    or revision <= 0:
                raise ValueError(
                    f"{where}.revision must be a positive integer")
            members = raw.get("members")
            if not isinstance(members, list) or not members:
                raise ValueError(
                    f"{where}.members must be a non-empty list")
            if not all(isinstance(value, str) and value
                       for value in members):
                raise ValueError(
                    f"{where}.members must be a list of non-empty strings")
            if len(set(members)) != len(members):
                raise ValueError(
                    f"{where}.members must not contain duplicate members")
            # The frozen snapshot belongs to a stored group, and its revision
            # cannot be ahead of the group's current revision in the same
            # document.
            group = groups.get(group_id)
            if group is None:
                raise ValueError(
                    f"{where} references an unknown group: {group_id}")
            if revision > group.revision:
                raise ValueError(
                    f"{where} revision {revision} exceeds the current group "
                    f"revision {group.revision}")
            # The initiator must be a registered device frozen into the
            # snapshot (revocation is not a bar: the session is immutable).
            if initiator_device_id not in device_index:
                raise ValueError(
                    f"{where} references an unknown initiator device: "
                    f"{initiator_device_id}")
            if initiator_device_id not in members:
                raise ValueError(
                    f"{where} initiator must be a frozen member: "
                    f"{initiator_device_id}")
            if session_id in group_sessions:
                raise ValueError(
                    "duplicate group session in state: "
                    f"{session_id}")
            # Session ids are a shared keyspace for messages and delivery; a
            # 1:1 and a group session sharing an id would be ambiguous.
            if session_id in sessions:
                raise ValueError(
                    f"duplicate session in state: {session_id}")
            group_sessions[session_id] = GroupSession(
                session_id=session_id, group_id=group_id,
                initiator_device_id=initiator_device_id,
                ephemeral_key=ephemeral_key, members=list(members),
                revision=revision, created_at=created_at)

        # Group-session rotation records. Older version-1 files predate the
        # section: it is absent and treated as empty. A present section is
        # fully validated — every record references a stored group (its
        # group_id must match the predecessor's group), a predecessor and a
        # successor group session, the rotation_id/successor/predecessor are
        # each unique (no id reuse, no forking a predecessor, and no two
        # records pointing at one successor), the actor is the group's
        # registered creator, and the frozen revision/members/created_at agree
        # exactly with the successor snapshot. A contradiction refuses
        # startup rather than silently dropping the record.
        group_session_rotations: Dict[str, GroupSessionRotation] = {}
        rotation_predecessors: Set[str] = set()
        rotation_successors: Set[str] = set()
        for index, raw in enumerate(raw_group_session_rotations):
            where = f"group_session_rotations[{index}]"
            if not isinstance(raw, dict):
                raise ValueError(f"{where} must be an object")
            rotation_id = raw.get("rotation_id")
            predecessor_session_id = raw.get("predecessor_session_id")
            successor_session_id = raw.get("successor_session_id")
            group_id = raw.get("group_id")
            actor_device_id = raw.get("actor_device_id")
            created_at = raw.get("created_at")
            for name, value in (("rotation_id", rotation_id),
                                ("predecessor_session_id",
                                 predecessor_session_id),
                                ("successor_session_id", successor_session_id),
                                ("group_id", group_id),
                                ("actor_device_id", actor_device_id),
                                ("created_at", created_at)):
                if not isinstance(value, str) or not value:
                    raise ValueError(
                        f"{where}.{name} must be a non-empty string")
            revision = raw.get("revision")
            if not isinstance(revision, int) or isinstance(revision, bool) \
                    or revision <= 0:
                raise ValueError(
                    f"{where}.revision must be a positive integer")
            members = raw.get("members")
            if not isinstance(members, list) or not members:
                raise ValueError(
                    f"{where}.members must be a non-empty list")
            if not all(isinstance(value, str) and value for value in members):
                raise ValueError(
                    f"{where}.members must be a list of non-empty strings")
            if len(set(members)) != len(members):
                raise ValueError(
                    f"{where}.members must not contain duplicate members")
            if rotation_id in group_session_rotations:
                raise ValueError(
                    f"duplicate group session rotation in state: "
                    f"{rotation_id}")
            if predecessor_session_id == successor_session_id:
                raise ValueError(
                    f"{where} successor session must differ from its "
                    f"predecessor: {successor_session_id}")
            predecessor = group_sessions.get(predecessor_session_id)
            if predecessor is None:
                raise ValueError(
                    f"{where} references an unknown predecessor session: "
                    f"{predecessor_session_id}")
            successor = group_sessions.get(successor_session_id)
            if successor is None:
                raise ValueError(
                    f"{where} references an unknown successor session: "
                    f"{successor_session_id}")
            if predecessor_session_id in rotation_predecessors:
                raise ValueError(
                    f"{where} forks predecessor session: "
                    f"{predecessor_session_id}")
            if successor_session_id in rotation_successors:
                raise ValueError(
                    f"{where} successor session already produced by another "
                    f"rotation: {successor_session_id}")
            group = groups.get(group_id)
            if group is None:
                raise ValueError(
                    f"{where} references an unknown group: {group_id}")
            if group_id != predecessor.group_id \
                    or group_id != successor.group_id:
                raise ValueError(
                    f"{where} group_id does not match the rotated sessions")
            if actor_device_id not in device_index:
                raise ValueError(
                    f"{where} references an unknown actor device: "
                    f"{actor_device_id}")
            # Only the (immutable) group creator can commit a rotation.
            if actor_device_id != group.creator_device_id:
                raise ValueError(
                    f"{where} actor is not the group creator: "
                    f"{actor_device_id}")
            # The frozen record must equal the successor snapshot it produced.
            if successor.initiator_device_id != actor_device_id \
                    or successor.revision != revision \
                    or list(successor.members) != list(members) \
                    or successor.created_at != created_at:
                raise ValueError(
                    f"{where} frozen values do not match the successor "
                    f"session snapshot")
            # Revisions only advance; a successor cannot predate its
            # predecessor's frozen revision.
            if revision < predecessor.revision:
                raise ValueError(
                    f"{where} revision {revision} predates the predecessor "
                    f"revision {predecessor.revision}")
            rotation_predecessors.add(predecessor_session_id)
            rotation_successors.add(successor_session_id)
            group_session_rotations[rotation_id] = GroupSessionRotation(
                rotation_id=rotation_id,
                predecessor_session_id=predecessor_session_id,
                successor_session_id=successor_session_id,
                group_id=group_id, actor_device_id=actor_device_id,
                revision=revision, members=list(members),
                created_at=created_at)

        messages: Dict[str, List[Message]] = {}
        for sid, stream in raw_messages.items():
            if not isinstance(sid, str) or not isinstance(stream, list):
                raise ValueError("messages must map session_id to a list")
            # Every stream belongs to a saved session (1:1 or group); a key
            # that names no session is a dangling reference.
            if sid not in sessions and sid not in group_sessions:
                raise ValueError(
                    f"messages reference an unknown session: {sid}")
            group_session = group_sessions.get(sid)
            parsed: List[Message] = []
            seen_message_ids: Set[str] = set()
            seen_nonces: Set[str] = set()
            for index, raw in enumerate(stream):
                where = f"messages[{sid}][{index}]"
                if not isinstance(raw, dict):
                    raise ValueError(f"{where} must be an object")
                try:
                    envelope_sid = raw["session_id"]
                    sender = raw["sender_device_id"]
                    message_id = raw["message_id"]
                    sequence = raw["sequence"]
                    nonce = raw["nonce"]
                    ciphertext = raw["ciphertext"]
                    created_at = raw["created_at"]
                except KeyError as error:
                    raise ValueError(
                        f"{where} missing field: {error.args[0]}") from None
                if not all(isinstance(value, str) and value for value in (
                        envelope_sid, sender, message_id, nonce, ciphertext,
                        created_at)):
                    raise ValueError(
                        f"{where} string fields must be non-empty strings")
                # The envelope must agree with the stream key it is filed
                # under; a mismatch means the document is internally
                # inconsistent.
                if envelope_sid != sid:
                    raise ValueError(
                        f"{where} session_id does not match its stream key")
                if not isinstance(sequence, int) or isinstance(sequence, bool):
                    raise ValueError(f"{where} sequence must be an integer")
                # Sequences run 1..n with no gaps or reordering, exactly as
                # append_message enforces for live traffic.
                if sequence != index + 1:
                    raise ValueError(
                        f"{where} sequence must run from 1 without gaps")
                if message_id in seen_message_ids:
                    raise ValueError(
                        f"{where} duplicates message_id in session: "
                        f"{message_id}")
                if nonce in seen_nonces:
                    raise ValueError(
                        f"{where} duplicates nonce in session: {nonce}")
                # The sender must be a registered device. A revoked device's
                # historical messages stay recoverable, so revocation is not
                # checked here; group sessions additionally require the
                # sender to be in the frozen member snapshot.
                if sender not in device_index:
                    raise ValueError(
                        f"{where} references an unknown sender device: "
                        f"{sender}")
                if group_session is not None \
                        and sender not in group_session.members:
                    raise ValueError(
                        f"{where} sender is not a frozen member of the "
                        f"group session: {sender}")
                seen_message_ids.add(message_id)
                seen_nonces.add(nonce)
                parsed.append(Message(
                    session_id=envelope_sid, sender_device_id=sender,
                    message_id=message_id, sequence=sequence, nonce=nonce,
                    ciphertext=ciphertext, created_at=created_at))
            messages[sid] = parsed

        # Session-scoped replay protection. Older version-1 files predate the
        # section: rebuild it from stored message history, which lists every
        # nonce ever accepted. A present section must map session_id to a
        # duplicate-free list of plain strings and must equal the nonce set
        # reconstructed from the messages section exactly — a section that
        # omits accepted nonces would silently re-open replay, and one that
        # adds unknown nonces is corrupt; both refuse startup.
        rebuilt_nonces: Dict[str, Set[str]] = {
            sid: {message.nonce for message in stream}
            for sid, stream in messages.items() if stream
        }
        used_nonces: Dict[str, Set[str]] = {}
        if raw_used_nonces is None:
            used_nonces = rebuilt_nonces
        else:
            if not isinstance(raw_used_nonces, dict):
                raise ValueError("used_nonces must be an object")
            parsed_nonces: Dict[str, Set[str]] = {}
            for sid, nonce_list in raw_used_nonces.items():
                if not isinstance(sid, str) or not isinstance(nonce_list, list) \
                        or not all(isinstance(value, str)
                                   for value in nonce_list):
                    raise ValueError(
                        "used_nonces must map session_id to a list of strings")
                if len(set(nonce_list)) != len(nonce_list):
                    raise ValueError(
                        f"used_nonces[{sid}] contains duplicate nonces")
                parsed_nonces[sid] = set(nonce_list)
            if parsed_nonces != rebuilt_nonces:
                raise ValueError(
                    "used_nonces does not match the stored message nonces")
            used_nonces = parsed_nonces

        # Idempotent message submissions. Older version-1 files predate the
        # section: it is absent and treated as empty. A present section is
        # fully validated — request_id unique, every record referencing a
        # stored session (1:1 or group) and one of its stored messages, with
        # the six frozen envelope fields (and the frozen created_at) agreeing
        # exactly with that message. Any contradiction refuses startup rather
        # than silently dropping the idempotency guarantee.
        message_submissions: Dict[str, MessageSubmission] = {}
        for index, raw in enumerate(raw_message_submissions):
            where = f"message_submissions[{index}]"
            if not isinstance(raw, dict):
                raise ValueError(f"{where} must be an object")
            try:
                request_id = raw["request_id"]
                s_session = raw["session_id"]
                s_sender = raw["sender_device_id"]
                s_message_id = raw["message_id"]
                s_sequence = raw["sequence"]
                s_nonce = raw["nonce"]
                s_ciphertext = raw["ciphertext"]
                s_created_at = raw["created_at"]
            except KeyError as error:
                raise ValueError(
                    f"{where} missing field: {error.args[0]}") from None
            if not all(isinstance(value, str) and value for value in (
                    request_id, s_session, s_sender, s_message_id, s_nonce,
                    s_ciphertext, s_created_at)):
                raise ValueError(
                    f"{where} string fields must be non-empty strings")
            if not isinstance(s_sequence, int) \
                    or isinstance(s_sequence, bool) or s_sequence < 1:
                raise ValueError(
                    f"{where} sequence must be a positive integer")
            if request_id in message_submissions:
                raise ValueError(
                    f"duplicate message submission in state: {request_id}")
            # The record belongs to a stored session and to one of that
            # session's stored messages; the frozen envelope must equal the
            # message it committed (the sender may since have been revoked —
            # historical submissions stay recoverable).
            if s_session not in sessions and s_session not in group_sessions:
                raise ValueError(
                    f"{where} references an unknown session: {s_session}")
            target = next((m for m in messages.get(s_session, [])
                           if m.message_id == s_message_id), None)
            if target is None:
                raise ValueError(
                    f"{where} references an unknown message: "
                    f"{s_session}/{s_message_id}")
            if (target.sender_device_id != s_sender
                    or target.sequence != s_sequence
                    or target.nonce != s_nonce
                    or target.ciphertext != s_ciphertext
                    or target.created_at != s_created_at):
                raise ValueError(
                    f"{where} frozen envelope does not match the stored "
                    f"message")
            message_submissions[request_id] = MessageSubmission(
                request_id=request_id, session_id=s_session,
                sender_device_id=s_sender, message_id=s_message_id,
                sequence=s_sequence, nonce=s_nonce, ciphertext=s_ciphertext,
                created_at=s_created_at)

        delivery: Dict[Tuple[str, str], MessageDelivery] = {}
        # Global inbox-lease index, keyed by lease_id. One id is durably
        # bound to exactly one device, one limit, one claim deadline, one
        # release timestamp and one renewal list across every delivery
        # record it appears on; a contradiction refuses startup. Built
        # while delivery records are parsed and cross-checked below
        # (sessions are already restored, so the owner device is known).
        lease_index: Dict[str, Tuple[str, int, str, Optional[str],
                                     Tuple[Tuple[str, str], ...]]] = {}
        for index, raw in enumerate(raw_delivery):
            where = f"delivery[{index}]"
            if not isinstance(raw, dict):
                raise ValueError(f"{where} must be an object")
            try:
                d_session = raw["session_id"]
                d_message = raw["message_id"]
                attempts = raw["attempts"]
                attempt_ids = raw["attempt_ids"]
                acked = raw["acked"]
                ack_sequence = raw["ack_sequence"]
            except KeyError as error:
                raise ValueError(
                    f"{where} missing field: {error.args[0]}") from None
            if not (isinstance(d_session, str) and d_session
                    and isinstance(d_message, str) and d_message):
                raise ValueError(
                    f"{where} session_id/message_id must be non-empty strings")
            if not isinstance(attempts, int) or isinstance(attempts, bool) \
                    or attempts < 0:
                raise ValueError(
                    f"{where} attempts must be a non-negative integer")
            if not isinstance(attempt_ids, list) or not all(
                    isinstance(value, str) and value for value in attempt_ids):
                raise ValueError(
                    f"{where} attempt_ids must be a list of non-empty strings")
            if len(set(attempt_ids)) != len(attempt_ids):
                raise ValueError(
                    f"{where} attempt_ids must not contain duplicates")
            # The counter is derived state: it must agree with the id list.
            if attempts != len(attempt_ids):
                raise ValueError(
                    f"{where} attempts must equal the number of attempt_ids")
            if not isinstance(acked, bool):
                raise ValueError(f"{where} acked must be a boolean")
            if not isinstance(ack_sequence, int) \
                    or isinstance(ack_sequence, bool) or ack_sequence < 0:
                raise ValueError(
                    f"{where} ack_sequence must be a non-negative integer")
            dkey = (d_session, d_message)
            if dkey in delivery:
                raise ValueError(
                    f"duplicate delivery record in state: {dkey}")
            # A delivery record only exists for a stored message; a record
            # naming no message is a dangling reference.
            target = next((m for m in messages.get(d_session, [])
                           if m.message_id == d_message), None)
            if target is None:
                raise ValueError(
                    f"{where} references an unknown message: {dkey}")
            # Inbox redelivery leases. Older version-1 files predate the
            # field: it is absent and treated as empty. A present field must
            # be a list of well-formed objects, one lease_id must not repeat
            # within a record, and each id binds one device/limit/deadline/
            # release timestamp globally. The per-lease ``released_at`` key
            # is itself a later extension: absent means null (never
            # released); a non-null value must be a canonical UTC timestamp.
            leases: List[MessageLease] = []
            if "leases" in raw:
                raw_leases = raw["leases"]
                if not isinstance(raw_leases, list):
                    raise ValueError(f"{where}.leases must be a list")
                owner_session = sessions.get(d_session)
                if owner_session is None:
                    # A delivery record carrying inbox leases can only belong
                    # to a 1:1 session (group delivery lives in
                    # group_delivery); a lease on anything else is malformed.
                    raise ValueError(
                        f"{where}.leases can only be attached to a 1:1 "
                        f"session delivery record")
                owner_device = owner_session.recipient_device_id
                seen_lease_ids: Set[str] = set()
                for l_index, raw_lease in enumerate(raw_leases):
                    l_where = f"{where}.leases[{l_index}]"
                    if not isinstance(raw_lease, dict):
                        raise ValueError(f"{l_where} must be an object")
                    try:
                        lease_id = raw_lease["lease_id"]
                        lease_limit = raw_lease["limit"]
                        leased_until = raw_lease["leased_until"]
                    except KeyError as error:
                        raise ValueError(
                            f"{l_where} missing field: {error.args[0]}") \
                            from None
                    if not (isinstance(lease_id, str) and lease_id):
                        raise ValueError(
                            f"{l_where}.lease_id must be a non-empty string")
                    if not isinstance(lease_limit, int) \
                            or isinstance(lease_limit, bool) \
                            or not 1 <= lease_limit <= 100:
                        raise ValueError(
                            f"{l_where}.limit must be an integer in 1..100")
                    if not (isinstance(leased_until, str) and leased_until):
                        raise ValueError(
                            f"{l_where}.leased_until must be a non-empty "
                            f"string")
                    released_at = raw_lease.get("released_at")
                    if released_at is not None \
                            and not _is_utc_microsecond_iso(released_at):
                        raise ValueError(
                            f"{l_where}.released_at must be null or a UTC "
                            f"ISO-8601 timestamp with six microsecond "
                            f"digits and a +00:00 offset")
                    # Lease renewals. Older files predate the key: absent
                    # means empty. Each item freezes the client-chosen
                    # renewal_id (unique within this lease) and the new
                    # effective deadline; deadlines must chain exactly
                    # +30s from the claim deadline, then item by item.
                    raw_renewals = raw_lease.get("renewals", [])
                    if not isinstance(raw_renewals, list):
                        raise ValueError(
                            f"{l_where}.renewals must be a list")
                    renewals: List[MessageLeaseRenewal] = []
                    seen_renewal_ids: Set[str] = set()
                    chain_from = leased_until
                    for r_index, raw_renewal in enumerate(raw_renewals):
                        r_where = f"{l_where}.renewals[{r_index}]"
                        if not isinstance(raw_renewal, dict):
                            raise ValueError(f"{r_where} must be an object")
                        try:
                            renewal_id = raw_renewal["renewal_id"]
                            renewal_until = raw_renewal["leased_until"]
                        except KeyError as error:
                            raise ValueError(
                                f"{r_where} missing field: "
                                f"{error.args[0]}") from None
                        if not (isinstance(renewal_id, str) and renewal_id):
                            raise ValueError(
                                f"{r_where}.renewal_id must be a non-empty "
                                f"string")
                        if not _is_utc_microsecond_iso(renewal_until):
                            raise ValueError(
                                f"{r_where}.leased_until must be a UTC "
                                f"ISO-8601 timestamp with six microsecond "
                                f"digits and a +00:00 offset")
                        if renewal_id in seen_renewal_ids:
                            raise ValueError(
                                f"{r_where} repeats renewal_id "
                                f"{renewal_id}")
                        seen_renewal_ids.add(renewal_id)
                        # Each renewal extends the previous effective
                        # deadline by exactly the lease lifetime.
                        try:
                            previous = datetime.fromisoformat(chain_from)
                        except (TypeError, ValueError):
                            previous = None
                        if previous is None or previous.tzinfo is None:
                            raise ValueError(
                                f"{r_where} cannot chain onto an "
                                f"unparseable deadline {chain_from!r}")
                        expected = (previous + timedelta(
                            seconds=INBOX_LEASE_SECONDS)) \
                            .isoformat(timespec="microseconds")
                        if renewal_until != expected:
                            raise ValueError(
                                f"{r_where}.leased_until must extend the "
                                f"previous deadline by exactly "
                                f"{INBOX_LEASE_SECONDS} seconds")
                        chain_from = renewal_until
                        renewals.append(MessageLeaseRenewal(
                            renewal_id=renewal_id,
                            leased_until=renewal_until))
                    # Lease completion. Older files predate the key: absent
                    # means null (never completed). A present value must be
                    # null or one object carrying exactly
                    # completion_id/outcome/completed_at in that order, the
                    # id a non-empty string, outcome one of the two wire
                    # values and the stamp a canonical UTC timestamp. A
                    # completion and a release are mutually exclusive
                    # terminal states.
                    completion: Optional[MessageLeaseCompletion] = None
                    if "completion" in raw_lease:
                        raw_completion = raw_lease["completion"]
                        if raw_completion is not None:
                            c_where = f"{l_where}.completion"
                            if not isinstance(raw_completion, dict):
                                raise ValueError(
                                    f"{c_where} must be null or an object")
                            if list(raw_completion) != [
                                    "completion_id", "outcome",
                                    "completed_at"]:
                                raise ValueError(
                                    f"{c_where} must have exactly the keys "
                                    f"'completion_id', 'outcome', "
                                    f"'completed_at' in order")
                            completion_id = raw_completion["completion_id"]
                            completion_outcome = raw_completion["outcome"]
                            completed_at = raw_completion["completed_at"]
                            if not (isinstance(completion_id, str)
                                    and completion_id):
                                raise ValueError(
                                    f"{c_where}.completion_id must be a "
                                    f"non-empty string")
                            if completion_outcome not in ("delivered",
                                                          "failed"):
                                raise ValueError(
                                    f"{c_where}.outcome must be one of "
                                    f"'delivered' or 'failed'")
                            if not _is_utc_microsecond_iso(completed_at):
                                raise ValueError(
                                    f"{c_where}.completed_at must be a UTC "
                                    f"ISO-8601 timestamp with six "
                                    f"microsecond digits and a +00:00 "
                                    f"offset")
                            completion = MessageLeaseCompletion(
                                completion_id=completion_id,
                                outcome=completion_outcome,
                                completed_at=completed_at)
                    if completion is not None and released_at is not None:
                        raise ValueError(
                            f"{l_where} cannot carry both a release and a "
                            f"completion")
                    if lease_id in seen_lease_ids:
                        raise ValueError(
                            f"{l_where} repeats lease_id {lease_id}")
                    seen_lease_ids.add(lease_id)
                    binding = (owner_device, lease_limit, leased_until,
                               released_at,
                               tuple((r.renewal_id, r.leased_until)
                                     for r in renewals),
                               None if completion is None
                               else (completion.completion_id,
                                     completion.outcome,
                                     completion.completed_at))
                    prior = lease_index.get(lease_id)
                    if prior is not None and prior != binding:
                        raise ValueError(
                            f"inbox lease {lease_id} is bound inconsistently "
                            f"across records (device/limit/leased_until/"
                            f"released_at/renewals/completion)")
                    lease_index[lease_id] = binding
                    leases.append(MessageLease(
                        lease_id=lease_id, limit=lease_limit,
                        leased_until=leased_until, released_at=released_at,
                        renewals=renewals, completion=completion))
            # The ack cursor mirrors the message once acked and is 0 before.
            if acked:
                if ack_sequence != target.sequence:
                    raise ValueError(
                        f"{where} ack_sequence must equal the message "
                        f"sequence {target.sequence}")
            elif ack_sequence != 0:
                raise ValueError(
                    f"{where} ack_sequence must be 0 while not acked")
            delivery[dkey] = MessageDelivery(
                attempts=attempts, attempt_ids=set(attempt_ids), acked=acked,
                ack_sequence=ack_sequence, leases=leases)

        # 1:1-inbox redelivery jobs. Older version-1 files predate the
        # section: it is absent and treated as empty. A present section is
        # fully validated — job_id unique, device_id a registered device,
        # state one of the four wire values, lease_id null or a lease that
        # exists, is owned by the job's device and agrees with the state
        # machine — together with the optional per-job ``recoveries``
        # history (absent on older items means empty): each record carries
        # exactly recovery_id/lease_id, the id is a non-empty string
        # globally unique across every job's recovery history, lease_id is
        # null (an empty recovery that ended the job) or a non-empty string
        # equal to its recovery_id whose inbox lease exists and belongs to
        # the job's device. The state machine is checked in both
        # directions: a pending job has neither a lease nor recoveries; a
        # running job's current lease is its dispatch lease (the job_id)
        # with no recovery history, or the uncompleted lease of the last
        # (non-empty) recovery; a succeeded job either never dispatched a
        # lease, carries a lease completed ``delivered`` (its dispatch
        # lease without recoveries, or the last recovery lease), or ended
        # on an empty recovery (last record lease_id null, no current
        # lease); a failed job's current lease completed ``failed``; every
        # earlier recovery lease must be uncompleted (it expired or was
        # released before the next recovery and never terminated the job).
        # Any contradiction refuses startup rather than silently dropping
        # the job state machine.
        redelivery_jobs: Dict[str, RedeliveryJob] = {}
        recovery_ids_seen: Set[str] = set()
        for index, raw in enumerate(raw_redelivery_jobs):
            where = f"redelivery_jobs[{index}]"
            if not isinstance(raw, dict):
                raise ValueError(f"{where} must be an object")
            try:
                j_job_id = raw["job_id"]
                j_device_id = raw["device_id"]
                j_state = raw["state"]
                j_lease_id = raw["lease_id"]
            except KeyError as error:
                raise ValueError(
                    f"{where} missing field: {error.args[0]}") from None
            if not (isinstance(j_job_id, str) and j_job_id):
                raise ValueError(
                    f"{where}.job_id must be a non-empty string")
            if not (isinstance(j_device_id, str) and j_device_id):
                raise ValueError(
                    f"{where}.device_id must be a non-empty string")
            if j_state not in REDELIVERY_JOB_STATES:
                raise ValueError(
                    f"{where}.state must be one of "
                    f"{sorted(REDELIVERY_JOB_STATES)}")
            if j_lease_id is not None and not (
                    isinstance(j_lease_id, str) and j_lease_id):
                raise ValueError(
                    f"{where}.lease_id must be null or a non-empty string")
            if j_job_id in redelivery_jobs:
                raise ValueError(
                    f"duplicate redelivery job in state: {j_job_id}")
            if j_device_id not in device_index:
                raise ValueError(
                    f"{where} references an unknown device: {j_device_id}")
            # Per-job recovery history. Older items predate the key: it is
            # absent and treated as empty. A present value must be a list
            # of well-formed records with globally unique recovery ids.
            raw_recoveries = raw.get("recoveries", [])
            if not isinstance(raw_recoveries, list):
                raise ValueError(f"{where}.recoveries must be a list")
            recoveries: List[RedeliveryJobRecovery] = []
            for rec_index, raw_recovery in enumerate(raw_recoveries):
                rec_where = f"{where}.recoveries[{rec_index}]"
                if not isinstance(raw_recovery, dict):
                    raise ValueError(f"{rec_where} must be an object")
                if list(raw_recovery) != ["recovery_id", "lease_id"]:
                    raise ValueError(
                        f"{rec_where} must have exactly the keys "
                        f"'recovery_id', 'lease_id' in order")
                rec_id = raw_recovery["recovery_id"]
                rec_lease_id = raw_recovery["lease_id"]
                if not (isinstance(rec_id, str) and rec_id):
                    raise ValueError(
                        f"{rec_where}.recovery_id must be a non-empty "
                        f"string")
                if rec_lease_id is not None and not (
                        isinstance(rec_lease_id, str) and rec_lease_id):
                    raise ValueError(
                        f"{rec_where}.lease_id must be null or a "
                        f"non-empty string")
                # The live entry point forbids recovery_id == this job's id
                # unconditionally (it would collide with the dispatch
                # lease), so no recovery record may reuse its own job_id.
                if rec_id == j_job_id:
                    raise ValueError(
                        f"{rec_where}.recovery_id must not equal the "
                        f"job's own dispatch lease id (job_id) {j_job_id}")
                # A non-empty recovery always leases under its own
                # recovery_id; an empty one records null.
                if rec_lease_id is not None \
                        and rec_lease_id != rec_id:
                    raise ValueError(
                        f"{rec_where}.lease_id must equal its recovery_id")
                if rec_id in recovery_ids_seen:
                    raise ValueError(
                        f"recovery_id repeats across redelivery jobs: "
                        f"{rec_id}")
                recovery_ids_seen.add(rec_id)
                recoveries.append(RedeliveryJobRecovery(
                    recovery_id=rec_id, lease_id=rec_lease_id))

            # Cancellation fields. Older four/five-key items predate them:
            # absent means null (never cancelled), and an explicit null is
            # accepted identically. The two must agree with the state: a
            # ``cancelled`` job carries a non-empty cancellation_id and a
            # canonical UTC cancelled_at, while every other state carries
            # neither. Any contradiction refuses startup.
            j_cancellation_id = raw.get("cancellation_id")
            j_cancelled_at = raw.get("cancelled_at")
            if j_cancellation_id is not None and not (
                    isinstance(j_cancellation_id, str)
                    and j_cancellation_id):
                raise ValueError(
                    f"{where}.cancellation_id must be null or a non-empty "
                    f"string")
            if j_cancelled_at is not None \
                    and not _is_utc_microsecond_iso(j_cancelled_at):
                raise ValueError(
                    f"{where}.cancelled_at must be null or a UTC ISO-8601 "
                    f"timestamp with six microsecond digits and a +00:00 "
                    f"offset")
            if j_state == REDELIVERY_JOB_CANCELLED:
                if j_cancellation_id is None or j_cancelled_at is None:
                    raise ValueError(
                        f"{where} is cancelled but is missing its "
                        f"cancellation_id/cancelled_at")
            elif j_cancellation_id is not None or j_cancelled_at is not None:
                raise ValueError(
                    f"{where} is not cancelled but carries "
                    f"cancellation_id/cancelled_at")

            dispatch_binding = lease_index.get(j_job_id)
            dispatch_completion = dispatch_binding[5] \
                if dispatch_binding is not None else None
            if not recoveries:
                # No recoveries: the original four-state rules against the
                # dispatch lease (whose id is the job_id).
                if j_state == REDELIVERY_JOB_PENDING:
                    if j_lease_id is not None:
                        raise ValueError(
                            f"{where} is pending but carries a lease_id")
                elif j_state == REDELIVERY_JOB_RUNNING:
                    if j_lease_id != j_job_id or dispatch_binding is None \
                            or dispatch_binding[0] != j_device_id \
                            or dispatch_completion is not None:
                        raise ValueError(
                            f"{where} is running but has no matching "
                            f"uncompleted dispatch lease owned by "
                            f"{j_device_id}")
                elif j_state == REDELIVERY_JOB_SUCCEEDED:
                    if j_lease_id is not None and (
                            dispatch_binding is None
                            or dispatch_binding[0] != j_device_id
                            or dispatch_completion is None
                            or dispatch_completion[1] != "delivered"):
                        raise ValueError(
                            f"{where} is succeeded but its lease did not "
                            f"complete 'delivered'")
                elif j_state == REDELIVERY_JOB_CANCELLED:
                    # A cancel while pending keeps no lease; a cancel while
                    # running releases the dispatch lease, so it must exist,
                    # belong to this device, carry a release timestamp and
                    # never have completed.
                    if j_lease_id is not None and (
                            j_lease_id != j_job_id
                            or dispatch_binding is None
                            or dispatch_binding[0] != j_device_id
                            or dispatch_completion is not None
                            or dispatch_binding[3] is None):
                        raise ValueError(
                            f"{where} is cancelled but its dispatch lease "
                            f"is missing, owned by another device, completed "
                            f"or not released")
                else:  # REDELIVERY_JOB_FAILED
                    if j_lease_id is None or dispatch_binding is None \
                            or dispatch_binding[0] != j_device_id \
                            or dispatch_completion is None \
                            or dispatch_completion[1] != "failed":
                        raise ValueError(
                            f"{where} is failed but its lease did not "
                            f"complete 'failed'")
            else:
                # A recovery history implies the job was dispatched: the
                # dispatch lease exists, belongs to this device and was
                # never completed (completing it would have terminated the
                # job before any recovery; a recovery only follows an
                # expired or released lease).
                if dispatch_binding is None \
                        or dispatch_binding[0] != j_device_id \
                        or dispatch_completion is not None:
                    raise ValueError(
                        f"{where} carries recoveries but has no matching "
                        f"uncompleted dispatch lease owned by "
                        f"{j_device_id}")
                if j_state == REDELIVERY_JOB_PENDING:
                    raise ValueError(
                        f"{where} is pending but carries recoveries")
                # A null-leased (empty) recovery ends the job succeeded;
                # only the last record may be null and then the job must be
                # succeeded with no current lease. Every non-null record
                # names a lease that exists and belongs to this device, and
                # every record but the last must be uncompleted (an
                # expired/released lease, never one that terminated the
                # job). A cancelled job is checked after the loop: its last
                # recovery lease must be the released current lease.
                for rec_position, record in enumerate(recoveries):
                    is_last = rec_position == len(recoveries) - 1
                    if record.lease_id is None:
                        if not (is_last
                                and j_state == REDELIVERY_JOB_SUCCEEDED):
                            raise ValueError(
                                f"{where}.recoveries[{rec_position}] is an "
                                f"empty recovery that may only end the job "
                                f"as its final record")
                        continue
                    rec_binding = lease_index.get(record.lease_id)
                    if rec_binding is None \
                            or rec_binding[0] != j_device_id:
                        raise ValueError(
                            f"{where}.recoveries[{rec_position}] names an "
                            f"unknown lease not owned by {j_device_id}: "
                            f"{record.lease_id}")
                    rec_completion = rec_binding[5]
                    if not is_last and rec_completion is not None:
                        raise ValueError(
                            f"{where}.recoveries[{rec_position}] completed "
                            f"before a later recovery")
                last_record = recoveries[-1]
                if j_state == REDELIVERY_JOB_RUNNING:
                    if last_record.lease_id is None \
                            or j_lease_id != last_record.lease_id:
                        raise ValueError(
                            f"{where} is running but its lease_id does not "
                            f"match its latest recovery lease")
                    last_binding = lease_index.get(j_lease_id)
                    if last_binding is None or last_binding[5] is not None:
                        raise ValueError(
                            f"{where} is running but its current recovery "
                            f"lease is missing or already completed")
                elif j_state == REDELIVERY_JOB_CANCELLED:
                    # A cancel while running after one or more recoveries
                    # releases the current (last, non-empty) recovery lease:
                    # it must be the job's lease_id, exist for this device,
                    # never have completed and carry a release timestamp.
                    if last_record.lease_id is None \
                            or j_lease_id != last_record.lease_id:
                        raise ValueError(
                            f"{where} is cancelled but its lease_id does "
                            f"not match its latest recovery lease")
                    last_binding = lease_index.get(j_lease_id)
                    if last_binding is None \
                            or last_binding[0] != j_device_id \
                            or last_binding[5] is not None \
                            or last_binding[3] is None:
                        raise ValueError(
                            f"{where} is cancelled but its current recovery "
                            f"lease is missing, completed or not released")
                elif j_state == REDELIVERY_JOB_SUCCEEDED:
                    if last_record.lease_id is None:
                        if j_lease_id is not None:
                            raise ValueError(
                                f"{where} is succeeded after an empty "
                                f"recovery but still carries a lease_id")
                    else:
                        if j_lease_id != last_record.lease_id:
                            raise ValueError(
                                f"{where} is succeeded but its lease_id "
                                f"does not match its latest recovery lease")
                        last_binding = lease_index.get(j_lease_id)
                        if last_binding is None \
                                or last_binding[5] is None \
                                or last_binding[5][1] != "delivered":
                            raise ValueError(
                                f"{where} is succeeded but its recovery "
                                f"lease did not complete 'delivered'")
                else:  # REDELIVERY_JOB_FAILED
                    if last_record.lease_id is None \
                            or j_lease_id != last_record.lease_id:
                        raise ValueError(
                            f"{where} is failed but its lease_id does not "
                            f"match its latest recovery lease")
                    last_binding = lease_index.get(j_lease_id)
                    if last_binding is None \
                            or last_binding[5] is None \
                            or last_binding[5][1] != "failed":
                        raise ValueError(
                            f"{where} is failed but its recovery lease did "
                            f"not complete 'failed'")
            redelivery_jobs[j_job_id] = RedeliveryJob(
                job_id=j_job_id, device_id=j_device_id, state=j_state,
                lease_id=j_lease_id, recoveries=recoveries,
                cancellation_id=j_cancellation_id,
                cancelled_at=j_cancelled_at)

        # Per-device group-session delivery records. Older version-1 files
        # predate the section: it is absent and treated as empty. A present
        # section holds one record per (session, message, device); every
        # record must name a stored group session, one of its stored
        # messages, and a device frozen into the session's member snapshot,
        # with the attempts counter agreeing with the dedup id list and the
        # ack cursor mirroring the message sequence exactly as the live
        # ack path writes it. Any contradiction refuses startup.
        group_delivery: Dict[Tuple[str, str, str], MessageDelivery] = {}
        for index, raw in enumerate(raw_group_delivery):
            where = f"group_delivery[{index}]"
            if not isinstance(raw, dict):
                raise ValueError(f"{where} must be an object")
            try:
                g_session = raw["session_id"]
                g_message = raw["message_id"]
                g_device = raw["device_id"]
                attempts = raw["attempts"]
                attempt_ids = raw["attempt_ids"]
                acked = raw["acked"]
                ack_sequence = raw["ack_sequence"]
            except KeyError as error:
                raise ValueError(
                    f"{where} missing field: {error.args[0]}") from None
            if not (isinstance(g_session, str) and g_session
                    and isinstance(g_message, str) and g_message
                    and isinstance(g_device, str) and g_device):
                raise ValueError(
                    f"{where} session_id/message_id/device_id must be "
                    "non-empty strings")
            if not isinstance(attempts, int) or isinstance(attempts, bool) \
                    or attempts < 0:
                raise ValueError(
                    f"{where} attempts must be a non-negative integer")
            if not isinstance(attempt_ids, list) or not all(
                    isinstance(value, str) and value for value in attempt_ids):
                raise ValueError(
                    f"{where} attempt_ids must be a list of non-empty strings")
            if len(set(attempt_ids)) != len(attempt_ids):
                raise ValueError(
                    f"{where} attempt_ids must not contain duplicates")
            # The counter is derived state: it must agree with the id list.
            if attempts != len(attempt_ids):
                raise ValueError(
                    f"{where} attempts must equal the number of attempt_ids")
            if not isinstance(acked, bool):
                raise ValueError(f"{where} acked must be a boolean")
            if not isinstance(ack_sequence, int) \
                    or isinstance(ack_sequence, bool) or ack_sequence < 0:
                raise ValueError(
                    f"{where} ack_sequence must be a non-negative integer")
            gkey = (g_session, g_message, g_device)
            if gkey in group_delivery:
                raise ValueError(
                    f"duplicate group delivery record in state: {gkey}")
            # The record belongs to a stored group session (never a 1:1 one)
            # and to a message stored in that session's stream.
            target_session = group_sessions.get(g_session)
            if target_session is None:
                raise ValueError(
                    f"{where} references an unknown group session: "
                    f"{g_session}")
            target = next((m for m in messages.get(g_session, [])
                           if m.message_id == g_message), None)
            if target is None:
                raise ValueError(
                    f"{where} references an unknown message: {gkey}")
            # The device must be registered (revocation after the record was
            # written is legitimate history, so revocation is not a bar) and
            # frozen into the member snapshot; the message's own sender never
            # holds delivery state. Later roster changes are irrelevant here.
            if g_device not in device_index:
                raise ValueError(
                    f"{where} references an unknown device: {g_device}")
            if g_device not in target_session.members:
                raise ValueError(
                    f"{where} device is not a frozen member of the group "
                    f"session: {g_device}")
            if g_device == target.sender_device_id:
                raise ValueError(
                    f"{where} device is the message sender: {g_device}")
            # The ack cursor mirrors the message once acked and is 0 before.
            if acked:
                if ack_sequence != target.sequence:
                    raise ValueError(
                        f"{where} ack_sequence must equal the message "
                        f"sequence {target.sequence}")
            elif ack_sequence != 0:
                raise ValueError(
                    f"{where} ack_sequence must be 0 while not acked")
            group_delivery[gkey] = MessageDelivery(
                attempts=attempts, attempt_ids=set(attempt_ids), acked=acked,
                ack_sequence=ack_sequence)

        # Per-device group-session sync cursors. Older version-1 files predate
        # the section: it is absent and treated as empty (every device starts
        # at cursor 0). A present section must be a list of well-formed
        # records; anything else makes the document malformed.
        group_sync_cursors: Dict[Tuple[str, str], GroupSyncCursor] = {}
        for index, raw in enumerate(raw_group_sync_cursors):
            if not isinstance(raw, dict):
                raise ValueError(
                    f"group_sync_cursors[{index}] must be an object")
            try:
                sid = raw["session_id"]
                did = raw["device_id"]
                cursor = raw["cursor"]
                updated_at = raw["updated_at"]
            except KeyError as error:
                raise ValueError(
                    f"group_sync_cursors[{index}] missing field: "
                    f"{error.args[0]}") from None
            if not (isinstance(sid, str) and sid
                    and isinstance(did, str) and did):
                raise ValueError(
                    f"group_sync_cursors[{index}] session_id/device_id must "
                    "be non-empty strings")
            if not isinstance(cursor, int) or isinstance(cursor, bool) \
                    or cursor < 0:
                raise ValueError(
                    f"group_sync_cursors[{index}] cursor must be a "
                    "non-negative integer")
            if not isinstance(updated_at, str) or not updated_at:
                raise ValueError(
                    f"group_sync_cursors[{index}] updated_at must be a "
                    "non-empty string")
            ckey = (sid, did)
            if ckey in group_sync_cursors:
                raise ValueError(
                    f"duplicate group sync cursor in state: {ckey}")
            # Association checks: a cursor belongs to a stored group session
            # and to a registered device frozen into that session's member
            # snapshot; its value cannot have passed the session's largest
            # stored sequence. Any mismatch is a malformed document — startup
            # is refused rather than silently dropping or clamping the record.
            target_session = group_sessions.get(sid)
            if target_session is None:
                raise ValueError(
                    f"group_sync_cursors[{index}] references an unknown "
                    f"group session: {sid}")
            device_key = device_index.get(did)
            target_device = devices.get(device_key) if device_key else None
            if target_device is None:
                raise ValueError(
                    f"group_sync_cursors[{index}] references an unknown "
                    f"device: {did}")
            if did not in target_session.members:
                raise ValueError(
                    f"group_sync_cursors[{index}] device is not a frozen "
                    f"member of the group session: {did}")
            stream = messages.get(sid, [])
            max_sequence = stream[-1].sequence if stream else 0
            if cursor > max_sequence:
                raise ValueError(
                    f"group_sync_cursors[{index}] cursor {cursor} exceeds the "
                    f"session's max sequence {max_sequence}")
            group_sync_cursors[ckey] = GroupSyncCursor(
                cursor=cursor, updated_at=updated_at)

        # Per-device unified 1:1/group-session sync cursors written by
        # GET /v1/sessions/{id}/sync and its checkpoint route. Older files
        # predate the section: it is absent and treated as empty (every device
        # starts at cursor 0). A present section must be a list of
        # well-formed records — malformed records, a duplicate
        # (session_id, device_id), a dangling session/device reference or a
        # device that is not a session participant, a cursor past the
        # session's max sequence, or an empty updated_at all make the
        # document malformed and refuse startup rather than silently dropping
        # the record.
        message_sync_cursors: Dict[Tuple[str, str], MessageSyncCursor] = {}
        for index, raw in enumerate(raw_message_sync_cursors):
            if not isinstance(raw, dict):
                raise ValueError(
                    f"message_sync_cursors[{index}] must be an object")
            try:
                sid = raw["session_id"]
                did = raw["device_id"]
                cursor = raw["cursor"]
                updated_at = raw["updated_at"]
            except KeyError as error:
                raise ValueError(
                    f"message_sync_cursors[{index}] missing field: "
                    f"{error.args[0]}") from None
            if not (isinstance(sid, str) and sid
                    and isinstance(did, str) and did):
                raise ValueError(
                    f"message_sync_cursors[{index}] session_id/device_id must "
                    "be non-empty strings")
            if not isinstance(cursor, int) or isinstance(cursor, bool) \
                    or cursor < 0:
                raise ValueError(
                    f"message_sync_cursors[{index}] cursor must be a "
                    "non-negative integer")
            if not isinstance(updated_at, str) or not updated_at:
                raise ValueError(
                    f"message_sync_cursors[{index}] updated_at must be a "
                    "non-empty string")
            ckey = (sid, did)
            if ckey in message_sync_cursors:
                raise ValueError(
                    f"duplicate message sync cursor in state: {ckey}")
            # Association checks: a cursor belongs to a stored 1:1 or group
            # session and to a registered device allowed to read it (an
            # endpoint of a 1:1 session, or a frozen group member); its value
            # cannot have passed the session's largest stored sequence.
            target_11 = sessions.get(sid)
            target_group = group_sessions.get(sid)
            if target_11 is None and target_group is None:
                raise ValueError(
                    f"message_sync_cursors[{index}] references an unknown "
                    f"session: {sid}")
            device_key = device_index.get(did)
            target_device = devices.get(device_key) if device_key else None
            if target_device is None:
                raise ValueError(
                    f"message_sync_cursors[{index}] references an unknown "
                    f"device: {did}")
            if target_11 is not None:
                is_participant = did in (target_11.initiator_device_id,
                                         target_11.recipient_device_id)
            else:
                is_participant = did in target_group.members
            if not is_participant:
                raise ValueError(
                    f"message_sync_cursors[{index}] device is not a "
                    f"participant of the session: {did}")
            stream = messages.get(sid, [])
            max_sequence = stream[-1].sequence if stream else 0
            if cursor > max_sequence:
                raise ValueError(
                    f"message_sync_cursors[{index}] cursor {cursor} exceeds "
                    f"the session's max sequence {max_sequence}")
            message_sync_cursors[ckey] = MessageSyncCursor(
                cursor=cursor, updated_at=updated_at)

        # Per-device key-audit chains. Older version-1 files predate the
        # section: it is absent and treated as empty, with no chain
        # completeness enforced. A present section is fully validated —
        # field types, a reference to a registered device, the recomputed
        # event hash, per-device seq 1..N consecutiveness and prev_hash
        # linkage — and every chain is replayed from scratch: the replayed
        # identity key, pre-key order/public keys/revocation flags and the
        # device revocation flag must reproduce the devices section exactly.
        # When the section is present every registered device must have a
        # chain (this server appends ``registered`` at registration time).
        # Any contradiction makes the document malformed: startup is refused
        # and the file is left untouched.
        key_events: Dict[str, List[KeyEvent]] = {}
        if raw_key_events is not None:
            if not isinstance(raw_key_events, list):
                raise ValueError(
                    "state document has a malformed top-level section")
            events_by_device: Dict[str, List[KeyEvent]] = {}
            for index, raw in enumerate(raw_key_events):
                where = f"key_events[{index}]"
                if not isinstance(raw, dict):
                    raise ValueError(f"{where} must be an object")
                event_device_id = raw.get("device_id")
                event_type = raw.get("type")
                payload = raw.get("payload")
                prev_hash = raw.get("prev_hash")
                event_hash = raw.get("hash")
                created_at = raw.get("created_at")
                seq = raw.get("seq")
                if not isinstance(event_device_id, str) or not event_device_id:
                    raise ValueError(
                        f"{where}.device_id must be a non-empty string")
                if not isinstance(seq, int) or isinstance(seq, bool) \
                        or seq < 1:
                    raise ValueError(
                        f"{where}.seq must be a positive integer")
                if event_type not in KEY_EVENT_TYPES:
                    raise ValueError(
                        f"{where}.type must be one of "
                        f"{sorted(KEY_EVENT_TYPES)}")
                if not isinstance(payload, dict):
                    raise ValueError(f"{where}.payload must be an object")
                if not isinstance(prev_hash, str):
                    raise ValueError(f"{where}.prev_hash must be a string")
                if not isinstance(event_hash, str) or not event_hash:
                    raise ValueError(
                        f"{where}.hash must be a non-empty string")
                if not isinstance(created_at, str) or not created_at:
                    raise ValueError(
                        f"{where}.created_at must be a non-empty string")
                if event_device_id not in device_index:
                    raise ValueError(
                        f"{where} references an unknown device: "
                        f"{event_device_id}")
                if key_event_hash(event_device_id, seq, event_type, payload,
                                  prev_hash, created_at) != event_hash:
                    raise ValueError(
                        f"{where}.hash does not match the event contents")
                events_by_device.setdefault(event_device_id, []).append(
                    KeyEvent(device_id=event_device_id, seq=seq,
                             type=event_type, payload=payload,
                             prev_hash=prev_hash, hash=event_hash,
                             created_at=created_at))
            for chained_device in device_index:
                if chained_device not in events_by_device:
                    raise ValueError(
                        f"device has no key_events chain: {chained_device}")
            for event_device_id, chain in events_by_device.items():
                chain.sort(key=lambda event: event.seq)
                for position, event in enumerate(chain):
                    if event.seq != position + 1:
                        raise ValueError(
                            f"key_events chain of device {event_device_id} "
                            f"must run consecutively from seq 1")
                    expected_prev = chain[position - 1].hash if position else ""
                    if event.prev_hash != expected_prev:
                        raise ValueError(
                            f"key_events chain of device {event_device_id} "
                            f"has a broken prev_hash link at seq {event.seq}")
                key_events[event_device_id] = chain
                self._replay_key_events(
                    devices[device_index[event_device_id]], chain)

        with self._lock:
            self._devices = devices
            self._device_index = device_index
            self._sessions = sessions
            self._prekey_claims = prekey_claims
            self._prekey_batch_claims = prekey_batch_claims
            self._claim_session_bindings = claim_session_bindings
            self._batch_claim_session_bindings = batch_claim_session_bindings
            self._groups = groups
            self._group_sessions = group_sessions
            self._group_session_rotations = group_session_rotations
            self._rotation_by_predecessor = {
                rotation.predecessor_session_id: rotation
                for rotation in group_session_rotations.values()}
            self._rotation_by_successor = {
                rotation.successor_session_id: rotation
                for rotation in group_session_rotations.values()}
            self._messages = messages
            self._delivery = delivery
            self._group_delivery = group_delivery
            self._used_nonces = used_nonces
            self._group_sync_cursors = group_sync_cursors
            self._message_sync_cursors = message_sync_cursors
            self._message_submissions = message_submissions
            self._redelivery_jobs = redelivery_jobs
            self._key_events = key_events
            # A file without the section predates the audit chain: every
            # registered device is chainless and gets a lazily-built anchor
            # before the first persisted change. A present section has just
            # been fully validated and every device carries a chain, so
            # nothing is pending.
            self._pending_anchor_devices = (
                list(device_index) if raw_key_events is None else None)

    def integrity_evaluate(
            self,
            read_payload: Callable[[], Tuple[Dict[str, Any], int]],
            expected_commit_seq: Callable[[], int],
            ) -> Tuple[int, str]:
        """Verify the durable document against the live state under the lock.

        Runs *read_payload* — which reads, parses and envelope-checks the
        ``version=1`` state file and returns ``(payload, commit_seq)`` with
        ``version``/``commit_seq`` stripped (a legacy file without the field
        reads as generation 0) — while holding the same store lock every
        mutation and key-event append uses, so neither the in-memory state
        nor the file (rewritten only under this lock) can advance between
        reading the file and comparing it. *expected_commit_seq* is also
        evaluated under this lock and must equal the on-disk generation; the
        payload is then restored into a fresh store and its canonical
        snapshot is compared, section by section, with the live store's
        canonical snapshot. The returned hash is the SHA-256 of the on-disk
        state's canonical compact JSON.

        Raises :class:`ValueError`/`:class:`TypeError` for any semantic
        validation failure, a generation mismatch, or any divergence between
        the file and memory; nothing is mutated in that case (the restored
        copy is discarded).
        """
        with self._lock:
            payload, commit_seq = read_payload()
            if not isinstance(commit_seq, int) or isinstance(
                    commit_seq, bool) or commit_seq < 0:
                raise ValueError(
                    "state document 'commit_seq' must be a non-negative "
                    "integer")
            expected = expected_commit_seq()
            if commit_seq != expected:
                raise ValueError(
                    "state file commit_seq "
                    f"{commit_seq} does not match the last committed "
                    f"generation {expected}")
            if not isinstance(payload, dict):
                raise ValueError("state document must be a JSON object")
            unknown_sections = set(payload) - set(INTEGRITY_SECTION_KEYS)
            if unknown_sections:
                raise ValueError(
                    "state document has unknown sections: "
                    f"{sorted(unknown_sections)}")
            live = canonical_integrity_snapshot(self.snapshot_state())
            restored_store = DeviceStore()
            restored_store.restore_state(copy.deepcopy(payload))
            on_disk = canonical_integrity_snapshot(
                restored_store.snapshot_state())
            if on_disk != live:
                raise ValueError(
                    "state file is inconsistent with the in-memory snapshot")
            return commit_seq, integrity_state_hash(on_disk)
