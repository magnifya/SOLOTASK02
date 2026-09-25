"""Domain model: devices and their signed pre-keys.

Only public keys and identifiers are represented here; plaintext messages and
private keys are never stored.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string (``...+00:00``)."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class SignedPreKey:
    """A signed pre-key: an identifier plus a public key, revocable.

    Pre-keys are kept in insertion order; revoked keys remain stored (so the
    fact of revocation is durable) but are excluded from public listings.
    """

    key_id: str
    public_key: str
    revoked: bool = False
    #: True once the key has been handed out by a (committed) pre-key claim.
    #: A consumed key behaves like a revoked one for listings and claims, but
    #: the distinct flag records *why* it left the available pool. Like
    #: revocation, consumption is durable and never reset.
    consumed: bool = False


@dataclass
class Device:
    """A registered device belonging to a user."""

    user_id: str
    device_id: str
    identity_key: str
    registered_at: str = field(default_factory=utc_now_iso)
    #: Timestamp of the last identity-key rotation. A device starts unrotated,
    #: so this equals ``registered_at`` until the key is actually replaced.
    rotated_at: Optional[str] = None
    prekeys: List[SignedPreKey] = field(default_factory=list)
    revoked: bool = False

    def __post_init__(self) -> None:
        if self.rotated_at is None:
            self.rotated_at = self.registered_at


@dataclass
class KeyEvent:
    """One link of a device's append-only key-audit chain.

    Every committed key-material change of a device (registration, identity
    rotation, pre-key add/revoke, device revoke) appends exactly one event.
    ``seq`` starts at 1 and advances by one per event; ``prev_hash`` is empty
    for the first event and afterwards chains to the previous event's
    ``hash``. ``hash`` is the lowercase hex SHA-256 of the UTF-8 canonical
    JSON of the event without the ``hash`` field (keys sorted, compact
    separators, Unicode written as-is). Only public key material and
    identifiers are ever recorded.
    """

    device_id: str
    seq: int
    type: str
    payload: Dict[str, Any]
    prev_hash: str
    hash: str
    created_at: str = field(default_factory=utc_now_iso)


@dataclass
class PreKeyClaim:
    """The durable record of one successful one-time pre-key claim.

    A claim hands out the recipient's first un-revoked, un-consumed pre-key
    and is idempotent on ``claim_id``: repeating a claim returns this same
    record and never consumes a second key. The record freezes the public
    material returned to the claimant (the recipient's identity key and the
    claimed pre-key's public key) together with the UTC ``claimed_at``
    timestamp, so a replayed claim returns byte-identical values even if the
    device later rotates or revokes.
    """

    claim_id: str
    recipient_device_id: str
    key_id: str
    identity_key: str
    public_key: str
    claimed_at: str = field(default_factory=utc_now_iso)


@dataclass
class BatchClaimDevice:
    """One frozen device entry of a successful multi-device batch claim.

    Freezes the device's (claimed-at) identity key together with the one
    pre-key handed out for that device, so a replayed batch claim returns
    byte-identical material even if the device later rotates or revokes.
    """

    device_id: str
    identity_key: str
    key_id: str
    public_key: str


@dataclass
class PreKeyBatchClaim:
    """The durable record of one successful user-wide batch pre-key claim.

    A batch claim enumerates every active (un-revoked) device of a user in
    registration order and hands out each device's first un-revoked,
    un-consumed pre-key in one atomic transaction. The record is idempotent on
    ``claim_id`` (a shared, globally unique namespace with single claims):
    repeating it returns this same frozen record and never consumes a second
    key. ``devices`` stays in the registration order captured at claim time.
    """

    claim_id: str
    user_id: str
    devices: List[BatchClaimDevice] = field(default_factory=list)
    claimed_at: str = field(default_factory=utc_now_iso)


@dataclass
class ClaimSessionBinding:
    """Durable binding of one pre-key claim to the session it established.

    Exactly one session may ever be established per ``claim_id``; this record
    is written together with that session and makes a repeated
    ``POST /v1/sessions/from-claim`` a conflict. It freezes the claimed
    recipient/pre-key material the session was built from so the one-session
    guarantee survives a restart.
    """

    claim_id: str
    session_id: str
    recipient_device_id: str
    prekey_id: str
    identity_key: str
    public_key: str
    created_at: str = field(default_factory=utc_now_iso)


@dataclass
class BatchClaimSessionEntry:
    """One frozen recipient entry of a batch-claim session set.

    Repeats the per-device material frozen by the originating batch claim
    (recipient device, claimed pre-key and the public material), so the
    recovery check can compare the binding against both the batch claim
    record and the session snapshot without looking at any device's current
    (possibly rotated) identity key.
    """

    recipient_device_id: str
    prekey_id: str
    identity_key: str
    public_key: str
    session_id: str


@dataclass
class BatchClaimSessionBinding:
    """Durable binding of one batch claim to the session set it established.

    A batch claim establishes at most one session per claimed device, and
    the whole set is created together with this one record in a single
    locked transaction (all sessions or none). ``entries`` keeps the batch
    claim's frozen device order. Only the initiator's id, the frozen
    recipient/pre-key material, session ids and timestamps are retained.
    """

    claim_id: str
    initiator_device_id: str
    entries: List[BatchClaimSessionEntry] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now_iso)


@dataclass
class Message:
    """An encrypted message envelope stored inside a session.

    Only the ciphertext and its metadata are retained: the server never sees
    plaintext and never holds the keys that could produce it. ``sequence`` is
    the sender-chosen position in the session's message stream (starting at 1,
    strictly consecutive); ``created_at`` is the server's receive timestamp.
    """

    session_id: str
    sender_device_id: str
    message_id: str
    sequence: int
    nonce: str
    ciphertext: str
    created_at: str = field(default_factory=utc_now_iso)


@dataclass
class MessageSubmission:
    """The durable idempotency record of one committed message submission.

    Written together with the message it accepted (same locked transaction),
    keyed by the client-chosen ``request_id`` (globally unique). The record
    freezes the six envelope fields plus the message's ``created_at`` so a
    replayed submission returns the first response byte-identically, even if
    the sender device is revoked afterwards.
    """

    request_id: str
    session_id: str
    sender_device_id: str
    message_id: str
    sequence: int
    nonce: str
    ciphertext: str
    created_at: str = field(default_factory=utc_now_iso)


@dataclass
class MessageLease:
    """One 1:1-inbox redelivery lease held on a message.

    A successful ``POST /v1/devices/{device_id}/inbox/claim`` leases up to
    ``limit`` still-unleased (or lease-expired) unacked messages for 30
    seconds. The lease is recorded on every leased message's
    :class:`MessageDelivery` record (inside ``delivery[].leases``), so the
    same ``lease_id`` — globally unique — is durably bound to exactly one
    device, one limit and one ``leased_until`` timestamp. Only identifiers,
    the small integer limit and the UTC deadline are retained.
    """

    lease_id: str
    limit: int
    leased_until: str


@dataclass
class MessageDelivery:
    """Reliable-delivery state for one stored message of a session.

    Tracks the recipient-side retry attempts (keyed/deduped by the client's
    non-empty ``attempt_id``), whether the recipient has acknowledged the
    message, the per-recipient ack sequence cursor, and the inbox redelivery
    leases currently or previously held on the message (active and expired
    ones alike — an expired lease is history that a new ``lease_id`` may
    supersede but never erase). Only identifiers and counters are kept —
    never message plaintext or keys.
    """

    attempts: int = 0
    attempt_ids: Set[str] = field(default_factory=set)
    acked: bool = False
    ack_sequence: int = 0
    leases: List[MessageLease] = field(default_factory=list)


@dataclass
class Group:
    """A member group created by one (creator) device.

    Membership keeps insertion order (creator first, then each added member)
    so the public ``members`` listing is stable across repeated reads. The
    ``revision`` starts at 1 and advances by one on every successful
    membership change; the creator is fixed for the group's lifetime.
    """

    group_id: str
    creator_device_id: str
    members: List[str] = field(default_factory=list)
    revision: int = 1
    created_at: str = field(default_factory=utc_now_iso)


@dataclass
class GroupSession:
    """An immutable snapshot of a negotiated group session.

    The member list is frozen at creation time: later group membership
    changes never alter it, and only frozen members may read messages sent
    into the session. Only public material (the initiator's ephemeral public
    key) and identifiers are retained.
    """

    session_id: str
    group_id: str
    initiator_device_id: str
    ephemeral_key: str
    members: List[str]
    revision: int
    created_at: str = field(default_factory=utc_now_iso)


@dataclass
class GroupSessionRotation:
    """The durable record of one group-session rotation.

    Rotating a predecessor group session creates a fresh group session whose
    member list and revision are frozen from the group's state at commit
    time; the predecessor snapshot itself is never altered. The record is
    idempotent on ``rotation_id`` for the same predecessor: replaying it
    returns the original successor. A given predecessor may be rotated at
    most once (no forks), and a given ``rotation_id`` may succeed for at
    most one predecessor.
    """

    rotation_id: str
    predecessor_session_id: str
    successor_session_id: str
    group_id: str
    actor_device_id: str
    revision: int
    members: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now_iso)


@dataclass
class GroupSyncCursor:
    """Per-device read cursor for syncing one group session's messages.

    Records the sequence up to which a frozen member device has consumed the
    session's message stream, together with the timestamp of the last forward
    checkpoint (``updated_at``). A device has one cursor per group session;
    the initial cursor sits at sequence 0 and is created lazily on the first
    check-pointed advance. Only the integer cursor and its timestamp are kept.
    """

    cursor: int = 0
    updated_at: str = field(default_factory=utc_now_iso)


@dataclass
class MessageSyncCursor:
    """Per-device read cursor for syncing one session's messages (1:1 or group).

    Records the sequence up to which a device has consumed a session's message
    stream, together with the timestamp of the last forward checkpoint
    (``updated_at``). A device has one cursor per session it may read; the
    initial cursor sits at sequence 0 and is created lazily on the first
    check-pointed advance. Only the integer cursor and its timestamp are kept.
    """

    cursor: int = 0
    updated_at: str = field(default_factory=utc_now_iso)


@dataclass
class Session:
    """An immutable snapshot of a negotiated session.

    Only public material is retained: the initiator's ephemeral public key,
    the recipient's identity public key and the public key of the signed
    pre-key that was used. Private keys, shared secrets and plaintext messages
    are never stored. The record is frozen at creation time, so later
    revocations of devices or pre-keys do not alter it.
    """

    session_id: str
    initiator_device_id: str
    recipient_device_id: str
    prekey_id: str
    ephemeral_key: str
    identity_key: str
    public_key: str
    created_at: str = field(default_factory=utc_now_iso)
