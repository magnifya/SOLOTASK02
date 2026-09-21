"""Domain model: devices and their signed pre-keys.

Only public keys and identifiers are represented here; plaintext messages and
private keys are never stored.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set


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
class MessageDelivery:
    """Reliable-delivery state for one stored message of a session.

    Tracks the recipient-side retry attempts (keyed/deduped by the client's
    non-empty ``attempt_id``), whether the recipient has acknowledged the
    message, and the per-recipient ack sequence cursor. Only identifiers and
    counters are kept — never message plaintext or keys.
    """

    attempts: int = 0
    attempt_ids: Set[str] = field(default_factory=set)
    acked: bool = False
    ack_sequence: int = 0


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
