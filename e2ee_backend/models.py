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


@dataclass
class Group:
    """A mutable member group created by one of its devices.

    ``members`` holds the device ids in stable insertion order (the order in
    which they were supplied when the group was created or later joined). Every successful membership change increments
    ``revision`` (starting at 1); the creator is fixed for the group's life.
    Only identifiers are retained.
    """

    group_id: str
    creator_device_id: str
    members: List[str] = field(default_factory=list)
    revision: int = 1
    created_at: str = field(default_factory=utc_now_iso)


@dataclass
class GroupSession:
    """A group session whose membership is frozen at creation time.

    ``members`` is an ordered snapshot of the group's members then; later
    group membership changes never alter it. Message reads in a group session
    are restricted to these frozen members, even if a member device is later
    revoked. Only public material (the initiator's ephemeral public key) and
    identifiers are retained.
    """

    session_id: str
    group_id: str
    initiator_device_id: str
    ephemeral_key: str
    members: List[str] = field(default_factory=list)
    revision: int = 1
    created_at: str = field(default_factory=utc_now_iso)
