"""Domain model: devices and their signed pre-keys.

Only public keys and identifiers are represented here; plaintext messages and
private keys are never stored.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List


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
    prekeys: List[SignedPreKey] = field(default_factory=list)
    revoked: bool = False


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
