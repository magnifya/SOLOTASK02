"""Per-device hash-chained key audit log.

Every key-lifecycle change appends an immutable :class:`~.models.KeyEvent`
to the owning device's chain. Chains never cross devices: a device's
``seq`` starts at 1, its first ``prev_hash`` is the empty string and each
following event carries its predecessor's ``hash``.

Event payload shapes
--------------------

``registered``
    ``{"prekeys": [{"key_id", "public_key"}, ...]}`` — the pre-keys of the
    registration body in their submitted order.

``identity_rotated``
    ``{"old_identity_key", "new_identity_key"}``.

``prekey_added`` / ``prekey_revoked``
    ``{"key_id", "public_key"}``.

``device_revoked``
    ``{}`` — the revocation retires the device and all of its pre-keys.

The event ``hash`` is the lowercase-hex SHA-256 of the canonical JSON of
every event field except ``hash``: object keys sorted, compact separators,
non-ASCII characters left as-is (``ensure_ascii=False``), UTF-8 encoded.
"""
from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Dict

from .models import KeyEvent, utc_now_iso

#: Audit event types, appended on the matching key-lifecycle transition.
EVENT_REGISTERED = "registered"
EVENT_IDENTITY_ROTATED = "identity_rotated"
EVENT_PREKEY_ADDED = "prekey_added"
EVENT_PREKEY_REVOKED = "prekey_revoked"
EVENT_DEVICE_REVOKED = "device_revoked"

#: Every event type understood by this build, in lifecycle-neutral order.
KEY_EVENT_TYPES = frozenset({
    EVENT_REGISTERED,
    EVENT_IDENTITY_ROTATED,
    EVENT_PREKEY_ADDED,
    EVENT_PREKEY_REVOKED,
    EVENT_DEVICE_REVOKED,
})

#: Fields carried by (and hashed over for) every audit event except ``hash``.
KEY_EVENT_FIELDS = (
    "device_id", "seq", "type", "payload", "prev_hash", "created_at")


def canonical_event_json(event: Dict[str, Any]) -> bytes:
    """Return the canonical UTF-8 JSON bytes hashed for one audit event.

    The ``hash`` field (if present) is dropped, the remaining object's keys
    are sorted recursively, separators are compact, and non-ASCII characters
    survive verbatim (the input is UTF-8 encoded, never ASCII-escaped).
    """
    body = {name: copy.deepcopy(event[name])
            for name in KEY_EVENT_FIELDS if name in event}
    return json.dumps(body, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def compute_event_hash(event: Dict[str, Any]) -> str:
    """Return the lowercase-hex SHA-256 of an event's canonical JSON."""
    return hashlib.sha256(canonical_event_json(event)).hexdigest()


def build_key_event(device_id: str, seq: int, event_type: str,
                    payload: Dict[str, Any], prev_hash: str,
                    created_at: str | None = None) -> KeyEvent:
    """Construct one audit event with its ``hash`` filled in."""
    event = KeyEvent(
        device_id=device_id,
        seq=seq,
        type=event_type,
        payload=copy.deepcopy(payload),
        prev_hash=prev_hash,
        hash="",
        created_at=created_at if created_at is not None else utc_now_iso(),
    )
    event.hash = compute_event_hash(key_event_to_dict(event))
    return event


def key_event_to_dict(event: KeyEvent) -> Dict[str, Any]:
    """Return the plain dict form of an event (same shape as the snapshot)."""
    return {
        "device_id": event.device_id,
        "seq": event.seq,
        "type": event.type,
        "payload": copy.deepcopy(event.payload),
        "prev_hash": event.prev_hash,
        "hash": event.hash,
        "created_at": event.created_at,
    }
