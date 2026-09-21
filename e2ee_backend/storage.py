"""Thread-safe in-memory device storage.

Devices live for the lifetime of the server process. Only public keys and
identifiers are retained. Reads return pre-key ids in the exact insertion
order, so repeated requests list them identically.
"""
from __future__ import annotations

import threading
import uuid
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .models import (
    BatchClaimDevice,
    ClaimSessionBinding,
    Device,
    Group,
    GroupSession,
    GroupSyncCursor,
    Message,
    MessageDelivery,
    PreKeyBatchClaim,
    PreKeyClaim,
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

#: Outcome codes for a failed atomic message append.
MESSAGE_SESSION_UNKNOWN = "session_unknown"
MESSAGE_SENDER_INACTIVE = "sender_inactive"
MESSAGE_DUPLICATE_ID = "duplicate_message_id"
MESSAGE_BAD_SEQUENCE = "bad_sequence"
MESSAGE_DUPLICATE_NONCE = "duplicate_nonce"

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


class DeviceStore:
    """In-memory store keyed by ``(user_id, device_id)``.

    Device ids are additionally indexed globally, because the public GET route
    addresses a device by ``device_id`` alone.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
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
        self._groups: Dict[str, Group] = {}
        self._group_sessions: Dict[str, GroupSession] = {}
        # Per-device group-session read cursors, keyed by
        # (session_id, device_id); created lazily on the first advance.
        self._group_sync_cursors: Dict[Tuple[str, str], GroupSyncCursor] = {}
        self._messages: Dict[str, List[Message]] = {}
        # Per-session set of nonces already accepted for replay protection.
        # Keyed independently of the streams so a nonce is scoped to a session.
        self._used_nonces: Dict[str, Set[str]] = {}
        # Delivery state keyed by (session_id, message_id).
        self._delivery: Dict[Tuple[str, str], MessageDelivery] = {}
        # Called (under the lock) after any state mutation, for persistence.
        self.on_change: Optional[Callable[[], None]] = None

    def _notify_change(self) -> None:
        """Invoke the persistence hook after a committed mutation."""
        if self.on_change is not None:
            self.on_change()

    def add_device(self, device: Device) -> bool:
        """Insert a device.

        Return ``False`` (and store nothing) when the same ``device_id`` is
        already registered, whether under the same user or another one.
        """
        key = (device.user_id, device.device_id)
        with self._lock:
            if key in self._devices or device.device_id in self._device_index:
                return False
            self._devices[key] = device
            self._device_index[device.device_id] = key
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
        """Mark one of the device's pre-keys revoked. Return ``False`` if absent."""
        with self._lock:
            for prekey in device.prekeys:
                if prekey.key_id == key_id:
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
            device.revoked = True
            for prekey in device.prekeys:
                prekey.revoked = True
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
                    prekey.revoked = True
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
                device.identity_key = identity_key
                device.rotated_at = utc_now_iso()
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
            prekey = SignedPreKey(key_id=key_id, public_key=public_key)
            device.prekeys.append(prekey)
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
            self._notify_change()
            return message

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
                         ) -> Tuple[Session, Message]:
        """Resolve and authorize a (session, message, recipient) triple.

        Must be called while holding the store lock. Raises
        :class:`DeliveryError` with the mapped reason on any failure.
        """
        session = self._sessions.get(session_id)
        if session is None:
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
        if device_id != session.recipient_device_id:
            raise DeliveryError(DELIVERY_DEVICE_MISMATCH)
        return session, message

    def retry_message(self, session_id: str, message_id: str,
                      device_id: str, attempt_id: str
                      ) -> Tuple[Dict[str, Any], bool]:
        """Atomically record one delivery attempt for a message.

        The recipient must be the session's active recipient. Each distinct
        ``attempt_id`` is counted exactly once, so retries with the same id
        are idempotent. Returns ``(view, created)`` where ``created`` says
        whether this is the first ever attempt for the message (201 vs 200).
        Nothing changes on a failed authorization check.
        """
        with self._lock:
            _, message = self._delivery_target(
                session_id, message_id, device_id)
            key = (session_id, message_id)
            state = self._delivery.get(key)
            created = state is None
            if created:
                state = MessageDelivery()
                self._delivery[key] = state
            if attempt_id not in state.attempt_ids:
                state.attempt_ids.add(attempt_id)
                state.attempts += 1
            view = self._delivery_view(session_id, message, state)
            self._notify_change()
            return view, created

    def ack_message(self, session_id: str, message_id: str, device_id: str,
                    sequence: int) -> Tuple[Dict[str, Any], bool]:
        """Atomically acknowledge a message for the active recipient.

        ``sequence`` must equal the message's stored sequence. The first ack
        marks the message acked; repeated acks are idempotent. Returns
        ``(view, first_ack)``.
        """
        with self._lock:
            _, message = self._delivery_target(
                session_id, message_id, device_id)
            if sequence != message.sequence:
                raise DeliveryError(DELIVERY_BAD_SEQUENCE)
            key = (session_id, message_id)
            state = self._delivery.get(key)
            first_ack = state is None or not state.acked
            if state is None:
                state = MessageDelivery()
                self._delivery[key] = state
            state.acked = True
            state.ack_sequence = sequence
            view = self._delivery_view(session_id, message, state)
            self._notify_change()
            return view, first_ack

    def message_delivery_status(self, session_id: str, message_id: str,
                                device_id: str) -> Dict[str, Any]:
        """Atomically read one message's delivery status for its recipient."""
        with self._lock:
            _, message = self._delivery_target(
                session_id, message_id, device_id)
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
            } for (sid, mid), state in self._delivery.items()]
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
            return {"devices": devices, "sessions": sessions,
                    "prekey_claims": prekey_claims,
                    "prekey_batch_claims": prekey_batch_claims,
                    "claim_session_bindings": claim_session_bindings,
                    "groups": groups, "group_sessions": group_sessions,
                    "messages": messages, "delivery": delivery,
                    "used_nonces": used_nonces,
                    "group_sync_cursors": group_sync_cursors}

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
        raw_groups = state.get("groups", [])
        raw_group_sessions = state.get("group_sessions", [])
        raw_messages = state.get("messages", {})
        raw_delivery = state.get("delivery", [])
        raw_used_nonces = state.get("used_nonces")
        raw_group_sync_cursors = state.get("group_sync_cursors", [])
        if not (isinstance(raw_devices, list) and isinstance(raw_sessions, list)
                and isinstance(raw_prekey_claims, list)
                and isinstance(raw_prekey_batch_claims, list)
                and isinstance(raw_claim_session_bindings, list)
                and isinstance(raw_groups, list)
                and isinstance(raw_group_sessions, list)
                and isinstance(raw_messages, dict)
                and isinstance(raw_delivery, list)
                and isinstance(raw_group_sync_cursors, list)):
            raise ValueError("state document has a malformed top-level section")

        devices: Dict[Tuple[str, str], Device] = {}
        device_index: Dict[str, Tuple[str, str]] = {}
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

        delivery: Dict[Tuple[str, str], MessageDelivery] = {}
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

        with self._lock:
            self._devices = devices
            self._device_index = device_index
            self._sessions = sessions
            self._prekey_claims = prekey_claims
            self._prekey_batch_claims = prekey_batch_claims
            self._claim_session_bindings = claim_session_bindings
            self._groups = groups
            self._group_sessions = group_sessions
            self._messages = messages
            self._delivery = delivery
            self._used_nonces = used_nonces
            self._group_sync_cursors = group_sync_cursors
