"""Thread-safe in-memory device storage.

Devices live for the lifetime of the server process. Only public keys and
identifiers are retained. Reads return pre-key ids in the exact insertion
order, so repeated requests list them identically.
"""
from __future__ import annotations

import threading
import uuid
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .models import Device, Group, GroupSession, Message, MessageDelivery, Session, SignedPreKey, utc_now_iso

#: Outcome codes for a failed atomic session creation.
SESSION_INITIATOR_UNKNOWN = "initiator_unknown"
SESSION_RECIPIENT_UNKNOWN = "recipient_unknown"
SESSION_PREKEY_UNKNOWN = "prekey_unknown"
SESSION_INITIATOR_REVOKED = "initiator_revoked"
SESSION_RECIPIENT_REVOKED = "recipient_revoked"
SESSION_PREKEY_REVOKED = "prekey_revoked"

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

#: Outcome codes for group / group-session operations.
GROUP_UNKNOWN = "group_unknown"
GROUP_DUPLICATE_ID = "group_duplicate_id"
GROUP_MEMBER_DUPLICATE = "group_member_duplicate"
GROUP_CREATOR_UNKNOWN = "group_creator_unknown"
GROUP_CREATOR_INACTIVE = "group_creator_inactive"
GROUP_MEMBER_UNKNOWN = "group_member_unknown"
GROUP_MEMBER_INACTIVE = "group_member_inactive"
GROUP_ACTOR_UNKNOWN = "group_actor_unknown"
GROUP_DEVICE_UNKNOWN = "group_device_unknown"
GROUP_DEVICE_INACTIVE = "group_device_inactive"
GROUP_FORBIDDEN = "group_forbidden"
GROUP_NOT_MEMBER = "group_not_member"
GROUP_INITIATOR_UNKNOWN = "group_initiator_unknown"
GROUP_INITIATOR_INACTIVE = "group_initiator_inactive"


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


class GroupError(Exception):
    """An atomic group or group-session operation failed; nothing changed."""

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
        self._messages: Dict[str, List[Message]] = {}
        # Per-session set of nonces already accepted for replay protection.
        # Keyed independently of the streams so a nonce is scoped to a session.
        self._used_nonces: Dict[str, Set[str]] = {}
        # Delivery state keyed by (session_id, message_id).
        self._delivery: Dict[Tuple[str, str], MessageDelivery] = {}
        # Groups and their frozen-membership sessions.
        self._groups: Dict[str, Group] = {}
        self._group_sessions: Dict[str, GroupSession] = {}
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
        """Return key ids of non-revoked pre-keys, in stable insertion order."""
        with self._lock:
            return [pk.key_id for pk in device.prekeys if not pk.revoked]

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
                               if not pk.revoked],
                "registered_at": device.registered_at,
            }

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
            # A group session accepts messages only from its frozen members;
            # members added after the session was created are not on the list.
            if group_session is not None \
                    and sender_device_id not in group_session.members:
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
            # Group-session reads are restricted to the frozen membership.
            if group_session is not None \
                    and device_id not in group_session.members:
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

    # -- groups ------------------------------------------------------------

    def _require_active_device(self, device_id: str) -> Device:
        """Look up a device under the lock and require it to be non-revoked.

        Raises :class:`GroupError` ``group_device_unknown`` /
        ``group_device_inactive``. Must be called while holding the lock.
        """
        key = self._device_index.get(device_id)
        device = self._devices.get(key) if key is not None else None
        if device is None:
            raise GroupError(GROUP_DEVICE_UNKNOWN)
        if device.revoked:
            raise GroupError(GROUP_DEVICE_INACTIVE)
        return device

    @staticmethod
    def group_view(group: Group) -> Dict[str, Any]:
        """Copy a group into its public five-field view."""
        return {
            "group_id": group.group_id,
            "creator_device_id": group.creator_device_id,
            "members": list(group.members),
            "revision": group.revision,
            "created_at": group.created_at,
        }

    @staticmethod
    def group_session_view(session: GroupSession) -> Dict[str, Any]:
        """Copy a group session into its public seven-field frozen view."""
        return {
            "session_id": session.session_id,
            "group_id": session.group_id,
            "initiator_device_id": session.initiator_device_id,
            "ephemeral_key": session.ephemeral_key,
            "members": list(session.members),
            "revision": session.revision,
            "created_at": session.created_at,
        }

    def create_group(self, group_id: str, creator_device_id: str,
                     member_device_ids: List[str]) -> Group:
        """Atomically create a group.

        The creator and every listed member must reference an active device.
        The creator is always a member exactly once: when it appears in the
        member list that ordering is kept, otherwise it is prepended. On any
        failure nothing is written and :class:`GroupError` carries the reason.
        """
        with self._lock:
            if group_id in self._groups:
                raise GroupError(GROUP_DUPLICATE_ID)
            creator_key = self._device_index.get(creator_device_id)
            creator = (self._devices.get(creator_key)
                       if creator_key is not None else None)
            if creator is None:
                raise GroupError(GROUP_CREATOR_UNKNOWN)
            if creator.revoked:
                raise GroupError(GROUP_CREATOR_INACTIVE)
            members: List[str] = []
            seen: Set[str] = set()
            for device_id in member_device_ids:
                member_key = self._device_index.get(device_id)
                member = (self._devices.get(member_key)
                          if member_key is not None else None)
                if member is None:
                    raise GroupError(GROUP_MEMBER_UNKNOWN)
                if member.revoked:
                    raise GroupError(GROUP_MEMBER_INACTIVE)
                if device_id in seen:
                    raise GroupError(GROUP_MEMBER_DUPLICATE)
                seen.add(device_id)
                members.append(device_id)
            if creator_device_id not in seen:
                members.insert(0, creator_device_id)
            group = Group(
                group_id=group_id,
                creator_device_id=creator_device_id,
                members=members)
            self._groups[group_id] = group
            self._notify_change()
            return group

    def get_group(self, group_id: str) -> Optional[Group]:
        """Return the group with this id, or ``None``."""
        with self._lock:
            return self._groups.get(group_id)

    def add_group_member(self, group_id: str, actor_device_id: str,
                         device_id: str) -> Tuple[Group, bool]:
        """Atomically add a member; only the creator may authorize it.

        Returns ``(group, created)``: adding a device that is not yet a member
        appends it, increments ``revision`` and reports ``created`` True (201);
        an existing member is idempotent (200, ``created`` False). A revoked
        device can never be added. All checks and the write happen under the
        store lock; a failure changes nothing.
        """
        with self._lock:
            group = self._groups.get(group_id)
            if group is None:
                raise GroupError(GROUP_UNKNOWN)
            if actor_device_id not in self._device_index:
                raise GroupError(GROUP_ACTOR_UNKNOWN)
            if actor_device_id != group.creator_device_id:
                raise GroupError(GROUP_FORBIDDEN)
            if device_id in group.members:
                return group, False
            self._require_active_device(device_id)
            group.members.append(device_id)
            group.revision += 1
            self._notify_change()
            return group, True

    def remove_group_member(self, group_id: str, actor_device_id: str,
                            device_id: str) -> Group:
        """Atomically remove a member; only the creator may authorize it.

        Removing a device that is not a member (including a repeated removal)
        is idempotent and still returns 200 with the state unchanged. A
        successful removal increments ``revision`` exactly once.
        """
        with self._lock:
            group = self._groups.get(group_id)
            if group is None:
                raise GroupError(GROUP_UNKNOWN)
            if actor_device_id not in self._device_index:
                raise GroupError(GROUP_ACTOR_UNKNOWN)
            if actor_device_id != group.creator_device_id:
                raise GroupError(GROUP_FORBIDDEN)
            if device_id not in self._device_index:
                raise GroupError(GROUP_DEVICE_UNKNOWN)
            if device_id in group.members:
                group.members.remove(device_id)
                group.revision += 1
                self._notify_change()
            return group

    def create_group_session(self, group_id: str,
                             initiator_device_id: str,
                             ephemeral_key: str) -> GroupSession:
        """Atomically create a group session with the membership frozen.

        The group must exist and the initiator must be an active current
        member. The frozen member list and ``revision`` are copied from the
        group at this linearization point; later membership changes do not
        affect the session. Every call creates a fresh ``session_id``.
        """
        with self._lock:
            group = self._groups.get(group_id)
            if group is None:
                raise GroupError(GROUP_UNKNOWN)
            initiator_key = self._device_index.get(initiator_device_id)
            initiator = (self._devices.get(initiator_key)
                         if initiator_key is not None else None)
            if initiator is None:
                raise GroupError(GROUP_INITIATOR_UNKNOWN)
            if initiator.revoked:
                raise GroupError(GROUP_INITIATOR_INACTIVE)
            if initiator_device_id not in group.members:
                raise GroupError(GROUP_NOT_MEMBER)
            session_id = uuid.uuid4().hex
            while session_id in self._sessions or session_id in self._group_sessions:
                session_id = uuid.uuid4().hex
            session = GroupSession(
                session_id=session_id,
                group_id=group_id,
                initiator_device_id=initiator_device_id,
                ephemeral_key=ephemeral_key,
                members=list(group.members),
                revision=group.revision)
            self._group_sessions[session_id] = session
            self._notify_change()
            return session

    def get_group_session(self, session_id: str) -> Optional[GroupSession]:
        """Return the group session with this id, or ``None``."""
        with self._lock:
            return self._group_sessions.get(session_id)

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
                                 "revoked": pk.revoked}
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
            groups = [{
                "group_id": group.group_id,
                "creator_device_id": group.creator_device_id,
                "members": list(group.members),
                "revision": group.revision,
                "created_at": group.created_at,
            } for group in self._groups.values()]
            group_sessions = [{
                "session_id": session.session_id,
                "group_id": session.group_id,
                "initiator_device_id": session.initiator_device_id,
                "ephemeral_key": session.ephemeral_key,
                "members": list(session.members),
                "revision": session.revision,
                "created_at": session.created_at,
            } for session in self._group_sessions.values()]
            return {"devices": devices, "sessions": sessions,
                    "messages": messages, "delivery": delivery,
                    "used_nonces": used_nonces,
                    "groups": groups, "group_sessions": group_sessions}

    def restore_state(self, state: Dict[str, Any]) -> None:
        """Replace all in-memory state from a persisted (version-stripped) doc.

        Raises :class:`ValueError` when the document is malformed; the current
        in-memory state is only replaced after the whole document parses.
        """
        if not isinstance(state, dict):
            raise ValueError("state document must be a JSON object")

        raw_devices = state.get("devices", [])
        raw_sessions = state.get("sessions", [])
        raw_messages = state.get("messages", {})
        raw_delivery = state.get("delivery", [])
        raw_used_nonces = state.get("used_nonces")
        raw_groups = state.get("groups", [])
        raw_group_sessions = state.get("group_sessions", [])
        if not (isinstance(raw_devices, list) and isinstance(raw_sessions, list)
                and isinstance(raw_messages, dict)
                and isinstance(raw_delivery, list)
                and isinstance(raw_groups, list)
                and isinstance(raw_group_sessions, list)):
            raise ValueError("state document has a malformed top-level section")

        devices: Dict[Tuple[str, str], Device] = {}
        device_index: Dict[str, Tuple[str, str]] = {}
        for index, raw in enumerate(raw_devices):
            if not isinstance(raw, dict):
                raise ValueError(f"devices[{index}] must be an object")
            try:
                prekeys = []
                for pk in raw["prekeys"]:
                    if not isinstance(pk, dict):
                        raise ValueError("prekey must be an object")
                    prekeys.append(SignedPreKey(
                        key_id=pk["key_id"], public_key=pk["public_key"],
                        revoked=bool(pk.get("revoked", False))))
                device = Device(
                    user_id=raw["user_id"], device_id=raw["device_id"],
                    identity_key=raw["identity_key"],
                    registered_at=raw["registered_at"],
                    # Older version-1 files predate this field; then it equals
                    # registered_at (Device.__post_init__ fills in the default).
                    rotated_at=raw.get("rotated_at"),
                    prekeys=prekeys, revoked=bool(raw.get("revoked", False)))
            except KeyError as error:
                raise ValueError(
                    f"devices[{index}] missing field: {error.args[0]}") from None
            key = (device.user_id, device.device_id)
            if key in devices or device.device_id in device_index:
                raise ValueError(
                    f"duplicate device in state: {device.device_id}")
            devices[key] = device
            device_index[device.device_id] = key

        sessions: Dict[str, Session] = {}
        for index, raw in enumerate(raw_sessions):
            if not isinstance(raw, dict):
                raise ValueError(f"sessions[{index}] must be an object")
            try:
                session = Session(
                    session_id=raw["session_id"],
                    initiator_device_id=raw["initiator_device_id"],
                    recipient_device_id=raw["recipient_device_id"],
                    prekey_id=raw["prekey_id"],
                    ephemeral_key=raw["ephemeral_key"],
                    identity_key=raw["identity_key"],
                    public_key=raw["public_key"],
                    created_at=raw["created_at"])
            except KeyError as error:
                raise ValueError(
                    f"sessions[{index}] missing field: {error.args[0]}") from None
            if session.session_id in sessions:
                raise ValueError(
                    f"duplicate session in state: {session.session_id}")
            sessions[session.session_id] = session

        messages: Dict[str, List[Message]] = {}
        for sid, stream in raw_messages.items():
            if not isinstance(sid, str) or not isinstance(stream, list):
                raise ValueError("messages must map session_id to a list")
            parsed: List[Message] = []
            for index, raw in enumerate(stream):
                if not isinstance(raw, dict):
                    raise ValueError(f"messages[{sid}][{index}] must be an object")
                try:
                    parsed.append(Message(
                        session_id=raw["session_id"],
                        sender_device_id=raw["sender_device_id"],
                        message_id=raw["message_id"],
                        sequence=raw["sequence"], nonce=raw["nonce"],
                        ciphertext=raw["ciphertext"],
                        created_at=raw["created_at"]))
                except KeyError as error:
                    raise ValueError(
                        f"messages[{sid}][{index}] missing field: "
                        f"{error.args[0]}") from None
            messages[sid] = parsed

        # Session-scoped replay protection. Older version-1 files predate the
        # section: rebuild it from stored message history, which lists every
        # nonce ever accepted. A present section must map session_id to a list
        # of plain strings; anything else is a malformed document.
        used_nonces: Dict[str, Set[str]] = {}
        for sid, stream in messages.items():
            rebuilt = used_nonces.setdefault(sid, set())
            for message in stream:
                rebuilt.add(message.nonce)
        if raw_used_nonces is not None:
            if not isinstance(raw_used_nonces, dict):
                raise ValueError("used_nonces must be an object")
            for sid, nonce_list in raw_used_nonces.items():
                if not isinstance(sid, str) or not isinstance(nonce_list, list) \
                        or not all(isinstance(value, str)
                                   for value in nonce_list):
                    raise ValueError(
                        "used_nonces must map session_id to a list of strings")
                used_nonces.setdefault(sid, set()).update(nonce_list)

        delivery: Dict[Tuple[str, str], MessageDelivery] = {}
        for index, raw in enumerate(raw_delivery):
            if not isinstance(raw, dict):
                raise ValueError(f"delivery[{index}] must be an object")
            try:
                attempt_ids = raw["attempt_ids"]
                if not isinstance(attempt_ids, list) or not all(
                        isinstance(value, str) for value in attempt_ids):
                    raise ValueError("attempt_ids must be a list of strings")
                record = MessageDelivery(
                    attempts=int(raw["attempts"]),
                    attempt_ids=set(attempt_ids),
                    acked=bool(raw["acked"]),
                    ack_sequence=int(raw["ack_sequence"]))
                dkey = (raw["session_id"], raw["message_id"])
            except KeyError as error:
                raise ValueError(
                    f"delivery[{index}] missing field: {error.args[0]}") from None
            if dkey in delivery:
                raise ValueError(
                    f"duplicate delivery record in state: {dkey}")
            delivery[dkey] = record

        groups: Dict[str, Group] = {}
        for index, raw in enumerate(raw_groups):
            if not isinstance(raw, dict):
                raise ValueError(f"groups[{index}] must be an object")
            try:
                members = raw["members"]
                revision = raw["revision"]
                if not isinstance(members, list) or not all(
                        isinstance(value, str) and value
                        for value in members):
                    raise ValueError("members must be a list of non-empty strings")
                if not isinstance(revision, int) or isinstance(revision, bool) \
                        or revision < 1:
                    raise ValueError("revision must be a positive integer")
                group = Group(
                    group_id=raw["group_id"],
                    creator_device_id=raw["creator_device_id"],
                    members=list(members),
                    revision=revision,
                    created_at=raw["created_at"])
            except KeyError as error:
                raise ValueError(
                    f"groups[{index}] missing field: {error.args[0]}") from None
            if group.group_id in groups:
                raise ValueError(
                    f"duplicate group in state: {group.group_id}")
            groups[group.group_id] = group

        group_sessions: Dict[str, GroupSession] = {}
        for index, raw in enumerate(raw_group_sessions):
            if not isinstance(raw, dict):
                raise ValueError(f"group_sessions[{index}] must be an object")
            try:
                members = raw["members"]
                revision = raw["revision"]
                if not isinstance(members, list) or not all(
                        isinstance(value, str) and value
                        for value in members):
                    raise ValueError("members must be a list of non-empty strings")
                if not isinstance(revision, int) or isinstance(revision, bool) \
                        or revision < 1:
                    raise ValueError("revision must be a positive integer")
                group_session = GroupSession(
                    session_id=raw["session_id"],
                    group_id=raw["group_id"],
                    initiator_device_id=raw["initiator_device_id"],
                    ephemeral_key=raw["ephemeral_key"],
                    members=list(members),
                    revision=revision,
                    created_at=raw["created_at"])
            except KeyError as error:
                raise ValueError(
                    f"group_sessions[{index}] missing field: "
                    f"{error.args[0]}") from None
            if group_session.group_id not in groups:
                raise ValueError(
                    f"group_sessions[{index}] references unknown group: "
                    f"{group_session.group_id}")
            if group_session.session_id in sessions \
                    or group_session.session_id in group_sessions:
                raise ValueError(
                    "duplicate group session in state: "
                    f"{group_session.session_id}")
            group_sessions[group_session.session_id] = group_session

        with self._lock:
            self._devices = devices
            self._device_index = device_index
            self._sessions = sessions
            self._messages = messages
            self._delivery = delivery
            self._used_nonces = used_nonces
            self._groups = groups
            self._group_sessions = group_sessions
