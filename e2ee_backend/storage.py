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
    Device,
    Group,
    GroupSession,
    GroupSyncCursor,
    Message,
    MessageDelivery,
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


def _is_nonempty_str(value: Any) -> bool:
    """Return True for a plain, non-empty ``str``.

    ``bool``/numbers/``None``/containers all fail: ``isinstance(True, str)``
    is already False, so this is the single type-and-content gate used by
    state recovery for every identifier and string-valued field.
    """
    return isinstance(value, str) and value != ""


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
        raw_groups = state.get("groups", [])
        raw_group_sessions = state.get("group_sessions", [])
        raw_messages = state.get("messages", {})
        raw_delivery = state.get("delivery", [])
        raw_used_nonces = state.get("used_nonces")
        raw_group_sync_cursors = state.get("group_sync_cursors", [])
        if not (isinstance(raw_devices, list) and isinstance(raw_sessions, list)
                and isinstance(raw_groups, list)
                and isinstance(raw_group_sessions, list)
                and isinstance(raw_messages, dict)
                and isinstance(raw_delivery, list)
                and isinstance(raw_group_sync_cursors, list)):
            raise ValueError("state document has a malformed top-level section")

        devices: Dict[Tuple[str, str], Device] = {}
        device_index: Dict[str, Tuple[str, str]] = {}
        for index, raw in enumerate(raw_devices):
            if not isinstance(raw, dict):
                raise ValueError(f"devices[{index}] must be an object")
            where = f"devices[{index}]"
            try:
                user_id = raw["user_id"]
                device_id = raw["device_id"]
                identity_key = raw["identity_key"]
                registered_at = raw["registered_at"]
                raw_prekeys = raw["prekeys"]
            except KeyError as error:
                raise ValueError(
                    f"{where} missing field: {error.args[0]}") from None
            # Every identifier and string field must be a non-empty string;
            # booleans/numbers/null are rejected.
            if not all(_is_nonempty_str(value) for value in
                       (user_id, device_id, identity_key, registered_at)):
                raise ValueError(
                    f"{where} string fields must be non-empty strings")
            # Older version-1 files predate rotated_at; an absent value is
            # filled from registered_at by Device.__post_init__. A present
            # value must still be a non-empty string.
            rotated_at = raw.get("rotated_at")
            if rotated_at is not None and not _is_nonempty_str(rotated_at):
                raise ValueError(
                    f"{where} rotated_at must be a non-empty string")
            revoked = raw.get("revoked", False)
            if not isinstance(revoked, bool):
                raise ValueError(f"{where} revoked must be a boolean")
            if not isinstance(raw_prekeys, list):
                raise ValueError(f"{where} prekeys must be a list")
            prekeys = []
            seen_key_ids: Set[str] = set()
            for pk_index, pk in enumerate(raw_prekeys):
                pk_where = f"{where}.prekeys[{pk_index}]"
                if not isinstance(pk, dict):
                    raise ValueError(f"{pk_where} must be an object")
                key_id = pk.get("key_id")
                public_key = pk.get("public_key")
                if not (_is_nonempty_str(key_id)
                        and _is_nonempty_str(public_key)):
                    raise ValueError(
                        f"{pk_where} key_id/public_key must be non-empty "
                        "strings")
                pk_revoked = pk.get("revoked", False)
                if not isinstance(pk_revoked, bool):
                    raise ValueError(
                        f"{pk_where} revoked must be a boolean")
                if key_id in seen_key_ids:
                    raise ValueError(
                        f"{where} has a duplicate prekey key_id: {key_id}")
                seen_key_ids.add(key_id)
                prekeys.append(SignedPreKey(
                    key_id=key_id, public_key=public_key,
                    revoked=pk_revoked))
            device = Device(
                user_id=user_id, device_id=device_id,
                identity_key=identity_key,
                registered_at=registered_at,
                rotated_at=rotated_at,
                prekeys=prekeys, revoked=revoked)
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
            where = f"sessions[{index}]"
            try:
                session_id = raw["session_id"]
                initiator_id = raw["initiator_device_id"]
                recipient_id = raw["recipient_device_id"]
                prekey_id = raw["prekey_id"]
                ephemeral_key = raw["ephemeral_key"]
                identity_key = raw["identity_key"]
                public_key = raw["public_key"]
                created_at = raw["created_at"]
            except KeyError as error:
                raise ValueError(
                    f"{where} missing field: {error.args[0]}") from None
            # Every identifier, both snapshot public keys, the ephemeral key
            # and the timestamp must be non-empty strings; booleans, numbers
            # and null are rejected.
            if not all(_is_nonempty_str(value) for value in (
                    session_id, initiator_id, recipient_id, prekey_id,
                    ephemeral_key, identity_key, public_key, created_at)):
                raise ValueError(
                    f"{where} fields must be non-empty strings")
            if session_id in sessions:
                raise ValueError(
                    f"duplicate session in state: {session_id}")
            # Both endpoints must name registered devices. A device revoked
            # after the session was frozen keeps its historical sessions, so
            # revocation is not checked here.
            initiator = devices.get(device_index.get(initiator_id)) \
                if initiator_id in device_index else None
            if initiator is None:
                raise ValueError(
                    f"{where} initiator_device_id is not a registered "
                    f"device: {initiator_id}")
            recipient = devices.get(device_index.get(recipient_id)) \
                if recipient_id in device_index else None
            if recipient is None:
                raise ValueError(
                    f"{where} recipient_device_id is not a registered "
                    f"device: {recipient_id}")
            # The claimed pre-key must be one of the recipient's own keys
            # (revoked keys remain stored and still count: the snapshot was
            # taken at session creation and is immutable).
            if not any(pk.key_id == prekey_id for pk in recipient.prekeys):
                raise ValueError(
                    f"{where} prekey_id {prekey_id} does not belong to "
                    f"recipient device {recipient_id}")
            session = Session(
                session_id=session_id,
                initiator_device_id=initiator_id,
                recipient_device_id=recipient_id,
                prekey_id=prekey_id,
                ephemeral_key=ephemeral_key,
                identity_key=identity_key,
                public_key=public_key,
                created_at=created_at)
            sessions[session.session_id] = session

        groups: Dict[str, Group] = {}
        for index, raw in enumerate(raw_groups):
            if not isinstance(raw, dict):
                raise ValueError(f"groups[{index}] must be an object")
            where = f"groups[{index}]"
            try:
                group_id = raw["group_id"]
                creator_id = raw["creator_device_id"]
                members = raw["members"]
                revision = raw["revision"]
                created_at = raw["created_at"]
            except KeyError as error:
                raise ValueError(
                    f"{where} missing field: {error.args[0]}") from None
            if not (_is_nonempty_str(group_id)
                    and _is_nonempty_str(creator_id)
                    and _is_nonempty_str(created_at)):
                raise ValueError(
                    f"{where} group_id/creator_device_id/created_at must be "
                    "non-empty strings")
            if not isinstance(revision, int) or isinstance(revision, bool) \
                    or revision < 1:
                raise ValueError(
                    f"{where} revision must be a positive integer")
            if not isinstance(members, list) or not members:
                raise ValueError(
                    f"{where} members must be a non-empty list")
            if not all(_is_nonempty_str(value) for value in members):
                raise ValueError(
                    f"{where} members must be non-empty strings")
            if len(set(members)) != len(members):
                raise ValueError(
                    f"{where} members must not contain duplicates")
            # The creator is a registered device and is frozen as the first
            # member; a creator revoked later keeps its historical groups.
            if creator_id not in device_index:
                raise ValueError(
                    f"{where} creator_device_id is not a registered device: "
                    f"{creator_id}")
            if members[0] != creator_id:
                raise ValueError(
                    f"{where} creator_device_id must be the first member")
            group = Group(
                group_id=group_id,
                creator_device_id=creator_id,
                members=list(members),
                revision=revision,
                created_at=created_at)
            if group.group_id in groups:
                raise ValueError(
                    f"duplicate group in state: {group.group_id}")
            groups[group.group_id] = group

        group_sessions: Dict[str, GroupSession] = {}
        for index, raw in enumerate(raw_group_sessions):
            if not isinstance(raw, dict):
                raise ValueError(f"group_sessions[{index}] must be an object")
            where = f"group_sessions[{index}]"
            try:
                gs_session_id = raw["session_id"]
                group_id = raw["group_id"]
                initiator_id = raw["initiator_device_id"]
                ephemeral_key = raw["ephemeral_key"]
                members = raw["members"]
                revision = raw["revision"]
                created_at = raw["created_at"]
            except KeyError as error:
                raise ValueError(
                    f"{where} missing field: "
                    f"{error.args[0]}") from None
            if not all(_is_nonempty_str(value) for value in (
                    gs_session_id, group_id, initiator_id, ephemeral_key,
                    created_at)):
                raise ValueError(
                    f"{where} fields must be non-empty strings")
            if not isinstance(revision, int) or isinstance(revision, bool) \
                    or revision < 1:
                raise ValueError(
                    f"{where} revision must be a positive integer")
            if not isinstance(members, list) or not members:
                raise ValueError(
                    f"{where} members must be a non-empty list")
            if not all(_is_nonempty_str(value) for value in members):
                raise ValueError(
                    f"{where} members must be non-empty strings")
            if len(set(members)) != len(members):
                raise ValueError(
                    f"{where} members must not contain duplicates")
            if gs_session_id in group_sessions:
                raise ValueError(
                    "duplicate group session in state: "
                    f"{gs_session_id}")
            # The id spaces of 1:1 and group sessions are shared (messages are
            # keyed by session_id alone), so an id taken by a 1:1 session can
            # never also name a group session.
            if gs_session_id in sessions:
                raise ValueError(
                    "group session id collides with a 1:1 session: "
                    f"{gs_session_id}")
            # The session must belong to a stored group; a frozen snapshot
            # naming no group is a dangling reference.
            group = groups.get(group_id)
            if group is None:
                raise ValueError(
                    f"{where} references an unknown group: {group_id}")
            # The initiator must have been frozen into this member snapshot.
            if initiator_id not in members:
                raise ValueError(
                    f"{where} initiator_device_id is not a frozen member: "
                    f"{initiator_id}")
            # The frozen revision cannot be ahead of the group's current
            # revision: sessions only snapshot revisions the group reached.
            if revision > group.revision:
                raise ValueError(
                    f"{where} revision {revision} exceeds the current group "
                    f"revision {group.revision}")
            group_session = GroupSession(
                session_id=gs_session_id,
                group_id=group_id,
                initiator_device_id=initiator_id,
                ephemeral_key=ephemeral_key,
                members=list(members),
                revision=revision,
                created_at=created_at)
            group_sessions[group_session.session_id] = group_session

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
            self._groups = groups
            self._group_sessions = group_sessions
            self._messages = messages
            self._delivery = delivery
            self._used_nonces = used_nonces
            self._group_sync_cursors = group_sync_cursors
