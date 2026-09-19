"""Thread-safe in-memory device storage.

Devices live for the lifetime of the server process. Only public keys and
identifiers are retained. Reads return pre-key ids in the exact insertion
order, so repeated requests list them identically.
"""
from __future__ import annotations

import threading
import uuid
from typing import Any, Dict, List, Optional, Tuple

from .models import Device, Message, Session, SignedPreKey

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

#: Outcome code for a failed message listing.
MESSAGE_DEVICE_INACTIVE = "device_inactive"

#: Outcome codes for failed reliable-delivery operations (retry/ack/status).
DELIVERY_MESSAGE_UNKNOWN = "message_unknown"
DELIVERY_DEVICE_INACTIVE = "device_inactive"
DELIVERY_BAD_SEQUENCE = "bad_sequence"


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
    """An atomic retry/ack/status check failed (nothing was written)."""

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
        self._persister: Any = None

    # -- persistence hooks ---------------------------------------------------

    def attach_persister(self, persister: Any) -> None:
        """Attach a persister called with a full snapshot after every write.

        The persister only needs a ``save(snapshot: dict)`` method; it is
        invoked while the store lock is held, so snapshots are serialized and
        never interleave with a concurrent mutation.
        """
        with self._lock:
            self._persister = persister

    def _persist_locked(self) -> None:
        if self._persister is not None:
            self._persister.save(self.snapshot())

    def snapshot(self) -> Dict[str, Any]:
        """Return a JSON-serializable snapshot of the entire store."""
        with self._lock:
            return {
                "version": 1,
                "devices": [
                    {
                        "user_id": device.user_id,
                        "device_id": device.device_id,
                        "identity_key": device.identity_key,
                        "registered_at": device.registered_at,
                        "revoked": device.revoked,
                        "prekeys": [
                            {"key_id": pk.key_id, "public_key": pk.public_key,
                             "revoked": pk.revoked}
                            for pk in device.prekeys
                        ],
                    }
                    for device in self._devices.values()
                ],
                "sessions": [
                    {
                        "session_id": s.session_id,
                        "initiator_device_id": s.initiator_device_id,
                        "recipient_device_id": s.recipient_device_id,
                        "prekey_id": s.prekey_id,
                        "ephemeral_key": s.ephemeral_key,
                        "identity_key": s.identity_key,
                        "public_key": s.public_key,
                        "created_at": s.created_at,
                    }
                    for s in self._sessions.values()
                ],
                "messages": {
                    session_id: [
                        {
                            "session_id": m.session_id,
                            "sender_device_id": m.sender_device_id,
                            "message_id": m.message_id,
                            "sequence": m.sequence,
                            "nonce": m.nonce,
                            "ciphertext": m.ciphertext,
                            "created_at": m.created_at,
                            "attempts": m.attempts,
                            "attempt_ids": list(m.attempt_ids),
                            "acked": m.acked,
                        }
                        for m in stream
                    ]
                    for session_id, stream in self._messages.items()
                },
            }

    @classmethod
    def from_snapshot(cls, data: Dict[str, Any]) -> "DeviceStore":
        """Rebuild a store from :meth:`snapshot` data.

        Raises ``ValueError`` when the shape is not a version-1 snapshot.
        """
        if not isinstance(data, dict) or data.get("version") != 1:
            raise ValueError("unsupported snapshot version")
        store = cls()
        for record in data["devices"]:
            device = Device(
                user_id=record["user_id"],
                device_id=record["device_id"],
                identity_key=record["identity_key"],
                registered_at=record["registered_at"],
                prekeys=[SignedPreKey(pk["key_id"], pk["public_key"],
                                      pk.get("revoked", False))
                         for pk in record["prekeys"]],
                revoked=record.get("revoked", False),
            )
            key = (device.user_id, device.device_id)
            store._devices[key] = device
            store._device_index[device.device_id] = key
        for record in data["sessions"]:
            session = Session(
                session_id=record["session_id"],
                initiator_device_id=record["initiator_device_id"],
                recipient_device_id=record["recipient_device_id"],
                prekey_id=record["prekey_id"],
                ephemeral_key=record["ephemeral_key"],
                identity_key=record["identity_key"],
                public_key=record["public_key"],
                created_at=record["created_at"],
            )
            store._sessions[session.session_id] = session
        for session_id, stream in data["messages"].items():
            store._messages[session_id] = [
                Message(
                    session_id=record["session_id"],
                    sender_device_id=record["sender_device_id"],
                    message_id=record["message_id"],
                    sequence=record["sequence"],
                    nonce=record["nonce"],
                    ciphertext=record["ciphertext"],
                    created_at=record["created_at"],
                    attempts=record.get("attempts", 0),
                    attempt_ids=list(record.get("attempt_ids", [])),
                    acked=record.get("acked", False),
                )
                for record in stream
            ]
        return store

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
            self._persist_locked()
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
                    self._persist_locked()
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
            self._persist_locked()
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
                    self._persist_locked()
                    return device, True
            return device, False

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
            self._persist_locked()
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

        The session lookup, sender revocation check, duplicate-id check and
        sequence-continuity check all happen while holding the store lock (the
        same lock device revocations take), so a concurrent revocation is
        linearized either wholly before this call (then it fails) or wholly
        after it (then the message is retained). On any failure nothing is
        written and :class:`MessageCreateError` carries the reason.
        """
        with self._lock:
            if session_id not in self._sessions:
                raise MessageCreateError(MESSAGE_SESSION_UNKNOWN)

            sender_key = self._device_index.get(sender_device_id)
            sender = (self._devices.get(sender_key)
                      if sender_key is not None else None)
            if sender is None or sender.revoked:
                raise MessageCreateError(MESSAGE_SENDER_INACTIVE)

            stream = self._messages.setdefault(session_id, [])
            if any(m.message_id == message_id for m in stream):
                raise MessageCreateError(MESSAGE_DUPLICATE_ID)
            expected = stream[-1].sequence + 1 if stream else 1
            if sequence != expected:
                raise MessageCreateError(MESSAGE_BAD_SEQUENCE)

            message = Message(
                session_id=session_id,
                sender_device_id=sender_device_id,
                message_id=message_id,
                sequence=sequence,
                nonce=nonce,
                ciphertext=ciphertext,
            )
            stream.append(message)
            self._persist_locked()
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
            if session_id not in self._sessions:
                raise MessageListError(MESSAGE_SESSION_UNKNOWN)

            device_key = self._device_index.get(device_id)
            device = (self._devices.get(device_key)
                      if device_key is not None else None)
            if device is None or device.revoked:
                raise MessageListError(MESSAGE_DEVICE_INACTIVE)

            stream = self._messages.get(session_id, [])
            page = [m for m in stream if m.sequence > after][:limit]
            next_after = page[-1].sequence if page else after
            return [self.message_view(m) for m in page], next_after

    # -- reliable delivery (retry / ack / status) ----------------------------

    @staticmethod
    def delivery_view(message: Message) -> Dict[str, Any]:
        """Copy one message's delivery state into its public five-field view."""
        return {
            "session_id": message.session_id,
            "message_id": message.message_id,
            "status": "acked" if message.acked else "pending",
            "attempts": message.attempts,
            "sequence": message.sequence,
        }

    def _find_message_locked(self, session_id: str,
                             message_id: str) -> Tuple[Session, Message]:
        """Look up session and message; raise :class:`DeliveryError` if absent."""
        session = self._sessions.get(session_id)
        if session is None:
            raise DeliveryError(MESSAGE_SESSION_UNKNOWN)
        stream = self._messages.get(session_id, [])
        message = next((m for m in stream if m.message_id == message_id), None)
        if message is None:
            raise DeliveryError(DELIVERY_MESSAGE_UNKNOWN)
        return session, message

    def _check_recipient_locked(self, session: Session, device_id: str) -> None:
        """Require *device_id* to be the session's active recipient device."""
        if device_id != session.recipient_device_id:
            raise DeliveryError(DELIVERY_DEVICE_INACTIVE)
        key = self._device_index.get(device_id)
        device = self._devices.get(key) if key is not None else None
        if device is None or device.revoked:
            raise DeliveryError(DELIVERY_DEVICE_INACTIVE)

    def record_attempt(self, session_id: str, message_id: str,
                       device_id: str, attempt_id: str
                       ) -> Tuple[Dict[str, Any], bool]:
        """Atomically record one delivery attempt for a message.

        Returns ``(view, created)``: ``created`` is ``True`` (HTTP 201) only
        for a brand-new *attempt_id* on a not-yet-acked message; a repeated
        *attempt_id* and any retry after acknowledgement return ``False``
        (HTTP 200) without incrementing ``attempts``. All checks run under
        the store lock, linearized with revocations and acks.
        """
        with self._lock:
            session, message = self._find_message_locked(session_id, message_id)
            self._check_recipient_locked(session, device_id)
            if message.acked or attempt_id in message.attempt_ids:
                return self.delivery_view(message), False
            message.attempt_ids.append(attempt_id)
            message.attempts += 1
            self._persist_locked()
            return self.delivery_view(message), True

    def ack_message(self, session_id: str, device_id: str,
                    message_id: str, sequence: int
                    ) -> Tuple[Dict[str, Any], bool]:
        """Atomically acknowledge one message.

        Returns ``(view, created)``: ``created`` is ``True`` (HTTP 201) for
        the first acknowledgement; repeats return ``False`` (HTTP 200). The
        supplied *sequence* must match the message's own sequence.
        """
        with self._lock:
            session, message = self._find_message_locked(session_id, message_id)
            self._check_recipient_locked(session, device_id)
            if sequence != message.sequence:
                raise DeliveryError(DELIVERY_BAD_SEQUENCE)
            if message.acked:
                return self.delivery_view(message), False
            message.acked = True
            self._persist_locked()
            return self.delivery_view(message), True

    def delivery_status(self, session_id: str, message_id: str,
                        device_id: str) -> Dict[str, Any]:
        """Atomically read one message's delivery state.

        The caller's device must be an active participant of the session
        (initiator or recipient); anything else is a 409 at the API layer.
        """
        with self._lock:
            session, message = self._find_message_locked(session_id, message_id)
            participants = (session.initiator_device_id,
                            session.recipient_device_id)
            if device_id not in participants:
                raise DeliveryError(DELIVERY_DEVICE_INACTIVE)
            key = self._device_index.get(device_id)
            device = self._devices.get(key) if key is not None else None
            if device is None or device.revoked:
                raise DeliveryError(DELIVERY_DEVICE_INACTIVE)
            return self.delivery_view(message)
