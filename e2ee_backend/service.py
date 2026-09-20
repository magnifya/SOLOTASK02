"""Business logic and request validation for device registration/queries.

The service never sees plaintext messages or private keys: it validates and
stores identifiers and public-key material only.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .crypto import is_nonempty_string, load_public_key
from .models import Device, SignedPreKey
from .storage import (
    DELIVERY_BAD_SEQUENCE,
    DELIVERY_DEVICE_INACTIVE,
    DELIVERY_DEVICE_MISMATCH,
    DELIVERY_MESSAGE_UNKNOWN,
    DELIVERY_SESSION_UNKNOWN,
    DEVICE_REVOKED,
    DEVICE_UNKNOWN,
    MESSAGE_BAD_SEQUENCE,
    MESSAGE_DEVICE_INACTIVE,
    MESSAGE_DUPLICATE_ID,
    MESSAGE_SENDER_INACTIVE,
    MESSAGE_SESSION_UNKNOWN,
    PREKEY_CONFLICT,
    SESSION_INITIATOR_REVOKED,
    SESSION_INITIATOR_UNKNOWN,
    SESSION_PREKEY_REVOKED,
    SESSION_PREKEY_UNKNOWN,
    SESSION_RECIPIENT_REVOKED,
    SESSION_RECIPIENT_UNKNOWN,
    DeviceStore,
    DeviceUpdateError,
    DeliveryError,
    MessageCreateError,
    MessageListError,
    SessionCreateError,
)

_REQUIRED_SCALAR_FIELDS = ("user_id", "device_id", "identity_key")
_SESSION_SCALAR_FIELDS = (
    "initiator_device_id", "recipient_device_id", "prekey_id", "ephemeral_key")
_MESSAGE_STRING_FIELDS = (
    "session_id", "sender_device_id", "message_id", "nonce", "ciphertext")

#: Maps a storage-level message-append failure reason to (HTTP status, field).
_MESSAGE_CREATE_ERROR_MAP = {
    MESSAGE_SESSION_UNKNOWN: (404, "session_id"),
    MESSAGE_SENDER_INACTIVE: (409, "sender_device_id"),
    MESSAGE_DUPLICATE_ID: (409, "message_id"),
    MESSAGE_BAD_SEQUENCE: (409, "sequence"),
}

#: Maps a storage-level session failure reason to (HTTP status, field name).
_SESSION_ERROR_MAP = {
    SESSION_INITIATOR_UNKNOWN: (404, "initiator_device_id"),
    SESSION_RECIPIENT_UNKNOWN: (404, "recipient_device_id"),
    SESSION_PREKEY_UNKNOWN: (404, "prekey_id"),
    SESSION_INITIATOR_REVOKED: (409, "initiator_device_id"),
    SESSION_RECIPIENT_REVOKED: (409, "recipient_device_id"),
    SESSION_PREKEY_REVOKED: (409, "prekey_id"),
}


class ServiceError(Exception):
    """A validation/lookup failure carrying an HTTP status and a field name."""

    def __init__(self, message: str, field: str = "", status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.field = field
        self.status_code = status_code

    def to_body(self) -> Dict[str, Any]:
        """JSON-serializable error body naming the offending field."""
        body: Dict[str, Any] = {"message": self.message}
        if self.field:
            body["field"] = self.field
        return body


class DeviceService:
    """Validates payloads and coordinates the :class:`DeviceStore`."""

    def __init__(self, store: Optional[DeviceStore] = None) -> None:
        self.store = store if store is not None else DeviceStore()

    # -- registration ------------------------------------------------------

    def register(self, payload: object) -> Dict[str, Any]:
        """Validate and persist a registration payload; return the 201 body."""
        if not isinstance(payload, dict):
            raise ServiceError("request body must be a JSON object", "request_body")

        for name in _REQUIRED_SCALAR_FIELDS:
            if name not in payload:
                raise ServiceError(f"missing required field: {name}", name)
            if not is_nonempty_string(payload[name]):
                raise ServiceError(f"field must be a non-empty string: {name}", name)

        if "signed_prekeys" not in payload:
            raise ServiceError("missing required field: signed_prekeys",
                               "signed_prekeys")
        raw_prekeys = payload["signed_prekeys"]
        if not isinstance(raw_prekeys, list):
            raise ServiceError("field must be an array: signed_prekeys",
                               "signed_prekeys")

        prekeys: List[SignedPreKey] = []
        seen_key_ids: set = set()
        for index, element in enumerate(raw_prekeys):
            prefix = f"signed_prekeys[{index}]"
            if not isinstance(element, dict):
                raise ServiceError(f"array element must be an object: {prefix}", prefix)
            for subfield in ("key_id", "public_key"):
                path = f"{prefix}.{subfield}"
                if subfield not in element:
                    raise ServiceError(f"missing required field: {path}", path)
                if not is_nonempty_string(element[subfield]):
                    raise ServiceError(
                        f"field must be a non-empty string: {path}", path)
            if element["key_id"] in seen_key_ids:
                raise ServiceError(
                    f"duplicate key_id in signed_prekeys: {element['key_id']}",
                    f"{prefix}.key_id")
            if load_public_key(element["public_key"]) is None:
                raise ServiceError(
                    f"field is not a valid public key: {prefix}.public_key",
                    f"{prefix}.public_key")
            seen_key_ids.add(element["key_id"])
            prekeys.append(SignedPreKey(element["key_id"], element["public_key"]))

        if load_public_key(payload["identity_key"]) is None:
            raise ServiceError("field is not a valid public key: identity_key",
                               "identity_key")

        device = Device(
            user_id=payload["user_id"],
            device_id=payload["device_id"],
            identity_key=payload["identity_key"],
            prekeys=prekeys,
        )
        if not self.store.add_device(device):
            raise ServiceError(
                f"device_id already registered: {device.device_id}",
                "device_id", status_code=409)

        return {"device_id": device.device_id,
                "registered_at": device.registered_at}

    # -- lookup ------------------------------------------------------------

    def get_device(self, device_id: str) -> Dict[str, Any]:
        """Return the public device record; raise :class:`ServiceError` if absent."""
        view = self.store.public_view(device_id)
        if view is None:
            raise ServiceError(f"device not found: {device_id}",
                               "device_id", status_code=404)
        return view

    # -- revocation --------------------------------------------------------

    def revoke_device(self, device_id: str) -> Dict[str, Any]:
        """Revoke a device (and its pre-keys); idempotent, 404 when unknown."""
        device = self.store.revoke_device(device_id)
        if device is None:
            raise ServiceError(f"device not found: {device_id}",
                               "device_id", status_code=404)
        return {"device_id": device.device_id, "revoked": True}

    def revoke_prekey(self, device_id: str, key_id: str) -> Dict[str, Any]:
        """Revoke one pre-key; idempotent, 404 on unknown device/key."""
        device, key_found = self.store.revoke_prekey_by_id(device_id, key_id)
        if device is None:
            raise ServiceError(f"device not found: {device_id}",
                               "device_id", status_code=404)
        if not key_found:
            raise ServiceError(f"pre-key not found: {key_id}",
                               "key_id", status_code=404)
        return {"device_id": device.device_id, "key_id": key_id, "revoked": True}

    # -- identity rotation & pre-key replenishment -------------------------

    def rotate_identity_key(self, device_id: str,
                            payload: object) -> Dict[str, Any]:
        """Validate and apply an identity-key rotation.

        The new ``identity_key`` must be a non-empty, valid public key. The
        same value as the current key is a no-op that leaves ``rotated_at``
        unchanged; a changed value stamps a fresh UTC ISO-8601 ``rotated_at``.
        Unknown device is 404, revoked device is 409 (both field
        ``device_id``); nothing is written on failure.
        """
        if not isinstance(payload, dict):
            raise ServiceError("request body must be a JSON object",
                               "request_body")
        if "identity_key" not in payload:
            raise ServiceError("missing required field: identity_key",
                               "identity_key")
        if not is_nonempty_string(payload["identity_key"]):
            raise ServiceError(
                "field must be a non-empty string: identity_key",
                "identity_key")
        if load_public_key(payload["identity_key"]) is None:
            raise ServiceError(
                "field is not a valid public key: identity_key",
                "identity_key")

        try:
            view, _changed = self.store.rotate_identity_key(
                device_id, payload["identity_key"])
        except DeviceUpdateError as error:
            raise self._device_update_error(error, device_id)
        return view

    def add_prekey(self, device_id: str,
                   payload: object) -> Tuple[Dict[str, Any], int]:
        """Validate and append a pre-key, or confirm an identical existing one.

        ``key_id`` and ``public_key`` must be non-empty strings and the key
        must parse as a public key. A new id appends in order (201); an
        existing id with the same, non-revoked key is idempotent (200); a
        changed key or a revoked id conflicts (409/key_id). Unknown device is
        404, revoked device 409 (field ``device_id``). Returns
        ``(body, status_code)``.
        """
        if not isinstance(payload, dict):
            raise ServiceError("request body must be a JSON object",
                               "request_body")
        for name in ("key_id", "public_key"):
            if name not in payload:
                raise ServiceError(f"missing required field: {name}", name)
            if not is_nonempty_string(payload[name]):
                raise ServiceError(
                    f"field must be a non-empty string: {name}", name)
        if load_public_key(payload["public_key"]) is None:
            raise ServiceError(
                "field is not a valid public key: public_key", "public_key")

        try:
            view, created = self.store.add_prekey(
                device_id, payload["key_id"], payload["public_key"])
        except DeviceUpdateError as error:
            raise self._device_update_error(error, device_id)
        return view, 201 if created else 200

    @staticmethod
    def _device_update_error(error: DeviceUpdateError,
                             device_id: str) -> ServiceError:
        """Translate a rotation/add-pre-key storage failure to a ServiceError."""
        if error.reason == DEVICE_UNKNOWN:
            return ServiceError(f"device not found: {device_id}",
                                "device_id", status_code=404)
        if error.reason == DEVICE_REVOKED:
            return ServiceError("device_id is revoked",
                                "device_id", status_code=409)
        if error.reason == PREKEY_CONFLICT:
            return ServiceError(
                "key_id already exists with a different key or is revoked",
                "key_id", status_code=409)
        raise  # pragma: no cover - defensive

    # -- sessions ----------------------------------------------------------

    def create_session(self, payload: object) -> Dict[str, Any]:
        """Validate a session-negotiation payload and atomically create it.

        Every request creates a fresh session (a new ``session_id``); repeated
        POSTs are never deduplicated.
        """
        if not isinstance(payload, dict):
            raise ServiceError("request body must be a JSON object",
                               "request_body")

        for name in _SESSION_SCALAR_FIELDS:
            if name not in payload:
                raise ServiceError(f"missing required field: {name}", name)
            if not is_nonempty_string(payload[name]):
                raise ServiceError(
                    f"field must be a non-empty string: {name}", name)

        if load_public_key(payload["ephemeral_key"]) is None:
            raise ServiceError(
                "field is not a valid public key: ephemeral_key",
                "ephemeral_key")

        if payload["initiator_device_id"] == payload["recipient_device_id"]:
            raise ServiceError(
                "recipient_device_id must differ from initiator_device_id",
                "recipient_device_id")

        try:
            session = self.store.create_session(
                payload["initiator_device_id"],
                payload["recipient_device_id"],
                payload["prekey_id"],
                payload["ephemeral_key"])
        except SessionCreateError as error:
            status_code, field = _SESSION_ERROR_MAP[error.reason]
            if status_code == 404:
                if field == "prekey_id":
                    message = f"pre-key not found: {payload['prekey_id']}"
                else:
                    device_id = payload[field]
                    message = f"device not found: {device_id}"
            else:
                message = f"{field} is revoked"
            raise ServiceError(message, field, status_code=status_code)

        return self.store.session_view(session.session_id)  # type: ignore[return-value]

    def get_session(self, session_id: str) -> Dict[str, Any]:
        """Return the immutable eight-field session snapshot; 404 if unknown."""
        view = self.store.session_view(session_id)
        if view is None:
            raise ServiceError(f"session not found: {session_id}",
                               "session_id", status_code=404)
        return view

    # -- messages ----------------------------------------------------------

    def post_message(self, payload: object) -> Dict[str, Any]:
        """Validate a message-send payload and atomically append the message.

        ``sequence`` must continue the session's stream (starting at 1, no
        gaps or duplicates) and ``message_id`` must be unique within the
        session; violations are 409s, as is a revoked/unknown sender device.
        """
        if not isinstance(payload, dict):
            raise ServiceError("request body must be a JSON object",
                               "request_body")

        for name in _MESSAGE_STRING_FIELDS:
            if name not in payload:
                raise ServiceError(f"missing required field: {name}", name)
            if not is_nonempty_string(payload[name]):
                raise ServiceError(
                    f"field must be a non-empty string: {name}", name)

        if "sequence" not in payload:
            raise ServiceError("missing required field: sequence", "sequence")
        sequence = payload["sequence"]
        # bool is a subclass of int; reject it explicitly.
        if not isinstance(sequence, int) or isinstance(sequence, bool):
            raise ServiceError("field must be an integer: sequence", "sequence")

        try:
            message = self.store.append_message(
                payload["session_id"],
                payload["sender_device_id"],
                payload["message_id"],
                sequence,
                payload["nonce"],
                payload["ciphertext"])
        except MessageCreateError as error:
            status_code, field = _MESSAGE_CREATE_ERROR_MAP[error.reason]
            if error.reason == MESSAGE_SESSION_UNKNOWN:
                message_text = f"session not found: {payload['session_id']}"
            elif error.reason == MESSAGE_SENDER_INACTIVE:
                message_text = "sender_device_id is not an active device"
            elif error.reason == MESSAGE_DUPLICATE_ID:
                message_text = (f"message_id already exists in session: "
                                f"{payload['message_id']}")
            else:
                message_text = (f"sequence must continue the session stream "
                                f"(got {sequence})")
            raise ServiceError(message_text, field, status_code=status_code)

        return self.store.message_view(message)

    def list_messages(self, session_id: str, device_id: str, after: int,
                      limit: int) -> Dict[str, Any]:
        """Return one page of a session's messages plus the resume cursor."""
        try:
            messages, next_after = self.store.message_page(
                session_id, device_id, after, limit)
        except MessageListError as error:
            if error.reason == MESSAGE_SESSION_UNKNOWN:
                raise ServiceError(f"session not found: {session_id}",
                                   "session_id", status_code=404)
            if error.reason == MESSAGE_DEVICE_INACTIVE:
                raise ServiceError("device_id is not an active device",
                                   "device_id", status_code=409)
            raise  # pragma: no cover - defensive
        return {"messages": messages, "next_after": next_after}

    # -- reliable delivery -------------------------------------------------

    #: Maps a delivery failure reason to (HTTP status, field name).
    _DELIVERY_ERROR_MAP = {
        DELIVERY_SESSION_UNKNOWN: (404, "session_id"),
        DELIVERY_MESSAGE_UNKNOWN: (404, "message_id"),
        DELIVERY_DEVICE_INACTIVE: (409, "device_id"),
        DELIVERY_DEVICE_MISMATCH: (409, "device_id"),
        DELIVERY_BAD_SEQUENCE: (409, "sequence"),
    }

    def _delivery_error(self, error: DeliveryError,
                        session_id: str, message_id: str,
                        sequence: Optional[int] = None) -> ServiceError:
        """Translate a storage :class:`DeliveryError` into a ServiceError."""
        status_code, field = self._DELIVERY_ERROR_MAP[error.reason]
        if error.reason == DELIVERY_SESSION_UNKNOWN:
            text = f"session not found: {session_id}"
        elif error.reason == DELIVERY_MESSAGE_UNKNOWN:
            text = f"message not found: {message_id}"
        elif error.reason == DELIVERY_BAD_SEQUENCE:
            text = f"sequence does not match the message (got {sequence})"
        else:
            text = "device_id is not the active recipient of this session"
        return ServiceError(text, field, status_code=status_code)

    def retry_message(self, session_id: str, message_id: str,
                      payload: object) -> Tuple[Dict[str, Any], int]:
        """Validate and record one delivery attempt for a message.

        Returns ``(body, 201)`` for the first attempt and ``(body, 200)`` for
        a repeated attempt. A repeated ``attempt_id`` is not counted again;
        retries after an ack still succeed with status ``acked``.
        """
        if not isinstance(payload, dict):
            raise ServiceError("request body must be a JSON object",
                               "request_body")
        for name in ("device_id", "attempt_id"):
            if name not in payload:
                raise ServiceError(f"missing required field: {name}", name)
            if not is_nonempty_string(payload[name]):
                raise ServiceError(
                    f"field must be a non-empty string: {name}", name)

        try:
            view, created = self.store.retry_message(
                session_id, message_id,
                payload["device_id"], payload["attempt_id"])
        except DeliveryError as error:
            raise self._delivery_error(error, session_id, message_id)
        return view, 201 if created else 200

    def ack_message(self, session_id: str, payload: object
                    ) -> Tuple[Dict[str, Any], int]:
        """Validate and record an acknowledgement.

        The session is taken from the URL; the body carries ``device_id``,
        ``message_id`` and the integer ``sequence``. The first ack returns 201
        and marks the message acked; repeated acks return 200 and leave the
        acked state in place.
        """
        if not isinstance(payload, dict):
            raise ServiceError("request body must be a JSON object",
                               "request_body")
        for name in ("device_id", "message_id"):
            if name not in payload:
                raise ServiceError(f"missing required field: {name}", name)
            if not is_nonempty_string(payload[name]):
                raise ServiceError(
                    f"field must be a non-empty string: {name}", name)
        if "sequence" not in payload:
            raise ServiceError("missing required field: sequence", "sequence")
        sequence = payload["sequence"]
        if not isinstance(sequence, int) or isinstance(sequence, bool):
            raise ServiceError("field must be an integer: sequence", "sequence")

        try:
            view, first_ack = self.store.ack_message(
                session_id, payload["message_id"],
                payload["device_id"], sequence)
        except DeliveryError as error:
            raise self._delivery_error(
                error, session_id, payload["message_id"], sequence)
        return view, 201 if first_ack else 200

    def message_status(self, session_id: str, message_id: str,
                       device_id: str) -> Dict[str, Any]:
        """Return one message's delivery status for its active recipient."""
        try:
            return self.store.message_delivery_status(
                session_id, message_id, device_id)
        except DeliveryError as error:
            raise self._delivery_error(error, session_id, message_id)
