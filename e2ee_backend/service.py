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
    DELIVERY_MESSAGE_UNKNOWN,
    MESSAGE_BAD_SEQUENCE,
    MESSAGE_DEVICE_INACTIVE,
    MESSAGE_DUPLICATE_ID,
    MESSAGE_SENDER_INACTIVE,
    MESSAGE_SESSION_UNKNOWN,
    SESSION_INITIATOR_REVOKED,
    SESSION_INITIATOR_UNKNOWN,
    SESSION_PREKEY_REVOKED,
    SESSION_PREKEY_UNKNOWN,
    SESSION_RECIPIENT_REVOKED,
    SESSION_RECIPIENT_UNKNOWN,
    DeliveryError,
    DeviceStore,
    MessageCreateError,
    MessageListError,
    SessionCreateError,
)

_REQUIRED_SCALAR_FIELDS = ("user_id", "device_id", "identity_key")
_SESSION_SCALAR_FIELDS = (
    "initiator_device_id", "recipient_device_id", "prekey_id", "ephemeral_key")
_MESSAGE_STRING_FIELDS = (
    "session_id", "sender_device_id", "message_id", "nonce", "ciphertext")
_RETRY_STRING_FIELDS = ("device_id", "attempt_id")
_ACK_STRING_FIELDS = ("device_id", "message_id")

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

    # -- reliable delivery (retry / ack / status) ----------------------------

    @staticmethod
    def _require_object(payload: object) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            raise ServiceError("request body must be a JSON object",
                               "request_body")
        return payload

    @staticmethod
    def _require_nonempty_strings(payload: Dict[str, Any],
                                  fields: Tuple[str, ...]) -> None:
        for name in fields:
            if name not in payload:
                raise ServiceError(f"missing required field: {name}", name)
            if not is_nonempty_string(payload[name]):
                raise ServiceError(
                    f"field must be a non-empty string: {name}", name)

    @staticmethod
    def _require_integer(payload: Dict[str, Any], name: str) -> int:
        if name not in payload:
            raise ServiceError(f"missing required field: {name}", name)
        value = payload[name]
        # bool is a subclass of int; reject it explicitly.
        if not isinstance(value, int) or isinstance(value, bool):
            raise ServiceError(f"field must be an integer: {name}", name)
        return value

    @staticmethod
    def _delivery_error(error: DeliveryError, session_id: str,
                        message_id: str) -> ServiceError:
        if error.reason == MESSAGE_SESSION_UNKNOWN:
            return ServiceError(f"session not found: {session_id}",
                                "session_id", status_code=404)
        if error.reason == DELIVERY_MESSAGE_UNKNOWN:
            return ServiceError(f"message not found: {message_id}",
                                "message_id", status_code=404)
        if error.reason == DELIVERY_DEVICE_INACTIVE:
            return ServiceError("device_id is not the active recipient",
                                "device_id", status_code=409)
        if error.reason == DELIVERY_BAD_SEQUENCE:
            return ServiceError("sequence does not match the message",
                                "sequence", status_code=409)
        raise  # pragma: no cover - defensive

    def retry_message(self, session_id: str, message_id: str,
                      payload: object) -> Tuple[Dict[str, Any], bool]:
        """Record one delivery attempt; return ``(body, created)``.

        ``created`` is ``True`` (HTTP 201) only for a new ``attempt_id`` on a
        pending message; duplicate attempts and retries after acknowledgement
        return ``created=False`` (HTTP 200) without incrementing ``attempts``.
        """
        body = self._require_object(payload)
        self._require_nonempty_strings(body, _RETRY_STRING_FIELDS)
        try:
            return self.store.record_attempt(
                session_id, message_id, body["device_id"], body["attempt_id"])
        except DeliveryError as error:
            raise self._delivery_error(error, session_id, message_id)

    def ack_message(self, session_id: str,
                    payload: object) -> Tuple[Dict[str, Any], bool]:
        """Acknowledge one message; return ``(body, created)``.

        The first acknowledgement flips the message to ``acked`` (HTTP 201);
        repeats are idempotent (HTTP 200). ``sequence`` must match the
        message's own sequence number.
        """
        body = self._require_object(payload)
        self._require_nonempty_strings(body, _ACK_STRING_FIELDS)
        sequence = self._require_integer(body, "sequence")
        try:
            return self.store.ack_message(
                session_id, body["device_id"], body["message_id"], sequence)
        except DeliveryError as error:
            raise self._delivery_error(
                error, session_id, str(body.get("message_id", "")))

    def message_status(self, session_id: str, message_id: str,
                       device_id: str) -> Dict[str, Any]:
        """Return one message's delivery state (status/attempts/sequence)."""
        try:
            return self.store.delivery_status(session_id, message_id, device_id)
        except DeliveryError as error:
            raise self._delivery_error(error, session_id, message_id)
