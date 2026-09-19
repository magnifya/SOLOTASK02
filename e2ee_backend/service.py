"""Business logic and request validation for device registration/queries.

The service never sees plaintext messages or private keys: it validates and
stores identifiers and public-key material only.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .crypto import is_nonempty_string, load_public_key
from .models import Device, SignedPreKey
from .storage import (
    SESSION_INITIATOR_REVOKED,
    SESSION_INITIATOR_UNKNOWN,
    SESSION_PREKEY_REVOKED,
    SESSION_PREKEY_UNKNOWN,
    SESSION_RECIPIENT_REVOKED,
    SESSION_RECIPIENT_UNKNOWN,
    DeviceStore,
    SessionCreateError,
)

_REQUIRED_SCALAR_FIELDS = ("user_id", "device_id", "identity_key")
_SESSION_SCALAR_FIELDS = (
    "initiator_device_id", "recipient_device_id", "prekey_id", "ephemeral_key")

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
