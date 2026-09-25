"""Business logic and request validation for device registration/queries.

The service never sees plaintext messages or private keys: it validates and
stores identifiers and public-key material only.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .crypto import is_nonempty_string, load_public_key
from .models import Device, SignedPreKey
from .persistence import IntegrityCheckError
from .storage import (
    BATCH_CLAIM_NO_ACTIVE_DEVICE,
    BATCH_CLAIM_NO_PREKEY,
    BATCH_CLAIM_USER_UNKNOWN,
    BATCH_SESSION_CLAIM_UNKNOWN,
    BATCH_SESSION_CLAIM_WRONG_KIND,
    BATCH_SESSION_DEVICE_SET_MISMATCH,
    BATCH_SESSION_DUPLICATE,
    BATCH_SESSION_INITIATOR_IN_SNAPSHOT,
    BATCH_SESSION_INITIATOR_REVOKED,
    BATCH_SESSION_INITIATOR_UNKNOWN,
    BATCH_SESSION_PREKEY_REVOKED,
    BATCH_SESSION_RECIPIENT_REVOKED,
    CLAIM_ID_CONFLICT,
    CLAIM_NO_PREKEY,
    CLAIM_RECIPIENT_REVOKED,
    CLAIM_RECIPIENT_UNKNOWN,
    CLAIM_SESSION_CLAIM_UNKNOWN,
    CLAIM_SESSION_DUPLICATE,
    CLAIM_SESSION_INITIATOR_REVOKED,
    CLAIM_SESSION_INITIATOR_UNKNOWN,
    CLAIM_SESSION_PREKEY_REVOKED,
    CLAIM_SESSION_RECIPIENT_REVOKED,
    DELIVERY_BAD_SEQUENCE,
    DELIVERY_DEVICE_INACTIVE,
    DELIVERY_DEVICE_MISMATCH,
    DELIVERY_MESSAGE_UNKNOWN,
    DELIVERY_SESSION_UNKNOWN,
    DEVICE_REVOKED,
    DEVICE_UNKNOWN,
    GROUP_ACTOR_NOT_CREATOR,
    GROUP_ACTOR_REVOKED,
    GROUP_ACTOR_UNKNOWN,
    GROUP_CREATOR_REVOKED,
    GROUP_CREATOR_UNKNOWN,
    GROUP_DEVICE_REVOKED,
    GROUP_DEVICE_UNKNOWN,
    GROUP_DUPLICATE_ID,
    GROUP_SESSION_GROUP_UNKNOWN,
    GROUP_SESSION_INITIATOR_INACTIVE,
    GROUP_SESSION_INITIATOR_NOT_MEMBER,
    GROUP_SESSION_INITIATOR_UNKNOWN,
    GROUP_UNKNOWN,
    MESSAGE_BAD_SEQUENCE,
    MESSAGE_DEVICE_INACTIVE,
    MESSAGE_DUPLICATE_ID,
    MESSAGE_DUPLICATE_NONCE,
    MESSAGE_REQUEST_ID_CONFLICT,
    MESSAGE_SENDER_INACTIVE,
    MESSAGE_SESSION_UNKNOWN,
    MESSAGE_SYNC_CURSOR_CONFLICT,
    MESSAGE_SYNC_DEVICE_INACTIVE,
    MESSAGE_SYNC_DEVICE_NOT_PARTICIPANT,
    MESSAGE_SYNC_DEVICE_UNKNOWN,
    MESSAGE_SYNC_SESSION_UNKNOWN,
    INBOX_RETRY_MESSAGE_ACKED,
    INBOX_RETRY_MESSAGE_UNKNOWN,
    INBOX_RETRY_NOT_RECIPIENT,
    INBOX_RETRY_SESSION_UNKNOWN,
    INBOX_LEASE_COMPLETION_CONFLICT,
    INBOX_LEASE_CONFLICT,
    INBOX_LEASE_DEVICE_INACTIVE,
    INBOX_LEASE_DEVICE_UNKNOWN,
    INBOX_LEASE_NOT_DELIVERED,
    INBOX_LEASE_NOT_FOUND,
    INBOX_LEASE_UNAVAILABLE,
    PREKEY_CONFLICT,
    REDELIVERY_JOB_CONFLICT,
    REDELIVERY_JOB_DEVICE_INACTIVE,
    REDELIVERY_JOB_DEVICE_UNKNOWN,
    REDELIVERY_JOB_LEASE_OCCUPIED,
    REDELIVERY_JOB_NOT_FOUND,
    REDELIVERY_JOB_RECOVERY_CONFLICT,
    REDELIVERY_JOB_RECOVERY_LEASE_ACTIVE,
    REDELIVERY_JOB_RECOVERY_PARTIAL_REPLAY,
    REDELIVERY_JOB_RECOVERY_STATE,
    ROTATION_ACTOR_NOT_CREATOR,
    ROTATION_ACTOR_REVOKED,
    ROTATION_ACTOR_UNKNOWN,
    ROTATION_PREDECESSOR_ROTATED,
    ROTATION_REVISION_MISMATCH,
    ROTATION_ID_CONFLICT,
    ROTATION_SESSION_UNKNOWN,
    SESSION_INITIATOR_REVOKED,
    SESSION_INITIATOR_UNKNOWN,
    SESSION_PREKEY_CONSUMED,
    SESSION_PREKEY_REVOKED,
    SESSION_PREKEY_UNKNOWN,
    SESSION_RECIPIENT_REVOKED,
    SESSION_RECIPIENT_UNKNOWN,
    SYNC_CURSOR_CONFLICT,
    SYNC_DEVICE_INACTIVE,
    SYNC_DEVICE_NOT_MEMBER,
    SYNC_DEVICE_UNKNOWN,
    SYNC_SESSION_UNKNOWN,
    DeviceStore,
    DeviceUpdateError,
    BatchClaimSessionError,
    ClaimSessionError,
    DeliveryError,
    GroupError,
    GroupSessionRotationError,
    GroupSyncError,
    MessageCreateError,
    MessageListError,
    InboxRetryBatchError,
    InboxLeaseError,
    MessageSyncAckBatchError,
    MessageSyncError,
    PreKeyClaimError,
    PreKeyBatchClaimError,
    RedeliveryJobError,
    RedeliveryJobRecoverBatchError,
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
    MESSAGE_DUPLICATE_NONCE: (409, "nonce"),
    MESSAGE_REQUEST_ID_CONFLICT: (409, "request_id"),
}

#: Maps a storage-level session failure reason to (HTTP status, field name).
_SESSION_ERROR_MAP = {
    SESSION_INITIATOR_UNKNOWN: (404, "initiator_device_id"),
    SESSION_RECIPIENT_UNKNOWN: (404, "recipient_device_id"),
    SESSION_PREKEY_UNKNOWN: (404, "prekey_id"),
    SESSION_INITIATOR_REVOKED: (409, "initiator_device_id"),
    SESSION_RECIPIENT_REVOKED: (409, "recipient_device_id"),
    SESSION_PREKEY_REVOKED: (409, "prekey_id"),
    SESSION_PREKEY_CONSUMED: (409, "prekey_id"),
}

#: Maps a storage-level session-from-claim failure to (HTTP status, field).
_CLAIM_SESSION_ERROR_MAP = {
    CLAIM_SESSION_CLAIM_UNKNOWN: (404, "claim_id"),
    CLAIM_SESSION_DUPLICATE: (409, "claim_id"),
    CLAIM_SESSION_INITIATOR_UNKNOWN: (404, "initiator_device_id"),
    CLAIM_SESSION_INITIATOR_REVOKED: (409, "initiator_device_id"),
    CLAIM_SESSION_RECIPIENT_REVOKED: (409, "recipient_device_id"),
    CLAIM_SESSION_PREKEY_REVOKED: (409, "prekey_id"),
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
        #: Set by :func:`e2ee_backend.persistence.attach_persistence` to the
        #: durable state store; ``None`` marks the purely in-memory mode, in
        #: which the integrity probe answers 409/field=data_file.
        self.integrity_state_store: Optional[Any] = None

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

    # -- key-audit chain ---------------------------------------------------

    def list_key_events(self, device_id: str, after: int,
                        limit: int) -> Dict[str, Any]:
        """Return one ascending page of a device's key-audit chain.

        ``after`` must be a non-negative integer (0 replays from the chain's
        first event) and ``limit`` an integer in 1..100. The page carries the
        events with ``seq > after`` (at most ``limit``), ``next_after`` — the
        last returned ``seq``, or ``after`` itself when the page is empty —
        and ``has_more``. An unknown device is 404/field=device_id; a revoked
        device's chain stays readable.
        """
        if not isinstance(after, int) or isinstance(after, bool) or after < 0:
            raise ServiceError(
                "field must be a non-negative integer: after", "after")
        if not isinstance(limit, int) or isinstance(limit, bool) \
                or not 1 <= limit <= 100:
            raise ServiceError(
                "field must be an integer in 1..100: limit", "limit")
        page = self.store.key_events_page(device_id, after, limit)
        if page is None:
            raise ServiceError(f"device not found: {device_id}",
                               "device_id", status_code=404)
        events, next_after, has_more = page
        return {"events": events, "next_after": next_after,
                "has_more": has_more}

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

    # -- one-time pre-key claims ------------------------------------------

    def claim_prekey(self, payload: object) -> Tuple[Dict[str, Any], int]:
        """Validate a claim payload and atomically claim one pre-key.

        ``recipient_device_id`` and ``claim_id`` must be non-empty strings.
        The first available (registered-order, un-revoked, un-consumed)
        pre-key is marked consumed and returned with 201. A repeated
        ``claim_id`` returns the original response with 200 and consumes no
        further key; a different ``claim_id`` claims the next available key.
        Unknown recipient is 404, revoked recipient 409 (field
        ``recipient_device_id``), and no available key is 409/field
        ``prekey_id``. Returns ``(body, status_code)``.
        """
        if not isinstance(payload, dict):
            raise ServiceError("request body must be a JSON object",
                               "request_body")
        for name in ("recipient_device_id", "claim_id"):
            if name not in payload:
                raise ServiceError(f"missing required field: {name}", name)
            if not is_nonempty_string(payload[name]):
                raise ServiceError(
                    f"field must be a non-empty string: {name}", name)

        try:
            view, created = self.store.claim_prekey(
                payload["recipient_device_id"], payload["claim_id"])
        except PreKeyClaimError as error:
            raise self._claim_error(error, payload["recipient_device_id"])
        return view, 201 if created else 200

    @staticmethod
    def _claim_error(error: PreKeyClaimError,
                     recipient_device_id: str) -> ServiceError:
        """Translate a claim storage failure into a ServiceError."""
        if error.reason == CLAIM_RECIPIENT_UNKNOWN:
            return ServiceError(f"device not found: {recipient_device_id}",
                                "recipient_device_id", status_code=404)
        if error.reason == CLAIM_RECIPIENT_REVOKED:
            return ServiceError("recipient_device_id is revoked",
                                "recipient_device_id", status_code=409)
        if error.reason == CLAIM_ID_CONFLICT:
            return ServiceError(
                "claim_id was already used by a different kind of claim",
                "claim_id", status_code=409)
        return ServiceError("no pre-key available for this device",
                            "prekey_id", status_code=409)

    # -- user-wide batch pre-key claims -----------------------------------

    def claim_prekey_batch(self, payload: object) -> Tuple[Dict[str, Any], int]:
        """Validate a batch-claim payload and atomically claim per device.

        ``user_id`` and ``claim_id`` must be non-empty strings. Every active
        device registered to the user (registration order) contributes its
        first available (un-revoked, un-consumed) pre-key; all keys are
        consumed in one locked transaction or none at all. Success returns
        201 with ``claim_id``, ``user_id``, ``claimed_at`` and ``devices``;
        a repeated batch ``claim_id`` returns the original frozen response
        with 200 and consumes nothing. A user with no device is 404/field
        ``user_id``; a user with no active device is 409/field
        ``device_id``; an active device with no available key is
        409/field ``prekey_id``. Returns ``(body, status_code)``.
        """
        if not isinstance(payload, dict):
            raise ServiceError("request body must be a JSON object",
                               "request_body")
        for name in ("user_id", "claim_id"):
            if name not in payload:
                raise ServiceError(f"missing required field: {name}", name)
            if not is_nonempty_string(payload[name]):
                raise ServiceError(
                    f"field must be a non-empty string: {name}", name)

        try:
            view, created = self.store.claim_prekey_batch(
                payload["user_id"], payload["claim_id"])
        except PreKeyBatchClaimError as error:
            raise self._batch_claim_error(error, payload["user_id"])
        return view, 201 if created else 200

    @staticmethod
    def _batch_claim_error(error: PreKeyBatchClaimError,
                           user_id: str) -> ServiceError:
        """Translate a batch-claim storage failure into a ServiceError."""
        if error.reason == BATCH_CLAIM_USER_UNKNOWN:
            return ServiceError(f"user not found: {user_id}",
                                "user_id", status_code=404)
        if error.reason == BATCH_CLAIM_NO_ACTIVE_DEVICE:
            return ServiceError("user has no active device",
                                "device_id", status_code=409)
        if error.reason == CLAIM_ID_CONFLICT:
            return ServiceError(
                "claim_id was already used by a different kind of claim",
                "claim_id", status_code=409)
        return ServiceError("no pre-key available for one of the user's "
                            "devices", "prekey_id", status_code=409)

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
            elif error.reason == SESSION_PREKEY_CONSUMED:
                message = (f"prekey_id has already been claimed: "
                           f"{payload['prekey_id']}")
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

    def create_session_from_claim(self, payload: object) -> Dict[str, Any]:
        """Validate a from-claim session payload and atomically create it.

        ``claim_id``, ``initiator_device_id`` and ``ephemeral_key`` must be
        non-empty strings; the ephemeral key must parse with the existing
        public-key encoding. The recipient and pre-key are not client-supplied:
        they come from the frozen claim record. Each ``claim_id`` establishes
        at most one session — a repeat is 409/field=claim_id. On success the
        existing eight-field session snapshot is returned (201 at the HTTP
        layer).
        """
        if not isinstance(payload, dict):
            raise ServiceError("request body must be a JSON object",
                               "request_body")

        for name in ("claim_id", "initiator_device_id", "ephemeral_key"):
            if name not in payload:
                raise ServiceError(f"missing required field: {name}", name)
            if not is_nonempty_string(payload[name]):
                raise ServiceError(
                    f"field must be a non-empty string: {name}", name)

        if load_public_key(payload["ephemeral_key"]) is None:
            raise ServiceError(
                "field is not a valid public key: ephemeral_key",
                "ephemeral_key")

        try:
            session = self.store.create_session_from_claim(
                payload["claim_id"],
                payload["initiator_device_id"],
                payload["ephemeral_key"])
        except ClaimSessionError as error:
            status_code, field = _CLAIM_SESSION_ERROR_MAP[error.reason]
            if error.reason == CLAIM_SESSION_CLAIM_UNKNOWN:
                message = f"claim not found: {payload['claim_id']}"
            elif error.reason == CLAIM_SESSION_DUPLICATE:
                message = (f"claim_id has already established a session: "
                           f"{payload['claim_id']}")
            elif status_code == 404:
                message = f"device not found: {payload['initiator_device_id']}"
            else:
                message = f"{field} is revoked"
            raise ServiceError(message, field, status_code=status_code)

        return self.store.session_view(session.session_id)  # type: ignore[return-value]

    def create_sessions_from_batch_claim(self, payload: object
                                          ) -> Dict[str, Any]:
        """Validate a batch-claim session payload and atomically create all.

        ``claim_id`` and ``initiator_device_id`` must be non-empty strings;
        ``ephemeral_keys`` must be a non-empty array whose elements each
        carry a unique non-empty ``device_id`` and a valid public key
        ``ephemeral_key``. The request's device set must equal the frozen
        batch-claim snapshot (a mismatch is a 400 naming the whole array, or
        the offending item's ``device_id`` path when an extra device is
        given). An unknown batch claim id (including one naming a single
        claim) is a 404/field=claim_id; an already-bound batch claim is a
        409/claim_id. The initiator is 404/409 on unknown/revoked, and an
        initiator that is itself one of the claimed devices is a 400 naming
        ``initiator_device_id`` with no sessions created. A recipient device
        or claimed pre-key revoked after the claim is a 409 naming
        ``recipient_device_id`` / ``prekey_id``. On success the body carries
        ``claim_id`` and ``sessions``, the eight-field session views listed
        in the claim's frozen device order.
        """
        if not isinstance(payload, dict):
            raise ServiceError("request body must be a JSON object",
                               "request_body")
        for name in ("claim_id", "initiator_device_id"):
            if name not in payload:
                raise ServiceError(f"missing required field: {name}", name)
            if not is_nonempty_string(payload[name]):
                raise ServiceError(
                    f"field must be a non-empty string: {name}", name)

        if "ephemeral_keys" not in payload:
            raise ServiceError("missing required field: ephemeral_keys",
                               "ephemeral_keys")
        raw_entries = payload["ephemeral_keys"]
        if not isinstance(raw_entries, list) or not raw_entries:
            raise ServiceError(
                "field must be a non-empty array: ephemeral_keys",
                "ephemeral_keys")
        ordered: List[Tuple[str, str]] = []
        seen_devices: set = set()
        for index, element in enumerate(raw_entries):
            prefix = f"ephemeral_keys[{index}]"
            if not isinstance(element, dict):
                raise ServiceError(f"array element must be an object: {prefix}",
                                   prefix)
            for subfield in ("device_id", "ephemeral_key"):
                path = f"{prefix}.{subfield}"
                if subfield not in element:
                    raise ServiceError(f"missing required field: {path}", path)
                if not is_nonempty_string(element[subfield]):
                    raise ServiceError(
                        f"field must be a non-empty string: {path}", path)
            if element["device_id"] in seen_devices:
                raise ServiceError(
                    f"duplicate device_id in ephemeral_keys: "
                    f"{element['device_id']}",
                    f"{prefix}.device_id")
            if load_public_key(element["ephemeral_key"]) is None:
                raise ServiceError(
                    f"field is not a valid public key: {prefix}.ephemeral_key",
                    f"{prefix}.ephemeral_key")
            seen_devices.add(element["device_id"])
            ordered.append((element["device_id"], element["ephemeral_key"]))

        try:
            binding = self.store.create_sessions_from_batch_claim(
                payload["claim_id"], payload["initiator_device_id"], ordered)
        except BatchClaimSessionError as error:
            raise self._batch_session_error(error, payload, raw_entries)

        sessions = [self.store.session_view(entry.session_id)
                    for entry in binding.entries]
        return {"claim_id": binding.claim_id, "sessions": sessions}

    @staticmethod
    def _batch_session_error(error: BatchClaimSessionError,
                              payload: Dict[str, Any],
                              raw_entries: List[Dict[str, Any]]
                              ) -> ServiceError:
        """Translate a batch-session storage failure into a ServiceError."""
        reason = error.reason
        if reason == BATCH_SESSION_CLAIM_UNKNOWN:
            return ServiceError(
                f"batch claim not found: {payload['claim_id']}",
                "claim_id", status_code=404)
        if reason == BATCH_SESSION_CLAIM_WRONG_KIND:
            # claim_ids share one namespace: an id consumed by a single
            # claim is known but occupied by another claim kind -> 409.
            return ServiceError(
                "claim_id belongs to a single-device claim and cannot "
                "establish batch sessions",
                "claim_id", status_code=409)
        if reason == BATCH_SESSION_DUPLICATE:
            return ServiceError(
                f"claim_id has already established batch sessions: "
                f"{payload['claim_id']}", "claim_id", status_code=409)
        if reason == BATCH_SESSION_INITIATOR_UNKNOWN:
            return ServiceError(
                f"device not found: {payload['initiator_device_id']}",
                "initiator_device_id", status_code=404)
        if reason == BATCH_SESSION_INITIATOR_REVOKED:
            return ServiceError("initiator_device_id is revoked",
                                "initiator_device_id", status_code=409)
        if reason == BATCH_SESSION_INITIATOR_IN_SNAPSHOT:
            return ServiceError(
                "initiator_device_id must not be one of the claimed devices",
                "initiator_device_id", status_code=400)
        if reason == BATCH_SESSION_DEVICE_SET_MISMATCH:
            if error.detail:
                index = next((i for i, element in enumerate(raw_entries)
                              if element.get("device_id") == error.detail), 0)
                field = f"ephemeral_keys[{index}].device_id"
                message = (f"device is not part of the claim snapshot: "
                           f"{error.detail}")
            else:
                field = "ephemeral_keys"
                message = ("ephemeral_keys devices must equal the claim "
                           "snapshot devices")
            return ServiceError(message, field, status_code=400)
        if reason == BATCH_SESSION_RECIPIENT_REVOKED:
            return ServiceError(
                f"recipient_device_id is revoked: {error.detail}",
                "recipient_device_id", status_code=409)
        # BATCH_SESSION_PREKEY_REVOKED
        return ServiceError(
            f"prekey_id is revoked: {error.detail}",
            "prekey_id", status_code=409)

    # -- groups ------------------------------------------------------------

    #: Maps a storage-level group-membership failure to (HTTP status, field).
    _GROUP_MEMBER_ERROR_MAP = {
        GROUP_UNKNOWN: (404, "group_id"),
        GROUP_ACTOR_UNKNOWN: (404, "actor_device_id"),
        GROUP_ACTOR_REVOKED: (409, "actor_device_id"),
        GROUP_ACTOR_NOT_CREATOR: (409, "actor_device_id"),
        GROUP_DEVICE_UNKNOWN: (404, "device_id"),
        GROUP_DEVICE_REVOKED: (409, "device_id"),
    }

    def create_group(self, payload: object) -> Dict[str, Any]:
        """Validate a group-creation payload and atomically create the group.

        ``group_id`` and ``creator_device_id`` must be non-empty strings and
        ``member_device_ids`` a non-empty array of non-empty strings. The
        creator must be an active device; it always becomes the first member.
        A repeated ``group_id`` is a 409 naming ``group_id``.
        """
        if not isinstance(payload, dict):
            raise ServiceError("request body must be a JSON object",
                               "request_body")
        for name in ("group_id", "creator_device_id"):
            if name not in payload:
                raise ServiceError(f"missing required field: {name}", name)
            if not is_nonempty_string(payload[name]):
                raise ServiceError(
                    f"field must be a non-empty string: {name}", name)

        if "member_device_ids" not in payload:
            raise ServiceError("missing required field: member_device_ids",
                               "member_device_ids")
        raw_members = payload["member_device_ids"]
        if not isinstance(raw_members, list) or not raw_members:
            raise ServiceError(
                "field must be a non-empty array: member_device_ids",
                "member_device_ids")
        for index, element in enumerate(raw_members):
            if not is_nonempty_string(element):
                raise ServiceError(
                    "every member must be a non-empty string: "
                    f"member_device_ids[{index}]",
                    "member_device_ids")

        try:
            group = self.store.create_group(
                payload["group_id"], payload["creator_device_id"],
                list(raw_members))
        except GroupError as error:
            if error.reason == GROUP_DUPLICATE_ID:
                raise ServiceError(
                    f"group_id already exists: {payload['group_id']}",
                    "group_id", status_code=409)
            if error.reason == GROUP_CREATOR_UNKNOWN:
                raise ServiceError(
                    f"device not found: {payload['creator_device_id']}",
                    "creator_device_id", status_code=404)
            raise ServiceError("creator_device_id is revoked",
                               "creator_device_id", status_code=409)
        return self.store.group_view(group)

    def get_group(self, group_id: str) -> Dict[str, Any]:
        """Return the group's public snapshot; 404/field=group_id if unknown."""
        group = self.store.get_group(group_id)
        if group is None:
            raise ServiceError(f"group not found: {group_id}",
                               "group_id", status_code=404)
        return self.store.group_view(group)

    def _validate_member_change(self, payload: object
                                ) -> Tuple[str, str]:
        """Validate the shared body of add/remove-member requests."""
        if not isinstance(payload, dict):
            raise ServiceError("request body must be a JSON object",
                               "request_body")
        for name in ("actor_device_id", "device_id"):
            if name not in payload:
                raise ServiceError(f"missing required field: {name}", name)
            if not is_nonempty_string(payload[name]):
                raise ServiceError(
                    f"field must be a non-empty string: {name}", name)
        return payload["actor_device_id"], payload["device_id"]

    def _group_member_error(self, error: GroupError,
                            group_id: str) -> ServiceError:
        """Translate a storage membership failure into a ServiceError."""
        status_code, field = self._GROUP_MEMBER_ERROR_MAP[error.reason]
        if error.reason == GROUP_UNKNOWN:
            text = f"group not found: {group_id}"
        elif error.reason == GROUP_ACTOR_UNKNOWN:
            text = "actor_device_id is not a registered device"
        elif error.reason == GROUP_DEVICE_UNKNOWN:
            text = "device_id is not a registered device"
        elif field == "actor_device_id":
            text = "actor_device_id is revoked or is not the group creator"
        else:
            text = "device_id is revoked and cannot be added to the group"
        return ServiceError(text, field, status_code=status_code)

    def add_group_member(self, group_id: str,
                         payload: object) -> Tuple[Dict[str, Any], int]:
        """Validate and add one member; 201 created, 200 when already a member.

        Only the creator may change membership. Unknown group/actor/target
        device are 404 with the corresponding field; a revoked or
        non-creator actor and a revoked target are 409.
        """
        actor_device_id, device_id = self._validate_member_change(payload)
        try:
            group, created = self.store.add_group_member(
                group_id, actor_device_id, device_id)
        except GroupError as error:
            raise self._group_member_error(error, group_id)
        return self.store.group_view(group), 201 if created else 200

    def remove_group_member(self, group_id: str, payload: object
                            ) -> Dict[str, Any]:
        """Validate and remove one member; always 200 (idempotent).

        Only the creator may remove members. Removing a device that is not
        on the roster is a no-op; an id that has never been registered is
        404/field=device_id.
        """
        actor_device_id, device_id = self._validate_member_change(payload)
        try:
            group = self.store.remove_group_member(
                group_id, actor_device_id, device_id)
        except GroupError as error:
            raise self._group_member_error(error, group_id)
        return self.store.group_view(group)

    # -- group sessions ----------------------------------------------------

    def create_group_session(self, payload: object) -> Dict[str, Any]:
        """Validate a group-session payload and atomically create one.

        Every request creates a fresh session (a new ``session_id``);
        repeated POSTs are never deduplicated. The member list is frozen at
        creation. The initiator must be an active current member.
        """
        if not isinstance(payload, dict):
            raise ServiceError("request body must be a JSON object",
                               "request_body")
        for name in ("group_id", "initiator_device_id", "ephemeral_key"):
            if name not in payload:
                raise ServiceError(f"missing required field: {name}", name)
            if not is_nonempty_string(payload[name]):
                raise ServiceError(
                    f"field must be a non-empty string: {name}", name)

        try:
            session = self.store.create_group_session(
                payload["group_id"], payload["initiator_device_id"],
                payload["ephemeral_key"])
        except GroupError as error:
            if error.reason == GROUP_SESSION_GROUP_UNKNOWN:
                raise ServiceError(
                    f"group not found: {payload['group_id']}",
                    "group_id", status_code=404)
            if error.reason == GROUP_SESSION_INITIATOR_UNKNOWN:
                raise ServiceError(
                    f"device not found: {payload['initiator_device_id']}",
                    "initiator_device_id", status_code=404)
            raise ServiceError(
                "initiator_device_id is revoked or is not a group member",
                "initiator_device_id", status_code=409)
        return self.store.group_session_view(session)

    def get_group_session(self, session_id: str) -> Dict[str, Any]:
        """Return the frozen group-session snapshot; 404/field=session_id."""
        session = self.store.get_group_session(session_id)
        if session is None:
            raise ServiceError(f"session not found: {session_id}",
                               "session_id", status_code=404)
        return self.store.group_session_view(session)

    #: Maps a rotation failure reason to (HTTP status, field name).
    _ROTATION_ERROR_MAP = {
        ROTATION_SESSION_UNKNOWN: (404, "session_id"),
        ROTATION_ACTOR_UNKNOWN: (404, "actor_device_id"),
        ROTATION_ACTOR_REVOKED: (409, "actor_device_id"),
        ROTATION_ACTOR_NOT_CREATOR: (409, "actor_device_id"),
        ROTATION_REVISION_MISMATCH: (409, "expected_revision"),
        ROTATION_ID_CONFLICT: (409, "rotation_id"),
        ROTATION_PREDECESSOR_ROTATED: (409, "session_id"),
    }

    def rotate_group_session(self, predecessor_session_id: str,
                             payload: object
                             ) -> Tuple[Dict[str, Any], int]:
        """Validate a rotation payload and atomically rotate the predecessor.

        ``rotation_id`` and ``actor_device_id`` must be non-empty strings,
        ``ephemeral_key`` a non-empty valid public key and
        ``expected_revision`` a positive integer; the first rotation returns
        201, an idempotent replay (same id on the same predecessor) 200 with
        the original response.
        """
        if not isinstance(payload, dict):
            raise ServiceError("request body must be a JSON object",
                               "request_body")
        for name in ("rotation_id", "actor_device_id"):
            if name not in payload:
                raise ServiceError(f"missing required field: {name}", name)
            if not is_nonempty_string(payload[name]):
                raise ServiceError(
                    f"field must be a non-empty string: {name}", name)
        if "ephemeral_key" not in payload:
            raise ServiceError(
                "missing required field: ephemeral_key", "ephemeral_key")
        if not is_nonempty_string(payload["ephemeral_key"]):
            raise ServiceError(
                "field must be a non-empty string: ephemeral_key",
                "ephemeral_key")
        if load_public_key(payload["ephemeral_key"]) is None:
            raise ServiceError(
                "field is not a valid public key: ephemeral_key",
                "ephemeral_key")
        if "expected_revision" not in payload:
            raise ServiceError(
                "missing required field: expected_revision",
                "expected_revision")
        expected_revision = payload["expected_revision"]
        if not isinstance(expected_revision, int) \
                or isinstance(expected_revision, bool) \
                or expected_revision <= 0:
            raise ServiceError(
                "field must be a positive integer: expected_revision",
                "expected_revision")

        try:
            successor, rotation, created = self.store.rotate_group_session(
                predecessor_session_id, payload["rotation_id"],
                payload["actor_device_id"], payload["ephemeral_key"],
                expected_revision)
        except GroupSessionRotationError as error:
            status_code, field = self._ROTATION_ERROR_MAP[error.reason]
            if error.reason == ROTATION_SESSION_UNKNOWN:
                message = f"session not found: {predecessor_session_id}"
            elif error.reason == ROTATION_ACTOR_UNKNOWN:
                message = ("actor_device_id is not a registered device: "
                           f"{payload['actor_device_id']}")
            elif error.reason == ROTATION_REVISION_MISMATCH:
                message = (
                    "expected_revision does not match the group's current "
                    "revision")
            elif error.reason == ROTATION_ID_CONFLICT:
                message = (
                    "rotation_id has already rotated another session: "
                    f"{payload['rotation_id']}")
            elif error.reason == ROTATION_PREDECESSOR_ROTATED:
                message = "session has already been rotated"
            else:
                message = (
                    "actor_device_id is revoked or is not the group creator")
            raise ServiceError(message, field, status_code=status_code)
        return self.store.rotation_view(successor, rotation), \
            201 if created else 200

    # -- group-session sync ------------------------------------------------

    #: Maps a group-session sync failure reason to (HTTP status, field name).
    #: Only an unknown session is a 404; every device-side problem (unknown,
    #: revoked, or outside the frozen member set) is a 409 naming device_id,
    #: mirroring the group-session message read contract.
    _SYNC_ERROR_MAP = {
        SYNC_SESSION_UNKNOWN: (404, "session_id"),
        SYNC_DEVICE_UNKNOWN: (409, "device_id"),
        SYNC_DEVICE_INACTIVE: (409, "device_id"),
        SYNC_DEVICE_NOT_MEMBER: (409, "device_id"),
        SYNC_CURSOR_CONFLICT: (409, "cursor"),
    }

    def _sync_error(self, error: GroupSyncError,
                    session_id: str) -> ServiceError:
        """Translate a storage :class:`GroupSyncError` into a ServiceError."""
        status_code, field = self._SYNC_ERROR_MAP[error.reason]
        if error.reason == SYNC_SESSION_UNKNOWN:
            text = f"session not found: {session_id}"
        elif error.reason == SYNC_CURSOR_CONFLICT:
            text = "cursor is out of range or moved backwards"
        elif error.reason == SYNC_DEVICE_UNKNOWN:
            text = "device_id is not a registered device"
        elif error.reason == SYNC_DEVICE_INACTIVE:
            text = "device_id is revoked"
        else:
            text = "device_id is not a frozen member of this group session"
        return ServiceError(text, field, status_code=status_code)

    def sync_group_messages(self, session_id: str, device_id: str,
                            after: Optional[int], limit: int) -> Dict[str, Any]:
        """Return one ascending sync page of a group session's messages.

        With *after* ``None`` the device's stored cursor is the start and is
        advanced under the store lock (an empty page leaves it unchanged); an
        explicit non-negative *after* queries from there without touching the
        stored cursor. Authorization follows the frozen member set.
        """
        try:
            return self.store.group_sync_page(
                session_id, device_id, after, limit)
        except GroupSyncError as error:
            raise self._sync_error(error, session_id)

    def sync_group_checkpoint(self, session_id: str,
                              payload: object) -> Tuple[Dict[str, Any], int]:
        """Validate and move a device's group-session sync cursor.

        ``device_id`` must be a non-empty string and ``cursor`` an integer in
        ``0..max_sequence``. A forward move returns 201 with a refreshed
        ``updated_at``; an equal cursor returns 200 unchanged; a backward or
        out-of-range cursor returns 409/field=cursor. Unknown session is
        404/session_id; an unknown, revoked or non-frozen-member device is
        409/device_id.
        """
        if not isinstance(payload, dict):
            raise ServiceError("request body must be a JSON object",
                               "request_body")
        if "device_id" not in payload:
            raise ServiceError("missing required field: device_id", "device_id")
        if not is_nonempty_string(payload["device_id"]):
            raise ServiceError(
                "field must be a non-empty string: device_id", "device_id")
        if "cursor" not in payload:
            raise ServiceError("missing required field: cursor", "cursor")
        cursor = payload["cursor"]
        # bool is a subclass of int; reject it explicitly.
        if not isinstance(cursor, int) or isinstance(cursor, bool):
            raise ServiceError("field must be an integer: cursor", "cursor")
        if cursor < 0:
            raise ServiceError("field must be a non-negative integer: cursor",
                               "cursor")

        try:
            view, advanced = self.store.group_sync_checkpoint(
                session_id, payload["device_id"], cursor)
        except GroupSyncError as error:
            raise self._sync_error(error, session_id)
        return view, 201 if advanced else 200

    # -- unified 1:1/group-session sync ------------------------------------

    #: Maps a unified session-sync failure reason to (HTTP status, field).
    #: An unknown session (neither 1:1 nor group) is a 404; every device-side
    #: problem (unknown, revoked, or not a session participant) is a 409
    #: naming device_id, mirroring both message-read contracts.
    _MESSAGE_SYNC_ERROR_MAP = {
        MESSAGE_SYNC_SESSION_UNKNOWN: (404, "session_id"),
        MESSAGE_SYNC_DEVICE_UNKNOWN: (409, "device_id"),
        MESSAGE_SYNC_DEVICE_INACTIVE: (409, "device_id"),
        MESSAGE_SYNC_DEVICE_NOT_PARTICIPANT: (409, "device_id"),
        MESSAGE_SYNC_CURSOR_CONFLICT: (409, "cursor"),
    }

    def _message_sync_error(self, error: MessageSyncError,
                            session_id: str) -> ServiceError:
        """Translate a storage :class:`MessageSyncError` into a ServiceError."""
        status_code, field = self._MESSAGE_SYNC_ERROR_MAP[error.reason]
        if error.reason == MESSAGE_SYNC_SESSION_UNKNOWN:
            text = f"session not found: {session_id}"
        elif error.reason == MESSAGE_SYNC_CURSOR_CONFLICT:
            text = "cursor is out of range or moved backwards"
        elif error.reason == MESSAGE_SYNC_DEVICE_UNKNOWN:
            text = "device_id is not a registered device"
        elif error.reason == MESSAGE_SYNC_DEVICE_INACTIVE:
            text = "device_id is revoked"
        else:
            text = "device_id is not a participant of this session"
        return ServiceError(text, field, status_code=status_code)

    def sync_session_messages(self, session_id: str, device_id: str,
                              after: Optional[int], limit: int) -> Dict[str, Any]:
        """Return one ascending sync page of a 1:1 or group session's messages.

        With *after* ``None`` the device's stored cursor is the start and is
        advanced under the store lock (an empty page leaves it unchanged); an
        explicit non-negative *after* queries from there without touching the
        stored cursor. A 1:1 session authorizes its two endpoints; a group
        session the frozen member set. Unknown session is 404/session_id; an
        unknown, revoked or non-participant device is 409/device_id.
        """
        try:
            return self.store.message_sync_page(
                session_id, device_id, after, limit)
        except MessageSyncError as error:
            raise self._message_sync_error(error, session_id)

    def sync_session_checkpoint(self, session_id: str,
                                payload: object) -> Tuple[Dict[str, Any], int]:
        """Validate and move a device's unified session-sync cursor.

        ``device_id`` must be a non-empty string and ``cursor`` an integer in
        ``0..max_sequence``. A forward move returns 201 with a refreshed
        ``updated_at``; an equal cursor returns 200 unchanged; a backward or
        out-of-range cursor returns 409/field=cursor. Unknown session is
        404/session_id; an unknown, revoked or non-participant device is
        409/device_id.
        """
        if not isinstance(payload, dict):
            raise ServiceError("request body must be a JSON object",
                               "request_body")
        if "device_id" not in payload:
            raise ServiceError("missing required field: device_id", "device_id")
        if not is_nonempty_string(payload["device_id"]):
            raise ServiceError(
                "field must be a non-empty string: device_id", "device_id")
        if "cursor" not in payload:
            raise ServiceError("missing required field: cursor", "cursor")
        cursor = payload["cursor"]
        # bool is a subclass of int; reject it explicitly.
        if not isinstance(cursor, int) or isinstance(cursor, bool):
            raise ServiceError("field must be an integer: cursor", "cursor")
        if cursor < 0:
            raise ServiceError("field must be a non-negative integer: cursor",
                               "cursor")

        try:
            view, advanced = self.store.message_sync_checkpoint(
                session_id, payload["device_id"], cursor)
        except MessageSyncError as error:
            raise self._message_sync_error(error, session_id)
        return view, 201 if advanced else 200

    def sync_session_ack(self, session_id: str,
                         payload: object) -> Tuple[Dict[str, Any], int]:
        """Validate and apply a batch offline-message acknowledgement.

        The body carries a non-empty ``device_id`` and an integer non-negative
        non-bool ``cursor``. Advancing the cursor acknowledges, one by one and
        in the same locked transaction as the cursor advance, every receivable
        message in ``(old_cursor, cursor]`` (a 1:1 session acks for the
        recipient; a group session acks for the frozen member device and skips
        the device's own messages) without changing any retry ``attempts``. A
        forward move returns 201 with a fresh UTC ISO-8601 ``updated_at``; an
        equal cursor returns 200 with the timestamp untouched and writes
        nothing (a never-advanced cursor reports the session's
        ``created_at``). A cursor below the stored cursor or above the
        session's max sequence is 409/field=cursor. Unknown session is
        404/session_id; an unknown, revoked or unauthorized device (recipient
        only for 1:1, frozen member for a group) is 409/device_id.
        """
        if not isinstance(payload, dict):
            raise ServiceError("request body must be a JSON object",
                               "request_body")
        if "device_id" not in payload:
            raise ServiceError("missing required field: device_id", "device_id")
        if not is_nonempty_string(payload["device_id"]):
            raise ServiceError(
                "field must be a non-empty string: device_id", "device_id")
        if "cursor" not in payload:
            raise ServiceError("missing required field: cursor", "cursor")
        cursor = payload["cursor"]
        # bool is a subclass of int; reject it explicitly.
        if not isinstance(cursor, int) or isinstance(cursor, bool):
            raise ServiceError("field must be an integer: cursor", "cursor")
        if cursor < 0:
            raise ServiceError("field must be a non-negative integer: cursor",
                               "cursor")

        try:
            view, advanced = self.store.message_sync_ack(
                session_id, payload["device_id"], cursor)
        except MessageSyncError as error:
            raise self._message_sync_error(error, session_id)
        return view, 201 if advanced else 200

    def sync_device_ack_batch(self, device_id: str,
                               payload: object) -> Tuple[Dict[str, Any], int]:
        """Validate and apply one device's multi-session batch sync-ack.

        ``POST /v1/devices/{device_id}/sync/ack-batch``. The body must be an
        object carrying a non-empty ``items`` array of objects, each with a
        non-empty string ``session_id`` and a non-negative non-bool integer
        ``cursor``, with no repeated session. Shape errors are reported, in
        order, as 400/field ``request_body`` (bad/non-object body),
        ``items`` (missing/not-a-non-empty-array), ``items[i]`` (non-object
        element or repeated session) or ``items[i].session_id`` /
        ``items[i].cursor`` for the offending field.

        The device is then resolved (unknown/revoked -> 409/field
        ``device_id``) and items are validated in array order, the first error
        aborting the whole batch with nothing written: an unknown session is
        404/``items[i].session_id``; a group session or a 1:1 session the
        device is not the recipient of is 409/``items[i].session_id``; a cursor
        below the stored cursor (0 initially) or above the session's max
        sequence is 409/``items[i].cursor``. On success the body is
        ``device_id`` then ``results``; results keep input order and each item
        is ``session_id``/``cursor``/``updated_at``. Status is 201 when at
        least one cursor advanced and 200 otherwise.
        """
        if not isinstance(payload, dict):
            raise ServiceError("request body must be a JSON object",
                               "request_body")
        if "items" not in payload:
            raise ServiceError("missing required field: items", "items")
        raw_items = payload["items"]
        if not isinstance(raw_items, list) or not raw_items:
            raise ServiceError(
                "field must be a non-empty array: items", "items")

        items: List[Tuple[str, int]] = []
        seen_sessions: set = set()
        for index, element in enumerate(raw_items):
            item_field = f"items[{index}]"
            if not isinstance(element, dict):
                raise ServiceError(
                    f"array element must be an object: {item_field}",
                    item_field)
            session_field = f"{item_field}.session_id"
            if "session_id" not in element:
                raise ServiceError(
                    f"missing required field: {session_field}", session_field)
            if not is_nonempty_string(element["session_id"]):
                raise ServiceError(
                    f"field must be a non-empty string: {session_field}",
                    session_field)
            cursor_field = f"{item_field}.cursor"
            if "cursor" not in element:
                raise ServiceError(
                    f"missing required field: {cursor_field}", cursor_field)
            cursor = element["cursor"]
            # bool is a subclass of int; reject it explicitly.
            if not isinstance(cursor, int) or isinstance(cursor, bool):
                raise ServiceError(
                    f"field must be an integer: {cursor_field}", cursor_field)
            if cursor < 0:
                raise ServiceError(
                    f"field must be a non-negative integer: {cursor_field}",
                    cursor_field)
            if element["session_id"] in seen_sessions:
                raise ServiceError(
                    "duplicate session_id in items: "
                    f"{element['session_id']}", item_field)
            seen_sessions.add(element["session_id"])
            items.append((element["session_id"], cursor))

        try:
            results, any_advanced = self.store.message_sync_ack_batch(
                device_id, items)
        except MessageSyncError as error:
            # Batch-level device failure (unknown/revoked): 409/device_id.
            raise self._message_sync_error(error, device_id)
        except MessageSyncAckBatchError as error:
            session_id = items[error.index][0]
            status_code, _ = self._MESSAGE_SYNC_ERROR_MAP[error.reason]
            if error.reason == MESSAGE_SYNC_SESSION_UNKNOWN:
                field = f"items[{error.index}].session_id"
                text = f"session not found: {session_id}"
            elif error.reason == MESSAGE_SYNC_CURSOR_CONFLICT:
                field = f"items[{error.index}].cursor"
                text = "cursor is out of range or moved backwards"
            else:
                # A group session, or a 1:1 session the device is not the
                # recipient of, is not ack-able here.
                field = f"items[{error.index}].session_id"
                text = ("session is a group session or device is not its "
                        "recipient")
            raise ServiceError(text, field, status_code=status_code)
        body = {"device_id": device_id, "results": results}
        return body, 201 if any_advanced else 200

    def device_inbox(self, device_id: str, limit: int) -> Dict[str, Any]:
        """Return one device's aggregated 1:1 offline-inbox page (read-only).

        ``GET /v1/devices/{device_id}/inbox``. ``limit`` must be an integer in
        1..100 (the HTTP layer defaults it to 100 and rejects repeats,
        non-decimals and out-of-range values with 400/field=limit). An
        unknown or revoked device is 409/field=device_id. On success the body
        is ``device_id``, ``messages``, ``has_more`` in that key order;
        ``messages`` holds at most ``limit`` seven-field envelopes (the
        ``message_view`` key order) of the unacked messages of every 1:1
        session the device is the recipient of — group sessions and other
        devices' sessions never contribute — ordered by
        ``(session.created_at, session_id, sequence)``; ``has_more`` says
        whether the locked snapshot had further entries. The query is purely
        read-only: it shares the store lock with submission, ack and
        revocation but writes nothing, advances no cursor or ``commit_seq``
        and touches no ``attempts``, so an unchanged state answers
        byte-identically and a restart rebuilds the same view from
        ``messages`` and ``delivery``.
        """
        if not isinstance(limit, int) or isinstance(limit, bool) \
                or not 1 <= limit <= 100:
            raise ServiceError(
                "field must be an integer in 1..100: limit", "limit")
        try:
            return self.store.device_inbox(device_id, limit)
        except MessageSyncError as error:
            if error.reason == MESSAGE_SYNC_DEVICE_UNKNOWN:
                raise ServiceError("device_id is not a registered device",
                                   "device_id", status_code=409)
            raise ServiceError("device_id is revoked",
                               "device_id", status_code=409)

    def device_inbox_wait(self, device_id: str, limit: int,
                          timeout_ms: int) -> Dict[str, Any]:
        """Long-poll one device's aggregated 1:1 offline inbox (read-only).

        ``GET /v1/devices/{device_id}/inbox/wait``. ``limit`` must be a
        non-boolean integer in 1..100 (HTTP default 100) and ``timeout_ms`` a
        non-boolean integer in 0..30000 (HTTP default 30000); the HTTP layer
        additionally rejects repeats/non-decimals with 400 on the matching
        field before this is reached. An unknown or revoked device is
        409/field=device_id. The success body is identical in shape and key
        order to :meth:`device_inbox` (``device_id``, ``messages``,
        ``has_more``); an already-populated inbox returns immediately, and an
        empty one waits on the store condition until a message becomes
        deliverable, the device is revoked or the monotonic deadline passes,
        in which case the body carries ``messages=[]`` and
        ``has_more=false``. The wait is purely read-only: it writes nothing,
        advances no cursor or ``commit_seq`` and touches no lease or
        ``attempts``.
        """
        if not isinstance(limit, int) or isinstance(limit, bool) \
                or not 1 <= limit <= 100:
            raise ServiceError(
                "field must be an integer in 1..100: limit", "limit")
        if not isinstance(timeout_ms, int) or isinstance(timeout_ms, bool) \
                or not 0 <= timeout_ms <= 30000:
            raise ServiceError(
                "field must be an integer in 0..30000: timeout_ms",
                "timeout_ms")
        try:
            return self.store.device_inbox_wait(device_id, limit, timeout_ms)
        except MessageSyncError as error:
            if error.reason == MESSAGE_SYNC_DEVICE_UNKNOWN:
                raise ServiceError("device_id is not a registered device",
                                   "device_id", status_code=409)
            raise ServiceError("device_id is revoked",
                               "device_id", status_code=409)

    def inbox_retry_batch(self, device_id: str,
                          payload: object) -> Tuple[Dict[str, Any], int]:
        """Validate and apply one device's batch retry of 1:1 inbox messages.

        ``POST /v1/devices/{device_id}/inbox/retry-batch``. The body must be
        an object carrying a non-empty string ``attempt_id`` and a non-empty
        ``items`` array of objects, each with a non-empty string
        ``session_id`` and ``message_id``, with no repeated
        ``(session_id, message_id)`` pair. Shape errors are reported, in
        order, as 400/field ``request_body`` (bad/non-object body),
        ``attempt_id`` (missing/empty/non-string), ``items``
        (missing/not-a-non-empty-array), ``items[i]`` (non-object element or
        a repeated pair) or ``items[i].session_id`` /
        ``items[i].message_id`` for the offending field.

        The device is then resolved (unknown/revoked -> 409/field
        ``device_id``) and items are prechecked in array order, the first
        error aborting the whole batch with nothing written: an unknown
        session is 404/``items[i].session_id``; a group session or a 1:1
        session the device is not the recipient of is
        409/``items[i].session_id``; an unknown message is
        404/``items[i].message_id``; an already-acked message is
        409/``items[i].message_id``. On success the body is ``device_id``
        then ``results``; results keep input order and each item is
        ``session_id``/``message_id``/``attempts``. Status is 201 when at
        least one message newly recorded the attempt id and 200 (with no
        durable write) when every id was a replay.
        """
        if not isinstance(payload, dict):
            raise ServiceError("request body must be a JSON object",
                               "request_body")
        if "attempt_id" not in payload:
            raise ServiceError("missing required field: attempt_id",
                               "attempt_id")
        if not is_nonempty_string(payload["attempt_id"]):
            raise ServiceError(
                "field must be a non-empty string: attempt_id", "attempt_id")
        if "items" not in payload:
            raise ServiceError("missing required field: items", "items")
        raw_items = payload["items"]
        if not isinstance(raw_items, list) or not raw_items:
            raise ServiceError(
                "field must be a non-empty array: items", "items")

        items: List[Tuple[str, str]] = []
        seen_pairs: set = set()
        for index, element in enumerate(raw_items):
            item_field = f"items[{index}]"
            if not isinstance(element, dict):
                raise ServiceError(
                    f"array element must be an object: {item_field}",
                    item_field)
            session_field = f"{item_field}.session_id"
            if "session_id" not in element:
                raise ServiceError(
                    f"missing required field: {session_field}", session_field)
            if not is_nonempty_string(element["session_id"]):
                raise ServiceError(
                    f"field must be a non-empty string: {session_field}",
                    session_field)
            message_field = f"{item_field}.message_id"
            if "message_id" not in element:
                raise ServiceError(
                    f"missing required field: {message_field}", message_field)
            if not is_nonempty_string(element["message_id"]):
                raise ServiceError(
                    f"field must be a non-empty string: {message_field}",
                    message_field)
            pair = (element["session_id"], element["message_id"])
            if pair in seen_pairs:
                raise ServiceError(
                    "duplicate (session_id, message_id) pair in items: "
                    f"({pair[0]}, {pair[1]})", item_field)
            seen_pairs.add(pair)
            items.append(pair)

        try:
            results, any_new = self.store.inbox_retry_batch(
                device_id, payload["attempt_id"], items)
        except MessageSyncError as error:
            # Batch-level device failure (unknown/revoked): 409/device_id.
            raise self._message_sync_error(error, device_id)
        except InboxRetryBatchError as error:
            session_id, message_id = items[error.index]
            item_field = f"items[{error.index}]"
            if error.reason == INBOX_RETRY_SESSION_UNKNOWN:
                raise ServiceError(f"session not found: {session_id}",
                                   f"{item_field}.session_id",
                                   status_code=404)
            if error.reason == INBOX_RETRY_NOT_RECIPIENT:
                raise ServiceError(
                    "session is a group session or device is not its "
                    "recipient", f"{item_field}.session_id",
                    status_code=409)
            if error.reason == INBOX_RETRY_MESSAGE_UNKNOWN:
                raise ServiceError(f"message not found: {message_id}",
                                   f"{item_field}.message_id",
                                   status_code=404)
            raise ServiceError(
                "message has already been acknowledged",
                f"{item_field}.message_id", status_code=409)
        body = {"device_id": device_id, "results": results}
        return body, 201 if any_new else 200

    def inbox_claim(self, device_id: str,
                    payload: object) -> Tuple[Dict[str, Any], int]:
        """Validate and apply one 1:1-inbox redelivery lease claim.

        ``POST /v1/devices/{device_id}/inbox/claim``. The body must be a
        JSON object carrying a non-empty string ``lease_id`` and a
        non-boolean integer ``limit`` in 1..100. A bad/non-object body is
        400/field ``request_body``; a missing or malformed field is 400 with
        the corresponding ``field`` (``lease_id`` / ``limit``).

        The path device must be registered and not revoked, else
        409/field=device_id (checked in the store under the lock, so it is
        linearized against revocation). An occupied ``lease_id`` replayed for
        the same device and the same limit is idempotent: 200 with the first
        response (non-empty leases keep their original ``leased_until``); the
        same id for another device or with another limit is 409/field
        lease_id. A fresh claim that leases at least one message returns 201
        with the UTC ISO-8601 deadline (30 seconds out, six microsecond
        digits, ``+00:00``); an empty selection returns 200 with
        ``leased_until`` null and writes nothing (the id stays free).
        """
        if not isinstance(payload, dict):
            raise ServiceError("request body must be a JSON object",
                               "request_body")
        if "lease_id" not in payload:
            raise ServiceError("missing required field: lease_id",
                               "lease_id")
        if not is_nonempty_string(payload["lease_id"]):
            raise ServiceError(
                "field must be a non-empty string: lease_id", "lease_id")
        if "limit" not in payload:
            raise ServiceError("missing required field: limit", "limit")
        limit = payload["limit"]
        if not isinstance(limit, int) or isinstance(limit, bool) \
                or not 1 <= limit <= 100:
            raise ServiceError(
                "field must be an integer in 1..100: limit", "limit")

        try:
            body, status_code, _ = self.store.inbox_claim(
                device_id, payload["lease_id"], limit)
        except InboxLeaseError as error:
            if error.reason == INBOX_LEASE_CONFLICT:
                raise ServiceError(
                    "lease_id is already used by another device or with a "
                    "different limit", "lease_id", status_code=409)
            if error.reason == INBOX_LEASE_DEVICE_UNKNOWN:
                raise ServiceError("device_id is not a registered device",
                                   "device_id", status_code=409)
            raise ServiceError("device_id is revoked",
                               "device_id", status_code=409)
        return body, status_code

    def inbox_release(self, device_id: str,
                      lease_id: str) -> Tuple[Dict[str, Any], int]:
        """Release one occupied 1:1-inbox lease before its deadline.

        ``POST /v1/devices/{device_id}/inbox/leases/{lease_id}/release``
        (no request body; the HTTP layer rejects a non-empty one with
        400/field ``request_body``). A never-committed ``lease_id`` is
        404/field ``lease_id``; a lease owned by another device is
        409/field ``lease_id``; both are decided in the store under the
        lock, ahead of the path device's state. A first release on a
        revoked device is 409/field ``device_id``.

        A first release returns 201 with ``device_id``, ``lease_id``,
        ``released_at`` (UTC ISO-8601, six microsecond digits, ``+00:00``)
        and ``released_count`` (the number of messages the lease had
        claimed) in that key order, and persists one generation; a repeat
        release returns 200 with the first response byte-identically, even
        if the device has since been revoked. The released lease stops
        withholding its messages from new claims; replaying the original
        claim still returns its frozen first response and reactivates
        nothing.
        """
        try:
            return self.store.inbox_release(device_id, lease_id)
        except InboxLeaseError as error:
            if error.reason == INBOX_LEASE_NOT_FOUND:
                raise ServiceError(f"lease not found: {lease_id}",
                                   "lease_id", status_code=404)
            if error.reason == INBOX_LEASE_CONFLICT:
                raise ServiceError(
                    "lease_id is owned by another device",
                    "lease_id", status_code=409)
            if error.reason == INBOX_LEASE_UNAVAILABLE:
                raise ServiceError(
                    "lease is already completed and cannot be released",
                    "lease_id", status_code=409)
            if error.reason == INBOX_LEASE_DEVICE_UNKNOWN:
                raise ServiceError("device_id is not a registered device",
                                   "device_id", status_code=409)
            raise ServiceError("device_id is revoked",
                               "device_id", status_code=409)

    def inbox_lease_renew(self, device_id: str, lease_id: str,
                          payload: object) -> Tuple[Dict[str, Any], int]:
        """Validate and apply one 1:1-inbox lease renewal.

        ``POST /v1/devices/{device_id}/inbox/leases/{lease_id}/renew``.
        The body must be a JSON object carrying a non-empty string
        ``renewal_id``. A bad/non-object body is 400/field
        ``request_body``; a missing, empty or wrongly typed field is
        400/field ``renewal_id``.

        The lease is resolved in the store under the lock, ahead of the
        path device's state: a never-committed ``lease_id`` is 404/field
        ``lease_id`` and a lease owned by another device is 409/field
        ``lease_id``. Replaying the same ``renewal_id`` on the same lease
        returns the frozen first response with 200 (ids may recur on
        other leases). Only a first renewal checks the device (unknown or
        revoked -> 409/field ``device_id``) and the lease state (already
        released or expired -> 409/field ``lease_id``). A first renewal
        returns 201 with ``device_id``, ``lease_id``, ``renewal_id`` and
        the new ``leased_until`` (the previous effective deadline plus
        exactly 30 seconds, UTC ISO-8601 with six microsecond digits and
        ``+00:00``), and persists one generation.
        """
        if not isinstance(payload, dict):
            raise ServiceError("request body must be a JSON object",
                               "request_body")
        if "renewal_id" not in payload:
            raise ServiceError("missing required field: renewal_id",
                               "renewal_id")
        if not is_nonempty_string(payload["renewal_id"]):
            raise ServiceError(
                "field must be a non-empty string: renewal_id",
                "renewal_id")

        try:
            return self.store.inbox_lease_renew(
                device_id, lease_id, payload["renewal_id"])
        except InboxLeaseError as error:
            if error.reason == INBOX_LEASE_NOT_FOUND:
                raise ServiceError(f"lease not found: {lease_id}",
                                   "lease_id", status_code=404)
            if error.reason == INBOX_LEASE_CONFLICT:
                raise ServiceError(
                    "lease_id is owned by another device",
                    "lease_id", status_code=409)
            if error.reason == INBOX_LEASE_UNAVAILABLE:
                raise ServiceError(
                    "lease is released or expired and cannot be renewed",
                    "lease_id", status_code=409)
            if error.reason == INBOX_LEASE_DEVICE_UNKNOWN:
                raise ServiceError("device_id is not a registered device",
                                   "device_id", status_code=409)
            raise ServiceError("device_id is revoked",
                               "device_id", status_code=409)

    def inbox_lease_complete(self, device_id: str, lease_id: str,
                             payload: object) -> Tuple[Dict[str, Any], int]:
        """Validate and apply one 1:1-inbox lease completion.

        ``POST /v1/devices/{device_id}/inbox/leases/{lease_id}/complete``.
        The body must be a JSON object carrying a non-empty string
        ``completion_id`` and an ``outcome`` of exactly ``delivered`` or
        ``failed``. A bad/non-object body is 400/field ``request_body``; a
        missing, empty or wrongly typed ``completion_id`` is
        400/completion_id and a missing or non-allowed ``outcome`` is
        400/outcome.

        The lease is resolved in the store under the lock, ahead of the
        path device's state: a never-committed ``lease_id`` is 404/field
        ``lease_id`` and a lease owned by another device is 409/field
        ``lease_id``. Replaying the same ``completion_id`` on the same
        lease returns the frozen first response with 200 (ids may recur on
        other leases), winning over every later state; reusing the id with
        a different outcome, or completing the lease again under another
        id, is 409/field ``completion_id``. Only a first completion checks
        the device (unknown or revoked -> 409/field ``device_id``) and the
        lease state (already released or expired -> 409/field
        ``lease_id``). A first completion returns 201 with ``device_id``,
        ``lease_id``, ``completion_id``, ``outcome`` and ``completed_at``
        (UTC ISO-8601 with six microsecond digits and ``+00:00``) in that
        key order, and persists one generation. Completion ends the lease:
        its still-unacked messages may be claimed again (``delivered`` is
        not an acknowledgement) and no renewal or first release follows.
        """
        if not isinstance(payload, dict):
            raise ServiceError("request body must be a JSON object",
                               "request_body")
        if "completion_id" not in payload:
            raise ServiceError("missing required field: completion_id",
                               "completion_id")
        if not is_nonempty_string(payload["completion_id"]):
            raise ServiceError(
                "field must be a non-empty string: completion_id",
                "completion_id")
        if "outcome" not in payload:
            raise ServiceError("missing required field: outcome", "outcome")
        outcome = payload["outcome"]
        if not isinstance(outcome, str) or outcome not in ("delivered",
                                                           "failed"):
            raise ServiceError(
                "field must be one of 'delivered' or 'failed': outcome",
                "outcome")

        try:
            return self.store.inbox_lease_complete(
                device_id, lease_id, payload["completion_id"], outcome)
        except InboxLeaseError as error:
            if error.reason == INBOX_LEASE_NOT_FOUND:
                raise ServiceError(f"lease not found: {lease_id}",
                                   "lease_id", status_code=404)
            if error.reason == INBOX_LEASE_CONFLICT:
                raise ServiceError(
                    "lease_id is owned by another device",
                    "lease_id", status_code=409)
            if error.reason == INBOX_LEASE_COMPLETION_CONFLICT:
                raise ServiceError(
                    "completion_id is already used on this lease or the "
                    "lease has already been completed",
                    "completion_id", status_code=409)
            if error.reason == INBOX_LEASE_UNAVAILABLE:
                raise ServiceError(
                    "lease is released or expired and cannot be completed",
                    "lease_id", status_code=409)
            if error.reason == INBOX_LEASE_DEVICE_UNKNOWN:
                raise ServiceError("device_id is not a registered device",
                                   "device_id", status_code=409)
            raise ServiceError("device_id is revoked",
                               "device_id", status_code=409)

    def inbox_lease_ack(self, device_id: str,
                        lease_id: str) -> Tuple[Dict[str, Any], int]:
        """Bulk-acknowledge every message of one delivered 1:1-inbox lease.

        ``POST /v1/devices/{device_id}/inbox/leases/{lease_id}/ack`` (no
        request body; the HTTP layer rejects a non-empty one with
        400/field ``request_body``). The lease is resolved in the store
        under the lock, ahead of the path device's state: a
        never-committed ``lease_id`` is 404/field ``lease_id`` and a lease
        owned by another device is 409/field ``lease_id``.

        A lease whose messages are already all acked answers 200 and
        writes nothing — even if the device has since been revoked; that
        replay takes precedence. Otherwise only a lease completed with
        ``completion.outcome == "delivered"`` may be acked: an active,
        expired, released or ``failed`` lease is 409/field ``lease_id``.
        A deliverable lease on an unknown or revoked device is
        409/field ``device_id``. A first ack returns 201, sets every
        leased message's delivery record acked with ``ack_sequence`` the
        message sequence (attempts and attempt ids unchanged), and
        persists one generation. The body keys are ``device_id``,
        ``lease_id``, ``acked`` (always ``true``) and ``message_count``
        (the number of messages the lease claimed) in that order.
        """
        try:
            return self.store.inbox_lease_ack(device_id, lease_id)
        except InboxLeaseError as error:
            if error.reason == INBOX_LEASE_NOT_FOUND:
                raise ServiceError(f"lease not found: {lease_id}",
                                   "lease_id", status_code=404)
            if error.reason == INBOX_LEASE_CONFLICT:
                raise ServiceError(
                    "lease_id is owned by another device",
                    "lease_id", status_code=409)
            if error.reason == INBOX_LEASE_NOT_DELIVERED:
                raise ServiceError(
                    "lease can only be acknowledged after a 'delivered' "
                    "completion", "lease_id", status_code=409)
            if error.reason == INBOX_LEASE_DEVICE_UNKNOWN:
                raise ServiceError("device_id is not a registered device",
                                   "device_id", status_code=409)
            raise ServiceError("device_id is revoked",
                               "device_id", status_code=409)

    def inbox_lease_get(self, device_id: str,
                        lease_id: str) -> Dict[str, Any]:
        """Return one occupied 1:1-inbox lease's current state (read-only).

        ``GET /v1/devices/{device_id}/inbox/leases/{lease_id}``. The GET
        takes no request body and no query parameters: the HTTP layer
        rejects a non-empty body with 400/field ``request_body`` and any
        query string with 400/field ``query``. A never-committed
        ``lease_id`` is 404/field ``lease_id`` and a lease owned by another
        device is 409/field ``lease_id``, both decided in the store under
        the lock ahead of everything else; a matching lease is returned
        even if its device has since been revoked, and the lookup is
        purely read-only (no write, no ``commit_seq`` change). On success
        the body keys are ``device_id``, ``lease_id``, ``limit``,
        ``state``, ``leased_until``, ``released_at``, ``completion`` and
        ``messages`` in that order; ``state`` is one of ``active``,
        ``expired``, ``released`` or ``completed``.
        """
        try:
            return self.store.inbox_lease_get(device_id, lease_id)
        except InboxLeaseError as error:
            if error.reason == INBOX_LEASE_NOT_FOUND:
                raise ServiceError(f"lease not found: {lease_id}",
                                   "lease_id", status_code=404)
            if error.reason == INBOX_LEASE_CONFLICT:
                raise ServiceError(
                    "lease_id is owned by another device",
                    "lease_id", status_code=409)
            raise

    def inbox_leases(self, device_id: str, state: str, after: int,
                     limit: int) -> Dict[str, Any]:
        """Return one page of a device's 1:1-inbox lease history (read-only).

        ``GET /v1/devices/{device_id}/inbox/leases``. The GET takes no
        request body (the HTTP layer rejects a non-empty one with
        400/field ``request_body``) and only the single-valued query
        parameters ``state``, ``after`` and ``limit`` (any other parameter
        is 400/field ``query``). ``state`` defaults to ``all`` and must be
        one of ``all``/``active``/``expired``/``released``/``completed``;
        ``after`` defaults to 0 and must be an unsigned decimal integer;
        ``limit`` defaults to 100 and must be in 1..100 (the HTTP layer
        enforces the decimal shape and single occurrence, answering
        400 with the parameter name as ``field``).

        Under the store lock the per-message lease copies are deduplicated
        by ``lease_id``, ordered by the initial claim ``leased_until`` and
        then the ``lease_id`` code points, filtered by current state and
        paged (skip *after*, take *limit*). An unknown device is
        404/field ``device_id``; a revoked device's history stays
        readable. On success the body keys are ``device_id``, ``leases``,
        ``next_after`` and ``has_more`` in that order; each lease item is
        ``lease_id``, ``state``, ``leased_until`` and ``message_count``
        with the state and effective deadline identical to the single-lease
        query. The lookup writes nothing and advances no ``commit_seq``.
        """
        if state not in ("all", "active", "expired", "released",
                         "completed"):
            raise ServiceError(
                "field must be one of 'all', 'active', 'expired', "
                "'released' or 'completed': state", "state")
        if not isinstance(after, int) or isinstance(after, bool) or after < 0:
            raise ServiceError(
                "field must be a non-negative integer: after", "after")
        if not isinstance(limit, int) or isinstance(limit, bool) \
                or not 1 <= limit <= 100:
            raise ServiceError(
                "field must be an integer in 1..100: limit", "limit")
        page = self.store.inbox_leases_page(device_id, state, after, limit)
        if page is None:
            raise ServiceError(f"device not found: {device_id}",
                               "device_id", status_code=404)
        return page

    def inbox_job(self, payload: object) -> Tuple[Dict[str, Any], int]:
        """Validate and apply one 1:1-inbox redelivery job operation.

        ``POST /v1/inbox-jobs``. The body must be a JSON object carrying
        non-empty strings ``device_id``, ``job_id`` and ``op``, with ``op``
        one of ``queue``, ``dispatch``, ``status`` or ``recover``. A
        bad/non-object body is 400/field ``request_body``; a missing,
        empty or wrongly typed field is 400 with the corresponding ``field``
        (``device_id`` / ``job_id`` / ``op``), as is an ``op`` outside the
        four verbs. A ``recover`` additionally requires a non-empty string
        ``recovery_id``; it missing, empty or wrongly typed is
        400/field ``recovery_id``.

        The device must be registered and not revoked, else 409/field
        ``device_id`` (checked in the store under the lock, ahead of the
        job lookup, so it is linearized against revocation). ``queue``
        creates the job ``pending`` (201); replaying the same ``job_id``
        on the same device returns its current view (200); the same
        ``job_id`` on another device is 409/field ``job_id``.
        ``dispatch``/``status`` on a never-queued id are 404/field
        ``job_id``; ``status`` is read-only (200) and ``dispatch`` on a
        non-pending job is a replay (200). A first ``dispatch`` on a
        pending job leases up to 100 unacked, currently unleased inbox
        messages under a ``lease_id`` equal to the ``job_id``: a non-empty
        selection moves the job to ``running``, an empty one to
        ``succeeded`` — both 201. Completing that lease ``delivered`` /
        ``failed`` moves the job to ``succeeded`` / ``failed`` in the same
        locked transaction.

        ``recover`` re-establishes the lease of a ``running`` job whose
        current lease has expired or been released: an exact replay of the
        same ``recovery_id`` on the same job returns the current view
        (200) without writing; the id reused on another job or already
        occupied as an inbox lease id is 409/field ``recovery_id``; an
        unknown/cross-device job is 404/409/field ``job_id``; a
        still-valid lease is 409/field ``lease_id`` and any other job
        state is 409/field ``job_id``. A first recovery leases up to 100
        currently claimable messages under an ordinary lease named by the
        ``recovery_id`` (job stays ``running`` with that new
        ``lease_id``), or ends the job ``succeeded`` with ``lease_id``
        null when nothing remains — both 201. The response keys are
        ``job_id``, ``device_id``, ``state``, ``lease_id`` in that order.
        """
        if not isinstance(payload, dict):
            raise ServiceError("request body must be a JSON object",
                               "request_body")
        for name in ("device_id", "job_id", "op"):
            if name not in payload:
                raise ServiceError(f"missing required field: {name}", name)
            if not is_nonempty_string(payload[name]):
                raise ServiceError(
                    f"field must be a non-empty string: {name}", name)
        op = payload["op"]
        if op not in ("queue", "dispatch", "status", "recover"):
            raise ServiceError(
                "field must be one of 'queue', 'dispatch', 'status' or "
                "'recover': op",
                "op")
        recovery_id = None
        if op == "recover":
            if "recovery_id" not in payload:
                raise ServiceError(
                    "missing required field: recovery_id", "recovery_id")
            if not is_nonempty_string(payload["recovery_id"]):
                raise ServiceError(
                    "field must be a non-empty string: recovery_id",
                    "recovery_id")
            recovery_id = payload["recovery_id"]
        try:
            if op == "recover":
                return self.store.redelivery_job_recover(
                    payload["device_id"], payload["job_id"], recovery_id)
            return self.store.redelivery_job_submit(
                payload["device_id"], payload["job_id"], op)
        except RedeliveryJobError as error:
            if error.reason == REDELIVERY_JOB_NOT_FOUND:
                raise ServiceError(
                    f"job not found: {payload['job_id']}",
                    "job_id", status_code=404)
            if error.reason == REDELIVERY_JOB_RECOVERY_CONFLICT:
                raise ServiceError(
                    "recovery_id is already used by another job or occupied "
                    "as a lease id", "recovery_id", status_code=409)
            if error.reason == REDELIVERY_JOB_RECOVERY_LEASE_ACTIVE:
                raise ServiceError(
                    "the job's lease is still valid and cannot be recovered",
                    "lease_id", status_code=409)
            if error.reason == REDELIVERY_JOB_RECOVERY_STATE:
                raise ServiceError(
                    "only a running job with an expired or released lease "
                    "can be recovered", "job_id", status_code=409)
            if error.reason in (REDELIVERY_JOB_CONFLICT,
                                REDELIVERY_JOB_LEASE_OCCUPIED):
                raise ServiceError(
                    "job_id is already used by another device or its lease "
                    "is occupied", "job_id", status_code=409)
            if error.reason == REDELIVERY_JOB_DEVICE_UNKNOWN:
                raise ServiceError("device_id is not a registered device",
                                   "device_id", status_code=409)
            raise ServiceError("device_id is revoked",
                               "device_id", status_code=409)

    def inbox_job_recover_batch(
            self, payload: object) -> Tuple[Dict[str, Any], int]:
        """Validate and atomically apply a batch of redelivery-job recoveries.

        ``POST /v1/inbox-jobs/recover-batch``. The body must be a JSON
        object carrying a non-empty string ``device_id`` and a non-empty
        ``items`` array; each item is an object with a non-empty string
        ``job_id`` and ``recovery_id``, and neither field may repeat across
        items. Shape errors are reported, in order, as 400/field
        ``request_body`` (bad/non-object body), ``device_id``
        (missing/empty/non-string), ``items`` (missing/not-a-non-empty
        array), ``items[i]`` (non-object element or a repeated
        ``job_id``/``recovery_id``) or ``items[i].job_id`` /
        ``items[i].recovery_id`` for the offending field.

        The device is then resolved (unknown/revoked -> 409/field
        ``device_id``) and every item is prechecked in array order with the
        single-job ``op=recover`` rules, the first error aborting the whole
        batch with nothing written and its ``field`` prefixed to
        ``items[i].``: an unknown job is 404/``items[i].job_id``, a job of
        another device 409/``items[i].job_id``, a ``recovery_id`` already
        committed on another job, equal to any job id or occupied as an
        inbox lease id 409/``items[i].recovery_id``, a non-``running`` job
        409/``items[i].job_id`` and a still-valid lease
        409/``items[i].lease_id``. An item replaying its own committed
        recovery skips those checks; when every item is such a replay the
        batch answers 200 with the current views and writes nothing, and a
        mix of replays and first-time items conflicts 409 with the first
        replayed item's ``items[i].recovery_id``.

        A first-time batch recovers every job in input order (each with the
        usual 100-message limit and 30-second lease) and commits once
        (201). The body keys are ``device_id`` then ``results``; results
        keep input order and each item is ``job_id``, ``state``,
        ``lease_id`` in that order.
        """
        if not isinstance(payload, dict):
            raise ServiceError("request body must be a JSON object",
                               "request_body")
        if "device_id" not in payload:
            raise ServiceError("missing required field: device_id",
                               "device_id")
        if not is_nonempty_string(payload["device_id"]):
            raise ServiceError(
                "field must be a non-empty string: device_id", "device_id")
        device_id = payload["device_id"]
        if "items" not in payload:
            raise ServiceError("missing required field: items", "items")
        raw_items = payload["items"]
        if not isinstance(raw_items, list) or not raw_items:
            raise ServiceError(
                "field must be a non-empty array: items", "items")

        items: List[Tuple[str, str]] = []
        seen_job_ids: set = set()
        seen_recovery_ids: set = set()
        for index, element in enumerate(raw_items):
            item_field = f"items[{index}]"
            if not isinstance(element, dict):
                raise ServiceError(
                    f"array element must be an object: {item_field}",
                    item_field)
            job_field = f"{item_field}.job_id"
            if "job_id" not in element:
                raise ServiceError(
                    f"missing required field: {job_field}", job_field)
            if not is_nonempty_string(element["job_id"]):
                raise ServiceError(
                    f"field must be a non-empty string: {job_field}",
                    job_field)
            recovery_field = f"{item_field}.recovery_id"
            if "recovery_id" not in element:
                raise ServiceError(
                    f"missing required field: {recovery_field}",
                    recovery_field)
            if not is_nonempty_string(element["recovery_id"]):
                raise ServiceError(
                    f"field must be a non-empty string: {recovery_field}",
                    recovery_field)
            if element["job_id"] in seen_job_ids:
                raise ServiceError(
                    "duplicate job_id in items: "
                    f"{element['job_id']}", item_field)
            if element["recovery_id"] in seen_recovery_ids:
                raise ServiceError(
                    "duplicate recovery_id in items: "
                    f"{element['recovery_id']}", item_field)
            seen_job_ids.add(element["job_id"])
            seen_recovery_ids.add(element["recovery_id"])
            items.append((element["job_id"], element["recovery_id"]))

        try:
            results, status_code = self.store.redelivery_job_recover_batch(
                device_id, items)
        except RedeliveryJobError as error:
            # Batch-level device failure (unknown/revoked): 409/device_id.
            if error.reason == REDELIVERY_JOB_DEVICE_UNKNOWN:
                raise ServiceError("device_id is not a registered device",
                                   "device_id", status_code=409)
            raise ServiceError("device_id is revoked",
                               "device_id", status_code=409)
        except RedeliveryJobRecoverBatchError as error:
            job_id, recovery_id = items[error.index]
            item_field = f"items[{error.index}]"
            if error.reason == REDELIVERY_JOB_NOT_FOUND:
                raise ServiceError(f"job not found: {job_id}",
                                   f"{item_field}.job_id", status_code=404)
            if error.reason == REDELIVERY_JOB_CONFLICT:
                raise ServiceError(
                    "job_id is already used by another device",
                    f"{item_field}.job_id", status_code=409)
            if error.reason == REDELIVERY_JOB_RECOVERY_CONFLICT:
                raise ServiceError(
                    "recovery_id is already used by another job or occupied "
                    "as a lease id", f"{item_field}.recovery_id",
                    status_code=409)
            if error.reason == REDELIVERY_JOB_RECOVERY_LEASE_ACTIVE:
                raise ServiceError(
                    "the job's lease is still valid and cannot be recovered",
                    f"{item_field}.lease_id", status_code=409)
            if error.reason == REDELIVERY_JOB_RECOVERY_STATE:
                raise ServiceError(
                    "only a running job with an expired or released lease "
                    "can be recovered", f"{item_field}.job_id",
                    status_code=409)
            raise ServiceError(
                "recovery_id replays an already committed recovery and "
                "cannot be mixed with first-time items",
                f"{item_field}.recovery_id", status_code=409)
        body = {"device_id": device_id, "results": results}
        return body, status_code

    # -- messages ----------------------------------------------------------

    def post_message(self, payload: object) -> Dict[str, Any]:
        """Validate a message-send payload and atomically append the message.

        ``sequence`` must continue the session's stream (starting at 1, no
        gaps or duplicates), ``message_id`` must be unique within the session,
        and ``nonce`` must not have been accepted before in the same session;
        each violation is a 409 naming the offending field, as is a
        revoked/unknown sender device. The identical nonce in a different
        session is allowed.
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
            raise self._message_create_error(error, payload)

        return self.store.message_view(message)

    @staticmethod
    def _message_create_error(error: MessageCreateError,
                              payload: Dict[str, Any]) -> ServiceError:
        """Translate a storage :class:`MessageCreateError` into a ServiceError."""
        status_code, field = _MESSAGE_CREATE_ERROR_MAP[error.reason]
        if error.reason == MESSAGE_SESSION_UNKNOWN:
            message_text = f"session not found: {payload['session_id']}"
        elif error.reason == MESSAGE_SENDER_INACTIVE:
            message_text = "sender_device_id is not an active device"
        elif error.reason == MESSAGE_DUPLICATE_ID:
            message_text = (f"message_id already exists in session: "
                            f"{payload['message_id']}")
        elif error.reason == MESSAGE_BAD_SEQUENCE:
            message_text = (f"sequence must continue the session stream "
                            f"(got {payload['sequence']})")
        elif error.reason == MESSAGE_REQUEST_ID_CONFLICT:
            message_text = (f"request_id was already used with different "
                            f"fields: {payload['request_id']}")
        else:
            message_text = (f"nonce already used in this session: "
                            f"{payload['nonce']}")
        return ServiceError(message_text, field, status_code=status_code)

    def submit_message(self, payload: object) -> Tuple[Dict[str, Any], int]:
        """Validate and atomically commit one idempotent message submission.

        ``request_id`` must be a non-empty string; the six envelope fields
        follow the same validation as :meth:`post_message`. A first-seen id
        commits the message and returns ``(body, 201)``; a replay with the
        identical envelope returns the original response with 200 (even if
        the sender was revoked since); the same id with any changed field —
        or reused across sessions — is a 409 naming ``request_id``. A failed
        submission never consumes its id. Returns ``(body, status_code)``.
        """
        if not isinstance(payload, dict):
            raise ServiceError("request body must be a JSON object",
                               "request_body")

        if "request_id" not in payload:
            raise ServiceError("missing required field: request_id",
                               "request_id")
        if not is_nonempty_string(payload["request_id"]):
            raise ServiceError(
                "field must be a non-empty string: request_id", "request_id")

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
            view, created = self.store.submit_message(
                payload["request_id"],
                payload["session_id"],
                payload["sender_device_id"],
                payload["message_id"],
                sequence,
                payload["nonce"],
                payload["ciphertext"])
        except MessageCreateError as error:
            raise self._message_create_error(error, payload)
        return view, 201 if created else 200

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

    # -- persistence integrity --------------------------------------------

    def persistence_integrity(self) -> Dict[str, Any]:
        """Read-only probe of the durable state file (no parameters/body).

        Only available when persistence is enabled (``--data-file`` or
        ``$E2EE_DATA_FILE``); the purely in-memory mode answers
        409/field=data_file. With a file, the document is read under the
        store lock and must parse as version=1, carry a non-negative
        ``commit_seq``, pass the full startup semantic validation, and match
        the in-memory snapshot exactly; success returns
        ``{"commit_seq", "state_hash", "consistent": true}`` (that key
        order). Any parse/version/semantic failure or generation/snapshot
        mismatch raises :class:`ServiceError` as 503/field=data_file without
        changing memory, the file/inode, cursors or the generation.
        """
        if self.integrity_state_store is None:
            raise ServiceError(
                "persistence is not enabled; start the server with a data "
                "file to inspect state integrity",
                "data_file", status_code=409)
        try:
            return self.integrity_state_store.integrity_report(self.store)
        except IntegrityCheckError as error:
            raise ServiceError(str(error), "data_file",
                               status_code=503) from None

    def persistence_integrity_history(self) -> Dict[str, Any]:
        """Read-only probe of the append-only integrity-history sidecar.

        Answers ``409/field=data_file`` whenever the state document carries
        no ``integrity_log_version`` marker — the purely in-memory mode and a
        file-mode store that has not made its first (migrating) commit. With
        the marker, the sidecar is read under the store lock and fully
        verified (structure, hash chain, per-entry hashes) and its last entry
        must equal the current committed generation and the live state hash;
        success returns ``{"commit_seq", "entries"}`` in that key order with
        ``commit_seq`` identical to the state generation and the last entry.
        Any parse/chain/hash/tail failure is 503/field=data_file and changes
        nothing.
        """
        if self.integrity_state_store is None:
            raise ServiceError(
                "persistence is not enabled; start the server with a data "
                "file to inspect state integrity history",
                "data_file", status_code=409)
        try:
            report = self.integrity_state_store.history_report(self.store)
        except IntegrityCheckError as error:
            raise ServiceError(str(error), "data_file",
                               status_code=503) from None
        if report is None:
            raise ServiceError(
                "integrity history is not enabled; the state file has not "
                "committed since the integrity log was introduced",
                "data_file", status_code=409)
        return report

    def persistence_integrity_history_page(self, after: int,
                                           limit: int) -> Dict[str, Any]:
        """Read-only ascending page of the integrity-history sidecar.

        ``after`` must be a non-negative integer (0 replays from the first
        entry) and ``limit`` an integer in 1..100 (the HTTP layer defaults
        them to 0 and 100). Availability and verification are identical to
        :meth:`persistence_integrity_history`: 409/field=data_file without
        the ``integrity_log_version`` marker (in-memory or not-yet-migrated
        file), and 503/field=data_file for any parse/version/generation/
        chain/hash/read failure. On success returns
        ``{"commit_seq", "entries", "next_after", "has_more"}`` in that key
        order — the verified entries with ``commit_seq > after`` (ascending,
        at most ``limit``), ``next_after`` equal to *after* on an empty page
        and otherwise the last entry's generation, and ``has_more`` saying
        whether a later entry exists. The probe changes nothing.
        """
        if not isinstance(after, int) or isinstance(after, bool) or after < 0:
            raise ServiceError(
                "field must be a non-negative integer: after", "after")
        if not isinstance(limit, int) or isinstance(limit, bool) \
                or not 1 <= limit <= 100:
            raise ServiceError(
                "field must be an integer in 1..100: limit", "limit")
        if self.integrity_state_store is None:
            raise ServiceError(
                "persistence is not enabled; start the server with a data "
                "file to inspect state integrity history",
                "data_file", status_code=409)
        try:
            report = self.integrity_state_store.history_page_report(
                self.store, after, limit)
        except IntegrityCheckError as error:
            raise ServiceError(str(error), "data_file",
                               status_code=503) from None
        if report is None:
            raise ServiceError(
                "integrity history is not enabled; the state file has not "
                "committed since the integrity log was introduced",
                "data_file", status_code=409)
        return report
