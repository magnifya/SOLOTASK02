"""Business logic and request validation for device registration/queries.

The service never sees plaintext messages or private keys: it validates and
stores identifiers and public-key material only.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .crypto import (
    CryptoError,
    decode_ed25519_signature,
    identity_fingerprint,
    is_fingerprint_format,
    is_nonempty_string,
    load_ed25519_public_key,
    load_public_key,
    parse_signed_message,
    verify_signed_prekey,
)
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
    CLEANUP_CHECKPOINT_AFTER_CONFLICT,
    CLEANUP_CHECKPOINT_EXPECTED_CONFLICT,
    CLEANUP_LEASE_CONSUMER_BUSY,
    CLEANUP_LEASE_CONSUMER_MISMATCH,
    CLEANUP_LEASE_ID_CONFLICT,
    CLEANUP_LEASE_NOT_FOUND,
    CLEANUP_LEASE_RENEW_CONFLICT,
    CLEANUP_LEASE_STATE_CONFLICT,
    DELIVERY_BAD_SEQUENCE,
    DELIVERY_DEVICE_INACTIVE,
    DELIVERY_DEVICE_MISMATCH,
    DELIVERY_MESSAGE_UNKNOWN,
    DELIVERY_SESSION_UNKNOWN,
    DEVICE_REVOKED,
    DEVICE_UNKNOWN,
    DEVICE_REVOCATION_NOT_ED25519,
    DEVICE_REVOCATION_SIGNATURE_INVALID,
    DEVICE_REVOCATION_VERSION_MISMATCH,
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
    IDENTITY_ROTATION_NOT_ED25519,
    IDENTITY_ROTATION_SIGNATURE_INVALID,
    IDENTITY_ROTATION_VERSION_MISMATCH,
    LEASE_EVENT_CURSOR_EXPECTED_CONFLICT,
    LEASE_SUBSCRIPTION_AFTER_CONFLICT,
    LEASE_SUBSCRIPTION_EXPECTED_CONFLICT,
    LEASE_SUBSCRIPTION_FILTER_CONFLICT,
    LEASE_SUBSCRIPTION_NOT_FOUND,
    MESSAGE_BAD_SEQUENCE,
    MESSAGE_DEVICE_INACTIVE,
    MESSAGE_DUPLICATE_ID,
    MESSAGE_DUPLICATE_NONCE,
    MESSAGE_IDENTITY_KEY_CONFLICT,
    MESSAGE_MESSAGE_UNKNOWN,
    MESSAGE_PROOF_ABSENT,
    MESSAGE_REQUEST_ID_CONFLICT,
    MESSAGE_SENDER_INACTIVE,
    MESSAGE_SESSION_UNKNOWN,
    MESSAGE_SIGNATURE_INVALID,
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
    INBOX_LEASE_COMPLETE_PARTIAL_REPLAY,
    INBOX_LEASE_ACK_CONFLICT,
    INBOX_LEASE_ACK_PARTIAL_REPLAY,
    INBOX_LEASE_RENEW_PARTIAL_REPLAY,
    INBOX_LEASE_RELEASE_PARTIAL_REPLAY,
    INBOX_LEASE_CONFLICT,
    INBOX_LEASE_DEVICE_INACTIVE,
    INBOX_LEASE_DEVICE_UNKNOWN,
    INBOX_LEASE_NOT_DELIVERED,
    INBOX_LEASE_NOT_FOUND,
    INBOX_LEASE_UNAVAILABLE,
    PREKEY_CONFLICT,
    PREKEY_IDENTITY_NOT_ED25519,
    PREKEY_SIGNATURE_INVALID,
    PROOF_DEVICE_UNKNOWN,
    PROOF_PREKEY_UNKNOWN,
    REDELIVERY_JOB_CONFLICT,
    REDELIVERY_JOB_DEVICE_INACTIVE,
    REDELIVERY_JOB_DEVICE_UNKNOWN,
    REDELIVERY_JOB_LEASE_OCCUPIED,
    REDELIVERY_JOB_NOT_FOUND,
    REDELIVERY_JOB_CANCELLATION_CONFLICT,
    REDELIVERY_JOB_CANCEL_PARTIAL_REPLAY,
    REDELIVERY_JOB_CANCEL_STATE,
    REDELIVERY_JOB_DISPATCH_PARTIAL_REPLAY,
    REDELIVERY_JOB_EVENT_AFTER_CONFLICT,
    REDELIVERY_JOB_EVENT_CHECKPOINT_CONSUMER_INVALID,
    REDELIVERY_JOB_EVENT_CHECKPOINT_SEQ_CONFLICT,
    REDELIVERY_JOB_EVENT_GC_BATCH_REQUEST_ID_CONFLICT,
    REDELIVERY_JOB_EVENT_GC_CONSUMER_UNKNOWN,
    REDELIVERY_JOB_EVENT_GC_NO_VALID_CONSUMER,
    REDELIVERY_JOB_EVENT_GC_SEQ_CONFLICT,
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
    SESSION_ROTATED,
    SESSION_ROTATION_ACTOR_NOT_INITIATOR,
    SESSION_ROTATION_ACTOR_REVOKED,
    SESSION_ROTATION_ACTOR_UNKNOWN,
    SESSION_ROTATION_ID_CONFLICT,
    SESSION_ROTATION_PREDECESSOR_ROTATED,
    SESSION_ROTATION_PREKEY_REVOKED,
    SESSION_ROTATION_PREKEY_UNKNOWN,
    SESSION_ROTATION_RECIPIENT_REVOKED,
    SESSION_ROTATION_SESSION_UNKNOWN,
    SYNC_BAD_SEQUENCE,
    SYNC_CURSOR_CONFLICT,
    SYNC_DEVICE_INACTIVE,
    SYNC_DEVICE_NOT_MEMBER,
    SYNC_DEVICE_UNKNOWN,
    SYNC_MESSAGE_UNKNOWN,
    SYNC_SESSION_UNKNOWN,
    SYNC_SELF_SENDER,
    DeviceStore,
    DeviceUpdateError,
    IdentityVerificationError,
    BatchClaimSessionError,
    ClaimSessionError,
    DeliveryError,
    GroupError,
    GroupSessionRotationError,
    GroupSyncError,
    GroupSyncAckBatchError,
    MessageCreateError,
    MessageListError,
    InboxRetryBatchError,
    InboxLeaseError,
    InboxLeaseCompleteBatchError,
    InboxLeaseAckBatchError,
    InboxLeaseRenewBatchError,
    InboxLeaseReleaseBatchError,
    InboxLeaseStatusBatchError,
    MessageSyncAckBatchError,
    MessageSyncError,
    PreKeyClaimError,
    PreKeyBatchClaimError,
    RedeliveryJobError,
    RedeliveryJobCancelBatchError,
    RedeliveryJobDispatchBatchError,
    RedeliveryJobRecoverBatchError,
    RedeliveryJobStatusBatchError,
    SessionCreateError,
    SessionRotationError,
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
    SESSION_ROTATED: (409, "session_id"),
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
        return self._publish_device(device)

    def register_verified(self, payload: object) -> Dict[str, Any]:
        """Validate a verified registration with signed pre-key proofs.

        Same shape as :meth:`register` plus a non-empty ``signature`` per
        signed pre-key. The ``identity_key`` must be an Ed25519 public key
        and every signature must be a standard-base64 64-byte Ed25519
        signature over the domain-separated canonical proof of that entry's
        ``device_id``/``key_id``/``public_key``/``user_id``. Every check
        passes before any state is written: a malformed field, a
        non-Ed25519/illegal identity key, an illegal public key, a duplicate
        ``key_id``, bad base64 or signature length, or a proof that does not
        verify is a 400 naming the exact field path; nothing is registered
        and no pre-key is consumed. A duplicate ``(user_id, device_id)`` is
        the same 409/field=device_id as ordinary registration.
        """
        if not isinstance(payload, dict):
            raise ServiceError("request body must be a JSON object", "request_body")

        for name in _REQUIRED_SCALAR_FIELDS:
            if name not in payload:
                raise ServiceError(f"missing required field: {name}", name)
            if not is_nonempty_string(payload[name]):
                raise ServiceError(f"field must be a non-empty string: {name}", name)

        identity = load_ed25519_public_key(payload["identity_key"])
        if identity is None:
            raise ServiceError(
                "field is not a valid Ed25519 public key: identity_key",
                "identity_key")

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
            for subfield in ("key_id", "public_key", "signature"):
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
            signature = decode_ed25519_signature(element["signature"])
            if signature is None:
                raise ServiceError(
                    f"field must be a standard base64 64-byte Ed25519 "
                    f"signature: {prefix}.signature",
                    f"{prefix}.signature")
            if not verify_signed_prekey(
                    identity, signature,
                    payload["user_id"], payload["device_id"],
                    element["key_id"], element["public_key"]):
                raise ServiceError(
                    f"signed pre-key proof failed verification: {prefix}.signature",
                    f"{prefix}.signature")
            seen_key_ids.add(element["key_id"])
            # Keep the published proof verbatim, frozen against the identity
            # public key presented with this registration; ordinary
            # publication (register()) stores no proof.
            prekeys.append(SignedPreKey(
                element["key_id"], element["public_key"],
                signature=element["signature"],
                proof_identity_key=payload["identity_key"]))

        device = Device(
            user_id=payload["user_id"],
            device_id=payload["device_id"],
            identity_key=payload["identity_key"],
            prekeys=prekeys,
        )
        return self._publish_device(device)

    def _publish_device(self, device: Device) -> Dict[str, Any]:
        """Insert a fully validated device and return the 201 body.

        Shared by ordinary and verified registration; a duplicate device id
        is a 409/field=device_id.
        """
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

    def get_prekey_proof(self, device_id: str,
                         key_id: str) -> Dict[str, str]:
        """Return the frozen published proof of one signed pre-key.

        The device is resolved before the key: an unknown device is
        404/field=device_id, a key it does not own is 404/field=key_id.
        Revoked devices/keys and consumed keys are still found — saved
        proofs are immutable history. A key without a saved proof
        (ordinary publication, legacy data, or any non-verified origin) is
        409/field=signature. The query is read-only: it consumes no key and
        changes no audit chain, cursor, delivery state or commit
        generation.
        """
        proof = self.store.prekey_proof(device_id, key_id)
        if proof == PROOF_DEVICE_UNKNOWN:
            raise ServiceError(f"device not found: {device_id}",
                               "device_id", status_code=404)
        if proof == PROOF_PREKEY_UNKNOWN:
            raise ServiceError(f"pre-key not found: {key_id}",
                               "key_id", status_code=404)
        if proof is None:
            raise ServiceError(
                "pre-key has no published signed proof: signature",
                "signature", status_code=409)
        return proof

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

    def revoke_device_verified(self, device_id: str,
                               payload: object) -> Dict[str, Any]:
        """Validate and apply a signature-authorized device revocation.

        The body is a JSON object with ``expected_version`` and
        ``signature``. ``expected_version`` is the device's current
        ``identity_key_version`` (as shown by the fingerprint query): a
        positive integer (booleans are refused) that must equal the
        current version at commit time. ``signature`` is standard base64
        decoding to exactly 64 bytes — an Ed25519 signature verified
        against the device's *current* identity key over the
        domain-separated canonical revocation message
        (``E2EE-DEVICE-REVOCATION-V1``) for the stored ``user_id`` and the
        request's path-decoded ``device_id`` / ``expected_version``.
        Extra fields are ignored.

        A body that is not an object is 400/field=request_body; a missing
        or non-positive/boolean/non-integer version is
        400/field=expected_version; a missing, empty, wrong-type,
        bad-encoding or wrong-length signature is 400/field=signature.
        The device/identity/version/signature checks then run atomically
        in the store: unknown device is 404/field=device_id, a current
        key that is not Ed25519 400/field=identity_key, a version
        mismatch 409/field=expected_version and a failed verification
        400/field=signature. Unlike the ordinary revoke entry, an already
        revoked device is not short-circuited: the authorization is still
        fully checked there, under the same lock as an identity rotation,
        so only an authorization still valid at commit time succeeds.

        Success returns the ordinary revocation body
        (``{"device_id", "revoked": true}``, 200): a first-time call
        revokes the device and all of its pre-keys and appends one
        ``device_revoked`` event; a valid replay against an already
        revoked device changes no state, audit chain or commit
        generation. The identity key, its version, timestamps, existing
        sessions, messages and other devices are never changed. On
        failure nothing is written.
        """
        if not isinstance(payload, dict):
            raise ServiceError("request body must be a JSON object",
                               "request_body")
        if "expected_version" not in payload:
            raise ServiceError(
                "missing required field: expected_version",
                "expected_version")
        expected_version = payload["expected_version"]
        if not isinstance(expected_version, int) \
                or isinstance(expected_version, bool):
            raise ServiceError(
                "field must be a positive integer: expected_version",
                "expected_version")
        if expected_version < 1:
            raise ServiceError(
                "field must be a positive integer: expected_version",
                "expected_version")
        if "signature" not in payload:
            raise ServiceError("missing required field: signature",
                               "signature")
        signature = decode_ed25519_signature(payload["signature"])
        if signature is None:
            raise ServiceError(
                "field must be a standard base64 64-byte Ed25519 "
                "signature: signature", "signature")

        try:
            device = self.store.revoke_device_verified(
                device_id, expected_version, signature)
        except DeviceUpdateError as error:
            raise self._revocation_verified_error(error, device_id)
        return {"device_id": device.device_id, "revoked": True}

    @staticmethod
    def _revocation_verified_error(error: DeviceUpdateError,
                                   device_id: str) -> ServiceError:
        """Translate a verified-revocation storage failure to a ServiceError."""
        if error.reason == DEVICE_UNKNOWN:
            return ServiceError(f"device not found: {device_id}",
                                "device_id", status_code=404)
        if error.reason == DEVICE_REVOCATION_NOT_ED25519:
            return ServiceError(
                "stored identity key is not an Ed25519 public key: "
                "identity_key", "identity_key")
        if error.reason == DEVICE_REVOCATION_VERSION_MISMATCH:
            return ServiceError(
                "expected_version does not match the device's current "
                "identity_key_version",
                "expected_version", status_code=409)
        if error.reason == DEVICE_REVOCATION_SIGNATURE_INVALID:
            return ServiceError(
                "device revocation authorization failed verification: "
                "signature", "signature")
        raise  # pragma: no cover - defensive

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

    def rotate_identity_key_verified(self, device_id: str,
                                     payload: object) -> Dict[str, Any]:
        """Validate and apply a signature-authorized identity-key rotation.

        The body is a JSON object with ``identity_key``, ``expected_version``
        and ``signature``. ``identity_key`` must be a non-empty string that
        parses as an Ed25519 public key (the usual encodings); the device's
        *current* key must also be Ed25519 so the authorization can be
        checked against it. ``expected_version`` is the
        ``identity_key_version`` shown by the fingerprint query: a positive
        integer (booleans are refused) that must equal the device's current
        version. ``signature`` is standard base64 decoding to exactly 64
        bytes — an Ed25519 signature verified against the device's current
        identity key over the domain-separated canonical rotation message
        (``E2EE-IDENTITY-ROTATION-V1``) for the stored ``user_id`` and the
        request's ``device_id`` / ``identity_key`` / ``expected_version``.

        A body that is not an object is 400/field=request_body; a missing
        field, empty string, wrong type, out-of-range version, bad signature
        encoding or wrong key algorithm is 400 naming the field. The
        device/identity/version/signature checks then run atomically in the
        store: unknown device is 404/field=device_id, revoked 409/field=
        device_id, a non-Ed25519 current key 400/field=identity_key, a
        version mismatch 409/field=expected_version and a failed
        verification 400/field=signature. Success returns the ordinary
        rotation view (200): rotating to the same raw key with a valid
        authorization is a no-op that changes no timestamp, version, audit
        event or commit generation; a different key refreshes the identity
        and timestamp, raises the version by one and appends the usual
        ``identity_rotated`` event. On failure nothing is written.
        """
        if not isinstance(payload, dict):
            raise ServiceError("request body must be a JSON object",
                               "request_body")
        for name in ("identity_key", "expected_version", "signature"):
            if name not in payload:
                raise ServiceError(f"missing required field: {name}", name)
        if not is_nonempty_string(payload["identity_key"]):
            raise ServiceError(
                "field must be a non-empty string: identity_key",
                "identity_key")
        if not is_nonempty_string(payload["signature"]):
            raise ServiceError(
                "field must be a non-empty string: signature", "signature")
        expected_version = payload["expected_version"]
        if not isinstance(expected_version, int) \
                or isinstance(expected_version, bool):
            raise ServiceError(
                "field must be a positive integer: expected_version",
                "expected_version")
        if expected_version < 1:
            raise ServiceError(
                "field must be a positive integer: expected_version",
                "expected_version")
        signature = decode_ed25519_signature(payload["signature"])
        if signature is None:
            raise ServiceError(
                "field must be a standard base64 64-byte Ed25519 "
                "signature: signature", "signature")
        if load_ed25519_public_key(payload["identity_key"]) is None:
            raise ServiceError(
                "field is not a valid Ed25519 public key: identity_key",
                "identity_key")

        try:
            view, _changed = self.store.rotate_identity_key_verified(
                device_id, payload["identity_key"], expected_version,
                signature)
        except DeviceUpdateError as error:
            raise self._rotation_verified_error(error, device_id)
        return view

    @staticmethod
    def _rotation_verified_error(error: DeviceUpdateError,
                                 device_id: str) -> ServiceError:
        """Translate a verified-rotation storage failure to a ServiceError."""
        if error.reason == DEVICE_UNKNOWN:
            return ServiceError(f"device not found: {device_id}",
                                "device_id", status_code=404)
        if error.reason == DEVICE_REVOKED:
            return ServiceError("device_id is revoked",
                                "device_id", status_code=409)
        if error.reason == IDENTITY_ROTATION_NOT_ED25519:
            return ServiceError(
                "stored identity key is not an Ed25519 public key: "
                "identity_key", "identity_key")
        if error.reason == IDENTITY_ROTATION_VERSION_MISMATCH:
            return ServiceError(
                "expected_version does not match the device's current "
                "identity_key_version",
                "expected_version", status_code=409)
        if error.reason == IDENTITY_ROTATION_SIGNATURE_INVALID:
            return ServiceError(
                "identity rotation proof failed verification: signature",
                "signature")
        raise  # pragma: no cover - defensive

    # -- identity fingerprint & explicit verification ---------------------

    def _identity_peer_and_verifier(
            self, device_id: str,
            verifier_device_id: str) -> Tuple[Device, Device]:
        """Resolve the two devices of a fingerprint/verification request.

        The peer (*device_id*) is checked first, then the verifier: unknown
        or revoked is a 404 naming ``device_id`` or ``verifier_device_id``
        respectively. The two ids must differ.
        """
        peer = self.store.find_by_device_id(device_id)
        if peer is None or peer.revoked:
            raise ServiceError(f"device not found: {device_id}",
                               "device_id", status_code=404)
        verifier = self.store.find_by_device_id(verifier_device_id)
        if verifier is None or verifier.revoked:
            raise ServiceError(
                f"verifier device not found: {verifier_device_id}",
                "verifier_device_id", status_code=404)
        return peer, verifier

    def identity_fingerprint_view(self, device_id: str,
                                  verifier_device_id: str) -> Dict[str, Any]:
        """Return the verifier's identity-fingerprint view of a peer.

        Answers ``GET /v1/devices/{device_id}/identity-fingerprint``: the
        peer's current ``identity_key``, its domain-separated 64-char
        lowercase hex ``fingerprint``, the current ``identity_key_version``
        (1-based; same-key rotations leave it fixed, different-key rotations
        raise it), the pair's ``verification_status`` (``unverified``,
        ``verified`` or ``changed``) and the active ``verification_id`` (null
        until a first verification).
        """
        peer, _verifier = self._identity_peer_and_verifier(
            device_id, verifier_device_id)
        fingerprint = identity_fingerprint(peer.identity_key)
        # A registered identity key always parses; keep an explicit guard so
        # a corrupt record can never leak a null fingerprint.
        if fingerprint is None:
            raise ServiceError(
                "stored identity key is not a valid public key: "
                "identity_key", "identity_key")
        return self.store.identity_fingerprint_view(
            verifier_device_id, device_id, fingerprint)

    def verify_identity(self, device_id: str,
                        payload: object) -> Tuple[Dict[str, Any], int]:
        """Explicitly confirm a peer's fingerprint.

        Handles ``POST .../identity-verifications``. The body carries three
        non-empty strings: ``verifier_device_id``,
        ``verification_id`` and ``expected_fingerprint``; the last must be 64
        lowercase hexadecimal characters. A confirmation that matches the
        peer's current fingerprint writes the first record (201); repeating
        the same id for the same verifier/peer/fingerprint while it is still
        active is an idempotent replay (200). Reusing an id for another pair,
        reusing a superseded id after the peer's key changed, or confirming a
        pair that already has an active (still-matching) record under another
        id is 409/field=verification_id; a well-formed fingerprint that does
        not match the peer's current key is 409/field=expected_fingerprint.
        Unknown or revoked devices are 404 (``device_id`` checked first, then
        ``verifier_device_id``). Returns ``(body, status_code)``.
        """
        if not isinstance(payload, dict):
            raise ServiceError("request body must be a JSON object",
                               "request_body")
        for name in ("verifier_device_id", "verification_id",
                     "expected_fingerprint"):
            if name not in payload:
                raise ServiceError(f"missing required field: {name}", name)
            if not is_nonempty_string(payload[name]):
                raise ServiceError(
                    f"field must be a non-empty string: {name}", name)
        verifier_device_id = payload["verifier_device_id"]
        verification_id = payload["verification_id"]
        expected_fingerprint = payload["expected_fingerprint"]
        if verifier_device_id == device_id:
            raise ServiceError(
                "verifier_device_id must differ from device_id",
                "verifier_device_id")
        if not is_fingerprint_format(expected_fingerprint):
            raise ServiceError(
                "field must be 64 lowercase hexadecimal characters: "
                "expected_fingerprint", "expected_fingerprint")

        peer, _verifier = self._identity_peer_and_verifier(
            device_id, verifier_device_id)
        current_fingerprint = identity_fingerprint(peer.identity_key)
        if current_fingerprint is None:
            raise ServiceError(
                "stored identity key is not a valid public key: "
                "identity_key", "identity_key")
        if expected_fingerprint != current_fingerprint:
            raise ServiceError(
                "expected_fingerprint does not match the device's current "
                "identity key",
                "expected_fingerprint", status_code=409)
        try:
            view, created = self.store.confirm_identity_verification(
                verifier_device_id, device_id, verification_id,
                current_fingerprint)
        except IdentityVerificationError:
            raise ServiceError(
                "verification_id conflicts with an existing verification",
                "verification_id", status_code=409) from None
        return view, 201 if created else 200

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

    def add_prekey_verified(self, device_id: str,
                            payload: object) -> Tuple[Dict[str, Any], int]:
        """Validate and append an identity-authorized (signed) pre-key.

        The body is a JSON object with three non-empty strings:
        ``key_id``, ``public_key`` and ``signature``. ``public_key`` uses the
        usual public-key encodings; ``signature`` is standard base64 decoding
        to exactly 64 bytes — an Ed25519 signature verified against the
        device's *current* identity key over the same domain-separated
        canonical proof (``E2EE-SIGNED-PREKEY-V1``) as verified registration,
        for the stored ``user_id`` and the request's ``device_id`` /
        ``key_id`` / ``public_key`` strings taken verbatim.

        A body that is not an object is 400/field=request_body; a missing or
        wrongly-typed field, an invalid public key, or an invalid signature
        encoding is 400 naming ``key_id``, ``public_key`` or ``signature``.
        The proof and the device/identity checks then run atomically in the
        store: unknown device is 404/field=device_id, a revoked device
        409/field=device_id, a current identity key that is not Ed25519
        400/field=identity_key, and a proof that does not verify
        400/field=signature. A verified new id appends and returns 201; the
        same id with the same, non-revoked key and a valid proof is
        idempotent (200); the same id with a changed key or a revoked id is
        409/field=key_id. On failure no device, pre-key, audit-chain, sync
        cursor or durable generation is changed. Returns ``(body,
        status_code)``.
        """
        if not isinstance(payload, dict):
            raise ServiceError("request body must be a JSON object",
                               "request_body")
        for name in ("key_id", "public_key", "signature"):
            if name not in payload:
                raise ServiceError(f"missing required field: {name}", name)
            if not is_nonempty_string(payload[name]):
                raise ServiceError(
                    f"field must be a non-empty string: {name}", name)
        if load_public_key(payload["public_key"]) is None:
            raise ServiceError(
                "field is not a valid public key: public_key", "public_key")
        signature = decode_ed25519_signature(payload["signature"])
        if signature is None:
            raise ServiceError(
                "field must be a standard base64 64-byte Ed25519 "
                "signature: signature", "signature")

        try:
            view, created = self.store.add_prekey_verified(
                device_id, payload["key_id"], payload["public_key"],
                signature, payload["signature"])
        except DeviceUpdateError as error:
            raise self._device_update_error(error, device_id)
        return view, 201 if created else 200

    def add_prekeys_verified_batch(self, device_id: str,
                                   payload: object) -> Tuple[Dict[str, Any], int]:
        """Validate and append a batch of identity-authorized pre-keys.

        The body is a JSON object whose ``signed_prekeys`` is a non-empty
        array; each element carries the same three non-empty strings as the
        single-item verified entry (``key_id``, ``public_key``,
        ``signature``), with the same public-key and signature encodings and
        the same ``E2EE-SIGNED-PREKEY-V1`` proof verified against the
        device's *current* identity key over the verbatim request strings.

        A body that is not an object is 400/field=request_body; a missing,
        non-array or empty ``signed_prekeys`` is 400/field=signed_prekeys; a
        non-object element is 400 naming ``signed_prekeys[i]`` (0-based); a
        missing/wrongly-typed/empty field, an invalid public key or an
        invalid signature encoding is 400 naming
        ``signed_prekeys[i].<field>``; a repeated ``key_id`` is 400 at its
        second occurrence, naming ``signed_prekeys[i].key_id``. The
        device/identity checks and per-element proof verification and
        conflict decisions then run atomically in the store, in array order
        with each element's proof checked before its conflict decision:
        unknown device is 404/field=device_id, a revoked device
        409/field=device_id, a current identity key that is not Ed25519
        400/field=identity_key, a proof that does not verify 400 naming
        ``signed_prekeys[i].signature``, and an existing id with a changed
        key or a revoked id 409 naming ``signed_prekeys[i].key_id``.

        The batch is all-or-nothing: when every element passes, each new id
        is appended in request order (one ``prekey_added`` event and one
        frozen proof each, a single durable commit generation for the whole
        batch) and the response is 201; an existing id with the same,
        non-revoked key is idempotent (no proof rewrite, no event, no
        consumed-flag reset), and a batch of only idempotent elements returns
        200 without consuming a commit generation. On any failure no device,
        pre-key, audit-chain, sync cursor or durable generation is changed.
        The response carries ``device_id`` and ``signed_prekeys`` with each
        element's ``key_id``/``public_key`` in request order. Returns
        ``(body, status_code)``.
        """
        if not isinstance(payload, dict):
            raise ServiceError("request body must be a JSON object",
                               "request_body")
        if "signed_prekeys" not in payload:
            raise ServiceError("missing required field: signed_prekeys",
                               "signed_prekeys")
        raw_prekeys = payload["signed_prekeys"]
        if not isinstance(raw_prekeys, list) or not raw_prekeys:
            raise ServiceError(
                "field must be a non-empty array: signed_prekeys",
                "signed_prekeys")

        items: List[Dict[str, Any]] = []
        seen_key_ids: set = set()
        for index, element in enumerate(raw_prekeys):
            prefix = f"signed_prekeys[{index}]"
            if not isinstance(element, dict):
                raise ServiceError(
                    f"array element must be an object: {prefix}", prefix)
            for subfield in ("key_id", "public_key", "signature"):
                path = f"{prefix}.{subfield}"
                if subfield not in element:
                    raise ServiceError(f"missing required field: {path}", path)
                if not is_nonempty_string(element[subfield]):
                    raise ServiceError(
                        f"field must be a non-empty string: {path}", path)
            if element["key_id"] in seen_key_ids:
                raise ServiceError(
                    f"duplicate key_id in signed_prekeys: "
                    f"{element['key_id']}", f"{prefix}.key_id")
            if load_public_key(element["public_key"]) is None:
                raise ServiceError(
                    f"field is not a valid public key: {prefix}.public_key",
                    f"{prefix}.public_key")
            signature = decode_ed25519_signature(element["signature"])
            if signature is None:
                raise ServiceError(
                    f"field must be a standard base64 64-byte Ed25519 "
                    f"signature: {prefix}.signature", f"{prefix}.signature")
            seen_key_ids.add(element["key_id"])
            items.append({"key_id": element["key_id"],
                          "public_key": element["public_key"],
                          "signature": signature,
                          "signature_text": element["signature"]})

        try:
            view, created = self.store.add_prekeys_verified_batch(
                device_id, items)
        except DeviceUpdateError as error:
            raise self._device_batch_update_error(error, device_id)
        return view, 201 if created else 200

    @staticmethod
    def _device_batch_update_error(error: DeviceUpdateError,
                                   device_id: str) -> ServiceError:
        """Translate a batch pre-key storage failure to a ServiceError.

        Element-level failures carry the 0-based request-array index and are
        reported under that element's field path; device-level failures map
        exactly as in the single-item entry.
        """
        prefix = (f"signed_prekeys[{error.index}]"
                  if error.index is not None else "")
        if error.reason == PREKEY_SIGNATURE_INVALID:
            return ServiceError(
                "signed pre-key proof failed verification: "
                f"{prefix}.signature", f"{prefix}.signature")
        if error.reason == PREKEY_CONFLICT:
            return ServiceError(
                "key_id already exists with a different key or is revoked",
                f"{prefix}.key_id", status_code=409)
        return DeviceService._device_update_error(error, device_id)

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
        if error.reason == PREKEY_IDENTITY_NOT_ED25519:
            return ServiceError(
                "field is not a valid Ed25519 public key: identity_key",
                "identity_key")
        if error.reason == PREKEY_SIGNATURE_INVALID:
            return ServiceError(
                "signed pre-key proof failed verification: signature",
                "signature")
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

    #: Maps a 1:1 session rotation failure to (HTTP status, field name).
    _SESSION_ROTATION_ERROR_MAP = {
        SESSION_ROTATION_SESSION_UNKNOWN: (404, "session_id"),
        SESSION_ROTATION_ACTOR_UNKNOWN: (404, "actor_device_id"),
        SESSION_ROTATION_ACTOR_REVOKED: (409, "actor_device_id"),
        SESSION_ROTATION_ACTOR_NOT_INITIATOR: (409, "actor_device_id"),
        SESSION_ROTATION_RECIPIENT_REVOKED: (409, "recipient_device_id"),
        SESSION_ROTATION_PREKEY_UNKNOWN: (404, "prekey_id"),
        SESSION_ROTATION_PREKEY_REVOKED: (409, "prekey_id"),
        SESSION_ROTATION_ID_CONFLICT: (409, "rotation_id"),
        SESSION_ROTATION_PREDECESSOR_ROTATED: (409, "session_id"),
    }

    def rotate_session(self, predecessor_session_id: str,
                       payload: object) -> Tuple[Dict[str, Any], int]:
        """Validate a 1:1 session rotation payload and atomically rotate it.

        ``rotation_id``, ``actor_device_id`` and ``prekey_id`` must be
        non-empty strings and ``ephemeral_key`` a non-empty valid public key.
        The actor must be the predecessor's un-revoked initiator and the
        recipient still active; the pre-key must be a valid (un-revoked,
        un-consumed) pre-key of the recipient. The first rotation returns
        201; an idempotent replay (same id on the same predecessor) returns
        200 with the original response. The successor keeps the two
        endpoints and starts its message stream at sequence 1.
        """
        if not isinstance(payload, dict):
            raise ServiceError("request body must be a JSON object",
                               "request_body")
        for name in ("rotation_id", "actor_device_id", "prekey_id"):
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

        try:
            successor, rotation, created = self.store.rotate_session(
                predecessor_session_id, payload["rotation_id"],
                payload["actor_device_id"], payload["prekey_id"],
                payload["ephemeral_key"])
        except SessionRotationError as error:
            status_code, field = \
                self._SESSION_ROTATION_ERROR_MAP[error.reason]
            if error.reason == SESSION_ROTATION_SESSION_UNKNOWN:
                message = f"session not found: {predecessor_session_id}"
            elif error.reason == SESSION_ROTATION_ACTOR_UNKNOWN:
                message = ("actor_device_id is not a registered device: "
                           f"{payload['actor_device_id']}")
            elif error.reason == SESSION_ROTATION_ACTOR_NOT_INITIATOR:
                message = ("actor_device_id is not the initiator of this "
                           "session")
            elif error.reason == SESSION_ROTATION_RECIPIENT_REVOKED:
                message = "recipient_device_id is revoked"
            elif error.reason == SESSION_ROTATION_PREKEY_UNKNOWN:
                message = (f"pre-key not found for the recipient: "
                           f"{payload['prekey_id']}")
            elif error.reason == SESSION_ROTATION_PREKEY_REVOKED:
                message = (f"prekey_id is revoked or already consumed: "
                           f"{payload['prekey_id']}")
            elif error.reason == SESSION_ROTATION_ID_CONFLICT:
                message = (
                    "rotation_id has already rotated another session: "
                    f"{payload['rotation_id']}")
            else:
                message = "session has already been rotated"
            raise ServiceError(message, field, status_code=status_code)
        return self.store.session_rotation_view(successor, rotation), \
            201 if created else 200

    def get_session_rotation(self, session_id: str) -> Dict[str, Any]:
        """Return the rotation view for a predecessor or successor session.

        Answers with the twelve-field rotation view (the successor's eight
        session fields plus the four rotation fields) when *session_id* is
        either end of a committed rotation; 404/field=session_id when the
        session is unknown or never took part in one.
        """
        record = self.store.get_session_rotation_record(session_id)
        if record is None:
            raise ServiceError(f"session not found: {session_id}",
                               "session_id", status_code=404)
        successor = self.store.session_view(record.successor_session_id)
        if successor is None:
            # A committed record always references a stored successor
            # (restore_state enforces it); defensive only.
            raise ServiceError(f"session not found: {session_id}",
                               "session_id", status_code=404)
        body = dict(successor)
        body["rotation_id"] = record.rotation_id
        body["predecessor_session_id"] = record.predecessor_session_id
        body["successor_session_id"] = record.successor_session_id
        body["predecessor_last_sequence"] = record.predecessor_last_sequence
        return body

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

    def sync_group_ack_batch(self, device_id: str,
                             payload: object) -> Tuple[Dict[str, Any], int]:
        """Validate and apply one device's multi-group-session batch ack.

        ``POST /v1/devices/{device_id}/group-sync/ack-batch``. The body must
        be an object carrying only a non-empty ``items`` array of objects,
        each with exactly a non-empty string ``session_id`` and a
        non-negative non-bool non-float integer ``cursor``, with no repeated
        session. Shape errors are reported, in order, as 400/field
        ``request_body`` (bad/non-object body), the offending key name (a
        top-level key other than ``items``), ``items``
        (missing/not-a-non-empty-array), ``items[i]`` (non-object element,
        an unexpected element key or a repeated session) or
        ``items[i].session_id`` / ``items[i].cursor`` for the offending
        field.

        The device is then resolved (unknown/revoked -> 409/field
        ``device_id``) and items are validated in array order, the first
        error aborting the whole batch with nothing written: a session that
        is not a group session is 409/``items[i].session_id`` (an unknown
        session id is 404); a group session the device was not frozen into
        is 409/``items[i].session_id``; a cursor below the stored group
        cursor (0 initially) or above the session's max sequence is
        409/``items[i].cursor``. On success the body is ``device_id`` then
        ``results``; results keep input order and each item is
        ``session_id``/``cursor``/``updated_at``. Status is 201 when at
        least one cursor advanced and 200 otherwise.
        """
        if not isinstance(payload, dict):
            raise ServiceError("request body must be a JSON object",
                               "request_body")
        extras = [key for key in payload if key != "items"]
        if extras:
            raise ServiceError(
                f"unexpected field: {extras[0]}", extras[0])
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
            element_extras = [key for key in element
                              if key not in ("session_id", "cursor")]
            if element_extras:
                raise ServiceError(
                    f"unexpected field: {item_field}.{element_extras[0]}",
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
            # bool is a subclass of int; reject it and floats explicitly.
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
            results, any_advanced = self.store.group_sync_ack_batch(
                device_id, items)
        except GroupSyncError as error:
            # Batch-level device failure (unknown/revoked): 409/device_id.
            raise self._sync_error(error, device_id)
        except GroupSyncAckBatchError as error:
            session_id = items[error.index][0]
            if error.reason == SYNC_SESSION_UNKNOWN:
                field = f"items[{error.index}].session_id"
                text = f"session not found: {session_id}"
                status_code = 404
            elif error.reason == SYNC_CURSOR_CONFLICT:
                field = f"items[{error.index}].cursor"
                text = "cursor is out of range or moved backwards"
                status_code = 409
            else:
                # A 1:1 session, or a group session the device was not
                # frozen into, is not group-ack-able here.
                field = f"items[{error.index}].session_id"
                text = ("session is not a group session or device is not a "
                        "frozen member")
                status_code = 409
            raise ServiceError(text, field, status_code=status_code)
        body = {"device_id": device_id, "results": results}
        return body, 201 if any_advanced else 200

    def sync_group_ack_messages(self, device_id: str,
                                payload: object) -> Tuple[Dict[str, Any], int]:
        """Validate and apply one device's selective group-message acks.

        ``POST /v1/devices/{device_id}/group-sync/ack-messages``. The body
        must be an object carrying only a non-empty ``items`` array of
        objects, each with exactly a non-empty string ``session_id``, a
        non-empty string ``message_id`` and a non-bool non-float
        non-negative integer ``sequence``, with no repeated
        ``(session_id, message_id)`` pair. Shape errors are reported, in
        order, as 400/field ``request_body`` (bad/non-object body), the
        offending key name (a top-level key other than ``items``),
        ``items`` (missing/not-a-non-empty-array), ``items[i]``
        (non-object element, an unexpected element key or a repeated
        session/message pair) or ``items[i].session_id`` /
        ``items[i].message_id`` / ``items[i].sequence`` for the offending
        field.

        The device is then resolved (unknown/revoked -> 409/field
        ``device_id``) and items are prechecked in array order, the first
        error aborting the whole batch with nothing written: an unknown
        session id is 404/``items[i].session_id``; a 1:1 session or a group
        session the device was not frozen into is
        409/``items[i].session_id`` (a removed member still acks, a
        later-added member cannot); an unknown message is
        404/``items[i].message_id``; a mismatched sequence is
        409/``items[i].sequence``; a message the device itself sent is
        409/``items[i].message_id``. On success the body is ``device_id``
        then ``results``; results keep input order and each item is
        ``session_id``/``message_id``/``acked`` with ``acked`` true. Status
        is 201 when at least one message is acknowledged for the first
        time and 200 when every item was already acked (nothing written).
        """
        if not isinstance(payload, dict):
            raise ServiceError("request body must be a JSON object",
                               "request_body")
        extras = [key for key in payload if key != "items"]
        if extras:
            raise ServiceError(
                f"unexpected field: {extras[0]}", extras[0])
        if "items" not in payload:
            raise ServiceError("missing required field: items", "items")
        raw_items = payload["items"]
        if not isinstance(raw_items, list) or not raw_items:
            raise ServiceError(
                "field must be a non-empty array: items", "items")

        items: List[Tuple[str, str, int]] = []
        seen_pairs: set = set()
        for index, element in enumerate(raw_items):
            item_field = f"items[{index}]"
            if not isinstance(element, dict):
                raise ServiceError(
                    f"array element must be an object: {item_field}",
                    item_field)
            element_extras = [key for key in element
                              if key not in ("session_id", "message_id",
                                             "sequence")]
            if element_extras:
                raise ServiceError(
                    f"unexpected field: {item_field}.{element_extras[0]}",
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
            sequence_field = f"{item_field}.sequence"
            if "sequence" not in element:
                raise ServiceError(
                    f"missing required field: {sequence_field}",
                    sequence_field)
            sequence = element["sequence"]
            # bool is a subclass of int; reject it and floats explicitly.
            if not isinstance(sequence, int) or isinstance(sequence, bool):
                raise ServiceError(
                    f"field must be an integer: {sequence_field}",
                    sequence_field)
            if sequence < 0:
                raise ServiceError(
                    f"field must be a non-negative integer: {sequence_field}",
                    sequence_field)
            pair = (element["session_id"], element["message_id"])
            if pair in seen_pairs:
                raise ServiceError(
                    "duplicate session_id/message_id pair in items: "
                    f"{pair[0]}/{pair[1]}", item_field)
            seen_pairs.add(pair)
            items.append((element["session_id"], element["message_id"],
                          sequence))

        try:
            results, any_first_ack = self.store.group_sync_ack_messages(
                device_id, items)
        except GroupSyncError as error:
            # Batch-level device failure (unknown/revoked): 409/device_id.
            raise self._sync_error(error, device_id)
        except GroupSyncAckBatchError as error:
            session_id, message_id, _ = items[error.index]
            if error.reason == SYNC_SESSION_UNKNOWN:
                field = f"items[{error.index}].session_id"
                text = f"session not found: {session_id}"
                status_code = 404
            elif error.reason == SYNC_MESSAGE_UNKNOWN:
                field = f"items[{error.index}].message_id"
                text = f"message not found: {message_id}"
                status_code = 404
            elif error.reason == SYNC_BAD_SEQUENCE:
                field = f"items[{error.index}].sequence"
                text = "sequence does not match the stored message"
                status_code = 409
            elif error.reason == SYNC_SELF_SENDER:
                field = f"items[{error.index}].message_id"
                text = "a device cannot acknowledge its own message"
                status_code = 409
            else:
                # A 1:1 session, or a group session the device was not
                # frozen into, is not group-ack-able here.
                field = f"items[{error.index}].session_id"
                text = ("session is not a group session or device is not a "
                        "frozen member")
                status_code = 409
            raise ServiceError(text, field, status_code=status_code)
        body = {"device_id": device_id, "results": results}
        return body, 201 if any_first_ack else 200

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

    def device_group_inbox(self, device_id: str, limit: int) -> Dict[str, Any]:
        """Return one device's aggregated group offline-inbox page (read-only).

        ``GET /v1/devices/{device_id}/group-inbox``. ``limit`` must be an
        integer in 1..100 (the HTTP layer defaults it to 100 and rejects
        repeats, non-decimals, out-of-range values and any other parameter
        with 400/field limit or query). An unknown or revoked device is
        409/field=device_id. On success the body is ``device_id``,
        ``messages``, ``has_more`` in that key order; ``messages`` holds at
        most ``limit`` seven-field envelopes (each already carrying
        ``session_id``) of the group messages the device has not yet
        acknowledged as a frozen member — 1:1 messages, the device's own
        messages and messages of sessions the device was not frozen into
        never contribute, while a member removed after the freeze still
        sees its frozen sessions — ordered by
        ``(session.created_at, session_id, sequence)``; ``has_more`` says
        whether the locked snapshot had further entries. The query is
        purely read-only: it writes nothing, advances no cursor or
        ``commit_seq`` and touches no delivery record, so an unchanged
        state answers byte-identically.
        """
        if not isinstance(limit, int) or isinstance(limit, bool) \
                or not 1 <= limit <= 100:
            raise ServiceError(
                "field must be an integer in 1..100: limit", "limit")
        try:
            return self.store.device_group_inbox(device_id, limit)
        except MessageSyncError as error:
            if error.reason == MESSAGE_SYNC_DEVICE_UNKNOWN:
                raise ServiceError("device_id is not a registered device",
                                   "device_id", status_code=409)
            raise ServiceError("device_id is revoked",
                               "device_id", status_code=409)

    def device_group_inbox_wait(self, device_id: str, limit: int,
                                timeout_ms: int) -> Dict[str, Any]:
        """Long-poll one device's aggregated group offline inbox (read-only).

        ``GET /v1/devices/{device_id}/group-inbox/wait``. ``limit`` must be a
        non-boolean integer in 1..100 (HTTP default 100) and
        ``timeout_ms`` a non-boolean integer in 0..30000 (HTTP default
        30000); the HTTP layer additionally rejects a request body, extra
        parameters, repeats and non-decimals with 400 on the matching field
        (request_body, query, limit or timeout_ms) before this is reached.
        An unknown or revoked device is 409/field=device_id. The success
        body is identical in shape and key order to
        :meth:`device_group_inbox`; an already-populated inbox returns
        immediately, and an empty one waits on the store condition without
        holding the store lock until a group message becomes deliverable,
        the device is revoked (checked first on wakeup) or the monotonic
        deadline passes, in which case the body carries ``messages=[]`` and
        ``has_more=false``. The wait writes nothing, advances no cursor or
        ``commit_seq`` and touches no delivery record.
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
            return self.store.device_group_inbox_wait(
                device_id, limit, timeout_ms)
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

    def group_inbox_retry_batch(self, device_id: str,
                                payload: object
                                ) -> Tuple[Dict[str, Any], int]:
        """Validate and apply one device's batch retry of group messages.

        ``POST /v1/devices/{device_id}/group-inbox/retry-batch``. The body
        must be an object carrying only a non-empty string ``attempt_id``
        and a non-empty ``items`` array of objects, each with only a
        non-empty string ``session_id`` and ``message_id``, with no
        repeated ``(session_id, message_id)`` pair. Shape errors are
        reported, in order, as 400/field ``request_body`` (bad/non-object
        body), the offending top-level key name (any key other than
        ``attempt_id``/``items``), ``attempt_id``
        (missing/empty/non-string), ``items``
        (missing/not-a-non-empty-array), ``items[i]`` (non-object element,
        an unexpected element key or a repeated pair) or
        ``items[i].session_id`` / ``items[i].message_id`` for the
        offending field.

        The device is then resolved (unknown/revoked -> 409/field
        ``device_id``) ahead of every item; items are prechecked in array
        order, the first error aborting the whole batch with nothing
        written: a session that is not a group session is
        404/``items[i].session_id``; a group session the device was not
        frozen into, or a message the device itself sent, is
        409/``items[i].session_id`` (a removed member still retries, a
        later-added member cannot); an unknown message is
        404/``items[i].message_id``; an already-acked message is
        409/``items[i].message_id``. On success the body is ``device_id``
        then ``results``; results keep input order and each item is
        ``session_id``/``message_id``/``attempts``. Status is 201 when at
        least one message newly recorded the attempt id and 200 (with no
        durable write and no ``commit_seq`` advance) when every id was a
        replay.
        """
        if not isinstance(payload, dict):
            raise ServiceError("request body must be a JSON object",
                               "request_body")
        extras = [key for key in payload
                  if key not in ("attempt_id", "items")]
        if extras:
            raise ServiceError(
                f"unexpected field: {extras[0]}", extras[0])
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
            element_extras = [key for key in element
                              if key not in ("session_id", "message_id")]
            if element_extras:
                raise ServiceError(
                    f"unexpected field: {item_field}.{element_extras[0]}",
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
            results, any_new = self.store.group_inbox_retry_batch(
                device_id, payload["attempt_id"], items)
        except MessageSyncError as error:
            # Batch-level device failure (unknown/revoked): 409/device_id.
            raise self._message_sync_error(error, device_id)
        except InboxRetryBatchError as error:
            session_id, message_id = items[error.index]
            item_field = f"items[{error.index}]"
            if error.reason == INBOX_RETRY_SESSION_UNKNOWN:
                raise ServiceError(
                    f"session is not a group session: {session_id}",
                    f"{item_field}.session_id", status_code=404)
            if error.reason == INBOX_RETRY_NOT_RECIPIENT:
                raise ServiceError(
                    "device is not a frozen member of the session or is "
                    "the message sender", f"{item_field}.session_id",
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

    def group_inbox_claim(self, device_id: str,
                          payload: object) -> Tuple[Dict[str, Any], int]:
        """Validate and apply one group-inbox redelivery lease claim.

        ``POST /v1/devices/{device_id}/group-inbox/claim``. The body must
        be a JSON object carrying only a non-empty string ``lease_id`` and
        a non-boolean integer ``limit`` in 1..100. A bad/non-object body is
        400/field ``request_body``; an unexpected top-level key is 400 with
        that key's name; a missing or malformed field is 400 with the
        corresponding ``field`` (``lease_id`` / ``limit``).

        The path device must be registered and not revoked, else
        409/field=device_id (checked in the store under the lock, so it is
        linearized against revocation). The ``lease_id`` is globally bound:
        replaying it for the same device and the same limit is idempotent
        (200 with the first, frozen response and no write); using it for
        another device, with another limit, or after it was committed in
        the 1:1-inbox namespace is 409/field ``lease_id``, ahead of the
        device state. A fresh claim leasing at least one message returns
        201 with the UTC ISO-8601 deadline (30 seconds out, six
        microsecond digits, ``+00:00``); an empty selection returns 200
        with ``leased_until`` null and writes nothing (the id stays free,
        no delivery record is created).
        """
        if not isinstance(payload, dict):
            raise ServiceError("request body must be a JSON object",
                               "request_body")
        extras = [key for key in payload
                  if key not in ("lease_id", "limit")]
        if extras:
            raise ServiceError(
                f"unexpected field: {extras[0]}", extras[0])
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
            body, status_code, _ = self.store.group_inbox_claim(
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

    def group_inbox_release(self, device_id: str,
                            lease_id: str) -> Tuple[Dict[str, Any], int]:
        """Release one occupied group-inbox lease before its deadline.

        ``POST /v1/devices/{device_id}/group-inbox/leases/{lease_id}
        /release`` (no request body and no query string; the HTTP layer
        rejects them with 400/field ``request_body`` / ``query``). A
        never-committed ``lease_id`` is 404/field ``lease_id``; one owned
        by another device — or committed in the 1:1 namespace, since the
        id is global — is 409/field ``lease_id``; both are decided in the
        store under the lock, ahead of the path device's state. A first
        release on an unknown/revoked device is 409/field ``device_id``.

        A first release returns 201 with ``device_id``, ``lease_id``,
        ``released_at`` (UTC ISO-8601, six microsecond digits, ``+00:00``)
        and ``released_count`` in that key order, and persists one
        generation; a repeat release returns 200 with the first response
        byte-identically, even if the device has since been revoked. The
        released lease stops withholding its messages from new group
        claims; replaying the original claim still returns its frozen
        first response and reactivates nothing.
        """
        try:
            return self.store.group_inbox_release(device_id, lease_id)
        except InboxLeaseError as error:
            if error.reason == INBOX_LEASE_NOT_FOUND:
                raise ServiceError(f"lease not found: {lease_id}",
                                   "lease_id", status_code=404)
            if error.reason == INBOX_LEASE_CONFLICT:
                raise ServiceError(
                    "lease_id is owned by another device",
                    "lease_id", status_code=409)
            if error.reason == INBOX_LEASE_DEVICE_UNKNOWN:
                raise ServiceError("device_id is not a registered device",
                                   "device_id", status_code=409)
            raise ServiceError("device_id is revoked",
                               "device_id", status_code=409)

    def group_inbox_lease_renew(self, device_id: str, lease_id: str,
                                payload: object) -> Tuple[Dict[str, Any],
                                                          int]:
        """Validate and apply one group-inbox lease renewal.

        ``POST /v1/devices/{device_id}/group-inbox/leases/{lease_id}
        /renew``. The body must be a JSON object carrying only a non-empty
        string ``renewal_id``: a bad/non-object body is 400/field
        ``request_body``; an unexpected top-level key is 400 with that
        key's name; a missing, empty or wrongly typed field is 400/field
        ``renewal_id``.

        The lease is resolved in the store under the lock, ahead of the
        path device's state: a never-committed ``lease_id`` is 404/field
        ``lease_id`` and one owned by another device — or committed in the
        1:1 namespace, since the id is global — is 409/field ``lease_id``.
        Replaying the same ``renewal_id`` on the same lease returns the
        frozen first response with 200 (ids may recur on other leases, in
        either inbox namespace). Only a first renewal checks the device
        (unknown or revoked -> 409/field ``device_id``) and the lease
        state (already released or expired -> 409/field ``lease_id``). A
        first renewal returns 201 with ``device_id``, ``lease_id``,
        ``renewal_id`` and the new ``leased_until`` (the previous
        effective deadline plus exactly 30 seconds, UTC ISO-8601 with six
        microsecond digits and ``+00:00``), and persists one generation.
        """
        if not isinstance(payload, dict):
            raise ServiceError("request body must be a JSON object",
                               "request_body")
        extras = [key for key in payload if key != "renewal_id"]
        if extras:
            raise ServiceError(
                f"unexpected field: {extras[0]}", extras[0])
        if "renewal_id" not in payload:
            raise ServiceError("missing required field: renewal_id",
                               "renewal_id")
        if not is_nonempty_string(payload["renewal_id"]):
            raise ServiceError(
                "field must be a non-empty string: renewal_id",
                "renewal_id")

        try:
            return self.store.group_inbox_lease_renew(
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

    def group_inbox_lease_get(self, device_id: str,
                              lease_id: str) -> Dict[str, Any]:
        """Return one occupied group-inbox lease's current state (read-only).

        ``GET /v1/devices/{device_id}/group-inbox/leases/{lease_id}``.
        The GET takes no request body and no query parameters: the HTTP
        layer rejects a non-empty body with 400/field ``request_body`` and
        any query string with 400/field ``query``. A never-committed
        ``lease_id`` is 404/field ``lease_id`` and one owned by another
        device — or committed in the 1:1 namespace, since the id is
        global — is 409/field ``lease_id``, both decided in the store
        under the lock ahead of the path device's state; a matching
        lease is returned even if its device has since been revoked, and
        the lookup is purely read-only (no write, no ``commit_seq``
        change). On success the body keys are ``device_id``,
        ``lease_id``, ``limit``, ``state``, ``leased_until``,
        ``released_at`` and ``messages`` in that order; ``state`` is one
        of ``active``, ``expired`` or ``released``.
        """
        try:
            return self.store.group_inbox_lease_get(device_id, lease_id)
        except InboxLeaseError as error:
            if error.reason == INBOX_LEASE_NOT_FOUND:
                raise ServiceError(f"lease not found: {lease_id}",
                                   "lease_id", status_code=404)
            if error.reason == INBOX_LEASE_CONFLICT:
                raise ServiceError(
                    "lease_id is owned by another device",
                    "lease_id", status_code=409)
            raise

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

    def inbox_jobs_page(self, device_id: str, state: str, after: int,
                        limit: int) -> Dict[str, Any]:
        """Return one page of a device's redelivery-job history (read-only).

        ``GET /v1/devices/{device_id}/inbox-jobs``. The GET takes no
        request body (the HTTP layer rejects a non-empty one with
        400/field ``request_body``) and only the single-valued query
        parameters ``state``, ``after`` and ``limit`` (any other parameter
        is 400/field ``query``). ``state`` defaults to ``all`` and must be
        one of ``all``/``pending``/``running``/``succeeded``/``failed``/
        ``cancelled``; ``after`` defaults to 0 and must be an unsigned
        decimal integer in 0..2**63-1; ``limit`` defaults to 100 and must
        be in 1..100 (the HTTP layer enforces the decimal shape and single
        occurrence, answering 400 with the parameter name as ``field``).

        Under the store lock the jobs are listed in the order their first
        successful ``queue`` committed, filtered by state and paged (skip
        *after*, take *limit*). An unknown device is 404/field
        ``device_id``; a revoked device's history stays readable. On
        success the body keys are ``device_id``, ``jobs``, ``next_after``
        and ``has_more`` in that order; each job item is ``job_id``,
        ``state``, ``lease_id``, ``cancellation_id`` and ``cancelled_at``
        with the latter three a string or ``null`` exactly as persisted.
        The lookup writes nothing and advances no ``commit_seq``.
        """
        if state not in ("all", "pending", "running", "succeeded",
                         "failed", "cancelled"):
            raise ServiceError(
                "field must be one of 'all', 'pending', 'running', "
                "'succeeded', 'failed' or 'cancelled': state", "state")
        if not isinstance(after, int) or isinstance(after, bool) \
                or not 0 <= after <= 2**63 - 1:
            raise ServiceError(
                "field must be an integer in 0..2^63-1: after", "after")
        if not isinstance(limit, int) or isinstance(limit, bool) \
                or not 1 <= limit <= 100:
            raise ServiceError(
                "field must be an integer in 1..100: limit", "limit")
        page = self.store.redelivery_jobs_page(device_id, state, after,
                                               limit)
        if page is None:
            raise ServiceError(f"device not found: {device_id}",
                               "device_id", status_code=404)
        return page

    def inbox_job_events_page(self, device_id: str, after: int,
                              limit: int) -> Dict[str, Any]:
        """Return one page of a device's redelivery-job events (read-only).

        ``GET /v1/devices/{device_id}/inbox-job-events``. The GET takes no
        request body (the HTTP layer rejects a non-empty one with
        400/field ``request_body``) and only the single-valued query
        parameters ``after`` and ``limit`` (any other parameter is
        400/field ``query``). ``after`` defaults to 0 and must be an
        unsigned decimal integer in 0..2**63-1; ``limit`` defaults to 100
        and must be in 1..100 (the HTTP layer enforces the decimal shape
        and single occurrence, answering 400 with the parameter name as
        ``field``).

        Under the store lock the device's lifecycle events are taken in
        chain order (``seq`` ascending, contiguous from the retention
        watermark + 1): the page is the events with ``seq > after``, at
        most *limit* of them. An unknown device is 404/field
        ``device_id``; a revoked device's chain stays readable. An
        ``after`` below the device's retention watermark is 409/field
        ``after`` (those events were pruned). On success the body keys
        are ``device_id``, ``events``, ``next_after`` and ``has_more`` in
        that order; each event item is
        ``seq``, ``job_id``, ``type`` and ``state`` in that order, with
        ``type`` the operation name (``queue``/``dispatch``/``recover``/
        ``cancel``/``complete``) and ``state`` the job's state right after
        that commit. An empty page echoes ``next_after=after``; otherwise
        it is the last returned event's ``seq``. The lookup writes nothing
        and advances no ``commit_seq``.
        """
        if not isinstance(after, int) or isinstance(after, bool) \
                or not 0 <= after <= 2**63 - 1:
            raise ServiceError(
                "field must be an integer in 0..2^63-1: after", "after")
        if not isinstance(limit, int) or isinstance(limit, bool) \
                or not 1 <= limit <= 100:
            raise ServiceError(
                "field must be an integer in 1..100: limit", "limit")
        try:
            page = self.store.redelivery_job_events_page(device_id, after,
                                                         limit)
        except RedeliveryJobError as error:
            if error.reason == REDELIVERY_JOB_EVENT_AFTER_CONFLICT:
                raise ServiceError(
                    "after is below the device's retention watermark",
                    "after", status_code=409)
            raise
        if page is None:
            raise ServiceError(f"device not found: {device_id}",
                               "device_id", status_code=404)
        return page

    def inbox_job_event_checkpoint(
            self, device_id: str,
            payload: object) -> Tuple[Dict[str, Any], int]:
        """Read or advance one consumer's job-event checkpoint.

        ``POST /v1/devices/{device_id}/inbox-job-events/checkpoint``. The
        body must be a JSON object carrying exactly ``consumer_id`` (a
        non-empty string) and ``seq`` (``null`` or a non-negative,
        non-boolean integer). A bad/non-object body is 400/field
        ``request_body``; a missing, empty or wrongly typed field is 400
        with the corresponding ``field`` (``consumer_id`` / ``seq``), as is
        the first extra key in payload order.

        An unknown device is 404/field ``device_id``; a revoked device
        stays checkpointable. A consumer whose checkpoint record is
        invalid — its retention lease was revoked or expired, or its
        stored ``seq`` fell below the device's retention watermark — is
        409/field ``consumer_id`` for reads and advances alike.
        ``seq=null`` is a read-only query (200): a
        pair that never checkpointed reports ``seq`` 0 and ``updated_at``
        null, otherwise the stored values. An integer ``seq`` greater than
        the device's last event ``seq`` or below this consumer's stored
        value is 409/field ``seq``; an equal ``seq`` is an idempotent
        no-op (200, the timestamp untouched); a strictly greater one
        advances the checkpoint and refreshes ``updated_at`` (201).
        Checkpoints are independent per ``(device_id, consumer_id)`` pair.
        On success the body keys are ``device_id``, ``consumer_id``,
        ``seq`` and ``updated_at`` in that order, with ``updated_at`` null
        or a UTC ISO-8601 timestamp (six microsecond digits, ``+00:00``).
        """
        if not isinstance(payload, dict):
            raise ServiceError("request body must be a JSON object",
                               "request_body")
        # The body carries exactly consumer_id and seq; any other key is
        # 400 with that field (the first extra key, in payload order).
        extras = [key for key in payload if key not in ("consumer_id", "seq")]
        if extras:
            raise ServiceError(
                f"unexpected field: {extras[0]}", extras[0])
        if "consumer_id" not in payload:
            raise ServiceError(
                "missing required field: consumer_id", "consumer_id")
        if not is_nonempty_string(payload["consumer_id"]):
            raise ServiceError(
                "field must be a non-empty string: consumer_id",
                "consumer_id")
        if "seq" not in payload:
            raise ServiceError("missing required field: seq", "seq")
        seq = payload["seq"]
        if seq is not None:
            # bool is a subclass of int; reject it explicitly.
            if not isinstance(seq, int) or isinstance(seq, bool):
                raise ServiceError(
                    "field must be an integer or null: seq", "seq")
            if seq < 0:
                raise ServiceError(
                    "field must be a non-negative integer or null: seq",
                    "seq")
        try:
            result = self.store.redelivery_job_event_checkpoint(
                device_id, payload["consumer_id"], seq)
        except RedeliveryJobError as error:
            if error.reason == REDELIVERY_JOB_EVENT_CHECKPOINT_SEQ_CONFLICT:
                raise ServiceError(
                    "seq exceeds the last event seq or moves backwards",
                    "seq", status_code=409)
            if error.reason == REDELIVERY_JOB_EVENT_CHECKPOINT_CONSUMER_INVALID:
                raise ServiceError(
                    "the consumer's checkpoint is not valid: its retention "
                    "lease was revoked or expired, or its seq fell below "
                    "the retention watermark",
                    "consumer_id", status_code=409)
            raise
        if result is None:
            raise ServiceError(f"device not found: {device_id}",
                               "device_id", status_code=404)
        view, advanced = result
        return view, 201 if advanced else 200

    def event_gc(self, device_id: str,
                 payload: object) -> Tuple[Dict[str, Any], int]:
        """Touch/revoke a consumer's event-retention lease or prune events.

        ``POST /v1/event-gc/{device_id}``. The body must be a JSON object
        carrying exactly ``consumer``, ``op`` and ``seq``; a
        bad/non-object body is 400/field ``request_body``, a missing,
        wrongly typed or extra field is 400 with that field (the first
        extra key, in payload order). ``op`` must be one of ``touch``,
        ``revoke`` or ``prune``. ``touch``/``revoke`` require a non-empty
        string ``consumer`` and a null ``seq``; ``prune`` requires a null
        ``consumer`` and a non-negative, non-boolean integer ``seq``.

        An unknown device is 404/field ``device_id``; a revoked device
        stays usable. ``touch`` registers the consumer at the current
        retention watermark (an existing consumer keeps its checkpoint)
        with a fresh 30-day lease and answers 201 with keys ``device_id``,
        ``consumer``, ``seq``, ``expires``. ``revoke`` drops the lease: an
        unknown consumer is 404/field ``consumer``, the first revoke is
        201, a replay 200 (same keys, ``expires`` null). ``prune`` deletes
        the device's events up to ``seq``: with no valid consumer lease it
        is 409/field ``consumer``, a ``seq`` outside [watermark, minimum
        valid checkpoint] is 409/field ``seq``, an equal ``seq`` is 200
        (``removed`` 0), a greater one 201 (keys ``device_id``, ``seq``,
        ``removed``). Timestamps are UTC ISO-8601 (six microsecond
        digits, ``+00:00``).
        """
        if not isinstance(payload, dict):
            raise ServiceError("request body must be a JSON object",
                               "request_body")
        # The body carries exactly consumer, op and seq; any other key is
        # 400 with that field (the first extra key, in payload order).
        extras = [key for key in payload
                  if key not in ("consumer", "op", "seq")]
        if extras:
            raise ServiceError(
                f"unexpected field: {extras[0]}", extras[0])
        for name in ("consumer", "op", "seq"):
            if name not in payload:
                raise ServiceError(f"missing required field: {name}", name)
        op = payload["op"]
        if not is_nonempty_string(op):
            raise ServiceError("field must be a non-empty string: op", "op")
        if op not in ("touch", "revoke", "prune"):
            raise ServiceError(
                "field must be one of 'touch', 'revoke' or 'prune': op",
                "op")
        consumer = payload["consumer"]
        seq = payload["seq"]
        if op in ("touch", "revoke"):
            if not is_nonempty_string(consumer):
                raise ServiceError(
                    "field must be a non-empty string: consumer",
                    "consumer")
            if seq is not None:
                raise ServiceError("field must be null: seq", "seq")
        else:
            if consumer is not None:
                raise ServiceError("field must be null: consumer",
                                   "consumer")
            # bool is a subclass of int; reject it explicitly.
            if not isinstance(seq, int) or isinstance(seq, bool) \
                    or seq < 0:
                raise ServiceError(
                    "field must be a non-negative integer: seq", "seq")
        try:
            result = self.store.redelivery_job_event_gc(
                device_id, op, consumer, seq)
        except RedeliveryJobError as error:
            if error.reason == REDELIVERY_JOB_EVENT_GC_CONSUMER_UNKNOWN:
                raise ServiceError(
                    f"consumer not found: {consumer}",
                    "consumer", status_code=404)
            if error.reason == REDELIVERY_JOB_EVENT_GC_NO_VALID_CONSUMER:
                raise ServiceError(
                    "no valid consumer lease withholds events from "
                    "pruning", "consumer", status_code=409)
            if error.reason == REDELIVERY_JOB_EVENT_GC_SEQ_CONFLICT:
                raise ServiceError(
                    "seq is outside [retention watermark, minimum valid "
                    "consumer checkpoint]", "seq", status_code=409)
            raise
        if result is None:
            raise ServiceError(f"device not found: {device_id}",
                               "device_id", status_code=404)
        return result

    def event_gc_observe(self, device_id: str, after: int,
                         limit: int) -> Dict[str, Any]:
        """Return one page of a device's event-retention records.

        ``GET /v1/event-gc/{device_id}``. The GET takes no request body
        (the HTTP layer rejects a non-empty one with 400/field
        ``request_body``) and only the single-valued query parameters
        ``after`` and ``limit`` (any other parameter is 400/field
        ``query``), using the same paging contract as the job-events
        page: ``after`` defaults to 0 and is an unsigned decimal integer
        in 0..2**63-1, ``limit`` defaults to 100 and is in 1..100 (the
        HTTP layer enforces the decimal shape and single occurrence,
        answering 400 with the parameter name as ``field``).

        Under the store lock the device's retention registrations —
        revoked and expired consumers included — are ordered by
        ``consumer_id`` Unicode code point; the page is the records at
        the zero-based offset *after*, at most *limit* of them. An
        unknown device is 404/field ``device_id``; a revoked device
        stays observable. On success the body keys are ``device_id``,
        ``watermark``, ``first_seq``, ``last_seq``, ``consumers``,
        ``next_after`` and ``has_more`` in that order; the two seqs are
        the bounds of the surviving event chain and are both null when
        it is empty. Each item is ``consumer_id``, ``seq``,
        ``updated_at``, ``active`` and ``expires`` in that order, with
        ``active`` true only for a lease that is neither revoked nor
        expired and whose ``seq`` is at least the watermark.
        ``next_after`` is *after* plus the page length (equal to *after*
        for an empty page) and ``has_more`` says whether further
        records follow. The lookup writes nothing and advances no
        ``commit_seq``.
        """
        if not isinstance(after, int) or isinstance(after, bool) \
                or not 0 <= after <= 2**63 - 1:
            raise ServiceError(
                "field must be an integer in 0..2^63-1: after", "after")
        if not isinstance(limit, int) or isinstance(limit, bool) \
                or not 1 <= limit <= 100:
            raise ServiceError(
                "field must be an integer in 1..100: limit", "limit")
        page = self.store.redelivery_job_event_gc_observe(
            device_id, after, limit)
        if page is None:
            raise ServiceError(f"device not found: {device_id}",
                               "device_id", status_code=404)
        return page

    def event_gc_cleanup_expired(
            self, device_id: str) -> Tuple[Dict[str, Any], int]:
        """Delete a device's expired event-retention registrations.

        ``POST /v1/event-gc/{device_id}/cleanup-expired``. The call takes
        no query parameters and no request body (the HTTP layer rejects
        either with 400/field ``query`` or ``request_body``). An unknown
        device is 404/field ``device_id``; a revoked device stays
        cleanable. Every registration whose ``expires`` is non-null and
        already due is deleted (a revoke record is kept — its
        ``expires`` is null), after which the consumer is unregistered.
        With nothing to delete the call writes nothing (200,
        ``removed`` 0); otherwise the deletions commit once and the
        answer is 201. On success the body keys are ``device_id`` and
        ``removed`` in that order.
        """
        result = self.store.redelivery_job_event_gc_cleanup_expired(
            device_id)
        if result is None:
            raise ServiceError(f"device not found: {device_id}",
                               "device_id", status_code=404)
        return result

    def event_gc_cleanup_expired_batch(
            self, payload: object) -> Tuple[Dict[str, Any], int]:
        """Preview or clean up expired retention registrations for a page.

        ``POST /v1/event-gc-batch/cleanup-expired``. The call takes no
        query parameters (the HTTP layer rejects any with 400/field
        ``query``) and the body must be a JSON object carrying ``mode``,
        ``device_ids``, ``after`` and ``limit``, plus an optional
        ``request_id`` for a ``commit``; a bad/non-object body is
        400/field ``request_body``. ``mode`` must be one of ``preview``
        or ``commit``; ``device_ids`` must be a non-empty array of unique
        non-empty strings; ``after`` must be a non-negative, non-boolean
        integer and ``limit`` a non-boolean integer in 1..100. A missing,
        wrongly typed or out-of-range field is 400 with that field name,
        an extra top-level key is 400 with that key (the first, in payload
        order), and an illegal (non-string/empty) or repeated element is
        400/field ``device_ids[i]``. A ``request_id`` is only accepted on
        ``commit`` and must be a non-empty string (400/field
        ``request_id`` otherwise, including on ``preview``).

        The input list is paged from the zero-based offset ``after`` for
        at most ``limit`` entries and the whole page is handled under the
        one store lock at one instant, so a revoked device stays
        cleanable. Each unknown device answers with the three counters
        null and ``error`` ``{"status": 404, "field": "device_id"}``
        (keys in that order); each known device answers with non-negative
        integer counters and ``error`` null. ``preview`` only counts
        registrations whose ``expires`` is non-null and due: ``removed``
        is always 0, the call writes nothing and answers 200. ``commit``
        deletes that set across the page in one transaction: with at
        least one deletion the answer is 201, otherwise 200; a data-file
        failure rolls the whole batch back and the HTTP layer answers
        503/field ``data_file``.

        A ``commit`` carrying ``request_id`` is idempotent across
        restarts: a replay with the same id and the same ``device_ids``
        (order included), ``after`` and ``limit`` returns the first
        status code and a byte-identical frozen response without writing
        or consuming a commit generation, even if the state has since
        changed; the same id with a different payload is 409/field
        ``request_id``; concurrent commits with the same id execute
        exactly once. The deletions, the frozen response and the
        idempotency record commit in the one locked transaction, so even
        a deletion-less idempotent commit persists exactly once
        (``commit_seq`` plus one). On success the body keys are ``mode``,
        ``results``, ``next_after`` and ``has_more`` in that order;
        results keep input order, each item ``device_id``,
        ``watermark``, ``expired``, ``removed`` and ``error`` in that
        order; ``next_after`` is ``after`` plus the page length and
        ``has_more`` says whether further input entries follow.
        """
        if not isinstance(payload, dict):
            raise ServiceError("request body must be a JSON object",
                               "request_body")
        # The body carries exactly mode, device_ids, after, limit and the
        # optional request_id; any other key is 400 with that field (the
        # first extra key, in payload order).
        allowed = ("mode", "device_ids", "after", "limit", "request_id")
        extras = [key for key in payload if key not in allowed]
        if extras:
            raise ServiceError(
                f"unexpected field: {extras[0]}", extras[0])
        for name in ("mode", "device_ids", "after", "limit"):
            if name not in payload:
                raise ServiceError(f"missing required field: {name}", name)
        mode = payload["mode"]
        if not is_nonempty_string(mode) or mode not in ("preview",
                                                        "commit"):
            raise ServiceError(
                "field must be one of 'preview' or 'commit': mode", "mode")
        request_id = payload.get("request_id")
        if "request_id" in payload:
            # request_id is a commit-only option: on preview it stays the
            # 400 it was as an extra key; on commit it must be a non-empty
            # string.
            if mode != "commit" or not is_nonempty_string(request_id):
                raise ServiceError(
                    "field must be a non-empty string: request_id",
                    "request_id")
        raw_device_ids = payload["device_ids"]
        if not isinstance(raw_device_ids, list) or not raw_device_ids:
            raise ServiceError(
                "field must be a non-empty array: device_ids", "device_ids")
        after = payload["after"]
        # bool is a subclass of int; reject it explicitly.
        if not isinstance(after, int) or isinstance(after, bool) or after < 0:
            raise ServiceError(
                "field must be a non-negative integer: after", "after")
        limit = payload["limit"]
        if (not isinstance(limit, int) or isinstance(limit, bool)
                or not 1 <= limit <= 100):
            raise ServiceError(
                "field must be an integer in 1..100: limit", "limit")
        device_ids: List[str] = []
        seen_device_ids: set = set()
        for index, element in enumerate(raw_device_ids):
            item_field = f"device_ids[{index}]"
            if not is_nonempty_string(element) or element in seen_device_ids:
                raise ServiceError(
                    "array elements must be unique non-empty strings: "
                    f"{item_field}", item_field)
            seen_device_ids.add(element)
            device_ids.append(element)
        try:
            return self.store.redelivery_job_event_gc_cleanup_expired_batch(
                mode, device_ids, after, limit,
                request_id if mode == "commit" else None)
        except RedeliveryJobError as error:
            if error.reason == \
                    REDELIVERY_JOB_EVENT_GC_BATCH_REQUEST_ID_CONFLICT:
                raise ServiceError(
                    "request_id was already committed with a different "
                    "payload", "request_id", status_code=409)
            raise

    def event_gc_batch_checkpoint(
            self, payload: object) -> Tuple[Dict[str, Any], int]:
        """Read or advance one consumer's batch-cleanup audit checkpoint.

        ``POST /v1/event-gc-batch/checkpoint``. The call takes no query
        parameters (the HTTP layer rejects any with 400/field ``query``)
        and the body must be a JSON object carrying exactly
        ``consumer_id``, ``expected`` and ``after``; a bad/non-object body
        is 400/field ``request_body``, a missing, wrongly typed or extra
        field is 400 with that field (the first extra key, in payload
        order). ``consumer_id`` must be a non-empty string. ``expected``
        and ``after`` come as a pair: both ``null`` for a read-only query
        or both non-boolean integers in 0..2^63-1 for a
        compare-and-advance; exactly one of them ``null`` is 400/field
        ``expected``.

        A read-only query answers 200 with the stored checkpoint — a
        consumer that never advanced reads as ``after`` 0 with a null
        ``updated_at`` — and writes nothing. A compare-and-advance whose
        ``expected`` differs from the consumer's current checkpoint (0
        when none is stored) is 409/field ``expected``; an ``after``
        below the current checkpoint or beyond the number of committed
        batch-cleanup audit records is 409/field ``after``; an ``after``
        equal to the current checkpoint is an idempotent no-op (200, the
        timestamp untouched, nothing written); a strictly greater one
        advances the checkpoint and refreshes ``updated_at`` (201). The
        decision shares the one store lock with the batch cleanup commits,
        so concurrent advances with the same ``expected`` linearize to at
        most one 201; a data-file failure on the advance is
        503/field ``data_file`` with the in-memory state, both files and
        the commit generation rolled back. On success the body keys are
        ``consumer_id``, ``after`` and ``updated_at`` in that order, with
        ``updated_at`` null or a UTC ISO-8601 timestamp (six microsecond
        digits, ``+00:00``).
        """
        if not isinstance(payload, dict):
            raise ServiceError("request body must be a JSON object",
                               "request_body")
        # The body carries exactly consumer_id, expected and after; any
        # other key is 400 with that field (the first extra key, in
        # payload order).
        extras = [key for key in payload
                  if key not in ("consumer_id", "expected", "after")]
        if extras:
            raise ServiceError(
                f"unexpected field: {extras[0]}", extras[0])
        for name in ("consumer_id", "expected", "after"):
            if name not in payload:
                raise ServiceError(f"missing required field: {name}", name)
        if not is_nonempty_string(payload["consumer_id"]):
            raise ServiceError(
                "field must be a non-empty string: consumer_id",
                "consumer_id")
        expected = payload["expected"]
        after = payload["after"]
        if (expected is None) != (after is None):
            # expected/after come as a pair: both null (read-only) or both
            # integers (compare-and-advance); exactly one null is
            # 400/expected.
            raise ServiceError(
                "expected and after must both be null or both be "
                "integers", "expected")
        if expected is not None:
            # bool is a subclass of int; reject it explicitly.
            if not isinstance(expected, int) or isinstance(expected, bool) \
                    or not 0 <= expected <= 2**63 - 1:
                raise ServiceError(
                    "field must be an integer in 0..2^63-1 or null: "
                    "expected", "expected")
            if not isinstance(after, int) or isinstance(after, bool) \
                    or not 0 <= after <= 2**63 - 1:
                raise ServiceError(
                    "field must be an integer in 0..2^63-1 or null: after",
                    "after")
        try:
            view, advanced = self.store.cleanup_checkpoint(
                payload["consumer_id"], expected, after)
        except RedeliveryJobError as error:
            if error.reason == CLEANUP_CHECKPOINT_EXPECTED_CONFLICT:
                raise ServiceError(
                    "expected does not match the consumer's current "
                    "checkpoint", "expected", status_code=409)
            if error.reason == CLEANUP_CHECKPOINT_AFTER_CONFLICT:
                raise ServiceError(
                    "after moves backwards or exceeds the number of "
                    "committed batch-cleanup audit records",
                    "after", status_code=409)
            raise
        return view, 201 if advanced else 200

    def event_gc_batch_consume(
            self, payload: object) -> Tuple[Dict[str, Any], int]:
        """Atomically pull one page of audit records for a consumer.

        ``POST /v1/event-gc-batch/consume``. The call takes no query
        parameters (the HTTP layer rejects any with 400/field ``query``)
        and the body must be a JSON object carrying exactly
        ``consumer_id``, ``expected`` and ``limit``; a bad/non-object body
        is 400/field ``request_body``, a missing, wrongly typed or extra
        field is 400 with that field (the first extra key, in payload
        order). ``consumer_id`` must be a non-empty string, ``expected`` a
        non-boolean integer in 0..2^63-1 and ``limit`` a non-boolean
        integer in 1..100.

        The consumer's checkpoint starts at 0; an ``expected`` differing
        from the current checkpoint is 409/field ``expected``. On a match
        the store returns, under the one store lock, the committed
        batch-cleanup audit records right after the checkpoint in commit
        order (at most ``limit`` of them) and advances the checkpoint to
        just past the last record of the page. A non-empty page answers
        201 with a fresh UTC ISO-8601 ``updated_at`` (six microsecond
        digits, ``+00:00``); an empty page answers 200, leaves the offset
        and the timestamp untouched and writes nothing. Concurrent pulls
        with the same ``expected`` linearize to at most one 201, the rest
        409; a data-file failure on a non-empty page is
        503/field ``data_file`` with the checkpoint, the commit
        generation and both files rolled back. On success the body keys
        are ``consumer_id``, ``records``, ``next_after``, ``has_more``
        and ``updated_at`` in that order; each record uses the six-key
        audit wire view (``request_id``, ``device_ids``, ``after``,
        ``limit``, ``status``, ``response``) with the frozen nested key
        order, and ``has_more`` says whether further records follow.
        """
        if not isinstance(payload, dict):
            raise ServiceError("request body must be a JSON object",
                               "request_body")
        # The body carries exactly consumer_id, expected and limit; any
        # other key is 400 with that field (the first extra key, in
        # payload order).
        extras = [key for key in payload
                  if key not in ("consumer_id", "expected", "limit")]
        if extras:
            raise ServiceError(
                f"unexpected field: {extras[0]}", extras[0])
        for name in ("consumer_id", "expected", "limit"):
            if name not in payload:
                raise ServiceError(f"missing required field: {name}", name)
        if not is_nonempty_string(payload["consumer_id"]):
            raise ServiceError(
                "field must be a non-empty string: consumer_id",
                "consumer_id")
        expected = payload["expected"]
        # bool is a subclass of int; reject it explicitly.
        if not isinstance(expected, int) or isinstance(expected, bool) \
                or not 0 <= expected <= 2**63 - 1:
            raise ServiceError(
                "field must be an integer in 0..2^63-1: expected",
                "expected")
        limit = payload["limit"]
        if not isinstance(limit, int) or isinstance(limit, bool) \
                or not 1 <= limit <= 100:
            raise ServiceError(
                "field must be an integer in 1..100: limit", "limit")
        try:
            view, consumed = self.store.cleanup_consume(
                payload["consumer_id"], expected, limit)
        except RedeliveryJobError as error:
            if error.reason == CLEANUP_CHECKPOINT_EXPECTED_CONFLICT:
                raise ServiceError(
                    "expected does not match the consumer's current "
                    "checkpoint", "expected", status_code=409)
            raise
        return view, 201 if consumed else 200

    def event_gc_batch_claim(
            self, payload: object) -> Tuple[Dict[str, Any], int]:
        """Claim one non-advancing audit page under a 30-second lease.

        ``POST /v1/event-gc-batch/claim``. The call takes no query
        parameters (the HTTP layer rejects any with 400/field ``query``)
        and the body must be a JSON object carrying exactly
        ``consumer_id``, ``lease_id``, ``expected`` and ``limit``; a
        bad/non-object body is 400/field ``request_body``, a missing,
        wrongly typed or extra field is 400 with that field (the first
        extra key, in payload order). ``consumer_id`` and ``lease_id``
        must be non-empty strings, ``expected`` a non-boolean integer in
        0..2^63-1 and ``limit`` a non-boolean integer in 1..100.

        The ``lease_id`` is resolved first, under the one store lock: an
        exact-payload replay (same consumer, expected and limit), whether
        the lease is still active, already acknowledged or expired,
        answers 200 with the byte-identical first response, while the id
        committed with another payload is 409/field ``lease_id``. A new
        id then requires ``expected`` to equal the consumer's current
        checkpoint (0 when none is stored) — otherwise 409/field
        ``expected`` — and the consumer must hold no other
        unacknowledged, unexpired lease — otherwise 409/field
        ``consumer_id``. The response keys are ``consumer_id``,
        ``lease_id``, ``records`` (the six-key audit wire view of the up
        to ``limit`` records starting at ``expected``), ``next_after``
        and ``expires`` in that order; ``expires`` is the claim time
        plus 30 seconds as a UTC ISO-8601 timestamp (six microsecond
        digits, ``+00:00``). The checkpoint is never advanced: the lease
        is acknowledged once a later checkpoint/consume move reaches
        ``next_after``; before that and while unexpired it blocks
        another claim by the same consumer, while an acknowledged or
        expired lease lets the consumer claim again. An empty page
        answers 200 and occupies neither the lease id nor a commit
        generation; a non-empty page answers 201 and persists once, so a
        data-file failure is 503/field ``data_file`` with the lease, the
        commit generation and both files rolled back.
        """
        if not isinstance(payload, dict):
            raise ServiceError("request body must be a JSON object",
                               "request_body")
        # The body carries exactly consumer_id, lease_id, expected and
        # limit; any other key is 400 with that field (the first extra
        # key, in payload order).
        extras = [key for key in payload
                  if key not in ("consumer_id", "lease_id", "expected",
                                 "limit")]
        if extras:
            raise ServiceError(
                f"unexpected field: {extras[0]}", extras[0])
        for name in ("consumer_id", "lease_id", "expected", "limit"):
            if name not in payload:
                raise ServiceError(f"missing required field: {name}", name)
        if not is_nonempty_string(payload["consumer_id"]):
            raise ServiceError(
                "field must be a non-empty string: consumer_id",
                "consumer_id")
        if not is_nonempty_string(payload["lease_id"]):
            raise ServiceError(
                "field must be a non-empty string: lease_id", "lease_id")
        expected = payload["expected"]
        # bool is a subclass of int; reject it explicitly.
        if not isinstance(expected, int) or isinstance(expected, bool) \
                or not 0 <= expected <= 2**63 - 1:
            raise ServiceError(
                "field must be an integer in 0..2^63-1: expected",
                "expected")
        limit = payload["limit"]
        if not isinstance(limit, int) or isinstance(limit, bool) \
                or not 1 <= limit <= 100:
            raise ServiceError(
                "field must be an integer in 1..100: limit", "limit")
        try:
            return self.store.cleanup_claim(
                payload["consumer_id"], payload["lease_id"], expected,
                limit)
        except RedeliveryJobError as error:
            if error.reason == CLEANUP_LEASE_ID_CONFLICT:
                raise ServiceError(
                    "lease_id was already committed with a different "
                    "payload", "lease_id", status_code=409)
            if error.reason == CLEANUP_CHECKPOINT_EXPECTED_CONFLICT:
                raise ServiceError(
                    "expected does not match the consumer's current "
                    "checkpoint", "expected", status_code=409)
            if error.reason == CLEANUP_LEASE_CONSUMER_BUSY:
                raise ServiceError(
                    "the consumer already holds another unacknowledged, "
                    "unexpired lease", "consumer_id", status_code=409)
            raise

    def event_gc_batch_lease_op(
            self, payload: object) -> Tuple[Dict[str, Any], int]:
        """Confirm or release one batch-cleanup audit claim lease.

        ``POST /v1/event-gc-batch/lease``. The call takes no query
        parameters (the HTTP layer rejects any with 400/field ``query``)
        and the body must be a JSON object carrying exactly
        ``consumer_id``, ``lease_id``, ``expected`` and ``op``; a
        bad/non-object body is 400/field ``request_body``, a missing,
        wrongly typed or extra field is 400 with that field (the first
        extra key, in payload order). ``consumer_id`` and ``lease_id``
        must be non-empty strings, ``expected`` a non-boolean integer in
        0..2^63-1 and ``op`` exactly ``confirm`` or ``release``.

        Under the one store lock the lease is resolved first: an unknown
        ``lease_id`` is 404/field ``lease_id`` and a lease owned by
        another consumer is 409/field ``consumer_id``. An
        unacknowledged lease whose 30-second deadline has passed is
        409/field ``lease_id`` for either op, before the
        compare-and-set; once the checkpoint has already reached
        ``next_after`` a confirm is a read-only 200 (regardless of
        ``expected``) and a release is 409/lease_id. A first
        resolution on an active, unacknowledged lease requires
        ``expected`` to equal both the lease's starting offset and the
        consumer's current checkpoint (else 409/field ``expected``).
        The same op replayed answers 200 with the byte-identical first
        response; the other op after a terminal resolution is
        409/lease_id. A first confirm advances the checkpoint to
        ``next_after`` (201); a first release leaves the checkpoint
        untouched and unblocks a later claim (201). The response keys
        are ``consumer_id``, ``lease_id``, ``op`` and ``after`` in that
        order; ``after`` is ``next_after`` for a confirm and
        ``expected`` for a release. A data-file failure is
        503/field ``data_file`` with everything rolled back.
        """
        if not isinstance(payload, dict):
            raise ServiceError("request body must be a JSON object",
                               "request_body")
        # The body carries exactly consumer_id, lease_id, expected and
        # op; any other key is 400 with that field (the first extra key,
        # in payload order).
        extras = [key for key in payload
                  if key not in ("consumer_id", "lease_id", "expected",
                                 "op")]
        if extras:
            raise ServiceError(
                f"unexpected field: {extras[0]}", extras[0])
        for name in ("consumer_id", "lease_id", "expected", "op"):
            if name not in payload:
                raise ServiceError(f"missing required field: {name}", name)
        if not is_nonempty_string(payload["consumer_id"]):
            raise ServiceError(
                "field must be a non-empty string: consumer_id",
                "consumer_id")
        if not is_nonempty_string(payload["lease_id"]):
            raise ServiceError(
                "field must be a non-empty string: lease_id", "lease_id")
        expected = payload["expected"]
        # bool is a subclass of int; reject it explicitly.
        if not isinstance(expected, int) or isinstance(expected, bool) \
                or not 0 <= expected <= 2**63 - 1:
            raise ServiceError(
                "field must be an integer in 0..2^63-1: expected",
                "expected")
        op = payload["op"]
        if op not in ("confirm", "release"):
            raise ServiceError(
                "field must be 'confirm' or 'release': op", "op")
        try:
            return self.store.cleanup_lease_op(
                payload["consumer_id"], payload["lease_id"], expected, op)
        except RedeliveryJobError as error:
            if error.reason == CLEANUP_LEASE_NOT_FOUND:
                raise ServiceError(
                    "unknown cleanup lease", "lease_id", status_code=404)
            if error.reason == CLEANUP_LEASE_CONSUMER_MISMATCH:
                raise ServiceError(
                    "the cleanup lease belongs to another consumer",
                    "consumer_id", status_code=409)
            if error.reason == CLEANUP_CHECKPOINT_EXPECTED_CONFLICT:
                raise ServiceError(
                    "expected does not match the lease's starting offset "
                    "and the consumer's current checkpoint",
                    "expected", status_code=409)
            if error.reason == CLEANUP_LEASE_STATE_CONFLICT:
                raise ServiceError(
                    "the cleanup lease cannot be resolved with this op "
                    "in its current state", "lease_id", status_code=409)
            raise

    def event_gc_batch_lease_renew(
            self, payload: object) -> Tuple[Dict[str, Any], int]:
        """Renew one batch-cleanup audit claim lease for 30 seconds.

        ``POST /v1/event-gc-batch/lease/renew``. The call takes no query
        parameters (the HTTP layer rejects any with 400/field ``query``)
        and the body must be a JSON object carrying exactly
        ``consumer_id``, ``lease_id`` and ``renewal_id``; a
        bad/non-object body is 400/field ``request_body``, a missing,
        wrongly typed or extra field is 400 with that field (the first
        extra key, in payload order). All three fields must be
        non-empty strings.

        Under the one store lock the lease is resolved first: an unknown
        ``lease_id`` is 404/field ``lease_id`` and a lease owned by
        another consumer is 409/field ``consumer_id``. A replay of the
        same ``renewal_id`` on the same lease answers 200 with the
        byte-identical first response (the id only has to be unique
        within one lease). A first renewal requires the lease to be not
        terminal, not expired and still unacknowledged with the
        consumer's checkpoint exactly at the lease's ``expected``, and
        fewer than ten renewals to exist — otherwise 409/field
        ``lease_id``. A first renewal answers 201 and extends the
        current effective deadline (the claim value initially, the last
        renewal's afterwards) by exactly 30 seconds. The response keys
        are ``consumer_id``, ``lease_id``, ``renewal_id`` and
        ``expires`` in that order; ``expires`` is a UTC ISO-8601
        timestamp (six microsecond digits, ``+00:00``). A first renewal
        persists once (commit_seq + 1); a replay writes nothing. A
        data-file failure is 503/field ``data_file`` with everything
        rolled back.
        """
        if not isinstance(payload, dict):
            raise ServiceError("request body must be a JSON object",
                               "request_body")
        # The body carries exactly consumer_id, lease_id and
        # renewal_id; any other key is 400 with that field (the first
        # extra key, in payload order).
        extras = [key for key in payload
                  if key not in ("consumer_id", "lease_id",
                                 "renewal_id")]
        if extras:
            raise ServiceError(
                f"unexpected field: {extras[0]}", extras[0])
        for name in ("consumer_id", "lease_id", "renewal_id"):
            if name not in payload:
                raise ServiceError(f"missing required field: {name}", name)
        if not is_nonempty_string(payload["consumer_id"]):
            raise ServiceError(
                "field must be a non-empty string: consumer_id",
                "consumer_id")
        if not is_nonempty_string(payload["lease_id"]):
            raise ServiceError(
                "field must be a non-empty string: lease_id", "lease_id")
        if not is_nonempty_string(payload["renewal_id"]):
            raise ServiceError(
                "field must be a non-empty string: renewal_id",
                "renewal_id")
        try:
            return self.store.cleanup_lease_renew(
                payload["consumer_id"], payload["lease_id"],
                payload["renewal_id"])
        except RedeliveryJobError as error:
            if error.reason == CLEANUP_LEASE_NOT_FOUND:
                raise ServiceError(
                    "unknown cleanup lease", "lease_id", status_code=404)
            if error.reason == CLEANUP_LEASE_CONSUMER_MISMATCH:
                raise ServiceError(
                    "the cleanup lease belongs to another consumer",
                    "consumer_id", status_code=409)
            if error.reason == CLEANUP_LEASE_RENEW_CONFLICT:
                raise ServiceError(
                    "the cleanup lease cannot be renewed in its current "
                    "state", "lease_id", status_code=409)
            raise

    def event_gc_batch_checkpoints(
            self, after: int, limit: int) -> Dict[str, Any]:
        """Page the batch-cleanup audit consumer checkpoints.

        ``GET /v1/event-gc-batch/checkpoints``. The GET takes no request
        body (the HTTP layer rejects a non-empty one with 400/field
        ``request_body``) and only the single-valued query parameters
        ``after`` and ``limit`` (any other parameter is 400/field
        ``query``), using the same paging contract as the request audit
        list: ``after`` defaults to 0 and is an unsigned decimal integer
        in 0..2**63-1, ``limit`` defaults to 100 and is in 1..100 (the
        HTTP layer enforces the decimal shape and single occurrence,
        answering 400 with the parameter name as ``field``).

        Under the store lock the checkpoint records are taken in
        creation order; the page is the records at the zero-based
        offset *after*, at most *limit* of them (consumers that never
        advanced have no record and are not listed). On success the
        body keys are ``audit_count``, ``checkpoints``, ``next_after``
        and ``has_more`` in that order; ``audit_count`` is the number
        of committed batch-cleanup audit records and each item is
        ``consumer_id``, ``after``, ``updated_at`` (the checkpoint
        view) plus ``pending`` — ``audit_count`` minus the item's
        ``after``. ``next_after`` is *after* plus the page length
        (equal to *after* for an empty page) and ``has_more`` says
        whether further records follow. The lookup writes nothing and
        advances no ``commit_seq``.
        """
        if not isinstance(after, int) or isinstance(after, bool) \
                or not 0 <= after <= 2**63 - 1:
            raise ServiceError(
                "field must be an integer in 0..2^63-1: after", "after")
        if not isinstance(limit, int) or isinstance(limit, bool) \
                or not 1 <= limit <= 100:
            raise ServiceError(
                "field must be an integer in 1..100: limit", "limit")
        return self.store.cleanup_checkpoints_page(after, limit)

    def event_gc_batch_leases(
            self, consumer_id: Optional[str], after: int,
            limit: int) -> Dict[str, Any]:
        """Page the batch-cleanup audit claim leases.

        ``GET /v1/event-gc-batch/leases``. The GET takes no request
        body (the HTTP layer rejects a non-empty one with 400/field
        ``request_body``) and only the single-valued query parameters
        ``consumer_id``, ``after`` and ``limit`` (any other parameter
        is 400/field ``query``). ``consumer_id`` is optional: omitted,
        every consumer's leases are listed; given, it must be a
        non-empty string (a repeated or empty value is 400/field
        ``consumer_id``). ``after`` defaults to 0 and is an ASCII
        decimal integer in 0..2**63-1 and ``limit`` defaults to 100 and
        is in 1..100 (the HTTP layer enforces the decimal shape and
        single occurrence, answering 400 with the parameter name as
        ``field``).

        Under the store lock the committed leases are taken in creation
        order, filtered to *consumer_id* when one is given, and the page
        is the matches at the zero-based offset *after*, at most
        *limit* of them. On success the body keys are ``leases``,
        ``next_after`` and ``has_more`` in that order; each item is
        ``lease_id``, ``consumer_id``, ``expected``, ``next_after``,
        ``expires``, ``effective_expires``, ``renewal_count`` and
        ``state`` in that order. The state is decided at one query
        instant: a released lease is ``released``; an explicitly
        confirmed one or one whose consumer checkpoint has reached
        ``next_after`` is ``confirmed``; a lease whose effective
        deadline (last renewal value, else the claim ``expires``) is at
        or before now is ``expired``; otherwise ``active``.
        ``next_after`` is *after* plus the page length (equal to
        *after* for an empty page) and ``has_more`` says whether
        further matching leases follow. The lookup writes nothing and
        advances no ``commit_seq``.
        """
        if consumer_id is not None and (
                not isinstance(consumer_id, str) or not consumer_id):
            raise ServiceError(
                "field must be a non-empty string: consumer_id",
                "consumer_id")
        if not isinstance(after, int) or isinstance(after, bool) \
                or not 0 <= after <= 2**63 - 1:
            raise ServiceError(
                "field must be an integer in 0..2^63-1: after", "after")
        if not isinstance(limit, int) or isinstance(limit, bool) \
                or not 1 <= limit <= 100:
            raise ServiceError(
                "field must be an integer in 1..100: limit", "limit")
        return self.store.cleanup_leases_page(consumer_id, after, limit)

    def event_gc_batch_lease_get(self, lease_id: str) -> Dict[str, Any]:
        """Return one committed batch-cleanup audit claim lease.

        ``GET /v1/event-gc-batch/leases/{lease_id}``. The GET takes no
        request body (the HTTP layer rejects a non-empty one with
        400/field ``request_body``) and no query parameters (any is
        400/field ``query``); the path identifier is a single non-empty
        segment strictly percent-decoded as UTF-8 by the HTTP layer (an
        empty segment or a deeper path is 404/field ``lease_id`` and a
        bad escape or invalid UTF-8 is 400/field ``lease_id``). An
        uncommitted *lease_id* (only a non-empty claim ever occupies an
        id) is 404/field ``lease_id``.

        On success the body keys are ``lease_id``, ``consumer_id``,
        ``expected``, ``next_after``, ``limit``, ``expires``,
        ``renewals``, ``terminal``, ``checkpoint``,
        ``effective_expires`` and ``state`` in that order:
        ``expected``/``next_after``/``limit``/``checkpoint`` are
        non-negative integers (``checkpoint`` is the owner consumer's
        current value, 0 without a record), ``renewals`` keeps commit
        order with each item ``renewal_id`` then ``expires`` (both
        non-empty strings), ``terminal`` is ``None``/``confirm``/
        ``release`` and ``state`` is the same single-instant
        classification as the paged view. The lookup is purely read-only
        (no write, no ``commit_seq`` change) and shares the store lock
        with every mutation.
        """
        if not isinstance(lease_id, str) or not lease_id:
            raise ServiceError(
                "field must be a non-empty string: lease_id",
                "lease_id", status_code=404)
        body = self.store.cleanup_lease_get(lease_id)
        if body is None:
            raise ServiceError(f"lease not found: {lease_id}",
                               "lease_id", status_code=404)
        return body

    def event_gc_batch_lease_events(
            self, consumer_id: Optional[str], lease_id: Optional[str],
            after: int, limit: int) -> Dict[str, Any]:
        """Page the batch-cleanup audit lease lifecycle event stream.

        ``GET /v1/event-gc-batch/lease-events``. The GET takes no
        request body (the HTTP layer rejects a non-empty one with
        400/field ``request_body``) and only the single-valued query
        parameters ``consumer_id``, ``lease_id``, ``after`` and
        ``limit`` (any other parameter is 400/field ``query``). Both
        ``consumer_id`` and ``lease_id`` are optional: omitted they
        match every consumer/lease, given they must each be a non-empty
        string (a repeated or empty value is 400 with that field).
        ``after`` defaults to 0 and is an ASCII decimal integer in
        0..2**63-1 and ``limit`` defaults to 100 and is in 1..100 (the
        HTTP layer enforces the decimal shape and single occurrence,
        answering 400 with the parameter name as ``field``).

        Under the one store lock the global event stream (a single
        chain across every lease; ``seq`` is consecutive from 1) is
        filtered by *consumer_id*/*lease_id* when given and the page is
        the matching events with ``seq > after`` in seq order, at most
        *limit* of them. On success the body keys are ``events``,
        ``next_after`` and ``has_more`` in that order; each item is
        ``seq``, ``lease_id``, ``consumer_id`` and ``type`` in that
        order, with ``type`` one of ``claim``/``renew``/``confirm``/
        ``release``/``implicit_confirm``. An empty page echoes
        ``next_after=after``; otherwise it is the last returned
        event's ``seq``; ``has_more`` says whether a matching event
        follows the page. The lookup writes nothing and advances no
        ``commit_seq``.
        """
        if consumer_id is not None and (
                not isinstance(consumer_id, str) or not consumer_id):
            raise ServiceError(
                "field must be a non-empty string: consumer_id",
                "consumer_id")
        if lease_id is not None and (
                not isinstance(lease_id, str) or not lease_id):
            raise ServiceError(
                "field must be a non-empty string: lease_id", "lease_id")
        if not isinstance(after, int) or isinstance(after, bool) \
                or not 0 <= after <= 2**63 - 1:
            raise ServiceError(
                "field must be an integer in 0..2^63-1: after", "after")
        if not isinstance(limit, int) or isinstance(limit, bool) \
                or not 1 <= limit <= 100:
            raise ServiceError(
                "field must be an integer in 1..100: limit", "limit")
        return self.store.cleanup_lease_events_page(
            consumer_id, lease_id, after, limit)

    def event_gc_batch_lease_event_consume(
            self, payload: object) -> Tuple[Dict[str, Any], int]:
        """Subscribe to, and pull one page of, the cleanup lease events.

        ``POST /v1/event-gc-batch/lease-events/consume``. The call
        takes no query parameters (the HTTP layer rejects any with
        400/field ``query``) and the body must be a JSON object
        carrying exactly ``subscriber_id``, ``expected`` and
        ``limit``; a bad/non-object body is 400/field ``request_body``,
        a missing, wrongly typed or extra field is 400 with that field
        (the first extra key, in payload order). ``subscriber_id``
        must be a non-empty string, ``expected`` a non-boolean
        integer in 0..2^63-1 and ``limit`` a non-boolean integer in
        1..100.

        A subscriber's cursor starts at 0; a brand-new subscriber
        naming a non-zero ``expected`` and an existing subscriber
        naming an ``expected`` that differs from its stored cursor
        (the last ``next_after``) are both 409/field ``expected``. On a
        match the store returns, under the one store lock, the global
        lifecycle events right after the cursor in seq order (at most
        ``limit`` of them) and advances the cursor to the last event's
        seq. A brand-new subscriber reading an empty stream (expected
        0) still registers a cursor record at after 0 and answers
        ``201``; any later empty page answers 200, leaves the cursor
        untouched and writes nothing. A non-empty page — a newly
        created subscription or a forward advance — also answers 201.
        Concurrent pulls with the same ``expected`` linearize to at
        most one 201, the rest 409; a data-file failure on a created
        or advanced cursor is 503/field ``data_file`` with the cursor,
        both files and the commit generation rolled back. On success
        the body keys are ``subscriber_id``, ``events``,
        ``next_after`` and ``has_more`` in that order; each event keeps
        the existing four-key event view (``seq``, ``lease_id``,
        ``consumer_id``, ``type``), an empty page echoes
        ``next_after=expected`` and ``has_more`` says whether a later
        event follows.
        """
        if not isinstance(payload, dict):
            raise ServiceError("request body must be a JSON object",
                               "request_body")
        # The body carries exactly subscriber_id, expected and limit;
        # any other key is 400 with that field (the first extra key, in
        # payload order).
        extras = [key for key in payload
                  if key not in ("subscriber_id", "expected", "limit")]
        if extras:
            raise ServiceError(
                f"unexpected field: {extras[0]}", extras[0])
        for name in ("subscriber_id", "expected", "limit"):
            if name not in payload:
                raise ServiceError(f"missing required field: {name}", name)
        if not is_nonempty_string(payload["subscriber_id"]):
            raise ServiceError(
                "field must be a non-empty string: subscriber_id",
                "subscriber_id")
        expected = payload["expected"]
        # bool is a subclass of int; reject it explicitly.
        if not isinstance(expected, int) or isinstance(expected, bool) \
                or not 0 <= expected <= 2**63 - 1:
            raise ServiceError(
                "field must be an integer in 0..2^63-1: expected",
                "expected")
        limit = payload["limit"]
        if not isinstance(limit, int) or isinstance(limit, bool) \
                or not 1 <= limit <= 100:
            raise ServiceError(
                "field must be an integer in 1..100: limit", "limit")
        try:
            view, advanced = self.store.lease_event_cursor_consume(
                payload["subscriber_id"], expected, limit)
        except RedeliveryJobError as error:
            if error.reason == LEASE_EVENT_CURSOR_EXPECTED_CONFLICT:
                raise ServiceError(
                    "expected does not match the subscriber's current "
                    "cursor", "expected", status_code=409)
            raise
        return view, 201 if advanced else 200

    def lease_subscription_register(
            self, payload: object) -> Tuple[Dict[str, Any], int]:
        """Register one named binding on the cleanup-lease event stream.

        ``POST /v1/lease-subs``. The call takes no query parameters
        (the HTTP layer rejects any with 400/field ``query``) and the
        body must be a JSON object carrying exactly
        ``subscriber_id``, ``consumer_id`` and ``lease_id``; a
        bad/non-object body is 400/field ``request_body``, a missing,
        wrongly typed or extra field is 400 with that field (the first
        extra key, in payload order). ``subscriber_id`` must be a
        non-empty string; ``consumer_id`` and ``lease_id`` must each
        be a non-empty string or ``null`` (a null filter matches
        every value).

        A first registration answers 201 with the binding at position
        0. Re-registering the same id with the identical filters is
        an idempotent replay (200, the frozen first-registration view
        whose ``after`` stays 0 even if the binding later advanced via
        an ack, nothing written); the same id with different filters
        is 409/field ``subscriber_id``. The registration commits once
        under the one store lock, so a data-file failure is
        503/field ``data_file`` with the in-memory state, both files
        and the commit generation rolled back. On success the body
        keys are ``subscriber_id``, ``consumer_id``, ``lease_id`` and
        ``after`` in that order; ``after`` is 0.
        """
        if not isinstance(payload, dict):
            raise ServiceError("request body must be a JSON object",
                               "request_body")
        extras = [key for key in payload
                  if key not in ("subscriber_id", "consumer_id",
                                 "lease_id")]
        if extras:
            raise ServiceError(
                f"unexpected field: {extras[0]}", extras[0])
        for name in ("subscriber_id", "consumer_id", "lease_id"):
            if name not in payload:
                raise ServiceError(f"missing required field: {name}", name)
        if not is_nonempty_string(payload["subscriber_id"]):
            raise ServiceError(
                "field must be a non-empty string: subscriber_id",
                "subscriber_id")

        def _filter(name: str) -> Optional[str]:
            value = payload[name]
            if value is None:
                return None
            if not is_nonempty_string(value):
                raise ServiceError(
                    "field must be a non-empty string or null: " + name,
                    name)
            return value

        consumer_id = _filter("consumer_id")
        lease_id = _filter("lease_id")
        try:
            return self.store.lease_subscription_register(
                payload["subscriber_id"], consumer_id, lease_id)
        except RedeliveryJobError as error:
            if error.reason == LEASE_SUBSCRIPTION_FILTER_CONFLICT:
                raise ServiceError(
                    "subscriber_id is already registered with different "
                    "filters", "subscriber_id", status_code=409)
            raise

    def lease_subscription_page(
            self, subscriber_id: str, limit: int) -> Dict[str, Any]:
        """Page one named binding's filtered event stream without advancing.

        ``GET /v1/lease-subs/{subscriber_id}``. The GET takes no
        request body and only the single-valued query parameter
        ``limit`` (the HTTP layer rejects a body with
        400/field ``request_body`` and any other parameter with
        400/field ``query``), defaulting to 100 and restricted to
        1..100 (the strict decimal shape is enforced by the HTTP
        layer). An unknown *subscriber_id* is 404/field
        ``subscriber_id`` and the lookup never creates the binding.
        Under the one store lock the global lifecycle event stream is
        filtered by the binding's ``consumer_id``/``lease_id`` (a null
        filter matches all), the page is the matching events with a
        ``seq`` greater than the binding's stored position in seq
        order (at most ``limit`` of them), and the stored position is
        never advanced. On success the body keys are
        ``subscriber_id``, ``events``, ``next_after`` and
        ``has_more`` in that order; each event keeps the four-key
        event view (``seq``, ``lease_id``, ``consumer_id``,
        ``type``), an empty page echoes the stored position as
        ``next_after`` and ``has_more`` says whether a matching event
        follows.
        """
        if not isinstance(limit, int) or isinstance(limit, bool) \
                or not 1 <= limit <= 100:
            raise ServiceError(
                "field must be an integer in 1..100: limit", "limit")
        try:
            return self.store.lease_subscription_page(
                subscriber_id, limit)
        except RedeliveryJobError as error:
            if error.reason == LEASE_SUBSCRIPTION_NOT_FOUND:
                raise ServiceError(
                    f"subscriber not found: {subscriber_id}",
                    "subscriber_id", status_code=404)
            raise

    def lease_subscription_ack(
            self, subscriber_id: str, payload: object) \
            -> Tuple[Dict[str, Any], int]:
        """Advance one named binding's position to a matching event seq.

        ``POST /v1/lease-subs/{subscriber_id}/ack``. The call takes no
        query parameters (the HTTP layer rejects any with
        400/field ``query``) and the body must be a JSON object
        carrying exactly ``expected`` and ``after``; a bad/non-object
        body is 400/field ``request_body``, a missing, wrongly typed
        or extra field is 400 with that field (the first extra key, in
        payload order). Both values must be non-boolean non-negative
        integers (no upper bound on the wire: a position above the
        stream is a state conflict, not a 400).

        An unknown *subscriber_id* is 404/field ``subscriber_id`` and
        the call never creates the binding. Under the one store lock,
        an ``expected`` differing from the binding's stored position
        is 409/field ``expected``; an ``after`` below the stored
        position, above the highest global event seq or at a seq
        whose event does not match the binding's filters is
        409/field ``after``; an equal value is an idempotent no-op
        (200, nothing written); a strictly greater matching value
        advances once and answers 201. The advance commits once, so a
        data-file failure is 503/field ``data_file`` with the
        position, both files and the commit generation rolled back.
        On success the body keys are ``subscriber_id`` and ``after``
        in that order.
        """
        if not isinstance(payload, dict):
            raise ServiceError("request body must be a JSON object",
                               "request_body")
        extras = [key for key in payload
                  if key not in ("expected", "after")]
        if extras:
            raise ServiceError(
                f"unexpected field: {extras[0]}", extras[0])
        for name in ("expected", "after"):
            if name not in payload:
                raise ServiceError(f"missing required field: {name}", name)
        expected = payload["expected"]
        after = payload["after"]
        # Any non-negative (non-boolean) integer is a well-formed
        # value; an out-of-range position is a state conflict
        # (409/expected or 409/after), not a field 400.
        for name, value in (("expected", expected), ("after", after)):
            if not isinstance(value, int) or isinstance(value, bool) \
                    or value < 0:
                raise ServiceError(
                    "field must be a non-negative integer: " + name, name)
        try:
            return self.store.lease_subscription_ack(
                subscriber_id, expected, after)
        except RedeliveryJobError as error:
            if error.reason == LEASE_SUBSCRIPTION_NOT_FOUND:
                raise ServiceError(
                    f"subscriber not found: {subscriber_id}",
                    "subscriber_id", status_code=404)
            if error.reason == LEASE_SUBSCRIPTION_EXPECTED_CONFLICT:
                raise ServiceError(
                    "expected does not match the subscriber's current "
                    "position", "expected", status_code=409)
            if error.reason == LEASE_SUBSCRIPTION_AFTER_CONFLICT:
                raise ServiceError(
                    "after moves backwards, is out of bounds or is not "
                    "the seq of a matching event",
                    "after", status_code=409)
            raise

    def event_gc_batch_cleanup_request_get(
            self, request_id: str) -> Dict[str, Any]:
        """Return one committed batch cleanup idempotency record.

        ``GET /v1/event-gc-batch/cleanup-expired/requests/{request_id}``.
        The GET takes no request body (the HTTP layer rejects a non-empty
        one with 400/field ``request_body``) and no query parameters (any
        is 400/field ``query``); the path identifier is a single non-empty
        segment strictly percent-decoded as UTF-8 by the HTTP layer (a bad
        escape or invalid UTF-8 is 400/field ``request_id``). An
        uncommitted *request_id* is 404/field ``request_id``. The lookup is
        purely read-only (no write, no ``commit_seq`` change) and shares
        the store lock with the commits. On success the body keys are
        ``request_id``, ``device_ids``, ``after``, ``limit``, ``status``
        and ``response`` in that order; ``device_ids`` keeps the committed
        order and ``response`` is the frozen first response with every
        nested key order and value untouched.
        """
        body = self.store.event_gc_batch_cleanup_request_get(request_id)
        if body is None:
            raise ServiceError(f"request not found: {request_id}",
                               "request_id", status_code=404)
        return body

    def event_gc_batch_cleanup_request_list(
            self, after: int, limit: int) -> Dict[str, Any]:
        """Page committed batch cleanup idempotency records in commit order.

        ``GET /v1/event-gc-batch/cleanup-expired/requests``. The GET takes
        no request body (the HTTP layer rejects a non-empty one with
        400/field ``request_body``) and only the single-valued query
        parameters ``after`` and ``limit`` (any other parameter is
        400/field ``query``), using the same paging contract as the
        event-gc observation page: ``after`` defaults to 0 and is an
        unsigned decimal integer in 0..2**63-1, ``limit`` defaults to 100
        and is in 1..100 (the HTTP layer enforces the decimal shape and
        single occurrence, answering 400 with the parameter name as
        ``field``).

        Under the store lock the records are taken in commit order; the
        page is the records at the zero-based offset *after*, at most
        *limit* of them. On success the body keys are ``requests``,
        ``next_after`` and ``has_more`` in that order; each item uses the
        same key order and frozen ``response`` as the detail view.
        ``next_after`` is *after* plus the page length (equal to *after*
        for an empty page) and ``has_more`` says whether further records
        follow. The lookup writes nothing and advances no ``commit_seq``.
        """
        if not isinstance(after, int) or isinstance(after, bool) \
                or not 0 <= after <= 2**63 - 1:
            raise ServiceError(
                "field must be an integer in 0..2^63-1: after", "after")
        if not isinstance(limit, int) or isinstance(limit, bool) \
                or not 1 <= limit <= 100:
            raise ServiceError(
                "field must be an integer in 1..100: limit", "limit")
        return self.store.event_gc_batch_cleanup_request_list(after, limit)

    def inbox_job_get(self, device_id: str, job_id: str) -> Dict[str, Any]:
        """Return one redelivery job's detail with its recovery chain.

        ``GET /v1/devices/{device_id}/inbox-jobs/{job_id}``. The GET takes
        no request body (the HTTP layer rejects a non-empty one with
        400/field ``request_body``) and no query parameters (any is
        400/field ``query``); the two path identifiers are single
        non-empty segments strictly percent-decoded as UTF-8 by the HTTP
        layer. An unknown device is 404/field ``device_id`` and a revoked
        device's jobs stay readable; a never-queued job is 404/field
        ``job_id`` and a job committed for another device is 409/field
        ``job_id``. The lookup is purely read-only (no write, no
        ``commit_seq`` change). On success the body keys are ``job_id``,
        ``device_id``, ``state``, ``lease_id``, ``recoveries``,
        ``cancellation_id`` and ``cancelled_at`` in that order;
        ``recoveries`` keeps commit order, each item ``recovery_id`` then
        ``lease_id`` (string or null).
        """
        try:
            return self.store.redelivery_job_get(device_id, job_id)
        except RedeliveryJobError as error:
            if error.reason == REDELIVERY_JOB_DEVICE_UNKNOWN:
                raise ServiceError(f"device not found: {device_id}",
                                   "device_id", status_code=404)
            if error.reason == REDELIVERY_JOB_NOT_FOUND:
                raise ServiceError(f"job not found: {job_id}",
                                   "job_id", status_code=404)
            if error.reason == REDELIVERY_JOB_CONFLICT:
                raise ServiceError(
                    "job_id is owned by another device",
                    "job_id", status_code=409)
            raise

    def inbox_job(self, payload: object) -> Tuple[Dict[str, Any], int]:
        """Validate and apply one 1:1-inbox redelivery job operation.

        ``POST /v1/inbox-jobs``. The body must be a JSON object carrying
        non-empty strings ``device_id``, ``job_id`` and ``op``, with ``op``
        one of ``queue``, ``dispatch``, ``status``, ``recover`` or
        ``cancel``. A bad/non-object body is 400/field ``request_body``; a
        missing, empty or wrongly typed field is 400 with the corresponding
        ``field`` (``device_id`` / ``job_id`` / ``op``), as is an ``op``
        outside the five verbs. A ``recover`` additionally requires a
        non-empty string ``recovery_id``; it missing, empty or wrongly typed
        is 400/field ``recovery_id``. A ``cancel`` body carries exactly the
        three base keys plus a non-empty string ``cancellation_id``: an
        extra key is 400 with that field, and a missing/empty/wrong
        ``cancellation_id`` is 400/field ``cancellation_id``.

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
        null when nothing remains — both 201.

        ``cancel`` ends a ``pending``/``running`` job as ``cancelled``:
        an unknown/cross-device job is 404/409/field ``job_id`` and a
        ``succeeded``/``failed`` terminal job is 409/field ``job_id``.
        Cancelling a ``pending`` job keeps ``lease_id`` null; cancelling a
        ``running`` job releases its current lease (so the messages are
        claimable again) while the job keeps that lease id. The same
        ``cancellation_id`` on the same job is an idempotent replay (200,
        the frozen first response), a different id on an already cancelled
        job is 409/field ``cancellation_id``. A first cancel is 201. The
        response keys are ``job_id``, ``device_id``, ``state``,
        ``lease_id`` in that order.
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
        if op not in ("queue", "dispatch", "status", "recover", "cancel"):
            raise ServiceError(
                "field must be one of 'queue', 'dispatch', 'status', "
                "'recover' or 'cancel': op",
                "op")
        recovery_id = None
        cancellation_id = None
        if op == "recover":
            if "recovery_id" not in payload:
                raise ServiceError(
                    "missing required field: recovery_id", "recovery_id")
            if not is_nonempty_string(payload["recovery_id"]):
                raise ServiceError(
                    "field must be a non-empty string: recovery_id",
                    "recovery_id")
            recovery_id = payload["recovery_id"]
        elif op == "cancel":
            # A cancel body carries exactly the three base keys plus the
            # non-empty string cancellation_id; any other key is 400 with
            # that field (the first extra key, in payload order).
            allowed = {"device_id", "job_id", "op", "cancellation_id"}
            extras = [key for key in payload if key not in allowed]
            if extras:
                raise ServiceError(
                    f"unexpected field: {extras[0]}", extras[0])
            if "cancellation_id" not in payload:
                raise ServiceError(
                    "missing required field: cancellation_id",
                    "cancellation_id")
            if not is_nonempty_string(payload["cancellation_id"]):
                raise ServiceError(
                    "field must be a non-empty string: cancellation_id",
                    "cancellation_id")
            cancellation_id = payload["cancellation_id"]
        try:
            if op == "recover":
                return self.store.redelivery_job_recover(
                    payload["device_id"], payload["job_id"], recovery_id)
            if op == "cancel":
                return self.store.redelivery_job_cancel(
                    payload["device_id"], payload["job_id"], cancellation_id)
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
            if error.reason == REDELIVERY_JOB_CANCELLATION_CONFLICT:
                raise ServiceError(
                    "the job is already cancelled under a different "
                    "cancellation_id", "cancellation_id", status_code=409)
            if error.reason == REDELIVERY_JOB_CANCEL_STATE:
                raise ServiceError(
                    "a succeeded or failed job cannot be cancelled",
                    "job_id", status_code=409)
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

    def inbox_job_dispatch_batch(
            self, payload: object) -> Tuple[Dict[str, Any], int]:
        """Validate and atomically apply a batch of redelivery-job dispatches.

        ``POST /v1/inbox-jobs/dispatch-batch``. The body must be a JSON
        object carrying exactly a non-empty string ``device_id`` and a
        non-empty ``items`` array (any other top-level key is
        400/that field); each item is an object carrying exactly a
        non-empty string ``job_id`` and no ``job_id`` may repeat across
        items. Shape errors are reported, in order, as 400/field
        ``request_body`` (bad/non-object body), ``device_id``
        (missing/empty/non-string), ``items`` (missing/not-a-non-empty
        array), ``items[i]`` (non-object element, an unexpected key, or a
        repeated ``job_id``) or ``items[i].job_id`` for the offending field.

        The device is then resolved (unknown/revoked -> 409/field
        ``device_id``) and every item is prechecked in array order with the
        single-job ``op=dispatch`` rules, the first error aborting the whole
        batch with nothing written and its ``field`` prefixed to
        ``items[i].``: an unknown job is 404/``items[i].job_id`` and a job
        of another device or one whose id is already occupied as an inbox
        lease id is 409/``items[i].job_id``.

        An item whose job is no longer ``pending`` replays its single-job
        dispatch; when every item is such a replay the batch answers 200
        with the current views and writes nothing, and a mix of replays and
        first-time (pending) items conflicts 409 with the first replayed
        item's ``items[i].job_id``. A first-time batch dispatches every
        pending job in input order (each with the usual 100-message limit
        and a 30-second lease named by its own ``job_id``; earlier items'
        fresh leases immediately withhold their messages from later items;
        a non-empty pick moves the job to ``running``, an empty pick to
        ``succeeded`` with ``lease_id`` null) and commits once (201). The
        body keys are ``device_id`` then ``results``; results keep input
        order and each item is ``job_id``, ``state``, ``lease_id`` in that
        order.
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
        # The body carries exactly device_id and items; any other
        # top-level key is 400 with that field (the first extra key, in
        # payload order).
        extras = [key for key in payload if key not in ("device_id", "items")]
        if extras:
            raise ServiceError(
                f"unexpected field: {extras[0]}", extras[0])

        job_ids: List[str] = []
        seen_job_ids: set = set()
        for index, element in enumerate(raw_items):
            item_field = f"items[{index}]"
            if not isinstance(element, dict):
                raise ServiceError(
                    f"array element must be an object: {item_field}",
                    item_field)
            # Each item carries exactly job_id; any other key is 400 at the
            # item level (items[i]).
            if any(key != "job_id" for key in element):
                raise ServiceError(
                    f"array element must carry only job_id: {item_field}",
                    item_field)
            job_field = f"{item_field}.job_id"
            if "job_id" not in element:
                raise ServiceError(
                    f"missing required field: {job_field}", job_field)
            if not is_nonempty_string(element["job_id"]):
                raise ServiceError(
                    f"field must be a non-empty string: {job_field}",
                    job_field)
            if element["job_id"] in seen_job_ids:
                raise ServiceError(
                    "duplicate job_id in items: "
                    f"{element['job_id']}", item_field)
            seen_job_ids.add(element["job_id"])
            job_ids.append(element["job_id"])

        try:
            results, status_code = self.store.redelivery_job_dispatch_batch(
                device_id, job_ids)
        except RedeliveryJobError as error:
            # Batch-level device failure (unknown/revoked): 409/device_id.
            if error.reason == REDELIVERY_JOB_DEVICE_UNKNOWN:
                raise ServiceError("device_id is not a registered device",
                                   "device_id", status_code=409)
            raise ServiceError("device_id is revoked",
                               "device_id", status_code=409)
        except RedeliveryJobDispatchBatchError as error:
            job_id = job_ids[error.index]
            item_field = f"items[{error.index}]"
            if error.reason == REDELIVERY_JOB_NOT_FOUND:
                raise ServiceError(f"job not found: {job_id}",
                                   f"{item_field}.job_id", status_code=404)
            if error.reason in (REDELIVERY_JOB_CONFLICT,
                                REDELIVERY_JOB_LEASE_OCCUPIED):
                raise ServiceError(
                    "job_id is already used by another device or its lease "
                    "is occupied", f"{item_field}.job_id", status_code=409)
            raise ServiceError(
                "the item replays an already applied dispatch and cannot be "
                "mixed with first-time items",
                f"{item_field}.job_id", status_code=409)
        body = {"device_id": device_id, "results": results}
        return body, status_code

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
            # Each item carries exactly job_id and recovery_id; any other
            # key is 400 at the item level (items[i]).
            if any(key not in ("job_id", "recovery_id")
                   for key in element):
                raise ServiceError(
                    f"array element must carry only job_id and "
                    f"recovery_id: {item_field}", item_field)
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

    def inbox_job_cancel_batch(
            self, payload: object) -> Tuple[Dict[str, Any], int]:
        """Validate and atomically apply a batch of redelivery-job cancels.

        ``POST /v1/inbox-jobs/cancel-batch``. The body must be a JSON
        object carrying exactly a non-empty string ``device_id`` and a
        non-empty ``items`` array (any other top-level key is
        400/that field); each item is an object carrying exactly a
        non-empty string ``job_id`` and ``cancellation_id``, and neither
        field may repeat across items. Shape errors are reported, in
        order, as 400/field ``request_body`` (bad/non-object body),
        ``device_id`` (missing/empty/non-string), ``items``
        (missing/not-a-non-empty array), ``items[i]`` (non-object
        element, an unexpected key, or a repeated ``job_id``/
        ``cancellation_id``) or ``items[i].job_id`` /
        ``items[i].cancellation_id`` for the offending field.

        The device is then resolved (unknown/revoked -> 409/field
        ``device_id``) and every item is prechecked in array order with
        the single-job ``op=cancel`` rules, the first error aborting the
        whole batch with nothing written and its ``field`` prefixed to
        ``items[i].``: an unknown job is 404/``items[i].job_id``, a job
        of another device or a ``succeeded``/``failed`` terminal job is
        409/``items[i].job_id``, and an already cancelled job named under
        a different ``cancellation_id`` is
        409/``items[i].cancellation_id``. An item replaying its own
        committed cancellation skips those checks; when every item is
        such a replay the batch answers 200 with the current views and
        writes nothing, and a mix of replays and first-time items
        conflicts 409 with the first replayed item's
        ``items[i].cancellation_id``.

        A first-time batch cancels every job in input order — a pending
        job is cancelled directly, a running job also releases its
        current lease so the messages can be claimed again — and commits
        once (201). The body keys are ``device_id`` then ``results``;
        results keep input order and each item is ``job_id``, ``state``,
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
        # The body carries exactly device_id and items; any other
        # top-level key is 400 with that field (the first extra key, in
        # payload order), mirroring the single-job op=cancel tightening.
        extras = [key for key in payload
                  if key not in ("device_id", "items")]
        if extras:
            raise ServiceError(
                f"unexpected field: {extras[0]}", extras[0])

        items: List[Tuple[str, str]] = []
        seen_job_ids: set = set()
        seen_cancellation_ids: set = set()
        for index, element in enumerate(raw_items):
            item_field = f"items[{index}]"
            if not isinstance(element, dict):
                raise ServiceError(
                    f"array element must be an object: {item_field}",
                    item_field)
            # Each item carries exactly job_id and cancellation_id; any
            # other key is 400 at the item level (items[i]).
            if any(key not in ("job_id", "cancellation_id")
                   for key in element):
                raise ServiceError(
                    "array element must carry only job_id and "
                    f"cancellation_id: {item_field}", item_field)
            job_field = f"{item_field}.job_id"
            if "job_id" not in element:
                raise ServiceError(
                    f"missing required field: {job_field}", job_field)
            if not is_nonempty_string(element["job_id"]):
                raise ServiceError(
                    f"field must be a non-empty string: {job_field}",
                    job_field)
            cancellation_field = f"{item_field}.cancellation_id"
            if "cancellation_id" not in element:
                raise ServiceError(
                    f"missing required field: {cancellation_field}",
                    cancellation_field)
            if not is_nonempty_string(element["cancellation_id"]):
                raise ServiceError(
                    f"field must be a non-empty string: "
                    f"{cancellation_field}", cancellation_field)
            if element["job_id"] in seen_job_ids:
                raise ServiceError(
                    "duplicate job_id in items: "
                    f"{element['job_id']}", item_field)
            if element["cancellation_id"] in seen_cancellation_ids:
                raise ServiceError(
                    "duplicate cancellation_id in items: "
                    f"{element['cancellation_id']}", item_field)
            seen_job_ids.add(element["job_id"])
            seen_cancellation_ids.add(element["cancellation_id"])
            items.append((element["job_id"], element["cancellation_id"]))

        try:
            results, status_code = self.store.redelivery_job_cancel_batch(
                device_id, items)
        except RedeliveryJobError as error:
            # Batch-level device failure (unknown/revoked): 409/device_id.
            if error.reason == REDELIVERY_JOB_DEVICE_UNKNOWN:
                raise ServiceError("device_id is not a registered device",
                                   "device_id", status_code=409)
            raise ServiceError("device_id is revoked",
                               "device_id", status_code=409)
        except RedeliveryJobCancelBatchError as error:
            job_id, cancellation_id = items[error.index]
            item_field = f"items[{error.index}]"
            if error.reason == REDELIVERY_JOB_NOT_FOUND:
                raise ServiceError(f"job not found: {job_id}",
                                   f"{item_field}.job_id", status_code=404)
            if error.reason == REDELIVERY_JOB_CONFLICT:
                raise ServiceError(
                    "job_id is already used by another device",
                    f"{item_field}.job_id", status_code=409)
            if error.reason == REDELIVERY_JOB_CANCEL_STATE:
                raise ServiceError(
                    "a succeeded or failed job cannot be cancelled",
                    f"{item_field}.job_id", status_code=409)
            if error.reason == REDELIVERY_JOB_CANCELLATION_CONFLICT:
                raise ServiceError(
                    "the job is already cancelled under a different "
                    "cancellation_id",
                    f"{item_field}.cancellation_id", status_code=409)
            raise ServiceError(
                "cancellation_id replays an already committed "
                "cancellation and cannot be mixed with first-time items",
                f"{item_field}.cancellation_id", status_code=409)
        body = {"device_id": device_id, "results": results}
        return body, status_code

    def inbox_job_complete_batch(
            self, payload: object) -> Tuple[Dict[str, Any], int]:
        """Validate and atomically apply a batch of inbox lease completions.

        ``POST /v1/inbox-jobs/complete-batch``. The body must be a JSON
        object carrying a non-empty string ``device_id`` and a non-empty
        ``items`` array; each item is an object with a non-empty string
        ``lease_id`` and ``completion_id`` and an ``outcome`` of exactly
        ``delivered`` or ``failed``; neither ``lease_id`` nor
        ``completion_id`` may repeat across items. Shape errors are
        reported, in order, as 400/field ``request_body`` (bad/non-object
        body), ``device_id`` (missing/empty/non-string), ``items``
        (missing/not-a-non-empty array), ``items[i]`` (non-object element,
        an unexpected key, or a repeated ``lease_id``/``completion_id``) or
        ``items[i].lease_id`` / ``items[i].completion_id`` /
        ``items[i].outcome`` for the offending field.

        The device is then resolved (unknown/revoked -> 409/field
        ``device_id``) and every item is prechecked in array order with the
        single-lease completion rules, the first error aborting the whole
        batch with nothing written and its ``field`` prefixed to
        ``items[i].``: an unknown lease is 404/``items[i].lease_id``, a
        lease of another device or an unfinished but released/expired lease
        is 409/``items[i].lease_id`` and an already-completed lease named
        under another ``completion_id`` (or a changed outcome) is
        409/``items[i].completion_id``. An item replaying its own committed
        completion (same id and outcome) skips those checks; when every
        item is such a replay the batch answers 200 with the frozen
        responses and writes nothing, and a mix of replays and first-time
        items conflicts 409 with the first replayed item's
        ``items[i].completion_id``.

        A first-time batch completes every lease in input order with one
        shared UTC timestamp and commits once (201); each item's associated
        running redelivery job moves to ``succeeded``/``failed`` with its
        outcome. The body keys are ``device_id`` then ``results``; results
        keep input order and each item is ``lease_id``, ``completion_id``,
        ``outcome``, ``completed_at`` in that order (a replay carries its
        frozen ``completed_at``).
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
        # The body carries exactly device_id and items; any other
        # top-level key is 400 with that field (the first extra key, in
        # payload order).
        extras = [key for key in payload
                  if key not in ("device_id", "items")]
        if extras:
            raise ServiceError(
                f"unexpected field: {extras[0]}", extras[0])

        items: List[Tuple[str, str, str]] = []
        seen_lease_ids: set = set()
        seen_completion_ids: set = set()
        for index, element in enumerate(raw_items):
            item_field = f"items[{index}]"
            if not isinstance(element, dict):
                raise ServiceError(
                    f"array element must be an object: {item_field}",
                    item_field)
            # Each item carries exactly lease_id, completion_id and
            # outcome; any other key is 400 at the item level (items[i]).
            if any(key not in ("lease_id", "completion_id", "outcome")
                   for key in element):
                raise ServiceError(
                    "array element must carry only lease_id, "
                    f"completion_id and outcome: {item_field}", item_field)
            lease_field = f"{item_field}.lease_id"
            if "lease_id" not in element:
                raise ServiceError(
                    f"missing required field: {lease_field}", lease_field)
            if not is_nonempty_string(element["lease_id"]):
                raise ServiceError(
                    f"field must be a non-empty string: {lease_field}",
                    lease_field)
            completion_field = f"{item_field}.completion_id"
            if "completion_id" not in element:
                raise ServiceError(
                    f"missing required field: {completion_field}",
                    completion_field)
            if not is_nonempty_string(element["completion_id"]):
                raise ServiceError(
                    f"field must be a non-empty string: "
                    f"{completion_field}", completion_field)
            outcome_field = f"{item_field}.outcome"
            if "outcome" not in element:
                raise ServiceError(
                    f"missing required field: {outcome_field}",
                    outcome_field)
            outcome = element["outcome"]
            if not isinstance(outcome, str) or outcome not in ("delivered",
                                                               "failed"):
                raise ServiceError(
                    "field must be one of 'delivered' or 'failed': "
                    f"{outcome_field}", outcome_field)
            if element["lease_id"] in seen_lease_ids:
                raise ServiceError(
                    "duplicate lease_id in items: "
                    f"{element['lease_id']}", item_field)
            if element["completion_id"] in seen_completion_ids:
                raise ServiceError(
                    "duplicate completion_id in items: "
                    f"{element['completion_id']}", item_field)
            seen_lease_ids.add(element["lease_id"])
            seen_completion_ids.add(element["completion_id"])
            items.append((element["lease_id"],
                          element["completion_id"], outcome))

        try:
            results, status_code = self.store.inbox_lease_complete_batch(
                device_id, items)
        except InboxLeaseError as error:
            # Batch-level device failure (unknown/revoked): 409/device_id.
            if error.reason == INBOX_LEASE_DEVICE_UNKNOWN:
                raise ServiceError("device_id is not a registered device",
                                   "device_id", status_code=409)
            raise ServiceError("device_id is revoked",
                               "device_id", status_code=409)
        except InboxLeaseCompleteBatchError as error:
            lease_id, completion_id, _outcome = items[error.index]
            item_field = f"items[{error.index}]"
            if error.reason == INBOX_LEASE_NOT_FOUND:
                raise ServiceError(f"lease not found: {lease_id}",
                                   f"{item_field}.lease_id", status_code=404)
            if error.reason == INBOX_LEASE_CONFLICT:
                raise ServiceError(
                    "lease_id is owned by another device",
                    f"{item_field}.lease_id", status_code=409)
            if error.reason == INBOX_LEASE_UNAVAILABLE:
                raise ServiceError(
                    "lease is released or expired and cannot be completed",
                    f"{item_field}.lease_id", status_code=409)
            if error.reason == INBOX_LEASE_COMPLETION_CONFLICT:
                raise ServiceError(
                    "completion_id is already used on this lease or the "
                    "lease has already been completed",
                    f"{item_field}.completion_id", status_code=409)
            raise ServiceError(
                "the item replays an already committed completion and "
                "cannot be mixed with first-time items",
                f"{item_field}.completion_id", status_code=409)
        body = {"device_id": device_id, "results": results}
        return body, status_code

    def inbox_job_ack_batch(
            self, payload: object) -> Tuple[Dict[str, Any], int]:
        """Validate and atomically apply a batch of inbox lease acks.

        ``POST /v1/inbox-jobs/ack-batch``. The body must be a JSON object
        carrying exactly a non-empty string ``device_id`` and a non-empty
        ``items`` array; each item is an object carrying exactly the
        non-empty strings ``lease_id`` and ``ack_id``; neither ``lease_id``
        nor ``ack_id`` may repeat across items. Shape errors are reported,
        in order, as 400/field ``request_body`` (bad/non-object body),
        ``device_id`` (missing/empty/non-string), ``items``
        (missing/not-a-non-empty array), the first extra top-level key,
        ``items[i]`` (non-object element, an unexpected key, or a repeated
        ``lease_id``/``ack_id``) or ``items[i].lease_id`` /
        ``items[i].ack_id`` for the offending field.

        The device is then resolved (unknown/revoked -> 409/field
        ``device_id``) and every item is prechecked in array order, the
        first error aborting the whole batch with nothing written and its
        ``field`` prefixed to ``items[i].``: an unknown lease is
        404/``items[i].lease_id``, a lease of another device or one
        without a ``delivered`` completion is 409/``items[i].lease_id``,
        and an already-acknowledged lease named under another ``ack_id``
        is 409/``items[i].ack_id``. An item replaying its own committed
        ack (same id) skips the remaining checks; when every item is such
        a replay the batch answers 200 with the frozen responses and
        writes nothing, and a mix of replays and first-time items
        conflicts 409 with the first replayed item's
        ``items[i].ack_id``.

        A first-time batch acknowledges every lease in input order — each
        leased message's delivery record is set acked with
        ``ack_sequence`` the message sequence (attempts untouched) and the
        lease freezes its ``ack_id`` — and commits once (201). The body
        keys are ``device_id`` then ``results``; results keep input order
        and each item is ``lease_id``, ``ack_id``, ``message_count`` in
        that order (a replay carries its frozen values).
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
        # The body carries exactly device_id and items; any other
        # top-level key is 400 with that field (the first extra key, in
        # payload order).
        extras = [key for key in payload
                  if key not in ("device_id", "items")]
        if extras:
            raise ServiceError(
                f"unexpected field: {extras[0]}", extras[0])

        items: List[Tuple[str, str]] = []
        seen_lease_ids: set = set()
        seen_ack_ids: set = set()
        for index, element in enumerate(raw_items):
            item_field = f"items[{index}]"
            if not isinstance(element, dict):
                raise ServiceError(
                    f"array element must be an object: {item_field}",
                    item_field)
            # Each item carries exactly lease_id and ack_id; any other key
            # is 400 at the item level (items[i]).
            if any(key not in ("lease_id", "ack_id") for key in element):
                raise ServiceError(
                    "array element must carry only lease_id and ack_id: "
                    f"{item_field}", item_field)
            lease_field = f"{item_field}.lease_id"
            if "lease_id" not in element:
                raise ServiceError(
                    f"missing required field: {lease_field}", lease_field)
            if not is_nonempty_string(element["lease_id"]):
                raise ServiceError(
                    f"field must be a non-empty string: {lease_field}",
                    lease_field)
            ack_field = f"{item_field}.ack_id"
            if "ack_id" not in element:
                raise ServiceError(
                    f"missing required field: {ack_field}", ack_field)
            if not is_nonempty_string(element["ack_id"]):
                raise ServiceError(
                    f"field must be a non-empty string: {ack_field}",
                    ack_field)
            if element["lease_id"] in seen_lease_ids:
                raise ServiceError(
                    "duplicate lease_id in items: "
                    f"{element['lease_id']}", item_field)
            if element["ack_id"] in seen_ack_ids:
                raise ServiceError(
                    "duplicate ack_id in items: "
                    f"{element['ack_id']}", item_field)
            seen_lease_ids.add(element["lease_id"])
            seen_ack_ids.add(element["ack_id"])
            items.append((element["lease_id"], element["ack_id"]))

        try:
            results, status_code = self.store.inbox_lease_ack_batch(
                device_id, items)
        except InboxLeaseError as error:
            # Batch-level device failure (unknown/revoked): 409/device_id.
            if error.reason == INBOX_LEASE_DEVICE_UNKNOWN:
                raise ServiceError("device_id is not a registered device",
                                   "device_id", status_code=409)
            raise ServiceError("device_id is revoked",
                               "device_id", status_code=409)
        except InboxLeaseAckBatchError as error:
            lease_id, ack_id = items[error.index]
            item_field = f"items[{error.index}]"
            if error.reason == INBOX_LEASE_NOT_FOUND:
                raise ServiceError(f"lease not found: {lease_id}",
                                   f"{item_field}.lease_id", status_code=404)
            if error.reason == INBOX_LEASE_CONFLICT:
                raise ServiceError(
                    "lease_id is owned by another device",
                    f"{item_field}.lease_id", status_code=409)
            if error.reason == INBOX_LEASE_NOT_DELIVERED:
                raise ServiceError(
                    "lease can only be acknowledged after a 'delivered' "
                    "completion", f"{item_field}.lease_id", status_code=409)
            if error.reason == INBOX_LEASE_ACK_CONFLICT:
                raise ServiceError(
                    "ack_id is already used on this lease or the lease "
                    "has already been acknowledged",
                    f"{item_field}.ack_id", status_code=409)
            raise ServiceError(
                "the item replays an already committed acknowledgement "
                "and cannot be mixed with first-time items",
                f"{item_field}.ack_id", status_code=409)
        body = {"device_id": device_id, "results": results}
        return body, status_code

    def inbox_job_renew_batch(
            self, payload: object) -> Tuple[Dict[str, Any], int]:
        """Validate and atomically apply a batch of inbox lease renewals.

        ``POST /v1/inbox-jobs/renew-batch``. The body must be a JSON
        object carrying a non-empty string ``device_id`` and a non-empty
        ``items`` array; each item is an object carrying exactly the
        non-empty strings ``lease_id`` and ``renewal_id``; neither
        ``lease_id`` nor ``renewal_id`` may repeat across items. Shape
        errors are reported, in order, as 400/field ``request_body``
        (bad/non-object body), ``device_id`` (missing/empty/non-string),
        ``items`` (missing/not-a-non-empty array), the first extra
        top-level key, ``items[i]`` (non-object element, an unexpected
        key, or a repeated ``lease_id``/``renewal_id``) or
        ``items[i].lease_id`` / ``items[i].renewal_id`` for the offending
        field.

        The device is then resolved (unknown/revoked -> 409/field
        ``device_id``) and every item is prechecked in array order at one
        uniform batch instant, the first error aborting the whole batch
        with nothing written and its ``field`` prefixed to ``items[i].``:
        an unknown lease is 404/``items[i].lease_id`` and a lease owned by
        another device, or a first-time renewal naming a lease already
        released, completed or expired by the batch instant, is
        409/``items[i].lease_id``. An item replaying a renewal already
        committed on the same lease (same renewal_id) skips the remaining
        checks; when every item is such a replay the batch answers 200 with
        the frozen responses and writes nothing, and a mix of replays and
        first-time items conflicts 409 with the first replayed item's
        ``items[i].renewal_id``.

        A first-time batch renews every lease in input order — each one's
        current effective deadline extended by exactly 30 seconds — and
        commits once (201). The body keys are ``device_id`` then
        ``results``; results keep input order and each item is
        ``lease_id``, ``renewal_id``, ``leased_until`` in that order (a
        replay carries its frozen values).
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
        # The body carries exactly device_id and items; any other
        # top-level key is 400 with that field (the first extra key, in
        # payload order).
        extras = [key for key in payload
                  if key not in ("device_id", "items")]
        if extras:
            raise ServiceError(
                f"unexpected field: {extras[0]}", extras[0])

        items: List[Tuple[str, str]] = []
        seen_lease_ids: set = set()
        seen_renewal_ids: set = set()
        for index, element in enumerate(raw_items):
            item_field = f"items[{index}]"
            if not isinstance(element, dict):
                raise ServiceError(
                    f"array element must be an object: {item_field}",
                    item_field)
            # Each item carries exactly lease_id and renewal_id; any other
            # key is 400 at the item level (items[i]).
            if any(key not in ("lease_id", "renewal_id") for key in element):
                raise ServiceError(
                    "array element must carry only lease_id and "
                    f"renewal_id: {item_field}", item_field)
            lease_field = f"{item_field}.lease_id"
            if "lease_id" not in element:
                raise ServiceError(
                    f"missing required field: {lease_field}", lease_field)
            if not is_nonempty_string(element["lease_id"]):
                raise ServiceError(
                    f"field must be a non-empty string: {lease_field}",
                    lease_field)
            renewal_field = f"{item_field}.renewal_id"
            if "renewal_id" not in element:
                raise ServiceError(
                    f"missing required field: {renewal_field}",
                    renewal_field)
            if not is_nonempty_string(element["renewal_id"]):
                raise ServiceError(
                    f"field must be a non-empty string: {renewal_field}",
                    renewal_field)
            if element["lease_id"] in seen_lease_ids:
                raise ServiceError(
                    "duplicate lease_id in items: "
                    f"{element['lease_id']}", item_field)
            if element["renewal_id"] in seen_renewal_ids:
                raise ServiceError(
                    "duplicate renewal_id in items: "
                    f"{element['renewal_id']}", item_field)
            seen_lease_ids.add(element["lease_id"])
            seen_renewal_ids.add(element["renewal_id"])
            items.append((element["lease_id"], element["renewal_id"]))

        try:
            results, status_code = self.store.inbox_lease_renew_batch(
                device_id, items)
        except InboxLeaseError as error:
            # Batch-level device failure (unknown/revoked): 409/device_id.
            if error.reason == INBOX_LEASE_DEVICE_UNKNOWN:
                raise ServiceError("device_id is not a registered device",
                                   "device_id", status_code=409)
            raise ServiceError("device_id is revoked",
                               "device_id", status_code=409)
        except InboxLeaseRenewBatchError as error:
            lease_id, renewal_id = items[error.index]
            item_field = f"items[{error.index}]"
            if error.reason == INBOX_LEASE_NOT_FOUND:
                raise ServiceError(f"lease not found: {lease_id}",
                                   f"{item_field}.lease_id", status_code=404)
            if error.reason == INBOX_LEASE_CONFLICT:
                raise ServiceError(
                    "lease_id is owned by another device",
                    f"{item_field}.lease_id", status_code=409)
            if error.reason == INBOX_LEASE_UNAVAILABLE:
                raise ServiceError(
                    "lease is released, completed or expired and cannot be "
                    "renewed", f"{item_field}.lease_id", status_code=409)
            raise ServiceError(
                "the item replays an already committed renewal and cannot "
                "be mixed with first-time items",
                f"{item_field}.renewal_id", status_code=409)
        body = {"device_id": device_id, "results": results}
        return body, status_code

    def inbox_job_release_batch(
            self, payload: object) -> Tuple[Dict[str, Any], int]:
        """Validate and atomically apply a batch of 1:1-inbox releases.

        ``POST /v1/inbox-jobs/release-batch``. The body must be a JSON
        object carrying a non-empty string ``device_id`` and a non-empty
        ``items`` array; each item is an object carrying exactly the
        non-empty string ``lease_id``, and no ``lease_id`` may repeat
        across items. Shape errors are reported, in order, as 400/field
        ``request_body`` (bad/non-object body), ``device_id``
        (missing/empty/non-string), ``items`` (missing/not-a-non-empty
        array), the first extra top-level key, ``items[i]`` (non-object
        element, an unexpected key, or a repeated ``lease_id``) or
        ``items[i].lease_id`` for the offending field.

        The device is then resolved (unknown/revoked -> 409/field
        ``device_id``) and every item is prechecked in array order, the
        first error aborting the whole batch with nothing written and its
        ``field`` prefixed to ``items[i].``: an unknown lease is
        404/``items[i].lease_id`` and a lease owned by another device, or
        a first-time release naming an already completed lease, is
        409/``items[i].lease_id``. An already-released lease is an exact
        replay and skips the completion check; when every item is such a
        replay the batch answers 200 with the frozen responses and writes
        nothing, and a mix of replays and first-time items conflicts 409
        with the first replayed item's ``items[i].lease_id``.

        A first-time batch releases every lease in input order — an
        expired but uncompleted lease may still be released — stamping
        one shared UTC ``released_at`` (six microsecond digits,
        ``+00:00``) onto every delivery record, and commits once (201).
        The body keys are ``device_id`` then ``results``; results keep
        input order and each item is ``lease_id``, ``released_at``,
        ``released_count`` in that order (a replay carries its frozen
        values).
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
        # The body carries exactly device_id and items; any other
        # top-level key is 400 with that field (the first extra key, in
        # payload order).
        extras = [key for key in payload
                  if key not in ("device_id", "items")]
        if extras:
            raise ServiceError(
                f"unexpected field: {extras[0]}", extras[0])

        lease_ids: List[str] = []
        seen_lease_ids: set = set()
        for index, element in enumerate(raw_items):
            item_field = f"items[{index}]"
            if not isinstance(element, dict):
                raise ServiceError(
                    f"array element must be an object: {item_field}",
                    item_field)
            # Each item carries exactly lease_id; any other key is 400 at
            # the item level (items[i]).
            if any(key != "lease_id" for key in element):
                raise ServiceError(
                    "array element must carry only lease_id: "
                    f"{item_field}", item_field)
            lease_field = f"{item_field}.lease_id"
            if "lease_id" not in element:
                raise ServiceError(
                    f"missing required field: {lease_field}", lease_field)
            if not is_nonempty_string(element["lease_id"]):
                raise ServiceError(
                    f"field must be a non-empty string: {lease_field}",
                    lease_field)
            if element["lease_id"] in seen_lease_ids:
                raise ServiceError(
                    "duplicate lease_id in items: "
                    f"{element['lease_id']}", item_field)
            seen_lease_ids.add(element["lease_id"])
            lease_ids.append(element["lease_id"])

        try:
            results, status_code = self.store.inbox_lease_release_batch(
                device_id, lease_ids)
        except InboxLeaseError as error:
            # Batch-level device failure (unknown/revoked): 409/device_id.
            if error.reason == INBOX_LEASE_DEVICE_UNKNOWN:
                raise ServiceError("device_id is not a registered device",
                                   "device_id", status_code=409)
            raise ServiceError("device_id is revoked",
                               "device_id", status_code=409)
        except InboxLeaseReleaseBatchError as error:
            lease_id = lease_ids[error.index]
            item_field = f"items[{error.index}]"
            if error.reason == INBOX_LEASE_NOT_FOUND:
                raise ServiceError(f"lease not found: {lease_id}",
                                   f"{item_field}.lease_id",
                                   status_code=404)
            if error.reason == INBOX_LEASE_CONFLICT:
                raise ServiceError(
                    "lease_id is owned by another device",
                    f"{item_field}.lease_id", status_code=409)
            if error.reason == INBOX_LEASE_UNAVAILABLE:
                raise ServiceError(
                    "lease is already completed and cannot be released",
                    f"{item_field}.lease_id", status_code=409)
            raise ServiceError(
                "the item replays an already committed release and cannot "
                "be mixed with first-time items",
                f"{item_field}.lease_id", status_code=409)
        body = {"device_id": device_id, "results": results}
        return body, status_code

    def inbox_job_lease_status_batch(
            self, payload: object) -> Tuple[Dict[str, Any], int]:
        """Validate and read a batch of 1:1-inbox lease states (read-only).

        ``POST /v1/inbox-jobs/lease-status-batch``. The body must be a
        JSON object carrying exactly a non-empty string ``device_id`` and
        a non-empty ``items`` array; each item is an object carrying
        exactly the non-empty string ``lease_id``, and no ``lease_id``
        may repeat across items. Shape errors are reported, in order, as
        400/field ``request_body`` (bad/non-object body), ``device_id``
        (missing/empty/non-string), ``items`` (missing/not-a-non-empty
        array), the first extra top-level key, ``items[i]`` (non-object
        element, an unexpected key, or a repeated ``lease_id``) or
        ``items[i].lease_id`` for the offending field.

        The device is then resolved (unknown/revoked -> 409/field
        ``device_id``) and every item is prechecked in array order, the
        first error aborting the whole batch with its ``field`` prefixed
        to ``items[i].``: an unknown lease is 404/``items[i].lease_id``
        and a lease owned by another device is 409/``items[i].lease_id``.

        The query is purely read-only — it shares the store lock with the
        mutating operations but writes nothing, advances no
        ``commit_seq`` and changes no state — and always answers 200 on
        success. The body keys are ``device_id`` then ``results``;
        results keep input order and each item is ``lease_id``,
        ``state``, ``leased_until``, ``released_at``, ``completion``,
        ``message_count`` in that order, every ``state`` decided against
        one uniform batch instant (a completion wins, then a release,
        then active while the effective deadline is still in the future,
        else expired).
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
        # The body carries exactly device_id and items; any other
        # top-level key is 400 with that field (the first extra key, in
        # payload order).
        extras = [key for key in payload
                  if key not in ("device_id", "items")]
        if extras:
            raise ServiceError(
                f"unexpected field: {extras[0]}", extras[0])

        lease_ids: List[str] = []
        seen_lease_ids: set = set()
        for index, element in enumerate(raw_items):
            item_field = f"items[{index}]"
            if not isinstance(element, dict):
                raise ServiceError(
                    f"array element must be an object: {item_field}",
                    item_field)
            # Each item carries exactly lease_id; any other key is 400 at
            # the item level (items[i]).
            if any(key != "lease_id" for key in element):
                raise ServiceError(
                    "array element must carry only lease_id: "
                    f"{item_field}", item_field)
            lease_field = f"{item_field}.lease_id"
            if "lease_id" not in element:
                raise ServiceError(
                    f"missing required field: {lease_field}", lease_field)
            if not is_nonempty_string(element["lease_id"]):
                raise ServiceError(
                    f"field must be a non-empty string: {lease_field}",
                    lease_field)
            if element["lease_id"] in seen_lease_ids:
                raise ServiceError(
                    "duplicate lease_id in items: "
                    f"{element['lease_id']}", item_field)
            seen_lease_ids.add(element["lease_id"])
            lease_ids.append(element["lease_id"])

        try:
            results = self.store.inbox_lease_status_batch(
                device_id, lease_ids)
        except InboxLeaseError as error:
            # Batch-level device failure (unknown/revoked): 409/device_id.
            if error.reason == INBOX_LEASE_DEVICE_UNKNOWN:
                raise ServiceError("device_id is not a registered device",
                                   "device_id", status_code=409)
            raise ServiceError("device_id is revoked",
                               "device_id", status_code=409)
        except InboxLeaseStatusBatchError as error:
            lease_id = lease_ids[error.index]
            item_field = f"items[{error.index}]"
            if error.reason == INBOX_LEASE_NOT_FOUND:
                raise ServiceError(f"lease not found: {lease_id}",
                                   f"{item_field}.lease_id",
                                   status_code=404)
            raise ServiceError(
                "lease_id is owned by another device",
                f"{item_field}.lease_id", status_code=409)
        body = {"device_id": device_id, "results": results}
        return body, 200

    def inbox_job_status_batch(
            self, payload: object) -> Tuple[Dict[str, Any], int]:
        """Validate and read a batch of redelivery-job states (read-only).

        ``POST /v1/inbox-jobs/status-batch``. The body must be a JSON
        object carrying exactly a non-empty string ``device_id`` and a
        non-empty ``items`` array; each item is an object carrying exactly
        the non-empty string ``job_id``, and no ``job_id`` may repeat
        across items. Shape errors are reported, in order, as
        400/field ``request_body`` (bad/non-object body), ``device_id``
        (missing/empty/non-string), ``items`` (missing/not-a-non-empty
        array), the first extra top-level key, ``items[i]`` (non-object
        element, an unexpected key, or a repeated ``job_id``) or
        ``items[i].job_id`` for the offending field.

        The device is then resolved under the store lock: an unknown
        device is 404/field ``device_id`` while, unlike the mutating and
        lease batch routes, a revoked device stays queryable (as for the
        single-job GET). Every item is prechecked in array order, the first
        error aborting the whole batch with its ``field`` prefixed to
        ``items[i].``: a never-queued job is 404/``items[i].job_id`` and a
        job committed for another device is 409/``items[i].job_id``.

        The query is purely read-only — it shares the store lock with the
        mutating operations but writes nothing, advances no
        ``commit_seq`` and changes no state — and always answers 200 on
        success. The body keys are ``device_id`` then ``results``; results
        keep input order and each item is ``job_id``, ``state``,
        ``lease_id``, ``recoveries``, ``cancellation_id`` and
        ``cancelled_at`` in that order, with ``recoveries`` in commit
        order (each item ``recovery_id`` then ``lease_id``).
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
        # The body carries exactly device_id and items; any other
        # top-level key is 400 with that field (the first extra key, in
        # payload order).
        extras = [key for key in payload if key not in ("device_id", "items")]
        if extras:
            raise ServiceError(
                f"unexpected field: {extras[0]}", extras[0])

        job_ids: List[str] = []
        seen_job_ids: set = set()
        for index, element in enumerate(raw_items):
            item_field = f"items[{index}]"
            if not isinstance(element, dict):
                raise ServiceError(
                    f"array element must be an object: {item_field}",
                    item_field)
            # Each item carries exactly job_id; any other key is 400 at
            # the item level (items[i]).
            if any(key != "job_id" for key in element):
                raise ServiceError(
                    f"array element must carry only job_id: {item_field}",
                    item_field)
            job_field = f"{item_field}.job_id"
            if "job_id" not in element:
                raise ServiceError(
                    f"missing required field: {job_field}", job_field)
            if not is_nonempty_string(element["job_id"]):
                raise ServiceError(
                    f"field must be a non-empty string: {job_field}",
                    job_field)
            if element["job_id"] in seen_job_ids:
                raise ServiceError(
                    "duplicate job_id in items: "
                    f"{element['job_id']}", item_field)
            seen_job_ids.add(element["job_id"])
            job_ids.append(element["job_id"])

        try:
            results = self.store.redelivery_job_status_batch(
                device_id, job_ids)
        except RedeliveryJobError as error:
            # A revoked device stays queryable; only an unknown device
            # fails, and it does so as 404/device_id (like the GET).
            if error.reason == REDELIVERY_JOB_DEVICE_UNKNOWN:
                raise ServiceError(f"device not found: {device_id}",
                                   "device_id", status_code=404)
            raise
        except RedeliveryJobStatusBatchError as error:
            job_id = job_ids[error.index]
            item_field = f"items[{error.index}]"
            if error.reason == REDELIVERY_JOB_NOT_FOUND:
                raise ServiceError(f"job not found: {job_id}",
                                   f"{item_field}.job_id",
                                   status_code=404)
            raise ServiceError(
                "job_id is owned by another device",
                f"{item_field}.job_id", status_code=409)
        body = {"device_id": device_id, "results": results}
        return body, 200

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
        elif error.reason == SESSION_ROTATED:
            message_text = (
                "session has been rotated; send new messages to its "
                "successor session")
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

    #: Maps the verified submission's storage failures to
    #: (HTTP status, field). The message-id/sequence/nonce errors reuse the
    #: ordinary mapping; identity-key and signature failures are unique to
    #: this entry.
    _VERIFIED_MESSAGE_ERROR_MAP = {
        MESSAGE_SESSION_UNKNOWN: (404, "session_id"),
        MESSAGE_SENDER_INACTIVE: (409, "sender_device_id"),
        MESSAGE_DUPLICATE_ID: (409, "message_id"),
        MESSAGE_BAD_SEQUENCE: (409, "sequence"),
        MESSAGE_DUPLICATE_NONCE: (409, "nonce"),
        MESSAGE_REQUEST_ID_CONFLICT: (409, "request_id"),
        SESSION_ROTATED: (409, "session_id"),
        MESSAGE_IDENTITY_KEY_CONFLICT: (409, "identity_key"),
        MESSAGE_SIGNATURE_INVALID: (400, "signature"),
    }

    def submit_verified_message(self, payload: object
                                ) -> Tuple[Dict[str, Any], int]:
        """Validate and commit one signature-verified message submission.

        ``POST /v1/messages/submit-verified`` accepts the eight fields
        :func:`e2ee_backend.crypto.sign_message` returns (the six committed
        envelope fields plus ``identity_key`` and ``signature``) together
        with a non-empty UTF-8 ``request_id``; extra fields are ignored and
        every string is kept verbatim. The ``request_id`` namespace is
        independent of the ordinary :meth:`submit_message` entry.

        Validation order: a body that is not an object is
        400/field=request_body and an invalid ``request_id`` is
        400/field=request_id; the eight fields then follow exactly the
        structure, UTF-8 and encoding rules of
        :func:`e2ee_backend.crypto.verify_message`, a structural failure
        being 400 naming the original field. Only afterwards, inside one
        locked transaction linearized against identity rotation and
        revocation, does replay come first (identical eight fields return
        the first response with 200 even after revocation/rotation/session
        rotation; any changed field is 409/request_id), then the existing
        session-existence, rotation-closed and sender-eligibility checks,
        then the submitted key must name the sender device's *current*
        Ed25519 key (a non-Ed25519 current key or a different key is
        409/identity_key), the signature must verify against that current
        key (400/signature), and finally the ordinary message-id, sequence
        and nonce conflict checks run in their existing order. A first
        success returns 201 with the ordinary idempotent-submission body
        plus the frozen ``identity_key`` and ``signature``; a failed
        submission consumes no id and advances no sequence, nonce, delivery
        state or cursor. Returns ``(body, status_code)``.
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
        try:
            payload["request_id"].encode("utf-8")
        except UnicodeEncodeError:
            raise ServiceError(
                "field must be encodable as UTF-8: request_id",
                "request_id") from None
        try:
            fields = parse_signed_message(payload)
        except CryptoError as error:
            raise ServiceError(error.message, error.field) from None
        signature = decode_ed25519_signature(fields["signature"])
        # parse_signed_message already guaranteed a canonical 64-byte
        # signature; the decode is repeated here only to obtain the raw bytes.
        assert signature is not None

        try:
            view, created = self.store.submit_verified_message(
                payload["request_id"],
                fields["session_id"],
                fields["sender_device_id"],
                fields["message_id"],
                fields["sequence"],
                fields["nonce"],
                fields["ciphertext"],
                fields["identity_key"],
                signature)
        except MessageCreateError as error:
            raise self._verified_message_error(error, fields)
        return view, 201 if created else 200

    @staticmethod
    def _verified_message_error(error: MessageCreateError,
                                payload: Dict[str, Any]) -> ServiceError:
        """Translate a verified-submission storage failure to a ServiceError."""
        status_code, field = \
            DeviceService._VERIFIED_MESSAGE_ERROR_MAP[error.reason]
        if error.reason == MESSAGE_SESSION_UNKNOWN:
            message_text = f"session not found: {payload['session_id']}"
        elif error.reason == MESSAGE_SENDER_INACTIVE:
            message_text = "sender_device_id is not an active device"
        elif error.reason == MESSAGE_DUPLICATE_ID:
            message_text = (f"message_id already exists in session: "
                            f"{payload['message_id']}")
        elif error.reason == MESSAGE_BAD_SEQUENCE:
            message_text = ("sequence must continue the session stream "
                            f"(got {payload['sequence']})")
        elif error.reason == MESSAGE_REQUEST_ID_CONFLICT:
            message_text = "request_id was already used with different fields"
        elif error.reason == SESSION_ROTATED:
            message_text = (
                "session has been rotated; send new messages to its "
                "successor session")
        elif error.reason == MESSAGE_IDENTITY_KEY_CONFLICT:
            message_text = (
                "identity_key does not match the sender device's current "
                "Ed25519 identity key")
        elif error.reason == MESSAGE_SIGNATURE_INVALID:
            message_text = (
                "message signature failed verification against the sender "
                "device's current identity key: signature")
        else:
            message_text = (f"nonce already used in this session: "
                            f"{payload['nonce']}")
        return ServiceError(message_text, field, status_code=status_code)

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

    def message_proof(self, session_id: str, message_id: str,
                      device_id: str) -> Dict[str, Any]:
        """Return one message's frozen signature proof (a pure read).

        The session is resolved first (unknown is 404/field=session_id),
        then the reader's eligibility: a 1:1 session is readable by any
        active registered device, a group session only by an active device
        frozen into its member snapshot — an unknown, revoked or
        ineligible device is 409/field=device_id. An unknown message id is
        404/field=message_id; a message that exists but was not committed
        through the verified submission entry (ordinary submit, direct
        send or legacy data) has no proof and is 409/field=signature — the
        query never backfills one. The answer freezes the first verified
        submission's eight values (the six envelope fields plus
        ``identity_key`` and ``signature``), so identity rotations, sender
        revocations, group roster changes and session rotations after the
        commit do not alter it. The query changes no cursor, delivery
        state, audit chain or commit generation.
        """
        try:
            return self.store.message_proof(session_id, message_id,
                                            device_id)
        except MessageListError as error:
            if error.reason == MESSAGE_SESSION_UNKNOWN:
                raise ServiceError(f"session not found: {session_id}",
                                   "session_id", status_code=404)
            if error.reason == MESSAGE_DEVICE_INACTIVE:
                raise ServiceError(
                    "device_id is not an active eligible reader",
                    "device_id", status_code=409)
            if error.reason == MESSAGE_MESSAGE_UNKNOWN:
                raise ServiceError(f"message not found: {message_id}",
                                   "message_id", status_code=404)
            if error.reason == MESSAGE_PROOF_ABSENT:
                raise ServiceError(
                    "message has no saved signature proof: signature",
                    "signature", status_code=409)
            raise  # pragma: no cover - defensive

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
