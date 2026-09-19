"""Local AES-256-GCM envelope encryption for E2EE messages.

These helpers run entirely on the client side; the server only ever stores
and forwards the resulting ciphertext. The wire envelope is::

    session_id / nonce / ciphertext

where ``nonce`` is 12 random bytes and ``ciphertext`` is the GCM ciphertext
with the 16-byte authentication tag appended, both base64-encoded. The
session id (UTF-8) is bound as additional authenticated data (AAD), so a
sealed envelope cannot be moved to another session without failing
authentication.
"""
from __future__ import annotations

import base64
import binascii
import os
from typing import Any, Dict

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

#: AES-256 key length, recommended GCM nonce length and GCM tag length.
KEY_BYTES = 32
NONCE_BYTES = 12
TAG_BYTES = 16


class EnvelopeError(Exception):
    """A field of the encryption/decryption request was invalid."""

    def __init__(self, field: str, message: str = "") -> None:
        super().__init__(message or field)
        self.field = field
        self.message = message or field

    def to_body(self) -> Dict[str, Any]:
        """JSON-serializable error body naming the offending field."""
        return {"message": self.message, "field": self.field}


def _b64_decode(value: str, field: str) -> bytes:
    """Decode standard base64 (padding optional) or raise on *field*."""
    try:
        padded = value + "=" * (-len(value) % 4)
        return base64.b64decode(padded, validate=True)
    except (binascii.Error, ValueError):
        raise EnvelopeError(field, f"field is not valid base64: {field}") from None


def _require_string(payload: Dict[str, Any], field: str,
                    *, allow_empty: bool = False) -> str:
    if not isinstance(payload, dict):
        raise EnvelopeError("request_body", "request body must be a JSON object")
    if field not in payload:
        raise EnvelopeError(field, f"missing required field: {field}")
    value = payload[field]
    if not isinstance(value, str) or (not allow_empty and not value):
        raise EnvelopeError(
            field, f"field must be a non-empty string: {field}")
    return value


def _decode_key(payload: Dict[str, Any]) -> bytes:
    raw = _b64_decode(_require_string(payload, "key"), "key")
    if len(raw) != KEY_BYTES:
        raise EnvelopeError(
            "key", f"key must decode to {KEY_BYTES} bytes, got {len(raw)}")
    return raw


def seal_message(payload: Dict[str, Any]) -> Dict[str, str]:
    """Encrypt a UTF-8 plaintext into a session-bound envelope.

    Input: ``session_id`` (non-empty str, used verbatim as AAD), ``key``
    (base64, exactly 32 bytes) and ``plaintext`` (UTF-8 string). Output:
    ``session_id``, ``nonce`` (base64 of 12 random bytes) and
    ``ciphertext`` (base64 ciphertext with the 16-byte GCM tag appended).
    """
    session_id = _require_string(payload, "session_id")
    key = _decode_key(payload)
    if "plaintext" not in payload:
        raise EnvelopeError("plaintext", "missing required field: plaintext")
    if not isinstance(payload["plaintext"], str):
        raise EnvelopeError("plaintext", "field must be a string: plaintext")

    nonce = os.urandom(NONCE_BYTES)
    aesgcm = AESGCM(key)
    sealed = aesgcm.encrypt(nonce, payload["plaintext"].encode("utf-8"),
                            session_id.encode("utf-8"))
    return {
        "session_id": session_id,
        "nonce": base64.b64encode(nonce).decode("ascii"),
        "ciphertext": base64.b64encode(sealed).decode("ascii"),
    }


def open_message(payload: Dict[str, Any]) -> Dict[str, str]:
    """Decrypt an envelope and authenticate the session binding.

    Input adds ``nonce`` (base64, 12 bytes) and ``ciphertext`` (base64,
    GCM tag appended) to the :func:`seal_message` inputs. Returns
    ``session_id`` and ``plaintext`` (UTF-8). A bad key/nonce encoding is a
    400-style error on that field; an authentication failure (wrong key,
    tampered ciphertext or AAD mismatch) is reported on ``ciphertext``.
    """
    session_id = _require_string(payload, "session_id")
    key = _decode_key(payload)
    nonce = _b64_decode(_require_string(payload, "nonce"), "nonce")
    if len(nonce) != NONCE_BYTES:
        raise EnvelopeError(
            "nonce", f"nonce must decode to {NONCE_BYTES} bytes, got {len(nonce)}")
    sealed = _b64_decode(_require_string(payload, "ciphertext"), "ciphertext")
    if len(sealed) < TAG_BYTES:
        raise EnvelopeError(
            "ciphertext", "ciphertext is shorter than the 16-byte GCM tag")

    aesgcm = AESGCM(key)
    try:
        raw_plaintext = aesgcm.decrypt(nonce, sealed,
                                       session_id.encode("utf-8"))
    except InvalidTag:
        raise EnvelopeError(
            "ciphertext",
            "authentication failed: wrong key, tampered ciphertext or "
            "session_id mismatch") from None
    try:
        plaintext = raw_plaintext.decode("utf-8")
    except UnicodeDecodeError:
        raise EnvelopeError(
            "plaintext", "decrypted payload is not valid UTF-8") from None
    return {"session_id": session_id, "plaintext": plaintext}
