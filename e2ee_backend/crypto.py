"""Public-key parsing and validation helpers built on the ``cryptography`` library.

The server only ever sees *public* key material. These helpers accept the common
wire encodings clients use to publish identity keys and signed pre-keys:

* PEM text (SubjectPublicKeyInfo),
* base64- or hex-encoded DER (SubjectPublicKeyInfo),
* raw 32-byte X25519/Ed25519 points (base64- or hex-encoded).
"""
from __future__ import annotations

import base64
import binascii
import os

from cryptography.exceptions import InvalidTag, UnsupportedAlgorithm
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, x25519
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from typing import Optional

#: AES-GCM nonce size in bytes (96-bit nonces, as recommended for GCM).
GCM_NONCE_BYTES = 12
#: AES-GCM authentication tag size in bytes (appended to the ciphertext).
GCM_TAG_BYTES = 16
#: AES-256 key size in bytes.
AES_KEY_BYTES = 32


class CryptoError(Exception):
    """A local encryption/decryption failure naming the offending field."""

    def __init__(self, message: str, field: str) -> None:
        super().__init__(message)
        self.message = message
        self.field = field


def _decode_strict_base64(value: str, field: str) -> bytes:
    """Decode standard base64; raise :class:`CryptoError` on failure."""
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        raise CryptoError(f"field must be valid base64: {field}", field) from None


def _load_aes_key(key_b64: str) -> AESGCM:
    """Decode a base64 AES-256 key (exactly 32 bytes) into an AES-GCM cipher."""
    if not is_nonempty_string(key_b64):
        raise CryptoError("field must be a non-empty string: key", "key")
    key = _decode_strict_base64(key_b64, "key")
    if len(key) != AES_KEY_BYTES:
        raise CryptoError(
            f"key must decode to {AES_KEY_BYTES} bytes (got {len(key)})", "key")
    return AESGCM(key)


def encrypt_message(session_id: str, key_b64: str,
                    plaintext: str) -> dict:
    """Encrypt *plaintext* (UTF-8) with AES-256-GCM.

    The 12-byte nonce is generated randomly; the session id is bound to the
    ciphertext as additional authenticated data. Returns a dict with
    ``session_id``, ``nonce`` and ``ciphertext`` (both base64; the ciphertext
    carries the 16-byte GCM tag appended).
    """
    if not is_nonempty_string(session_id):
        raise CryptoError("field must be a non-empty string: session_id",
                          "session_id")
    if not isinstance(plaintext, str):
        raise CryptoError("field must be a string: plaintext", "plaintext")
    cipher = _load_aes_key(key_b64)
    nonce = os.urandom(GCM_NONCE_BYTES)
    ciphertext = cipher.encrypt(nonce, plaintext.encode("utf-8"),
                                session_id.encode("utf-8"))
    return {
        "session_id": session_id,
        "nonce": base64.b64encode(nonce).decode("ascii"),
        "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
    }


def decrypt_message(session_id: str, key_b64: str, nonce_b64: str,
                    ciphertext_b64: str) -> dict:
    """Decrypt a payload produced by :func:`encrypt_message`.

    Returns a dict with ``session_id`` and the UTF-8 ``plaintext``. Any
    decoding or authentication failure raises :class:`CryptoError` naming the
    offending field.
    """
    if not is_nonempty_string(session_id):
        raise CryptoError("field must be a non-empty string: session_id",
                          "session_id")
    cipher = _load_aes_key(key_b64)

    if not is_nonempty_string(nonce_b64):
        raise CryptoError("field must be a non-empty string: nonce", "nonce")
    nonce = _decode_strict_base64(nonce_b64, "nonce")
    if len(nonce) != GCM_NONCE_BYTES:
        raise CryptoError(
            f"nonce must decode to {GCM_NONCE_BYTES} bytes (got {len(nonce)})",
            "nonce")

    if not is_nonempty_string(ciphertext_b64):
        raise CryptoError("field must be a non-empty string: ciphertext",
                          "ciphertext")
    ciphertext = _decode_strict_base64(ciphertext_b64, "ciphertext")
    if len(ciphertext) < GCM_TAG_BYTES:
        raise CryptoError("ciphertext is too short to carry a GCM tag",
                          "ciphertext")

    try:
        plaintext = cipher.decrypt(nonce, ciphertext,
                                   session_id.encode("utf-8"))
    except InvalidTag:
        raise CryptoError("ciphertext failed authentication", "ciphertext") \
            from None
    try:
        text = plaintext.decode("utf-8")
    except UnicodeDecodeError:
        raise CryptoError("plaintext is not valid UTF-8", "plaintext") \
            from None
    return {"session_id": session_id, "plaintext": text}


def is_nonempty_string(value: object) -> bool:
    """Return True iff *value* is a ``str`` that contains at least one character."""
    return isinstance(value, str) and len(value) > 0


def _decode_base64(value: str) -> Optional[bytes]:
    """Decode standard base64 (padding optional); return ``None`` on failure."""
    try:
        padded = value + "=" * (-len(value) % 4)
        return base64.b64decode(padded, validate=True)
    except (binascii.Error, ValueError):
        return None


def _decode_hex(value: str) -> Optional[bytes]:
    """Decode a hex string; return ``None`` on failure."""
    try:
        return binascii.unhexlify(value)
    except (binascii.Error, ValueError):
        return None


def _load_raw_point(data: bytes) -> Optional[object]:
    """Parse a raw 32-byte X25519 or Ed25519 public point."""
    if len(data) != 32:
        return None
    for loader in (x25519.X25519PublicKey.from_public_bytes,
                   ed25519.Ed25519PublicKey.from_public_bytes):
        try:
            return loader(data)
        except (ValueError, UnsupportedAlgorithm):
            continue
    return None


def _load_der(data: bytes) -> Optional[object]:
    try:
        return serialization.load_der_public_key(data)
    except (ValueError, UnsupportedAlgorithm, TypeError):
        return None


def load_public_key(value: str) -> Optional[object]:
    """Best-effort parse of a public key.

    Returns the parsed key object, or ``None`` when it cannot be parsed. Only
    public material is ever handled here; private keys are not loaded on purpose.
    """
    if not is_nonempty_string(value):
        return None

    # 1) PEM text.
    try:
        return serialization.load_pem_public_key(value.encode("ascii"))
    except (ValueError, UnsupportedAlgorithm, TypeError):
        pass

    # 2) Encoded bytes: collect every plausible decoding. A hex string can also
    #    be valid base64 (e.g. a 64-char hex digest), so both are candidates and
    #    the first one that actually parses as a key wins.
    candidates: list[bytes] = []
    blob = _decode_base64(value)
    if blob is not None:
        candidates.append(blob)
    blob = _decode_hex(value)
    if blob is not None and blob not in candidates:
        candidates.append(blob)

    # 3) Raw 32-byte point (X25519/Ed25519), otherwise DER SubjectPublicKeyInfo.
    for candidate in candidates:
        raw = _load_raw_point(candidate)
        if raw is not None:
            return raw
        der = _load_der(candidate)
        if der is not None:
            return der
    return None
