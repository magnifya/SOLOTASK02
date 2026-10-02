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
import hashlib
import json
import os

from cryptography.exceptions import (
    InvalidSignature,
    InvalidTag,
    UnsupportedAlgorithm,
)
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

#: Domain-separation prefix for the signed pre-key proof message.
SIGNED_PREKEY_PROOF_PREFIX = "E2EE-SIGNED-PREKEY-V1"
#: Domain-separation prefix for the envelope-authenticated message AAD.
MESSAGE_ENVELOPE_PREFIX = "E2EE-MESSAGE-ENVELOPE-V1"
#: Domain-separation prefix for the identity-key fingerprint message.
IDENTITY_FINGERPRINT_PREFIX = "E2EE-IDENTITY-FINGERPRINT-V1"
#: Length of a fingerprint: SHA-256 rendered as lowercase hexadecimal.
IDENTITY_FINGERPRINT_HEX_LEN = 64
#: Length of an Ed25519 signature in bytes.
ED25519_SIGNATURE_BYTES = 64


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


def _normalize_envelope(sender_device_id: object, message_id: object,
                        sequence: object) -> Optional[tuple]:
    """Validate the optional message-envelope binding as an all-or-nothing group.

    Returns ``None`` when all three values are omitted (legacy protocol);
    returns ``(sender_device_id, message_id, sequence)`` when all three are
    present and valid. Partial provision, empty/non-string identifiers, or a
    sequence that is not a positive integer (booleans excluded) raises
    :class:`CryptoError`, reporting the first bad field in fixed order
    (``sender_device_id``, ``message_id``, ``sequence``).
    """
    provided = [value is not None for value in
                (sender_device_id, message_id, sequence)]
    if not any(provided):
        return None
    if not is_nonempty_string(sender_device_id):
        raise CryptoError(
            "field must be a non-empty string: sender_device_id",
            "sender_device_id")
    if not is_nonempty_string(message_id):
        raise CryptoError(
            "field must be a non-empty string: message_id", "message_id")
    # ``bool`` is a subclass of ``int``; reject it explicitly.
    if not isinstance(sequence, int) or isinstance(sequence, bool) \
            or sequence < 1:
        raise CryptoError(
            "field must be a positive integer: sequence", "sequence")
    return sender_device_id, message_id, sequence


def _message_aad(session_id: str,
                 envelope: Optional[tuple]) -> bytes:
    """Build the GCM AAD for a message, selecting legacy vs. envelope mode.

    Legacy mode authenticates only the raw session-id bytes. Envelope mode
    authenticates ``E2EE-MESSAGE-ENVELOPE-V1``, one newline and compact JSON
    (sorted keys, non-ASCII kept as-is) carrying ``session_id``,
    ``sender_device_id``, ``message_id`` and the integer ``sequence``.
    """
    if envelope is None:
        return session_id.encode("utf-8")
    sender_device_id, message_id, sequence = envelope
    document = json.dumps(
        {"message_id": message_id, "sender_device_id": sender_device_id,
         "sequence": sequence, "session_id": session_id},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return (MESSAGE_ENVELOPE_PREFIX + "\n" + document).encode("utf-8")


def encrypt_message(session_id: str, key_b64: str,
                    plaintext: str,
                    sender_device_id: Optional[str] = None,
                    message_id: Optional[str] = None,
                    sequence: Optional[int] = None) -> dict:
    """Encrypt *plaintext* (UTF-8) with AES-256-GCM.

    The 12-byte nonce is generated randomly; the session id is bound to the
    ciphertext as additional authenticated data. When *sender_device_id*,
    *message_id* and *sequence* are all provided, the envelope-mode AAD also
    binds those values; when all three are omitted the legacy session-only
    AAD is used. Returns a dict with ``session_id``, ``nonce`` and
    ``ciphertext`` (both base64; the ciphertext carries the 16-byte GCM tag
    appended).
    """
    if not is_nonempty_string(session_id):
        raise CryptoError("field must be a non-empty string: session_id",
                          "session_id")
    envelope = _normalize_envelope(sender_device_id, message_id, sequence)
    if not isinstance(plaintext, str):
        raise CryptoError("field must be a string: plaintext", "plaintext")
    cipher = _load_aes_key(key_b64)
    nonce = os.urandom(GCM_NONCE_BYTES)
    ciphertext = cipher.encrypt(nonce, plaintext.encode("utf-8"),
                                _message_aad(session_id, envelope))
    return {
        "session_id": session_id,
        "nonce": base64.b64encode(nonce).decode("ascii"),
        "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
    }


def decrypt_message(session_id: str, key_b64: str, nonce_b64: str,
                    ciphertext_b64: str,
                    sender_device_id: Optional[str] = None,
                    message_id: Optional[str] = None,
                    sequence: Optional[int] = None) -> dict:
    """Decrypt a payload produced by :func:`encrypt_message`.

    Returns a dict with ``session_id`` and the UTF-8 ``plaintext``. Any
    decoding or authentication failure raises :class:`CryptoError` naming the
    offending field. The envelope arguments must match encryption exactly:
    all omitted for legacy payloads, all provided for envelope payloads.
    There is no fallback between the two AAD schemes.
    """
    if not is_nonempty_string(session_id):
        raise CryptoError("field must be a non-empty string: session_id",
                          "session_id")
    envelope = _normalize_envelope(sender_device_id, message_id, sequence)
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
                                   _message_aad(session_id, envelope))
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


def _load_ed25519_raw(data: bytes) -> Optional[ed25519.Ed25519PublicKey]:
    """Parse a raw 32-byte Ed25519 public point; return ``None`` on failure."""
    if len(data) != 32:
        return None
    try:
        return ed25519.Ed25519PublicKey.from_public_bytes(data)
    except (ValueError, UnsupportedAlgorithm):
        return None


def _load_ed25519_der(data: bytes) -> Optional[ed25519.Ed25519PublicKey]:
    """Parse a DER SubjectPublicKeyInfo key; keep it only when it is Ed25519."""
    try:
        key = serialization.load_der_public_key(data)
    except (ValueError, UnsupportedAlgorithm, TypeError):
        return None
    return key if isinstance(key, ed25519.Ed25519PublicKey) else None


def load_ed25519_public_key(value: str) -> Optional[ed25519.Ed25519PublicKey]:
    """Parse *value* strictly as an Ed25519 public key.

    Accepts the same encodings as :func:`load_public_key` — PEM text,
    standard base64 (padding optional) or hex DER SubjectPublicKeyInfo, and
    standard base64 or hex of a raw 32-byte point — but only an Ed25519 key
    is accepted: an X25519 key or any other algorithm yields ``None``.
    """
    if not is_nonempty_string(value):
        return None

    # 1) PEM text.
    try:
        key = serialization.load_pem_public_key(value.encode("ascii"))
    except (ValueError, UnsupportedAlgorithm, TypeError):
        pass
    else:
        return key if isinstance(key, ed25519.Ed25519PublicKey) else None

    # 2) Encoded bytes: a hex string can also be valid base64, so both
    #    decodings are candidates; the first one that parses as Ed25519 wins.
    candidates: list[bytes] = []
    blob = _decode_base64(value)
    if blob is not None:
        candidates.append(blob)
    blob = _decode_hex(value)
    if blob is not None and blob not in candidates:
        candidates.append(blob)

    # 3) Raw 32-byte Ed25519 point first, otherwise DER SubjectPublicKeyInfo.
    for candidate in candidates:
        raw = _load_ed25519_raw(candidate)
        if raw is not None:
            return raw
        der = _load_ed25519_der(candidate)
        if der is not None:
            return der
    return None


def decode_ed25519_signature(value: object) -> Optional[bytes]:
    """Decode a standard-base64 Ed25519 signature that must be 64 bytes.

    Returns the raw 64-byte signature, or ``None`` when *value* is not a
    string, is not canonical standard base64 (correct padding and alphabet
    only — no whitespace or URL-safe alphabet), or does not decode to
    exactly 64 bytes. Re-encoding the decoded bytes must reproduce the
    input, so non-canonical padding/spellings are refused too.
    """
    if not isinstance(value, str):
        return None
    try:
        raw = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        return None
    if len(raw) != ED25519_SIGNATURE_BYTES:
        return None
    if base64.b64encode(raw).decode("ascii") != value:
        return None
    return raw


def signed_prekey_proof_message(user_id: str, device_id: str, key_id: str,
                                public_key: str) -> bytes:
    """Build the exact bytes an identity key signs for one signed pre-key.

    The message is the domain prefix ``E2EE-SIGNED-PREKEY-V1``, one newline,
    then compact JSON of the four fields with keys sorted
    (``device_id``, ``key_id``, ``public_key``, ``user_id``) and Unicode
    written as-is. The string values are the request's original strings, so
    the caller passes them through unmodified.
    """
    document = json.dumps(
        {"device_id": device_id, "key_id": key_id,
         "public_key": public_key, "user_id": user_id},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return (SIGNED_PREKEY_PROOF_PREFIX + "\n" + document).encode("utf-8")


def verify_signed_prekey(identity_key: ed25519.Ed25519PublicKey,
                         signature: bytes, user_id: str, device_id: str,
                         key_id: str, public_key: str) -> bool:
    """Verify one signed pre-key proof against the Ed25519 identity key.

    Returns ``True`` iff *signature* is valid over
    :func:`signed_prekey_proof_message` for the four field values.
    """
    message = signed_prekey_proof_message(
        user_id, device_id, key_id, public_key)
    try:
        identity_key.verify(signature, message)
    except InvalidSignature:
        return False
    return True


def canonical_public_key_bytes(value: str) -> Optional[bytes]:
    """Return the canonical encoding of the public key in *value*.

    Accepts the same encodings as :func:`load_public_key` (PEM text,
    base64/hex DER SubjectPublicKeyInfo, or base64/hex raw 32-byte
    X25519/Ed25519 point). Two wire spellings of the same key canonicalize
    to the same bytes: an X25519/Ed25519 key in any encoding becomes its raw
    32-byte point, while any other key type becomes its DER
    SubjectPublicKeyInfo. Returns ``None`` when *value* does not parse as a
    public key.
    """
    key = load_public_key(value)
    if key is None:
        return None
    if isinstance(key, (x25519.X25519PublicKey, ed25519.Ed25519PublicKey)):
        return key.public_bytes(serialization.Encoding.Raw,
                                serialization.PublicFormat.Raw)
    return key.public_bytes(serialization.Encoding.DER,
                            serialization.PublicFormat.SubjectPublicKeyInfo)


def identity_fingerprint(identity_key: str) -> Optional[str]:
    """Compute the domain-separated SHA-256 fingerprint of an identity key.

    The fingerprint is the lowercase hex SHA-256 of the UTF-8 bytes of the
    ``E2EE-IDENTITY-FINGERPRINT-V1`` domain prefix, one newline and the
    canonical public-key bytes (see
    :func:`canonical_public_key_bytes`). Because the key is canonicalized
    first, PEM, DER (base64 or hex) and raw-point (base64 or hex) spellings
    of the same key yield the same 64-character fingerprint. Returns
    ``None`` when *identity_key* does not parse as a public key.
    """
    canonical = canonical_public_key_bytes(identity_key)
    if canonical is None:
        return None
    message = (IDENTITY_FINGERPRINT_PREFIX + "\n").encode("utf-8") + canonical
    return hashlib.sha256(message).hexdigest()


def is_fingerprint_format(value: object) -> bool:
    """Return True iff *value* is 64 lowercase hexadecimal characters."""
    return (isinstance(value, str)
            and len(value) == IDENTITY_FINGERPRINT_HEX_LEN
            and all(char in "0123456789abcdef" for char in value))
