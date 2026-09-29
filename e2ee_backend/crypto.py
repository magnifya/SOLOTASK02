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
import json
import os

from cryptography.exceptions import InvalidTag, UnsupportedAlgorithm
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.asymmetric import ed25519, x25519
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from typing import Dict, Optional

#: AES-GCM nonce size in bytes (96-bit nonces, as recommended for GCM).
GCM_NONCE_BYTES = 12
#: AES-GCM authentication tag size in bytes (appended to the ciphertext).
GCM_TAG_BYTES = 16
#: AES-256 key size in bytes.
AES_KEY_BYTES = 32
#: HKDF info label for locally derived session keys.
SESSION_KEY_INFO = b"e2ee-session-key-v1"


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


#: Snapshot fields that carry public X25519 material. ``initiator_identity_key``
#: is an extra local-only field: the eight-field server snapshot never stores
#: the initiator's identity public key, but the recipient needs it to compute
#: the initiator-identity/pre-key agreement of the three shared secrets.
_SESSION_PUBLIC_FIELDS = (
    "ephemeral_key",
    "identity_key",
    "public_key",
    "initiator_identity_key",
)


def load_session_snapshot(session_json: object) -> Dict[str, str]:
    """Parse and validate the frozen session JSON used for local derivation.

    Accepts either a JSON string (e.g. the output of ``show-session``) or an
    already-parsed mapping. Every required field must be a non-empty string.
    The three server-snapshot public keys and the local-only
    ``initiator_identity_key`` must all parse as X25519 public keys. Any
    failure raises :class:`CryptoError` with a precise ``field``:
    ``session_json`` for parse/shape problems, ``session_id`` for the id, or
    the offending key field for non-X25519 material.
    """
    if isinstance(session_json, (bytes, bytearray)):
        try:
            session_json = bytes(session_json).decode("utf-8")
        except UnicodeDecodeError:
            raise CryptoError("session_json must be valid UTF-8 JSON",
                              "session_json") from None
    if isinstance(session_json, str):
        try:
            snapshot = json.loads(session_json)
        except (json.JSONDecodeError, ValueError):
            raise CryptoError("session_json must be a JSON object",
                              "session_json") from None
    else:
        snapshot = session_json
    if not isinstance(snapshot, dict):
        raise CryptoError("session_json must be a JSON object", "session_json")

    for name in ("session_id", *_SESSION_PUBLIC_FIELDS):
        if name not in snapshot:
            raise CryptoError(f"missing required field: {name}",
                              "session_id" if name == "session_id" else name)
        if not is_nonempty_string(snapshot[name]):
            raise CryptoError(
                f"field must be a non-empty string: {name}",
                "session_id" if name == "session_id" else name)

    for name in _SESSION_PUBLIC_FIELDS:
        if load_x25519_public_key(snapshot[name], name) is None:
            raise CryptoError(f"field is not a valid X25519 public key: {name}",
                              name)
    return {name: snapshot[name]
            for name in ("session_id", *_SESSION_PUBLIC_FIELDS)}


def load_x25519_public_key(value: str, field: str) -> Optional[x25519.X25519PublicKey]:
    """Parse *value* strictly as an X25519 public key; return ``None`` otherwise.

    Accepts PEM SubjectPublicKeyInfo, base64/hex DER SubjectPublicKeyInfo, or a
    base64/hex raw 32-byte point. Ed25519 and other curves are rejected.
    """
    if not is_nonempty_string(value):
        return None

    try:
        key = serialization.load_pem_public_key(value.encode("ascii"))
    except (ValueError, UnsupportedAlgorithm, TypeError):
        key = None
    if isinstance(key, x25519.X25519PublicKey):
        return key

    candidates: list[bytes] = []
    blob = _decode_base64(value)
    if blob is not None:
        candidates.append(blob)
    blob = _decode_hex(value)
    if blob is not None and blob not in candidates:
        candidates.append(blob)

    for candidate in candidates:
        if len(candidate) == 32:
            try:
                return x25519.X25519PublicKey.from_public_bytes(candidate)
            except (ValueError, UnsupportedAlgorithm, TypeError):
                pass
        try:
            der_key = serialization.load_der_public_key(candidate)
        except (ValueError, UnsupportedAlgorithm, TypeError):
            continue
        if isinstance(der_key, x25519.X25519PublicKey):
            return der_key
    return None


def load_x25519_private_key(value: object,
                            field: str) -> x25519.X25519PrivateKey:
    """Parse an X25519 private key in PEM PKCS#8, DER, or raw 32-byte form.

    PEM text must be PKCS#8; other encodings accept base64/hex DER PKCS#8 or
    base64/hex of the raw 32-byte scalar. Anything missing, malformed or not
    X25519 raises :class:`CryptoError` naming *field*.
    """
    if not is_nonempty_string(value):
        raise CryptoError(f"missing or invalid field: {field}", field)
    assert isinstance(value, str)

    try:
        key = serialization.load_pem_private_key(
            value.encode("ascii"), password=None)
    except (ValueError, UnsupportedAlgorithm, TypeError):
        key = None
    if isinstance(key, x25519.X25519PrivateKey):
        return key

    candidates: list[bytes] = []
    blob = _decode_base64(value)
    if blob is not None:
        candidates.append(blob)
    blob = _decode_hex(value)
    if blob is not None and blob not in candidates:
        candidates.append(blob)

    for candidate in candidates:
        if len(candidate) == 32:
            try:
                return x25519.X25519PrivateKey.from_private_bytes(candidate)
            except (ValueError, UnsupportedAlgorithm, TypeError):
                pass
        try:
            der_key = serialization.load_der_private_key(
                candidate, password=None)
        except (ValueError, UnsupportedAlgorithm, TypeError):
            continue
        if isinstance(der_key, x25519.X25519PrivateKey):
            return der_key
    raise CryptoError(f"field is not a valid X25519 private key: {field}", field)


def _require_matching_private_key(private_key: x25519.X25519PrivateKey,
                                  public_value: str, field: str) -> None:
    """Ensure *private_key* corresponds to the snapshot's *public_value*."""
    public_key = load_x25519_public_key(public_value, field)
    if public_key is None:
        raise CryptoError(
            f"field is not a valid X25519 public key: {field}", field)
    derived = private_key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    expected = public_key.public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    if derived != expected:
        raise CryptoError(
            f"private key does not match the snapshot public key: {field}",
            field)


def _x25519_exchange(private_key: x25519.X25519PrivateKey,
                    public_value: str, field: str) -> bytes:
    """Perform one X25519 ECDH against a snapshot public key."""
    public_key = load_x25519_public_key(public_value, field)
    if public_key is None:
        raise CryptoError(
            f"field is not a valid X25519 public key: {field}", field)
    try:
        return private_key.exchange(public_key)
    except ValueError:
        raise CryptoError(f"X25519 key agreement failed for field: {field}",
                          field) from None


def derive_session_key(session_json: object, role: object,
                       identity_private: object,
                       ephemeral_private: object = None,
                       prekey_private: object = None) -> Dict[str, str]:
    """Locally derive the shared AES-256 session key from one side's keys.

    *role* is ``"initiator"`` (identity + ephemeral private keys) or
    ``"recipient"`` (identity + pre-key private keys). Both sides compute the
    same three X25519 shared secrets, in the same order::

        DH(IK_initiator, SPK_recipient)
        || DH(EK_initiator, IK_recipient)
        || DH(EK_initiator, SPK_recipient)

    HKDF-SHA256 with ``salt = UTF-8(session_id)`` and
    ``info = b"e2ee-session-key-v1"`` then yields the 32-byte key. Every
    validation failure raises :class:`CryptoError` with the precise field.
    """
    snapshot = load_session_snapshot(session_json)

    if not is_nonempty_string(role) or role not in ("initiator", "recipient"):
        raise CryptoError("role must be 'initiator' or 'recipient'", "role")

    if not is_nonempty_string(identity_private):
        raise CryptoError(
            "missing or invalid field: identity_private_key",
            "identity_private_key")
    identity = load_x25519_private_key(
        identity_private, "identity_private_key")

    if role == "initiator":
        if not is_nonempty_string(ephemeral_private):
            raise CryptoError(
                "missing or invalid field: ephemeral_private_key",
                "ephemeral_private_key")
        ephemeral = load_x25519_private_key(
            ephemeral_private, "ephemeral_private_key")
        _require_matching_private_key(
            identity, snapshot["initiator_identity_key"],
            "identity_private_key")
        _require_matching_private_key(
            ephemeral, snapshot["ephemeral_key"],
            "ephemeral_private_key")
        secrets = (
            _x25519_exchange(identity, snapshot["public_key"],
                             "identity_private_key"),
            _x25519_exchange(ephemeral, snapshot["identity_key"],
                             "ephemeral_private_key"),
            _x25519_exchange(ephemeral, snapshot["public_key"],
                             "ephemeral_private_key"),
        )
    else:
        if not is_nonempty_string(prekey_private):
            raise CryptoError(
                "missing or invalid field: prekey_private_key",
                "prekey_private_key")
        prekey = load_x25519_private_key(
            prekey_private, "prekey_private_key")
        _require_matching_private_key(
            identity, snapshot["identity_key"], "identity_private_key")
        _require_matching_private_key(
            prekey, snapshot["public_key"], "prekey_private_key")
        secrets = (
            _x25519_exchange(prekey, snapshot["initiator_identity_key"],
                             "prekey_private_key"),
            _x25519_exchange(identity, snapshot["ephemeral_key"],
                             "identity_private_key"),
            _x25519_exchange(prekey, snapshot["ephemeral_key"],
                             "prekey_private_key"),
        )

    session_id = snapshot["session_id"]
    key = HKDF(
        algorithm=hashes.SHA256(),
        length=AES_KEY_BYTES,
        salt=session_id.encode("utf-8"),
        info=SESSION_KEY_INFO,
    ).derive(b"".join(secrets))
    return {
        "session_id": session_id,
        "key": base64.b64encode(key).decode("ascii"),
    }
