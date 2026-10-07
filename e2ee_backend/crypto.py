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
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, x25519
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from typing import Any, List, Optional, Tuple

#: AES-GCM nonce size in bytes (96-bit nonces, as recommended for GCM).
GCM_NONCE_BYTES = 12
#: AES-GCM authentication tag size in bytes (appended to the ciphertext).
GCM_TAG_BYTES = 16
#: AES-256 key size in bytes.
AES_KEY_BYTES = 32

#: Domain-separation prefix for the signed pre-key proof message.
SIGNED_PREKEY_PROOF_PREFIX = "E2EE-SIGNED-PREKEY-V1"
#: Domain-separation prefix for the authorized identity-rotation message.
IDENTITY_ROTATION_PREFIX = "E2EE-IDENTITY-ROTATION-V1"
#: Domain-separation prefix for the signature-authorized device revocation.
DEVICE_REVOCATION_PREFIX = "E2EE-DEVICE-REVOCATION-V1"
#: Domain-separation prefix for the signature-authorized group-membership
#: change (add/remove) message.
GROUP_MEMBERSHIP_PREFIX = "E2EE-GROUP-MEMBERSHIP-V1"
#: Domain-separation prefix for the signature-authorized group-session
#: rotation message.
GROUP_SESSION_ROTATION_PREFIX = "E2EE-GROUP-SESSION-ROTATION-V1"
#: Domain-separation prefix for the identity-key fingerprint message.
IDENTITY_FINGERPRINT_PREFIX = "E2EE-IDENTITY-FINGERPRINT-V1"
#: Domain-separation prefix for the message-envelope AAD document.
MESSAGE_ENVELOPE_PREFIX = "E2EE-MESSAGE-ENVELOPE-V1"
#: Domain-separation prefix for the offline signed ciphertext envelope.
SIGNED_MESSAGE_PREFIX = "E2EE-SIGNED-MESSAGE-V1"
#: Domain-separation prefix for the one-to-one session-key HKDF info.
SESSION_KEY_INFO_PREFIX = "E2EE-SESSION-KEY-V1"
#: Domain-separation prefix for the signature-authorized batch sync-ack.
SYNC_ACK_PREFIX = "E2EE-SYNC-ACK-V1"
#: Domain-separation prefix for the signature-authorized batch group
#: sync-ack.
GROUP_SYNC_ACK_PREFIX = "E2EE-GROUP-SYNC-ACK-V1"
#: Domain-separation prefix for the signature-authorized single-message
#: delivery ack.
MESSAGE_ACK_PREFIX = "E2EE-MESSAGE-ACK-V1"
#: Length of a derived one-to-one session key in bytes.
SESSION_KEY_BYTES = 32
#: Fixed HKDF salt for session-key derivation: 32 zero bytes.
SESSION_KEY_SALT = b"\x00" * SESSION_KEY_BYTES
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


def message_envelope_aad(session_id: str, sender_device_id: str,
                         message_id: str, sequence: int) -> bytes:
    """Build the AAD bytes that bind a ciphertext to its message envelope.

    The document is the domain prefix ``E2EE-MESSAGE-ENVELOPE-V1``, one
    newline, then compact JSON of the four fields with keys sorted
    (``message_id``, ``sender_device_id``, ``sequence``, ``session_id``) and
    Unicode written as-is. The string values are used exactly as given — no
    trimming or normalization — and ``sequence`` is serialized as a JSON
    integer.
    """
    document = json.dumps(
        {"message_id": message_id, "sender_device_id": sender_device_id,
         "sequence": sequence, "session_id": session_id},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return (MESSAGE_ENVELOPE_PREFIX + "\n" + document).encode("utf-8")


def _encode_utf8_strict(value: str, field: str) -> bytes:
    """Encode *value* as strict UTF-8; raise :class:`CryptoError` on failure.

    A string that cannot be encoded (e.g. one carrying lone surrogates) is
    reported under *field* without exposing the raw encoding exception, and
    the value is never silently repaired or truncated.
    """
    try:
        return value.encode("utf-8")
    except UnicodeEncodeError:
        raise CryptoError(f"field must be encodable as UTF-8: {field}",
                          field) from None


def _validate_envelope_metadata(sender_device_id: object,
                                message_id: object,
                                sequence: object) -> None:
    """Validate the envelope metadata group once any field is present.

    Reports the first missing or invalid field in the fixed order
    ``sender_device_id``, ``message_id``, ``sequence``.
    """
    if not is_nonempty_string(sender_device_id):
        raise CryptoError(
            "field must be a non-empty string: sender_device_id",
            "sender_device_id")
    if not is_nonempty_string(message_id):
        raise CryptoError("field must be a non-empty string: message_id",
                          "message_id")
    if isinstance(sequence, bool) or not isinstance(sequence, int) \
            or sequence < 1:
        raise CryptoError("field must be a positive integer: sequence",
                          "sequence")


def _message_aad(session_id: str, sender_device_id: Optional[str],
                 message_id: Optional[str],
                 sequence: Optional[int]) -> bytes:
    """Pick the AAD for one message: legacy session-only or full envelope.

    With all three metadata fields omitted the legacy AAD (the session id's
    UTF-8 bytes) is used; otherwise the metadata group is validated and the
    envelope AAD is built. There is no mixing: exactly one AAD is tried.

    A text field that cannot be encoded as strict UTF-8 raises
    :class:`CryptoError`; the first unencodable field is reported in the
    order ``session_id``, ``sender_device_id``, ``message_id``.
    """
    if sender_device_id is None and message_id is None and sequence is None:
        return _encode_utf8_strict(session_id, "session_id")
    _validate_envelope_metadata(sender_device_id, message_id, sequence)
    _encode_utf8_strict(session_id, "session_id")
    _encode_utf8_strict(sender_device_id, "sender_device_id")
    _encode_utf8_strict(message_id, "message_id")
    return message_envelope_aad(session_id, sender_device_id, message_id,
                                sequence)


def encrypt_message(session_id: str, key_b64: str,
                    plaintext: str,
                    sender_device_id: Optional[str] = None,
                    message_id: Optional[str] = None,
                    sequence: Optional[int] = None) -> dict:
    """Encrypt *plaintext* (UTF-8) with AES-256-GCM.

    The 12-byte nonce is generated randomly; the session id is bound to the
    ciphertext as additional authenticated data. Returns a dict with
    ``session_id``, ``nonce`` and ``ciphertext`` (both base64; the ciphertext
    carries the 16-byte GCM tag appended).

    When *sender_device_id*, *message_id* and *sequence* are all given, the
    AAD instead becomes the message-envelope document (see
    :func:`message_envelope_aad`) binding those values to the ciphertext.
    The three fields form a group: supplying only some of them, empty
    strings, wrong types, or a non-positive/boolean sequence raises
    :class:`CryptoError` naming the first offending field.

    A text field that cannot be encoded as strict UTF-8 (e.g. one carrying
    lone surrogates) raises :class:`CryptoError` naming the first such field
    in the order ``session_id``, ``sender_device_id``, ``message_id``,
    ``plaintext``; the value is never repaired or truncated.
    """
    if not is_nonempty_string(session_id):
        raise CryptoError("field must be a non-empty string: session_id",
                          "session_id")
    if not isinstance(plaintext, str):
        raise CryptoError("field must be a string: plaintext", "plaintext")
    cipher = _load_aes_key(key_b64)
    aad = _message_aad(session_id, sender_device_id, message_id, sequence)
    nonce = os.urandom(GCM_NONCE_BYTES)
    ciphertext = cipher.encrypt(nonce, _encode_utf8_strict(plaintext,
                                                           "plaintext"), aad)
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
    offending field. The optional envelope metadata selects the same AAD
    mode as encryption; a mode or metadata mismatch fails authentication
    with ``field=ciphertext`` — the other mode is never retried. A text
    field that cannot be encoded as strict UTF-8 raises :class:`CryptoError`
    naming the first such field in the order ``session_id``,
    ``sender_device_id``, ``message_id``.
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

    aad = _message_aad(session_id, sender_device_id, message_id, sequence)
    try:
        plaintext = cipher.decrypt(nonce, ciphertext, aad)
    except InvalidTag:
        raise CryptoError("ciphertext failed authentication", "ciphertext") \
            from None
    try:
        text = plaintext.decode("utf-8")
    except UnicodeDecodeError:
        raise CryptoError("plaintext is not valid UTF-8", "plaintext") \
            from None
    return {"session_id": session_id, "plaintext": text}


def _decode_private_key_bytes(value: object) -> bytes:
    """Decode a canonical standard-base64 raw 32-byte private key.

    Raises :class:`CryptoError` with ``field=private_key`` unless *value* is
    a non-empty string of canonical standard base64 (correct alphabet and
    padding, re-encoding reproduces the input) that decodes to exactly 32
    bytes.
    """
    if not is_nonempty_string(value):
        raise CryptoError("field must be a non-empty string: private_key",
                          "private_key")
    raw = _decode_strict_base64(value, "private_key")
    if len(raw) != 32:
        raise CryptoError(
            f"private_key must decode to 32 bytes (got {len(raw)})",
            "private_key")
    if base64.b64encode(raw).decode("ascii") != value:
        raise CryptoError("private_key must be canonical standard base64",
                          "private_key")
    return raw


def _load_x25519_peer_public_key(value: object) -> x25519.X25519PublicKey:
    """Parse the peer public key strictly as an X25519 public key.

    Accepts the same encodings as :func:`load_public_key` (PEM text,
    base64/hex DER SubjectPublicKeyInfo, or base64/hex raw 32-byte point),
    but only an X25519 key is accepted: a raw 32-byte point is interpreted
    as X25519, while a key carrying an Ed25519 or any other algorithm
    identifier is rejected with ``field=peer_public_key``.
    """
    if not is_nonempty_string(value):
        raise CryptoError("field must be a non-empty string: peer_public_key",
                          "peer_public_key")
    key = load_public_key(value)
    if not isinstance(key, x25519.X25519PublicKey):
        raise CryptoError(
            "peer_public_key must be an X25519 public key",
            "peer_public_key")
    return key


def derive_session_key(session_id: str, private_key: str,
                       peer_public_key: str) -> dict:
    """Derive the shared one-to-one session key purely locally.

    Both parties run the same protocol: the X25519 shared secret between
    *private_key* (canonical standard-base64 raw 32-byte private key) and
    *peer_public_key* (any encoding accepted by :func:`load_public_key`,
    X25519 only) feeds HKDF-SHA256 with a 32-zero-byte salt and an info of
    the ``E2EE-SESSION-KEY-V1`` prefix, one newline and *session_id*'s
    UTF-8 bytes, producing 32 bytes. The initiator passes its ephemeral
    private key and the snapshot's ``public_key``; the recipient passes the
    corresponding pre-key private key and the snapshot's ``ephemeral_key``,
    and both obtain the same key.

    Returns a dict with ``session_id`` and ``key`` (standard base64 of the
    32-byte key, ready for :func:`encrypt_message`/:func:`decrypt_message`).
    Nothing is persisted and no server is contacted. Invalid inputs raise
    :class:`CryptoError` naming the first offending field in the order
    ``session_id``, ``private_key``, ``peer_public_key``; a failed exchange
    or an all-zero shared secret is reported as ``peer_public_key``.
    """
    if not is_nonempty_string(session_id):
        raise CryptoError("field must be a non-empty string: session_id",
                          "session_id")
    try:
        info = (SESSION_KEY_INFO_PREFIX + "\n" + session_id).encode("utf-8")
    except UnicodeEncodeError:
        raise CryptoError("session_id must be encodable as UTF-8",
                          "session_id") from None

    private_bytes = _decode_private_key_bytes(private_key)
    peer_key = _load_x25519_peer_public_key(peer_public_key)

    private = x25519.X25519PrivateKey.from_private_bytes(private_bytes)
    try:
        shared = private.exchange(peer_key)
    except ValueError:
        # X25519 rejects an all-zero shared secret (low-order peer point).
        raise CryptoError(
            "peer_public_key does not yield a usable shared secret",
            "peer_public_key") from None

    key = HKDF(algorithm=hashes.SHA256(), length=SESSION_KEY_BYTES,
               salt=SESSION_KEY_SALT, info=info).derive(shared)
    return {
        "session_id": session_id,
        "key": base64.b64encode(key).decode("ascii"),
    }


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


def sign_prekey_proof(user_id: object, device_id: object, key_id: object,
                      public_key: object, private_key: object) -> dict:
    """Create a signed pre-key proof purely locally from an identity key.

    The local counterpart of :func:`verify_prekey_proof`: the Ed25519
    identity key in *private_key* signs the public
    ``E2EE-SIGNED-PREKEY-V1`` message (see
    :func:`signed_prekey_proof_message`) over the four literal field values
    *user_id*, *device_id*, *key_id* and *public_key*, so the result is the
    same six-string shape a proof query returns
    (``user_id``/``device_id``/``key_id``/``public_key``/``identity_key``/
    ``signature``) and is accepted by :func:`verify_prekey_proof` and by the
    verified registration / verified pre-key replenishment entries.

    The identifiers and *public_key* are copied through exactly as given —
    no trimming or normalization, so Chinese characters, slashes and leading
    or trailing spaces are preserved byte-for-byte. *public_key* must parse
    under :func:`load_public_key` (the same range the publication entries
    accept). *private_key* must be canonical standard base64 (correct
    alphabet and padding, re-encoding reproduces the input — no whitespace
    or URL-safe alphabet) of exactly the raw 32-byte Ed25519 private key
    seed; PEM/DER/PKCS8 or any other spelling is rejected. The seed never
    appears in the result: ``identity_key`` is the canonical standard base64
    of the corresponding raw 32-byte Ed25519 public point and ``signature``
    is canonical standard base64 of the deterministic 64-byte signature, so
    identical inputs always return an identical object.

    No server is contacted, no backend state is read or written, and
    nothing is persisted. Every failure raises :class:`CryptoError`
    reporting the first offending input in the order *user_id*,
    *device_id*, *key_id*, *public_key*, *private_key* — an empty/wrong-type
    value or a text value that cannot be encoded as UTF-8 names that
    identifier, an unparsable public key names ``public_key`` and a
    non-canonical or wrong-length private key names ``private_key``; no
    underlying exception or the private value is exposed.
    """
    for name, value in (("user_id", user_id), ("device_id", device_id),
                        ("key_id", key_id), ("public_key", public_key)):
        if not is_nonempty_string(value):
            raise CryptoError(f"field must be a non-empty string: {name}",
                              name)
        # A string carrying lone surrogates cannot sign: the proof message is
        # UTF-8 JSON of the literal values. The failure keeps the field name.
        _encode_utf8_strict(value, name)
    if load_public_key(public_key) is None:
        raise CryptoError("public_key must be a public key", "public_key")
    private_bytes = _decode_private_key_bytes(private_key)
    try:
        identity = ed25519.Ed25519PrivateKey.from_private_bytes(private_bytes)
        message = signed_prekey_proof_message(
            user_id, device_id, key_id, public_key)
        signature = identity.sign(message)
        identity_raw = identity.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    except (ValueError, UnsupportedAlgorithm, TypeError):
        # Defensive: the seed already decoded to exactly 32 bytes, which the
        # Ed25519 loader accepts; keep any platform quirk inside the contract.
        raise CryptoError("private_key is not a usable Ed25519 private key",
                          "private_key") from None
    return {
        "user_id": user_id,
        "device_id": device_id,
        "key_id": key_id,
        "public_key": public_key,
        "identity_key": base64.b64encode(identity_raw).decode("ascii"),
        "signature": base64.b64encode(signature).decode("ascii"),
    }


def identity_rotation_proof_message(user_id: str, device_id: str,
                                    identity_key: str,
                                    expected_version: int) -> bytes:
    """Build the exact bytes the current identity key signs to authorize a rotation.

    The message is the domain prefix ``E2EE-IDENTITY-ROTATION-V1``, one
    newline, then compact JSON of the four fields with keys sorted
    (``device_id``, ``expected_version``, ``identity_key``, ``user_id``) and
    Unicode written as-is. ``user_id`` is the device's registered value; the
    other strings are the request's original strings, and
    ``expected_version`` is serialized as a JSON integer.
    """
    document = json.dumps(
        {"device_id": device_id, "expected_version": expected_version,
         "identity_key": identity_key, "user_id": user_id},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return (IDENTITY_ROTATION_PREFIX + "\n" + document).encode("utf-8")


def verify_identity_rotation(identity_key: ed25519.Ed25519PublicKey,
                             signature: bytes, user_id: str, device_id: str,
                             new_identity_key: str,
                             expected_version: int) -> bool:
    """Verify one identity-rotation authorization against the Ed25519 key.

    Returns ``True`` iff *signature* is valid over
    :func:`identity_rotation_proof_message` for the four field values.
    """
    message = identity_rotation_proof_message(
        user_id, device_id, new_identity_key, expected_version)
    try:
        identity_key.verify(signature, message)
    except InvalidSignature:
        return False
    return True


def device_revocation_proof_message(user_id: str, device_id: str,
                                    expected_version: int) -> bytes:
    """Build the exact bytes the current identity key signs to revoke a device.

    The message is the domain prefix ``E2EE-DEVICE-REVOCATION-V1``, one
    newline, then compact JSON of the three fields with keys sorted
    (``device_id``, ``expected_version``, ``user_id``) and Unicode written
    as-is. ``user_id`` is the device's registered value and *device_id* is
    the request's path-decoded value; the strings are the stored/requested
    original strings — no trimming or normalization — and
    *expected_version* is serialized as a JSON integer.
    """
    document = json.dumps(
        {"device_id": device_id, "expected_version": expected_version,
         "user_id": user_id},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return (DEVICE_REVOCATION_PREFIX + "\n" + document).encode("utf-8")


def verify_device_revocation(identity_key: ed25519.Ed25519PublicKey,
                             signature: bytes, user_id: str, device_id: str,
                             expected_version: int) -> bool:
    """Verify one signature-authorized device revocation against the Ed25519 key.

    Returns ``True`` iff *signature* is valid over
    :func:`device_revocation_proof_message` for the three field values.
    """
    message = device_revocation_proof_message(
        user_id, device_id, expected_version)
    try:
        identity_key.verify(signature, message)
    except InvalidSignature:
        return False
    return True


def group_membership_proof_message(group_id: str, operation: str,
                                   actor_device_id: str, device_id: str,
                                   expected_revision: int) -> bytes:
    """Build the exact bytes the creator's identity key signs for a member change.

    The message is the domain prefix ``E2EE-GROUP-MEMBERSHIP-V1``, one
    newline, then compact JSON of the five fields with keys sorted
    (``actor_device_id``, ``device_id``, ``expected_revision``, ``group_id``,
    ``operation``) and Unicode written as-is. ``group_id`` is the
    request's path-decoded value and the other strings are the request's
    original strings — no trimming or normalization — and
    ``expected_revision`` is serialized as a JSON integer.
    """
    document = json.dumps(
        {"actor_device_id": actor_device_id, "device_id": device_id,
         "expected_revision": expected_revision, "group_id": group_id,
         "operation": operation},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return (GROUP_MEMBERSHIP_PREFIX + "\n" + document).encode("utf-8")


def verify_group_membership(identity_key: ed25519.Ed25519PublicKey,
                            signature: bytes, group_id: str, operation: str,
                            actor_device_id: str, device_id: str,
                            expected_revision: int) -> bool:
    """Verify one signature-authorized membership change against the Ed25519 key.

    Returns ``True`` iff *signature* is valid over
    :func:`group_membership_proof_message` for the five field values.
    """
    message = group_membership_proof_message(
        group_id, operation, actor_device_id, device_id, expected_revision)
    try:
        identity_key.verify(signature, message)
    except InvalidSignature:
        return False
    return True


def group_session_rotation_proof_message(
        user_id: str, group_id: str, predecessor_session_id: str,
        rotation_id: str, actor_device_id: str, ephemeral_key: str,
        expected_revision: int, expected_version: int) -> bytes:
    """Build the exact bytes the creator's identity key signs for a rotation.

    The message is the domain prefix ``E2EE-GROUP-SESSION-ROTATION-V1``, one
    newline, then compact JSON of the eight fields with keys sorted
    (``actor_device_id``, ``ephemeral_key``, ``expected_revision``,
    ``expected_version``, ``group_id``, ``predecessor_session_id``,
    ``rotation_id``, ``user_id``) and Unicode written as-is. ``user_id``,
    ``group_id`` and ``predecessor_session_id`` are the stored values; the
    other strings are the request's original strings — no trimming or
    normalization — and ``expected_revision`` / ``expected_version`` are
    serialized as JSON integers.
    """
    document = json.dumps(
        {"actor_device_id": actor_device_id, "ephemeral_key": ephemeral_key,
         "expected_revision": expected_revision,
         "expected_version": expected_version, "group_id": group_id,
         "predecessor_session_id": predecessor_session_id,
         "rotation_id": rotation_id, "user_id": user_id},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return (GROUP_SESSION_ROTATION_PREFIX + "\n" + document).encode("utf-8")


def verify_group_session_rotation(
        identity_key: ed25519.Ed25519PublicKey, signature: bytes,
        user_id: str, group_id: str, predecessor_session_id: str,
        rotation_id: str, actor_device_id: str, ephemeral_key: str,
        expected_revision: int, expected_version: int) -> bool:
    """Verify one signature-authorized group-session rotation.

    Returns ``True`` iff *signature* is valid over
    :func:`group_session_rotation_proof_message` for the eight field values.
    """
    message = group_session_rotation_proof_message(
        user_id, group_id, predecessor_session_id, rotation_id,
        actor_device_id, ephemeral_key, expected_revision, expected_version)
    try:
        identity_key.verify(signature, message)
    except InvalidSignature:
        return False
    return True


def sync_ack_proof_message(user_id: str, device_id: str,
                           expected_version: int,
                           items: List[Tuple[str, int]]) -> bytes:
    """Build the exact bytes the identity key signs for a batch sync-ack.

    The message is the domain prefix ``E2EE-SYNC-ACK-V1``, one newline,
    then compact JSON of the four fields with keys sorted at every level
    (``device_id``, ``expected_version``, ``items``, ``user_id``) and
    Unicode written as-is. ``user_id`` is the device's registered value,
    ``device_id`` the request's path-decoded value and
    ``expected_version`` is serialized as a JSON integer; *items* are the
    already-validated ``(session_id, cursor)`` pairs in request order —
    each contributes exactly its verbatim ``session_id`` string and
    integer ``cursor``, and the array order is preserved.
    """
    document = json.dumps(
        {"device_id": device_id, "expected_version": expected_version,
         "items": [{"session_id": session_id, "cursor": cursor}
                   for session_id, cursor in items],
         "user_id": user_id},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return (SYNC_ACK_PREFIX + "\n" + document).encode("utf-8")


def verify_sync_ack(identity_key: ed25519.Ed25519PublicKey,
                    signature: bytes, user_id: str, device_id: str,
                    expected_version: int,
                    items: List[Tuple[str, int]]) -> bool:
    """Verify one signature-authorized batch sync-ack against the Ed25519 key.

    Returns ``True`` iff *signature* is valid over
    :func:`sync_ack_proof_message` for the four field values.
    """
    message = sync_ack_proof_message(
        user_id, device_id, expected_version, items)
    try:
        identity_key.verify(signature, message)
    except InvalidSignature:
        return False
    return True


def group_sync_ack_proof_message(user_id: str, device_id: str,
                                 expected_version: int,
                                 items: List[Tuple[str, int]]) -> bytes:
    """Build the exact bytes the identity key signs for a group sync-ack.

    The message is the domain prefix ``E2EE-GROUP-SYNC-ACK-V1``, one
    newline, then compact JSON of the four fields with keys sorted at every
    level (``device_id``, ``expected_version``, ``items``, ``user_id``) and
    Unicode written as-is. ``user_id`` is the device's registered value,
    ``device_id`` the request's path-decoded value and ``expected_version``
    is serialized as a JSON integer; *items* are the already-validated
    ``(session_id, cursor)`` pairs in request order — each contributes
    exactly its verbatim ``session_id`` string and integer ``cursor``, and
    the array order is preserved.
    """
    document = json.dumps(
        {"device_id": device_id, "expected_version": expected_version,
         "items": [{"session_id": session_id, "cursor": cursor}
                   for session_id, cursor in items],
         "user_id": user_id},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return (GROUP_SYNC_ACK_PREFIX + "\n" + document).encode("utf-8")


def verify_group_sync_ack(identity_key: ed25519.Ed25519PublicKey,
                          signature: bytes, user_id: str, device_id: str,
                          expected_version: int,
                          items: List[Tuple[str, int]]) -> bool:
    """Verify one signature-authorized batch group sync-ack.

    Returns ``True`` iff *signature* is valid over
    :func:`group_sync_ack_proof_message` for the four field values.
    """
    message = group_sync_ack_proof_message(
        user_id, device_id, expected_version, items)
    try:
        identity_key.verify(signature, message)
    except InvalidSignature:
        return False
    return True


def message_ack_proof_message(user_id: str, device_id: str, message_id: str,
                              sequence: int, session_id: str,
                              expected_version: int) -> bytes:
    """Build the exact bytes the identity key signs for a verified ack.

    The message is the domain prefix ``E2EE-MESSAGE-ACK-V1``, one newline,
    then compact JSON of the six fields with keys sorted at every level
    (``device_id``, ``expected_version``, ``message_id``, ``sequence``,
    ``session_id``, ``user_id``) and Unicode written as-is. ``user_id`` is
    the device's registered value, ``session_id`` the request's
    path-decoded value, ``device_id``/``message_id`` the request's verbatim
    strings and ``sequence``/``expected_version`` are serialized as JSON
    integers.
    """
    document = json.dumps(
        {"device_id": device_id, "expected_version": expected_version,
         "message_id": message_id, "sequence": sequence,
         "session_id": session_id, "user_id": user_id},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return (MESSAGE_ACK_PREFIX + "\n" + document).encode("utf-8")


def verify_message_ack(identity_key: ed25519.Ed25519PublicKey,
                       signature: bytes, user_id: str, device_id: str,
                       message_id: str, sequence: int, session_id: str,
                       expected_version: int) -> bool:
    """Verify one signature-authorized message ack against the Ed25519 key.

    Returns ``True`` iff *signature* is valid over
    :func:`message_ack_proof_message` for the six field values.
    """
    message = message_ack_proof_message(
        user_id, device_id, message_id, sequence, session_id,
        expected_version)
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


def verify_prekey_proof(proof: object, user_id: object, device_id: object,
                        key_id: object,
                        expected_fingerprint: object) -> dict:
    """Verify one published signed pre-key proof fully offline.

    *proof* is the frozen six-string proof object (``user_id``,
    ``device_id``, ``key_id``, ``public_key``, ``identity_key``,
    ``signature``) as returned by the proof query; extra fields are ignored
    and the input mapping is never modified. *user_id*, *device_id* and
    *key_id* are the expected identifiers: each is compared against the
    proof's own string exactly as given — no trimming of spaces and no
    normalization of Chinese characters or slashes. *expected_fingerprint*
    is the trusted 64-lowercase-hex fingerprint of the identity key.

    The check is purely local and historical: the proof is verified against
    the identity public key frozen inside it (the key that authorized the
    pre-key at publication time), so an old proof still verifies with the
    old trusted fingerprint after an identity rotation, while the new
    identity's fingerprint simply does not match. Consumption or revocation
    of the pre-key does not change this offline verdict.

    The identity key must be an Ed25519 public key in any encoding accepted
    by :func:`load_ed25519_public_key`; the pre-key public key must parse
    under :func:`load_public_key`. The fingerprint follows
    :func:`identity_fingerprint`, so an equivalent re-encoding of the same
    identity key still matches; the signature, however, signs the literal
    ``public_key`` string, so re-encoding the pre-key public key without
    re-signing fails verification. The fingerprint must equal the proof's
    frozen identity key's fingerprint before the signature is checked.

    On success returns a new dict with the proof's six original string
    values plus ``fingerprint``. Any failure raises :class:`CryptoError`:
    a non-object proof is ``field=proof``; a missing/wrong-type/empty
    required string, an invalid public key, a bad fingerprint format or an
    ownership mismatch names the corresponding field; a fingerprint
    mismatch is ``expected_fingerprint``; a non-canonical-base64 or
    non-64-byte signature and a failed verification are ``signature``.
    """
    if not isinstance(proof, dict):
        raise CryptoError("proof must be a JSON object", "proof")
    for name, value in (("user_id", user_id), ("device_id", device_id),
                        ("key_id", key_id)):
        if not is_nonempty_string(value):
            raise CryptoError(f"field must be a non-empty string: {name}",
                              name)
    if not is_nonempty_string(expected_fingerprint):
        raise CryptoError(
            "field must be a non-empty string: expected_fingerprint",
            "expected_fingerprint")
    if not is_fingerprint_format(expected_fingerprint):
        raise CryptoError(
            "expected_fingerprint must be 64 lowercase hexadecimal "
            "characters", "expected_fingerprint")
    for name in ("user_id", "device_id", "key_id", "public_key",
                 "identity_key", "signature"):
        if not is_nonempty_string(proof.get(name)):
            raise CryptoError(f"field must be a non-empty string: {name}",
                              name)
    for name, expected in (("user_id", user_id), ("device_id", device_id),
                           ("key_id", key_id)):
        if proof[name] != expected:
            raise CryptoError(
                f"proof does not match the expected {name}", name)
    identity_key = load_ed25519_public_key(proof["identity_key"])
    if identity_key is None:
        raise CryptoError("identity_key must be an Ed25519 public key",
                          "identity_key")
    if load_public_key(proof["public_key"]) is None:
        raise CryptoError("public_key must be a public key", "public_key")
    fingerprint = identity_fingerprint(proof["identity_key"])
    if fingerprint != expected_fingerprint:
        raise CryptoError(
            "expected_fingerprint does not match the proof's identity key",
            "expected_fingerprint")
    signature = decode_ed25519_signature(proof["signature"])
    if signature is None:
        raise CryptoError(
            "signature must be canonical standard base64 of a 64-byte "
            "Ed25519 signature", "signature")
    if not verify_signed_prekey(identity_key, signature, proof["user_id"],
                                proof["device_id"], proof["key_id"],
                                proof["public_key"]):
        raise CryptoError(
            "signed pre-key proof failed verification: signature",
            "signature")
    return {
        "user_id": proof["user_id"],
        "device_id": proof["device_id"],
        "key_id": proof["key_id"],
        "public_key": proof["public_key"],
        "identity_key": proof["identity_key"],
        "signature": proof["signature"],
        "fingerprint": fingerprint,
    }


#: Required fields of a committed ciphertext envelope, in the order their
#: validation errors are reported: the same six fields message submission
#: freezes (``session_id``, ``sender_device_id``, ``message_id``,
#: ``sequence``, ``nonce``, ``ciphertext``).
_SIGNED_MESSAGE_ENVELOPE_FIELDS = (
    "session_id", "sender_device_id", "message_id", "sequence", "nonce",
    "ciphertext")


def _decode_canonical_base64_bytes(value: object, field: str, *,
                                   exact: Optional[int] = None,
                                   minimum: Optional[int] = None) -> bytes:
    """Decode canonical standard base64 with an exact or minimum length.

    Raises :class:`CryptoError` under *field* unless *value* is a non-empty
    string of canonical standard base64 (correct alphabet and padding;
    re-encoding the decoded bytes reproduces the input verbatim — no
    whitespace, URL-safe alphabet or non-canonical trailing bits) whose
    decoded length equals *exact* bytes or is at least *minimum* bytes.
    """
    if not is_nonempty_string(value):
        raise CryptoError(f"field must be a non-empty string: {field}", field)
    raw = _decode_strict_base64(value, field)
    if exact is not None and len(raw) != exact:
        raise CryptoError(
            f"{field} must decode to {exact} bytes (got {len(raw)})", field)
    if minimum is not None and len(raw) < minimum:
        raise CryptoError(
            f"{field} must decode to at least {minimum} bytes "
            f"(got {len(raw)})", field)
    if base64.b64encode(raw).decode("ascii") != value:
        raise CryptoError(f"{field} must be canonical standard base64", field)
    return raw


def _signed_message_bytes(fields: dict) -> bytes:
    """Build the exact bytes signed for one six-field ciphertext envelope.

    The document is the domain prefix ``E2EE-SIGNED-MESSAGE-V1``, one
    newline, then compact JSON of the six envelope fields with keys sorted
    (``ciphertext``, ``message_id``, ``nonce``, ``sender_device_id``,
    ``sequence``, ``session_id``) and Unicode written as-is; ``sequence`` is
    serialized as a JSON integer. The string values are the envelope's
    literal values — no trimming or normalization.
    """
    document = json.dumps(
        {name: fields[name] for name in _SIGNED_MESSAGE_ENVELOPE_FIELDS},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return (SIGNED_MESSAGE_PREFIX + "\n" + document).encode("utf-8")


def _validate_signed_envelope_fields(envelope: dict) -> dict:
    """Validate and copy the six signed envelope fields.

    Reports the first missing or invalid field in the fixed order
    ``session_id``, ``sender_device_id``, ``message_id``, ``sequence``,
    ``nonce``, ``ciphertext``: the three identifiers are non-empty strings
    encodable as strict UTF-8 (spaces, Chinese characters and slashes kept
    verbatim), ``sequence`` is a positive integer that is not a boolean,
    ``nonce`` is canonical standard base64 of exactly 12 bytes and
    ``ciphertext`` is canonical standard base64 of at least 16 bytes.
    Returns a fresh dict holding the six original values; the input mapping
    is never read-modified.
    """
    for name in ("session_id", "sender_device_id", "message_id"):
        value = envelope.get(name)
        if not is_nonempty_string(value):
            raise CryptoError(
                f"field must be a non-empty string: {name}", name)
        _encode_utf8_strict(value, name)
    sequence = envelope.get("sequence")
    if isinstance(sequence, bool) or not isinstance(sequence, int) \
            or sequence < 1:
        raise CryptoError("field must be a positive integer: sequence",
                          "sequence")
    nonce = envelope.get("nonce")
    _decode_canonical_base64_bytes(
        nonce, "nonce", exact=GCM_NONCE_BYTES)
    ciphertext = envelope.get("ciphertext")
    _decode_canonical_base64_bytes(
        ciphertext, "ciphertext", minimum=GCM_TAG_BYTES)
    return {
        "session_id": envelope["session_id"],
        "sender_device_id": envelope["sender_device_id"],
        "message_id": envelope["message_id"],
        "sequence": sequence,
        "nonce": nonce,
        "ciphertext": ciphertext,
    }


def sign_message(envelope: object, private_key: object) -> dict:
    """Sign one committed ciphertext envelope purely offline with an Ed25519 key.

    *envelope* is the same six-field object a message submission freezes:
    ``session_id``, ``sender_device_id``, ``message_id``, ``sequence``,
    ``nonce`` and ``ciphertext``; extra fields are ignored and the input
    mapping is never modified. *private_key* is the canonical
    standard-base64 raw 32-byte Ed25519 private key seed (same encoding as
    :func:`sign_prekey_proof`); PEM/DER/PKCS#8 spellings are rejected and
    the seed never appears in the result.

    The signature is the deterministic Ed25519 signature over the public
    ``E2EE-SIGNED-MESSAGE-V1`` message: the domain prefix, one newline, then
    compact JSON of the six fields with keys sorted and Unicode written as-is
    (``sequence`` serialized as a JSON integer). Identifier strings are used
    exactly as given — spaces, Chinese characters and slashes are preserved
    byte-for-byte; ``sequence`` must be a positive integer (booleans are
    rejected), ``nonce`` canonical standard base64 of exactly 12 bytes and
    ``ciphertext`` canonical standard base64 of at least 16 bytes.

    Returns a new dict with the six envelope values verbatim plus
    ``identity_key`` (canonical standard base64 of the raw 32-byte Ed25519
    public point derived from the seed) and ``signature`` (canonical
    standard base64 of the 64-byte signature); identical inputs always
    return an identical object, directly accepted by
    :func:`verify_message`. No server is contacted, no backend state is read
    or written, and nothing is persisted. Every failure raises
    :class:`CryptoError`: a non-object envelope is ``field=envelope``; a
    missing/wrong-type/empty field, a bad ``sequence`` or non-canonical
    base64 names that field; private-key errors are ``field=private_key``.
    """
    if not isinstance(envelope, dict):
        raise CryptoError("envelope must be a JSON object", "envelope")
    fields = _validate_signed_envelope_fields(envelope)
    private_bytes = _decode_private_key_bytes(private_key)
    message = _signed_message_bytes(fields)
    try:
        identity = ed25519.Ed25519PrivateKey.from_private_bytes(private_bytes)
        signature = identity.sign(message)
        identity_raw = identity.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    except (ValueError, UnsupportedAlgorithm, TypeError):
        # Defensive: the seed already decoded to exactly 32 bytes, which the
        # Ed25519 loader accepts; keep any platform quirk inside the contract.
        raise CryptoError("private_key is not a usable Ed25519 private key",
                          "private_key") from None
    result = {name: fields[name] for name in _SIGNED_MESSAGE_ENVELOPE_FIELDS}
    result["identity_key"] = base64.b64encode(identity_raw).decode("ascii")
    result["signature"] = base64.b64encode(signature).decode("ascii")
    return result


def parse_signed_message(envelope: object) -> dict:
    """Validate the eight fields of a :func:`sign_message` object structurally.

    This is the request-shape check shared by the verified message-submission
    entry: it runs exactly the structure/UTF-8/encoding rules of
    :func:`verify_message` without the expected-session, expected-device and
    trusted-fingerprint comparisons (the server supplies those itself, from
    live state). *envelope* must be an object carrying the six committed
    envelope fields plus non-empty ``identity_key`` and ``signature``
    strings; the three identifiers are non-empty strict-UTF-8 strings kept
    verbatim, ``sequence`` a positive non-boolean integer, ``nonce``
    canonical standard base64 of exactly 12 bytes, ``ciphertext`` canonical
    standard base64 of at least 16 bytes, ``identity_key`` an Ed25519 public
    key in any encoding accepted by :func:`load_ed25519_public_key`, and
    ``signature`` canonical standard base64 of exactly 64 bytes. The
    signature itself is *not* verified here.

    Returns a fresh dict holding the eight original string/integer values
    verbatim (the input mapping is never modified); extra fields are
    ignored. Every failure raises :class:`CryptoError` naming the first
    offending field, exactly as :func:`verify_message` would
    (``envelope`` for a non-object).
    """
    if not isinstance(envelope, dict):
        raise CryptoError("envelope must be a JSON object", "envelope")
    fields = _validate_signed_envelope_fields(envelope)
    identity_value = envelope.get("identity_key")
    signature_value = envelope.get("signature")
    if not is_nonempty_string(identity_value):
        raise CryptoError(
            "field must be a non-empty string: identity_key", "identity_key")
    if not is_nonempty_string(signature_value):
        raise CryptoError(
            "field must be a non-empty string: signature", "signature")
    if load_ed25519_public_key(identity_value) is None:
        raise CryptoError("identity_key must be an Ed25519 public key",
                          "identity_key")
    if decode_ed25519_signature(signature_value) is None:
        raise CryptoError(
            "signature must be canonical standard base64 of a 64-byte "
            "Ed25519 signature", "signature")
    result = {name: fields[name] for name in _SIGNED_MESSAGE_ENVELOPE_FIELDS}
    result["identity_key"] = identity_value
    result["signature"] = signature_value
    return result


def verify_message_signature(identity_key: "ed25519.Ed25519PublicKey",
                             signature: bytes, fields: dict) -> bool:
    """Verify an Ed25519 signature over one six-field signed envelope.

    Returns ``True`` iff *signature* is valid over
    :func:`_signed_message_bytes` for the six envelope values in *fields*.
    """
    try:
        identity_key.verify(signature, _signed_message_bytes(fields))
    except InvalidSignature:
        return False
    return True


def same_ed25519_public_key(left: "ed25519.Ed25519PublicKey",
                            right: "ed25519.Ed25519PublicKey") -> bool:
    """Return whether two parsed Ed25519 public keys name the same point."""
    return left.public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw) == \
        right.public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw)


def verify_message(envelope: object, expected_session_id: object,
                   expected_sender_device_id: object,
                   expected_fingerprint: object) -> dict:
    """Verify one signed ciphertext envelope fully offline.

    *envelope* is the frozen eight-field object returned by
    :func:`sign_message`: the six committed envelope fields
    (``session_id``, ``sender_device_id``, ``message_id``, ``sequence``,
    ``nonce``, ``ciphertext``) plus ``identity_key`` and ``signature``;
    extra fields are ignored and the input mapping is never modified.
    *expected_session_id* and *expected_sender_device_id* are compared
    against the envelope's own strings exactly as given — no trimming of
    spaces and no normalization of Chinese characters or slashes.
    *expected_fingerprint* is the trusted 64-lowercase-hex fingerprint of
    the sender identity key, computed with :func:`identity_fingerprint`.

    The check is purely local and historical: the envelope is verified
    against the identity public key frozen inside it, so an old envelope
    still verifies with the old trusted fingerprint after an identity
    rotation or revocation, while a new identity's fingerprint simply does
    not match. The identity key must be an Ed25519 public key in any
    encoding accepted by :func:`load_ed25519_public_key`; an equivalent
    re-encoding of the same key still passes because the fingerprint
    canonicalizes the key and the signature covers only the six envelope
    fields, not the key spelling. The expected identifiers and the
    fingerprint must match before the signature is checked; changing any
    signed field fails verification.

    On success returns a new dict with the eight original field values
    (each string preserved verbatim, including the ``identity_key``
    spelling) plus ``fingerprint``. Any failure raises :class:`CryptoError`
    and no partial result is returned: a non-object envelope is
    ``field=envelope``; a missing/wrong-type/empty signed field, an invalid
    ``sequence`` or non-canonical base64 names that field; an invalid or
    mismatching expected identifier is ``session_id`` or
    ``sender_device_id``; a fingerprint that is not 64 lowercase hex
    characters or does not match is ``expected_fingerprint``; an
    unparsable public key is ``identity_key``; a non-canonical-base64 or
    non-64-byte signature and a failed verification are ``signature``.
    """
    if not isinstance(envelope, dict):
        raise CryptoError("envelope must be a JSON object", "envelope")
    if not is_nonempty_string(expected_session_id):
        raise CryptoError(
            "field must be a non-empty string: session_id", "session_id")
    if not is_nonempty_string(expected_sender_device_id):
        raise CryptoError(
            "field must be a non-empty string: sender_device_id",
            "sender_device_id")
    if not is_nonempty_string(expected_fingerprint):
        raise CryptoError(
            "field must be a non-empty string: expected_fingerprint",
            "expected_fingerprint")
    if not is_fingerprint_format(expected_fingerprint):
        raise CryptoError(
            "expected_fingerprint must be 64 lowercase hexadecimal "
            "characters", "expected_fingerprint")

    fields = _validate_signed_envelope_fields(envelope)
    for name in ("identity_key", "signature"):
        if not is_nonempty_string(envelope.get(name)):
            raise CryptoError(
                f"field must be a non-empty string: {name}", name)

    if fields["session_id"] != expected_session_id:
        raise CryptoError(
            "envelope does not match the expected session_id", "session_id")
    if fields["sender_device_id"] != expected_sender_device_id:
        raise CryptoError(
            "envelope does not match the expected sender_device_id",
            "sender_device_id")

    identity_key = load_ed25519_public_key(envelope["identity_key"])
    if identity_key is None:
        raise CryptoError("identity_key must be an Ed25519 public key",
                          "identity_key")
    fingerprint = identity_fingerprint(envelope["identity_key"])
    if fingerprint != expected_fingerprint:
        raise CryptoError(
            "expected_fingerprint does not match the envelope's identity key",
            "expected_fingerprint")

    signature = decode_ed25519_signature(envelope["signature"])
    if signature is None:
        raise CryptoError(
            "signature must be canonical standard base64 of a 64-byte "
            "Ed25519 signature", "signature")
    try:
        identity_key.verify(signature, _signed_message_bytes(fields))
    except InvalidSignature:
        raise CryptoError(
            "signed message failed verification: signature",
            "signature") from None

    result = {name: envelope[name]
              for name in (*_SIGNED_MESSAGE_ENVELOPE_FIELDS,
                           "identity_key", "signature")}
    result["fingerprint"] = fingerprint
    return result


#: Required fields of the public eight-field one-to-one session snapshot, in
#: the order their validation errors are reported.
_SESSION_SNAPSHOT_FIELDS = (
    "session_id", "initiator_device_id", "recipient_device_id", "prekey_id",
    "ephemeral_key", "identity_key", "public_key", "created_at")

#: The two roles accepted by ``derive_verified_session_key``: the initiator
#: derives with its ephemeral private key, the recipient with the pre-key
#: private key; both end up with the same session key.
_ROLE_INITIATOR = "initiator"
_ROLE_RECIPIENT = "recipient"

#: Sentinel for an omitted ``role``: omitting the parameter selects the
#: initiator, while an explicit ``None`` (or any other non-string value) is
#: rejected as ``field=role``.
_ROLE_OMITTED = object()


def derive_verified_session_key(session: object, proof: object,
                                private_key: object, user_id: object,
                                expected_fingerprint: object, *,
                                role: object = _ROLE_OMITTED) -> dict:
    """Derive a one-to-one session key from a verified pre-key proof.

    This is the purely local counterpart to :func:`derive_session_key`: it
    cross-checks a frozen public session snapshot (the eight fields returned
    by the public session query) against a frozen signed pre-key proof and a
    trusted identity fingerprint before deriving the same X25519+HKDF key the
    other party obtains from :func:`derive_session_key`. No server is
    contacted, no backend state is read or written, and neither input mapping
    is modified.

    *session* must be a JSON object whose eight snapshot fields
    (``session_id``, ``initiator_device_id``, ``recipient_device_id``,
    ``prekey_id``, ``ephemeral_key``, ``identity_key``, ``public_key``,
    ``created_at``) are all non-empty strings; extra fields are ignored.

    *proof* is the same frozen six-string object accepted by
    :func:`verify_prekey_proof`. Its ``device_id`` must equal the snapshot's
    ``recipient_device_id`` and its ``key_id``/``public_key`` must equal the
    snapshot's ``prekey_id``/``public_key`` verbatim. The proof's
    ``identity_key`` need not equal the snapshot's ``identity_key`` string
    (encodings may differ): both must parse as Ed25519 keys and name the same
    actual key. The proof itself — ownership of *user_id*, fingerprint of
    *expected_fingerprint* and signature — is verified with exactly the rules
    of :func:`verify_prekey_proof`; hence an old proof with a matching old
    snapshot still derives under the old trusted fingerprint after an identity
    rotation, pre-key consumption or revocation. Both roles name the
    recipient's proof: *user_id* and *expected_fingerprint* always describe
    the recipient.

    *role* selects which private key *private_key* carries. Omitted or the
    literal string ``"initiator"``, it is the initiator's ephemeral private
    key and its public point must equal the snapshot's ``ephemeral_key``.
    The literal string ``"recipient"`` makes it the recipient's pre-key
    private key, whose public point must equal the snapshot's ``public_key``.
    Any other value — an explicit ``None`` or empty string, a non-string, or
    a different string — raises :class:`CryptoError` with ``field=role``.
    In both roles *private_key* uses the same canonical standard-base64 raw
    32-byte encoding as :func:`derive_session_key`, and both the ephemeral
    key and the pre-key ``public_key`` must be X25519 (a raw 32-byte point is
    interpreted as X25519 under the existing rules; an algorithm-identified
    Ed25519 key is rejected).

    Returns a new dict with only ``session_id`` and ``key`` (the same values
    :func:`derive_session_key` returns for the snapshot's ``session_id``, the
    role's private key and the other party's public key). Every failure
    raises :class:`CryptoError`: a non-object snapshot is ``field=session``
    and a missing/wrong-type/empty snapshot field names that field; proof
    verification errors keep the field names used by
    :func:`verify_prekey_proof` (mismatches with the snapshot's device,
    pre-key id and pre-key public key are reported as ``device_id``,
    ``key_id`` and ``public_key``); an identity-key parse failure or an
    identity key that does not name the same actual key is ``identity_key``;
    private-key encoding errors are ``private_key``; an illegal or
    mismatching ephemeral public key is ``ephemeral_key``; a non-X25519
    pre-key is ``public_key``. For the initiator a private key that does not
    match the snapshot's ``ephemeral_key`` is ``ephemeral_key`` and a failed
    X25519 exchange with the pre-key is ``public_key``; for the recipient a
    private key that does not match the snapshot's ``public_key`` is
    ``public_key`` and a failed X25519 exchange with the ephemeral key (or an
    all-zero shared secret) is ``ephemeral_key``.
    """
    # 0) The role selects the rest of the flow; only the two literal strings
    #    are accepted, with an omitted role meaning the initiator. The role
    #    is normalized to the module constant so the branch below can use
    #    identity comparison.
    if role is _ROLE_OMITTED or role == _ROLE_INITIATOR:
        role = _ROLE_INITIATOR
    elif role == _ROLE_RECIPIENT:
        role = _ROLE_RECIPIENT
    else:
        raise CryptoError(
            'role must be "initiator" or "recipient"', "role")

    # 1) Snapshot shape: object first, then each required non-empty string.
    if not isinstance(session, dict):
        raise CryptoError("session must be a JSON object", "session")
    for name in _SESSION_SNAPSHOT_FIELDS:
        if not is_nonempty_string(session.get(name)):
            raise CryptoError(
                f"field must be a non-empty string: {name}", name)

    # 2) The proof verifies ownership, fingerprint and signature exactly as
    #    the standalone offline check does. The snapshot's recipient device
    #    is the expected device; mismatches keep their proof field names.
    verify_prekey_proof(proof, user_id, session["recipient_device_id"],
                        session["prekey_id"], expected_fingerprint)
    assert isinstance(proof, dict)  # guaranteed by verify_prekey_proof

    # 3) Frozen cross-checks against the snapshot. device_id/key_id/public_key
    #    must be byte-for-byte the same published strings; identity keys are
    #    compared as actual keys, so equivalent encodings are accepted.
    if proof["public_key"] != session["public_key"]:
        raise CryptoError(
            "public_key does not match the session snapshot", "public_key")
    snapshot_identity = load_ed25519_public_key(session["identity_key"])
    if snapshot_identity is None:
        raise CryptoError("identity_key must be an Ed25519 public key",
                          "identity_key")
    proof_identity = load_ed25519_public_key(proof["identity_key"])
    assert proof_identity is not None  # verify_prekey_proof accepted it
    if snapshot_identity.public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw) != proof_identity.public_bytes(
                serialization.Encoding.Raw, serialization.PublicFormat.Raw):
        raise CryptoError(
            "identity_key does not match the session snapshot",
            "identity_key")

    # 4) The private key must be valid; which snapshot public key it must
    #    match, and which one it exchanges with, depends on the role. The
    #    initiator's ephemeral private key matches the snapshot's
    #    ephemeral_key and exchanges with the pre-key; the recipient's
    #    pre-key private key matches the snapshot's public_key and exchanges
    #    with the ephemeral key. Public points are compared as actual keys,
    #    so equivalent encodings of the same key are accepted.
    private_bytes = _decode_private_key_bytes(private_key)
    private = x25519.X25519PrivateKey.from_private_bytes(private_bytes)
    private_public = private.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    if role is _ROLE_INITIATOR:
        own_field, peer_field = "ephemeral_key", "public_key"
    else:
        own_field, peer_field = "public_key", "ephemeral_key"
    own = _load_x25519_peer_public_key_for(session[own_field], own_field)
    if private_public != own.public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw):
        raise CryptoError(
            f"private_key does not match the session's {own_field}",
            own_field)
    peer = _load_x25519_peer_public_key_for(session[peer_field], peer_field)

    # 5) Run the existing X25519+HKDF construction with the snapshot's
    #    literal session id string so both roles — and both parties — obtain
    #    a key byte-identical to the derive_session_key result. A failed
    #    exchange (e.g. a low-order peer point, which X25519 rejects as an
    #    all-zero shared secret) is reported under the peer key's field.
    try:
        info = (SESSION_KEY_INFO_PREFIX + "\n"
                + session["session_id"]).encode("utf-8")
    except UnicodeEncodeError:
        raise CryptoError("session_id must be encodable as UTF-8",
                          "session_id") from None
    try:
        shared = private.exchange(peer)
    except ValueError:
        raise CryptoError(
            f"{peer_field} does not yield a usable shared secret",
            peer_field) from None
    key = HKDF(algorithm=hashes.SHA256(), length=SESSION_KEY_BYTES,
               salt=SESSION_KEY_SALT, info=info).derive(shared)
    return {
        "session_id": session["session_id"],
        "key": base64.b64encode(key).decode("ascii"),
    }


def _load_x25519_peer_public_key_for(value: Any, field: str):
    """Parse *value* strictly as X25519, reporting errors under *field*.

    Mirrors :func:`_load_x25519_peer_public_key` but lets the caller name the
    offending snapshot field (``ephemeral_key`` or ``public_key``) instead of
    the generic ``peer_public_key``.
    """
    if not is_nonempty_string(value):
        raise CryptoError(f"field must be a non-empty string: {field}", field)
    key = load_public_key(value)
    if not isinstance(key, x25519.X25519PublicKey):
        raise CryptoError(f"{field} must be an X25519 public key", field)
    return key
