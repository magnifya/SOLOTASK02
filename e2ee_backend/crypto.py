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
from typing import Any, Optional

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
#: Domain-separation prefix for the identity-key fingerprint message.
IDENTITY_FINGERPRINT_PREFIX = "E2EE-IDENTITY-FINGERPRINT-V1"
#: Domain-separation prefix for the message-envelope AAD document.
MESSAGE_ENVELOPE_PREFIX = "E2EE-MESSAGE-ENVELOPE-V1"
#: Domain-separation prefix for the one-to-one session-key HKDF info.
SESSION_KEY_INFO_PREFIX = "E2EE-SESSION-KEY-V1"
#: Length of a derived one-to-one session key in bytes.
SESSION_KEY_BYTES = 32
#: Fixed HKDF salt for session-key derivation: 32 zero bytes.
SESSION_KEY_SALT = b"\x00" * SESSION_KEY_BYTES
#: Length of a fingerprint: SHA-256 rendered as lowercase hexadecimal.
IDENTITY_FINGERPRINT_HEX_LEN = 64
#: Length of an Ed25519 signature in bytes.
ED25519_SIGNATURE_BYTES = 64

#: ``derive_verified_session_key`` role naming the initiator (the default).
SESSION_ROLE_INITIATOR = "initiator"
#: ``derive_verified_session_key`` role naming the recipient.
SESSION_ROLE_RECIPIENT = "recipient"


class _RoleOmitted:
    """Sentinel for an omitted *role*: distinct from any explicit value."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<role omitted>"


#: Default for *role*: an omitted role derives as the initiator, while any
#: explicit value other than the two role strings is rejected.
_ROLE_OMITTED = _RoleOmitted()


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
    """
    if sender_device_id is None and message_id is None and sequence is None:
        return session_id.encode("utf-8")
    _validate_envelope_metadata(sender_device_id, message_id, sequence)
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
    """
    if not is_nonempty_string(session_id):
        raise CryptoError("field must be a non-empty string: session_id",
                          "session_id")
    if not isinstance(plaintext, str):
        raise CryptoError("field must be a string: plaintext", "plaintext")
    cipher = _load_aes_key(key_b64)
    aad = _message_aad(session_id, sender_device_id, message_id, sequence)
    nonce = os.urandom(GCM_NONCE_BYTES)
    ciphertext = cipher.encrypt(nonce, plaintext.encode("utf-8"), aad)
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
    with ``field=ciphertext`` — the other mode is never retried.
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


#: Required fields of the public eight-field one-to-one session snapshot, in
#: the order their validation errors are reported.
_SESSION_SNAPSHOT_FIELDS = (
    "session_id", "initiator_device_id", "recipient_device_id", "prekey_id",
    "ephemeral_key", "identity_key", "public_key", "created_at")


def derive_verified_session_key(session: object, proof: object,
                                private_key: object, user_id: object,
                                expected_fingerprint: object,
                                role: object = _ROLE_OMITTED) -> dict:
    """Derive a one-to-one session key from a verified pre-key proof.

    This is the purely local counterpart to :func:`derive_session_key`: it
    cross-checks a frozen public session snapshot (the eight fields returned
    by the public session query) against a frozen signed pre-key proof and a
    trusted identity fingerprint before deriving the same X25519+HKDF key the
    other party obtains from :func:`derive_session_key` with the matching
    private key. No server is contacted, no backend state is read or written,
    and neither input mapping is modified.

    *role* selects which side derives the key. Omitted or ``"initiator"``
    (the default), *private_key* is the initiator's ephemeral private key and
    the peer is the snapshot's pre-key ``public_key``; with ``"recipient"``,
    *private_key* is the recipient's pre-key private key and the peer is the
    snapshot's ``ephemeral_key``. Both roles yield the byte-identical key.
    Any other explicit value — a non-string, an empty string, ``None`` or an
    unknown string — raises :class:`CryptoError` with ``field=role``.

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
    rotation, pre-key consumption or revocation. Both roles verify the
    recipient's proof: *user_id* and *expected_fingerprint* always name the
    recipient side.

    *private_key* uses the same canonical standard-base64 raw 32-byte
    encoding as :func:`derive_session_key`. For the initiator its public
    point must equal the snapshot's ``ephemeral_key``; for the recipient it
    must name the same actual X25519 public key as the snapshot's
    ``public_key`` (equivalent encodings of that key are accepted). Both the
    ephemeral key and the pre-key ``public_key`` must be X25519 (a raw
    32-byte point is interpreted as X25519 under the existing rules; an
    algorithm-identified Ed25519 key is rejected).

    Returns a new dict with only ``session_id`` and ``key`` (standard base64
    of the 32-byte key, ready for :func:`encrypt_message`/
    :func:`decrypt_message`). Every failure raises :class:`CryptoError`: an
    invalid *role* is ``role``; a non-object snapshot is ``field=session``
    and a missing/wrong-type/empty snapshot field names that field; proof
    verification errors keep the field names used by
    :func:`verify_prekey_proof` (mismatches with the snapshot's device,
    pre-key id and pre-key public key are reported as ``device_id``,
    ``key_id`` and ``public_key``); an identity-key parse failure or an
    identity key that does not name the same actual key is ``identity_key``;
    private-key encoding errors are ``private_key``. For the initiator an
    illegal or mismatching ephemeral public key is ``ephemeral_key`` and a
    non-X25519 pre-key or a failed X25519 exchange is ``public_key``; for the
    recipient a non-X25519 or mismatching pre-key is ``public_key`` and an
    illegal ephemeral key, a failed exchange or an all-zero shared secret is
    ``ephemeral_key``.
    """
    if role is _ROLE_OMITTED or role == SESSION_ROLE_INITIATOR:
        is_recipient = False
    elif role == SESSION_ROLE_RECIPIENT:
        is_recipient = True
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

    # 4) The private key must be valid, and its public point must match the
    #    snapshot public key of the caller's own side: the ephemeral key for
    #    the initiator, the pre-key public_key for the recipient (X25519
    #    only, raw points follow the existing interpretation rules, and
    #    equivalent encodings of the same key compare equal).
    private_bytes = _decode_private_key_bytes(private_key)
    private = x25519.X25519PrivateKey.from_private_bytes(private_bytes)
    if is_recipient:
        own_field, peer_field = "public_key", "ephemeral_key"
    else:
        own_field, peer_field = "ephemeral_key", "public_key"
    own = _load_x25519_peer_public_key_for(session[own_field], own_field)
    if private.public_key().public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw) != own.public_bytes(
                serialization.Encoding.Raw, serialization.PublicFormat.Raw):
        raise CryptoError(
            f"private_key does not match the session's {own_field}",
            own_field)

    # 5) The peer public key must be X25519; then run the existing
    #    X25519+HKDF construction with the snapshot's literal session id
    #    string so the key is byte-identical to the other party's
    #    derive_session_key result. A failed exchange (e.g. a low-order peer
    #    point, which also covers an all-zero shared secret) is reported as
    #    the peer key's field.
    peer = _load_x25519_peer_public_key_for(session[peer_field], peer_field)
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
