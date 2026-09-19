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

from cryptography.exceptions import UnsupportedAlgorithm
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, x25519
from typing import Optional


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
