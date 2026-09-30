"""Tests for public-key parsing."""
import base64
import unittest

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519, x25519

from e2ee_backend.crypto import (
    decode_ed25519_signature,
    is_nonempty_string,
    load_ed25519_public_key,
    load_public_key,
    verify_ed25519,
)


def _raw_x25519_b64() -> str:
    key = x25519.X25519PrivateKey.generate().public_key()
    raw = key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return base64.b64encode(raw).decode()


class CryptoTest(unittest.TestCase):
    def test_nonempty_string(self) -> None:
        self.assertTrue(is_nonempty_string("a"))
        self.assertFalse(is_nonempty_string(""))
        self.assertFalse(is_nonempty_string(1))
        self.assertFalse(is_nonempty_string(None))

    def test_raw_x25519_base64_and_hex(self) -> None:
        encoded = _raw_x25519_b64()
        self.assertIsNotNone(load_public_key(encoded))
        raw = base64.b64decode(encoded)
        self.assertIsNotNone(load_public_key(raw.hex()))

    def test_pem_ed25519(self) -> None:
        key = ed25519.Ed25519PrivateKey.generate().public_key()
        pem = key.public_bytes(serialization.Encoding.PEM,
                               serialization.PublicFormat.SubjectPublicKeyInfo)
        self.assertIsNotNone(load_public_key(pem.decode("ascii")))

    def test_der_ec_key(self) -> None:
        key = ec.generate_private_key(ec.SECP256R1()).public_key()
        der = key.public_bytes(serialization.Encoding.DER,
                               serialization.PublicFormat.SubjectPublicKeyInfo)
        self.assertIsNotNone(load_public_key(base64.b64encode(der).decode()))
        self.assertIsNotNone(load_public_key(der.hex()))

    def test_garbage_and_empty(self) -> None:
        self.assertIsNone(load_public_key("not-a-key"))
        self.assertIsNone(load_public_key(""))
        self.assertIsNone(load_public_key("abcd"))  # 4 hex chars -> 2 bytes


def _spki(key, encoding):
    return key.public_bytes(encoding,
                            serialization.PublicFormat.SubjectPublicKeyInfo)


class Ed25519LoaderTest(unittest.TestCase):
    def test_raw_ed25519_point_base64_and_hex(self) -> None:
        key = ed25519.Ed25519PrivateKey.generate().public_key()
        raw = key.public_bytes(serialization.Encoding.Raw,
                               serialization.PublicFormat.Raw)
        b64 = base64.b64encode(raw).decode()
        self.assertIsInstance(load_ed25519_public_key(b64),
                              ed25519.Ed25519PublicKey)
        self.assertIsInstance(load_ed25519_public_key(raw.hex()),
                              ed25519.Ed25519PublicKey)

    def test_pem_and_der_ed25519(self) -> None:
        key = ed25519.Ed25519PrivateKey.generate().public_key()
        pem = _spki(key, serialization.Encoding.PEM).decode("ascii")
        der_b64 = base64.b64encode(
            _spki(key, serialization.Encoding.DER)).decode()
        self.assertIsInstance(load_ed25519_public_key(pem),
                              ed25519.Ed25519PublicKey)
        self.assertIsInstance(load_ed25519_public_key(der_b64),
                              ed25519.Ed25519PublicKey)

    def test_algorithm_marked_x25519_rejected(self) -> None:
        key = x25519.X25519PrivateKey.generate().public_key()
        pem = _spki(key, serialization.Encoding.PEM).decode("ascii")
        der_b64 = base64.b64encode(
            _spki(key, serialization.Encoding.DER)).decode()
        self.assertIsNone(load_ed25519_public_key(pem))
        self.assertIsNone(load_ed25519_public_key(der_b64))

    def test_ec_key_rejected(self) -> None:
        key = ec.generate_private_key(ec.SECP256R1()).public_key()
        der_b64 = base64.b64encode(
            _spki(key, serialization.Encoding.DER)).decode()
        self.assertIsNone(load_ed25519_public_key(der_b64))

    def test_garbage_empty_wrong_length(self) -> None:
        self.assertIsNone(load_ed25519_public_key("not-a-key"))
        self.assertIsNone(load_ed25519_public_key(""))
        self.assertIsNone(load_ed25519_public_key(
            base64.b64encode(b"\x00" * 31).decode()))


class Ed25519SignatureTest(unittest.TestCase):
    def setUp(self) -> None:
        self.private = ed25519.Ed25519PrivateKey.generate()
        raw = self.private.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        self.identity_b64 = base64.b64encode(raw).decode()
        self.message = b"E2EE-SIGNED-PREKEY-V1\n" + \
            b'{"x":1}'

    def _signature_b64(self):
        return base64.b64encode(self.private.sign(self.message)).decode()

    def test_decode_valid_signature(self) -> None:
        raw = decode_ed25519_signature(self._signature_b64())
        self.assertIsNotNone(raw)
        self.assertEqual(len(raw), 64)

    def test_decode_rejects_bad_inputs(self) -> None:
        valid = self._signature_b64()
        self.assertIsNone(decode_ed25519_signature(None))
        self.assertIsNone(decode_ed25519_signature(123))
        self.assertIsNone(decode_ed25519_signature(b"bytes"))
        self.assertIsNone(decode_ed25519_signature(""))
        self.assertIsNone(decode_ed25519_signature(valid[:-2] + "--"))
        self.assertIsNone(decode_ed25519_signature(valid.rstrip("=")))
        self.assertIsNone(decode_ed25519_signature(
            base64.b64encode(b"\x00" * 63).decode()))
        self.assertIsNone(decode_ed25519_signature(
            base64.b64encode(b"\x00" * 65).decode()))

    def test_verify_valid_signature(self) -> None:
        self.assertTrue(verify_ed25519(
            self.identity_b64, self.message, self._signature_b64()))

    def test_verify_rejects_tampered_message_or_signature(self) -> None:
        valid = self._signature_b64()
        self.assertFalse(verify_ed25519(
            self.identity_b64, self.message + b" ", valid))
        # Flip the first data character (a full quantum, so the bytes change).
        flipped = ("A" if valid[0] != "A" else "B") + valid[1:]
        self.assertFalse(verify_ed25519(
            self.identity_b64, self.message, flipped))

    def test_verify_rejects_other_identity_and_bad_key(self) -> None:
        other = ed25519.Ed25519PrivateKey.generate().public_key()
        other_b64 = base64.b64encode(other.public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw)) \
            .decode()
        self.assertFalse(verify_ed25519(
            other_b64, self.message, self._signature_b64()))
        self.assertFalse(verify_ed25519(
            "not-a-key", self.message, self._signature_b64()))
        self.assertFalse(verify_ed25519(
            self.identity_b64, self.message, "not-base64!!"))


if __name__ == "__main__":
    unittest.main()
