"""Tests for public-key parsing."""
import base64
import unittest

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519, x25519

from e2ee_backend.crypto import is_nonempty_string, load_public_key


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


if __name__ == "__main__":
    unittest.main()
