"""Tests for the local one-to-one session-key derivation helper and CLI."""
import base64
import json
import subprocess
import sys
import unittest

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, x25519

from e2ee_backend.crypto import (CryptoError, decrypt_message,
                                 derive_session_key, encrypt_message)


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _x25519_keypair() -> tuple:
    private = x25519.X25519PrivateKey.generate()
    public = private.public_key()
    return private, public


def _raw_public_b64(public) -> str:
    return _b64(public.public_bytes(serialization.Encoding.Raw,
                                    serialization.PublicFormat.Raw))


def _der_public_b64(public) -> str:
    return _b64(public.public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo))


class DeriveSessionKeyTest(unittest.TestCase):
    def setUp(self) -> None:
        # Initiator ephemeral key pair and recipient pre-key pair.
        self.eph_private, self.eph_public = _x25519_keypair()
        self.pre_private, self.pre_public = _x25519_keypair()
        self.eph_private_b64 = _b64(
            self.eph_private.private_bytes(
                serialization.Encoding.Raw,
                serialization.PrivateFormat.Raw,
                serialization.NoEncryption()))
        self.pre_private_b64 = _b64(
            self.pre_private.private_bytes(
                serialization.Encoding.Raw,
                serialization.PrivateFormat.Raw,
                serialization.NoEncryption()))
        self.eph_public_b64 = _raw_public_b64(self.eph_public)
        self.pre_public_b64 = _raw_public_b64(self.pre_public)

    def _initiator_key(self, session_id: str = "sess-1") -> dict:
        return derive_session_key(session_id, self.eph_private_b64,
                                  self.pre_public_b64)

    def _recipient_key(self, session_id: str = "sess-1") -> dict:
        return derive_session_key(session_id, self.pre_private_b64,
                                  self.eph_public_b64)

    def test_both_parties_derive_the_same_key(self) -> None:
        initiator = self._initiator_key()
        recipient = self._recipient_key()
        self.assertEqual(initiator, recipient)
        self.assertEqual(set(initiator), {"session_id", "key"})
        self.assertEqual(initiator["session_id"], "sess-1")
        self.assertEqual(len(base64.b64decode(initiator["key"])), 32)

    def test_derivation_is_deterministic(self) -> None:
        self.assertEqual(self._initiator_key(), self._initiator_key())

    def test_different_session_id_derives_a_different_key(self) -> None:
        self.assertNotEqual(self._initiator_key("sess-1")["key"],
                            self._initiator_key("sess-2")["key"])

    def test_session_id_is_used_verbatim(self) -> None:
        # No trimming or normalization: whitespace and Chinese are kept.
        padded = self._initiator_key(" 会话 ")
        trimmed = self._initiator_key("会话")
        self.assertNotEqual(padded["key"], trimmed["key"])
        self.assertEqual(padded["session_id"], " 会话 ")

    def test_derived_key_works_with_encrypt_decrypt(self) -> None:
        key = self._initiator_key()["key"]
        enc = encrypt_message("sess-1", key, "hello 世界")
        dec = decrypt_message("sess-1", key, enc["nonce"], enc["ciphertext"])
        self.assertEqual(dec["plaintext"], "hello 世界")
        # Envelope AAD mode works with the derived key too.
        enc = encrypt_message("sess-1", key, "hi", sender_device_id="d",
                              message_id="m", sequence=1)
        dec = decrypt_message("sess-1", key, enc["nonce"], enc["ciphertext"],
                              sender_device_id="d", message_id="m", sequence=1)
        self.assertEqual(dec["plaintext"], "hi")

    def test_peer_public_key_accepts_existing_encodings(self) -> None:
        expected = self._initiator_key()["key"]
        # DER SubjectPublicKeyInfo (base64 and hex) and PEM spellings.
        der = _der_public_b64(self.pre_public)
        self.assertEqual(
            derive_session_key("sess-1", self.eph_private_b64, der)["key"],
            expected)
        hex_der = base64.b64decode(der).hex()
        self.assertEqual(
            derive_session_key("sess-1", self.eph_private_b64, hex_der)["key"],
            expected)
        pem = self.pre_public.public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo).decode("ascii")
        self.assertEqual(
            derive_session_key("sess-1", self.eph_private_b64, pem)["key"],
            expected)
        # Hex of the raw 32-byte point is also an existing encoding.
        raw_hex = base64.b64decode(self.pre_public_b64).hex()
        self.assertEqual(
            derive_session_key("sess-1", self.eph_private_b64, raw_hex)["key"],
            expected)

    def test_session_id_must_be_a_nonempty_utf8_string(self) -> None:
        for bad in (None, "", 1, b"sess", "\ud800"):
            with self.assertRaises(CryptoError) as ctx:
                derive_session_key(bad, self.eph_private_b64,
                                   self.pre_public_b64)
            self.assertEqual(ctx.exception.field, "session_id", repr(bad))

    def test_private_key_must_be_canonical_base64_of_32_bytes(self) -> None:
        cases = [None, "", 1, "not base64!!!", _b64(b"short"), _b64(b"x" * 33)]
        # Non-canonical spelling: same bytes, different final character.
        canonical = _b64(b"\x01" * 32)
        flipped = canonical[:-2] + ("R" if canonical[-2] != "R" else "Q") + "="
        if base64.b64decode(flipped) == b"\x01" * 32:
            cases.append(flipped)
        for bad in cases:
            with self.assertRaises(CryptoError) as ctx:
                derive_session_key("sess-1", bad, self.pre_public_b64)
            self.assertEqual(ctx.exception.field, "private_key", repr(bad))

    def test_ed25519_peer_public_key_is_rejected(self) -> None:
        ed = ed25519.Ed25519PrivateKey.generate().public_key()
        # Only algorithm-identified spellings are rejected; a raw 32-byte
        # point is always interpreted as X25519 per the protocol.
        for spelling in (_der_public_b64(ed),
                         ed.public_bytes(
                             serialization.Encoding.PEM,
                             serialization.PublicFormat.SubjectPublicKeyInfo)
                         .decode("ascii")):
            with self.assertRaises(CryptoError) as ctx:
                derive_session_key("sess-1", self.eph_private_b64, spelling)
            self.assertEqual(ctx.exception.field, "peer_public_key")

    def test_unparseable_peer_public_key_is_rejected(self) -> None:
        for bad in (None, "", 1, "not a key", _b64(b"short")):
            with self.assertRaises(CryptoError) as ctx:
                derive_session_key("sess-1", self.eph_private_b64, bad)
            self.assertEqual(ctx.exception.field, "peer_public_key", repr(bad))

    def test_all_zero_shared_secret_is_reported_as_peer_public_key(self) -> None:
        # The all-zero X25519 public point yields an all-zero shared secret.
        low_order = _b64(b"\x00" * 32)
        with self.assertRaises(CryptoError) as ctx:
            derive_session_key("sess-1", self.eph_private_b64, low_order)
        self.assertEqual(ctx.exception.field, "peer_public_key")

    def test_first_invalid_field_is_reported_in_order(self) -> None:
        with self.assertRaises(CryptoError) as ctx:
            derive_session_key("", "bad", "bad")
        self.assertEqual(ctx.exception.field, "session_id")
        with self.assertRaises(CryptoError) as ctx:
            derive_session_key("sess-1", "bad", "bad")
        self.assertEqual(ctx.exception.field, "private_key")


class DeriveSessionKeyCLITest(unittest.TestCase):
    def _run(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "e2ee_backend", *arguments],
            capture_output=True, text=True, timeout=15)

    def setUp(self) -> None:
        eph_private, eph_public = _x25519_keypair()
        pre_private, pre_public = _x25519_keypair()
        self.eph_private_b64 = _b64(eph_private.private_bytes(
            serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
            serialization.NoEncryption()))
        self.pre_private_b64 = _b64(pre_private.private_bytes(
            serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
            serialization.NoEncryption()))
        self.eph_public_b64 = _raw_public_b64(eph_public)
        self.pre_public_b64 = _raw_public_b64(pre_public)

    def test_cli_both_parties_derive_the_same_key(self) -> None:
        initiator = self._run(
            "derive-session-key", "--session-id", "sess-1",
            "--private-key", self.eph_private_b64,
            "--peer-public-key", self.pre_public_b64)
        self.assertEqual(initiator.returncode, 0, initiator.stderr)
        self.assertEqual(initiator.stderr, "")
        line = initiator.stdout.strip()
        self.assertEqual(line.count("\n"), 0)
        recipient = self._run(
            "derive-session-key", "--session-id", "sess-1",
            "--private-key", self.pre_private_b64,
            "--peer-public-key", self.eph_public_b64)
        self.assertEqual(recipient.returncode, 0, recipient.stderr)
        self.assertEqual(json.loads(line), json.loads(recipient.stdout))
        self.assertEqual(set(json.loads(line)), {"session_id", "key"})

    def test_cli_missing_option_exits_2_with_json_error(self) -> None:
        cases = [
            (("--private-key", self.eph_private_b64,
              "--peer-public-key", self.pre_public_b64), "session_id"),
            (("--session-id", "sess-1",
              "--peer-public-key", self.pre_public_b64), "private_key"),
            (("--session-id", "sess-1",
              "--private-key", self.eph_private_b64), "peer_public_key"),
        ]
        for arguments, field in cases:
            result = self._run("derive-session-key", *arguments)
            self.assertEqual(result.returncode, 2, arguments)
            self.assertEqual(result.stdout, "")
            self.assertNotIn("Traceback", result.stderr)
            error = json.loads(result.stderr)
            self.assertEqual(error["field"], field)
            self.assertIn("message", error)

    def test_cli_invalid_inputs_exit_2_with_json_error(self) -> None:
        cases = [
            (("--session-id", "", "--private-key", self.eph_private_b64,
              "--peer-public-key", self.pre_public_b64), "session_id"),
            (("--session-id", "sess-1", "--private-key", "bad",
              "--peer-public-key", self.pre_public_b64), "private_key"),
            (("--session-id", "sess-1", "--private-key",
              self.eph_private_b64, "--peer-public-key", "bad"),
             "peer_public_key"),
        ]
        for arguments, field in cases:
            result = self._run("derive-session-key", *arguments)
            self.assertEqual(result.returncode, 2, arguments)
            self.assertEqual(result.stdout, "")
            self.assertNotIn("Traceback", result.stderr)
            self.assertEqual(json.loads(result.stderr)["field"], field)


if __name__ == "__main__":
    unittest.main()
