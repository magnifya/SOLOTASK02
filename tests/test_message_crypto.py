"""Tests for the local AES-256-GCM encrypt/decrypt helpers and CLI commands."""
import base64
import json
import os
import subprocess
import sys
import unittest

from e2ee_backend.crypto import CryptoError, decrypt_message, encrypt_message


def _key_b64() -> str:
    return base64.b64encode(os.urandom(32)).decode()


class MessageCryptoTest(unittest.TestCase):
    def test_roundtrip_restores_utf8_plaintext(self) -> None:
        key = _key_b64()
        enc = encrypt_message("sess-1", key, "héllo wörld 你好")
        self.assertEqual(set(enc), {"session_id", "nonce", "ciphertext"})
        self.assertEqual(enc["session_id"], "sess-1")
        self.assertEqual(len(base64.b64decode(enc["nonce"])), 12)
        # Ciphertext = plaintext bytes + 16-byte GCM tag.
        plaintext_bytes = "héllo wörld 你好".encode("utf-8")
        self.assertEqual(len(base64.b64decode(enc["ciphertext"])),
                         len(plaintext_bytes) + 16)

        dec = decrypt_message("sess-1", key, enc["nonce"], enc["ciphertext"])
        self.assertEqual(dec, {"session_id": "sess-1",
                               "plaintext": "héllo wörld 你好"})

    def test_nonces_are_random_per_encryption(self) -> None:
        key = _key_b64()
        first = encrypt_message("s", key, "same")
        second = encrypt_message("s", key, "same")
        self.assertNotEqual(first["nonce"], second["nonce"])
        self.assertNotEqual(first["ciphertext"], second["ciphertext"])

    def test_session_id_is_authenticated_as_aad(self) -> None:
        key = _key_b64()
        enc = encrypt_message("sess-1", key, "hello")
        with self.assertRaises(CryptoError) as ctx:
            decrypt_message("sess-2", key, enc["nonce"], enc["ciphertext"])
        self.assertEqual(ctx.exception.field, "ciphertext")

    def test_key_must_be_32_bytes_of_base64(self) -> None:
        with self.assertRaises(CryptoError) as ctx:
            encrypt_message("s", "not base64!!!", "x")
        self.assertEqual(ctx.exception.field, "key")
        with self.assertRaises(CryptoError) as ctx:
            encrypt_message("s", base64.b64encode(b"short").decode(), "x")
        self.assertEqual(ctx.exception.field, "key")

    def test_nonce_must_be_12_bytes_of_base64(self) -> None:
        key = _key_b64()
        enc = encrypt_message("s", key, "x")
        with self.assertRaises(CryptoError) as ctx:
            decrypt_message("s", key, "!!!", enc["ciphertext"])
        self.assertEqual(ctx.exception.field, "nonce")
        with self.assertRaises(CryptoError) as ctx:
            decrypt_message("s", key, base64.b64encode(b"short").decode(),
                            enc["ciphertext"])
        self.assertEqual(ctx.exception.field, "nonce")

    def test_ciphertext_must_decode_and_carry_a_tag(self) -> None:
        key = _key_b64()
        enc = encrypt_message("s", key, "x")
        with self.assertRaises(CryptoError) as ctx:
            decrypt_message("s", key, enc["nonce"], "!!!")
        self.assertEqual(ctx.exception.field, "ciphertext")
        with self.assertRaises(CryptoError) as ctx:
            decrypt_message("s", key, enc["nonce"],
                            base64.b64encode(b"tiny").decode())
        self.assertEqual(ctx.exception.field, "ciphertext")

    def test_tampered_ciphertext_fails_authentication(self) -> None:
        key = _key_b64()
        enc = encrypt_message("s", key, "x")
        raw = bytearray(base64.b64decode(enc["ciphertext"]))
        raw[0] ^= 1
        with self.assertRaises(CryptoError) as ctx:
            decrypt_message("s", key, enc["nonce"],
                            base64.b64encode(bytes(raw)).decode())
        self.assertEqual(ctx.exception.field, "ciphertext")

    def test_wrong_key_fails_authentication(self) -> None:
        enc = encrypt_message("s", _key_b64(), "x")
        with self.assertRaises(CryptoError) as ctx:
            decrypt_message("s", _key_b64(), enc["nonce"], enc["ciphertext"])
        self.assertEqual(ctx.exception.field, "ciphertext")


class MessageCryptoCLITest(unittest.TestCase):
    def _run(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "e2ee_backend", *arguments],
            capture_output=True, text=True, timeout=15)

    def test_encrypt_decrypt_roundtrip_via_cli(self) -> None:
        key = _key_b64()
        result = self._run("encrypt-message", "--session-id", "s1",
                           "--key", key, "--plaintext", "hello 世界")
        self.assertEqual(result.returncode, 0, result.stderr)
        line = result.stdout.strip()
        self.assertEqual(line.count("\n"), 0)
        enc = json.loads(line)
        self.assertEqual(set(enc), {"session_id", "nonce", "ciphertext"})

        result = self._run("decrypt-message", "--session-id", "s1",
                           "--key", key, "--nonce", enc["nonce"],
                           "--ciphertext", enc["ciphertext"])
        self.assertEqual(result.returncode, 0, result.stderr)
        dec = json.loads(result.stdout)
        self.assertEqual(dec, {"session_id": "s1", "plaintext": "hello 世界"})

    def test_encrypt_with_bad_key_exits_nonzero_with_json_error(self) -> None:
        result = self._run("encrypt-message", "--session-id", "s1",
                           "--key", "bad", "--plaintext", "x")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertNotIn("Traceback", result.stderr)
        error = json.loads(result.stderr)
        self.assertEqual(error["field"], "key")

    def test_decrypt_with_tampered_ciphertext_exits_nonzero(self) -> None:
        key = _key_b64()
        enc = json.loads(self._run("encrypt-message", "--session-id", "s1",
                                   "--key", key, "--plaintext", "x").stdout)
        raw = bytearray(base64.b64decode(enc["ciphertext"]))
        raw[-1] ^= 1
        result = self._run("decrypt-message", "--session-id", "s1",
                           "--key", key, "--nonce", enc["nonce"],
                           "--ciphertext",
                           base64.b64encode(bytes(raw)).decode())
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("Traceback", result.stderr)
        error = json.loads(result.stderr)
        self.assertEqual(error["field"], "ciphertext")


if __name__ == "__main__":
    unittest.main()
