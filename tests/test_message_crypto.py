"""Tests for the local AES-256-GCM encrypt/decrypt helpers and CLI commands."""
import base64
import json
import os
import subprocess
import sys
import unittest

from e2ee_backend.crypto import (CryptoError, decrypt_message, encrypt_message,
                                 message_envelope_aad)


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


class MessageEnvelopeCryptoTest(unittest.TestCase):
    """Tests for the optional message-envelope metadata AAD mode."""

    def _encrypt(self, key: str, plaintext: str = "hello",
                 session_id: str = "sess-1", sender: str = "dev-1",
                 message_id: str = "msg-1", sequence: int = 1) -> dict:
        return encrypt_message(session_id, key, plaintext,
                               sender_device_id=sender, message_id=message_id,
                               sequence=sequence)

    def test_envelope_aad_document_format(self) -> None:
        aad = message_envelope_aad("sess-1", "dev-1", "msg-1", 7)
        self.assertEqual(
            aad,
            b'E2EE-MESSAGE-ENVELOPE-V1\n'
            b'{"message_id":"msg-1","sender_device_id":"dev-1",'
            b'"sequence":7,"session_id":"sess-1"}')

    def test_envelope_aad_keeps_unicode_and_escapes_strings(self) -> None:
        aad = message_envelope_aad("会话", 'dev"1\n', "m", 2)
        self.assertEqual(
            aad.decode("utf-8"),
            'E2EE-MESSAGE-ENVELOPE-V1\n'
            '{"message_id":"m","sender_device_id":"dev\\"1\\n",'
            '"sequence":2,"session_id":"会话"}')

    def test_roundtrip_output_shape_matches_legacy(self) -> None:
        key = _key_b64()
        enc = self._encrypt(key)
        self.assertEqual(set(enc), {"session_id", "nonce", "ciphertext"})
        dec = decrypt_message("sess-1", key, enc["nonce"], enc["ciphertext"],
                              sender_device_id="dev-1", message_id="msg-1",
                              sequence=1)
        self.assertEqual(dec, {"session_id": "sess-1", "plaintext": "hello"})

    def test_roundtrip_empty_plaintext(self) -> None:
        key = _key_b64()
        enc = self._encrypt(key, plaintext="")
        dec = decrypt_message("sess-1", key, enc["nonce"], enc["ciphertext"],
                              sender_device_id="dev-1", message_id="msg-1",
                              sequence=1)
        self.assertEqual(dec["plaintext"], "")

    def test_roundtrip_chinese_plaintext_and_quoted_ids(self) -> None:
        key = _key_b64()
        sender = 'dev"ice\n一号'
        message_id = 'msg"\\"\n二号'
        enc = encrypt_message("会话-1", key, "你好，世界",
                              sender_device_id=sender, message_id=message_id,
                              sequence=42)
        dec = decrypt_message("会话-1", key, enc["nonce"], enc["ciphertext"],
                              sender_device_id=sender, message_id=message_id,
                              sequence=42)
        self.assertEqual(dec, {"session_id": "会话-1",
                               "plaintext": "你好，世界"})

    def test_each_metadata_change_fails_as_ciphertext(self) -> None:
        key = _key_b64()
        enc = self._encrypt(key)
        for overrides in ({"sender_device_id": "dev-2"},
                          {"message_id": "msg-2"},
                          {"sequence": 2}):
            kwargs = {"sender_device_id": "dev-1", "message_id": "msg-1",
                      "sequence": 1}
            kwargs.update(overrides)
            with self.assertRaises(CryptoError) as ctx:
                decrypt_message("sess-1", key, enc["nonce"],
                                enc["ciphertext"], **kwargs)
            self.assertEqual(ctx.exception.field, "ciphertext")

    def test_tampered_ciphertext_fails_in_envelope_mode(self) -> None:
        key = _key_b64()
        enc = self._encrypt(key)
        raw = bytearray(base64.b64decode(enc["ciphertext"]))
        raw[0] ^= 1
        with self.assertRaises(CryptoError) as ctx:
            decrypt_message("sess-1", key, enc["nonce"],
                            base64.b64encode(bytes(raw)).decode(),
                            sender_device_id="dev-1", message_id="msg-1",
                            sequence=1)
        self.assertEqual(ctx.exception.field, "ciphertext")

    def test_wrong_key_fails_in_envelope_mode(self) -> None:
        enc = self._encrypt(_key_b64())
        with self.assertRaises(CryptoError) as ctx:
            decrypt_message("sess-1", _key_b64(), enc["nonce"],
                            enc["ciphertext"], sender_device_id="dev-1",
                            message_id="msg-1", sequence=1)
        self.assertEqual(ctx.exception.field, "ciphertext")

    def test_cross_mode_decryption_fails_both_ways(self) -> None:
        key = _key_b64()
        legacy = encrypt_message("sess-1", key, "x")
        with self.assertRaises(CryptoError) as ctx:
            decrypt_message("sess-1", key, legacy["nonce"],
                            legacy["ciphertext"], sender_device_id="dev-1",
                            message_id="msg-1", sequence=1)
        self.assertEqual(ctx.exception.field, "ciphertext")

        enveloped = self._encrypt(key)
        with self.assertRaises(CryptoError) as ctx:
            decrypt_message("sess-1", key, enveloped["nonce"],
                            enveloped["ciphertext"])
        self.assertEqual(ctx.exception.field, "ciphertext")

    def test_metadata_group_is_all_or_none(self) -> None:
        key = _key_b64()
        enc = encrypt_message("s", key, "x")
        partials = [
            ({"sender_device_id": "d"}, "message_id"),
            ({"message_id": "m"}, "sender_device_id"),
            ({"sequence": 1}, "sender_device_id"),
            ({"sender_device_id": "d", "message_id": "m"}, "sequence"),
            ({"sender_device_id": "d", "sequence": 1}, "message_id"),
            ({"message_id": "m", "sequence": 1}, "sender_device_id"),
        ]
        for kwargs, field in partials:
            with self.assertRaises(CryptoError) as ctx:
                encrypt_message("s", key, "x", **kwargs)
            self.assertEqual(ctx.exception.field, field)
            with self.assertRaises(CryptoError) as ctx:
                decrypt_message("s", key, enc["nonce"], enc["ciphertext"],
                                **kwargs)
            self.assertEqual(ctx.exception.field, field)

    def test_metadata_empty_strings_and_wrong_types(self) -> None:
        key = _key_b64()
        cases = [
            ({"sender_device_id": "", "message_id": "m", "sequence": 1},
             "sender_device_id"),
            ({"sender_device_id": 1, "message_id": "m", "sequence": 1},
             "sender_device_id"),
            ({"sender_device_id": "d", "message_id": "", "sequence": 1},
             "message_id"),
            ({"sender_device_id": "d", "message_id": None, "sequence": 1},
             "message_id"),
        ]
        for kwargs, field in cases:
            with self.assertRaises(CryptoError) as ctx:
                encrypt_message("s", key, "x", **kwargs)
            self.assertEqual(ctx.exception.field, field)

    def test_sequence_must_be_a_positive_integer(self) -> None:
        key = _key_b64()
        for bad in (0, -1, True, False, 1.5, "1", None):
            with self.assertRaises(CryptoError) as ctx:
                encrypt_message("s", key, "x", sender_device_id="d",
                                message_id="m", sequence=bad)
            self.assertEqual(ctx.exception.field, "sequence",
                             f"sequence={bad!r}")
        # Boundary: 1 and a large integer are valid.
        for good in (1, 2**53):
            enc = encrypt_message("s", key, "x", sender_device_id="d",
                                  message_id="m", sequence=good)
            dec = decrypt_message("s", key, enc["nonce"], enc["ciphertext"],
                                  sender_device_id="d", message_id="m",
                                  sequence=good)
            self.assertEqual(dec["plaintext"], "x")

    def test_first_invalid_field_is_reported_in_order(self) -> None:
        key = _key_b64()
        with self.assertRaises(CryptoError) as ctx:
            encrypt_message("s", key, "x", sender_device_id="",
                            message_id="", sequence=0)
        self.assertEqual(ctx.exception.field, "sender_device_id")
        with self.assertRaises(CryptoError) as ctx:
            encrypt_message("s", key, "x", sender_device_id="d",
                            message_id="", sequence=0)
        self.assertEqual(ctx.exception.field, "message_id")


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


class MessageEnvelopeCLITest(unittest.TestCase):
    """CLI tests for the optional message-envelope metadata flags."""

    def _run(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "e2ee_backend", *arguments],
            capture_output=True, text=True, timeout=15)

    def _encrypt_envelope(self, key: str, *extra: str) -> dict:
        result = self._run("encrypt-message", "--session-id", "s1",
                           "--key", key, "--plaintext", "hello 世界",
                           "--sender-device-id", "dev-1",
                           "--message-id", "msg-1", "--sequence", "3", *extra)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_envelope_roundtrip_via_cli(self) -> None:
        key = _key_b64()
        enc = self._encrypt_envelope(key)
        self.assertEqual(set(enc), {"session_id", "nonce", "ciphertext"})
        self.assertEqual(enc["session_id"], "s1")

        result = self._run("decrypt-message", "--session-id", "s1",
                           "--key", key, "--nonce", enc["nonce"],
                           "--ciphertext", enc["ciphertext"],
                           "--sender-device-id", "dev-1",
                           "--message-id", "msg-1", "--sequence", "3")
        self.assertEqual(result.returncode, 0, result.stderr)
        line = result.stdout.strip()
        self.assertEqual(line.count("\n"), 0)
        self.assertEqual(json.loads(line),
                         {"session_id": "s1", "plaintext": "hello 世界"})

    def test_envelope_roundtrip_with_quoted_and_multiline_ids(self) -> None:
        key = _key_b64()
        sender = 'dev"ice\n1'
        message_id = 'm"\n2'
        enc = json.loads(self._run(
            "encrypt-message", "--session-id", "s1", "--key", key,
            "--plaintext", "", "--sender-device-id", sender,
            "--message-id", message_id, "--sequence", "1").stdout)
        result = self._run("decrypt-message", "--session-id", "s1",
                           "--key", key, "--nonce", enc["nonce"],
                           "--ciphertext", enc["ciphertext"],
                           "--sender-device-id", sender,
                           "--message-id", message_id, "--sequence", "1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["plaintext"], "")

    def test_cli_metadata_mismatch_exits_2_with_ciphertext_field(self) -> None:
        key = _key_b64()
        enc = self._encrypt_envelope(key)
        result = self._run("decrypt-message", "--session-id", "s1",
                           "--key", key, "--nonce", enc["nonce"],
                           "--ciphertext", enc["ciphertext"],
                           "--sender-device-id", "dev-2",
                           "--message-id", "msg-1", "--sequence", "3")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertNotIn("Traceback", result.stderr)
        error = json.loads(result.stderr)
        self.assertEqual(error["field"], "ciphertext")
        self.assertIn("message", error)

    def test_cli_cross_mode_decryption_fails(self) -> None:
        key = _key_b64()
        legacy = json.loads(self._run("encrypt-message", "--session-id", "s1",
                                      "--key", key, "--plaintext", "x").stdout)
        result = self._run("decrypt-message", "--session-id", "s1",
                           "--key", key, "--nonce", legacy["nonce"],
                           "--ciphertext", legacy["ciphertext"],
                           "--sender-device-id", "dev-1",
                           "--message-id", "msg-1", "--sequence", "1")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stderr)["field"], "ciphertext")

        enveloped = self._encrypt_envelope(key)
        result = self._run("decrypt-message", "--session-id", "s1",
                           "--key", key, "--nonce", enveloped["nonce"],
                           "--ciphertext", enveloped["ciphertext"])
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stderr)["field"], "ciphertext")

    def test_cli_invalid_sequence_uses_crypto_error_contract(self) -> None:
        key = _key_b64()
        for bad in ("abc", "0", "-2", "1.5"):
            result = self._run("encrypt-message", "--session-id", "s1",
                               "--key", key, "--plaintext", "x",
                               "--sender-device-id", "dev-1",
                               "--message-id", "msg-1", "--sequence", bad)
            self.assertEqual(result.returncode, 2, bad)
            self.assertEqual(result.stdout, "")
            self.assertNotIn("Traceback", result.stderr)
            error = json.loads(result.stderr)
            self.assertEqual(error["field"], "sequence", bad)
            self.assertIn("message", error)

    def test_cli_partial_metadata_reports_first_missing_field(self) -> None:
        key = _key_b64()
        cases = [
            (("--sender-device-id", "d"), "message_id"),
            (("--message-id", "m"), "sender_device_id"),
            (("--sequence", "1"), "sender_device_id"),
            (("--sender-device-id", "d", "--message-id", "m"), "sequence"),
        ]
        for flags, field in cases:
            result = self._run("encrypt-message", "--session-id", "s1",
                               "--key", key, "--plaintext", "x", *flags)
            self.assertEqual(result.returncode, 2, flags)
            self.assertEqual(result.stdout, "")
            self.assertNotIn("Traceback", result.stderr)
            self.assertEqual(json.loads(result.stderr)["field"], field)

    def test_cli_empty_metadata_string_is_rejected(self) -> None:
        key = _key_b64()
        result = self._run("encrypt-message", "--session-id", "s1",
                           "--key", key, "--plaintext", "x",
                           "--sender-device-id", "", "--message-id", "m",
                           "--sequence", "1")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stderr)["field"],
                         "sender_device_id")


if __name__ == "__main__":
    unittest.main()
