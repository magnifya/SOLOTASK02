"""Tests for the local AES-256-GCM encrypt/decrypt helpers and CLI commands."""
import base64
import json
import os
import subprocess
import sys
import unittest

from e2ee_backend.crypto import (
    MESSAGE_ENVELOPE_PREFIX,
    CryptoError,
    _message_aad,
    decrypt_message,
    encrypt_message,
)


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
    def test_envelope_roundtrip_returns_same_fields_as_legacy(self) -> None:
        key = _key_b64()
        enc = encrypt_message("sess-1", key, "hello 你好",
                              "dev-1", "msg-1", 3)
        self.assertEqual(set(enc), {"session_id", "nonce", "ciphertext"})
        self.assertEqual(enc["session_id"], "sess-1")
        dec = decrypt_message("sess-1", key, enc["nonce"], enc["ciphertext"],
                              "dev-1", "msg-1", 3)
        self.assertEqual(dec, {"session_id": "sess-1",
                               "plaintext": "hello 你好"})

    def test_envelope_aad_is_prefix_newline_and_sorted_compact_json(self) -> None:
        aad = _message_aad("sess-1", ("dev-1", "msg-1", 3))
        document = (
            '{"message_id":"msg-1","sender_device_id":"dev-1",'
            '"sequence":3,"session_id":"sess-1"}')
        self.assertEqual(
            aad,
            (MESSAGE_ENVELOPE_PREFIX + "\n" + document).encode("utf-8"))

    def test_envelope_keeps_non_ascii_and_escapes_quotes_and_newlines(self) -> None:
        aad = _message_aad("会 话", ('设备 "A"', "标识\n第二行", 42))
        text = aad.decode("utf-8")
        self.assertTrue(text.startswith(MESSAGE_ENVELOPE_PREFIX + "\n"))
        document = text[len(MESSAGE_ENVELOPE_PREFIX) + 1:]
        self.assertIn('"sender_device_id":"设备 \\"A\\""', document)
        self.assertIn('"message_id":"标识\\n第二行"', document)
        self.assertIn('"session_id":"会 话"', document)

    def test_empty_plaintext_roundtrips(self) -> None:
        key = _key_b64()
        enc = encrypt_message("s", key, "", "d", "m", 1)
        self.assertEqual(len(base64.b64decode(enc["ciphertext"])), 16)
        dec = decrypt_message("s", key, enc["nonce"], enc["ciphertext"],
                              "d", "m", 1)
        self.assertEqual(dec["plaintext"], "")

    def test_special_characters_in_identifiers_roundtrip(self) -> None:
        key = _key_b64()
        sender = 'dev "quoted" \n line'
        message = "msg\t中文\\backslash"
        enc = encrypt_message("s", key, "p", sender, message, 9007199254740993)
        dec = decrypt_message("s", key, enc["nonce"], enc["ciphertext"],
                              sender, message, 9007199254740993)
        self.assertEqual(dec["plaintext"], "p")

    def test_changed_binding_value_fails_with_field_ciphertext(self) -> None:
        key = _key_b64()
        enc = encrypt_message("s", key, "x", "dev-1", "msg-1", 3)
        variants = [
            ("other session", dict(session_id="s2")),
            ("other sender", dict(sender_device_id="dev-2")),
            ("other message", dict(message_id="msg-2")),
            ("other sequence", dict(sequence=4)),
        ]
        for label, overrides in variants:
            with self.subTest(label), self.assertRaises(CryptoError) as ctx:
                decrypt_message(
                    overrides.get("session_id", "s"), key,
                    enc["nonce"], enc["ciphertext"],
                    overrides.get("sender_device_id", "dev-1"),
                    overrides.get("message_id", "msg-1"),
                    overrides.get("sequence", 3))
            self.assertEqual(ctx.exception.field, "ciphertext")

    def test_envelope_tampered_ciphertext_and_wrong_key_fail(self) -> None:
        key = _key_b64()
        enc = encrypt_message("s", key, "x", "d", "m", 1)
        raw = bytearray(base64.b64decode(enc["ciphertext"]))
        raw[0] ^= 1
        with self.assertRaises(CryptoError) as ctx:
            decrypt_message("s", key, enc["nonce"],
                            base64.b64encode(bytes(raw)).decode(),
                            "d", "m", 1)
        self.assertEqual(ctx.exception.field, "ciphertext")
        with self.assertRaises(CryptoError) as ctx:
            decrypt_message("s", _key_b64(), enc["nonce"],
                            enc["ciphertext"], "d", "m", 1)
        self.assertEqual(ctx.exception.field, "ciphertext")

    def test_cross_mode_decryption_fails_without_fallback(self) -> None:
        key = _key_b64()
        legacy = encrypt_message("s", key, "x")
        envelope = encrypt_message("s", key, "x", "d", "m", 1)
        # Envelope ciphertext decrypted with the legacy (session-only) AAD.
        with self.assertRaises(CryptoError) as ctx:
            decrypt_message("s", key, envelope["nonce"],
                            envelope["ciphertext"])
        self.assertEqual(ctx.exception.field, "ciphertext")
        # Legacy ciphertext decrypted with envelope AAD.
        with self.assertRaises(CryptoError) as ctx:
            decrypt_message("s", key, legacy["nonce"],
                            legacy["ciphertext"], "d", "m", 1)
        self.assertEqual(ctx.exception.field, "ciphertext")

    def test_envelope_metadata_does_not_change_output_fields(self) -> None:
        key = _key_b64()
        first = encrypt_message("s", key, "x", "d", "m", 1)
        second = encrypt_message("s", key, "x", "d", "m", 1)
        self.assertNotEqual(first["nonce"], second["nonce"])
        self.assertEqual(set(first), set(second))

    def test_partial_envelope_reports_first_missing_field_in_order(self) -> None:
        key = _key_b64()
        cases = [
            (dict(sender_device_id="d"), "message_id"),
            (dict(message_id="m"), "sender_device_id"),
            (dict(sequence=1), "sender_device_id"),
            (dict(sender_device_id="d", message_id="m"), "sequence"),
            (dict(sender_device_id="d", sequence=1), "message_id"),
            (dict(message_id="m", sequence=1), "sender_device_id"),
        ]
        for kwargs, field in cases:
            with self.subTest(kwargs):
                with self.assertRaises(CryptoError) as ctx:
                    encrypt_message("s", key, "x",
                                    kwargs.get("sender_device_id"),
                                    kwargs.get("message_id"),
                                    kwargs.get("sequence"))
                self.assertEqual(ctx.exception.field, field)

    def test_empty_or_typed_wrong_identifiers_report_own_field(self) -> None:
        key = _key_b64()
        bad_sender_values = ["", 1, 1.5, b"d", ["d"], {"d": 1}, True]
        for value in bad_sender_values:
            with self.subTest("sender", value=value):
                with self.assertRaises(CryptoError) as ctx:
                    encrypt_message("s", key, "x", value, "m", 1)
                self.assertEqual(ctx.exception.field, "sender_device_id")
        bad_message_values = ["", 2, None, b"m", ["m"], False]
        # ``None`` here means "omitted"; exercise it only while the sender is
        # present so the sender check passes and the message check is reached.
        for value in bad_message_values:
            if value is None:
                continue
            with self.subTest("message", value=value):
                with self.assertRaises(CryptoError) as ctx:
                    encrypt_message("s", key, "x", "d", value, 1)
                self.assertEqual(ctx.exception.field, "message_id")

    def test_sequence_must_be_positive_integer_and_not_bool(self) -> None:
        key = _key_b64()
        bad_sequences = [0, -1, 1.0, "1", True, False, None, 1j, [1]]
        for value in bad_sequences:
            if value is None:
                # Omitted sequence with both identifiers present.
                with self.assertRaises(CryptoError) as ctx:
                    encrypt_message("s", key, "x", "d", "m", None)
                self.assertEqual(ctx.exception.field, "sequence")
                continue
            with self.subTest(value=value):
                with self.assertRaises(CryptoError) as ctx:
                    encrypt_message("s", key, "x", "d", "m", value)
                self.assertEqual(ctx.exception.field, "sequence")

    def test_large_positive_integer_sequence_roundtrips(self) -> None:
        key = _key_b64()
        enc = encrypt_message("s", key, "x", "d", "m", 2 ** 63)
        dec = decrypt_message("s", key, enc["nonce"], enc["ciphertext"],
                              "d", "m", 2 ** 63)
        self.assertEqual(dec["plaintext"], "x")

    def test_validation_runs_for_decrypt_too(self) -> None:
        key = _key_b64()
        enc = encrypt_message("s", key, "x", "d", "m", 1)
        with self.assertRaises(CryptoError) as ctx:
            decrypt_message("s", key, enc["nonce"], enc["ciphertext"],
                            "d", "m", 0)
        self.assertEqual(ctx.exception.field, "sequence")
        with self.assertRaises(CryptoError) as ctx:
            decrypt_message("s", key, enc["nonce"], enc["ciphertext"],
                            "d", "", 1)
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

    def test_envelope_roundtrip_via_cli_with_unicode_and_punctuation(self) -> None:
        key = _key_b64()
        sender = '设备 "A"'
        message = "msg-1\nnext"
        result = self._run("encrypt-message", "--session-id", "s1",
                           "--key", key, "--plaintext", "中文\n明文",
                           "--sender-device-id", sender,
                           "--message-id", message, "--sequence", "042")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        enc = json.loads(result.stdout)
        self.assertEqual(set(enc), {"session_id", "nonce", "ciphertext"})

        result = self._run("decrypt-message", "--session-id", "s1",
                           "--key", key, "--nonce", enc["nonce"],
                           "--ciphertext", enc["ciphertext"],
                           "--sender-device-id", sender,
                           "--message-id", message, "--sequence", "42")
        self.assertEqual(result.returncode, 0, result.stderr)
        dec = json.loads(result.stdout)
        self.assertEqual(dec, {"session_id": "s1",
                               "plaintext": "中文\n明文"})

    def test_envelope_tampered_binding_fails_via_cli(self) -> None:
        key = _key_b64()
        enc = json.loads(self._run(
            "encrypt-message", "--session-id", "s1", "--key", key,
            "--plaintext", "x", "--sender-device-id", "d1",
            "--message-id", "m1", "--sequence", "1").stdout)
        result = self._run(
            "decrypt-message", "--session-id", "s1", "--key", key,
            "--nonce", enc["nonce"], "--ciphertext", enc["ciphertext"],
            "--sender-device-id", "d2", "--message-id", "m1",
            "--sequence", "1")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertEqual(result.stderr.count("\n"), 1)
        self.assertNotIn("Traceback", result.stderr)
        error = json.loads(result.stderr)
        self.assertEqual(error["field"], "ciphertext")

    def test_cross_mode_decryption_fails_via_cli(self) -> None:
        key = _key_b64()
        legacy = json.loads(self._run(
            "encrypt-message", "--session-id", "s1", "--key", key,
            "--plaintext", "x").stdout)
        envelope = json.loads(self._run(
            "encrypt-message", "--session-id", "s1", "--key", key,
            "--plaintext", "x", "--sender-device-id", "d",
            "--message-id", "m", "--sequence", "1").stdout)
        result = self._run(
            "decrypt-message", "--session-id", "s1", "--key", key,
            "--nonce", envelope["nonce"],
            "--ciphertext", envelope["ciphertext"])
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stderr)["field"], "ciphertext")
        result = self._run(
            "decrypt-message", "--session-id", "s1", "--key", key,
            "--nonce", legacy["nonce"], "--ciphertext", legacy["ciphertext"],
            "--sender-device-id", "d", "--message-id", "m",
            "--sequence", "1")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stderr)["field"], "ciphertext")

    def test_partial_envelope_flags_via_cli_report_fields_in_order(self) -> None:
        key = _key_b64()
        base = ["encrypt-message", "--session-id", "s1", "--key", key,
                "--plaintext", "x"]
        cases = [
            (["--sender-device-id", "d"], "message_id"),
            (["--message-id", "m"], "sender_device_id"),
            (["--sequence", "1"], "sender_device_id"),
            (["--sender-device-id", "d", "--message-id", "m"], "sequence"),
            (["--sender-device-id", "d", "--sequence", "1"], "message_id"),
            (["--message-id", "m", "--sequence", "1"],
             "sender_device_id"),
        ]
        for flags, field in cases:
            result = self._run(*(base + flags))
            self.assertEqual(result.returncode, 2, flags)
            self.assertEqual(result.stdout, "")
            self.assertNotIn("Traceback", result.stderr)
            error = json.loads(result.stderr)
            self.assertEqual(error["field"], field)

    def test_empty_identifier_flags_via_cli_fail_exit_2(self) -> None:
        key = _key_b64()
        result = self._run("encrypt-message", "--session-id", "s1",
                           "--key", key, "--plaintext", "x",
                           "--sender-device-id", "",
                           "--message-id", "m", "--sequence", "1")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stderr)["field"],
                         "sender_device_id")
        result = self._run("encrypt-message", "--session-id", "s1",
                           "--key", key, "--plaintext", "x",
                           "--sender-device-id", "d",
                           "--message-id", "", "--sequence", "1")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stderr)["field"], "message_id")

    def test_bad_sequence_flag_via_cli_uses_crypto_error_contract(self) -> None:
        key = _key_b64()
        for value in ["0", "-1", "1.0", "abc", "true", " 1", "1 ",
                      "0x1", "+1", "1e3"]:
            result = self._run(
                "encrypt-message", "--session-id", "s1", "--key", key,
                "--plaintext", "x", "--sender-device-id", "d",
                "--message-id", "m", "--sequence", value)
            self.assertEqual(result.returncode, 2, value)
            self.assertEqual(result.stdout, "", value)
            self.assertNotIn("Traceback", result.stderr)
            error = json.loads(result.stderr)
            self.assertEqual(error["field"], "sequence", value)

    def test_bad_sequence_ordering_via_cli_still_reports_missing_sender(self):
        key = _key_b64()
        # A bad sequence must not mask an earlier missing/empty field.
        result = self._run(
            "encrypt-message", "--session-id", "s1", "--key", key,
            "--plaintext", "x", "--message-id", "m", "--sequence", "abc")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stderr)["field"],
                         "sender_device_id")


if __name__ == "__main__":
    unittest.main()
