"""Tests for the local AES-256-GCM message envelope helpers."""
import base64
import os
import unittest

from e2ee_backend.envelope import (
    KEY_BYTES,
    NONCE_BYTES,
    TAG_BYTES,
    EnvelopeError,
    open_message,
    seal_message,
)


def _key_b64() -> str:
    return base64.b64encode(os.urandom(KEY_BYTES)).decode()


class EnvelopeRoundTripTest(unittest.TestCase):
    def test_seal_returns_envelope_with_random_nonce(self) -> None:
        key = _key_b64()
        first = seal_message(
            {"session_id": "s1", "key": key, "plaintext": "hello"})
        self.assertEqual(set(first), {"session_id", "nonce", "ciphertext"})
        self.assertEqual(first["session_id"], "s1")
        self.assertEqual(len(base64.b64decode(first["nonce"])), NONCE_BYTES)
        # Ciphertext is plaintext length plus the 16-byte GCM tag.
        self.assertEqual(len(base64.b64decode(first["ciphertext"])),
                         len("hello") + TAG_BYTES)
        second = seal_message(
            {"session_id": "s1", "key": key, "plaintext": "hello"})
        self.assertNotEqual(first["nonce"], second["nonce"])
        self.assertNotEqual(first["ciphertext"], second["ciphertext"])

    def test_roundtrip_preserves_utf8_plaintext(self) -> None:
        key = _key_b64()
        plaintext = "héllo 世界 🌍"
        sealed = seal_message(
            {"session_id": "s-1", "key": key, "plaintext": plaintext})
        opened = open_message({
            "session_id": "s-1", "key": key,
            "nonce": sealed["nonce"], "ciphertext": sealed["ciphertext"]})
        self.assertEqual(opened, {"session_id": "s-1", "plaintext": plaintext})

    def test_empty_plaintext_roundtrips(self) -> None:
        key = _key_b64()
        sealed = seal_message(
            {"session_id": "s", "key": key, "plaintext": ""})
        opened = open_message({
            "session_id": "s", "key": key,
            "nonce": sealed["nonce"], "ciphertext": sealed["ciphertext"]})
        self.assertEqual(opened["plaintext"], "")


class EnvelopeAadTest(unittest.TestCase):
    def _sealed(self):
        key = _key_b64()
        return key, seal_message(
            {"session_id": "s1", "key": key, "plaintext": "secret"})

    def test_wrong_session_id_fails_authentication(self) -> None:
        key, sealed = self._sealed()
        with self.assertRaises(EnvelopeError) as ctx:
            open_message({"session_id": "s2", "key": key,
                          "nonce": sealed["nonce"],
                          "ciphertext": sealed["ciphertext"]})
        self.assertEqual(ctx.exception.field, "ciphertext")

    def test_wrong_key_fails_authentication(self) -> None:
        _, sealed = self._sealed()
        with self.assertRaises(EnvelopeError) as ctx:
            open_message({"session_id": "s1", "key": _key_b64(),
                          "nonce": sealed["nonce"],
                          "ciphertext": sealed["ciphertext"]})
        self.assertEqual(ctx.exception.field, "ciphertext")

    def test_tampered_ciphertext_fails_authentication(self) -> None:
        key, sealed = self._sealed()
        raw = bytearray(base64.b64decode(sealed["ciphertext"]))
        raw[0] ^= 0xFF
        sealed["ciphertext"] = base64.b64encode(bytes(raw)).decode()
        with self.assertRaises(EnvelopeError) as ctx:
            open_message({"session_id": "s1", "key": key,
                          "nonce": sealed["nonce"],
                          "ciphertext": sealed["ciphertext"]})
        self.assertEqual(ctx.exception.field, "ciphertext")


class EnvelopeValidationTest(unittest.TestCase):
    def _assert_field(self, fn, payload: object, field: str) -> None:
        with self.assertRaises(EnvelopeError) as ctx:
            fn(payload)
        self.assertEqual(ctx.exception.field, field)

    def test_seal_missing_or_bad_fields(self) -> None:
        key = _key_b64()
        good = {"session_id": "s1", "key": key, "plaintext": "x"}
        for field in ("session_id", "key", "plaintext"):
            payload = dict(good)
            del payload[field]
            self._assert_field(seal_message, payload, field)
        self._assert_field(seal_message,
                           {"session_id": "", "key": key, "plaintext": "x"},
                           "session_id")
        self._assert_field(seal_message,
                           {"session_id": "s1", "key": 7, "plaintext": "x"},
                           "key")
        self._assert_field(seal_message,
                           {"session_id": "s1", "key": key, "plaintext": 7},
                           "plaintext")

    def test_key_must_be_base64_and_32_bytes(self) -> None:
        self._assert_field(
            seal_message,
            {"session_id": "s1", "key": "not*base64*", "plaintext": "x"},
            "key")
        self._assert_field(
            seal_message,
            {"session_id": "s1", "key": base64.b64encode(b"short").decode(),
             "plaintext": "x"},
            "key")

    def test_open_bad_nonce_and_ciphertext(self) -> None:
        key, sealed = self._sealed()
        base = {"session_id": "s1", "key": key,
                "nonce": sealed["nonce"], "ciphertext": sealed["ciphertext"]}
        for field in ("session_id", "key", "nonce", "ciphertext"):
            payload = dict(base)
            del payload[field]
            self._assert_field(open_message, payload, field)
        self._assert_field(
            open_message, {**base, "nonce": "####"}, "nonce")
        self._assert_field(
            open_message, {**base, "ciphertext": "####"}, "ciphertext")
        self._assert_field(
            open_message,
            {**base, "nonce": base64.b64encode(b"short").decode()},
            "nonce")
        self._assert_field(
            open_message,
            {**base, "ciphertext": base64.b64encode(b"short").decode()},
            "ciphertext")

    @staticmethod
    def _sealed():
        key = _key_b64()
        return key, seal_message(
            {"session_id": "s1", "key": key, "plaintext": "secret"})


if __name__ == "__main__":
    unittest.main()
