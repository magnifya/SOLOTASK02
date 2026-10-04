"""Tests for offline Ed25519 ciphertext-envelope signing and verification.

Covers the Python entry points ``sign_message`` and ``verify_message``.
Signing is purely local: an Ed25519 identity private key seed signs the
public ``E2EE-SIGNED-MESSAGE-V1`` message over the same six envelope fields
a message submission freezes (``session_id``, ``sender_device_id``,
``message_id``, ``sequence``, ``nonce``, ``ciphertext``), producing the
six envelope fields plus ``identity_key`` and ``signature``. Verification
is purely local and historical: the envelope's own session/device
identifiers and the trusted fingerprint are checked first, and only then
is the signature verified against the identity key frozen inside the
envelope. No server is contacted, no backend state is read or written, and
the private seed never appears in the result.
"""
import base64
import json
import os
import unittest

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, x25519

from e2ee_backend.crypto import (CryptoError, identity_fingerprint,
                                 sign_message, verify_message)

_ENVELOPE_FIELDS = ("session_id", "sender_device_id", "message_id",
                    "sequence", "nonce", "ciphertext")
_SIGNED_FIELDS = (*_ENVELOPE_FIELDS, "identity_key", "signature")


def _seed_b64(private) -> str:
    return base64.b64encode(private.private_bytes(
        serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
        serialization.NoEncryption())).decode()


def _raw_b64(public) -> str:
    return base64.b64encode(public.public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw)).decode()


def _der_b64(public) -> str:
    return base64.b64encode(public.public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo)).decode()


class SignMessageTest(unittest.TestCase):
    def setUp(self) -> None:
        self.private = ed25519.Ed25519PrivateKey.generate()
        self.seed = _seed_b64(self.private)
        self.identity = _raw_b64(self.private.public_key())
        self.envelope = {
            "session_id": "sess-1",
            "sender_device_id": "dev-1",
            "message_id": "msg-1",
            "sequence": 1,
            "nonce": base64.b64encode(os.urandom(12)).decode(),
            "ciphertext": base64.b64encode(os.urandom(32)).decode(),
        }

    _DEFAULT = object()

    def _sign(self, envelope=_DEFAULT, private_key=_DEFAULT):
        return sign_message(
            self.envelope if envelope is self._DEFAULT else envelope,
            self.seed if private_key is self._DEFAULT else private_key)

    def test_success_returns_six_fields_plus_identity_and_signature(self):
        signed = self._sign()
        self.assertEqual(set(signed), set(_SIGNED_FIELDS))
        for name in _ENVELOPE_FIELDS:
            self.assertEqual(signed[name], self.envelope[name])
        self.assertNotIn("private_key", signed)
        self.assertNotIn(self.seed, json.dumps(signed))

    def test_extra_fields_ignored_and_input_not_modified(self):
        envelope = dict(self.envelope, extra="ignored", another=7)
        snapshot = json.dumps(envelope, sort_keys=True, ensure_ascii=False)
        signed = self._sign(envelope)
        self.assertEqual(set(signed), set(_SIGNED_FIELDS))
        self.assertEqual(
            json.dumps(envelope, sort_keys=True, ensure_ascii=False), snapshot)

    def test_returned_object_is_independent_of_input(self):
        signed = self._sign()
        signed["session_id"] = "mutated"
        self.assertEqual(self.envelope["session_id"], "sess-1")

    def test_identifiers_preserved_verbatim(self):
        envelope = dict(self.envelope, session_id=" 会话/a ",
                        sender_device_id=" dev / 一 ",
                        message_id=" 消息/ m ")
        signed = self._sign(envelope)
        self.assertEqual(signed["session_id"], " 会话/a ")
        self.assertEqual(signed["sender_device_id"], " dev / 一 ")
        self.assertEqual(signed["message_id"], " 消息/ m ")

    def test_identity_key_is_canonical_raw_public_point(self):
        signed = self._sign()
        self.assertEqual(signed["identity_key"], self.identity)
        raw = base64.b64decode(signed["identity_key"], validate=True)
        self.assertEqual(len(raw), 32)
        self.assertEqual(base64.b64encode(raw).decode(),
                         signed["identity_key"])
        self.assertEqual(
            ed25519.Ed25519PublicKey.from_public_bytes(raw),
            self.private.public_key())

    def test_signature_is_canonical_base64_64_bytes(self):
        signed = self._sign()
        raw = base64.b64decode(signed["signature"], validate=True)
        self.assertEqual(len(raw), 64)
        self.assertEqual(base64.b64encode(raw).decode(),
                         signed["signature"])

    def test_signature_follows_public_protocol(self):
        signed = self._sign()
        document = json.dumps(
            {name: self.envelope[name] for name in _ENVELOPE_FIELDS},
            sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        message = ("E2EE-SIGNED-MESSAGE-V1\n" + document).encode("utf-8")
        self.private.public_key().verify(
            base64.b64decode(signed["signature"]), message)
        # Key order inside the JSON is the sorted six-field order.
        self.assertEqual(
            json.dumps({name: self.envelope[name]
                        for name in _ENVELOPE_FIELDS},
                       sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False),
            '{"ciphertext":%s,"message_id":"msg-1","nonce":%s,'
            '"sender_device_id":"dev-1","sequence":1,'
            '"session_id":"sess-1"}'
            % (json.dumps(self.envelope["ciphertext"]),
               json.dumps(self.envelope["nonce"])))

    def test_unicode_is_not_escaped_in_signed_document(self):
        envelope = dict(self.envelope, session_id="会话/s",
                        message_id="消息 m")
        signed = self._sign(envelope)
        document = json.dumps(
            {name: envelope[name] for name in _ENVELOPE_FIELDS},
            sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        message = ("E2EE-SIGNED-MESSAGE-V1\n" + document).encode("utf-8")
        # Verifies byte-for-byte: escaped JSON would not match.
        self.private.public_key().verify(
            base64.b64decode(signed["signature"]), message)

    def test_deterministic_same_inputs_same_output(self):
        first = self._sign()
        self.assertEqual(self._sign(), first)
        self.assertEqual(self._sign(), self._sign())

    def test_all_zero_seed_is_a_valid_ed25519_seed(self):
        signed = self._sign(
            private_key=base64.b64encode(b"\x00" * 32).decode())
        identity = ed25519.Ed25519PublicKey.from_public_bytes(
            base64.b64decode(signed["identity_key"]))
        document = json.dumps(
            {name: self.envelope[name] for name in _ENVELOPE_FIELDS},
            sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        identity.verify(
            base64.b64decode(signed["signature"]),
            ("E2EE-SIGNED-MESSAGE-V1\n" + document).encode("utf-8"))

    def test_ciphertext_may_be_exactly_sixteen_bytes(self):
        envelope = dict(self.envelope,
                        ciphertext=base64.b64encode(b"\x00" * 16).decode())
        signed = self._sign(envelope)
        self.assertEqual(signed["ciphertext"], envelope["ciphertext"])

    # -- error handling -----------------------------------------------------

    def _assert_field(self, field, **kwargs):
        with self.assertRaises(CryptoError) as ctx:
            self._sign(**kwargs)
        self.assertEqual(ctx.exception.field, field)

    def test_non_object_envelope_names_envelope(self):
        for bad in (None, [], "x", 7, True, b"{}"):
            self._assert_field("envelope", envelope=bad)

    def test_missing_fields_name_that_field(self):
        for name in _ENVELOPE_FIELDS:
            envelope = dict(self.envelope)
            del envelope[name]
            self._assert_field(name, envelope=envelope)

    def test_empty_or_wrong_typed_string_fields_name_that_field(self):
        for name in ("session_id", "sender_device_id", "message_id"):
            for bad in ("", 0, 7, True, False, b"x", ["x"], {"x": 1}, None):
                self._assert_field(name, envelope=dict(self.envelope,
                                                       **{name: bad}))

    def test_sequence_must_be_positive_integer_not_boolean(self):
        for bad in (True, False, 0, -1, 1.0, "1", None, [], {}):
            self._assert_field("sequence",
                               envelope=dict(self.envelope, sequence=bad))
        # Large positive integers are accepted.
        signed = self._sign(envelope=dict(self.envelope, sequence=2 ** 62))
        self.assertEqual(signed["sequence"], 2 ** 62)

    def test_nonce_must_be_canonical_base64_of_twelve_bytes(self):
        for bad in (
                "", None, 7, True,
                "!!!",
                base64.b64encode(b"x" * 11).decode(),
                base64.b64encode(b"x" * 13).decode(),
                self.envelope["nonce"] + " ",
                " " + self.envelope["nonce"],
                self.envelope["nonce"] + "\n",
                self.envelope["nonce"].replace("+", "-")
                if "+" in self.envelope["nonce"] else None):
            if bad is None:
                continue
            self._assert_field("nonce",
                               envelope=dict(self.envelope, nonce=bad))

    def test_nonce_encoding_has_no_ignored_trailing_bits(self):
        # Twelve bytes encode to exactly 16 standard-base64 characters with
        # no padding: every character bit is significant, so there is no
        # non-canonical spelling to reject -- re-encoding always reproduces
        # the string, and another canonical 12-byte nonce signs fine.
        canonical = self.envelope["nonce"]
        self.assertEqual(len(canonical), 16)
        self.assertEqual(
            base64.b64encode(base64.b64decode(canonical)).decode(),
            canonical)
        other = base64.b64encode(os.urandom(12)).decode()
        signed = self._sign(envelope=dict(self.envelope, nonce=other))
        self.assertEqual(signed["nonce"], other)

    def test_ciphertext_non_canonical_trailing_bits_rejected(self):
        # Sixteen bytes encode with "==": the high two bits of the last
        # data character are the only significant ones, so replacing it
        # with another character sharing those bits decodes to the same
        # bytes but must still be refused as non-canonical.
        canonical = base64.b64encode(b"\x00" * 16).decode()
        self.assertEqual(canonical[-2:], "==")
        self.assertEqual(canonical[-3], "A")
        non_canonical = canonical[:-3] + "B" + "=="
        self.assertEqual(base64.b64decode(non_canonical), b"\x00" * 16)
        self.assertNotEqual(non_canonical, canonical)
        self._assert_field(
            "ciphertext",
            envelope=dict(self.envelope, ciphertext=non_canonical))

    def test_ciphertext_must_be_canonical_base64_of_at_least_16_bytes(self):
        for bad in (
                "", None, 7,
                "!!!",
                base64.b64encode(b"x" * 15).decode(),
                self.envelope["ciphertext"].rstrip("="),
                self.envelope["ciphertext"] + " "):
            self._assert_field("ciphertext",
                               envelope=dict(self.envelope, ciphertext=bad))

    def test_unencodable_utf8_names_that_identifier(self):
        surrogate = "ud800\ud800"
        for name in ("session_id", "sender_device_id", "message_id"):
            self._assert_field(name,
                               envelope=dict(self.envelope,
                                             **{name: surrogate}))

    def test_private_key_wrong_formats_rejected(self):
        pkcs8_der = base64.b64encode(self.private.private_bytes(
            serialization.Encoding.DER, serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption())).decode()
        pkcs8_pem = self.private.private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption()).decode()
        for bad in (
                None, "", "not base64!", "--__", pkcs8_der, pkcs8_pem,
                base64.b64encode(b"\x00" * 31).decode(),
                base64.b64encode(b"\x00" * 33).decode(),
                base64.b64encode(b"\x00" * 64).decode(),
                self.seed + " ", " " + self.seed, self.seed + "\n",
                self.seed.rstrip("="), self.seed + "="):
            self._assert_field("private_key", private_key=bad)

    def test_private_key_hex_seed_refused(self):
        self._assert_field(
            "private_key",
            private_key=base64.b64decode(self.seed).hex())

    def test_errors_reported_in_envelope_field_order_then_private_key(self):
        envelope = dict(self.envelope, session_id="", sender_device_id="",
                        message_id="", sequence=False, nonce="!",
                        ciphertext="!")
        self._assert_field("session_id", envelope=envelope,
                           private_key="bad")
        envelope["session_id"] = "s"
        self._assert_field("sender_device_id", envelope=envelope,
                           private_key="bad")
        envelope["sender_device_id"] = "d"
        self._assert_field("message_id", envelope=envelope,
                           private_key="bad")
        envelope["message_id"] = "m"
        self._assert_field("sequence", envelope=envelope,
                           private_key="bad")
        envelope["sequence"] = 1
        self._assert_field("nonce", envelope=envelope, private_key="bad")
        envelope["nonce"] = self.envelope["nonce"]
        self._assert_field("ciphertext", envelope=envelope,
                           private_key="bad")
        envelope["ciphertext"] = self.envelope["ciphertext"]
        self._assert_field("private_key", envelope=envelope,
                           private_key="bad")

    def test_error_message_never_echoes_private_key(self):
        with self.assertRaises(CryptoError) as ctx:
            self._sign(private_key="top-secret-value")
        self.assertNotIn("top-secret-value", ctx.exception.message)
        with self.assertRaises(CryptoError) as ctx:
            self._sign(private_key=self.seed + " ")
        self.assertNotIn(self.seed, ctx.exception.message)


class VerifyMessageTest(unittest.TestCase):
    def setUp(self) -> None:
        self.private = ed25519.Ed25519PrivateKey.generate()
        self.seed = _seed_b64(self.private)
        self.identity = _raw_b64(self.private.public_key())
        self.fingerprint = identity_fingerprint(self.identity)
        self.sid = " 会话/sess "
        self.dev = " dev/一 "
        self.envelope = {
            "session_id": self.sid,
            "sender_device_id": self.dev,
            "message_id": " 消息/1 ",
            "sequence": 4,
            "nonce": base64.b64encode(os.urandom(12)).decode(),
            "ciphertext": base64.b64encode(os.urandom(40)).decode(),
        }
        self.signed = sign_message(self.envelope, self.seed)

    _DEFAULT = object()

    def _verify(self, envelope=_DEFAULT, session_id=_DEFAULT,
                sender_device_id=_DEFAULT, fingerprint=_DEFAULT):
        return verify_message(
            self.signed if envelope is self._DEFAULT else envelope,
            self.sid if session_id is self._DEFAULT else session_id,
            self.dev if sender_device_id is self._DEFAULT else sender_device_id,
            self.fingerprint if fingerprint is self._DEFAULT else fingerprint)

    def test_success_returns_eight_fields_plus_fingerprint(self):
        result = self._verify()
        self.assertEqual(set(result), set(_SIGNED_FIELDS) | {"fingerprint"})
        for name in _SIGNED_FIELDS:
            self.assertEqual(result[name], self.signed[name])
        self.assertEqual(result["fingerprint"], self.fingerprint)
        self.assertNotIn("private_key", result)

    def test_success_roundtrips_with_sign_message(self):
        result = self._verify()
        for name in _ENVELOPE_FIELDS:
            self.assertEqual(result[name], self.envelope[name])

    def test_extra_fields_ignored_and_input_not_modified(self):
        envelope = dict(self.signed, extra="ignored")
        snapshot = json.dumps(envelope, sort_keys=True, ensure_ascii=False)
        result = self._verify(envelope)
        self.assertEqual(set(result), set(_SIGNED_FIELDS) | {"fingerprint"})
        self.assertEqual(
            json.dumps(envelope, sort_keys=True, ensure_ascii=False), snapshot)

    def test_identifiers_compared_verbatim_without_trim_or_normalization(self):
        # The signed envelope keeps spaces/Chinese/slashes; only the exact
        # expected strings match.
        self._verify()
        with self.assertRaises(CryptoError) as ctx:
            self._verify(session_id=self.sid.strip())
        self.assertEqual(ctx.exception.field, "session_id")
        with self.assertRaises(CryptoError) as ctx:
            self._verify(sender_device_id=self.dev.strip())
        self.assertEqual(ctx.exception.field, "sender_device_id")
        with self.assertRaises(CryptoError) as ctx:
            self._verify(session_id="会话/sess")
        self.assertEqual(ctx.exception.field, "session_id")

    def test_equivalent_identity_key_encodings_still_verify(self):
        public = self.private.public_key()
        raw = public.public_bytes(serialization.Encoding.Raw,
                                  serialization.PublicFormat.Raw)
        for encoding in (_der_b64(public),
                         base64.b64encode(
                             public.public_bytes(
                                 serialization.Encoding.DER,
                                 serialization.PublicFormat.
                                 SubjectPublicKeyInfo)).decode(),
                         public.public_bytes(
                             serialization.Encoding.DER,
                             serialization.PublicFormat.
                             SubjectPublicKeyInfo).hex(),
                         public.public_bytes(
                             serialization.Encoding.PEM,
                             serialization.PublicFormat.
                             SubjectPublicKeyInfo).decode(),
                         raw.hex()):
            with self.subTest(encoding=encoding[:16]):
                envelope = dict(self.signed, identity_key=encoding)
                result = self._verify(envelope)
                # The original spelling is preserved in the result.
                self.assertEqual(result["identity_key"], encoding)
                self.assertEqual(result["fingerprint"], self.fingerprint)

    def test_tampering_signed_string_ids_names_expected_identifier(self):
        # session_id / sender_device_id mismatches surface on the expected
        # identifier (checked before the signature).
        bad = dict(self.signed, session_id=self.sid + "x")
        with self.assertRaises(CryptoError) as ctx:
            self._verify(bad)
        self.assertEqual(ctx.exception.field, "session_id")
        bad = dict(self.signed, sender_device_id=self.dev + "x")
        with self.assertRaises(CryptoError) as ctx:
            self._verify(bad)
        self.assertEqual(ctx.exception.field, "sender_device_id")

    def test_tampering_other_signed_fields_fails_signature(self):
        bad = dict(self.signed, message_id=self.signed["message_id"] + "x")
        with self.assertRaises(CryptoError) as ctx:
            self._verify(bad)
        self.assertEqual(ctx.exception.field, "signature")
        for name in ("nonce", "ciphertext"):
            raw = bytearray(base64.b64decode(self.signed[name]))
            raw[0] ^= 1
            bad = dict(self.signed,
                       **{name: base64.b64encode(bytes(raw)).decode()})
            with self.assertRaises(CryptoError) as ctx:
                self._verify(bad)
            self.assertEqual(ctx.exception.field, "signature", name)
        bad = dict(self.signed, sequence=self.signed["sequence"] + 1)
        with self.assertRaises(CryptoError) as ctx:
            self._verify(bad)
        self.assertEqual(ctx.exception.field, "signature")

    def test_signature_from_other_key_fails(self):
        other = ed25519.Ed25519PrivateKey.generate()
        other_identity = _raw_b64(other.public_key())
        other_fp = identity_fingerprint(other_identity)
        foreign = sign_message(self.envelope, _seed_b64(other))
        # Trusted fingerprint does not match the foreign key.
        with self.assertRaises(CryptoError) as ctx:
            self._verify(foreign)
        self.assertEqual(ctx.exception.field, "expected_fingerprint")
        # Even trusting the foreign fingerprint, mixing our signature with
        # the foreign key fails signature verification.
        mixed = dict(foreign, signature=self.signed["signature"])
        with self.assertRaises(CryptoError) as ctx:
            self._verify(mixed, fingerprint=other_fp)
        self.assertEqual(ctx.exception.field, "signature")

    def test_old_envelope_still_verifies_after_identity_rotation(self):
        # Verification is historical/offline: a new identity and its
        # fingerprint do not affect an envelope frozen under the old key.
        rotated = ed25519.Ed25519PrivateKey.generate()
        new_fingerprint = identity_fingerprint(_raw_b64(
            rotated.public_key()))
        result = self._verify()
        self.assertEqual(result["fingerprint"], self.fingerprint)
        with self.assertRaises(CryptoError) as ctx:
            self._verify(fingerprint=new_fingerprint)
        self.assertEqual(ctx.exception.field, "expected_fingerprint")
        # The old frozen envelope verifies again and again, identically.
        again = self._verify()
        self.assertEqual(again, result)

    def test_no_partial_result_on_failure(self):
        # A failure raises before any dict is produced; callers receive
        # nothing to work with.
        with self.assertRaises(CryptoError):
            self._verify(fingerprint="0" * 64)

    # -- error attribution --------------------------------------------------

    def test_non_object_envelope_names_envelope(self):
        for bad in (None, [], "x", 7, True, b"{}"):
            with self.assertRaises(CryptoError) as ctx:
                verify_message(bad, self.sid, self.dev, self.fingerprint)
            self.assertEqual(ctx.exception.field, "envelope")

    def test_invalid_expected_identifiers_name_their_field(self):
        for bad in ("", None, 7, True, b"s", ["s"]):
            with self.assertRaises(CryptoError) as ctx:
                verify_message(self.signed, bad, self.dev,
                               self.fingerprint)
            self.assertEqual(ctx.exception.field, "session_id")
            with self.assertRaises(CryptoError) as ctx:
                verify_message(self.signed, self.sid, bad,
                               self.fingerprint)
            self.assertEqual(ctx.exception.field, "sender_device_id")

    def test_mismatched_expected_identifiers_name_their_field(self):
        with self.assertRaises(CryptoError) as ctx:
            self._verify(session_id="other")
        self.assertEqual(ctx.exception.field, "session_id")
        with self.assertRaises(CryptoError) as ctx:
            self._verify(sender_device_id="other")
        self.assertEqual(ctx.exception.field, "sender_device_id")

    def test_expected_fingerprint_format(self):
        for bad in ("", None, 7, True,
                    "z" * 64, "A" * 64,
                    self.fingerprint[:-1],
                    self.fingerprint[:-1] + "g",
                    " " + self.fingerprint):
            with self.assertRaises(CryptoError) as ctx:
                self._verify(fingerprint=bad)
            self.assertEqual(ctx.exception.field, "expected_fingerprint")

    def test_fingerprint_mismatch_names_expected_fingerprint(self):
        other = ed25519.Ed25519PrivateKey.generate()
        other_fp = identity_fingerprint(_raw_b64(other.public_key()))
        with self.assertRaises(CryptoError) as ctx:
            self._verify(fingerprint=other_fp)
        self.assertEqual(ctx.exception.field, "expected_fingerprint")

    def test_envelope_field_errors_keep_their_field_names(self):
        for name in _ENVELOPE_FIELDS:
            envelope = dict(self.signed)
            del envelope[name]
            with self.assertRaises(CryptoError) as ctx:
                self._verify(envelope)
            self.assertEqual(ctx.exception.field, name)
        for name in ("session_id", "sender_device_id", "message_id"):
            with self.assertRaises(CryptoError) as ctx:
                self._verify(dict(self.signed, **{name: ""}))
            self.assertEqual(ctx.exception.field, name)
        with self.assertRaises(CryptoError) as ctx:
            self._verify(dict(self.signed, sequence=True))
        self.assertEqual(ctx.exception.field, "sequence")
        with self.assertRaises(CryptoError) as ctx:
            self._verify(dict(
                self.signed,
                nonce=base64.b64encode(b"x" * 11).decode()))
        self.assertEqual(ctx.exception.field, "nonce")
        with self.assertRaises(CryptoError) as ctx:
            self._verify(dict(
                self.signed,
                ciphertext=base64.b64encode(b"x" * 15).decode()))
        self.assertEqual(ctx.exception.field, "ciphertext")

    def test_identity_key_rejects_non_ed25519_and_unparsable_values(self):
        x_public = x25519.X25519PrivateKey.generate().public_key()
        x_der_b64 = base64.b64encode(x_public.public_bytes(
            serialization.Encoding.DER,
            serialization.PublicFormat.SubjectPublicKeyInfo)).decode()
        x_pem = x_public.public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo).decode()
        for bad in (x_der_b64, x_pem, "not-a-key", "!!!", "", 7, None,
                    base64.b64encode(b"\x00" * 31).decode(),
                    base64.b64encode(b"\x00" * 33).decode()):
            with self.assertRaises(CryptoError) as ctx:
                self._verify(dict(self.signed, identity_key=bad))
            self.assertEqual(ctx.exception.field, "identity_key", bad)

    def test_foreign_raw_point_is_rejected_at_fingerprint_stage(self):
        # A raw 32-byte X25519 point is also a syntactically valid raw
        # Ed25519 point under the existing key-loading range, so it cannot
        # be distinguished at parse time; it names a different identity and
        # the trusted-fingerprint comparison (which runs before signature
        # verification) rejects it.
        foreign = _raw_b64(
            x25519.X25519PrivateKey.generate().public_key())
        with self.assertRaises(CryptoError) as ctx:
            self._verify(dict(self.signed, identity_key=foreign))
        self.assertEqual(ctx.exception.field, "expected_fingerprint")

    def test_missing_identity_key_or_signature_names_that_field(self):
        for name in ("identity_key", "signature"):
            envelope = dict(self.signed)
            del envelope[name]
            with self.assertRaises(CryptoError) as ctx:
                self._verify(envelope)
            self.assertEqual(ctx.exception.field, name)
            with self.assertRaises(CryptoError) as ctx:
                self._verify(dict(self.signed, **{name: ""}))
            self.assertEqual(ctx.exception.field, name)

    def test_bad_signature_encoding_names_signature(self):
        for bad in (
                None, 7, "!!!",
                self.signed["signature"].rstrip("="),
                self.signed["signature"] + " ",
                base64.b64encode(b"\x00" * 63).decode(),
                base64.b64encode(b"\x00" * 65).decode()):
            with self.assertRaises(CryptoError) as ctx:
                self._verify(dict(self.signed, signature=bad))
            self.assertEqual(ctx.exception.field, "signature", bad)

    def test_signature_bit_flip_names_signature(self):
        raw = bytearray(base64.b64decode(self.signed["signature"]))
        raw[0] ^= 1
        with self.assertRaises(CryptoError) as ctx:
            self._verify(dict(
                self.signed,
                signature=base64.b64encode(bytes(raw)).decode()))
        self.assertEqual(ctx.exception.field, "signature")

    def test_checks_run_in_documented_order(self):
        # Expected identifiers and fingerprint are validated before the
        # envelope body: an invalid expected id with an invalid envelope
        # still reports the expected id first.
        with self.assertRaises(CryptoError) as ctx:
            verify_message("not-an-object", "", self.dev, self.fingerprint)
        self.assertEqual(ctx.exception.field, "envelope")
        envelope = dict(self.signed, nonce="bad")
        with self.assertRaises(CryptoError) as ctx:
            verify_message(envelope, "", self.dev, self.fingerprint)
        self.assertEqual(ctx.exception.field, "session_id")
        with self.assertRaises(CryptoError) as ctx:
            verify_message(envelope, self.sid, "", self.fingerprint)
        self.assertEqual(ctx.exception.field, "sender_device_id")
        with self.assertRaises(CryptoError) as ctx:
            verify_message(envelope, self.sid, self.dev, "bad")
        self.assertEqual(ctx.exception.field, "expected_fingerprint")


if __name__ == "__main__":
    unittest.main()
