"""Tests for the offline trusted-fingerprint signed pre-key proof verification.

Covers the Python entry point ``verify_prekey_proof`` and the
``verify-prekey-proof`` CLI command (real subprocess). The check is purely
local: a frozen six-string proof is compared against the expected
identifiers as-is, the proof's frozen identity key must match the trusted
fingerprint, and only then is the E2EE-SIGNED-PREKEY-V1 signature verified.
No server is contacted and no backend state is read or written.
"""
import base64
import json
import os
import subprocess
import sys
import tempfile
import unittest

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, x25519

from e2ee_backend.crypto import (CryptoError, identity_fingerprint,
                                 signed_prekey_proof_message,
                                 verify_prekey_proof)


def _raw_b64(key) -> str:
    raw = key.public_bytes(serialization.Encoding.Raw,
                           serialization.PublicFormat.Raw)
    return base64.b64encode(raw).decode()


def _new_identity():
    private = ed25519.Ed25519PrivateKey.generate()
    return private, _raw_b64(private.public_key())


def _new_prekey() -> str:
    return _raw_b64(x25519.X25519PrivateKey.generate().public_key())


def _sign(private, user_id, device_id, key_id, public_key) -> str:
    message = signed_prekey_proof_message(user_id, device_id, key_id,
                                          public_key)
    return base64.b64encode(private.sign(message)).decode()


def _der_b64_ed25519(raw_b64: str) -> str:
    public = ed25519.Ed25519PublicKey.from_public_bytes(
        base64.b64decode(raw_b64))
    der = public.public_bytes(serialization.Encoding.DER,
                              serialization.PublicFormat.SubjectPublicKeyInfo)
    return base64.b64encode(der).decode()


def _der_b64_x25519(raw_b64: str) -> str:
    public = x25519.X25519PublicKey.from_public_bytes(
        base64.b64decode(raw_b64))
    der = public.public_bytes(serialization.Encoding.DER,
                              serialization.PublicFormat.SubjectPublicKeyInfo)
    return base64.b64encode(der).decode()


class VerifyPrekeyProofCryptoTest(unittest.TestCase):
    def setUp(self) -> None:
        self.private, self.identity = _new_identity()
        self.public_key = _new_prekey()
        self.signature = _sign(self.private, "u1", "d1", "k1",
                               self.public_key)
        self.fingerprint = identity_fingerprint(self.identity)
        self.proof = {
            "user_id": "u1", "device_id": "d1", "key_id": "k1",
            "public_key": self.public_key, "identity_key": self.identity,
            "signature": self.signature,
        }

    _UNSET = object()

    def _verify(self, proof=_UNSET, user_id="u1", device_id="d1", key_id="k1",
                fingerprint=_UNSET):
        return verify_prekey_proof(
            self.proof if proof is self._UNSET else proof, user_id, device_id,
            key_id,
            self.fingerprint if fingerprint is self._UNSET else fingerprint)

    def _assert_field(self, field, **kwargs) -> None:
        with self.assertRaises(CryptoError) as ctx:
            self._verify(**kwargs)
        self.assertEqual(ctx.exception.field, field)

    def test_success_returns_six_original_fields_plus_fingerprint(self):
        result = self._verify()
        self.assertEqual(
            set(result),
            {"user_id", "device_id", "key_id", "public_key", "identity_key",
             "signature", "fingerprint"})
        for name in ("user_id", "device_id", "key_id", "public_key",
                     "identity_key", "signature"):
            self.assertEqual(result[name], self.proof[name])
        self.assertEqual(result["fingerprint"], self.fingerprint)

    def test_extra_fields_ignored_and_input_not_modified(self):
        proof = dict(self.proof, extra="ignored", nested={"a": 1})
        snapshot = json.loads(json.dumps(proof))
        result = self._verify(proof=proof)
        self.assertNotIn("extra", result)
        self.assertNotIn("nested", result)
        self.assertEqual(proof, snapshot)

    def test_identifiers_compared_verbatim(self):
        private, identity = _new_identity()
        public_key = _new_prekey()
        user_id = " 用户/alice "
        device_id = "dev / 1"
        key_id = "键\tk"
        signature = _sign(private, user_id, device_id, key_id, public_key)
        proof = {"user_id": user_id, "device_id": device_id,
                 "key_id": key_id, "public_key": public_key,
                 "identity_key": identity, "signature": signature}
        result = verify_prekey_proof(proof, user_id, device_id, key_id,
                                     identity_fingerprint(identity))
        self.assertEqual(result["user_id"], user_id)
        self.assertEqual(result["device_id"], device_id)
        self.assertEqual(result["key_id"], key_id)
        # No trimming or normalization: a trimmed expectation mismatches.
        with self.assertRaises(CryptoError) as ctx:
            verify_prekey_proof(proof, user_id.strip(), device_id, key_id,
                                identity_fingerprint(identity))
        self.assertEqual(ctx.exception.field, "user_id")

    def test_proof_not_an_object(self):
        for bad in (None, [], "proof", 3, True):
            self._assert_field("proof", proof=bad)

    def test_missing_or_empty_proof_fields_name_the_field(self):
        for name in ("user_id", "device_id", "key_id", "public_key",
                     "identity_key", "signature"):
            for bad in (None, "", 7, True):
                proof = dict(self.proof)
                proof[name] = bad
                self._assert_field(name, proof=proof)
            proof = dict(self.proof)
            del proof[name]
            self._assert_field(name, proof=proof)

    def test_missing_or_empty_expected_identifiers_name_the_field(self):
        for bad in (None, "", 5):
            self._assert_field("user_id", user_id=bad)
            self._assert_field("device_id", device_id=bad)
            self._assert_field("key_id", key_id=bad)

    def test_ownership_mismatch_names_the_field(self):
        self._assert_field("user_id", user_id="u2")
        self._assert_field("device_id", device_id="d2")
        self._assert_field("key_id", key_id="k2")

    def test_fingerprint_format_errors(self):
        for bad in (None, "", "zz" * 32, self.fingerprint.upper(),
                    self.fingerprint[:-1], self.fingerprint + "0", 64):
            self._assert_field("expected_fingerprint", fingerprint=bad)

    def test_fingerprint_mismatch(self):
        _, other_identity = _new_identity()
        self._assert_field("expected_fingerprint",
                           fingerprint=identity_fingerprint(other_identity))

    def test_identity_key_must_be_ed25519(self):
        # A DER SubjectPublicKeyInfo carrying the X25519 algorithm
        # identifier is rejected; so is any unparsable string.
        x_identity_der = _der_b64_x25519(_new_prekey())
        proof = dict(self.proof, identity_key=x_identity_der)
        self._assert_field("identity_key", proof=proof,
                           fingerprint=identity_fingerprint(x_identity_der))
        for bad in ("not-a-key", "!!!!", self.public_key[:-2]):
            proof = dict(self.proof, identity_key=bad)
            self._assert_field("identity_key", proof=proof)

    def test_public_key_must_parse(self):
        private, identity = _new_identity()
        signature = _sign(private, "u1", "d1", "k1", "not-a-key")
        proof = dict(self.proof, public_key="not-a-key",
                     identity_key=identity, signature=signature)
        self._assert_field("public_key", proof=proof,
                           fingerprint=identity_fingerprint(identity))

    def test_signature_must_be_canonical_base64_64_bytes(self):
        for bad in ("not base64!", "AAAA",  # not 64 bytes
                    base64.b64encode(b"\x00" * 32).decode(),
                    self.signature.rstrip("="),  # non-canonical padding
                    self.signature + " "):
            proof = dict(self.proof, signature=bad)
            self._assert_field("signature", proof=proof)

    def test_signature_verification_failure(self):
        other_private, _ = _new_identity()
        bad = _sign(other_private, "u1", "d1", "k1", self.public_key)
        proof = dict(self.proof, signature=bad)
        self._assert_field("signature", proof=proof)

    def test_equivalent_identity_encoding_still_matches_fingerprint(self):
        # The fingerprint canonicalizes the identity key, so a DER-base64
        # spelling of the same key matches the raw-point fingerprint — but
        # the signature signs the literal strings, so the proof must have
        # been signed with that exact identity_key spelling... it was not:
        # the signature does not cover identity_key, so it still verifies.
        der_identity = _der_b64_ed25519(self.identity)
        proof = dict(self.proof, identity_key=der_identity)
        result = self._verify(proof=proof)
        self.assertEqual(result["identity_key"], der_identity)
        self.assertEqual(result["fingerprint"], self.fingerprint)

    def test_reencoded_prekey_public_key_fails_verification(self):
        # Same pre-key point in DER base64: equivalent encoding, but the
        # signature binds the literal public_key string, so it fails.
        der_public = _der_b64_x25519(self.public_key)
        proof = dict(self.proof, public_key=der_public)
        self._assert_field("signature", proof=proof)
        # Re-signing the new spelling verifies again.
        resigned = _sign(self.private, "u1", "d1", "k1", der_public)
        proof = dict(self.proof, public_key=der_public, signature=resigned)
        result = self._verify(proof=proof)
        self.assertEqual(result["public_key"], der_public)

    def test_historical_proof_after_identity_rotation(self):
        # The proof freezes the identity key at publication time: after a
        # rotation the old trusted fingerprint still verifies the old proof,
        # and the new identity's fingerprint mismatches.
        _, new_identity = _new_identity()
        result = self._verify()
        self.assertEqual(result["identity_key"], self.identity)
        self._assert_field("expected_fingerprint",
                           fingerprint=identity_fingerprint(new_identity))


class VerifyPrekeyProofCLITest(unittest.TestCase):
    def setUp(self) -> None:
        self.private, self.identity = _new_identity()
        self.public_key = _new_prekey()
        self.signature = _sign(self.private, "u1", "d1", "k1",
                               self.public_key)
        self.fingerprint = identity_fingerprint(self.identity)
        self.proof = {
            "user_id": "u1", "device_id": "d1", "key_id": "k1",
            "public_key": self.public_key, "identity_key": self.identity,
            "signature": self.signature,
        }

    def _run(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "e2ee_backend", "verify-prekey-proof",
             *arguments],
            capture_output=True, text=True, timeout=15)

    def _run_ok(self, proof_arg: str) -> dict:
        result = self._run(
            "--proof", proof_arg, "--user-id", "u1", "--device-id", "d1",
            "--key-id", "k1", "--expected-fingerprint", self.fingerprint)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        line = result.stdout.strip()
        self.assertEqual(line.count("\n"), 0)
        return json.loads(line)

    def _run_error(self, *arguments: str) -> dict:
        result = self._run(*arguments)
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertNotIn("Traceback", result.stderr)
        line = result.stderr.strip()
        self.assertEqual(line.count("\n"), 0)
        body = json.loads(line)
        self.assertEqual(set(body), {"message", "field"})
        return body

    def _good_args(self, proof_arg: str):
        return ("--proof", proof_arg, "--user-id", "u1", "--device-id", "d1",
                "--key-id", "k1", "--expected-fingerprint", self.fingerprint)

    def test_success_with_inline_json(self):
        body = self._run_ok(json.dumps(self.proof))
        self.assertEqual(
            set(body),
            {"user_id", "device_id", "key_id", "public_key", "identity_key",
             "signature", "fingerprint"})
        for name, value in self.proof.items():
            self.assertEqual(body[name], value)
        self.assertEqual(body["fingerprint"], self.fingerprint)

    def test_success_with_at_file_and_extra_fields(self):
        proof = dict(self.proof, extra="ignored")
        with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", suffix=".json", delete=False) as handle:
            json.dump(proof, handle)
            path = handle.name
        try:
            body = self._run_ok("@" + path)
        finally:
            os.unlink(path)
        self.assertNotIn("extra", body)
        self.assertEqual(body["signature"], self.signature)

    def test_invalid_json_and_non_object_are_proof_errors(self):
        for bad in ("{not json", "[1,2]", '"text"', "3"):
            body = self._run_error(*self._good_args(bad))
            self.assertEqual(body["field"], "proof")

    def test_missing_proof_option_is_proof_error(self):
        body = self._run_error(
            "--user-id", "u1", "--device-id", "d1", "--key-id", "k1",
            "--expected-fingerprint", self.fingerprint)
        self.assertEqual(body["field"], "proof")

    def test_unreadable_file_is_proof_error(self):
        body = self._run_error(*self._good_args("@/nonexistent/proof.json"))
        self.assertEqual(body["field"], "proof")

    def test_non_utf8_file_is_proof_error(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            f.write(b'{"user_id": "\xff\xfe"}')
            path = f.name
        try:
            body = self._run_error(*self._good_args("@" + path))
        finally:
            os.unlink(path)
        self.assertEqual(body["field"], "proof")

    def test_missing_option_names_its_field(self):
        proof_arg = json.dumps(self.proof)
        for option, field in (("--user-id", "user_id"),
                              ("--device-id", "device_id"),
                              ("--key-id", "key_id"),
                              ("--expected-fingerprint",
                               "expected_fingerprint")):
            args = ["--proof", proof_arg, "--user-id", "u1",
                    "--device-id", "d1", "--key-id", "k1",
                    "--expected-fingerprint", self.fingerprint]
            index = args.index(option)
            del args[index:index + 2]
            body = self._run_error(*args)
            self.assertEqual(body["field"], field)

    def test_ownership_mismatch(self):
        args = list(self._good_args(json.dumps(self.proof)))
        args[args.index("--user-id") + 1] = "u2"
        body = self._run_error(*args)
        self.assertEqual(body["field"], "user_id")

    def test_fingerprint_mismatch(self):
        _, other = _new_identity()
        args = list(self._good_args(json.dumps(self.proof)))
        args[args.index("--expected-fingerprint") + 1] = \
            identity_fingerprint(other)
        body = self._run_error(*args)
        self.assertEqual(body["field"], "expected_fingerprint")

    def test_bad_signature(self):
        other_private, _ = _new_identity()
        proof = dict(self.proof, signature=_sign(
            other_private, "u1", "d1", "k1", self.public_key))
        body = self._run_error(*self._good_args(json.dumps(proof)))
        self.assertEqual(body["field"], "signature")


if __name__ == "__main__":
    unittest.main()
