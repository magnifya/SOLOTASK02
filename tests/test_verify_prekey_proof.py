"""Tests for the offline trusted-fingerprint pre-key proof verification.

Covers the Python entry ``verify_prekey_proof`` and the local CLI command
``verify-prekey-proof``: both run purely offline (no server, no backend
state), check the proof's ownership strings verbatim, pin the frozen
identity key to a trusted fingerprint, and only then verify the
E2EE-SIGNED-PREKEY-V1 signature.
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


def _proof(private, user_id, device_id, key_id, public_key) -> str:
    message = signed_prekey_proof_message(
        user_id, device_id, key_id, public_key)
    return base64.b64encode(private.sign(message)).decode()


class VerifyPrekeyProofTest(unittest.TestCase):
    def setUp(self) -> None:
        self.private, self.identity = _new_identity()
        self.public_key = _new_prekey()
        self.signature = _proof(self.private, "u1", "d1", "k1",
                                self.public_key)
        self.fingerprint = identity_fingerprint(self.identity)
        self.proof = {
            "user_id": "u1", "device_id": "d1", "key_id": "k1",
            "public_key": self.public_key, "identity_key": self.identity,
            "signature": self.signature,
        }

    _UNSET = object()

    def _verify(self, proof=_UNSET, user_id="u1", device_id="d1",
                key_id="k1", fingerprint=_UNSET):
        return verify_prekey_proof(
            self.proof if proof is self._UNSET else proof, user_id,
            device_id, key_id,
            self.fingerprint if fingerprint is self._UNSET else fingerprint)

    def _assert_field(self, field, **kwargs) -> None:
        with self.assertRaises(CryptoError) as ctx:
            self._verify(**kwargs)
        self.assertEqual(ctx.exception.field, field)

    def test_success_returns_six_original_fields_plus_fingerprint(self):
        result = self._verify()
        self.assertEqual(result, {
            "user_id": "u1", "device_id": "d1", "key_id": "k1",
            "public_key": self.public_key, "identity_key": self.identity,
            "signature": self.signature,
            "fingerprint": self.fingerprint})

    def test_extra_fields_ignored_and_input_not_modified(self):
        proof = dict(self.proof, extra="x", fingerprint="0" * 64)
        before = dict(proof)
        result = self._verify(proof=proof)
        self.assertEqual(set(result), set(self.proof) | {"fingerprint"})
        self.assertEqual(proof, before)

    def test_identifiers_compared_verbatim(self):
        padded = dict(self.proof, user_id=" u1 ", device_id="d/1",
                      key_id="键1")
        padded["signature"] = _proof(self.private, " u1 ", "d/1", "键1",
                                     self.public_key)
        result = verify_prekey_proof(padded, " u1 ", "d/1", "键1",
                                     self.fingerprint)
        self.assertEqual(result["user_id"], " u1 ")
        # No trimming or normalization: the untrimmed expectations fail.
        self._assert_field("user_id", proof=padded, user_id="u1")
        self._assert_field("device_id", proof=padded, user_id=" u1 ",
                           device_id="d1")
        self._assert_field("key_id", proof=padded, user_id=" u1 ",
                           device_id="d/1", key_id="键1 ")

    def test_ownership_mismatch_names_the_identifier(self):
        self._assert_field("user_id", user_id="u2")
        self._assert_field("device_id", device_id="d2")
        self._assert_field("key_id", key_id="k2")

    def test_equivalent_identity_encoding_still_matches(self):
        # The fingerprint canonicalizes the identity key, so a DER/hex/PEM
        # spelling of the same key verifies against the same fingerprint.
        public = ed25519.Ed25519PublicKey.from_public_bytes(
            base64.b64decode(self.identity))
        der = public.public_bytes(
            serialization.Encoding.DER,
            serialization.PublicFormat.SubjectPublicKeyInfo)
        pem = public.public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo).decode("ascii")
        raw_hex = base64.b64decode(self.identity).hex()
        for spelling in (base64.b64encode(der).decode(), der.hex(), pem,
                         raw_hex):
            proof = dict(self.proof, identity_key=spelling)
            with self.subTest(spelling=spelling[:16]):
                result = self._verify(proof=proof)
                self.assertEqual(result["identity_key"], spelling)
                self.assertEqual(result["fingerprint"], self.fingerprint)

    def test_reencoded_prekey_public_key_fails_signature(self):
        # The pre-key public key string is signed verbatim: an equivalent
        # encoding of the same point was never signed.
        raw_hex = base64.b64decode(self.public_key).hex()
        proof = dict(self.proof, public_key=raw_hex)
        self._assert_field("signature", proof=proof)

    def test_proof_must_be_an_object(self):
        for bad in (None, [], "proof", 42, ["x"]):
            with self.subTest(bad=bad):
                self._assert_field("proof", proof=bad)

    def test_missing_or_empty_proof_fields_name_the_field(self):
        for name in ("user_id", "device_id", "key_id", "public_key",
                     "identity_key", "signature"):
            for bad in (None, "", 7):
                with self.subTest(name=name, bad=bad):
                    proof = dict(self.proof)
                    if bad is None:
                        del proof[name]
                    else:
                        proof[name] = bad
                    self._assert_field(name, proof=proof)

    def test_invalid_public_key(self):
        self._assert_field("public_key",
                           proof=dict(self.proof, public_key="not-a-key"))

    def test_identity_key_must_be_ed25519(self):
        self._assert_field("identity_key",
                           proof=dict(self.proof, identity_key="not-a-key"))
        # A DER X25519 key carries an algorithm identifier and is rejected;
        # only Ed25519 identity keys are accepted.
        x25519_der = x25519.X25519PrivateKey.generate().public_key(
            ).public_bytes(
                serialization.Encoding.DER,
                serialization.PublicFormat.SubjectPublicKeyInfo)
        proof = dict(self.proof,
                     identity_key=base64.b64encode(x25519_der).decode())
        self._assert_field("identity_key", proof=proof)

    def test_signature_encoding_and_length(self):
        for bad in ("not-base64!!", "abc",
                    base64.b64encode(b"\x00" * 63).decode(),
                    base64.b64encode(b"\x00" * 65).decode(),
                    # Valid 64-byte signature, non-canonical spelling.
                    self.signature.rstrip("=")):
            with self.subTest(bad=bad[:12]):
                self._assert_field("signature",
                                   proof=dict(self.proof, signature=bad))

    def test_foreign_signature_fails_verification(self):
        other, _ = _new_identity()
        forged = _proof(other, "u1", "d1", "k1", self.public_key)
        self._assert_field("signature",
                           proof=dict(self.proof, signature=forged))

    def test_expected_identifiers_must_be_nonempty_strings(self):
        for field, kwargs in (("user_id", {"user_id": ""}),
                              ("device_id", {"device_id": None}),
                              ("key_id", {"key_id": 3})):
            with self.subTest(field=field):
                self._assert_field(field, **kwargs)

    def test_expected_fingerprint_format(self):
        for bad in (None, "", "0" * 63, "0" * 65, "A" * 64, "g" * 64, 64):
            with self.subTest(bad=bad):
                self._assert_field("expected_fingerprint", fingerprint=bad)

    def test_fingerprint_mismatch(self):
        _, other_identity = _new_identity()
        other_fingerprint = identity_fingerprint(other_identity)
        self.assertNotEqual(other_fingerprint, self.fingerprint)
        self._assert_field("expected_fingerprint",
                           fingerprint=other_fingerprint)

    def test_fingerprint_checked_before_signature(self):
        # A broken signature is not reported when the fingerprint already
        # fails to match.
        _, other_identity = _new_identity()
        proof = dict(self.proof, signature=base64.b64encode(
            b"\x00" * 64).decode())
        self._assert_field("expected_fingerprint", proof=proof,
                           fingerprint=identity_fingerprint(other_identity))

    def test_history_rotation_scenario(self):
        # After an identity rotation the old trusted fingerprint still
        # verifies the old proof; the new identity's fingerprint does not.
        _, new_identity = _new_identity()
        new_fingerprint = identity_fingerprint(new_identity)
        self.assertEqual(self._verify()["fingerprint"], self.fingerprint)
        self._assert_field("expected_fingerprint",
                           fingerprint=new_fingerprint)


class VerifyPrekeyProofCliTest(unittest.TestCase):
    def setUp(self) -> None:
        self.private, self.identity = _new_identity()
        self.public_key = _new_prekey()
        self.signature = _proof(self.private, "u1", "d1", "k1",
                                self.public_key)
        self.fingerprint = identity_fingerprint(self.identity)
        self.proof = {
            "user_id": "u1", "device_id": "d1", "key_id": "k1",
            "public_key": self.public_key, "identity_key": self.identity,
            "signature": self.signature,
        }
        self.directory = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(
            self.directory, ignore_errors=True))

    def _run(self, *argv):
        env = dict(os.environ, PYTHONPATH=".")
        return subprocess.run(
            [sys.executable, "-m", "e2ee_backend", "verify-prekey-proof",
             *argv],
            capture_output=True, text=True, env=env, timeout=15)

    def _base_args(self, proof):
        return ["--proof", proof, "--user-id", "u1", "--device-id", "d1",
                "--key-id", "k1", "--expected-fingerprint", self.fingerprint]

    def _assert_cli_error(self, result, field):
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        lines = result.stderr.strip().splitlines()
        self.assertEqual(len(lines), 1)
        body = json.loads(lines[0])
        self.assertEqual(set(body), {"message", "field"})
        self.assertEqual(body["field"], field)
        self.assertNotIn("Traceback", result.stderr)

    def test_success_stdout_single_line_exit_0(self):
        result = self._run(*self._base_args(json.dumps(self.proof)))
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stderr, "")
        body = json.loads(result.stdout.strip())
        self.assertEqual(body, dict(self.proof,
                                    fingerprint=self.fingerprint))

    def test_proof_from_at_file(self):
        path = os.path.join(self.directory, "proof.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.proof, handle)
        result = self._run(*self._base_args("@" + path))
        self.assertEqual(result.returncode, 0)
        self.assertEqual(json.loads(result.stdout.strip())["key_id"], "k1")

    def test_invalid_json_and_non_object_are_proof_field(self):
        for bad in ("not json", "[1,2]", '"text"', "42"):
            with self.subTest(bad=bad):
                self._assert_cli_error(self._run(*self._base_args(bad)),
                                       "proof")

    def test_unreadable_and_non_utf8_file_are_proof_field(self):
        missing = os.path.join(self.directory, "missing.json")
        self._assert_cli_error(self._run(*self._base_args("@" + missing)),
                               "proof")
        binary = os.path.join(self.directory, "binary.json")
        with open(binary, "wb") as handle:
            handle.write(b"\xff\xfe{}")
        self._assert_cli_error(self._run(*self._base_args("@" + binary)),
                               "proof")

    def test_missing_options_use_the_json_contract(self):
        cases = [
            (["--user-id", "u1", "--device-id", "d1", "--key-id", "k1",
              "--expected-fingerprint", self.fingerprint], "proof"),
            (["--proof", json.dumps(self.proof), "--device-id", "d1",
              "--key-id", "k1", "--expected-fingerprint", self.fingerprint],
             "user_id"),
            (["--proof", json.dumps(self.proof), "--user-id", "u1",
              "--key-id", "k1", "--expected-fingerprint", self.fingerprint],
             "device_id"),
            (["--proof", json.dumps(self.proof), "--user-id", "u1",
              "--device-id", "d1",
              "--expected-fingerprint", self.fingerprint], "key_id"),
            (["--proof", json.dumps(self.proof), "--user-id", "u1",
              "--device-id", "d1", "--key-id", "k1"], "expected_fingerprint"),
        ]
        for argv, field in cases:
            with self.subTest(field=field):
                self._assert_cli_error(self._run(*argv), field)

    def test_verification_failure_exit_2(self):
        result = self._run(*self._base_args(json.dumps(
            dict(self.proof, key_id="k2"))))
        self._assert_cli_error(result, "key_id")


if __name__ == "__main__":
    unittest.main()
