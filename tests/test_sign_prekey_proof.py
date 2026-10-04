"""Tests for local signed pre-key proof generation.

Covers the Python entry point ``sign_prekey_proof`` and the
``sign-prekey-proof`` CLI command (real subprocess). Signing is purely
local: an Ed25519 identity private key seed signs the public
``E2EE-SIGNED-PREKEY-V1`` message, producing exactly the six-string proof
shape the proof query returns and ``verify_prekey_proof`` consumes. No
server is contacted, no backend state is read or written, and the private
seed never appears in the result.
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
                                 sign_prekey_proof, verify_prekey_proof)
from e2ee_backend.service import DeviceService


def _raw_b64(key) -> str:
    raw = key.public_bytes(serialization.Encoding.Raw,
                           serialization.PublicFormat.Raw)
    return base64.b64encode(raw).decode()


def _seed_b64(private) -> str:
    return base64.b64encode(private.private_bytes(
        serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
        serialization.NoEncryption())).decode()


def _new_identity():
    private = ed25519.Ed25519PrivateKey.generate()
    return private, _raw_b64(private.public_key())


def _new_prekey() -> str:
    return _raw_b64(x25519.X25519PrivateKey.generate().public_key())


def _der_b64_public(public) -> str:
    der = public.public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo)
    return base64.b64encode(der).decode()


def _pem_public(public) -> str:
    return public.public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo).decode()


_PROOF_FIELDS = ("user_id", "device_id", "key_id", "public_key",
                 "identity_key", "signature")

#: Repository root, prepended onto PYTHONPATH so subprocesses resolve the
#: package regardless of their working directory.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class SignPrekeyProofCryptoTest(unittest.TestCase):
    def setUp(self) -> None:
        self.private, self.identity = _new_identity()
        self.seed = _seed_b64(self.private)
        self.public_key = _new_prekey()

    _DEFAULT = object()

    def _sign(self, user_id=_DEFAULT, device_id=_DEFAULT, key_id=_DEFAULT,
              public_key=_DEFAULT, private_key=_DEFAULT):
        return sign_prekey_proof(
            "u1" if user_id is self._DEFAULT else user_id,
            "d1" if device_id is self._DEFAULT else device_id,
            "k1" if key_id is self._DEFAULT else key_id,
            self.public_key if public_key is self._DEFAULT else public_key,
            self.seed if private_key is self._DEFAULT else private_key)

    def test_success_returns_six_nonempty_strings(self) -> None:
        proof = self._sign()
        self.assertEqual(set(proof), set(_PROOF_FIELDS))
        for name in _PROOF_FIELDS:
            self.assertIsInstance(proof[name], str)
            self.assertTrue(proof[name])
        # No private material rides along in the result.
        self.assertNotIn("private_key", proof)
        self.assertNotIn(self.seed, json.dumps(proof))

    def test_identifiers_and_public_key_preserved_verbatim(self) -> None:
        user_id = " 用户/alice "
        device_id = " dev / 一 "
        key_id = " 键/k "
        proof = self._sign(user_id=user_id, device_id=device_id,
                           key_id=key_id)
        self.assertEqual(proof["user_id"], user_id)
        self.assertEqual(proof["device_id"], device_id)
        self.assertEqual(proof["key_id"], key_id)
        self.assertEqual(proof["public_key"], self.public_key)

    def test_identity_key_is_canonical_raw_public_point(self) -> None:
        proof = self._sign()
        self.assertEqual(proof["identity_key"], self.identity)
        raw = base64.b64decode(proof["identity_key"], validate=True)
        self.assertEqual(len(raw), 32)
        # Canonical standard base64: re-encoding reproduces the string.
        self.assertEqual(base64.b64encode(raw).decode(), proof["identity_key"])
        self.assertEqual(
            ed25519.Ed25519PublicKey.from_public_bytes(raw),
            self.private.public_key())

    def test_signature_is_canonical_base64_64_bytes(self) -> None:
        proof = self._sign()
        raw = base64.b64decode(proof["signature"], validate=True)
        self.assertEqual(len(raw), 64)
        self.assertEqual(base64.b64encode(raw).decode(), proof["signature"])

    def test_signature_follows_public_protocol_and_is_deterministic(self) -> None:
        # Independent reference construction over the documented message must
        # produce byte-identical output (Ed25519 is deterministic).
        message = signed_prekey_proof_message(
            "u1", "d1", "k1", self.public_key)
        self.assertEqual(
            message,
            b"E2EE-SIGNED-PREKEY-V1\n"
            + json.dumps(
                {"device_id": "d1", "key_id": "k1",
                 "public_key": self.public_key, "user_id": "u1"},
                sort_keys=True, separators=(",", ":"),
                ensure_ascii=False).encode("utf-8"))
        expected = base64.b64encode(self.private.sign(message)).decode()
        first = self._sign()
        self.assertEqual(first["signature"], expected)
        # Repeated calls with the same inputs return an identical object.
        self.assertEqual(self._sign(), first)
        self.assertEqual(self._sign(), self._sign())

    def test_manual_verification_accepts_proof(self) -> None:
        proof = self._sign(user_id=" 用户 ", device_id="a/b", key_id=" k / 1 ")
        identity = ed25519.Ed25519PublicKey.from_public_bytes(
            base64.b64decode(proof["identity_key"]))
        identity.verify(
            base64.b64decode(proof["signature"]),
            signed_prekey_proof_message(proof["user_id"], proof["device_id"],
                                        proof["key_id"], proof["public_key"]))

    def test_verify_prekey_proof_accepts_generated_proof(self) -> None:
        proof = self._sign()
        result = verify_prekey_proof(
            proof, "u1", "d1", "k1", identity_fingerprint(self.identity))
        self.assertEqual(result["fingerprint"],
                         identity_fingerprint(self.identity))
        for name in _PROOF_FIELDS:
            self.assertEqual(result[name], proof[name])

    def test_inputs_are_not_modified(self) -> None:
        user_id, device_id, key_id = "u1", "d1", "k1"
        snapshot = (user_id, device_id, key_id, self.public_key, self.seed)
        self._sign()
        self.assertEqual(
            (user_id, device_id, key_id, self.public_key, self.seed),
            snapshot)

    def test_accepted_public_key_encodings(self) -> None:
        x_public = x25519.X25519PublicKey.from_public_bytes(
            base64.b64decode(self.public_key))
        ed_public = ed25519.Ed25519PrivateKey.generate().public_key()
        raw_ed = _raw_b64(ed_public)
        for value, point in (
                (self.public_key, x_public),                       # raw b64
                (base64.b64encode(
                    x_public.public_bytes(
                        serialization.Encoding.Raw,
                        serialization.PublicFormat.Raw)).decode(),
                 x_public),                                        # raw again
                (x_public.public_bytes(
                    serialization.Encoding.Raw,
                    serialization.PublicFormat.Raw).hex(),
                 x_public),                                        # raw hex
                (_der_b64_public(x_public), x_public),             # DER b64
                (_pem_public(x_public), x_public),                 # PEM
                (raw_ed, ed_public),                               # Ed25519 raw
                (_der_b64_public(ed_public), ed_public)):          # Ed25519 DER
            with self.subTest(value=value[:20]):
                proof = self._sign(public_key=value)
                # The literal spelling is what gets signed and frozen.
                self.assertEqual(proof["public_key"], value)
                verify_prekey_proof(
                    proof, "u1", "d1", "k1",
                    identity_fingerprint(self.identity))

    def test_verified_registration_and_replenishment_accept_proofs(self) -> None:
        service = DeviceService()
        proof = self._sign()
        service.register_verified({
            "user_id": "u1", "device_id": "d1",
            "identity_key": proof["identity_key"],
            "signed_prekeys": [{
                "key_id": "k1", "public_key": proof["public_key"],
                "signature": proof["signature"]}],
        })
        frozen = service.get_prekey_proof("d1", "k1")
        self.assertEqual(frozen, proof)

        # A locally signed proof also drives the replenishment entry, and the
        # query-returned proof is byte-for-byte the locally prepared material.
        k2 = _new_prekey()
        proof_k2 = self._sign(key_id="k2", public_key=k2)
        _, status = service.add_prekey_verified("d1", {
            "key_id": "k2", "public_key": k2,
            "signature": proof_k2["signature"]})
        self.assertEqual(status, 201)
        self.assertEqual(service.get_prekey_proof("d1", "k2"), proof_k2)
        # The public six fields are directly submittable as one object too.
        verify_prekey_proof(
            service.get_prekey_proof("d1", "k2"), "u1", "d1", "k2",
            identity_fingerprint(self.identity))

    def test_wrong_seed_gives_unverifiable_proof_but_distinct_identity(self) -> None:
        # Sanity: a different seed yields a different identity key and the
        # proof only verifies under that seed's fingerprint.
        other_private, other_identity = _new_identity()
        proof = self._sign(private_key=_seed_b64(other_private))
        self.assertEqual(proof["identity_key"], other_identity)
        with self.assertRaises(CryptoError) as ctx:
            verify_prekey_proof(proof, "u1", "d1", "k1",
                                identity_fingerprint(self.identity))
        self.assertEqual(ctx.exception.field, "expected_fingerprint")
        verify_prekey_proof(proof, "u1", "d1", "k1",
                            identity_fingerprint(other_identity))

    def test_all_zero_seed_is_a_valid_ed25519_seed(self) -> None:
        proof = self._sign(private_key=base64.b64encode(b"\x00" * 32).decode())
        identity = ed25519.Ed25519PublicKey.from_public_bytes(
            base64.b64decode(proof["identity_key"]))
        identity.verify(
            base64.b64decode(proof["signature"]),
            signed_prekey_proof_message("u1", "d1", "k1", self.public_key))

    # -- error handling -----------------------------------------------------

    def _assert_field(self, field, **kwargs) -> None:
        with self.assertRaises(CryptoError) as ctx:
            self._sign(**kwargs)
        self.assertEqual(ctx.exception.field, field)

    def test_missing_empty_or_wrong_typed_inputs_name_first_field(self) -> None:
        for bad in (None, "", 0, 7, True, False, b"u1", ["u1"], {"u": 1}):
            self._assert_field("user_id", user_id=bad)
            self._assert_field("device_id", device_id=bad)
            self._assert_field("key_id", key_id=bad)
            self._assert_field("public_key", public_key=bad)
            self._assert_field("private_key", private_key=bad)

    def test_unencodable_utf8_names_that_identifier(self) -> None:
        surrogate = "ud800\ud800"
        self._assert_field("user_id", user_id=surrogate)
        self._assert_field("device_id", device_id=surrogate)
        self._assert_field("key_id", key_id=surrogate)
        # A valid public key with an unencodable identifier still reports the
        # identifier, not the key.
        self._assert_field("key_id", key_id=surrogate,
                           private_key=self.seed)

    def test_invalid_public_key_names_public_key(self) -> None:
        for bad in ("not-a-key", "!!!!", self.public_key[:-3] + "!!!",
                    "   ", "\n\t"):
            self._assert_field("public_key", public_key=bad)

    def test_private_key_wrong_formats_rejected(self) -> None:
        # PKCS#8 DER (48 bytes) and PEM text are *key formats*, not the raw
        # 32-byte seed: both must be refused.
        pkcs8_der = base64.b64encode(self.private.private_bytes(
            serialization.Encoding.DER, serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption())).decode()
        pkcs8_pem = self.private.private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption()).decode()
        for bad in (
                pkcs8_der, pkcs8_pem,
                base64.b64encode(b"\x00" * 31).decode(),   # too short
                base64.b64encode(b"\x00" * 33).decode(),   # too long
                base64.b64encode(b"\x00" * 64).decode(),   # expanded scalar
                "not base64!", "--__",                     # urlsafe alphabet
                self.seed + " ", " " + self.seed,          # whitespace
                self.seed + "\n",
                self.seed.rstrip("="),                     # missing padding
                self.seed + "="):                          # extra padding
            with self.subTest(bad=bad[:16]):
                self._assert_field("private_key", private_key=bad)

    def test_private_key_urlsafe_alphabet_rejected(self) -> None:
        # Build a 32-byte seed whose standard spelling contains '+'/'.', then
        # replace them with URL-safe '-'/'_': same bytes intent, refused.
        for _ in range(64):
            raw = os.urandom(32)
            standard = base64.b64encode(raw).decode()
            if "+" in standard or "/" in standard:
                urlsafe = standard.replace("+", "-").replace("/", "_")
                self._assert_field("private_key", private_key=urlsafe)
                # The standard spelling itself succeeds.
                self.assertEqual(self._sign(private_key=standard)[
                    "signature"], self._sign(private_key=standard)["signature"])
                return
        self.skipTest("no sampled seed needed urlsafe characters")

    def test_private_key_hex_seed_refused(self) -> None:
        # 64 hex chars decode under base64 too, but never to 32 bytes (48),
        # so a hex seed spelling is refused rather than misread.
        raw = base64.b64decode(self.seed)
        self._assert_field("private_key", private_key=raw.hex())

    def test_errors_reported_in_input_order(self) -> None:
        # Everything wrong -> user_id first.
        self._assert_field(
            "user_id", user_id="", device_id="", key_id="",
            public_key="bad", private_key="bad")
        # user_id valid -> device_id, then key_id, then public_key, then key.
        self._assert_field("device_id", user_id="u", device_id="",
                           key_id="", public_key="bad", private_key="bad")
        self._assert_field("key_id", user_id="u", device_id="d", key_id="",
                           public_key="bad", private_key="bad")
        self._assert_field("public_key", user_id="u", device_id="d",
                           key_id="k", public_key="bad",
                           private_key="bad")
        self._assert_field("private_key", user_id="u", device_id="d",
                           key_id="k", public_key=self.public_key,
                           private_key="bad")

    def test_errors_hide_underlying_exception_and_private_value(self) -> None:
        secret = self.seed
        for kwargs in (
                dict(private_key=secret.rstrip("=")),
                dict(private_key="not base64!"),
                dict(public_key="bad", private_key="not base64!")):
            with self.assertRaises(CryptoError) as ctx:
                self._sign(**kwargs)
            self.assertNotIn(secret, ctx.exception.message)
            self.assertNotIn("Traceback", ctx.exception.message)


class SignPrekeyProofCLITest(unittest.TestCase):
    def setUp(self) -> None:
        self.private, self.identity = _new_identity()
        self.seed = _seed_b64(self.private)
        self.public_key = _new_prekey()
        self.fingerprint = identity_fingerprint(self.identity)
        self.env = dict(os.environ, PYTHONIOENCODING="utf-8")
        existing_pythonpath = os.environ.get("PYTHONPATH", "")
        self.env["PYTHONPATH"] = (
            _REPO_ROOT + (os.pathsep + existing_pythonpath
                          if existing_pythonpath else ""))

    def _run(self, *arguments: str, text: bool = True):
        return subprocess.run(
            [sys.executable, "-m", "e2ee_backend", "sign-prekey-proof",
             *arguments],
            capture_output=True, env=self.env, timeout=15, text=text)

    def _good_args(self, user_id="u1", device_id="d1", key_id="k1",
                   public_key=None, private_key=None):
        return ("--user-id", user_id, "--device-id", device_id,
                "--key-id", key_id,
                "--public-key",
                self.public_key if public_key is None else public_key,
                "--private-key", self.seed if private_key is None
                else private_key)

    def test_success_single_line_six_fields(self) -> None:
        result = self._run(*self._good_args())
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        line = result.stdout.rstrip("\n")
        self.assertEqual(line.count("\n"), 0)
        body = json.loads(line)
        self.assertEqual(body, {
            "user_id": "u1", "device_id": "d1", "key_id": "k1",
            "public_key": self.public_key, "identity_key": self.identity,
            "signature": sign_prekey_proof(
                "u1", "d1", "k1", self.public_key, self.seed)["signature"]})

    def test_non_ascii_and_spaces_passthrough_unescaped(self) -> None:
        # Bytes capture: the contract promises the literal UTF-8 text, not
        # JSON \u escapes.
        result = self._run(*self._good_args(
            user_id=" 用户/alice ", device_id=" dev/一 ", key_id=" 键/k "),
            text=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, b"")
        raw = result.stdout.rstrip(b"\n")
        self.assertEqual(raw.count(b"\n"), 0)
        self.assertIn(" 用户/alice ".encode("utf-8"), raw)
        self.assertIn("一".encode("utf-8"), raw)
        body = json.loads(raw.decode("utf-8"))
        self.assertEqual(body["user_id"], " 用户/alice ")
        self.assertEqual(body["device_id"], " dev/一 ")
        self.assertEqual(body["key_id"], " 键/k ")

    def test_two_invocations_are_identical(self) -> None:
        first = self._run(*self._good_args())
        second = self._run(*self._good_args())
        self.assertEqual(first.stdout, second.stdout)

    def test_output_verifies_with_verify_prekey_proof_cli(self) -> None:
        signed = self._run(*self._good_args())
        proof = json.loads(signed.stdout)
        verified = subprocess.run(
            [sys.executable, "-m", "e2ee_backend", "verify-prekey-proof",
             "--proof", json.dumps(proof),
             "--user-id", "u1", "--device-id", "d1", "--key-id", "k1",
             "--expected-fingerprint", self.fingerprint],
            capture_output=True, env=self.env, timeout=15, text=True)
        self.assertEqual(verified.returncode, 0, verified.stderr)
        self.assertEqual(verified.stderr, "")
        body = json.loads(verified.stdout)
        self.assertEqual(body["fingerprint"], self.fingerprint)
        self.assertEqual(body["signature"], proof["signature"])

    def _run_error(self, *arguments: str) -> dict:
        result = self._run(*arguments)
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertNotIn("Traceback", result.stderr)
        line = result.stderr.rstrip("\n")
        self.assertEqual(line.count("\n"), 0)
        body = json.loads(line)
        self.assertEqual(set(body), {"message", "field"})
        return body

    def test_missing_each_option_names_its_field(self) -> None:
        for option, field in (("--user-id", "user_id"),
                              ("--device-id", "device_id"),
                              ("--key-id", "key_id"),
                              ("--public-key", "public_key"),
                              ("--private-key", "private_key")):
            args = list(self._good_args())
            index = args.index(option)
            del args[index:index + 2]
            with self.subTest(option=option):
                body = self._run_error(*args)
                self.assertEqual(body["field"], field)

    def test_empty_value_names_its_field(self) -> None:
        for option, field in (("--user-id", "user_id"),
                              ("--device-id", "device_id"),
                              ("--key-id", "key_id"),
                              ("--public-key", "public_key"),
                              ("--private-key", "private_key")):
            args = list(self._good_args())
            index = args.index(option)
            args[index + 1] = ""
            with self.subTest(option=option):
                body = self._run_error(*args)
                self.assertEqual(body["field"], field)

    def test_bad_public_and_private_keys(self) -> None:
        body = self._run_error(*self._good_args(public_key="not-a-key"))
        self.assertEqual(body["field"], "public_key")
        for bad in ("not base64!", self.seed.rstrip("="), self.seed + " ",
                    base64.b64encode(b"\x00" * 31).decode()):
            body = self._run_error(
                *self._good_args(public_key=self.public_key,
                                 private_key=bad))
            self.assertEqual(body["field"], "private_key")

    def test_multiple_errors_report_first_in_input_order(self) -> None:
        body = self._run_error(
            "--user-id", "", "--device-id", "", "--key-id", "",
            "--public-key", "bad", "--private-key", "bad")
        self.assertEqual(body["field"], "user_id")
        body = self._run_error(
            "--user-id", "u", "--device-id", "d", "--key-id", "k",
            "--public-key", "bad", "--private-key", "bad")
        self.assertEqual(body["field"], "public_key")

    def test_error_message_never_echoes_private_key(self) -> None:
        body = self._run_error(*self._good_args(private_key=self.seed + " "))
        self.assertNotIn(self.seed, body["message"])
        body = self._run_error(*self._good_args(
            private_key="top-secret-value"))
        self.assertNotIn("top-secret-value", body["message"])

    def test_at_prefix_is_direct_text_not_file_indirection(self) -> None:
        # Every option is a direct text value: an @path-looking private key is
        # validated as base64 (and rejected), never opened.
        body = self._run_error(*self._good_args(
            private_key="@/nonexistent/seed.b64"))
        self.assertEqual(body["field"], "private_key")
        # @-text identifiers are likewise taken literally and signed as-is.
        result = self._run(*self._good_args(user_id="@alice"))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["user_id"], "@alice")

    def test_success_does_not_touch_filesystem_state(self) -> None:
        # Signing in a fresh empty directory leaves no artifacts behind.
        directory = tempfile.mkdtemp()
        before = set(os.listdir(directory))
        result = subprocess.run(
            [sys.executable, "-m", "e2ee_backend", "sign-prekey-proof",
             *self._good_args()],
            capture_output=True, env=self.env, timeout=15, text=True,
            cwd=directory)
        try:
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(set(os.listdir(directory)), before)
        finally:
            os.rmdir(directory)


if __name__ == "__main__":
    unittest.main()
