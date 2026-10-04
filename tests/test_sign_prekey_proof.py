"""Tests for local signed pre-key proof generation.

Covers the Python entry point ``sign_prekey_proof`` and the
``sign-prekey-proof`` CLI command (real subprocess). Generation is purely
local: it uses a canonical standard-base64 raw 32-byte Ed25519 identity
private key seed to sign the public E2EE-SIGNED-PREKEY-V1 message and returns
the same six-field proof object the proof query returns. The produced proof
must verify with ``verify_prekey_proof`` and be publishable verbatim through
the verified registration and verified pre-key entry points (service and
real loopback HTTP). No server is contacted, no backend state is touched and
neither the private seed nor the result is ever persisted.
"""
import base64
import json
import subprocess
import sys
import threading
import unittest
from http.client import HTTPConnection

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, x25519

from e2ee_backend.crypto import (CryptoError, identity_fingerprint,
                                 sign_prekey_proof, verify_prekey_proof)
from e2ee_backend.http_app import create_server
from e2ee_backend.service import DeviceService


def _raw_b64(key) -> str:
    raw = key.public_bytes(serialization.Encoding.Raw,
                           serialization.PublicFormat.Raw)
    return base64.b64encode(raw).decode()


def _seed_b64(private) -> str:
    """Canonical standard base64 of an Ed25519 private key's 32-byte seed."""
    seed = private.private_bytes(
        serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
        serialization.NoEncryption())
    return base64.b64encode(seed).decode()


def _new_identity():
    private = ed25519.Ed25519PrivateKey.generate()
    return private, _raw_b64(private.public_key()), _seed_b64(private)


def _new_prekey() -> str:
    return _raw_b64(x25519.X25519PrivateKey.generate().public_key())


def _der_b64_x25519(raw_b64: str) -> str:
    public = x25519.X25519PublicKey.from_public_bytes(
        base64.b64decode(raw_b64))
    der = public.public_bytes(serialization.Encoding.DER,
                              serialization.PublicFormat.SubjectPublicKeyInfo)
    return base64.b64encode(der).decode()


_SIX_FIELDS = ("user_id", "device_id", "key_id", "public_key",
               "identity_key", "signature")


class SignPrekeyProofCryptoTest(unittest.TestCase):
    def setUp(self) -> None:
        self.private, self.identity, self.seed = _new_identity()
        self.public_key = _new_prekey()
        self.fingerprint = identity_fingerprint(self.identity)

    def _sign(self, user_id="u1", device_id="d1", key_id="k1",
              public_key=None, private_key=None):
        return sign_prekey_proof(
            user_id, device_id, key_id,
            self.public_key if public_key is None else public_key,
            self.seed if private_key is None else private_key)

    def _assert_field(self, field, **kwargs) -> None:
        with self.assertRaises(CryptoError) as ctx:
            self._sign(**kwargs)
        self.assertEqual(ctx.exception.field, field)

    def test_success_returns_exactly_six_fields(self):
        proof = self._sign()
        self.assertEqual(set(proof), set(_SIX_FIELDS))
        self.assertNotIn("private_key", proof)
        for name in _SIX_FIELDS:
            self.assertIsInstance(proof[name], str)
            self.assertTrue(proof[name])

    def test_identity_key_is_canonical_base64_raw_32_byte_point(self):
        proof = self._sign()
        self.assertEqual(proof["identity_key"], self.identity)
        raw = base64.b64decode(proof["identity_key"], validate=True)
        self.assertEqual(len(raw), 32)
        # Canonical spelling: re-encoding reproduces the string exactly.
        self.assertEqual(
            base64.b64encode(raw).decode(), proof["identity_key"])

    def test_signature_is_canonical_base64_64_bytes(self):
        proof = self._sign()
        raw = base64.b64decode(proof["signature"], validate=True)
        self.assertEqual(len(raw), 64)
        self.assertEqual(base64.b64encode(raw).decode(), proof["signature"])

    def test_signing_is_deterministic(self):
        first = self._sign()
        second = self._sign()
        self.assertEqual(first, second)

    def test_identifiers_and_public_key_preserved_verbatim(self):
        user_id = " 用户/张三 "
        device_id = " dev/01 "
        key_id = " 键/k "
        public_key = _new_prekey()
        proof = sign_prekey_proof(user_id, device_id, key_id, public_key,
                                  self.seed)
        self.assertEqual(proof["user_id"], user_id)
        self.assertEqual(proof["device_id"], device_id)
        self.assertEqual(proof["key_id"], key_id)
        self.assertEqual(proof["public_key"], public_key)

    def test_der_and_pem_spellings_of_public_key_preserved(self):
        for spelling in (_der_b64_x25519(self.public_key),):
            proof = self._sign(public_key=spelling)
            self.assertEqual(proof["public_key"], spelling)
        pem = x25519.X25519PublicKey.from_public_bytes(
            base64.b64decode(self.public_key)).public_bytes(
                serialization.Encoding.PEM,
                serialization.PublicFormat.SubjectPublicKeyInfo).decode()
        proof = self._sign(public_key=pem)
        self.assertEqual(proof["public_key"], pem)

    def test_proof_verifies_with_verify_prekey_proof(self):
        proof = self._sign()
        result = verify_prekey_proof(proof, "u1", "d1", "k1",
                                     self.fingerprint)
        for name in _SIX_FIELDS:
            self.assertEqual(result[name], proof[name])
        self.assertEqual(result["fingerprint"], self.fingerprint)

    def test_verbatim_unicode_identifiers_round_trip(self):
        user_id = " 用户/alice "
        device_id = "dev / 1"
        key_id = "键\tk"
        proof = sign_prekey_proof(user_id, device_id, key_id,
                                  self.public_key, self.seed)
        result = verify_prekey_proof(proof, user_id, device_id, key_id,
                                     identity_fingerprint(proof["identity_key"]))
        self.assertEqual(result["user_id"], user_id)
        self.assertEqual(result["device_id"], device_id)
        self.assertEqual(result["key_id"], key_id)

    def test_proof_rejected_for_wrong_fingerprint(self):
        proof = self._sign()
        _, other_identity, _ = _new_identity()
        with self.assertRaises(CryptoError) as ctx:
            verify_prekey_proof(proof, "u1", "d1", "k1",
                                identity_fingerprint(other_identity))
        self.assertEqual(ctx.exception.field, "expected_fingerprint")

    def test_interop_with_verified_registration_service(self):
        service = DeviceService()
        proof = self._sign()
        body = service.register_verified({
            "user_id": "u1", "device_id": "d1",
            "identity_key": proof["identity_key"],
            "signed_prekeys": [{
                "key_id": proof["key_id"],
                "public_key": proof["public_key"],
                "signature": proof["signature"]}],
        })
        self.assertIn("registered_at", body)
        stored = service.get_prekey_proof("d1", "k1")
        self.assertEqual(stored, proof)

    def test_interop_with_add_prekey_verified_service(self):
        # Ordinary registration first (no proofs), then the locally signed
        # proof is submitted to the verified single-item replenishment.
        service = DeviceService()
        service.register({
            "user_id": "u1", "device_id": "d1",
            "identity_key": self.identity, "signed_prekeys": []})
        proof = self._sign(key_id="k9")
        body, status = service.add_prekey_verified("d1", {
            "key_id": proof["key_id"],
            "public_key": proof["public_key"],
            "signature": proof["signature"]})
        self.assertEqual(status, 201)
        self.assertEqual(service.get_prekey_proof("d1", "k9"), proof)
        # Idempotent replay with the same locally produced proof.
        _, status = service.add_prekey_verified("d1", {
            "key_id": proof["key_id"],
            "public_key": proof["public_key"],
            "signature": proof["signature"]})
        self.assertEqual(status, 200)

    def test_interop_with_verified_batch_service(self):
        service = DeviceService()
        service.register({
            "user_id": "u1", "device_id": "d1",
            "identity_key": self.identity, "signed_prekeys": []})
        proofs = [self._sign(key_id=f"k{i}") for i in range(3)]
        body, status = service.add_prekeys_verified_batch("d1", {
            "signed_prekeys": [{
                "key_id": p["key_id"], "public_key": p["public_key"],
                "signature": p["signature"]} for p in proofs]})
        self.assertEqual(status, 201)
        for proof in proofs:
            self.assertEqual(
                service.get_prekey_proof("d1", proof["key_id"]), proof)

    def test_empty_or_wrong_type_inputs_name_first_field_in_order(self):
        bad_values = (None, "", 7, True, b"u1", ["u1"])

        def expect(field, **values):
            kwargs = {"user_id": "u1", "device_id": "d1", "key_id": "k1",
                      "public_key": self.public_key,
                      "private_key": self.seed}
            kwargs.update(values)
            with self.assertRaises(CryptoError) as ctx:
                sign_prekey_proof(**kwargs)
            self.assertEqual(ctx.exception.field, field)

        for bad in bad_values:
            expect("user_id", user_id=bad)
            expect("device_id", device_id=bad)
            expect("key_id", key_id=bad)
            expect("public_key", public_key=bad)
            expect("private_key", private_key=bad)

    def test_multiple_errors_report_first_in_input_order(self):
        # All five invalid: user_id wins.
        self._assert_field(
            "user_id", user_id="", device_id="", key_id="",
            public_key="", private_key="")
        # From device_id onward invalid: device_id wins.
        self._assert_field(
            "device_id", user_id="u1", device_id="", key_id="",
            public_key="bad", private_key="bad")
        # public_key invalid but private_key also invalid: public_key wins.
        self._assert_field(
            "public_key", user_id="u1", device_id="d1", key_id="k1",
            public_key="not-a-key", private_key="not-a-key")

    def test_unencodable_utf8_identifier_names_that_field(self):
        lone_surrogate = "ud\ud800"
        self._assert_field("user_id", user_id=lone_surrogate)
        self._assert_field("device_id", user_id="u1",
                           device_id=lone_surrogate)
        self._assert_field("key_id", user_id="u1", device_id="d1",
                           key_id=lone_surrogate)
        # public_key that is valid UTF-8 text but not a key is a public_key
        # error, regardless of a valid private key.
        self._assert_field("public_key", user_id="u1", device_id="d1",
                           key_id="k1", public_key="not-a-key")

    def test_invalid_public_key_names_public_key(self):
        for bad in ("not-a-key", "!!!!", "AAAA",
                    "x" * 64, base64.b64encode(b"\x01\x02\x03").decode()):
            self._assert_field("public_key", public_key=bad)

    # -- private key encoding strictness ----------------------------------

    def test_private_key_rejects_other_formats(self):
        pem = self.private.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption()).decode()
        der = self.private.private_bytes(
            serialization.Encoding.DER,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption())
        for bad in (pem, base64.b64encode(der).decode()):
            self._assert_field("private_key", private_key=bad)

    def test_private_key_rejects_wrong_length(self):
        for length in (0, 1, 31, 33, 64):
            self._assert_field(
                "private_key",
                private_key=base64.b64encode(b"\x00" * length).decode())

    def test_private_key_rejects_non_standard_alphabet_and_padding(self):
        # 32 0xff bytes encode to standard base64 starting with "//"; the
        # URL-safe spelling ("__") is a different alphabet and is refused.
        standard = base64.b64encode(b"\xff" * 32).decode()
        self.assertTrue(standard.startswith("//"))
        urlsafe = base64.urlsafe_b64encode(b"\xff" * 32).decode()
        unpadded = self.seed.rstrip("=")
        # A canonical spelling for 32 zero bytes, used to prove a changed
        # letter case is a non-canonical spelling.
        zero_canonical = base64.b64encode(b"\x00" * 32).decode()
        for bad in (
                urlsafe,                       # '-' / '_' alphabet
                self.seed + " ",               # trailing whitespace
                " " + self.seed,               # leading whitespace
                self.seed + "\n",              # newline
                unpadded,                      # canonical padding required
                unpadded + "==",               # wrong padding amount
                "*" * 43,                      # illegal characters
                zero_canonical.lower(),        # lowercased spelling
        ):
            self._assert_field("private_key", private_key=bad)

    def test_private_key_canonical_spelling_accepted(self):
        # 32 zero bytes: canonical standard base64 is a valid Ed25519 seed.
        zero_seed = base64.b64encode(b"\x00" * 32).decode()
        proof = self._sign(private_key=zero_seed)
        self.assertEqual(
            proof["identity_key"],
            _raw_b64(ed25519.Ed25519PrivateKey.from_private_bytes(
                b"\x00" * 32).public_key()))

    def test_error_message_does_not_echo_private_key(self):
        secret = "AAAA" + self.seed[4:]
        try:
            self._sign(private_key=secret + " ")
        except CryptoError as error:
            self.assertNotIn(secret, error.message)
            self.assertNotIn(self.seed, error.message)
        else:
            self.fail("expected CryptoError")

    def test_inputs_not_modified(self):
        snapshot = (self.seed,)
        self._sign()
        self.assertEqual((self.seed,), snapshot)


class SignPrekeyProofInteropHTTPTest(unittest.TestCase):
    """The locally produced proof publishes through the real HTTP routes."""

    def setUp(self) -> None:
        self.private, self.identity, self.seed = _new_identity()
        self.server, _service = create_server(
            "127.0.0.1", 0, DeviceService())
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def _request(self, method: str, path: str, body=None):
        connection = HTTPConnection("127.0.0.1",
                                    self.server.server_address[1],
                                    timeout=10)
        data = json.dumps(body).encode() if body is not None else None
        headers = {}
        if data is not None:
            headers["Content-Type"] = "application/json"
        connection.request(method, path, body=data, headers=headers)
        response = connection.getresponse()
        payload = json.loads(response.read().decode())
        connection.close()
        return response.status, payload

    def test_register_verified_then_proof_query_and_local_verify(self):
        public_key = _new_prekey()
        proof = sign_prekey_proof("u1", "d1", "k1", public_key, self.seed)
        status, body = self._request(
            "POST", "/v1/devices/verified", {
                "user_id": "u1", "device_id": "d1",
                "identity_key": proof["identity_key"],
                "signed_prekeys": [{
                    "key_id": proof["key_id"],
                    "public_key": proof["public_key"],
                    "signature": proof["signature"]}]})
        self.assertEqual(status, 201, body)

        status, stored = self._request(
            "GET", "/v1/devices/d1/prekeys/k1/proof")
        self.assertEqual(status, 200, stored)
        self.assertEqual(stored, proof)

        result = verify_prekey_proof(
            stored, "u1", "d1", "k1",
            identity_fingerprint(self.identity))
        self.assertEqual(result["signature"], proof["signature"])

    def test_add_prekey_verified_route_accepts_local_proof(self):
        # Register ordinarily with the same identity, then sign locally and
        # publish through the single-item verified replenishment route.
        status, _ = self._request(
            "POST", "/v1/devices", {
                "user_id": "u1", "device_id": "d2",
                "identity_key": self.identity, "signed_prekeys": []})
        self.assertEqual(status, 201)
        public_key = _new_prekey()
        proof = sign_prekey_proof("u1", "d2", "kA", public_key, self.seed)
        status, body = self._request(
            "POST", "/v1/devices/d2/prekeys/verified", {
                "key_id": proof["key_id"],
                "public_key": proof["public_key"],
                "signature": proof["signature"]})
        self.assertEqual(status, 201, body)
        status, stored = self._request(
            "GET", "/v1/devices/d2/prekeys/kA/proof")
        self.assertEqual(status, 200, stored)
        self.assertEqual(stored, proof)


class SignPrekeyProofCLITest(unittest.TestCase):
    def setUp(self) -> None:
        self.private, self.identity, self.seed = _new_identity()
        self.public_key = _new_prekey()
        self.fingerprint = identity_fingerprint(self.identity)

    def _run(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "e2ee_backend", "sign-prekey-proof",
             *arguments],
            capture_output=True, text=True, timeout=15)

    def _good_args(self, user_id="u1", device_id="d1", key_id="k1",
                   public_key=None, private_key=None):
        return ("--user-id", user_id, "--device-id", device_id,
                "--key-id", key_id,
                "--public-key", self.public_key
                if public_key is None else public_key,
                "--private-key", self.seed if private_key is None
                else private_key)

    def test_success_stdout_single_line_json_exit0(self):
        result = self._run(*self._good_args())
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        line = result.stdout.strip()
        self.assertEqual(line.count("\n"), 0)
        body = json.loads(line)
        self.assertEqual(set(body), set(_SIX_FIELDS))
        self.assertEqual(body["user_id"], "u1")
        self.assertEqual(body["device_id"], "d1")
        self.assertEqual(body["key_id"], "k1")
        self.assertEqual(body["public_key"], self.public_key)
        self.assertEqual(body["identity_key"], self.identity)

    def test_cli_output_verifies_locally_and_is_deterministic(self):
        first = self._run(*self._good_args())
        second = self._run(*self._good_args())
        self.assertEqual(first.stdout, second.stdout)
        proof = json.loads(first.stdout)
        result = verify_prekey_proof(proof, "u1", "d1", "k1",
                                     self.fingerprint)
        self.assertEqual(result["fingerprint"], self.fingerprint)

    def test_verbatim_identifiers_with_spaces_slashes_and_chinese(self):
        args = self._good_args(user_id=" 用户/张三 ", device_id=" dev/01 ",
                               key_id=" 键/k ")
        result = self._run(*args)
        self.assertEqual(result.returncode, 0, result.stderr)
        body = json.loads(result.stdout)
        self.assertEqual(body["user_id"], " 用户/张三 ")
        self.assertEqual(body["device_id"], " dev/01 ")
        self.assertEqual(body["key_id"], " 键/k ")

    def test_missing_option_names_its_field_exit2(self):
        for option, field in (("--user-id", "user_id"),
                              ("--device-id", "device_id"),
                              ("--key-id", "key_id"),
                              ("--public-key", "public_key"),
                              ("--private-key", "private_key")):
            args = list(self._good_args())
            index = args.index(option)
            del args[index:index + 2]
            result = self._run(*args)
            self.assertEqual(result.returncode, 2, result.stdout)
            self.assertEqual(result.stdout, "")
            self.assertNotIn("Traceback", result.stderr)
            body = json.loads(result.stderr.strip())
            self.assertEqual(set(body), {"message", "field"})
            self.assertEqual(body["field"], field)

    def test_missing_subcommand_and_unknown_option_do_not_traceback(self):
        # Unknown option is argparse's own exit 2; it must not carry a
        # traceback (stdout empty is argparse's own usage-on-stderr layout,
        # which still never mentions the private seed).
        result = self._run("--bogus")
        self.assertEqual(result.returncode, 2)
        self.assertNotIn("Traceback", result.stderr)
        self.assertNotIn(self.seed, result.stderr)

    def test_invalid_inputs_exit2_with_field_json(self):
        cases = [
            (self._good_args(user_id=""), "user_id"),
            (self._good_args(device_id=""), "device_id"),
            (self._good_args(key_id=""), "key_id"),
            (self._good_args(public_key="not-a-key"), "public_key"),
            (self._good_args(private_key="not-base64!"), "private_key"),
            (self._good_args(private_key="AAAA"), "private_key"),
            (self._good_args(private_key=self.seed + " "), "private_key"),
            (self._good_args(
                private_key=base64.urlsafe_b64encode(b"\xff" * 32).decode()),
             "private_key"),
        ]
        for args, field in cases:
            result = self._run(*args)
            self.assertEqual(result.returncode, 2,
                             f"{field}: {result.stdout}")
            self.assertEqual(result.stdout, "")
            self.assertNotIn("Traceback", result.stderr)
            line = result.stderr.strip()
            self.assertEqual(line.count("\n"), 0)
            body = json.loads(line)
            self.assertEqual(set(body), {"message", "field"})
            self.assertEqual(body["field"], field)

    def test_first_invalid_field_wins(self):
        result = self._run(
            "--user-id", "", "--device-id", "", "--key-id", "",
            "--public-key", "", "--private-key", "")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stderr)["field"], "user_id")

    def test_stderr_does_not_echo_private_key(self):
        result = self._run(*self._good_args(private_key=self.seed + "!"))
        self.assertEqual(result.returncode, 2)
        self.assertNotIn(self.seed, result.stderr)

    def test_at_path_is_treated_as_literal_text_not_a_file(self):
        # Direct text values only: an @path is not dereferenced; it reaches
        # the crypto layer verbatim and fails as a private key encoding.
        result = self._run(*self._good_args(private_key="@/tmp/whatever"))
        self.assertEqual(result.returncode, 2)
        body = json.loads(result.stderr)
        self.assertEqual(body["field"], "private_key")
        self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
