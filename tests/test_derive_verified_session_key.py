"""Tests for the initiator-side verified session-key derivation entry point.

Covers the Python entry point ``derive_verified_session_key`` and the
``derive-verified-session-key`` CLI command (real subprocess). The initiator
cross-checks a frozen public eight-field session snapshot against a frozen
signed pre-key proof and a trusted identity fingerprint, then derives the
same X25519+HKDF key the recipient obtains from ``derive_session_key`` with
the matching pre-key private key. The check is purely local: no server is
contacted, no backend state is read or written, and the input objects are
never modified.
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

from e2ee_backend.crypto import (CryptoError, decrypt_message,
                                 derive_session_key,
                                 derive_verified_session_key,
                                 encrypt_message, identity_fingerprint,
                                 signed_prekey_proof_message)


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _raw_b64(key) -> str:
    return _b64(key.public_bytes(serialization.Encoding.Raw,
                                 serialization.PublicFormat.Raw))


def _der_b64(key) -> str:
    return _b64(key.public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo))


def _pem(key) -> str:
    return key.public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo).decode("ascii")


def _x25519_private_b64(private) -> str:
    return _b64(private.private_bytes(
        serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
        serialization.NoEncryption()))


class DeriveVerifiedSessionKeyCryptoTest(unittest.TestCase):
    USER_ID = " 用户/alice "
    DEVICE_ID = "dev / 1"
    KEY_ID = "键\tk"
    SESSION_ID = " 会话-1 "

    def setUp(self) -> None:
        self.identity_private = ed25519.Ed25519PrivateKey.generate()
        self.identity = _raw_b64(self.identity_private.public_key())
        self.fingerprint = identity_fingerprint(self.identity)

        self.prekey_private = x25519.X25519PrivateKey.generate()
        self.prekey_public = _raw_b64(self.prekey_private.public_key())
        self.prekey_private_b64 = _x25519_private_b64(self.prekey_private)

        self.ephemeral_private = x25519.X25519PrivateKey.generate()
        self.ephemeral_public = _raw_b64(self.ephemeral_private.public_key())
        self.ephemeral_private_b64 = _x25519_private_b64(
            self.ephemeral_private)

        self.proof = self._proof(self.prekey_public)
        self.session = self._session(
            self.ephemeral_public, self.prekey_public, self.identity)

    def _proof(self, public_key: str, *, identity: str = None,
               signer=None) -> dict:
        signer = signer or self.identity_private
        identity = self.identity if identity is None else identity
        signature = _b64(signer.sign(signed_prekey_proof_message(
            self.USER_ID, self.DEVICE_ID, self.KEY_ID, public_key)))
        return {
            "user_id": self.USER_ID, "device_id": self.DEVICE_ID,
            "key_id": self.KEY_ID, "public_key": public_key,
            "identity_key": identity, "signature": signature,
        }

    def _session(self, ephemeral_key: str, public_key: str,
                 identity_key: str) -> dict:
        return {
            "session_id": self.SESSION_ID,
            "initiator_device_id": "initiator-device",
            "recipient_device_id": self.DEVICE_ID,
            "prekey_id": self.KEY_ID,
            "ephemeral_key": ephemeral_key,
            "identity_key": identity_key,
            "public_key": public_key,
            "created_at": "2026-01-02T03:04:05Z",
        }

    _UNSET = object()

    def _derive(self, session=_UNSET, proof=_UNSET, private_key=_UNSET,
                user_id=_UNSET, fingerprint=_UNSET):
        return derive_verified_session_key(
            self.session if session is self._UNSET else session,
            self.proof if proof is self._UNSET else proof,
            self.ephemeral_private_b64
            if private_key is self._UNSET else private_key,
            self.USER_ID if user_id is self._UNSET else user_id,
            self.fingerprint if fingerprint is self._UNSET else fingerprint)

    def _recipient_key(self) -> dict:
        return derive_session_key(self.SESSION_ID, self.prekey_private_b64,
                                  self.ephemeral_public)

    def _assert_field(self, field: str, **kwargs) -> None:
        with self.assertRaises(CryptoError) as ctx:
            self._derive(**kwargs)
        self.assertEqual(ctx.exception.field, field)

    def test_success_matches_recipient_derivation(self) -> None:
        result = self._derive()
        self.assertEqual(set(result), {"session_id", "key"})
        self.assertEqual(result["session_id"], self.SESSION_ID)
        self.assertEqual(result, self._recipient_key())
        self.assertEqual(len(base64.b64decode(result["key"])), 32)

    def test_derived_key_interoperates_with_message_crypto(self) -> None:
        key = self._derive()["key"]
        enc = encrypt_message(self.SESSION_ID, key, "hello 世界",
                              sender_device_id="initiator-device",
                              message_id="m1", sequence=1)
        dec = decrypt_message(self.SESSION_ID, key, enc["nonce"],
                              enc["ciphertext"],
                              sender_device_id="initiator-device",
                              message_id="m1", sequence=1)
        self.assertEqual(dec["plaintext"], "hello 世界")
        # The recipient, deriving with the existing entry, decrypts too.
        recipient_key = self._recipient_key()["key"]
        dec = decrypt_message(self.SESSION_ID, recipient_key, enc["nonce"],
                              enc["ciphertext"],
                              sender_device_id="initiator-device",
                              message_id="m1", sequence=1)
        self.assertEqual(dec["plaintext"], "hello 世界")

    def test_extra_fields_ignored_and_inputs_not_modified(self) -> None:
        session = dict(self.session, extra="ignored", nested={"a": 1})
        proof = dict(self.proof, extra="ignored")
        session_before = json.loads(json.dumps(session))
        proof_before = json.loads(json.dumps(proof))
        result = self._derive(session=session, proof=proof)
        self.assertEqual(set(result), {"session_id", "key"})
        self.assertEqual(session, session_before)
        self.assertEqual(proof, proof_before)

    def test_equivalent_identity_encodings_match_by_actual_key(self) -> None:
        expected = self._recipient_key()
        # The snapshot carries the identity key as DER/PEM while the proof
        # carries the raw point (and vice versa): same actual key verifies.
        self.assertEqual(
            self._derive(
                session=self._session(self.ephemeral_public,
                                      self.prekey_public,
                                      _der_b64(self.identity_private
                                               .public_key())))["key"],
            expected["key"])
        self.assertEqual(
            self._derive(
                session=self._session(self.ephemeral_public,
                                      self.prekey_public,
                                      _pem(self.identity_private
                                           .public_key())))["key"],
            expected["key"])
        self.assertEqual(
            self._derive(proof=self._proof(
                self.prekey_public,
                identity=_der_b64(self.identity_private.public_key())))["key"],
            expected["key"])

    def test_equivalent_x25519_encodings_derive_the_same_key(self) -> None:
        # DER spellings of the ephemeral key and pre-key. The signature signs
        # the literal pre-key public_key string, so the DER spelling must be
        # re-signed; the snapshot's public_key stays the proof's exact string.
        prekey_der = _der_b64(self.prekey_private.public_key())
        session = self._session(
            _der_b64(self.ephemeral_private.public_key()), prekey_der,
            self.identity)
        result = self._derive(session=session,
                              proof=self._proof(prekey_der))
        self.assertEqual(result, self._recipient_key())

    def test_historical_materials_derive_with_old_trusted_fingerprint(self) -> None:
        # An identity rotation does not invalidate the frozen proof/snapshot:
        # the old trusted fingerprint still derives; a new identity's
        # fingerprint does not match the old proof.
        self.assertEqual(self._derive(), self._recipient_key())
        new_identity = ed25519.Ed25519PrivateKey.generate().public_key()
        self._assert_field(
            "expected_fingerprint",
            fingerprint=identity_fingerprint(_raw_b64(new_identity)))

    def test_session_must_be_an_object(self) -> None:
        for bad in (None, [], "session", 3, True):
            self._assert_field("session", session=bad)

    def test_missing_or_empty_snapshot_fields_name_the_field(self) -> None:
        fields = ("session_id", "initiator_device_id",
                  "recipient_device_id", "prekey_id", "ephemeral_key",
                  "identity_key", "public_key", "created_at")
        for name in fields:
            snapshot = dict(self.session)
            del snapshot[name]
            self._assert_field(name, session=snapshot)
            for bad in (None, "", 7, True):
                self._assert_field(name, session=dict(self.session, **{name: bad}))

    def test_proof_must_be_an_object(self) -> None:
        for bad in (None, [], "proof", 3, True):
            self._assert_field("proof", proof=bad)

    def test_proof_ownership_mismatches_keep_proof_field_names(self) -> None:
        self._assert_field("user_id", user_id="someone-else")
        self._assert_field(
            "device_id",
            session=self._session(self.ephemeral_public, self.prekey_public,
                                  self.identity) | {"recipient_device_id": "d2"})
        self._assert_field(
            "key_id",
            session=self._session(self.ephemeral_public, self.prekey_public,
                                  self.identity) | {"prekey_id": "k2"})

    def test_fingerprint_errors(self) -> None:
        for bad in (None, "", "zz" * 32, self.fingerprint.upper(),
                    self.fingerprint[:-1], 64):
            self._assert_field("expected_fingerprint", fingerprint=bad)

    def test_signature_errors_are_signature(self) -> None:
        # A validly signed proof for a different identity fails verification.
        other = ed25519.Ed25519PrivateKey.generate()
        self._assert_field(
            "signature",
            proof=self._proof(self.prekey_public, signer=other))
        # A non-64-byte signature is likewise a signature error.
        self._assert_field(
            "signature",
            proof=dict(self.proof, signature=_b64(b"\x00" * 32)))

    def test_proof_public_key_string_mismatch_is_public_key(self) -> None:
        # The proof is internally valid (re-signed over its own public_key
        # string), but that string no longer equals the snapshot's string.
        other_prekey = _raw_b64(x25519.X25519PrivateKey.generate().public_key())
        self._assert_field(
            "public_key", proof=self._proof(other_prekey))

    def test_reencoded_prekey_without_resigning_is_signature(self) -> None:
        # Same pre-key point as DER, but the snapshot keeps the raw string and
        # the proof was not re-signed: the literal-string signature fails.
        proof = dict(self.proof,
                     public_key=_der_b64(self.prekey_private.public_key()))
        self._assert_field("signature", proof=proof)

    def test_identity_key_actual_mismatch_is_identity_key(self) -> None:
        other_private = ed25519.Ed25519PrivateKey.generate()
        other_raw = _raw_b64(other_private.public_key())
        proof = self._proof(self.prekey_public, identity=other_raw,
                            signer=other_private)
        self._assert_field(
            "identity_key", proof=proof,
            fingerprint=identity_fingerprint(other_raw))

    def test_snapshot_identity_key_must_be_ed25519(self) -> None:
        # A DER SubjectPublicKeyInfo carrying X25519 is rejected, as is a
        # string that does not parse.
        x_der = _der_b64(x25519.X25519PrivateKey.generate().public_key())
        self._assert_field(
            "identity_key",
            session=self._session(self.ephemeral_public, self.prekey_public,
                                  x_der))
        self._assert_field(
            "identity_key",
            session=self._session(self.ephemeral_public, self.prekey_public,
                                  "not-a-key"))

    def test_private_key_errors_are_private_key(self) -> None:
        for bad in (None, "", 1, "not base64!!!", _b64(b"short"),
                    _b64(b"x" * 33)):
            self._assert_field("private_key", private_key=bad)

    def test_private_key_mismatching_ephemeral_is_ephemeral_key(self) -> None:
        other = _x25519_private_b64(x25519.X25519PrivateKey.generate())
        self._assert_field("ephemeral_key", private_key=other)

    def test_ephemeral_key_must_be_x25519(self) -> None:
        # A raw 32-byte Ed25519 point is X25519 under the raw-point rule and
        # simply won't match the private key; an algorithm-identified Ed25519
        # key is rejected outright as ephemeral_key.
        ed_der = _der_b64(ed25519.Ed25519PrivateKey.generate().public_key())
        self._assert_field(
            "ephemeral_key",
            session=self._session(ed_der, self.prekey_public, self.identity))

    def test_ephemeral_key_unparsable_is_ephemeral_key(self) -> None:
        self._assert_field(
            "ephemeral_key",
            session=self._session("not-a-key", self.prekey_public,
                                  self.identity))

    def test_prekey_must_be_x25519(self) -> None:
        ed_der = _der_b64(ed25519.Ed25519PrivateKey.generate().public_key())
        session = self._session(self.ephemeral_public, ed_der, self.identity)
        self._assert_field("public_key", session=session,
                           proof=self._proof(ed_der))

    def test_low_order_prekey_is_public_key(self) -> None:
        low_order = _b64(b"\x00" * 32)
        session = self._session(self.ephemeral_public, low_order,
                                self.identity)
        self._assert_field("public_key", session=session,
                           proof=self._proof(low_order))


class DeriveVerifiedSessionKeyCLITest(unittest.TestCase):
    def setUp(self) -> None:
        self.identity_private = ed25519.Ed25519PrivateKey.generate()
        identity = _raw_b64(self.identity_private.public_key())
        self.fingerprint = identity_fingerprint(identity)
        self.prekey_private = x25519.X25519PrivateKey.generate()
        prekey_public = _raw_b64(self.prekey_private.public_key())
        self.ephemeral_private = x25519.X25519PrivateKey.generate()
        ephemeral_public = _raw_b64(self.ephemeral_private.public_key())
        self.ephemeral_private_b64 = _x25519_private_b64(
            self.ephemeral_private)
        self.prekey_private_b64 = _x25519_private_b64(self.prekey_private)
        self.session = {
            "session_id": "sess-cli",
            "initiator_device_id": "ini",
            "recipient_device_id": "d1", "prekey_id": "k1",
            "ephemeral_key": ephemeral_public, "identity_key": identity,
            "public_key": prekey_public,
            "created_at": "2026-01-02T03:04:05Z",
        }
        self.proof = {
            "user_id": "u1", "device_id": "d1", "key_id": "k1",
            "public_key": prekey_public, "identity_key": identity,
            "signature": _b64(self.identity_private.sign(
                signed_prekey_proof_message(
                    "u1", "d1", "k1", prekey_public))),
        }

    def _run(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "e2ee_backend",
             "derive-verified-session-key", *arguments],
            capture_output=True, text=True, timeout=20)

    def _args(self, session_arg=None, proof_arg=None):
        return [
            "--session",
            json.dumps(self.session) if session_arg is None else session_arg,
            "--proof",
            json.dumps(self.proof) if proof_arg is None else proof_arg,
            "--private-key", self.ephemeral_private_b64,
            "--user-id", "u1", "--expected-fingerprint", self.fingerprint,
        ]

    def _assert_error(self, result: subprocess.CompletedProcess,
                      field: str) -> dict:
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertNotIn("Traceback", result.stderr)
        line = result.stderr.strip()
        self.assertEqual(line.count("\n"), 0)
        body = json.loads(line)
        self.assertEqual(set(body), {"message", "field"})
        self.assertEqual(body["field"], field)
        return body

    def test_cli_success_inline_matches_recipient(self) -> None:
        result = self._run(*self._args())
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        line = result.stdout.strip()
        self.assertEqual(line.count("\n"), 0)
        body = json.loads(line)
        self.assertEqual(set(body), {"session_id", "key"})
        recipient = derive_session_key(
            "sess-cli", self.prekey_private_b64,
            self.session["ephemeral_key"])
        self.assertEqual(body, recipient)

    def test_cli_success_with_at_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            session_path = os.path.join(directory, "session.json")
            proof_path = os.path.join(directory, "proof.json")
            with open(session_path, "w", encoding="utf-8") as handle:
                json.dump(self.session, handle)
            with open(proof_path, "w", encoding="utf-8") as handle:
                json.dump(dict(self.proof, extra="ignored"), handle)
            result = self._run(*self._args("@" + session_path,
                                           "@" + proof_path))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["session_id"], "sess-cli")

    def test_cli_missing_option_names_its_field(self) -> None:
        for option, field in (("--session", "session"), ("--proof", "proof"),
                              ("--private-key", "private_key"),
                              ("--user-id", "user_id"),
                              ("--expected-fingerprint",
                               "expected_fingerprint")):
            args = self._args()
            index = args.index(option)
            del args[index:index + 2]
            self._assert_error(self._run(*args), field)

    def test_cli_invalid_json_and_non_objects(self) -> None:
        for option, value, field in (
                ("--session", "{not json", "session"),
                ("--session", "[1,2]", "session"),
                ("--session", '"text"', "session"),
                ("--proof", "{not json", "proof"),
                ("--proof", "42", "proof"),
                ("--proof", "[]", "proof")):
            args = self._args()
            args[args.index(option) + 1] = value
            self._assert_error(self._run(*args), field)

    def test_cli_unreadable_and_non_utf8_files_name_the_object(self) -> None:
        args = self._args()
        args[args.index("--session") + 1] = "@/nonexistent/session.json"
        self._assert_error(self._run(*args), "session")
        args = self._args()
        args[args.index("--proof") + 1] = "@/nonexistent/proof.json"
        self._assert_error(self._run(*args), "proof")
        with tempfile.NamedTemporaryFile(suffix=".json",
                                         delete=False) as handle:
            handle.write(b'{"session_id": "\xff\xfe"}')
            bad_path = handle.name
        try:
            args = self._args(session_arg="@" + bad_path)
            self._assert_error(self._run(*args), "session")
        finally:
            os.unlink(bad_path)

    def test_cli_crypto_failures_keep_field_contract(self) -> None:
        # Wrong private key: the public point does not match ephemeral_key.
        other_private = _x25519_private_b64(x25519.X25519PrivateKey.generate())
        args = self._args()
        args[args.index("--private-key") + 1] = other_private
        self._assert_error(self._run(*args), "ephemeral_key")
        # Wrong trusted fingerprint.
        other_identity = ed25519.Ed25519PrivateKey.generate().public_key()
        args = self._args()
        args[args.index("--expected-fingerprint") + 1] = \
            identity_fingerprint(_raw_b64(other_identity))
        self._assert_error(self._run(*args), "expected_fingerprint")
        # Wrong user id.
        args = self._args()
        args[args.index("--user-id") + 1] = "u2"
        self._assert_error(self._run(*args), "user_id")


if __name__ == "__main__":
    unittest.main()
