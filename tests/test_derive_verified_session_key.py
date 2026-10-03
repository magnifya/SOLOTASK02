"""Tests for the initiator-side verified session-key derivation.

Covers the Python entry point ``derive_verified_session_key`` and the
``derive-verified-session-key`` CLI command (real subprocess). The check is
purely local: a frozen eight-field session snapshot is cross-checked against
a trusted signed pre-key proof, the ephemeral private key must match the
snapshot's ephemeral public key, and only then does the exact
X25519 + HKDF-SHA256 protocol of ``derive_session_key`` run — so the derived
key is identical to what the recipient derives and interoperates with
``encrypt_message``/``decrypt_message``. No server is contacted and no
backend state is read or written.
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
                                 derive_verified_session_key, encrypt_message,
                                 identity_fingerprint,
                                 signed_prekey_proof_message)


def _raw_b64(key) -> str:
    raw = key.public_bytes(serialization.Encoding.Raw,
                           serialization.PublicFormat.Raw)
    return base64.b64encode(raw).decode()


def _private_b64(private) -> str:
    raw = private.private_bytes(serialization.Encoding.Raw,
                                serialization.PrivateFormat.Raw,
                                serialization.NoEncryption())
    return base64.b64encode(raw).decode()


def _new_identity():
    private = ed25519.Ed25519PrivateKey.generate()
    return private, _raw_b64(private.public_key())


def _new_x25519():
    private = x25519.X25519PrivateKey.generate()
    return private, _raw_b64(private.public_key())


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


class DeriveVerifiedSessionKeyCryptoTest(unittest.TestCase):
    def setUp(self) -> None:
        self.id_private, self.identity = _new_identity()
        self.prekey_private, self.prekey_public = _new_x25519()
        self.eph_private, self.eph_public = _new_x25519()
        self.eph_private_b64 = _private_b64(self.eph_private)
        self.prekey_private_b64 = _private_b64(self.prekey_private)
        self.signature = _sign(self.id_private, "u1", "d1", "k1",
                               self.prekey_public)
        self.fingerprint = identity_fingerprint(self.identity)
        self.proof = {
            "user_id": "u1", "device_id": "d1", "key_id": "k1",
            "public_key": self.prekey_public, "identity_key": self.identity,
            "signature": self.signature,
        }
        self.session = {
            "session_id": "s1", "initiator_device_id": "d0",
            "recipient_device_id": "d1", "prekey_id": "k1",
            "ephemeral_key": self.eph_public, "identity_key": self.identity,
            "public_key": self.prekey_public, "created_at": "2026-01-01",
        }

    _UNSET = object()

    def _derive(self, session=_UNSET, proof=_UNSET, private_key=_UNSET,
                user_id="u1", fingerprint=_UNSET):
        return derive_verified_session_key(
            self.session if session is self._UNSET else session,
            self.proof if proof is self._UNSET else proof,
            self.eph_private_b64 if private_key is self._UNSET else private_key,
            user_id,
            self.fingerprint if fingerprint is self._UNSET else fingerprint)

    def _assert_field(self, field, **kwargs) -> None:
        with self.assertRaises(CryptoError) as ctx:
            self._derive(**kwargs)
        self.assertEqual(ctx.exception.field, field)

    def test_success_returns_only_session_id_and_key(self):
        result = self._derive()
        self.assertEqual(set(result), {"session_id", "key"})
        self.assertEqual(result["session_id"], "s1")

    def test_key_matches_existing_protocol_on_both_sides(self):
        result = self._derive()
        initiator = derive_session_key("s1", self.eph_private_b64,
                                       self.prekey_public)
        recipient = derive_session_key("s1", self.prekey_private_b64,
                                       self.eph_public)
        self.assertEqual(result, initiator)
        self.assertEqual(result, recipient)

    def test_derived_key_interoperates_with_message_crypto(self):
        result = self._derive()
        envelope = encrypt_message("s1", result["key"], "hello 世界",
                                   sender_device_id="d0", message_id="m1",
                                   sequence=1)
        opened = decrypt_message("s1", result["key"], envelope["nonce"],
                                 envelope["ciphertext"], sender_device_id="d0",
                                 message_id="m1", sequence=1)
        self.assertEqual(opened["plaintext"], "hello 世界")

    def test_extra_fields_ignored_and_inputs_not_modified(self):
        session = dict(self.session, extra="ignored", nested={"a": 1})
        proof = dict(self.proof, extra="ignored")
        session_snapshot = json.loads(json.dumps(session))
        proof_snapshot = json.loads(json.dumps(proof))
        result = self._derive(session=session, proof=proof)
        self.assertEqual(set(result), {"session_id", "key"})
        self.assertEqual(session, session_snapshot)
        self.assertEqual(proof, proof_snapshot)

    def test_session_not_an_object(self):
        for bad in (None, [], "session", 3, True):
            self._assert_field("session", session=bad)

    def test_missing_or_empty_session_fields_name_the_field(self):
        for name in ("session_id", "initiator_device_id",
                     "recipient_device_id", "prekey_id", "ephemeral_key",
                     "identity_key", "public_key", "created_at"):
            for bad in (None, "", 7, True):
                session = dict(self.session)
                session[name] = bad
                self._assert_field(name, session=session)
            session = dict(self.session)
            del session[name]
            self._assert_field(name, session=session)

    def test_proof_errors_keep_their_fields(self):
        self._assert_field("proof", proof=None)
        self._assert_field("proof", proof=[1, 2])
        other_private, _ = _new_identity()
        proof = dict(self.proof, signature=_sign(
            other_private, "u1", "d1", "k1", self.prekey_public))
        self._assert_field("signature", proof=proof)
        _, other_identity = _new_identity()
        self._assert_field("expected_fingerprint",
                           fingerprint=identity_fingerprint(other_identity))
        self._assert_field("user_id", user_id="u2")

    def test_proof_device_and_prekey_id_checked_against_snapshot(self):
        session = dict(self.session, recipient_device_id="d2")
        self._assert_field("device_id", session=session)
        session = dict(self.session, prekey_id="k2")
        self._assert_field("key_id", session=session)

    def test_proof_public_key_must_equal_snapshot_verbatim(self):
        # Same pre-key point in DER base64: an equivalent encoding, but the
        # snapshot string must match the proof string as-is.
        session = dict(self.session,
                       public_key=_der_b64_x25519(self.prekey_public))
        self._assert_field("public_key", session=session)

    def test_identity_keys_compared_by_actual_key(self):
        # A DER spelling of the same identity key in the snapshot matches.
        session = dict(self.session,
                       identity_key=_der_b64_ed25519(self.identity))
        result = self._derive(session=session)
        self.assertEqual(result["session_id"], "s1")
        # A different identity key does not.
        _, other_identity = _new_identity()
        session = dict(self.session, identity_key=other_identity)
        self._assert_field("identity_key", session=session)
        session = dict(self.session, identity_key="not-a-key")
        self._assert_field("identity_key", session=session)

    def test_private_key_errors(self):
        for bad in (None, "", 7, "not base64!",
                    base64.b64encode(b"\x00" * 16).decode(),
                    self.eph_private_b64.rstrip("=")):
            self._assert_field("private_key", private_key=bad)

    def test_ephemeral_key_invalid_or_mismatched(self):
        for bad in ("not-a-key", _der_b64_ed25519(self.identity)):
            session = dict(self.session, ephemeral_key=bad)
            self._assert_field("ephemeral_key", session=session)
        # A well-formed ephemeral key that the private key does not match.
        _, other_public = _new_x25519()
        session = dict(self.session, ephemeral_key=other_public)
        self._assert_field("ephemeral_key", session=session)
        # Equivalent encodings of the matching key are accepted.
        session = dict(self.session,
                       ephemeral_key=_der_b64_x25519(self.eph_public))
        result = self._derive(session=session)
        self.assertEqual(result["session_id"], "s1")

    def test_prekey_must_be_x25519_and_exchange_must_work(self):
        session = dict(self.session,
                       public_key=_der_b64_ed25519(self.identity))
        proof = dict(self.proof, public_key=session["public_key"],
                     signature=_sign(self.id_private, "u1", "d1", "k1",
                                     session["public_key"]))
        self._assert_field("public_key", session=session, proof=proof)
        # A low-order point makes the X25519 exchange fail.
        low_order = base64.b64encode(b"\x00" * 32).decode()
        session = dict(self.session, public_key=low_order)
        proof = dict(self.proof, public_key=low_order,
                     signature=_sign(self.id_private, "u1", "d1", "k1",
                                     low_order))
        self._assert_field("public_key", session=session, proof=proof)

    def test_historical_material_still_derives(self):
        # Only frozen material is consulted: after an identity rotation the
        # old proof and matching old snapshot still derive with the old
        # trusted fingerprint, while the new fingerprint mismatches.
        _, new_identity = _new_identity()
        result = self._derive()
        self.assertEqual(set(result), {"session_id", "key"})
        self._assert_field("expected_fingerprint",
                           fingerprint=identity_fingerprint(new_identity))


class DeriveVerifiedSessionKeyCLITest(unittest.TestCase):
    def setUp(self) -> None:
        self.id_private, self.identity = _new_identity()
        self.prekey_private, self.prekey_public = _new_x25519()
        self.eph_private, self.eph_public = _new_x25519()
        self.eph_private_b64 = _private_b64(self.eph_private)
        self.signature = _sign(self.id_private, "u1", "d1", "k1",
                               self.prekey_public)
        self.fingerprint = identity_fingerprint(self.identity)
        self.proof = {
            "user_id": "u1", "device_id": "d1", "key_id": "k1",
            "public_key": self.prekey_public, "identity_key": self.identity,
            "signature": self.signature,
        }
        self.session = {
            "session_id": "s1", "initiator_device_id": "d0",
            "recipient_device_id": "d1", "prekey_id": "k1",
            "ephemeral_key": self.eph_public, "identity_key": self.identity,
            "public_key": self.prekey_public, "created_at": "2026-01-01",
        }

    def _run(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "e2ee_backend",
             "derive-verified-session-key", *arguments],
            capture_output=True, text=True, timeout=15)

    def _good_args(self, session_arg: str, proof_arg: str):
        return ("--session", session_arg, "--proof", proof_arg,
                "--private-key", self.eph_private_b64,
                "--user-id", "u1", "--expected-fingerprint", self.fingerprint)

    def _run_ok(self, session_arg: str, proof_arg: str) -> dict:
        result = self._run(*self._good_args(session_arg, proof_arg))
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

    def test_success_with_inline_json(self):
        body = self._run_ok(json.dumps(self.session), json.dumps(self.proof))
        self.assertEqual(set(body), {"session_id", "key"})
        self.assertEqual(body["session_id"], "s1")
        expected = derive_session_key("s1", self.eph_private_b64,
                                      self.prekey_public)
        self.assertEqual(body["key"], expected["key"])

    def test_success_with_at_files_and_extra_fields(self):
        session = dict(self.session, extra="ignored")
        proof = dict(self.proof, extra="ignored")
        paths = []
        try:
            for element in (session, proof):
                with tempfile.NamedTemporaryFile(
                        "w", encoding="utf-8", suffix=".json",
                        delete=False) as handle:
                    json.dump(element, handle)
                    paths.append(handle.name)
            body = self._run_ok("@" + paths[0], "@" + paths[1])
        finally:
            for path in paths:
                os.unlink(path)
        self.assertEqual(set(body), {"session_id", "key"})

    def test_invalid_json_and_non_object_are_session_or_proof_errors(self):
        good_session = json.dumps(self.session)
        good_proof = json.dumps(self.proof)
        for bad in ("{not json", "[1,2]", '"text"', "3"):
            body = self._run_error(*self._good_args(bad, good_proof))
            self.assertEqual(body["field"], "session")
            body = self._run_error(*self._good_args(good_session, bad))
            self.assertEqual(body["field"], "proof")

    def test_unreadable_and_non_utf8_files(self):
        good = json.dumps(self.session)
        body = self._run_error(*self._good_args("@/nonexistent/session.json",
                                                json.dumps(self.proof)))
        self.assertEqual(body["field"], "session")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            f.write(b'{"session_id": "\xff\xfe"}')
            path = f.name
        try:
            body = self._run_error(*self._good_args("@" + path,
                                                    json.dumps(self.proof)))
            self.assertEqual(body["field"], "session")
            body = self._run_error(*self._good_args(good, "@" + path))
            self.assertEqual(body["field"], "proof")
        finally:
            os.unlink(path)

    def test_missing_option_names_its_field(self):
        args = list(self._good_args(json.dumps(self.session),
                                    json.dumps(self.proof)))
        for option, field in (("--session", "session"),
                              ("--proof", "proof"),
                              ("--private-key", "private_key"),
                              ("--user-id", "user_id"),
                              ("--expected-fingerprint",
                               "expected_fingerprint")):
            reduced = list(args)
            index = reduced.index(option)
            del reduced[index:index + 2]
            body = self._run_error(*reduced)
            self.assertEqual(body["field"], field)

    def test_verification_and_mismatch_failures(self):
        good = self._good_args(json.dumps(self.session),
                               json.dumps(self.proof))
        # Trusted fingerprint mismatch.
        _, other_identity = _new_identity()
        args = list(good)
        args[args.index("--expected-fingerprint") + 1] = \
            identity_fingerprint(other_identity)
        self.assertEqual(self._run_error(*args)["field"],
                         "expected_fingerprint")
        # Ephemeral private key does not match the snapshot.
        _, other_public = _new_x25519()
        session = dict(self.session, ephemeral_key=other_public)
        args = list(self._good_args(json.dumps(session),
                                    json.dumps(self.proof)))
        self.assertEqual(self._run_error(*args)["field"], "ephemeral_key")
        # Proof public key disagrees with the snapshot.
        session = dict(self.session,
                       public_key=_der_b64_x25519(self.prekey_public))
        args = list(self._good_args(json.dumps(session),
                                    json.dumps(self.proof)))
        self.assertEqual(self._run_error(*args)["field"], "public_key")


if __name__ == "__main__":
    unittest.main()
