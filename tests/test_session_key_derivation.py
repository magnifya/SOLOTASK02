"""Tests for local X3DH-style session-key derivation and its CLI command."""
from __future__ import annotations

import base64
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519, x25519
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from e2ee_backend.crypto import (CryptoError, decrypt_message,
                                 derive_session_key, encrypt_message)


def _pair():
    private = x25519.X25519PrivateKey.generate()
    return private, private.public_key()


def _pub_b64(public_key) -> str:
    return base64.b64encode(public_key.public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw)).decode()


def _priv_raw(private_key) -> bytes:
    return private_key.private_bytes(
        serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
        serialization.NoEncryption())


def _encodings(private_key):
    raw = _priv_raw(private_key)
    der = private_key.private_bytes(
        serialization.Encoding.DER, serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption())
    pem = private_key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption()).decode("ascii")
    return (pem, base64.b64encode(der).decode(), der.hex(), raw.hex(),
            base64.b64encode(raw).decode())


def _material():
    ik_a, ik_a_pub = _pair()
    ek_a, ek_a_pub = _pair()
    ik_b, ik_b_pub = _pair()
    spk_b, spk_b_pub = _pair()
    snapshot = {
        "session_id": "0123abcd" * 4,
        "initiator_device_id": "laptop",
        "recipient_device_id": "phone",
        "prekey_id": "pk-1",
        "ephemeral_key": _pub_b64(ek_a_pub),
        "identity_key": _pub_b64(ik_b_pub),
        "public_key": _pub_b64(spk_b_pub),
        "initiator_identity_key": _pub_b64(ik_a_pub),
        "created_at": "2026-09-29T00:00:00+00:00",
    }
    return dict(snapshot=snapshot, ik_a=ik_a, ek_a=ek_a, ik_b=ik_b,
                spk_b=spk_b, spk_b_pub=spk_b_pub, ik_b_pub=ik_b_pub)


class DeriveSessionKeyTest(unittest.TestCase):
    def setUp(self):
        self.m = _material()
        self.snapshot = self.m["snapshot"]

    def _initiator(self, snapshot=None, identity=None, ephemeral=None):
        m = self.m
        return derive_session_key(
            json.dumps(self.snapshot if snapshot is None else snapshot),
            "initiator",
            _encodings(m["ik_a"])[4] if identity is None else identity,
            _encodings(m["ek_a"])[0] if ephemeral is None else ephemeral)

    def _recipient(self, snapshot=None, identity=None, prekey=None):
        m = self.m
        return derive_session_key(
            self.snapshot if snapshot is None else snapshot, "recipient",
            _encodings(m["ik_b"])[1] if identity is None else identity,
            prekey_private=_encodings(m["spk_b"])[3] if prekey is None
            else prekey)

    def test_both_sides_derive_same_key(self):
        a = self._initiator()
        b = self._recipient()
        self.assertEqual(a["session_id"], self.snapshot["session_id"])
        self.assertEqual(a["key"], b["key"])
        self.assertEqual(len(base64.b64decode(a["key"])), 32)

    def test_key_matches_independent_hkdf_reference(self):
        m = self.m
        shared = (m["ik_a"].exchange(m["spk_b_pub"])
                  + m["ek_a"].exchange(m["ik_b_pub"])
                  + m["ek_a"].exchange(m["spk_b_pub"]))
        expected = HKDF(
            algorithm=hashes.SHA256(), length=32,
            salt=self.snapshot["session_id"].encode("utf-8"),
            info=b"e2ee-session-key-v1").derive(shared)
        self.assertEqual(self._initiator()["key"],
                         base64.b64encode(expected).decode("ascii"))

    def test_deterministic_across_repeats(self):
        keys = {self._initiator()["key"] for _ in range(4)}
        keys.add(self._recipient()["key"])
        self.assertEqual(len(keys), 1)

    def test_all_private_key_encodings_agree(self):
        reference = self._initiator()["key"]
        for identity in _encodings(self.m["ik_a"]):
            self.assertEqual(self._initiator(identity=identity)["key"],
                             reference)
        for prekey in _encodings(self.m["spk_b"]):
            self.assertEqual(self._recipient(prekey=prekey)["key"], reference)

    def test_tampered_snapshot_fails_or_diverges(self):
        good = self._initiator()["key"]
        tampered = dict(self.snapshot, ephemeral_key=_pub_b64(_pair()[1]))
        with self.assertRaises(CryptoError) as caught:
            self._initiator(tampered)
        self.assertEqual(caught.exception.field, "ephemeral_private_key")
        tampered = dict(self.snapshot, public_key=_pub_b64(_pair()[1]))
        try:
            key = self._recipient(tampered)["key"]
        except CryptoError as error:
            self.assertEqual(error.field, "prekey_private_key")
        else:
            self.assertNotEqual(key, good)
        tampered = dict(self.snapshot, session_id="deadbeef" * 4)
        derived = self._initiator(tampered)["key"]
        self.assertEqual(derived, self._recipient(tampered)["key"])
        self.assertNotEqual(derived, good)

    def test_mismatched_private_keys_rejected(self):
        other, _ = _pair()
        cases = ((self._initiator, {"identity": _encodings(other)[0]},
                  "identity_private_key"),
                 (self._initiator, {"ephemeral": _encodings(other)[3]},
                  "ephemeral_private_key"),
                 (self._recipient, {"prekey": _encodings(other)[0]},
                  "prekey_private_key"),
                 (self._initiator,
                  {"identity": _encodings(self.m["ik_b"])[0]},
                  "identity_private_key"))
        for func, kwargs, field in cases:
            with self.assertRaises(CryptoError) as caught:
                func(**kwargs)
            self.assertEqual(caught.exception.field, field)

    def test_non_x25519_private_and_public_keys_rejected(self):
        ed_pem = ed25519.Ed25519PrivateKey.generate().private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption()).decode("ascii")
        ec_pem = ec.generate_private_key(ec.SECP256R1()).private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption()).decode("ascii")
        for bad in (ed_pem, ec_pem, "nonsense", ""):
            with self.assertRaises(CryptoError) as caught:
                self._initiator(identity=bad)
            self.assertEqual(caught.exception.field, "identity_private_key")
        ed_pub = ed25519.Ed25519PrivateKey.generate().public_key(
            ).public_bytes(
                serialization.Encoding.PEM,
                serialization.PublicFormat.SubjectPublicKeyInfo).decode()
        with self.assertRaises(CryptoError) as caught:
            self._recipient(dict(self.snapshot, identity_key=ed_pub))
        self.assertEqual(caught.exception.field, "identity_key")

    def test_bad_snapshot_shape_and_missing_fields(self):
        for text, field in (("{bad", "session_json"), ("[1]", "session_json"),
                            ('"x"', "session_json"), ("12", "session_json")):
            with self.assertRaises(CryptoError) as caught:
                derive_session_key(
                    text, "initiator", _encodings(self.m["ik_a"])[4],
                    _encodings(self.m["ek_a"])[0])
            self.assertEqual(caught.exception.field, field)
        for name, field in (
                ("session_id", "session_id"),
                ("ephemeral_key", "ephemeral_key"),
                ("identity_key", "identity_key"),
                ("public_key", "public_key"),
                ("initiator_identity_key", "initiator_identity_key")):
            broken = dict(self.snapshot)
            del broken[name]
            with self.assertRaises(CryptoError) as caught:
                self._recipient(broken)
            self.assertEqual(caught.exception.field, field)
        with self.assertRaises(CryptoError) as caught:
            self._recipient(dict(self.snapshot, session_id=9))
        self.assertEqual(caught.exception.field, "session_id")
        with self.assertRaises(CryptoError) as caught:
            self._initiator(dict(self.snapshot, public_key=None))
        self.assertEqual(caught.exception.field, "public_key")

    def test_bad_role(self):
        for role in ("", None, "man-in-the-middle", 3):
            with self.assertRaises(CryptoError) as caught:
                derive_session_key(self.snapshot, role,
                                   _encodings(self.m["ik_a"])[0])
            self.assertEqual(caught.exception.field, "role")

    def test_missing_role_specific_private_key(self):
        with self.assertRaises(CryptoError) as caught:
            derive_session_key(self.snapshot, "initiator",
                               _encodings(self.m["ik_a"])[0])
        self.assertEqual(caught.exception.field, "ephemeral_private_key")
        with self.assertRaises(CryptoError) as caught:
            derive_session_key(self.snapshot, "recipient",
                               _encodings(self.m["ik_b"])[0])
        self.assertEqual(caught.exception.field, "prekey_private_key")
        with self.assertRaises(CryptoError) as caught:
            derive_session_key(self.snapshot, "initiator", None, "x")
        self.assertEqual(caught.exception.field, "identity_private_key")

    def test_aes_gcm_roundtrip_with_derived_key(self):
        key = self._initiator()["key"]
        session_id = self.snapshot["session_id"]
        envelope = encrypt_message(session_id, key, "hello derive-key")
        self.assertEqual(
            decrypt_message(session_id, key, envelope["nonce"],
                            envelope["ciphertext"])["plaintext"],
            "hello derive-key")


class DeriveSessionKeyCliTest(unittest.TestCase):
    def setUp(self):
        self.m = _material()
        self.repo = Path(__file__).resolve().parents[1]
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        self.session_file = root / "session.json"
        self.session_file.write_text(json.dumps(self.m["snapshot"]),
                                     encoding="utf-8")
        self.ik_a = root / "ik_a.pem"
        self.ek_a = root / "ek_a.hex"
        self.ik_b = root / "ik_b.der"
        self.spk_b = root / "spk_b.b64"
        self.ik_a.write_text(_encodings(self.m["ik_a"])[0])
        self.ek_a.write_text(_encodings(self.m["ek_a"])[3])
        self.ik_b.write_bytes(base64.b64encode(self.m["ik_b"].private_bytes(
            serialization.Encoding.DER, serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption())))
        self.spk_b.write_text(base64.b64encode(_priv_raw(self.m["spk_b"]))
                              .decode())

    def tearDown(self):
        self.tempdir.cleanup()

    def _run(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "e2ee_backend", "derive-session-key",
             *args],
            cwd=self.repo, capture_output=True, text=True, timeout=20)

    def _initiator_cli(self):
        return self._run(
            "--session-json", "@" + str(self.session_file),
            "--role", "initiator",
            "--identity-private-key", "@" + str(self.ik_a),
            "--ephemeral-private-key", "@" + str(self.ek_a))

    def test_cli_both_sides_repeatable(self):
        a1 = self._initiator_cli()
        a2 = self._initiator_cli()
        b = self._run(
            "--session-json", "@" + str(self.session_file),
            "--role", "recipient",
            "--identity-private-key", "@" + str(self.ik_b),
            "--prekey-private-key", "@" + str(self.spk_b))
        self.assertEqual(a1.returncode, 0, a1.stderr)
        self.assertEqual(a2.returncode, 0, a2.stderr)
        self.assertEqual(b.returncode, 0, b.stderr)
        self.assertEqual(a1.stdout, a2.stdout)
        self.assertEqual(json.loads(a1.stdout)["key"],
                         json.loads(b.stdout)["key"])
        self.assertEqual(a1.stdout.count("\n"), 1)
        self.assertEqual(a1.stderr, "")

    def test_cli_inline_session_json_and_raw_keys(self):
        result = self._run(
            "--session-json", json.dumps(self.m["snapshot"]),
            "--role", "recipient",
            "--identity-private-key", _encodings(self.m["ik_b"])[4],
            "--prekey-private-key", _encodings(self.m["spk_b"])[0])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["key"],
                         json.loads(self._initiator_cli().stdout)["key"])

    def test_cli_error_emits_single_json_line_nonzero(self):
        result = self._run(
            "--session-json", "@/nonexistent/session.json",
            "--role", "initiator",
            "--identity-private-key", "@" + str(self.ik_a),
            "--ephemeral-private-key", "@" + str(self.ek_a))
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        error = json.loads(result.stderr)
        self.assertEqual(error["field"], "session_json")
        self.assertTrue(error["message"])

        result = self._run(
            "--session-json", "@" + str(self.session_file),
            "--role", "initiator",
            "--identity-private-key", "@" + str(self.ik_b),
            "--ephemeral-private-key", "@" + str(self.ek_a))
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stderr)["field"],
                         "identity_private_key")

        result = self._run(
            "--session-json", "@" + str(self.session_file),
            "--role", "recipient",
            "--identity-private-key", "@" + str(self.ik_b))
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(json.loads(result.stderr)["field"],
                         "prekey_private_key")

        result = self._run(
            "--session-json", "@" + str(self.session_file),
            "--role", "bogus",
            "--identity-private-key", "@" + str(self.ik_b))
        self.assertEqual(json.loads(result.stderr)["field"], "role")

        bad_json = Path(self.tempdir.name) / "bad.json"
        bad_json.write_text("not-json", encoding="utf-8")
        result = self._run(
            "--session-json", "@" + str(bad_json),
            "--role", "initiator",
            "--identity-private-key", "@" + str(self.ik_a),
            "--ephemeral-private-key", "@" + str(self.ek_a))
        self.assertEqual(json.loads(result.stderr)["field"], "session_json")

    def test_cli_never_prints_private_material(self):
        result = self._initiator_cli()
        self.assertNotIn(_priv_raw(self.m["ik_a"]).hex(), result.stdout)
        self.assertNotIn(_encodings(self.m["ek_a"])[0], result.stdout)
