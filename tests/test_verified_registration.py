"""Tests for verified registration: POST /v1/devices/verified.

The verified route accepts the same registration shape plus an Ed25519
``signature`` per signed pre-key; the device is only published when the
identity key is an Ed25519 key and every pre-key proof verifies over the
domain-separated canonical message. Failures name the exact field path and
write no state.
"""
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from http.client import HTTPConnection

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, x25519

from e2ee_backend.crypto import signed_prekey_proof_message
from e2ee_backend.http_app import create_server
from e2ee_backend.persistence import attach_persistence
from e2ee_backend.service import DeviceService, ServiceError


def _raw_b64(key) -> str:
    raw = key.public_bytes(serialization.Encoding.Raw,
                           serialization.PublicFormat.Raw)
    return base64.b64encode(raw).decode()


def _der_b64(key) -> str:
    der = key.public_bytes(serialization.Encoding.DER,
                           serialization.PublicFormat.SubjectPublicKeyInfo)
    return base64.b64encode(der).decode()


def _new_identity():
    """Return an Ed25519 (private key, base64 raw public key) pair."""
    private = ed25519.Ed25519PrivateKey.generate()
    return private, _raw_b64(private.public_key())


def _new_prekey() -> str:
    return _raw_b64(x25519.X25519PrivateKey.generate().public_key())


def _proof(private, user_id, device_id, key_id, public_key) -> str:
    message = signed_prekey_proof_message(
        user_id, device_id, key_id, public_key)
    return base64.b64encode(private.sign(message)).decode()


def _verified_payload(private=None, identity_key=None, user_id="u1",
                      device_id="d1", prekeys=None, signature_for=None):
    """Build a valid verified-registration payload.

    Each entry of *prekeys* is a key_id (the public key is generated); the
    matching Ed25519 proof is produced with *private*. *signature_for* may
    map a key_id to an override signature value.
    """
    if private is None:
        private, generated_identity = _new_identity()
        if identity_key is None:
            identity_key = generated_identity
    elif identity_key is None:
        identity_key = _raw_b64(private.public_key())
    entries = []
    for key_id in prekeys or ("k1", "k2"):
        public_key = _new_prekey()
        signature = (signature_for or {}).get(key_id)
        if signature is None:
            signature = _proof(private, user_id, device_id,
                               key_id, public_key)
        entries.append({"key_id": key_id, "public_key": public_key,
                        "signature": signature})
    return {
        "user_id": user_id,
        "device_id": device_id,
        "identity_key": identity_key,
        "signed_prekeys": entries,
    }


class ProofMessageTest(unittest.TestCase):
    def test_message_is_prefix_newline_then_sorted_compact_json(self) -> None:
        # Built independently of the helper to pin the wire contract.
        document = json.dumps(
            {"device_id": "d1", "key_id": "k1",
             "public_key": "pk", "user_id": "u1"},
            sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        expected = ("E2EE-SIGNED-PREKEY-V1\n" + document).encode("utf-8")
        self.assertEqual(
            signed_prekey_proof_message("u1", "d1", "k1", "pk"), expected)
        # Key order in the JSON is fixed regardless of insertion order.
        self.assertEqual(
            json.loads(expected.split(b"\n", 1)[1].decode()),
            {"device_id": "d1", "key_id": "k1",
             "public_key": "pk", "user_id": "u1"})

    def test_message_uses_request_values_including_unicode(self) -> None:
        message = signed_prekey_proof_message("用户", "dæ", "k1", "p#1")
        # Unicode is written as-is, not \u-escaped.
        self.assertIn("用户".encode("utf-8"), message)
        self.assertIn("dæ".encode("utf-8"), message)


class RegisterVerifiedServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = DeviceService()

    def _register(self, payload) -> None:
        body = self.service.register_verified(payload)
        self.assertEqual(set(body), {"device_id", "registered_at"})
        self.assertEqual(body["device_id"], payload["device_id"])

    def test_success_publishes_device_and_prekeys_in_order(self) -> None:
        payload = _verified_payload()
        self._register(payload)
        view = self.service.get_device("d1")
        self.assertEqual(view["identity_key"], payload["identity_key"])
        self.assertEqual(view["prekey_ids"], ["k1", "k2"])

    def test_single_prekey_success(self) -> None:
        payload = _verified_payload(prekeys=("only",))
        self._register(payload)
        self.assertEqual(self.service.get_device("d1")["prekey_ids"], ["only"])

    def test_der_ed25519_identity_accepted(self) -> None:
        private = ed25519.Ed25519PrivateKey.generate()
        identity_key = _der_b64(private.public_key())
        payload = _verified_payload(private=private,
                                    identity_key=identity_key)
        self._register(payload)

    def test_pem_ed25519_identity_accepted(self) -> None:
        private = ed25519.Ed25519PrivateKey.generate()
        identity_key = private.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo).decode()
        payload = _verified_payload(private=private,
                                    identity_key=identity_key)
        self._register(payload)

    def test_unicode_strings_signed_and_verified(self) -> None:
        private, identity_key = _new_identity()
        public_key = _new_prekey()
        payload = {
            "user_id": "用户",
            "device_id": "dæ-1",
            "identity_key": identity_key,
            "signed_prekeys": [{
                "key_id": "密钥①",
                "public_key": public_key,
                "signature": _proof(private, "用户", "dæ-1",
                                    "密钥①", public_key)}],
        }
        self._register(payload)

    # -- identity key failures --------------------------------------------

    def test_missing_identity_key(self) -> None:
        payload = _verified_payload()
        del payload["identity_key"]
        self._assert_field(payload, "identity_key")

    def test_identity_key_wrong_type(self) -> None:
        payload = _verified_payload()
        payload["identity_key"] = 123
        self._assert_field(payload, "identity_key")

    def test_identity_key_empty(self) -> None:
        payload = _verified_payload()
        payload["identity_key"] = ""
        self._assert_field(payload, "identity_key")

    def test_identity_key_garbage(self) -> None:
        payload = _verified_payload()
        payload["identity_key"] = "not-a-key"
        self._assert_field(payload, "identity_key")

    def test_identity_key_x25519_der_is_rejected(self) -> None:
        # Raw 32 bytes are algorithm-agnostic; a DER SPKI carries the OID,
        # so an X25519 DER key must be refused as not-Ed25519.
        x25519_der = _der_b64(
            x25519.X25519PrivateKey.generate().public_key())
        payload = _verified_payload()
        payload["identity_key"] = x25519_der
        self._assert_field(payload, "identity_key")

    # -- body / array shape failures --------------------------------------

    def test_body_must_be_object(self) -> None:
        with self.assertRaises(ServiceError) as caught:
            self.service.register_verified(["not", "an", "object"])
        self.assertEqual(caught.exception.field, "request_body")

    def test_missing_signed_prekeys(self) -> None:
        payload = _verified_payload()
        del payload["signed_prekeys"]
        self._assert_field(payload, "signed_prekeys")

    def test_signed_prekeys_wrong_type(self) -> None:
        payload = _verified_payload()
        payload["signed_prekeys"] = {"k1": {}}
        self._assert_field(payload, "signed_prekeys")

    def test_element_must_be_object(self) -> None:
        payload = _verified_payload()
        payload["signed_prekeys"][0] = "k1"
        self._assert_field(payload, "signed_prekeys[0]")

    def test_missing_scalar_fields(self) -> None:
        payload = _verified_payload()
        del payload["user_id"]
        self._assert_field(payload, "user_id")
        payload["user_id"] = "u1"
        del payload["device_id"]
        self._assert_field(payload, "device_id")

    def test_scalar_field_wrong_type(self) -> None:
        payload = _verified_payload(device_id="dt")
        payload["user_id"] = True
        self._assert_field(payload, "user_id")

    # -- per-element failures ----------------------------------------------

    def test_missing_key_id(self) -> None:
        payload = _verified_payload()
        del payload["signed_prekeys"][1]["key_id"]
        self._assert_field(payload, "signed_prekeys[1].key_id")

    def test_missing_public_key(self) -> None:
        payload = _verified_payload()
        del payload["signed_prekeys"][0]["public_key"]
        self._assert_field(payload, "signed_prekeys[0].public_key")

    def test_missing_signature(self) -> None:
        payload = _verified_payload()
        del payload["signed_prekeys"][0]["signature"]
        self._assert_field(payload, "signed_prekeys[0].signature")

    def test_signature_empty(self) -> None:
        payload = _verified_payload(
            signature_for={"k1": ""})
        self._assert_field(payload, "signed_prekeys[0].signature")

    def test_signature_wrong_type(self) -> None:
        payload = _verified_payload()
        payload["signed_prekeys"][0]["signature"] = 42
        self._assert_field(payload, "signed_prekeys[0].signature")

    def test_key_id_wrong_type(self) -> None:
        payload = _verified_payload()
        payload["signed_prekeys"][0]["key_id"] = None
        self._assert_field(payload, "signed_prekeys[0].key_id")

    def test_public_key_wrong_type(self) -> None:
        payload = _verified_payload()
        payload["signed_prekeys"][0]["public_key"] = ["x"]
        self._assert_field(payload, "signed_prekeys[0].public_key")

    def test_public_key_illegal(self) -> None:
        payload = _verified_payload()
        payload["signed_prekeys"][0]["public_key"] = "garbage"
        self._assert_field(payload, "signed_prekeys[0].public_key")

    def test_duplicate_key_id(self) -> None:
        payload = _verified_payload(prekeys=("k1", "k1"))
        self._assert_field(payload, "signed_prekeys[1].key_id")

    def test_signature_bad_base64(self) -> None:
        payload = _verified_payload(
            signature_for={"k1": "not base64!"})
        self._assert_field(payload, "signed_prekeys[0].signature")

    def test_signature_wrong_length(self) -> None:
        # Valid base64 that decodes to fewer than 64 bytes.
        short = base64.b64encode(b"\x00" * 32).decode()
        payload = _verified_payload(signature_for={"k1": short})
        self._assert_field(payload, "signed_prekeys[0].signature")

    def test_signature_non_canonical_base64(self) -> None:
        # A 64-byte signature with its required padding stripped is not
        # canonical standard base64 and must be refused.
        private = ed25519.Ed25519PrivateKey.generate()
        payload = _verified_payload(
            private=private, identity_key=_raw_b64(private.public_key()))
        sig = payload["signed_prekeys"][0]["signature"]
        self.assertTrue(sig.endswith("="))
        payload["signed_prekeys"][0]["signature"] = sig.rstrip("=")
        self._assert_field(payload, "signed_prekeys[0].signature")

    def test_signature_for_different_message_fails(self) -> None:
        # Flip one byte of an otherwise well-formed 64-byte signature.
        private = ed25519.Ed25519PrivateKey.generate()
        payload = _verified_payload(private=private,
                                    identity_key=_raw_b64(private.public_key()))
        good = base64.b64decode(payload["signed_prekeys"][0]["signature"])
        tampered = bytes([good[0] ^ 0x01]) + good[1:]
        payload["signed_prekeys"][0]["signature"] = \
            base64.b64encode(tampered).decode()
        self._assert_field(payload, "signed_prekeys[0].signature")

    def test_signature_over_other_fields_fails(self) -> None:
        private, identity_key = _new_identity()
        # Proof is signed for device_id "other", but requested as "d1".
        payload = _verified_payload(
            private=private, identity_key=identity_key, device_id="d1",
            signature_for={"k1": None})
        public_key = payload["signed_prekeys"][0]["public_key"]
        payload["signed_prekeys"][0]["signature"] = _proof(
            private, "u1", "other-device", "k1", public_key)
        self._assert_field(payload, "signed_prekeys[0].signature")

    def test_wrong_identity_key_signature_fails(self) -> None:
        signer, _ = _new_identity()
        _, other_identity = _new_identity()
        payload = _verified_payload(
            private=signer, identity_key=other_identity)
        self._assert_field(payload, "signed_prekeys[0].signature")

    # -- atomicity ----------------------------------------------------------

    def test_failure_writes_nothing_and_consumes_no_key(self) -> None:
        good = _verified_payload(device_id="d1")
        self.service.register_verified(good)
        available_before = self.service.get_device("d1")["prekey_ids"]

        bad = _verified_payload(device_id="d2",
                                signature_for={"k1": "garbage"})
        self._assert_field(bad, "signed_prekeys[0].signature")
        # The rejected device was never created...
        with self.assertRaises(ServiceError) as caught:
            self.service.get_device("d2")
        self.assertEqual(caught.exception.status_code, 404)
        # ...and the earlier device's keys are untouched.
        self.assertEqual(self.service.get_device("d1")["prekey_ids"],
                         available_before)

    def test_second_bad_entry_fails_the_whole_batch(self) -> None:
        payload = _verified_payload(device_id="db", prekeys=("k1", "k2"),
                                    signature_for={"k2": "garbage"})
        self._assert_field(payload, "signed_prekeys[1].signature")
        with self.assertRaises(ServiceError) as caught:
            self.service.get_device("db")
        self.assertEqual(caught.exception.status_code, 404)

    # -- conflict ------------------------------------------------------------

    def test_duplicate_is_409_field_device_id(self) -> None:
        payload = _verified_payload()
        self.service.register_verified(payload)
        with self.assertRaises(ServiceError) as caught:
            self.service.register_verified(dict(payload))
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "device_id")

    def test_conflict_with_legacy_registration(self) -> None:
        legacy = {
            "user_id": "u1", "device_id": "shared",
            "identity_key": _raw_b64(
                x25519.X25519PrivateKey.generate().public_key()),
            "signed_prekeys": [{"key_id": "k1",
                                "public_key": _new_prekey()}]}
        self.service.register(legacy)
        verified = _verified_payload(device_id="shared")
        with self.assertRaises(ServiceError) as caught:
            self.service.register_verified(verified)
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "device_id")

    def test_legacy_route_still_accepts_unsigned_prekeys(self) -> None:
        legacy = {
            "user_id": "u1", "device_id": "legacy",
            "identity_key": _raw_b64(
                x25519.X25519PrivateKey.generate().public_key()),
            "signed_prekeys": [{"key_id": "k1",
                                "public_key": _new_prekey()}]}
        body = self.service.register(legacy)
        self.assertEqual(body["device_id"], "legacy")

    def _assert_field(self, payload, field) -> None:
        with self.assertRaises(ServiceError) as caught:
            self.service.register_verified(payload)
        self.assertEqual(caught.exception.status_code, 400)
        self.assertEqual(caught.exception.field, field)


class RegisterVerifiedHTTPTest(unittest.TestCase):
    def setUp(self) -> None:
        self.server, _ = create_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _request(self, method: str, path: str, body: object = None):
        connection = HTTPConnection("127.0.0.1", self.port, timeout=5)
        payload = json.dumps(body) if body is not None else None
        headers = {"Content-Type": "application/json"} if payload else {}
        connection.request(method, path, body=payload, headers=headers)
        response = connection.getresponse()
        data = response.read().decode("utf-8")
        connection.close()
        return response.status, json.loads(data)

    def test_success_is_201(self) -> None:
        status, body = self._request(
            "POST", "/v1/devices/verified", _verified_payload())
        self.assertEqual(status, 201)
        self.assertEqual(set(body), {"device_id", "registered_at"})

    def test_bad_body_is_400_request_body(self) -> None:
        connection = HTTPConnection("127.0.0.1", self.port, timeout=5)
        connection.request("POST", "/v1/devices/verified", body=b"{nope",
                           headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        body = json.loads(response.read())
        connection.close()
        self.assertEqual(response.status, 400)
        self.assertEqual(body["field"], "request_body")

    def test_bad_signature_is_400_named_path(self) -> None:
        payload = _verified_payload(device_id="d2",
                                    signature_for={"k1": "garbage"})
        status, body = self._request("POST", "/v1/devices/verified", payload)
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "signed_prekeys[0].signature")

    def test_non_ed25519_identity_is_400_identity_key(self) -> None:
        payload = _verified_payload()
        payload["identity_key"] = _der_b64(
            x25519.X25519PrivateKey.generate().public_key())
        status, body = self._request("POST", "/v1/devices/verified", payload)
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "identity_key")

    def test_duplicate_is_409_device_id(self) -> None:
        payload = _verified_payload(device_id="dup")
        self.assertEqual(
            self._request("POST", "/v1/devices/verified", payload)[0], 201)
        status, body = self._request("POST", "/v1/devices/verified", payload)
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "device_id")

    def test_failure_does_not_publish(self) -> None:
        payload = _verified_payload(device_id="d3",
                                    signature_for={"k1": "garbage"})
        self._request("POST", "/v1/devices/verified", payload)
        status, body = self._request("GET", "/v1/devices/d3")
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "device_id")

    def test_verified_device_supports_claims_and_sessions(self) -> None:
        private, identity_key = _new_identity()
        payload = _verified_payload(private=private, identity_key=identity_key,
                                    device_id="d4", prekeys=("k1",))
        self._request("POST", "/v1/devices/verified", payload)
        status, body = self._request(
            "POST", "/v1/prekeys/claim",
            {"recipient_device_id": "d4", "claim_id": "c1"})
        self.assertEqual(status, 201)
        self.assertEqual(body["identity_key"], identity_key)
        self.assertEqual(body["key_id"], "k1")
        status, show = self._request("GET", "/v1/devices/d4")
        self.assertEqual(status, 200)
        self.assertEqual(show["prekey_ids"], [])  # the one key was consumed

    def test_legacy_route_unchanged(self) -> None:
        legacy = {
            "user_id": "u1", "device_id": "old",
            "identity_key": _raw_b64(
                x25519.X25519PrivateKey.generate().public_key()),
            "signed_prekeys": [{"key_id": "k1",
                                "public_key": _new_prekey()}]}
        status, body = self._request("POST", "/v1/devices", legacy)
        self.assertEqual(status, 201)
        self.assertEqual(set(body), {"device_id", "registered_at"})

    def test_verified_route_rejects_unsigned_entry(self) -> None:
        payload = _verified_payload(device_id="d5")
        del payload["signed_prekeys"][0]["signature"]
        status, body = self._request("POST", "/v1/devices/verified", payload)
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "signed_prekeys[0].signature")


class RegisterVerifiedPersistenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.path = os.path.join(self.directory, "state.json")

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_verified_registration_persists_and_restarts(self) -> None:
        service = DeviceService()
        attach_persistence(service, self.path)
        payload = _verified_payload(device_id="persist")
        body = service.register_verified(payload)
        registered_at = body["registered_at"]

        # Restart: a brand-new service attached to the same file restores
        # the verified device, its Ed25519 identity and the pre-key order.
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        view = restarted.get_device("persist")
        self.assertEqual(view["identity_key"], payload["identity_key"])
        self.assertEqual(view["prekey_ids"], ["k1", "k2"])
        self.assertEqual(view["registered_at"], registered_at)

        # The registered key-audit event was anchored by the same write.
        events = restarted.list_key_events("persist", 0, 100)["events"]
        self.assertEqual(events[0]["type"], "registered")
        self.assertEqual(
            events[0]["payload"]["identity_key"], payload["identity_key"])

        # The one-time claim and consumption semantics survive the restart.
        claim_body, status = restarted.claim_prekey(
            {"recipient_device_id": "persist", "claim_id": "c1"})
        self.assertEqual(status, 201)
        self.assertEqual(claim_body["key_id"], "k1")


class RegisterVerifiedCLITest(unittest.TestCase):
    def setUp(self) -> None:
        self.server, _ = create_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        self.private, self.identity_key = _new_identity()
        self.public_key = _new_prekey()
        self.signature = _proof(self.private, "u1", "cli1",
                                "k1", self.public_key)

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _run(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "e2ee_backend",
             "--base-url", self.base_url, *arguments],
            capture_output=True, text=True, timeout=15)

    def test_success_prints_single_line_json(self) -> None:
        result = self._run(
            "register-verified", "--user-id", "u1", "--device-id", "cli1",
            "--identity-key", self.identity_key,
            "--prekey", f"k1:{self.public_key}:{self.signature}")
        self.assertEqual(result.returncode, 0, result.stderr)
        line = result.stdout.strip()
        self.assertEqual(line.count("\n"), 0)
        self.assertEqual(set(json.loads(line)),
                         {"device_id", "registered_at"})

    def test_bad_proof_is_stderr_json_exit_1(self) -> None:
        result = self._run(
            "register-verified", "--user-id", "u1", "--device-id", "cli2",
            "--identity-key", self.identity_key,
            "--prekey", f"k1:{self.public_key}:AAAA")
        self.assertEqual(result.returncode, 1)
        body = json.loads(result.stderr.strip())
        self.assertEqual(body["field"], "signed_prekeys[0].signature")

    def test_malformed_spec_fails_locally_exit_2(self) -> None:
        result = self._run(
            "register-verified", "--user-id", "u1", "--device-id", "cli3",
            "--identity-key", self.identity_key,
            "--prekey", "k1:only-two-fields")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stderr.strip())["field"],
                         "signed_prekeys")

    def test_at_file_prekey_entry(self) -> None:
        signature = _proof(self.private, "u1", "cli4",
                           "k1", self.public_key)
        path = os.path.join(tempfile.mkdtemp(), "prekey.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"key_id": "k1", "public_key": self.public_key,
                       "signature": signature}, handle)
        try:
            result = self._run(
                "register-verified", "--user-id", "u1",
                "--device-id", "cli4",
                "--identity-key", self.identity_key,
                "--prekey", f"@{path}")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout.strip())["device_id"],
                             "cli4")
        finally:
            shutil.rmtree(os.path.dirname(path), ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
