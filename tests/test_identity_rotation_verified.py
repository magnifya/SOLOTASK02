"""Tests for signature-authorized identity-key rotation.

Covers the service, HTTP (real loopback socket), CLI (real subprocess),
key-audit and persistence layers for:

* POST /v1/devices/{device_id}/identity-key/rotate-verified

The new entry rotates a device's identity key only when a standard-base64
64-byte Ed25519 signature verifies against the device's *current* identity
key over the domain-separated canonical rotation message
(``E2EE-IDENTITY-ROTATION-V1``) and the request's ``expected_version``
matches the device's current ``identity_key_version``. A verified same-key
rotation is a state-free idempotent replay; a verified different-key
rotation refreshes the identity, timestamp and version and appends the
usual ``identity_rotated`` audit event. Nothing changes on failure.
"""
import base64
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from http.client import HTTPConnection

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, x25519

from e2ee_backend.crypto import identity_rotation_message
from e2ee_backend.http_app import create_server
from e2ee_backend.persistence import attach_persistence
from e2ee_backend.service import DeviceService, ServiceError


def _raw_b64(key) -> str:
    raw = key.public_bytes(serialization.Encoding.Raw,
                           serialization.PublicFormat.Raw)
    return base64.b64encode(raw).decode()


def _new_identity():
    private = ed25519.Ed25519PrivateKey.generate()
    return private, _raw_b64(private.public_key())


def _new_prekey() -> str:
    return _raw_b64(x25519.X25519PrivateKey.generate().public_key())


def _authorization(private, user_id, device_id, identity_key,
                   expected_version) -> str:
    message = identity_rotation_message(
        user_id, device_id, identity_key, expected_version)
    return base64.b64encode(private.sign(message)).decode()


class RotateVerifiedServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = DeviceService()
        self.private, self.identity = _new_identity()
        self.prekey = _new_prekey()
        from e2ee_backend.crypto import signed_prekey_proof_message
        proof = base64.b64encode(self.private.sign(
            signed_prekey_proof_message("u1", "d1", "k1",
                                        self.prekey))).decode()
        self.service.register_verified({
            "user_id": "u1", "device_id": "d1",
            "identity_key": self.identity,
            "signed_prekeys": [{
                "key_id": "k1", "public_key": self.prekey,
                "signature": proof}],
        })
        self.registered_at = self.service.get_device("d1")["registered_at"]

    def _payload(self, identity_key=None, expected_version=1,
                 signature=None, private=None):
        if identity_key is None:
            _, identity_key = _new_identity()
        if signature is None:
            signer = private or self.private
            signature = _authorization(
                signer, "u1", "d1", identity_key, expected_version)
        return {"identity_key": identity_key,
                "expected_version": expected_version,
                "signature": signature}

    def test_valid_rotation_200_updates_identity_and_version(self) -> None:
        _, new_identity = _new_identity()
        body = self.service.rotate_identity_key_verified(
            "d1", self._payload(identity_key=new_identity))
        self.assertEqual(set(body),
                         {"device_id", "identity_key", "rotated_at"})
        self.assertEqual(body["identity_key"], new_identity)
        self.assertNotEqual(body["rotated_at"], self.registered_at)
        self.assertTrue(body["rotated_at"].endswith("+00:00"))
        device = self.service.store.find_by_device_id("d1")
        self.assertEqual(device.identity_key, new_identity)
        self.assertEqual(device.identity_key_version, 2)

    def test_same_key_valid_authorization_is_state_free(self) -> None:
        events_before = self.service.list_key_events("d1", 0, 100)["events"]
        body = self.service.rotate_identity_key_verified(
            "d1", self._payload(identity_key=self.identity))
        self.assertEqual(body["identity_key"], self.identity)
        self.assertEqual(body["rotated_at"], self.registered_at)
        device = self.service.store.find_by_device_id("d1")
        self.assertEqual(device.identity_key_version, 1)
        self.assertEqual(device.rotated_at, self.registered_at)
        self.assertEqual(
            self.service.list_key_events("d1", 0, 100)["events"],
            events_before)

    def test_chained_rotations_bump_version_each_time(self) -> None:
        private_b, identity_b = _new_identity()
        self.service.rotate_identity_key_verified(
            "d1", self._payload(identity_key=identity_b))
        device = self.service.store.find_by_device_id("d1")
        self.assertEqual(device.identity_key_version, 2)
        # Rotate back to the original key, authorized by identity_b.
        body = self.service.rotate_identity_key_verified(
            "d1", self._payload(identity_key=self.identity,
                                expected_version=2, private=private_b))
        self.assertEqual(body["identity_key"], self.identity)
        device = self.service.store.find_by_device_id("d1")
        self.assertEqual(device.identity_key_version, 3)
        events = self.service.list_key_events("d1", 0, 100)["events"]
        self.assertEqual([e["type"] for e in events],
                         ["registered", "identity_rotated",
                          "identity_rotated"])
        self.assertEqual(events[1]["payload"],
                         {"old_identity_key": self.identity,
                          "new_identity_key": identity_b})
        self.assertEqual(events[2]["payload"],
                         {"old_identity_key": identity_b,
                          "new_identity_key": self.identity})

    def test_body_not_object_400_request_body(self) -> None:
        for payload in (None, [], "x", 42):
            with self.subTest(payload=payload):
                with self.assertRaises(ServiceError) as ctx:
                    self.service.rotate_identity_key_verified("d1", payload)
                self.assertEqual(ctx.exception.status_code, 400)
                self.assertEqual(ctx.exception.field, "request_body")

    def test_missing_fields_400(self) -> None:
        _, key = _new_identity()
        cases = [
            ({}, "identity_key"),
            ({"identity_key": key}, "expected_version"),
            ({"identity_key": key, "expected_version": 1}, "signature"),
        ]
        for payload, field in cases:
            with self.subTest(field=field):
                with self.assertRaises(ServiceError) as ctx:
                    self.service.rotate_identity_key_verified("d1", payload)
                self.assertEqual(ctx.exception.status_code, 400)
                self.assertEqual(ctx.exception.field, field)

    def test_bad_identity_key_400(self) -> None:
        # A raw 32-byte point cannot be told apart by curve (see the
        # verified-prekey tests); the X25519-in-DER case is covered by
        # test_x25519_new_key_rejected_400_identity_key.
        for bad in ("", 123, "not-a-public-key"):
            payload = self._payload(identity_key=bad)
            with self.subTest(bad=bad):
                with self.assertRaises(ServiceError) as ctx:
                    self.service.rotate_identity_key_verified("d1", payload)
                self.assertEqual(ctx.exception.status_code, 400)
                self.assertEqual(ctx.exception.field, "identity_key")

    def test_x25519_new_key_rejected_400_identity_key(self) -> None:
        x25519_der = base64.b64encode(
            x25519.X25519PrivateKey.generate().public_key().public_bytes(
                serialization.Encoding.DER,
                serialization.PublicFormat.SubjectPublicKeyInfo)).decode()
        with self.assertRaises(ServiceError) as ctx:
            self.service.rotate_identity_key_verified(
                "d1", self._payload(identity_key=x25519_der))
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.field, "identity_key")

    def test_bad_expected_version_400(self) -> None:
        for bad in (True, False, 0, -1, 1.5, "1", None):
            payload = self._payload(expected_version=bad)
            with self.subTest(bad=bad):
                with self.assertRaises(ServiceError) as ctx:
                    self.service.rotate_identity_key_verified("d1", payload)
                self.assertEqual(ctx.exception.status_code, 400)
                self.assertEqual(ctx.exception.field, "expected_version")

    def test_bad_signature_encoding_400_signature(self) -> None:
        good = base64.b64encode(b"\x00" * 64).decode()
        for bad in ("", 8, "@@@@", "abc", "a" * 88,
                    good.replace("+", "-").replace("/", "_"),
                    good.rstrip("="),
                    base64.b64encode(b"\x00" * 63).decode(),
                    base64.b64encode(b"\x00" * 65).decode()):
            payload = self._payload(signature=bad)
            with self.subTest(bad=bad):
                with self.assertRaises(ServiceError) as ctx:
                    self.service.rotate_identity_key_verified("d1", payload)
                self.assertEqual(ctx.exception.status_code, 400)
                self.assertEqual(ctx.exception.field, "signature")

    def test_unknown_device_404(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.rotate_identity_key_verified(
                "ghost", self._payload())
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "device_id")

    def test_revoked_device_409(self) -> None:
        self.service.revoke_device("d1")
        with self.assertRaises(ServiceError) as ctx:
            self.service.rotate_identity_key_verified(
                "d1", self._payload())
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "device_id")

    def test_non_ed25519_current_key_400_identity_key(self) -> None:
        # A device registered through the ordinary (unsigned) entry with an
        # X25519 identity key cannot authorize a verified rotation.
        x25519_der = base64.b64encode(
            x25519.X25519PrivateKey.generate().public_key().public_bytes(
                serialization.Encoding.DER,
                serialization.PublicFormat.SubjectPublicKeyInfo)).decode()
        self.service.register({
            "user_id": "u2", "device_id": "d2",
            "identity_key": x25519_der,
            "signed_prekeys": [{"key_id": "k1", "public_key": _new_prekey()}],
        })
        with self.assertRaises(ServiceError) as ctx:
            self.service.rotate_identity_key_verified(
                "d2", self._payload())
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.field, "identity_key")

    def test_version_mismatch_409_expected_version(self) -> None:
        _, identity_b = _new_identity()
        self.service.rotate_identity_key_verified(
            "d1", self._payload(identity_key=identity_b))
        # The version is now 2; a request naming 1 conflicts even with a
        # signature that would verify for version 1's signer.
        with self.assertRaises(ServiceError) as ctx:
            self.service.rotate_identity_key_verified(
                "d1", self._payload(expected_version=1))
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "expected_version")
        with self.assertRaises(ServiceError) as ctx:
            self.service.rotate_identity_key_verified(
                "d1", self._payload(expected_version=3))
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "expected_version")

    def test_wrong_signer_400_signature(self) -> None:
        other, _ = _new_identity()
        with self.assertRaises(ServiceError) as ctx:
            self.service.rotate_identity_key_verified(
                "d1", self._payload(private=other))
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.field, "signature")

    def test_tampered_fields_400_signature(self) -> None:
        _, new_identity = _new_identity()
        good = self._payload(identity_key=new_identity)
        # Signature over new_identity, request names another key.
        _, other_identity = _new_identity()
        tampered = dict(good, identity_key=other_identity)
        with self.assertRaises(ServiceError) as ctx:
            self.service.rotate_identity_key_verified("d1", tampered)
        self.assertEqual(ctx.exception.field, "signature")
        # Signature over version 1, request names version 2: the version
        # check runs before signature verification, so this is the 409.
        tampered = dict(good, expected_version=2)
        with self.assertRaises(ServiceError) as ctx:
            self.service.rotate_identity_key_verified("d1", tampered)
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "expected_version")

    def test_uses_stored_user_id_in_message(self) -> None:
        _, new_identity = _new_identity()
        signature = _authorization(
            self.private, "attacker", "d1", new_identity, 1)
        with self.assertRaises(ServiceError) as ctx:
            self.service.rotate_identity_key_verified(
                "d1", {"identity_key": new_identity, "expected_version": 1,
                       "signature": signature})
        self.assertEqual(ctx.exception.field, "signature")

    def test_failure_writes_nothing(self) -> None:
        events_before = self.service.list_key_events("d1", 0, 100)["events"]
        other, _ = _new_identity()
        with self.assertRaises(ServiceError):
            self.service.rotate_identity_key_verified(
                "d1", self._payload(private=other))
        device = self.service.store.find_by_device_id("d1")
        self.assertEqual(device.identity_key, self.identity)
        self.assertEqual(device.identity_key_version, 1)
        self.assertEqual(device.rotated_at, self.registered_at)
        self.assertEqual(
            self.service.list_key_events("d1", 0, 100)["events"],
            events_before)

    def test_ordinary_rotate_route_still_works(self) -> None:
        _, new_identity = _new_identity()
        body = self.service.rotate_identity_key(
            "d1", {"identity_key": new_identity})
        self.assertEqual(body["identity_key"], new_identity)
        device = self.service.store.find_by_device_id("d1")
        self.assertEqual(device.identity_key_version, 2)


class RotateVerifiedHTTPTest(unittest.TestCase):
    def setUp(self) -> None:
        self.server, self.service = create_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        self.private, self.identity = _new_identity()
        prekey = _new_prekey()
        from e2ee_backend.crypto import signed_prekey_proof_message
        proof = base64.b64encode(self.private.sign(
            signed_prekey_proof_message("u1", "d1", "k1", prekey))).decode()
        self.service.register_verified({
            "user_id": "u1", "device_id": "d1",
            "identity_key": self.identity,
            "signed_prekeys": [{"key_id": "k1", "public_key": prekey,
                                "signature": proof}],
        })

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _request(self, method, path, body=None, raw=None):
        connection = HTTPConnection("127.0.0.1", self.port, timeout=5)
        payload = raw if raw is not None else (
            json.dumps(body) if body is not None else None)
        headers = {"Content-Type": "application/json"} if payload else {}
        connection.request(method, path, body=payload, headers=headers)
        response = connection.getresponse()
        data = response.read().decode("utf-8")
        connection.close()
        return response.status, json.loads(data)

    def _payload(self, identity_key=None, expected_version=1, private=None):
        if identity_key is None:
            _, identity_key = _new_identity()
        signer = private or self.private
        return {"identity_key": identity_key,
                "expected_version": expected_version,
                "signature": _authorization(
                    signer, "u1", "d1", identity_key, expected_version)}

    def test_valid_rotation_200(self) -> None:
        _, new_identity = _new_identity()
        status, body = self._request(
            "POST", "/v1/devices/d1/identity-key/rotate-verified",
            self._payload(identity_key=new_identity))
        self.assertEqual(status, 200)
        self.assertEqual(set(body),
                         {"device_id", "identity_key", "rotated_at"})
        self.assertEqual(body["identity_key"], new_identity)

    def test_non_object_body_400_request_body(self) -> None:
        for raw in ("[]", "null", '"x"', "42", "not-json"):
            status, body = self._request(
                "POST", "/v1/devices/d1/identity-key/rotate-verified",
                raw=raw)
            self.assertEqual(status, 400, raw)
            self.assertEqual(body["field"], "request_body")

    def test_version_mismatch_409(self) -> None:
        status, body = self._request(
            "POST", "/v1/devices/d1/identity-key/rotate-verified",
            self._payload(expected_version=2))
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "expected_version")

    def test_bool_version_400(self) -> None:
        payload = self._payload()
        payload["expected_version"] = True
        status, body = self._request(
            "POST", "/v1/devices/d1/identity-key/rotate-verified", payload)
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "expected_version")

    def test_bad_signature_400(self) -> None:
        _, new_identity = _new_identity()
        status, body = self._request(
            "POST", "/v1/devices/d1/identity-key/rotate-verified",
            {"identity_key": new_identity, "expected_version": 1,
             "signature": "not-base64"})
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "signature")

    def test_proof_failure_400_signature(self) -> None:
        other, _ = _new_identity()
        status, body = self._request(
            "POST", "/v1/devices/d1/identity-key/rotate-verified",
            self._payload(private=other))
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "signature")

    def test_unknown_device_404(self) -> None:
        status, body = self._request(
            "POST", "/v1/devices/ghost/identity-key/rotate-verified",
            self._payload())
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "device_id")

    def test_revoked_device_409(self) -> None:
        self.service.revoke_device("d1")
        status, body = self._request(
            "POST", "/v1/devices/d1/identity-key/rotate-verified",
            self._payload())
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "device_id")

    def test_ordinary_rotate_route_still_works(self) -> None:
        _, new_identity = _new_identity()
        status, body = self._request(
            "POST", "/v1/devices/d1/identity-key/rotate",
            {"identity_key": new_identity})
        self.assertEqual(status, 200)
        self.assertEqual(body["identity_key"], new_identity)


class RotateVerifiedCLITest(unittest.TestCase):
    def setUp(self) -> None:
        self.server, service = create_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        self.private, self.identity = _new_identity()
        prekey = _new_prekey()
        from e2ee_backend.crypto import signed_prekey_proof_message
        proof = base64.b64encode(self.private.sign(
            signed_prekey_proof_message("u1", "d1", "k1", prekey))).decode()
        service.register_verified({
            "user_id": "u1", "device_id": "d1",
            "identity_key": self.identity,
            "signed_prekeys": [{"key_id": "k1", "public_key": prekey,
                                "signature": proof}],
        })

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _run(self, *arguments) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "e2ee_backend", "--base-url",
             self.base_url, *arguments],
            capture_output=True, text=True, timeout=15)

    def test_success_stdout_zero(self) -> None:
        _, new_identity = _new_identity()
        signature = _authorization(self.private, "u1", "d1", new_identity, 1)
        result = self._run("rotate-identity-key-verified",
                           "--device-id", "d1",
                           "--identity-key", new_identity,
                           "--expected-version", "1",
                           "--signature", signature)
        self.assertEqual(result.returncode, 0, result.stderr)
        body = json.loads(result.stdout.strip())
        self.assertEqual(set(body),
                         {"device_id", "identity_key", "rotated_at"})
        self.assertEqual(body["identity_key"], new_identity)
        self.assertFalse(result.stderr.strip())

    def test_same_key_idempotent_stdout_zero(self) -> None:
        signature = _authorization(self.private, "u1", "d1", self.identity, 1)
        result = self._run("rotate-identity-key-verified",
                           "--device-id", "d1",
                           "--identity-key", self.identity,
                           "--expected-version", "1",
                           "--signature", signature)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout.strip())["identity_key"],
                         self.identity)

    def test_version_conflict_stderr_nonzero(self) -> None:
        _, new_identity = _new_identity()
        signature = _authorization(self.private, "u1", "d1", new_identity, 2)
        result = self._run("rotate-identity-key-verified",
                           "--device-id", "d1",
                           "--identity-key", new_identity,
                           "--expected-version", "2",
                           "--signature", signature)
        self.assertEqual(result.returncode, 1)
        self.assertFalse(result.stdout.strip())
        body = json.loads(result.stderr.strip())
        self.assertEqual(body["field"], "expected_version")

    def test_unknown_device_stderr_nonzero(self) -> None:
        _, new_identity = _new_identity()
        signature = _authorization(self.private, "u1", "d1", new_identity, 1)
        result = self._run("rotate-identity-key-verified",
                           "--device-id", "ghost",
                           "--identity-key", new_identity,
                           "--expected-version", "1",
                           "--signature", signature)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(json.loads(result.stderr.strip())["field"],
                         "device_id")

    def test_bad_signature_stderr_signature(self) -> None:
        _, new_identity = _new_identity()
        result = self._run("rotate-identity-key-verified",
                           "--device-id", "d1",
                           "--identity-key", new_identity,
                           "--expected-version", "1",
                           "--signature", "not-base64")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(json.loads(result.stderr.strip())["field"],
                         "signature")


class RotateVerifiedPersistenceTest(unittest.TestCase):
    def _register(self, service, private, identity) -> None:
        prekey = _new_prekey()
        from e2ee_backend.crypto import signed_prekey_proof_message
        proof = base64.b64encode(private.sign(
            signed_prekey_proof_message("u1", "d1", "k1", prekey))).decode()
        service.register_verified({
            "user_id": "u1", "device_id": "d1",
            "identity_key": identity,
            "signed_prekeys": [{"key_id": "k1", "public_key": prekey,
                                "signature": proof}],
        })

    def test_rotation_and_version_survive_restart(self) -> None:
        path = tempfile.mktemp(suffix=".json")
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        private, identity = _new_identity()
        _, new_identity = _new_identity()

        first = DeviceService()
        attach_persistence(first, path)
        self._register(first, private, identity)
        first.rotate_identity_key_verified("d1", {
            "identity_key": new_identity, "expected_version": 1,
            "signature": _authorization(private, "u1", "d1",
                                        new_identity, 1)})

        second = DeviceService()
        attach_persistence(second, path)
        device = second.store.find_by_device_id("d1")
        self.assertEqual(device.identity_key, new_identity)
        self.assertEqual(device.identity_key_version, 2)
        self.assertTrue(device.rotated_at.endswith("+00:00"))
        events = second.list_key_events("d1", 0, 100)["events"]
        self.assertEqual([e["type"] for e in events],
                         ["registered", "identity_rotated"])
        # The version check survives the restart: replaying version 1
        # conflicts, and a same-key replay at version 2 needs a signature
        # by the *current* key.
        with self.assertRaises(ServiceError) as ctx:
            second.rotate_identity_key_verified("d1", {
                "identity_key": new_identity, "expected_version": 1,
                "signature": _authorization(private, "u1", "d1",
                                            new_identity, 1)})
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "expected_version")

    def test_failure_advances_no_generation_and_leaves_no_event(self) -> None:
        path = tempfile.mktemp(suffix=".json")
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        private, identity = _new_identity()
        service = DeviceService()
        store = attach_persistence(service, path)
        self._register(service, private, identity)
        generation_after_register = store.commit_seq

        other, _ = _new_identity()
        _, new_identity = _new_identity()
        with self.assertRaises(ServiceError):
            service.rotate_identity_key_verified("d1", {
                "identity_key": new_identity, "expected_version": 1,
                "signature": _authorization(other, "u1", "d1",
                                            new_identity, 1)})
        self.assertEqual(store.commit_seq, generation_after_register)
        self.assertEqual(
            [e["type"] for e in
             service.list_key_events("d1", 0, 100)["events"]],
            ["registered"])
        device = service.store.find_by_device_id("d1")
        self.assertEqual(device.identity_key, identity)
        self.assertEqual(device.identity_key_version, 1)

    def test_same_key_replay_consumes_no_generation(self) -> None:
        path = tempfile.mktemp(suffix=".json")
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        private, identity = _new_identity()
        service = DeviceService()
        store = attach_persistence(service, path)
        self._register(service, private, identity)
        generation_after_register = store.commit_seq

        service.rotate_identity_key_verified("d1", {
            "identity_key": identity, "expected_version": 1,
            "signature": _authorization(private, "u1", "d1", identity, 1)})
        self.assertEqual(store.commit_seq, generation_after_register)
        device = service.store.find_by_device_id("d1")
        self.assertEqual(device.identity_key_version, 1)
        self.assertEqual(device.rotated_at, device.registered_at)

    def test_persist_failure_rolls_back_the_rotation(self) -> None:
        from e2ee_backend.persistence import PersistenceUnavailable

        path = tempfile.mktemp(suffix=".json")
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        private, identity = _new_identity()
        service = DeviceService()
        state_store = attach_persistence(service, path)
        self._register(service, private, identity)

        def fail_save(_pending) -> None:
            raise OSError("simulated disk failure")

        state_store.save = fail_save  # type: ignore[assignment]
        _, new_identity = _new_identity()
        # The durable write fails inside the locked transaction: the HTTP
        # layer answers 503/data_file and the in-memory mutation is rolled
        # back to the last committed state.
        with self.assertRaises(PersistenceUnavailable):
            service.rotate_identity_key_verified("d1", {
                "identity_key": new_identity, "expected_version": 1,
                "signature": _authorization(private, "u1", "d1",
                                            new_identity, 1)})
        device = service.store.find_by_device_id("d1")
        self.assertEqual(device.identity_key, identity)
        self.assertEqual(device.identity_key_version, 1)
        self.assertEqual(device.rotated_at, device.registered_at)
        self.assertEqual(
            [e["type"] for e in
             service.list_key_events("d1", 0, 100)["events"]],
            ["registered"])


class RotateVerifiedInteractionTest(unittest.TestCase):
    """Verified rotation leaves sessions, proofs and other devices alone."""

    def setUp(self) -> None:
        self.service = DeviceService()
        self.private_a, self.identity_a = _new_identity()
        self.private_b, self.identity_b = _new_identity()
        from e2ee_backend.crypto import signed_prekey_proof_message
        prekey_a = _new_prekey()
        proof_a = base64.b64encode(self.private_a.sign(
            signed_prekey_proof_message("u1", "alice", "k1",
                                        prekey_a))).decode()
        self.service.register_verified({
            "user_id": "u1", "device_id": "alice",
            "identity_key": self.identity_a,
            "signed_prekeys": [{"key_id": "k1", "public_key": prekey_a,
                                "signature": proof_a}],
        })
        self.prekey_b = _new_prekey()
        proof_b = base64.b64encode(self.private_b.sign(
            signed_prekey_proof_message("u2", "bob", "k1",
                                        self.prekey_b))).decode()
        self.service.register_verified({
            "user_id": "u2", "device_id": "bob",
            "identity_key": self.identity_b,
            "signed_prekeys": [{"key_id": "k1", "public_key": self.prekey_b,
                                "signature": proof_b}],
        })
        self.session = self.service.create_session({
            "initiator_device_id": "alice", "recipient_device_id": "bob",
            "prekey_id": "k1", "ephemeral_key": _new_prekey()})

    def _rotate_bob(self, identity_key, version=1, private=None):
        signer = private or self.private_b
        return self.service.rotate_identity_key_verified("bob", {
            "identity_key": identity_key, "expected_version": version,
            "signature": _authorization(signer, "u2", "bob",
                                        identity_key, version)})

    def test_existing_session_snapshot_stays_frozen(self) -> None:
        _, new_identity = _new_identity()
        self._rotate_bob(new_identity)
        snapshot = self.service.get_session(self.session["session_id"])
        self.assertEqual(snapshot["identity_key"], self.identity_b)

    def test_saved_prekey_proof_is_not_replaced(self) -> None:
        _, new_identity = _new_identity()
        self._rotate_bob(new_identity)
        proof = self.service.get_prekey_proof("bob", "k1")
        self.assertEqual(proof["identity_key"], self.identity_b)
        self.assertEqual(proof["public_key"], self.prekey_b)

    def test_other_device_is_unaffected(self) -> None:
        _, new_identity = _new_identity()
        self._rotate_bob(new_identity)
        alice = self.service.store.find_by_device_id("alice")
        self.assertEqual(alice.identity_key, self.identity_a)
        self.assertEqual(alice.identity_key_version, 1)
        events = self.service.list_key_events("alice", 0, 100)["events"]
        self.assertEqual([e["type"] for e in events], ["registered"])


if __name__ == "__main__":
    unittest.main()
