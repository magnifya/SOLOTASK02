"""Tests for signature-authorized identity-key rotation.

Covers the service, HTTP (real loopback socket), CLI (real subprocess),
key-audit and persistence layers for:

* POST /v1/devices/{device_id}/identity-key/rotate-verified

The rotation applies only when the standard-base64 64-byte Ed25519
signature verifies over the domain-separated canonical rotation message
(``E2EE-IDENTITY-ROTATION-V1``) against the device's *current* identity
key, and ``expected_version`` equals the current ``identity_key_version``.
A same-key authorized rotation is a no-op; a different key refreshes the
identity and timestamp, raises the version and appends the usual
``identity_rotated`` audit event. Nothing changes on failure.
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

from e2ee_backend.crypto import identity_rotation_proof_message
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


def _x25519_b64() -> str:
    return _raw_b64(x25519.X25519PrivateKey.generate().public_key())


def _x25519_der_b64() -> str:
    """X25519 public key as base64 DER SPKI — never parses as Ed25519."""
    der = x25519.X25519PrivateKey.generate().public_key().public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo)
    return base64.b64encode(der).decode()


def _authorization(private, user_id, device_id, identity_key,
                   expected_version) -> str:
    message = identity_rotation_proof_message(
        user_id, device_id, identity_key, expected_version)
    return base64.b64encode(private.sign(message)).decode()


def _register_payload(device_id="d1", user_id="u1", identity_key=None,
                      prekeys=None) -> dict:
    return {
        "user_id": user_id,
        "device_id": device_id,
        "identity_key": identity_key or _x25519_b64(),
        "signed_prekeys": (prekeys if prekeys is not None
                           else [{"key_id": "k1",
                                  "public_key": _x25519_b64()}]),
    }


class RotateVerifiedServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = DeviceService()
        self.private, self.identity = _new_identity()
        self.service.register(
            _register_payload(identity_key=self.identity))
        self.registered_at = self.service.get_device("d1")["registered_at"]

    def _payload(self, identity_key=None, expected_version=1,
                 signature=None, private=None, user_id="u1",
                 device_id="d1"):
        if identity_key is None:
            _, identity_key = _new_identity()
        if signature is None:
            signer = private or self.private
            signature = _authorization(
                signer, user_id, device_id, identity_key, expected_version)
        return {"identity_key": identity_key,
                "expected_version": expected_version,
                "signature": signature}

    def _version(self) -> int:
        return self.service.store.find_by_device_id("d1").identity_key_version

    def test_valid_authorization_rotates_200(self) -> None:
        _, new_key = _new_identity()
        body = self.service.rotate_identity_key_verified(
            "d1", self._payload(identity_key=new_key))
        self.assertEqual(set(body),
                         {"device_id", "identity_key", "rotated_at"})
        self.assertEqual(body["identity_key"], new_key)
        self.assertNotEqual(body["rotated_at"], self.registered_at)
        self.assertTrue(body["rotated_at"].endswith("+00:00"))
        self.assertEqual(self._version(), 2)
        events = self.service.list_key_events("d1", 0, 100)["events"]
        self.assertEqual([e["type"] for e in events],
                         ["registered", "identity_rotated"])
        self.assertEqual(events[-1]["payload"],
                         {"old_identity_key": self.identity,
                          "new_identity_key": new_key})

    def test_same_key_valid_authorization_is_noop(self) -> None:
        body = self.service.rotate_identity_key_verified(
            "d1", self._payload(identity_key=self.identity))
        self.assertEqual(body["identity_key"], self.identity)
        self.assertEqual(body["rotated_at"], self.registered_at)
        self.assertEqual(self._version(), 1)
        events = self.service.list_key_events("d1", 0, 100)["events"]
        self.assertEqual([e["type"] for e in events], ["registered"])

    def test_rotate_back_to_old_key_still_raises_version(self) -> None:
        private_b, key_b = _new_identity()
        self.service.rotate_identity_key_verified(
            "d1", self._payload(identity_key=key_b))
        self.assertEqual(self._version(), 2)
        # Rotate back to the original key, authorized by the *current* key
        # (key_b): the version climbs, never resets.
        body = self.service.rotate_identity_key_verified(
            "d1", self._payload(identity_key=self.identity,
                                expected_version=2,
                                private=private_b))
        self.assertEqual(body["identity_key"], self.identity)
        self.assertEqual(self._version(), 3)
        events = self.service.list_key_events("d1", 0, 100)["events"]
        self.assertEqual([e["type"] for e in events],
                         ["registered", "identity_rotated",
                          "identity_rotated"])

    def test_body_not_object_400_request_body(self) -> None:
        for payload in (None, [], "x", 42):
            with self.subTest(payload=payload):
                with self.assertRaises(ServiceError) as ctx:
                    self.service.rotate_identity_key_verified("d1", payload)
                self.assertEqual(ctx.exception.status_code, 400)
                self.assertEqual(ctx.exception.field, "request_body")

    def test_missing_or_typed_fields_400(self) -> None:
        _, new_key = _new_identity()
        good_sig = _authorization(self.private, "u1", "d1", new_key, 1)
        cases = [
            ({}, "identity_key"),
            ({"identity_key": new_key}, "expected_version"),
            ({"identity_key": new_key, "expected_version": 1}, "signature"),
            ({"identity_key": "", "expected_version": 1,
              "signature": good_sig}, "identity_key"),
            ({"identity_key": 9, "expected_version": 1,
              "signature": good_sig}, "identity_key"),
            ({"identity_key": new_key, "expected_version": 1,
              "signature": ""}, "signature"),
            ({"identity_key": new_key, "expected_version": 1,
              "signature": 8}, "signature"),
        ]
        for payload, field in cases:
            with self.subTest(field=field):
                with self.assertRaises(ServiceError) as ctx:
                    self.service.rotate_identity_key_verified("d1", payload)
                self.assertEqual(ctx.exception.status_code, 400)
                self.assertEqual(ctx.exception.field, field)

    def test_expected_version_must_be_positive_int_no_bool(self) -> None:
        for bad in (True, False, 0, -1, "1", 1.5, None):
            with self.subTest(bad=bad):
                with self.assertRaises(ServiceError) as ctx:
                    self.service.rotate_identity_key_verified(
                        "d1", self._payload(expected_version=bad))
                self.assertEqual(ctx.exception.status_code, 400)
                self.assertEqual(ctx.exception.field, "expected_version")

    def test_new_key_must_be_ed25519(self) -> None:
        # Raw 32 bytes cannot distinguish the two curves, but the DER OID
        # can: an X25519 key in DER/SPKI form is not an Ed25519 key.
        for bad in ("not-a-key", _x25519_der_b64()):
            with self.subTest(bad=bad):
                with self.assertRaises(ServiceError) as ctx:
                    self.service.rotate_identity_key_verified(
                        "d1", self._payload(identity_key=bad))
                self.assertEqual(ctx.exception.status_code, 400)
                self.assertEqual(ctx.exception.field, "identity_key")

    def test_bad_signature_encoding_400_signature(self) -> None:
        _, new_key = _new_identity()
        for bad in ("@@@@", "abc", "a" * 88,
                    base64.b64encode(b"\x00" * 63).decode(),
                    base64.b64encode(b"\x00" * 65).decode()):
            payload = self._payload(identity_key=new_key, signature=bad)
            with self.subTest(bad=bad):
                with self.assertRaises(ServiceError) as ctx:
                    self.service.rotate_identity_key_verified("d1", payload)
                self.assertEqual(ctx.exception.status_code, 400)
                self.assertEqual(ctx.exception.field, "signature")

    def test_signature_must_be_canonical_base64(self) -> None:
        _, new_key = _new_identity()
        good = _authorization(self.private, "u1", "d1", new_key, 1)
        # URL-safe alphabet and stripped padding are not canonical.
        for bad in (good.replace("+", "-").replace("/", "_"),
                    good.rstrip("=")):
            payload = self._payload(identity_key=new_key, signature=bad)
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

    def test_current_key_not_ed25519_400_identity_key(self) -> None:
        service = DeviceService()
        x_key = _x25519_der_b64()
        service.register(_register_payload(identity_key=x_key))
        _, new_key = _new_identity()
        payload = {"identity_key": new_key, "expected_version": 1,
                   "signature": _authorization(
                       self.private, "u1", "d1", new_key, 1)}
        with self.assertRaises(ServiceError) as ctx:
            service.rotate_identity_key_verified("d1", payload)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.field, "identity_key")

    def test_version_mismatch_409_expected_version(self) -> None:
        for bad_version in (2, 99):
            with self.subTest(bad_version=bad_version):
                with self.assertRaises(ServiceError) as ctx:
                    self.service.rotate_identity_key_verified(
                        "d1", self._payload(expected_version=bad_version))
                self.assertEqual(ctx.exception.status_code, 409)
                self.assertEqual(ctx.exception.field, "expected_version")

    def test_version_mismatch_checked_before_signature(self) -> None:
        # A stale version with an otherwise valid signature is still 409.
        _, new_key = _new_identity()
        payload = self._payload(identity_key=new_key, expected_version=2)
        with self.assertRaises(ServiceError) as ctx:
            self.service.rotate_identity_key_verified("d1", payload)
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "expected_version")

    def test_bad_signature_400_signature(self) -> None:
        _, new_key = _new_identity()
        other, _ = _new_identity()
        cases = [
            # Signed by a different key.
            self._payload(identity_key=new_key, private=other),
            # Signed over a different new key than the request carries.
            {"identity_key": new_key, "expected_version": 1,
             "signature": _authorization(
                 self.private, "u1", "d1", self.identity, 1)},
            # Signed for a different device id.
            self._payload(identity_key=new_key, device_id="d2"),
            # Signed for a different user id.
            self._payload(identity_key=new_key, user_id="u2"),
        ]
        for payload in cases:
            with self.subTest(payload=payload):
                with self.assertRaises(ServiceError) as ctx:
                    self.service.rotate_identity_key_verified("d1", payload)
                self.assertEqual(ctx.exception.status_code, 400)
                self.assertEqual(ctx.exception.field, "signature")

    def test_failure_leaves_state_untouched(self) -> None:
        _, new_key = _new_identity()
        with self.assertRaises(ServiceError):
            self.service.rotate_identity_key_verified(
                "d1", self._payload(identity_key=new_key,
                                    expected_version=7))
        device = self.service.store.find_by_device_id("d1")
        self.assertEqual(device.identity_key, self.identity)
        self.assertEqual(device.rotated_at, self.registered_at)
        self.assertEqual(self._version(), 1)
        events = self.service.list_key_events("d1", 0, 100)["events"]
        self.assertEqual([e["type"] for e in events], ["registered"])

    def test_rotation_does_not_touch_session_snapshot_or_other_device(
            self) -> None:
        bob_private, bob_key = _new_identity()
        self.service.register(_register_payload(
            device_id="bob", identity_key=bob_key,
            prekeys=[{"key_id": "p1", "public_key": _x25519_b64()}]))
        session = self.service.create_session({
            "initiator_device_id": "d1", "recipient_device_id": "bob",
            "prekey_id": "p1", "ephemeral_key": _x25519_b64()})
        _, new_bob_key = _new_identity()
        self.service.rotate_identity_key_verified(
            "bob", {"identity_key": new_bob_key, "expected_version": 1,
                    "signature": _authorization(
                        bob_private, "u1", "bob", new_bob_key, 1)})
        # The existing session snapshot keeps the frozen identity key.
        snapshot = self.service.get_session(session["session_id"])
        self.assertEqual(snapshot["identity_key"], bob_key)
        # The other device is untouched.
        d1 = self.service.store.find_by_device_id("d1")
        self.assertEqual(d1.identity_key, self.identity)
        self.assertEqual(d1.identity_key_version, 1)


class RotateVerifiedHTTPTest(unittest.TestCase):
    def setUp(self) -> None:
        self.server, self.service = create_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        self.private, self.identity = _new_identity()
        self.service.register(
            _register_payload(identity_key=self.identity))

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _request(self, method: str, path: str, body: object = None,
                 raw: str | None = None):
        connection = HTTPConnection("127.0.0.1", self.port, timeout=5)
        payload = raw if raw is not None else (
            json.dumps(body) if body is not None else None)
        headers = {"Content-Type": "application/json"} if payload else {}
        connection.request(method, path, body=payload, headers=headers)
        response = connection.getresponse()
        data = response.read().decode("utf-8")
        connection.close()
        return response.status, json.loads(data)

    def _payload(self, identity_key=None, expected_version=1) -> dict:
        if identity_key is None:
            _, identity_key = _new_identity()
        return {"identity_key": identity_key,
                "expected_version": expected_version,
                "signature": _authorization(
                    self.private, "u1", "d1", identity_key,
                    expected_version)}

    def test_rotate_verified_200(self) -> None:
        _, new_key = _new_identity()
        status, body = self._request(
            "POST", "/v1/devices/d1/identity-key/rotate-verified",
            self._payload(identity_key=new_key))
        self.assertEqual(status, 200)
        self.assertEqual(body["identity_key"], new_key)
        self.assertTrue(body["rotated_at"].endswith("+00:00"))

    def test_invalid_json_400_request_body(self) -> None:
        status, body = self._request(
            "POST", "/v1/devices/d1/identity-key/rotate-verified",
            raw="{not json")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "request_body")

    def test_non_object_400_request_body(self) -> None:
        status, body = self._request(
            "POST", "/v1/devices/d1/identity-key/rotate-verified", [1, 2])
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "request_body")

    def test_bool_version_400(self) -> None:
        payload = self._payload()
        payload["expected_version"] = True
        status, body = self._request(
            "POST", "/v1/devices/d1/identity-key/rotate-verified", payload)
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "expected_version")

    def test_unknown_device_404(self) -> None:
        status, body = self._request(
            "POST", "/v1/devices/ghost/identity-key/rotate-verified",
            self._payload())
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "device_id")

    def test_version_mismatch_409(self) -> None:
        status, body = self._request(
            "POST", "/v1/devices/d1/identity-key/rotate-verified",
            self._payload(expected_version=5))
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "expected_version")

    def test_bad_signature_400(self) -> None:
        payload = self._payload()
        payload["signature"] = base64.b64encode(b"\x00" * 64).decode()
        status, body = self._request(
            "POST", "/v1/devices/d1/identity-key/rotate-verified", payload)
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "signature")

    def test_plain_rotate_route_still_works(self) -> None:
        new_key = _x25519_b64()
        status, body = self._request(
            "POST", "/v1/devices/d1/identity-key/rotate",
            {"identity_key": new_key})
        self.assertEqual(status, 200)
        self.assertEqual(body["identity_key"], new_key)


class RotateVerifiedCLITest(unittest.TestCase):
    def setUp(self) -> None:
        self.server, service = create_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        self.private, self.identity = _new_identity()
        service.register(_register_payload(identity_key=self.identity))

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _run(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "e2ee_backend", "--base-url",
             self.base_url, *arguments],
            capture_output=True, text=True, timeout=15)

    def test_rotate_verified_success(self) -> None:
        _, new_key = _new_identity()
        signature = _authorization(self.private, "u1", "d1", new_key, 1)
        result = self._run("rotate-identity-key-verified",
                           "--device-id", "d1",
                           "--identity-key", new_key,
                           "--expected-version", "1",
                           "--signature", signature)
        self.assertEqual(result.returncode, 0, result.stderr)
        body = json.loads(result.stdout.strip())
        self.assertEqual(set(body),
                         {"device_id", "identity_key", "rotated_at"})
        self.assertEqual(body["identity_key"], new_key)

    def test_rotate_verified_failure_stderr_nonzero(self) -> None:
        _, new_key = _new_identity()
        signature = _authorization(self.private, "u1", "d1", new_key, 1)
        result = self._run("rotate-identity-key-verified",
                           "--device-id", "d1",
                           "--identity-key", new_key,
                           "--expected-version", "9",
                           "--signature", signature)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(json.loads(result.stderr.strip())["field"],
                         "expected_version")

    def test_rotate_verified_bad_signature_exit_1(self) -> None:
        _, new_key = _new_identity()
        result = self._run("rotate-identity-key-verified",
                           "--device-id", "d1",
                           "--identity-key", new_key,
                           "--expected-version", "1",
                           "--signature",
                           base64.b64encode(b"\x00" * 64).decode())
        self.assertEqual(result.returncode, 1)
        self.assertEqual(json.loads(result.stderr.strip())["field"],
                         "signature")


class RotateVerifiedPersistenceTest(unittest.TestCase):
    def test_version_and_rotation_survive_restart(self) -> None:
        path = tempfile.mktemp(suffix=".json")
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        private, identity = _new_identity()
        new_private, new_key = _new_identity()

        first = DeviceService()
        attach_persistence(first, path)
        first.register(_register_payload(identity_key=identity))
        first.rotate_identity_key_verified(
            "d1", {"identity_key": new_key, "expected_version": 1,
                   "signature": _authorization(
                       private, "u1", "d1", new_key, 1)})

        second = DeviceService()
        attach_persistence(second, path)
        device = second.store.find_by_device_id("d1")
        self.assertEqual(device.identity_key, new_key)
        self.assertEqual(device.identity_key_version, 2)
        # The version check still applies after the restart: replaying the
        # original request is a 409/expected_version, not a no-op.
        with self.assertRaises(ServiceError) as ctx:
            second.rotate_identity_key_verified(
                "d1", {"identity_key": new_key, "expected_version": 1,
                       "signature": _authorization(
                           private, "u1", "d1", new_key, 1)})
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "expected_version")
        # A same-key rotation authorized by the current key at the current
        # version is a no-op.
        body = second.rotate_identity_key_verified(
            "d1", {"identity_key": new_key, "expected_version": 2,
                   "signature": _authorization(
                       new_private, "u1", "d1", new_key, 2)})
        self.assertEqual(body["identity_key"], new_key)
        self.assertEqual(
            second.store.find_by_device_id("d1").identity_key_version, 2)

    def test_failure_advances_no_generation_and_leaves_no_event(self) -> None:
        path = tempfile.mktemp(suffix=".json")
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        private, identity = _new_identity()
        service = DeviceService()
        store = attach_persistence(service, path)
        service.register(_register_payload(identity_key=identity))
        generation_after_register = store.commit_seq

        _, new_key = _new_identity()
        other, _ = _new_identity()
        with self.assertRaises(ServiceError):
            service.rotate_identity_key_verified(
                "d1", {"identity_key": new_key, "expected_version": 1,
                       "signature": _authorization(
                           other, "u1", "d1", new_key, 1)})
        self.assertEqual(store.commit_seq, generation_after_register)
        device = service.store.find_by_device_id("d1")
        self.assertEqual(device.identity_key, identity)
        self.assertEqual(device.identity_key_version, 1)
        events = service.list_key_events("d1", 0, 100)["events"]
        self.assertEqual([e["type"] for e in events], ["registered"])

    def test_persist_failure_rolls_back_the_rotation(self) -> None:
        from e2ee_backend.persistence import PersistenceUnavailable

        path = tempfile.mktemp(suffix=".json")
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        private, identity = _new_identity()
        service = DeviceService()
        state_store = attach_persistence(service, path)
        service.register(_register_payload(identity_key=identity))
        rotated_at = service.store.find_by_device_id("d1").rotated_at

        def fail_save(_pending) -> None:
            raise OSError("simulated disk failure")

        state_store.save = fail_save  # type: ignore[assignment]
        _, new_key = _new_identity()
        # The durable write fails inside the locked transaction: the HTTP
        # layer answers 503/data_file and the in-memory mutation is rolled
        # back to the last committed state.
        with self.assertRaises(PersistenceUnavailable):
            service.rotate_identity_key_verified(
                "d1", {"identity_key": new_key, "expected_version": 1,
                       "signature": _authorization(
                           private, "u1", "d1", new_key, 1)})
        device = service.store.find_by_device_id("d1")
        self.assertEqual(device.identity_key, identity)
        self.assertEqual(device.rotated_at, rotated_at)
        self.assertEqual(device.identity_key_version, 1)
        events = service.list_key_events("d1", 0, 100)["events"]
        self.assertEqual([e["type"] for e in events], ["registered"])


if __name__ == "__main__":
    unittest.main()
