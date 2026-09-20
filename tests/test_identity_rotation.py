"""Tests for identity-key rotation and pre-key replenishment.

Covers the service, HTTP (real loopback socket), CLI (real subprocess) and
persistence layers for:

* POST /v1/devices/{device_id}/identity-key/rotate
* POST /v1/devices/{device_id}/prekeys
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
from cryptography.hazmat.primitives.asymmetric import x25519

from e2ee_backend.http_app import create_server
from e2ee_backend.persistence import attach_persistence
from e2ee_backend.service import DeviceService, ServiceError


def _raw_key_b64() -> str:
    key = x25519.X25519PrivateKey.generate().public_key()
    raw = key.public_bytes(serialization.Encoding.Raw,
                           serialization.PublicFormat.Raw)
    return base64.b64encode(raw).decode()


def _register_payload(device_id: str = "d1", user_id: str = "u1",
                      identity_key: str | None = None,
                      prekeys: list | None = None) -> dict:
    return {
        "user_id": user_id,
        "device_id": device_id,
        "identity_key": identity_key or _raw_key_b64(),
        "signed_prekeys": (prekeys if prekeys is not None
                           else [{"key_id": "k1", "public_key": _raw_key_b64()}]),
    }


class RotationServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = DeviceService()
        self.old_key = _raw_key_b64()
        self.new_key = _raw_key_b64()
        self.service.register(_register_payload(identity_key=self.old_key))
        self.registered_at = self.service.get_device("d1")["registered_at"]

    def test_rotated_at_starts_equal_to_registered_at(self) -> None:
        view = self.service.store.find_by_device_id("d1")
        self.assertEqual(view.rotated_at, view.registered_at)

    def test_same_key_is_idempotent_and_keeps_timestamp(self) -> None:
        body = self.service.rotate_identity_key(
            "d1", {"identity_key": self.old_key})
        self.assertEqual(set(body),
                         {"device_id", "identity_key", "rotated_at"})
        self.assertEqual(body["identity_key"], self.old_key)
        self.assertEqual(body["rotated_at"], self.registered_at)
        self.assertEqual(self.service.get_device("d1")["identity_key"],
                         self.old_key)

    def test_changed_key_updates_timestamp(self) -> None:
        body = self.service.rotate_identity_key(
            "d1", {"identity_key": self.new_key})
        self.assertEqual(body["device_id"], "d1")
        self.assertEqual(body["identity_key"], self.new_key)
        self.assertNotEqual(body["rotated_at"], self.registered_at)
        self.assertTrue(body["rotated_at"].endswith("+00:00"))
        self.assertEqual(self.service.get_device("d1")["identity_key"],
                         self.new_key)

    def test_validation_errors_are_400_identity_key(self) -> None:
        for payload in ({}, {"identity_key": ""},
                        {"identity_key": 123},
                        {"identity_key": "not-a-public-key"}):
            with self.subTest(payload=payload):
                with self.assertRaises(ServiceError) as ctx:
                    self.service.rotate_identity_key("d1", payload)
                self.assertEqual(ctx.exception.status_code, 400)
                self.assertEqual(ctx.exception.field, "identity_key")
        # Failure leaves the stored key untouched.
        self.assertEqual(self.service.get_device("d1")["identity_key"],
                         self.old_key)

    def test_unknown_device_is_404(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.rotate_identity_key(
                "ghost", {"identity_key": self.new_key})
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "device_id")

    def test_revoked_device_is_409(self) -> None:
        self.service.revoke_device("d1")
        with self.assertRaises(ServiceError) as ctx:
            self.service.rotate_identity_key(
                "d1", {"identity_key": self.new_key})
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "device_id")


class AddPrekeyServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = DeviceService()
        self.key_a = _raw_key_b64()
        self.service.register(
            _register_payload(prekeys=[{"key_id": "k1", "public_key": self.key_a}]))

    def test_new_id_appends_in_order_201(self) -> None:
        key_b = _raw_key_b64()
        body, status = self.service.add_prekey(
            "d1", {"key_id": "k2", "public_key": key_b})
        self.assertEqual(status, 201)
        self.assertEqual(body, {"device_id": "d1", "key_id": "k2",
                                "public_key": key_b})
        self.assertEqual(self.service.get_device("d1")["prekey_ids"],
                         ["k1", "k2"])

    def test_same_id_same_key_not_revoked_is_200_idempotent(self) -> None:
        body, status = self.service.add_prekey(
            "d1", {"key_id": "k1", "public_key": self.key_a})
        self.assertEqual(status, 200)
        self.assertEqual(body, {"device_id": "d1", "key_id": "k1",
                                "public_key": self.key_a})
        self.assertEqual(self.service.get_device("d1")["prekey_ids"], ["k1"])

    def test_same_id_changed_key_is_409_key_id(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.add_prekey(
                "d1", {"key_id": "k1", "public_key": _raw_key_b64()})
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "key_id")

    def test_revoked_id_is_409_key_id(self) -> None:
        self.service.revoke_prekey("d1", "k1")
        with self.assertRaises(ServiceError) as ctx:
            self.service.add_prekey(
                "d1", {"key_id": "k1", "public_key": self.key_a})
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "key_id")

    def test_validation_errors(self) -> None:
        cases = [({}, "key_id"),
                 ({"key_id": "x"}, "public_key"),
                 ({"key_id": "", "public_key": self.key_a}, "key_id"),
                 ({"key_id": 9, "public_key": self.key_a}, "key_id"),
                 ({"key_id": "x", "public_key": ""}, "public_key"),
                 ({"key_id": "x", "public_key": "not-a-key"}, "public_key")]
        for payload, field in cases:
            with self.subTest(payload=payload):
                with self.assertRaises(ServiceError) as ctx:
                    self.service.add_prekey("d1", payload)
                self.assertEqual(ctx.exception.status_code, 400)
                self.assertEqual(ctx.exception.field, field)

    def test_unknown_device_is_404(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.add_prekey(
                "ghost", {"key_id": "x", "public_key": self.key_a})
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "device_id")

    def test_revoked_device_is_409(self) -> None:
        self.service.revoke_device("d1")
        with self.assertRaises(ServiceError) as ctx:
            self.service.add_prekey(
                "d1", {"key_id": "x", "public_key": self.key_a})
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "device_id")


class RotationInteractionTest(unittest.TestCase):
    """Rotation affects only new sessions; existing snapshots stay frozen."""

    def setUp(self) -> None:
        self.service = DeviceService()
        self.alice = _raw_key_b64()
        self.alice_new = _raw_key_b64()
        self.bob = _raw_key_b64()
        self.service.register(_register_payload(
            device_id="alice", identity_key=self.alice,
            prekeys=[{"key_id": "p1", "public_key": self.alice}]))
        self.service.register(_register_payload(
            device_id="bob", identity_key=self.bob,
            prekeys=[{"key_id": "p1", "public_key": self.bob}]))
        self.session = self.service.create_session({
            "initiator_device_id": "alice", "recipient_device_id": "bob",
            "prekey_id": "p1", "ephemeral_key": _raw_key_b64()})

    def test_rotation_does_not_change_existing_snapshot(self) -> None:
        self.service.rotate_identity_key(
            "bob", {"identity_key": _raw_key_b64()})
        snapshot = self.service.get_session(self.session["session_id"])
        self.assertEqual(snapshot["identity_key"], self.bob)

    def test_revoked_prekey_excluded_new_prekey_appends(self) -> None:
        self.service.add_prekey("bob", {"key_id": "p2",
                                        "public_key": _raw_key_b64()})
        self.service.revoke_prekey("bob", "p1")
        self.assertEqual(self.service.get_device("bob")["prekey_ids"], ["p2"])


class RotationHTTPTest(unittest.TestCase):
    def setUp(self) -> None:
        self.server, self.service = create_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        self.old_key = _raw_key_b64()
        self.service.register(
            _register_payload(device_id="d1", identity_key=self.old_key))

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

    def test_rotate_changed_key_200(self) -> None:
        new_key = _raw_key_b64()
        status, body = self._request(
            "POST", "/v1/devices/d1/identity-key/rotate",
            {"identity_key": new_key})
        self.assertEqual(status, 200)
        self.assertEqual(body["identity_key"], new_key)
        self.assertTrue(body["rotated_at"].endswith("+00:00"))

    def test_rotate_bad_key_400_field(self) -> None:
        status, body = self._request(
            "POST", "/v1/devices/d1/identity-key/rotate",
            {"identity_key": "nope"})
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "identity_key")

    def test_rotate_unknown_device_404(self) -> None:
        status, body = self._request(
            "POST", "/v1/devices/ghost/identity-key/rotate",
            {"identity_key": _raw_key_b64()})
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "device_id")

    def test_add_prekey_201_then_200_then_409(self) -> None:
        key_b = _raw_key_b64()
        status, body = self._request(
            "POST", "/v1/devices/d1/prekeys",
            {"key_id": "k9", "public_key": key_b})
        self.assertEqual(status, 201)
        self.assertEqual(body["key_id"], "k9")

        status, body = self._request(
            "POST", "/v1/devices/d1/prekeys",
            {"key_id": "k9", "public_key": key_b})
        self.assertEqual(status, 200)

        status, body = self._request(
            "POST", "/v1/devices/d1/prekeys",
            {"key_id": "k9", "public_key": _raw_key_b64()})
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "key_id")

    def test_add_prekey_bad_public_key_400(self) -> None:
        status, body = self._request(
            "POST", "/v1/devices/d1/prekeys",
            {"key_id": "kx", "public_key": "nope"})
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "public_key")


class RotationCLITest(unittest.TestCase):
    def setUp(self) -> None:
        self.server, service = create_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        service.register(_register_payload(device_id="d1"))

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _run(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "e2ee_backend", "--base-url",
             self.base_url, *arguments],
            capture_output=True, text=True, timeout=15)

    def test_rotate_identity_key_success(self) -> None:
        result = self._run("rotate-identity-key", "--device-id", "d1",
                           "--identity-key", _raw_key_b64())
        self.assertEqual(result.returncode, 0, result.stderr)
        body = json.loads(result.stdout.strip())
        self.assertEqual(set(body),
                         {"device_id", "identity_key", "rotated_at"})

    def test_rotate_identity_key_failure_stderr_nonzero(self) -> None:
        result = self._run("rotate-identity-key", "--device-id", "ghost",
                           "--identity-key", _raw_key_b64())
        self.assertEqual(result.returncode, 1)
        self.assertEqual(json.loads(result.stderr.strip())["field"],
                         "device_id")

    def test_add_prekey_created_and_idempotent(self) -> None:
        key = _raw_key_b64()
        first = self._run("add-prekey", "--device-id", "d1",
                          "--key-id", "kz", "--public-key", key)
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(json.loads(first.stdout.strip())["key_id"], "kz")

        second = self._run("add-prekey", "--device-id", "d1",
                           "--key-id", "kz", "--public-key", key)
        self.assertEqual(second.returncode, 0, second.stderr)

        conflict = self._run("add-prekey", "--device-id", "d1",
                             "--key-id", "kz", "--public-key", _raw_key_b64())
        self.assertEqual(conflict.returncode, 1)
        self.assertEqual(json.loads(conflict.stderr.strip())["field"],
                         "key_id")


class RotationPersistenceTest(unittest.TestCase):
    def test_rotation_and_added_prekey_survive_restart(self) -> None:
        path = tempfile.mktemp(suffix=".json")
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        new_key = _raw_key_b64()
        added_key = _raw_key_b64()

        first = DeviceService()
        attach_persistence(first, path)
        first.register(_register_payload(device_id="d1"))
        first.rotate_identity_key("d1", {"identity_key": new_key})
        first.add_prekey("d1", {"key_id": "k2", "public_key": added_key})

        second = DeviceService()
        attach_persistence(second, path)
        device = second.store.find_by_device_id("d1")
        self.assertEqual(device.identity_key, new_key)
        self.assertTrue(device.rotated_at.endswith("+00:00"))
        self.assertEqual(second.get_device("d1")["prekey_ids"],
                         ["k1", "k2"])

    def test_legacy_doc_without_rotated_at_defaults_to_registered_at(self) -> None:
        path = tempfile.mktemp(suffix=".json")
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        key = _raw_key_b64()
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({
                "version": 1,
                "devices": [{
                    "user_id": "u1", "device_id": "d1",
                    "identity_key": key, "registered_at":
                    "2026-01-01T00:00:00+00:00", "revoked": False,
                    "prekeys": []}],
                "sessions": [], "messages": {}, "delivery": [],
            }, handle)
        service = DeviceService()
        attach_persistence(service, path)
        device = service.store.find_by_device_id("d1")
        self.assertEqual(device.rotated_at, device.registered_at)


if __name__ == "__main__":
    unittest.main()
