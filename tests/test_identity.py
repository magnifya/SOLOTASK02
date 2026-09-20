"""Tests for identity-key rotation and pre-key replenishment.

Covers the service validation/status mapping, the HTTP routes over a real
socket, the snapshot-freeze guarantee (rotation only affects sessions
negotiated afterwards), and persistence of ``rotated_at`` across a state
snapshot/restore.
"""
import base64
import json
import threading
import unittest
from http.client import HTTPConnection

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import x25519

from e2ee_backend.http_app import create_server
from e2ee_backend.service import DeviceService, ServiceError
from e2ee_backend.storage import DeviceStore


def _raw_key_b64() -> str:
    key = x25519.X25519PrivateKey.generate().public_key()
    raw = key.public_bytes(serialization.Encoding.Raw,
                           serialization.PublicFormat.Raw)
    return base64.b64encode(raw).decode()


def _register_payload(device_id: str = "d1", user_id: str = "u1",
                      identity_key: str | None = None,
                      prekeys=None) -> dict:
    if prekeys is None:
        prekeys = [{"key_id": "k1", "public_key": _raw_key_b64()}]
    return {
        "user_id": user_id,
        "device_id": device_id,
        "identity_key": identity_key or _raw_key_b64(),
        "signed_prekeys": prekeys,
    }


class RotateIdentityServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = DeviceService()
        self.identity_key = _raw_key_b64()
        self.service.register(
            _register_payload(identity_key=self.identity_key))

    def test_rotated_at_starts_equal_to_registered_at(self) -> None:
        body = self.service.rotate_identity_key(
            "d1", {"identity_key": self.identity_key})
        view = self.service.get_device("d1")
        self.assertEqual(body["rotated_at"], view["registered_at"])

    def test_same_key_is_idempotent_and_keeps_timestamp(self) -> None:
        before = self.service.rotate_identity_key(
            "d1", {"identity_key": self.identity_key})["rotated_at"]
        after = self.service.rotate_identity_key(
            "d1", {"identity_key": self.identity_key})
        self.assertEqual(after["identity_key"], self.identity_key)
        self.assertEqual(after["rotated_at"], before)

    def test_different_key_updates_value_and_timestamp(self) -> None:
        new_key = _raw_key_b64()
        body = self.service.rotate_identity_key(
            "d1", {"identity_key": new_key})
        self.assertEqual(set(body),
                         {"device_id", "identity_key", "rotated_at"})
        self.assertEqual(body["device_id"], "d1")
        self.assertEqual(body["identity_key"], new_key)
        self.assertTrue(body["rotated_at"].endswith("+00:00"))
        self.assertEqual(self.service.get_device("d1")["identity_key"], new_key)

    def test_validation_errors_are_400_identity_key(self) -> None:
        for payload in ({}, {"identity_key": ""}, {"identity_key": 123},
                        {"identity_key": "not-a-key"}, {"other": "x"}):
            with self.assertRaises(ServiceError) as ctx:
                self.service.rotate_identity_key("d1", payload)
            self.assertEqual(ctx.exception.status_code, 400, payload)
            self.assertEqual(ctx.exception.field, "identity_key", payload)

    def test_unknown_device_is_404(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.rotate_identity_key(
                "ghost", {"identity_key": _raw_key_b64()})
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "device_id")

    def test_revoked_device_is_409(self) -> None:
        self.service.revoke_device("d1")
        with self.assertRaises(ServiceError) as ctx:
            self.service.rotate_identity_key(
                "d1", {"identity_key": _raw_key_b64()})
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "device_id")

    def test_rejected_rotation_does_not_change_state(self) -> None:
        original = self.service.get_device("d1")
        self.service.revoke_device("d1")
        with self.assertRaises(ServiceError):
            self.service.rotate_identity_key(
                "d1", {"identity_key": _raw_key_b64()})
        # Identity stays the original even though a new key was offered.
        view = self.service.store.find_by_device_id("d1")
        self.assertEqual(view.identity_key, original["identity_key"])


class AddPrekeyServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = DeviceService()
        self.service.register(
            _register_payload(prekeys=[{"key_id": "k1",
                                        "public_key": _raw_key_b64()}]))

    def test_new_id_is_appended_and_201(self) -> None:
        public_key = _raw_key_b64()
        body, status = self.service.add_prekey(
            "d1", {"key_id": "k2", "public_key": public_key})
        self.assertEqual(status, 201)
        self.assertEqual(body, {"device_id": "d1", "key_id": "k2",
                                "public_key": public_key})
        self.assertEqual(self.service.get_device("d1")["prekey_ids"],
                         ["k1", "k2"])

    def test_same_id_same_key_is_idempotent_200(self) -> None:
        public_key = self.service.store.find_by_device_id(
            "d1").prekeys[0].public_key
        body, status = self.service.add_prekey(
            "d1", {"key_id": "k1", "public_key": public_key})
        self.assertEqual(status, 200)
        self.assertEqual(body["public_key"], public_key)
        # No duplicate appended.
        self.assertEqual(self.service.get_device("d1")["prekey_ids"], ["k1"])

    def test_same_id_different_key_is_409(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.add_prekey(
                "d1", {"key_id": "k1", "public_key": _raw_key_b64()})
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "key_id")

    def test_revoked_id_is_409(self) -> None:
        public_key = self.service.store.find_by_device_id(
            "d1").prekeys[0].public_key
        self.service.revoke_prekey("d1", "k1")
        with self.assertRaises(ServiceError) as ctx:
            self.service.add_prekey(
                "d1", {"key_id": "k1", "public_key": public_key})
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "key_id")
        # Still revoked / still absent from the listing.
        self.assertEqual(self.service.get_device("d1")["prekey_ids"], [])

    def test_field_validation_400(self) -> None:
        cases = [({}, "key_id"),
                 ({"key_id": "", "public_key": _raw_key_b64()}, "key_id"),
                 ({"key_id": 9, "public_key": _raw_key_b64()}, "key_id"),
                 ({"key_id": "k9"}, "public_key"),
                 ({"key_id": "k9", "public_key": 1}, "public_key"),
                 ({"key_id": "k9", "public_key": "bad"}, "public_key")]
        for payload, field in cases:
            with self.assertRaises(ServiceError) as ctx:
                self.service.add_prekey("d1", payload)
            self.assertEqual(ctx.exception.status_code, 400, payload)
            self.assertEqual(ctx.exception.field, field, payload)

    def test_unknown_and_revoked_device(self) -> None:
        payload = {"key_id": "k9", "public_key": _raw_key_b64()}
        with self.assertRaises(ServiceError) as ctx:
            self.service.add_prekey("ghost", payload)
        self.assertEqual((ctx.exception.status_code, ctx.exception.field),
                         (404, "device_id"))
        self.service.revoke_device("d1")
        with self.assertRaises(ServiceError) as ctx:
            self.service.add_prekey("d1", payload)
        self.assertEqual((ctx.exception.status_code, ctx.exception.field),
                         (409, "device_id"))


class SnapshotFreezeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = DeviceService()
        self.old_identity = _raw_key_b64()
        self.prekey = _raw_key_b64()
        self.service.register(_register_payload(
            device_id="alice", user_id="u", identity_key=_raw_key_b64(),
            prekeys=[]))
        self.service.register(_register_payload(
            device_id="bob", user_id="u", identity_key=self.old_identity,
            prekeys=[{"key_id": "p1", "public_key": self.prekey}]))
        self.ephemeral = _raw_key_b64()

    def _negotiate(self, prekey_id: str = "p1") -> dict:
        return self.service.create_session({
            "initiator_device_id": "alice",
            "recipient_device_id": "bob",
            "prekey_id": prekey_id,
            "ephemeral_key": self.ephemeral,
        })

    def test_rotation_only_affects_new_sessions(self) -> None:
        old = self._negotiate()
        self.assertEqual(old["identity_key"], self.old_identity)

        new_identity = _raw_key_b64()
        self.service.rotate_identity_key(
            "bob", {"identity_key": new_identity})

        # The existing snapshot is frozen at the old identity key.
        self.assertEqual(
            self.service.get_session(old["session_id"])["identity_key"],
            self.old_identity)
        # A newly negotiated session carries the rotated identity key.
        new = self._negotiate()
        self.assertNotEqual(new["session_id"], old["session_id"])
        self.assertEqual(new["identity_key"], new_identity)

    def test_replenished_prekey_is_usable_revoked_is_not(self) -> None:
        fresh = _raw_key_b64()
        _, status = self.service.add_prekey(
            "bob", {"key_id": "p2", "public_key": fresh})
        self.assertEqual(status, 201)
        session = self._negotiate("p2")
        self.assertEqual(session["public_key"], fresh)

        self.service.revoke_prekey("bob", "p1")
        with self.assertRaises(ServiceError) as ctx:
            self._negotiate("p1")
        self.assertEqual((ctx.exception.status_code, ctx.exception.field),
                         (409, "prekey_id"))
        self.assertEqual(self.service.get_device("bob")["prekey_ids"], ["p2"])


class RotatedAtPersistenceTest(unittest.TestCase):
    def test_snapshot_and_restore_keeps_rotated_at(self) -> None:
        store = DeviceStore()
        service = DeviceService(store)
        identity_key = _raw_key_b64()
        service.register(_register_payload(identity_key=identity_key))
        rotated = _raw_key_b64()
        body = service.rotate_identity_key(
            "d1", {"identity_key": rotated})

        snapshot = store.snapshot_state()
        record = next(d for d in snapshot["devices"]
                      if d["device_id"] == "d1")
        self.assertEqual(record["identity_key"], rotated)
        self.assertEqual(record["rotated_at"], body["rotated_at"])

        restored = DeviceStore()
        restored.restore_state(snapshot)
        device = restored.find_by_device_id("d1")
        self.assertEqual(device.identity_key, rotated)
        self.assertEqual(device.rotated_at, body["rotated_at"])

    def test_legacy_document_without_rotated_at_defaults_to_registered(self) -> None:
        store = DeviceStore()
        store.restore_state({
            "devices": [{
                "user_id": "u", "device_id": "old",
                "identity_key": "ik", "registered_at": "t0",
                "revoked": False, "prekeys": [],
            }],
            "sessions": [], "messages": {}, "delivery": [],
        })
        self.assertEqual(store.find_by_device_id("old").rotated_at, "t0")


class IdentityHTTPTest(unittest.TestCase):
    def setUp(self) -> None:
        self.server, self.service = create_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        self.identity_key = _raw_key_b64()
        self._register()

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

    def _register(self) -> None:
        status, _ = self._request("POST", "/v1/devices", _register_payload(
            identity_key=self.identity_key))
        self.assertEqual(status, 201)

    def test_rotate_routes_and_status(self) -> None:
        status, body = self._request(
            "POST", "/v1/devices/d1/identity-key/rotate",
            {"identity_key": self.identity_key})
        self.assertEqual(status, 200)
        self.assertEqual(set(body),
                         {"device_id", "identity_key", "rotated_at"})

        new_key = _raw_key_b64()
        status, body = self._request(
            "POST", "/v1/devices/d1/identity-key/rotate",
            {"identity_key": new_key})
        self.assertEqual(status, 200)
        self.assertEqual(body["identity_key"], new_key)

        status, body = self._request(
            "POST", "/v1/devices/d1/identity-key/rotate",
            {"identity_key": "bad"})
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "identity_key")

        status, body = self._request(
            "POST", "/v1/devices/ghost/identity-key/rotate",
            {"identity_key": new_key})
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "device_id")

    def test_add_prekey_routes_and_status(self) -> None:
        public_key = _raw_key_b64()
        status, body = self._request(
            "POST", "/v1/devices/d1/prekeys",
            {"key_id": "k9", "public_key": public_key})
        self.assertEqual(status, 201)
        self.assertEqual(body, {"device_id": "d1", "key_id": "k9",
                                "public_key": public_key})

        status, _ = self._request(
            "POST", "/v1/devices/d1/prekeys",
            {"key_id": "k9", "public_key": public_key})
        self.assertEqual(status, 200)

        status, body = self._request(
            "POST", "/v1/devices/d1/prekeys",
            {"key_id": "k9", "public_key": _raw_key_b64()})
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "key_id")

        status, body = self._request(
            "POST", "/v1/devices/d1/prekeys", {"key_id": "k10"})
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "public_key")

    def test_add_prekey_route_does_not_swallow_revoke_route(self) -> None:
        # The existing .../prekeys/{id}/revoke route must still resolve.
        status, body = self._request(
            "POST", "/v1/devices/d1/prekeys/k1/revoke", {})
        self.assertEqual(status, 200)
        self.assertTrue(body["revoked"])


if __name__ == "__main__":
    unittest.main()
