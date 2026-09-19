"""Tests for the service layer: validation, status codes and isolation."""
import base64
import threading
import unittest

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import x25519

from e2ee_backend.models import Device, SignedPreKey
from e2ee_backend.service import DeviceService, ServiceError
from e2ee_backend.storage import DeviceStore


def _raw_key_b64() -> str:
    key = x25519.X25519PrivateKey.generate().public_key()
    raw = key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return base64.b64encode(raw).decode()


def _payload(device_id: str = "d1", user_id: str = "u1") -> dict:
    return {
        "user_id": user_id,
        "device_id": device_id,
        "identity_key": _raw_key_b64(),
        "signed_prekeys": [
            {"key_id": "k1", "public_key": _raw_key_b64()},
            {"key_id": "k2", "public_key": _raw_key_b64()},
        ],
    }


class ServiceRegistrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = DeviceService()

    def test_register_returns_only_contract_fields(self) -> None:
        body = self.service.register(_payload())
        self.assertEqual(set(body), {"device_id", "registered_at"})
        self.assertEqual(body["device_id"], "d1")
        self.assertIsInstance(body["registered_at"], str)
        self.assertTrue(body["registered_at"])

    def test_duplicate_device_same_user_is_conflict(self) -> None:
        self.service.register(_payload())
        with self.assertRaises(ServiceError) as ctx:
            self.service.register(_payload())
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "device_id")

    def test_device_id_is_unique_even_across_users(self) -> None:
        self.service.register(_payload(device_id="shared", user_id="u1"))
        with self.assertRaises(ServiceError) as ctx:
            self.service.register(_payload(device_id="shared", user_id="u2"))
        self.assertEqual(ctx.exception.status_code, 409)

    def test_empty_prekey_array_is_accepted(self) -> None:
        payload = _payload(device_id="d-empty")
        payload["signed_prekeys"] = []
        body = self.service.register(payload)
        self.assertEqual(body["device_id"], "d-empty")


class ServiceValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = DeviceService()

    def _assert_400(self, payload: object, field: str) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.register(payload)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.field, field)

    def test_body_must_be_object(self) -> None:
        self._assert_400(["nope"], "request_body")
        self._assert_400("nope", "request_body")

    def test_missing_scalar_fields(self) -> None:
        for field in ("user_id", "device_id", "identity_key"):
            payload = _payload(device_id=f"d-{field}")
            del payload[field]
            self._assert_400(payload, field)

    def test_missing_signed_prekeys(self) -> None:
        payload = _payload(device_id="d-x")
        del payload["signed_prekeys"]
        self._assert_400(payload, "signed_prekeys")

    def test_empty_or_wrong_type_scalars(self) -> None:
        self._assert_400({**_payload(device_id="a"), "user_id": ""}, "user_id")
        self._assert_400({**_payload(device_id="b"), "device_id": 7}, "device_id")
        self._assert_400({**_payload(device_id="c"), "identity_key": None},
                         "identity_key")

    def test_signed_prekeys_must_be_array(self) -> None:
        self._assert_400({**_payload(device_id="e"), "signed_prekeys": {}},
                         "signed_prekeys")

    def test_prekey_element_must_be_object(self) -> None:
        payload = _payload(device_id="f")
        payload["signed_prekeys"] = ["x"]
        self._assert_400(payload, "signed_prekeys[0]")

    def test_prekey_missing_subfields(self) -> None:
        payload = _payload(device_id="g")
        payload["signed_prekeys"] = [{"key_id": "k1"}]
        self._assert_400(payload, "signed_prekeys[0].public_key")

        payload = _payload(device_id="h")
        payload["signed_prekeys"] = [{"public_key": _raw_key_b64()}]
        self._assert_400(payload, "signed_prekeys[0].key_id")

    def test_prekey_empty_subfields(self) -> None:
        payload = _payload(device_id="i")
        payload["signed_prekeys"] = [{"key_id": "", "public_key": _raw_key_b64()}]
        self._assert_400(payload, "signed_prekeys[0].key_id")

    def test_prekey_invalid_public_key(self) -> None:
        payload = _payload(device_id="j")
        payload["signed_prekeys"] = [{"key_id": "k1", "public_key": "garbage"}]
        self._assert_400(payload, "signed_prekeys[0].public_key")

    def test_invalid_identity_key(self) -> None:
        self._assert_400({**_payload(device_id="k"), "identity_key": "garbage"},
                         "identity_key")

    def test_duplicate_key_id_in_one_request(self) -> None:
        payload = _payload(device_id="l")
        payload["signed_prekeys"][1]["key_id"] = "k1"
        self._assert_400(payload, "signed_prekeys[1].key_id")


class ServiceQueryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = DeviceService()
        self.service.register(_payload())

    def test_show_returns_contract_fields(self) -> None:
        body = self.service.get_device("d1")
        self.assertEqual(set(body),
                         {"identity_key", "prekey_ids", "registered_at"})

    def test_prekey_order_is_stable_and_complete(self) -> None:
        first = self.service.get_device("d1")
        second = self.service.get_device("d1")
        self.assertEqual(first, second)
        self.assertEqual(first["prekey_ids"], ["k1", "k2"])

    def test_unknown_device_is_404(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.get_device("ghost")
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "device_id")

    def test_revoked_prekeys_excluded_others_unaffected(self) -> None:
        service = DeviceService()
        service.register(_payload(device_id="d1", user_id="u1"))
        service.register(_payload(device_id="d2", user_id="u1"))

        device = service.store.find_by_device_id("d1")
        self.assertTrue(service.store.revoke_prekey(device, "k1"))

        self.assertEqual(service.get_device("d1")["prekey_ids"], ["k2"])
        # d2 under the same user must be untouched.
        self.assertEqual(service.get_device("d2")["prekey_ids"], ["k1", "k2"])


class ServiceRevocationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = DeviceService()
        self.service.register(_payload(device_id="d1", user_id="u1"))
        self.service.register(_payload(device_id="d2", user_id="u1"))

    def test_revoke_device_body(self) -> None:
        body = self.service.revoke_device("d1")
        self.assertEqual(body, {"device_id": "d1", "revoked": True})

    def test_revoke_device_is_idempotent(self) -> None:
        first = self.service.revoke_device("d1")
        second = self.service.revoke_device("d1")
        self.assertEqual(first, second)

    def test_revoke_unknown_device_is_404(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.revoke_device("ghost")
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "device_id")

    def test_revoked_device_has_empty_prekeys_but_kept_identity_and_timestamp(self) -> None:
        before = self.service.get_device("d1")
        self.service.revoke_device("d1")
        after = self.service.get_device("d1")
        self.assertEqual(after["prekey_ids"], [])
        self.assertEqual(after["identity_key"], before["identity_key"])
        self.assertEqual(after["registered_at"], before["registered_at"])

    def test_revoke_single_prekey_excludes_only_target(self) -> None:
        body = self.service.revoke_prekey("d1", "k1")
        self.assertEqual(body,
                         {"device_id": "d1", "key_id": "k1", "revoked": True})
        self.assertEqual(self.service.get_device("d1")["prekey_ids"], ["k2"])
        # Other device under the same user is untouched, order preserved.
        self.assertEqual(self.service.get_device("d2")["prekey_ids"], ["k1", "k2"])

    def test_revoke_prekey_is_idempotent(self) -> None:
        first = self.service.revoke_prekey("d1", "k1")
        second = self.service.revoke_prekey("d1", "k1")
        self.assertEqual(first, second)
        self.assertEqual(self.service.get_device("d1")["prekey_ids"], ["k2"])

    def test_revoke_prekey_unknown_device_is_404(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.revoke_prekey("ghost", "k1")
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "device_id")

    def test_revoke_unknown_prekey_is_404_and_leaves_keys_intact(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.revoke_prekey("d1", "nope")
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "key_id")
        self.assertEqual(self.service.get_device("d1")["prekey_ids"], ["k1", "k2"])

    def test_revoke_device_after_single_prekey_still_lists_empty(self) -> None:
        self.service.revoke_prekey("d1", "k1")
        self.service.revoke_device("d1")
        self.assertEqual(self.service.get_device("d1")["prekey_ids"], [])


class ServiceConcurrencyTest(unittest.TestCase):
    def test_gets_observe_only_before_or_after_snapshots(self) -> None:
        service = DeviceService()
        service.register(_payload(device_id="d1", user_id="u1"))

        observed = []
        stop = threading.Event()

        def reader() -> None:
            while not stop.is_set():
                ids = tuple(service.get_device("d1")["prekey_ids"])
                observed.append(ids)

        threads = [threading.Thread(target=reader) for _ in range(4)]
        for thread in threads:
            thread.start()
        service.revoke_device("d1")
        stop.set()
        for thread in threads:
            thread.join(timeout=2)

        # Only the full pre-revoke list or the empty post-revoke list may
        # appear: a partially-revoked snapshot would violate linearizability.
        self.assertTrue(observed)
        self.assertTrue(set(observed) <= {("k1", "k2"), ()})


class StorageIsolationTest(unittest.TestCase):
    def test_devices_under_same_user_are_independent(self) -> None:
        store = DeviceStore()
        a = Device("u1", "a", "ik-a", prekeys=[SignedPreKey("ka", "pa")])
        b = Device("u1", "b", "ik-b", prekeys=[SignedPreKey("kb", "pb")])
        self.assertTrue(store.add_device(a))
        self.assertTrue(store.add_device(b))
        self.assertEqual(store.active_prekey_ids(a), ["ka"])
        self.assertEqual(store.active_prekey_ids(b), ["kb"])
        store.revoke_prekey(a, "ka")
        self.assertEqual(store.active_prekey_ids(a), [])
        self.assertEqual(store.active_prekey_ids(b), ["kb"])


if __name__ == "__main__":
    unittest.main()
