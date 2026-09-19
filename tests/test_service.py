"""Tests for the service layer: validation, status codes and isolation."""
import base64
import threading
import unittest
from datetime import datetime

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


class _SessionTestBase(unittest.TestCase):
    """Registers an initiator ``da`` and a recipient ``db`` with one pre-key."""

    INITIATOR = "da"
    RECIPIENT = "db"
    PREKEY = "pk"

    def setUp(self) -> None:
        self.service = DeviceService()
        self.init_identity = _raw_key_b64()
        self.recp_identity = _raw_key_b64()
        self.recp_prekey = _raw_key_b64()
        self.service.register({
            "user_id": "u1", "device_id": self.INITIATOR,
            "identity_key": self.init_identity,
            "signed_prekeys": [{"key_id": "ki", "public_key": _raw_key_b64()}]})
        self.service.register({
            "user_id": "u2", "device_id": self.RECIPIENT,
            "identity_key": self.recp_identity,
            "signed_prekeys": [{"key_id": self.PREKEY,
                                "public_key": self.recp_prekey}]})

    def _session_payload(self, **overrides: object) -> dict:
        payload = {
            "initiator_device_id": self.INITIATOR,
            "recipient_device_id": self.RECIPIENT,
            "prekey_id": self.PREKEY,
            "ephemeral_key": _raw_key_b64(),
        }
        payload.update(overrides)
        return payload


class SessionNegotiationTest(_SessionTestBase):
    def test_created_session_has_exactly_eight_fields(self) -> None:
        body = self.service.create_session(self._session_payload())
        self.assertEqual(set(body), {
            "session_id", "initiator_device_id", "recipient_device_id",
            "prekey_id", "ephemeral_key", "identity_key", "public_key",
            "created_at"})

    def test_echoes_four_input_fields(self) -> None:
        ephemeral = _raw_key_b64()
        body = self.service.create_session(self._session_payload(
            ephemeral_key=ephemeral))
        self.assertEqual(body["initiator_device_id"], self.INITIATOR)
        self.assertEqual(body["recipient_device_id"], self.RECIPIENT)
        self.assertEqual(body["prekey_id"], self.PREKEY)
        self.assertEqual(body["ephemeral_key"], ephemeral)

    def test_identity_and_public_key_come_from_recipient(self) -> None:
        body = self.service.create_session(self._session_payload())
        self.assertEqual(body["identity_key"], self.recp_identity)
        self.assertEqual(body["public_key"], self.recp_prekey)

    def test_session_id_is_unique_and_nonempty(self) -> None:
        first = self.service.create_session(self._session_payload())
        second = self.service.create_session(self._session_payload())
        self.assertTrue(first["session_id"])
        self.assertNotEqual(first["session_id"], second["session_id"])

    def test_repeated_post_creates_a_new_session(self) -> None:
        payload = self._session_payload()
        first = self.service.create_session(payload)
        second = self.service.create_session(payload)
        self.assertNotEqual(first["session_id"], second["session_id"])
        self.assertEqual(self.service.get_session(first["session_id"])["session_id"],
                         first["session_id"])
        self.assertEqual(self.service.get_session(second["session_id"])["session_id"],
                         second["session_id"])

    def test_created_at_is_utc_iso8601_with_zulu_offset(self) -> None:
        from datetime import timedelta

        body = self.service.create_session(self._session_payload())
        stamp = body["created_at"]
        self.assertTrue(stamp.endswith("+00:00"))
        parsed = datetime.fromisoformat(stamp)
        self.assertEqual(parsed.utcoffset(), timedelta(0))


class SessionValidationTest(_SessionTestBase):
    def _assert_400(self, payload: object, field: str) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.create_session(payload)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.field, field)

    def test_body_must_be_object(self) -> None:
        self._assert_400(["nope"], "request_body")
        self._assert_400("nope", "request_body")

    def test_missing_fields(self) -> None:
        for name in ("initiator_device_id", "recipient_device_id",
                     "prekey_id", "ephemeral_key"):
            payload = self._session_payload()
            del payload[name]
            self._assert_400(payload, name)

    def test_fields_must_be_nonempty_strings(self) -> None:
        self._assert_400(self._session_payload(initiator_device_id=""),
                         "initiator_device_id")
        self._assert_400(self._session_payload(recipient_device_id=7),
                         "recipient_device_id")
        self._assert_400(self._session_payload(prekey_id=None), "prekey_id")
        self._assert_400(self._session_payload(ephemeral_key=["x"]),
                         "ephemeral_key")

    def test_ephemeral_key_must_use_public_key_encoding(self) -> None:
        self._assert_400(self._session_payload(ephemeral_key="not-a-key"),
                         "ephemeral_key")

    def test_initiator_equal_recipient_is_400_on_recipient_field(self) -> None:
        self._assert_400(
            self._session_payload(recipient_device_id=self.INITIATOR),
            "recipient_device_id")


class SessionLookupTest(_SessionTestBase):
    def test_unknown_initiator_is_404(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.create_session(
                self._session_payload(initiator_device_id="ghost"))
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "initiator_device_id")

    def test_unknown_recipient_is_404(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.create_session(
                self._session_payload(recipient_device_id="ghost"))
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "recipient_device_id")

    def test_unknown_prekey_is_404(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.create_session(
                self._session_payload(prekey_id="ghost"))
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "prekey_id")

    def test_revoked_initiator_is_409(self) -> None:
        self.service.revoke_device(self.INITIATOR)
        with self.assertRaises(ServiceError) as ctx:
            self.service.create_session(self._session_payload())
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "initiator_device_id")

    def test_revoked_recipient_is_409(self) -> None:
        self.service.revoke_device(self.RECIPIENT)
        with self.assertRaises(ServiceError) as ctx:
            self.service.create_session(self._session_payload())
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "recipient_device_id")

    def test_revoked_prekey_is_409(self) -> None:
        self.service.revoke_prekey(self.RECIPIENT, self.PREKEY)
        with self.assertRaises(ServiceError) as ctx:
            self.service.create_session(self._session_payload())
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "prekey_id")

    def test_failed_creation_writes_nothing(self) -> None:
        cases = (
            self._session_payload(initiator_device_id="ghost"),
            self._session_payload(recipient_device_id="ghost"),
            self._session_payload(prekey_id="ghost"),
        )
        for payload in cases:
            with self.assertRaises(ServiceError):
                self.service.create_session(payload)
        self.service.revoke_prekey(self.RECIPIENT, self.PREKEY)
        with self.assertRaises(ServiceError):
            self.service.create_session(self._session_payload())
        self.assertEqual(self.service.store._sessions, {})


class SessionSnapshotTest(_SessionTestBase):
    def test_get_unknown_session_is_404(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.get_session("deadbeef")
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "session_id")

    def test_get_returns_same_snapshot_as_create(self) -> None:
        created = self.service.create_session(self._session_payload())
        fetched = self.service.get_session(created["session_id"])
        self.assertEqual(fetched, created)

    def test_revocation_does_not_change_snapshot(self) -> None:
        created = self.service.create_session(self._session_payload())
        self.service.revoke_prekey(self.RECIPIENT, self.PREKEY)
        self.service.revoke_device(self.RECIPIENT)
        self.service.revoke_device(self.INITIATOR)
        self.assertEqual(self.service.get_session(created["session_id"]), created)


class SessionAtomicityTest(unittest.TestCase):
    """Creation and revocation are linearized under one lock: revoke-first
    makes creation fail (409, nothing written); create-first succeeds and the
    session snapshot is retained despite the subsequent revocation.

    The atomic contract is the storage-level critical section, so the race
    pits ``store.create_session`` directly against the revocation, with tiny
    randomized stalls so both orderings are actually exercised.
    """

    def test_create_vs_device_revoke(self) -> None:
        self._race(lambda store: store.revoke_device("db"),
                   "recipient_revoked", "recipient_device_id")

    def test_create_vs_prekey_revoke(self) -> None:
        self._race(lambda store: store.revoke_prekey_by_id("db", "pk"),
                   "prekey_revoked", "prekey_id")

    def _race(self, revoke: object, revoked_reason: str,
              conflict_field: str) -> None:
        import random
        import time

        from e2ee_backend.storage import SessionCreateError

        rng = random.Random(0xC0FFEE)
        seen_created = 0
        seen_conflict = 0
        for _ in range(500):
            service = DeviceService()
            service.register({
                "user_id": "u1", "device_id": "da",
                "identity_key": _raw_key_b64(),
                "signed_prekeys": []})
            service.register({
                "user_id": "u2", "device_id": "db",
                "identity_key": _raw_key_b64(),
                "signed_prekeys": [{"key_id": "pk",
                                    "public_key": _raw_key_b64()}]})
            store = service.store
            ephemeral = _raw_key_b64()

            barrier = threading.Barrier(2)
            outcome: list = []

            def create() -> None:
                barrier.wait()
                time.sleep(rng.random() * 25e-6)
                try:
                    session = store.create_session(
                        "da", "db", "pk", ephemeral)
                    outcome.append(("created", session.session_id))
                except SessionCreateError as error:
                    outcome.append(("conflict", error.reason))

            def do_revoke() -> None:
                barrier.wait()
                time.sleep(rng.random() * 25e-6)
                revoke(store)

            threads = [threading.Thread(target=create),
                       threading.Thread(target=do_revoke)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=5)

            kind, value = outcome[0]
            if kind == "created":
                seen_created += 1
                # Create linearized first: one session exists and its snapshot
                # survives the revocation that landed afterwards.
                self.assertEqual(len(store._sessions), 1)
                view = service.get_session(value)
                self.assertEqual(view["session_id"], value)
                self.assertEqual(view["ephemeral_key"], ephemeral)
            else:
                seen_conflict += 1
                # Revocation linearized first: correct reason, nothing written.
                self.assertEqual(value, revoked_reason)
                self.assertEqual(store._sessions, {})

            if seen_created and seen_conflict:
                break

        self.assertGreater(seen_created, 0,
                           "create-first ordering never observed")
        self.assertGreater(seen_conflict, 0,
                           "revoke-first ordering never observed")


if __name__ == "__main__":
    unittest.main()
