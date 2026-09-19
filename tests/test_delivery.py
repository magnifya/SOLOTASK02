"""Tests for reliable delivery (retry/ack/status) and durable persistence."""
import base64
import json
import os
import tempfile
import unittest

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import x25519

from e2ee_backend.persistence import (
    FilePersister,
    PersistenceError,
    load_store,
)
from e2ee_backend.service import DeviceService, ServiceError


def _raw_key_b64() -> str:
    key = x25519.X25519PrivateKey.generate().public_key()
    raw = key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return base64.b64encode(raw).decode()


def _register_payload(device_id: str, user_id: str = "u1") -> dict:
    return {
        "user_id": user_id,
        "device_id": device_id,
        "identity_key": _raw_key_b64(),
        "signed_prekeys": [{"key_id": "k1", "public_key": _raw_key_b64()}],
    }


def _message_payload(session_id: str, message_id: str = "m1",
                     sequence: int = 1, sender: str = "d1") -> dict:
    return {
        "session_id": session_id,
        "sender_device_id": sender,
        "message_id": message_id,
        "sequence": sequence,
        "nonce": base64.b64encode(b"0123456789ab").decode(),
        "ciphertext": base64.b64encode(b"ciphertext-and-tag").decode(),
    }


class DeliveryFixture:
    """A service with two devices, one session, and one delivered message."""

    def __init__(self, service: DeviceService = None) -> None:
        self.service = service if service is not None else DeviceService()
        self.service.register(_register_payload("d1"))
        self.service.register(_register_payload("d2"))
        session = self.service.create_session({
            "initiator_device_id": "d1",
            "recipient_device_id": "d2",
            "prekey_id": "k1",
            "ephemeral_key": _raw_key_b64(),
        })
        self.session_id = session["session_id"]
        self.service.post_message(_message_payload(self.session_id))

    def retry(self, attempt_id: str = "a1", device_id: str = "d2",
              message_id: str = "m1", session_id: str = None):
        return self.service.retry_message(
            session_id or self.session_id, message_id,
            {"device_id": device_id, "attempt_id": attempt_id})

    def ack(self, message_id: str = "m1", device_id: str = "d2",
            sequence: int = 1, session_id: str = None):
        return self.service.ack_message(
            session_id or self.session_id,
            {"device_id": device_id, "message_id": message_id,
             "sequence": sequence})

    def status(self, message_id: str = "m1", device_id: str = "d2",
               session_id: str = None):
        return self.service.message_status(
            session_id or self.session_id, message_id, device_id)


class RetryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fx = DeliveryFixture()

    def test_first_retry_is_201_and_counts(self) -> None:
        body, created = self.fx.retry()
        self.assertTrue(created)
        self.assertEqual(body, {"session_id": self.fx.session_id,
                                "message_id": "m1", "status": "pending",
                                "attempts": 1, "sequence": 1})

    def test_duplicate_attempt_is_200_and_does_not_count(self) -> None:
        self.fx.retry("a1")
        body, created = self.fx.retry("a1")
        self.assertFalse(created)
        self.assertEqual(body["attempts"], 1)

    def test_distinct_attempts_count_separately(self) -> None:
        self.fx.retry("a1")
        body, created = self.fx.retry("a2")
        self.assertTrue(created)
        self.assertEqual(body["attempts"], 2)

    def test_retry_after_ack_is_200_and_does_not_count(self) -> None:
        self.fx.retry("a1")
        self.fx.ack()
        body, created = self.fx.retry("a2")
        self.assertFalse(created)
        self.assertEqual(body["status"], "acked")
        self.assertEqual(body["attempts"], 1)

    def test_missing_and_empty_fields_are_400(self) -> None:
        for payload in ({}, {"device_id": "d2"}, {"attempt_id": "a1"},
                        {"device_id": "", "attempt_id": "a1"},
                        {"device_id": "d2", "attempt_id": 7}):
            with self.assertRaises(ServiceError) as ctx:
                self.fx.service.retry_message(
                    self.fx.session_id, "m1", payload)
            self.assertEqual(ctx.exception.status_code, 400, payload)
            self.assertIn(ctx.exception.field, ("device_id", "attempt_id"))

    def test_unknown_session_is_404(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.fx.retry(session_id="ghost")
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "session_id")

    def test_unknown_message_is_404(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.fx.retry(message_id="ghost")
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "message_id")

    def test_wrong_device_is_409(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.fx.retry(device_id="d1")  # initiator, not recipient
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "device_id")

    def test_unknown_device_is_409(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.fx.retry(device_id="ghost")
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "device_id")

    def test_revoked_recipient_is_409(self) -> None:
        self.fx.service.revoke_device("d2")
        with self.assertRaises(ServiceError) as ctx:
            self.fx.retry()
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "device_id")


class AckTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fx = DeliveryFixture()

    def test_first_ack_is_201_and_marks_acked(self) -> None:
        body, created = self.fx.ack()
        self.assertTrue(created)
        self.assertEqual(body["status"], "acked")
        self.assertEqual(body["sequence"], 1)
        self.assertEqual(self.fx.status()["status"], "acked")

    def test_duplicate_ack_is_200(self) -> None:
        self.fx.ack()
        body, created = self.fx.ack()
        self.assertFalse(created)
        self.assertEqual(body["status"], "acked")

    def test_sequence_mismatch_is_409(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.fx.ack(sequence=2)
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "sequence")
        self.assertEqual(self.fx.status()["status"], "pending")

    def test_non_integer_sequence_is_400(self) -> None:
        for bad in ("1", 1.5, True, None):
            with self.assertRaises(ServiceError) as ctx:
                self.fx.ack(sequence=bad)
            self.assertEqual(ctx.exception.status_code, 400, bad)
            self.assertEqual(ctx.exception.field, "sequence")

    def test_missing_fields_are_400(self) -> None:
        for payload in ({}, {"device_id": "d2"},
                        {"device_id": "d2", "message_id": "m1"},
                        {"device_id": "d2", "message_id": "", "sequence": 1}):
            with self.assertRaises(ServiceError) as ctx:
                self.fx.service.ack_message(self.fx.session_id, payload)
            self.assertEqual(ctx.exception.status_code, 400, payload)

    def test_unknown_session_and_message_are_404(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.fx.ack(session_id="ghost")
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "session_id")
        with self.assertRaises(ServiceError) as ctx:
            self.fx.ack(message_id="ghost")
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "message_id")

    def test_wrong_or_revoked_device_is_409(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.fx.ack(device_id="d1")
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "device_id")
        self.fx.service.revoke_device("d2")
        with self.assertRaises(ServiceError) as ctx:
            self.fx.ack()
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "device_id")


class StatusTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fx = DeliveryFixture()

    def test_status_returns_five_fields(self) -> None:
        self.fx.retry("a1")
        body = self.fx.status()
        self.assertEqual(body, {"session_id": self.fx.session_id,
                                "message_id": "m1", "status": "pending",
                                "attempts": 1, "sequence": 1})

    def test_initiator_may_query_status(self) -> None:
        body = self.fx.status(device_id="d1")
        self.assertEqual(body["status"], "pending")

    def test_unknown_ids_are_404(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.fx.status(session_id="ghost")
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "session_id")
        with self.assertRaises(ServiceError) as ctx:
            self.fx.status(message_id="ghost")
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "message_id")

    def test_outsider_or_revoked_device_is_409(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.fx.status(device_id="ghost")
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "device_id")
        self.fx.service.revoke_device("d2")
        with self.assertRaises(ServiceError) as ctx:
            self.fx.status(device_id="d2")
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "device_id")


class PersistenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "state.json")

    def _service(self) -> DeviceService:
        store = load_store(self.path)
        store.attach_persister(FilePersister(self.path))
        return DeviceService(store)

    def test_missing_file_starts_empty_and_is_created(self) -> None:
        service = self._service()
        self.assertFalse(os.path.exists(self.path))
        service.register(_register_payload("d1"))
        self.assertTrue(os.path.exists(self.path))
        with open(self.path, encoding="utf-8") as handle:
            data = json.load(handle)
        self.assertEqual(data["version"], 1)

    def test_restart_restores_dedup_sequence_revocation_and_ack(self) -> None:
        service = self._service()
        fx = DeliveryFixture(service)
        fx.retry("a1")
        fx.ack()

        # "Restart": rebuild the service purely from the data file.
        restored = self._service()

        # Ack state survives: status is acked, attempts kept.
        status = restored.message_status(fx.session_id, "m1", "d1")
        self.assertEqual(status["status"], "acked")
        self.assertEqual(status["attempts"], 1)
        # Retry dedup survives: the same attempt_id does not count again.
        body, created = restored.retry_message(
            fx.session_id, "m1", {"device_id": "d2", "attempt_id": "a1"})
        self.assertFalse(created)
        self.assertEqual(body["attempts"], 1)
        # Message-id dedup and the sequence cursor survive.
        with self.assertRaises(ServiceError) as ctx:
            restored.post_message(_message_payload(fx.session_id))
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "message_id")
        with self.assertRaises(ServiceError) as ctx:
            restored.post_message(
                _message_payload(fx.session_id, "m2", sequence=3))
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "sequence")
        body = restored.post_message(
            _message_payload(fx.session_id, "m2", sequence=2))
        self.assertEqual(body["sequence"], 2)

        # Revocation survives a restart as well.
        restored.revoke_device("d2")
        restarted = self._service()
        self.assertEqual(restarted.get_device("d2")["prekey_ids"], [])
        with self.assertRaises(ServiceError) as ctx:
            restarted.retry_message(fx.session_id, "m1",
                                    {"device_id": "d2", "attempt_id": "a3"})
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "device_id")

    def test_corrupt_file_refuses_to_load(self) -> None:
        with open(self.path, "w", encoding="utf-8") as handle:
            handle.write("{not json")
        with self.assertRaises(PersistenceError):
            load_store(self.path)

    def test_wrong_version_refuses_to_load(self) -> None:
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump({"version": 2, "devices": [], "sessions": [],
                       "messages": {}}, handle)
        with self.assertRaises(PersistenceError):
            load_store(self.path)

    def test_save_is_atomic_and_leaves_no_temp_files(self) -> None:
        service = self._service()
        service.register(_register_payload("d1"))
        service.register(_register_payload("d2"))
        self.assertEqual(os.listdir(self.dir), ["state.json"])
        restored = self._service()
        self.assertIsNotNone(restored.get_device("d1"))
        self.assertIsNotNone(restored.get_device("d2"))


if __name__ == "__main__":
    unittest.main()
