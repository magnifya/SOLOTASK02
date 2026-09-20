"""Tests for reliable delivery (retry/ack/status) and durable state."""
from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.client import HTTPConnection

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import x25519

from e2ee_backend.http_app import create_server
from e2ee_backend.persistence import (
    STATE_VERSION,
    JsonStateStore,
    StateFileError,
    attach_persistence,
)
from e2ee_backend.service import DeviceService, ServiceError


def _raw_key_b64() -> str:
    key = x25519.X25519PrivateKey.generate().public_key()
    raw = key.public_bytes(serialization.Encoding.Raw,
                           serialization.PublicFormat.Raw)
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
        # Distinct per message id: a session rejects nonce replays.
        "nonce": base64.b64encode(f"nonce-{message_id}".encode()).decode(),
        "ciphertext": base64.b64encode(b"ciphertext-and-tag").decode(),
    }


class DeliveryFixture:
    """A service with devices d1/d2, one session d1->d2 and one message m1."""

    def __init__(self) -> None:
        self.service = DeviceService()
        self.service.register(_register_payload("d1"))
        self.service.register(_register_payload("d2"))
        session = self.service.create_session({
            "initiator_device_id": "d1",
            "recipient_device_id": "d2",
            "prekey_id": "k1",
            "ephemeral_key": _raw_key_b64(),
        })
        self.session_id = session["session_id"]
        self.service.post_message(
            _message_payload(self.session_id))


class RetryServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = DeliveryFixture()
        self.service = self.fixture.service
        self.sid = self.fixture.session_id

    def _retry(self, **overrides):
        payload = {"device_id": "d2", "attempt_id": "a1"}
        payload.update(overrides)
        return self.service.retry_message(self.sid, "m1", payload)

    def test_first_retry_is_201_pending_attempts_one(self) -> None:
        body, status = self._retry()
        self.assertEqual(status, 201)
        self.assertEqual(set(body),
                         {"session_id", "message_id", "status",
                          "attempts", "sequence"})
        self.assertEqual(body["session_id"], self.sid)
        self.assertEqual(body["message_id"], "m1")
        self.assertEqual(body["status"], "pending")
        self.assertEqual(body["attempts"], 1)
        self.assertEqual(body["sequence"], 1)

    def test_same_attempt_id_is_200_and_not_counted(self) -> None:
        self.assertEqual(self._retry()[1], 201)
        body, status = self._retry()
        self.assertEqual(status, 200)
        self.assertEqual(body["attempts"], 1)

    def test_new_attempt_id_is_200_and_increments(self) -> None:
        self._retry()
        body, status = self._retry(attempt_id="a2")
        self.assertEqual(status, 200)
        self.assertEqual(body["attempts"], 2)

    def test_retry_after_ack_is_200_and_stays_acked(self) -> None:
        self.service.ack_message(self.sid, {
            "device_id": "d2", "message_id": "m1", "sequence": 1})
        body, status = self._retry(attempt_id="a9")
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "acked")
        self.assertEqual(body["attempts"], 1)

    def test_body_must_be_object(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.retry_message(self.sid, "m1", ["nope"])
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.field, "request_body")

    def test_missing_or_empty_fields_are_400(self) -> None:
        for payload, field in (
            ({"attempt_id": "a"}, "device_id"),
            ({"device_id": "d2"}, "attempt_id"),
            ({"device_id": "", "attempt_id": "a"}, "device_id"),
            ({"device_id": "d2", "attempt_id": ""}, "attempt_id"),
            ({"device_id": 5, "attempt_id": "a"}, "device_id"),
            ({"device_id": "d2", "attempt_id": None}, "attempt_id"),
        ):
            with self.assertRaises(ServiceError) as ctx:
                self.service.retry_message(self.sid, "m1", payload)
            self.assertEqual(ctx.exception.status_code, 400)
            self.assertEqual(ctx.exception.field, field, payload)

    def test_unknown_session_is_404_session_id(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.retry_message("ghost", "m1",
                                       {"device_id": "d2", "attempt_id": "a"})
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "session_id")

    def test_unknown_message_is_404_message_id(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.retry_message(self.sid, "ghost",
                                       {"device_id": "d2", "attempt_id": "a"})
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "message_id")

    def test_wrong_or_unknown_or_revoked_device_is_409_device_id(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self._retry(device_id="d1")  # session initiator, not recipient
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "device_id")

        with self.assertRaises(ServiceError) as ctx:
            self._retry(device_id="ghost")
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "device_id")

        self.service.revoke_device("d2")
        with self.assertRaises(ServiceError) as ctx:
            self._retry()
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "device_id")

    def test_failed_retry_does_not_count(self) -> None:
        with self.assertRaises(ServiceError):
            self._retry(device_id="ghost")
        body, status = self._retry()
        self.assertEqual(status, 201)
        self.assertEqual(body["attempts"], 1)


class AckServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = DeliveryFixture()
        self.service = self.fixture.service
        self.sid = self.fixture.session_id

    def _ack(self, **overrides):
        payload = {"device_id": "d2", "message_id": "m1", "sequence": 1}
        payload.update(overrides)
        return self.service.ack_message(self.sid, payload)

    def test_first_ack_is_201_acked(self) -> None:
        body, status = self._ack()
        self.assertEqual(status, 201)
        self.assertEqual(body["status"], "acked")
        self.assertEqual(body["attempts"], 0)
        self.assertEqual(body["sequence"], 1)

    def test_repeated_ack_is_200_and_idempotent(self) -> None:
        self.assertEqual(self._ack()[1], 201)
        body, status = self._ack()
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "acked")

    def test_ack_without_prior_retry_is_allowed(self) -> None:
        body, status = self._ack()
        self.assertEqual(status, 201)
        self.assertEqual(body["attempts"], 0)

    def test_validation_errors(self) -> None:
        for payload, field in (
            ({"message_id": "m1", "sequence": 1}, "device_id"),
            ({"device_id": "d2", "sequence": 1}, "message_id"),
            ({"device_id": "d2", "message_id": "m1"}, "sequence"),
            ({"device_id": "d2", "message_id": "m1", "sequence": "1"},
             "sequence"),
            ({"device_id": "d2", "message_id": "m1", "sequence": True},
             "sequence"),
            ({"device_id": "d2", "message_id": "m1", "sequence": 1.5},
             "sequence"),
            ({"device_id": "", "message_id": "m1", "sequence": 1},
             "device_id"),
        ):
            with self.assertRaises(ServiceError) as ctx:
                self.service.ack_message(self.sid, payload)
            self.assertEqual(ctx.exception.status_code, 400, payload)
            self.assertEqual(ctx.exception.field, field, payload)

    def test_sequence_mismatch_is_409_sequence(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self._ack(sequence=2)
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "sequence")

    def test_ack_lookup_errors(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.ack_message("ghost",
                                     {"device_id": "d2", "message_id": "m1",
                                      "sequence": 1})
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "session_id")

        with self.assertRaises(ServiceError) as ctx:
            self._ack(message_id="ghost")
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "message_id")

        with self.assertRaises(ServiceError) as ctx:
            self._ack(device_id="d1")
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "device_id")

    def test_failed_ack_does_not_ack(self) -> None:
        with self.assertRaises(ServiceError):
            self._ack(sequence=9)
        body, status = self._ack()
        self.assertEqual(status, 201)
        self.assertEqual(body["status"], "acked")


class StatusServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = DeliveryFixture()
        self.service = self.fixture.service
        self.sid = self.fixture.session_id

    def test_status_without_activity_is_pending_zero_attempts(self) -> None:
        body = self.service.message_status(self.sid, "m1", "d2")
        self.assertEqual(body["status"], "pending")
        self.assertEqual(body["attempts"], 0)
        self.assertEqual(body["sequence"], 1)

    def test_status_reflects_retry_and_ack(self) -> None:
        self.service.retry_message(
            self.sid, "m1", {"device_id": "d2", "attempt_id": "a1"})
        body = self.service.message_status(self.sid, "m1", "d2")
        self.assertEqual(body["status"], "pending")
        self.assertEqual(body["attempts"], 1)

        self.service.ack_message(
            self.sid, {"device_id": "d2", "message_id": "m1", "sequence": 1})
        body = self.service.message_status(self.sid, "m1", "d2")
        self.assertEqual(body["status"], "acked")

    def test_status_errors(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.message_status("ghost", "m1", "d2")
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "session_id")

        with self.assertRaises(ServiceError) as ctx:
            self.service.message_status(self.sid, "ghost", "d2")
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "message_id")

        with self.assertRaises(ServiceError) as ctx:
            self.service.message_status(self.sid, "m1", "d1")
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "device_id")

        with self.assertRaises(ServiceError) as ctx:
            self.service.message_status(self.sid, "m1", "ghost")
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "device_id")


class HTTPDeliveryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.server, self.service = create_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        for device_id in ("d1", "d2"):
            self._request("POST", "/v1/devices",
                          _register_payload(device_id))
        _, session = self._request("POST", "/v1/sessions", {
            "initiator_device_id": "d1",
            "recipient_device_id": "d2",
            "prekey_id": "k1",
            "ephemeral_key": _raw_key_b64(),
        })
        self.sid = session["session_id"]
        self._request("POST", "/v1/messages",
                      _message_payload(self.sid))

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

    def test_retry_lifecycle(self) -> None:
        status, body = self._request(
            "POST", f"/v1/messages/{self.sid}/retry/m1",
            {"device_id": "d2", "attempt_id": "a1"})
        self.assertEqual(status, 201)
        self.assertEqual(body["status"], "pending")
        self.assertEqual(body["attempts"], 1)

        status, body = self._request(
            "POST", f"/v1/messages/{self.sid}/retry/m1",
            {"device_id": "d2", "attempt_id": "a1"})
        self.assertEqual(status, 200)
        self.assertEqual(body["attempts"], 1)

        status, body = self._request(
            "POST", f"/v1/messages/{self.sid}/retry/m1",
            {"device_id": "d2", "attempt_id": "a2"})
        self.assertEqual(status, 200)
        self.assertEqual(body["attempts"], 2)

    def test_ack_lifecycle_and_retry_after_ack(self) -> None:
        status, body = self._request(
            "POST", f"/v1/messages/{self.sid}/acks",
            {"device_id": "d2", "message_id": "m1", "sequence": 1})
        self.assertEqual(status, 201)
        self.assertEqual(body["status"], "acked")

        status, body = self._request(
            "POST", f"/v1/messages/{self.sid}/acks",
            {"device_id": "d2", "message_id": "m1", "sequence": 1})
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "acked")

        status, body = self._request(
            "POST", f"/v1/messages/{self.sid}/retry/m1",
            {"device_id": "d2", "attempt_id": "a1"})
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "acked")

    def test_status_endpoint(self) -> None:
        status, body = self._request(
            "GET", f"/v1/messages/{self.sid}/status/m1?device_id=d2")
        self.assertEqual(status, 200)
        self.assertEqual(set(body),
                         {"session_id", "message_id", "status",
                          "attempts", "sequence"})
        self.assertEqual(body["status"], "pending")

    def test_http_error_matrix(self) -> None:
        def post_retry(body):
            return self._request(
                "POST", f"/v1/messages/{self.sid}/retry/m1", body)

        status, body = post_retry({"device_id": "d2"})
        self.assertEqual((status, body["field"]), (400, "attempt_id"))
        status, body = post_retry({"attempt_id": "a"})
        self.assertEqual((status, body["field"]), (400, "device_id"))

        status, body = self._request(
            "POST", f"/v1/messages/ghost/retry/m1",
            {"device_id": "d2", "attempt_id": "a"})
        self.assertEqual((status, body["field"]), (404, "session_id"))

        status, body = self._request(
            "POST", f"/v1/messages/{self.sid}/retry/ghost",
            {"device_id": "d2", "attempt_id": "a"})
        self.assertEqual((status, body["field"]), (404, "message_id"))

        status, body = post_retry({"device_id": "d1", "attempt_id": "a"})
        self.assertEqual((status, body["field"]), (409, "device_id"))
        status, body = post_retry({"device_id": "ghost", "attempt_id": "a"})
        self.assertEqual((status, body["field"]), (409, "device_id"))

        status, body = self._request(
            "POST", f"/v1/messages/{self.sid}/acks",
            {"device_id": "d2", "message_id": "m1", "sequence": 5})
        self.assertEqual((status, body["field"]), (409, "sequence"))
        status, body = self._request(
            "POST", f"/v1/messages/{self.sid}/acks",
            {"device_id": "d2", "message_id": "m1"})
        self.assertEqual((status, body["field"]), (400, "sequence"))

    def test_status_parameter_and_lookup_errors(self) -> None:
        base = f"/v1/messages/{self.sid}/status/m1"
        for path, field in (
            (base, "device_id"),
            (base + "?device_id=", "device_id"),
            (base + "?device_id=d1&device_id=d2", "device_id"),
        ):
            status, body = self._request("GET", path)
            self.assertEqual((status, body["field"]), (400, field), path)

        status, body = self._request(
            "GET", f"/v1/messages/ghost/status/m1?device_id=d2")
        self.assertEqual((status, body["field"]), (404, "session_id"))
        status, body = self._request(
            "GET", f"/v1/messages/{self.sid}/status/ghost?device_id=d2")
        self.assertEqual((status, body["field"]), (404, "message_id"))
        status, body = self._request("GET", base + "?device_id=d1")
        self.assertEqual((status, body["field"]), (409, "device_id"))

    def test_revoked_recipient_is_409_on_all_three(self) -> None:
        self._request("POST", "/v1/devices/d2/revoke")
        status, body = self._request(
            "POST", f"/v1/messages/{self.sid}/retry/m1",
            {"device_id": "d2", "attempt_id": "a"})
        self.assertEqual((status, body["field"]), (409, "device_id"))
        status, body = self._request(
            "POST", f"/v1/messages/{self.sid}/acks",
            {"device_id": "d2", "message_id": "m1", "sequence": 1})
        self.assertEqual((status, body["field"]), (409, "device_id"))
        status, body = self._request(
            "GET", f"/v1/messages/{self.sid}/status/m1?device_id=d2")
        self.assertEqual((status, body["field"]), (409, "device_id"))


class PersistenceTest(unittest.TestCase):
    def _service_with_state(self) -> DeliveryFixture:
        return DeliveryFixture()

    def test_missing_file_is_created_version_one(self) -> None:
        directory = tempfile.mkdtemp()
        path = os.path.join(directory, "state.json")
        service = DeviceService()
        attach_persistence(service, path)
        self.assertTrue(os.path.exists(path))
        with open(path, encoding="utf-8") as handle:
            document = json.load(handle)
        self.assertEqual(document["version"], STATE_VERSION)

    def test_round_trip_preserves_delivery_dedup_and_ack(self) -> None:
        directory = tempfile.mkdtemp()
        path = os.path.join(directory, "state.json")
        fixture = self._service_with_state()
        attach_persistence(fixture.service, path)
        fixture.service.retry_message(
            fixture.session_id, "m1",
            {"device_id": "d2", "attempt_id": "a1"})
        fixture.service.retry_message(
            fixture.session_id, "m1",
            {"device_id": "d2", "attempt_id": "a2"})
        fixture.service.ack_message(
            fixture.session_id,
            {"device_id": "d2", "message_id": "m1", "sequence": 1})

        restored = DeviceService()
        attach_persistence(restored, path)
        # same attempt id is not re-counted after restart
        body, status = restored.retry_message(
            fixture.session_id, "m1",
            {"device_id": "d2", "attempt_id": "a1"})
        self.assertEqual(status, 200)
        self.assertEqual(body["attempts"], 2)
        self.assertEqual(body["status"], "acked")
        # ack stays acked (repeat ack is 200, not 201)
        _, ack_status = restored.ack_message(
            fixture.session_id,
            {"device_id": "d2", "message_id": "m1", "sequence": 1})
        self.assertEqual(ack_status, 200)

    def test_sequence_cursor_survives_restart(self) -> None:
        directory = tempfile.mkdtemp()
        path = os.path.join(directory, "state.json")
        fixture = self._service_with_state()
        attach_persistence(fixture.service, path)
        restored = DeviceService()
        attach_persistence(restored, path)
        body = restored.post_message(
            _message_payload(fixture.session_id, message_id="m2", sequence=2))
        self.assertEqual(body["sequence"], 2)
        # cursor really advanced: reusing sequence 2 is a sequence conflict
        with self.assertRaises(ServiceError) as ctx:
            restored.post_message(
                _message_payload(fixture.session_id, message_id="m3",
                                 sequence=2))
        self.assertEqual(ctx.exception.field, "sequence")
        # and the stream resumes at 3
        body = restored.post_message(
            _message_payload(fixture.session_id, message_id="m3", sequence=3))
        self.assertEqual(body["sequence"], 3)

    def test_revocation_survives_restart(self) -> None:
        directory = tempfile.mkdtemp()
        path = os.path.join(directory, "state.json")
        fixture = self._service_with_state()
        attach_persistence(fixture.service, path)
        fixture.service.revoke_device("d2")

        restored = DeviceService()
        attach_persistence(restored, path)
        with self.assertRaises(ServiceError) as ctx:
            restored.message_status(fixture.session_id, "m1", "d2")
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "device_id")
        # revoked prekey filter is also preserved
        self.assertEqual(restored.get_device("d2")["prekey_ids"], [])

    def test_every_change_is_persisted_atomically(self) -> None:
        directory = tempfile.mkdtemp()
        path = os.path.join(directory, "state.json")
        fixture = self._service_with_state()
        attach_persistence(fixture.service, path)
        fixture.service.retry_message(
            fixture.session_id, "m1",
            {"device_id": "d2", "attempt_id": "a1"})
        # the file on disk is always a complete, parseable document
        with open(path, encoding="utf-8") as handle:
            document = json.load(handle)
        self.assertEqual(document["delivery"][0]["attempts"], 1)
        # no temp files left behind
        self.assertEqual(
            [name for name in os.listdir(directory) if name.endswith(".tmp")],
            [])

    def test_corrupt_file_is_rejected(self) -> None:
        directory = tempfile.mkdtemp()
        path = os.path.join(directory, "state.json")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("{not json")
        with self.assertRaises(StateFileError):
            attach_persistence(DeviceService(), path)

    def test_wrong_version_is_rejected(self) -> None:
        directory = tempfile.mkdtemp()
        path = os.path.join(directory, "state.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"version": 99, "devices": []}, handle)
        with self.assertRaises(StateFileError):
            attach_persistence(DeviceService(), path)

    def test_non_object_document_is_rejected(self) -> None:
        directory = tempfile.mkdtemp()
        path = os.path.join(directory, "state.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump([1, 2, 3], handle)
        with self.assertRaises(StateFileError):
            attach_persistence(DeviceService(), path)

    def test_malformed_state_payload_is_rejected(self) -> None:
        directory = tempfile.mkdtemp()
        path = os.path.join(directory, "state.json")
        JsonStateStore(path).save({"devices": "not-a-list"})
        # A structurally malformed (but valid-JSON) file is a corrupt file,
        # not an in-memory programming error: startup refuses cleanly.
        with self.assertRaises(StateFileError):
            attach_persistence(DeviceService(), path)

    def test_used_nonce_set_survives_restart(self) -> None:
        directory = tempfile.mkdtemp()
        path = os.path.join(directory, "state.json")
        fixture = self._service_with_state()
        attach_persistence(fixture.service, path)
        nonce_a = base64.b64encode(b"AAAAAAAAAAAAAAAA").decode()
        fixture.service.post_message(_message_payload(
            fixture.session_id, message_id="m2", sequence=2,
        ) | {"nonce": nonce_a})
        # The set is part of the version-1 document.
        with open(path, encoding="utf-8") as handle:
            document = json.load(handle)
        self.assertEqual(document["used_nonces"][fixture.session_id],
                         sorted({_message_payload(fixture.session_id)["nonce"],
                                 nonce_a}))

        restored = DeviceService()
        attach_persistence(restored, path)
        # The historical nonce is still consumed: replaying it is 409/nonce.
        with self.assertRaises(ServiceError) as ctx:
            restored.post_message(_message_payload(
                fixture.session_id, message_id="m3", sequence=3,
            ) | {"nonce": nonce_a})
        self.assertEqual(ctx.exception.field, "nonce")
        # A fresh nonce at the resumed cursor is accepted.
        body = restored.post_message(_message_payload(
            fixture.session_id, message_id="m3", sequence=3))
        self.assertEqual(body["sequence"], 3)

    def test_legacy_file_without_nonce_section_rebuilds_from_messages(self) -> None:
        directory = tempfile.mkdtemp()
        path = os.path.join(directory, "state.json")
        fixture = self._service_with_state()
        attach_persistence(fixture.service, path)
        # Simulate a pre-replay-protection version-1 file: strip the section.
        with open(path, encoding="utf-8") as handle:
            document = json.load(handle)
        del document["used_nonces"]
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)

        restored = DeviceService()
        attach_persistence(restored, path)  # must not refuse to start
        # The set was rebuilt from stored message history, so the historical
        # nonce (the fixture m1 nonce) is already consumed.
        historical_nonce = _message_payload(fixture.session_id)["nonce"]
        with self.assertRaises(ServiceError) as ctx:
            restored.post_message(_message_payload(
                fixture.session_id, message_id="m2", sequence=2,
            ) | {"nonce": historical_nonce})
        self.assertEqual(ctx.exception.field, "nonce")

    def test_malformed_nonce_section_is_rejected(self) -> None:
        directory = tempfile.mkdtemp()
        path = os.path.join(directory, "state.json")
        fixture = self._service_with_state()
        attach_persistence(fixture.service, path)
        with open(path, encoding="utf-8") as handle:
            document = json.load(handle)
        for bad_section in (["not", "an", "object"],
                            {"sid": "not-a-list"},
                            {"sid": [1, 2, 3]}):
            document["used_nonces"] = bad_section
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(document, handle)
            with self.assertRaises(StateFileError):
                attach_persistence(DeviceService(), path)


class CLIDeliveryTest(unittest.TestCase):
    """Real-subprocess tests for retry-message/ack-message/message-status."""

    def setUp(self) -> None:
        self.server, _ = create_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        for device_id in ("d1", "d2"):
            self._run("register", "--user-id", "u1", "--device-id", device_id,
                      "--identity-key", _raw_key_b64(),
                      "--prekey", f"k1:{_raw_key_b64()}")
        result = self._run("create-session",
                           "--initiator-device-id", "d1",
                           "--recipient-device-id", "d2",
                           "--prekey-id", "k1",
                           "--ephemeral-key", _raw_key_b64())
        self.sid = json.loads(result.stdout)["session_id"]
        self._run("send-message", "--session-id", self.sid,
                  "--sender-device-id", "d1", "--message-id", "m1",
                  "--sequence", "1", "--nonce", "bm9uY2UxMjM0NTY3",
                  "--ciphertext", "Y2lw")

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _run(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "e2ee_backend", "--base-url",
             self.base_url, *arguments],
            capture_output=True, text=True, timeout=15)

    def test_retry_ack_status_success_on_stdout(self) -> None:
        result = self._run("retry-message", self.sid, "m1",
                           "--device-id", "d2", "--attempt-id", "a1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        self.assertEqual(result.stdout.count("\n"), 1)
        body = json.loads(result.stdout)
        self.assertEqual(body["attempts"], 1)
        self.assertEqual(body["status"], "pending")

        # duplicate attempt still exits 0 (200)
        result = self._run("retry-message", self.sid, "m1",
                           "--device-id", "d2", "--attempt-id", "a1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["attempts"], 1)

        result = self._run("ack-message", self.sid,
                           "--device-id", "d2", "--message-id", "m1",
                           "--sequence", "1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "acked")

        result = self._run("message-status", self.sid, "m1",
                           "--device-id", "d2")
        self.assertEqual(result.returncode, 0, result.stderr)
        body = json.loads(result.stdout)
        self.assertEqual(body["status"], "acked")
        self.assertEqual(body["attempts"], 1)

    def test_failure_prints_json_on_stderr_and_exits_nonzero(self) -> None:
        result = self._run("retry-message", self.sid, "ghost",
                           "--device-id", "d2", "--attempt-id", "a1")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertEqual(json.loads(result.stderr)["field"], "message_id")

        result = self._run("ack-message", self.sid,
                           "--device-id", "d2", "--message-id", "m1",
                           "--sequence", "9")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(json.loads(result.stderr)["field"], "sequence")

        result = self._run("message-status", self.sid, "m1",
                           "--device-id", "ghost")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(json.loads(result.stderr)["field"], "device_id")

    def test_connection_failure_is_field_server(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "e2ee_backend",
             "--base-url", "http://127.0.0.1:1",
             "message-status", self.sid, "m1", "--device-id", "d2"],
            capture_output=True, text=True, timeout=15)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(json.loads(result.stderr)["field"], "server")
        self.assertNotIn("Traceback", result.stderr)


class ServePersistenceTest(unittest.TestCase):
    """``serve --data-file`` restores delivery state across real restarts."""

    def setUp(self) -> None:
        directory = tempfile.mkdtemp()
        self.data_file = os.path.join(directory, "state.json")
        self.port = 0  # chosen below, fixed for both runs
        self.process = None

    def tearDown(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()

    def _serve(self, port: int, capture_stderr: bool = False
               ) -> subprocess.Popen:
        process = subprocess.Popen(
            [sys.executable, "-m", "e2ee_backend", "serve",
             "--host", "127.0.0.1", "--port", str(port),
             "--data-file", self.data_file],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE if capture_stderr else subprocess.DEVNULL,
            text=True)
        # Wait until the port answers (or the process exits on bad state).
        deadline = time.time() + 10
        while time.time() < deadline:
            if process.poll() is not None:
                return process
            try:
                connection = HTTPConnection("127.0.0.1", port, timeout=0.5)
                connection.request("GET", "/v1/devices/probe")
                connection.getresponse().read()
                connection.close()
                return process
            except OSError:
                time.sleep(0.1)
        process.kill()
        self.fail("server did not start in time")

    def _cli(self, port: int, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "e2ee_backend",
             "--base-url", f"http://127.0.0.1:{port}", *arguments],
            capture_output=True, text=True, timeout=15)

    def test_state_survives_restart(self) -> None:
        port = 18201
        self.process = self._serve(port)
        self.assertIsNone(self.process.poll())

        key = _raw_key_b64
        self._cli(port, "register", "--user-id", "u", "--device-id", "d1",
                  "--identity-key", key(),
                  "--prekey", f"k1:{key()}")
        self._cli(port, "register", "--user-id", "u", "--device-id", "d2",
                  "--identity-key", key(),
                  "--prekey", f"k1:{key()}")
        result = self._cli(port, "create-session",
                           "--initiator-device-id", "d1",
                           "--recipient-device-id", "d2",
                           "--prekey-id", "k1", "--ephemeral-key", key())
        sid = json.loads(result.stdout)["session_id"]
        self._cli(port, "send-message", "--session-id", sid,
                  "--sender-device-id", "d1", "--message-id", "m1",
                  "--sequence", "1", "--nonce", "bm9uY2UxMjM0NTY3",
                  "--ciphertext", "Y2lw")
        self._cli(port, "retry-message", sid, "m1",
                  "--device-id", "d2", "--attempt-id", "a1")
        self._cli(port, "ack-message", sid, "--device-id", "d2",
                  "--message-id", "m1", "--sequence", "1")

        self.process.terminate()
        self.process.wait(timeout=5)

        # Restart: dedup, acked status and sequence cursor all survive.
        self.process = self._serve(port)
        result = self._cli(port, "retry-message", sid, "m1",
                           "--device-id", "d2", "--attempt-id", "a1")
        self.assertEqual(result.returncode, 0, result.stderr)
        body = json.loads(result.stdout)
        self.assertEqual(body["attempts"], 1)
        self.assertEqual(body["status"], "acked")

        result = self._cli(port, "send-message", "--session-id", sid,
                           "--sender-device-id", "d1", "--message-id", "m2",
                           "--sequence", "2", "--nonce", "bm9uY2UtbTI=",
                           "--ciphertext", "Y2lw")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["sequence"], 2)

        # The m1 nonce remains consumed across the restart: a fresh envelope
        # replaying it is 409/nonce and does not advance the stream.
        replay = self._cli(port, "send-message", "--session-id", sid,
                           "--sender-device-id", "d1", "--message-id", "m3",
                           "--sequence", "3",
                           "--nonce", "bm9uY2UxMjM0NTY3",
                           "--ciphertext", "Y2lw")
        self.assertNotEqual(replay.returncode, 0)
        self.assertEqual(json.loads(replay.stderr)["field"], "nonce")
        result = self._cli(port, "pull-messages", sid, "--device-id", "d2")
        body = json.loads(result.stdout)
        self.assertEqual([m["message_id"] for m in body["messages"]],
                         ["m1", "m2"])

    def test_corrupt_file_makes_serve_refuse_to_start(self) -> None:
        with open(self.data_file, "w", encoding="utf-8") as handle:
            handle.write("{broken")
        port = 18202
        process = self._serve(port, capture_stderr=True)
        self.assertIsNotNone(process.poll())
        assert process.stderr is not None
        error = json.loads(process.stderr.read())
        process.stderr.close()
        self.assertEqual(error["field"], "data_file")
        self.process = None  # already exited; tearDown must not terminate it


if __name__ == "__main__":
    unittest.main()
