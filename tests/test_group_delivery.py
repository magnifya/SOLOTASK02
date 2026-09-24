"""Tests for per-device reliable delivery (retry/ack/status) in group sessions."""
from __future__ import annotations

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
from e2ee_backend.persistence import (
    PersistenceUnavailable,
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
        "nonce": base64.b64encode(f"nonce-{message_id}".encode()).decode(),
        "ciphertext": base64.b64encode(b"ciphertext-and-tag").decode(),
    }


class GroupDeliveryFixture:
    """Group g1 (creator d1, members d1/d2/d3), one group session, message m1
    sent by d1. d4 is a registered device outside the group."""

    def __init__(self) -> None:
        self.service = DeviceService()
        for device_id in ("d1", "d2", "d3", "d4"):
            self.service.register(_register_payload(device_id))
        self.service.create_group({
            "group_id": "g1", "creator_device_id": "d1",
            "member_device_ids": ["d1", "d2", "d3"]})
        session = self.service.create_group_session({
            "group_id": "g1", "initiator_device_id": "d1",
            "ephemeral_key": _raw_key_b64()})
        self.session_id = session["session_id"]
        self.service.post_message(_message_payload(self.session_id))


class GroupRetryServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = GroupDeliveryFixture()
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

    def test_attempt_ids_are_deduped_per_device_not_per_message(self) -> None:
        self.assertEqual(self._retry()[1], 201)
        # The same attempt_id under another device is that device's first.
        body, status = self._retry(device_id="d3")
        self.assertEqual(status, 201)
        self.assertEqual(body["attempts"], 1)
        # And the first device's counter is untouched.
        status_body = self.service.message_status(self.sid, "m1", "d2")
        self.assertEqual(status_body["attempts"], 1)

    def test_retry_after_ack_is_200_and_stays_acked(self) -> None:
        self.service.ack_message(self.sid, {
            "device_id": "d2", "message_id": "m1", "sequence": 1})
        body, status = self._retry(attempt_id="a9")
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "acked")
        self.assertEqual(body["attempts"], 1)

    def test_unknown_session_or_message_is_404(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.retry_message("ghost", "m1",
                                       {"device_id": "d2", "attempt_id": "a"})
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "session_id")

        with self.assertRaises(ServiceError) as ctx:
            self.service.retry_message(self.sid, "ghost",
                                       {"device_id": "d2", "attempt_id": "a"})
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "message_id")

    def test_device_errors_are_409_device_id(self) -> None:
        for device_id in ("d1",      # the sender
                          "d4",      # not a frozen member
                          "ghost"):  # unknown device
            with self.assertRaises(ServiceError) as ctx:
                self._retry(device_id=device_id)
            self.assertEqual(ctx.exception.status_code, 409, device_id)
            self.assertEqual(ctx.exception.field, "device_id", device_id)

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


class GroupAckServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = GroupDeliveryFixture()
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

    def test_acks_are_independent_per_device(self) -> None:
        self.assertEqual(self._ack()[1], 201)
        # Another frozen member's first ack is still 201...
        body, status = self._ack(device_id="d3")
        self.assertEqual(status, 201)
        # ...and its own repeat is 200 while d2 stays acked.
        self.assertEqual(self._ack(device_id="d3")[1], 200)
        status_body = self.service.message_status(self.sid, "m1", "d2")
        self.assertEqual(status_body["status"], "acked")

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

        for device_id in ("d1", "d4", "ghost"):
            with self.assertRaises(ServiceError) as ctx:
                self._ack(device_id=device_id)
            self.assertEqual(ctx.exception.status_code, 409, device_id)
            self.assertEqual(ctx.exception.field, "device_id", device_id)

    def test_failed_ack_does_not_ack(self) -> None:
        with self.assertRaises(ServiceError):
            self._ack(sequence=9)
        body, status = self._ack()
        self.assertEqual(status, 201)
        self.assertEqual(body["status"], "acked")


class GroupStatusServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = GroupDeliveryFixture()
        self.service = self.fixture.service
        self.sid = self.fixture.session_id

    def test_status_without_activity_is_pending_zero_attempts(self) -> None:
        body = self.service.message_status(self.sid, "m1", "d2")
        self.assertEqual(body["status"], "pending")
        self.assertEqual(body["attempts"], 0)
        self.assertEqual(body["sequence"], 1)

    def test_status_reflects_only_that_device(self) -> None:
        self.service.retry_message(
            self.sid, "m1", {"device_id": "d2", "attempt_id": "a1"})
        self.service.ack_message(
            self.sid, {"device_id": "d3", "message_id": "m1", "sequence": 1})
        d2 = self.service.message_status(self.sid, "m1", "d2")
        self.assertEqual((d2["status"], d2["attempts"]), ("pending", 1))
        d3 = self.service.message_status(self.sid, "m1", "d3")
        self.assertEqual((d3["status"], d3["attempts"]), ("acked", 0))

    def test_status_errors(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.message_status("ghost", "m1", "d2")
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "session_id")

        with self.assertRaises(ServiceError) as ctx:
            self.service.message_status(self.sid, "ghost", "d2")
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "message_id")

        for device_id in ("d1", "d4", "ghost"):
            with self.assertRaises(ServiceError) as ctx:
                self.service.message_status(self.sid, "m1", device_id)
            self.assertEqual(ctx.exception.status_code, 409, device_id)
            self.assertEqual(ctx.exception.field, "device_id", device_id)


class FrozenMembershipTest(unittest.TestCase):
    """Later group roster changes never alter the frozen delivery scope."""

    def setUp(self) -> None:
        self.fixture = GroupDeliveryFixture()
        self.service = self.fixture.service
        self.sid = self.fixture.session_id

    def test_removed_member_keeps_delivery_rights(self) -> None:
        self.service.remove_group_member(
            "g1", {"actor_device_id": "d1", "device_id": "d3"})
        body, status = self.service.retry_message(
            self.sid, "m1", {"device_id": "d3", "attempt_id": "a1"})
        self.assertEqual(status, 201)
        body, status = self.service.ack_message(
            self.sid, {"device_id": "d3", "message_id": "m1", "sequence": 1})
        self.assertEqual(status, 201)
        self.assertEqual(body["status"], "acked")

    def test_added_member_gains_no_delivery_rights(self) -> None:
        self.service.add_group_member(
            "g1", {"actor_device_id": "d1", "device_id": "d4"})
        with self.assertRaises(ServiceError) as ctx:
            self.service.retry_message(
                self.sid, "m1", {"device_id": "d4", "attempt_id": "a1"})
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "device_id")
        with self.assertRaises(ServiceError) as ctx:
            self.service.message_status(self.sid, "m1", "d4")
        self.assertEqual(ctx.exception.status_code, 409)


class GroupDeliveryPersistenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.path = os.path.join(self.directory, "state.json")
        self.fixture = GroupDeliveryFixture()
        self.sid = self.fixture.session_id

    def _document(self) -> dict:
        with open(self.path, encoding="utf-8") as handle:
            return json.load(handle)

    def test_round_trip_preserves_per_device_dedup_and_ack(self) -> None:
        service = self.fixture.service
        attach_persistence(service, self.path)
        service.retry_message(self.sid, "m1",
                              {"device_id": "d2", "attempt_id": "a1"})
        service.retry_message(self.sid, "m1",
                              {"device_id": "d2", "attempt_id": "a2"})
        service.ack_message(self.sid, {
            "device_id": "d2", "message_id": "m1", "sequence": 1})
        service.retry_message(self.sid, "m1",
                              {"device_id": "d3", "attempt_id": "a1"})

        document = self._document()
        self.assertEqual(document["version"], 1)
        self.assertEqual(len(document["group_delivery"]), 2)

        restored = DeviceService()
        attach_persistence(restored, self.path)
        # d2: same attempt id is not re-counted; ack survived the restart.
        body, status = restored.retry_message(
            self.sid, "m1", {"device_id": "d2", "attempt_id": "a1"})
        self.assertEqual(status, 200)
        self.assertEqual(body["attempts"], 2)
        self.assertEqual(body["status"], "acked")
        _, ack_status = restored.ack_message(
            self.sid, {"device_id": "d2", "message_id": "m1", "sequence": 1})
        self.assertEqual(ack_status, 200)
        # d3: pending with its own single attempt.
        body = restored.message_status(self.sid, "m1", "d3")
        self.assertEqual((body["status"], body["attempts"]), ("pending", 1))

    def test_missing_group_delivery_section_loads_as_empty(self) -> None:
        service = self.fixture.service
        attach_persistence(service, self.path)
        service.retry_message(self.sid, "m1",
                              {"device_id": "d2", "attempt_id": "a1"})
        document = self._document()
        del document["group_delivery"]
        # Emulate a pre-marker legacy file: drop the marker and sidecar too.
        del document["integrity_log_version"]
        os.unlink(self.path + ".integrity")
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)

        restored = DeviceService()
        attach_persistence(restored, self.path)  # must not refuse to start
        body = restored.message_status(self.sid, "m1", "d2")
        self.assertEqual((body["status"], body["attempts"]), ("pending", 0))

    def test_contradictory_section_refuses_startup_without_overwrite(self) -> None:
        service = self.fixture.service
        attach_persistence(service, self.path)
        service.retry_message(self.sid, "m1",
                              {"device_id": "d2", "attempt_id": "a1"})
        document = self._document()
        record = document["group_delivery"][0]

        def corrupt(mutate) -> None:
            broken = json.loads(json.dumps(document))
            mutate(broken)
            with open(self.path, "w", encoding="utf-8") as handle:
                json.dump(broken, handle)
            before = open(self.path, "rb").read()
            with self.assertRaises(StateFileError):
                attach_persistence(DeviceService(), self.path)
            # The contradictory file is left exactly as it was.
            self.assertEqual(open(self.path, "rb").read(), before)

        corrupt(lambda d: d["group_delivery"].append(
            dict(record, session_id="ghost")))
        corrupt(lambda d: d["group_delivery"].append(
            dict(record, message_id="ghost")))
        corrupt(lambda d: d["group_delivery"].append(
            dict(record, device_id="d4")))  # not a frozen member
        corrupt(lambda d: d["group_delivery"].append(
            dict(record, device_id="d3", attempts=5)))  # != len(attempt_ids)
        corrupt(lambda d: d["group_delivery"].append(
            dict(record, device_id="d3", acked=True, ack_sequence=0)))
        corrupt(lambda d: d["group_delivery"].append(record))  # duplicate
        corrupt(lambda d: d.__setitem__("group_delivery", "not-a-list"))

    def test_failed_persist_rolls_back_group_delivery(self) -> None:
        from e2ee_backend.persistence import JsonStateStore

        service = self.fixture.service
        attach_persistence(service, self.path)
        real_save = JsonStateStore.save

        def failing_save(self, state):
            raise OSError("disk full")

        JsonStateStore.save = failing_save
        try:
            with self.assertRaises(PersistenceUnavailable):
                service.retry_message(
                    self.sid, "m1", {"device_id": "d2", "attempt_id": "a1"})
        finally:
            JsonStateStore.save = real_save

        # The failed mutation is visible nowhere: status is back to empty
        # and the first durable retry is still a 201 with attempts=1.
        body = service.message_status(self.sid, "m1", "d2")
        self.assertEqual((body["status"], body["attempts"]), ("pending", 0))
        body, status = service.retry_message(
            self.sid, "m1", {"device_id": "d2", "attempt_id": "a1"})
        self.assertEqual(status, 201)
        self.assertEqual(body["attempts"], 1)


class GroupDeliveryCLITest(unittest.TestCase):
    """The three delivery CLI commands accept a group session_id."""

    def setUp(self) -> None:
        self.server, _ = create_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        for device_id in ("d1", "d2", "d3"):
            self._request("POST", "/v1/devices",
                          _register_payload(device_id))
        self._request("POST", "/v1/groups", {
            "group_id": "g1", "creator_device_id": "d1",
            "member_device_ids": ["d1", "d2", "d3"]})
        _, session = self._request("POST", "/v1/group-sessions", {
            "group_id": "g1", "initiator_device_id": "d1",
            "ephemeral_key": _raw_key_b64()})
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

    def _run(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "e2ee_backend", "--base-url",
             self.base_url, *arguments],
            capture_output=True, text=True, timeout=15)

    def test_retry_ack_status_lifecycle_on_group_session(self) -> None:
        result = self._run("retry-message", self.sid, "m1",
                           "--device-id", "d2", "--attempt-id", "a1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        body = json.loads(result.stdout)
        self.assertEqual((body["status"], body["attempts"]), ("pending", 1))

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
        self.assertEqual((body["status"], body["attempts"]), ("acked", 1))

    def test_failures_print_json_on_stderr_and_exit_nonzero(self) -> None:
        result = self._run("retry-message", self.sid, "m1",
                           "--device-id", "d1", "--attempt-id", "a1")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertEqual(json.loads(result.stderr)["field"], "device_id")

        result = self._run("ack-message", self.sid,
                           "--device-id", "d2", "--message-id", "m1",
                           "--sequence", "9")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(json.loads(result.stderr)["field"], "sequence")

        result = self._run("message-status", self.sid, "m1",
                           "--device-id", "d4")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(json.loads(result.stderr)["field"], "device_id")


if __name__ == "__main__":
    unittest.main()
