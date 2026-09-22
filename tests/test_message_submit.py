"""Tests for idempotent message submission (POST /v1/messages/submit).

Covers the service contract (201 first commit, 200 byte-identical replay —
even after the sender is revoked — 409/request_id on any changed field or
cross-session reuse, 400 validation, failures never consuming the id),
concurrency (one write per request_id), durable recovery of the
``message_submissions`` section (round-trip plus malformed-section startup
refusal), the strengthened ``group_delivery`` recovery checks, the HTTP
route over a real socket, and the ``submit-message`` CLI subcommand.
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
from cryptography.hazmat.primitives.asymmetric import x25519

from e2ee_backend.http_app import create_server
from e2ee_backend.persistence import StateFileError, attach_persistence
from e2ee_backend.service import DeviceService, ServiceError
from e2ee_backend.storage import DeviceStore


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


def _submit_payload(session_id: str, request_id: str = "req-1",
                    message_id: str = "m1", sequence: int = 1,
                    sender: str = "d1") -> dict:
    return {
        "request_id": request_id,
        "session_id": session_id,
        "sender_device_id": sender,
        "message_id": message_id,
        "sequence": sequence,
        "nonce": base64.b64encode(f"nonce-{message_id}".encode()).decode(),
        "ciphertext": base64.b64encode(b"ciphertext-and-tag").decode(),
    }


class ServiceSubmitFixture:
    """A service with two devices and one 1:1 session between them."""

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


class ServiceSubmitMessageTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = ServiceSubmitFixture()
        self.service = self.fixture.service
        self.session_id = self.fixture.session_id

    def _submit(self, **overrides):
        payload = _submit_payload(
            overrides.pop("session_id", self.session_id),
            request_id=overrides.pop("request_id", "req-1"),
            message_id=overrides.pop("message_id", "m1"))
        payload.update(overrides)
        return self.service.submit_message(payload)

    def test_first_submit_is_201_with_eight_fields(self) -> None:
        body, status = self._submit()
        self.assertEqual(status, 201)
        self.assertEqual(set(body), {"request_id", "session_id",
                                     "sender_device_id", "message_id",
                                     "sequence", "nonce", "ciphertext",
                                     "created_at"})
        self.assertEqual(body["request_id"], "req-1")
        self.assertEqual(body["sequence"], 1)
        self.assertTrue(body["created_at"].endswith("+00:00"))

    def test_identical_replay_is_200_with_original_body(self) -> None:
        first, _ = self._submit()
        replay, status = self._submit()
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)

    def test_replay_after_sender_revoked_still_returns_original(self) -> None:
        first, _ = self._submit()
        self.service.revoke_device("d1")
        replay, status = self._submit()
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)

    def test_changed_field_is_409_request_id(self) -> None:
        self._submit()
        for override in ({"ciphertext": "other"},
                         {"nonce": "other-nonce"},
                         {"message_id": "m2"},
                         {"sequence": 2},
                         {"sender_device_id": "d2"}):
            with self.assertRaises(ServiceError) as ctx:
                self._submit(**override)
            self.assertEqual(ctx.exception.status_code, 409, override)
            self.assertEqual(ctx.exception.field, "request_id", override)

    def test_request_id_reused_across_sessions_is_409(self) -> None:
        self._submit()
        other = self.service.create_session({
            "initiator_device_id": "d2",
            "recipient_device_id": "d1",
            "prekey_id": "k1",
            "ephemeral_key": _raw_key_b64(),
        })
        with self.assertRaises(ServiceError) as ctx:
            self._submit(session_id=other["session_id"])
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "request_id")

    def test_failed_submit_does_not_consume_request_id(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self._submit(sequence=5)
        self.assertEqual(ctx.exception.field, "sequence")
        body, status = self._submit()
        self.assertEqual(status, 201)
        self.assertEqual(body["sequence"], 1)

    def test_missing_or_empty_request_id_is_400(self) -> None:
        payload = _submit_payload(self.session_id)
        del payload["request_id"]
        with self.assertRaises(ServiceError) as ctx:
            self.service.submit_message(payload)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.field, "request_id")
        for bad in ("", 7, None, True):
            payload = _submit_payload(self.session_id)
            payload["request_id"] = bad
            with self.assertRaises(ServiceError) as ctx:
                self.service.submit_message(payload)
            self.assertEqual(ctx.exception.status_code, 400, bad)
            self.assertEqual(ctx.exception.field, "request_id", bad)

    def test_envelope_validation_matches_post_message(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self._submit(session_id="ghost")
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "session_id")

        self.service.revoke_device("d1")
        with self.assertRaises(ServiceError) as ctx:
            self._submit(request_id="req-2")
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "sender_device_id")

    def test_duplicate_message_id_and_nonce_still_conflict(self) -> None:
        first, _ = self._submit()
        with self.assertRaises(ServiceError) as ctx:
            self._submit(request_id="req-2", sequence=2)
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "message_id")
        with self.assertRaises(ServiceError) as ctx:
            self._submit(request_id="req-3", message_id="m2", sequence=2,
                         nonce=first["nonce"])
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "nonce")

    def test_message_is_stored_once_and_pullable(self) -> None:
        self._submit()
        self._submit()
        page = self.service.list_messages(self.session_id, "d2", 0, 100)
        self.assertEqual(len(page["messages"]), 1)
        self.assertEqual(page["messages"][0]["message_id"], "m1")

    def test_concurrent_same_request_id_writes_one_message(self) -> None:
        results = []
        errors = []

        def worker() -> None:
            try:
                results.append(self._submit())
            except ServiceError as error:  # pragma: no cover - unexpected
                errors.append(error)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(errors, [])
        self.assertEqual(len(results), 8)
        self.assertEqual(sum(1 for _, status in results if status == 201), 1)
        bodies = {json.dumps(body, sort_keys=True) for body, _ in results}
        self.assertEqual(len(bodies), 1)
        page = self.service.list_messages(self.session_id, "d2", 0, 100)
        self.assertEqual(len(page["messages"]), 1)

    def test_group_session_submit_follows_frozen_members(self) -> None:
        self.service.register(_register_payload("d3"))
        self.service.create_group({
            "group_id": "g1", "creator_device_id": "d1",
            "member_device_ids": ["d2"]})
        session = self.service.create_group_session({
            "group_id": "g1", "initiator_device_id": "d1",
            "ephemeral_key": _raw_key_b64()})
        body, status = self._submit(session_id=session["session_id"])
        self.assertEqual(status, 201)
        self.assertEqual(body["session_id"], session["session_id"])
        # A device outside the frozen member set cannot submit.
        with self.assertRaises(ServiceError) as ctx:
            self._submit(session_id=session["session_id"],
                         request_id="req-2", message_id="m2", sequence=2,
                         sender_device_id="d3")
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "sender_device_id")


class SubmitPersistenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.path = os.path.join(self.directory, "state.json")

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def _persisted_service(self) -> DeviceService:
        service = DeviceService()
        attach_persistence(service, self.path)
        return service

    def _reset_file(self) -> None:
        if os.path.exists(self.path):
            os.unlink(self.path)

    def _fixture(self, service: DeviceService) -> str:
        service.register(_register_payload("d1"))
        service.register(_register_payload("d2"))
        session = service.create_session({
            "initiator_device_id": "d1",
            "recipient_device_id": "d2",
            "prekey_id": "k1",
            "ephemeral_key": _raw_key_b64(),
        })
        return session["session_id"]

    def _document(self) -> dict:
        with open(self.path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _restore(self, document: dict) -> None:
        payload = {key: value for key, value in document.items()
                   if key != "version"}
        DeviceStore().restore_state(payload)

    def test_replay_survives_restart(self) -> None:
        service = self._persisted_service()
        session_id = self._fixture(service)
        first, status = service.submit_message(
            _submit_payload(session_id))
        self.assertEqual(status, 201)

        restarted = self._persisted_service()
        replay, status = restarted.submit_message(
            _submit_payload(session_id))
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)
        # The id stays globally unique across the restart.
        with self.assertRaises(ServiceError) as ctx:
            restarted.submit_message(
                _submit_payload(session_id, message_id="m2", sequence=2))
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "request_id")

    def test_old_file_without_section_loads_empty(self) -> None:
        service = self._persisted_service()
        self._fixture(service)
        document = self._document()
        del document["message_submissions"]
        self._restore(document)  # must not raise

    def _submitted_document(self) -> dict:
        self._reset_file()
        service = self._persisted_service()
        session_id = self._fixture(service)
        service.submit_message(_submit_payload(session_id))
        return self._document()

    def test_section_must_be_a_list(self) -> None:
        document = self._submitted_document()
        document["message_submissions"] = {"request_id": "req-1"}
        with self.assertRaises(ValueError):
            self._restore(document)

    def test_duplicate_request_id_is_rejected(self) -> None:
        document = self._submitted_document()
        record = document["message_submissions"][0]
        document["message_submissions"].append(dict(record))
        with self.assertRaises(ValueError):
            self._restore(document)

    def test_dangling_session_reference_is_rejected(self) -> None:
        document = self._submitted_document()
        document["message_submissions"][0]["session_id"] = "ghost"
        with self.assertRaises(ValueError):
            self._restore(document)

    def test_dangling_message_reference_is_rejected(self) -> None:
        document = self._submitted_document()
        document["message_submissions"][0]["message_id"] = "ghost"
        with self.assertRaises(ValueError):
            self._restore(document)

    def test_envelope_mismatch_is_rejected(self) -> None:
        for field, value in (("sender_device_id", "d2"),
                             ("sequence", 7),
                             ("nonce", "other"),
                             ("ciphertext", "other"),
                             ("created_at", "other")):
            document = self._submitted_document()
            document["message_submissions"][0][field] = value
            with self.assertRaises(ValueError, msg=field):
                self._restore(document)

    def test_missing_field_is_rejected(self) -> None:
        document = self._submitted_document()
        del document["message_submissions"][0]["request_id"]
        with self.assertRaises(ValueError):
            self._restore(document)

    def test_malformed_section_refuses_startup_without_overwriting(self) -> None:
        document = self._submitted_document()
        document["message_submissions"][0]["nonce"] = "tampered"
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        with open(self.path, "rb") as handle:
            original = handle.read()
        with self.assertRaises(StateFileError):
            self._persisted_service()
        with open(self.path, "rb") as handle:
            self.assertEqual(handle.read(), original)


class GroupDeliveryRecoveryTest(unittest.TestCase):
    """Strengthened group_delivery recovery checks."""

    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.path = os.path.join(self.directory, "state.json")

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def _document_with_delivery(self) -> dict:
        service = DeviceService()
        attach_persistence(service, self.path)
        for device_id in ("creator", "alice", "bob"):
            service.register(_register_payload(device_id))
        service.create_group({
            "group_id": "g1", "creator_device_id": "creator",
            "member_device_ids": ["alice", "bob"]})
        session = service.create_group_session({
            "group_id": "g1", "initiator_device_id": "creator",
            "ephemeral_key": _raw_key_b64()})
        self.session_id = session["session_id"]
        service.post_message({
            "session_id": self.session_id, "sender_device_id": "creator",
            "message_id": "m1", "sequence": 1, "nonce": "n1",
            "ciphertext": "ct"})
        service.retry_message(self.session_id, "m1", {
            "device_id": "alice", "attempt_id": "a1"})
        with open(self.path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _restore(self, document: dict) -> None:
        payload = {key: value for key, value in document.items()
                   if key != "version"}
        DeviceStore().restore_state(payload)

    def test_unregistered_device_is_rejected(self) -> None:
        document = self._document_with_delivery()
        document["group_delivery"][0]["device_id"] = "ghost"
        with self.assertRaises(ValueError):
            self._restore(document)

    def test_sender_device_is_rejected(self) -> None:
        document = self._document_with_delivery()
        document["group_delivery"][0]["device_id"] = "creator"
        with self.assertRaises(ValueError):
            self._restore(document)

    def test_revoked_device_record_is_valid_history(self) -> None:
        document = self._document_with_delivery()
        for device in document["devices"]:
            if device["device_id"] == "alice":
                device["revoked"] = True
        # The revocation is simulated as predating the key-audit feature: a
        # legacy version-1 file carries no key_events section, so no chain
        # has to account for the revoked flag.
        document.pop("key_events", None)
        store = DeviceStore()
        payload = {key: value for key, value in document.items()
                   if key != "version"}
        store.restore_state(payload)  # must not raise
        # Writes for the revoked device still conflict after recovery.
        service = DeviceService(store)
        with self.assertRaises(ServiceError) as ctx:
            service.retry_message(self.session_id, "m1", {
                "device_id": "alice", "attempt_id": "a2"})
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "device_id")


class HTTPSubmitMessageTest(unittest.TestCase):
    def setUp(self) -> None:
        self.server, _ = create_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        for device_id in ("d1", "d2"):
            status, _ = self._request("POST", "/v1/devices",
                                      _register_payload(device_id))
            self.assertEqual(status, 201)
        status, session = self._request("POST", "/v1/sessions", {
            "initiator_device_id": "d1",
            "recipient_device_id": "d2",
            "prekey_id": "k1",
            "ephemeral_key": _raw_key_b64(),
        })
        self.assertEqual(status, 201)
        self.session_id = session["session_id"]

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _request(self, method: str, path: str, body: object = None):
        connection = HTTPConnection("127.0.0.1", self.port, timeout=5)
        payload = json.dumps(body) if body is not None else None
        headers = {"Content-Type": "application/json"} if payload is not None else {}
        connection.request(method, path, body=payload, headers=headers)
        response = connection.getresponse()
        data = response.read().decode("utf-8")
        connection.close()
        return response.status, json.loads(data)

    def test_submit_replay_and_conflict(self) -> None:
        payload = _submit_payload(self.session_id)
        status, first = self._request("POST", "/v1/messages/submit", payload)
        self.assertEqual(status, 201)
        self.assertEqual(first["request_id"], "req-1")

        status, replay = self._request("POST", "/v1/messages/submit", payload)
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)

        changed = dict(payload, ciphertext="other")
        status, body = self._request("POST", "/v1/messages/submit", changed)
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "request_id")

    def test_missing_request_id_is_400(self) -> None:
        payload = _submit_payload(self.session_id)
        del payload["request_id"]
        status, body = self._request("POST", "/v1/messages/submit", payload)
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "request_id")

    def test_plain_post_message_route_unchanged(self) -> None:
        payload = _submit_payload(self.session_id)
        del payload["request_id"]
        status, body = self._request("POST", "/v1/messages", payload)
        self.assertEqual(status, 201)
        self.assertNotIn("request_id", body)


class CLISubmitMessageTest(unittest.TestCase):
    def setUp(self) -> None:
        self.server, _ = create_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        for device_id in ("d1", "d2"):
            self.assertEqual(self._run(
                "register", "--user-id", "u1", "--device-id", device_id,
                "--identity-key", _raw_key_b64(),
                "--prekey", f"k1:{_raw_key_b64()}").returncode, 0)
        result = self._run(
            "create-session", "--initiator-device-id", "d1",
            "--recipient-device-id", "d2", "--prekey-id", "k1",
            "--ephemeral-key", _raw_key_b64())
        self.assertEqual(result.returncode, 0, result.stderr)
        self.session_id = json.loads(result.stdout)["session_id"]

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _run(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "e2ee_backend", "--base-url", self.base_url,
             *arguments],
            capture_output=True, text=True, timeout=15)

    def _submit(self, request_id: str = "req-1", message_id: str = "m1",
                sequence: str = "1", ciphertext: str = "Y3Q="):
        return self._run(
            "submit-message", "--request-id", request_id,
            "--session-id", self.session_id, "--sender-device-id", "d1",
            "--message-id", message_id, "--sequence", sequence,
            "--nonce", "bjE=", "--ciphertext", ciphertext)

    def test_submit_then_replay_prints_same_json_on_stdout(self) -> None:
        result = self._submit()
        self.assertEqual(result.returncode, 0, result.stderr)
        line = result.stdout.strip()
        self.assertEqual(line.count("\n"), 0)
        body = json.loads(line)
        self.assertEqual(set(body), {"request_id", "session_id",
                                     "sender_device_id", "message_id",
                                     "sequence", "nonce", "ciphertext",
                                     "created_at"})
        replay = self._submit()
        self.assertEqual(replay.returncode, 0, replay.stderr)
        self.assertEqual(replay.stdout, result.stdout)

    def test_conflict_prints_error_on_stderr_nonzero(self) -> None:
        self.assertEqual(self._submit().returncode, 0)
        result = self._submit(ciphertext="b3RoZXI=")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        self.assertEqual(json.loads(result.stderr.strip())["field"],
                         "request_id")


if __name__ == "__main__":
    unittest.main()
