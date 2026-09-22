"""Tests for the idempotent message submit (POST /v1/messages/submit)."""
from __future__ import annotations

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


def _submit_payload(session_id: str, request_id: str = "r1",
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


class SubmitServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = ServiceSubmitFixture()
        self.service = self.fixture.service
        self.session_id = self.fixture.session_id

    def _submit(self, **overrides):
        payload = _submit_payload(
            overrides.pop("session_id", self.session_id),
            request_id=overrides.pop("request_id", "r1"),
            message_id=overrides.pop("message_id", "m1"),
            sequence=overrides.pop("sequence", 1))
        payload.update(overrides)
        return self.service.submit_message(payload)

    def test_first_submit_is_201_with_request_id_and_seven_fields(self):
        body, status = self._submit()
        self.assertEqual(status, 201)
        self.assertEqual(set(body), {"request_id", "session_id",
                                     "sender_device_id", "message_id",
                                     "sequence", "nonce", "ciphertext",
                                     "created_at"})
        self.assertEqual(body["request_id"], "r1")
        self.assertEqual(body["sequence"], 1)
        self.assertTrue(body["created_at"].endswith("+00:00"))

    def test_exact_replay_is_200_with_the_first_response(self):
        first, _ = self._submit()
        replay, status = self._submit()
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)
        # The replay did not append a second message.
        messages, _ = self.service.store.message_page(
            self.session_id, "d2", 0, 100)
        self.assertEqual(len(messages), 1)

    def test_replay_succeeds_even_after_sender_revoked(self):
        first, _ = self._submit()
        self.service.revoke_device("d1")
        replay, status = self._submit()
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)

    def test_changed_fields_conflict_with_field_request_id(self):
        self._submit()
        for override in ({"ciphertext": "eW90aGVy"},
                         {"nonce": base64.b64encode(b"other").decode()},
                         {"message_id": "m9"},
                         {"sequence": 9},
                         {"sender_device_id": "d2"},
                         {"session_id": "ghost"}):
            with self.assertRaises(ServiceError) as caught:
                self._submit(**override)
            self.assertEqual(caught.exception.status_code, 409, override)
            self.assertEqual(caught.exception.field, "request_id", override)

    def test_same_id_in_another_session_conflicts(self):
        self._submit()
        other = self.service.create_session({
            "initiator_device_id": "d1",
            "recipient_device_id": "d2",
            "prekey_id": "k1",
            "ephemeral_key": _raw_key_b64(),
        })
        with self.assertRaises(ServiceError) as caught:
            self._submit(session_id=other["session_id"])
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "request_id")

    def test_failed_submit_does_not_consume_the_request_id(self):
        # Bad sequence (stream is empty, 9 does not continue it): 409 and the
        # request_id stays free.
        with self.assertRaises(ServiceError) as caught:
            self._submit(sequence=9)
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "sequence")
        body, status = self._submit()
        self.assertEqual(status, 201)
        self.assertEqual(body["sequence"], 1)

    def test_request_id_validation(self):
        for payload in (
                _submit_payload(self.session_id),
                {**_submit_payload(self.session_id), "request_id": ""},
                {**_submit_payload(self.session_id), "request_id": 7}):
            if "request_id" in payload and payload["request_id"] == "r1":
                del payload["request_id"]
            with self.assertRaises(ServiceError) as caught:
                self.service.submit_message(payload)
            self.assertEqual(caught.exception.status_code, 400, payload)
            self.assertEqual(caught.exception.field, "request_id", payload)

    def test_envelope_validation_matches_plain_post(self):
        payload = _submit_payload(self.session_id)
        del payload["nonce"]
        with self.assertRaises(ServiceError) as caught:
            self.service.submit_message(payload)
        self.assertEqual(caught.exception.status_code, 400)
        self.assertEqual(caught.exception.field, "nonce")

        payload = _submit_payload(self.session_id)
        payload["sequence"] = "1"
        with self.assertRaises(ServiceError) as caught:
            self.service.submit_message(payload)
        self.assertEqual(caught.exception.status_code, 400)
        self.assertEqual(caught.exception.field, "sequence")

        with self.assertRaises(ServiceError) as caught:
            self.service.submit_message("not-an-object")
        self.assertEqual(caught.exception.status_code, 400)
        self.assertEqual(caught.exception.field, "request_body")

    def test_unknown_session_is_404_session_id(self):
        with self.assertRaises(ServiceError) as caught:
            self._submit(session_id="ghost")
        self.assertEqual(caught.exception.status_code, 404)
        self.assertEqual(caught.exception.field, "session_id")

    def test_concurrent_same_id_writes_exactly_one_message(self):
        outcomes = []
        errors = []

        def submit():
            try:
                _, status = self._submit()
                outcomes.append(status)
            except ServiceError as error:  # pragma: no cover - unexpected
                errors.append(error)

        threads = [threading.Thread(target=submit) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(errors, [])
        self.assertEqual(sorted(outcomes), [200] * 7 + [201])
        messages, _ = self.service.store.message_page(
            self.session_id, "d2", 0, 100)
        self.assertEqual(len(messages), 1)

    def test_plain_post_is_unaffected_by_submissions(self):
        self._submit()
        # The plain endpoint still appends with its own (non-idempotent)
        # semantics and sees the submitted message in the stream.
        body = self.service.post_message({
            "session_id": self.session_id,
            "sender_device_id": "d1",
            "message_id": "m2",
            "sequence": 2,
            "nonce": base64.b64encode(b"nonce-m2").decode(),
            "ciphertext": "Y2lwaGVy",
        })
        self.assertEqual(body["sequence"], 2)
        self.assertNotIn("request_id", body)


class SubmitHTTPTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = ServiceSubmitFixture()
        self.server, _ = create_server("127.0.0.1", 0, self.fixture.service)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        self.session_id = self.fixture.session_id

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _post(self, path: str, payload: object):
        connection = HTTPConnection("127.0.0.1", self.port, timeout=5)
        connection.request("POST", path, body=json.dumps(payload),
                           headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        body = json.loads(response.read().decode("utf-8"))
        connection.close()
        return response.status, body

    def test_submit_replay_and_conflict_over_http(self):
        payload = _submit_payload(self.session_id)
        status, body = self._post("/v1/messages/submit", payload)
        self.assertEqual(status, 201)
        self.assertEqual(body["request_id"], "r1")

        status, replay = self._post("/v1/messages/submit", payload)
        self.assertEqual(status, 200)
        self.assertEqual(replay, body)

        status, error = self._post(
            "/v1/messages/submit", {**payload, "ciphertext": "eG90aGVy"})
        self.assertEqual(status, 409)
        self.assertEqual(error["field"], "request_id")

    def test_submit_validation_errors_over_http(self):
        status, error = self._post("/v1/messages/submit", {"request_id": ""})
        self.assertEqual(status, 400)
        self.assertEqual(error["field"], "request_id")

        status, error = self._post(
            "/v1/messages/submit", _submit_payload("ghost"))
        self.assertEqual(status, 404)
        self.assertEqual(error["field"], "session_id")

    def test_plain_post_route_unchanged(self):
        status, body = self._post("/v1/messages", {
            "session_id": self.session_id,
            "sender_device_id": "d1",
            "message_id": "m1",
            "sequence": 1,
            "nonce": base64.b64encode(b"nonce-m1").decode(),
            "ciphertext": "Y2lwaGVy",
        })
        self.assertEqual(status, 201)
        self.assertNotIn("request_id", body)


class SubmitPersistenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.path = os.path.join(self.directory, "state.json")
        self.fixture = ServiceSubmitFixture()
        self.service = self.fixture.service
        self.session_id = self.fixture.session_id

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def _document(self) -> dict:
        with open(self.path, encoding="utf-8") as handle:
            return json.load(handle)

    def _rewrite(self, document: dict) -> None:
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)

    def test_round_trip_replays_after_restart(self):
        attach_persistence(self.service, self.path)
        first, status = self.service.submit_message(
            _submit_payload(self.session_id))
        self.assertEqual(status, 201)
        document = self._document()
        self.assertEqual(len(document["message_submissions"]), 1)

        restored = DeviceService()
        attach_persistence(restored, self.path)
        replay, status = restored.submit_message(
            _submit_payload(self.session_id))
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)
        # The stream continues where the persisted state left off.
        payload = _submit_payload(self.session_id, request_id="r2",
                                  message_id="m2", sequence=2)
        _, status = restored.submit_message(payload)
        self.assertEqual(status, 201)

    def test_missing_section_loads_as_empty(self):
        attach_persistence(self.service, self.path)
        document = self._document()
        self.assertIn("message_submissions", document)
        del document["message_submissions"]
        self._rewrite(document)

        restored = DeviceService()
        attach_persistence(restored, self.path)  # must not refuse to start
        _, status = restored.submit_message(_submit_payload(self.session_id))
        self.assertEqual(status, 201)

    def test_contradictory_section_refuses_startup_without_overwrite(self):
        attach_persistence(self.service, self.path)
        self.service.submit_message(_submit_payload(self.session_id))
        good = self._document()
        record = good["message_submissions"][0]

        variants = [
            # duplicate request_id
            good["message_submissions"] + [record],
            # unknown session reference
            [{**record, "session_id": "ghost"}],
            # unknown message reference
            [{**record, "message_id": "ghost"}],
            # envelope field contradicting the stored message
            [{**record, "ciphertext": "eG90aGVy"}],
            [{**record, "sequence": 9}],
        ]
        for submissions in variants:
            self._rewrite({**good, "message_submissions": submissions})
            before = open(self.path, "rb").read()
            with self.assertRaises(StateFileError):
                attach_persistence(DeviceService(), self.path)
            # The refused file is left untouched.
            self.assertEqual(open(self.path, "rb").read(), before)

    def test_persistence_failure_rolls_back_message_and_record(self):
        state_store = attach_persistence(self.service, self.path)

        def fail_save(state):  # noqa: ANN001 - mimics JsonStateStore.save
            raise OSError("simulated disk failure")

        state_store.save = fail_save  # type: ignore[assignment]
        with self.assertRaises(PersistenceUnavailable):
            self.service.submit_message(_submit_payload(self.session_id))
        # Neither the message nor the idempotency record survived.
        messages, _ = self.service.store.message_page(
            self.session_id, "d2", 0, 100)
        self.assertEqual(messages, [])
        self.assertEqual(self.service.store._message_submissions, {})

        # After the disk is repaired the same request_id is still usable.
        del state_store.save  # type: ignore[attr-defined]
        _, status = self.service.submit_message(
            _submit_payload(self.session_id))
        self.assertEqual(status, 201)


class GroupDeliveryRestoreTest(unittest.TestCase):
    """The strengthened group_delivery recovery checks."""

    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.path = os.path.join(self.directory, "state.json")
        self.service = DeviceService()
        for device_id in ("d1", "d2", "d3"):
            self.service.register(_register_payload(device_id))
        self.service.create_group({
            "group_id": "g1", "creator_device_id": "d1",
            "member_device_ids": ["d1", "d2", "d3"]})
        session = self.service.create_group_session({
            "group_id": "g1", "initiator_device_id": "d1",
            "ephemeral_key": _raw_key_b64()})
        self.session_id = session["session_id"]
        self.service.post_message({
            "session_id": self.session_id,
            "sender_device_id": "d1",
            "message_id": "m1",
            "sequence": 1,
            "nonce": base64.b64encode(b"nonce-m1").decode(),
            "ciphertext": "Y2lwaGVy",
        })
        attach_persistence(self.service, self.path)

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def _document_with_delivery(self, device_id: str) -> dict:
        with open(self.path, encoding="utf-8") as handle:
            document = json.load(handle)
        document["group_delivery"] = [{
            "session_id": self.session_id,
            "message_id": "m1",
            "device_id": device_id,
            "attempts": 1,
            "attempt_ids": ["a1"],
            "acked": False,
            "ack_sequence": 0,
        }]
        return document

    def _refuses(self, document: dict) -> None:
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        with self.assertRaises(StateFileError):
            attach_persistence(DeviceService(), self.path)

    def test_unregistered_device_is_refused(self):
        self._refuses(self._document_with_delivery("ghost"))

    def test_sender_device_is_refused(self):
        self._refuses(self._document_with_delivery("d1"))

    def test_revoked_device_is_valid_history_but_writes_stay_409(self):
        self.service.retry_message(self.session_id, "m1",
                                   {"device_id": "d2", "attempt_id": "a1"})
        self.service.revoke_device("d2")

        restored = DeviceService()
        attach_persistence(restored, self.path)  # must not refuse to start
        # The revoked device's record survived the restart as valid history.
        self.assertIn((self.session_id, "m1", "d2"),
                      restored.store._group_delivery)
        # Writes still reject the revoked device with 409/device_id, exactly
        # as the live path enforced before the restart.
        with self.assertRaises(ServiceError) as caught:
            restored.retry_message(self.session_id, "m1",
                                   {"device_id": "d2", "attempt_id": "a2"})
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "device_id")


class CLISubmitTest(unittest.TestCase):
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

    def _submit(self, request_id: str, message_id: str,
                sequence: int) -> subprocess.CompletedProcess:
        nonce = base64.b64encode(f"nonce-{message_id}".encode()).decode()
        return self._run("submit-message", "--request-id", request_id,
                         "--session-id", self.session_id,
                         "--sender-device-id", "d1",
                         "--message-id", message_id,
                         "--sequence", str(sequence),
                         "--nonce", nonce,
                         "--ciphertext", "Y2lwaGVydGV4dA==")

    def test_submit_and_replay_exit_zero_on_stdout(self):
        first = self._submit("r1", "m1", 1)
        self.assertEqual(first.returncode, 0, first.stderr)
        body = json.loads(first.stdout)
        self.assertEqual(body["request_id"], "r1")
        self.assertEqual(body["message_id"], "m1")

        replay = self._submit("r1", "m1", 1)
        self.assertEqual(replay.returncode, 0, replay.stderr)
        self.assertEqual(json.loads(replay.stdout), body)

    def test_conflict_exits_nonzero_with_single_line_json_on_stderr(self):
        self.assertEqual(self._submit("r1", "m1", 1).returncode, 0)
        conflict = self._run("submit-message", "--request-id", "r1",
                             "--session-id", self.session_id,
                             "--sender-device-id", "d1",
                             "--message-id", "m9",
                             "--sequence", "1",
                             "--nonce", "bm9uY2UtbTk=",
                             "--ciphertext", "Y2lwaGVy")
        self.assertNotEqual(conflict.returncode, 0)
        self.assertEqual(conflict.stdout, "")
        line = conflict.stderr.strip()
        self.assertEqual(line.count("\n"), 0)
        self.assertEqual(json.loads(line)["field"], "request_id")


if __name__ == "__main__":
    unittest.main()
