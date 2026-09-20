"""Tests for message sending/pulling: service, HTTP and CLI layers."""
import base64
import json
import subprocess
import sys
import threading
import unittest
from http.client import HTTPConnection

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import x25519

from e2ee_backend.http_app import create_server
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
        # Distinct per message id: a session rejects nonce replays, so each
        # message of a multi-message stream carries its own deterministic nonce.
        "nonce": base64.b64encode(f"nonce-{message_id}".encode()).decode(),
        "ciphertext": base64.b64encode(b"ciphertext-and-tag").decode(),
    }


class ServiceMessageFixture:
    """A service with two devices and one session between them."""

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


class ServiceMessageTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = ServiceMessageFixture()
        self.service = self.fixture.service
        self.session_id = self.fixture.session_id

    def _post(self, **overrides) -> dict:
        # Build the payload from the (possibly overridden) message_id so its
        # deterministic nonce tracks the message, then apply the rest verbatim.
        payload = _message_payload(
            overrides.get("session_id", self.session_id),
            message_id=overrides.get("message_id", "m1"))
        payload.update(overrides)
        return self.service.post_message(payload)

    def test_post_returns_full_envelope_with_created_at(self) -> None:
        body = self._post()
        self.assertEqual(set(body), {"session_id", "sender_device_id",
                                     "message_id", "sequence", "nonce",
                                     "ciphertext", "created_at"})
        self.assertEqual(body["sequence"], 1)
        self.assertTrue(body["created_at"].endswith("+00:00"))

    def test_missing_field_is_400_with_field_name(self) -> None:
        payload = _message_payload(self.session_id)
        del payload["nonce"]
        with self.assertRaises(ServiceError) as ctx:
            self.service.post_message(payload)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.field, "nonce")

    def test_wrong_type_is_400_with_field_name(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self._post(ciphertext=42)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.field, "ciphertext")

    def test_non_integer_sequence_is_400(self) -> None:
        for bad in ("1", 1.5, True, None):
            with self.assertRaises(ServiceError) as ctx:
                self._post(sequence=bad)
            self.assertEqual(ctx.exception.status_code, 400, bad)
            self.assertEqual(ctx.exception.field, "sequence")

    def test_unknown_session_is_404(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self._post(session_id="ghost")
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "session_id")

    def test_unknown_sender_is_409(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self._post(sender_device_id="ghost")
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "sender_device_id")

    def test_revoked_sender_is_409_and_writes_nothing(self) -> None:
        self.service.revoke_device("d1")
        with self.assertRaises(ServiceError) as ctx:
            self._post()
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "sender_device_id")
        page = self.service.list_messages(self.session_id, "d2", 0, 100)
        self.assertEqual(page["messages"], [])
        self.assertEqual(page["next_after"], 0)

    def test_duplicate_message_id_is_409(self) -> None:
        self._post()
        with self.assertRaises(ServiceError) as ctx:
            self._post(sequence=2)
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "message_id")

    def test_sequence_must_start_at_one_and_continue(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self._post(sequence=2)
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "sequence")

        self._post(message_id="m1", sequence=1)
        self._post(message_id="m2", sequence=2)
        with self.assertRaises(ServiceError) as ctx:
            self._post(message_id="m3", sequence=4)
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "sequence")
        with self.assertRaises(ServiceError) as ctx:
            self._post(message_id="m3", sequence=0)
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "sequence")
        self._post(message_id="m3", sequence=3)

    def test_failed_post_does_not_advance_sequence(self) -> None:
        self._post(message_id="m1", sequence=1)
        with self.assertRaises(ServiceError):
            self._post(message_id="m2", sequence=5)
        body = self._post(message_id="m2", sequence=2)
        self.assertEqual(body["sequence"], 2)

    def test_sequences_are_independent_per_session(self) -> None:
        other = self.service.create_session({
            "initiator_device_id": "d2",
            "recipient_device_id": "d1",
            "prekey_id": "k1",
            "ephemeral_key": _raw_key_b64(),
        })
        self._post(message_id="m1", sequence=1)
        body = self._post(session_id=other["session_id"],
                          message_id="m1", sequence=1)
        self.assertEqual(body["session_id"], other["session_id"])

    def test_replayed_nonce_is_409_and_writes_nothing(self) -> None:
        nonce_a = base64.b64encode(b"AAAAAAAAAAAAAAAA").decode()
        self._post(message_id="m1", sequence=1, nonce=nonce_a)
        # Fresh id and the correct next sequence, but the session already saw
        # this exact nonce: a replay, rejected as 409/nonce.
        with self.assertRaises(ServiceError) as ctx:
            self._post(message_id="m2", sequence=2, nonce=nonce_a)
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "nonce")

        # Nothing was written and the sequence cursor did not advance: the
        # page still holds only m1, and a legitimate m2/2 still goes through.
        page = self.service.list_messages(self.session_id, "d2", 0, 100)
        self.assertEqual([m["message_id"] for m in page["messages"]], ["m1"])
        self.assertEqual(page["next_after"], 1)
        body = self._post(message_id="m2", sequence=2,
                          nonce=base64.b64encode(b"BBBBBBBBBBBBBBBB").decode())
        self.assertEqual(body["sequence"], 2)

    def test_identical_nonce_is_allowed_in_another_session(self) -> None:
        nonce_a = base64.b64encode(b"AAAAAAAAAAAAAAAA").decode()
        other = self.service.create_session({
            "initiator_device_id": "d2",
            "recipient_device_id": "d1",
            "prekey_id": "k1",
            "ephemeral_key": _raw_key_b64(),
        })
        self._post(message_id="m1", sequence=1, nonce=nonce_a)
        # Same nonce (and even the same message_id/sequence) in another
        # session is an independent context and is accepted.
        body = self._post(session_id=other["session_id"],
                          message_id="m1", sequence=1, nonce=nonce_a)
        self.assertEqual(body["session_id"], other["session_id"])

    def test_nonce_is_compared_as_the_raw_string(self) -> None:
        # Byte-for-byte string equality: spacing/case differences are distinct
        # nonces, while the exact repeat is the replay.
        self._post(message_id="m1", sequence=1, nonce="nonce-1")
        self._post(message_id="m2", sequence=2, nonce="nonce-1 ")
        self._post(message_id="m3", sequence=3, nonce="Nonce-1")
        with self.assertRaises(ServiceError) as ctx:
            self._post(message_id="m4", sequence=4, nonce="nonce-1")
        self.assertEqual(ctx.exception.field, "nonce")

    def test_replay_does_not_change_delivery_state(self) -> None:
        nonce_a = base64.b64encode(b"AAAAAAAAAAAAAAAA").decode()
        self._post(message_id="m1", sequence=1, nonce=nonce_a)
        with self.assertRaises(ServiceError):
            self._post(message_id="m2", sequence=2, nonce=nonce_a)
        # m1 has never been retried: the rejected replay records no attempt.
        view, status = self.service.retry_message(
            self.session_id, "m1",
            {"device_id": "d2", "attempt_id": "a1"})
        self.assertEqual(status, 201)
        self.assertEqual(view["attempts"], 1)
        self.assertEqual(view["status"], "pending")

    def test_check_priority_dup_id_then_bad_sequence_then_nonce(self) -> None:
        nonce_a = base64.b64encode(b"AAAAAAAAAAAAAAAA").decode()
        self._post(message_id="m1", sequence=1, nonce=nonce_a)
        # Same id AND same nonce: duplicate message_id wins.
        with self.assertRaises(ServiceError) as ctx:
            self._post(message_id="m1", sequence=2, nonce=nonce_a)
        self.assertEqual(ctx.exception.field, "message_id")
        # Fresh id but wrong sequence AND reused nonce: sequence wins.
        with self.assertRaises(ServiceError) as ctx:
            self._post(message_id="m2", sequence=9, nonce=nonce_a)
        self.assertEqual(ctx.exception.field, "sequence")

    def test_concurrent_identical_envelope_is_linearized_to_one_append(self):
        import threading

        nonce_a = base64.b64encode(b"AAAAAAAAAAAAAAAA").decode()
        results = []

        def send() -> None:
            try:
                self._post(message_id="m1", sequence=1, nonce=nonce_a)
                results.append("ok")
            except ServiceError:
                results.append("conflict")

        threads = [threading.Thread(target=send) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(results.count("ok"), 1)
        self.assertEqual(results.count("conflict"), 7)
        page = self.service.list_messages(self.session_id, "d2", 0, 100)
        self.assertEqual([m["message_id"] for m in page["messages"]], ["m1"])

    def test_list_paginates_ascending_with_next_after(self) -> None:
        for index in range(1, 6):
            self._post(message_id=f"m{index}", sequence=index)

        page = self.service.list_messages(self.session_id, "d2", 0, 2)
        self.assertEqual([m["message_id"] for m in page["messages"]],
                         ["m1", "m2"])
        self.assertEqual(page["next_after"], 2)

        page = self.service.list_messages(
            self.session_id, "d2", page["next_after"], 2)
        self.assertEqual([m["message_id"] for m in page["messages"]],
                         ["m3", "m4"])
        self.assertEqual(page["next_after"], 4)

        page = self.service.list_messages(
            self.session_id, "d2", page["next_after"], 2)
        self.assertEqual([m["message_id"] for m in page["messages"]], ["m5"])
        self.assertEqual(page["next_after"], 5)

        page = self.service.list_messages(
            self.session_id, "d2", page["next_after"], 2)
        self.assertEqual(page["messages"], [])
        self.assertEqual(page["next_after"], 5)

    def test_list_message_bodies_match_post_bodies(self) -> None:
        posted = self._post()
        page = self.service.list_messages(self.session_id, "d2", 0, 100)
        self.assertEqual(page["messages"], [posted])

    def test_list_unknown_session_is_404(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.list_messages("ghost", "d1", 0, 100)
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "session_id")

    def test_list_with_revoked_reader_is_409(self) -> None:
        self.service.revoke_device("d2")
        with self.assertRaises(ServiceError) as ctx:
            self.service.list_messages(self.session_id, "d2", 0, 100)
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "device_id")

    def test_stored_messages_survive_sender_revocation(self) -> None:
        self._post()
        self.service.revoke_device("d1")
        page = self.service.list_messages(self.session_id, "d2", 0, 100)
        self.assertEqual(len(page["messages"]), 1)

    def test_revoking_other_device_does_not_affect_session(self) -> None:
        self.service.register(_register_payload("d3", user_id="u2"))
        self.service.revoke_device("d3")
        body = self._post()
        self.assertEqual(body["sequence"], 1)


class HTTPMessageTest(unittest.TestCase):
    def setUp(self) -> None:
        self.server, self.service = create_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        for device_id in ("d1", "d2"):
            self._request("POST", "/v1/devices", _register_payload(device_id))
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

    def _post_message(self, message_id: str = "m1", sequence: int = 1):
        return self._request("POST", "/v1/messages",
                             _message_payload(self.session_id, message_id,
                                              sequence))

    def test_post_message_is_201_with_created_at(self) -> None:
        status, body = self._post_message()
        self.assertEqual(status, 201)
        self.assertEqual(set(body), {"session_id", "sender_device_id",
                                     "message_id", "sequence", "nonce",
                                     "ciphertext", "created_at"})
        self.assertTrue(body["created_at"].endswith("+00:00"))

    def test_post_message_errors_carry_field(self) -> None:
        status, body = self._request("POST", "/v1/messages", {"session_id": "x"})
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "sender_device_id")

        status, body = self._request(
            "POST", "/v1/messages",
            _message_payload("ghost"))
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "session_id")

        self._post_message()
        status, body = self._post_message(message_id="m1", sequence=2)
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "message_id")
        status, body = self._post_message(message_id="m2", sequence=5)
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "sequence")

    def test_replayed_nonce_is_409_with_field_nonce(self) -> None:
        nonce_a = base64.b64encode(b"AAAAAAAAAAAAAAAA").decode()
        status, body = self._request(
            "POST", "/v1/messages",
            _message_payload(self.session_id, "m1", 1)
            | {"nonce": nonce_a})
        self.assertEqual(status, 201)
        status, body = self._request(
            "POST", "/v1/messages",
            _message_payload(self.session_id, "m2", 2)
            | {"nonce": nonce_a})
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "nonce")
        # Nothing was written: the stream still holds only m1.
        status, body = self._request(
            "GET", f"/v1/messages/{self.session_id}?device_id=d2")
        self.assertEqual([m["message_id"] for m in body["messages"]], ["m1"])

    def test_same_nonce_in_another_session_is_201(self) -> None:
        nonce_a = base64.b64encode(b"AAAAAAAAAAAAAAAA").decode()
        _, other = self._request("POST", "/v1/sessions", {
            "initiator_device_id": "d2",
            "recipient_device_id": "d1",
            "prekey_id": "k1",
            "ephemeral_key": _raw_key_b64(),
        })
        for session_id in (self.session_id, other["session_id"]):
            status, _ = self._request(
                "POST", "/v1/messages",
                _message_payload(session_id, "m1", 1) | {"nonce": nonce_a})
            self.assertEqual(status, 201, session_id)

    def test_get_messages_roundtrip_and_pagination(self) -> None:
        for index in range(1, 4):
            self.assertEqual(self._post_message(f"m{index}", index)[0], 201)

        status, body = self._request(
            "GET", f"/v1/messages/{self.session_id}?device_id=d2")
        self.assertEqual(status, 200)
        self.assertEqual(set(body), {"messages", "next_after"})
        self.assertEqual([m["message_id"] for m in body["messages"]],
                         ["m1", "m2", "m3"])
        self.assertEqual(body["next_after"], 3)

        status, body = self._request(
            "GET", f"/v1/messages/{self.session_id}?device_id=d2&after=1&limit=1")
        self.assertEqual(status, 200)
        self.assertEqual([m["message_id"] for m in body["messages"]], ["m2"])
        self.assertEqual(body["next_after"], 2)

        status, body = self._request(
            "GET", f"/v1/messages/{self.session_id}?device_id=d2&after=3")
        self.assertEqual(body["messages"], [])
        self.assertEqual(body["next_after"], 3)

    def test_get_messages_parameter_errors(self) -> None:
        base = f"/v1/messages/{self.session_id}"
        for path, field in (
            (base, "device_id"),
            (base + "?device_id=", "device_id"),
            (base + "?device_id=d1&after=-1", "after"),
            (base + "?device_id=d1&after=abc", "after"),
            (base + "?device_id=d1&limit=0", "limit"),
            (base + "?device_id=d1&limit=101", "limit"),
            (base + "?device_id=d1&limit=two", "limit"),
        ):
            status, body = self._request("GET", path)
            self.assertEqual(status, 400, path)
            self.assertEqual(body["field"], field, path)

    def test_get_messages_unknown_session_and_device(self) -> None:
        status, body = self._request("GET", "/v1/messages/ghost?device_id=d1")
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "session_id")

        status, body = self._request(
            "GET", f"/v1/messages/{self.session_id}?device_id=ghost")
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "device_id")


class CLIMessageTest(unittest.TestCase):
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

    def _send(self, message_id: str, sequence: int) -> subprocess.CompletedProcess:
        # One deterministic nonce per message id; a session rejects replays.
        nonce = base64.b64encode(f"nonce-{message_id}".encode()).decode()
        return self._run("send-message", "--session-id", self.session_id,
                         "--sender-device-id", "d1",
                         "--message-id", message_id,
                         "--sequence", str(sequence),
                         "--nonce", nonce,
                         "--ciphertext", "Y2lwaGVydGV4dA==")

    def test_send_and_pull_roundtrip(self) -> None:
        result = self._send("m1", 1)
        self.assertEqual(result.returncode, 0, result.stderr)
        sent = json.loads(result.stdout)
        self.assertEqual(sent["message_id"], "m1")
        self.assertEqual(sent["sequence"], 1)
        self.assertTrue(sent["created_at"].endswith("+00:00"))

        self.assertEqual(self._send("m2", 2).returncode, 0)

        result = self._run("pull-messages", self.session_id, "--device-id", "d2")
        self.assertEqual(result.returncode, 0, result.stderr)
        line = result.stdout.strip()
        self.assertEqual(line.count("\n"), 0)
        body = json.loads(line)
        self.assertEqual([m["message_id"] for m in body["messages"]],
                         ["m1", "m2"])
        self.assertEqual(body["next_after"], 2)

        result = self._run("pull-messages", self.session_id,
                           "--device-id", "d2", "--after", "2")
        body = json.loads(result.stdout)
        self.assertEqual(body["messages"], [])
        self.assertEqual(body["next_after"], 2)

    def test_send_conflict_exits_nonzero_with_json_error(self) -> None:
        self.assertEqual(self._send("m1", 1).returncode, 0)
        result = self._send("m1", 2)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        error = json.loads(result.stderr)
        self.assertEqual(error["field"], "message_id")

    def test_replayed_nonce_exits_nonzero_with_field_nonce(self) -> None:
        nonce = base64.b64encode(b"AAAAAAAAAAAAAAAA").decode()
        first = self._run("send-message", "--session-id", self.session_id,
                          "--sender-device-id", "d1", "--message-id", "m1",
                          "--sequence", "1", "--nonce", nonce,
                          "--ciphertext", "Y2lw")
        self.assertEqual(first.returncode, 0, first.stderr)
        # Fresh id and correct sequence, but the session already saw the nonce.
        replay = self._run("send-message", "--session-id", self.session_id,
                           "--sender-device-id", "d1", "--message-id", "m2",
                           "--sequence", "2", "--nonce", nonce,
                           "--ciphertext", "Y2lw")
        self.assertNotEqual(replay.returncode, 0)
        self.assertEqual(replay.stdout, "")
        self.assertEqual(json.loads(replay.stderr)["field"], "nonce")
        # The rejected replay did not land in the stream.
        pulled = self._run("pull-messages", self.session_id,
                           "--device-id", "d2")
        body = json.loads(pulled.stdout)
        self.assertEqual([m["message_id"] for m in body["messages"]], ["m1"])

    def test_same_nonce_in_other_session_exits_zero(self) -> None:
        nonce = base64.b64encode(b"AAAAAAAAAAAAAAAA").decode()
        other = self._run("create-session",
                          "--initiator-device-id", "d2",
                          "--recipient-device-id", "d1",
                          "--prekey-id", "k1",
                          "--ephemeral-key", _raw_key_b64())
        other_sid = json.loads(other.stdout)["session_id"]
        for session_id in (self.session_id, other_sid):
            result = self._run("send-message", "--session-id", session_id,
                               "--sender-device-id", "d1",
                               "--message-id", "m1", "--sequence", "1",
                               "--nonce", nonce, "--ciphertext", "Y2lw")
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_pull_unknown_session_exits_nonzero(self) -> None:
        result = self._run("pull-messages", "ghost", "--device-id", "d1")
        self.assertNotEqual(result.returncode, 0)
        error = json.loads(result.stderr)
        self.assertEqual(error["field"], "session_id")


if __name__ == "__main__":
    unittest.main()
