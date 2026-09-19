"""Tests for session negotiation (POST /v1/sessions) and snapshot queries."""
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


class SessionServiceTest(unittest.TestCase):
    """Service-level validation and snapshot semantics."""

    def setUp(self) -> None:
        self.service = DeviceService()
        self.initiator_identity = _raw_key_b64()
        self.recipient_identity = _raw_key_b64()
        self.prekey_public = _raw_key_b64()
        self.ephemeral = _raw_key_b64()
        self.service.register({
            "user_id": "u1", "device_id": "init",
            "identity_key": self.initiator_identity,
            "signed_prekeys": [{"key_id": "i1", "public_key": _raw_key_b64()}],
        })
        self.service.register({
            "user_id": "u2", "device_id": "recv",
            "identity_key": self.recipient_identity,
            "signed_prekeys": [{"key_id": "r1", "public_key": self.prekey_public}],
        })

    def _payload(self, **overrides) -> dict:
        payload = {
            "initiator_device_id": "init",
            "recipient_device_id": "recv",
            "prekey_id": "r1",
            "ephemeral_key": self.ephemeral,
        }
        payload.update(overrides)
        return payload

    def _create(self, **overrides) -> dict:
        return self.service.create_session(self._payload(**overrides))

    # -- 201 contract ------------------------------------------------------

    def test_create_returns_eight_fields_and_echoes_request(self) -> None:
        body = self._create()
        self.assertEqual(set(body), {
            "initiator_device_id", "recipient_device_id", "prekey_id",
            "ephemeral_key", "session_id", "identity_key", "public_key",
            "created_at"})
        self.assertEqual(body["initiator_device_id"], "init")
        self.assertEqual(body["recipient_device_id"], "recv")
        self.assertEqual(body["prekey_id"], "r1")
        self.assertEqual(body["ephemeral_key"], self.ephemeral)
        self.assertEqual(body["identity_key"], self.recipient_identity)
        self.assertEqual(body["public_key"], self.prekey_public)
        self.assertTrue(body["session_id"])
        self.assertTrue(body["created_at"].endswith("+00:00"))

    def test_repeated_post_creates_new_unique_session(self) -> None:
        first = self._create()
        second = self._create()
        self.assertNotEqual(first["session_id"], second["session_id"])

    def test_get_returns_same_snapshot(self) -> None:
        created = self._create()
        self.assertEqual(self.service.get_session(created["session_id"]), created)

    def test_snapshot_unchanged_after_revocation(self) -> None:
        created = self._create()
        self.service.revoke_device("recv")
        self.assertEqual(self.service.get_session(created["session_id"]), created)

    # -- 400 validation ----------------------------------------------------

    def test_missing_each_field_is_400_and_names_it(self) -> None:
        for field in ("initiator_device_id", "recipient_device_id",
                      "prekey_id", "ephemeral_key"):
            payload = self._payload()
            del payload[field]
            with self.assertRaises(ServiceError) as caught:
                self.service.create_session(payload)
            self.assertEqual(caught.exception.status_code, 400)
            self.assertEqual(caught.exception.field, field)

    def test_wrong_type_is_400_and_names_field(self) -> None:
        for bad in (None, 1, True, [], {}, ""):
            with self.assertRaises(ServiceError) as caught:
                self._create(prekey_id=bad)
            self.assertEqual(caught.exception.status_code, 400)
            self.assertEqual(caught.exception.field, "prekey_id")

    def test_bad_ephemeral_key_encoding_is_400(self) -> None:
        with self.assertRaises(ServiceError) as caught:
            self._create(ephemeral_key="!!!not-a-key!!!")
        self.assertEqual(caught.exception.status_code, 400)
        self.assertEqual(caught.exception.field, "ephemeral_key")

    def test_same_device_is_400_field_recipient(self) -> None:
        with self.assertRaises(ServiceError) as caught:
            self._create(recipient_device_id="init")
        self.assertEqual(caught.exception.status_code, 400)
        self.assertEqual(caught.exception.field, "recipient_device_id")

    def test_non_object_body_is_400(self) -> None:
        with self.assertRaises(ServiceError) as caught:
            self.service.create_session(["not", "a", "dict"])
        self.assertEqual(caught.exception.status_code, 400)

    # -- 404 unknown -------------------------------------------------------

    def test_unknown_initiator_is_404(self) -> None:
        with self.assertRaises(ServiceError) as caught:
            self._create(initiator_device_id="ghost")
        self.assertEqual(caught.exception.status_code, 404)
        self.assertEqual(caught.exception.field, "initiator_device_id")

    def test_unknown_recipient_is_404(self) -> None:
        with self.assertRaises(ServiceError) as caught:
            self._create(recipient_device_id="ghost")
        self.assertEqual(caught.exception.status_code, 404)
        self.assertEqual(caught.exception.field, "recipient_device_id")

    def test_unknown_prekey_is_404(self) -> None:
        with self.assertRaises(ServiceError) as caught:
            self._create(prekey_id="nope")
        self.assertEqual(caught.exception.status_code, 404)
        self.assertEqual(caught.exception.field, "prekey_id")

    def test_prekey_of_other_device_is_404(self) -> None:
        # "i1" exists on the initiator, not on the recipient.
        with self.assertRaises(ServiceError) as caught:
            self._create(prekey_id="i1")
        self.assertEqual(caught.exception.status_code, 404)
        self.assertEqual(caught.exception.field, "prekey_id")

    def test_unknown_session_get_is_404(self) -> None:
        with self.assertRaises(ServiceError) as caught:
            self.service.get_session("ghost")
        self.assertEqual(caught.exception.status_code, 404)
        self.assertEqual(caught.exception.field, "session_id")

    # -- 409 revoked -------------------------------------------------------

    def test_revoked_initiator_is_409(self) -> None:
        self.service.revoke_device("init")
        with self.assertRaises(ServiceError) as caught:
            self._create()
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "initiator_device_id")

    def test_revoked_recipient_is_409(self) -> None:
        self.service.revoke_device("recv")
        with self.assertRaises(ServiceError) as caught:
            self._create()
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "recipient_device_id")

    def test_revoked_prekey_is_409(self) -> None:
        self.service.revoke_prekey("recv", "r1")
        with self.assertRaises(ServiceError) as caught:
            self._create()
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "prekey_id")

    def test_failed_create_writes_nothing(self) -> None:
        self.service.revoke_device("recv")
        with self.assertRaises(ServiceError):
            self._create()
        # The failed attempt stored no session at all.
        self.assertEqual(self.service.store._sessions, {})


class SessionHTTPTest(unittest.TestCase):
    """Full HTTP round-trip over a real loopback socket."""

    def setUp(self) -> None:
        self.server, _ = create_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.ephemeral = _raw_key_b64()
        self._register("init", "u1")
        self._register("recv", "u2")

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

    def _register(self, device_id: str, user_id: str) -> None:
        status, _ = self._request("POST", "/v1/devices", {
            "user_id": user_id, "device_id": device_id,
            "identity_key": _raw_key_b64(),
            "signed_prekeys": [{"key_id": "k1", "public_key": _raw_key_b64()}],
        })
        self.assertEqual(status, 201)

    def _session_payload(self, **overrides) -> dict:
        payload = {
            "initiator_device_id": "init",
            "recipient_device_id": "recv",
            "prekey_id": "k1",
            "ephemeral_key": self.ephemeral,
        }
        payload.update(overrides)
        return payload

    def test_post_201_then_get_200_same_body(self) -> None:
        status, created = self._request("POST", "/v1/sessions",
                                        self._session_payload())
        self.assertEqual(status, 201)
        self.assertEqual(len(created), 8)
        status, fetched = self._request(
            "GET", f"/v1/sessions/{created['session_id']}")
        self.assertEqual(status, 200)
        self.assertEqual(fetched, created)

    def test_post_missing_field_is_400(self) -> None:
        payload = self._session_payload()
        del payload["ephemeral_key"]
        status, body = self._request("POST", "/v1/sessions", payload)
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "ephemeral_key")

    def test_post_bad_ephemeral_key_is_400(self) -> None:
        status, body = self._request(
            "POST", "/v1/sessions", self._session_payload(ephemeral_key="nope"))
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "ephemeral_key")

    def test_post_unknown_device_is_404(self) -> None:
        status, body = self._request(
            "POST", "/v1/sessions", self._session_payload(recipient_device_id="ghost"))
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "recipient_device_id")

    def test_post_revoked_prekey_is_409(self) -> None:
        self.assertEqual(
            self._request("POST", "/v1/devices/recv/prekeys/k1/revoke")[0], 200)
        status, body = self._request("POST", "/v1/sessions",
                                     self._session_payload())
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "prekey_id")

    def test_get_unknown_session_is_404(self) -> None:
        status, body = self._request("GET", "/v1/sessions/ghost")
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "session_id")

    def test_get_session_subpath_is_404(self) -> None:
        status, body = self._request("GET", "/v1/sessions/a/b")
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "session_id")

    def test_snapshot_survives_revocation_over_http(self) -> None:
        _, created = self._request("POST", "/v1/sessions", self._session_payload())
        self.assertEqual(self._request("POST", "/v1/devices/recv/revoke")[0], 200)
        status, fetched = self._request(
            "GET", f"/v1/sessions/{created['session_id']}")
        self.assertEqual(status, 200)
        self.assertEqual(fetched, created)


class SessionCLITest(unittest.TestCase):
    """CLI subcommands against a real server."""

    def setUp(self) -> None:
        self.server, _ = create_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self._register("init", "u1")
        self._register("recv", "u2")

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _run(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "e2ee_backend", "--base-url", self.base_url,
             *arguments],
            capture_output=True, text=True, timeout=15)

    def _register(self, device_id: str, user_id: str) -> None:
        result = self._run(
            "register", "--user-id", user_id, "--device-id", device_id,
            "--identity-key", _raw_key_b64(),
            "--prekey", f"k1:{_raw_key_b64()}")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_create_and_show_session(self) -> None:
        result = self._run(
            "create-session", "--initiator-device-id", "init",
            "--recipient-device-id", "recv", "--prekey-id", "k1",
            "--ephemeral-key", _raw_key_b64())
        self.assertEqual(result.returncode, 0, result.stderr)
        line = result.stdout.strip()
        self.assertEqual(line.count("\n"), 0)
        created = json.loads(line)
        self.assertEqual(len(created), 8)

        shown = self._run("show-session", created["session_id"])
        self.assertEqual(shown.returncode, 0, shown.stderr)
        self.assertEqual(json.loads(shown.stdout.strip()), created)

    def test_create_session_error_goes_to_stderr(self) -> None:
        result = self._run(
            "create-session", "--initiator-device-id", "init",
            "--recipient-device-id", "ghost", "--prekey-id", "k1",
            "--ephemeral-key", _raw_key_b64())
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        self.assertEqual(json.loads(result.stderr.strip())["field"],
                         "recipient_device_id")

    def test_show_unknown_session(self) -> None:
        result = self._run("show-session", "ghost")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(json.loads(result.stderr.strip())["field"], "session_id")


if __name__ == "__main__":
    unittest.main()
