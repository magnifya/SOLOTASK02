"""Tests for device/pre-key revocation: service, HTTP and CLI layers."""
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


class ServiceRevokeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = DeviceService()
        self.service.register(_payload())

    def test_revoke_device_contract_and_idempotence(self) -> None:
        body = self.service.revoke_device("d1")
        self.assertEqual(body, {"device_id": "d1", "revoked": True})
        self.assertEqual(self.service.revoke_device("d1"), body)

    def test_revoke_unknown_device_is_404(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.revoke_device("ghost")
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "device_id")

    def test_revoked_device_lists_no_prekeys_but_keeps_record(self) -> None:
        before = self.service.get_device("d1")
        self.service.revoke_device("d1")
        after = self.service.get_device("d1")
        self.assertEqual(after["prekey_ids"], [])
        self.assertEqual(after["identity_key"], before["identity_key"])
        self.assertEqual(after["registered_at"], before["registered_at"])

    def test_revoke_prekey_contract_and_idempotence(self) -> None:
        body = self.service.revoke_prekey("d1", "k1")
        self.assertEqual(body, {"device_id": "d1", "key_id": "k1",
                                "revoked": True})
        self.assertEqual(self.service.revoke_prekey("d1", "k1"), body)

    def test_revoke_prekey_unknown_device_is_404(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.revoke_prekey("ghost", "k1")
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "device_id")

    def test_revoke_unknown_prekey_is_404(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.revoke_prekey("d1", "ghost")
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "key_id")

    def test_single_key_revoke_leaves_order_and_peers_untouched(self) -> None:
        self.service.register(_payload(device_id="d2"))
        self.service.revoke_prekey("d1", "k1")
        self.assertEqual(self.service.get_device("d1")["prekey_ids"], ["k2"])
        self.assertEqual(self.service.get_device("d2")["prekey_ids"],
                         ["k1", "k2"])


class HTTPRevokeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.server, _ = create_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        self._request("POST", "/v1/devices", _payload())

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

    def test_revoke_device_200_then_empty_prekeys(self) -> None:
        status, body = self._request("POST", "/v1/devices/d1/revoke")
        self.assertEqual(status, 200)
        self.assertEqual(body, {"device_id": "d1", "revoked": True})
        # Idempotent.
        self.assertEqual(self._request("POST", "/v1/devices/d1/revoke"),
                         (status, body))

        status, shown = self._request("GET", "/v1/devices/d1")
        self.assertEqual(status, 200)
        self.assertEqual(shown["prekey_ids"], [])
        self.assertTrue(shown["identity_key"])
        self.assertTrue(shown["registered_at"])

    def test_revoke_device_unknown_is_404(self) -> None:
        status, body = self._request("POST", "/v1/devices/ghost/revoke")
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "device_id")

    def test_revoke_prekey_200_then_excluded(self) -> None:
        status, body = self._request("POST", "/v1/devices/d1/prekeys/k1/revoke")
        self.assertEqual(status, 200)
        self.assertEqual(body, {"device_id": "d1", "key_id": "k1",
                                "revoked": True})
        # Idempotent.
        self.assertEqual(
            self._request("POST", "/v1/devices/d1/prekeys/k1/revoke"),
            (status, body))

        status, shown = self._request("GET", "/v1/devices/d1")
        self.assertEqual(status, 200)
        self.assertEqual(shown["prekey_ids"], ["k2"])

    def test_revoke_prekey_unknown_device_is_404(self) -> None:
        status, body = self._request("POST", "/v1/devices/ghost/prekeys/k1/revoke")
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "device_id")

    def test_revoke_unknown_prekey_is_404(self) -> None:
        status, body = self._request("POST", "/v1/devices/d1/prekeys/ghost/revoke")
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "key_id")


class CLIRevokeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.server, _ = create_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _run(self, *arguments: str, base_url: str = None) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "e2ee_backend", "--base-url",
             base_url or self.base_url, *arguments],
            capture_output=True, text=True, timeout=15)

    def _register(self, device_id: str = "d1") -> None:
        result = self._run(
            "register", "--user-id", "u1", "--device-id", device_id,
            "--identity-key", _raw_key_b64(),
            "--prekey", f"k1:{_raw_key_b64()}",
            "--prekey", f"k2:{_raw_key_b64()}")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_revoke_device_success_and_idempotence(self) -> None:
        self._register()
        result = self._run("revoke-device", "--device-id", "d1")
        self.assertEqual(result.returncode, 0, result.stderr)
        line = result.stdout.strip()
        self.assertEqual(line.count("\n"), 0)
        self.assertEqual(json.loads(line), {"device_id": "d1", "revoked": True})
        repeat = self._run("revoke-device", "--device-id", "d1")
        self.assertEqual(repeat.returncode, 0)
        self.assertEqual(repeat.stdout, result.stdout)

        shown = self._run("show", "d1")
        self.assertEqual(json.loads(shown.stdout.strip())["prekey_ids"], [])

    def test_revoke_device_unknown_names_field(self) -> None:
        result = self._run("revoke-device", "--device-id", "ghost")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(json.loads(result.stderr.strip())["field"], "device_id")

    def test_revoke_prekey_success_and_idempotence(self) -> None:
        self._register()
        result = self._run("revoke-prekey", "--device-id", "d1", "--key-id", "k1")
        self.assertEqual(result.returncode, 0, result.stderr)
        line = result.stdout.strip()
        self.assertEqual(line.count("\n"), 0)
        self.assertEqual(json.loads(line),
                         {"device_id": "d1", "key_id": "k1", "revoked": True})
        repeat = self._run("revoke-prekey", "--device-id", "d1", "--key-id", "k1")
        self.assertEqual(repeat.returncode, 0)
        self.assertEqual(repeat.stdout, result.stdout)

        shown = self._run("show", "d1")
        self.assertEqual(json.loads(shown.stdout.strip())["prekey_ids"], ["k2"])

    def test_revoke_prekey_unknown_device_and_key(self) -> None:
        self._register()
        result = self._run("revoke-prekey", "--device-id", "ghost", "--key-id", "k1")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(json.loads(result.stderr.strip())["field"], "device_id")

        result = self._run("revoke-prekey", "--device-id", "d1", "--key-id", "ghost")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(json.loads(result.stderr.strip())["field"], "key_id")

    def test_connection_failure_is_single_line_json_without_traceback(self) -> None:
        dead_url = f"http://127.0.0.1:{self.port + 1}"
        commands = [
            ("register", "--user-id", "u1", "--device-id", "d1",
             "--identity-key", _raw_key_b64()),
            ("show", "d1"),
            ("revoke-device", "--device-id", "d1"),
            ("revoke-prekey", "--device-id", "d1", "--key-id", "k1"),
        ]
        for command in commands:
            result = self._run(*command, base_url=dead_url)
            self.assertNotEqual(result.returncode, 0, command)
            self.assertNotIn("Traceback", result.stderr)
            line = result.stderr.strip()
            self.assertEqual(line.count("\n"), 0)
            self.assertEqual(json.loads(line)["field"], "server", command)
            self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
