"""端到端测试：HTTP 接口与命令行。"""

import json
import threading
import unittest
import urllib.error
import urllib.request

from e2e_backend.cli import main as cli_main
from e2e_backend.server import create_server
from e2e_backend.store import DeviceStore


class ServerTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = DeviceStore()
        cls.server = create_server("127.0.0.1", 0, cls.store)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = "http://127.0.0.1:%d" % cls.port

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def request(self, method, path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            self.base + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as err:
            return err.code, json.loads(err.read())

    def make_payload(self, **overrides):
        payload = {
            "user_id": "user-1",
            "device_id": "dev-1",
            "identity_key": "IK",
            "signed_prekeys": [
                {"key_id": "k1", "public_key": "PK1"},
                {"key_id": "k2", "public_key": "PK2"},
            ],
        }
        payload.update(overrides)
        return payload


class TestRegister(ServerTestCase):
    def test_register_success_returns_201(self):
        status, body = self.request("POST", "/v1/devices", self.make_payload())
        self.assertEqual(status, 201)
        self.assertEqual(body["device_id"], "dev-1")
        self.assertIn("registered_at", body)

    def test_duplicate_device_returns_409(self):
        payload = self.make_payload(device_id="dev-dup")
        self.assertEqual(self.request("POST", "/v1/devices", payload)[0], 201)
        status, _ = self.request("POST", "/v1/devices", payload)
        self.assertEqual(status, 409)

    def test_same_device_id_under_other_user_is_allowed(self):
        payload = self.make_payload(user_id="user-2", device_id="dev-dup")
        status, _ = self.request("POST", "/v1/devices", payload)
        self.assertEqual(status, 201)

    def test_missing_required_field_returns_400_with_field_name(self):
        for field_name in ("user_id", "device_id", "identity_key", "signed_prekeys"):
            payload = self.make_payload(device_id="dev-400-" + field_name)
            del payload[field_name]
            status, body = self.request("POST", "/v1/devices", payload)
            self.assertEqual(status, 400, field_name)
            self.assertEqual(body["field"], field_name)

    def test_malformed_prekey_element_returns_400(self):
        payload = self.make_payload(
            device_id="dev-badprekey",
            signed_prekeys=[{"key_id": "k1"}],
        )
        status, body = self.request("POST", "/v1/devices", payload)
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "signed_prekeys[0].public_key")

    def test_prekeys_not_a_list_returns_400(self):
        payload = self.make_payload(device_id="dev-notlist", signed_prekeys="k1")
        status, body = self.request("POST", "/v1/devices", payload)
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "signed_prekeys")


class TestShow(ServerTestCase):
    def test_show_returns_identity_key_and_prekey_ids(self):
        self.request("POST", "/v1/devices", self.make_payload(device_id="dev-show"))
        status, body = self.request("GET", "/v1/devices/dev-show")
        self.assertEqual(status, 200)
        self.assertEqual(body["identity_key"], "IK")
        self.assertEqual(body["prekey_ids"], ["k1", "k2"])
        self.assertIn("registered_at", body)

    def test_prekey_ids_order_is_stable_across_requests(self):
        self.request("POST", "/v1/devices", self.make_payload(device_id="dev-stable"))
        first = self.request("GET", "/v1/devices/dev-stable")[1]["prekey_ids"]
        for _ in range(5):
            self.assertEqual(self.request("GET", "/v1/devices/dev-stable")[1]["prekey_ids"], first)

    def test_unknown_device_returns_404(self):
        status, _ = self.request("GET", "/v1/devices/does-not-exist")
        self.assertEqual(status, 404)

    def test_revoked_prekeys_are_excluded(self):
        self.request("POST", "/v1/devices", self.make_payload(device_id="dev-revoke"))
        device = self.store.get("dev-revoke")
        device.signed_prekeys[0].revoked = True
        _, body = self.request("GET", "/v1/devices/dev-revoke")
        self.assertEqual(body["prekey_ids"], ["k2"])

    def test_devices_do_not_affect_each_other(self):
        self.request("POST", "/v1/devices", self.make_payload(device_id="dev-a"))
        self.request(
            "POST",
            "/v1/devices",
            self.make_payload(
                device_id="dev-b",
                identity_key="IK-B",
                signed_prekeys=[{"key_id": "kb", "public_key": "PKB"}],
            ),
        )
        self.store.get("dev-a").signed_prekeys[0].revoked = True
        _, body_b = self.request("GET", "/v1/devices/dev-b")
        self.assertEqual(body_b["prekey_ids"], ["kb"])
        self.assertEqual(body_b["identity_key"], "IK-B")


class TestCli(ServerTestCase):
    def test_register_and_show_print_single_line_json(self):
        import contextlib
        import io

        argv = [
            "--server", self.base,
            "register",
            "--user-id", "cli-user",
            "--device-id", "cli-dev",
            "--identity-key", "IK-CLI",
            "--prekey", "k1:PK1",
        ]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            exit_code = cli_main(argv)
        self.assertEqual(exit_code, 0)
        line = out.getvalue().strip()
        self.assertNotIn("\n", line)
        body = json.loads(line)
        self.assertEqual(body["device_id"], "cli-dev")
        self.assertIn("registered_at", body)

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            exit_code = cli_main(["--server", self.base, "show", "cli-dev"])
        self.assertEqual(exit_code, 0)
        body = json.loads(out.getvalue().strip())
        self.assertEqual(body["identity_key"], "IK-CLI")
        self.assertEqual(body["prekey_ids"], ["k1"])


if __name__ == "__main__":
    unittest.main()
