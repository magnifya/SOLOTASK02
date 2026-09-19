"""End-to-end tests for the register/show CLI subcommands."""
import base64
import json
import subprocess
import sys
import threading
import unittest

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import x25519

from e2ee_backend.http_app import create_server


def _raw_key_b64() -> str:
    key = x25519.X25519PrivateKey.generate().public_key()
    raw = key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return base64.b64encode(raw).decode()


class CLITest(unittest.TestCase):
    def setUp(self) -> None:
        self.server, _ = create_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _run(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "e2ee_backend", "--base-url", self.base_url,
             *arguments],
            capture_output=True, text=True, timeout=15)

    def test_register_prints_single_line_contract_json(self) -> None:
        result = self._run(
            "register", "--user-id", "u1", "--device-id", "d1",
            "--identity-key", _raw_key_b64(),
            "--prekey", f"k1:{_raw_key_b64()}",
            "--prekey", f"k2:{_raw_key_b64()}")
        self.assertEqual(result.returncode, 0, result.stderr)
        line = result.stdout.strip()
        self.assertEqual(line.count("\n"), 0)
        self.assertEqual(set(json.loads(line)), {"device_id", "registered_at"})

    def test_show_prints_single_line_contract_json(self) -> None:
        self._run("register", "--user-id", "u1", "--device-id", "d1",
                  "--identity-key", _raw_key_b64(),
                  "--prekey", f"k1:{_raw_key_b64()}")
        result = self._run("show", "d1")
        self.assertEqual(result.returncode, 0, result.stderr)
        line = result.stdout.strip()
        self.assertEqual(line.count("\n"), 0)
        body = json.loads(line)
        self.assertEqual(set(body),
                         {"identity_key", "prekey_ids", "registered_at"})
        self.assertEqual(body["prekey_ids"], ["k1"])

        repeated = self._run("show", "d1")
        self.assertEqual(repeated.stdout, result.stdout)

    def test_register_conflict_exit_code_and_json_error(self) -> None:
        args = ("register", "--user-id", "u1", "--device-id", "dup",
                "--identity-key", _raw_key_b64(),
                "--prekey", f"k1:{_raw_key_b64()}")
        self.assertEqual(self._run(*args).returncode, 0)
        result = self._run(*args)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(json.loads(result.stderr.strip())["field"], "device_id")

    def test_show_missing_device(self) -> None:
        result = self._run("show", "ghost")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(json.loads(result.stderr.strip())["field"], "device_id")

    def test_register_invalid_key_names_field(self) -> None:
        result = self._run("register", "--user-id", "u1", "--device-id", "bad",
                           "--identity-key", "garbage")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(json.loads(result.stderr.strip())["field"],
                         "identity_key")

    def test_malformed_prekey_spec_fails_locally(self) -> None:
        result = self._run("register", "--user-id", "u1", "--device-id", "bad2",
                           "--identity-key", _raw_key_b64(),
                           "--prekey", "no-separator")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stderr.strip())["field"],
                         "signed_prekeys")

    def _register_device(self, device_id: str = "d1") -> None:
        result = self._run(
            "register", "--user-id", "u1", "--device-id", device_id,
            "--identity-key", _raw_key_b64(),
            "--prekey", f"k1:{_raw_key_b64()}",
            "--prekey", f"k2:{_raw_key_b64()}")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_revoke_device_success_and_idempotent(self) -> None:
        self._register_device()
        result = self._run("revoke-device", "--device-id", "d1")
        self.assertEqual(result.returncode, 0, result.stderr)
        line = result.stdout.strip()
        self.assertEqual(line.count("\n"), 0)
        self.assertEqual(json.loads(line),
                         {"device_id": "d1", "revoked": True})
        again = self._run("revoke-device", "--device-id", "d1")
        self.assertEqual(again.returncode, 0, again.stderr)
        self.assertEqual(again.stdout, result.stdout)
        shown = self._run("show", "d1")
        self.assertEqual(json.loads(shown.stdout.strip())["prekey_ids"], [])

    def test_revoke_device_unknown_is_error_json(self) -> None:
        result = self._run("revoke-device", "--device-id", "ghost")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout.strip(), "")
        self.assertEqual(json.loads(result.stderr.strip())["field"],
                         "device_id")

    def test_revoke_prekey_success_and_idempotent(self) -> None:
        self._register_device()
        result = self._run("revoke-prekey", "--device-id", "d1",
                           "--key-id", "k1")
        self.assertEqual(result.returncode, 0, result.stderr)
        line = result.stdout.strip()
        self.assertEqual(line.count("\n"), 0)
        self.assertEqual(json.loads(line),
                         {"device_id": "d1", "key_id": "k1", "revoked": True})
        again = self._run("revoke-prekey", "--device-id", "d1",
                          "--key-id", "k1")
        self.assertEqual(again.returncode, 0, again.stderr)
        self.assertEqual(again.stdout, result.stdout)
        shown = self._run("show", "d1")
        self.assertEqual(json.loads(shown.stdout.strip())["prekey_ids"], ["k2"])

    def test_revoke_prekey_unknown_device_and_key(self) -> None:
        self._register_device()
        no_device = self._run("revoke-prekey", "--device-id", "ghost",
                              "--key-id", "k1")
        self.assertEqual(no_device.returncode, 1)
        self.assertEqual(json.loads(no_device.stderr.strip())["field"],
                         "device_id")
        no_key = self._run("revoke-prekey", "--device-id", "d1",
                           "--key-id", "nope")
        self.assertEqual(no_key.returncode, 1)
        self.assertEqual(json.loads(no_key.stderr.strip())["field"], "key_id")

    def test_connection_failure_reports_field_server_without_traceback(self) -> None:
        import sys as _sys
        dead = subprocess.run(
            [_sys.executable, "-m", "e2ee_backend",
             "--base-url", "http://127.0.0.1:1", "show", "d1"],
            capture_output=True, text=True, timeout=15)
        self.assertEqual(dead.returncode, 1)
        self.assertEqual(dead.stdout.strip(), "")
        body = json.loads(dead.stderr.strip())
        self.assertEqual(body["field"], "server")
        self.assertNotIn("Traceback", dead.stderr)

        for arguments in (["revoke-device", "--device-id", "d1"],
                          ["revoke-prekey", "--device-id", "d1",
                           "--key-id", "k1"],
                          ["register", "--user-id", "u1", "--device-id", "dx",
                           "--identity-key", _raw_key_b64()]):
            result = subprocess.run(
                [_sys.executable, "-m", "e2ee_backend",
                 "--base-url", "http://127.0.0.1:1", *arguments],
                capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 1, arguments)
            self.assertEqual(json.loads(result.stderr.strip())["field"],
                             "server", arguments)
            self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
