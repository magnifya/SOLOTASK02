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


if __name__ == "__main__":
    unittest.main()
