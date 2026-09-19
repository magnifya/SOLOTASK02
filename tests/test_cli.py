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
                           "--identity-key", _raw_key_b64()],
                          ["create-session", "--initiator-device-id", "da",
                           "--recipient-device-id", "db", "--prekey-id", "pk",
                           "--ephemeral-key", _raw_key_b64()],
                          ["show-session", "sid"],
                          ["send-message", "--session-id", "sid",
                           "--sender-device-id", "da", "--message-id", "m1",
                           "--sequence", "1", "--nonce", "n",
                           "--ciphertext", "c"],
                          ["pull-messages", "sid", "--device-id", "da"]):
            result = subprocess.run(
                [_sys.executable, "-m", "e2ee_backend",
                 "--base-url", "http://127.0.0.1:1", *arguments],
                capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 1, arguments)
            self.assertEqual(json.loads(result.stderr.strip())["field"],
                             "server", arguments)
            self.assertNotIn("Traceback", result.stderr)


class CLISessionTest(unittest.TestCase):
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

    def _register(self, device_id: str) -> None:
        result = self._run(
            "register", "--user-id", f"u-{device_id}", "--device-id", device_id,
            "--identity-key", _raw_key_b64(),
            "--prekey", f"pk:{_raw_key_b64()}")
        self.assertEqual(result.returncode, 0, result.stderr)

    def _create_session(self) -> dict:
        result = self._run(
            "create-session", "--initiator-device-id", "da",
            "--recipient-device-id", "db", "--prekey-id", "pk",
            "--ephemeral-key", _raw_key_b64())
        self.assertEqual(result.returncode, 0, result.stderr)
        line = result.stdout.strip()
        self.assertEqual(line.count("\n"), 0)
        return json.loads(line)

    def test_create_session_prints_single_line_eight_field_json(self) -> None:
        self._register("da")
        self._register("db")
        body = self._create_session()
        self.assertEqual(set(body), {
            "session_id", "initiator_device_id", "recipient_device_id",
            "prekey_id", "ephemeral_key", "identity_key", "public_key",
            "created_at"})

    def test_show_session_prints_identical_snapshot(self) -> None:
        self._register("da")
        self._register("db")
        created = self._create_session()
        result = self._run("show-session", created["session_id"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout.strip()), created)
        repeated = self._run("show-session", created["session_id"])
        self.assertEqual(repeated.stdout, result.stdout)

    def test_show_session_unknown_is_error_json(self) -> None:
        result = self._run("show-session", "ghost")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout.strip(), "")
        self.assertEqual(json.loads(result.stderr.strip())["field"],
                         "session_id")

    def test_create_session_validation_error_names_field(self) -> None:
        self._register("da")
        self._register("db")
        result = self._run(
            "create-session", "--initiator-device-id", "da",
            "--recipient-device-id", "db", "--prekey-id", "pk",
            "--ephemeral-key", "garbage")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(json.loads(result.stderr.strip())["field"],
                         "ephemeral_key")

    def test_create_session_unknown_recipient_is_404_json(self) -> None:
        self._register("da")
        result = self._run(
            "create-session", "--initiator-device-id", "da",
            "--recipient-device-id", "ghost", "--prekey-id", "pk",
            "--ephemeral-key", _raw_key_b64())
        self.assertEqual(result.returncode, 1)
        self.assertEqual(json.loads(result.stderr.strip())["field"],
                         "recipient_device_id")

    def test_create_session_revoked_recipient_is_409_json(self) -> None:
        self._register("da")
        self._register("db")
        revoked = self._run("revoke-device", "--device-id", "db")
        self.assertEqual(revoked.returncode, 0, revoked.stderr)
        result = self._run(
            "create-session", "--initiator-device-id", "da",
            "--recipient-device-id", "db", "--prekey-id", "pk",
            "--ephemeral-key", _raw_key_b64())
        self.assertEqual(result.returncode, 1)
        body = json.loads(result.stderr.strip())
        self.assertEqual(body["field"], "recipient_device_id")


class CLIMessageTest(unittest.TestCase):
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

    def _run(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "e2ee_backend", "--base-url", self.base_url,
             *arguments],
            capture_output=True, text=True, timeout=15)

    def _register(self, device_id: str) -> None:
        result = self._run(
            "register", "--user-id", f"u-{device_id}", "--device-id", device_id,
            "--identity-key", _raw_key_b64(),
            "--prekey", f"pk:{_raw_key_b64()}")
        self.assertEqual(result.returncode, 0, result.stderr)

    def _session(self) -> str:
        self._register("da")
        self._register("db")
        result = self._run(
            "create-session", "--initiator-device-id", "da",
            "--recipient-device-id", "db", "--prekey-id", "pk",
            "--ephemeral-key", _raw_key_b64())
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout.strip())["session_id"]

    def test_send_and_pull_single_line_json(self) -> None:
        session_id = self._session()
        send = self._run(
            "send-message", "--session-id", session_id,
            "--sender-device-id", "da", "--message-id", "m1",
            "--sequence", "1", "--nonce", "bm9uY2U=",
            "--ciphertext", "Y3Q=")
        self.assertEqual(send.returncode, 0, send.stderr)
        line = send.stdout.strip()
        self.assertEqual(line.count("\n"), 0)
        body = json.loads(line)
        self.assertEqual(set(body), {
            "session_id", "sender_device_id", "message_id", "sequence",
            "nonce", "ciphertext", "created_at"})
        self.assertTrue(body["created_at"].endswith("+00:00"))

        pull = self._run("pull-messages", session_id, "--device-id", "db")
        self.assertEqual(pull.returncode, 0, pull.stderr)
        page = json.loads(pull.stdout.strip())
        self.assertEqual(set(page), {"messages", "next_after"})
        self.assertEqual([m["message_id"] for m in page["messages"]], ["m1"])
        self.assertEqual(page["next_after"], 1)
        self.assertEqual(page["messages"][0]["ciphertext"], "Y3Q=")

    def test_send_conflict_is_stderr_json_nonzero(self) -> None:
        session_id = self._session()
        first = self._run(
            "send-message", "--session-id", session_id,
            "--sender-device-id", "da", "--message-id", "dup",
            "--sequence", "1", "--nonce", "n", "--ciphertext", "c")
        self.assertEqual(first.returncode, 0, first.stderr)
        # Reusing the id with the next sequence is a duplicate-id conflict.
        result = self._run(
            "send-message", "--session-id", session_id,
            "--sender-device-id", "da", "--message-id", "dup",
            "--sequence", "2", "--nonce", "n2", "--ciphertext", "c2")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout.strip(), "")
        self.assertEqual(json.loads(result.stderr.strip())["field"],
                         "message_id")

    def test_send_non_integer_sequence_fails_locally(self) -> None:
        session_id = self._session()
        result = self._run(
            "send-message", "--session-id", session_id,
            "--sender-device-id", "da", "--message-id", "m1",
            "--sequence", "abc", "--nonce", "n", "--ciphertext", "c")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stderr.strip())["field"],
                         "sequence")

    def test_pull_unknown_session_is_stderr_json(self) -> None:
        result = self._run("pull-messages", "ghost", "--device-id", "da")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(json.loads(result.stderr.strip())["field"],
                         "session_id")

    def test_pull_paging_query_params(self) -> None:
        session_id = self._session()
        for sequence in range(1, 4):
            result = self._run(
                "send-message", "--session-id", session_id,
                "--sender-device-id", "da", "--message-id", f"m{sequence}",
                "--sequence", str(sequence),
                "--nonce", f"n{sequence}", "--ciphertext", f"c{sequence}")
            self.assertEqual(result.returncode, 0, result.stderr)
        result = self._run("pull-messages", session_id, "--device-id", "db",
                           "--after", "1", "--limit", "1")
        self.assertEqual(result.returncode, 0, result.stderr)
        page = json.loads(result.stdout.strip())
        self.assertEqual([m["sequence"] for m in page["messages"]], [2])
        self.assertEqual(page["next_after"], 2)


class CLIEnvelopeTest(unittest.TestCase):
    """The encrypt/decrypt commands run locally and never touch the server."""

    def _run(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "e2ee_backend", *arguments],
            capture_output=True, text=True, timeout=15)

    def test_encrypt_decrypt_roundtrip(self) -> None:
        import base64 as _b64
        import os as _os

        key = _b64.b64encode(_os.urandom(32)).decode()
        encrypted = self._run(
            "encrypt-message", "--session-id", "s1", "--key", key,
            "--plaintext", "héllo 世界")
        self.assertEqual(encrypted.returncode, 0, encrypted.stderr)
        envelope = json.loads(encrypted.stdout.strip())
        self.assertEqual(set(envelope),
                         {"session_id", "nonce", "ciphertext"})
        self.assertEqual(len(_b64.b64decode(envelope["nonce"])), 12)

        decrypted = self._run(
            "decrypt-message", "--session-id", "s1", "--key", key,
            "--nonce", envelope["nonce"],
            "--ciphertext", envelope["ciphertext"])
        self.assertEqual(decrypted.returncode, 0, decrypted.stderr)
        self.assertEqual(json.loads(decrypted.stdout.strip()),
                         {"session_id": "s1", "plaintext": "héllo 世界"})

    def test_decrypt_wrong_session_reports_ciphertext(self) -> None:
        import base64 as _b64
        import os as _os

        key = _b64.b64encode(_os.urandom(32)).decode()
        envelope = json.loads(self._run(
            "encrypt-message", "--session-id", "s1", "--key", key,
            "--plaintext", "secret").stdout.strip())
        result = self._run(
            "decrypt-message", "--session-id", "s2", "--key", key,
            "--nonce", envelope["nonce"],
            "--ciphertext", envelope["ciphertext"])
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout.strip(), "")
        self.assertEqual(json.loads(result.stderr.strip())["field"],
                         "ciphertext")

    def test_encrypt_bad_key_reports_field_key(self) -> None:
        result = self._run(
            "encrypt-message", "--session-id", "s1", "--key", "AAAA",
            "--plaintext", "x")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(json.loads(result.stderr.strip())["field"], "key")
        self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
