"""End-to-end subprocess tests for the group CLI subcommands."""
import base64
import json
import subprocess
import sys
import threading
import unittest

from cryptography.hazmat.primitives.asymmetric import x25519
from cryptography.hazmat.primitives import serialization

from e2ee_backend.http_app import create_server
from e2ee_backend.models import Device


def _valid_key() -> str:
    raw = x25519.X25519PrivateKey.generate().public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return base64.b64encode(raw).decode()


class GroupCLITest(unittest.TestCase):
    def setUp(self) -> None:
        self.server, self.service = create_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        for device_id in ("creator", "alice", "bob", "carol"):
            self.service.store.add_device(Device("u", device_id, "ik"))

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _run(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "e2ee_backend", "--base-url", self.base_url,
             *arguments],
            capture_output=True, text=True, timeout=15)

    def _json(self, stream: str) -> dict:
        line = stream.strip()
        self.assertEqual(line.count("\n"), 0)
        return json.loads(line)

    def test_group_create_show(self) -> None:
        result = self._run("group-create", "--group-id", "g1",
                           "--creator-device-id", "creator",
                           "--member-device-id", "alice",
                           "--member-device-id", "bob")
        self.assertEqual(result.returncode, 0, result.stderr)
        body = self._json(result.stdout)
        self.assertEqual(set(body),
                         {"group_id", "revision", "members", "created_at"})
        self.assertEqual(body["members"], ["creator", "alice", "bob"])

        shown = self._run("group-show", "g1")
        self.assertEqual(shown.returncode, 0, shown.stderr)
        self.assertEqual(self._json(shown.stdout), body)

    def test_group_create_conflict_goes_to_stderr_nonzero(self) -> None:
        args = ("group-create", "--group-id", "g1",
                "--creator-device-id", "creator",
                "--member-device-id", "alice")
        self.assertEqual(self._run(*args).returncode, 0)
        result = self._run(*args)
        self.assertEqual(result.returncode, 1)
        self.assertFalse(result.stdout.strip())
        self.assertEqual(self._json(result.stderr)["field"], "group_id")

    def test_group_show_unknown(self) -> None:
        result = self._run("group-show", "missing")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(self._json(result.stderr)["field"], "group_id")

    def test_group_add_member_201_and_200(self) -> None:
        self._run("group-create", "--group-id", "g1",
                  "--creator-device-id", "creator",
                  "--member-device-id", "alice")
        first = self._run("group-add-member", "--group-id", "g1",
                          "--actor-device-id", "creator",
                          "--device-id", "carol")
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(self._json(first.stdout)["revision"], 2)
        repeat = self._run("group-add-member", "--group-id", "g1",
                           "--actor-device-id", "creator",
                           "--device-id", "carol")
        self.assertEqual(repeat.returncode, 0, repeat.stderr)
        self.assertEqual(self._json(repeat.stdout)["revision"], 2)

    def test_group_add_member_unauthorized(self) -> None:
        self._run("group-create", "--group-id", "g1",
                  "--creator-device-id", "creator",
                  "--member-device-id", "alice")
        result = self._run("group-add-member", "--group-id", "g1",
                           "--actor-device-id", "alice",
                           "--device-id", "carol")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(self._json(result.stderr)["field"],
                         "actor_device_id")

    def test_group_remove_member(self) -> None:
        self._run("group-create", "--group-id", "g1",
                  "--creator-device-id", "creator",
                  "--member-device-id", "alice",
                  "--member-device-id", "bob")
        result = self._run("group-remove-member", "--group-id", "g1",
                           "--actor-device-id", "creator",
                           "--device-id", "alice")
        self.assertEqual(result.returncode, 0, result.stderr)
        body = self._json(result.stdout)
        self.assertNotIn("alice", body["members"])
        self.assertEqual(body["revision"], 2)
        # Idempotent repeat still succeeds, revision unchanged.
        repeat = self._run("group-remove-member", "--group-id", "g1",
                           "--actor-device-id", "creator",
                           "--device-id", "alice")
        self.assertEqual(repeat.returncode, 0, repeat.stderr)
        self.assertEqual(self._json(repeat.stdout)["revision"], 2)

    def test_group_session_create_show_freeze(self) -> None:
        self._run("group-create", "--group-id", "g1",
                  "--creator-device-id", "creator",
                  "--member-device-id", "alice")
        first = self._run("create-group-session", "--group-id", "g1",
                          "--initiator-device-id", "creator",
                          "--ephemeral-key", "ephemeral-key")
        self.assertEqual(first.returncode, 0, first.stderr)
        body = self._json(first.stdout)
        self.assertEqual(set(body), {"session_id", "group_id",
                                     "initiator_device_id", "ephemeral_key",
                                     "revision", "members", "created_at"})
        self.assertEqual(body["members"], ["creator", "alice"])
        session_id = body["session_id"]

        shown = self._run("show-group-session", session_id)
        self.assertEqual(shown.returncode, 0, shown.stderr)
        self.assertEqual(self._json(shown.stdout), body)

        # A roster change after creation does not touch the snapshot.
        self._run("group-add-member", "--group-id", "g1",
                  "--actor-device-id", "creator", "--device-id", "carol")
        frozen = self._json(self._run("show-group-session",
                                      session_id).stdout)
        self.assertEqual(frozen["members"], ["creator", "alice"])

    def test_group_session_repeated_submission_creates_new(self) -> None:
        self._run("group-create", "--group-id", "g1",
                  "--creator-device-id", "creator",
                  "--member-device-id", "alice")
        args = ("create-group-session", "--group-id", "g1",
                "--initiator-device-id", "creator", "--ephemeral-key", "epk")
        first = self._json(self._run(*args).stdout)
        second = self._json(self._run(*args).stdout)
        self.assertNotEqual(first["session_id"], second["session_id"])

    def test_show_group_session_unknown(self) -> None:
        result = self._run("show-group-session", "missing")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(self._json(result.stderr)["field"], "session_id")

    def test_rotate_group_session_create_and_replay(self) -> None:
        self._run("group-create", "--group-id", "g1",
                  "--creator-device-id", "creator",
                  "--member-device-id", "alice")
        session = self._json(self._run(
            "create-group-session", "--group-id", "g1",
            "--initiator-device-id", "creator",
            "--ephemeral-key", "ephemeral-key").stdout)
        first = self._run("rotate-group-session", session["session_id"],
                          "--rotation-id", "rot-1",
                          "--actor-device-id", "creator",
                          "--ephemeral-key", _valid_key(),
                          "--expected-revision", "1")
        self.assertEqual(first.returncode, 0, first.stderr)
        body = self._json(first.stdout)
        self.assertEqual(set(body), {
            "session_id", "group_id", "initiator_device_id",
            "ephemeral_key", "revision", "members", "created_at",
            "rotation_id", "predecessor_session_id"})
        self.assertEqual(body["rotation_id"], "rot-1")
        self.assertEqual(body["predecessor_session_id"],
                         session["session_id"])
        self.assertEqual(body["members"], ["creator", "alice"])

        # Replaying the same rotation id returns 200 (still exit 0) with the
        # original response.
        replay = self._run("rotate-group-session", session["session_id"],
                           "--rotation-id", "rot-1",
                           "--actor-device-id", "creator",
                           "--ephemeral-key", _valid_key(),
                           "--expected-revision", "1")
        self.assertEqual(replay.returncode, 0, replay.stderr)
        self.assertEqual(self._json(replay.stdout), body)

    def test_rotate_group_session_fork_conflict_stderr(self) -> None:
        self._run("group-create", "--group-id", "g1",
                  "--creator-device-id", "creator",
                  "--member-device-id", "alice")
        session = self._json(self._run(
            "create-group-session", "--group-id", "g1",
            "--initiator-device-id", "creator",
            "--ephemeral-key", "ephemeral-key").stdout)
        self.assertEqual(self._run(
            "rotate-group-session", session["session_id"],
            "--rotation-id", "rot-1", "--actor-device-id", "creator",
            "--ephemeral-key", _valid_key(), "--expected-revision",
            "1").returncode, 0)
        result = self._run(
            "rotate-group-session", session["session_id"],
            "--rotation-id", "rot-2", "--actor-device-id", "creator",
            "--ephemeral-key", _valid_key(), "--expected-revision",
            "1")
        self.assertEqual(result.returncode, 1)
        self.assertFalse(result.stdout.strip())
        self.assertEqual(self._json(result.stderr)["field"], "session_id")

    def test_rotate_group_session_unknown_predecessor(self) -> None:
        result = self._run(
            "rotate-group-session", "missing",
            "--rotation-id", "rot-1", "--actor-device-id", "creator",
            "--ephemeral-key", _valid_key(), "--expected-revision",
            "1")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(self._json(result.stderr)["field"], "session_id")


if __name__ == "__main__":
    unittest.main()
