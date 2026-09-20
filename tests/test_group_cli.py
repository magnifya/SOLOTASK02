"""End-to-end subprocess tests for the group CLI subcommands."""
import json
import subprocess
import sys
import threading
import unittest

from e2ee_backend.http_app import create_server
from e2ee_backend.models import Device


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

    def _group_session_with_messages(self, count=3) -> str:
        self._run("group-create", "--group-id", "g1",
                  "--creator-device-id", "creator",
                  "--member-device-id", "alice")
        created = self._run("create-group-session", "--group-id", "g1",
                            "--initiator-device-id", "creator",
                            "--ephemeral-key", "epk")
        session_id = self._json(created.stdout)["session_id"]
        for sequence in range(1, count + 1):
            self._run("send-message", "--session-id", session_id,
                      "--sender-device-id", "creator",
                      "--message-id", f"m{sequence}",
                      "--sequence", str(sequence),
                      "--nonce", f"n{sequence}", "--ciphertext", "ct")
        return session_id

    def test_sync_group_messages_paging_and_resume(self) -> None:
        sid = self._group_session_with_messages()
        first = self._run("sync-group-messages", sid,
                          "--device-id", "alice", "--limit", "2")
        self.assertEqual(first.returncode, 0, first.stderr)
        body = self._json(first.stdout)
        self.assertEqual([m["sequence"] for m in body["messages"]], [1, 2])
        self.assertEqual(body["next_cursor"], 2)
        self.assertTrue(body["has_more"])
        # Omitted --after resumes from the device's stored cursor.
        second = self._run("sync-group-messages", sid, "--device-id", "alice")
        body = self._json(second.stdout)
        self.assertEqual([m["sequence"] for m in body["messages"]], [3])
        self.assertEqual(body["next_cursor"], 3)
        self.assertFalse(body["has_more"])
        # Explicit --after is a one-shot read and does not move the cursor.
        explicit = self._run("sync-group-messages", sid,
                             "--device-id", "alice", "--after", "1")
        body = self._json(explicit.stdout)
        self.assertEqual([m["sequence"] for m in body["messages"]], [2, 3])
        self.assertEqual(body["next_cursor"], 3)
        # Stored cursor is still 3, so an omitted-after read is now empty.
        again = self._run("sync-group-messages", sid, "--device-id", "alice")
        body = self._json(again.stdout)
        self.assertEqual(body["messages"], [])
        self.assertEqual(body["next_cursor"], 3)

    def test_sync_group_messages_unknown_session(self) -> None:
        result = self._run("sync-group-messages", "missing",
                           "--device-id", "alice")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(self._json(result.stderr)["field"], "session_id")

    def test_sync_checkpoint_forward_same_backward(self) -> None:
        sid = self._group_session_with_messages()
        forward = self._run("sync-checkpoint", sid,
                            "--device-id", "creator", "--cursor", "3")
        self.assertEqual(forward.returncode, 0, forward.stderr)
        body = self._json(forward.stdout)
        self.assertEqual(body["cursor"], 3)
        self.assertIn("updated_at", body)
        timestamp = body["updated_at"]

        same = self._run("sync-checkpoint", sid,
                         "--device-id", "creator", "--cursor", "3")
        self.assertEqual(same.returncode, 0, same.stderr)
        self.assertEqual(self._json(same.stdout)["updated_at"], timestamp)

        backward = self._run("sync-checkpoint", sid,
                             "--device-id", "creator", "--cursor", "2")
        self.assertEqual(backward.returncode, 1)
        self.assertFalse(backward.stdout.strip())
        self.assertEqual(self._json(backward.stderr)["field"], "cursor")

    def test_sync_checkpoint_out_of_range_is_400(self) -> None:
        sid = self._group_session_with_messages()
        result = self._run("sync-checkpoint", sid,
                           "--device-id", "creator", "--cursor", "99")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(self._json(result.stderr)["field"], "cursor")

    def test_sync_checkpoint_non_member_409(self) -> None:
        sid = self._group_session_with_messages()
        result = self._run("sync-checkpoint", sid,
                           "--device-id", "carol", "--cursor", "0")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(self._json(result.stderr)["field"], "device_id")


if __name__ == "__main__":
    unittest.main()
