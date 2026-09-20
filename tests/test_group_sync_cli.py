"""End-to-end subprocess tests for the group-sync CLI subcommands."""
import json
import subprocess
import sys
import threading
import unittest

from e2ee_backend.http_app import create_server
from e2ee_backend.models import Device


class GroupSyncCLITest(unittest.TestCase):
    def setUp(self) -> None:
        self.server, self.service = create_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        for device_id in ("creator", "alice", "bob", "carol"):
            self.service.store.add_device(Device("u", device_id, "ik"))
        self.service.create_group({
            "group_id": "g1", "creator_device_id": "creator",
            "member_device_ids": ["alice", "bob"]})
        session = self.service.create_group_session({
            "group_id": "g1", "initiator_device_id": "creator",
            "ephemeral_key": "epk"})
        self.sid = session["session_id"]
        for sequence in range(1, 4):
            self.service.post_message({
                "session_id": self.sid, "sender_device_id": "creator",
                "message_id": f"m{sequence}", "sequence": sequence,
                "nonce": f"n{sequence}", "ciphertext": "ct"})

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

    def test_sync_group_messages_paging(self) -> None:
        first = self._run("sync-group-messages", self.sid,
                          "--device-id", "alice", "--limit", "2")
        self.assertEqual(first.returncode, 0, first.stderr)
        body = self._json(first.stdout)
        self.assertEqual([m["sequence"] for m in body["messages"]], [1, 2])
        self.assertEqual(body["next_cursor"], 2)
        self.assertTrue(body["has_more"])
        # Default call resumes from the stored cursor.
        second = self._run("sync-group-messages", self.sid,
                           "--device-id", "alice")
        body = self._json(second.stdout)
        self.assertEqual([m["sequence"] for m in body["messages"]], [3])
        self.assertEqual(body["next_cursor"], 3)

    def test_sync_group_messages_explicit_after(self) -> None:
        result = self._run("sync-group-messages", self.sid,
                           "--device-id", "alice", "--after", "0",
                           "--limit", "1")
        self.assertEqual(result.returncode, 0, result.stderr)
        body = self._json(result.stdout)
        self.assertEqual([m["sequence"] for m in body["messages"]], [1])
        self.assertEqual(body["next_cursor"], 1)

    def test_sync_group_messages_unknown_session_stderr_nonzero(self) -> None:
        result = self._run("sync-group-messages", "missing",
                           "--device-id", "alice")
        self.assertEqual(result.returncode, 1)
        self.assertFalse(result.stdout.strip())
        self.assertEqual(self._json(result.stderr)["field"], "session_id")

    def test_sync_group_messages_non_member_stderr_nonzero(self) -> None:
        result = self._run("sync-group-messages", self.sid,
                           "--device-id", "carol")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(self._json(result.stderr)["field"], "device_id")

    def test_sync_checkpoint_forward_and_same(self) -> None:
        first = self._run("sync-checkpoint", self.sid,
                          "--device-id", "alice", "--cursor", "2")
        self.assertEqual(first.returncode, 0, first.stderr)
        body = self._json(first.stdout)
        self.assertEqual(body["cursor"], 2)
        self.assertEqual(set(body),
                         {"session_id", "device_id", "cursor", "updated_at"})
        timestamp = body["updated_at"]
        repeat = self._run("sync-checkpoint", self.sid,
                           "--device-id", "alice", "--cursor", "2")
        self.assertEqual(repeat.returncode, 0, repeat.stderr)
        self.assertEqual(self._json(repeat.stdout)["updated_at"], timestamp)

    def test_sync_checkpoint_backward_stderr_nonzero(self) -> None:
        self._run("sync-checkpoint", self.sid,
                  "--device-id", "alice", "--cursor", "2")
        result = self._run("sync-checkpoint", self.sid,
                           "--device-id", "alice", "--cursor", "1")
        self.assertEqual(result.returncode, 1)
        self.assertFalse(result.stdout.strip())
        self.assertEqual(self._json(result.stderr)["field"], "cursor")

    def test_sync_checkpoint_unknown_session(self) -> None:
        result = self._run("sync-checkpoint", "missing",
                           "--device-id", "alice", "--cursor", "0")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(self._json(result.stderr)["field"], "session_id")


if __name__ == "__main__":
    unittest.main()
