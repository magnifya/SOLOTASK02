"""Coordination tests for concurrent ``serve`` processes on one state file.

Only ``--data-file``/``$E2EE_DATA_FILE`` deployments take a process-exclusive
lock on a same-directory sibling lock file. These tests cover:

* the lock API directly (exclusion, release, never truncating the lock file);
* a second ``serve`` on the same file is refused with one stderr JSON line
  carrying ``field=data_file`` and exit code 1, without touching the formal
  file, while the holder keeps serving;
* flag and env-var lock the same path; in-memory servers never contend;
* normal shutdown (SIGINT) and ``SIGKILL`` both release the lock so a
  successor starts and fully recovers devices, groups, messages, nonces,
  delivery/ack state, sync cursors and checkpoints, and revocations.
"""
from __future__ import annotations

import base64
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from http.client import HTTPConnection

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import x25519

from e2ee_backend.locking import (
    StateFileLocked,
    StateFileLock,
    acquire_state_file_lock,
    lock_path_for,
)


def _raw_key_b64() -> str:
    key = x25519.X25519PrivateKey.generate().public_key()
    raw = key.public_bytes(serialization.Encoding.Raw,
                           serialization.PublicFormat.Raw)
    return base64.b64encode(raw).decode()


def _free_port() -> int:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


def _wait_until_listening(port: int, timeout: float = 10.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.05)
    raise AssertionError(f"server on port {port} did not start")


def _http(port: int, method: str, path: str, body: object = None):
    connection = HTTPConnection("127.0.0.1", port, timeout=5)
    payload = json.dumps(body) if body is not None else None
    headers = {"Content-Type": "application/json"} if payload else {}
    connection.request(method, path, body=payload, headers=headers)
    response = connection.getresponse()
    data = response.read().decode("utf-8")
    connection.close()
    return response.status, json.loads(data)


def _register(port: int, device_id: str):
    status, body = _http(port, "POST", "/v1/devices", {
        "user_id": "u1",
        "device_id": device_id,
        "identity_key": _raw_key_b64(),
        "signed_prekeys": [{"key_id": "k1", "public_key": _raw_key_b64()}],
    })
    assert status == 201, (status, body)
    return body


class StateFileLockApiTest(unittest.TestCase):
    """Direct tests for the lock primitive (no server processes)."""

    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.state_path = os.path.join(self.directory, "state.json")

    def test_lock_file_is_a_same_directory_sibling(self) -> None:
        self.assertEqual(lock_path_for(self.state_path),
                         os.path.join(self.directory, "state.json.lock"))
        lock = acquire_state_file_lock(self.state_path)
        try:
            self.assertTrue(os.path.isfile(lock.lock_path))
            self.assertEqual(os.path.dirname(os.path.abspath(lock.lock_path)),
                             os.path.dirname(os.path.abspath(self.state_path)))
            # The empty lock file is never written to.
            self.assertEqual(open(lock.lock_path, "rb").read(), b"")
        finally:
            lock.release()

    def test_second_acquire_in_same_process_is_refused(self) -> None:
        lock = acquire_state_file_lock(self.state_path)
        try:
            with self.assertRaises(StateFileLocked):
                acquire_state_file_lock(self.state_path)
        finally:
            lock.release()
        # Once released the same path is lockable again.
        with acquire_state_file_lock(self.state_path):
            pass

    def test_release_is_idempotent_and_context_manager_unlocks(self) -> None:
        lock = acquire_state_file_lock(self.state_path)
        lock.release()
        lock.release()
        with StateFileLock(self.state_path) as other:
            self.assertIsNotNone(other._fd)

    def test_existing_lock_file_is_opened_not_truncated(self) -> None:
        marker = b"do-not-touch"
        lock_path = lock_path_for(self.state_path)
        with open(lock_path, "wb") as handle:
            handle.write(marker)
        with acquire_state_file_lock(self.state_path):
            self.assertEqual(open(lock_path, "rb").read(), marker)
        # Released without altering the pre-existing bytes.
        self.assertEqual(open(lock_path, "rb").read(), marker)

    def test_distinct_state_files_do_not_contend(self) -> None:
        first = acquire_state_file_lock(self.state_path)
        try:
            other = acquire_state_file_lock(
                os.path.join(self.directory, "other-state.json"))
            other.release()
        finally:
            first.release()

    def test_unwritable_directory_raises_oserror_not_locked(self) -> None:
        # A regular file where the lock directory should be: makedirs/open fails
        # with a plain OSError, never a false "already locked".
        blocker = os.path.join(self.directory, "a-file")
        with open(blocker, "wb"):
            pass
        with self.assertRaises(OSError):
            acquire_state_file_lock(
                os.path.join(blocker, "nested", "state.json"))


class ServeLockingTest(unittest.TestCase):
    """Subprocess tests for the ``serve`` process-level lock."""

    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.data_file = os.path.join(self.directory, "state.json")
        self.processes = []

    def tearDown(self) -> None:
        for proc in self.processes:
            if proc.poll() is None:
                proc.send_signal(signal.SIGKILL)
                proc.wait(timeout=5)

    def _serve(self, data_file=None, env_data_file=False, port=None):
        port = port or _free_port()
        argv = [sys.executable, "-m", "e2ee_backend", "serve",
                "--host", "127.0.0.1", "--port", str(port)]
        env = os.environ.copy()
        if env_data_file:
            env["E2EE_DATA_FILE"] = data_file
        else:
            env.pop("E2EE_DATA_FILE", None)
            if data_file is not None:
                argv.extend(["--data-file", data_file])
        proc = subprocess.Popen(
            argv, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True)
        self.processes.append(proc)
        return proc, port

    def _start_ready(self, **kwargs):
        proc, port = self._serve(**kwargs)
        try:
            _wait_until_listening(port)
        except AssertionError:
            proc.wait(timeout=5)
            raise AssertionError(
                f"serve failed to start: {proc.stderr.read()}")
        return proc, port

    def _stop_with_sigint(self, proc) -> None:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=10)

    def test_second_serve_same_file_is_refused_with_contract(self) -> None:
        proc_a, port_a = self._start_ready(data_file=self.data_file)
        # The holder serves normally and persists into the formal file.
        _register(port_a, "alice")
        before = open(self.data_file, "rb").read()
        before_ino = os.stat(self.data_file).st_ino
        lock_path = lock_path_for(self.data_file)
        self.assertTrue(os.path.isfile(lock_path))

        proc_b, port_b = self._serve(data_file=self.data_file)
        result = proc_b.wait(timeout=10)
        self.assertEqual(result, 1)
        self.assertEqual(proc_b.stdout.read(), "")
        line = proc_b.stderr.read().strip()
        self.assertEqual(line.count("\n"), 0)
        self.assertEqual(json.loads(line)["field"], "data_file")

        # Refusal must not overwrite, truncate, or replace the formal file.
        self.assertEqual(open(self.data_file, "rb").read(), before)
        self.assertEqual(os.stat(self.data_file).st_ino, before_ino)

        # The holder is still alive and still the owner of the state.
        status, body = _http(port_a, "GET", "/v1/devices/alice")
        self.assertEqual(status, 200)
        self.assertIn("registered_at", body)

        self._stop_with_sigint(proc_a)

    def test_flag_and_env_var_share_one_lock(self) -> None:
        proc_a, _ = self._start_ready(env_data_file=True,
                                      data_file=self.data_file)
        # --data-file against the same path must see the env-started holder.
        proc_b, _ = self._serve(data_file=self.data_file)
        self.assertEqual(proc_b.wait(timeout=10), 1)
        self.assertEqual(json.loads(proc_b.stderr.read().strip())["field"],
                         "data_file")
        # A second env-var process is refused too.
        proc_c, _ = self._serve(env_data_file=True, data_file=self.data_file)
        self.assertEqual(proc_c.wait(timeout=10), 1)
        self.assertEqual(json.loads(proc_c.stderr.read().strip())["field"],
                         "data_file")
        self._stop_with_sigint(proc_a)

    def test_two_in_memory_servers_coexist(self) -> None:
        # No flag and no env var: no lock is taken and nothing is on disk.
        proc_a, port_a = self._start_ready()
        proc_b, port_b = self._start_ready()
        _register(port_a, "alice")
        _register(port_b, "alice")
        self.assertFalse(os.path.exists(self.data_file))
        self.assertFalse(os.path.exists(lock_path_for(self.data_file)))
        self._stop_with_sigint(proc_a)
        self._stop_with_sigint(proc_b)

    def test_lock_released_after_normal_shutdown(self) -> None:
        proc_a, port_a = self._start_ready(data_file=self.data_file)
        _register(port_a, "alice")
        self._stop_with_sigint(proc_a)
        # A successor starts on the same file immediately (lock auto-released)
        # and sees the committed state.
        proc_b, port_b = self._start_ready(data_file=self.data_file)
        try:
            status, body = _http(port_b, "GET", "/v1/devices/alice")
            self.assertEqual(status, 200)
            self.assertIn("registered_at", body)
        finally:
            self._stop_with_sigint(proc_b)

    def test_lock_released_after_sigkill_and_state_recoverable(self) -> None:
        proc_a, port_a = self._start_ready(data_file=self.data_file)
        _register(port_a, "alice")
        registered_at = _http(port_a, "GET", "/v1/devices/alice")[1][
            "registered_at"]
        proc_a.send_signal(signal.SIGKILL)
        proc_a.wait(timeout=5)

        proc_b, port_b = self._start_ready(data_file=self.data_file)
        try:
            status, body = _http(port_b, "GET", "/v1/devices/alice")
            self.assertEqual(status, 200)
            self.assertEqual(body["registered_at"], registered_at)
        finally:
            self._stop_with_sigint(proc_b)

    def test_successor_recovers_full_entity_state_after_kill(self) -> None:
        proc_a, port = self._start_ready(data_file=self.data_file)

        # Devices, a group and a frozen group session.
        for device_id in ("creator", "alice", "bob", "dave"):
            _register(port, device_id)
        status, group = _http(port, "POST", "/v1/groups", {
            "group_id": "g1", "creator_device_id": "creator",
            "member_device_ids": ["alice", "bob"]})
        self.assertEqual(status, 201)
        status, group_session = _http(port, "POST", "/v1/group-sessions", {
            "group_id": "g1", "initiator_device_id": "creator",
            "ephemeral_key": "epk-g1"})
        self.assertEqual(status, 201)
        gsid = group_session["session_id"]

        # Three group messages; alice syncs the first page (cursor -> 2),
        # bob's cursor is check-pointed at 2.
        for sequence in (1, 2, 3):
            status, _ = _http(port, "POST", "/v1/messages", {
                "session_id": gsid, "sender_device_id": "creator",
                "message_id": f"g{sequence}", "sequence": sequence,
                "nonce": base64.b64encode(f"g-nonce-{sequence}".encode())
                .decode(),
                "ciphertext": base64.b64encode(b"g-ct").decode()})
            self.assertEqual(status, 201)
        status, alice_page = _http(
            port, "GET", f"/v1/group-sessions/{gsid}/sync"
                          f"?device_id=alice&limit=2")
        self.assertEqual(status, 200)
        self.assertEqual(alice_page["next_cursor"], 2)
        status, checkpoint = _http(
            port, "POST", f"/v1/group-sessions/{gsid}/sync/checkpoint",
            {"device_id": "bob", "cursor": 2})
        self.assertEqual(status, 201)
        checkpoint_updated_at = checkpoint["updated_at"]

        # A 1:1 session with a message, a retry, and an ack (delivery state),
        # plus a consumed nonce and a separate revoked device.
        status, session = _http(port, "POST", "/v1/sessions", {
            "initiator_device_id": "creator",
            "recipient_device_id": "bob",
            "prekey_id": "k1",
            "ephemeral_key": _raw_key_b64()})
        self.assertEqual(status, 201)
        sid = session["session_id"]
        m1_nonce = base64.b64encode(b"p2p-nonce-1").decode()
        status, message = _http(port, "POST", "/v1/messages", {
            "session_id": sid, "sender_device_id": "creator",
            "message_id": "p1", "sequence": 1, "nonce": m1_nonce,
            "ciphertext": base64.b64encode(b"p2p-ct").decode()})
        self.assertEqual(status, 201)
        status, retry = _http(
            port, "POST", f"/v1/messages/{sid}/retry/p1",
            {"device_id": "bob", "attempt_id": "a1"})
        self.assertEqual(status, 201)
        self.assertEqual(retry["attempts"], 1)
        status, ack = _http(port, "POST", f"/v1/messages/{sid}/acks", {
            "device_id": "bob", "message_id": "p1", "sequence": 1})
        self.assertEqual(status, 201)
        self.assertEqual(ack["status"], "acked")
        status, _ = _http(port, "POST", "/v1/devices/dave/revoke")
        self.assertEqual(status, 200)

        # Hard kill: no graceful shutdown, the kernel must drop the flock.
        proc_a.send_signal(signal.SIGKILL)
        proc_a.wait(timeout=5)

        proc_b, port = self._start_ready(data_file=self.data_file)
        try:
            # Group + frozen group session.
            status, body = _http(port, "GET", "/v1/groups/g1")
            self.assertEqual(status, 200)
            self.assertEqual(body["members"], ["creator", "alice", "bob"])
            status, body = _http(port, "GET",
                                 f"/v1/group-sessions/{gsid}")
            self.assertEqual(status, 200)
            self.assertEqual(body["members"], ["creator", "alice", "bob"])
            self.assertEqual(body["revision"], group["revision"])

            # alice's stored cursor survived: she resumes at seq 3, not 1.
            status, body = _http(
                port, "GET",
                f"/v1/group-sessions/{gsid}/sync?device_id=alice")
            self.assertEqual(status, 200)
            self.assertEqual([m["sequence"] for m in body["messages"]], [3])
            self.assertEqual(body["next_cursor"], 3)

            # bob's checkpoint cursor AND its updated_at survived: repeating
            # the same checkpoint is the 200 idempotent no-op.
            status, body = _http(
                port, "POST", f"/v1/group-sessions/{gsid}/sync/checkpoint",
                {"device_id": "bob", "cursor": 2})
            self.assertEqual(status, 200)
            self.assertEqual(body["updated_at"], checkpoint_updated_at)
            status, body = _http(
                port, "GET",
                f"/v1/group-sessions/{gsid}/sync?device_id=bob")
            self.assertEqual(status, 200)
            self.assertEqual([m["sequence"] for m in body["messages"]], [3])

            # Message history + delivery: ack and retry dedup persisted.
            status, body = _http(
                port, "GET",
                f"/v1/messages/{sid}/status/p1?device_id=bob")
            self.assertEqual(status, 200)
            self.assertEqual(body["status"], "acked")
            self.assertEqual(body["attempts"], 1)
            status, body = _http(
                port, "POST", f"/v1/messages/{sid}/retry/p1",
                {"device_id": "bob", "attempt_id": "a1"})
            self.assertEqual(status, 200)
            self.assertEqual(body["attempts"], 1)
            status, body = _http(
                port, "POST", f"/v1/messages/{sid}/retry/p1",
                {"device_id": "bob", "attempt_id": "a2"})
            self.assertEqual(status, 200)
            self.assertEqual(body["attempts"], 2)

            # Nonce set persisted: the old nonce is still consumed (409),
            # while the stream resumes at sequence 2 and new writes commit.
            status, body = _http(port, "POST", "/v1/messages", {
                "session_id": sid, "sender_device_id": "creator",
                "message_id": "p2", "sequence": 2, "nonce": m1_nonce,
                "ciphertext": base64.b64encode(b"p2p-ct").decode()})
            self.assertEqual(status, 409)
            self.assertEqual(body["field"], "nonce")
            status, body = _http(port, "POST", "/v1/messages", {
                "session_id": sid, "sender_device_id": "creator",
                "message_id": "p2", "sequence": 2,
                "nonce": base64.b64encode(b"p2p-nonce-2").decode(),
                "ciphertext": base64.b64encode(b"p2p-ct").decode()})
            self.assertEqual(status, 201)
            self.assertEqual(body["sequence"], 2)

            # Revocation persisted (the public view omits the flag, so check
            # the version=1 document directly; an advisory flock does not
            # block plain readers).
            with open(self.data_file, encoding="utf-8") as handle:
                document = json.load(handle)
            self.assertEqual(document["version"], 1)
            revoked = {d["device_id"]: d["revoked"]
                       for d in document["devices"]}
            self.assertTrue(revoked["dave"])
            self.assertFalse(revoked["creator"])
            cursors = {(c["session_id"], c["device_id"]): c["cursor"]
                       for c in document["group_sync_cursors"]}
            self.assertEqual(cursors.get((gsid, "alice")), 3)
        finally:
            self._stop_with_sigint(proc_b)

        # Clean shutdown leaves no transaction temporaries behind, only the
        # formal file and its sibling lock file.
        leftovers = [name for name in os.listdir(self.directory)
                     if name.endswith(".tmp") or name.endswith(".bak")]
        self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()
