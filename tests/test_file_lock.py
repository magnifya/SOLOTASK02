"""Tests for the process-level state-file lock coordinating concurrent serves.

Only ``serve --data-file`` / ``$E2EE_DATA_FILE`` is affected: a second live
process on the same state file must be refused with the existing startup
contract (stderr single-line JSON, ``field=data_file``, exit 1) without
touching the formal file; a lock released by a normal exit or a SIGKILL lets
a later process start and recover. In-memory mode takes no lock at all.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from http.client import HTTPConnection

from e2ee_backend.file_lock import StateFileLock, StateFileLocked

_MODULE = [sys.executable, "-m", "e2ee_backend"]
_PORT = 18411


class StateFileLockUnitTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_lock_lives_beside_the_state_file(self) -> None:
        lock = StateFileLock(os.path.join(self.directory, "state.json"))
        self.assertEqual(
            lock.path, os.path.join(self.directory, ".state.json.lock"))
        # The name is neither a .state-*.tmp nor a .state-*.bak leftover.
        self.assertFalse(os.path.basename(lock.path).startswith(".state-"))
        self.assertFalse(lock.path.endswith((".tmp", ".bak")))

    def test_second_acquire_fails_until_release(self) -> None:
        first = StateFileLock(os.path.join(self.directory, "state.json"))
        second = StateFileLock(os.path.join(self.directory, "state.json"))
        first.acquire()
        try:
            with self.assertRaises(StateFileLocked):
                second.acquire()
        finally:
            first.release()
        # Releasing (even via the context manager) frees the lock.
        with second:
            with self.assertRaises(StateFileLocked):
                StateFileLock(os.path.join(self.directory, "state.json")) \
                    .acquire()

    def test_distinct_state_files_get_distinct_locks(self) -> None:
        one = StateFileLock(os.path.join(self.directory, "one.json"))
        two = StateFileLock(os.path.join(self.directory, "two.json"))
        one.acquire()
        two.acquire()  # must not block or raise
        one.release()
        two.release()

    def test_release_is_idempotent_and_lock_file_is_never_truncated(self) -> None:
        path = os.path.join(self.directory, "state.json")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write('{"version": 1}')
        before = open(path, "rb").read()
        ino_before = os.stat(path).st_ino
        lock = StateFileLock(path)
        lock.acquire()
        lock.release()
        lock.release()
        # Acquiring the lock creates a sibling file and never touches the
        # state file: bytes and inode stay identical.
        self.assertEqual(open(path, "rb").read(), before)
        self.assertEqual(os.stat(path).st_ino, ino_before)
        self.assertTrue(os.path.exists(lock.path))


class ServeLockSubprocessTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.data_file = os.path.join(self.directory, "state.json")
        self.processes = []

    def tearDown(self) -> None:
        for process in self.processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        shutil.rmtree(self.directory, ignore_errors=True)

    def _serve(self, data_file=None, env=None, port=_PORT,
               capture_stderr=False) -> subprocess.Popen:
        environment = dict(os.environ)
        if env:
            environment.update(env)
        argv = _MODULE + ["serve", "--host", "127.0.0.1",
                          "--port", str(port)]
        if data_file is not None:
            argv += ["--data-file", data_file]
        process = subprocess.Popen(
            argv,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE if capture_stderr else subprocess.DEVNULL,
            env=environment)
        self.processes.append(process)
        return process

    def _wait_serving(self, process: subprocess.Popen, port: int) -> None:
        deadline = time.time() + 10
        while time.time() < deadline:
            if process.poll() is not None:
                self.fail(f"serve exited early with code {process.returncode}")
            try:
                connection = HTTPConnection("127.0.0.1", port, timeout=0.5)
                connection.request("GET", "/v1/devices/probe")
                connection.getresponse().read()
                connection.close()
                return
            except OSError:
                time.sleep(0.1)
        process.kill()
        self.fail("server did not start in time")

    def _wait_exit(self, process: subprocess.Popen) -> None:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            self.fail("rejected serve did not exit")

    def test_second_serve_is_refused_with_data_file_contract(self) -> None:
        first = self._serve(self.data_file)
        self._wait_serving(first, _PORT)
        self.assertTrue(
            os.path.exists(os.path.join(self.directory,
                                        ".state.json.lock")))

        # Put content into the formal state file so a refusal can be proven to
        # leave it byte- and inode-identical.
        register = subprocess.run(
            _MODULE + ["--base-url", f"http://127.0.0.1:{_PORT}",
                       "register", "--user-id", "u", "--device-id", "d1",
                       "--identity-key", "01" * 32,
                       "--prekey", "k1:" + "02" * 32],
            capture_output=True, text=True, timeout=15)
        self.assertEqual(register.returncode, 0, register.stderr)
        before = open(self.data_file, "rb").read()
        inode_before = os.stat(self.data_file).st_ino

        second = self._serve(self.data_file, capture_stderr=True)
        self._wait_exit(second)
        self.assertEqual(second.returncode, 1)
        error = second.stderr.read().strip()
        self.assertEqual(len(error.splitlines()), 1, error)
        body = json.loads(error)
        self.assertEqual(body.get("field"), "data_file")
        # The rejected process neither overwrote nor truncated anything and
        # left no staged temporary files behind.
        self.assertEqual(open(self.data_file, "rb").read(), before)
        self.assertEqual(os.stat(self.data_file).st_ino, inode_before)
        names = [name for name in os.listdir(self.directory)
                 if name.endswith((".tmp", ".bak"))]
        self.assertEqual(names, [])
        # The first process keeps serving.
        self.assertIsNone(first.poll())

    def test_lock_from_environment_is_equally_exclusive(self) -> None:
        first = self._serve(env={"E2EE_DATA_FILE": self.data_file})
        self._wait_serving(first, _PORT)
        second = self._serve(env={"E2EE_DATA_FILE": self.data_file},
                             capture_stderr=True)
        self._wait_exit(second)
        self.assertEqual(second.returncode, 1)
        self.assertEqual(json.loads(second.stderr.read().strip())["field"],
                         "data_file")

    def test_restart_after_sigterm_recovers_state(self) -> None:
        first = self._serve(self.data_file)
        self._wait_serving(first, _PORT)
        register = subprocess.run(
            _MODULE + ["--base-url", f"http://127.0.0.1:{_PORT}",
                       "register", "--user-id", "u", "--device-id", "d1",
                       "--identity-key", "01" * 32,
                       "--prekey", "k1:" + "02" * 32],
            capture_output=True, text=True, timeout=15)
        self.assertEqual(register.returncode, 0, register.stderr)
        first.terminate()
        first.wait(timeout=5)

        second = self._serve(self.data_file)
        self._wait_serving(second, _PORT)
        show = subprocess.run(
            _MODULE + ["--base-url", f"http://127.0.0.1:{_PORT}",
                       "show", "d1"],
            capture_output=True, text=True, timeout=15)
        self.assertEqual(show.returncode, 0, show.stderr)
        self.assertEqual(json.loads(show.stdout)["prekey_ids"], ["k1"])

    def test_lock_auto_releases_after_sigkill(self) -> None:
        first = self._serve(self.data_file)
        self._wait_serving(first, _PORT)
        first.kill()
        first.wait(timeout=5)
        # The stale lock file is still on disk, but the kernel dropped the
        # flock with the process, so a new holder starts and restores state.
        self.assertTrue(
            os.path.exists(os.path.join(self.directory,
                                        ".state.json.lock")))
        second = self._serve(self.data_file)
        self._wait_serving(second, _PORT)
        self.assertIsNone(second.poll())

    def test_two_in_memory_serves_coexist_without_any_lock(self) -> None:
        one = self._serve(port=_PORT)
        two = self._serve(port=_PORT + 1)
        self._wait_serving(one, _PORT)
        self._wait_serving(two, _PORT + 1)
        self.assertIsNone(one.poll())
        self.assertIsNone(two.poll())
        self.assertEqual(os.listdir(self.directory), [])


if __name__ == "__main__":
    unittest.main()
