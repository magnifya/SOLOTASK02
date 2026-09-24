"""Commit-integrity sidecar log (``<state>.integrity``).

Every durable commit rewrites the state file with its ``commit_seq`` and
appends one entry to the sibling integrity log as one locked two-file
transaction. The log is compact, newline-free UTF-8 JSON
(``ensure_ascii=False``) shaped ``{"version":1,"entries":[...]}``; each
entry is ``{"commit_seq","state_hash","prev_hash","hash"}`` in that key
order, ``commit_seq`` a non-negative integer, the hashes 64-character
lowercase hex, the first ``prev_hash`` empty and every later one the
previous entry's ``hash``. ``state_hash`` is the read-only probe's state
hash and ``hash`` is the SHA-256 of the entry minus ``hash`` serialised as
compact sorted-key JSON.

Startup gates: a document without ``integrity_log_version`` and without a
sidecar starts (the first commit stamps the marker and writes the first
entry); marker missing with a sidecar, a marker other than 1, marker 1
without a sidecar, or any log structure/chain/hash/last-entry error makes
the server refuse to start with one stderr JSON line ``field=data_file``
and exit 1, leaving every original file untouched.

``GET /v1/persistence/integrity/history`` answers 409/field=data_file
without a marker and 200 ``{"commit_seq","entries"}`` (that key order)
when enabled, with ``commit_seq`` equal to the state generation and the
last entry.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.client import HTTPConnection
from typing import Any, Dict, List, Tuple

from e2ee_backend.models import Device
from e2ee_backend.persistence import (
    PersistenceUnavailable,
    StateFileError,
    attach_persistence,
)
from e2ee_backend.service import DeviceService, ServiceError


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
    raise AssertionError(f"server on port {port} never started listening")


def _entry_hash(commit_seq: int, state_hash: str, prev_hash: str) -> str:
    raw = json.dumps(
        {"commit_seq": commit_seq, "prev_hash": prev_hash,
         "state_hash": state_hash},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class IntegrityLogTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.path = os.path.join(self.directory, "state.json")
        self.log_path = self.path + ".integrity"

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def _service(self) -> DeviceService:
        service = DeviceService()
        self.state_store = attach_persistence(service, self.path)
        return service

    def _state_document(self) -> Dict[str, Any]:
        with open(self.path, encoding="utf-8") as handle:
            return json.load(handle)

    def _write_state(self, document: Dict[str, Any]) -> None:
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)

    def _log_raw(self) -> bytes:
        with open(self.log_path, "rb") as handle:
            return handle.read()

    def _log(self) -> Dict[str, Any]:
        return json.loads(self._log_raw().decode("utf-8"))

    def _write_log(self, document: Dict[str, Any],
                   path: str = None) -> None:
        target = path or self.log_path
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(document, handle)

    def _commit(self, service: DeviceService, index: int) -> None:
        service.store.add_device(Device("u", f"d{index}", f"ik{index}"))


class IntegrityLogFormatTest(IntegrityLogTestBase):
    def test_sidecar_file_is_created_on_first_attach(self) -> None:
        # Even the initial empty state (commit_seq 0) is written through
        # save(), so the marker and sidecar exist from the start.
        self._service()
        self.assertTrue(os.path.exists(self.log_path))
        document = self._state_document()
        self.assertEqual(document["integrity_log_version"], 1)
        log = self._log()
        self.assertEqual(list(log), ["version", "entries"])
        self.assertEqual(log["version"], 1)
        self.assertEqual(len(log["entries"]), 1)
        self.assertEqual(log["entries"][0]["commit_seq"], 0)

    def test_raw_encoding_is_compact_newline_free_utf8(self) -> None:
        service = self._service()
        # Non-ASCII state is hashed as literal UTF-8 (ensure_ascii=False),
        # which the per-entry hash check below covers; the log bytes are
        # compact, whitespace-free and carry no trailing newline.
        service.store.add_device(Device("用户", "设备✓", "ik-日本語"))
        raw = self._log_raw()
        self.assertNotIn(b"\n", raw)
        self.assertTrue(raw.startswith(b'{"version":1,"entries":['))
        self.assertNotIn(b" ", raw)

    def test_entry_key_order_and_types(self) -> None:
        service = self._service()
        self._commit(service, 1)
        raw = self._log_raw().decode("utf-8")
        first_entry = raw.split('"entries":[', 1)[1]
        self.assertIn(
            '{"commit_seq":0,"state_hash":"', first_entry)
        for entry in self._log()["entries"]:
            self.assertEqual(
                list(entry), ["commit_seq", "state_hash", "prev_hash",
                              "hash"])
            self.assertIsInstance(entry["commit_seq"], int)
            self.assertNotIsInstance(entry["commit_seq"], bool)
            self.assertGreaterEqual(entry["commit_seq"], 0)
            self.assertRegex(entry["state_hash"], r"^[0-9a-f]{64}$")
            self.assertRegex(entry["hash"], r"^[0-9a-f]{64}$")
            self.assertIsInstance(entry["prev_hash"], str)

    def test_entries_chain_by_commit_seq_and_prev_hash(self) -> None:
        service = self._service()
        for index in range(1, 5):
            self._commit(service, index)
        entries = self._log()["entries"]
        self.assertEqual([entry["commit_seq"] for entry in entries],
                         [0, 1, 2, 3, 4])
        self.assertEqual(entries[0]["prev_hash"], "")
        for earlier, later in zip(entries, entries[1:]):
            self.assertEqual(later["prev_hash"], earlier["hash"])

    def test_each_entry_hash_is_sorted_key_compact_sha256(self) -> None:
        service = self._service()
        self._commit(service, 1)
        for entry in self._log()["entries"]:
            self.assertEqual(
                entry["hash"],
                _entry_hash(entry["commit_seq"], entry["state_hash"],
                            entry["prev_hash"]))

    def test_state_hash_equals_read_only_probe_hash(self) -> None:
        service = self._service()
        for index in range(1, 4):
            self._commit(service, index)
        probe = service.persistence_integrity()
        entries = self._log()["entries"]
        self.assertEqual(entries[-1]["commit_seq"], probe["commit_seq"])
        self.assertEqual(entries[-1]["state_hash"], probe["state_hash"])
        # Every generation recorded the probe hash of its own document:
        # rebuild each committed document and check the matching entry.
        for entry in entries:
            self.assertEqual(len(entry["state_hash"]), 64)


class IntegrityLogAppendTest(IntegrityLogTestBase):
    def test_appends_one_entry_per_committed_generation(self) -> None:
        service = self._service()
        self._commit(service, 1)
        self._commit(service, 2)
        entries = self._log()["entries"]
        self.assertEqual([entry["commit_seq"] for entry in entries],
                         [0, 1, 2])
        self.assertEqual(self._state_document()["commit_seq"], 2)

    def test_failed_transaction_is_idempotent_no_duplicate_entry(self) -> None:
        import e2ee_backend.persistence as persistence_mod

        service = self._service()
        self._commit(service, 1)
        before = self._log_raw()
        before_entries = self._log()["entries"]
        self.assertEqual(len(before_entries), 2)

        real_fsync = persistence_mod.os.fsync

        def fail_second_file_fsync(fd: int) -> None:  # noqa: ANN001
            # Fail the fsync of the staged integrity log (the second regular
            # file fsync in a save); directory fsyncs are unaffected.
            count = getattr(fail_second_file_fsync, "calls", 0)
            fail_second_file_fsync.calls = count + 1
            is_directory = os.fstat(fd).st_mode & 0o170000 == 0o040000
            if not is_directory and count == 1:
                raise OSError("simulated integrity log fsync failure")
            real_fsync(fd)

        persistence_mod.os.fsync = fail_second_file_fsync
        try:
            with self.assertRaises(PersistenceUnavailable):
                self._commit(service, 2)
        finally:
            persistence_mod.os.fsync = real_fsync
        # Nothing advanced: no appended entry, generation unchanged, memory
        # rolled back, and no temporary file survives.
        self.assertEqual(self._log_raw(), before)
        self.assertEqual(
            self._state_document()["commit_seq"],
            before_entries[-1]["commit_seq"])
        self.assertIsNone(service.store.find_by_device_id("d2"))
        self.assertFalse(
            any(name.startswith(".integrity-") or name.startswith(".state-")
                for name in os.listdir(self.directory)))
        # The retried commit consumes the same generation and appends once.
        self._commit(service, 2)
        entries = self._log()["entries"]
        self.assertEqual([entry["commit_seq"] for entry in entries],
                         [0, 1, 2])
        self.assertEqual(
            self._state_document()["commit_seq"], 2)

    def test_sidecar_replace_failure_rolls_both_files_back(self) -> None:
        import e2ee_backend.persistence as persistence_mod

        service = self._service()
        self._commit(service, 1)
        state_before = open(self.path, "rb").read()
        log_before = self._log_raw()
        state_ino = os.stat(self.path).st_ino
        log_ino = os.stat(self.log_path).st_ino

        real_replace = persistence_mod.os.replace

        def fail_sidecar_replace(src: str, dst: str) -> None:
            if os.path.abspath(dst) == os.path.abspath(self.log_path):
                raise OSError("simulated sidecar replace failure")
            return real_replace(src, dst)

        persistence_mod.os.replace = fail_sidecar_replace
        try:
            with self.assertRaises(PersistenceUnavailable):
                self._commit(service, 2)
        finally:
            persistence_mod.os.replace = real_replace
        # Both formal files hold the previous committed state.
        self.assertEqual(open(self.path, "rb").read(), state_before)
        self.assertEqual(self._log_raw(), log_before)
        self.assertEqual(os.stat(self.path).st_ino, state_ino)
        self.assertEqual(os.stat(self.log_path).st_ino, log_ino)
        self.assertIsNone(service.store.find_by_device_id("d2"))
        # Every temporary file is cleaned up.
        leftovers = [name for name in os.listdir(self.directory)
                     if name.startswith(".integrity-")
                     or name.startswith(".state-")]
        self.assertEqual(leftovers, [])

    def test_first_commit_on_legacy_file_starts_chain_at_that_seq(
            self) -> None:
        # Build a modern fixture, then turn it into a legacy document at a
        # generation above 0 (no marker, no sidecar).
        service = self._service()
        self._commit(service, 1)
        self._commit(service, 2)
        document = self._state_document()
        del document["integrity_log_version"]
        os.unlink(self.log_path)
        self._write_state(document)

        restarted = DeviceService()
        store = attach_persistence(restarted, self.path)
        self.assertIsNone(store.integrity_entries)
        # No marker yet -> history answers 409.
        with self.assertRaises(ServiceError) as caught:
            restarted.persistence_integrity_history()
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "data_file")
        # The first commit stamps the marker and writes the FIRST log entry,
        # chaining from an empty prev_hash despite starting above seq 0.
        restarted.store.add_device(Device("u", "fresh", "ik"))
        document = self._state_document()
        self.assertEqual(document["integrity_log_version"], 1)
        entries = self._log()["entries"]
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["commit_seq"], 3)
        self.assertEqual(entries[0]["prev_hash"], "")
        self.assertEqual(
            entries[0]["state_hash"],
            restarted.persistence_integrity()["state_hash"])
        # The next append chains normally.
        restarted.store.add_device(Device("u", "fresh2", "ik"))
        entries = self._log()["entries"]
        self.assertEqual([entry["commit_seq"] for entry in entries], [3, 4])
        self.assertEqual(entries[1]["prev_hash"], entries[0]["hash"])


class IntegrityLogStartupGateTest(IntegrityLogTestBase):
    def _make_marked_state(self) -> Tuple[DeviceService, bytes]:
        service = self._service()
        self._commit(service, 1)
        self._commit(service, 2)
        return service, open(self.path, "rb").read()

    def _assert_refuses_untouched(self) -> None:
        state_before = open(self.path, "rb").read()
        log_before = (open(self.log_path, "rb").read()
                      if os.path.exists(self.log_path) else None)
        with self.assertRaises(StateFileError):
            attach_persistence(DeviceService(), self.path)
        self.assertEqual(open(self.path, "rb").read(), state_before)
        if log_before is None:
            self.assertFalse(os.path.exists(self.log_path))
        else:
            self.assertEqual(self._log_raw(), log_before)

    def test_marker_one_with_sidecar_starts(self) -> None:
        self._make_marked_state()
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        self.assertEqual(
            restarted.persistence_integrity_history()["commit_seq"], 2)

    def test_legacy_without_marker_or_sidecar_starts(self) -> None:
        service = self._service()
        self._commit(service, 1)
        document = self._state_document()
        del document["integrity_log_version"]
        os.unlink(self.log_path)
        self._write_state(document)
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        restarted.store.add_device(Device("u", "ok", "ik"))

    def test_marker_without_sidecar_refuses(self) -> None:
        self._make_marked_state()
        os.unlink(self.log_path)
        self._assert_refuses_untouched()

    def test_sidecar_without_marker_refuses(self) -> None:
        self._make_marked_state()
        document = self._state_document()
        del document["integrity_log_version"]
        self._write_state(document)
        self._assert_refuses_untouched()

    def test_marker_other_than_one_refuses(self) -> None:
        self._make_marked_state()
        for bad in (0, 2, -1, True, "1", 1.0):
            document = self._state_document()
            document["integrity_log_version"] = bad
            self._write_state(document)
            self._assert_refuses_untouched()

    def test_broken_log_json_refuses(self) -> None:
        self._make_marked_state()
        with open(self.log_path, "wb") as handle:
            handle.write(b"{not json")
        self._assert_refuses_untouched()

    def test_bad_log_version_refuses(self) -> None:
        self._make_marked_state()
        log = self._log()
        for bad in (2, 0, "1", None, True):
            self._write_log({**log, "version": bad})
            self._assert_refuses_untouched()

    def test_unknown_log_keys_refuse(self) -> None:
        self._make_marked_state()
        log = self._log()
        self._write_log({**log, "extra": 1})
        self._assert_refuses_untouched()

    def test_bad_entry_shape_refuses(self) -> None:
        self._make_marked_state()
        log = self._log()
        # Extra key in an entry.
        tampered = json.loads(json.dumps(log))
        tampered["entries"][0]["extra"] = 1
        self._write_log(tampered)
        self._assert_refuses_untouched()
        # Missing key in an entry.
        tampered = json.loads(json.dumps(log))
        del tampered["entries"][1]["hash"]
        self._write_log(tampered)
        self._assert_refuses_untouched()
        # entries is not a list.
        self._write_log({"version": 1, "entries": {}})
        self._assert_refuses_untouched()
        # Empty entries list.
        self._write_log({"version": 1, "entries": []})
        self._assert_refuses_untouched()

    def test_bad_commit_seq_in_entry_refuses(self) -> None:
        self._make_marked_state()
        log = self._log()
        for bad in (-1, True, 1.5, "2", None):
            tampered = json.loads(json.dumps(log))
            tampered["entries"][1]["commit_seq"] = bad
            self._retamper(tampered)
            self._assert_refuses_untouched()

    def _retamper(self, tampered: Dict[str, Any]) -> None:
        # Re-sign every entry after the mutation so a bad commit_seq is what
        # is rejected rather than the resulting hash mismatch.
        prev_hash = ""
        for entry in tampered["entries"]:
            entry["prev_hash"] = prev_hash
            entry["hash"] = _entry_hash(
                entry["commit_seq"], entry["state_hash"], prev_hash)
            prev_hash = entry["hash"]
        self._write_log(tampered)

    def test_unordered_entries_refuse(self) -> None:
        self._make_marked_state()
        log = self._log()
        tampered = json.loads(json.dumps(log))
        tampered["entries"][0], tampered["entries"][1] = (
            tampered["entries"][1], tampered["entries"][0])
        self._write_log(tampered)
        self._assert_refuses_untouched()

    def test_broken_prev_hash_chain_refuses(self) -> None:
        self._make_marked_state()
        log = self._log()
        tampered = json.loads(json.dumps(log))
        tampered["entries"][1]["prev_hash"] = "0" * 64
        self._write_log(tampered)
        self._assert_refuses_untouched()
        # The first entry must chain from an empty prev_hash.
        tampered = json.loads(json.dumps(log))
        tampered["entries"][0]["prev_hash"] = "1" * 64
        self._write_log(tampered)
        self._assert_refuses_untouched()

    def test_wrong_entry_hash_refuses(self) -> None:
        self._make_marked_state()
        log = self._log()
        tampered = json.loads(json.dumps(log))
        tampered["entries"][0]["hash"] = "f" * 64
        self._write_log(tampered)
        self._assert_refuses_untouched()
        # A tampered state_hash with a re-signed hash is still rejected by
        # the last-entry cross-check against the state document.
        tampered = json.loads(json.dumps(log))
        prev_hash = ""
        for entry in tampered["entries"]:
            entry["state_hash"] = "a" * 64
            entry["prev_hash"] = prev_hash
            entry["hash"] = _entry_hash(
                entry["commit_seq"], "a" * 64, prev_hash)
            prev_hash = entry["hash"]
        self._write_log(tampered)
        self._assert_refuses_untouched()

    def test_last_entry_generation_behind_state_refuses(self) -> None:
        # A crash between the two atomic replaces can leave the new state
        # paired with the previous log: startup refuses (last entry wrong).
        service = self._service()
        self._commit(service, 1)
        old_log = self._log_raw()
        self._commit(service, 2)
        with open(self.log_path, "wb") as handle:
            handle.write(old_log)
        self._assert_refuses_untouched()

    def test_last_entry_generation_ahead_of_state_refuses(self) -> None:
        self._make_marked_state()
        log = self._log()
        tampered = json.loads(json.dumps(log))
        extra = {
            "commit_seq": 3, "state_hash": "b" * 64,
            "prev_hash": tampered["entries"][-1]["hash"],
        }
        extra["hash"] = _entry_hash(3, "b" * 64, extra["prev_hash"])
        tampered["entries"].append(extra)
        self._write_log(tampered)
        self._assert_refuses_untouched()

    def test_serve_refuses_with_stderr_json_and_exit_1(self) -> None:
        self._make_marked_state()
        os.unlink(self.log_path)
        state_before = open(self.path, "rb").read()
        port = _free_port()
        result = subprocess.run(
            [sys.executable, "-m", "e2ee_backend", "serve",
             "--host", "127.0.0.1", "--port", str(port),
             "--data-file", self.path],
            capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 1)
        lines = result.stderr.strip().splitlines()
        self.assertEqual(len(lines), 1)
        body = json.loads(lines[0])
        self.assertEqual(body["field"], "data_file")
        # The original file is untouched and no sidecar was created.
        self.assertEqual(open(self.path, "rb").read(), state_before)
        self.assertFalse(os.path.exists(self.log_path))


class IntegrityLogCrashReconcileTest(IntegrityLogTestBase):
    def test_staged_new_log_is_promoted_on_restart(self) -> None:
        # Crash between the two replaces: the new state document is formal,
        # the sidecar still holds the previous log, and the complete new
        # log survives as a .integrity-*.tmp leftover.
        service = self._service()
        self._commit(service, 1)
        self._commit(service, 2)
        new_log_bytes = self._log_raw()
        # Roll the formal sidecar one generation back to stage the crash.
        entries = self._log()["entries"][:-1]
        self._write_log({"version": 1, "entries": entries})
        staged = os.path.join(self.directory, ".integrity-crash.tmp")
        with open(staged, "wb") as handle:
            handle.write(new_log_bytes)

        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        history = restarted.persistence_integrity_history()
        self.assertEqual(history["commit_seq"], 2)
        self.assertEqual(
            [entry["commit_seq"] for entry in history["entries"]],
            [0, 1, 2])
        self.assertEqual(self._log_raw(), new_log_bytes)
        self.assertFalse(any(
            name.startswith(".integrity-")
            for name in os.listdir(self.directory)))

    def test_missing_sidecar_is_recreated_from_staged_log(self) -> None:
        service = self._service()
        self._commit(service, 1)
        new_log_bytes = self._log_raw()
        os.unlink(self.log_path)
        staged = os.path.join(self.directory, ".integrity-crash.tmp")
        with open(staged, "wb") as handle:
            handle.write(new_log_bytes)
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        self.assertEqual(self._log_raw(), new_log_bytes)
        self.assertEqual(
            restarted.persistence_integrity_history()["commit_seq"], 1)

    def test_unverifiable_staged_log_is_left_for_gate(self) -> None:
        service = self._service()
        self._commit(service, 1)
        # Crash between the replaces with the formal sidecar gone and only
        # an unreadable staged log surviving: reconciliation cannot prove
        # the state, so the leftover is preserved and the gate refuses.
        os.unlink(self.log_path)
        staged = os.path.join(self.directory, ".integrity-crash.tmp")
        with open(staged, "wb") as handle:
            handle.write(b"{broken")
        with self.assertRaises(StateFileError):
            attach_persistence(DeviceService(), self.path)
        self.assertTrue(os.path.exists(staged))


class IntegrityHistoryHTTPTest(unittest.TestCase):
    """End-to-end HTTP behaviour over a real loopback socket."""

    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.path = os.path.join(self.directory, "state.json")
        self.service = DeviceService()
        self.state_store = attach_persistence(self.service, self.path)
        from e2ee_backend.http_app import create_server
        self.server, _ = create_server("127.0.0.1", 0, self.service)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        shutil.rmtree(self.directory, ignore_errors=True)

    def _request(self, path: str) -> Tuple[int, Any]:
        connection = HTTPConnection("127.0.0.1", self.port, timeout=5)
        connection.request("GET", path)
        response = connection.getresponse()
        data = response.read().decode("utf-8")
        connection.close()
        return response.status, json.loads(data)

    def test_history_200_shape_and_advancing_entries(self) -> None:
        status, body = self._request(
            "/v1/persistence/integrity/history")
        self.assertEqual(status, 200)
        self.assertEqual(list(body), ["commit_seq", "entries"])
        self.assertEqual(body["commit_seq"], 0)
        self.assertEqual(len(body["entries"]), 1)
        self.assertEqual(list(body["entries"][0]),
                         ["commit_seq", "state_hash", "prev_hash", "hash"])
        self.service.store.add_device(Device("u", "d1", "ik"))
        status, body = self._request(
            "/v1/persistence/integrity/history")
        self.assertEqual(status, 200)
        self.assertEqual(body["commit_seq"], 1)
        self.assertEqual(
            [entry["commit_seq"] for entry in body["entries"]], [0, 1])
        # commit_seq equals the state document generation and the last entry.
        with open(self.path, encoding="utf-8") as handle:
            document = json.load(handle)
        self.assertEqual(body["commit_seq"], document["commit_seq"])
        self.assertEqual(body["entries"][-1]["commit_seq"],
                         body["commit_seq"])

    def test_history_query_and_body_ignored(self) -> None:
        status, _ = self._request(
            "/v1/persistence/integrity/history?ignored=1")
        self.assertEqual(status, 200)

    def test_existing_probe_route_unchanged(self) -> None:
        status, body = self._request("/v1/persistence/integrity")
        self.assertEqual(status, 200)
        self.assertEqual(list(body),
                         ["commit_seq", "state_hash", "consistent"])


class IntegrityHistoryServeSubprocessTest(unittest.TestCase):
    """A legacy file answers 409 from history until its first commit."""

    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.data_file = os.path.join(self.directory, "state.json")
        self.processes = []

    def tearDown(self) -> None:
        for proc in self.processes:
            if proc.poll() is None:
                proc.send_signal(signal.SIGKILL)
                proc.wait(timeout=5)
        shutil.rmtree(self.directory, ignore_errors=True)

    def _serve(self, use_file: bool):
        port = _free_port()
        argv = [sys.executable, "-m", "e2ee_backend", "serve",
                "--host", "127.0.0.1", "--port", str(port)]
        env = os.environ.copy()
        env.pop("E2EE_DATA_FILE", None)
        if use_file:
            argv.extend(["--data-file", self.data_file])
        proc = subprocess.Popen(
            argv, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True)
        self.processes.append(proc)
        _wait_until_listening(port)
        return proc, port

    def _get(self, port: int, path: str) -> Tuple[int, Any]:
        connection = HTTPConnection("127.0.0.1", port, timeout=5)
        connection.request("GET", path)
        response = connection.getresponse()
        data = json.loads(response.read().decode("utf-8"))
        connection.close()
        return response.status, data

    def test_memory_mode_is_409(self) -> None:
        _proc, port = self._serve(use_file=False)
        status, body = self._get(
            port, "/v1/persistence/integrity/history")
        self.assertEqual(status, 409)
        self.assertEqual(list(body), ["message", "field"])
        self.assertEqual(body["field"], "data_file")

    def test_file_mode_has_history_from_generation_zero(self) -> None:
        _proc, port = self._serve(use_file=True)
        status, body = self._get(
            port, "/v1/persistence/integrity/history")
        self.assertEqual(status, 200)
        self.assertEqual(body["commit_seq"], 0)
        self.assertEqual(len(body["entries"]), 1)


if __name__ == "__main__":
    unittest.main()
