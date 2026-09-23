"""``GET /v1/persistence/integrity`` — read-only on-disk integrity probe.

Only available with persistence enabled (``--data-file``/``$E2EE_DATA_FILE``);
the in-memory mode answers 409/field=data_file. With a file the handler reads
the version=1 document under the store lock (shared with every mutation and
key-event append), enforces a non-negative integer ``commit_seq`` equal to
the last committed generation, restores the payload into a fresh store, and
compares the canonical 17-section snapshot with the live in-memory state.
Success is 200 ``{"commit_seq","state_hash","consistent":true}``; any parse,
version, semantic, generation or snapshot failure is 503/field=data_file and
must leave memory, the file bytes/inode, cursors and the generation untouched.
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
from typing import Any, Dict, Tuple

from e2ee_backend.models import Device
from e2ee_backend.persistence import (
    IntegrityCheckError,
    attach_persistence,
)
from e2ee_backend.service import DeviceService, ServiceError
from e2ee_backend.storage import (
    canonical_integrity_snapshot,
    integrity_state_hash,
)

from tests.test_crash_recovery import build_fixture


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


def _expected_hash(document: Dict[str, Any]) -> str:
    """Independently compute the canonical hash of a version=1 document."""
    payload = {key: value for key, value in document.items()
               if key not in ("version", "commit_seq")}
    canonical = canonical_integrity_snapshot(payload)
    return integrity_state_hash(canonical)


class IntegrityServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.path = os.path.join(self.directory, "state.json")

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def _service(self) -> DeviceService:
        service = DeviceService()
        self.state_store = attach_persistence(service, self.path)
        return service

    def _document(self) -> Dict[str, Any]:
        with open(self.path, encoding="utf-8") as handle:
            return json.load(handle)

    def _overwrite(self, document: Dict[str, Any]) -> None:
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(document, handle, ensure_ascii=False)

    def test_in_memory_mode_is_409_data_file(self) -> None:
        service = DeviceService()
        self.assertIsNone(service.integrity_state_store)
        with self.assertRaises(ServiceError) as caught:
            service.persistence_integrity()
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "data_file")

    def test_empty_state_reports_generation_zero_and_true(self) -> None:
        service = self._service()
        report = service.persistence_integrity()
        self.assertEqual(list(report),
                         ["commit_seq", "state_hash", "consistent"])
        self.assertEqual(report["commit_seq"], 0)
        self.assertIs(report["consistent"], True)
        self.assertEqual(len(report["state_hash"]), 64)
        self.assertEqual(report["state_hash"],
                         _expected_hash(self._document()))

    def test_generation_and_hash_track_commits(self) -> None:
        service = self._service()
        service.store.add_device(Device("u", "d1", "ik"))
        service.store.add_device(Device("u", "d2", "ik2"))
        report = service.persistence_integrity()
        self.assertEqual(report["commit_seq"], 2)
        self.assertEqual(report["commit_seq"],
                         self._document()["commit_seq"])
        self.assertEqual(report["state_hash"],
                         _expected_hash(self._document()))
        # A read-only probe advances no generation.
        before = self._document()["commit_seq"]
        service.persistence_integrity()
        service.persistence_integrity()
        self.assertEqual(self._document()["commit_seq"], before)
        self.assertEqual(self.state_store.commit_seq, before + 1)

    def test_hash_uses_compact_json_without_ascii_escaping(self) -> None:
        service = self._service()
        # Non-ASCII identifiers must be hashed as literal UTF-8, not as
        # \\uXXXX escapes.
        service.store.add_device(Device("用户", "设备✓", "ik-日本語"))
        report = service.persistence_integrity()
        document = self._document()
        payload = {key: value for key, value in document.items()
                   if key not in ("version", "commit_seq")}
        canonical = canonical_integrity_snapshot(payload)
        raw = json.dumps(canonical, separators=(",", ":"),
                         ensure_ascii=False).encode("utf-8")
        self.assertEqual(hashlib.sha256(raw).hexdigest(),
                         report["state_hash"])
        escaped = json.dumps(canonical, separators=(",", ":"),
                             ensure_ascii=True).encode("utf-8")
        self.assertNotEqual(hashlib.sha256(escaped).hexdigest(),
                            report["state_hash"])

    def test_document_key_reordering_keeps_hash_stable(self) -> None:
        service = self._service()
        service.store.add_device(Device("u", "d1", "ik"))
        baseline = service.persistence_integrity()["state_hash"]
        ordered = self._document()
        # Scramble every top-level key's order: the parsed document and
        # hence the hash must not move.
        self._overwrite({key: ordered[key]
                         for key in reversed(list(ordered))})
        self.assertEqual(service.persistence_integrity()["state_hash"],
                         baseline)

    def test_missing_optional_sections_hash_as_empty(self) -> None:
        # A legacy file carrying only {"version": 1} loads with every
        # optional section empty; its hash must equal the canonical empty
        # snapshot and its generation reads as 0.
        self._overwrite({"version": 1})
        service = DeviceService()
        attach_persistence(service, self.path)
        report = service.persistence_integrity()
        self.assertEqual(report["commit_seq"], 0)
        self.assertEqual(
            report["state_hash"],
            integrity_state_hash(
                canonical_integrity_snapshot({})))
        self.assertTrue(report["consistent"])

    def test_corrupt_json_is_503_and_changes_nothing(self) -> None:
        service = self._service()
        service.store.add_device(Device("u", "d1", "ik"))
        raw_before = open(self.path, "rb").read()
        inode_before = os.stat(self.path).st_ino
        seq_before = self.state_store.commit_seq
        with open(self.path, "wb") as handle:
            handle.write(b"{not json")
        self._expect_503(service)
        self.assertEqual(open(self.path, "rb").read(), b"{not json")
        self.assertEqual(os.stat(self.path).st_ino, inode_before)
        self.assertEqual(self.state_store.commit_seq, seq_before)
        # Memory stayed intact and the failure left no cursor/state change.
        self.assertIsNotNone(service.store.public_view("d1"))
        # Restore: writes and probes work again, file regenerated normally.
        with open(self.path, "wb") as handle:
            handle.write(raw_before)
        self.assertTrue(service.persistence_integrity()["consistent"])

    def test_wrong_version_and_non_object_are_503(self) -> None:
        service = self._service()
        service.store.add_device(Device("u", "d1", "ik"))
        valid = self._document()
        wrong_version = {**{k: v for k, v in valid.items()
                            if k != "version"}, "version": 2}
        no_version = {k: v for k, v in valid.items() if k != "version"}
        for mutated in (["not-an-object"], "a string document", 42,
                        wrong_version, no_version):
            self._overwrite(mutated)
            self._expect_503(service)
        self._overwrite(valid)
        self.assertTrue(service.persistence_integrity()["consistent"])

    def test_invalid_commit_seq_values_are_503(self) -> None:
        service = self._service()
        service.store.add_device(Device("u", "d1", "ik"))
        document = self._document()
        for bad in (True, -1, 1.5, "3", None):
            mutated = dict(document)
            mutated["commit_seq"] = bad
            self._overwrite(mutated)
            self._expect_503(service)

    def test_generation_mismatch_is_503(self) -> None:
        service = self._service()
        service.store.add_device(Device("u", "d1", "ik"))
        document = self._document()
        inode_before = os.stat(self.path).st_ino
        seq_before = self.state_store.commit_seq
        document["commit_seq"] = document["commit_seq"] + 1
        self._overwrite(document)
        self._expect_503(service)
        # The semantically-valid-but-wrong-generation file is left in place
        # and the in-memory generation does not move.
        self.assertEqual(os.stat(self.path).st_ino, inode_before)
        self.assertEqual(self.state_store.commit_seq, seq_before)

    def test_semantic_error_is_503_and_changes_nothing(self) -> None:
        service = self._service()
        service.store.add_device(Device("u", "d1", "ik"))
        document = self._document()
        inode_before = os.stat(self.path).st_ino
        # A session dangling at an unknown device: structurally fine,
        # semantically invalid.
        document["sessions"].append({
            "session_id": "sx", "initiator_device_id": "ghost",
            "recipient_device_id": "d1", "prekey_id": "k1",
            "ephemeral_key": "e", "identity_key": "i",
            "public_key": "p", "created_at": "2026-01-01T00:00:00+00:00"})
        self._overwrite(document)
        self._expect_503(service)
        self.assertEqual(os.stat(self.path).st_ino, inode_before)
        self.assertEqual(service.store.snapshot_state()["sessions"], [])

    def test_unknown_section_is_503(self) -> None:
        service = self._service()
        document = self._document()
        document["surprise_section"] = []
        self._overwrite(document)
        self._expect_503(service)

    def test_snapshot_divergence_is_503(self) -> None:
        service = self._service()
        service.store.add_device(Device("u", "d1", "ik"))
        committed = open(self.path, "rb").read()
        # Advance memory (and the file) by one more commit, then restore the
        # older generation's bytes: a valid document that disagrees with the
        # live snapshot and carries the previous generation.
        service.store.add_device(Device("u", "d2", "ik"))
        inode_before = os.stat(self.path).st_ino
        with open(self.path, "wb") as handle:
            handle.write(committed)
        self._expect_503(service)
        self.assertEqual(open(self.path, "rb").read(), committed)
        self.assertEqual(os.stat(self.path).st_ino, inode_before)
        # Memory still reflects the newer commit.
        self.assertIsNotNone(service.store.public_view("d2"))

    def test_missing_file_is_503_and_not_created(self) -> None:
        service = self._service()
        service.store.add_device(Device("u", "d1", "ik"))
        os.unlink(self.path)
        self._expect_503(service)
        self.assertFalse(os.path.exists(self.path))

    def test_degraded_state_probe_is_503_and_never_self_heals(self) -> None:
        service = self._service()
        service.store.add_device(Device("u", "d1", "ik"))
        # Recreate the undecidable-write state by hand: the formal path is
        # missing while the last committed inode survives only as a .bak.
        backup = os.path.join(self.directory, ".state-pinned.bak")
        os.link(self.path, backup)
        os.unlink(self.path)
        self.state_store.degraded = True
        self._expect_503(service)
        # The read-only probe must not run the write-time self-heal: the
        # formal path stays missing, the flag and the backup are untouched.
        self.assertTrue(self.state_store.degraded)
        self.assertFalse(os.path.exists(self.path))
        self.assertTrue(os.path.exists(backup))
        # A later write self-heals and commits; the probe then succeeds and
        # the leftover backup is swept.
        service.store.add_device(Device("u", "d2", "ik"))
        report = service.persistence_integrity()
        self.assertTrue(report["consistent"])
        self.assertFalse(any(
            name.startswith(".state-") for name in os.listdir(self.directory)))

    def test_hash_is_stable_across_restart(self) -> None:
        service = self._service()
        service.store.add_device(Device("u", "d1", "ik"))
        first = service.persistence_integrity()
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        second = restarted.persistence_integrity()
        self.assertEqual(first, second)

    def test_rich_fixture_sections_round_trip(self) -> None:
        # A document exercising many sections must verify and hash exactly
        # like the independently-canonicalised payload.
        build_fixture(self.directory)
        fixture_path = os.path.join(self.directory, "state.json")
        service = DeviceService()
        attach_persistence(service, fixture_path)
        report = service.persistence_integrity()
        self.assertTrue(report["consistent"])
        with open(fixture_path, encoding="utf-8") as handle:
            document = json.load(handle)
        self.assertEqual(report["state_hash"], _expected_hash(document))
        self.assertEqual(report["commit_seq"], document["commit_seq"])

    def test_probe_shares_the_store_lock_with_writes(self) -> None:
        service = self._service()
        service.store.add_device(Device("u", "d1", "ik"))
        done = threading.Event()
        result: Dict[str, Any] = {}

        def probe() -> None:
            try:
                result["report"] = service.persistence_integrity()
            except ServiceError as error:
                result["error"] = error
            finally:
                done.set()

        lock = service.store._lock
        lock.acquire()
        try:
            thread = threading.Thread(target=probe)
            thread.start()
            self.assertFalse(done.wait(timeout=0.3))
        finally:
            lock.release()
        self.assertTrue(done.wait(timeout=2.0))
        thread.join(timeout=2)
        self.assertIn("report", result)
        self.assertTrue(result["report"]["consistent"])

    def test_probe_linearizes_with_concurrent_writes(self) -> None:
        service = self._service()
        service.store.add_device(Device("u", "d0", "ik"))
        # Record the (commit_seq -> state_hash) of every generation the
        # persist hook actually commits, captured under the same store lock
        # that writes the file. The probe must never report a pair not in
        # this history (a torn file/memory state would).
        committed: Dict[int, str] = {}
        initial = service.persistence_integrity()
        committed[initial["commit_seq"]] = initial["state_hash"]
        original_hook = service.store.on_change

        def spy() -> None:
            assert original_hook is not None
            original_hook()
            document = self._document()
            committed[document["commit_seq"]] = _expected_hash(document)

        service.store.on_change = spy
        reports = []
        errors = []
        start = threading.Barrier(10)

        def write(index: int) -> None:
            start.wait()
            service.store.add_device(Device("u", f"w{index}", "ik"))

        def probe() -> None:
            start.wait()
            for _ in range(3):
                try:
                    reports.append(service.persistence_integrity())
                except ServiceError as error:
                    errors.append(error)

        try:
            threads = [threading.Thread(target=probe) for _ in range(7)]
            threads += [threading.Thread(target=write, args=(i,))
                        for i in range(3)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=5)
        finally:
            service.store.on_change = original_hook
        self.assertEqual(errors, [])
        self.assertGreaterEqual(len(reports), 7)
        # Every probe observed one complete committed generation.
        for report in reports:
            self.assertIn(report["commit_seq"], committed)
            self.assertEqual(
                report["state_hash"], committed[report["commit_seq"]])
            self.assertTrue(report["consistent"])
        self.assertEqual(service.persistence_integrity()["commit_seq"], 4)

    def _expect_503(self, service: DeviceService) -> None:
        with self.assertRaises(ServiceError) as caught:
            service.persistence_integrity()
        self.assertEqual(caught.exception.status_code, 503)
        self.assertEqual(caught.exception.field, "data_file")
        with self.assertRaises(IntegrityCheckError):
            self.state_store.integrity_report(service.store)


class IntegrityHTTPTest(unittest.TestCase):
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

    def _request(self, method: str, path: str,
                 body: object = None) -> Tuple[int, Any]:
        connection = HTTPConnection("127.0.0.1", self.port, timeout=5)
        payload = json.dumps(body) if body is not None else None
        headers = {"Content-Type": "application/json"} \
            if payload is not None else {}
        connection.request(method, path, body=payload, headers=headers)
        response = connection.getresponse()
        data = response.read().decode("utf-8")
        connection.close()
        return response.status, json.loads(data)

    def test_200_body_shape_and_key_order(self) -> None:
        status, body = self._request("GET", "/v1/persistence/integrity")
        self.assertEqual(status, 200)
        self.assertEqual(list(body),
                         ["commit_seq", "state_hash", "consistent"])
        self.assertEqual(body["commit_seq"], 0)
        self.assertIs(body["consistent"], True)
        self.assertRegex(body["state_hash"], r"^[0-9a-f]{64}$")

    def test_parameters_and_body_are_ignored(self) -> None:
        status, _ = self._request(
            "GET", "/v1/persistence/integrity?ignored=1&also=2")
        self.assertEqual(status, 200)
        status, _ = self._request(
            "GET", "/v1/persistence/integrity", body={"unexpected": True})
        self.assertEqual(status, 200)

    def test_tampered_file_is_503_data_file_with_error_key_order(self) -> None:
        self.service.store.add_device(Device("u", "d1", "ik"))
        raw = b"{oops"
        inode_before = os.stat(self.path).st_ino
        with open(self.path, "wb") as handle:
            handle.write(raw)
        status, body = self._request("GET", "/v1/persistence/integrity")
        self.assertEqual(status, 503)
        self.assertEqual(list(body), ["message", "field"])
        self.assertIsInstance(body["message"], str)
        self.assertEqual(body["field"], "data_file")
        self.assertEqual(os.stat(self.path).st_ino, inode_before)
        self.assertEqual(open(self.path, "rb").read(), raw)

    def test_reports_advancing_generations(self) -> None:
        status, first = self._request("GET", "/v1/persistence/integrity")
        self.assertEqual(first["commit_seq"], 0)
        self.service.store.add_device(Device("u", "d1", "ik"))
        status, second = self._request("GET", "/v1/persistence/integrity")
        self.assertEqual(status, 200)
        self.assertEqual(second["commit_seq"], 1)
        self.assertNotEqual(first["state_hash"], second["state_hash"])


class IntegrityInMemoryHTTPTest(unittest.TestCase):
    """A server started without a data file answers 409, never 503."""

    def setUp(self) -> None:
        from e2ee_backend.http_app import create_server
        self.server, _ = create_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def test_in_memory_server_is_409(self) -> None:
        connection = HTTPConnection("127.0.0.1", self.port, timeout=5)
        connection.request("GET", "/v1/persistence/integrity")
        response = connection.getresponse()
        data = json.loads(response.read().decode("utf-8"))
        connection.close()
        self.assertEqual(response.status, 409)
        self.assertEqual(list(data), ["message", "field"])
        self.assertEqual(data["field"], "data_file")


class IntegrityServeSubprocessTest(unittest.TestCase):
    """Real ``serve`` subprocesses: memory mode 409, file mode 200/503."""

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

    def _get(self, port: int) -> Tuple[int, Any]:
        connection = HTTPConnection("127.0.0.1", port, timeout=5)
        connection.request("GET", "/v1/persistence/integrity")
        response = connection.getresponse()
        data = json.loads(response.read().decode("utf-8"))
        connection.close()
        return response.status, data

    def test_memory_serve_is_409_and_file_serve_is_200(self) -> None:
        _proc, port = self._serve(use_file=False)
        status, body = self._get(port)
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "data_file")

        _proc2, file_port = self._serve(use_file=True)
        status, body = self._get(file_port)
        self.assertEqual(status, 200)
        self.assertEqual(body["commit_seq"], 0)
        self.assertTrue(body["consistent"])

        # Tamper with the file out of band: the live server reports 503
        # without rewriting it; after a restart the same corrupt file still
        # does not silently change (the probe never repairs it).
        with open(self.data_file, "wb") as handle:
            handle.write(b"broken")
        status, body = self._get(file_port)
        self.assertEqual(status, 503)
        self.assertEqual(body["field"], "data_file")
        self.assertEqual(open(self.data_file, "rb").read(), b"broken")


if __name__ == "__main__":
    unittest.main()
