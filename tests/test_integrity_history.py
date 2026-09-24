"""``GET /v1/persistence/integrity/history`` and its append-only sidecar.

Beside the durable state file ``state.json`` the store maintains a sibling
``state.json.integrity``: one compact UTF-8 JSON document
``{"version":1,"entries":[...]}`` with one entry per durable commit, keyed
``commit_seq``/``state_hash``/``prev_hash``/``hash`` in that order. Entries
ascend by ``commit_seq``, the first has an empty ``prev_hash`` and each later
one links the previous entry's ``hash``; ``hash`` is the SHA-256 of the entry
without ``hash`` (sorted-key compact JSON, ``ensure_ascii=False``) and
``state_hash`` is the probe hash of that generation's canonical state.

The state document records the format with ``integrity_log_version=1`` (set on
the first real commit, immediately after ``commit_seq``). A legacy document
without the marker and without a sidecar still starts and migrates on its
first commit; any marker/sidecar disagreement, broken chain or mismatched tail
refuses startup with a single ``field=data_file`` stderr line and exit 1, the
on-disk files untouched. The history endpoint answers 409/field=data_file
without the marker and otherwise 200 ``{"commit_seq","entries"}`` with
``commit_seq`` equal to both the state generation and the last entry.
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
    deadline = __import__("time").time() + timeout
    while __import__("time").time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            __import__("time").sleep(0.05)
    raise AssertionError(f"server on port {port} never started listening")


def _entry_hash(entry: Dict[str, Any]) -> str:
    """Independently recompute an integrity entry's chain hash."""
    document = {"commit_seq": entry["commit_seq"],
                "state_hash": entry["state_hash"],
                "prev_hash": entry["prev_hash"]}
    raw = json.dumps(document, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class IntegrityHistoryServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.path = os.path.join(self.directory, "state.json")
        self.sidecar = self.path + ".integrity"

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def _service(self) -> DeviceService:
        service = DeviceService()
        self.state_store = attach_persistence(service, self.path)
        return service

    def _state_doc(self) -> Dict[str, Any]:
        with open(self.path, encoding="utf-8") as handle:
            return json.load(handle)

    def _sidecar_doc(self) -> Dict[str, Any]:
        with open(self.sidecar, encoding="utf-8") as handle:
            return json.load(handle)

    def _write_sidecar(self, document: Dict[str, Any]) -> None:
        with open(self.sidecar, "w", encoding="utf-8") as handle:
            json.dump(document, handle)

    def _write_state(self, document: Dict[str, Any]) -> None:
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)

    # -- enablement / 409 -------------------------------------------------

    def test_in_memory_mode_is_409_data_file(self) -> None:
        service = DeviceService()
        with self.assertRaises(ServiceError) as caught:
            service.persistence_integrity_history()
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "data_file")

    def test_bootstrap_file_has_no_marker_or_sidecar(self) -> None:
        self._service()
        self.assertNotIn("integrity_log_version", self._state_doc())
        self.assertFalse(os.path.exists(self.sidecar))

    def test_history_is_409_before_the_first_commit(self) -> None:
        service = self._service()
        with self.assertRaises(ServiceError) as caught:
            service.persistence_integrity_history()
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "data_file")

    def test_first_commit_stamps_marker_and_creates_sidecar(self) -> None:
        service = self._service()
        service.store.add_device(Device("u", "d1", "ik"))
        self.assertEqual(self._state_doc()["integrity_log_version"], 1)
        self.assertTrue(os.path.exists(self.sidecar))
        report = service.persistence_integrity_history()
        self.assertEqual(list(report), ["commit_seq", "entries"])
        self.assertEqual(report["commit_seq"], 1)
        self.assertEqual(len(report["entries"]), 1)
        entry = report["entries"][0]
        self.assertEqual(list(entry),
                         ["commit_seq", "state_hash", "prev_hash", "hash"])
        self.assertEqual(entry["commit_seq"], 1)
        self.assertEqual(entry["prev_hash"], "")
        self.assertRegex(entry["state_hash"], r"^[0-9a-f]{64}$")
        self.assertRegex(entry["hash"], r"^[0-9a-f]{64}$")
        self.assertEqual(entry["hash"], _entry_hash(entry))

    # -- chain growth -----------------------------------------------------

    def test_entries_chain_across_commits(self) -> None:
        service = self._service()
        for index in range(1, 5):
            service.store.add_device(Device("u", f"d{index}", "ik"))
        report = service.persistence_integrity_history()
        self.assertEqual(report["commit_seq"], 4)
        entries = report["entries"]
        self.assertEqual([e["commit_seq"] for e in entries], [1, 2, 3, 4])
        prev = ""
        for index, entry in enumerate(entries):
            self.assertEqual(entry["hash"], _entry_hash(entry))
            self.assertEqual(entry["prev_hash"], prev)
            prev = entry["hash"]
        # commit_seq equals both the state generation and the last entry.
        self.assertEqual(report["commit_seq"],
                         self._state_doc()["commit_seq"])
        self.assertEqual(report["commit_seq"], entries[-1]["commit_seq"])

    def test_state_hash_matches_the_probe_at_every_generation(self) -> None:
        service = self._service()
        for index in range(1, 4):
            service.store.add_device(Device("u", f"d{index}", "ik"))
        probe = service.persistence_integrity()
        history = service.persistence_integrity_history()
        self.assertEqual(history["entries"][-1]["state_hash"],
                         probe["state_hash"])
        self.assertEqual(history["commit_seq"], probe["commit_seq"])

    def test_idempotent_replays_append_nothing(self) -> None:
        service = self._service()
        service.store.add_device(Device("u", "d1", "ik"))
        before = len(service.persistence_integrity_history()["entries"])
        # No-op identity rotation performs no persistence transaction.
        service.store.rotate_identity_key("d1", "ik")
        service.store.rotate_identity_key("d1", "ik")
        after = service.persistence_integrity_history()
        self.assertEqual(len(after["entries"]), before)
        self.assertEqual(after["commit_seq"], 1)

    def test_read_only_probes_append_nothing(self) -> None:
        service = self._service()
        service.store.add_device(Device("u", "d1", "ik"))
        raw_before = open(self.sidecar, "rb").read()
        service.persistence_integrity()
        service.persistence_integrity_history()
        service.persistence_integrity_history()
        self.assertEqual(open(self.sidecar, "rb").read(), raw_before)

    def test_history_survives_restart(self) -> None:
        service = self._service()
        for index in range(1, 4):
            service.store.add_device(Device("u", f"d{index}", "ik"))
        first = service.persistence_integrity_history()
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        second = restarted.persistence_integrity_history()
        self.assertEqual(first, second)
        # And continues consecutively after the restart.
        restarted.store.add_device(Device("u", "d4", "ik"))
        third = restarted.persistence_integrity_history()
        self.assertEqual([e["commit_seq"] for e in third["entries"]],
                         [1, 2, 3, 4])

    # -- on-disk encoding -------------------------------------------------

    def test_sidecar_is_compact_without_spaces_or_trailing_newline(self) -> None:
        service = self._service()
        # A non-ASCII registration flows into the state_hash (hashed over the
        # literal-UTF-8 canonical state) even though the sidecar itself only
        # records the resulting hex digest.
        service.store.add_device(Device("用户", "设备✓", "ik-日本語"))
        raw = open(self.sidecar, "rb").read()
        # Compact: no whitespace beyond the JSON tokens, no trailing newline.
        self.assertNotIn(b"\n", raw)
        self.assertNotIn(b" ", raw)
        self.assertFalse(raw.endswith(b"\n"))
        document = json.loads(raw.decode("utf-8"))
        self.assertEqual(list(document), ["version", "entries"])
        # Independently recompute the state_hash with literal UTF-8 compact
        # canonical JSON and confirm it equals the sidecar's tail hash and
        # differs from the ASCII-escaped digest of the same canonical state.
        from e2ee_backend.storage import (
            canonical_integrity_snapshot, integrity_state_hash)
        state_payload = {
            key: value for key, value in self._state_doc().items()
            if key not in ("version", "commit_seq",
                           "integrity_log_version")}
        canonical = canonical_integrity_snapshot(state_payload)
        expected = integrity_state_hash(canonical)
        self.assertEqual(document["entries"][-1]["state_hash"], expected)
        escaped = hashlib.sha256(
            json.dumps(canonical, separators=(",", ":"),
                       ensure_ascii=True).encode("utf-8")).hexdigest()
        self.assertNotEqual(escaped, expected)

    # -- legacy migration -------------------------------------------------

    def test_legacy_file_starts_without_marker_or_sidecar(self) -> None:
        # A pre-feature file with no marker and no sidecar must attach.
        service = self._service()
        for index in range(3):
            service.store.add_device(Device("u", f"d{index}", "ik"))
        document = self._state_doc()
        document.pop("integrity_log_version")
        self._write_state(document)
        os.remove(self.sidecar)
        legacy = DeviceService()
        store = attach_persistence(legacy, self.path)
        self.assertFalse(store.integrity_log_enabled)
        with self.assertRaises(ServiceError) as caught:
            legacy.persistence_integrity_history()
        self.assertEqual(caught.exception.status_code, 409)

    def test_legacy_file_anchors_chain_at_its_first_commit(self) -> None:
        service = self._service()
        for index in range(3):
            service.store.add_device(Device("u", f"d{index}", "ik"))
        document = self._state_doc()
        document.pop("integrity_log_version")
        self._write_state(document)
        os.remove(self.sidecar)

        legacy = DeviceService()
        attach_persistence(legacy, self.path)
        legacy.store.add_device(Device("u", "new", "ik"))
        on_disk = self._state_doc()
        sidecar = self._sidecar_doc()
        self.assertEqual(on_disk["integrity_log_version"], 1)
        self.assertEqual(on_disk["commit_seq"], 4)
        self.assertEqual(len(sidecar["entries"]), 1)
        entry = sidecar["entries"][0]
        # The anchor sits at the generation of the migrating commit (4), with
        # an empty prev_hash, and the chain verifies and restarts cleanly.
        self.assertEqual(entry["commit_seq"], 4)
        self.assertEqual(entry["prev_hash"], "")
        self.assertEqual(entry["hash"], _entry_hash(entry))
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        report = restarted.persistence_integrity_history()
        self.assertEqual(report["commit_seq"], 4)
        self.assertEqual([e["commit_seq"] for e in report["entries"]], [4])

    # -- transactional failure -------------------------------------------

    def test_sidecar_write_failure_rolls_back_and_appends_once_on_retry(
            self) -> None:
        import e2ee_backend.persistence as persistence_mod
        service = self._service()
        service.store.add_device(Device("u", "d1", "ik"))
        state_bytes = open(self.path, "rb").read()
        log_bytes = open(self.sidecar, "rb").read()

        real_replace = persistence_mod.os.replace
        sidecar_target = os.path.abspath(self.sidecar)

        def fail_sidecar_replace(src: str, dst: str) -> None:
            if os.path.abspath(dst) == sidecar_target:
                raise OSError("simulated sidecar replace failure")
            return real_replace(src, dst)

        persistence_mod.os.replace = fail_sidecar_replace
        try:
            with self.assertRaises(PersistenceUnavailable):
                service.store.add_device(Device("u", "d2", "ik"))
        finally:
            persistence_mod.os.replace = real_replace

        # Memory rolled back; neither file advanced; no tmp staged file.
        self.assertIsNone(service.store.find_by_device_id("d2"))
        self.assertEqual(open(self.path, "rb").read(), state_bytes)
        self.assertEqual(open(self.sidecar, "rb").read(), log_bytes)
        self.assertFalse(
            any(name.startswith(".integrity-")
                for name in os.listdir(self.directory)))
        # The failed generation is reused by the repaired commit: the entry
        # for generation 2 appears exactly once.
        service.store.add_device(Device("u", "d2", "ik"))
        entries = self._sidecar_doc()["entries"]
        self.assertEqual([e["commit_seq"] for e in entries], [1, 2])
        self.assertEqual(self._state_doc()["commit_seq"], 2)

    # -- runtime tampering -> 503, files untouched -----------------------

    def _expect_history_503(self, service: DeviceService) -> None:
        with self.assertRaises(ServiceError) as caught:
            service.persistence_integrity_history()
        self.assertEqual(caught.exception.status_code, 503)
        self.assertEqual(caught.exception.field, "data_file")

    def test_broken_chain_is_503_and_read_only(self) -> None:
        service = self._service()
        for index in range(1, 4):
            service.store.add_device(Device("u", f"d{index}", "ik"))
        raw_before = open(self.sidecar, "rb").read()
        inode_before = os.stat(self.sidecar).st_ino
        document = self._sidecar_doc()
        document["entries"][1]["prev_hash"] = "0" * 64
        self._write_sidecar(document)
        self._expect_history_503(service)
        self.assertEqual(os.stat(self.sidecar).st_ino, inode_before)
        # Restore: the probe works again without any self-heal.
        with open(self.sidecar, "wb") as handle:
            handle.write(raw_before)
        self.assertEqual(service.persistence_integrity_history()["commit_seq"],
                         3)

    def test_bad_entry_hash_is_503(self) -> None:
        service = self._service()
        service.store.add_device(Device("u", "d1", "ik"))
        document = self._sidecar_doc()
        document["entries"][0]["state_hash"] = "a" * 64
        self._write_sidecar(document)
        self._expect_history_503(service)

    def test_missing_sidecar_with_marker_is_503(self) -> None:
        service = self._service()
        service.store.add_device(Device("u", "d1", "ik"))
        os.remove(self.sidecar)
        self._expect_history_503(service)

    def test_wrong_log_version_is_503(self) -> None:
        service = self._service()
        service.store.add_device(Device("u", "d1", "ik"))
        document = self._sidecar_doc()
        document["version"] = 2
        self._write_sidecar(document)
        self._expect_history_503(service)

    def test_tail_generation_mismatch_is_503(self) -> None:
        service = self._service()
        service.store.add_device(Device("u", "d1", "ik"))
        document = self._sidecar_doc()
        document["entries"][0]["commit_seq"] = 99
        # Keep the entry hash internally consistent so the failure is the
        # tail/generation binding, not the per-entry hash.
        entry = document["entries"][0]
        entry["hash"] = _entry_hash(entry)
        self._write_sidecar(document)
        self._expect_history_503(service)

    def test_bad_marker_value_is_503(self) -> None:
        service = self._service()
        service.store.add_device(Device("u", "d1", "ik"))
        document = self._state_doc()
        document["integrity_log_version"] = 2
        self._write_state(document)
        self._expect_history_503(service)

    def test_missing_marker_at_runtime_is_409(self) -> None:
        # A live server whose marker is removed out of band answers 409 (no
        # marker) rather than 503; the strict pairing refusal is a startup
        # gate, applied on the next restart.
        service = self._service()
        service.store.add_device(Device("u", "d1", "ik"))
        document = self._state_doc()
        document.pop("integrity_log_version")
        self._write_state(document)
        with self.assertRaises(ServiceError) as caught:
            service.persistence_integrity_history()
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "data_file")

    def test_history_linearizes_with_concurrent_commits(self) -> None:
        service = self._service()
        start = threading.Barrier(8)
        reports: List[Dict[str, Any]] = []
        errors: List[ServiceError] = []

        def write(index: int) -> None:
            start.wait()
            service.store.add_device(Device("u", f"w{index}", "ik"))

        def probe() -> None:
            start.wait()
            for _ in range(5):
                try:
                    reports.append(service.persistence_integrity_history())
                except ServiceError as error:  # pragma: no cover
                    errors.append(error)

        threads = [threading.Thread(target=probe) for _ in range(6)]
        threads += [threading.Thread(target=write, args=(i,)) for i in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
        self.assertEqual(errors, [])
        self.assertTrue(reports)
        # Every observed tail is one complete committed generation with a
        # self-consistent chain.
        for report in reports:
            entries = report["entries"]
            self.assertEqual(entries[-1]["commit_seq"], report["commit_seq"])
            prev = ""
            for entry in entries:
                self.assertEqual(entry["prev_hash"], prev)
                self.assertEqual(entry["hash"], _entry_hash(entry))
                prev = entry["hash"]


class IntegrityHistoryStartupTest(unittest.TestCase):
    """The marker/sidecar pairing gate refuses startup, files untouched."""

    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.path = os.path.join(self.directory, "state.json")
        self.sidecar = self.path + ".integrity"
        service = DeviceService()
        attach_persistence(service, self.path)
        for index in range(1, 3):
            service.store.add_device(Device("u", f"d{index}", "ik"))

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def _state(self) -> Dict[str, Any]:
        return json.load(open(self.path, encoding="utf-8"))

    def _sidecar(self) -> Dict[str, Any]:
        return json.load(open(self.sidecar, encoding="utf-8"))

    def _write_state(self, document: Dict[str, Any]) -> None:
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)

    def _write_sidecar(self, document: Dict[str, Any]) -> None:
        with open(self.sidecar, "w", encoding="utf-8") as handle:
            json.dump(document, handle)

    def _assert_refuses_untouched(self) -> None:
        state_before = open(self.path, "rb").read()
        state_ino = os.stat(self.path).st_ino
        sidecar_before = open(self.sidecar, "rb").read()
        sidecar_ino = os.stat(self.sidecar).st_ino
        with self.assertRaises(StateFileError):
            attach_persistence(DeviceService(), self.path)
        self.assertEqual(open(self.path, "rb").read(), state_before)
        self.assertEqual(os.stat(self.path).st_ino, state_ino)
        self.assertEqual(open(self.sidecar, "rb").read(), sidecar_before)
        self.assertEqual(os.stat(self.sidecar).st_ino, sidecar_ino)

    def test_marker_without_sidecar_refuses(self) -> None:
        os.remove(self.sidecar)
        state_before = open(self.path, "rb").read()
        state_ino = os.stat(self.path).st_ino
        with self.assertRaises(StateFileError):
            attach_persistence(DeviceService(), self.path)
        self.assertEqual(open(self.path, "rb").read(), state_before)
        self.assertEqual(os.stat(self.path).st_ino, state_ino)

    def test_sidecar_without_marker_refuses(self) -> None:
        document = self._state()
        document.pop("integrity_log_version")
        self._write_state(document)
        self._assert_refuses_untouched()

    def test_wrong_marker_value_refuses(self) -> None:
        document = self._state()
        document["integrity_log_version"] = 2
        self._write_state(document)
        self._assert_refuses_untouched()

    def test_bad_log_structure_refuses(self) -> None:
        for mutated in ({"version": 2, "entries": []},
                        {"version": 1, "entries": {}},
                        {"version": 1},
                        ["version", 1],
                        {"version": 1, "entries": [], "extra": 1}):
            self._write_sidecar(mutated)  # type: ignore[arg-type]
            self._assert_refuses_untouched()

    def test_empty_entries_refuses(self) -> None:
        self._write_sidecar({"version": 1, "entries": []})
        self._assert_refuses_untouched()

    def test_broken_chain_refuses(self) -> None:
        document = self._sidecar()
        document["entries"][1]["prev_hash"] = "f" * 64
        self._write_sidecar(document)
        self._assert_refuses_untouched()

    def test_bad_entry_hash_refuses(self) -> None:
        document = self._sidecar()
        document["entries"][0]["hash"] = "0" * 64
        self._write_sidecar(document)
        self._assert_refuses_untouched()

    def test_non_consecutive_generation_refuses(self) -> None:
        document = self._sidecar()
        document["entries"][1]["commit_seq"] = 5
        entry = document["entries"][1]
        entry["hash"] = _entry_hash(entry)
        self._write_sidecar(document)
        self._assert_refuses_untouched()

    def test_tail_generation_mismatch_refuses(self) -> None:
        # The state advances one generation beyond the sidecar's tail.
        state = self._state()
        state["commit_seq"] = state["commit_seq"] + 1
        self._write_state(state)
        with self.assertRaises(StateFileError):
            attach_persistence(DeviceService(), self.path)

    def test_tail_state_hash_mismatch_refuses(self) -> None:
        document = self._sidecar()
        document["entries"][-1]["state_hash"] = "9" * 64
        entry = document["entries"][-1]
        entry["hash"] = _entry_hash(entry)
        self._write_sidecar(document)
        self._assert_refuses_untouched()

    def test_consistent_pairing_starts(self) -> None:
        # The untouched, consistent fixture starts cleanly.
        service = DeviceService()
        store = attach_persistence(service, self.path)
        self.assertTrue(store.integrity_log_enabled)
        self.assertEqual(
            service.persistence_integrity_history()["commit_seq"], 2)

    def test_serve_refuses_with_single_stderr_line_exit_1(self) -> None:
        document = self._sidecar()
        document["entries"][-1]["hash"] = "0" * 64
        self._write_sidecar(document)
        result = subprocess.run(
            [sys.executable, "-m", "e2ee_backend", "serve",
             "--host", "127.0.0.1", "--port", str(_free_port()),
             "--data-file", self.path],
            capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 1)
        lines = result.stderr.strip().splitlines()
        self.assertEqual(len(lines), 1)
        body = json.loads(lines[0])
        self.assertEqual(body["field"], "data_file")
        # Files untouched by the refused start.
        self.assertEqual(self._sidecar()["entries"][-1]["hash"], "0" * 64)


class IntegrityHistoryHTTPTest(unittest.TestCase):
    """End-to-end HTTP behaviour for the history route."""

    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.path = os.path.join(self.directory, "state.json")
        self.service = DeviceService()
        attach_persistence(self.service, self.path)
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

    def test_pre_commit_is_409(self) -> None:
        status, body = self._request(
            "GET", "/v1/persistence/integrity/history")
        self.assertEqual(status, 409)
        self.assertEqual(list(body), ["message", "field"])
        self.assertEqual(body["field"], "data_file")

    def test_post_commit_is_200_with_ordered_body(self) -> None:
        self.service.store.add_device(Device("u", "d1", "ik"))
        status, body = self._request(
            "GET", "/v1/persistence/integrity/history")
        self.assertEqual(status, 200)
        self.assertEqual(list(body), ["commit_seq", "entries"])
        self.assertEqual(body["commit_seq"], 1)
        self.assertEqual(body["commit_seq"],
                         body["entries"][-1]["commit_seq"])
        self.assertEqual(list(body["entries"][0]),
                         ["commit_seq", "state_hash", "prev_hash", "hash"])

    def test_parameters_and_body_are_ignored(self) -> None:
        self.service.store.add_device(Device("u", "d1", "ik"))
        status, _ = self._request(
            "GET", "/v1/persistence/integrity/history?x=1",
            body={"unexpected": True})
        self.assertEqual(status, 200)

    def test_tampered_sidecar_is_503_data_file(self) -> None:
        self.service.store.add_device(Device("u", "d1", "ik"))
        sidecar = self.path + ".integrity"
        with open(sidecar, encoding="utf-8") as handle:
            document = json.load(handle)
        document["entries"][0]["hash"] = "0" * 64
        with open(sidecar, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        status, body = self._request(
            "GET", "/v1/persistence/integrity/history")
        self.assertEqual(status, 503)
        self.assertEqual(list(body), ["message", "field"])
        self.assertEqual(body["field"], "data_file")


class IntegrityHistoryServeSubprocessTest(unittest.TestCase):
    """Memory mode answers 409; a migrated file answers 200 over real socket."""

    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.data_file = os.path.join(self.directory, "state.json")
        self.processes: List[subprocess.Popen] = []

    def tearDown(self) -> None:
        for proc in self.processes:
            if proc.poll() is None:
                proc.send_signal(signal.SIGKILL)
                proc.wait(timeout=5)
        shutil.rmtree(self.directory, ignore_errors=True)

    def _serve(self, use_file: bool) -> Tuple[subprocess.Popen, int]:
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
        connection.request("GET", "/v1/persistence/integrity/history")
        response = connection.getresponse()
        data = json.loads(response.read().decode("utf-8"))
        connection.close()
        return response.status, data

    def test_memory_is_409_and_legacy_file_is_409_then_200(self) -> None:
        _proc, port = self._serve(use_file=False)
        status, body = self._get(port)
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "data_file")

        _proc2, file_port = self._serve(use_file=True)
        status, body = self._get(file_port)
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "data_file")

        # First real commit migrates; the history endpoint now answers 200.
        import base64
        from cryptography.hazmat.primitives.asymmetric import x25519
        from cryptography.hazmat.primitives import serialization

        def _raw_key_b64() -> str:
            raw = x25519.X25519PrivateKey.generate().public_key().public_bytes(
                serialization.Encoding.Raw,
                serialization.PublicFormat.Raw)
            return base64.b64encode(raw).decode()

        connection = HTTPConnection("127.0.0.1", file_port, timeout=5)
        connection.request(
            "POST", "/v1/devices",
            body=json.dumps({
                "user_id": "u", "device_id": "d1",
                "identity_key": _raw_key_b64(),
                "signed_prekeys": [
                    {"key_id": "k1", "public_key": _raw_key_b64()}]}),
            headers={"Content-Type": "application/json"})
        self.assertEqual(connection.getresponse().status, 201)
        connection.close()
        status, body = self._get(file_port)
        self.assertEqual(status, 200)
        self.assertEqual(list(body), ["commit_seq", "entries"])
        self.assertEqual(body["commit_seq"], 1)
        self.assertEqual(len(body["entries"]), 1)


if __name__ == "__main__":
    unittest.main()
