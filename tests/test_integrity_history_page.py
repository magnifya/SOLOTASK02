"""``GET /v1/persistence/integrity/history/page`` — paged integrity audit.

The page endpoint shares the sidecar and verification of
``/v1/persistence/integrity/history`` but returns one ascending cursor page:
``commit_seq > after`` (strictly), at most ``limit``. The 200 body is ordered
``commit_seq``/``entries``/``next_after``/``has_more``; an empty page keeps
``next_after == after`` and otherwise it is the last returned generation,
with ``has_more`` reporting a successor. ``after``/``limit`` are single
valued (a repeat is 400), default to 0/100, and must be strict decimal
non-negative integers with ``limit`` in 1..100 — otherwise 400 with
``field`` naming the parameter. Availability is unchanged: in-memory and
not-yet-migrated file mode answer 409/field=data_file; any parse/version/
generation/chain/hash/read failure answers 503/field=data_file. The CLI
``integrity-history-page`` mirrors the route, with a connection failure
becoming one stderr line field=server and a non-zero exit.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from http.client import HTTPConnection
from typing import Any, Dict, List, Tuple

from e2ee_backend.models import Device
from e2ee_backend.persistence import attach_persistence
from e2ee_backend.service import DeviceService, ServiceError


def _entry_hash(entry: Dict[str, Any]) -> str:
    """Independently recompute an integrity entry's chain hash."""
    document = {"commit_seq": entry["commit_seq"],
                "state_hash": entry["state_hash"],
                "prev_hash": entry["prev_hash"]}
    raw = json.dumps(document, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class IntegrityHistoryPageServiceTest(unittest.TestCase):
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

    def _commit(self, service: DeviceService, count: int) -> None:
        existing = getattr(self, "_committed", 0)
        for index in range(existing + 1, existing + count + 1):
            service.store.add_device(Device("u", f"d{index}", "ik"))
        self._committed = existing + count

    def _sidecar_doc(self) -> Dict[str, Any]:
        with open(self.sidecar, encoding="utf-8") as handle:
            return json.load(handle)

    def _write_sidecar(self, document: Dict[str, Any]) -> None:
        with open(self.sidecar, "w", encoding="utf-8") as handle:
            json.dump(document, handle)

    # -- availability / 409 ----------------------------------------------

    def test_in_memory_mode_is_409_data_file(self) -> None:
        service = DeviceService()
        with self.assertRaises(ServiceError) as caught:
            service.persistence_integrity_history_page(0, 100)
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "data_file")

    def test_bootstrap_file_is_409_before_the_first_commit(self) -> None:
        service = self._service()
        with self.assertRaises(ServiceError) as caught:
            service.persistence_integrity_history_page(0, 100)
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "data_file")

    # -- 200 shape and paging --------------------------------------------

    def test_first_commit_page_shape_and_entry_order(self) -> None:
        service = self._service()
        self._commit(service, 1)
        report = service.persistence_integrity_history_page(0, 100)
        self.assertEqual(list(report),
                         ["commit_seq", "entries", "next_after", "has_more"])
        self.assertEqual(report["commit_seq"], 1)
        self.assertEqual(report["next_after"], 1)
        self.assertFalse(report["has_more"])
        entries = report["entries"]
        self.assertEqual(len(entries), 1)
        entry = entries[0]
        self.assertEqual(list(entry),
                         ["commit_seq", "state_hash", "prev_hash", "hash"])
        self.assertEqual(entry["commit_seq"], 1)
        self.assertEqual(entry["prev_hash"], "")
        self.assertRegex(entry["state_hash"], r"^[0-9a-f]{64}$")
        self.assertRegex(entry["hash"], r"^[0-9a-f]{64}$")
        self.assertEqual(entry["hash"], _entry_hash(entry))

    def test_page_is_strictly_after_ascending_and_limited(self) -> None:
        service = self._service()
        self._commit(service, 5)
        report = service.persistence_integrity_history_page(2, 2)
        self.assertEqual(report["commit_seq"], 5)
        self.assertEqual([e["commit_seq"] for e in report["entries"]], [3, 4])
        self.assertEqual(report["next_after"], 4)
        self.assertTrue(report["has_more"])

    def test_page_ending_at_the_tail_has_no_more(self) -> None:
        service = self._service()
        self._commit(service, 5)
        report = service.persistence_integrity_history_page(3, 2)
        self.assertEqual([e["commit_seq"] for e in report["entries"]], [4, 5])
        self.assertEqual(report["next_after"], 5)
        self.assertFalse(report["has_more"])

    def test_empty_page_keeps_after_and_reports_no_more(self) -> None:
        service = self._service()
        self._commit(service, 3)
        report = service.persistence_integrity_history_page(3, 100)
        self.assertEqual(report["entries"], [])
        self.assertEqual(report["next_after"], 3)
        self.assertFalse(report["has_more"])

    def test_after_beyond_the_tail_is_empty_with_next_after_preserved(self) -> None:
        service = self._service()
        self._commit(service, 3)
        report = service.persistence_integrity_history_page(99, 100)
        self.assertEqual(report["entries"], [])
        self.assertEqual(report["next_after"], 99)
        self.assertFalse(report["has_more"])

    def test_defaults_page_the_whole_chain(self) -> None:
        service = self._service()
        self._commit(service, 4)
        report = service.persistence_integrity_history_page(0, 100)
        self.assertEqual([e["commit_seq"] for e in report["entries"]],
                         [1, 2, 3, 4])
        self.assertEqual(report["next_after"], 4)
        self.assertFalse(report["has_more"])

    def test_limit_one_walks_every_generation_via_next_after(self) -> None:
        service = self._service()
        self._commit(service, 4)
        seen: List[int] = []
        after = 0
        for _ in range(10):  # an upper bound far beyond the chain length
            report = service.persistence_integrity_history_page(after, 1)
            if not report["entries"]:
                self.assertFalse(report["has_more"])
                break
            self.assertEqual(len(report["entries"]), 1)
            self.assertEqual(report["next_after"],
                             report["entries"][-1]["commit_seq"])
            seen.extend(e["commit_seq"] for e in report["entries"])
            after = report["next_after"]
        self.assertEqual(seen, [1, 2, 3, 4])

    def test_entries_remain_a_verified_chain(self) -> None:
        service = self._service()
        self._commit(service, 5)
        report = service.persistence_integrity_history_page(1, 3)
        prev = report["entries"][0]["prev_hash"]
        for entry in report["entries"]:
            self.assertEqual(entry["prev_hash"], prev)
            self.assertEqual(entry["hash"], _entry_hash(entry))
            prev = entry["hash"]

    def test_probe_is_read_only(self) -> None:
        service = self._service()
        self._commit(service, 2)
        raw = open(self.sidecar, "rb").read()
        service.persistence_integrity_history_page(0, 1)
        service.persistence_integrity_history_page(2, 100)
        service.persistence_integrity_history_page(99, 100)
        self.assertEqual(open(self.sidecar, "rb").read(), raw)

    # -- service-level argument validation (400) -------------------------

    def test_service_rejects_invalid_arguments(self) -> None:
        service = self._service()
        self._commit(service, 1)
        for after, limit, field in (
                (-1, 100, "after"),
                (True, 100, "after"),  # type: ignore[arg-type]
                ("1", 100, "after"),  # type: ignore[arg-type]
                (0, 0, "limit"),
                (0, 101, "limit"),
                (0, True, "limit"),  # type: ignore[arg-type]
                (0, "5", "limit")):  # type: ignore[arg-type]
            with self.assertRaises(ServiceError) as caught:
                service.persistence_integrity_history_page(after, limit)
            self.assertEqual(caught.exception.status_code, 400)
            self.assertEqual(caught.exception.field, field)

    # -- 503 on a broken sidecar, files untouched ------------------------

    def test_broken_chain_is_503_data_file(self) -> None:
        service = self._service()
        self._commit(service, 3)
        inode_before = os.stat(self.sidecar).st_ino
        document = self._sidecar_doc()
        document["entries"][1]["prev_hash"] = "0" * 64
        self._write_sidecar(document)
        with self.assertRaises(ServiceError) as caught:
            service.persistence_integrity_history_page(0, 100)
        self.assertEqual(caught.exception.status_code, 503)
        self.assertEqual(caught.exception.field, "data_file")
        self.assertEqual(os.stat(self.sidecar).st_ino, inode_before)


class IntegrityHistoryPageHTTPTest(unittest.TestCase):
    """End-to-end HTTP behaviour for the paged history route."""

    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.path = os.path.join(self.directory, "state.json")
        self.sidecar = self.path + ".integrity"
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

    def _request(self, query: str = "") -> Tuple[int, Any]:
        connection = HTTPConnection("127.0.0.1", self.port, timeout=5)
        connection.request(
            "GET", "/v1/persistence/integrity/history/page" + query)
        response = connection.getresponse()
        data = response.read().decode("utf-8")
        connection.close()
        return response.status, json.loads(data)

    def _commit(self, count: int) -> None:
        existing = getattr(self, "_committed", 0)
        for index in range(existing + 1, existing + count + 1):
            self.service.store.add_device(Device("u", f"d{index}", "ik"))
        self._committed = existing + count

    def test_pre_commit_is_409(self) -> None:
        status, body = self._request()
        self.assertEqual(status, 409)
        self.assertEqual(list(body), ["message", "field"])
        self.assertEqual(body["field"], "data_file")

    def test_post_commit_is_200_with_ordered_body(self) -> None:
        self._commit(3)
        status, body = self._request("?after=1&limit=1")
        self.assertEqual(status, 200)
        self.assertEqual(list(body),
                         ["commit_seq", "entries", "next_after", "has_more"])
        self.assertEqual(body["commit_seq"], 3)
        self.assertEqual([e["commit_seq"] for e in body["entries"]], [2])
        self.assertEqual(body["next_after"], 2)
        self.assertTrue(body["has_more"])
        self.assertEqual(list(body["entries"][0]),
                         ["commit_seq", "state_hash", "prev_hash", "hash"])

    def test_defaults_apply(self) -> None:
        self._commit(2)
        status, body = self._request()
        self.assertEqual(status, 200)
        self.assertEqual([e["commit_seq"] for e in body["entries"]], [1, 2])
        self.assertEqual(body["next_after"], 2)
        self.assertFalse(body["has_more"])

    def test_extra_parameters_are_ignored(self) -> None:
        self._commit(1)
        status, _ = self._request("?junk=1&limit=2&other=x")
        self.assertEqual(status, 200)

    def test_repeated_params_are_400_with_field(self) -> None:
        for query, field in (
                ("?after=1&after=2", "after"),
                ("?limit=1&limit=2", "limit")):
            status, body = self._request(query)
            self.assertEqual(status, 400, query)
            self.assertEqual(body["field"], field, query)

    def test_after_must_be_decimal_non_negative(self) -> None:
        for query in ("?after=-1", "?after=x", "?after=1.0",
                      "?after=%201", "?after=1%20", "?after=%2B1",
                      "?after=0x1", "?after=", "?after=1.5"):
            status, body = self._request(query)
            self.assertEqual(status, 400, query)
            self.assertEqual(body["field"], "after", query)

    def test_leading_zero_is_a_valid_decimal(self) -> None:
        self._commit(2)
        status, body = self._request("?after=01")
        self.assertEqual(status, 200)
        self.assertEqual([e["commit_seq"] for e in body["entries"]], [2])

    def test_limit_must_be_in_1_100(self) -> None:
        for query in ("?limit=0", "?limit=101", "?limit=-1",
                      "?limit=x", "?limit=1.0"):
            status, body = self._request(query)
            self.assertEqual(status, 400, query)
            self.assertEqual(body["field"], "limit", query)

    def test_tampered_sidecar_is_503_data_file(self) -> None:
        self._commit(2)
        with open(self.sidecar, encoding="utf-8") as handle:
            document = json.load(handle)
        document["entries"][0]["hash"] = "0" * 64
        with open(self.sidecar, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        status, body = self._request()
        self.assertEqual(status, 503)
        self.assertEqual(list(body), ["message", "field"])
        self.assertEqual(body["field"], "data_file")


class IntegrityHistoryPageMemoryHTTPTest(unittest.TestCase):
    """A purely in-memory server answers 409 even with valid paging params."""

    def setUp(self) -> None:
        from e2ee_backend.http_app import create_server
        self.server, self.service = create_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def test_memory_mode_is_409(self) -> None:
        connection = HTTPConnection("127.0.0.1", self.port, timeout=5)
        connection.request("GET",
                           "/v1/persistence/integrity/history/page")
        response = connection.getresponse()
        body = json.loads(response.read().decode("utf-8"))
        connection.close()
        self.assertEqual(response.status, 409)
        self.assertEqual(body["field"], "data_file")

    def test_bad_params_are_400_even_without_persistence(self) -> None:
        # Parameter validation (400) precedes the availability gate (409).
        connection = HTTPConnection("127.0.0.1", self.port, timeout=5)
        connection.request(
            "GET",
            "/v1/persistence/integrity/history/page?limit=0&after=1&after=2")
        response = connection.getresponse()
        body = json.loads(response.read().decode("utf-8"))
        connection.close()
        self.assertEqual(response.status, 400)
        self.assertEqual(body["field"], "after")


class IntegrityHistoryPageCLITest(unittest.TestCase):
    """The integrity-history-page subcommand prints one JSON line."""

    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.path = os.path.join(self.directory, "state.json")
        self.service = DeviceService()
        attach_persistence(self.service, self.path)
        from e2ee_backend.http_app import create_server
        self.server, _ = create_server("127.0.0.1", 0, self.service)
        self.port = self.server.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        for index in range(1, 4):
            self.service.store.add_device(Device("u", f"d{index}", "ik"))

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        shutil.rmtree(self.directory, ignore_errors=True)

    def _run(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "e2ee_backend",
             "--base-url", self.base_url,
             "integrity-history-page", *arguments],
            capture_output=True, text=True, timeout=15)

    @staticmethod
    def _single_line(stream: str) -> Dict[str, Any]:
        line = stream.strip()
        # Exactly one physical line.
        assert "\n" not in line
        return json.loads(line)

    def test_success_prints_one_compact_stdout_line(self) -> None:
        result = self._run("--after", "1", "--limit", "1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(result.stderr.strip())
        body = self._single_line(result.stdout)
        self.assertEqual(list(body),
                         ["commit_seq", "entries", "next_after", "has_more"])
        self.assertEqual([e["commit_seq"] for e in body["entries"]], [2])
        self.assertEqual(body["next_after"], 2)
        self.assertTrue(body["has_more"])
        # Compact JSON: no whitespace outside strings.
        self.assertNotIn(" ", result.stdout.strip())

    def test_default_arguments_page_from_zero(self) -> None:
        result = self._run()
        self.assertEqual(result.returncode, 0, result.stderr)
        body = self._single_line(result.stdout)
        self.assertEqual([e["commit_seq"] for e in body["entries"]],
                         [1, 2, 3])
        self.assertFalse(body["has_more"])

    def test_409_is_one_stderr_line_nonzero(self) -> None:
        # A different, purely in-memory server answers 409.
        from e2ee_backend.http_app import create_server
        server, _ = create_server("127.0.0.1", 0)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            result = subprocess.run(
                [sys.executable, "-m", "e2ee_backend",
                 "--base-url", f"http://127.0.0.1:{port}",
                 "integrity-history-page"],
                capture_output=True, text=True, timeout=15)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(result.stdout.strip())
        body = self._single_line(result.stderr)
        self.assertEqual(body["field"], "data_file")

    def test_400_is_one_stderr_line_nonzero(self) -> None:
        result = self._run("--limit", "0")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(result.stdout.strip())
        body = self._single_line(result.stderr)
        self.assertEqual(body["field"], "limit")

    def test_connection_error_is_one_stderr_line_field_server(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "e2ee_backend",
             "--base-url", "http://127.0.0.1:1",
             "integrity-history-page"],
            capture_output=True, text=True, timeout=15)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(result.stdout.strip())
        self.assertEqual(len(result.stderr.strip().splitlines()), 1)
        body = json.loads(result.stderr.strip())
        self.assertEqual(body["field"], "server")


if __name__ == "__main__":
    unittest.main()
