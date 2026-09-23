"""Integrity audit endpoint: ``GET /v1/persistence/integrity``.

The endpoint is only available when persistence is attached (a purely
in-memory server answers 409/field=data_file). Under the store lock it
re-reads the version=1 state file, applies the startup validation, and
compares the payload with the in-memory snapshot. Success is 200 with the
ordered keys ``commit_seq`` (the previous, non-negative generation),
``state_hash`` (SHA-256 lowercase hex of the canonical 17-section snapshot
without ``version``/``commit_seq``, keys sorted, compact JSON with
``ensure_ascii=False`` encoded as UTF-8) and ``consistent`` (true). Any
parse/version/semantic error and any generation or snapshot mismatch is
503/field=data_file and changes nothing — not memory, the file or its
inode, the cursors, or the commit generation.
"""
import base64
import hashlib
import json
import os
import shutil
import tempfile
import threading
import unittest
from http.client import HTTPConnection

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import x25519

from e2ee_backend.http_app import create_server
from e2ee_backend.persistence import (
    INTEGRITY_SECTION_ORDER,
    attach_persistence,
)
from e2ee_backend.service import DeviceService

_MAPPING_SECTIONS = ("messages", "used_nonces")


def _raw_key_b64() -> str:
    key = x25519.X25519PrivateKey.generate().public_key()
    raw = key.public_bytes(serialization.Encoding.Raw,
                           serialization.PublicFormat.Raw)
    return base64.b64encode(raw).decode()


def _register_payload(device_id: str, user_id: str = "u1") -> dict:
    return {
        "user_id": user_id,
        "device_id": device_id,
        "identity_key": _raw_key_b64(),
        "signed_prekeys": [{"key_id": "k1", "public_key": _raw_key_b64()}],
    }


def _canonical(document: dict) -> dict:
    """Project a state document onto the canonical 17-section snapshot."""
    payload = {key: value for key, value in document.items()
               if key not in ("version", "commit_seq")}
    return {name: payload.get(name, {} if name in _MAPPING_SECTIONS else [])
            for name in INTEGRITY_SECTION_ORDER}


def _expected_hash(document: dict) -> str:
    canonical = json.dumps(_canonical(document), sort_keys=True,
                           separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class _ServerBase(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.path = os.path.join(self.directory, "state.json")
        self.service = DeviceService()
        self.state_store = attach_persistence(self.service, self.path)
        self.server, _ = create_server("127.0.0.1", 0, service=self.service)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        shutil.rmtree(self.directory, ignore_errors=True)

    def _request(self, method: str, path: str, body: object = None):
        connection = HTTPConnection("127.0.0.1", self.port, timeout=5)
        payload = json.dumps(body) if body is not None else None
        headers = {"Content-Type": "application/json"} if payload else {}
        connection.request(method, path, body=payload, headers=headers)
        response = connection.getresponse()
        raw = response.read().decode("utf-8")
        connection.close()
        return response.status, raw

    def _integrity(self):
        return self._request("GET", "/v1/persistence/integrity")

    def _read_file(self) -> bytes:
        with open(self.path, "rb") as handle:
            return handle.read()

    def _read_doc(self) -> dict:
        with open(self.path, encoding="utf-8") as handle:
            return json.load(handle)


class IntegritySuccessTest(_ServerBase):
    def test_empty_state_audits_consistent_at_generation_zero(self) -> None:
        status, raw = self._integrity()
        self.assertEqual(status, 200)
        # Response key order: commit_seq, state_hash, consistent.
        self.assertTrue(raw.startswith('{"commit_seq":'))
        body = json.loads(raw)
        self.assertEqual(list(body.keys()),
                         ["commit_seq", "state_hash", "consistent"])
        self.assertEqual(body["commit_seq"], 0)
        self.assertIs(body["consistent"], True)
        self.assertEqual(body["state_hash"], _expected_hash(self._read_doc()))

    def test_hash_and_generation_track_committed_mutations(self) -> None:
        for index in range(2):
            status, _ = self._request("POST", "/v1/devices",
                                      _register_payload(f"d{index}"))
            self.assertEqual(status, 201)
            status, raw = self._integrity()
            self.assertEqual(status, 200)
            body = json.loads(raw)
            self.assertEqual(body["commit_seq"], index + 1)
            self.assertIs(body["consistent"], True)
            self.assertEqual(body["state_hash"],
                             _expected_hash(self._read_doc()))

    def test_legacy_file_without_commit_seq_audits_as_generation_zero(
            self) -> None:
        # Rewrite the formal file as a legacy document: no commit_seq, no
        # key_events section. Restart so the service resumes from it.
        document = self._read_doc()
        del document["commit_seq"]
        del document["key_events"]
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        service = DeviceService()
        attach_persistence(service, self.path)
        self.service = service
        self.server, _ = create_server("127.0.0.1", 0, service=service)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()

        status, raw = self._integrity()
        self.assertEqual(status, 200)
        body = json.loads(raw)
        self.assertEqual(body["commit_seq"], 0)
        self.assertIs(body["consistent"], True)
        self.assertEqual(body["state_hash"], _expected_hash(self._read_doc()))

    def test_audit_changes_nothing(self) -> None:
        status, _ = self._request("POST", "/v1/devices",
                                  _register_payload("d1"))
        self.assertEqual(status, 201)
        before_bytes = self._read_file()
        before_inode = os.stat(self.path).st_ino
        before_snapshot = self.service.store.snapshot_state()
        before_generation = self.state_store.commit_seq

        status, _ = self._integrity()
        self.assertEqual(status, 200)

        self.assertEqual(self._read_file(), before_bytes)
        self.assertEqual(os.stat(self.path).st_ino, before_inode)
        self.assertEqual(self.service.store.snapshot_state(), before_snapshot)
        self.assertEqual(self.state_store.commit_seq, before_generation)


class IntegrityUnavailableTest(unittest.TestCase):
    """A purely in-memory server has no state file to audit."""

    def setUp(self) -> None:
        self.server, _ = create_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def test_in_memory_server_answers_409_data_file(self) -> None:
        connection = HTTPConnection("127.0.0.1", self.port, timeout=5)
        connection.request("GET", "/v1/persistence/integrity")
        response = connection.getresponse()
        raw = response.read().decode("utf-8")
        connection.close()
        self.assertEqual(response.status, 409)
        # Error key order: message, field.
        self.assertTrue(raw.startswith('{"message":'))
        body = json.loads(raw)
        self.assertEqual(list(body.keys()), ["message", "field"])
        self.assertIsInstance(body["message"], str)
        self.assertEqual(body["field"], "data_file")


class IntegrityFailureTest(_ServerBase):
    """Every audit failure is 503/field=data_file and changes nothing."""

    def _assert_503_data_file(self, raw: str, status: int) -> None:
        self.assertEqual(status, 503)
        self.assertTrue(raw.startswith('{"message":'))
        body = json.loads(raw)
        self.assertEqual(list(body.keys()), ["message", "field"])
        self.assertIsInstance(body["message"], str)
        self.assertEqual(body["field"], "data_file")

    def _assert_nothing_changed(self, before_bytes: bytes,
                                before_inode: int,
                                before_snapshot: dict,
                                before_generation: int) -> None:
        self.assertEqual(self._read_file(), before_bytes)
        self.assertEqual(os.stat(self.path).st_ino, before_inode)
        self.assertEqual(self.service.store.snapshot_state(), before_snapshot)
        self.assertEqual(self.state_store.commit_seq, before_generation)

    def _tamper_and_audit(self, tamper) -> None:
        status, _ = self._request("POST", "/v1/devices",
                                  _register_payload("d1"))
        self.assertEqual(status, 201)
        before_snapshot = self.service.store.snapshot_state()
        before_generation = self.state_store.commit_seq
        tamper()
        before_bytes = self._read_file()
        before_inode = os.stat(self.path).st_ino

        status, raw = self._integrity()
        self._assert_503_data_file(raw, status)
        self._assert_nothing_changed(before_bytes, before_inode,
                                     before_snapshot, before_generation)

    def test_unparseable_file_is_503(self) -> None:
        def tamper() -> None:
            with open(self.path, "wb") as handle:
                handle.write(b"{not json")

        self._tamper_and_audit(tamper)

    def test_wrong_version_is_503(self) -> None:
        def tamper() -> None:
            document = self._read_doc()
            document["version"] = 2
            with open(self.path, "w", encoding="utf-8") as handle:
                json.dump(document, handle)

        self._tamper_and_audit(tamper)

    def test_malformed_commit_seq_is_503(self) -> None:
        def tamper() -> None:
            document = self._read_doc()
            document["commit_seq"] = "1"
            with open(self.path, "w", encoding="utf-8") as handle:
                json.dump(document, handle)

        self._tamper_and_audit(tamper)

    def test_generation_mismatch_is_503(self) -> None:
        def tamper() -> None:
            document = self._read_doc()
            document["commit_seq"] += 1
            with open(self.path, "w", encoding="utf-8") as handle:
                json.dump(document, handle)

        self._tamper_and_audit(tamper)

    def test_snapshot_mismatch_is_503(self) -> None:
        def tamper() -> None:
            document = self._read_doc()
            document["devices"][0]["revoked"] = True
            with open(self.path, "w", encoding="utf-8") as handle:
                json.dump(document, handle)

        self._tamper_and_audit(tamper)

    def test_semantically_malformed_payload_is_503(self) -> None:
        def tamper() -> None:
            document = self._read_doc()
            document["devices"] = "not-a-list"
            with open(self.path, "w", encoding="utf-8") as handle:
                json.dump(document, handle)

        self._tamper_and_audit(tamper)

    def test_missing_file_is_503(self) -> None:
        status, _ = self._request("POST", "/v1/devices",
                                  _register_payload("d1"))
        self.assertEqual(status, 201)
        before_snapshot = self.service.store.snapshot_state()
        before_generation = self.state_store.commit_seq
        os.unlink(self.path)

        status, raw = self._integrity()
        self._assert_503_data_file(raw, status)
        self.assertFalse(os.path.exists(self.path))
        self.assertEqual(self.service.store.snapshot_state(), before_snapshot)
        self.assertEqual(self.state_store.commit_seq, before_generation)


if __name__ == "__main__":
    unittest.main()
