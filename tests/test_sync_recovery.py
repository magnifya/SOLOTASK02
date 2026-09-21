"""Offline recovery and durable-transaction tests for group-session sync.

Covers:
* restart resumption from per-device saved cursors (cursor/updated_at/messages)
  and query-only explicit ``after``;
* startup refusal when ``group_sync_cursors`` is present but malformed in
  shape, type, range, key or referential structure -- without overwriting the
  old file or leaving a temp file behind;
* 503/field=data_file with full in-memory rollback when the temp-file write,
  fsync, or atomic replace fails during a default sync, a forward checkpoint
  or a device revocation;
* linearization of concurrent sync/checkpoint vs device revocation;
* CLI failure contract (single-line stderr JSON, non-zero exit).
"""
import argparse
import contextlib
import io
import json
import os
import tempfile
import threading
import unittest
from http.client import HTTPConnection

from e2ee_backend import cli
from e2ee_backend.http_app import create_server
from e2ee_backend.models import Device
from e2ee_backend.persistence import StateFileError, attach_persistence
from e2ee_backend.service import DeviceService
from e2ee_backend.storage import PersistenceError


def _build_world(service: DeviceService, with_carol: bool = False) -> str:
    """Register devices, a group, a frozen group session and three messages."""
    device_ids = ["creator", "alice", "bob"]
    if with_carol:
        device_ids.append("carol")
    for device_id in device_ids:
        service.store.add_device(Device("u", device_id, "ik"))
    service.create_group({
        "group_id": "g1", "creator_device_id": "creator",
        "member_device_ids": ["alice", "bob"]})
    session = service.create_group_session({
        "group_id": "g1", "initiator_device_id": "creator",
        "ephemeral_key": "epk"})
    sid = session["session_id"]
    for sequence in range(1, 4):
        service.post_message({
            "session_id": sid, "sender_device_id": "creator",
            "message_id": f"m{sequence}", "sequence": sequence,
            "nonce": f"n{sequence}", "ciphertext": "ct"})
    return sid


class RestartRecoveryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.path = os.path.join(self.directory, "state.json")
        self.service = DeviceService()
        self.sid = _build_world(self.service)
        attach_persistence(self.service, self.path)

    def _restart(self) -> DeviceService:
        restored = DeviceService()
        attach_persistence(restored, self.path)
        return restored

    def test_default_sync_resumes_from_saved_cursor_after_restart(self) -> None:
        body = self.service.sync_group_messages(self.sid, "alice", None, 2)
        self.assertEqual([m["sequence"] for m in body["messages"]], [1, 2])
        self.service.sync_group_checkpoint(
            self.sid, {"device_id": "bob", "cursor": 1})

        restored = self._restart()
        # Alice resumes past sequences 1-2; only m3 is returned.
        body = restored.sync_group_messages(self.sid, "alice", None, 100)
        self.assertEqual([m["sequence"] for m in body["messages"]], [3])
        self.assertEqual(body["next_cursor"], 3)
        self.assertFalse(body["has_more"])
        # The next default call is an empty page that does not move the cursor.
        body = restored.sync_group_messages(self.sid, "alice", None, 100)
        self.assertEqual(body["messages"], [])
        self.assertEqual(body["next_cursor"], 3)
        # Bob's independently checkpointed cursor survived too.
        record = restored.store._group_sync_cursors[(self.sid, "bob")]
        self.assertEqual(record.cursor, 1)

    def test_explicit_after_after_restart_is_query_only(self) -> None:
        self.service.sync_group_messages(self.sid, "alice", None, 100)
        restored = self._restart()

        body = restored.sync_group_messages(self.sid, "alice", 0, 100)
        self.assertEqual([m["sequence"] for m in body["messages"]], [1, 2, 3])
        self.assertEqual(body["next_cursor"], 3)
        # The saved cursor was neither read nor written by the explicit query.
        self.assertEqual(
            restored.store._group_sync_cursors[(self.sid, "alice")].cursor, 3)
        body = restored.sync_group_messages(self.sid, "alice", None, 100)
        self.assertEqual(body["messages"], [])
        self.assertEqual(body["next_cursor"], 3)
        with open(self.path, encoding="utf-8") as handle:
            document = json.load(handle)
        alice_records = [row for row in document["group_sync_cursors"]
                         if row["device_id"] == "alice"]
        self.assertEqual(len(alice_records), 1)
        self.assertEqual(alice_records[0]["cursor"], 3)

    def test_cursor_timestamp_and_messages_survive_restart(self) -> None:
        body, status = self.service.sync_group_checkpoint(
            self.sid, {"device_id": "alice", "cursor": 2})
        self.assertEqual(status, 201)
        timestamp = body["updated_at"]

        restored = self._restart()
        record = restored.store._group_sync_cursors[(self.sid, "alice")]
        self.assertEqual(record.cursor, 2)
        self.assertEqual(record.updated_at, timestamp)
        # Repeating the same checkpoint after restart stays a 200 no-op and
        # keeps the timestamp recovered from the file.
        body, status = restored.sync_group_checkpoint(
            self.sid, {"device_id": "alice", "cursor": 2})
        self.assertEqual(status, 200)
        self.assertEqual(body["updated_at"], timestamp)
        # The message stream backing the pages is intact, in strict order.
        page = restored.sync_group_messages(self.sid, "bob", 0, 100)
        self.assertEqual([m["sequence"] for m in page["messages"]], [1, 2, 3])
        self.assertEqual([m["message_id"] for m in page["messages"]],
                         ["m1", "m2", "m3"])

    def test_per_device_cursors_independent_after_restart(self) -> None:
        self.service.sync_group_messages(self.sid, "alice", None, 1)
        self.service.sync_group_messages(self.sid, "bob", None, 100)
        restored = self._restart()

        body = restored.sync_group_messages(self.sid, "alice", None, 100)
        self.assertEqual([m["sequence"] for m in body["messages"]], [2, 3])
        body = restored.sync_group_messages(self.sid, "bob", None, 100)
        self.assertEqual(body["messages"], [])
        self.assertEqual(body["next_cursor"], 3)
        self.assertEqual(
            restored.store._group_sync_cursors[(self.sid, "alice")].cursor, 3)
        self.assertEqual(
            restored.store._group_sync_cursors[(self.sid, "bob")].cursor, 3)


class MalformedCursorSectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.path = os.path.join(self.directory, "state.json")
        service = DeviceService()
        self.sid = _build_world(service, with_carol=True)
        attach_persistence(service, self.path)
        service.sync_group_messages(self.sid, "alice", None, 100)
        with open(self.path, encoding="utf-8") as handle:
            self.base_document = json.load(handle)

    def _assert_refused_without_touching_file(self, document) -> None:
        raw = json.dumps(document).encode("utf-8")
        with open(self.path, "wb") as handle:
            handle.write(raw)
        with self.assertRaises(StateFileError):
            attach_persistence(DeviceService(), self.path)
        # The malformed file must survive startup refusal byte-for-byte.
        with open(self.path, "rb") as handle:
            self.assertEqual(handle.read(), raw)
        self.assertEqual(
            [name for name in os.listdir(self.directory)
             if name.endswith(".tmp")], [])

    def _cursor_row(self, **overrides):
        row = {"session_id": self.sid, "device_id": "alice",
               "cursor": 2, "updated_at": "2026-01-01T00:00:00+00:00"}
        row.update(overrides)
        return row

    def test_shape_type_range_and_duplicate_failures(self) -> None:
        bad_sections = [
            "nope",
            [self._cursor_row(cursor="2")],
            [self._cursor_row(cursor=True)],
            [self._cursor_row(cursor=-1)],
            [self._cursor_row(updated_at=5)],
            [{"session_id": self.sid, "device_id": "alice",
              "cursor": 2}],
            [self._cursor_row(session_id="")],
            [self._cursor_row(device_id="")],
            [self._cursor_row(), self._cursor_row(updated_at="later")],
        ]
        for section in bad_sections:
            document = dict(self.base_document)
            document["group_sync_cursors"] = section
            self._assert_refused_without_touching_file(document)

    def test_referential_failures(self) -> None:
        # Cursor points at a session that does not exist.
        document = dict(self.base_document)
        document["group_sync_cursors"] = [self._cursor_row(session_id="ghost")]
        self._assert_refused_without_touching_file(document)
        # Cursor points at a device that was never registered.
        document = dict(self.base_document)
        document["group_sync_cursors"] = [self._cursor_row(device_id="ghost")]
        self._assert_refused_without_touching_file(document)
        # carol is registered but outside the frozen member set.
        document = dict(self.base_document)
        document["group_sync_cursors"] = [self._cursor_row(device_id="carol")]
        self._assert_refused_without_touching_file(document)
        # Cursor past the session's largest stored sequence.
        document = dict(self.base_document)
        document["group_sync_cursors"] = [self._cursor_row(cursor=99)]
        self._assert_refused_without_touching_file(document)

    def test_empty_session_only_admits_zero_cursor(self) -> None:
        document = dict(self.base_document)
        document["messages"] = {}
        document["group_sync_cursors"] = [self._cursor_row(cursor=1)]
        self._assert_refused_without_touching_file(document)
        # cursor 0 on the empty session is structurally fine.
        document["group_sync_cursors"] = [self._cursor_row(cursor=0)]
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        restored = DeviceService()
        attach_persistence(restored, self.path)
        self.assertEqual(
            restored.store._group_sync_cursors[(self.sid, "alice")].cursor, 0)

    def test_old_file_without_section_still_loads_empty(self) -> None:
        document = dict(self.base_document)
        del document["group_sync_cursors"]
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        restored = DeviceService()
        attach_persistence(restored, self.path)
        self.assertEqual(restored.store._group_sync_cursors, {})


class PersistenceFailureRollbackTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.path = os.path.join(self.directory, "state.json")
        self.service = DeviceService()
        self.sid = _build_world(self.service)
        attach_persistence(self.service, self.path)
        self.server, _ = create_server("127.0.0.1", 0, self.service)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _request(self, method: str, url: str, body: object = None):
        connection = HTTPConnection("127.0.0.1", self.port, timeout=5)
        payload = json.dumps(body) if body is not None else None
        headers = {"Content-Type": "application/json"} if payload else {}
        connection.request(method, url, body=payload, headers=headers)
        response = connection.getresponse()
        data = response.read().decode("utf-8")
        connection.close()
        return response.status, json.loads(data)

    def _sync_url(self, device_id: str, after: str = "") -> str:
        suffix = f"&after={after}" if after else ""
        return (f"/v1/group-sessions/{self.sid}/sync"
                f"?device_id={device_id}&limit=100{suffix}")

    @contextlib.contextmanager
    def _failing_fsync(self):
        original = os.fsync

        def fail(_fileno):
            raise OSError("simulated fsync failure")

        os.fsync = fail
        try:
            yield
        finally:
            os.fsync = original

    def test_failed_sync_advance_returns_503_and_rolls_back(self) -> None:
        with open(self.path, "rb") as handle:
            before = handle.read()
        with self._failing_fsync():
            status, body = self._request("GET", self._sync_url("alice"))
        self.assertEqual(status, 503)
        self.assertEqual(body["field"], "data_file")
        # Memory: the cursor advance never happened.
        self.assertNotIn((self.sid, "alice"),
                         self.service.store._group_sync_cursors)
        # File: previous document retained; temp file cleaned up.
        with open(self.path, "rb") as handle:
            self.assertEqual(handle.read(), before)
        self.assertEqual(
            [n for n in os.listdir(self.directory) if n.endswith(".tmp")], [])
        # After the storage heals, the same request works and persists.
        status, body = self._request("GET", self._sync_url("alice"))
        self.assertEqual(status, 200)
        self.assertEqual([m["sequence"] for m in body["messages"]], [1, 2, 3])
        self.assertEqual(
            self.service.store._group_sync_cursors[(self.sid, "alice")].cursor,
            3)

    def test_failed_checkpoint_returns_503_and_rolls_back(self) -> None:
        with open(self.path, "rb") as handle:
            before = handle.read()
        with self._failing_fsync():
            status, body = self._request(
                "POST", f"/v1/group-sessions/{self.sid}/sync/checkpoint",
                {"device_id": "alice", "cursor": 2})
        self.assertEqual(status, 503)
        self.assertEqual(body["field"], "data_file")
        self.assertNotIn((self.sid, "alice"),
                         self.service.store._group_sync_cursors)
        with open(self.path, "rb") as handle:
            self.assertEqual(handle.read(), before)
        # The failed 201 leaves no record, so the same checkpoint is still 201.
        status, body = self._request(
            "POST", f"/v1/group-sessions/{self.sid}/sync/checkpoint",
            {"device_id": "alice", "cursor": 2})
        self.assertEqual(status, 201)
        self.assertEqual(body["cursor"], 2)

    def test_failed_revocation_returns_503_and_rolls_back(self) -> None:
        with open(self.path, "rb") as handle:
            before = handle.read()
        with self._failing_fsync():
            status, body = self._request("POST", "/v1/devices/bob/revoke")
        self.assertEqual(status, 503)
        self.assertEqual(body["field"], "data_file")
        # Memory: bob is still an active syncing device.
        self.assertFalse(self.service.store.find_by_device_id("bob").revoked)
        with open(self.path, "rb") as handle:
            self.assertEqual(handle.read(), before)
        status, body = self._request("GET", self._sync_url("bob"))
        self.assertEqual(status, 200)
        self.assertEqual([m["sequence"] for m in body["messages"]], [1, 2, 3])
        # And the revoked flag was never persisted: a restart agrees.
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        self.assertFalse(restarted.store.find_by_device_id("bob").revoked)

    def test_readonly_and_empty_page_requests_do_not_write(self) -> None:
        # Advance alice to the end successfully first.
        self._request("GET", self._sync_url("alice"))
        with open(self.path, "rb") as handle:
            before = handle.read()
        with self._failing_fsync():
            # Explicit after is query-only even with a broken durable layer.
            status, body = self._request("GET", self._sync_url("alice", "0"))
            self.assertEqual(status, 200)
            self.assertEqual(len(body["messages"]), 3)
            # An empty page at the saved cursor does not advance or write.
            status, body = self._request("GET", self._sync_url("alice"))
            self.assertEqual(status, 200)
            self.assertEqual(body["messages"], [])
            self.assertEqual(body["next_cursor"], 3)
        with open(self.path, "rb") as handle:
            self.assertEqual(handle.read(), before)

    def test_business_rule_failure_does_not_invoke_persistence(self) -> None:
        calls = []
        original = self.service.store.on_change

        def counting_hook():
            calls.append(1)
            return original()

        self.service.store.on_change = counting_hook
        # Out-of-range checkpoint: 409/cursor, no persistence attempted.
        status, body = self._request(
            "POST", f"/v1/group-sessions/{self.sid}/sync/checkpoint",
            {"device_id": "alice", "cursor": 99})
        self.assertEqual((status, body["field"]), (409, "cursor"))
        # Sync by a non-frozen member: 409/device_id, no persistence.
        status, body = self._request("GET", self._sync_url("carol"))
        self.assertEqual((status, body["field"]), (409, "device_id"))
        self.assertEqual(calls, [])


class StoreTransactionRollbackTest(unittest.TestCase):
    def test_persistence_error_restores_memory_without_persistence_attached(
            self) -> None:
        service = DeviceService()
        sid = _build_world(service)
        # No on_change hook: plain in-memory behavior is unchanged, the error
        # can never surface.
        body = service.sync_group_messages(sid, "alice", None, 100)
        self.assertEqual(len(body["messages"]), 3)
        self.assertIsNone(service.store.on_change)

    def test_hook_failure_raises_persistence_error_and_restores_snapshot(
            self) -> None:
        service = DeviceService()
        sid = _build_world(service)

        def fail():
            raise RuntimeError("disk gone")

        service.store.on_change = fail
        with self.assertRaises(PersistenceError):
            service.store.group_sync_page(sid, "alice", None, 100)
        self.assertNotIn((sid, "alice"),
                         service.store._group_sync_cursors)
        with self.assertRaises(PersistenceError):
            service.store.group_sync_checkpoint(sid, "alice", 2)
        self.assertNotIn((sid, "alice"),
                         service.store._group_sync_cursors)
        with self.assertRaises(PersistenceError):
            service.store.revoke_device("bob")
        self.assertFalse(service.store.find_by_device_id("bob").revoked)
        # A replaced hook sees a consistent, pre-failure store.
        service.store.on_change = None
        body = service.sync_group_messages(sid, "alice", None, 100)
        self.assertEqual([m["sequence"] for m in body["messages"]], [1, 2, 3])


class ConcurrentRevocationLinearizationTest(unittest.TestCase):
    def test_syncs_and_revocation_linearize_to_success_or_conflict(self) -> None:
        directory = tempfile.mkdtemp()
        path = os.path.join(directory, "state.json")
        service = DeviceService()
        sid = _build_world(service)
        attach_persistence(service, path)
        server, _ = create_server("127.0.0.1", 0, service)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        outcomes = []
        outcomes_lock = threading.Lock()

        def request(method, url, body=None):
            connection = HTTPConnection("127.0.0.1", port, timeout=10)
            payload = json.dumps(body) if body is not None else None
            headers = {"Content-Type": "application/json"} if payload else {}
            connection.request(method, url, body=payload, headers=headers)
            response = connection.getresponse()
            data = json.loads(response.read().decode("utf-8"))
            connection.close()
            return response.status, data

        def sync_device(device_id):
            try:
                while True:
                    status, body = request(
                        "GET",
                        f"/v1/group-sessions/{sid}/sync"
                        f"?device_id={device_id}&limit=1")
                    with outcomes_lock:
                        outcomes.append(("sync", status, body.get("field")))
                    if status == 409:
                        self.assertEqual(body.get("field"), "device_id")
                        return
                    self.assertEqual(status, 200, body)
                    if not body["messages"]:
                        return
            except Exception as error:  # pragma: no cover - fail the test
                outcomes.append(("sync-exc", repr(error), None))

        def checkpoint_device(device_id):
            try:
                for cursor in (1, 2, 3):
                    status, body = request(
                        "POST",
                        f"/v1/group-sessions/{sid}/sync/checkpoint",
                        {"device_id": device_id, "cursor": cursor})
                    with outcomes_lock:
                        outcomes.append(
                            ("checkpoint", status, body.get("field")))
                    if status == 409:
                        # Racing the revocation: the only legal conflict once
                        # revoke is linearized first names device_id.
                        self.assertEqual(body.get("field"), "device_id", body)
                        return
                    self.assertEqual(status, 201, body)
            except Exception as error:  # pragma: no cover - fail the test
                outcomes.append(("checkpoint-exc", repr(error), None))

        def revoke_device(device_id):
            status, body = request("POST", f"/v1/devices/{device_id}/revoke")
            with outcomes_lock:
                outcomes.append(("revoke", status, body.get("field")))
            self.assertEqual(status, 200, body)

        # Alice only ever default-syncs; bob only ever checkpoints, so no
        # 409/cursor can arise from two advancers racing the same cursor.
        syncer = threading.Thread(target=sync_device, args=("alice",))
        checkpointer = threading.Thread(target=checkpoint_device, args=("bob",))
        syncer.start()
        checkpointer.start()

        # Revoke each device after a short delay to race in-flight requests.
        def revoke_later():
            for device_id in ("alice", "bob"):
                threading.Event().wait(0.01)
                revoke_device(device_id)

        revoker = threading.Thread(target=revoke_later)
        revoker.start()
        syncer.join(timeout=10)
        checkpointer.join(timeout=10)
        revoker.join(timeout=5)

        # No 5xx and no exception outcomes: every operation was either an
        # ordered success or a single linearized 409/device_id.
        self.assertFalse(
            [row for row in outcomes if row[0].endswith("exc")], outcomes)
        self.assertTrue(
            all(200 <= row[1] < 500 for row in outcomes), outcomes)
        self.assertTrue(
            all(row[2] in (None, "device_id") for row in outcomes), outcomes)
        server.shutdown()
        server.server_close()

        # The persisted file is a self-consistent restart point: it parses,
        # loads through the strict referential checks, cursors stay in range
        # and per-device, and the revocations survived.
        restarted = DeviceService()
        attach_persistence(restarted, path)
        for key, record in restarted.store._group_sync_cursors.items():
            self.assertEqual(key[0], sid)
            self.assertIn(record.cursor, range(0, 4))
        self.assertTrue(restarted.store.find_by_device_id("alice").revoked)
        self.assertTrue(restarted.store.find_by_device_id("bob").revoked)


class CLIPersistenceFailureContractTest(unittest.TestCase):
    def test_503_is_single_line_stderr_json_with_nonzero_exit(self) -> None:
        original = cli._request_json
        cli._request_json = (
            lambda *a, **k: (503, {"message": "state file unavailable: boom",
                                   "field": "data_file"}))
        try:
            stdout, stderr = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(stdout), \
                    contextlib.redirect_stderr(stderr):
                code = cli.cmd_sync_checkpoint(argparse.Namespace(
                    base_url="http://127.0.0.1:1", session_id="s1",
                    device_id="alice", cursor=2))
        finally:
            cli._request_json = original
        self.assertEqual(code, 1)
        self.assertFalse(stdout.getvalue())
        line = stderr.getvalue().strip()
        self.assertEqual(line.count("\n"), 0)
        self.assertEqual(json.loads(line)["field"], "data_file")


if __name__ == "__main__":
    unittest.main()
