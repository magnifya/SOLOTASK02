"""Tests for durable transaction semantics, failure recovery and restart.

Covers:
* a failed durable write rolls the in-memory store back to the last
  persisted state, leaves the old file and its inode in place, cleans the
  temporary file, and surfaces PersistenceUnavailable;
* the HTTP layer answers 503/field=data_file for such failures;
* a restarted service resumes default syncs from each device's saved cursor,
  while explicit ``after`` stays a pure query;
* malformed ``group_sync_cursors`` sections (records, types, ranges,
  duplicate keys, dangling/wrong associations) make startup refuse without
  touching the file;
* concurrent revocation is linearized against cursor advances, and when the
  durable write fails neither memory nor the file advances.
"""
import json
import os
import shutil
import tempfile
import threading
import unittest
from copy import deepcopy
from http.client import HTTPConnection
from typing import Any, Dict, Tuple

import e2ee_backend.persistence as persistence_mod
from e2ee_backend.http_app import create_server
from e2ee_backend.models import Device
from e2ee_backend.persistence import (
    PersistenceUnavailable,
    StateFileError,
    attach_persistence,
)
from e2ee_backend.service import DeviceService, ServiceError


def build_fixture(directory: str,
                  with_carol: bool = False,
                  carol_member: bool = False,
                  with_empty_session: bool = False
                  ) -> Tuple[DeviceService, Any, str, str]:
    """Create a persisted service with one group session holding seq 1..3.

    Returns ``(service, state_store, path, session_id)``. The frozen members
    are creator/alice/bob (plus carol when *carol_member*); carol is merely
    registered when *with_carol* is set without *carol_member*.
    """
    path = os.path.join(directory, "state.json")
    service = DeviceService()
    state_store = attach_persistence(service, path)
    device_ids = ["creator", "alice", "bob"] + (["carol"] if with_carol else [])
    for device_id in device_ids:
        service.store.add_device(Device("u", device_id, "ik"))
    members = ["alice", "bob"] + (["carol"] if carol_member else [])
    service.create_group({
        "group_id": "g1", "creator_device_id": "creator",
        "member_device_ids": members})
    session = service.create_group_session({
        "group_id": "g1", "initiator_device_id": "creator",
        "ephemeral_key": "epk"})
    sid = session["session_id"]
    for sequence in range(1, 4):
        service.post_message({
            "session_id": sid, "sender_device_id": "creator",
            "message_id": f"m{sequence}", "sequence": sequence,
            "nonce": f"n{sequence}", "ciphertext": "ct"})
    if with_empty_session:
        service.create_group({"group_id": "g2",
                              "creator_device_id": "creator",
                              "member_device_ids": ["alice"]})
        service.create_group_session({"group_id": "g2",
                                      "initiator_device_id": "creator",
                                      "ephemeral_key": "epk2"})
    return service, state_store, path, sid


class WriteFailureRollbackTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.service, self.state_store, self.path, self.sid = build_fixture(
            self.directory)
        with open(self.path, "rb") as handle:
            self.good_bytes = handle.read()
        self.good_stat = os.stat(self.path)

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def _tmp_files(self):
        return [name for name in os.listdir(self.directory)
                if name.endswith(".tmp")]

    def _fail_writes(self) -> None:
        def raise_oserror(state):  # noqa: ANN001 - mimics JsonStateStore.save
            raise OSError("simulated disk failure")

        self.state_store.save = raise_oserror  # type: ignore[assignment]

    def _assert_file_unchanged(self) -> None:
        with open(self.path, "rb") as handle:
            self.assertEqual(handle.read(), self.good_bytes)
        # os.replace never ran: the same inode still holds the old document.
        self.assertEqual(os.stat(self.path).st_ino, self.good_stat.st_ino)
        self.assertEqual(self._tmp_files(), [])

    def test_default_sync_advance_rolls_back(self) -> None:
        self._fail_writes()
        with self.assertRaises(PersistenceUnavailable):
            self.service.sync_group_messages(self.sid, "alice", None, 100)
        self.assertNotIn((self.sid, "alice"),
                         self.service.store._group_sync_cursors)
        self._assert_file_unchanged()

    def test_checkpoint_advance_rolls_back(self) -> None:
        self._fail_writes()
        with self.assertRaises(PersistenceUnavailable):
            self.service.sync_group_checkpoint(
                self.sid, {"device_id": "alice", "cursor": 2})
        self.assertNotIn((self.sid, "alice"),
                         self.service.store._group_sync_cursors)
        self._assert_file_unchanged()

    def test_device_revocation_rolls_back(self) -> None:
        self._fail_writes()
        with self.assertRaises(PersistenceUnavailable):
            self.service.revoke_device("alice")
        self.assertFalse(
            self.service.store.find_by_device_id("alice").revoked)
        self._assert_file_unchanged()

    def test_failed_sync_does_not_block_later_repaired_sync(self) -> None:
        self._fail_writes()
        with self.assertRaises(PersistenceUnavailable):
            self.service.sync_group_messages(self.sid, "alice", None, 100)
        # Repair the disk: the rolled-back store still starts at cursor 0.
        del self.state_store.save
        body = self.service.sync_group_messages(self.sid, "alice", None, 100)
        self.assertEqual([m["sequence"] for m in body["messages"]], [1, 2, 3])
        with open(self.path, encoding="utf-8") as handle:
            document = json.load(handle)
        cursors = {(c["session_id"], c["device_id"]): c["cursor"]
                   for c in document["group_sync_cursors"]}
        self.assertEqual(cursors[(self.sid, "alice")], 3)

    def test_replace_failure_keeps_old_file_and_removes_tmp(self) -> None:
        real_replace = persistence_mod.os.replace

        def fail_replace(src, dst):  # noqa: ANN001
            raise OSError("simulated replace failure")

        persistence_mod.os.replace = fail_replace
        try:
            with self.assertRaises(PersistenceUnavailable):
                self.service.revoke_device("alice")
        finally:
            persistence_mod.os.replace = real_replace
        self.assertFalse(
            self.service.store.find_by_device_id("alice").revoked)
        self._assert_file_unchanged()
        # The store keeps working after the repaired replace.
        body = self.service.revoke_device("alice")
        self.assertEqual(body["revoked"], True)


class DataFileHTTP503Test(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.service, self.state_store, _path, self.sid = build_fixture(
            self.directory)
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
        data = response.read().decode("utf-8")
        connection.close()
        return response.status, json.loads(data)

    def _fail_writes(self) -> None:
        def raise_oserror(state):  # noqa: ANN001
            raise OSError("simulated disk failure")

        self.state_store.save = raise_oserror  # type: ignore[assignment]

    def test_failed_checkpoint_is_503_data_file(self) -> None:
        self._fail_writes()
        status, body = self._request(
            "POST", f"/v1/group-sessions/{self.sid}/sync/checkpoint",
            {"device_id": "alice", "cursor": 2})
        self.assertEqual(status, 503)
        self.assertEqual(body["field"], "data_file")
        self.assertNotIn((self.sid, "alice"),
                         self.service.store._group_sync_cursors)

    def test_failed_default_sync_is_503_and_rolled_back(self) -> None:
        self._fail_writes()
        status, body = self._request(
            "GET", f"/v1/group-sessions/{self.sid}/sync?device_id=alice")
        self.assertEqual(status, 503)
        self.assertEqual(body["field"], "data_file")
        # Repaired retry starts from 0: the failed request advanced nothing.
        del self.state_store.save
        status, body = self._request(
            "GET", f"/v1/group-sessions/{self.sid}/sync?device_id=alice")
        self.assertEqual(status, 200)
        self.assertEqual([m["sequence"] for m in body["messages"]], [1, 2, 3])

    def test_failed_group_create_is_503_then_succeeds(self) -> None:
        self._fail_writes()
        status, body = self._request("POST", "/v1/groups", {
            "group_id": "g2", "creator_device_id": "creator",
            "member_device_ids": ["alice"]})
        self.assertEqual(status, 503)
        self.assertEqual(body["field"], "data_file")
        del self.state_store.save
        status, body = self._request("POST", "/v1/groups", {
            "group_id": "g2", "creator_device_id": "creator",
            "member_device_ids": ["alice"]})
        self.assertEqual(status, 201)
        self.assertEqual(body["group_id"], "g2")


class RestartCursorResumeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        service, _store, self.path, self.sid = build_fixture(self.directory)
        # alice consumed up to 2 via default sync; bob check-pointed at 1.
        service.sync_group_messages(self.sid, "alice", None, 2)
        self.alice_updated_at = service.store._group_sync_cursors[
            (self.sid, "alice")].updated_at
        service.sync_group_checkpoint(
            self.sid, {"device_id": "bob", "cursor": 1})
        self.bob_updated_at = service.store._group_sync_cursors[
            (self.sid, "bob")].updated_at

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def _restart(self) -> DeviceService:
        service = DeviceService()
        attach_persistence(service, self.path)
        return service

    def test_default_sync_resumes_from_saved_cursor_after_restart(self) -> None:
        service = self._restart()
        body = service.sync_group_messages(self.sid, "alice", None, 100)
        self.assertEqual([m["sequence"] for m in body["messages"]], [3])
        self.assertEqual(body["next_cursor"], 3)
        self.assertFalse(body["has_more"])
        # Empty page does not advance next_cursor.
        body = service.sync_group_messages(self.sid, "alice", None, 100)
        self.assertEqual(body["messages"], [])
        self.assertEqual(body["next_cursor"], 3)
        self.assertFalse(body["has_more"])

    def test_explicit_after_after_restart_is_pure_query(self) -> None:
        service = self._restart()
        body = service.sync_group_messages(self.sid, "alice", 0, 100)
        self.assertEqual([m["sequence"] for m in body["messages"]], [1, 2, 3])
        # Saved cursor is still 2: the default sync resumes at 3.
        record = service.store._group_sync_cursors[(self.sid, "alice")]
        self.assertEqual(record.cursor, 2)
        body = service.sync_group_messages(self.sid, "alice", None, 100)
        self.assertEqual([m["sequence"] for m in body["messages"]], [3])

    def test_per_device_cursors_and_timestamps_survive_restart(self) -> None:
        service = self._restart()
        # Confirming the saved cursors is 200 and keeps the persisted
        # timestamps, without advancing anything.
        view, status = service.sync_group_checkpoint(
            self.sid, {"device_id": "bob", "cursor": 1})
        self.assertEqual(status, 200)
        self.assertEqual(view["updated_at"], self.bob_updated_at)
        view, status = service.sync_group_checkpoint(
            self.sid, {"device_id": "alice", "cursor": 2})
        self.assertEqual(status, 200)
        self.assertEqual(view["updated_at"], self.alice_updated_at)
        # bob's default sync still resumes from his saved cursor 1.
        body = service.sync_group_messages(self.sid, "bob", None, 100)
        self.assertEqual([m["sequence"] for m in body["messages"]], [2, 3])

    def test_checkpoint_rules_still_hold_after_restart(self) -> None:
        service = self._restart()
        with self.assertRaises(ServiceError) as caught:
            service.sync_group_checkpoint(
                self.sid, {"device_id": "alice", "cursor": 1})
        self.assertEqual((caught.exception.status_code,
                          caught.exception.field), (409, "cursor"))
        with self.assertRaises(ServiceError) as caught:
            service.sync_group_checkpoint(
                self.sid, {"device_id": "alice", "cursor": 4})
        self.assertEqual((caught.exception.status_code,
                          caught.exception.field), (409, "cursor"))
        # A device that never check-pointed still gets 200 for cursor 0, and
        # no record is persisted.
        _, status = service.sync_group_checkpoint(
            self.sid, {"device_id": "creator", "cursor": 0})
        self.assertEqual(status, 200)
        self.assertNotIn((self.sid, "creator"),
                         service.store._group_sync_cursors)

    def test_empty_session_checkpoint_beyond_zero_conflicts_after_restart(
            self) -> None:
        directory = tempfile.mkdtemp()
        try:
            service, _store, path, _sid = build_fixture(
                directory, with_empty_session=True)
            sessions = service.store.snapshot_state()["group_sessions"]
            empty_sid = next(s["session_id"] for s in sessions
                             if s["group_id"] == "g2")
            service.sync_group_checkpoint(
                empty_sid, {"device_id": "creator", "cursor": 0})
            restarted = DeviceService()
            attach_persistence(restarted, path)
            with self.assertRaises(ServiceError) as caught:
                restarted.sync_group_checkpoint(
                    empty_sid, {"device_id": "creator", "cursor": 1})
            self.assertEqual((caught.exception.status_code,
                              caught.exception.field), (409, "cursor"))
        finally:
            shutil.rmtree(directory, ignore_errors=True)


class MalformedCursorSectionStartupTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        _service, _store, self.good_path, self.sid = build_fixture(
            self.directory, with_carol=True, with_empty_session=True)
        with open(self.good_path, encoding="utf-8") as handle:
            self.good_document = json.load(handle)
        sessions = self.good_document["group_sessions"]
        self.empty_sid = next(s["session_id"] for s in sessions
                              if s["group_id"] == "g2")

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def _assert_rejected_without_touching_file(
            self, document: Dict[str, Any]) -> None:
        path = os.path.join(self.directory, "bad.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        before = open(path, "rb").read()
        with self.assertRaises(StateFileError):
            attach_persistence(DeviceService(), path)
        # Startup must never overwrite a file it rejected.
        self.assertEqual(open(path, "rb").read(), before)

    def test_old_file_without_section_loads_empty(self) -> None:
        document = dict(self.good_document)
        del document["group_sync_cursors"]
        # A legacy file also predates the integrity-log marker/sidecar.
        del document["integrity_log_version"]
        path = os.path.join(self.directory, "legacy.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        service = DeviceService()
        attach_persistence(service, path)
        self.assertEqual(service.store._group_sync_cursors, {})

    def test_malformed_records_are_rejected(self) -> None:
        base = self.good_document
        good_record = {"session_id": self.sid, "device_id": "alice",
                       "cursor": 1, "updated_at": "2026-01-01T00:00:00+00:00"}

        def with_record(record: Any) -> Dict[str, Any]:
            document = dict(base)
            document["group_sync_cursors"] = [record]
            return document

        bad_records = [
            "not-a-list",  # wrong section type
            ["not-an-object"],  # record not an object
            {**good_record, "updated_at": ""},  # empty timestamp
            {**good_record, "cursor": "1"},  # cursor wrong type
            {**good_record, "cursor": True},  # cursor bool
            {**good_record, "cursor": -1},  # cursor negative
            {**good_record, "cursor": 4},  # cursor past max sequence
            {**good_record, "session_id": "missing-session"},  # dangling
            {**good_record, "device_id": "ghost"},  # unknown device
            {**good_record, "device_id": "carol"},  # not frozen member
            {"session_id": self.empty_sid, "device_id": "creator",
             "cursor": 1, "updated_at": "t"},  # empty session, cursor > 0
            {k: v for k, v in good_record.items() if k != "cursor"},  # field
        ]
        for record in bad_records:
            self._assert_rejected_without_touching_file(with_record(record))

    def test_duplicate_cursor_key_is_rejected(self) -> None:
        document = dict(self.good_document)
        document["group_sync_cursors"] = [
            {"session_id": self.sid, "device_id": "alice", "cursor": 1,
             "updated_at": "2026-01-01T00:00:00+00:00"},
            {"session_id": self.sid, "device_id": "alice", "cursor": 2,
             "updated_at": "2026-01-02T00:00:00+00:00"},
        ]
        self._assert_rejected_without_touching_file(document)

    def test_well_formed_section_still_loads(self) -> None:
        service = DeviceService()
        attach_persistence(service, self.good_path)
        self.assertEqual(
            service.store._group_sync_cursors, {})


class ConcurrentRevocationLinearizationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.service, self.state_store, _path, self.sid = build_fixture(
            self.directory, with_carol=True, carol_member=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_checkpoint_concurrent_with_revoke(self) -> None:
        outcomes = []
        outcomes_lock = threading.Lock()
        barrier = threading.Barrier(2)

        def checkpoint() -> None:
            barrier.wait()
            try:
                _view, status = self.service.sync_group_checkpoint(
                    self.sid, {"device_id": "carol", "cursor": 2})
                with outcomes_lock:
                    outcomes.append(("ok", status))
            except ServiceError as error:
                with outcomes_lock:
                    outcomes.append(("error", error.status_code, error.field))

        def revoke() -> None:
            barrier.wait()
            self.service.revoke_device("carol")

        threads = [threading.Thread(target=checkpoint),
                   threading.Thread(target=revoke)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)

        kind = outcomes[0][0]
        if kind == "ok":
            # Linearized before the revoke: exactly one 201, and the advanced
            # cursor must be durable.
            self.assertEqual(outcomes, [("ok", 201)])
            self.assertEqual(
                self.service.store._group_sync_cursors[
                    (self.sid, "carol")].cursor, 2)
            reloaded = DeviceService()
            attach_persistence(reloaded, os.path.join(self.directory,
                                                      "state.json"))
            self.assertEqual(
                reloaded.store._group_sync_cursors[
                    (self.sid, "carol")].cursor, 2)
        else:
            # Linearized after the revoke: 409/field=device_id only, and no
            # cursor record exists in memory or in the file.
            self.assertEqual(outcomes, [("error", 409, "device_id")])
            self.assertNotIn((self.sid, "carol"),
                             self.service.store._group_sync_cursors)

    def test_failed_writes_under_concurrency_advance_nothing(self) -> None:
        def raise_oserror(state):  # noqa: ANN001
            raise OSError("simulated disk failure")

        self.state_store.save = raise_oserror  # type: ignore[assignment]
        device_ids = ["creator", "alice", "bob"]
        errors = []
        errors_lock = threading.Lock()
        barrier = threading.Barrier(len(device_ids))

        def revoke(device_id: str) -> None:
            barrier.wait()
            try:
                self.service.revoke_device(device_id)
            except PersistenceUnavailable:
                with errors_lock:
                    errors.append(device_id)

        threads = [threading.Thread(target=revoke, args=(device_id,))
                   for device_id in device_ids]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)

        self.assertEqual(sorted(errors), sorted(device_ids))
        for device_id in device_ids:
            self.assertFalse(
                self.service.store.find_by_device_id(device_id).revoked)
        # The file still parses and also shows every device active.
        reloaded = DeviceService()
        attach_persistence(reloaded, os.path.join(self.directory,
                                                  "state.json"))
        for device_id in device_ids:
            self.assertFalse(
                reloaded.store.find_by_device_id(device_id).revoked)
        self.assertEqual(
            [name for name in os.listdir(self.directory)
             if name.endswith(".tmp")], [])


class _MalformedSectionStartupBase(unittest.TestCase):
    """Shared fixture: a persisted good document plus rejection helpers."""

    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        _service, _store, self.good_path, self.sid = build_fixture(
            self.directory, with_carol=True, with_empty_session=True)
        with open(self.good_path, encoding="utf-8") as handle:
            self.good_document = json.load(handle)

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def _assert_rejected_without_touching_file(
            self, document: Dict[str, Any]) -> None:
        path = os.path.join(self.directory, "bad.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        before = open(path, "rb").read()
        before_ino = os.stat(path).st_ino
        with self.assertRaises(StateFileError):
            attach_persistence(DeviceService(), path)
        # Startup must never overwrite a file it rejected: same bytes, same
        # inode (no atomic replace happened).
        self.assertEqual(open(path, "rb").read(), before)
        self.assertEqual(os.stat(path).st_ino, before_ino)

    def _with_section(self, name: str, value: Any) -> Dict[str, Any]:
        document = deepcopy(self.good_document)
        document[name] = value
        return document

    def _with_messages(self, stream: Any) -> Dict[str, Any]:
        document = deepcopy(self.good_document)
        document["messages"] = {self.sid: stream}
        return document

    def _good_stream(self) -> Any:
        return deepcopy(self.good_document["messages"][self.sid])


class MalformedMessageSectionStartupTest(_MalformedSectionStartupBase):
    """The messages section must be internally consistent to be loaded."""

    def test_section_must_be_an_object(self) -> None:
        for bad in ([], "nope", 3):
            self._assert_rejected_without_touching_file(
                self._with_section("messages", bad))

    def test_stream_key_must_name_a_saved_session(self) -> None:
        self._assert_rejected_without_touching_file(
            self._with_section("messages", {"ghost-session": []}))

    def test_envelope_session_id_must_match_stream_key(self) -> None:
        stream = self._good_stream()
        stream[0]["session_id"] = "some-other-session"
        self._assert_rejected_without_touching_file(
            self._with_messages(stream))

    def test_envelope_field_types_are_checked(self) -> None:
        for field, value in (("sequence", "1"), ("sequence", True),
                             ("nonce", ""), ("ciphertext", 7),
                             ("sender_device_id", ""), ("message_id", ""),
                             ("created_at", "")):
            stream = self._good_stream()
            stream[1][field] = value
            self._assert_rejected_without_touching_file(
                self._with_messages(stream))

    def test_sequence_must_run_from_one_without_gaps(self) -> None:
        for index, value in ((0, 0), (0, 2), (2, 4), (2, 2)):
            stream = self._good_stream()
            stream[index]["sequence"] = value
            self._assert_rejected_without_touching_file(
                self._with_messages(stream))

    def test_message_id_must_be_unique_within_session(self) -> None:
        stream = self._good_stream()
        stream[2]["message_id"] = "m1"
        self._assert_rejected_without_touching_file(
            self._with_messages(stream))

    def test_nonce_must_be_unique_within_session(self) -> None:
        stream = self._good_stream()
        stream[2]["nonce"] = "n1"
        self._assert_rejected_without_touching_file(
            self._with_messages(stream))

    def test_sender_must_be_a_registered_device(self) -> None:
        stream = self._good_stream()
        stream[0]["sender_device_id"] = "ghost"
        self._assert_rejected_without_touching_file(
            self._with_messages(stream))

    def test_group_sender_must_be_a_frozen_member(self) -> None:
        # carol is registered but not frozen into the group session.
        stream = self._good_stream()
        stream[0]["sender_device_id"] = "carol"
        self._assert_rejected_without_touching_file(
            self._with_messages(stream))

    def test_revoked_sender_history_still_loads(self) -> None:
        service = DeviceService()
        attach_persistence(service, self.good_path)
        service.revoke_device("creator")  # persisted: sender now revoked
        restarted = DeviceService()
        attach_persistence(restarted, self.good_path)
        body = restarted.sync_group_messages(self.sid, "alice", 0, 100)
        self.assertEqual([m["sequence"] for m in body["messages"]], [1, 2, 3])


class NonceSetConsistencyStartupTest(_MalformedSectionStartupBase):
    """used_nonces must equal the nonce set of the stored messages exactly."""

    def test_missing_nonce_is_rejected(self) -> None:
        document = self._with_section(
            "used_nonces", {self.sid: ["n1", "n3"]})
        self._assert_rejected_without_touching_file(document)

    def test_extra_nonce_is_rejected(self) -> None:
        document = self._with_section(
            "used_nonces", {self.sid: ["n1", "n2", "n3", "n9"]})
        self._assert_rejected_without_touching_file(document)

    def test_unknown_session_key_is_rejected(self) -> None:
        document = self._with_section(
            "used_nonces",
            {self.sid: ["n1", "n2", "n3"], "ghost-session": ["x"]})
        self._assert_rejected_without_touching_file(document)

    def test_duplicate_entries_are_rejected(self) -> None:
        document = self._with_section(
            "used_nonces", {self.sid: ["n1", "n1", "n2", "n3"]})
        self._assert_rejected_without_touching_file(document)

    def test_exact_section_loads(self) -> None:
        service = DeviceService()
        attach_persistence(service, self.good_path)
        self.assertEqual(service.store._used_nonces[self.sid],
                         {"n1", "n2", "n3"})


class MalformedDeliverySectionStartupTest(_MalformedSectionStartupBase):
    """Delivery records must reference stored messages and stay consistent."""

    def _record(self, **overrides: Any) -> Dict[str, Any]:
        record = {"session_id": self.sid, "message_id": "m2", "attempts": 1,
                  "attempt_ids": ["a1"], "acked": True, "ack_sequence": 2}
        record.update(overrides)
        return record

    def _with_delivery(self, *records: Any) -> Dict[str, Any]:
        return self._with_section("delivery", list(records))

    def test_dangling_message_reference_is_rejected(self) -> None:
        self._assert_rejected_without_touching_file(
            self._with_delivery(self._record(message_id="m9")))
        self._assert_rejected_without_touching_file(
            self._with_delivery(self._record(session_id="ghost-session")))

    def test_attempts_must_equal_attempt_id_count(self) -> None:
        self._assert_rejected_without_touching_file(
            self._with_delivery(self._record(attempts=2)))

    def test_attempt_ids_must_be_unique_strings(self) -> None:
        self._assert_rejected_without_touching_file(self._with_delivery(
            self._record(attempts=2, attempt_ids=["a1", "a1"])))
        self._assert_rejected_without_touching_file(self._with_delivery(
            self._record(attempt_ids=[1])))

    def test_field_types_are_checked(self) -> None:
        self._assert_rejected_without_touching_file(
            self._with_delivery(self._record(attempts="1")))
        self._assert_rejected_without_touching_file(
            self._with_delivery(self._record(acked=1)))
        self._assert_rejected_without_touching_file(
            self._with_delivery(self._record(ack_sequence="2")))

    def test_acked_record_must_mirror_message_sequence(self) -> None:
        self._assert_rejected_without_touching_file(
            self._with_delivery(self._record(ack_sequence=3)))

    def test_unacked_record_must_have_zero_ack_sequence(self) -> None:
        self._assert_rejected_without_touching_file(self._with_delivery(
            self._record(attempts=0, attempt_ids=[], acked=False,
                         ack_sequence=2)))

    def test_duplicate_record_key_is_rejected(self) -> None:
        self._assert_rejected_without_touching_file(
            self._with_delivery(self._record(), self._record()))

    def test_well_formed_record_loads(self) -> None:
        document = self._with_delivery(self._record())
        # A hand-written legacy document carries no integrity-log marker.
        del document["integrity_log_version"]
        path = os.path.join(self.directory, "with-delivery.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        service = DeviceService()
        attach_persistence(service, path)
        state = service.store._delivery[(self.sid, "m2")]
        self.assertTrue(state.acked)
        self.assertEqual(state.ack_sequence, 2)
        self.assertEqual(state.attempts, 1)
        self.assertEqual(state.attempt_ids, {"a1"})


if __name__ == "__main__":
    unittest.main()
