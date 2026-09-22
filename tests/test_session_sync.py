"""Tests for unified 1:1/group-session message sync and per-device checkpoints."""
import copy
import unittest

from e2ee_backend.models import Device, SignedPreKey
from e2ee_backend.service import DeviceService, ServiceError
from e2ee_backend.storage import (
    MESSAGE_SYNC_CURSOR_CONFLICT,
    MESSAGE_SYNC_DEVICE_INACTIVE,
    MESSAGE_SYNC_DEVICE_NOT_PARTICIPANT,
    MESSAGE_SYNC_DEVICE_UNKNOWN,
    MESSAGE_SYNC_SESSION_UNKNOWN,
    DeviceStore,
    MessageSyncError,
)


class _FixtureMixin:
    def _build(self) -> str:
        self.service = DeviceService()
        for device_id in ("alice", "bob", "carol"):
            self.service.store.add_device(Device(
                "u", device_id, "ik",
                prekeys=([SignedPreKey("pk", "pubk")]
                         if device_id == "bob" else [])))
        session = self.service.store.create_session(
            "alice", "bob", "pk", "ek")
        sid = session.session_id
        for sequence in range(1, 4):
            self.service.post_message({
                "session_id": sid, "sender_device_id": "alice",
                "message_id": f"m{sequence}", "sequence": sequence,
                "nonce": f"n{sequence}", "ciphertext": "ct"})
        return sid


class SessionSyncServiceTest(_FixtureMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.sid = self._build()

    def _error(self, callable_):
        with self.assertRaises(ServiceError) as caught:
            callable_()
        return caught.exception

    def test_sync_pages_ascending_with_next_cursor_and_has_more(self) -> None:
        body = self.service.sync_session_messages(self.sid, "bob", None, 2)
        self.assertEqual([m["sequence"] for m in body["messages"]], [1, 2])
        self.assertEqual(body["next_cursor"], 2)
        self.assertTrue(body["has_more"])
        body = self.service.sync_session_messages(self.sid, "bob", None, 100)
        self.assertEqual([m["sequence"] for m in body["messages"]], [3])
        self.assertEqual(body["next_cursor"], 3)
        self.assertFalse(body["has_more"])

    def test_sync_without_after_advances_stored_cursor(self) -> None:
        self.service.sync_session_messages(self.sid, "bob", None, 2)
        self.assertEqual(
            self.service.store._message_sync_cursors[(self.sid, "bob")].cursor,
            2)
        body = self.service.sync_session_messages(self.sid, "bob", None, 100)
        self.assertEqual([m["sequence"] for m in body["messages"]], [3])
        body = self.service.sync_session_messages(self.sid, "bob", None, 100)
        self.assertEqual(body["messages"], [])
        self.assertEqual(body["next_cursor"], 3)
        self.assertEqual(
            self.service.store._message_sync_cursors[(self.sid, "bob")].cursor,
            3)

    def test_explicit_after_does_not_move_stored_cursor(self) -> None:
        self.service.sync_session_messages(self.sid, "bob", None, 100)
        body = self.service.sync_session_messages(self.sid, "bob", 0, 100)
        self.assertEqual([m["sequence"] for m in body["messages"]], [1, 2, 3])
        self.assertEqual(body["next_cursor"], 3)
        self.assertEqual(
            self.service.store._message_sync_cursors[(self.sid, "bob")].cursor,
            3)
        body = self.service.sync_session_messages(self.sid, "bob", None, 100)
        self.assertEqual(body["messages"], [])
        self.assertEqual(body["next_cursor"], 3)

    def test_initiator_may_sync(self) -> None:
        body = self.service.sync_session_messages(self.sid, "alice", None, 100)
        self.assertEqual([m["sequence"] for m in body["messages"]], [1, 2, 3])

    def test_cursors_are_per_device(self) -> None:
        self.service.sync_session_messages(self.sid, "alice", None, 1)
        body = self.service.sync_session_messages(self.sid, "bob", None, 100)
        self.assertEqual([m["sequence"] for m in body["messages"]], [1, 2, 3])
        self.assertEqual(
            self.service.store._message_sync_cursors[(self.sid, "alice")].cursor,
            1)
        self.assertEqual(
            self.service.store._message_sync_cursors[(self.sid, "bob")].cursor,
            3)

    def test_sync_authorization(self) -> None:
        error = self._error(lambda: self.service.sync_session_messages(
            "missing", "alice", None, 100))
        self.assertEqual((error.status_code, error.field), (404, "session_id"))
        error = self._error(lambda: self.service.sync_session_messages(
            self.sid, "ghost", None, 100))
        self.assertEqual((error.status_code, error.field), (409, "device_id"))
        # carol is registered but neither initiator nor recipient.
        error = self._error(lambda: self.service.sync_session_messages(
            self.sid, "carol", None, 100))
        self.assertEqual((error.status_code, error.field), (409, "device_id"))
        self.service.store.revoke_device("bob")
        error = self._error(lambda: self.service.sync_session_messages(
            self.sid, "bob", None, 100))
        self.assertEqual((error.status_code, error.field), (409, "device_id"))

    def test_group_session_syncs_with_same_endpoint(self) -> None:
        for device_id in ("creator", "dave"):
            self.service.store.add_device(Device("u", device_id, "ik"))
        self.service.create_group({
            "group_id": "g1", "creator_device_id": "creator",
            "member_device_ids": ["dave"]})
        group_session = self.service.create_group_session({
            "group_id": "g1", "initiator_device_id": "creator",
            "ephemeral_key": "epk"})
        gsid = group_session["session_id"]
        self.service.post_message({
            "session_id": gsid, "sender_device_id": "creator",
            "message_id": "g1", "sequence": 1, "nonce": "gn1",
            "ciphertext": "ct"})
        body = self.service.sync_session_messages(gsid, "dave", None, 100)
        self.assertEqual([m["sequence"] for m in body["messages"]], [1])
        # A 1:1-only device has no rights on the group session.
        error = self._error(lambda: self.service.sync_session_messages(
            gsid, "alice", None, 100))
        self.assertEqual((error.status_code, error.field), (409, "device_id"))

    def test_checkpoint_forward_same_backward(self) -> None:
        body, status = self.service.sync_session_checkpoint(
            self.sid, {"device_id": "bob", "cursor": 2})
        self.assertEqual(status, 201)
        self.assertEqual(set(body),
                         {"session_id", "device_id", "cursor", "updated_at"})
        self.assertEqual(body["cursor"], 2)
        self.assertTrue(body["updated_at"])
        timestamp = body["updated_at"]
        body, status = self.service.sync_session_checkpoint(
            self.sid, {"device_id": "bob", "cursor": 2})
        self.assertEqual(status, 200)
        self.assertEqual(body["updated_at"], timestamp)
        error = self._error(lambda: self.service.sync_session_checkpoint(
            self.sid, {"device_id": "bob", "cursor": 1}))
        self.assertEqual((error.status_code, error.field), (409, "cursor"))
        error = self._error(lambda: self.service.sync_session_checkpoint(
            self.sid, {"device_id": "bob", "cursor": 4}))
        self.assertEqual((error.status_code, error.field), (409, "cursor"))

    def test_checkpoint_zero_on_fresh_device_is_200_noop(self) -> None:
        body, status = self.service.sync_session_checkpoint(
            self.sid, {"device_id": "bob", "cursor": 0})
        self.assertEqual(status, 200)
        self.assertEqual(body["cursor"], 0)
        self.assertNotIn((self.sid, "bob"),
                         self.service.store._message_sync_cursors)

    def test_checkpoint_validation(self) -> None:
        cases = [
            ({"device_id": "alice"}, "cursor"),
            ({"device_id": "alice", "cursor": "1"}, "cursor"),
            ({"device_id": "alice", "cursor": True}, "cursor"),
            ({"device_id": "alice", "cursor": -1}, "cursor"),
            ({"cursor": 1}, "device_id"),
            ({"device_id": "", "cursor": 1}, "device_id"),
            ([], "request_body"),
        ]
        for payload, field in cases:
            error = self._error(lambda: self.service.sync_session_checkpoint(
                self.sid, payload))
            self.assertEqual((error.status_code, error.field), (400, field),
                             payload)

    def test_checkpoint_authorization(self) -> None:
        error = self._error(lambda: self.service.sync_session_checkpoint(
            "missing", {"device_id": "alice", "cursor": 0}))
        self.assertEqual((error.status_code, error.field), (404, "session_id"))
        error = self._error(lambda: self.service.sync_session_checkpoint(
            self.sid, {"device_id": "ghost", "cursor": 0}))
        self.assertEqual((error.status_code, error.field), (409, "device_id"))
        error = self._error(lambda: self.service.sync_session_checkpoint(
            self.sid, {"device_id": "carol", "cursor": 0}))
        self.assertEqual((error.status_code, error.field), (409, "device_id"))

    def test_group_cursor_namespace_is_independent(self) -> None:
        self.service.sync_session_messages(self.sid, "bob", None, 100)
        self.assertIn((self.sid, "bob"),
                      self.service.store._message_sync_cursors)
        self.assertNotIn((self.sid, "bob"),
                         self.service.store._group_sync_cursors)

    def test_storage_reason_codes(self) -> None:
        store = self.service.store

        def reason(callable_):
            with self.assertRaises(MessageSyncError) as caught:
                callable_()
            return caught.exception.reason

        self.assertEqual(reason(lambda: store.message_sync_page(
            "missing", "alice", None, 100)), MESSAGE_SYNC_SESSION_UNKNOWN)
        self.assertEqual(reason(lambda: store.message_sync_page(
            self.sid, "ghost", None, 100)), MESSAGE_SYNC_DEVICE_UNKNOWN)
        self.assertEqual(reason(lambda: store.message_sync_page(
            self.sid, "carol", None, 100)),
            MESSAGE_SYNC_DEVICE_NOT_PARTICIPANT)
        store.revoke_device("bob")
        self.assertEqual(reason(lambda: store.message_sync_page(
            self.sid, "bob", None, 100)), MESSAGE_SYNC_DEVICE_INACTIVE)
        self.assertEqual(reason(lambda: store.message_sync_checkpoint(
            self.sid, "alice", 99)), MESSAGE_SYNC_CURSOR_CONFLICT)


class SessionSyncPersistenceTest(_FixtureMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.sid = self._build()

    def _round_trip(self, document):
        restored = DeviceStore()
        restored.restore_state(document)
        return restored

    def test_cursors_persist_in_snapshot(self) -> None:
        self.service.sync_session_messages(self.sid, "bob", None, 100)
        self.service.sync_session_checkpoint(
            self.sid, {"device_id": "alice", "cursor": 1})
        document = self.service.store.snapshot_state()
        restored = self._round_trip(document)
        self.assertEqual(
            restored._message_sync_cursors[(self.sid, "bob")].cursor, 3)
        alice = restored._message_sync_cursors[(self.sid, "alice")]
        self.assertEqual(alice.cursor, 1)
        self.assertTrue(alice.updated_at)

    def test_old_file_without_section_loads_empty(self) -> None:
        self.service.sync_session_messages(self.sid, "bob", None, 100)
        document = self.service.store.snapshot_state()
        del document["message_sync_cursors"]
        restored = self._round_trip(document)
        self.assertEqual(restored._message_sync_cursors, {})

    def test_malformed_section_rejected(self) -> None:
        document = self.service.store.snapshot_state()
        bad = copy.deepcopy(document)
        bad["message_sync_cursors"] = [
            {"session_id": self.sid, "device_id": "alice"}]
        with self.assertRaises(ValueError):
            DeviceStore().restore_state(bad)
        bad = copy.deepcopy(document)
        bad["message_sync_cursors"] = [
            {"session_id": self.sid, "device_id": "alice", "cursor": -1,
             "updated_at": "t"}]
        with self.assertRaises(ValueError):
            DeviceStore().restore_state(bad)
        bad = copy.deepcopy(document)
        bad["message_sync_cursors"] = "nope"
        with self.assertRaises(ValueError):
            DeviceStore().restore_state(bad)
        bad = copy.deepcopy(document)
        # Dangling session.
        bad["message_sync_cursors"] = [
            {"session_id": "nope", "device_id": "alice", "cursor": 0,
             "updated_at": "t"}]
        with self.assertRaises(ValueError):
            DeviceStore().restore_state(bad)
        bad = copy.deepcopy(document)
        # Non-participant device.
        bad["message_sync_cursors"] = [
            {"session_id": self.sid, "device_id": "carol", "cursor": 0,
             "updated_at": "t"}]
        with self.assertRaises(ValueError):
            DeviceStore().restore_state(bad)
        bad = copy.deepcopy(document)
        # Empty updated_at.
        bad["message_sync_cursors"] = [
            {"session_id": self.sid, "device_id": "alice", "cursor": 0,
             "updated_at": ""}]
        with self.assertRaises(ValueError):
            DeviceStore().restore_state(bad)
        bad = copy.deepcopy(document)
        # Duplicate (session, device) key.
        record = {"session_id": self.sid, "device_id": "alice", "cursor": 0,
                  "updated_at": "t"}
        bad["message_sync_cursors"] = [record, dict(record)]
        with self.assertRaises(ValueError):
            DeviceStore().restore_state(bad)


class SessionSyncRollbackTest(_FixtureMixin, unittest.TestCase):
    """A failed durable write rolls the cursor back and surfaces 503."""

    def setUp(self) -> None:
        import os
        import tempfile
        from e2ee_backend.persistence import (
            PersistenceUnavailable, attach_persistence)
        self.PersistenceUnavailable = PersistenceUnavailable
        self.directory = tempfile.mkdtemp()
        # Rebuild the fixture on a persisted service instead of the in-memory
        # one _FixtureMixin creates.
        from e2ee_backend.service import DeviceService as _Service
        self.service = _Service()
        self.state_store = attach_persistence(
            self.service, os.path.join(self.directory, "state.json"))
        for device_id in ("alice", "bob"):
            self.service.store.add_device(Device(
                "u", device_id, "ik",
                prekeys=([SignedPreKey("pk", "pubk")]
                         if device_id == "bob" else [])))
        session = self.service.store.create_session(
            "alice", "bob", "pk", "ek")
        self.sid = session.session_id
        for sequence in range(1, 4):
            self.service.post_message({
                "session_id": self.sid, "sender_device_id": "alice",
                "message_id": f"m{sequence}", "sequence": sequence,
                "nonce": f"n{sequence}", "ciphertext": "ct"})

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self.directory, ignore_errors=True)

    def _fail_writes(self) -> None:
        def raise_oserror(state):
            raise OSError("simulated disk failure")

        self.state_store.save = raise_oserror

    def test_failed_sync_advance_rolls_back(self) -> None:
        self._fail_writes()
        with self.assertRaises(self.PersistenceUnavailable):
            self.service.sync_session_messages(self.sid, "bob", None, 100)
        self.assertNotIn((self.sid, "bob"),
                         self.service.store._message_sync_cursors)

    def test_failed_checkpoint_advance_rolls_back(self) -> None:
        self._fail_writes()
        with self.assertRaises(self.PersistenceUnavailable):
            self.service.sync_session_checkpoint(
                self.sid, {"device_id": "bob", "cursor": 2})
        self.assertNotIn((self.sid, "bob"),
                         self.service.store._message_sync_cursors)


if __name__ == "__main__":
    unittest.main()
