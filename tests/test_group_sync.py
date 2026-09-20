"""Tests for group-session message sync and per-device checkpoints."""
import copy
import unittest

from e2ee_backend.models import Device
from e2ee_backend.service import DeviceService, ServiceError
from e2ee_backend.storage import (
    SYNC_CURSOR_CONFLICT,
    SYNC_DEVICE_INACTIVE,
    SYNC_DEVICE_NOT_MEMBER,
    SYNC_DEVICE_UNKNOWN,
    SYNC_SESSION_UNKNOWN,
    DeviceStore,
    GroupSyncError,
)


class GroupSyncServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = DeviceService()
        # Group logic only cares about identifiers and the active flag.
        for device_id in ("creator", "alice", "bob", "carol"):
            self.service.store.add_device(Device("u", device_id, "ik"))
        self.service.create_group({
            "group_id": "g1", "creator_device_id": "creator",
            "member_device_ids": ["alice", "bob"]})
        session = self.service.create_group_session({
            "group_id": "g1", "initiator_device_id": "creator",
            "ephemeral_key": "epk"})
        self.sid = session["session_id"]
        for sequence in range(1, 4):
            self.service.post_message({
                "session_id": self.sid, "sender_device_id": "creator",
                "message_id": f"m{sequence}", "sequence": sequence,
                "nonce": f"n{sequence}", "ciphertext": "ct"})

    def _error(self, callable_):
        with self.assertRaises(ServiceError) as caught:
            callable_()
        return caught.exception

    def test_sync_pages_ascending_with_next_cursor_and_has_more(self) -> None:
        body = self.service.sync_group_messages(self.sid, "alice", None, 2)
        self.assertEqual([m["sequence"] for m in body["messages"]], [1, 2])
        self.assertEqual(body["next_cursor"], 2)
        self.assertTrue(body["has_more"])
        body = self.service.sync_group_messages(self.sid, "alice", None, 100)
        self.assertEqual([m["sequence"] for m in body["messages"]], [3])
        self.assertEqual(body["next_cursor"], 3)
        self.assertFalse(body["has_more"])

    def test_sync_without_after_starts_at_and_advances_stored_cursor(self) -> None:
        # No cursor yet: starts at 0 and advances to the last returned seq.
        self.service.sync_group_messages(self.sid, "alice", None, 2)
        record = self.service.store._group_sync_cursors[(self.sid, "alice")]
        self.assertEqual(record.cursor, 2)
        # The next call resumes from the stored cursor.
        body = self.service.sync_group_messages(self.sid, "alice", None, 100)
        self.assertEqual([m["sequence"] for m in body["messages"]], [3])
        # An empty page keeps the cursor at 3 and returns it as next_cursor.
        body = self.service.sync_group_messages(self.sid, "alice", None, 100)
        self.assertEqual(body["messages"], [])
        self.assertEqual(body["next_cursor"], 3)
        self.assertEqual(
            self.service.store._group_sync_cursors[(self.sid, "alice")].cursor,
            3)

    def test_explicit_after_does_not_move_stored_cursor(self) -> None:
        # Seed a stored cursor at 3 for alice.
        self.service.sync_group_messages(self.sid, "alice", None, 100)
        self.assertEqual(
            self.service.store._group_sync_cursors[(self.sid, "alice")].cursor,
            3)
        # An explicit after=0 re-reads from 0 but must not touch the cursor.
        body = self.service.sync_group_messages(self.sid, "alice", 0, 100)
        self.assertEqual([m["sequence"] for m in body["messages"]], [1, 2, 3])
        self.assertEqual(body["next_cursor"], 3)
        self.assertEqual(
            self.service.store._group_sync_cursors[(self.sid, "alice")].cursor,
            3)
        # The stored cursor still resumes past the end.
        body = self.service.sync_group_messages(self.sid, "alice", None, 100)
        self.assertEqual(body["messages"], [])
        self.assertEqual(body["next_cursor"], 3)

    def test_cursors_are_per_device(self) -> None:
        self.service.sync_group_messages(self.sid, "alice", None, 1)
        body = self.service.sync_group_messages(self.sid, "bob", None, 100)
        self.assertEqual([m["sequence"] for m in body["messages"]], [1, 2, 3])
        self.assertEqual(
            self.service.store._group_sync_cursors[(self.sid, "alice")].cursor,
            1)
        self.assertEqual(
            self.service.store._group_sync_cursors[(self.sid, "bob")].cursor, 3)

    def test_sync_authorization(self) -> None:
        error = self._error(lambda: self.service.sync_group_messages(
            "missing", "alice", None, 100))
        self.assertEqual((error.status_code, error.field), (404, "session_id"))
        error = self._error(lambda: self.service.sync_group_messages(
            self.sid, "ghost", None, 100))
        self.assertEqual((error.status_code, error.field), (409, "device_id"))
        # carol is registered but not a frozen member -> 409/device_id.
        error = self._error(lambda: self.service.sync_group_messages(
            self.sid, "carol", None, 100))
        self.assertEqual((error.status_code, error.field), (409, "device_id"))
        # A revoked frozen member is rejected.
        self.service.store.revoke_device("bob")
        error = self._error(lambda: self.service.sync_group_messages(
            self.sid, "bob", None, 100))
        self.assertEqual((error.status_code, error.field), (409, "device_id"))

    def test_removed_member_still_syncs_later_added_member_does_not(self) -> None:
        # carol joins the current group after the freeze: must not sync.
        self.service.add_group_member("g1", {
            "actor_device_id": "creator", "device_id": "carol"})
        error = self._error(lambda: self.service.sync_group_messages(
            self.sid, "carol", None, 100))
        self.assertEqual((error.status_code, error.field), (409, "device_id"))
        # alice is removed from the current group but frozen into the session.
        self.service.remove_group_member("g1", {
            "actor_device_id": "creator", "device_id": "alice"})
        body = self.service.sync_group_messages(self.sid, "alice", None, 100)
        self.assertEqual([m["sequence"] for m in body["messages"]], [1, 2, 3])

    def test_checkpoint_forward_same_backward(self) -> None:
        body, status = self.service.sync_group_checkpoint(
            self.sid, {"device_id": "alice", "cursor": 2})
        self.assertEqual(status, 201)
        self.assertEqual(set(body),
                         {"session_id", "device_id", "cursor", "updated_at"})
        self.assertEqual(body["cursor"], 2)
        self.assertTrue(body["updated_at"])
        timestamp = body["updated_at"]
        # Repeating the same cursor is a 200 no-op with an unchanged timestamp.
        body, status = self.service.sync_group_checkpoint(
            self.sid, {"device_id": "alice", "cursor": 2})
        self.assertEqual(status, 200)
        self.assertEqual(body["cursor"], 2)
        self.assertEqual(body["updated_at"], timestamp)
        # Moving backwards conflicts.
        error = self._error(lambda: self.service.sync_group_checkpoint(
            self.sid, {"device_id": "alice", "cursor": 1}))
        self.assertEqual((error.status_code, error.field), (409, "cursor"))
        # Advancing beyond the max stored sequence conflicts.
        error = self._error(lambda: self.service.sync_group_checkpoint(
            self.sid, {"device_id": "alice", "cursor": 4}))
        self.assertEqual((error.status_code, error.field), (409, "cursor"))

    def test_checkpoint_zero_on_fresh_device_is_200_noop(self) -> None:
        body, status = self.service.sync_group_checkpoint(
            self.sid, {"device_id": "bob", "cursor": 0})
        self.assertEqual(status, 200)
        self.assertEqual(body["cursor"], 0)
        self.assertNotIn((self.sid, "bob"),
                         self.service.store._group_sync_cursors)

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
            error = self._error(lambda: self.service.sync_group_checkpoint(
                self.sid, payload))
            self.assertEqual((error.status_code, error.field), (400, field),
                             payload)

    def test_checkpoint_authorization(self) -> None:
        error = self._error(lambda: self.service.sync_group_checkpoint(
            "missing", {"device_id": "alice", "cursor": 0}))
        self.assertEqual((error.status_code, error.field), (404, "session_id"))
        error = self._error(lambda: self.service.sync_group_checkpoint(
            self.sid, {"device_id": "ghost", "cursor": 0}))
        self.assertEqual((error.status_code, error.field), (409, "device_id"))
        error = self._error(lambda: self.service.sync_group_checkpoint(
            self.sid, {"device_id": "carol", "cursor": 0}))
        self.assertEqual((error.status_code, error.field), (409, "device_id"))

    def test_failed_checkpoint_does_not_change_cursor(self) -> None:
        self.service.sync_group_checkpoint(
            self.sid, {"device_id": "alice", "cursor": 2})
        # A failed explicit sync auth must not advance anything either.
        error = self._error(lambda: self.service.sync_group_messages(
            self.sid, "carol", None, 100))
        self.assertEqual(error.status_code, 409)
        self.assertEqual(
            self.service.store._group_sync_cursors[(self.sid, "alice")].cursor,
            2)

    def test_storage_reason_codes(self) -> None:
        store = self.service.store

        def reason(callable_):
            with self.assertRaises(GroupSyncError) as caught:
                callable_()
            return caught.exception.reason

        self.assertEqual(reason(lambda: store.group_sync_page(
            "missing", "alice", None, 100)), SYNC_SESSION_UNKNOWN)
        self.assertEqual(reason(lambda: store.group_sync_page(
            self.sid, "ghost", None, 100)), SYNC_DEVICE_UNKNOWN)
        self.assertEqual(reason(lambda: store.group_sync_page(
            self.sid, "carol", None, 100)), SYNC_DEVICE_NOT_MEMBER)
        store.revoke_device("bob")
        self.assertEqual(reason(lambda: store.group_sync_page(
            self.sid, "bob", None, 100)), SYNC_DEVICE_INACTIVE)
        self.assertEqual(reason(lambda: store.group_sync_checkpoint(
            self.sid, "alice", 99)), SYNC_CURSOR_CONFLICT)


class GroupSyncPersistenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = DeviceService()
        for device_id in ("creator", "alice", "bob"):
            self.service.store.add_device(Device("u", device_id, "ik"))
        self.service.create_group({
            "group_id": "g1", "creator_device_id": "creator",
            "member_device_ids": ["alice", "bob"]})
        session = self.service.create_group_session({
            "group_id": "g1", "initiator_device_id": "creator",
            "ephemeral_key": "epk"})
        self.sid = session["session_id"]
        for sequence in range(1, 3):
            self.service.post_message({
                "session_id": self.sid, "sender_device_id": "creator",
                "message_id": f"m{sequence}", "sequence": sequence,
                "nonce": f"n{sequence}", "ciphertext": "ct"})

    def _round_trip(self, document):
        restored = DeviceStore()
        restored.restore_state(document)
        return restored

    def test_cursors_persist_in_snapshot(self) -> None:
        self.service.sync_group_messages(self.sid, "alice", None, 100)
        self.service.sync_group_checkpoint(
            self.sid, {"device_id": "bob", "cursor": 1})
        document = self.service.store.snapshot_state()
        restored = self._round_trip(document)
        self.assertEqual(restored._group_sync_cursors[(self.sid, "alice")].cursor,
                         2)
        bob = restored._group_sync_cursors[(self.sid, "bob")]
        self.assertEqual(bob.cursor, 1)
        self.assertTrue(bob.updated_at)

    def test_old_file_without_section_loads_empty(self) -> None:
        self.service.sync_group_messages(self.sid, "alice", None, 100)
        document = self.service.store.snapshot_state()
        del document["group_sync_cursors"]
        restored = self._round_trip(document)
        self.assertEqual(restored._group_sync_cursors, {})

    def test_malformed_section_rejected(self) -> None:
        document = self.service.store.snapshot_state()
        bad = copy.deepcopy(document)
        bad["group_sync_cursors"] = [{"session_id": self.sid, "device_id": "a"}]
        with self.assertRaises(ValueError):
            DeviceStore().restore_state(bad)
        bad = copy.deepcopy(document)
        bad["group_sync_cursors"] = [
            {"session_id": self.sid, "device_id": "a", "cursor": -1,
             "updated_at": "t"}]
        with self.assertRaises(ValueError):
            DeviceStore().restore_state(bad)
        bad = copy.deepcopy(document)
        bad["group_sync_cursors"] = "nope"
        with self.assertRaises(ValueError):
            DeviceStore().restore_state(bad)


if __name__ == "__main__":
    unittest.main()
