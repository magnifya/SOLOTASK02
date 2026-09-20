"""Tests for group-session per-device message sync and checkpoints.

Covers the storage/service semantics, the HTTP routes over a real socket,
and persistence of the per-device cursors (including the old-version-1
missing/empty and malformed-section rules).
"""
import json
import threading
import unittest
from http.client import HTTPConnection

from e2ee_backend.http_app import create_server
from e2ee_backend.models import Device
from e2ee_backend.service import DeviceService, ServiceError
from e2ee_backend.storage import (
    DeviceStore,
    GroupSyncError,
    SYNC_CURSOR_BACKWARD,
    SYNC_CURSOR_OUT_OF_RANGE,
    SYNC_DEVICE_INACTIVE,
    SYNC_DEVICE_NOT_MEMBER,
    SYNC_DEVICE_UNKNOWN,
    SYNC_SESSION_UNKNOWN,
)


def _build_populated():
    """Return (service, session_id) with a frozen group session and 3 msgs.

    Devices: ``a`` (creator) and ``b`` are frozen members; ``c`` is neither
    in the group nor the frozen snapshot.
    """
    service = DeviceService(DeviceStore())
    store = service.store
    for device_id in ("a", "b", "c"):
        store.add_device(Device("u", device_id, "ik"))
    store.create_group("g1", "a", ["b"])
    session_id = store.create_group_session("g1", "a", "ek").session_id
    for sequence in (1, 2, 3):
        store.append_message(
            session_id, "a", f"m{sequence}", sequence,
            f"n{sequence}", "ct")
    return service, session_id


class GroupSyncStorageTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service, self.sid = _build_populated()
        self.store = self.service.store

    def test_sync_without_after_advances_cursor(self) -> None:
        messages, next_cursor, has_more = self.store.sync_group_messages(
            self.sid, "b", None, 100)
        self.assertEqual([m["sequence"] for m in messages], [1, 2, 3])
        self.assertEqual(next_cursor, 3)
        self.assertFalse(has_more)
        # A subsequent omitted-after sync resumes at 3: empty, cursor kept.
        messages, next_cursor, has_more = self.store.sync_group_messages(
            self.sid, "b", None, 100)
        self.assertEqual(messages, [])
        self.assertEqual(next_cursor, 3)
        self.assertFalse(has_more)

    def test_sync_respects_limit_and_has_more(self) -> None:
        messages, next_cursor, has_more = self.store.sync_group_messages(
            self.sid, "b", None, 2)
        self.assertEqual([m["sequence"] for m in messages], [1, 2])
        self.assertEqual(next_cursor, 2)
        self.assertTrue(has_more)
        # Cursor advanced to 2; next page returns the remainder.
        messages, next_cursor, has_more = self.store.sync_group_messages(
            self.sid, "b", None, 2)
        self.assertEqual([m["sequence"] for m in messages], [3])
        self.assertEqual(next_cursor, 3)
        self.assertFalse(has_more)

    def test_full_page_with_no_remainder_has_more_false(self) -> None:
        messages, _next_cursor, has_more = self.store.sync_group_messages(
            self.sid, "b", 0, 3)
        self.assertEqual(len(messages), 3)
        self.assertFalse(has_more)

    def test_explicit_after_does_not_move_cursor(self) -> None:
        # First advance the stored cursor to 3 via omitted-after.
        self.store.sync_group_messages(self.sid, "b", None, 100)
        # An explicit read from 1 returns 2..3 but leaves the cursor at 3.
        messages, next_cursor, _ = self.store.sync_group_messages(
            self.sid, "b", 1, 100)
        self.assertEqual([m["sequence"] for m in messages], [2, 3])
        self.assertEqual(next_cursor, 3)
        # Omitted-after again resumes at the stored cursor (3): empty page.
        messages, next_cursor, _ = self.store.sync_group_messages(
            self.sid, "b", None, 100)
        self.assertEqual(messages, [])
        self.assertEqual(next_cursor, 3)

    def test_explicit_after_zero_does_not_advance_cursor(self) -> None:
        _messages, _nc, _ = self.store.sync_group_messages(
            self.sid, "b", 0, 2)
        # Cursor was never set; an omitted-after sync must start at 0.
        messages, next_cursor, _ = self.store.sync_group_messages(
            self.sid, "b", None, 100)
        self.assertEqual([m["sequence"] for m in messages], [1, 2, 3])
        self.assertEqual(next_cursor, 3)

    def test_sync_unknown_session(self) -> None:
        with self.assertRaises(GroupSyncError) as caught:
            self.store.sync_group_messages("missing", "b", None, 100)
        self.assertEqual(caught.exception.reason, SYNC_SESSION_UNKNOWN)

    def test_sync_unknown_device(self) -> None:
        with self.assertRaises(GroupSyncError) as caught:
            self.store.sync_group_messages(self.sid, "ghost", None, 100)
        self.assertEqual(caught.exception.reason, SYNC_DEVICE_UNKNOWN)

    def test_sync_revoked_frozen_member(self) -> None:
        self.store.revoke_device("b")
        with self.assertRaises(GroupSyncError) as caught:
            self.store.sync_group_messages(self.sid, "b", None, 100)
        self.assertEqual(caught.exception.reason, SYNC_DEVICE_INACTIVE)

    def test_sync_non_frozen_member(self) -> None:
        # ``c`` is registered and active but was never frozen into the session.
        with self.assertRaises(GroupSyncError) as caught:
            self.store.sync_group_messages(self.sid, "c", None, 100)
        self.assertEqual(caught.exception.reason, SYNC_DEVICE_NOT_MEMBER)

    def test_removed_member_can_still_sync_frozen_session(self) -> None:
        # Remove ``b`` from the current group after the freeze; the frozen
        # snapshot keeps them as a sync-capable member.
        self.store.remove_group_member("g1", "a", "b")
        messages, _next_cursor, _ = self.store.sync_group_messages(
            self.sid, "b", None, 100)
        self.assertEqual(len(messages), 3)

    def test_later_added_member_cannot_sync(self) -> None:
        # Add ``c`` to the group after the freeze; it is not on the snapshot.
        self.store.add_group_member("g1", "a", "c")
        with self.assertRaises(GroupSyncError) as caught:
            self.store.sync_group_messages(self.sid, "c", None, 100)
        self.assertEqual(caught.exception.reason, SYNC_DEVICE_NOT_MEMBER)

    def test_checkpoint_forward_same_backward(self) -> None:
        view, status = self.store.set_sync_checkpoint(self.sid, "a", 3)
        self.assertEqual(status, 201)
        self.assertEqual(view["cursor"], 3)
        self.assertIsNotNone(view["updated_at"])
        first_timestamp = view["updated_at"]

        view, status = self.store.set_sync_checkpoint(self.sid, "a", 3)
        self.assertEqual(status, 200)
        self.assertEqual(view["updated_at"], first_timestamp)

        with self.assertRaises(GroupSyncError) as caught:
            self.store.set_sync_checkpoint(self.sid, "a", 2)
        self.assertEqual(caught.exception.reason, SYNC_CURSOR_BACKWARD)

    def test_checkpoint_zero_on_fresh_device_is_same(self) -> None:
        view, status = self.store.set_sync_checkpoint(self.sid, "b", 0)
        self.assertEqual(status, 200)
        self.assertEqual(view["cursor"], 0)
        self.assertIsNone(view["updated_at"])

    def test_checkpoint_out_of_range(self) -> None:
        for bad in (-1, 4):
            with self.assertRaises(GroupSyncError) as caught:
                self.store.set_sync_checkpoint(self.sid, "a", bad)
            self.assertEqual(caught.exception.reason,
                             SYNC_CURSOR_OUT_OF_RANGE)

    def test_checkpoint_max_sequence_empty_session(self) -> None:
        # A fresh group session has no messages; only cursor 0 is valid.
        empty_sid = self.store.create_group_session("g1", "a", "ek").session_id
        view, status = self.store.set_sync_checkpoint(empty_sid, "a", 0)
        self.assertEqual(status, 200)
        with self.assertRaises(GroupSyncError):
            self.store.set_sync_checkpoint(empty_sid, "a", 1)

    def test_checkpoint_authorization(self) -> None:
        with self.assertRaises(GroupSyncError) as caught:
            self.store.set_sync_checkpoint("missing", "a", 0)
        self.assertEqual(caught.exception.reason, SYNC_SESSION_UNKNOWN)
        with self.assertRaises(GroupSyncError) as caught:
            self.store.set_sync_checkpoint(self.sid, "c", 0)
        self.assertEqual(caught.exception.reason, SYNC_DEVICE_NOT_MEMBER)
        self.store.revoke_device("b")
        with self.assertRaises(GroupSyncError) as caught:
            self.store.set_sync_checkpoint(self.sid, "b", 0)
        self.assertEqual(caught.exception.reason, SYNC_DEVICE_INACTIVE)

    def test_checkpoint_failure_leaves_cursor_unchanged(self) -> None:
        self.store.set_sync_checkpoint(self.sid, "a", 2)
        with self.assertRaises(GroupSyncError):
            self.store.set_sync_checkpoint(self.sid, "a", 1)
        view, status = self.store.set_sync_checkpoint(self.sid, "a", 2)
        self.assertEqual(status, 200)
        self.assertEqual(view["cursor"], 2)


class GroupSyncServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service, self.sid = _build_populated()

    def test_sync_response_shape(self) -> None:
        body = self.service.sync_group_messages(self.sid, "b", None, 2)
        self.assertEqual(set(body), {"messages", "next_cursor", "has_more"})
        self.assertEqual([m["sequence"] for m in body["messages"]], [1, 2])
        self.assertEqual(body["next_cursor"], 2)
        self.assertTrue(body["has_more"])

    def test_checkpoint_service_statuses(self) -> None:
        _view, status = self.service.sync_checkpoint(
            self.sid, {"device_id": "a", "cursor": 3})
        self.assertEqual(status, 201)
        _view, status = self.service.sync_checkpoint(
            self.sid, {"device_id": "a", "cursor": 3})
        self.assertEqual(status, 200)

    def test_checkpoint_validation(self) -> None:
        for payload, field in (
            ({}, "device_id"),
            ({"device_id": ""}, "device_id"),
            ({"device_id": "a"}, "cursor"),
            ({"device_id": "a", "cursor": "1"}, "cursor"),
            ({"device_id": "a", "cursor": True}, "cursor"),
            ({"cursor": 1}, "device_id"),
        ):
            with self.assertRaises(ServiceError) as caught:
                self.service.sync_checkpoint(self.sid, payload)
            self.assertEqual(caught.exception.status_code, 400, payload)
            self.assertEqual(caught.exception.field, field, payload)

    def test_checkpoint_out_of_range_is_400_cursor(self) -> None:
        with self.assertRaises(ServiceError) as caught:
            self.service.sync_checkpoint(
                self.sid, {"device_id": "a", "cursor": 9})
        self.assertEqual(caught.exception.status_code, 400)
        self.assertEqual(caught.exception.field, "cursor")

    def test_checkpoint_backward_is_409_cursor(self) -> None:
        self.service.sync_checkpoint(
            self.sid, {"device_id": "a", "cursor": 3})
        with self.assertRaises(ServiceError) as caught:
            self.service.sync_checkpoint(
                self.sid, {"device_id": "a", "cursor": 2})
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "cursor")

    def test_sync_unknown_session_is_404(self) -> None:
        with self.assertRaises(ServiceError) as caught:
            self.service.sync_group_messages("missing", "b", None, 100)
        self.assertEqual(caught.exception.status_code, 404)
        self.assertEqual(caught.exception.field, "session_id")

    def test_sync_unknown_device_is_409(self) -> None:
        with self.assertRaises(ServiceError) as caught:
            self.service.sync_group_messages(self.sid, "ghost", None, 100)
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "device_id")

    def test_sync_non_member_is_409(self) -> None:
        with self.assertRaises(ServiceError) as caught:
            self.service.sync_group_messages(self.sid, "c", None, 100)
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "device_id")


class GroupSyncHTTPTest(unittest.TestCase):
    def setUp(self) -> None:
        self.server, self.service = create_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        for device_id in ("a", "b", "c"):
            self.service.store.add_device(Device("u", device_id, "ik"))
        self.service.store.create_group("g1", "a", ["b"])
        self.sid = self.service.store.create_group_session(
            "g1", "a", "ek").session_id
        for sequence in (1, 2, 3):
            self.service.store.append_message(
                self.sid, "a", f"m{sequence}", sequence,
                f"n{sequence}", "ct")

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _request(self, method: str, path: str, body: object = None):
        connection = HTTPConnection("127.0.0.1", self.port, timeout=5)
        payload = json.dumps(body) if body is not None else None
        headers = {"Content-Type": "application/json"} if payload else {}
        connection.request(method, path, body=payload, headers=headers)
        response = connection.getresponse()
        data = response.read().decode("utf-8")
        connection.close()
        return response.status, json.loads(data)

    def _sync(self, query: str):
        return self._request(
            "GET", f"/v1/group-sessions/{self.sid}/sync?{query}")

    def test_sync_paging_and_cursor_resume(self) -> None:
        status, body = self._sync("device_id=b&limit=2")
        self.assertEqual(status, 200)
        self.assertEqual([m["sequence"] for m in body["messages"]], [1, 2])
        self.assertEqual(body["next_cursor"], 2)
        self.assertTrue(body["has_more"])
        # Omitted after resumes from the stored cursor.
        status, body = self._sync("device_id=b")
        self.assertEqual(status, 200)
        self.assertEqual([m["sequence"] for m in body["messages"]], [3])
        self.assertEqual(body["next_cursor"], 3)
        self.assertFalse(body["has_more"])
        status, body = self._sync("device_id=b")
        self.assertEqual(status, 200)
        self.assertEqual(body["messages"], [])
        self.assertEqual(body["next_cursor"], 3)

    def test_sync_query_param_validation(self) -> None:
        for query, field in (
            ("", "device_id"),
            ("device_id=", "device_id"),
            ("device_id=b&device_id=b", "device_id"),
            ("device_id=b&after=-1", "after"),
            ("device_id=b&after=x", "after"),
            ("device_id=b&after=1&after=2", "after"),
            ("device_id=b&limit=0", "limit"),
            ("device_id=b&limit=101", "limit"),
            ("device_id=b&limit=x", "limit"),
        ):
            status, body = self._sync(query)
            self.assertEqual(status, 400, query)
            self.assertEqual(body["field"], field, query)

    def test_sync_unknown_session_404(self) -> None:
        status, body = self._request(
            "GET", "/v1/group-sessions/missing/sync?device_id=b")
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "session_id")

    def test_sync_non_member_409(self) -> None:
        status, body = self._sync("device_id=c")
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "device_id")

    def test_sync_unknown_device_409(self) -> None:
        status, body = self._sync("device_id=ghost")
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "device_id")

    def test_checkpoint_lifecycle(self) -> None:
        path = f"/v1/group-sessions/{self.sid}/sync/checkpoint"
        status, body = self._request(
            "POST", path, {"device_id": "a", "cursor": 3})
        self.assertEqual(status, 201)
        self.assertEqual(set(body),
                         {"session_id", "device_id", "cursor", "updated_at"})
        self.assertEqual(body["cursor"], 3)
        timestamp = body["updated_at"]
        status, body = self._request(
            "POST", path, {"device_id": "a", "cursor": 3})
        self.assertEqual(status, 200)
        self.assertEqual(body["updated_at"], timestamp)
        status, body = self._request(
            "POST", path, {"device_id": "a", "cursor": 2})
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "cursor")

    def test_checkpoint_validation_and_404(self) -> None:
        path = f"/v1/group-sessions/{self.sid}/sync/checkpoint"
        status, body = self._request("POST", path, {"device_id": "a"})
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "cursor")
        status, body = self._request("POST", path, {"cursor": 1})
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "device_id")
        status, body = self._request(
            "POST", "/v1/group-sessions/missing/sync/checkpoint",
            {"device_id": "a", "cursor": 0})
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "session_id")
        status, body = self._request(
            "POST", path, {"device_id": "c", "cursor": 0})
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "device_id")


class GroupSyncPersistenceTest(unittest.TestCase):
    def _state(self, store: DeviceStore) -> dict:
        return store.snapshot_state()

    def test_cursors_persist_and_restore(self) -> None:
        service, sid = _build_populated()
        store = service.store
        # b syncs (implicit cursor advance) to 3; a sets an explicit 2.
        store.sync_group_messages(sid, "b", None, 100)
        store.set_sync_checkpoint(sid, "a", 2)
        state = self._state(store)
        self.assertIn("group_sync_cursors", state)

        restored = DeviceStore()
        restored.restore_state(state)
        # b resumes at 3 (empty page), a keeps cursor 2.
        messages, next_cursor, _ = restored.sync_group_messages(
            sid, "b", None, 100)
        self.assertEqual(messages, [])
        self.assertEqual(next_cursor, 3)
        _view, status = restored.set_sync_checkpoint(sid, "a", 2)
        self.assertEqual(status, 200)
        _view, status = restored.set_sync_checkpoint(sid, "a", 3)
        self.assertEqual(status, 201)

    def test_old_version1_file_without_section_loads_empty(self) -> None:
        service, sid = _build_populated()
        state = self._state(service.store)
        del state["group_sync_cursors"]
        restored = DeviceStore()
        restored.restore_state(state)  # must not raise
        messages, next_cursor, _ = restored.sync_group_messages(
            sid, "b", None, 100)
        self.assertEqual(len(messages), 3)
        self.assertEqual(next_cursor, 3)

    def test_malformed_section_refuses_restore(self) -> None:
        service, _sid = _build_populated()
        base = self._state(service.store)
        bad_sections = (
            "not-a-list",
            [{"session_id": "s", "device_id": "b"}],  # missing fields
            [{"session_id": "s", "device_id": "b",
              "cursor": -1, "updated_at": None}],
            [{"session_id": "s", "device_id": "b",
              "cursor": 1, "updated_at": 12345}],
            [{"session_id": "s", "cursor": 0, "updated_at": None}],
        )
        for section in bad_sections:
            with self.assertRaises(ValueError):
                DeviceStore().restore_state({**base,
                                             "group_sync_cursors": section})


if __name__ == "__main__":
    unittest.main()
