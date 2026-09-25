"""Tests for the offline batch ack endpoint (POST .../sync/ack)."""
import copy
import json
import threading
import unittest
from http.client import HTTPConnection

from e2ee_backend.http_app import create_server
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

    def _error(self, callable_):
        with self.assertRaises(ServiceError) as caught:
            callable_()
        return caught.exception


class SessionSyncAckServiceTest(_FixtureMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.sid = self._build()

    def test_forward_ack_marks_messages_and_advances_cursor(self) -> None:
        body, status = self.service.sync_session_ack(
            self.sid, {"device_id": "bob", "cursor": 2})
        self.assertEqual(status, 201)
        self.assertEqual(list(body),
                         ["session_id", "device_id", "cursor", "updated_at"])
        self.assertEqual(
            (body["session_id"], body["device_id"], body["cursor"]),
            (self.sid, "bob", 2))
        self.assertTrue(body["updated_at"].endswith("+00:00"))
        store = self.service.store
        self.assertEqual(
            store._message_sync_cursors[(self.sid, "bob")].cursor, 2)
        self.assertTrue(store._delivery[(self.sid, "m1")].acked)
        self.assertTrue(store._delivery[(self.sid, "m2")].acked)
        self.assertNotIn((self.sid, "m3"), store._delivery)
        # The remaining message is acked by a later forward move.
        body, status = self.service.sync_session_ack(
            self.sid, {"device_id": "bob", "cursor": 3})
        self.assertEqual(status, 201)
        self.assertTrue(store._delivery[(self.sid, "m3")].acked)

    def test_ack_preserves_attempts(self) -> None:
        self.service.retry_message(
            self.sid, "m1", {"device_id": "bob", "attempt_id": "a1"})
        self.service.sync_session_ack(
            self.sid, {"device_id": "bob", "cursor": 1})
        view = self.service.message_status(self.sid, "m1", "bob")
        self.assertEqual(view["status"], "acked")
        self.assertEqual(view["attempts"], 1)

    def test_equal_cursor_is_idempotent_200(self) -> None:
        body, status = self.service.sync_session_ack(
            self.sid, {"device_id": "bob", "cursor": 2})
        self.assertEqual(status, 201)
        timestamp = body["updated_at"]
        body, status = self.service.sync_session_ack(
            self.sid, {"device_id": "bob", "cursor": 2})
        self.assertEqual(status, 200)
        self.assertEqual(body["updated_at"], timestamp)

    def test_zero_ack_on_fresh_device_is_200_noop(self) -> None:
        session = self.service.store._sessions[self.sid]
        body, status = self.service.sync_session_ack(
            self.sid, {"device_id": "bob", "cursor": 0})
        self.assertEqual(status, 200)
        self.assertEqual(body["cursor"], 0)
        self.assertEqual(body["updated_at"], session.created_at)
        self.assertNotIn((self.sid, "bob"),
                         self.service.store._message_sync_cursors)
        self.assertEqual(self.service.store._delivery, {})

    def test_cursor_conflicts(self) -> None:
        self.service.sync_session_ack(
            self.sid, {"device_id": "bob", "cursor": 2})
        error = self._error(lambda: self.service.sync_session_ack(
            self.sid, {"device_id": "bob", "cursor": 1}))
        self.assertEqual((error.status_code, error.field), (409, "cursor"))
        error = self._error(lambda: self.service.sync_session_ack(
            self.sid, {"device_id": "bob", "cursor": 4}))
        self.assertEqual((error.status_code, error.field), (409, "cursor"))

    def test_validation(self) -> None:
        cases = [
            ({"device_id": "bob"}, "cursor"),
            ({"device_id": "bob", "cursor": "1"}, "cursor"),
            ({"device_id": "bob", "cursor": True}, "cursor"),
            ({"device_id": "bob", "cursor": -1}, "cursor"),
            ({"cursor": 1}, "device_id"),
            ({"device_id": "", "cursor": 1}, "device_id"),
            ({"device_id": 7, "cursor": 1}, "device_id"),
            ([], "request_body"),
            (None, "request_body"),
        ]
        for payload, field in cases:
            error = self._error(lambda: self.service.sync_session_ack(
                self.sid, payload))
            self.assertEqual((error.status_code, error.field), (400, field),
                             payload)

    def test_authorization(self) -> None:
        error = self._error(lambda: self.service.sync_session_ack(
            "missing", {"device_id": "bob", "cursor": 0}))
        self.assertEqual((error.status_code, error.field),
                         (404, "session_id"))
        error = self._error(lambda: self.service.sync_session_ack(
            self.sid, {"device_id": "ghost", "cursor": 0}))
        self.assertEqual((error.status_code, error.field), (409, "device_id"))
        # carol is registered but not a session participant.
        error = self._error(lambda: self.service.sync_session_ack(
            self.sid, {"device_id": "carol", "cursor": 0}))
        self.assertEqual((error.status_code, error.field), (409, "device_id"))
        # The 1:1 initiator may sync but may not batch-ack.
        error = self._error(lambda: self.service.sync_session_ack(
            self.sid, {"device_id": "alice", "cursor": 0}))
        self.assertEqual((error.status_code, error.field), (409, "device_id"))
        self.service.store.revoke_device("bob")
        error = self._error(lambda: self.service.sync_session_ack(
            self.sid, {"device_id": "bob", "cursor": 0}))
        self.assertEqual((error.status_code, error.field), (409, "device_id"))

    def test_group_batch_ack_skips_own_messages(self) -> None:
        for device_id in ("creator", "dave", "erin"):
            self.service.store.add_device(Device("u", device_id, "ik"))
        self.service.create_group({
            "group_id": "g1", "creator_device_id": "creator",
            "member_device_ids": ["dave", "erin"]})
        group_session = self.service.create_group_session({
            "group_id": "g1", "initiator_device_id": "creator",
            "ephemeral_key": "epk"})
        gsid = group_session["session_id"]
        senders = {"g1": "creator", "g2": "dave", "g3": "erin"}
        for sequence, (message_id, sender) in enumerate(
                senders.items(), start=1):
            self.service.post_message({
                "session_id": gsid, "sender_device_id": sender,
                "message_id": message_id, "sequence": sequence,
                "nonce": f"gn{sequence}", "ciphertext": "ct"})
        body, status = self.service.sync_session_ack(
            gsid, {"device_id": "dave", "cursor": 3})
        self.assertEqual(status, 201)
        self.assertEqual(body["cursor"], 3)
        store = self.service.store
        # dave acked the other members' messages but not their own.
        self.assertTrue(store._group_delivery[(gsid, "g1", "dave")].acked)
        self.assertNotIn((gsid, "g2", "dave"), store._group_delivery)
        self.assertTrue(store._group_delivery[(gsid, "g3", "dave")].acked)
        # Other devices are untouched.
        self.assertNotIn((gsid, "g1", "erin"), store._group_delivery)
        # A non-member cannot ack; a member removed after the freeze still
        # can.
        error = self._error(lambda: self.service.sync_session_ack(
            gsid, {"device_id": "alice", "cursor": 1}))
        self.assertEqual((error.status_code, error.field), (409, "device_id"))
        self.service.remove_group_member(
            "g1", {"actor_device_id": "creator", "device_id": "erin"})
        _, status = self.service.sync_session_ack(
            gsid, {"device_id": "erin", "cursor": 3})
        self.assertEqual(status, 201)
        self.assertTrue(store._group_delivery[(gsid, "g1", "erin")].acked)
        self.assertNotIn((gsid, "g3", "erin"), store._group_delivery)

    def test_storage_reason_codes(self) -> None:
        store = self.service.store

        def reason(callable_):
            with self.assertRaises(MessageSyncError) as caught:
                callable_()
            return caught.exception.reason

        self.assertEqual(reason(lambda: store.message_sync_ack(
            "missing", "bob", 0)), MESSAGE_SYNC_SESSION_UNKNOWN)
        self.assertEqual(reason(lambda: store.message_sync_ack(
            self.sid, "ghost", 0)), MESSAGE_SYNC_DEVICE_UNKNOWN)
        self.assertEqual(reason(lambda: store.message_sync_ack(
            self.sid, "carol", 0)), MESSAGE_SYNC_DEVICE_NOT_PARTICIPANT)
        self.assertEqual(reason(lambda: store.message_sync_ack(
            self.sid, "alice", 0)), MESSAGE_SYNC_DEVICE_NOT_PARTICIPANT)
        self.assertEqual(reason(lambda: store.message_sync_ack(
            self.sid, "bob", 99)), MESSAGE_SYNC_CURSOR_CONFLICT)
        store.revoke_device("bob")
        self.assertEqual(reason(lambda: store.message_sync_ack(
            self.sid, "bob", 0)), MESSAGE_SYNC_DEVICE_INACTIVE)


class SessionSyncAckPersistenceTest(_FixtureMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.sid = self._build()

    def test_ack_state_survives_snapshot_restore(self) -> None:
        self.service.retry_message(
            self.sid, "m1", {"device_id": "bob", "attempt_id": "a1"})
        self.service.sync_session_ack(
            self.sid, {"device_id": "bob", "cursor": 2})
        document = self.service.store.snapshot_state()
        restored = DeviceStore()
        restored.restore_state(copy.deepcopy(document))
        cursor = restored._message_sync_cursors[(self.sid, "bob")]
        self.assertEqual(cursor.cursor, 2)
        self.assertTrue(cursor.updated_at)
        m1 = restored._delivery[(self.sid, "m1")]
        self.assertTrue(m1.acked)
        self.assertEqual(m1.attempts, 1)
        self.assertEqual(m1.ack_sequence, 1)
        self.assertTrue(restored._delivery[(self.sid, "m2")].acked)
        self.assertNotIn((self.sid, "m3"), restored._delivery)


class SessionSyncAckRollbackTest(unittest.TestCase):
    """A failed durable write rolls back cursor and delivery together."""

    def setUp(self) -> None:
        import os
        import tempfile
        from e2ee_backend.persistence import (
            PersistenceUnavailable, attach_persistence)
        self.PersistenceUnavailable = PersistenceUnavailable
        self.directory = tempfile.mkdtemp()
        self.service = DeviceService()
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

    def test_failed_ack_rolls_back_cursor_and_delivery(self) -> None:
        self._fail_writes()
        with self.assertRaises(self.PersistenceUnavailable):
            self.service.sync_session_ack(
                self.sid, {"device_id": "bob", "cursor": 2})
        self.assertNotIn((self.sid, "bob"),
                         self.service.store._message_sync_cursors)
        self.assertEqual(self.service.store._delivery, {})

    def test_equal_ack_does_not_persist_or_consume_generation(self) -> None:
        self.service.sync_session_ack(
            self.sid, {"device_id": "bob", "cursor": 2})
        generation = self.state_store.commit_seq
        _, status = self.service.sync_session_ack(
            self.sid, {"device_id": "bob", "cursor": 2})
        self.assertEqual(status, 200)
        self.assertEqual(self.state_store.commit_seq, generation)


class SessionSyncAckHTTPTest(unittest.TestCase):
    def setUp(self) -> None:
        self.server, self.service = create_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        self.service.store.add_device(Device("u", "alice", "ik"))
        self.service.store.add_device(
            Device("u", "bob", "ik", prekeys=[SignedPreKey("pk", "pubk")]))
        session = self.service.store.create_session(
            "alice", "bob", "pk", "ek")
        self.sid = session.session_id
        for sequence in range(1, 4):
            self.service.post_message({
                "session_id": self.sid, "sender_device_id": "alice",
                "message_id": f"m{sequence}", "sequence": sequence,
                "nonce": f"n{sequence}", "ciphertext": "ct"})

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _request(self, method: str, path: str, body=None, raw=None):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        if raw is not None:
            data = raw
        else:
            data = json.dumps(body).encode("utf-8") if body is not None \
                else None
        headers = {"Content-Type": "application/json"} if data else {}
        conn.request(method, path, data, headers)
        response = conn.getresponse()
        return response.status, json.loads(response.read().decode("utf-8"))

    def test_ack_forward_then_equal(self) -> None:
        path = f"/v1/sessions/{self.sid}/sync/ack"
        status, body = self._request(
            "POST", path, {"device_id": "bob", "cursor": 2})
        self.assertEqual(status, 201)
        self.assertEqual(list(body),
                         ["session_id", "device_id", "cursor", "updated_at"])
        timestamp = body["updated_at"]
        status, body = self._request(
            "POST", path, {"device_id": "bob", "cursor": 2})
        self.assertEqual(status, 200)
        self.assertEqual(body["updated_at"], timestamp)
        # The acked state is visible through the delivery status endpoint.
        status, body = self._request(
            "GET", f"/v1/messages/{self.sid}/status/m1?device_id=bob")
        self.assertEqual((status, body["status"]), (200, "acked"))

    def test_ack_validation_and_auth(self) -> None:
        path = f"/v1/sessions/{self.sid}/sync/ack"
        status, body = self._request("POST", path, {"device_id": "bob"})
        self.assertEqual((status, body["field"]), (400, "cursor"))
        self.assertEqual(list(body), ["message", "field"])
        status, body = self._request(
            "POST", path, {"device_id": "bob", "cursor": True})
        self.assertEqual((status, body["field"]), (400, "cursor"))
        status, body = self._request("POST", path, {"cursor": 1})
        self.assertEqual((status, body["field"]), (400, "device_id"))
        status, body = self._request("POST", path, [1, 2])
        self.assertEqual((status, body["field"]), (400, "request_body"))
        status, body = self._request("POST", path, raw=b"{not json")
        self.assertEqual((status, body["field"]), (400, "request_body"))
        status, body = self._request(
            "POST", "/v1/sessions/missing/sync/ack",
            {"device_id": "bob", "cursor": 0})
        self.assertEqual((status, body["field"]), (404, "session_id"))
        status, body = self._request(
            "POST", path, {"device_id": "alice", "cursor": 1})
        self.assertEqual((status, body["field"]), (409, "device_id"))
        status, body = self._request(
            "POST", path, {"device_id": "bob", "cursor": 9})
        self.assertEqual((status, body["field"]), (409, "cursor"))

    def test_ack_malformed_path_is_404(self) -> None:
        status, body = self._request(
            "POST", "/v1/sessions/sync/ack",
            {"device_id": "bob", "cursor": 0})
        self.assertEqual((status, body["field"]), (404, "session_id"))


if __name__ == "__main__":
    unittest.main()
