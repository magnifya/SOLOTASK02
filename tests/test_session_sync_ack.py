"""Tests for the batch offline-message acknowledgement endpoint.

POST /v1/sessions/{session_id}/sync/ack commits the delivery
acknowledgements and the unified sync-cursor advance in one locked
transaction, for both 1:1 and group sessions.
"""
import json
import os
import shutil
import tempfile
import threading
import unittest
from http.client import HTTPConnection

from e2ee_backend.models import Device, SignedPreKey
from e2ee_backend.persistence import PersistenceUnavailable, attach_persistence
from e2ee_backend.service import DeviceService, ServiceError
from e2ee_backend.http_app import create_server


class _OneToOneMixin:
    def _build_11(self) -> str:
        self.service = DeviceService()
        self.service.store.add_device(Device("u", "alice", "ik"))
        self.service.store.add_device(
            Device("u", "bob", "ik", prekeys=[SignedPreKey("pk", "pubk")]))
        self.service.store.add_device(Device("u", "carol", "ik"))
        sid = self.service.store.create_session(
            "alice", "bob", "pk", "ek").session_id
        for sequence in range(1, 4):
            self.service.post_message({
                "session_id": sid, "sender_device_id": "alice",
                "message_id": f"m{sequence}", "sequence": sequence,
                "nonce": f"n{sequence}", "ciphertext": "ct"})
        return sid


class _GroupMixin:
    def _build_group(self) -> str:
        self.service = DeviceService()
        for device_id in ("alice", "bob", "carol"):
            self.service.store.add_device(Device("u", device_id, "ik"))
        self.service.create_group({
            "group_id": "g1", "creator_device_id": "alice",
            "member_device_ids": ["bob", "carol"]})
        gs = self.service.create_group_session({
            "group_id": "g1", "initiator_device_id": "alice",
            "ephemeral_key": "epk"})
        sid = gs["session_id"]
        # seq 1 by alice, seq 2 by bob, seq 3 by alice.
        self.service.post_message({
            "session_id": sid, "sender_device_id": "alice",
            "message_id": "g1", "sequence": 1, "nonce": "gn1",
            "ciphertext": "ct"})
        self.service.post_message({
            "session_id": sid, "sender_device_id": "bob",
            "message_id": "g2", "sequence": 2, "nonce": "gn2",
            "ciphertext": "ct"})
        self.service.post_message({
            "session_id": sid, "sender_device_id": "alice",
            "message_id": "g3", "sequence": 3, "nonce": "gn3",
            "ciphertext": "ct"})
        return sid


class SyncAckServiceTest(_OneToOneMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.sid = self._build_11()

    def _error(self, callable_):
        with self.assertRaises(ServiceError) as caught:
            callable_()
        return caught.exception

    def test_forward_acks_range_and_advances_cursor(self) -> None:
        body, status = self.service.sync_session_ack(
            self.sid, {"device_id": "bob", "cursor": 2})
        self.assertEqual(status, 201)
        self.assertEqual(list(body),
                         ["session_id", "device_id", "cursor", "updated_at"])
        self.assertEqual(body["session_id"], self.sid)
        self.assertEqual(body["device_id"], "bob")
        self.assertEqual(body["cursor"], 2)
        self.assertIn("+00:00", body["updated_at"])
        m1 = self.service.store._delivery[(self.sid, "m1")]
        m2 = self.service.store._delivery[(self.sid, "m2")]
        self.assertTrue(m1.acked)
        self.assertEqual(m1.ack_sequence, 1)
        self.assertEqual(m1.attempts, 0)
        self.assertTrue(m2.acked)
        self.assertEqual(m2.ack_sequence, 2)
        self.assertEqual(m2.attempts, 0)
        # Only messages up to the cursor were acked.
        self.assertNotIn((self.sid, "m3"), self.service.store._delivery)
        self.assertEqual(
            self.service.store._message_sync_cursors[(self.sid, "bob")].cursor,
            2)

    def test_ack_only_marks_unsynced_range(self) -> None:
        # Advance the cursor once, then ack the next range: only (1,3] acks.
        self.service.sync_session_ack(
            self.sid, {"device_id": "bob", "cursor": 1})
        self.assertTrue(self.service.store._delivery[(self.sid, "m1")].acked)
        self.assertNotIn((self.sid, "m2"), self.service.store._delivery)
        body, status = self.service.sync_session_ack(
            self.sid, {"device_id": "bob", "cursor": 3})
        self.assertEqual(status, 201)
        self.assertEqual(body["cursor"], 3)
        self.assertTrue(
            self.service.store._delivery[(self.sid, "m2")].acked)
        self.assertTrue(
            self.service.store._delivery[(self.sid, "m3")].acked)
        self.assertEqual(
            self.service.store._message_sync_cursors[(self.sid, "bob")].cursor,
            3)

    def test_equal_cursor_is_200_unchanged_and_does_not_persist(self) -> None:
        body, status = self.service.sync_session_ack(
            self.sid, {"device_id": "bob", "cursor": 2})
        self.assertEqual(status, 201)
        timestamp = body["updated_at"]
        body, status = self.service.sync_session_ack(
            self.sid, {"device_id": "bob", "cursor": 2})
        self.assertEqual(status, 200)
        self.assertEqual(body["updated_at"], timestamp)
        self.assertEqual(body["cursor"], 2)

    def test_zero_equal_on_fresh_cursor_reports_session_created_at(self) -> None:
        session = self.service.get_session(self.sid)
        body, status = self.service.sync_session_ack(
            self.sid, {"device_id": "bob", "cursor": 0})
        self.assertEqual(status, 200)
        self.assertEqual(body["cursor"], 0)
        self.assertEqual(body["updated_at"], session["created_at"])
        self.assertNotIn((self.sid, "bob"),
                         self.service.store._message_sync_cursors)
        self.assertEqual(self.service.store._delivery, {})

    def test_backward_and_over_max_are_409_cursor(self) -> None:
        self.service.sync_session_ack(
            self.sid, {"device_id": "bob", "cursor": 2})
        error = self._error(lambda: self.service.sync_session_ack(
            self.sid, {"device_id": "bob", "cursor": 1}))
        self.assertEqual((error.status_code, error.field), (409, "cursor"))
        error = self._error(lambda: self.service.sync_session_ack(
            self.sid, {"device_id": "bob", "cursor": 4}))
        self.assertEqual((error.status_code, error.field), (409, "cursor"))

    def test_only_recipient_may_ack_11(self) -> None:
        # The initiator has no receive side in a 1:1 session.
        error = self._error(lambda: self.service.sync_session_ack(
            self.sid, {"device_id": "alice", "cursor": 3}))
        self.assertEqual((error.status_code, error.field), (409, "device_id"))
        error = self._error(lambda: self.service.sync_session_ack(
            self.sid, {"device_id": "carol", "cursor": 3}))
        self.assertEqual((error.status_code, error.field), (409, "device_id"))

    def test_unknown_revoked_device_and_session(self) -> None:
        error = self._error(lambda: self.service.sync_session_ack(
            "missing", {"device_id": "bob", "cursor": 0}))
        self.assertEqual((error.status_code, error.field), (404, "session_id"))
        error = self._error(lambda: self.service.sync_session_ack(
            self.sid, {"device_id": "ghost", "cursor": 0}))
        self.assertEqual((error.status_code, error.field), (409, "device_id"))
        self.service.store.revoke_device("bob")
        error = self._error(lambda: self.service.sync_session_ack(
            self.sid, {"device_id": "bob", "cursor": 0}))
        self.assertEqual((error.status_code, error.field), (409, "device_id"))

    def test_body_validation(self) -> None:
        cases = [
            (None, "request_body"),
            ([], "request_body"),
            ("x", "request_body"),
            ({"device_id": "bob"}, "cursor"),
            ({"device_id": "bob", "cursor": "2"}, "cursor"),
            ({"device_id": "bob", "cursor": True}, "cursor"),
            ({"device_id": "bob", "cursor": 1.5}, "cursor"),
            ({"device_id": "bob", "cursor": -1}, "cursor"),
            ({"cursor": 1}, "device_id"),
            ({"device_id": "", "cursor": 1}, "device_id"),
            ({"device_id": 5, "cursor": 1}, "device_id"),
            ({"device_id": True, "cursor": 1}, "device_id"),
        ]
        for payload, field in cases:
            error = self._error(lambda: self.service.sync_session_ack(
                self.sid, payload))
            self.assertEqual((error.status_code, error.field), (400, field),
                             payload)

    def test_explicit_after_sync_has_no_side_effect_on_ack(self) -> None:
        # An explicit after read must not move the stored cursor, so the ack
        # still sees old cursor 0 and acks the whole range.
        self.service.sync_session_messages(self.sid, "bob", 0, 100)
        self.assertNotIn((self.sid, "bob"),
                         self.service.store._message_sync_cursors)
        body, status = self.service.sync_session_ack(
            self.sid, {"device_id": "bob", "cursor": 3})
        self.assertEqual(status, 201)
        for mid in ("m1", "m2", "m3"):
            self.assertTrue(
                self.service.store._delivery[(self.sid, mid)].acked)


class SyncAckGroupServiceTest(_GroupMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.sid = self._build_group()

    def test_frozen_member_acks_and_skips_own_messages(self) -> None:
        body, status = self.service.sync_session_ack(
            self.sid, {"device_id": "bob", "cursor": 3})
        self.assertEqual(status, 201)
        self.assertEqual(body["cursor"], 3)
        group_delivery = self.service.store._group_delivery
        # g1 and g3 were sent by alice: bob acks them.
        self.assertTrue(group_delivery[(self.sid, "g1", "bob")].acked)
        self.assertEqual(
            group_delivery[(self.sid, "g1", "bob")].ack_sequence, 1)
        self.assertTrue(group_delivery[(self.sid, "g3", "bob")].acked)
        # g2 was sent by bob itself: skipped, no record.
        self.assertNotIn((self.sid, "g2", "bob"), group_delivery)

    def test_attempts_are_unchanged(self) -> None:
        self.service.retry_message(
            self.sid, "g1", {"device_id": "bob", "attempt_id": "att-1"})
        self.service.sync_session_ack(
            self.sid, {"device_id": "bob", "cursor": 1})
        state = self.service.store._group_delivery[(self.sid, "g1", "bob")]
        self.assertTrue(state.acked)
        self.assertEqual(state.attempts, 1)
        self.assertEqual(state.attempt_ids, {"att-1"})

    def test_devices_ack_independently(self) -> None:
        self.service.sync_session_ack(
            self.sid, {"device_id": "bob", "cursor": 3})
        group_delivery = self.service.store._group_delivery
        # carol has acked nothing yet.
        self.assertNotIn((self.sid, "g1", "carol"), group_delivery)
        self.service.sync_session_ack(
            self.sid, {"device_id": "carol", "cursor": 3})
        self.assertTrue(group_delivery[(self.sid, "g1", "carol")].acked)
        self.assertTrue(group_delivery[(self.sid, "g2", "carol")].acked)
        # bob's own-message skip does not leak to carol.
        self.assertNotIn((self.sid, "g2", "bob"), group_delivery)

    def test_frozen_member_after_removal_still_acks(self) -> None:
        # carol is frozen; removing them from the current group does not widen
        # or shrink the frozen snapshot.
        self.service.remove_group_member(
            "g1", {"actor_device_id": "alice", "device_id": "carol"})
        body, status = self.service.sync_session_ack(
            self.sid, {"device_id": "carol", "cursor": 3})
        self.assertEqual(status, 201)
        self.assertTrue(
            self.service.store._group_delivery[
                (self.sid, "g1", "carol")].acked)

    def test_later_added_member_cannot_ack(self) -> None:
        self.service.store.add_device(Device("u", "dave", "ik"))
        self.service.add_group_member(
            "g1", {"actor_device_id": "alice", "device_id": "dave"})
        with self.assertRaises(ServiceError) as caught:
            self.service.sync_session_ack(
                self.sid, {"device_id": "dave", "cursor": 0})
        self.assertEqual(
            (caught.exception.status_code, caught.exception.field),
            (409, "device_id"))


class SyncAckPersistenceTest(_OneToOneMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.service = DeviceService()
        self.state_store = attach_persistence(
            self.service, os.path.join(self.directory, "state.json"))
        self.service.store.add_device(Device("u", "alice", "ik"))
        self.service.store.add_device(
            Device("u", "bob", "ik",
                   prekeys=[SignedPreKey("pk", "pubk")]))
        self.sid = self.service.store.create_session(
            "alice", "bob", "pk", "ek").session_id
        for sequence in range(1, 4):
            self.service.post_message({
                "session_id": self.sid, "sender_device_id": "alice",
                "message_id": f"m{sequence}", "sequence": sequence,
                "nonce": f"n{sequence}", "ciphertext": "ct"})

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_acks_and_cursor_persist_together(self) -> None:
        self.service.sync_session_ack(
            self.sid, {"device_id": "bob", "cursor": 3})
        path = self.state_store.path
        service = DeviceService()
        attach_persistence(service, path)
        for mid in ("m1", "m2", "m3"):
            state = service.store._delivery[(self.sid, mid)]
            self.assertTrue(state.acked)
        self.assertEqual(
            service.store._message_sync_cursors[(self.sid, "bob")].cursor, 3)

    def test_failed_write_rolls_back_acks_and_cursor(self) -> None:
        def raise_oserror(state):
            raise OSError("simulated disk failure")

        self.state_store.save = raise_oserror
        generation = self.state_store.commit_seq
        with self.assertRaises(PersistenceUnavailable):
            self.service.sync_session_ack(
                self.sid, {"device_id": "bob", "cursor": 2})
        self.assertEqual(self.service.store._delivery, {})
        self.assertNotIn((self.sid, "bob"),
                         self.service.store._message_sync_cursors)
        self.assertEqual(self.state_store.commit_seq, generation)

    def test_equal_ack_does_not_consume_generation(self) -> None:
        generation = self.state_store.commit_seq
        body, status = self.service.sync_session_ack(
            self.sid, {"device_id": "bob", "cursor": 0})
        self.assertEqual(status, 200)
        self.assertEqual(self.state_store.commit_seq, generation)


class SyncAckGroupPersistenceRollbackTest(_GroupMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.service = DeviceService()
        self.state_store = attach_persistence(
            self.service, os.path.join(self.directory, "state.json"))
        for device_id in ("alice", "bob"):
            self.service.store.add_device(Device("u", device_id, "ik"))
        self.service.create_group({
            "group_id": "g1", "creator_device_id": "alice",
            "member_device_ids": ["bob"]})
        gs = self.service.create_group_session({
            "group_id": "g1", "initiator_device_id": "alice",
            "ephemeral_key": "epk"})
        self.sid = gs["session_id"]
        self.service.post_message({
            "session_id": self.sid, "sender_device_id": "alice",
            "message_id": "g1", "sequence": 1, "nonce": "gn1",
            "ciphertext": "ct"})

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_failed_write_rolls_back_group_delivery_and_cursor(self) -> None:
        def raise_oserror(state):
            raise OSError("simulated disk failure")

        self.state_store.save = raise_oserror
        with self.assertRaises(PersistenceUnavailable):
            self.service.sync_session_ack(
                self.sid, {"device_id": "bob", "cursor": 1})
        self.assertEqual(self.service.store._group_delivery, {})
        self.assertNotIn((self.sid, "bob"),
                         self.service.store._message_sync_cursors)


class SyncAckHTTPTest(_OneToOneMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.sid = self._build_11()
        self.server, self.service = create_server(
            "127.0.0.1", 0, self.service)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _request(self, method: str, path: str, body=None, raw=None):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        if raw is not None:
            conn.request(method, path, raw,
                         {"Content-Type": "application/json"})
        else:
            data = json.dumps(body).encode("utf-8") if body is not None \
                else b""
            conn.request(method, path, data,
                         {"Content-Type": "application/json"})
        response = conn.getresponse()
        return response.status, json.loads(response.read().decode("utf-8"))

    def test_forward_and_equal(self) -> None:
        path = f"/v1/sessions/{self.sid}/sync/ack"
        status, body = self._request(
            "POST", path, {"device_id": "bob", "cursor": 2})
        self.assertEqual(status, 201)
        self.assertEqual(list(body),
                         ["session_id", "device_id", "cursor", "updated_at"])
        status, body = self._request(
            "POST", path, {"device_id": "bob", "cursor": 2})
        self.assertEqual(status, 200)

    def test_error_statuses(self) -> None:
        path = f"/v1/sessions/{self.sid}/sync/ack"
        status, body = self._request("POST", path, raw="{bad")
        self.assertEqual((status, body["field"]), (400, "request_body"))
        status, body = self._request("POST", path, [1])
        self.assertEqual((status, body["field"]), (400, "request_body"))
        status, body = self._request(
            "POST", path, {"device_id": "bob", "cursor": 9})
        self.assertEqual((status, body["field"]), (409, "cursor"))
        status, body = self._request(
            "POST", "/v1/sessions/missing/sync/ack",
            {"device_id": "bob", "cursor": 0})
        self.assertEqual((status, body["field"]), (404, "session_id"))
        status, body = self._request(
            "POST", path, {"device_id": "carol", "cursor": 0})
        self.assertEqual((status, body["field"]), (409, "device_id"))
        self.assertEqual(list(body), ["message", "field"])


if __name__ == "__main__":
    unittest.main()
