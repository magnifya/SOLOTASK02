"""Tests for the device offline inbox endpoint.

GET /v1/devices/{device_id}/inbox aggregates, read-only and under the store
lock, the unacknowledged messages of every 1:1 session the device is the
recipient of (group sessions and other devices' sessions never contribute),
ordered by (session.created_at, session_id, sequence).
"""
import json
import os
import shutil
import tempfile
import threading
import unittest
from http.client import HTTPConnection

from e2ee_backend.models import Device, SignedPreKey
from e2ee_backend.persistence import attach_persistence
from e2ee_backend.service import DeviceService, ServiceError
from e2ee_backend.http_app import create_server


class InboxMixin:
    def _build(self) -> None:
        self.service = DeviceService()
        self.service.store.add_device(Device("u", "alice", "ik"))
        # bob holds three pre-keys so several distinct 1:1 sessions can be
        # addressed to him; bob2 is a second device of the same user.
        self.service.store.add_device(Device(
            "u", "bob", "ik",
            prekeys=[SignedPreKey("pk1", "pubk1"),
                     SignedPreKey("pk2", "pubk2"),
                     SignedPreKey("pk3", "pubk3")]))
        self.service.store.add_device(
            Device("u", "bob2", "ik", prekeys=[SignedPreKey("pkB", "pubkB")]))
        self.service.store.add_device(
            Device("u", "carol", "ik", prekeys=[SignedPreKey("pkC", "pubkC")]))
        # Two sessions addressed to bob.
        self.sid1 = self.service.store.create_session(
            "alice", "bob", "pk1", "ek1").session_id
        self.sid2 = self.service.store.create_session(
            "carol", "bob", "pk2", "ek2").session_id
        for sequence in range(1, 4):
            self.service.post_message({
                "session_id": self.sid1, "sender_device_id": "alice",
                "message_id": f"a{sequence}", "sequence": sequence,
                "nonce": f"na{sequence}", "ciphertext": "ct"})
        for sequence in range(1, 3):
            self.service.post_message({
                "session_id": self.sid2, "sender_device_id": "carol",
                "message_id": f"b{sequence}", "sequence": sequence,
                "nonce": f"nb{sequence}", "ciphertext": "ct"})
        # A session addressed to bob's other device bob2 (must not leak).
        self.sid_other = self.service.store.create_session(
            "alice", "bob2", "pkB", "ekB").session_id
        self.service.post_message({
            "session_id": self.sid_other, "sender_device_id": "alice",
            "message_id": "o1", "sequence": 1,
            "nonce": "no1", "ciphertext": "ct"})
        # A session bob initiated towards carol (bob is not its recipient).
        self.sid_out = self.service.store.create_session(
            "bob", "carol", "pkC", "ekC").session_id
        self.service.post_message({
            "session_id": self.sid_out, "sender_device_id": "carol",
            "message_id": "c1", "sequence": 1,
            "nonce": "nc1", "ciphertext": "ct"})
        # A group session including bob (must not contribute).
        self.service.create_group({
            "group_id": "g1", "creator_device_id": "alice",
            "member_device_ids": ["bob"]})
        group_session = self.service.create_group_session({
            "group_id": "g1", "initiator_device_id": "alice",
            "ephemeral_key": "epk"})
        self.group_sid = group_session["session_id"]
        self.service.post_message({
            "session_id": self.group_sid, "sender_device_id": "alice",
            "message_id": "g1", "sequence": 1,
            "nonce": "ng1", "ciphertext": "ct"})

    def _inbox_ids(self, device_id="bob", limit=100):
        body = self.service.device_inbox(device_id, limit)
        return [m["message_id"] for m in body["messages"]]


class InboxServiceTest(InboxMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def _error(self, callable_):
        with self.assertRaises(ServiceError) as caught:
            callable_()
        return caught.exception

    def test_aggregates_unacked_in_session_order(self) -> None:
        body = self.service.device_inbox("bob", 100)
        self.assertEqual(list(body), ["device_id", "messages", "has_more"])
        self.assertEqual(body["device_id"], "bob")
        self.assertFalse(body["has_more"])
        self.assertEqual(self._inbox_ids(), ["a1", "a2", "a3", "b1", "b2"])
        for message in body["messages"]:
            self.assertEqual(list(message), [
                "session_id", "sender_device_id", "message_id", "sequence",
                "nonce", "ciphertext", "created_at"])
        self.assertEqual(body["messages"][0]["session_id"], self.sid1)
        self.assertEqual(body["messages"][0]["sender_device_id"], "alice")
        self.assertEqual(body["messages"][0]["sequence"], 1)
        self.assertEqual(body["messages"][3]["session_id"], self.sid2)

    def test_excludes_group_other_device_and_outgoing_sessions(self) -> None:
        ids = self._inbox_ids()
        self.assertNotIn("o1", ids)  # same user's other device bob2
        self.assertNotIn("c1", ids)  # bob is the initiator, not recipient
        self.assertNotIn("g1", ids)  # group session message
        # carol's own inbox sees the message addressed to her.
        self.assertEqual(self._inbox_ids("carol"), ["c1"])
        self.assertEqual(self._inbox_ids("bob2"), ["o1"])

    def test_limit_and_has_more(self) -> None:
        body = self.service.device_inbox("bob", 2)
        self.assertEqual([m["message_id"] for m in body["messages"]],
                         ["a1", "a2"])
        self.assertTrue(body["has_more"])
        body = self.service.device_inbox("bob", 5)
        self.assertEqual(len(body["messages"]), 5)
        self.assertFalse(body["has_more"])
        body = self.service.device_inbox("bob", 100)
        self.assertFalse(body["has_more"])

    def test_tie_break_by_session_id_code_points(self) -> None:
        # Equal created_at timestamps order by the session_id itself.
        created_at = self.service.store._sessions[self.sid1].created_at
        self.service.store._sessions[self.sid2].created_at = created_at
        first, second = sorted([self.sid1, self.sid2])
        body = self.service.device_inbox("bob", 1)
        expected_session = first
        self.assertEqual(body["messages"][0]["session_id"], expected_session)
        body = self.service.device_inbox("bob", 100)
        boundary = [m["session_id"] for m in body["messages"]]
        counts = {self.sid1: 3, self.sid2: 2}
        self.assertEqual(boundary,
                         [first] * counts[first] + [second] * counts[second])

    def test_acked_messages_disappear_via_ack_and_ack_batch(self) -> None:
        self.service.ack_message(self.sid1, {
            "device_id": "bob", "message_id": "a2", "sequence": 2})
        self.assertEqual(self._inbox_ids(), ["a1", "a3", "b1", "b2"])
        # The device sync/ack-batch confirms by per-session highest sequence.
        self.service.sync_device_ack_batch("bob", {
            "items": [{"session_id": self.sid1, "cursor": 3},
                      {"session_id": self.sid2, "cursor": 1}]})
        self.assertEqual(self._inbox_ids(), ["b2"])

    def test_unacked_delivery_record_still_listed(self) -> None:
        # A retry creates a delivery record without acking: still listed.
        self.service.retry_message(
            self.sid1, "a1", {"device_id": "bob", "attempt_id": "att-1"})
        self.assertIn("a1", self._inbox_ids())
        state = self.service.store._delivery[(self.sid1, "a1")]
        self.assertEqual(state.attempts, 1)
        self.assertFalse(state.acked)

    def test_device_unknown_or_revoked(self) -> None:
        error = self._error(lambda: self.service.device_inbox("ghost", 100))
        self.assertEqual((error.status_code, error.field),
                         (409, "device_id"))
        self.service.store.revoke_device("bob")
        error = self._error(lambda: self.service.device_inbox("bob", 100))
        self.assertEqual((error.status_code, error.field),
                         (409, "device_id"))

    def test_limit_validation(self) -> None:
        for bad in (0, 101, -1, "1", 1.5, True, None):
            error = self._error(lambda: self.service.device_inbox("bob", bad))
            self.assertEqual((error.status_code, error.field),
                             (400, "limit"), bad)

    def test_read_only_and_byte_identical(self) -> None:
        delivery_before = dict(self.service.store._delivery)
        cursors_before = dict(self.service.store._message_sync_cursors)
        first = self.service.device_inbox("bob", 100)
        second = self.service.device_inbox("bob", 100)
        self.assertEqual(json.dumps(first, ensure_ascii=False),
                         json.dumps(second, ensure_ascii=False))
        self.assertEqual(dict(self.service.store._delivery), delivery_before)
        self.assertEqual(dict(self.service.store._message_sync_cursors),
                         cursors_before)


class InboxPersistenceTest(InboxMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self._build()
        self.path = os.path.join(self.directory, "state.json")
        self.state_store = attach_persistence(self.service, self.path)

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_query_consumes_no_generation(self) -> None:
        generation = self.state_store.commit_seq
        self.service.device_inbox("bob", 100)
        self.service.device_inbox("bob", 2)
        self.assertEqual(self.state_store.commit_seq, generation)

    def test_restart_rebuilds_the_same_view(self) -> None:
        self.service.sync_device_ack_batch("bob", {
            "items": [{"session_id": self.sid1, "cursor": 2}]})
        expected = self.service.device_inbox("bob", 100)
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        self.assertEqual(restarted.device_inbox("bob", 100), expected)
        self.assertEqual(self._inbox_ids(), ["a3", "b1", "b2"])


class InboxHTTPTest(InboxMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()
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

    def _request(self, path: str):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("GET", path)
        response = conn.getresponse()
        return response.status, response.read()

    def test_default_limit_and_key_order(self) -> None:
        status, raw = self._request("/v1/devices/bob/inbox")
        self.assertEqual(status, 200)
        body = json.loads(raw.decode("utf-8"))
        self.assertEqual(list(body), ["device_id", "messages", "has_more"])
        self.assertEqual(len(body["messages"]), 5)
        self.assertFalse(body["has_more"])
        self.assertEqual(list(body["messages"][0]), [
            "session_id", "sender_device_id", "message_id", "sequence",
            "nonce", "ciphertext", "created_at"])

    def test_limit_param_and_has_more(self) -> None:
        status, raw = self._request("/v1/devices/bob/inbox?limit=2")
        self.assertEqual(status, 200)
        body = json.loads(raw.decode("utf-8"))
        self.assertEqual([m["message_id"] for m in body["messages"]],
                         ["a1", "a2"])
        self.assertTrue(body["has_more"])

    def test_limit_validation_errors(self) -> None:
        for query in ("limit=", "limit=abc", "limit=0", "limit=101",
                      "limit=-1", "limit=+1", "limit=1.5", "limit=%201",
                      "limit=1&limit=2", "limit=2&limit=2"):
            status, raw = self._request(f"/v1/devices/bob/inbox?{query}")
            body = json.loads(raw.decode("utf-8"))
            self.assertEqual(status, 400, query)
            self.assertEqual(list(body), ["message", "field"], query)
            self.assertEqual(body["field"], "limit", query)

    def test_device_errors_and_body_key_order(self) -> None:
        status, raw = self._request("/v1/devices/ghost/inbox")
        body = json.loads(raw.decode("utf-8"))
        self.assertEqual(status, 409)
        self.assertEqual(list(body), ["message", "field"])
        self.assertEqual(body["field"], "device_id")
        self.service.store.revoke_device("bob")
        status, raw = self._request("/v1/devices/bob/inbox")
        body = json.loads(raw.decode("utf-8"))
        self.assertEqual(status, 409)
        self.assertEqual(list(body), ["message", "field"])
        self.assertEqual(body["field"], "device_id")

    def test_unchanged_state_is_byte_identical(self) -> None:
        status1, raw1 = self._request("/v1/devices/bob/inbox?limit=3")
        status2, raw2 = self._request("/v1/devices/bob/inbox?limit=3")
        self.assertEqual((status1, raw1), (status2, raw2))

    def test_acked_items_vanish_over_http(self) -> None:
        self.service.sync_device_ack_batch("bob", {
            "items": [{"session_id": self.sid1, "cursor": 3},
                      {"session_id": self.sid2, "cursor": 2}]})
        status, raw = self._request("/v1/devices/bob/inbox")
        self.assertEqual(status, 200)
        body = json.loads(raw.decode("utf-8"))
        self.assertEqual(body["messages"], [])
        self.assertFalse(body["has_more"])


if __name__ == "__main__":
    unittest.main()
