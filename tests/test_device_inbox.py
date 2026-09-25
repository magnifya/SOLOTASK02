"""Tests for the device offline inbox endpoint.

GET /v1/devices/{device_id}/inbox aggregates, read-only and under the store
lock, the unconfirmed messages of every 1:1 session the device receives on
(group sessions excluded), ordered by (session created_at, session_id,
sequence) and capped by a strict single-value ``limit`` (default 100).
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
        # bob holds two pre-keys so two distinct 1:1 sessions addressed to
        # bob can be created; carol is bob's sibling device on the same user.
        self.service.store.add_device(Device(
            "u", "bob", "ik",
            prekeys=[SignedPreKey("pk1", "pubk1"),
                     SignedPreKey("pk2", "pubk2")]))
        self.service.store.add_device(Device(
            "u", "carol", "ik", prekeys=[SignedPreKey("pk3", "pubk3")]))
        self.sid1 = self.service.store.create_session(
            "alice", "bob", "pk1", "ek1").session_id
        self.sid2 = self.service.store.create_session(
            "alice", "bob", "pk2", "ek2").session_id
        # A session addressed to carol must never leak into bob's inbox.
        self.sid_carol = self.service.store.create_session(
            "alice", "carol", "pk3", "ek3").session_id
        for sequence in range(1, 4):
            self.service.post_message({
                "session_id": self.sid1, "sender_device_id": "alice",
                "message_id": f"a{sequence}", "sequence": sequence,
                "nonce": f"na{sequence}", "ciphertext": "ct"})
        for sequence in range(1, 3):
            self.service.post_message({
                "session_id": self.sid2, "sender_device_id": "alice",
                "message_id": f"b{sequence}", "sequence": sequence,
                "nonce": f"nb{sequence}", "ciphertext": "ct"})
        self.service.post_message({
            "session_id": self.sid_carol, "sender_device_id": "alice",
            "message_id": "c1", "sequence": 1, "nonce": "nc1",
            "ciphertext": "ct"})

    def _group_session(self) -> str:
        self.service.create_group({
            "group_id": "g1", "creator_device_id": "alice",
            "member_device_ids": ["bob"]})
        gs = self.service.create_group_session({
            "group_id": "g1", "initiator_device_id": "alice",
            "ephemeral_key": "epk"})
        sid = gs["session_id"]
        self.service.post_message({
            "session_id": sid, "sender_device_id": "alice",
            "message_id": "g1", "sequence": 1, "nonce": "gn1",
            "ciphertext": "ct"})
        return sid


class InboxServiceTest(InboxMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def _error(self, callable_):
        with self.assertRaises(ServiceError) as caught:
            callable_()
        return caught.exception

    def test_aggregates_unacked_1to1_messages_in_order(self) -> None:
        body = self.service.device_inbox("bob", 100)
        self.assertEqual(list(body), ["device_id", "messages", "has_more"])
        self.assertEqual(body["device_id"], "bob")
        self.assertFalse(body["has_more"])
        messages = body["messages"]
        self.assertEqual([m["message_id"] for m in messages],
                         ["a1", "a2", "a3", "b1", "b2"])
        for message in messages:
            self.assertEqual(
                list(message),
                ["session_id", "sender_device_id", "message_id", "sequence",
                 "nonce", "ciphertext", "created_at"])
        self.assertEqual([m["sequence"] for m in messages], [1, 2, 3, 1, 2])

    def test_session_created_at_then_session_id_then_sequence(self) -> None:
        # Force identical created_at on bob's two sessions: the tie breaks
        # on session_id codepoint order.
        first, second = sorted([self.sid1, self.sid2])
        sessions = self.service.store._sessions
        sessions[self.sid2].created_at = sessions[self.sid1].created_at
        body = self.service.device_inbox("bob", 100)
        expected = []
        for sid in (first, second):
            expected.extend(m.message_id
                            for m in self.service.store._messages[sid])
        self.assertEqual([m["message_id"] for m in body["messages"]],
                         expected)

    def test_other_device_of_same_user_is_excluded(self) -> None:
        body = self.service.device_inbox("bob", 100)
        self.assertNotIn("c1", [m["message_id"] for m in body["messages"]])
        carol = self.service.device_inbox("carol", 100)
        self.assertEqual([m["message_id"] for m in carol["messages"]], ["c1"])

    def test_group_session_messages_are_excluded(self) -> None:
        self._group_session()
        body = self.service.device_inbox("bob", 100)
        self.assertEqual([m["message_id"] for m in body["messages"]],
                         ["a1", "a2", "a3", "b1", "b2"])

    def test_limit_and_has_more(self) -> None:
        body = self.service.device_inbox("bob", 2)
        self.assertEqual([m["message_id"] for m in body["messages"]],
                         ["a1", "a2"])
        self.assertTrue(body["has_more"])
        body = self.service.device_inbox("bob", 5)
        self.assertEqual(len(body["messages"]), 5)
        self.assertFalse(body["has_more"])

    def test_acked_messages_disappear(self) -> None:
        # Per-message ack plus a sync-ack cursor advance.
        self.service.ack_message(self.sid1, {
            "device_id": "bob", "message_id": "a1", "sequence": 1})
        self.service.sync_session_ack(self.sid2, {
            "device_id": "bob", "cursor": 1})
        body = self.service.device_inbox("bob", 100)
        self.assertEqual([m["message_id"] for m in body["messages"]],
                         ["a2", "a3", "b2"])

    def test_ack_batch_clears_whole_sessions(self) -> None:
        self.service.sync_device_ack_batch("bob", {
            "items": [{"session_id": self.sid1, "cursor": 3},
                      {"session_id": self.sid2, "cursor": 2}]})
        body = self.service.device_inbox("bob", 100)
        self.assertEqual(body["messages"], [])
        self.assertFalse(body["has_more"])

    def test_read_only_and_byte_identical(self) -> None:
        self.service.retry_message(
            self.sid1, "a1", {"device_id": "bob", "attempt_id": "att-1"})
        before = (dict(self.service.store._message_sync_cursors),
                  {k: (v.attempts, set(v.attempt_ids), v.acked)
                   for k, v in self.service.store._delivery.items()})
        first = self.service.device_inbox("bob", 100)
        second = self.service.device_inbox("bob", 100)
        self.assertEqual(json.dumps(first), json.dumps(second))
        after = (dict(self.service.store._message_sync_cursors),
                 {k: (v.attempts, set(v.attempt_ids), v.acked)
                  for k, v in self.service.store._delivery.items()})
        self.assertEqual(before, after)

    def test_device_unknown_or_revoked_is_409(self) -> None:
        error = self._error(lambda: self.service.device_inbox("ghost", 100))
        self.assertEqual((error.status_code, error.field),
                         (409, "device_id"))
        self.service.store.revoke_device("bob")
        error = self._error(lambda: self.service.device_inbox("bob", 100))
        self.assertEqual((error.status_code, error.field),
                         (409, "device_id"))

    def test_limit_validation(self) -> None:
        for bad in (0, 101, -1, "100", 1.5, True, None):
            error = self._error(
                lambda: self.service.device_inbox("bob", bad))
            self.assertEqual((error.status_code, error.field),
                             (400, "limit"), bad)


class InboxPersistenceTest(InboxMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self._build()
        self.state_store = attach_persistence(
            self.service, os.path.join(self.directory, "state.json"))

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_inbox_consumes_no_generation(self) -> None:
        generation = self.state_store.commit_seq
        self.service.device_inbox("bob", 100)
        self.assertEqual(self.state_store.commit_seq, generation)

    def test_restart_rebuilds_inbox_from_messages_and_delivery(self) -> None:
        self.service.sync_device_ack_batch("bob", {
            "items": [{"session_id": self.sid1, "cursor": 2}]})
        expected = self.service.device_inbox("bob", 100)
        rebuilt = DeviceService()
        attach_persistence(rebuilt,
                           os.path.join(self.directory, "state.json"))
        self.assertEqual(rebuilt.device_inbox("bob", 100), expected)
        self.assertEqual([m["message_id"] for m in expected["messages"]],
                         ["a3", "b1", "b2"])


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

    def _get(self, path: str):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("GET", path)
        response = conn.getresponse()
        raw = response.read().decode("utf-8")
        return response.status, json.loads(raw), raw

    def test_default_limit_and_key_order(self) -> None:
        status, body, _ = self._get("/v1/devices/bob/inbox")
        self.assertEqual(status, 200)
        self.assertEqual(list(body), ["device_id", "messages", "has_more"])
        self.assertEqual(len(body["messages"]), 5)
        self.assertEqual(list(body["messages"][0]),
                         ["session_id", "sender_device_id", "message_id",
                          "sequence", "nonce", "ciphertext", "created_at"])

    def test_limit_applies(self) -> None:
        status, body, _ = self._get("/v1/devices/bob/inbox?limit=3")
        self.assertEqual(status, 200)
        self.assertEqual([m["message_id"] for m in body["messages"]],
                         ["a1", "a2", "a3"])
        self.assertTrue(body["has_more"])

    def test_limit_rejects_malformed_and_repeated(self) -> None:
        for query in ("limit=0", "limit=101", "limit=-1", "limit=+1",
                      "limit=1.5", "limit=abc", "limit=", "limit=%20",
                      "limit=1&limit=2", "limit=1&limit=1"):
            status, body, _ = self._get(f"/v1/devices/bob/inbox?{query}")
            self.assertEqual(status, 400, query)
            self.assertEqual(list(body), ["message", "field"], query)
            self.assertEqual(body["field"], "limit", query)

    def test_unknown_and_revoked_device_are_409(self) -> None:
        status, body, _ = self._get("/v1/devices/ghost/inbox")
        self.assertEqual(status, 409)
        self.assertEqual(list(body), ["message", "field"])
        self.assertEqual(body["field"], "device_id")
        self.service.store.revoke_device("bob")
        status, body, _ = self._get("/v1/devices/bob/inbox")
        self.assertEqual((status, body["field"]), (409, "device_id"))

    def test_response_is_byte_identical_when_state_is_unchanged(self) -> None:
        status1, _, raw1 = self._get("/v1/devices/bob/inbox?limit=4")
        status2, _, raw2 = self._get("/v1/devices/bob/inbox?limit=4")
        self.assertEqual(status1, status2)
        self.assertEqual(raw1, raw2)

    def test_acked_items_vanish_over_http(self) -> None:
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("POST", f"/v1/devices/bob/sync/ack-batch",
                     json.dumps({"items": [{"session_id": self.sid1,
                                            "cursor": 3}]}),
                     {"Content-Type": "application/json"})
        conn.getresponse().read()
        status, body, _ = self._get("/v1/devices/bob/inbox")
        self.assertEqual(status, 200)
        self.assertEqual([m["message_id"] for m in body["messages"]],
                         ["b1", "b2"])


if __name__ == "__main__":
    unittest.main()
