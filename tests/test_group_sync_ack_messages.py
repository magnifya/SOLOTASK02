"""Tests for the device-scoped selective group-message ack endpoint.

POST /v1/devices/{device_id}/group-sync/ack-messages acknowledges
scattered messages across several group sessions in one locked
transaction (one persistence notification), touching neither sync
cursors nor retry attempt state.
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


class AckMessagesMixin:
    def _build(self) -> None:
        self.service = DeviceService()
        self.service.store.add_device(Device("u", "alice", "ik"))
        self.service.store.add_device(
            Device("u", "bob", "ik",
                   prekeys=[SignedPreKey("pk1", "pubk1")]))
        self.service.store.add_device(Device("u", "carol", "ik"))
        self.service.create_group({
            "group_id": "g1", "creator_device_id": "alice",
            "member_device_ids": ["bob"]})
        self.gs1 = self.service.create_group_session({
            "group_id": "g1", "initiator_device_id": "alice",
            "ephemeral_key": "epk"})["session_id"]
        self.service.create_group({
            "group_id": "g2", "creator_device_id": "alice",
            "member_device_ids": ["bob", "carol"]})
        self.gs2 = self.service.create_group_session({
            "group_id": "g2", "initiator_device_id": "alice",
            "ephemeral_key": "epk"})["session_id"]
        for sequence in range(1, 4):
            self.service.post_message({
                "session_id": self.gs1, "sender_device_id": "alice",
                "message_id": f"a{sequence}", "sequence": sequence,
                "nonce": f"na{sequence}", "ciphertext": "ct"})
        self.service.post_message({
            "session_id": self.gs2, "sender_device_id": "alice",
            "message_id": "b1", "sequence": 1, "nonce": "nb1",
            "ciphertext": "ct"})
        self.service.post_message({
            "session_id": self.gs2, "sender_device_id": "carol",
            "message_id": "b2", "sequence": 2, "nonce": "nb2",
            "ciphertext": "ct"})
        # A 1:1 session for the non-group-session failure case.
        self.sid1 = self.service.store.create_session(
            "alice", "bob", "pk1", "ek1").session_id

    def _ack(self, device_id, items):
        return self.service.group_sync_ack_messages(
            device_id, {"items": items})


class AckMessagesServiceTest(AckMessagesMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def _error(self, callable_):
        with self.assertRaises(ServiceError) as caught:
            callable_()
        return caught.exception

    def test_first_ack_201_and_replay_200_same_results(self) -> None:
        items = [{"session_id": self.gs1, "message_id": "a1",
                  "sequence": 1},
                 {"session_id": self.gs1, "message_id": "a3",
                  "sequence": 3},
                 {"session_id": self.gs2, "message_id": "b2",
                  "sequence": 2}]
        body, status = self._ack("bob", items)
        self.assertEqual(status, 201)
        self.assertEqual(list(body), ["device_id", "results"])
        self.assertEqual(body["device_id"], "bob")
        self.assertEqual(body["results"], [
            {"session_id": self.gs1, "message_id": "a1", "acked": True},
            {"session_id": self.gs1, "message_id": "a3", "acked": True},
            {"session_id": self.gs2, "message_id": "b2", "acked": True}])
        for result in body["results"]:
            self.assertEqual(list(result),
                             ["session_id", "message_id", "acked"])
        replay, replay_status = self._ack("bob", items)
        self.assertEqual(replay_status, 200)
        self.assertEqual(replay, body)

    def test_acked_messages_leave_the_group_inbox(self) -> None:
        inbox = self.service.store.device_group_inbox("bob", 100)
        self.assertEqual([m["message_id"] for m in inbox["messages"]],
                         ["a1", "a2", "a3", "b1", "b2"])
        self._ack("bob", [
            {"session_id": self.gs1, "message_id": "a2", "sequence": 2},
            {"session_id": self.gs2, "message_id": "b1", "sequence": 1}])
        inbox = self.service.store.device_group_inbox("bob", 100)
        self.assertEqual([m["message_id"] for m in inbox["messages"]],
                         ["a1", "a3", "b2"])

    def test_cursors_and_attempts_unchanged(self) -> None:
        # Seed an attempt and a cursor that the ack must leave alone.
        self.service.retry_message(
            self.gs1, "a1", {"device_id": "bob", "attempt_id": "att1"})
        self.service.sync_group_checkpoint(
            self.gs1, {"device_id": "bob", "cursor": 1})
        self._ack("bob", [
            {"session_id": self.gs1, "message_id": "a2", "sequence": 2}])
        a2 = self.service.store.message_delivery_status(
            self.gs1, "a2", "bob")
        self.assertEqual(a2["status"], "acked")
        a1 = self.service.store.message_delivery_status(
            self.gs1, "a1", "bob")
        self.assertEqual(a1["attempts"], 1)
        # The selective ack neither advances nor rewinds the stored cursor.
        record = self.service.store._group_sync_cursors[(self.gs1, "bob")]
        self.assertEqual(record.cursor, 1)

    def test_shape_errors(self) -> None:
        def call(payload):
            return lambda: self.service.group_sync_ack_messages(
                "bob", payload)

        cases = [
            ([], "request_body"),
            ("x", "request_body"),
            ({"x": 1, "items": []}, "x"),
            ({}, "items"),
            ({"items": 1}, "items"),
            ({"items": []}, "items"),
            ({"items": [4]}, "items[0]"),
            ({"items": [{"session_id": self.gs1, "message_id": "a1",
                         "sequence": 1, "z": 2}]}, "items[0]"),
            ({"items": [{"session_id": "", "message_id": "a1",
                         "sequence": 1}]}, "items[0].session_id"),
            ({"items": [{"session_id": 7, "message_id": "a1",
                         "sequence": 1}]}, "items[0].session_id"),
            ({"items": [{"session_id": self.gs1, "message_id": "",
                         "sequence": 1}]}, "items[0].message_id"),
            ({"items": [{"session_id": self.gs1, "message_id": None,
                         "sequence": 1}]}, "items[0].message_id"),
            ({"items": [{"session_id": self.gs1,
                         "message_id": "a1"}]}, "items[0].sequence"),
            ({"items": [{"session_id": self.gs1, "message_id": "a1",
                         "sequence": True}]}, "items[0].sequence"),
            ({"items": [{"session_id": self.gs1, "message_id": "a1",
                         "sequence": -2}]}, "items[0].sequence"),
            ({"items": [{"session_id": self.gs1, "message_id": "a1",
                         "sequence": 1.5}]}, "items[0].sequence"),
            ({"items": [{"session_id": self.gs1, "message_id": "a1",
                         "sequence": 1},
                        {"session_id": self.gs1, "message_id": "a1",
                         "sequence": 1}]}, "items[1]"),
        ]
        for payload, field in cases:
            error = self._error(call(payload))
            self.assertEqual((error.status_code, error.field),
                             (400, field), payload)

    def test_device_errors(self) -> None:
        error = self._error(lambda: self._ack("nope", [
            {"session_id": self.gs1, "message_id": "a1",
             "sequence": 1}]))
        self.assertEqual((error.status_code, error.field),
                         (409, "device_id"))
        self.service.revoke_device("carol")
        error = self._error(lambda: self._ack("carol", [
            {"session_id": self.gs2, "message_id": "b1",
             "sequence": 1}]))
        self.assertEqual((error.status_code, error.field),
                         (409, "device_id"))

    def test_item_errors_in_input_order(self) -> None:
        def expect(items, status_code, field):
            error = self._error(lambda: self._ack("bob", items))
            self.assertEqual((error.status_code, error.field),
                             (status_code, field), items)

        good = {"session_id": self.gs1, "message_id": "a1",
                "sequence": 1}
        expect([{"session_id": "nope", "message_id": "a1",
                 "sequence": 1}], 404, "items[0].session_id")
        expect([{"session_id": self.sid1, "message_id": "a1",
                 "sequence": 1}], 409, "items[0].session_id")
        # carol is frozen into gs2 but not gs1.
        error = self._error(lambda: self._ack("carol", [good]))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[0].session_id"))
        expect([{"session_id": self.gs1, "message_id": "zzz",
                 "sequence": 1}], 404, "items[0].message_id")
        expect([{"session_id": self.gs1, "message_id": "a1",
                 "sequence": 9}], 409, "items[0].sequence")
        error = self._error(lambda: self._ack("alice", [
            {"session_id": self.gs1, "message_id": "a1",
             "sequence": 1}]))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[0].message_id"))
        # First error wins and nothing is written: a1 stays pending.
        expect([good,
                {"session_id": self.gs1, "message_id": "zzz",
                 "sequence": 2}], 404, "items[1].message_id")
        status = self.service.store.message_delivery_status(
            self.gs1, "a1", "bob")
        self.assertEqual(status["status"], "pending")

    def test_removed_member_acks_but_added_member_cannot(self) -> None:
        self.service.remove_group_member(
            "g1", {"actor_device_id": "alice", "device_id": "bob"})
        self.service.add_group_member(
            "g1", {"actor_device_id": "alice", "device_id": "carol"})
        _, status = self._ack("bob", [
            {"session_id": self.gs1, "message_id": "a1",
             "sequence": 1}])
        self.assertEqual(status, 201)
        error = self._error(lambda: self._ack("carol", [
            {"session_id": self.gs1, "message_id": "a2",
             "sequence": 2}]))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[0].session_id"))


class AckMessagesPersistenceTest(AckMessagesMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self._build()
        self.state_store = attach_persistence(
            self.service, os.path.join(self.directory, "state.json"))

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_batch_persists_with_one_generation_and_restarts(self) -> None:
        generation = self.state_store.commit_seq
        self._ack("bob", [
            {"session_id": self.gs1, "message_id": "a1", "sequence": 1},
            {"session_id": self.gs2, "message_id": "b2", "sequence": 2}])
        self.assertEqual(self.state_store.commit_seq, generation + 1)
        restarted = DeviceService()
        attach_persistence(
            restarted, os.path.join(self.directory, "state.json"))
        inbox = restarted.store.device_group_inbox("bob", 100)
        ids = [m["message_id"] for m in inbox["messages"]]
        self.assertNotIn("a1", ids)
        self.assertNotIn("b2", ids)
        _, status = restarted.group_sync_ack_messages("bob", {"items": [
            {"session_id": self.gs1, "message_id": "a1",
             "sequence": 1}]})
        self.assertEqual(status, 200)

    def test_replays_consume_no_generation(self) -> None:
        items = [{"session_id": self.gs1, "message_id": "a1",
                  "sequence": 1}]
        self._ack("bob", items)
        generation = self.state_store.commit_seq
        _, status = self._ack("bob", items)
        self.assertEqual(status, 200)
        self.assertEqual(self.state_store.commit_seq, generation)

    def test_failed_write_rolls_back_the_whole_batch(self) -> None:
        def raise_oserror(state):
            raise OSError("simulated disk failure")

        self.state_store.save = raise_oserror
        with self.assertRaises(PersistenceUnavailable):
            self._ack("bob", [
                {"session_id": self.gs1, "message_id": "a1",
                 "sequence": 1},
                {"session_id": self.gs2, "message_id": "b2",
                 "sequence": 2}])
        acked = [state for key, state
                 in self.service.store._group_delivery.items()
                 if key[2] == "bob" and state.acked]
        self.assertEqual(acked, [])
        # Nothing durable happened: the retry is a first ack.
        del self.state_store.save
        _, status = self._ack("bob", [
            {"session_id": self.gs1, "message_id": "a1",
             "sequence": 1}])
        self.assertEqual(status, 201)


class AckMessagesHTTPTest(AckMessagesMixin, unittest.TestCase):
    PATH = "/v1/devices/bob/group-sync/ack-messages"

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

    def _request(self, path, body=None, raw=None):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        if raw is not None:
            data = raw
        else:
            data = json.dumps(body).encode("utf-8") if body is not None \
                else b""
        conn.request("POST", path, data,
                     {"Content-Type": "application/json"})
        response = conn.getresponse()
        return response.status, json.loads(response.read().decode("utf-8"))

    def test_success_and_replay(self) -> None:
        status, body = self._request(self.PATH, {"items": [
            {"session_id": self.gs1, "message_id": "a1", "sequence": 1},
            {"session_id": self.gs2, "message_id": "b2", "sequence": 2}]})
        self.assertEqual(status, 201)
        self.assertEqual(body["results"][0]["acked"], True)
        status, body = self._request(self.PATH, {"items": [
            {"session_id": self.gs1, "message_id": "a1", "sequence": 1}]})
        self.assertEqual(status, 200)

    def test_query_body_and_item_errors(self) -> None:
        status, body = self._request(self.PATH + "?x=1", {"items": []})
        self.assertEqual((status, body["field"]), (400, "query"))
        status, body = self._request(self.PATH, raw="{bad")
        self.assertEqual((status, body["field"]), (400, "request_body"))
        status, body = self._request(self.PATH, {"items": []})
        self.assertEqual((status, body["field"]), (400, "items"))
        status, body = self._request(self.PATH, {"items": [9]})
        self.assertEqual((status, body["field"]), (400, "items[0]"))
        status, body = self._request(self.PATH, {"items": [
            {"session_id": "nope", "message_id": "a1", "sequence": 1}]})
        self.assertEqual((status, body["field"]),
                         (404, "items[0].session_id"))
        status, body = self._request(self.PATH, {"items": [
            {"session_id": self.gs1, "message_id": "a1", "sequence": 1},
            {"session_id": self.gs1, "message_id": "zzz",
             "sequence": 2}]})
        self.assertEqual((status, body["field"]),
                         (404, "items[1].message_id"))

    def test_unknown_device_is_409(self) -> None:
        status, body = self._request(
            "/v1/devices/nope/group-sync/ack-messages", {"items": [
                {"session_id": self.gs1, "message_id": "a1",
                 "sequence": 1}]})
        self.assertEqual((status, body["field"]), (409, "device_id"))


if __name__ == "__main__":
    unittest.main()
