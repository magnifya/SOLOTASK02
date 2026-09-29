"""Tests for the selective per-message group ack batch.

``POST /v1/devices/{device_id}/group-sync/ack-messages`` lets one device
acknowledge scattered messages of several group sessions in one call,
without moving group sync cursors, paging, attempts or attempt-dedup sets.
"""
from __future__ import annotations

import base64
import json
import os
import tempfile
import threading
import unittest
from http.client import HTTPConnection

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import x25519

from e2ee_backend.http_app import create_server
from e2ee_backend.persistence import (
    PersistenceUnavailable,
    attach_persistence,
)
from e2ee_backend.service import DeviceService, ServiceError


def _raw_key_b64() -> str:
    key = x25519.X25519PrivateKey.generate().public_key()
    raw = key.public_bytes(serialization.Encoding.Raw,
                           serialization.PublicFormat.Raw)
    return base64.b64encode(raw).decode()


def _register_payload(device_id: str, user_id: str = "u1") -> dict:
    return {
        "user_id": user_id,
        "device_id": device_id,
        "identity_key": _raw_key_b64(),
        "signed_prekeys": [{"key_id": "k1", "public_key": _raw_key_b64()}],
    }


def _message_payload(session_id: str, message_id: str, sequence: int,
                     sender: str = "d1") -> dict:
    return {
        "session_id": session_id,
        "sender_device_id": sender,
        "message_id": message_id,
        "sequence": sequence,
        "nonce": base64.b64encode(f"nonce-{message_id}".encode()).decode(),
        "ciphertext": base64.b64encode(b"ciphertext-and-tag").decode(),
    }


def _item(session_id: str, message_id: str, sequence: int) -> dict:
    return {"session_id": session_id, "message_id": message_id,
            "sequence": sequence}


class TwoGroupFixture:
    """d1/d2/d3/d4 registered; two frozen group sessions.

    s1 (group g1, members d1/d2/d3) holds m1@1 and m2@2 from d1; s2
    (group g2, members d1/d2) holds n1@1 from d1. d4 is in neither group.
    """

    def __init__(self) -> None:
        self.service = DeviceService()
        for device_id in ("d1", "d2", "d3", "d4"):
            self.service.register(_register_payload(device_id))
        self.service.create_group({
            "group_id": "g1", "creator_device_id": "d1",
            "member_device_ids": ["d1", "d2", "d3"]})
        self.service.create_group({
            "group_id": "g2", "creator_device_id": "d1",
            "member_device_ids": ["d1", "d2"]})
        s1 = self.service.create_group_session({
            "group_id": "g1", "initiator_device_id": "d1",
            "ephemeral_key": _raw_key_b64()})
        s2 = self.service.create_group_session({
            "group_id": "g2", "initiator_device_id": "d1",
            "ephemeral_key": _raw_key_b64()})
        self.s1 = s1["session_id"]
        self.s2 = s2["session_id"]
        self.service.post_message(_message_payload(self.s1, "m1", 1))
        self.service.post_message(_message_payload(self.s1, "m2", 2))
        self.service.post_message(_message_payload(self.s2, "n1", 1))


class AckMessagesServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = TwoGroupFixture()
        self.service = self.fixture.service
        self.s1 = self.fixture.s1
        self.s2 = self.fixture.s2

    def _ack(self, device_id: str, *items):
        return self.service.sync_group_ack_messages(
            device_id, {"items": list(items)})

    def test_first_acks_across_sessions_are_201_in_input_order(self) -> None:
        body, status = self._ack(
            "d2", _item(self.s2, "n1", 1), _item(self.s1, "m1", 1))
        self.assertEqual(status, 201)
        self.assertEqual(list(body), ["device_id", "results"])
        self.assertEqual(body["device_id"], "d2")
        self.assertEqual(body["results"], [
            {"session_id": self.s2, "message_id": "n1", "acked": True},
            {"session_id": self.s1, "message_id": "m1", "acked": True},
        ])
        for result in body["results"]:
            self.assertEqual(list(result),
                             ["session_id", "message_id", "acked"])
        # The target messages leave the group inbox; m2 in s1 stays.
        inbox = self.service.device_group_inbox("d2", 100)["messages"]
        self.assertEqual([m["message_id"] for m in inbox], ["m2"])
        self.assertEqual(
            self.service.message_status(self.s1, "m1", "d2")["status"],
            "acked")
        self.assertEqual(
            self.service.message_status(self.s2, "n1", "d2")["status"],
            "acked")

    def test_all_already_acked_is_200_and_writes_nothing(self) -> None:
        self.assertEqual(self._ack("d2", _item(self.s1, "m1", 1))[1], 201)
        self.assertEqual(
            self._ack("d2", _item(self.s1, "m1", 1),
                      _item(self.s2, "n1", 1))[1], 201)
        writes = []
        self.service.store.on_change = lambda: writes.append(1)
        body, status = self._ack(
            "d2", _item(self.s2, "n1", 1), _item(self.s1, "m1", 1))
        self.assertEqual(status, 200)
        self.assertEqual(writes, [])
        # Repeating the whole batch gives identical results and no write.
        again, again_status = self._ack(
            "d2", _item(self.s2, "n1", 1), _item(self.s1, "m1", 1))
        self.assertEqual(again_status, 200)
        self.assertEqual(again, body)
        self.assertEqual(writes, [])

    def test_cursor_paging_attempts_and_attempt_ids_untouched(self) -> None:
        # Advance d2's group sync cursor over m1 first.
        page = self.service.sync_group_messages(self.s1, "d2", None, 1)
        self.assertEqual([m["message_id"] for m in page["messages"]], ["m1"])
        cursor_before = self.service.store._group_sync_cursors[
            (self.s1, "d2")].cursor
        # Record a retry attempt, then ack selectively out of cursor order.
        view, _ = self.service.retry_message(
            self.s1, "m2", {"device_id": "d2", "attempt_id": "a1"})
        self.assertEqual(view["attempts"], 1)
        self._ack("d2", _item(self.s1, "m2", 2))
        record = self.service.store._group_delivery[(self.s1, "m2", "d2")]
        self.assertTrue(record.acked)
        self.assertEqual(record.ack_sequence, 2)
        self.assertEqual(record.attempts, 1)
        self.assertEqual(record.attempt_ids, {"a1"})
        self.assertEqual(
            self.service.store._group_sync_cursors[(self.s1, "d2")].cursor,
            cursor_before)
        # No cursor record is created for a session that never synced.
        self.assertNotIn((self.s2, "d2"),
                         self.service.store._group_sync_cursors)

    def test_first_error_aborts_the_whole_batch(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self._ack("d2",
                      _item(self.s1, "m1", 1),
                      _item(self.s1, "ghost", 1))
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "items[1].message_id")
        # The valid first item was not written.
        self.assertEqual(
            self.service.message_status(self.s1, "m1", "d2")["status"],
            "pending")

    def test_session_and_member_validation(self) -> None:
        # A 1:1 session is not ack-able through this group endpoint.
        one_to_one = self.service.create_session({
            "initiator_device_id": "d3",
            "recipient_device_id": "d2",
            "prekey_id": "k1",
            "ephemeral_key": _raw_key_b64(),
        })["session_id"]
        self.service.post_message(
            _message_payload(one_to_one, "p1", 1, sender="d3"))
        with self.assertRaises(ServiceError) as ctx:
            self._ack("d2", _item(one_to_one, "p1", 1))
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "items[0].session_id")
        # Unknown session.
        with self.assertRaises(ServiceError) as ctx:
            self._ack("d2", _item("ghost", "m1", 1))
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "items[0].session_id")
        # A device never frozen in cannot ack; a removed member still can.
        self.service.add_group_member(
            "g1", {"actor_device_id": "d1", "device_id": "d4"})
        with self.assertRaises(ServiceError) as ctx:
            self._ack("d4", _item(self.s1, "m1", 1))
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "items[0].session_id")
        self.service.remove_group_member(
            "g1", {"actor_device_id": "d1", "device_id": "d3"})
        _, status = self._ack("d3", _item(self.s1, "m1", 1))
        self.assertEqual(status, 201)

    def test_message_sequence_and_self_sender_validation(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self._ack("d2", _item(self.s1, "ghost", 1))
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "items[0].message_id")
        with self.assertRaises(ServiceError) as ctx:
            self._ack("d2", _item(self.s1, "m1", 2))
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "items[0].sequence")
        with self.assertRaises(ServiceError) as ctx:
            self._ack("d1", _item(self.s1, "m1", 1))
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "items[0].message_id")

    def test_unknown_and_revoked_device_are_409_device_id(self) -> None:
        self.service.revoke_device("d4")
        for device_id in ("ghost", "d4"):
            with self.assertRaises(ServiceError) as ctx:
                self._ack(device_id, _item(self.s1, "m1", 1))
            self.assertEqual(ctx.exception.status_code, 409, device_id)
            self.assertEqual(ctx.exception.field, "device_id", device_id)

    def test_body_shape_errors(self) -> None:
        def expect(payload, field):
            with self.assertRaises(ServiceError) as ctx:
                self.service.sync_group_ack_messages("d2", payload)
            self.assertEqual(ctx.exception.status_code, 400)
            self.assertEqual(ctx.exception.field, field)
            self.assertEqual(set(ctx.exception.to_body()),
                             {"message", "field"})

        expect(["not", "an", "object"], "request_body")
        expect({"items": [], "extra": 1}, "extra")
        expect({}, "items")
        expect({"items": []}, "items")
        expect({"items": {}}, "items")
        expect({"items": ["x"]}, "items[0]")
        expect({"items": [{"session_id": self.s1, "message_id": "m1",
                           "sequence": 1, "extra": 2}]}, "items[0]")
        expect({"items": [{"message_id": "m1", "sequence": 1}]},
               "items[0].session_id")
        expect({"items": [{"session_id": "", "message_id": "m1",
                           "sequence": 1}]}, "items[0].session_id")
        expect({"items": [{"session_id": 3, "message_id": "m1",
                           "sequence": 1}]}, "items[0].session_id")
        expect({"items": [{"session_id": self.s1, "sequence": 1}]},
               "items[0].message_id")
        expect({"items": [{"session_id": self.s1, "message_id": "",
                           "sequence": 1}]}, "items[0].message_id")
        expect({"items": [{"session_id": self.s1, "message_id": 5,
                           "sequence": 1}]}, "items[0].message_id")
        expect({"items": [{"session_id": self.s1, "message_id": "m1"}]},
               "items[0].sequence")
        for bad_sequence in (True, False, -1, 1.5, "1", None):
            expect({"items": [{"session_id": self.s1, "message_id": "m1",
                               "sequence": bad_sequence}]},
                   "items[0].sequence")
        # Repeated session/message pair, even with a different sequence.
        expect({"items": [_item(self.s1, "m1", 1),
                          _item(self.s1, "m1", 2)]}, "items[1]")


class AckMessagesPersistenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.path = os.path.join(self.directory, "state.json")
        self.fixture = TwoGroupFixture()
        self.service = self.fixture.service
        self.s1 = self.fixture.s1
        self.s2 = self.fixture.s2

    def _document(self) -> dict:
        with open(self.path, encoding="utf-8") as handle:
            return json.load(handle)

    def test_acks_persist_in_existing_group_delivery_section(self) -> None:
        attach_persistence(self.service, self.path)
        self.service.sync_group_ack_messages("d2", {"items": [
            _item(self.s1, "m1", 1), _item(self.s2, "n1", 1)]})
        document = self._document()
        self.assertEqual(document["version"], 1)
        self.assertEqual(len(document["group_delivery"]), 2)
        self.assertTrue(all(row["acked"]
                            for row in document["group_delivery"]))
        top_level = set(document)

        restored = DeviceService()
        attach_persistence(restored, self.path)
        # Replaying the batch after restart is a 200 no-write.
        body, status = restored.sync_group_ack_messages("d2", {"items": [
            _item(self.s1, "m1", 1), _item(self.s2, "n1", 1)]})
        self.assertEqual(status, 200)
        self.assertTrue(all(row["acked"] for row in body["results"]))
        self.assertEqual(set(self._document()), top_level)

    def test_failed_persist_rolls_back_the_whole_batch(self) -> None:
        from e2ee_backend.persistence import JsonStateStore

        attach_persistence(self.service, self.path)
        real_save = JsonStateStore.save

        def failing_save(self, state):
            raise OSError("disk full")

        JsonStateStore.save = failing_save
        try:
            with self.assertRaises(PersistenceUnavailable):
                self.service.sync_group_ack_messages("d2", {"items": [
                    _item(self.s1, "m1", 1),
                    _item(self.s1, "m2", 2),
                    _item(self.s2, "n1", 1)]})
        finally:
            JsonStateStore.save = real_save

        # Nothing was acknowledged in memory after the rollback.
        for session_id, message_id in ((self.s1, "m1"), (self.s1, "m2"),
                                       (self.s2, "n1")):
            self.assertEqual(
                self.service.message_status(
                    session_id, message_id, "d2")["status"],
                "pending")
        # The first durable retry after recovery is still a first ack 201.
        _, status = self.service.sync_group_ack_messages("d2", {"items": [
            _item(self.s1, "m1", 1)]})
        self.assertEqual(status, 201)


class AckMessagesHTTPTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = TwoGroupFixture()
        self.server, _ = create_server("127.0.0.1", 0,
                                       service=self.fixture.service)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        self.s1 = self.fixture.s1
        self.s2 = self.fixture.s2

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _raw(self, path: str, raw: bytes):
        connection = HTTPConnection("127.0.0.1", self.port, timeout=5)
        connection.request("POST", path, body=raw,
                           headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        data = response.read().decode("utf-8")
        connection.close()
        return response.status, json.loads(data)

    def _request(self, path: str, body: object):
        return self._raw(path, json.dumps(body).encode("utf-8"))

    def test_endpoint_lifecycle_over_http(self) -> None:
        path = "/v1/devices/d2/group-sync/ack-messages"
        status, body = self._request(path, {"items": [
            _item(self.s1, "m2", 2), _item(self.s2, "n1", 1)]})
        self.assertEqual(status, 201)
        self.assertEqual(body["device_id"], "d2")
        self.assertEqual([r["message_id"] for r in body["results"]],
                         ["m2", "n1"])
        status, body = self._request(path, {"items": [
            _item(self.s2, "n1", 1), _item(self.s1, "m2", 2)]})
        self.assertEqual(status, 200)
        self.assertEqual([r["message_id"] for r in body["results"]],
                         ["n1", "m2"])

    def test_bad_json_and_query_params(self) -> None:
        path = "/v1/devices/d2/group-sync/ack-messages"
        status, body = self._raw(path, b"{not json")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "request_body")
        self.assertEqual(set(body), {"message", "field"})
        for suffix in ("?foo=bar", "?foo", "?a=1&a=2"):
            status, body = self._request(path + suffix,
                                         {"items": [_item(self.s1, "m1", 1)]})
            self.assertEqual(status, 400, suffix)
            self.assertEqual(body["field"], "query", suffix)

    def test_device_and_item_errors_over_http(self) -> None:
        status, body = self._request(
            "/v1/devices/ghost/group-sync/ack-messages",
            {"items": [_item(self.s1, "m1", 1)]})
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "device_id")
        status, body = self._request(
            "/v1/devices/d2/group-sync/ack-messages",
            {"items": [_item(self.s1, "m1", 9)]})
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "items[0].sequence")


if __name__ == "__main__":
    unittest.main()
