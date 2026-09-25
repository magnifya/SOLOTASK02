"""Tests for the device-scoped multi-session sync-ack batch endpoint.

POST /v1/devices/{device_id}/sync/ack-batch commits delivery
acknowledgements and unified sync-cursor advances for several 1:1 sessions
in one locked transaction (one persistence notification).
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


class AckBatchMixin:
    def _build(self) -> None:
        self.service = DeviceService()
        self.service.store.add_device(Device("u", "alice", "ik"))
        # bob holds two pre-keys so two distinct 1:1 sessions addressed to
        # bob can be created.
        self.service.store.add_device(Device(
            "u", "bob", "ik",
            prekeys=[SignedPreKey("pk1", "pubk1"),
                     SignedPreKey("pk2", "pubk2")]))
        self.service.store.add_device(Device("u", "carol", "ik"))
        self.sid1 = self.service.store.create_session(
            "alice", "bob", "pk1", "ek1").session_id
        self.sid2 = self.service.store.create_session(
            "alice", "bob", "pk2", "ek2").session_id
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


class AckBatchServiceTest(AckBatchMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def _error(self, callable_):
        with self.assertRaises(ServiceError) as caught:
            callable_()
        return caught.exception

    def test_forward_batch_acks_ranges_and_shares_timestamp(self) -> None:
        body, status = self.service.sync_device_ack_batch("bob", {
            "items": [{"session_id": self.sid1, "cursor": 2},
                      {"session_id": self.sid2, "cursor": 1}]})
        self.assertEqual(status, 201)
        self.assertEqual(list(body), ["device_id", "results"])
        self.assertEqual(body["device_id"], "bob")
        results = body["results"]
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertEqual(list(result),
                             ["session_id", "cursor", "updated_at"])
        self.assertEqual(results[0]["session_id"], self.sid1)
        self.assertEqual(results[0]["cursor"], 2)
        self.assertEqual(results[1]["session_id"], self.sid2)
        self.assertEqual(results[1]["cursor"], 1)
        # Every advancing item shares one timestamp.
        self.assertEqual(results[0]["updated_at"], results[1]["updated_at"])
        self.assertIn("+00:00", results[0]["updated_at"])
        # Acks applied per range.
        delivery = self.service.store._delivery
        self.assertTrue(delivery[(self.sid1, "a1")].acked)
        self.assertEqual(delivery[(self.sid1, "a1")].ack_sequence, 1)
        self.assertTrue(delivery[(self.sid1, "a2")].acked)
        self.assertNotIn((self.sid1, "a3"), delivery)
        self.assertTrue(delivery[(self.sid2, "b1")].acked)
        self.assertNotIn((self.sid2, "b2"), delivery)
        cursors = self.service.store._message_sync_cursors
        self.assertEqual(cursors[(self.sid1, "bob")].cursor, 2)
        self.assertEqual(cursors[(self.sid2, "bob")].cursor, 1)

    def test_results_follow_input_order(self) -> None:
        body, _ = self.service.sync_device_ack_batch("bob", {
            "items": [{"session_id": self.sid2, "cursor": 1},
                      {"session_id": self.sid1, "cursor": 3}]})
        self.assertEqual([r["session_id"] for r in body["results"]],
                         [self.sid2, self.sid1])

    def test_all_equal_is_200_and_writes_nothing(self) -> None:
        self.service.sync_device_ack_batch("bob", {
            "items": [{"session_id": self.sid1, "cursor": 2}]})
        body, status = self.service.sync_device_ack_batch("bob", {
            "items": [{"session_id": self.sid1, "cursor": 2},
                      {"session_id": self.sid2, "cursor": 0}]})
        self.assertEqual(status, 200)
        self.assertEqual(body["results"][0]["cursor"], 2)
        # sid2 never advanced: cursor 0 reports that session's created_at.
        session2 = self.service.get_session(self.sid2)
        self.assertEqual(body["results"][1]["cursor"], 0)
        self.assertEqual(body["results"][1]["updated_at"],
                         session2["created_at"])
        self.assertNotIn((self.sid2, "bob"),
                         self.service.store._message_sync_cursors)

    def test_any_advance_makes_whole_batch_201(self) -> None:
        self.service.sync_device_ack_batch("bob", {
            "items": [{"session_id": self.sid1, "cursor": 2}]})
        _, status = self.service.sync_device_ack_batch("bob", {
            "items": [{"session_id": self.sid1, "cursor": 2},
                      {"session_id": self.sid2, "cursor": 1}]})
        self.assertEqual(status, 201)

    def test_retry_attempts_and_dedup_unchanged(self) -> None:
        self.service.retry_message(
            self.sid1, "a1", {"device_id": "bob", "attempt_id": "att-1"})
        self.service.sync_device_ack_batch("bob", {
            "items": [{"session_id": self.sid1, "cursor": 1}]})
        state = self.service.store._delivery[(self.sid1, "a1")]
        self.assertTrue(state.acked)
        self.assertEqual(state.attempts, 1)
        self.assertEqual(state.attempt_ids, {"att-1"})

    # -- validation --------------------------------------------------------

    def test_body_validation(self) -> None:
        cases = [
            (None, "request_body"),
            ([], "request_body"),
            ("x", "request_body"),
            ({}, "items"),
            ({"items": []}, "items"),
            ({"items": "x"}, "items"),
            ({"items": {}}, "items"),
            ({"items": ["x"]}, "items[0]"),
            ({"items": [[]]}, "items[0]"),
            ({"items": [{"cursor": 1}]}, "items[0].session_id"),
            ({"items": [{"session_id": "", "cursor": 1}]},
             "items[0].session_id"),
            ({"items": [{"session_id": 5, "cursor": 1}]},
             "items[0].session_id"),
            ({"items": [{"session_id": True, "cursor": 1}]},
             "items[0].session_id"),
            ({"items": [{"session_id": self.sid1}]}, "items[0].cursor"),
            ({"items": [{"session_id": self.sid1, "cursor": "1"}]},
             "items[0].cursor"),
            ({"items": [{"session_id": self.sid1, "cursor": True}]},
             "items[0].cursor"),
            ({"items": [{"session_id": self.sid1, "cursor": 1.5}]},
             "items[0].cursor"),
            ({"items": [{"session_id": self.sid1, "cursor": -1}]},
             "items[0].cursor"),
        ]
        for payload, field in cases:
            error = self._error(lambda: self.service.sync_device_ack_batch(
                "bob", payload))
            self.assertEqual((error.status_code, error.field), (400, field),
                             payload)

    def test_duplicate_session_is_400_on_the_item(self) -> None:
        error = self._error(lambda: self.service.sync_device_ack_batch("bob", {
            "items": [{"session_id": self.sid1, "cursor": 1},
                      {"session_id": self.sid1, "cursor": 2}]}))
        self.assertEqual((error.status_code, error.field),
                         (400, "items[1]"))
        # Nothing written.
        self.assertEqual(self.service.store._delivery, {})
        self.assertEqual(self.service.store._message_sync_cursors, {})

    def test_device_unknown_or_revoked(self) -> None:
        error = self._error(lambda: self.service.sync_device_ack_batch(
            "ghost", {"items": [{"session_id": self.sid1, "cursor": 1}]}))
        self.assertEqual((error.status_code, error.field),
                         (409, "device_id"))
        self.service.store.revoke_device("bob")
        error = self._error(lambda: self.service.sync_device_ack_batch(
            "bob", {"items": [{"session_id": self.sid1, "cursor": 1}]}))
        self.assertEqual((error.status_code, error.field),
                         (409, "device_id"))

    def test_session_unknown_is_404_on_item(self) -> None:
        error = self._error(lambda: self.service.sync_device_ack_batch("bob", {
            "items": [{"session_id": self.sid1, "cursor": 1},
                      {"session_id": "missing", "cursor": 0}]}))
        self.assertEqual((error.status_code, error.field),
                         (404, "items[1].session_id"))
        # First, valid item was not written either.
        self.assertEqual(self.service.store._delivery, {})

    def test_group_session_is_409_on_item(self) -> None:
        group_sid = self._group_session()
        error = self._error(lambda: self.service.sync_device_ack_batch("bob", {
            "items": [{"session_id": group_sid, "cursor": 1}]}))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[0].session_id"))

    def test_non_recipient_is_409_on_item(self) -> None:
        # alice is the initiator of sid1, not its recipient.
        error = self._error(lambda: self.service.sync_device_ack_batch(
            "alice", {"items": [{"session_id": self.sid1, "cursor": 1}]}))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[0].session_id"))

    def test_backward_and_over_max_are_409_on_cursor(self) -> None:
        self.service.sync_device_ack_batch("bob", {
            "items": [{"session_id": self.sid1, "cursor": 2}]})
        error = self._error(lambda: self.service.sync_device_ack_batch("bob", {
            "items": [{"session_id": self.sid1, "cursor": 1}]}))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[0].cursor"))
        error = self._error(lambda: self.service.sync_device_ack_batch("bob", {
            "items": [{"session_id": self.sid2, "cursor": 9}]}))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[0].cursor"))

    def test_first_error_in_array_order_writes_nothing(self) -> None:
        # Item 0 would succeed; item 1 fails (over max). Item 0 must not be
        # applied.
        error = self._error(lambda: self.service.sync_device_ack_batch("bob", {
            "items": [{"session_id": self.sid1, "cursor": 2},
                      {"session_id": self.sid2, "cursor": 9}]}))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[1].cursor"))
        self.assertEqual(self.service.store._delivery, {})
        self.assertEqual(self.service.store._message_sync_cursors, {})


class AckBatchPersistenceTest(AckBatchMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self._build()
        self.state_store = attach_persistence(
            self.service, os.path.join(self.directory, "state.json"))

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_whole_batch_persists_with_one_generation(self) -> None:
        generation = self.state_store.commit_seq
        self.service.sync_device_ack_batch("bob", {
            "items": [{"session_id": self.sid1, "cursor": 2},
                      {"session_id": self.sid2, "cursor": 1}]})
        self.assertEqual(self.state_store.commit_seq, generation + 1)

    def test_all_equal_consumes_no_generation(self) -> None:
        generation = self.state_store.commit_seq
        _, status = self.service.sync_device_ack_batch("bob", {
            "items": [{"session_id": self.sid1, "cursor": 0},
                      {"session_id": self.sid2, "cursor": 0}]})
        self.assertEqual(status, 200)
        self.assertEqual(self.state_store.commit_seq, generation)

    def test_failed_write_rolls_back_the_whole_batch(self) -> None:
        def raise_oserror(state):
            raise OSError("simulated disk failure")

        self.state_store.save = raise_oserror
        with self.assertRaises(PersistenceUnavailable):
            self.service.sync_device_ack_batch("bob", {
                "items": [{"session_id": self.sid1, "cursor": 2},
                          {"session_id": self.sid2, "cursor": 1}]})
        self.assertEqual(self.service.store._delivery, {})
        self.assertEqual(self.service.store._message_sync_cursors, {})


class AckBatchHTTPTest(AckBatchMixin, unittest.TestCase):
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
        path = "/v1/devices/bob/sync/ack-batch"
        status, body = self._request("POST", path, {
            "items": [{"session_id": self.sid1, "cursor": 2},
                      {"session_id": self.sid2, "cursor": 1}]})
        self.assertEqual(status, 201)
        self.assertEqual(list(body), ["device_id", "results"])
        self.assertEqual([r["session_id"] for r in body["results"]],
                         [self.sid1, self.sid2])
        status, body = self._request("POST", path, {
            "items": [{"session_id": self.sid1, "cursor": 2},
                      {"session_id": self.sid2, "cursor": 1}]})
        self.assertEqual(status, 200)

    def test_error_statuses_and_fields(self) -> None:
        path = "/v1/devices/bob/sync/ack-batch"
        status, body = self._request("POST", path, raw="{bad")
        self.assertEqual((status, body["field"]), (400, "request_body"))
        status, body = self._request("POST", path, {"items": []})
        self.assertEqual((status, body["field"]), (400, "items"))
        status, body = self._request("POST", path, {"items": [5]})
        self.assertEqual((status, body["field"]), (400, "items[0]"))
        status, body = self._request(
            "POST", path, {"items": [{"session_id": "x", "cursor": 9}]})
        self.assertEqual((status, body["field"]),
                         (404, "items[0].session_id"))
        status, body = self._request(
            "POST", path, {"items": [{"session_id": self.sid1, "cursor": 9}]})
        self.assertEqual((status, body["field"]),
                         (409, "items[0].cursor"))
        status, body = self._request(
            "POST", "/v1/devices/ghost/sync/ack-batch",
            {"items": [{"session_id": self.sid1, "cursor": 1}]})
        self.assertEqual((status, body["field"]), (409, "device_id"))

    def test_unknown_device_path_is_409_not_404(self) -> None:
        status, body = self._request(
            "POST", "/v1/devices/ghost/sync/ack-batch", raw='{"items":[]}')
        # Body validation (empty items) still runs and reports 400/items;
        # a well-formed body for an unknown device is 409/device_id.
        self.assertEqual((status, body["field"]), (400, "items"))
        status, body = self._request(
            "POST", "/v1/devices/ghost/sync/ack-batch",
            {"items": [{"session_id": self.sid1, "cursor": 1}]})
        self.assertEqual((status, body["field"]), (409, "device_id"))


if __name__ == "__main__":
    unittest.main()
