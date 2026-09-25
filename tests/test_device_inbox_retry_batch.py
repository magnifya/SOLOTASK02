"""Tests for the device-scoped 1:1 inbox retry batch endpoint.

POST /v1/devices/{device_id}/inbox/retry-batch records delivery attempts for
several 1:1 messages in one locked transaction (one persistence notification).
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


class RetryBatchMixin:
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


class RetryBatchServiceTest(RetryBatchMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def _error(self, callable_):
        with self.assertRaises(ServiceError) as caught:
            callable_()
        return caught.exception

    def _batch(self, *pairs, attempt_id="batch-1", device="bob"):
        return self.service.device_inbox_retry_batch(device, {
            "attempt_id": attempt_id,
            "items": [{"session_id": sid, "message_id": mid}
                      for sid, mid in pairs]})

    def test_new_attempts_are_201_and_counted(self) -> None:
        body, status = self._batch((self.sid1, "a1"), (self.sid2, "b1"))
        self.assertEqual(status, 201)
        self.assertEqual(list(body), ["device_id", "results"])
        self.assertEqual(body["device_id"], "bob")
        results = body["results"]
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertEqual(list(result),
                             ["session_id", "message_id", "attempts"])
        self.assertEqual(results[0], {"session_id": self.sid1,
                                      "message_id": "a1", "attempts": 1})
        self.assertEqual(results[1], {"session_id": self.sid2,
                                      "message_id": "b1", "attempts": 1})
        delivery = self.service.store._delivery
        self.assertEqual(delivery[(self.sid1, "a1")].attempt_ids,
                         {"batch-1"})
        self.assertEqual(delivery[(self.sid2, "b1")].attempt_ids,
                         {"batch-1"})

    def test_results_follow_input_order(self) -> None:
        body, _ = self._batch((self.sid2, "b2"), (self.sid1, "a3"),
                              (self.sid2, "b1"))
        self.assertEqual([(r["session_id"], r["message_id"])
                          for r in body["results"]],
                         [(self.sid2, "b2"), (self.sid1, "a3"),
                          (self.sid2, "b1")])

    def test_same_attempt_id_replay_is_200_and_not_counted(self) -> None:
        body, status = self._batch((self.sid1, "a1"), (self.sid2, "b1"))
        self.assertEqual(status, 201)
        body, status = self._batch((self.sid1, "a1"), (self.sid2, "b1"))
        self.assertEqual(status, 200)
        self.assertEqual([r["attempts"] for r in body["results"]], [1, 1])

    def test_partial_replay_is_still_201(self) -> None:
        self._batch((self.sid1, "a1"), (self.sid2, "b1"))
        # a1 replays batch-1, a2 is new (same batch attempt id joins both).
        _, status = self._batch((self.sid1, "a1"), (self.sid2, "b2"),
                                attempt_id="batch-1")
        self.assertEqual(status, 201)
        delivery = self.service.store._delivery
        self.assertEqual(delivery[(self.sid1, "a1")].attempts, 1)
        self.assertEqual(delivery[(self.sid2, "b2")].attempts, 1)

    def test_distinct_attempt_ids_increment(self) -> None:
        self._batch((self.sid1, "a1"), attempt_id="x")
        body, status = self._batch((self.sid1, "a1"), attempt_id="y")
        self.assertEqual(status, 201)
        self.assertEqual(body["results"][0]["attempts"], 2)

    def test_repeated_pair_within_one_batch_is_400_on_item(self) -> None:
        error = self._error(lambda: self._batch(
            (self.sid1, "a1"), (self.sid2, "b1"), (self.sid1, "a1")))
        self.assertEqual((error.status_code, error.field),
                         (400, "items[2]"))
        self.assertEqual(self.service.store._delivery, {})

    def test_same_session_distinct_messages_allowed(self) -> None:
        body, status = self._batch((self.sid1, "a1"), (self.sid1, "a2"))
        self.assertEqual(status, 201)
        self.assertEqual(len(body["results"]), 2)

    # -- validation --------------------------------------------------------

    def test_body_validation(self) -> None:
        cases = [
            (None, "request_body"),
            ([], "request_body"),
            ("x", "request_body"),
            ({}, "attempt_id"),
            ({"attempt_id": ""}, "attempt_id"),
            ({"attempt_id": 5}, "attempt_id"),
            ({"attempt_id": True}, "attempt_id"),
            ({"attempt_id": "a"}, "items"),
            ({"attempt_id": "a", "items": []}, "items"),
            ({"attempt_id": "a", "items": "x"}, "items"),
            ({"attempt_id": "a", "items": {}}, "items"),
            ({"attempt_id": "a", "items": ["x"]}, "items[0]"),
            ({"attempt_id": "a", "items": [[]]}, "items[0]"),
            ({"attempt_id": "a",
              "items": [{"message_id": "a1"}]}, "items[0].session_id"),
            ({"attempt_id": "a",
              "items": [{"session_id": "", "message_id": "a1"}]},
             "items[0].session_id"),
            ({"attempt_id": "a",
              "items": [{"session_id": 5, "message_id": "a1"}]},
             "items[0].session_id"),
            ({"attempt_id": "a",
              "items": [{"session_id": self.sid1}]},
             "items[0].message_id"),
            ({"attempt_id": "a",
              "items": [{"session_id": self.sid1, "message_id": ""}]},
             "items[0].message_id"),
            ({"attempt_id": "a",
              "items": [{"session_id": self.sid1, "message_id": 4}]},
             "items[0].message_id"),
        ]
        for payload, field in cases:
            error = self._error(
                lambda: self.service.device_inbox_retry_batch("bob", payload))
            self.assertEqual((error.status_code, error.field), (400, field),
                             payload)

    def test_device_unknown_or_revoked(self) -> None:
        error = self._error(lambda: self._batch(
            (self.sid1, "a1"), device="ghost"))
        self.assertEqual((error.status_code, error.field),
                         (409, "device_id"))
        self.service.store.revoke_device("bob")
        error = self._error(lambda: self._batch((self.sid1, "a1")))
        self.assertEqual((error.status_code, error.field),
                         (409, "device_id"))

    def test_session_unknown_is_404_on_item(self) -> None:
        error = self._error(lambda: self._batch(
            (self.sid1, "a1"), ("missing", "x")))
        self.assertEqual((error.status_code, error.field),
                         (404, "items[1].session_id"))
        # First, valid item was not written either.
        self.assertEqual(self.service.store._delivery, {})

    def test_group_session_is_409_on_item(self) -> None:
        group_sid = self._group_session()
        error = self._error(lambda: self._batch((group_sid, "g1")))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[0].session_id"))

    def test_non_recipient_is_409_on_item(self) -> None:
        # alice is the initiator of sid1, not its recipient; carol is a
        # stranger to the session.
        error = self._error(lambda: self._batch(
            (self.sid1, "a1"), device="alice"))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[0].session_id"))
        error = self._error(lambda: self._batch(
            (self.sid1, "a1"), device="carol"))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[0].session_id"))

    def test_message_unknown_is_404_on_message(self) -> None:
        error = self._error(lambda: self._batch((self.sid1, "ghost-msg")))
        self.assertEqual((error.status_code, error.field),
                         (404, "items[0].message_id"))

    def test_already_acked_is_409_on_message(self) -> None:
        self.service.ack_message(
            self.sid1, {"device_id": "bob", "message_id": "a1",
                        "sequence": 1})
        error = self._error(lambda: self._batch((self.sid1, "a1")))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[0].message_id"))

    def test_precheck_order(self) -> None:
        # A group-session item reports the session error before an unknown
        # message in the same item, and an unknown session is reported before
        # the acked check of a later valid item.
        group_sid = self._group_session()
        self.service.ack_message(
            self.sid1, {"device_id": "bob", "message_id": "a1",
                        "sequence": 1})
        error = self._error(lambda: self._batch(
            (group_sid, "g1"), (self.sid1, "a1")))
        self.assertEqual(error.field, "items[0].session_id")

    def test_first_error_in_array_order_writes_nothing(self) -> None:
        # Item 0 would succeed; item 1 fails (unknown message). Item 0 must
        # not be applied.
        error = self._error(lambda: self._batch(
            (self.sid1, "a1"), (self.sid2, "ghost")))
        self.assertEqual((error.status_code, error.field),
                         (404, "items[1].message_id"))
        self.assertEqual(self.service.store._delivery, {})

    def test_acked_state_unaffected_by_new_retries(self) -> None:
        # A fresh retry against a pending message never flips acked.
        self._batch((self.sid1, "a1"))
        state = self.service.store._delivery[(self.sid1, "a1")]
        self.assertFalse(state.acked)
        self.assertEqual(state.ack_sequence, 0)


class RetryBatchPersistenceTest(RetryBatchMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self._build()
        self.state_store = attach_persistence(
            self.service, os.path.join(self.directory, "state.json"))

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_whole_batch_persists_with_one_generation(self) -> None:
        generation = self.state_store.commit_seq
        self.service.device_inbox_retry_batch("bob", {
            "attempt_id": "batch-1",
            "items": [{"session_id": self.sid1, "message_id": "a1"},
                      {"session_id": self.sid2, "message_id": "b1"}]})
        self.assertEqual(self.state_store.commit_seq, generation + 1)

    def test_all_replay_consumes_no_generation_and_writes_nothing(self) -> None:
        payload = {"attempt_id": "batch-1",
                   "items": [{"session_id": self.sid1, "message_id": "a1"}]}
        self.service.device_inbox_retry_batch("bob", payload)
        generation = self.state_store.commit_seq
        _, status = self.service.device_inbox_retry_batch("bob", payload)
        self.assertEqual(status, 200)
        self.assertEqual(self.state_store.commit_seq, generation)

    def test_failed_write_rolls_back_the_whole_batch(self) -> None:
        def raise_oserror(state):
            raise OSError("simulated disk failure")

        self.state_store.save = raise_oserror
        with self.assertRaises(PersistenceUnavailable):
            self.service.device_inbox_retry_batch("bob", {
                "attempt_id": "batch-1",
                "items": [{"session_id": self.sid1, "message_id": "a1"},
                          {"session_id": self.sid2, "message_id": "b1"}]})
        self.assertEqual(self.service.store._delivery, {})


class RetryBatchHTTPTest(RetryBatchMixin, unittest.TestCase):
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

    def test_new_and_replay(self) -> None:
        path = "/v1/devices/bob/inbox/retry-batch"
        payload = {"attempt_id": "batch-1",
                   "items": [{"session_id": self.sid1, "message_id": "a1"},
                             {"session_id": self.sid2, "message_id": "b1"}]}
        status, body = self._request("POST", path, payload)
        self.assertEqual(status, 201)
        self.assertEqual(list(body), ["device_id", "results"])
        self.assertEqual([r["attempts"] for r in body["results"]], [1, 1])
        status, body = self._request("POST", path, payload)
        self.assertEqual(status, 200)
        self.assertEqual([r["attempts"] for r in body["results"]], [1, 1])

    def test_error_statuses_and_fields(self) -> None:
        path = "/v1/devices/bob/inbox/retry-batch"
        status, body = self._request("POST", path, raw="{bad")
        self.assertEqual((status, body["field"]), (400, "request_body"))
        status, body = self._request("POST", path, [])
        self.assertEqual((status, body["field"]), (400, "request_body"))
        status, body = self._request("POST", path, {"items": []})
        self.assertEqual((status, body["field"]), (400, "attempt_id"))
        status, body = self._request(
            "POST", path, {"attempt_id": "a", "items": [5]})
        self.assertEqual((status, body["field"]), (400, "items[0]"))
        status, body = self._request(
            "POST", path,
            {"attempt_id": "a",
             "items": [{"session_id": "x", "message_id": "y"}]})
        self.assertEqual((status, body["field"]),
                         (404, "items[0].session_id"))
        status, body = self._request(
            "POST", path,
            {"attempt_id": "a",
             "items": [{"session_id": self.sid1, "message_id": "x"}]})
        self.assertEqual((status, body["field"]),
                         (404, "items[0].message_id"))
        status, body = self._request(
            "POST", "/v1/devices/ghost/inbox/retry-batch",
            {"attempt_id": "a",
             "items": [{"session_id": self.sid1, "message_id": "a1"}]})
        self.assertEqual((status, body["field"]), (409, "device_id"))

    def test_unknown_device_path_with_malformed_body_is_400(self) -> None:
        status, body = self._request(
            "POST", "/v1/devices/ghost/inbox/retry-batch",
            {"attempt_id": "a", "items": []})
        self.assertEqual((status, body["field"]), (400, "items"))


if __name__ == "__main__":
    unittest.main()
