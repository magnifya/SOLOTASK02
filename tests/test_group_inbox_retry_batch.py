"""Tests for the device group-inbox retry-batch endpoint.

POST /v1/devices/{device_id}/group-inbox/retry-batch records one
attempt_id against several unacked group messages for one frozen member
device in one locked transaction (one persistence notification), with
the path device resolved ahead of ordered per-item prechecks.
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


class GroupRetryBatchMixin:
    def _build(self) -> None:
        self.service = DeviceService()
        # alice is the group creator/sender; bob and carol are frozen
        # recipient members; dave is registered outside the group.
        for device_id in ("alice", "bob", "carol", "dave"):
            self.service.store.add_device(Device(
                "u", device_id, "ik",
                prekeys=[SignedPreKey("pk", "pub")]))
        self.service.create_group({
            "group_id": "g1", "creator_device_id": "alice",
            "member_device_ids": ["alice", "bob", "carol"]})
        group_session = self.service.create_group_session({
            "group_id": "g1", "initiator_device_id": "alice",
            "ephemeral_key": "epk"})
        self.sid = group_session["session_id"]
        for sequence in (1, 2, 3):
            self.service.post_message({
                "session_id": self.sid,
                "sender_device_id": "alice",
                "message_id": f"m{sequence}", "sequence": sequence,
                "nonce": f"nm{sequence}", "ciphertext": "ct"})
        # A 1:1 session that must be rejected by the group endpoint.
        self.one2one = self.service.store.create_session(
            "alice", "bob", "pk", "ek1").session_id
        self.service.post_message({
            "session_id": self.one2one, "sender_device_id": "alice",
            "message_id": "p1", "sequence": 1,
            "nonce": "np1", "ciphertext": "ct"})

    def _retry_batch(self, device_id="bob", attempt_id="att-1", items=None):
        if items is None:
            items = [{"session_id": self.sid, "message_id": "m1"},
                     {"session_id": self.sid, "message_id": "m2"}]
        return self.service.group_inbox_retry_batch(
            device_id, {"attempt_id": attempt_id, "items": items})


class GroupRetryBatchServiceTest(GroupRetryBatchMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def _error(self, callable_):
        with self.assertRaises(ServiceError) as caught:
            callable_()
        return caught.exception

    def test_first_batch_counts_every_item(self) -> None:
        body, status = self._retry_batch()
        self.assertEqual(status, 201)
        self.assertEqual(list(body), ["device_id", "results"])
        self.assertEqual(body["device_id"], "bob")
        self.assertEqual(
            [(r["session_id"], r["message_id"], r["attempts"])
             for r in body["results"]],
            [(self.sid, "m1", 1), (self.sid, "m2", 1)])
        for result in body["results"]:
            self.assertEqual(list(result),
                             ["session_id", "message_id", "attempts"])

    def test_full_replay_is_200_and_counts_nothing(self) -> None:
        _, first = self._retry_batch(attempt_id="att-1")
        self.assertEqual(first, 201)
        body, second = self._retry_batch(attempt_id="att-1")
        self.assertEqual(second, 200)
        self.assertEqual([r["attempts"] for r in body["results"]], [1, 1])

    def test_new_attempt_id_counts_again_per_message(self) -> None:
        self._retry_batch(attempt_id="att-1")
        body, status = self._retry_batch(
            attempt_id="att-2",
            items=[{"session_id": self.sid, "message_id": "m1"}])
        self.assertEqual(status, 201)
        self.assertEqual(body["results"][0]["attempts"], 2)
        body, status = self._retry_batch(
            attempt_id="att-2",
            items=[{"session_id": self.sid, "message_id": "m1"}])
        self.assertEqual(status, 200)
        self.assertEqual(body["results"][0]["attempts"], 2)

    def test_results_keep_input_order(self) -> None:
        body, _ = self._retry_batch(items=[
            {"session_id": self.sid, "message_id": "m3"},
            {"session_id": self.sid, "message_id": "m1"}])
        self.assertEqual(
            [r["message_id"] for r in body["results"]], ["m3", "m1"])

    def test_shape_errors(self) -> None:
        cases = [
            ({"items": []}, "attempt_id"),
            ({"attempt_id": "", "items": [
                {"session_id": self.sid, "message_id": "m1"}]},
             "attempt_id"),
            ({"attempt_id": 7, "items": []}, "attempt_id"),
            ({"attempt_id": "att-1"}, "items"),
            ({"attempt_id": "att-1", "items": []}, "items"),
            ({"attempt_id": "att-1", "items": "x"}, "items"),
            ({"attempt_id": "att-1", "items": [9]}, "items[0]"),
            ({"attempt_id": "att-1", "items": [
                {"message_id": "m1"}]}, "items[0].session_id"),
            ({"attempt_id": "att-1", "items": [
                {"session_id": "", "message_id": "m1"}]},
             "items[0].session_id"),
            ({"attempt_id": "att-1", "items": [
                {"session_id": self.sid}]}, "items[0].message_id"),
            ({"attempt_id": "att-1", "items": [
                {"session_id": self.sid, "message_id": None}]},
             "items[0].message_id"),
            ({"attempt_id": "att-1", "items": [
                {"session_id": self.sid, "message_id": "m1"},
                {"session_id": self.sid, "message_id": "m1"}]},
             "items[1]"),
            ({"attempt_id": "att-1", "items": [
                {"session_id": self.sid, "message_id": "m1"}],
              "bogus": 1}, "bogus"),
            ({"attempt_id": "att-1", "items": [
                {"session_id": self.sid, "message_id": "m1",
                 "bogus": 1}]}, "items[0].bogus"),
            ([1, 2], "request_body"),
            ("x", "request_body"),
        ]
        for payload, field in cases:
            error = self._error(
                lambda payload=payload:
                self.service.group_inbox_retry_batch("bob", payload))
            self.assertEqual(error.status_code, 400, payload)
            self.assertEqual(error.to_body()["field"], field, payload)

    def test_device_unknown_or_revoked_is_409_before_items(self) -> None:
        payload = {"attempt_id": "att-1",
                   "items": [{"session_id": "ghost", "message_id": "m1"}]}
        error = self._error(
            lambda: self.service.group_inbox_retry_batch("ghost", payload))
        self.assertEqual((error.status_code, error.to_body()["field"]),
                         (409, "device_id"))
        self.service.revoke_device("carol")
        error = self._error(
            lambda: self.service.group_inbox_retry_batch("carol", payload))
        self.assertEqual((error.status_code, error.to_body()["field"]),
                         (409, "device_id"))

    def test_prechecks_in_order_write_nothing(self) -> None:
        cases = [
            ("bob", "ghost", "m1", 404, "items[0].session_id"),
            ("bob", self.one2one, "p1", 409, "items[0].session_id"),
            ("dave", self.sid, "m1", 409, "items[0].session_id"),
            ("bob", self.sid, "ghost", 404, "items[0].message_id"),
            ("alice", self.sid, "m1", 409, "items[0].session_id"),
        ]
        for device_id, session_id, message_id, status, field in cases:
            error = self._error(
                lambda device_id=device_id, session_id=session_id,
                message_id=message_id: self.service.group_inbox_retry_batch(
                    device_id, {"attempt_id": "att-1", "items": [
                        {"session_id": session_id,
                         "message_id": message_id}]}))
            self.assertEqual((error.status_code, error.to_body()["field"]),
                             (status, field), (session_id, message_id))
        self.assertEqual(self.service.store._group_delivery, {})

    def test_first_error_aborts_the_whole_batch(self) -> None:
        with self.assertRaises(ServiceError):
            self._retry_batch(items=[
                {"session_id": self.sid, "message_id": "m1"},
                {"session_id": self.sid, "message_id": "ghost"}])
        self.assertEqual(self.service.store._group_delivery, {})

    def test_acked_message_is_409_and_never_unacks(self) -> None:
        self.service.ack_message(self.sid, {
            "device_id": "carol", "message_id": "m1", "sequence": 1})
        error = self._error(
            lambda: self._retry_batch(device_id="carol"))
        self.assertEqual((error.status_code, error.to_body()["field"]),
                         (409, "items[0].message_id"))
        view = self.service.message_status(self.sid, "m1", "carol")
        self.assertEqual(view["status"], "acked")
        self.assertEqual(view["attempts"], 0)

    def test_member_removed_after_freeze_stays_allowed(self) -> None:
        self.service.remove_group_member("g1", {
            "actor_device_id": "alice", "device_id": "bob"})
        _, status = self._retry_batch()
        self.assertEqual(status, 201)

    def test_member_added_after_freeze_is_rejected(self) -> None:
        self.service.add_group_member("g1", {
            "actor_device_id": "alice", "device_id": "dave"})
        error = self._error(
            lambda: self._retry_batch(device_id="dave"))
        self.assertEqual((error.status_code, error.to_body()["field"]),
                         (409, "items[0].session_id"))

    def test_devices_have_independent_counters(self) -> None:
        body_bob, _ = self._retry_batch(
            device_id="bob", attempt_id="att-1")
        body_carol, status = self._retry_batch(
            device_id="carol", attempt_id="att-1")
        self.assertEqual(status, 201)
        self.assertEqual(
            [r["attempts"] for r in body_bob["results"]], [1, 1])
        self.assertEqual(
            [r["attempts"] for r in body_carol["results"]], [1, 1])


class GroupRetryBatchPersistenceTest(GroupRetryBatchMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()
        self.tmpdir = tempfile.mkdtemp()
        self.state_store = attach_persistence(
            self.service, os.path.join(self.tmpdir, "state.json"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_whole_batch_persists_with_one_generation(self) -> None:
        generation = self.state_store.commit_seq
        self._retry_batch()
        self.assertEqual(self.state_store.commit_seq, generation + 1)

    def test_full_replay_consumes_no_generation(self) -> None:
        self._retry_batch(attempt_id="att-1")
        generation = self.state_store.commit_seq
        _, status = self._retry_batch(attempt_id="att-1")
        self.assertEqual(status, 200)
        self.assertEqual(self.state_store.commit_seq, generation)

    def test_failed_write_rolls_back_the_whole_batch(self) -> None:
        def raise_oserror(state, bootstrap=False):
            raise OSError("simulated disk failure")

        self.state_store.save = raise_oserror
        with self.assertRaises(PersistenceUnavailable):
            self._retry_batch(device_id="carol")
        self.assertIsNone(
            self.service.store._group_delivery.get(
                (self.sid, "m1", "carol")))

    def test_restart_recovers_attempts_and_dedup_sets(self) -> None:
        self._retry_batch(attempt_id="att-1")
        restarted = DeviceService()
        attach_persistence(restarted, self.state_store.path)
        _, status = restarted.group_inbox_retry_batch("bob", {
            "attempt_id": "att-1",
            "items": [{"session_id": self.sid, "message_id": "m1"},
                      {"session_id": self.sid, "message_id": "m2"}]})
        self.assertEqual(status, 200)
        _, status = restarted.group_inbox_retry_batch("bob", {
            "attempt_id": "att-2",
            "items": [{"session_id": self.sid, "message_id": "m1"}]})
        self.assertEqual(status, 201)
        view = restarted.message_status(self.sid, "m1", "bob")
        self.assertEqual(view["attempts"], 2)
        with open(self.state_store.path) as handle:
            document = json.load(handle)
        self.assertEqual(document["version"], 1)
        self.assertIn("group_delivery", document)


class GroupRetryBatchHTTPTest(GroupRetryBatchMixin, unittest.TestCase):
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
        self.thread.join(timeout=5)

    def _post(self, device_id, body, raw=False):
        connection = HTTPConnection("127.0.0.1", self.port, timeout=5)
        data = body if raw else json.dumps(body)
        connection.request(
            "POST",
            f"/v1/devices/{device_id}/group-inbox/retry-batch",
            data, {"Content-Type": "application/json"})
        response = connection.getresponse()
        payload = json.loads(response.read())
        connection.close()
        return response.status, payload

    def test_endpoint_round_trip(self) -> None:
        status, body = self._post("bob", {
            "attempt_id": "att-1",
            "items": [{"session_id": self.sid, "message_id": "m1"}]})
        self.assertEqual(status, 201)
        self.assertEqual(body["results"][0]["attempts"], 1)
        status, _ = self._post("bob", {
            "attempt_id": "att-1",
            "items": [{"session_id": self.sid, "message_id": "m1"}]})
        self.assertEqual(status, 200)

    def test_malformed_json_is_400_request_body(self) -> None:
        status, body = self._post("bob", "{not json", raw=True)
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "request_body")

    def test_shape_error_field(self) -> None:
        status, body = self._post("bob", {"attempt_id": "att-1"})
        self.assertEqual((status, body["field"]), (400, "items"))

    def test_precondition_error_fields(self) -> None:
        status, body = self._post("dave", {
            "attempt_id": "att-1",
            "items": [{"session_id": self.sid, "message_id": "m1"}]})
        self.assertEqual((status, body["field"]),
                         (409, "items[0].session_id"))
        status, body = self._post("bob", {
            "attempt_id": "att-1",
            "items": [{"session_id": self.sid, "message_id": "ghost"}]})
        self.assertEqual((status, body["field"]),
                         (404, "items[0].message_id"))

    def test_unknown_device_is_409_device_id(self) -> None:
        status, body = self._post("ghost", {
            "attempt_id": "att-1",
            "items": [{"session_id": self.sid, "message_id": "m1"}]})
        self.assertEqual((status, body["field"]), (409, "device_id"))
