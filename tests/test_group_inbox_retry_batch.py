"""Tests for the device group-inbox retry-batch endpoint.

POST /v1/devices/{device_id}/group-inbox/retry-batch records one
attempt_id against several unacked group messages for one frozen member
device in one locked transaction (one persistence notification), with
ordered per-item prechecks.
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
        for device_id in ("alice", "bob", "carol", "dan"):
            self.service.store.add_device(Device("u", device_id, "ik"))
        # A 1:1 session (the group endpoint must reject it with 404).
        self.service.store.add_device(Device(
            "u", "eve", "ik",
            prekeys=[SignedPreKey("pk1", "pubk1")]))
        self.one_sid = self.service.store.create_session(
            "alice", "eve", "pk1", "ek1").session_id
        self.service.post_message({
            "session_id": self.one_sid, "sender_device_id": "alice",
            "message_id": "o1", "sequence": 1,
            "nonce": "no1", "ciphertext": "ct"})
        # A group whose frozen members are alice, bob and carol.
        self.service.create_group({
            "group_id": "g1", "creator_device_id": "alice",
            "member_device_ids": ["bob", "carol"]})
        group_session = self.service.create_group_session({
            "group_id": "g1", "initiator_device_id": "alice",
            "ephemeral_key": "epk"})
        self.sid = group_session["session_id"]
        for sequence in range(1, 4):
            self.service.post_message({
                "session_id": self.sid,
                "sender_device_id": "alice",
                "message_id": f"m{sequence}", "sequence": sequence,
                "nonce": f"n{sequence}", "ciphertext": "ct"})

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
        self.assertEqual(len(body["results"]), 2)
        for result in body["results"]:
            self.assertEqual(list(result),
                             ["session_id", "message_id", "attempts"])
        self.assertEqual(
            [(r["message_id"], r["attempts"]) for r in body["results"]],
            [("m1", 1), ("m2", 1)])

    def test_results_keep_input_order(self) -> None:
        body, _ = self._retry_batch(items=[
            {"session_id": self.sid, "message_id": "m3"},
            {"session_id": self.sid, "message_id": "m1"}])
        self.assertEqual([r["message_id"] for r in body["results"]],
                         ["m3", "m1"])

    def test_full_replay_is_200_and_counts_nothing(self) -> None:
        _, first = self._retry_batch(attempt_id="att-1")
        self.assertEqual(first, 201)
        body, second = self._retry_batch(attempt_id="att-1")
        self.assertEqual(second, 200)
        self.assertEqual([r["attempts"] for r in body["results"]], [1, 1])

    def test_mixed_replay_and_new_is_201(self) -> None:
        self._retry_batch(attempt_id="att-1", items=[
            {"session_id": self.sid, "message_id": "m1"}])
        body, status = self._retry_batch(attempt_id="att-1", items=[
            {"session_id": self.sid, "message_id": "m1"},
            {"session_id": self.sid, "message_id": "m2"}])
        self.assertEqual(status, 201)
        self.assertEqual([r["attempts"] for r in body["results"]], [1, 1])
        body, status = self._retry_batch(attempt_id="att-2", items=[
            {"session_id": self.sid, "message_id": "m1"},
            {"session_id": self.sid, "message_id": "m2"}])
        self.assertEqual(status, 201)
        self.assertEqual([r["attempts"] for r in body["results"]], [2, 2])

    def test_dedup_is_per_message_not_per_batch(self) -> None:
        self._retry_batch(attempt_id="att-1", items=[
            {"session_id": self.sid, "message_id": "m1"}])
        body, status = self._retry_batch(attempt_id="att-1", items=[
            {"session_id": self.sid, "message_id": "m1"},
            {"session_id": self.sid, "message_id": "m2"}])
        self.assertEqual(status, 201)
        self.assertEqual(
            [(r["message_id"], r["attempts"]) for r in body["results"]],
            [("m1", 1), ("m2", 1)])

    def test_devices_have_independent_counters(self) -> None:
        body, _ = self._retry_batch(device_id="bob", attempt_id="att-1")
        self.assertEqual([r["attempts"] for r in body["results"]], [1, 1])
        body, status = self._retry_batch(
            device_id="carol", attempt_id="att-1")
        self.assertEqual(status, 201)
        self.assertEqual([r["attempts"] for r in body["results"]], [1, 1])

    def test_body_shape_errors(self) -> None:
        cases = [
            (None, "request_body"),
            ([], "request_body"),
            ({"items": [{"session_id": self.sid, "message_id": "m1"}]},
             "attempt_id"),
            ({"attempt_id": "", "items": []}, "attempt_id"),
            ({"attempt_id": 7, "items": []}, "attempt_id"),
            ({"attempt_id": "a"}, "items"),
            ({"attempt_id": "a", "items": []}, "items"),
            ({"attempt_id": "a", "items": {}}, "items"),
            ({"attempt_id": "a", "items": [7]}, "items[0]"),
            ({"attempt_id": "a", "items": [{"message_id": "m1"}]},
             "items[0].session_id"),
            ({"attempt_id": "a",
              "items": [{"session_id": "", "message_id": "m1"}]},
             "items[0].session_id"),
            ({"attempt_id": "a",
              "items": [{"session_id": 9, "message_id": "m1"}]},
             "items[0].session_id"),
            ({"attempt_id": "a", "items": [{"session_id": self.sid}]},
             "items[0].message_id"),
            ({"attempt_id": "a",
              "items": [{"session_id": self.sid, "message_id": ""}]},
             "items[0].message_id"),
            ({"attempt_id": "a",
              "items": [{"session_id": self.sid, "message_id": 3}]},
             "items[0].message_id"),
        ]
        for payload, field in cases:
            error = self._error(lambda payload=payload:
                                self.service.group_inbox_retry_batch(
                                    "bob", payload))
            self.assertEqual((error.status_code, error.field), (400, field),
                             payload)

    def test_extra_fields_use_first_key(self) -> None:
        error = self._error(lambda: self.service.group_inbox_retry_batch(
            "bob", {"attempt_id": "a",
                    "items": [{"session_id": self.sid, "message_id": "m1"}],
                    "bogus": 1, "other": 2}))
        self.assertEqual((error.status_code, error.field), (400, "bogus"))
        error = self._error(lambda: self.service.group_inbox_retry_batch(
            "bob", {"zz": 1, "attempt_id": "a", "items": []}))
        self.assertEqual((error.status_code, error.field), (400, "zz"))
        error = self._error(lambda: self.service.group_inbox_retry_batch(
            "bob", {"attempt_id": "a", "items": [
                {"session_id": self.sid, "message_id": "m1",
                 "bogus": 1}]}))
        self.assertEqual((error.status_code, error.field),
                         (400, "items[0]"))

    def test_duplicate_pair_is_400_items_i(self) -> None:
        error = self._error(lambda: self._retry_batch(items=[
            {"session_id": self.sid, "message_id": "m1"},
            {"session_id": self.sid, "message_id": "m1"}]))
        self.assertEqual((error.status_code, error.field),
                         (400, "items[1]"))
        self.assertEqual(self.service.store._group_delivery, {})

    def test_unknown_or_revoked_device_is_409_before_items(self) -> None:
        error = self._error(lambda: self._retry_batch(
            device_id="ghost",
            items=[{"session_id": "nope", "message_id": "m1"}]))
        self.assertEqual((error.status_code, error.field),
                         (409, "device_id"))
        self.service.store.revoke_device("bob")
        error = self._error(lambda: self._retry_batch(device_id="bob"))
        self.assertEqual((error.status_code, error.field),
                         (409, "device_id"))

    def test_non_group_session_is_404_session(self) -> None:
        error = self._error(lambda: self._retry_batch(items=[
            {"session_id": "nope", "message_id": "m1"}]))
        self.assertEqual((error.status_code, error.field),
                         (404, "items[0].session_id"))
        error = self._error(lambda: self._retry_batch(items=[
            {"session_id": self.one_sid, "message_id": "o1"}]))
        self.assertEqual((error.status_code, error.field),
                         (404, "items[0].session_id"))

    def test_non_frozen_member_is_409_session(self) -> None:
        error = self._error(lambda: self._retry_batch(device_id="dan"))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[0].session_id"))

    def test_sender_is_409_session(self) -> None:
        error = self._error(lambda: self._retry_batch(device_id="alice"))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[0].session_id"))
        self.assertEqual(self.service.store._group_delivery, {})

    def test_unknown_message_is_404_message(self) -> None:
        error = self._error(lambda: self._retry_batch(items=[
            {"session_id": self.sid, "message_id": "ghost"}]))
        self.assertEqual((error.status_code, error.field),
                         (404, "items[0].message_id"))

    def test_acked_message_is_409_message(self) -> None:
        self.service.ack_message(self.sid, {
            "device_id": "bob", "message_id": "m1", "sequence": 1})
        error = self._error(lambda: self._retry_batch(items=[
            {"session_id": self.sid, "message_id": "m1"}]))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[0].message_id"))

    def test_first_error_in_array_order_writes_nothing(self) -> None:
        error = self._error(lambda: self._retry_batch(items=[
            {"session_id": self.sid, "message_id": "m2"},
            {"session_id": self.sid, "message_id": "ghost"},
            {"session_id": "nope", "message_id": "m1"}]))
        self.assertEqual((error.status_code, error.field),
                         (404, "items[1].message_id"))
        self.assertEqual(self.service.store._group_delivery, {})

    def test_frozen_snapshot_survives_roster_changes(self) -> None:
        # Removed after the freeze: still able to register attempts.
        self.service.remove_group_member("g1", {
            "actor_device_id": "alice", "device_id": "bob"})
        _, status = self._retry_batch(device_id="bob", attempt_id="att-1")
        self.assertEqual(status, 201)
        # Joined only after the freeze: never allowed.
        self.service.add_group_member("g1", {
            "actor_device_id": "alice", "device_id": "dan"})
        error = self._error(lambda: self._retry_batch(device_id="dan"))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[0].session_id"))


class GroupRetryBatchPersistenceTest(GroupRetryBatchMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self._build()
        self.state_store = attach_persistence(
            self.service, os.path.join(self.directory, "state.json"))

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

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
        def raise_oserror(state):
            raise OSError("simulated disk failure")

        self.state_store.save = raise_oserror
        with self.assertRaises(PersistenceUnavailable):
            self._retry_batch()
        self.assertEqual(self.service.store._group_delivery, {})

    def test_restart_recovers_attempts_and_dedup_sets(self) -> None:
        self._retry_batch(attempt_id="att-1")
        restarted = DeviceService()
        attach_persistence(restarted, self.state_store.path)
        view, status = restarted.group_inbox_retry_batch("bob", {
            "attempt_id": "att-1",
            "items": [{"session_id": self.sid, "message_id": "m1"},
                      {"session_id": self.sid, "message_id": "m2"}]})
        self.assertEqual(status, 200)
        self.assertEqual([r["attempts"] for r in view["results"]], [1, 1])
        _, status = restarted.group_inbox_retry_batch("bob", {
            "attempt_id": "att-2",
            "items": [{"session_id": self.sid, "message_id": "m1"}]})
        self.assertEqual(status, 201)
        status_view = restarted.message_status(self.sid, "m1", "bob")
        self.assertEqual(status_view["attempts"], 2)


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
        self.thread.join(timeout=2)

    def _request(self, path, body=None, raw=None):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        if raw is not None:
            conn.request("POST", path, raw,
                         {"Content-Type": "application/json"})
        else:
            data = json.dumps(body).encode("utf-8") if body is not None \
                else b""
            conn.request("POST", path, data,
                         {"Content-Type": "application/json"})
        response = conn.getresponse()
        raw_body = response.read()
        return response.status, raw_body, json.loads(raw_body.decode("utf-8"))

    def _path(self, device_id="bob"):
        return f"/v1/devices/{device_id}/group-inbox/retry-batch"

    def test_first_batch_and_replay(self) -> None:
        payload = {"attempt_id": "att-1", "items": [
            {"session_id": self.sid, "message_id": "m1"},
            {"session_id": self.sid, "message_id": "m2"}]}
        status, raw, body = self._request(self._path(), payload)
        self.assertEqual(status, 201)
        self.assertEqual(list(body), ["device_id", "results"])
        for result in body["results"]:
            self.assertEqual(list(result),
                             ["session_id", "message_id", "attempts"])
        self.assertEqual([r["attempts"] for r in body["results"]], [1, 1])
        status, raw, body = self._request(self._path(), payload)
        self.assertEqual(status, 200)
        self.assertEqual([r["attempts"] for r in body["results"]], [1, 1])

    def test_error_statuses_and_fields(self) -> None:
        status, raw, body = self._request(self._path(), raw="{bad")
        self.assertEqual((status, body["field"]), (400, "request_body"))
        status, raw, body = self._request(self._path(), [])
        self.assertEqual((status, body["field"]), (400, "request_body"))
        status, raw, body = self._request(
            self._path(), {"attempt_id": "a", "items": []})
        self.assertEqual((status, body["field"]), (400, "items"))
        status, raw, body = self._request(
            self._path(), {"attempt_id": 1,
                           "items": [{"session_id": "x", "message_id": "y"}]})
        self.assertEqual((status, body["field"]), (400, "attempt_id"))
        status, raw, body = self._request(
            self._path(), {"attempt_id": "a", "items": [5]})
        self.assertEqual((status, body["field"]), (400, "items[0]"))
        status, raw, body = self._request(
            self._path(), {"attempt_id": "a", "items": [
                {"session_id": "x", "message_id": "y", "z": 1}]})
        self.assertEqual((status, body["field"]), (400, "items[0]"))
        status, raw, body = self._request(
            self._path(), {"attempt_id": "a", "bogus": 1, "items": [
                {"session_id": "x", "message_id": "y"}]})
        self.assertEqual((status, body["field"]), (400, "bogus"))
        status, raw, body = self._request(
            self._path(), {"attempt_id": "a", "items": [
                {"session_id": "x", "message_id": "y"}]})
        self.assertEqual((status, body["field"]),
                         (404, "items[0].session_id"))
        status, raw, body = self._request(
            self._path(), {"attempt_id": "a", "items": [
                {"session_id": self.sid, "message_id": "y"}]})
        self.assertEqual((status, body["field"]),
                         (404, "items[0].message_id"))
        status, raw, body = self._request(
            self._path(), {"attempt_id": "a", "items": [
                {"session_id": self.sid, "message_id": "m1"}]}, )
        # alice is the sender: 409 session.
        status, raw, body = self._request(
            self._path("alice"), {"attempt_id": "a", "items": [
                {"session_id": self.sid, "message_id": "m1"}]})
        self.assertEqual((status, body["field"]),
                         (409, "items[0].session_id"))
        self.service.ack_message(self.sid, {
            "device_id": "bob", "message_id": "m1", "sequence": 1})
        status, raw, body = self._request(
            self._path(), {"attempt_id": "a", "items": [
                {"session_id": self.sid, "message_id": "m1"}]})
        self.assertEqual((status, body["field"]),
                         (409, "items[0].message_id"))
        status, raw, body = self._request(
            self._path("ghost"), {"attempt_id": "a", "items": [
                {"session_id": self.sid, "message_id": "m1"}]})
        self.assertEqual((status, body["field"]), (409, "device_id"))
        self.assertEqual(list(body), ["message", "field"])

    def test_error_body_key_order(self) -> None:
        status, raw, body = self._request(
            self._path(), {"attempt_id": "a", "items": []})
        self.assertEqual(status, 400)
        self.assertEqual(raw, b'{"message":"field must be a non-empty array: '
                             b'items","field":"items"}')

    def test_unknown_subpath_is_404(self) -> None:
        status, _, body = self._request(
            "/v1/devices//group-inbox/retry-batch",
            {"attempt_id": "a", "items": [
                {"session_id": self.sid, "message_id": "m1"}]})
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "device_id")


class GroupRetryBatchHTTPFailureTest(GroupRetryBatchMixin, unittest.TestCase):
    """The 503/data_file path needs a real data file to fail."""

    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self._build()
        self.state_store = attach_persistence(
            self.service, os.path.join(self.directory, "state.json"))
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
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_failed_write_is_503_data_file(self) -> None:
        def raise_oserror(state):
            raise OSError("simulated disk failure")

        self.state_store.save = raise_oserror
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("POST", "/v1/devices/bob/group-inbox/retry-batch",
                     json.dumps({"attempt_id": "att-1", "items": [
                         {"session_id": self.sid, "message_id": "m1"},
                         {"session_id": self.sid, "message_id": "m2"}]}),
                     {"Content-Type": "application/json"})
        response = conn.getresponse()
        self.assertEqual(response.status, 503)
        body = json.loads(response.read().decode("utf-8"))
        self.assertEqual(list(body), ["message", "field"])
        self.assertEqual(body["field"], "data_file")
        self.assertEqual(self.service.store._group_delivery, {})


if __name__ == "__main__":
    unittest.main()
