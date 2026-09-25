"""Tests for the device 1:1 inbox retry-batch endpoint.

POST /v1/devices/{device_id}/inbox/retry-batch records one attempt_id
against several unacked 1:1 messages in one locked transaction (one
persistence notification), with ordered per-item prechecks.
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
        # bob holds three pre-keys so several distinct 1:1 sessions can be
        # addressed to him; bob2 is a second device of the same user.
        self.service.store.add_device(Device(
            "u", "bob", "ik",
            prekeys=[SignedPreKey("pk1", "pubk1"),
                     SignedPreKey("pk2", "pubk2"),
                     SignedPreKey("pk3", "pubk3")]))
        self.service.store.add_device(
            Device("u", "carol", "ik",
                   prekeys=[SignedPreKey("pkC", "pubkC")]))
        # Two sessions addressed to bob.
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
        # A session bob initiated towards carol (bob is not its recipient).
        self.sid_out = self.service.store.create_session(
            "bob", "carol", "pkC", "ekC").session_id
        self.service.post_message({
            "session_id": self.sid_out, "sender_device_id": "carol",
            "message_id": "c1", "sequence": 1,
            "nonce": "nc1", "ciphertext": "ct"})
        # A group session including bob (must not be retry-able here).
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

    def _retry_batch(self, device_id="bob", attempt_id="att-1", items=None):
        if items is None:
            items = [{"session_id": self.sid1, "message_id": "a1"},
                     {"session_id": self.sid1, "message_id": "a2"},
                     {"session_id": self.sid2, "message_id": "b1"}]
        return self.service.inbox_retry_batch(
            device_id, {"attempt_id": attempt_id, "items": items})


class RetryBatchServiceTest(RetryBatchMixin, unittest.TestCase):
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
        self.assertEqual(len(body["results"]), 3)
        for result in body["results"]:
            self.assertEqual(list(result),
                             ["session_id", "message_id", "attempts"])
        self.assertEqual(
            [(r["session_id"], r["message_id"], r["attempts"])
             for r in body["results"]],
            [(self.sid1, "a1", 1), (self.sid1, "a2", 1),
             (self.sid2, "b1", 1)])

    def test_results_keep_input_order(self) -> None:
        items = [{"session_id": self.sid2, "message_id": "b1"},
                 {"session_id": self.sid1, "message_id": "a3"},
                 {"session_id": self.sid1, "message_id": "a1"}]
        body, _ = self._retry_batch(items=items)
        self.assertEqual(
            [(r["session_id"], r["message_id"]) for r in body["results"]],
            [(self.sid2, "b1"), (self.sid1, "a3"), (self.sid1, "a1")])

    def test_full_replay_is_200_and_counts_nothing(self) -> None:
        body, first = self._retry_batch(attempt_id="att-1")
        self.assertEqual(first, 201)
        body, second = self._retry_batch(attempt_id="att-1")
        self.assertEqual(second, 200)
        self.assertEqual([r["attempts"] for r in body["results"]], [1, 1, 1])

    def test_mixed_replay_and_new_is_201(self) -> None:
        self._retry_batch(attempt_id="att-1", items=[
            {"session_id": self.sid1, "message_id": "a1"},
            {"session_id": self.sid1, "message_id": "a2"}])
        # a1 already saw att-1 (replay); b1 has not (new).
        body, status = self._retry_batch(attempt_id="att-1", items=[
            {"session_id": self.sid1, "message_id": "a1"},
            {"session_id": self.sid2, "message_id": "b1"}])
        self.assertEqual(status, 201)
        self.assertEqual(
            [(r["message_id"], r["attempts"]) for r in body["results"]],
            [("a1", 1), ("b1", 1)])
        # A further distinct attempt id counts on both messages.
        body, status = self._retry_batch(attempt_id="att-2", items=[
            {"session_id": self.sid1, "message_id": "a1"},
            {"session_id": self.sid2, "message_id": "b1"}])
        self.assertEqual(status, 201)
        self.assertEqual(
            [(r["message_id"], r["attempts"]) for r in body["results"]],
            [("a1", 2), ("b1", 2)])

    def test_body_validation(self) -> None:
        for bad in (None, [], "x", 5, False):
            error = self._error(
                lambda bad=bad: self.service.inbox_retry_batch("bob", bad))
            self.assertEqual((error.status_code, error.field),
                             (400, "request_body"), bad)
        for payload in ({}, {"items": []}, {"attempt_id": "", "items": []},
                        {"attempt_id": 5, "items": []}):
            error = self._error(
                lambda payload=payload:
                self.service.inbox_retry_batch("bob", payload))
            self.assertEqual((error.status_code, error.field),
                             (400, "attempt_id"), payload)
        for payload in ({"attempt_id": "a"},
                        {"attempt_id": "a", "items": []},
                        {"attempt_id": "a", "items": "x"},
                        {"attempt_id": "a", "items": 5}):
            error = self._error(
                lambda payload=payload:
                self.service.inbox_retry_batch("bob", payload))
            self.assertEqual((error.status_code, error.field),
                             (400, "items"), payload)
        base = [{"session_id": self.sid1, "message_id": "a1"}]
        for element in (5, "x", None, ["a"]):
            error = self._error(
                lambda element=element: self.service.inbox_retry_batch(
                    "bob", {"attempt_id": "a", "items": [element]}))
            self.assertEqual((error.status_code, error.field),
                             (400, "items[0]"), element)
        for element in ({"message_id": "a1"},
                        {"session_id": "", "message_id": "a1"},
                        {"session_id": 7, "message_id": "a1"},
                        {"session_id": None, "message_id": "a1"}):
            error = self._error(
                lambda element=element: self.service.inbox_retry_batch(
                    "bob", {"attempt_id": "a", "items": [element]}))
            self.assertEqual((error.status_code, error.field),
                             (400, "items[0].session_id"), element)
        for element in ({"session_id": self.sid1},
                        {"session_id": self.sid1, "message_id": ""},
                        {"session_id": self.sid1, "message_id": 7},
                        {"session_id": self.sid1, "message_id": False}):
            error = self._error(
                lambda element=element: self.service.inbox_retry_batch(
                    "bob", {"attempt_id": "a", "items": [element]}))
            self.assertEqual((error.status_code, error.field),
                             (400, "items[0].message_id"), element)

    def test_duplicate_pair_is_400_on_the_item(self) -> None:
        error = self._error(lambda: self._retry_batch(items=[
            {"session_id": self.sid1, "message_id": "a1"},
            {"session_id": self.sid1, "message_id": "a2"},
            {"session_id": self.sid1, "message_id": "a1"}]))
        self.assertEqual((error.status_code, error.field),
                         (400, "items[2]"))
        # The same message id in a different session is a different pair.
        body, status = self._retry_batch(items=[
            {"session_id": self.sid1, "message_id": "a1"},
            {"session_id": self.sid2, "message_id": "b1"}])
        self.assertEqual(status, 201)
        self.assertEqual(len(body["results"]), 2)

    def test_device_unknown_or_revoked(self) -> None:
        error = self._error(lambda: self._retry_batch(device_id="ghost"))
        self.assertEqual((error.status_code, error.field),
                         (409, "device_id"))
        self.service.store.revoke_device("bob")
        error = self._error(lambda: self._retry_batch(device_id="bob"))
        self.assertEqual((error.status_code, error.field),
                         (409, "device_id"))

    def test_session_unknown_is_404_on_item(self) -> None:
        error = self._error(lambda: self._retry_batch(items=[
            {"session_id": "nope", "message_id": "a1"}]))
        self.assertEqual((error.status_code, error.field),
                         (404, "items[0].session_id"))

    def test_group_session_is_409_on_item(self) -> None:
        error = self._error(lambda: self._retry_batch(items=[
            {"session_id": self.group_sid, "message_id": "g1"}]))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[0].session_id"))

    def test_non_recipient_is_409_on_item(self) -> None:
        # sid_out is a 1:1 session addressed to carol, not bob.
        error = self._error(lambda: self._retry_batch(items=[
            {"session_id": self.sid_out, "message_id": "c1"}]))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[0].session_id"))

    def test_message_unknown_is_404_on_item(self) -> None:
        error = self._error(lambda: self._retry_batch(items=[
            {"session_id": self.sid1, "message_id": "ghost"}]))
        self.assertEqual((error.status_code, error.field),
                         (404, "items[0].message_id"))

    def test_acked_message_is_409_on_item(self) -> None:
        self.service.ack_message(self.sid1, {
            "device_id": "bob", "message_id": "a1", "sequence": 1})
        error = self._error(lambda: self._retry_batch(items=[
            {"session_id": self.sid1, "message_id": "a1"}]))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[0].message_id"))

    def test_precedence_session_before_message(self) -> None:
        # An unknown session reports the session field even with an unknown
        # message id.
        error = self._error(lambda: self._retry_batch(items=[
            {"session_id": "nope", "message_id": "ghost"}]))
        self.assertEqual((error.status_code, error.field),
                         (404, "items[0].session_id"))

    def test_first_error_in_array_order_writes_nothing(self) -> None:
        error = self._error(lambda: self._retry_batch(items=[
            {"session_id": self.sid1, "message_id": "a1"},
            {"session_id": self.sid1, "message_id": "ghost"},
            {"session_id": "nope", "message_id": "a1"}]))
        self.assertEqual((error.status_code, error.field),
                         (404, "items[1].message_id"))
        # No delivery record may be created for the prechecked first item.
        self.assertEqual(self.service.store._delivery, {})

    def test_failed_item_does_not_affect_other_sessions(self) -> None:
        error = self._error(lambda: self._retry_batch(items=[
            {"session_id": self.group_sid, "message_id": "g1"}]))
        self.assertEqual(error.field, "items[0].session_id")
        self.assertEqual(self.service.store._delivery, {})
        self.assertEqual(self.service.store._group_delivery, {})


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
        self.assertEqual(self.service.store._delivery, {})

    def test_restart_recovers_attempts_and_dedup_sets(self) -> None:
        self._retry_batch(attempt_id="att-1")
        restarted = DeviceService()
        attach_persistence(restarted, self.state_store.path)
        view, status = restarted.inbox_retry_batch("bob", {
            "attempt_id": "att-1",
            "items": [{"session_id": self.sid1, "message_id": "a1"},
                      {"session_id": self.sid2, "message_id": "b1"}]})
        self.assertEqual(status, 200)
        self.assertEqual([r["attempts"] for r in view["results"]], [1, 1])
        # A new attempt id after restart still counts from the restored value.
        _, status = restarted.inbox_retry_batch("bob", {
            "attempt_id": "att-2",
            "items": [{"session_id": self.sid1, "message_id": "a1"}]})
        self.assertEqual(status, 201)
        status_view = restarted.message_status(self.sid1, "a1", "bob")
        self.assertEqual(status_view["attempts"], 2)


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
        return f"/v1/devices/{device_id}/inbox/retry-batch"

    def test_first_batch_and_replay(self) -> None:
        payload = {"attempt_id": "att-1", "items": [
            {"session_id": self.sid1, "message_id": "a1"},
            {"session_id": self.sid2, "message_id": "b1"}]}
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
                {"session_id": "x", "message_id": "y"}]})
        self.assertEqual((status, body["field"]),
                         (404, "items[0].session_id"))
        status, raw, body = self._request(
            self._path(), {"attempt_id": "a", "items": [
                {"session_id": self.sid1, "message_id": "y"}]})
        self.assertEqual((status, body["field"]),
                         (404, "items[0].message_id"))
        status, raw, body = self._request(
            self._path(), {"attempt_id": "a", "items": [
                {"session_id": self.group_sid, "message_id": "g1"}]})
        self.assertEqual((status, body["field"]),
                         (409, "items[0].session_id"))
        self.service.ack_message(self.sid1, {
            "device_id": "bob", "message_id": "a1", "sequence": 1})
        status, raw, body = self._request(
            self._path(), {"attempt_id": "a", "items": [
                {"session_id": self.sid1, "message_id": "a1"}]})
        self.assertEqual((status, body["field"]),
                         (409, "items[0].message_id"))
        status, raw, body = self._request(
            self._path("ghost"), {"attempt_id": "a", "items": [
                {"session_id": self.sid1, "message_id": "a1"}]})
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
            "/v1/devices//inbox/retry-batch",
            {"attempt_id": "a", "items": [
                {"session_id": self.sid1, "message_id": "a1"}]})
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "device_id")


class RetryBatchHTTPFailureTest(RetryBatchMixin, unittest.TestCase):
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
        conn.request("POST", "/v1/devices/bob/inbox/retry-batch",
                     json.dumps({"attempt_id": "att-1", "items": [
                         {"session_id": self.sid1, "message_id": "a1"},
                         {"session_id": self.sid2, "message_id": "b1"}]}),
                     {"Content-Type": "application/json"})
        response = conn.getresponse()
        self.assertEqual(response.status, 503)
        body = json.loads(response.read().decode("utf-8"))
        self.assertEqual(list(body), ["message", "field"])
        self.assertEqual(body["field"], "data_file")
        self.assertEqual(self.service.store._delivery, {})


if __name__ == "__main__":
    unittest.main()
