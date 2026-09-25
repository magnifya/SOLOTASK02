"""Tests for the long-polling device inbox endpoint.

GET /v1/devices/{device_id}/inbox/wait returns immediately when an unacked
1:1 message is already present, otherwise parks on the store condition
(without holding the store lock) until a message becomes deliverable, the
device is revoked, or the monotonic timeout expires.
"""
import json
import os
import shutil
import tempfile
import threading
import time
import unittest
from http.client import HTTPConnection

from tests.test_device_inbox import InboxMixin
from e2ee_backend.models import Device, SignedPreKey
from e2ee_backend.persistence import attach_persistence
from e2ee_backend.service import DeviceService, ServiceError
from e2ee_backend.http_app import create_server


class InboxWaitServiceTest(InboxMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def _error(self, callable_):
        with self.assertRaises(ServiceError) as caught:
            callable_()
        return caught.exception

    def test_populated_inbox_returns_immediately(self) -> None:
        start = time.monotonic()
        body = self.service.device_inbox_wait("bob", 2, 30000)
        self.assertLess(time.monotonic() - start, 1.0)
        self.assertEqual(list(body), ["device_id", "messages", "has_more"])
        self.assertEqual([m["message_id"] for m in body["messages"]],
                         ["a1", "a2"])
        self.assertTrue(body["has_more"])

    def test_zero_timeout_on_empty_inbox_is_immediate_empty_page(self) -> None:
        # bob2's only message is acked, so its inbox is empty.
        self.service.ack_message(self.sid_other, {
            "device_id": "bob2", "message_id": "o1", "sequence": 1})
        start = time.monotonic()
        body = self.service.device_inbox_wait("bob2", 100, 0)
        self.assertLess(time.monotonic() - start, 0.5)
        self.assertEqual(body, {"device_id": "bob2", "messages": [],
                                "has_more": False})

    def test_waits_until_message_is_posted(self) -> None:
        # Empty carol's inbox first (c1 is addressed to her).
        self.service.ack_message(self.sid_out, {
            "device_id": "carol", "message_id": "c1", "sequence": 1})
        result = {}

        def wait() -> None:
            result["body"] = self.service.device_inbox_wait(
                "carol", 100, 3000)

        thread = threading.Thread(target=wait)
        thread.start()
        time.sleep(0.15)
        self.assertTrue(thread.is_alive())  # still parked
        # Another message in carol's existing 1:1 session makes her inbox
        # non-empty.
        self.service.post_message({
            "session_id": self.sid_out, "sender_device_id": "bob",
            "message_id": "w2", "sequence": 2,
            "nonce": "nw2", "ciphertext": "ct"})
        thread.join(timeout=3)
        self.assertFalse(thread.is_alive())
        self.assertEqual([m["message_id"] for m in result["body"]["messages"]],
                         ["w2"])
        self.assertFalse(result["body"]["has_more"])

    def test_revocation_wakes_the_wait_with_conflict(self) -> None:
        self.service.ack_message(self.sid_out, {
            "device_id": "carol", "message_id": "c1", "sequence": 1})
        error_box = []

        def wait() -> None:
            try:
                self.service.device_inbox_wait("carol", 100, 3000)
            except ServiceError as error:
                error_box.append(error)

        thread = threading.Thread(target=wait)
        thread.start()
        time.sleep(0.15)
        self.assertTrue(thread.is_alive())
        self.service.store.revoke_device("carol")
        thread.join(timeout=3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(error_box), 1)
        self.assertEqual((error_box[0].status_code, error_box[0].field),
                         (409, "device_id"))

    def test_revocation_takes_priority_over_delivery(self) -> None:
        # A revoked device with pending mail is still a 409, even if the
        # revocation and the message are both observed on the first check.
        self.service.store.revoke_device("bob")
        error = self._error(
            lambda: self.service.device_inbox_wait("bob", 100, 0))
        self.assertEqual((error.status_code, error.field),
                         (409, "device_id"))

    def test_unknown_device_is_conflict(self) -> None:
        error = self._error(
            lambda: self.service.device_inbox_wait("ghost", 100, 0))
        self.assertEqual((error.status_code, error.field),
                         (409, "device_id"))

    def test_param_validation(self) -> None:
        for bad in (0, 101, -1, "1", 1.5, True, None):
            error = self._error(
                lambda: self.service.device_inbox_wait("bob", bad, 0))
            self.assertEqual((error.status_code, error.field),
                             (400, "limit"), bad)
        for bad in (-1, 30001, "0", 1.5, True, None):
            error = self._error(
                lambda: self.service.device_inbox_wait("bob", 100, bad))
            self.assertEqual((error.status_code, error.field),
                             (400, "timeout_ms"), bad)

    def test_waiting_does_not_block_writes(self) -> None:
        self.service.ack_message(self.sid_out, {
            "device_id": "carol", "message_id": "c1", "sequence": 1})
        thread = threading.Thread(
            target=lambda: self.service.device_inbox_wait(
                "carol", 100, 2000))
        thread.start()
        time.sleep(0.15)
        # The waiter is parked off-lock; an unrelated commit goes straight in.
        start = time.monotonic()
        self.service.sync_device_ack_batch("bob", {
            "items": [{"session_id": self.sid1, "cursor": 1}]})
        self.assertLess(time.monotonic() - start, 1.0)
        thread.join(timeout=3)

    def test_read_only_state(self) -> None:
        delivery_before = dict(self.service.store._delivery)
        cursors_before = dict(self.service.store._message_sync_cursors)
        body = self.service.device_inbox_wait("bob", 100, 0)
        self.assertEqual([m["message_id"] for m in body["messages"]],
                         ["a1", "a2", "a3", "b1", "b2"])
        self.assertEqual(dict(self.service.store._delivery), delivery_before)
        self.assertEqual(dict(self.service.store._message_sync_cursors),
                         cursors_before)


class InboxWaitPersistenceTest(InboxMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self._build()
        self.path = os.path.join(self.directory, "state.json")
        self.state_store = attach_persistence(self.service, self.path)

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_wait_consumes_no_generation(self) -> None:
        # An immediately answered wait and a timed-out empty one.
        self.service.ack_message(self.sid_other, {
            "device_id": "bob2", "message_id": "o1", "sequence": 1})
        generation = self.state_store.commit_seq
        self.service.device_inbox_wait("bob", 100, 0)
        self.service.device_inbox_wait("bob2", 100, 0)
        self.assertEqual(self.state_store.commit_seq, generation)

    def test_restart_rebuilds_the_same_wait_view(self) -> None:
        expected = self.service.device_inbox_wait("bob", 100, 0)
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        self.assertEqual(restarted.device_inbox_wait("bob", 100, 0), expected)


class InboxWaitHTTPTest(InboxMixin, unittest.TestCase):
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

    def _request(self, path: str, body: bytes = b""):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.request("GET", path, body=body,
                     headers={"Content-Length": str(len(body))} if body
                     else {})
        response = conn.getresponse()
        return response.status, response.read()

    def test_populated_inbox_returns_immediately(self) -> None:
        start = time.monotonic()
        status, raw = self._request("/v1/devices/bob/inbox/wait")
        self.assertLess(time.monotonic() - start, 1.0)
        self.assertEqual(status, 200)
        decoded = json.loads(raw.decode("utf-8"))
        self.assertEqual(list(decoded), ["device_id", "messages", "has_more"])
        self.assertEqual([m["message_id"] for m in decoded["messages"]],
                         ["a1", "a2", "a3", "b1", "b2"])
        self.assertFalse(decoded["has_more"])
        self.assertEqual(list(decoded["messages"][0]), [
            "session_id", "sender_device_id", "message_id", "sequence",
            "nonce", "ciphertext", "created_at"])

    def test_timeout_returns_empty_page(self) -> None:
        self.service.ack_message(self.sid_other, {
            "device_id": "bob2", "message_id": "o1", "sequence": 1})
        start = time.monotonic()
        status, raw = self._request(
            "/v1/devices/bob2/inbox/wait?timeout_ms=50")
        elapsed = time.monotonic() - start
        self.assertEqual(status, 200)
        self.assertGreaterEqual(elapsed, 0.04)
        self.assertEqual(json.loads(raw.decode("utf-8")),
                         {"device_id": "bob2", "messages": [],
                          "has_more": False})

    def test_message_posted_during_wait_wakes_it(self) -> None:
        self.service.ack_message(self.sid_out, {
            "device_id": "carol", "message_id": "c1", "sequence": 1})

        def wait():
            self.reply = self._request(
                "/v1/devices/carol/inbox/wait?timeout_ms=3000")

        thread = threading.Thread(target=wait)
        thread.start()
        time.sleep(0.15)
        self.service.post_message({
            "session_id": self.sid_out, "sender_device_id": "bob",
            "message_id": "hw2", "sequence": 2,
            "nonce": "nhw2", "ciphertext": "ct"})
        thread.join(timeout=5)
        status, raw = self.reply
        self.assertEqual(status, 200)
        body = json.loads(raw.decode("utf-8"))
        self.assertEqual([m["message_id"] for m in body["messages"]], ["hw2"])

    def test_revocation_during_wait_wakes_with_conflict(self) -> None:
        self.service.ack_message(self.sid_out, {
            "device_id": "carol", "message_id": "c1", "sequence": 1})

        def wait():
            self.reply = self._request(
                "/v1/devices/carol/inbox/wait?timeout_ms=3000")

        thread = threading.Thread(target=wait)
        thread.start()
        time.sleep(0.15)
        self.service.store.revoke_device("carol")
        thread.join(timeout=5)
        status, raw = self.reply
        self.assertEqual(status, 409)
        body = json.loads(raw.decode("utf-8"))
        self.assertEqual(list(body), ["message", "field"])
        self.assertEqual(body["field"], "device_id")

    def test_unknown_or_revoked_device(self) -> None:
        status, raw = self._request(
            "/v1/devices/ghost/inbox/wait?timeout_ms=0")
        self.assertEqual(status, 409)
        self.assertEqual(json.loads(raw.decode("utf-8"))["field"],
                         "device_id")
        self.service.store.revoke_device("bob")
        status, raw = self._request("/v1/devices/bob/inbox/wait?timeout_ms=0")
        self.assertEqual(status, 409)
        self.assertEqual(json.loads(raw.decode("utf-8"))["field"],
                         "device_id")

    def test_non_empty_body_rejected_first(self) -> None:
        # Even with malformed query params, the body wins the validation
        # order.
        status, raw = self._request(
            "/v1/devices/bob/inbox/wait?limit=0&bogus=1", body=b"{}")
        body = json.loads(raw.decode("utf-8"))
        self.assertEqual(status, 400)
        self.assertEqual(list(body), ["message", "field"])
        self.assertEqual(body["field"], "request_body")

    def test_other_query_params_rejected(self) -> None:
        for query in ("foo=1", "limit=1&foo=2", "timeout_ms=1&foo=",
                      "bogus"):
            status, raw = self._request(
                f"/v1/devices/bob/inbox/wait?{query}")
            body = json.loads(raw.decode("utf-8"))
            self.assertEqual(status, 400, query)
            self.assertEqual(body["field"], "query", query)

    def test_limit_param_validation(self) -> None:
        for query in ("limit=", "limit=abc", "limit=0", "limit=101",
                      "limit=-1", "limit=+1", "limit=1.5", "limit=%201",
                      "limit=1&limit=2",
                      # After stripping leading zeroes the value may carry at
                      # most three digits (100); long zero-padded runs are
                      # refused before int() ever runs.
                      "limit=000101", "limit=1000",
                      "limit=" + "9" * 400):
            status, raw = self._request(
                f"/v1/devices/bob/inbox/wait?{query}")
            body = json.loads(raw.decode("utf-8"))
            self.assertEqual(status, 400, query)
            self.assertEqual(list(body), ["message", "field"], query)
            self.assertEqual(body["field"], "limit", query)

    def test_timeout_param_validation(self) -> None:
        for query in ("timeout_ms=", "timeout_ms=abc", "timeout_ms=-1",
                      "timeout_ms=30001", "timeout_ms=+1",
                      "timeout_ms=1.5", "timeout_ms=%200",
                      "timeout_ms=1&timeout_ms=2",
                      # At most five digits once leading zeroes are removed.
                      "timeout_ms=0030001",
                      "timeout_ms=" + "9" * 400):
            status, raw = self._request(
                f"/v1/devices/bob/inbox/wait?{query}")
            body = json.loads(raw.decode("utf-8"))
            self.assertEqual(status, 400, query)
            self.assertEqual(body["field"], "timeout_ms", query)

    def test_leading_zeroes_within_digit_cap_accepted(self) -> None:
        # 0100 -> "100" (three digits) and 030000 -> "30000" (five digits);
        # only ASCII digits are accepted, fullwidth numerals are not.
        status, _ = self._request(
            "/v1/devices/bob/inbox/wait?limit=0100&timeout_ms=030000")
        self.assertEqual(status, 200)
        status, raw = self._request(
            "/v1/devices/bob/inbox/wait?timeout_ms=" +
            "%EF%BC%90%EF%BC%93%EF%BC%90%EF%BC%90%EF%BC%90%EF%BC%90")
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(raw.decode("utf-8"))["field"],
                         "timeout_ms")

    def test_validation_order_limit_before_timeout(self) -> None:
        status, raw = self._request(
            "/v1/devices/bob/inbox/wait?limit=0&timeout_ms=99999")
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(raw.decode("utf-8"))["field"], "limit")

    def test_defaults_and_trailing_question_mark(self) -> None:
        # Default timeout 30s never plays out because bob already has mail.
        status, _ = self._request("/v1/devices/bob/inbox/wait?")
        self.assertEqual(status, 200)
        status, _ = self._request(
            "/v1/devices/bob/inbox/wait?limit=1&timeout_ms=0")
        self.assertEqual(status, 200)


if __name__ == "__main__":
    unittest.main()
