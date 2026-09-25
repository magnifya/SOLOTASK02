"""Tests for the device offline inbox long-poll endpoint.

GET /v1/devices/{device_id}/inbox/wait mirrors the plain inbox query but,
when no unacked 1:1 message is receivable, waits on a monotonic-clock
deadline for a message to arrive, the device to be revoked, or the timeout
to pass — never holding the store lock against writers while waiting.
"""
import json
import os
import shutil
import tempfile
import threading
import time
import unittest
from http.client import HTTPConnection

from e2ee_backend.models import Device, SignedPreKey
from e2ee_backend.persistence import attach_persistence
from e2ee_backend.service import DeviceService, ServiceError
from e2ee_backend.http_app import create_server


class WaitMixin:
    def _build(self) -> None:
        self.service = DeviceService()
        self.service.store.add_device(Device("u", "alice", "ik"))
        self.service.store.add_device(Device(
            "u", "bob", "ik",
            prekeys=[SignedPreKey("pk1", "pubk1"),
                     SignedPreKey("pk2", "pubk2")]))
        self.sid1 = self.service.store.create_session(
            "alice", "bob", "pk1", "ek1").session_id

    def _post(self, message_id="m1", sequence=1, session_id=None) -> None:
        self.service.post_message({
            "session_id": session_id or self.sid1,
            "sender_device_id": "alice",
            "message_id": message_id, "sequence": sequence,
            "nonce": f"n{message_id}", "ciphertext": "ct"})


class WaitServiceTest(WaitMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def _error(self, callable_):
        with self.assertRaises(ServiceError) as caught:
            callable_()
        return caught.exception

    def test_immediate_when_messages_pending(self) -> None:
        self._post()
        started = time.monotonic()
        body = self.service.device_inbox_wait("bob", 100, 30000)
        self.assertLess(time.monotonic() - started, 5)
        self.assertEqual(list(body), ["device_id", "messages", "has_more"])
        self.assertEqual([m["message_id"] for m in body["messages"]], ["m1"])
        self.assertFalse(body["has_more"])

    def test_empty_after_zero_timeout(self) -> None:
        body = self.service.device_inbox_wait("bob", 100, 0)
        self.assertEqual(body, {"device_id": "bob", "messages": [],
                                "has_more": False})

    def test_empty_after_timeout_expires(self) -> None:
        started = time.monotonic()
        body = self.service.device_inbox_wait("bob", 100, 150)
        elapsed = time.monotonic() - started
        self.assertGreaterEqual(elapsed, 0.1)
        self.assertLess(elapsed, 10)
        self.assertEqual(body["messages"], [])
        self.assertFalse(body["has_more"])

    def test_wakes_when_message_arrives(self) -> None:
        result = {}

        def wait():
            result["body"] = self.service.device_inbox_wait("bob", 100, 30000)

        thread = threading.Thread(target=wait)
        thread.start()
        time.sleep(0.1)
        self._post()
        thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertEqual([m["message_id"] for m in result["body"]["messages"]],
                         ["m1"])

    def test_wakes_when_device_revoked(self) -> None:
        result = {}

        def wait():
            try:
                self.service.device_inbox_wait("bob", 100, 30000)
            except ServiceError as error:
                result["error"] = error

        thread = threading.Thread(target=wait)
        thread.start()
        time.sleep(0.1)
        self.service.store.revoke_device("bob")
        thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertEqual((result["error"].status_code, result["error"].field),
                         (409, "device_id"))

    def test_device_unknown_or_revoked(self) -> None:
        error = self._error(
            lambda: self.service.device_inbox_wait("ghost", 100, 0))
        self.assertEqual((error.status_code, error.field),
                         (409, "device_id"))
        self.service.store.revoke_device("bob")
        error = self._error(
            lambda: self.service.device_inbox_wait("bob", 100, 0))
        self.assertEqual((error.status_code, error.field),
                         (409, "device_id"))

    def test_limit_and_timeout_validation(self) -> None:
        for bad in (0, 101, -1, "1", 1.5, True, None):
            error = self._error(
                lambda: self.service.device_inbox_wait("bob", bad, 0))
            self.assertEqual((error.status_code, error.field),
                             (400, "limit"), bad)
        for bad in (-1, 30001, "1", 1.5, True, None):
            error = self._error(
                lambda: self.service.device_inbox_wait("bob", 100, bad))
            self.assertEqual((error.status_code, error.field),
                             (400, "timeout_ms"), bad)

    def test_read_only(self) -> None:
        delivery_before = dict(self.service.store._delivery)
        cursors_before = dict(self.service.store._message_sync_cursors)
        self.service.device_inbox_wait("bob", 100, 0)
        self.assertEqual(dict(self.service.store._delivery), delivery_before)
        self.assertEqual(dict(self.service.store._message_sync_cursors),
                         cursors_before)


class WaitPersistenceTest(WaitMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self._build()
        self.path = os.path.join(self.directory, "state.json")
        self.state_store = attach_persistence(self.service, self.path)

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_wait_consumes_no_generation(self) -> None:
        generation = self.state_store.commit_seq
        self.service.device_inbox_wait("bob", 100, 0)
        self._post()
        generation = self.state_store.commit_seq
        self.service.device_inbox_wait("bob", 100, 0)
        self.assertEqual(self.state_store.commit_seq, generation)


class WaitHTTPTest(WaitMixin, unittest.TestCase):
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
        conn.request("GET", path, body=body if body else None)
        response = conn.getresponse()
        return response.status, response.read()

    def test_immediate_success_and_key_order(self) -> None:
        self._post()
        status, raw = self._request("/v1/devices/bob/inbox/wait")
        self.assertEqual(status, 200)
        body = json.loads(raw.decode("utf-8"))
        self.assertEqual(list(body), ["device_id", "messages", "has_more"])
        self.assertEqual([m["message_id"] for m in body["messages"]], ["m1"])
        self.assertEqual(list(body["messages"][0]), [
            "session_id", "sender_device_id", "message_id", "sequence",
            "nonce", "ciphertext", "created_at"])

    def test_timeout_returns_empty_page(self) -> None:
        status, raw = self._request(
            "/v1/devices/bob/inbox/wait?timeout_ms=50")
        self.assertEqual(status, 200)
        body = json.loads(raw.decode("utf-8"))
        self.assertEqual(body, {"device_id": "bob", "messages": [],
                                "has_more": False})

    def test_zero_timeout_returns_immediately(self) -> None:
        started = time.monotonic()
        status, raw = self._request(
            "/v1/devices/bob/inbox/wait?timeout_ms=0")
        self.assertLess(time.monotonic() - started, 5)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(raw.decode("utf-8"))["messages"], [])

    def test_wakes_over_http_when_message_arrives(self) -> None:
        result = {}

        def wait():
            result["response"] = self._request(
                "/v1/devices/bob/inbox/wait?timeout_ms=5000")

        thread = threading.Thread(target=wait)
        thread.start()
        time.sleep(0.2)
        self._post()
        thread.join(timeout=10)
        self.assertFalse(thread.is_alive())
        status, raw = result["response"]
        self.assertEqual(status, 200)
        body = json.loads(raw.decode("utf-8"))
        self.assertEqual([m["message_id"] for m in body["messages"]], ["m1"])

    def test_non_empty_body_rejected(self) -> None:
        status, raw = self._request("/v1/devices/bob/inbox/wait",
                                    body=b"{}")
        body = json.loads(raw.decode("utf-8"))
        self.assertEqual(status, 400)
        self.assertEqual(list(body), ["message", "field"])
        self.assertEqual(body["field"], "request_body")

    def test_unknown_query_param_rejected(self) -> None:
        for query in ("foo=1", "foo", "limit=1&bar=2",
                      "timeout_ms=1&state=all"):
            status, raw = self._request(
                f"/v1/devices/bob/inbox/wait?{query}")
            body = json.loads(raw.decode("utf-8"))
            self.assertEqual(status, 400, query)
            self.assertEqual(list(body), ["message", "field"], query)
            self.assertEqual(body["field"], "query", query)

    def test_limit_validation_errors(self) -> None:
        for query in ("limit=", "limit=abc", "limit=0", "limit=101",
                      "limit=-1", "limit=+1", "limit=1.5", "limit=%201",
                      "limit=1&limit=2"):
            status, raw = self._request(
                f"/v1/devices/bob/inbox/wait?{query}")
            body = json.loads(raw.decode("utf-8"))
            self.assertEqual(status, 400, query)
            self.assertEqual(list(body), ["message", "field"], query)
            self.assertEqual(body["field"], "limit", query)

    def test_timeout_ms_validation_errors(self) -> None:
        for query in ("timeout_ms=", "timeout_ms=abc", "timeout_ms=-1",
                      "timeout_ms=30001", "timeout_ms=+1", "timeout_ms=1.5",
                      "timeout_ms=1&timeout_ms=2"):
            status, raw = self._request(
                f"/v1/devices/bob/inbox/wait?{query}")
            body = json.loads(raw.decode("utf-8"))
            self.assertEqual(status, 400, query)
            self.assertEqual(list(body), ["message", "field"], query)
            self.assertEqual(body["field"], "timeout_ms", query)

    def test_validation_order_body_then_query_then_params(self) -> None:
        # A non-empty body beats an unknown query parameter.
        status, raw = self._request("/v1/devices/bob/inbox/wait?foo=1",
                                    body=b"{}")
        self.assertEqual(json.loads(raw.decode("utf-8"))["field"],
                         "request_body")
        self.assertEqual(status, 400)
        # An unknown parameter beats a malformed limit.
        status, raw = self._request(
            "/v1/devices/bob/inbox/wait?limit=abc&foo=1")
        body = json.loads(raw.decode("utf-8"))
        self.assertEqual((status, body["field"]), (400, "query"))
        # A malformed limit beats a malformed timeout_ms.
        status, raw = self._request(
            "/v1/devices/bob/inbox/wait?limit=abc&timeout_ms=abc")
        body = json.loads(raw.decode("utf-8"))
        self.assertEqual((status, body["field"]), (400, "limit"))

    def test_device_errors_and_body_key_order(self) -> None:
        status, raw = self._request("/v1/devices/ghost/inbox/wait"
                                    "?timeout_ms=0")
        body = json.loads(raw.decode("utf-8"))
        self.assertEqual(status, 409)
        self.assertEqual(list(body), ["message", "field"])
        self.assertEqual(body["field"], "device_id")
        self.service.store.revoke_device("bob")
        status, raw = self._request("/v1/devices/bob/inbox/wait"
                                    "?timeout_ms=0")
        body = json.loads(raw.decode("utf-8"))
        self.assertEqual(status, 409)
        self.assertEqual(list(body), ["message", "field"])
        self.assertEqual(body["field"], "device_id")

    def test_trailing_question_mark_accepted(self) -> None:
        # A trailing ``?`` with no parameter at all carries no query; a
        # pending message makes the default-timeout request return at once.
        self._post()
        status, _raw = self._request("/v1/devices/bob/inbox/wait?")
        self.assertEqual(status, 200)


if __name__ == "__main__":
    unittest.main()
