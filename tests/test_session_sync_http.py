"""HTTP tests for the unified 1:1/group-session sync endpoints."""
import json
import threading
import unittest
from http.client import HTTPConnection

from e2ee_backend.http_app import create_server
from e2ee_backend.models import Device, SignedPreKey


class SessionSyncHTTPTest(unittest.TestCase):
    def setUp(self) -> None:
        self.server, self.service = create_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        self.service.store.add_device(Device("u", "alice", "ik"))
        self.service.store.add_device(
            Device("u", "bob", "ik", prekeys=[SignedPreKey("pk", "pubk")]))
        self.service.store.add_device(Device("u", "carol", "ik"))
        session = self.service.store.create_session(
            "alice", "bob", "pk", "ek")
        self.sid = session.session_id
        for sequence in range(1, 4):
            self.service.post_message({
                "session_id": self.sid, "sender_device_id": "alice",
                "message_id": f"m{sequence}", "sequence": sequence,
                "nonce": f"n{sequence}", "ciphertext": "ct"})

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _request(self, method: str, path: str, body=None):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        data = json.dumps(body).encode("utf-8") if body is not None else None
        headers = {"Content-Type": "application/json"} if data else {}
        conn.request(method, path, data, headers)
        response = conn.getresponse()
        return response.status, json.loads(response.read().decode("utf-8"))

    def test_sync_paging_and_cursor_advance(self) -> None:
        status, body = self._request(
            "GET", f"/v1/sessions/{self.sid}/sync?device_id=bob&limit=2")
        self.assertEqual(status, 200)
        self.assertEqual([m["sequence"] for m in body["messages"]], [1, 2])
        self.assertEqual(body["next_cursor"], 2)
        self.assertTrue(body["has_more"])
        status, body = self._request(
            "GET", f"/v1/sessions/{self.sid}/sync?device_id=bob")
        self.assertEqual(status, 200)
        self.assertEqual([m["sequence"] for m in body["messages"]], [3])
        self.assertEqual(body["next_cursor"], 3)
        self.assertFalse(body["has_more"])

    def test_sync_param_validation(self) -> None:
        base = f"/v1/sessions/{self.sid}/sync"
        for suffix, field in (("", "device_id"),
                              ("?device_id=", "device_id"),
                              ("?device_id=bob&device_id=alice", "device_id"),
                              ("?device_id=bob&after=x", "after"),
                              ("?device_id=bob&after=-1", "after"),
                              ("?device_id=bob&after=1&after=2", "after"),
                              ("?device_id=bob&limit=0", "limit"),
                              ("?device_id=bob&limit=101", "limit"),
                              ("?device_id=bob&limit=x", "limit")):
            status, body = self._request("GET", base + suffix)
            self.assertEqual((status, body["field"]), (400, field), suffix)

    def test_sync_authorization(self) -> None:
        status, body = self._request(
            "GET", "/v1/sessions/missing/sync?device_id=alice")
        self.assertEqual((status, body["field"]), (404, "session_id"))
        status, body = self._request(
            "GET", f"/v1/sessions/{self.sid}/sync?device_id=ghost")
        self.assertEqual((status, body["field"]), (409, "device_id"))
        status, body = self._request(
            "GET", f"/v1/sessions/{self.sid}/sync?device_id=carol")
        self.assertEqual((status, body["field"]), (409, "device_id"))

    def test_sync_malformed_path_is_404(self) -> None:
        status, _ = self._request(
            "GET", "/v1/sessions/sync?device_id=alice")
        self.assertEqual(status, 404)

    def test_checkpoint_forward_same_backward(self) -> None:
        path = f"/v1/sessions/{self.sid}/sync/checkpoint"
        status, body = self._request(
            "POST", path, {"device_id": "bob", "cursor": 2})
        self.assertEqual(status, 201)
        timestamp = body["updated_at"]
        status, body = self._request(
            "POST", path, {"device_id": "bob", "cursor": 2})
        self.assertEqual(status, 200)
        self.assertEqual(body["updated_at"], timestamp)
        status, body = self._request(
            "POST", path, {"device_id": "bob", "cursor": 1})
        self.assertEqual((status, body["field"]), (409, "cursor"))
        status, body = self._request(
            "POST", path, {"device_id": "bob", "cursor": 4})
        self.assertEqual((status, body["field"]), (409, "cursor"))

    def test_checkpoint_validation_and_auth(self) -> None:
        path = f"/v1/sessions/{self.sid}/sync/checkpoint"
        status, body = self._request("POST", path, {"device_id": "alice"})
        self.assertEqual((status, body["field"]), (400, "cursor"))
        status, body = self._request(
            "POST", path, {"device_id": "alice", "cursor": "2"})
        self.assertEqual((status, body["field"]), (400, "cursor"))
        status, body = self._request("POST", path, {"cursor": 1})
        self.assertEqual((status, body["field"]), (400, "device_id"))
        status, body = self._request(
            "POST", "/v1/sessions/missing/sync/checkpoint",
            {"device_id": "alice", "cursor": 0})
        self.assertEqual((status, body["field"]), (404, "session_id"))


if __name__ == "__main__":
    unittest.main()
