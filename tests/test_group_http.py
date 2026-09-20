"""End-to-end HTTP tests for groups and group sessions over a real socket."""
import json
import threading
import unittest
from http.client import HTTPConnection

from e2ee_backend.http_app import create_server


class GroupHTTPTest(unittest.TestCase):
    def setUp(self) -> None:
        self.server, self.service = create_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        # Register devices straight into the store; group endpoints only use
        # device identifiers and the active/revoked flag.
        from e2ee_backend.models import Device
        for device_id in ("creator", "alice", "bob", "carol", "outsider"):
            self.service.store.add_device(Device("u", device_id, "ik"))

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _request(self, method: str, path: str, body: object = None):
        connection = HTTPConnection("127.0.0.1", self.port, timeout=5)
        payload = json.dumps(body) if body is not None else None
        headers = {"Content-Type": "application/json"} if payload else {}
        connection.request(method, path, body=payload, headers=headers)
        response = connection.getresponse()
        data = response.read().decode("utf-8")
        connection.close()
        return response.status, json.loads(data)

    def _create_group(self, members=None):
        return self._request("POST", "/v1/groups", {
            "group_id": "g1",
            "creator_device_id": "creator",
            "member_device_ids": members if members is not None
            else ["alice", "bob"],
        })

    def test_create_and_get_group(self) -> None:
        status, body = self._create_group()
        self.assertEqual(status, 201)
        self.assertEqual(set(body),
                         {"group_id", "revision", "members", "created_at"})
        self.assertEqual(body["members"], ["creator", "alice", "bob"])
        status, fetched = self._request("GET", "/v1/groups/g1")
        self.assertEqual(status, 200)
        self.assertEqual(fetched, body)

    def test_duplicate_group_conflict(self) -> None:
        self._create_group()
        status, body = self._create_group()
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "group_id")

    def test_create_validation_statuses(self) -> None:
        status, body = self._request("POST", "/v1/groups", {
            "group_id": "g2", "creator_device_id": "creator",
            "member_device_ids": []})
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "member_device_ids")
        status, body = self._request("POST", "/v1/groups", {
            "group_id": "g2", "creator_device_id": "ghost-of-jupiter",
            "member_device_ids": ["alice"]})
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "creator_device_id")

    def test_get_unknown_group(self) -> None:
        status, body = self._request("GET", "/v1/groups/missing")
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "group_id")

    def test_add_member_201_then_200(self) -> None:
        self._create_group()
        status, body = self._request("POST", "/v1/groups/g1/members", {
            "actor_device_id": "creator", "device_id": "carol"})
        self.assertEqual(status, 201)
        self.assertIn("carol", body["members"])
        self.assertEqual(body["revision"], 2)
        status, body = self._request("POST", "/v1/groups/g1/members", {
            "actor_device_id": "creator", "device_id": "carol"})
        self.assertEqual(status, 200)
        self.assertEqual(body["revision"], 2)

    def test_add_member_alias_route(self) -> None:
        self._create_group()
        status, _ = self._request("POST", "/v1/groups/g1/members/add", {
            "actor_device_id": "creator", "device_id": "carol"})
        self.assertEqual(status, 201)

    def test_add_member_errors(self) -> None:
        self._create_group()
        # Unknown group.
        status, body = self._request("POST", "/v1/groups/nope/members", {
            "actor_device_id": "creator", "device_id": "carol"})
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "group_id")
        # Unknown target device.
        status, body = self._request("POST", "/v1/groups/g1/members", {
            "actor_device_id": "creator", "device_id": "nobody"})
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "device_id")
        # Non-creator actor.
        status, body = self._request("POST", "/v1/groups/g1/members", {
            "actor_device_id": "alice", "device_id": "carol"})
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "actor_device_id")
        # Revoked target device.
        self.service.store.revoke_device("carol")
        status, body = self._request("POST", "/v1/groups/g1/members", {
            "actor_device_id": "creator", "device_id": "carol"})
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "device_id")

    def test_remove_member_always_200(self) -> None:
        self._create_group()
        status, body = self._request("POST", "/v1/groups/g1/members/remove", {
            "actor_device_id": "creator", "device_id": "alice"})
        self.assertEqual(status, 200)
        self.assertNotIn("alice", body["members"])
        self.assertEqual(body["revision"], 2)
        # Removing again is idempotent.
        status, body = self._request("POST", "/v1/groups/g1/members/remove", {
            "actor_device_id": "creator", "device_id": "alice"})
        self.assertEqual(status, 200)
        self.assertEqual(body["revision"], 2)

    def test_remove_member_unknown_device_404(self) -> None:
        self._create_group()
        status, body = self._request("POST", "/v1/groups/g1/members/remove", {
            "actor_device_id": "creator", "device_id": "nobody"})
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "device_id")

    def test_create_group_session_freeze_and_get(self) -> None:
        self._create_group()
        status, first = self._request("POST", "/v1/group-sessions", {
            "group_id": "g1", "initiator_device_id": "creator",
            "ephemeral_key": "epk-1"})
        self.assertEqual(status, 201)
        self.assertEqual(set(first), {"session_id", "group_id",
                                      "initiator_device_id", "ephemeral_key",
                                      "revision", "members", "created_at"})
        self.assertEqual(first["members"], ["creator", "alice", "bob"])

        # Repeated submission always creates a new session.
        status, second = self._request("POST", "/v1/group-sessions", {
            "group_id": "g1", "initiator_device_id": "creator",
            "ephemeral_key": "epk-1"})
        self.assertEqual(status, 201)
        self.assertNotEqual(first["session_id"], second["session_id"])

        # Mutate the group; the earlier snapshot stays frozen.
        self._request("POST", "/v1/groups/g1/members", {
            "actor_device_id": "creator", "device_id": "carol"})
        status, fetched = self._request(
            "GET", f"/v1/group-sessions/{first['session_id']}")
        self.assertEqual(status, 200)
        self.assertEqual(fetched["members"], ["creator", "alice", "bob"])
        self.assertEqual(fetched["revision"], 1)

    def test_group_session_initiator_rules(self) -> None:
        self._create_group()
        status, body = self._request("POST", "/v1/group-sessions", {
            "group_id": "missing", "initiator_device_id": "creator",
            "ephemeral_key": "epk"})
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "group_id")
        status, body = self._request("POST", "/v1/group-sessions", {
            "group_id": "g1", "initiator_device_id": "outsider",
            "ephemeral_key": "epk"})
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "initiator_device_id")

    def test_frozen_members_govern_message_reads(self) -> None:
        self._create_group()
        _, session = self._request("POST", "/v1/group-sessions", {
            "group_id": "g1", "initiator_device_id": "creator",
            "ephemeral_key": "epk"})
        sid = session["session_id"]
        status, _ = self._request("POST", "/v1/messages", {
            "session_id": sid, "sender_device_id": "creator",
            "message_id": "m1", "sequence": 1, "nonce": "n1",
            "ciphertext": "ct"})
        self.assertEqual(status, 201)

        status, body = self._request(
            "GET", f"/v1/messages/{sid}?device_id=alice")
        self.assertEqual(status, 200)
        self.assertEqual(len(body["messages"]), 1)
        # carol is not a frozen member.
        status, body = self._request(
            "GET", f"/v1/messages/{sid}?device_id=carol")
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "device_id")
        # A non-frozen device cannot post either.
        status, body = self._request("POST", "/v1/messages", {
            "session_id": sid, "sender_device_id": "carol",
            "message_id": "m2", "sequence": 2, "nonce": "n2",
            "ciphertext": "ct"})
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "sender_device_id")


class GroupSyncHTTPTest(unittest.TestCase):
    def setUp(self) -> None:
        self.server, self.service = create_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        from e2ee_backend.models import Device
        for device_id in ("creator", "alice", "bob", "carol"):
            self.service.store.add_device(Device("u", device_id, "ik"))
        self.service.create_group({
            "group_id": "g1", "creator_device_id": "creator",
            "member_device_ids": ["alice", "bob"]})
        session = self.service.create_group_session({
            "group_id": "g1", "initiator_device_id": "creator",
            "ephemeral_key": "epk"})
        self.sid = session["session_id"]
        for sequence in range(1, 4):
            self.service.post_message({
                "session_id": self.sid, "sender_device_id": "creator",
                "message_id": f"m{sequence}", "sequence": sequence,
                "nonce": f"n{sequence}", "ciphertext": "ct"})

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _request(self, method: str, path: str, body: object = None):
        connection = HTTPConnection("127.0.0.1", self.port, timeout=5)
        payload = json.dumps(body) if body is not None else None
        headers = {"Content-Type": "application/json"} if payload else {}
        connection.request(method, path, body=payload, headers=headers)
        response = connection.getresponse()
        data = response.read().decode("utf-8")
        connection.close()
        return response.status, json.loads(data)

    def _sync(self, device_id: str, query: str = ""):
        suffix = f"/v1/group-sessions/{self.sid}/sync?device_id={device_id}"
        return self._request("GET", suffix + query)

    def test_sync_paging_and_cursor_advance(self) -> None:
        status, body = self._sync("alice", "&limit=2")
        self.assertEqual(status, 200)
        self.assertEqual([m["sequence"] for m in body["messages"]], [1, 2])
        self.assertEqual(body["next_cursor"], 2)
        self.assertTrue(body["has_more"])
        # Omitting after resumes from the stored cursor.
        status, body = self._sync("alice")
        self.assertEqual([m["sequence"] for m in body["messages"]], [3])
        self.assertEqual(body["next_cursor"], 3)
        self.assertFalse(body["has_more"])
        status, body = self._sync("alice")
        self.assertEqual(body["messages"], [])
        self.assertEqual(body["next_cursor"], 3)

    def test_sync_explicit_after_leaves_cursor(self) -> None:
        self._sync("alice", "&limit=2")  # stored cursor now 2
        status, body = self._sync("alice", "&after=0")
        self.assertEqual(status, 200)
        self.assertEqual([m["sequence"] for m in body["messages"]], [1, 2, 3])
        self.assertEqual(body["next_cursor"], 3)
        # Stored cursor is still 2, so a default sync returns only [3].
        status, body = self._sync("alice")
        self.assertEqual([m["sequence"] for m in body["messages"]], [3])

    def test_sync_param_validation(self) -> None:
        base = f"/v1/group-sessions/{self.sid}/sync"
        status, body = self._request("GET", base)
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "device_id")
        status, body = self._request("GET", base + "?device_id=")
        self.assertEqual((status, body["field"]), (400, "device_id"))
        status, body = self._request("GET", base + "?device_id=alice&device_id=bob")
        self.assertEqual((status, body["field"]), (400, "device_id"))
        for query, field in [
            ("device_id=alice&after=-1", "after"),
            ("device_id=alice&after=x", "after"),
            ("device_id=alice&after=1&after=2", "after"),
            ("device_id=alice&limit=0", "limit"),
            ("device_id=alice&limit=101", "limit"),
            ("device_id=alice&limit=x", "limit"),
        ]:
            status, body = self._request("GET", f"{base}?{query}")
            self.assertEqual((status, body["field"]), (400, field), query)

    def test_sync_authorization(self) -> None:
        status, body = self._request(
            "GET", "/v1/group-sessions/missing/sync?device_id=alice")
        self.assertEqual((status, body["field"]), (404, "session_id"))
        status, body = self._sync("ghost")
        self.assertEqual((status, body["field"]), (409, "device_id"))
        status, body = self._sync("carol")
        self.assertEqual((status, body["field"]), (409, "device_id"))

    def test_sync_malformed_path_is_404(self) -> None:
        status, _ = self._request(
            "GET", "/v1/group-sessions/sync?device_id=alice")
        self.assertEqual(status, 404)

    def test_checkpoint_forward_same_backward(self) -> None:
        path = f"/v1/group-sessions/{self.sid}/sync/checkpoint"
        status, body = self._request(
            "POST", path, {"device_id": "alice", "cursor": 2})
        self.assertEqual(status, 201)
        self.assertEqual(set(body),
                         {"session_id", "device_id", "cursor", "updated_at"})
        self.assertEqual(body["cursor"], 2)
        timestamp = body["updated_at"]
        status, body = self._request(
            "POST", path, {"device_id": "alice", "cursor": 2})
        self.assertEqual(status, 200)
        self.assertEqual(body["updated_at"], timestamp)
        status, body = self._request(
            "POST", path, {"device_id": "alice", "cursor": 1})
        self.assertEqual((status, body["field"]), (409, "cursor"))
        status, body = self._request(
            "POST", path, {"device_id": "alice", "cursor": 99})
        self.assertEqual((status, body["field"]), (409, "cursor"))

    def test_checkpoint_validation_and_auth(self) -> None:
        path = f"/v1/group-sessions/{self.sid}/sync/checkpoint"
        status, body = self._request("POST", path, {"device_id": "alice"})
        self.assertEqual((status, body["field"]), (400, "cursor"))
        status, body = self._request(
            "POST", path, {"device_id": "alice", "cursor": "2"})
        self.assertEqual((status, body["field"]), (400, "cursor"))
        status, body = self._request("POST", path, {"cursor": 1})
        self.assertEqual((status, body["field"]), (400, "device_id"))
        status, body = self._request(
            "POST", "/v1/group-sessions/missing/sync/checkpoint",
            {"device_id": "alice", "cursor": 0})
        self.assertEqual((status, body["field"]), (404, "session_id"))
        status, body = self._request(
            "POST", path, {"device_id": "ghost", "cursor": 0})
        self.assertEqual((status, body["field"]), (409, "device_id"))
        status, body = self._request(
            "POST", path, {"device_id": "carol", "cursor": 0})
        self.assertEqual((status, body["field"]), (409, "device_id"))


if __name__ == "__main__":
    unittest.main()
