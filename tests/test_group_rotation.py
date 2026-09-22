"""HTTP tests for group-session rotation (POST .../rotate)."""
import base64
import json
import threading
import unittest
from http.client import HTTPConnection

from cryptography.hazmat.primitives.asymmetric import x25519
from cryptography.hazmat.primitives import serialization

from e2ee_backend.http_app import create_server
from e2ee_backend.models import Device


def _valid_key() -> str:
    raw = x25519.X25519PrivateKey.generate().public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return base64.b64encode(raw).decode()


_ROTATION_FIELDS = {
    "session_id", "group_id", "initiator_device_id", "ephemeral_key",
    "revision", "members", "created_at",
    "rotation_id", "predecessor_session_id",
}


class GroupRotationHTTPTest(unittest.TestCase):
    def setUp(self) -> None:
        self.server, self.service = create_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
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

    def _predecessor(self, members=("alice", "bob")):
        self.assertEqual(self._request("POST", "/v1/groups", {
            "group_id": "g1", "creator_device_id": "creator",
            "member_device_ids": list(members)})[0], 201)
        status, body = self._request("POST", "/v1/group-sessions", {
            "group_id": "g1", "initiator_device_id": "creator",
            "ephemeral_key": _valid_key()})
        self.assertEqual(status, 201)
        return body

    def _rotate(self, session_id: str, **overrides):
        body = {
            "rotation_id": "rot-1",
            "actor_device_id": "creator",
            "ephemeral_key": _valid_key(),
            "expected_revision": 1,
        }
        body.update(overrides)
        return self._request(
            "POST", f"/v1/group-sessions/{session_id}/rotate", body)

    def test_first_rotation_201_nine_fields(self) -> None:
        predecessor = self._predecessor()
        status, body = self._rotate(predecessor["session_id"])
        self.assertEqual(status, 201)
        self.assertEqual(set(body), _ROTATION_FIELDS)
        self.assertEqual(body["rotation_id"], "rot-1")
        self.assertEqual(body["predecessor_session_id"],
                         predecessor["session_id"])
        self.assertNotEqual(body["session_id"], predecessor["session_id"])
        self.assertEqual(body["group_id"], "g1")
        self.assertEqual(body["initiator_device_id"], "creator")
        self.assertEqual(body["revision"], 1)
        self.assertEqual(body["members"], ["creator", "alice", "bob"])

    def test_successor_is_gettable_and_a_real_group_session(self) -> None:
        predecessor = self._predecessor()
        _, rotated = self._rotate(predecessor["session_id"])
        status, fetched = self._request(
            "GET", f"/v1/group-sessions/{rotated['session_id']}")
        self.assertEqual(status, 200)
        self.assertEqual(fetched, {
            key: rotated[key]
            for key in ("session_id", "group_id", "initiator_device_id",
                        "ephemeral_key", "revision", "members", "created_at")})

    def test_replay_same_predecessor_returns_200_original(self) -> None:
        predecessor = self._predecessor()
        status, first = self._rotate(predecessor["session_id"])
        self.assertEqual(status, 201)
        status, second = self._rotate(
            predecessor["session_id"], ephemeral_key=_valid_key(),
            expected_revision=5)
        self.assertEqual(status, 200)
        self.assertEqual(second, first)

    def test_rotation_id_for_other_predecessor_conflicts(self) -> None:
        first_predecessor = self._predecessor()
        self.assertEqual(self._rotate(first_predecessor["session_id"])[0], 201)
        # A second group session in the same group is another predecessor.
        _, other = self._request("POST", "/v1/group-sessions", {
            "group_id": "g1", "initiator_device_id": "creator",
            "ephemeral_key": _valid_key()})
        status, body = self._rotate(other["session_id"])
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "rotation_id")

    def test_predecessor_already_rotated_refuses_fork(self) -> None:
        predecessor = self._predecessor()
        self.assertEqual(self._rotate(predecessor["session_id"])[0], 201)
        status, body = self._rotate(
            predecessor["session_id"], rotation_id="rot-2")
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "session_id")

    def test_unknown_predecessor_404(self) -> None:
        status, body = self._rotate("nope")
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "session_id")

    def test_actor_errors(self) -> None:
        predecessor = self._predecessor()
        status, body = self._rotate(
            predecessor["session_id"], actor_device_id="ghost")
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "actor_device_id")
        # Non-creator but active.
        status, body = self._rotate(
            predecessor["session_id"], actor_device_id="alice")
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "actor_device_id")
        # Revoked creator.
        self.service.store.revoke_device("creator")
        status, body = self._rotate(
            predecessor["session_id"], rotation_id="rot-2")
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "actor_device_id")

    def test_revision_mismatch(self) -> None:
        predecessor = self._predecessor()
        # A membership change bumps the group revision to 2.
        self.assertEqual(self._request(
            "POST", "/v1/groups/g1/members",
            {"actor_device_id": "creator", "device_id": "carol"})[0], 201)
        status, body = self._rotate(
            predecessor["session_id"], expected_revision=1)
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "expected_revision")

    def test_validation_errors_400(self) -> None:
        predecessor = self._predecessor()
        path = f"/v1/group-sessions/{predecessor['session_id']}/rotate"
        valid = {
            "rotation_id": "rot-1", "actor_device_id": "creator",
            "ephemeral_key": _valid_key(), "expected_revision": 1}
        for field, bad in (
                ("rotation_id", ""),
                ("rotation_id", 5),
                ("actor_device_id", ""),
                ("ephemeral_key", ""),
                ("ephemeral_key", "not-a-key"),
                ("expected_revision", 0),
                ("expected_revision", -1),
                ("expected_revision", 1.5),
                ("expected_revision", True)):
            body = dict(valid)
            body[field] = bad
            status, response = self._request("POST", path, body)
            self.assertEqual(status, 400, (field, bad, response))
            self.assertEqual(response["field"], field, (field, bad))
        # Missing fields.
        for field in tuple(valid):
            body = dict(valid)
            del body[field]
            status, response = self._request("POST", path, body)
            self.assertEqual(status, 400, field)
            self.assertEqual(response["field"], field)

    def test_frozen_members_and_revision_at_commit(self) -> None:
        # Group is created with revision 1, a session freezes revision 1;
        # then carol joins (revision 2). Rotation with expected_revision 2
        # must freeze the new roster and revision.
        predecessor = self._predecessor()
        self.assertEqual(self._request(
            "POST", "/v1/groups/g1/members",
            {"actor_device_id": "creator", "device_id": "carol"})[0], 201)
        status, body = self._rotate(
            predecessor["session_id"], rotation_id="rot-2",
            expected_revision=2)
        self.assertEqual(status, 201)
        self.assertEqual(body["revision"], 2)
        self.assertEqual(body["members"],
                         ["creator", "alice", "bob", "carol"])

    def test_predecessor_snapshot_unchanged(self) -> None:
        predecessor = self._predecessor()
        self.assertEqual(self._request(
            "POST", "/v1/groups/g1/members",
            {"actor_device_id": "creator", "device_id": "carol"})[0], 201)
        self._rotate(predecessor["session_id"], expected_revision=2)
        status, fetched = self._request(
            "GET", f"/v1/group-sessions/{predecessor['session_id']}")
        self.assertEqual(status, 200)
        self.assertEqual(fetched, predecessor)

    def test_rotation_chains(self) -> None:
        # A successor can itself be rotated with a fresh rotation id.
        predecessor = self._predecessor()
        _, first = self._rotate(predecessor["session_id"])
        self.assertEqual(self._request(
            "POST", "/v1/groups/g1/members",
            {"actor_device_id": "creator", "device_id": "carol"})[0], 201)
        status, second = self._rotate(
            first["session_id"], rotation_id="rot-2", expected_revision=2)
        self.assertEqual(status, 201)
        self.assertEqual(second["predecessor_session_id"],
                         first["session_id"])
        self.assertEqual(second["revision"], 2)
        # All session ids are distinct.
        ids = {predecessor["session_id"], first["session_id"],
               second["session_id"]}
        self.assertEqual(len(ids), 3)

    def test_new_session_ids_unique_under_concurrency(self) -> None:
        predecessor = self._predecessor()
        # Only one rotation can win per predecessor; the losers get 409 and
        # the winner's id must not collide with anything.
        results = []
        threads = []

        def go(rotation_id: str) -> None:
            connection = HTTPConnection("127.0.0.1", self.port, timeout=5)
            body = json.dumps({
                "rotation_id": rotation_id, "actor_device_id": "creator",
                "ephemeral_key": _valid_key(), "expected_revision": 1})
            connection.request(
                "POST",
                f"/v1/group-sessions/{predecessor['session_id']}/rotate",
                body, {"Content-Type": "application/json"})
            response = connection.getresponse()
            results.append((response.status,
                            json.loads(response.read().decode("utf-8"))))
            connection.close()

        for index in range(8):
            thread = threading.Thread(target=go, args=(f"rot-{index}",))
            threads.append(thread)
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        statuses = sorted(status for status, _ in results)
        self.assertEqual(statuses.count(201), 1)
        self.assertEqual(statuses.count(409), 7)
        winner = next(body for status, body in results if status == 201)
        self.assertNotEqual(winner["session_id"], predecessor["session_id"])

    def test_group_delivery_rejects_sender_and_unregistered(self) -> None:
        predecessor = self._predecessor()
        _, rotated = self._rotate(predecessor["session_id"])
        # The creator sends a message into the successor session.
        self.assertEqual(self._request("POST", "/v1/messages", {
            "session_id": rotated["session_id"],
            "sender_device_id": "creator", "message_id": "m1",
            "sequence": 1, "nonce": "n1", "ciphertext": "c"})[0], 201)
        # The sender cannot retry/ack for its own message.
        status, body = self._request(
            "POST",
            f"/v1/messages/{rotated['session_id']}/retry/m1",
            {"device_id": "creator", "attempt_id": "a1"})
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "device_id")
        # An unregistered device is rejected too.
        status, body = self._request(
            "POST",
            f"/v1/messages/{rotated['session_id']}/retry/m1",
            {"device_id": "ghost", "attempt_id": "a1"})
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "device_id")
        # A frozen non-sender member works.
        status, body = self._request(
            "POST",
            f"/v1/messages/{rotated['session_id']}/retry/m1",
            {"device_id": "alice", "attempt_id": "a1"})
        self.assertEqual(status, 201)


if __name__ == "__main__":
    unittest.main()
