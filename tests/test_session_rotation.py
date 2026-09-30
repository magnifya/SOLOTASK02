"""HTTP tests for 1:1 session rotation (POST .../rotate, GET .../rotation)."""
import base64
import json
import threading
import unittest
from http.client import HTTPConnection

from cryptography.hazmat.primitives.asymmetric import x25519
from cryptography.hazmat.primitives import serialization

from e2ee_backend.http_app import create_server
from e2ee_backend.models import Device, SignedPreKey


def _valid_key() -> str:
    raw = x25519.X25519PrivateKey.generate().public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return base64.b64encode(raw).decode()


_SESSION_EIGHT = {
    "session_id", "initiator_device_id", "recipient_device_id", "prekey_id",
    "ephemeral_key", "identity_key", "public_key", "created_at",
}
_ROTATION_EXTRA = {
    "rotation_id", "predecessor_session_id", "successor_session_id",
    "predecessor_last_sequence",
}


class SessionRotationHTTPTest(unittest.TestCase):
    def setUp(self) -> None:
        self.server, self.service = create_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        # Initiator a and recipient b; b has two prekeys so a rotation can
        # name a different (still valid) prekey than the first session.
        self.service.store.add_device(
            Device("u", "a", _valid_key(),
                   prekeys=[SignedPreKey("ak1", _valid_key())]))
        self.service.store.add_device(
            Device("u", "b", _valid_key(),
                   prekeys=[SignedPreKey("rk1", _valid_key()),
                            SignedPreKey("rk2", _valid_key())]))

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

    def _session(self, prekey_id: str = "rk1"):
        status, body = self._request("POST", "/v1/sessions", {
            "initiator_device_id": "a", "recipient_device_id": "b",
            "prekey_id": prekey_id, "ephemeral_key": _valid_key()})
        self.assertEqual(status, 201, body)
        return body

    def _rotate(self, session_id: str, **overrides):
        body = {
            "rotation_id": "rot-1",
            "actor_device_id": "a",
            "prekey_id": "rk2",
            "ephemeral_key": _valid_key(),
        }
        body.update(overrides)
        return self._request(
            "POST", f"/v1/sessions/{session_id}/rotate", body)

    # -- happy path --------------------------------------------------------

    def test_first_rotation_201_twelve_fields(self) -> None:
        predecessor = self._session()
        status, body = self._rotate(predecessor["session_id"])
        self.assertEqual(status, 201)
        self.assertEqual(set(body), _SESSION_EIGHT | _ROTATION_EXTRA)
        self.assertEqual(body["rotation_id"], "rot-1")
        self.assertEqual(body["predecessor_session_id"],
                         predecessor["session_id"])
        self.assertEqual(body["successor_session_id"], body["session_id"])
        self.assertNotEqual(body["session_id"], predecessor["session_id"])
        # Endpoints preserved; the successor uses the new prekey/ephemeral and
        # the recipient's current identity key.
        self.assertEqual(body["initiator_device_id"], "a")
        self.assertEqual(body["recipient_device_id"], "b")
        self.assertEqual(body["prekey_id"], "rk2")
        recipient = self.service.store._find_device("b")
        self.assertEqual(body["identity_key"], recipient.identity_key)
        self.assertEqual(body["predecessor_last_sequence"], 0)

    def test_successor_is_a_real_gettable_session(self) -> None:
        predecessor = self._session()
        _, rotated = self._rotate(predecessor["session_id"])
        status, fetched = self._request(
            "GET", f"/v1/sessions/{rotated['session_id']}")
        self.assertEqual(status, 200)
        self.assertEqual(set(fetched), _SESSION_EIGHT)
        for name in _SESSION_EIGHT:
            self.assertEqual(fetched[name], rotated[name])

    def test_predecessor_history_is_preserved(self) -> None:
        predecessor = self._session()
        # Two messages into the predecessor before rotating.
        for sequence, nonce in ((1, "n1"), (2, "n2")):
            status, _ = self._request("POST", "/v1/messages", {
                "session_id": predecessor["session_id"],
                "sender_device_id": "a", "message_id": f"m{sequence}",
                "sequence": sequence, "nonce": nonce, "ciphertext": "c"})
            self.assertEqual(status, 201)
        _, rotated = self._rotate(predecessor["session_id"])
        self.assertEqual(rotated["predecessor_last_sequence"], 2)
        # The old history is still readable, unchanged.
        status, page = self._request(
            "GET", f"/v1/messages/{predecessor['session_id']}?device_id=b")
        self.assertEqual(status, 200)
        self.assertEqual([m["sequence"] for m in page["messages"]], [1, 2])
        status, fetched = self._request(
            "GET", f"/v1/sessions/{predecessor['session_id']}")
        self.assertEqual(status, 200)
        self.assertEqual(fetched, predecessor)

    def test_successor_sequence_starts_at_one(self) -> None:
        predecessor = self._session()
        for sequence, nonce in ((1, "n1"), (2, "n2")):
            self._request("POST", "/v1/messages", {
                "session_id": predecessor["session_id"],
                "sender_device_id": "a", "message_id": f"m{sequence}",
                "sequence": sequence, "nonce": nonce, "ciphertext": "c"})
        _, rotated = self._rotate(predecessor["session_id"])
        # The successor is a fresh stream: sequence starts at 1.
        status, _ = self._request("POST", "/v1/messages", {
            "session_id": rotated["session_id"],
            "sender_device_id": "a", "message_id": "x1",
            "sequence": 1, "nonce": "x1", "ciphertext": "c"})
        self.assertEqual(status, 201)
        status, body = self._request("POST", "/v1/messages", {
            "session_id": rotated["session_id"],
            "sender_device_id": "a", "message_id": "x0",
            "sequence": 3, "nonce": "x3", "ciphertext": "c"})
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "sequence")

    def test_old_session_rejects_new_message(self) -> None:
        predecessor = self._session()
        self._rotate(predecessor["session_id"])
        status, body = self._request("POST", "/v1/messages", {
            "session_id": predecessor["session_id"],
            "sender_device_id": "a", "message_id": "m1",
            "sequence": 1, "nonce": "n1", "ciphertext": "c"})
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "session_id")

    def test_pre_rotation_submit_replay_still_200(self) -> None:
        predecessor = self._session()
        envelope = {"session_id": predecessor["session_id"],
                    "sender_device_id": "a", "message_id": "m1",
                    "sequence": 1, "nonce": "n1", "ciphertext": "c"}
        status, first = self._request(
            "POST", "/v1/messages/submit", {"request_id": "q1", **envelope})
        self.assertEqual(status, 201)
        self._rotate(predecessor["session_id"])
        # Same request_id replays the original 200 even though the predecessor
        # no longer accepts new writes.
        status, again = self._request(
            "POST", "/v1/messages/submit", {"request_id": "q1", **envelope})
        self.assertEqual(status, 200)
        self.assertEqual(again, first)
        # A *new* request_id aimed at the rotated predecessor is rejected.
        status, body = self._request(
            "POST", "/v1/messages/submit",
            {"request_id": "q2", **{**envelope, "message_id": "m2",
                                    "sequence": 2, "nonce": "n2"}})
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "session_id")

    # -- idempotency / conflicts -------------------------------------------

    def test_replay_same_predecessor_returns_200_original(self) -> None:
        predecessor = self._session()
        status, first = self._rotate(predecessor["session_id"])
        self.assertEqual(status, 201)
        status, second = self._rotate(
            predecessor["session_id"], ephemeral_key=_valid_key())
        self.assertEqual(status, 200)
        self.assertEqual(second, first)

    def test_rotation_id_for_other_predecessor_conflicts(self) -> None:
        first = self._session()
        self.assertEqual(self._rotate(first["session_id"])[0], 201)
        other = self._session()
        status, body = self._rotate(other["session_id"])
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "rotation_id")

    def test_predecessor_already_rotated_refuses_fork(self) -> None:
        predecessor = self._session()
        self.assertEqual(self._rotate(predecessor["session_id"])[0], 201)
        status, body = self._rotate(
            predecessor["session_id"], rotation_id="rot-2")
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "session_id")

    def test_rotation_chains(self) -> None:
        predecessor = self._session()
        _, first = self._rotate(predecessor["session_id"])
        # The successor may itself be rotated by its initiator.
        status, second = self._request(
            "POST", f"/v1/sessions/{first['session_id']}/rotate",
            {"rotation_id": "rot-2", "actor_device_id": "a",
             "prekey_id": "rk1", "ephemeral_key": _valid_key()})
        self.assertEqual(status, 201)
        self.assertEqual(second["predecessor_session_id"],
                         first["session_id"])
        self.assertEqual(second["predecessor_last_sequence"], 0)
        ids = {predecessor["session_id"], first["session_id"],
               second["session_id"]}
        self.assertEqual(len(ids), 3)

    # -- GET rotation ------------------------------------------------------

    def test_get_rotation_by_predecessor_or_successor(self) -> None:
        predecessor = self._session()
        _, rotated = self._rotate(predecessor["session_id"])
        status, by_pred = self._request(
            "GET", f"/v1/sessions/{predecessor['session_id']}/rotation")
        self.assertEqual(status, 200)
        self.assertEqual(by_pred, rotated)
        status, by_succ = self._request(
            "GET", f"/v1/sessions/{rotated['session_id']}/rotation")
        self.assertEqual(status, 200)
        self.assertEqual(by_succ, rotated)

    def test_get_rotation_unknown_404(self) -> None:
        for target in ("nope",):
            status, body = self._request(
                "GET", f"/v1/sessions/{target}/rotation")
            self.assertEqual(status, 404)
            self.assertEqual(body["field"], "session_id")
        # A real session that was never rotated is also 404.
        fresh = self._session()
        status, body = self._request(
            "GET", f"/v1/sessions/{fresh['session_id']}/rotation")
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "session_id")

    # -- validation / authorization ----------------------------------------

    def test_unknown_predecessor_404(self) -> None:
        status, body = self._rotate("nope")
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "session_id")

    def test_actor_errors(self) -> None:
        predecessor = self._session()
        status, body = self._rotate(
            predecessor["session_id"], actor_device_id="ghost")
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "actor_device_id")
        # Recipient is not the initiator.
        status, body = self._rotate(
            predecessor["session_id"], actor_device_id="b")
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "actor_device_id")

    def test_actor_revoked_409_but_replay_still_200(self) -> None:
        predecessor = self._session()
        _, first = self._rotate(predecessor["session_id"])
        self.service.store.revoke_device("a")
        status, body = self._rotate(
            predecessor["session_id"], rotation_id="rot-2")
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "actor_device_id")
        # The committed rotation still replays by its original id.
        status, again = self._rotate(predecessor["session_id"])
        self.assertEqual(status, 200)
        self.assertEqual(again, first)

    def test_recipient_revoked_409(self) -> None:
        predecessor = self._session()
        self.service.store.revoke_device("b")
        status, body = self._rotate(predecessor["session_id"])
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "recipient_device_id")

    def test_prekey_errors(self) -> None:
        predecessor = self._session()
        status, body = self._rotate(
            predecessor["session_id"], prekey_id="nope")
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "prekey_id")
        # A prekey owned by another device is also 404/prekey_id.
        status, body = self._rotate(
            predecessor["session_id"], prekey_id="ak1")
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "prekey_id")
        # Revoked recipient prekey.
        self.service.store.revoke_prekey_by_id("b", "rk2")
        status, body = self._rotate(
            predecessor["session_id"], rotation_id="rot-9")
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "prekey_id")

    def test_validation_errors_400(self) -> None:
        predecessor = self._session()
        path = f"/v1/sessions/{predecessor['session_id']}/rotate"
        valid = {
            "rotation_id": "rot-1", "actor_device_id": "a",
            "prekey_id": "rk2", "ephemeral_key": _valid_key()}
        for field, bad in (
                ("rotation_id", ""),
                ("rotation_id", 5),
                ("rotation_id", None),
                ("actor_device_id", ""),
                ("actor_device_id", 7),
                ("prekey_id", ""),
                ("prekey_id", 3),
                ("ephemeral_key", ""),
                ("ephemeral_key", 4),
                ("ephemeral_key", "not-a-key")):
            body = dict(valid)
            body[field] = bad
            status, response = self._request("POST", path, body)
            self.assertEqual(status, 400, (field, bad, response))
            self.assertEqual(response["field"], field, (field, bad))
        for field in tuple(valid):
            body = dict(valid)
            del body[field]
            status, response = self._request("POST", path, body)
            self.assertEqual(status, 400, field)
            self.assertEqual(response["field"], field, field)

    def test_body_must_be_object(self) -> None:
        predecessor = self._session()
        for bad in (["rot-1"], "rot-1", 42):
            status, body = self._request(
                "POST", f"/v1/sessions/{predecessor['session_id']}/rotate",
                bad)
            self.assertEqual(status, 400, bad)
            self.assertEqual(body["field"], "request_body", bad)

    def test_only_one_rotation_wins_under_concurrency(self) -> None:
        predecessor = self._session()
        results = []
        threads = []

        def go(rotation_id: str) -> None:
            connection = HTTPConnection("127.0.0.1", self.port, timeout=5)
            body = json.dumps({
                "rotation_id": rotation_id, "actor_device_id": "a",
                "prekey_id": "rk2", "ephemeral_key": _valid_key()})
            connection.request(
                "POST",
                f"/v1/sessions/{predecessor['session_id']}/rotate",
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


if __name__ == "__main__":
    unittest.main()
