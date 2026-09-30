"""Tests for one-to-one session rotation (POST /v1/sessions/{id}/rotate).

Covers the HTTP contract: 201 with twelve fields, successor material and
sequence restart, 200 byte-identical replay of the same rotation_id on the
same predecessor, GET .../rotation from either end, 400 validation on every
input field, 404/409 status/field mapping for actor/recipient/pre-key state,
the no-fork and global rotation_id rules, the old-session message cutoff
(409/session_id) while a pre-rotation submit keeps replaying 200 and the
successor accepts fresh messages from sequence 1, plus crash/restart
idempotency and malformed-section startup refusal.
"""
import base64
import json
import os
import shutil
import tempfile
import threading
import unittest
from http.client import HTTPConnection

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import x25519

from e2ee_backend.http_app import create_server
from e2ee_backend.persistence import StateFileError, attach_persistence
from e2ee_backend.service import DeviceService, ServiceError


def _raw_key_b64() -> str:
    key = x25519.X25519PrivateKey.generate().public_key()
    raw = key.public_bytes(serialization.Encoding.Raw,
                           serialization.PublicFormat.Raw)
    return base64.b64encode(raw).decode()


_SESSION_EIGHT = {"session_id", "initiator_device_id",
                  "recipient_device_id", "prekey_id", "ephemeral_key",
                  "identity_key", "public_key", "created_at"}
_ROTATION_EXTRA = {"rotation_id", "predecessor_session_id",
                   "successor_session_id", "predecessor_last_sequence"}
_ROTATION_FIELDS = _SESSION_EIGHT | _ROTATION_EXTRA


def _register_payload(device_id: str, key_ids=("k1", "k2")) -> dict:
    return {
        "user_id": "u1",
        "device_id": device_id,
        "identity_key": _raw_key_b64(),
        "signed_prekeys": [{"key_id": kid, "public_key": _raw_key_b64()}
                           for kid in key_ids],
    }


def _submit_payload(session_id: str, request_id: str = "req-1",
                    message_id: str = "m1", sequence: int = 1,
                    sender: str = "d1", nonce: str | None = None) -> dict:
    return {
        "request_id": request_id,
        "session_id": session_id,
        "sender_device_id": sender,
        "message_id": message_id,
        "sequence": sequence,
        "nonce": nonce or base64.b64encode(
            f"nonce-{message_id}".encode()).decode(),
        "ciphertext": base64.b64encode(b"ciphertext-and-tag").decode(),
    }


class _HTTPFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.server, self.service = create_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        self.service.register(_register_payload("d1"))
        self.service.register(_register_payload("d2", ("k1", "k2", "k3")))
        self.d2_identity = self.service.store._find_device("d2").identity_key
        self.d2_keys = {
            pk.key_id: pk.public_key
            for pk in self.service.store._find_device("d2").prekeys}

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

    def _session(self, prekey_id: str = "k1") -> dict:
        status, body = self._request("POST", "/v1/sessions", {
            "initiator_device_id": "d1",
            "recipient_device_id": "d2",
            "prekey_id": prekey_id,
            "ephemeral_key": _raw_key_b64()})
        self.assertEqual(status, 201)
        return body

    def _rotate(self, session_id: str, **overrides):
        body = {
            "rotation_id": "rot-1",
            "actor_device_id": "d1",
            "prekey_id": "k2",
            "ephemeral_key": _raw_key_b64(),
        }
        body.update(overrides)
        return self._request(
            "POST", f"/v1/sessions/{session_id}/rotate", body)


class SessionRotationHTTPTest(_HTTPFixture):
    def test_first_rotation_201_twelve_fields(self) -> None:
        predecessor = self._session()
        status, body = self._rotate(predecessor["session_id"])
        self.assertEqual(status, 201)
        self.assertEqual(set(body), _ROTATION_FIELDS)
        self.assertEqual(body["rotation_id"], "rot-1")
        self.assertEqual(body["predecessor_session_id"],
                         predecessor["session_id"])
        self.assertEqual(body["successor_session_id"], body["session_id"])
        self.assertNotEqual(body["session_id"], predecessor["session_id"])
        self.assertEqual(body["predecessor_last_sequence"], 0)
        # The successor preserves the endpoints and freezes current material.
        self.assertEqual(body["initiator_device_id"], "d1")
        self.assertEqual(body["recipient_device_id"], "d2")
        self.assertEqual(body["prekey_id"], "k2")
        self.assertEqual(body["identity_key"], self.d2_identity)
        self.assertEqual(body["public_key"], self.d2_keys["k2"])
        self.assertNotEqual(body["ephemeral_key"],
                            predecessor["ephemeral_key"])

    def test_predecessor_last_sequence_freezes_history_length(self) -> None:
        predecessor = self._session()
        for seq, mid in enumerate(("m1", "m2", "m3"), start=1):
            status, _ = self._request(
                "POST", "/v1/messages/submit",
                _submit_payload(predecessor["session_id"],
                                request_id=f"req-{mid}", message_id=mid,
                                sequence=seq))
            self.assertEqual(status, 201)
        _, body = self._rotate(predecessor["session_id"])
        self.assertEqual(body["predecessor_last_sequence"], 3)

    def test_successor_is_gettable_as_plain_session(self) -> None:
        predecessor = self._session()
        _, rotated = self._rotate(predecessor["session_id"])
        status, fetched = self._request(
            "GET", f"/v1/sessions/{rotated['session_id']}")
        self.assertEqual(status, 200)
        self.assertEqual(set(fetched), _SESSION_EIGHT)
        for key in _SESSION_EIGHT:
            self.assertEqual(fetched[key], rotated[key])
        # The predecessor snapshot is byte-unchanged.
        status, old = self._request(
            "GET", f"/v1/sessions/{predecessor['session_id']}")
        self.assertEqual(status, 200)
        self.assertEqual(old, predecessor)

    def test_get_rotation_from_predecessor_and_successor(self) -> None:
        predecessor = self._session()
        _, rotated = self._rotate(predecessor["session_id"])
        for session_id in (predecessor["session_id"],
                           rotated["session_id"]):
            status, body = self._request(
                "GET", f"/v1/sessions/{session_id}/rotation")
            self.assertEqual(status, 200)
            self.assertEqual(body, rotated)

    def test_get_rotation_unknown_or_unrotated_is_404(self) -> None:
        status, body = self._request("GET", "/v1/sessions/ghost/rotation")
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "session_id")
        session = self._session()
        status, body = self._request(
            "GET", f"/v1/sessions/{session['session_id']}/rotation")
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "session_id")

    def test_replay_same_predecessor_200_original(self) -> None:
        predecessor = self._session()
        status, first = self._rotate(predecessor["session_id"])
        self.assertEqual(status, 201)
        # Even the ephemeral key on the replay is ignored: original response.
        status, second = self._rotate(
            predecessor["session_id"], ephemeral_key=_raw_key_b64())
        self.assertEqual(status, 200)
        self.assertEqual(second, first)

    def test_replay_after_restart_is_stable(self) -> None:
        directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
        path = os.path.join(directory, "state.json")
        attach_persistence(self.service, path)
        predecessor = self._session()
        _, first = self._rotate(predecessor["session_id"])
        service = DeviceService()
        attach_persistence(service, path)
        self.assertEqual(
            len(service.store.snapshot_state()["session_rotations"]), 1)
        body, status = service.rotate_session(predecessor["session_id"], {
            "rotation_id": "rot-1", "actor_device_id": "d1",
            "prekey_id": "k1", "ephemeral_key": _raw_key_b64()})
        self.assertEqual(status, 200)
        self.assertEqual(body, first)

    def test_rotation_id_for_other_predecessor_conflicts(self) -> None:
        first = self._session()
        second = self._session()
        self.assertEqual(self._rotate(first["session_id"])[0], 201)
        status, body = self._rotate(
            second["session_id"], prekey_id="k2")
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "rotation_id")

    def test_predecessor_already_rotated_refuses_fork(self) -> None:
        predecessor = self._session()
        self.assertEqual(self._rotate(predecessor["session_id"])[0], 201)
        status, body = self._rotate(
            predecessor["session_id"], rotation_id="rot-2")
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "session_id")

    def test_only_one_winner_under_concurrency(self) -> None:
        # Only one rotation can win per predecessor; the losers get 409 and
        # no successor is left dangling.
        predecessor = self._session()
        results = []
        errors = []

        def go(rotation_id: str) -> None:
            try:
                status, body = self._rotate(
                    predecessor["session_id"], rotation_id=rotation_id,
                    prekey_id="k2" if rotation_id == "rot-1" else "k3")
                results.append((status, body))
            except Exception as error:  # pragma: no cover - test plumbing
                errors.append(error)

        threads = [threading.Thread(target=go, args=(f"rot-{i}",))
                   for i in range(1, 9)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
        self.assertFalse(errors)
        self.assertEqual(sum(1 for status, _ in results if status == 201), 1)
        self.assertEqual(sum(1 for status, _ in results if status == 409), 7)
        snapshot = self.service.store.snapshot_state()
        self.assertEqual(len(snapshot["session_rotations"]), 1)
        self.assertEqual(
            len(snapshot["sessions"]), 2)  # predecessor plus one successor

    def test_chains_and_successor_restarts_at_sequence_one(self) -> None:
        first = self._session()
        _, rot1 = self._rotate(first["session_id"], prekey_id="k2")
        # Rotate the successor with a fresh id; k1 of d2 is still available.
        status, rot2 = self._rotate(
            rot1["session_id"], rotation_id="rot-2", prekey_id="k1")
        self.assertEqual(status, 201)
        self.assertEqual(rot2["predecessor_session_id"], rot1["session_id"])
        # Messages on the newest successor start at seq 1.
        status, _ = self._request(
            "POST", "/v1/messages/submit",
            _submit_payload(rot2["session_id"], request_id="r-a",
                            message_id="a", sequence=1))
        self.assertEqual(status, 201)

    def test_unknown_predecessor_404(self) -> None:
        status, body = self._rotate("ghost")
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "session_id")

    def test_actor_errors(self) -> None:
        predecessor = self._session()
        # Unknown actor.
        status, body = self._rotate(
            predecessor["session_id"], actor_device_id="ghost")
        self.assertEqual((status, body["field"]), (404, "actor_device_id"))
        # The recipient is not the initiator.
        status, body = self._rotate(
            predecessor["session_id"], actor_device_id="d2")
        self.assertEqual((status, body["field"]), (409, "actor_device_id"))
        # Revoked initiator.
        self.service.revoke_device("d1")
        status, body = self._rotate(
            predecessor["session_id"], rotation_id="rot-2")
        self.assertEqual((status, body["field"]), (409, "actor_device_id"))

    def test_recipient_revoked_409(self) -> None:
        predecessor = self._session()
        self.service.revoke_device("d2")
        status, body = self._rotate(predecessor["session_id"])
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "recipient_device_id")

    def test_prekey_errors(self) -> None:
        predecessor = self._session()
        # Unknown pre-key id.
        status, body = self._rotate(
            predecessor["session_id"], prekey_id="ghost")
        self.assertEqual((status, body["field"]), (404, "prekey_id"))
        # An id belonging to another device (d3), not the recipient.
        self.service.register(_register_payload("d3", ("pk9",)))
        status, body = self._rotate(
            predecessor["session_id"], prekey_id="pk9")
        self.assertEqual((status, body["field"]), (404, "prekey_id"))
        # A revoked recipient pre-key.
        self.service.revoke_prekey("d2", "k2")
        status, body = self._rotate(
            predecessor["session_id"], rotation_id="rot-2")
        self.assertEqual((status, body["field"]), (409, "prekey_id"))

    def test_consumed_prekey_409(self) -> None:
        # Plain session creation does not consume the pre-key; a claim does.
        predecessor = self._session()
        status, claim = self._request("POST", "/v1/prekeys/claim", {
            "recipient_device_id": "d2", "claim_id": "c-1"})
        self.assertEqual(status, 201)  # hands out d2's first key, k1
        status, body = self._rotate(
            predecessor["session_id"], prekey_id=claim["key_id"])
        self.assertEqual((status, body["field"]), (409, "prekey_id"))

    def test_validation_errors_400(self) -> None:
        predecessor = self._session()
        path = f"/v1/sessions/{predecessor['session_id']}/rotate"
        base = {
            "rotation_id": "rot-1", "actor_device_id": "d1",
            "prekey_id": "k2", "ephemeral_key": _raw_key_b64(),
        }
        for field in ("rotation_id", "actor_device_id", "prekey_id"):
            for bad in ("", 5, None, [], {}):
                body = dict(base)
                body[field] = bad
                status, response = self._request("POST", path, body)
                self.assertEqual((status, response["field"]),
                                 (400, field), msg=(field, bad))
        for bad in ("", 5, None, [], {}, "not-a-key"):
            body = dict(base)
            body["ephemeral_key"] = bad
            status, response = self._request("POST", path, body)
            self.assertEqual((status, response["field"]),
                             (400, "ephemeral_key"), msg=bad)
        # Missing fields.
        for field in ("rotation_id", "actor_device_id", "prekey_id",
                      "ephemeral_key"):
            body = {key: value for key, value in base.items()
                    if key != field}
            status, response = self._request("POST", path, body)
            self.assertEqual((status, response["field"]), (400, field))
        # A non-object body.
        status, response = self._request("POST", path, ["not", "object"])
        self.assertEqual((status, response["field"]),
                         (400, "request_body"))


class SessionRotationMessagesTest(_HTTPFixture):
    def test_old_session_rejects_new_messages_but_replays_submit(self) -> None:
        predecessor = self._session()
        # One committed submission before the rotation.
        status, original = self._request(
            "POST", "/v1/messages/submit",
            _submit_payload(predecessor["session_id"]))
        self.assertEqual(status, 201)
        _, rotated = self._rotate(predecessor["session_id"])
        # The pre-rotation request_id keeps replaying its original 200.
        status, replay = self._request(
            "POST", "/v1/messages/submit",
            _submit_payload(predecessor["session_id"]))
        self.assertEqual(status, 200)
        self.assertEqual(replay, original)
        # A genuinely new message into the old session is a 409.
        status, body = self._request(
            "POST", "/v1/messages/submit",
            _submit_payload(predecessor["session_id"], request_id="req-2",
                            message_id="m2", sequence=2,
                            nonce=base64.b64encode(b"n2").decode()))
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "session_id")
        # Plain POST /v1/messages on the old session is rejected the same way.
        plain = {key: value for key, value in _submit_payload(
            predecessor["session_id"], message_id="m3", sequence=3,
            nonce=base64.b64encode(b"n3").decode()).items()
            if key != "request_id"}
        status, body = self._request("POST", "/v1/messages", plain)
        self.assertEqual((status, body["field"]), (409, "session_id"))
        # History is still readable on the predecessor.
        status, page = self._request(
            "GET",
            f"/v1/messages/{predecessor['session_id']}"
            "?device_id=d2&limit=10")
        self.assertEqual(status, 200)
        self.assertEqual([m["message_id"] for m in page["messages"]], ["m1"])
        # The successor accepts new messages starting at sequence 1, with a
        # message id/nonce that the old history also used (independent stream).
        status, _ = self._request(
            "POST", "/v1/messages/submit",
            _submit_payload(rotated["session_id"], request_id="req-3",
                            message_id="m1", sequence=1))
        self.assertEqual(status, 201)
        status, page = self._request(
            "GET",
            f"/v1/messages/{rotated['session_id']}"
            "?device_id=d2&limit=10")
        self.assertEqual(status, 200)
        self.assertEqual([m["sequence"] for m in page["messages"]], [1])

    def test_successor_delivery_and_inbox_work_normally(self) -> None:
        predecessor = self._session()
        _, rotated = self._rotate(predecessor["session_id"])
        self.assertEqual(self._request(
            "POST", "/v1/messages/submit",
            _submit_payload(rotated["session_id"], request_id="req-1"))[0],
            201)
        # The recipient sees the successor message in its inbox, not only the
        # (empty) old stream.
        status, inbox = self._request(
            "GET", "/v1/devices/d2/inbox?limit=10")
        self.assertEqual(status, 200)
        self.assertEqual(
            [m["session_id"] for m in inbox["messages"]],
            [rotated["session_id"]])


class SessionRotationRecoveryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.path = os.path.join(self.directory, "state.json")
        service = DeviceService()
        attach_persistence(service, self.path)
        service.register(_register_payload("d1"))
        service.register(_register_payload("d2"))
        session = service.create_session({
            "initiator_device_id": "d1", "recipient_device_id": "d2",
            "prekey_id": "k1", "ephemeral_key": _raw_key_b64()})
        self.predecessor_id = session["session_id"]
        self.rotated, status = service.rotate_session(self.predecessor_id, {
            "rotation_id": "rot-1", "actor_device_id": "d1",
            "prekey_id": "k2", "ephemeral_key": _raw_key_b64()})
        self.assertEqual(status, 201)
        with open(self.path, encoding="utf-8") as handle:
            self.document = json.load(handle)

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def _assert_rejected(self, document) -> None:
        path = os.path.join(self.directory, "bad.json")
        document.pop("integrity_log_version", None)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        before = open(path, "rb").read()
        with self.assertRaises(StateFileError):
            attach_persistence(DeviceService(), path)
        self.assertEqual(open(path, "rb").read(), before)

    def test_record_survives_restart_and_no_fork_holds(self) -> None:
        service = DeviceService()
        attach_persistence(service, self.path)
        record = service.store.snapshot_state()["session_rotations"][0]
        self.assertEqual(record["rotation_id"], "rot-1")
        self.assertEqual(record["predecessor_session_id"],
                         self.predecessor_id)
        self.assertEqual(record["successor_session_id"],
                         self.rotated["session_id"])
        with self.assertRaises(ServiceError) as caught:
            service.rotate_session(self.predecessor_id, {
                "rotation_id": "rot-2", "actor_device_id": "d1",
                "prekey_id": "k1", "ephemeral_key": _raw_key_b64()})
        self.assertEqual(caught.exception.status_code, 409)

    def test_legacy_file_without_section_loads_empty(self) -> None:
        document = dict(self.document)
        del document["session_rotations"]
        document.pop("integrity_log_version", None)
        path = os.path.join(self.directory, "candidate.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        service = DeviceService()
        attach_persistence(service, path)
        self.assertEqual(
            service.store.snapshot_state()["session_rotations"], [])

    def test_malformed_records_are_rejected(self) -> None:
        record = self.document["session_rotations"][0]
        bad = [
            "not-an-object",
            {k: v for k, v in record.items() if k != "rotation_id"},
            {**record, "rotation_id": ""},
            {**record, "predecessor_session_id": "missing"},
            {**record, "successor_session_id": "missing"},
            {**record, "actor_device_id": "ghost"},
            {**record, "actor_device_id": "d2"},
            {**record, "predecessor_last_sequence": -1},
            {**record, "predecessor_last_sequence": "1"},
            {**record, "predecessor_last_sequence": "0"},
            {**record, "predecessor_last_sequence": True},
            {**record, "created_at": ""},
        ]
        for mutated in bad:
            document = dict(self.document)
            document["session_rotations"] = [mutated]
            self._assert_rejected(document)

    def test_duplicate_and_fork_rejected(self) -> None:
        record = self.document["session_rotations"][0]
        # Duplicate rotation_id.
        document = dict(self.document)
        document["session_rotations"] = [record, dict(record)]
        self._assert_rejected(document)
        # Fork: the same predecessor named twice (craft a second successor).
        service = DeviceService()
        attach_persistence(service, self.path)
        _, second = service.rotate_session(self.rotated["session_id"], {
            "rotation_id": "rot-2", "actor_device_id": "d1",
            "prekey_id": "k1", "ephemeral_key": _raw_key_b64()})
        snapshot = service.store.snapshot_state()
        snapshot["session_rotations"][1][
            "predecessor_session_id"] = self.predecessor_id
        self._assert_rejected({"version": 1, **snapshot})


if __name__ == "__main__":
    unittest.main()
