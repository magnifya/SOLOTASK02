"""Tests for group-session rotation: POST /v1/group-sessions/{id}/rotate.

Covers the service contract (validation, idempotent replay, no-fork, frozen
commit-time snapshot), the HTTP status/field mapping, durable persistence
(round trip, missing section, contradictory-file refusal, failed-write
rollback), the CLI command, and the group-delivery device checks the rotation
successor inherits.
"""
from __future__ import annotations

import base64
import copy
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from http.client import HTTPConnection

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import x25519

from e2ee_backend.http_app import create_server
from e2ee_backend.models import Device
from e2ee_backend.persistence import (
    PersistenceUnavailable,
    StateFileError,
    attach_persistence,
)
from e2ee_backend.service import DeviceService, ServiceError


def _raw_key_b64() -> str:
    key = x25519.X25519PrivateKey.generate().public_key()
    raw = key.public_bytes(serialization.Encoding.Raw,
                           serialization.PublicFormat.Raw)
    return base64.b64encode(raw).decode()


class RotationFixture:
    """Group g1 (creator d1, members d1/d2/d3) and one group session."""

    def __init__(self, service: DeviceService | None = None) -> None:
        self.service = service if service is not None else DeviceService()
        for device_id in ("d1", "d2", "d3", "d4"):
            self.service.store.add_device(
                Device("u1", device_id, "identity-key"))
        self.service.create_group({
            "group_id": "g1", "creator_device_id": "d1",
            "member_device_ids": ["d2", "d3"]})
        self.session_id = self.service.create_group_session({
            "group_id": "g1", "initiator_device_id": "d1",
            "ephemeral_key": _raw_key_b64()})["session_id"]

    def payload(self, **overrides) -> dict:
        payload = {
            "rotation_id": "rot-1",
            "actor_device_id": "d1",
            "ephemeral_key": _raw_key_b64(),
            "expected_revision": 1,
        }
        payload.update(overrides)
        return payload


class RotationServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fx = RotationFixture()
        self.service = self.fx.service
        self.sid = self.fx.session_id

    def _rotate(self, **overrides):
        return self.service.rotate_group_session(
            self.sid, self.fx.payload(**overrides))

    def test_first_rotation_is_201_with_nine_fields(self) -> None:
        body, status = self._rotate()
        self.assertEqual(status, 201)
        self.assertEqual(set(body), {
            "session_id", "group_id", "initiator_device_id", "ephemeral_key",
            "revision", "members", "created_at", "rotation_id",
            "predecessor_session_id"})
        self.assertEqual(body["group_id"], "g1")
        self.assertEqual(body["initiator_device_id"], "d1")
        self.assertEqual(body["revision"], 1)
        self.assertEqual(body["members"], ["d1", "d2", "d3"])
        self.assertEqual(body["rotation_id"], "rot-1")
        self.assertEqual(body["predecessor_session_id"], self.sid)
        self.assertNotEqual(body["session_id"], self.sid)
        self.assertTrue(body["created_at"].endswith("+00:00"))
        # The successor exists as an ordinary group session.
        self.assertEqual(
            self.service.get_group_session(body["session_id"])["members"],
            ["d1", "d2", "d3"])

    def test_successor_id_is_unique_across_rotations(self) -> None:
        first, _ = self._rotate(rotation_id="rot-a")
        other = self.service.create_group_session({
            "group_id": "g1", "initiator_device_id": "d1",
            "ephemeral_key": _raw_key_b64()})
        second, _ = self.service.rotate_group_session(
            other["session_id"], self.fx.payload(rotation_id="rot-b"))
        self.assertNotEqual(first["session_id"], second["session_id"])
        self.assertNotIn(first["session_id"], (self.sid, other["session_id"]))

    def test_members_and_revision_freeze_at_commit_time(self) -> None:
        # Predecessor predates the roster change; the rotation's successor
        # takes the group's *current* snapshot (revision 2, d4 included).
        body, status = self.service.add_group_member("g1", {
            "actor_device_id": "d1", "device_id": "d4"})
        self.assertEqual(body["revision"], 2)
        rotated, status = self._rotate(expected_revision=2)
        self.assertEqual(status, 201)
        self.assertEqual(rotated["revision"], 2)
        self.assertEqual(rotated["members"], ["d1", "d2", "d3", "d4"])
        # The predecessor snapshot is untouched.
        predecessor = self.service.get_group_session(self.sid)
        self.assertEqual(predecessor["revision"], 1)
        self.assertEqual(predecessor["members"], ["d1", "d2", "d3"])

    def test_predecessor_is_unchanged_after_rotation(self) -> None:
        before = self.service.get_group_session(self.sid)
        self._rotate()
        after = self.service.get_group_session(self.sid)
        self.assertEqual(before, after)

    def test_same_rotation_id_replays_identical_response_200(self) -> None:
        first, first_status = self._rotate()
        # A replay carries a different ephemeral key and (nominally) revision;
        # the stored original response is returned unchanged.
        replay, replay_status = self._rotate(
            ephemeral_key=_raw_key_b64(), expected_revision=99)
        self.assertEqual(first_status, 201)
        self.assertEqual(replay_status, 200)
        self.assertEqual(replay, first)
        # No extra successor was created.
        rotations = self.service.store._group_session_rotations
        self.assertEqual(list(rotations), ["rot-1"])

    def test_same_rotation_id_other_predecessor_is_409(self) -> None:
        self._rotate()
        other = self.service.create_group_session({
            "group_id": "g1", "initiator_device_id": "d1",
            "ephemeral_key": _raw_key_b64()})
        with self.assertRaises(ServiceError) as ctx:
            self.service.rotate_group_session(
                other["session_id"], self.fx.payload())
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "rotation_id")
        # The conflict created nothing on the other predecessor.
        self.assertNotIn(other["session_id"],
                         self.service.store._rotation_by_predecessor)

    def test_predecessor_rotated_by_other_id_cannot_fork(self) -> None:
        self._rotate()
        with self.assertRaises(ServiceError) as ctx:
            self._rotate(rotation_id="rot-2")
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "session_id")
        # Exactly one successor exists for this predecessor.
        records = [r for r in
                   self.service.store._group_session_rotations.values()
                   if r.predecessor_session_id == self.sid]
        self.assertEqual(len(records), 1)

    def test_unknown_predecessor_is_404_session_id(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.rotate_group_session(
                "ghost-session", self.fx.payload(rotation_id="rot-x"))
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "session_id")

    def test_unknown_actor_is_404_actor_device_id(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self._rotate(rotation_id="rot-x", actor_device_id="ghost")
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "actor_device_id")

    def test_revoked_actor_is_409_actor_device_id(self) -> None:
        self.service.revoke_device("d1")
        with self.assertRaises(ServiceError) as ctx:
            self._rotate(rotation_id="rot-x")
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "actor_device_id")

    def test_non_creator_actor_is_409_actor_device_id(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self._rotate(rotation_id="rot-x", actor_device_id="d2")
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "actor_device_id")

    def test_revision_mismatch_is_409_expected_revision(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self._rotate(rotation_id="rot-x", expected_revision=2)
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "expected_revision")

    def test_successor_can_itself_be_rotated(self) -> None:
        first, _ = self._rotate(rotation_id="rot-a")
        second, status = self.service.rotate_group_session(
            first["session_id"],
            self.fx.payload(rotation_id="rot-b"))
        self.assertEqual(status, 201)
        self.assertEqual(second["predecessor_session_id"],
                         first["session_id"])
        self.assertEqual(second["members"], first["members"])

    def test_failed_rotation_writes_nothing(self) -> None:
        before = copy.deepcopy(self.service.store.snapshot_state())
        with self.assertRaises(ServiceError):
            self._rotate(rotation_id="rot-x", expected_revision=42)
        self.assertEqual(self.service.store.snapshot_state(), before)

    # -- payload validation (all 400) --------------------------------------

    def test_validation_errors_are_400_with_named_fields(self) -> None:
        valid = self.fx.payload(rotation_id="rot-x")

        def expect_400(payload, field):
            with self.assertRaises(ServiceError) as ctx:
                self.service.rotate_group_session(self.sid, payload)
            self.assertEqual(ctx.exception.status_code, 400, payload)
            self.assertEqual(ctx.exception.field, field, payload)

        expect_400("not-an-object", "request_body")
        for name in ("rotation_id", "actor_device_id"):
            missing = {k: v for k, v in valid.items() if k != name}
            expect_400(missing, name)
            expect_400(dict(valid, **{name: ""}), name)
            expect_400(dict(valid, **{name: 7}), name)
        missing = {k: v for k, v in valid.items()
                   if k != "ephemeral_key"}
        expect_400(missing, "ephemeral_key")
        expect_400(dict(valid, ephemeral_key=""), "ephemeral_key")
        expect_400(dict(valid, ephemeral_key=7), "ephemeral_key")
        expect_400(dict(valid, ephemeral_key="not-a-key"), "ephemeral_key")
        missing = {k: v for k, v in valid.items()
                   if k != "expected_revision"}
        expect_400(missing, "expected_revision")
        for bad in (0, -1, "1", True, 1.5, None):
            expect_400(dict(valid, expected_revision=bad),
                       "expected_revision")


class RotationHTTPTest(unittest.TestCase):
    def setUp(self) -> None:
        self.server, self.service = create_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        self.fx = RotationFixture(self.service)
        self.sid = self.fx.session_id
        # Rotation needs a second, initially-unrotated predecessor for some
        # error-ordering cases.
        self.unrotated = self.service.create_group_session({
            "group_id": "g1", "initiator_device_id": "d1",
            "ephemeral_key": _raw_key_b64()})["session_id"]

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _request(self, method, path, body=None):
        connection = HTTPConnection("127.0.0.1", self.port, timeout=5)
        payload = json.dumps(body) if body is not None else None
        headers = {"Content-Type": "application/json"} if payload else {}
        connection.request(method, path, body=payload, headers=headers)
        response = connection.getresponse()
        data = response.read().decode("utf-8")
        connection.close()
        return response.status, json.loads(data)

    def _payload(self, **overrides):
        payload = {
            "rotation_id": "rot-1", "actor_device_id": "d1",
            "ephemeral_key": _raw_key_b64(), "expected_revision": 1}
        payload.update(overrides)
        return payload

    def test_rotate_201_then_replay_200(self) -> None:
        path = f"/v1/group-sessions/{self.sid}/rotate"
        payload = self._payload()
        status, first = self._request("POST", path, payload)
        self.assertEqual(status, 201)
        self.assertEqual(set(first), {
            "session_id", "group_id", "initiator_device_id", "ephemeral_key",
            "revision", "members", "created_at", "rotation_id",
            "predecessor_session_id"})
        status, replay = self._request("POST", path, payload)
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)

    def test_http_error_fields(self) -> None:
        cases = [
            ("/v1/group-sessions/ghost/rotate",
             self._payload(rotation_id="rot-x"), 404, "session_id"),
            (f"/v1/group-sessions/{self.unrotated}/rotate",
             self._payload(rotation_id="rot-x", actor_device_id="ghost"),
             404, "actor_device_id"),
            (f"/v1/group-sessions/{self.unrotated}/rotate",
             self._payload(rotation_id="rot-x", actor_device_id="d2"),
             409, "actor_device_id"),
            (f"/v1/group-sessions/{self.unrotated}/rotate",
             self._payload(rotation_id="rot-x", expected_revision=7),
             409, "expected_revision"),
            (f"/v1/group-sessions/{self.unrotated}/rotate",
             self._payload(rotation_id="rot-x", ephemeral_key="bad"),
             400, "ephemeral_key"),
        ]
        for path, payload, code, field in cases:
            status, body = self._request("POST", path, payload)
            self.assertEqual((status, body["field"]), (code, field),
                             (path, payload))

    def test_id_conflict_and_no_fork_over_http(self) -> None:
        path = f"/v1/group-sessions/{self.sid}/rotate"
        status, _ = self._request("POST", path, self._payload())
        self.assertEqual(status, 201)
        # Same id against another predecessor -> 409/rotation_id.
        other = self.service.create_group_session({
            "group_id": "g1", "initiator_device_id": "d1",
            "ephemeral_key": _raw_key_b64()})["session_id"]
        status, body = self._request(
            "POST", f"/v1/group-sessions/{other}/rotate", self._payload())
        self.assertEqual((status, body["field"]), (409, "rotation_id"))
        # Another id against the rotated predecessor -> 409/session_id.
        status, body = self._request(
            "POST", path, self._payload(rotation_id="rot-2"))
        self.assertEqual((status, body["field"]), (409, "session_id"))

    def test_malformed_rotate_path_is_404(self) -> None:
        status, _ = self._request(
            "POST", "/v1/group-sessions//rotate", self._payload())
        self.assertEqual(status, 404)
        status, _ = self._request(
            "POST", "/v1/group-sessions/a/b/rotate", self._payload())
        self.assertEqual(status, 404)

    def test_successor_inherits_group_delivery_device_checks(self) -> None:
        # Rotate, then post a message into the successor; the sender itself
        # and an unregistered device are rejected by group delivery.
        status, rotated = self._request(
            "POST", f"/v1/group-sessions/{self.sid}/rotate", self._payload())
        self.assertEqual(status, 201)
        successor = rotated["session_id"]
        status, _ = self._request("POST", "/v1/messages", {
            "session_id": successor, "sender_device_id": "d1",
            "message_id": "m1", "sequence": 1, "nonce": "n1",
            "ciphertext": "ct"})
        self.assertEqual(status, 201)
        # The sender cannot retry/ack its own message.
        status, body = self._request(
            "POST", f"/v1/messages/{successor}/retry/m1",
            {"device_id": "d1", "attempt_id": "a1"})
        self.assertEqual((status, body["field"]), (409, "device_id"))
        # An unregistered device cannot either.
        status, body = self._request(
            "POST", f"/v1/messages/{successor}/retry/m1",
            {"device_id": "ghost", "attempt_id": "a1"})
        self.assertEqual((status, body["field"]), (409, "device_id"))
        # A frozen non-sender member can.
        status, _ = self._request(
            "POST", f"/v1/messages/{successor}/retry/m1",
            {"device_id": "d2", "attempt_id": "a1"})
        self.assertEqual(status, 201)


class RotationPersistenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.path = os.path.join(self.directory, "state.json")
        self.fx = RotationFixture()
        self.service = self.fx.service
        self.sid = self.fx.session_id
        # A second, never-rotated predecessor of the same group snapshot.
        self.other_sid = self.service.create_group_session({
            "group_id": "g1", "initiator_device_id": "d1",
            "ephemeral_key": _raw_key_b64()})["session_id"]

    def _document(self) -> dict:
        with open(self.path, encoding="utf-8") as handle:
            return json.load(handle)

    def test_round_trip_preserves_idempotency_and_no_fork(self) -> None:
        attach_persistence(self.service, self.path)
        body, status = self.service.rotate_group_session(
            self.sid, self.fx.payload())
        self.assertEqual(status, 201)

        document = self._document()
        self.assertEqual(document["version"], 1)
        self.assertEqual(len(document["group_session_rotations"]), 1)
        stored = document["group_session_rotations"][0]
        self.assertEqual(stored["rotation_id"], "rot-1")
        self.assertEqual(stored["predecessor_session_id"], self.sid)
        self.assertEqual(stored["session_id"], body["session_id"])
        self.assertEqual(stored["members"], ["d1", "d2", "d3"])
        self.assertEqual(stored["revision"], 1)

        restored = DeviceService()
        attach_persistence(restored, self.path)
        # Replay after restart -> 200 with the original response.
        replay, replay_status = restored.rotate_group_session(
            self.sid, self.fx.payload())
        self.assertEqual(replay_status, 200)
        self.assertEqual(replay["session_id"], body["session_id"])
        # The predecessor still cannot fork after restart.
        with self.assertRaises(ServiceError) as ctx:
            restored.rotate_group_session(
                self.sid, self.fx.payload(rotation_id="rot-other"))
        self.assertEqual(ctx.exception.field, "session_id")
        self.assertEqual(ctx.exception.status_code, 409)

    def test_missing_section_loads_as_empty(self) -> None:
        attach_persistence(self.service, self.path)
        self.service.rotate_group_session(self.sid, self.fx.payload())
        document = self._document()
        del document["group_session_rotations"]
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)

        restored = DeviceService()
        attach_persistence(restored, self.path)  # must not refuse to start
        self.assertEqual(restored.store._group_session_rotations, {})
        self.assertEqual(restored.store._rotation_by_predecessor, {})
        # Without the section the committed rotation is forgotten, so the id
        # and predecessor are free again.
        body, status = restored.rotate_group_session(
            self.sid, self.fx.payload())
        self.assertEqual(status, 201)
        self.assertNotEqual(
            body["session_id"],
            document["group_sessions"][-1]["session_id"])

    def test_contradictory_section_refuses_startup_without_overwrite(self):
        attach_persistence(self.service, self.path)
        self.service.rotate_group_session(self.sid, self.fx.payload())
        document = self._document()
        record = document["group_session_rotations"][0]
        other_session = next(
            s for s in document["group_sessions"]
            if s["session_id"] not in (record["session_id"],
                                       record["predecessor_session_id"]))

        def corrupt(mutate) -> None:
            broken = json.loads(json.dumps(document))
            mutate(broken)
            with open(self.path, "w", encoding="utf-8") as handle:
                json.dump(broken, handle)
            before = open(self.path, "rb").read()
            with self.assertRaises(StateFileError):
                attach_persistence(DeviceService(), self.path)
            # The contradictory file is left exactly as it was.
            self.assertEqual(open(self.path, "rb").read(), before)

        corrupt(lambda d: d["group_session_rotations"].append(
            dict(record, rotation_id="dup",
                 predecessor_session_id="ghost-session")))
        corrupt(lambda d: d["group_session_rotations"].append(
            dict(record, rotation_id="dup", session_id="ghost-session")))
        # Fork: two successors for the same predecessor.
        corrupt(lambda d: d["group_session_rotations"].append(
            dict(record, rotation_id="dup",
                 session_id=other_session["session_id"])))
        # The same successor claimed by a second rotation record.
        corrupt(lambda d: d["group_session_rotations"].append(
            dict(record, rotation_id="dup",
                 predecessor_session_id=other_session["session_id"])))
        corrupt(lambda d: d["group_session_rotations"].append(record))
        corrupt(lambda d: d["group_session_rotations"].append(
            dict(record, rotation_id="dup", members=["d1"])))
        corrupt(lambda d: d["group_session_rotations"].append(
            dict(record, rotation_id="dup", members=record["members"][:],
                 revision=99)))
        corrupt(lambda d: d["group_session_rotations"].append(
            dict(record, rotation_id="dup",
                 ephemeral_key=_raw_key_b64())))
        corrupt(lambda d: d["group_session_rotations"].append(
            dict(record, rotation_id="dup", actor_device_id="d2")))
        corrupt(lambda d: d["group_session_rotations"].append(
            dict(record, rotation_id="dup", group_id="g-other")))
        corrupt(lambda d: d["group_session_rotations"].append(
            dict(record, rotation_id="dup",
                 session_id=record["predecessor_session_id"])))
        corrupt(lambda d: d.__setitem__("group_session_rotations", "nope"))

    def test_failed_persist_rolls_back_rotation(self) -> None:
        from e2ee_backend.persistence import JsonStateStore

        attach_persistence(self.service, self.path)
        real_save = JsonStateStore.save

        def failing_save(self, state):
            raise OSError("disk full")

        JsonStateStore.save = failing_save
        try:
            with self.assertRaises(PersistenceUnavailable):
                self.service.rotate_group_session(
                    self.sid, self.fx.payload())
        finally:
            JsonStateStore.save = real_save

        # The failed rotation is visible nowhere: the next attempt is 201.
        body, status = self.service.rotate_group_session(
            self.sid, self.fx.payload())
        self.assertEqual(status, 201)
        self.assertEqual(len(self._document()["group_session_rotations"]), 1)
        self.assertEqual(
            self._document()["group_session_rotations"][0]["session_id"],
            body["session_id"])


class RotationCLITest(unittest.TestCase):
    def setUp(self) -> None:
        self.server, _ = create_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        self.fx = RotationFixture(
            self.server.RequestHandlerClass.service)
        self.sid = self.fx.session_id

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _run(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "e2ee_backend", "--base-url",
             self.base_url, *arguments],
            capture_output=True, text=True, timeout=15)

    def test_rotate_cli_lifecycle(self) -> None:
        ephemeral = _raw_key_b64()
        result = self._run(
            "rotate-group-session", self.sid, "--rotation-id", "rot-1",
            "--actor-device-id", "d1", "--ephemeral-key", ephemeral,
            "--expected-revision", "1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        self.assertEqual(result.stdout.count("\n"), 1)
        body = json.loads(result.stdout)
        self.assertEqual(set(body), {
            "session_id", "group_id", "initiator_device_id", "ephemeral_key",
            "revision", "members", "created_at", "rotation_id",
            "predecessor_session_id"})
        self.assertEqual(body["rotation_id"], "rot-1")
        self.assertEqual(body["predecessor_session_id"], self.sid)

        # Replay still exits 0 with the same single-line JSON body.
        replay = self._run(
            "rotate-group-session", self.sid, "--rotation-id", "rot-1",
            "--actor-device-id", "d1", "--ephemeral-key", ephemeral,
            "--expected-revision", "1")
        self.assertEqual(replay.returncode, 0, replay.stderr)
        self.assertEqual(json.loads(replay.stdout), body)

    def test_rotate_cli_failure_goes_to_stderr_nonzero(self) -> None:
        result = self._run(
            "rotate-group-session", "ghost-session",
            "--rotation-id", "rot-x", "--actor-device-id", "d1",
            "--ephemeral-key", _raw_key_b64(), "--expected-revision", "1")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        error = json.loads(result.stderr)
        self.assertEqual(error["field"], "session_id")


if __name__ == "__main__":
    unittest.main()
