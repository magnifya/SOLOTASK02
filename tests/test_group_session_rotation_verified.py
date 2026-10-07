"""Tests for signature-authorized group-session rotation.

Covers the service, HTTP (real loopback socket), persistence and CLI layers
for:

* POST /v1/group-sessions/{session_id}/rotate-verified

The rotation commits only when the standard-base64 64-byte Ed25519 signature
verifies over the domain-separated canonical rotation message
(``E2EE-GROUP-SESSION-ROTATION-V1``) against the creator device's *current*
identity key, ``expected_revision`` equals the group's current ``revision``
and ``expected_version`` equals the creator's current
``identity_key_version``. A first rotation is 201 with the nine-field
rotation view; an identical replay on the same predecessor is 200 with the
original response even after later state changes; the same ``rotation_id``
with any field or the signature changed — or naming another predecessor —
is 409/field=rotation_id, and a predecessor already rotated under another
id is 409/field=session_id. The unsigned rotation endpoint, the frozen
member snapshots and legacy unsigned records keep their old behavior.
"""
import base64
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from http.client import HTTPConnection

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, x25519

from e2ee_backend.crypto import (
    group_session_rotation_proof_message,
    identity_rotation_proof_message,
)
from e2ee_backend.http_app import create_server
from e2ee_backend.models import Device
from e2ee_backend.persistence import (
    PersistenceUnavailable, StateFileError, attach_persistence)
from e2ee_backend.service import DeviceService, ServiceError


def _raw_b64(key) -> str:
    raw = key.public_bytes(serialization.Encoding.Raw,
                           serialization.PublicFormat.Raw)
    return base64.b64encode(raw).decode()


def _new_identity():
    private = ed25519.Ed25519PrivateKey.generate()
    return private, _raw_b64(private.public_key())


def _x25519_b64() -> str:
    return _raw_b64(x25519.X25519PrivateKey.generate().public_key())


def _x25519_der_b64() -> str:
    """X25519 public key as base64 DER SPKI — never parses as Ed25519."""
    der = x25519.X25519PrivateKey.generate().public_key().public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo)
    return base64.b64encode(der).decode()


def _authorization(private, group_id, predecessor_session_id, user_id,
                   rotation_id, actor, ephemeral_key, expected_revision,
                   expected_version) -> str:
    message = group_session_rotation_proof_message(
        group_id, predecessor_session_id, user_id, rotation_id, actor,
        ephemeral_key, expected_revision, expected_version)
    return base64.b64encode(private.sign(message)).decode()


def _register_payload(device_id, user_id="u1", identity_key=None) -> dict:
    return {
        "user_id": user_id,
        "device_id": device_id,
        "identity_key": identity_key or _x25519_b64(),
        "signed_prekeys": [{"key_id": "k1", "public_key": _x25519_b64()}],
    }


_ROTATION_FIELDS = {
    "session_id", "group_id", "initiator_device_id", "ephemeral_key",
    "revision", "members", "created_at",
    "rotation_id", "predecessor_session_id",
}


class _VerifiedRotationBase(unittest.TestCase):
    """Shared setup: an Ed25519 creator, one member and a group session."""

    def setUp(self) -> None:
        self.service = DeviceService()
        self.private, self.identity = _new_identity()
        self.service.register(_register_payload(
            "creator", identity_key=self.identity))
        self.service.register(_register_payload("alice", user_id="u2"))
        self.service.register(_register_payload("bob", user_id="u3"))
        self.service.create_group({
            "group_id": "g1", "creator_device_id": "creator",
            "member_device_ids": ["alice"]})
        self.ephemeral_key = _x25519_b64()
        self.predecessor = self.service.create_group_session({
            "group_id": "g1", "initiator_device_id": "creator",
            "ephemeral_key": _x25519_b64()})

    def _payload(self, rotation_id="rot-1", actor="creator",
                 ephemeral_key=None, expected_revision=1, expected_version=1,
                 signature=None, private=None, group_id="g1",
                 predecessor=None, user_id="u1"):
        if ephemeral_key is None:
            ephemeral_key = self.ephemeral_key
        if predecessor is None:
            predecessor = self.predecessor["session_id"]
        if signature is None:
            signature = _authorization(
                private or self.private, group_id, predecessor, user_id,
                rotation_id, actor, ephemeral_key, expected_revision,
                expected_version)
        return {"rotation_id": rotation_id, "actor_device_id": actor,
                "ephemeral_key": ephemeral_key,
                "expected_revision": expected_revision,
                "expected_version": expected_version,
                "signature": signature}

    def _rotate(self, payload=None, session_id=None):
        return self.service.rotate_group_session_verified(
            session_id or self.predecessor["session_id"],
            payload if payload is not None else self._payload())

    def _error(self, payload=None, session_id=None) -> ServiceError:
        with self.assertRaises(ServiceError) as caught:
            self._rotate(payload, session_id)
        return caught.exception


class GroupSessionRotationVerifiedServiceTest(_VerifiedRotationBase):
    def test_first_rotation_201_nine_fields(self) -> None:
        body, status = self._rotate()
        self.assertEqual(status, 201)
        self.assertEqual(set(body), _ROTATION_FIELDS)
        self.assertEqual(body["rotation_id"], "rot-1")
        self.assertEqual(body["predecessor_session_id"],
                         self.predecessor["session_id"])
        self.assertNotEqual(body["session_id"],
                            self.predecessor["session_id"])
        self.assertEqual(body["group_id"], "g1")
        self.assertEqual(body["initiator_device_id"], "creator")
        self.assertEqual(body["ephemeral_key"], self.ephemeral_key)
        self.assertEqual(body["revision"], 1)
        self.assertEqual(body["members"], ["creator", "alice"])

    def test_successor_is_a_real_group_session(self) -> None:
        body, _ = self._rotate()
        fetched = self.service.get_group_session(body["session_id"])
        self.assertEqual(fetched, {
            key: body[key]
            for key in ("session_id", "group_id", "initiator_device_id",
                        "ephemeral_key", "revision", "members",
                        "created_at")})

    def test_identical_replay_200_original_after_state_change(self) -> None:
        first, status = self._rotate()
        self.assertEqual(status, 201)
        # Move every checked state: group revision, creator identity
        # version, and revoke a member.
        self.service.add_group_member("g1", {
            "actor_device_id": "creator", "device_id": "bob"})
        new_private, new_identity = _new_identity()
        self.service.rotate_identity_key_verified("creator", {
            "identity_key": new_identity,
            "expected_version": 1,
            "signature": base64.b64encode(self.private.sign(
                identity_rotation_proof_message(
                    "u1", "creator", new_identity, 1))).decode()})
        second, status = self._rotate()
        self.assertEqual(status, 200)
        self.assertEqual(second, first)

    def test_replay_changed_field_or_signature_conflicts(self) -> None:
        self.assertEqual(self._rotate()[1], 201)
        for overrides in (
                {"ephemeral_key": _x25519_b64()},
                {"expected_revision": 2},
                {"expected_version": 2},
                {"signature": _authorization(
                    self.private, "g1", self.predecessor["session_id"],
                    "u1", "rot-1", "creator", self.ephemeral_key, 1, 2)}):
            error = self._error(self._payload(**overrides))
            self.assertEqual(error.status_code, 409)
            self.assertEqual(error.field, "rotation_id")

    def test_rotation_id_for_other_predecessor_conflicts(self) -> None:
        self.assertEqual(self._rotate()[1], 201)
        other = self.service.create_group_session({
            "group_id": "g1", "initiator_device_id": "creator",
            "ephemeral_key": _x25519_b64()})
        error = self._error(self._payload(predecessor=other["session_id"]),
                            session_id=other["session_id"])
        self.assertEqual(error.status_code, 409)
        self.assertEqual(error.field, "rotation_id")

    def test_predecessor_already_rotated_refuses_fork(self) -> None:
        self.assertEqual(self._rotate()[1], 201)
        error = self._error(self._payload(rotation_id="rot-2"))
        self.assertEqual(error.status_code, 409)
        self.assertEqual(error.field, "session_id")

    def test_unknown_session_404(self) -> None:
        error = self._error(self._payload(predecessor="missing"),
                            session_id="missing")
        self.assertEqual(error.status_code, 404)
        self.assertEqual(error.field, "session_id")

    def test_unknown_actor_404(self) -> None:
        error = self._error(self._payload(actor="ghost"))
        self.assertEqual(error.status_code, 404)
        self.assertEqual(error.field, "actor_device_id")

    def test_revoked_actor_409(self) -> None:
        self.service.revoke_device("creator")
        error = self._error()
        self.assertEqual(error.status_code, 409)
        self.assertEqual(error.field, "actor_device_id")

    def test_non_creator_actor_409(self) -> None:
        error = self._error(self._payload(actor="alice", user_id="u2"))
        self.assertEqual(error.status_code, 409)
        self.assertEqual(error.field, "actor_device_id")

    def test_non_ed25519_creator_identity_400(self) -> None:
        self.service.register(_register_payload(
            "carol", user_id="u4", identity_key=_x25519_der_b64()))
        self.service.create_group({
            "group_id": "g2", "creator_device_id": "carol",
            "member_device_ids": ["alice"]})
        session = self.service.create_group_session({
            "group_id": "g2", "initiator_device_id": "carol",
            "ephemeral_key": _x25519_b64()})
        payload = self._payload(actor="carol", user_id="u4", group_id="g2",
                                predecessor=session["session_id"])
        error = self._error(payload, session_id=session["session_id"])
        self.assertEqual(error.status_code, 400)
        self.assertEqual(error.field, "identity_key")

    def test_revision_mismatch_409(self) -> None:
        error = self._error(self._payload(expected_revision=2))
        self.assertEqual(error.status_code, 409)
        self.assertEqual(error.field, "expected_revision")

    def test_version_mismatch_409(self) -> None:
        error = self._error(self._payload(expected_version=2))
        self.assertEqual(error.status_code, 409)
        self.assertEqual(error.field, "expected_version")

    def test_bad_signature_400(self) -> None:
        wrong_private, _ = _new_identity()
        error = self._error(self._payload(private=wrong_private))
        self.assertEqual(error.status_code, 400)
        self.assertEqual(error.field, "signature")

    def test_signature_over_wrong_fields_400(self) -> None:
        # A valid signature over a different rotation_id does not authorize
        # this request.
        signature = _authorization(
            self.private, "g1", self.predecessor["session_id"], "u1",
            "rot-other", "creator", self.ephemeral_key, 1, 1)
        error = self._error(self._payload(signature=signature))
        self.assertEqual(error.status_code, 400)
        self.assertEqual(error.field, "signature")

    def test_validation_errors_400(self) -> None:
        good = self._payload()
        cases = []
        for field in ("rotation_id", "actor_device_id", "ephemeral_key",
                      "expected_revision", "expected_version", "signature"):
            missing = dict(good)
            del missing[field]
            cases.append((missing, field))
            emptied = dict(good, **{field: ""})
            cases.append((emptied, field))
        cases.append((dict(good, expected_revision=0), "expected_revision"))
        cases.append((dict(good, expected_revision=True),
                      "expected_revision"))
        cases.append((dict(good, expected_version=0), "expected_version"))
        cases.append((dict(good, expected_version="1"), "expected_version"))
        cases.append((dict(good, signature="!!!"), "signature"))
        # Valid base64 but not 64 bytes.
        cases.append((dict(good,
                           signature=base64.b64encode(b"\x00" * 32).decode()),
                      "signature"))
        # 64 bytes but non-canonical (URL-safe alphabet spelling).
        cases.append((dict(good, signature="A" * 88), "signature"))
        for payload, field in cases:
            with self.subTest(field=field, payload=payload):
                error = self._error(payload)
                self.assertEqual(error.status_code, 400)
                self.assertEqual(error.field, field)
        self.assertEqual(self._error("not-a-dict").field, "request_body")

    def test_failure_writes_nothing(self) -> None:
        wrong_private, _ = _new_identity()
        self._error(self._payload(private=wrong_private))
        # The failed attempt committed nothing: the same rotation_id can
        # still succeed with a valid authorization.
        body, status = self._rotate()
        self.assertEqual(status, 201)
        self.assertEqual(body["rotation_id"], "rot-1")


class GroupSessionRotationVerifiedHTTPTest(unittest.TestCase):
    def setUp(self) -> None:
        self.server, self.service = create_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        self.private, self.identity = _new_identity()
        self.assertEqual(self._request("POST", "/v1/devices",
                                       _register_payload(
                                           "creator",
                                           identity_key=self.identity))[0],
                         201)
        self.assertEqual(self._request(
            "POST", "/v1/devices",
            _register_payload("alice", user_id="u2"))[0], 201)
        self.assertEqual(self._request("POST", "/v1/groups", {
            "group_id": "g1", "creator_device_id": "creator",
            "member_device_ids": ["alice"]})[0], 201)
        status, self.predecessor = self._request(
            "POST", "/v1/group-sessions", {
                "group_id": "g1", "initiator_device_id": "creator",
                "ephemeral_key": _x25519_b64()})
        self.assertEqual(status, 201)
        self.ephemeral_key = _x25519_b64()

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

    def _payload(self, rotation_id="rot-1", **overrides):
        body = {
            "rotation_id": rotation_id,
            "actor_device_id": "creator",
            "ephemeral_key": self.ephemeral_key,
            "expected_revision": 1,
            "expected_version": 1,
        }
        body.update(overrides)
        body["signature"] = _authorization(
            self.private, "g1", self.predecessor["session_id"], "u1",
            body["rotation_id"], body["actor_device_id"],
            body["ephemeral_key"], body["expected_revision"],
            body["expected_version"])
        return body

    def _rotate(self, session_id, payload):
        return self._request(
            "POST", f"/v1/group-sessions/{session_id}/rotate-verified",
            payload)

    def test_rotate_verified_201_then_replay_200(self) -> None:
        payload = self._payload()
        status, body = self._rotate(self.predecessor["session_id"], payload)
        self.assertEqual(status, 201)
        self.assertEqual(set(body), _ROTATION_FIELDS)
        self.assertEqual(body["rotation_id"], "rot-1")
        status, fetched = self._request(
            "GET", f"/v1/group-sessions/{body['session_id']}")
        self.assertEqual(status, 200)
        self.assertEqual(fetched, {
            key: body[key]
            for key in ("session_id", "group_id", "initiator_device_id",
                        "ephemeral_key", "revision", "members",
                        "created_at")})
        status, replay = self._rotate(self.predecessor["session_id"],
                                      payload)
        self.assertEqual(status, 200)
        self.assertEqual(replay, body)

    def test_unknown_session_404(self) -> None:
        status, body = self._rotate("missing", self._payload())
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "session_id")

    def test_changed_replay_409_rotation_id(self) -> None:
        self.assertEqual(
            self._rotate(self.predecessor["session_id"],
                         self._payload())[0], 201)
        changed = self._payload(expected_version=2)
        status, body = self._rotate(self.predecessor["session_id"], changed)
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "rotation_id")

    def test_bad_signature_400(self) -> None:
        payload = self._payload()
        payload["signature"] = base64.b64encode(b"\x07" * 64).decode()
        status, body = self._rotate(self.predecessor["session_id"], payload)
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "signature")

    def test_missing_field_400(self) -> None:
        payload = self._payload()
        del payload["expected_version"]
        status, body = self._rotate(self.predecessor["session_id"], payload)
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "expected_version")


class GroupSessionRotationVerifiedPersistenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.path = tempfile.mktemp(suffix=".json")
        self.addCleanup(
            lambda: os.path.exists(self.path) and os.unlink(self.path))
        self.service, self.state_store = self._service(self.path)
        self.private, self.identity = _new_identity()
        self.service.register(_register_payload(
            "creator", identity_key=self.identity))
        self.service.register(_register_payload("alice", user_id="u2"))
        self.service.create_group({
            "group_id": "g1", "creator_device_id": "creator",
            "member_device_ids": ["alice"]})
        self.ephemeral_key = _x25519_b64()
        self.predecessor = self.service.create_group_session({
            "group_id": "g1", "initiator_device_id": "creator",
            "ephemeral_key": _x25519_b64()})

    def _service(self, path):
        service = DeviceService()
        store = attach_persistence(service, path)
        return service, store

    def _payload(self, rotation_id="rot-1", **overrides):
        body = {
            "rotation_id": rotation_id,
            "actor_device_id": "creator",
            "ephemeral_key": self.ephemeral_key,
            "expected_revision": 1,
            "expected_version": 1,
        }
        body.update(overrides)
        body["signature"] = _authorization(
            self.private, "g1", self.predecessor["session_id"], "u1",
            body["rotation_id"], body["actor_device_id"],
            body["ephemeral_key"], body["expected_revision"],
            body["expected_version"])
        return body

    def _document(self) -> dict:
        with open(self.path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _rewrite(self, document: dict) -> None:
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)

    def test_verified_rotation_survives_restart(self) -> None:
        first, status = self.service.rotate_group_session_verified(
            self.predecessor["session_id"], self._payload())
        self.assertEqual(status, 201)

        second_service, _ = self._service(self.path)
        replay, status = second_service.rotate_group_session_verified(
            self.predecessor["session_id"], self._payload())
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)
        # A changed field still conflicts after the restart.
        with self.assertRaises(ServiceError) as caught:
            second_service.rotate_group_session_verified(
                self.predecessor["session_id"],
                self._payload(expected_version=2))
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "rotation_id")

    def test_legacy_unsigned_record_still_recovers(self) -> None:
        first, status = self.service.rotate_group_session(
            self.predecessor["session_id"], {
                "rotation_id": "rot-legacy", "actor_device_id": "creator",
                "ephemeral_key": self.ephemeral_key,
                "expected_revision": 1})
        self.assertEqual(status, 201)
        # The on-disk record carries no authorization fields.
        (record,) = self._document()["group_session_rotations"]
        self.assertNotIn("identity_key", record)
        self.assertNotIn("expected_version", record)
        self.assertNotIn("signature", record)

        second_service, _ = self._service(self.path)
        replay, status = second_service.rotate_group_session(
            self.predecessor["session_id"], {
                "rotation_id": "rot-legacy", "actor_device_id": "creator",
                "ephemeral_key": self.ephemeral_key,
                "expected_revision": 1})
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)

    def test_verified_record_is_stored_frozen(self) -> None:
        self.service.rotate_group_session_verified(
            self.predecessor["session_id"], self._payload())
        (record,) = self._document()["group_session_rotations"]
        self.assertEqual(record["identity_key"], self.identity)
        self.assertEqual(record["expected_version"], 1)
        self.assertEqual(record["signature"], self._payload()["signature"])

    def test_tampered_signature_refuses_startup_without_overwrite(self) -> None:
        self.service.rotate_group_session_verified(
            self.predecessor["session_id"], self._payload())
        document = self._document()
        record = document["group_session_rotations"][0]
        record["signature"] = base64.b64encode(b"\x00" * 64).decode()
        self._rewrite(document)
        tampered = json.dumps(document, sort_keys=True)
        with self.assertRaises(StateFileError):
            self._service(self.path)
        # The refused file is left untouched.
        self.assertEqual(
            json.dumps(self._document(), sort_keys=True), tampered)

    def test_incomplete_verified_record_refuses_startup(self) -> None:
        self.service.rotate_group_session_verified(
            self.predecessor["session_id"], self._payload())
        document = self._document()
        del document["group_session_rotations"][0]["signature"]
        self._rewrite(document)
        with self.assertRaises(StateFileError):
            self._service(self.path)

    def test_tampered_frozen_identity_key_refuses_startup(self) -> None:
        self.service.rotate_group_session_verified(
            self.predecessor["session_id"], self._payload())
        document = self._document()
        _, other_identity = _new_identity()
        document["group_session_rotations"][0][
            "identity_key"] = other_identity
        self._rewrite(document)
        with self.assertRaises(StateFileError):
            self._service(self.path)

    def test_persist_failure_rolls_back_the_rotation(self) -> None:
        def fail_save(_pending) -> None:
            raise OSError("simulated disk failure")

        self.state_store.save = fail_save  # type: ignore[assignment]
        with self.assertRaises(PersistenceUnavailable):
            self.service.rotate_group_session_verified(
                self.predecessor["session_id"], self._payload())
        # Nothing was committed: the same rotation_id succeeds once the
        # state file is writable again.
        del self.state_store.save
        body, status = self.service.rotate_group_session_verified(
            self.predecessor["session_id"], self._payload())
        self.assertEqual(status, 201)
        self.assertEqual(body["rotation_id"], "rot-1")


class GroupSessionRotationVerifiedCLITest(unittest.TestCase):
    def setUp(self) -> None:
        self.server, self.service = create_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        self.private, self.identity = _new_identity()
        self.service.store.add_device(Device("u1", "creator", self.identity))
        self.service.store.add_device(Device("u2", "alice", "ik"))
        self.service.create_group({
            "group_id": "g1", "creator_device_id": "creator",
            "member_device_ids": ["alice"]})
        self.ephemeral_key = _x25519_b64()
        self.predecessor = self.service.create_group_session({
            "group_id": "g1", "initiator_device_id": "creator",
            "ephemeral_key": _x25519_b64()})

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _run(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "e2ee_backend", "--base-url",
             self.base_url, *arguments],
            capture_output=True, text=True, timeout=15)

    def _json(self, stream: str) -> dict:
        line = stream.strip()
        self.assertEqual(line.count("\n"), 0)
        return json.loads(line)

    def _args(self, rotation_id="rot-1", expected_version=1):
        signature = _authorization(
            self.private, "g1", self.predecessor["session_id"], "u1",
            rotation_id, "creator", self.ephemeral_key, 1, expected_version)
        return ("rotate-group-session-verified",
                self.predecessor["session_id"],
                "--rotation-id", rotation_id,
                "--actor-device-id", "creator",
                "--ephemeral-key", self.ephemeral_key,
                "--expected-revision", "1",
                "--expected-version", str(expected_version),
                "--signature", signature)

    def test_success_stdout_single_line_json(self) -> None:
        result = self._run(*self._args())
        self.assertEqual(result.returncode, 0, result.stderr)
        body = self._json(result.stdout)
        self.assertEqual(set(body), _ROTATION_FIELDS)
        self.assertEqual(body["rotation_id"], "rot-1")
        self.assertFalse(result.stderr.strip())
        # An identical replay also succeeds on stdout.
        replay = self._run(*self._args())
        self.assertEqual(replay.returncode, 0, replay.stderr)
        self.assertEqual(self._json(replay.stdout), body)

    def test_failure_stderr_single_line_json_nonzero(self) -> None:
        self.assertEqual(self._run(*self._args()).returncode, 0)
        result = self._run(*self._args(expected_version=2))
        self.assertEqual(result.returncode, 1)
        self.assertFalse(result.stdout.strip())
        self.assertEqual(self._json(result.stderr)["field"], "rotation_id")

    def test_unknown_session_stderr_nonzero(self) -> None:
        signature = _authorization(
            self.private, "g1", "missing", "u1", "rot-1", "creator",
            self.ephemeral_key, 1, 1)
        result = self._run(
            "rotate-group-session-verified", "missing",
            "--rotation-id", "rot-1", "--actor-device-id", "creator",
            "--ephemeral-key", self.ephemeral_key,
            "--expected-revision", "1", "--expected-version", "1",
            "--signature", signature)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(self._json(result.stderr)["field"], "session_id")


if __name__ == "__main__":
    unittest.main()
