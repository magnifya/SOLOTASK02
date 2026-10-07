"""Tests for signature-authorized group-session rotation.

Covers the service, HTTP (real loopback socket), CLI and persistence layers
for:

* POST /v1/group-sessions/{session_id}/rotate-verified
* the ``rotate-group-session-verified`` CLI subcommand

The rotation commits only when the standard-base64 64-byte Ed25519 signature
verifies over the domain-separated canonical rotation message
(``E2EE-GROUP-SESSION-ROTATION-V1``) against the creator device's *current*
identity key, ``expected_revision`` equals the group's current ``revision``
and ``expected_version`` equals the creator's current
``identity_key_version``. The first rotation is 201; an exact replay (same
``rotation_id`` on the same predecessor with identical fields and signature)
is 200 with the original response even after later state changes, while the
same id with any field or the signature changed — or naming another
predecessor — is 409/field=rotation_id. The committed record freezes the
creator's identity key, the authorized version and the signature; legacy
unsigned records still recover, and an incomplete or non-verifying signed
record refuses startup without overwriting the file.
"""
import base64
import json
import os
import shutil
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
    PersistenceUnavailable,
    StateFileError,
    attach_persistence,
)
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


def _authorization(private, predecessor_session_id, rotation_id="rot-1",
                   actor="creator", ephemeral_key=None, expected_revision=1,
                   expected_version=1, user_id="u1", group_id="g1") -> str:
    message = group_session_rotation_proof_message(
        user_id, group_id, predecessor_session_id, rotation_id, actor,
        ephemeral_key, expected_revision, expected_version)
    return base64.b64encode(private.sign(message)).decode()


_ROTATION_FIELDS = {
    "session_id", "group_id", "initiator_device_id", "ephemeral_key",
    "revision", "members", "created_at",
    "rotation_id", "predecessor_session_id",
}


class _RotationServiceBase(unittest.TestCase):
    """Shared setup: an Ed25519 creator, two members, one group session."""

    def setUp(self) -> None:
        self.service = DeviceService()
        self.private, self.identity = _new_identity()
        self.service.store.add_device(Device("u1", "creator", self.identity))
        self.service.store.add_device(Device("u2", "alice", _x25519_b64()))
        self.service.store.add_device(Device("u3", "bob", _x25519_b64()))
        self.service.create_group({
            "group_id": "g1", "creator_device_id": "creator",
            "member_device_ids": ["alice"]})
        self.ephemeral_key = _x25519_b64()
        self.predecessor = self.service.create_group_session({
            "group_id": "g1", "initiator_device_id": "creator",
            "ephemeral_key": _x25519_b64()})
        self.predecessor_id = self.predecessor["session_id"]

    def _payload(self, predecessor_id=None, rotation_id="rot-1",
                 actor="creator", ephemeral_key=None, expected_revision=1,
                 expected_version=1, signature=None, private=None,
                 group_id="g1") -> dict:
        predecessor_id = predecessor_id or self.predecessor_id
        ephemeral_key = ephemeral_key or self.ephemeral_key
        if signature is None:
            signature = _authorization(
                private or self.private, predecessor_id,
                rotation_id=rotation_id, actor=actor,
                ephemeral_key=ephemeral_key,
                expected_revision=expected_revision,
                expected_version=expected_version, group_id=group_id)
        return {"rotation_id": rotation_id, "actor_device_id": actor,
                "ephemeral_key": ephemeral_key,
                "expected_revision": expected_revision,
                "expected_version": expected_version,
                "signature": signature}


class GroupRotationVerifiedServiceTest(_RotationServiceBase):
    def test_first_rotation_201_nine_fields(self) -> None:
        body, status = self.service.rotate_group_session_verified(
            self.predecessor_id, self._payload())
        self.assertEqual(status, 201)
        self.assertEqual(set(body), _ROTATION_FIELDS)
        self.assertEqual(body["rotation_id"], "rot-1")
        self.assertEqual(body["predecessor_session_id"], self.predecessor_id)
        self.assertNotEqual(body["session_id"], self.predecessor_id)
        self.assertEqual(body["group_id"], "g1")
        self.assertEqual(body["initiator_device_id"], "creator")
        self.assertEqual(body["ephemeral_key"], self.ephemeral_key)
        self.assertEqual(body["revision"], 1)
        self.assertEqual(body["members"], ["creator", "alice"])

    def test_exact_replay_200_original_after_state_change(self) -> None:
        first, status = self.service.rotate_group_session_verified(
            self.predecessor_id, self._payload())
        self.assertEqual(status, 201)
        # State moves on: a member joins (revision 2) and the creator
        # rotates its identity key (identity_key_version 2).
        self.service.add_group_member("g1", {
            "actor_device_id": "creator", "device_id": "bob"})
        new_private, new_key = _new_identity()
        message = identity_rotation_proof_message("u1", "creator", new_key, 1)
        self.service.rotate_identity_key_verified("creator", {
            "identity_key": new_key, "expected_version": 1,
            "signature": base64.b64encode(
                self.private.sign(message)).decode()})
        # The exact original request still replays to the original response.
        second, status = self.service.rotate_group_session_verified(
            self.predecessor_id, self._payload())
        self.assertEqual(status, 200)
        self.assertEqual(second, first)

    def test_replay_changed_field_409_rotation_id(self) -> None:
        self.service.rotate_group_session_verified(
            self.predecessor_id, self._payload())
        # Same id and predecessor, but a different (validly signed) field.
        other_key = _x25519_b64()
        with self.assertRaises(ServiceError) as ctx:
            self.service.rotate_group_session_verified(
                self.predecessor_id, self._payload(ephemeral_key=other_key))
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "rotation_id")
        # A changed expected_version is a conflict too.
        with self.assertRaises(ServiceError) as ctx:
            self.service.rotate_group_session_verified(
                self.predecessor_id, self._payload(expected_version=2))
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "rotation_id")
        # A changed signature over the same fields is a conflict as well.
        other_private, _ = _new_identity()
        with self.assertRaises(ServiceError) as ctx:
            self.service.rotate_group_session_verified(
                self.predecessor_id, self._payload(private=other_private))
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "rotation_id")

    def test_rotation_id_for_other_predecessor_409(self) -> None:
        self.service.rotate_group_session_verified(
            self.predecessor_id, self._payload())
        other = self.service.create_group_session({
            "group_id": "g1", "initiator_device_id": "creator",
            "ephemeral_key": _x25519_b64()})
        with self.assertRaises(ServiceError) as ctx:
            self.service.rotate_group_session_verified(
                other["session_id"],
                self._payload(predecessor_id=other["session_id"]))
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "rotation_id")

    def test_predecessor_already_rotated_409_session_id(self) -> None:
        self.service.rotate_group_session_verified(
            self.predecessor_id, self._payload())
        with self.assertRaises(ServiceError) as ctx:
            self.service.rotate_group_session_verified(
                self.predecessor_id, self._payload(rotation_id="rot-2"))
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "session_id")

    def test_unknown_predecessor_404(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.rotate_group_session_verified(
                "ghost", self._payload(predecessor_id="ghost"))
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "session_id")

    def test_unknown_actor_404(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.rotate_group_session_verified(
                self.predecessor_id, self._payload(actor="ghost"))
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "actor_device_id")

    def test_revoked_actor_409(self) -> None:
        self.service.store.revoke_device("creator")
        with self.assertRaises(ServiceError) as ctx:
            self.service.rotate_group_session_verified(
                self.predecessor_id, self._payload())
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "actor_device_id")

    def test_non_creator_actor_409(self) -> None:
        alice_private, alice_key = _new_identity()
        self.service.store.add_device(Device("u4", "carol", alice_key))
        self.service.add_group_member("g1", {
            "actor_device_id": "creator", "device_id": "carol"})
        with self.assertRaises(ServiceError) as ctx:
            self.service.rotate_group_session_verified(
                self.predecessor_id,
                self._payload(actor="carol", private=alice_private,
                              expected_revision=2))
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "actor_device_id")

    def test_actor_key_not_ed25519_400_identity_key(self) -> None:
        service = DeviceService()
        service.store.add_device(
            Device("u1", "creator", _x25519_der_b64()))
        service.store.add_device(Device("u2", "alice", _x25519_b64()))
        service.create_group({"group_id": "g1",
                              "creator_device_id": "creator",
                              "member_device_ids": ["alice"]})
        predecessor = service.create_group_session({
            "group_id": "g1", "initiator_device_id": "creator",
            "ephemeral_key": _x25519_b64()})
        payload = self._payload(predecessor_id=predecessor["session_id"])
        with self.assertRaises(ServiceError) as ctx:
            service.rotate_group_session_verified(
                predecessor["session_id"], payload)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.field, "identity_key")

    def test_revision_mismatch_409(self) -> None:
        for bad_revision in (2, 3, 99):
            payload = self._payload(expected_revision=bad_revision)
            with self.subTest(bad_revision=bad_revision):
                with self.assertRaises(ServiceError) as ctx:
                    self.service.rotate_group_session_verified(
                        self.predecessor_id, payload)
                self.assertEqual(ctx.exception.status_code, 409)
                self.assertEqual(ctx.exception.field, "expected_revision")

    def test_version_mismatch_409(self) -> None:
        for bad_version in (2, 3, 99):
            payload = self._payload(expected_version=bad_version)
            with self.subTest(bad_version=bad_version):
                with self.assertRaises(ServiceError) as ctx:
                    self.service.rotate_group_session_verified(
                        self.predecessor_id, payload)
                self.assertEqual(ctx.exception.status_code, 409)
                self.assertEqual(ctx.exception.field, "expected_version")

    def test_bad_signature_400(self) -> None:
        other_private, _ = _new_identity()
        cases = [
            # Signed by a different key.
            self._payload(private=other_private),
            # Signed for a different rotation id.
            self._payload(signature=_authorization(
                self.private, self.predecessor_id, rotation_id="rot-9",
                ephemeral_key=self.ephemeral_key)),
            # Signed for a different predecessor.
            self._payload(signature=_authorization(
                self.private, "someone-else",
                ephemeral_key=self.ephemeral_key)),
            # Random 64 bytes.
            self._payload(signature=base64.b64encode(b"\x00" * 64).decode()),
        ]
        for payload in cases:
            with self.subTest(payload=payload):
                with self.assertRaises(ServiceError) as ctx:
                    self.service.rotate_group_session_verified(
                        self.predecessor_id, payload)
                self.assertEqual(ctx.exception.status_code, 400)
                self.assertEqual(ctx.exception.field, "signature")

    def test_check_order(self) -> None:
        # Unknown predecessor beats a bad revision: 404/session_id.
        with self.assertRaises(ServiceError) as ctx:
            self.service.rotate_group_session_verified(
                "ghost", self._payload(predecessor_id="ghost",
                                       expected_revision=9))
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "session_id")
        # Unknown actor beats a revision mismatch: 404/actor_device_id.
        with self.assertRaises(ServiceError) as ctx:
            self.service.rotate_group_session_verified(
                self.predecessor_id,
                self._payload(actor="ghost", expected_revision=9))
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "actor_device_id")
        # Revision mismatch beats a version mismatch: 409/expected_revision.
        with self.assertRaises(ServiceError) as ctx:
            self.service.rotate_group_session_verified(
                self.predecessor_id,
                self._payload(expected_revision=9, expected_version=9))
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "expected_revision")
        # Version mismatch beats a bad signature: 409/expected_version.
        payload = self._payload(
            expected_version=9,
            signature=base64.b64encode(b"\x00" * 64).decode())
        with self.assertRaises(ServiceError) as ctx:
            self.service.rotate_group_session_verified(
                self.predecessor_id, payload)
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "expected_version")

    # -- request-body validation -------------------------------------------

    def test_body_not_object_400_request_body(self) -> None:
        for payload in (None, [], "x", 42):
            with self.subTest(payload=payload):
                with self.assertRaises(ServiceError) as ctx:
                    self.service.rotate_group_session_verified(
                        self.predecessor_id, payload)
                self.assertEqual(ctx.exception.status_code, 400)
                self.assertEqual(ctx.exception.field, "request_body")

    def test_missing_fields_400(self) -> None:
        good = self._payload()
        for name in ("rotation_id", "actor_device_id", "ephemeral_key",
                     "expected_revision", "expected_version", "signature"):
            payload = dict(good)
            del payload[name]
            with self.subTest(missing=name):
                with self.assertRaises(ServiceError) as ctx:
                    self.service.rotate_group_session_verified(
                        self.predecessor_id, payload)
                self.assertEqual(ctx.exception.status_code, 400)
                self.assertEqual(ctx.exception.field, name)

    def test_bad_identifiers_400(self) -> None:
        for name in ("rotation_id", "actor_device_id", "ephemeral_key"):
            for bad in (None, "", 7, [], {}):
                payload = self._payload()
                payload[name] = bad
                with self.subTest(name=name, bad=bad):
                    with self.assertRaises(ServiceError) as ctx:
                        self.service.rotate_group_session_verified(
                            self.predecessor_id, payload)
                    self.assertEqual(ctx.exception.status_code, 400)
                    self.assertEqual(ctx.exception.field, name)

    def test_bad_ephemeral_key_400(self) -> None:
        payload = self._payload()
        payload["ephemeral_key"] = "not-a-key"
        with self.assertRaises(ServiceError) as ctx:
            self.service.rotate_group_session_verified(
                self.predecessor_id, payload)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.field, "ephemeral_key")

    def test_bad_expected_values_400(self) -> None:
        for name in ("expected_revision", "expected_version"):
            for bad in (None, True, False, 0, -1, "1", 1.5, [], {}):
                payload = self._payload()
                payload[name] = bad
                with self.subTest(name=name, bad=bad):
                    with self.assertRaises(ServiceError) as ctx:
                        self.service.rotate_group_session_verified(
                            self.predecessor_id, payload)
                    self.assertEqual(ctx.exception.status_code, 400)
                    self.assertEqual(ctx.exception.field, name)

    def test_bad_signature_encoding_400(self) -> None:
        for bad in (None, "", 8, [], "@@@@", "abc", "a" * 88,
                    base64.b64encode(b"\x00" * 63).decode(),
                    base64.b64encode(b"\x00" * 65).decode()):
            payload = self._payload()
            payload["signature"] = bad
            with self.subTest(bad=bad):
                with self.assertRaises(ServiceError) as ctx:
                    self.service.rotate_group_session_verified(
                        self.predecessor_id, payload)
                self.assertEqual(ctx.exception.status_code, 400)
                self.assertEqual(ctx.exception.field, "signature")

    def test_signature_must_be_canonical_base64(self) -> None:
        # A 64-byte value whose canonical encoding always contains '+' and
        # '/', so the URL-safe rewrite below is never a no-op.
        good = base64.b64encode(b"\xfb" * 64).decode()
        assert "+" in good and "/" in good
        for bad in (good.replace("+", "-").replace("/", "_"),
                    good.rstrip("=")):
            with self.subTest(bad=bad):
                payload = self._payload()
                payload["signature"] = bad
                with self.assertRaises(ServiceError) as ctx:
                    self.service.rotate_group_session_verified(
                        self.predecessor_id, payload)
                self.assertEqual(ctx.exception.status_code, 400)
                self.assertEqual(ctx.exception.field, "signature")

    def test_failure_writes_nothing(self) -> None:
        with self.assertRaises(ServiceError):
            self.service.rotate_group_session_verified(
                self.predecessor_id,
                self._payload(
                    signature=base64.b64encode(b"\x00" * 64).decode()))
        self.assertEqual(self.service.store.snapshot_state()[
            "group_session_rotations"], [])
        # The failed attempt consumed neither the id nor the predecessor.
        body, status = self.service.rotate_group_session_verified(
            self.predecessor_id, self._payload())
        self.assertEqual(status, 201)

    def test_signature_uses_current_identity_key_and_version(self) -> None:
        # Rotate the creator's identity key: authorizations signed by the
        # old key stop verifying and the version moves to 2.
        new_private, new_key = _new_identity()
        message = identity_rotation_proof_message("u1", "creator", new_key, 1)
        self.service.rotate_identity_key_verified("creator", {
            "identity_key": new_key, "expected_version": 1,
            "signature": base64.b64encode(
                self.private.sign(message)).decode()})
        # The old expected_version is now stale.
        with self.assertRaises(ServiceError) as ctx:
            self.service.rotate_group_session_verified(
                self.predecessor_id, self._payload())
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "expected_version")
        # And the old key's authorization no longer verifies, even naming
        # the current version.
        with self.assertRaises(ServiceError) as ctx:
            self.service.rotate_group_session_verified(
                self.predecessor_id, self._payload(expected_version=2))
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.field, "signature")
        # The new key with the new version authorizes the rotation.
        body, status = self.service.rotate_group_session_verified(
            self.predecessor_id,
            self._payload(private=new_private, expected_version=2))
        self.assertEqual(status, 201)


class GroupRotationVerifiedHTTPTest(unittest.TestCase):
    def setUp(self) -> None:
        self.server, self.service = create_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        self.private, self.identity = _new_identity()
        self.service.store.add_device(Device("u1", "creator", self.identity))
        self.service.store.add_device(Device("u2", "alice", _x25519_b64()))
        self.service.create_group({
            "group_id": "g1", "creator_device_id": "creator",
            "member_device_ids": ["alice"]})
        self.ephemeral_key = _x25519_b64()
        self.predecessor = self.service.create_group_session({
            "group_id": "g1", "initiator_device_id": "creator",
            "ephemeral_key": _x25519_b64()})
        self.predecessor_id = self.predecessor["session_id"]

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _request(self, method: str, path: str, body: object = None,
                 raw: str | None = None):
        connection = HTTPConnection("127.0.0.1", self.port, timeout=5)
        payload = raw if raw is not None else (
            json.dumps(body) if body is not None else None)
        headers = {"Content-Type": "application/json"} if payload else {}
        connection.request(method, path, body=payload, headers=headers)
        response = connection.getresponse()
        data = response.read().decode("utf-8")
        connection.close()
        return response.status, json.loads(data)

    def _payload(self, **overrides) -> dict:
        payload = {"rotation_id": "rot-1", "actor_device_id": "creator",
                   "ephemeral_key": self.ephemeral_key,
                   "expected_revision": 1, "expected_version": 1}
        payload.update(overrides)
        if "signature" not in payload:
            payload["signature"] = _authorization(
                self.private, self.predecessor_id,
                rotation_id=payload["rotation_id"],
                actor=payload["actor_device_id"],
                ephemeral_key=payload["ephemeral_key"],
                expected_revision=payload["expected_revision"],
                expected_version=payload["expected_version"])
        return payload

    def _rotate_verified(self, session_id: str, payload: dict):
        return self._request(
            "POST", f"/v1/group-sessions/{session_id}/rotate-verified",
            payload)

    def test_201_then_exact_replay_200(self) -> None:
        status, body = self._rotate_verified(
            self.predecessor_id, self._payload())
        self.assertEqual(status, 201)
        self.assertEqual(set(body), _ROTATION_FIELDS)
        self.assertEqual(body["predecessor_session_id"], self.predecessor_id)
        status, second = self._rotate_verified(
            self.predecessor_id, self._payload())
        self.assertEqual(status, 200)
        self.assertEqual(second, body)
        # The successor is an ordinary gettable group session.
        status, fetched = self._request(
            "GET", f"/v1/group-sessions/{body['session_id']}")
        self.assertEqual(status, 200)
        self.assertEqual(fetched, {
            key: body[key]
            for key in ("session_id", "group_id", "initiator_device_id",
                        "ephemeral_key", "revision", "members",
                        "created_at")})

    def test_invalid_json_400_request_body(self) -> None:
        status, body = self._request(
            "POST",
            f"/v1/group-sessions/{self.predecessor_id}/rotate-verified",
            raw="{not json")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "request_body")

    def test_missing_signature_400(self) -> None:
        payload = self._payload()
        del payload["signature"]
        status, body = self._rotate_verified(self.predecessor_id, payload)
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "signature")

    def test_bool_expected_version_400(self) -> None:
        status, body = self._rotate_verified(
            self.predecessor_id, self._payload(expected_version=True))
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "expected_version")

    def test_unknown_predecessor_404(self) -> None:
        status, body = self._rotate_verified("ghost", self._payload())
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "session_id")

    def test_revision_mismatch_409(self) -> None:
        status, body = self._rotate_verified(
            self.predecessor_id, self._payload(expected_revision=7))
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "expected_revision")

    def test_version_mismatch_409(self) -> None:
        status, body = self._rotate_verified(
            self.predecessor_id, self._payload(expected_version=7))
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "expected_version")

    def test_bad_signature_400(self) -> None:
        status, body = self._rotate_verified(
            self.predecessor_id,
            self._payload(
                signature=base64.b64encode(b"\x00" * 64).decode()))
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "signature")
        # The failed attempt wrote nothing: the correct payload still wins.
        status, _ = self._rotate_verified(self.predecessor_id,
                                          self._payload())
        self.assertEqual(status, 201)

    def test_changed_field_replay_409_rotation_id(self) -> None:
        status, _ = self._rotate_verified(self.predecessor_id,
                                          self._payload())
        self.assertEqual(status, 201)
        status, body = self._rotate_verified(
            self.predecessor_id,
            self._payload(ephemeral_key=_x25519_b64()))
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "rotation_id")

    def test_unsigned_rotate_route_still_works(self) -> None:
        status, body = self._request(
            "POST", f"/v1/group-sessions/{self.predecessor_id}/rotate", {
                "rotation_id": "rot-1", "actor_device_id": "creator",
                "ephemeral_key": _x25519_b64(), "expected_revision": 1})
        self.assertEqual(status, 201)
        self.assertEqual(set(body), _ROTATION_FIELDS)


class GroupRotationVerifiedPersistenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.directory, True)
        self.path = os.path.join(self.directory, "state.json")
        self.service = self._service(self.path)
        self.private, self.identity = _new_identity()
        self.service.store.add_device(Device("u1", "creator", self.identity))
        self.service.store.add_device(Device("u2", "alice", _x25519_b64()))
        self.service.create_group({
            "group_id": "g1", "creator_device_id": "creator",
            "member_device_ids": ["alice"]})
        self.ephemeral_key = _x25519_b64()
        self.predecessor = self.service.create_group_session({
            "group_id": "g1", "initiator_device_id": "creator",
            "ephemeral_key": _x25519_b64()})
        self.predecessor_id = self.predecessor["session_id"]

    @staticmethod
    def _service(path: str) -> DeviceService:
        service = DeviceService()
        attach_persistence(service, path)
        return service

    def _payload(self, rotation_id="rot-1") -> dict:
        return {"rotation_id": rotation_id, "actor_device_id": "creator",
                "ephemeral_key": self.ephemeral_key,
                "expected_revision": 1, "expected_version": 1,
                "signature": _authorization(
                    self.private, self.predecessor_id,
                    rotation_id=rotation_id,
                    ephemeral_key=self.ephemeral_key)}

    def _document(self) -> dict:
        with open(self.path, encoding="utf-8") as handle:
            return json.load(handle)

    def _assert_rejected(self, document) -> None:
        path = os.path.join(self.directory, "bad.json")
        # No sidecar accompanies this hand-built bad file; drop the marker so
        # refusal comes from the malformed content the test is exercising.
        document.pop("integrity_log_version", None)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        before = open(path, "rb").read()
        with self.assertRaises(StateFileError):
            attach_persistence(DeviceService(), path)
        # A rejected file must never be overwritten at startup.
        self.assertEqual(open(path, "rb").read(), before)

    def test_signed_record_shape_and_restart_replay(self) -> None:
        rotated, status = self.service.rotate_group_session_verified(
            self.predecessor_id, self._payload())
        self.assertEqual(status, 201)
        record = self._document()["group_session_rotations"][0]
        # The durable record freezes the creator's identity key, the
        # authorized version and the signature.
        self.assertEqual(record["identity_key"], self.identity)
        self.assertEqual(record["expected_version"], 1)
        self.assertEqual(record["signature"],
                         self._payload()["signature"])

        second = self._service(self.path)
        body, status = second.rotate_group_session_verified(
            self.predecessor_id, self._payload())
        self.assertEqual(status, 200)
        self.assertEqual(body, rotated)
        # The no-fork rule survived the restart.
        with self.assertRaises(ServiceError) as ctx:
            second.rotate_group_session_verified(
                self.predecessor_id, self._payload(rotation_id="rot-2"))
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "session_id")

    def test_legacy_unsigned_record_still_recovers(self) -> None:
        rotated, status = self.service.rotate_group_session(
            self.predecessor_id, {
                "rotation_id": "rot-1", "actor_device_id": "creator",
                "ephemeral_key": _x25519_b64(), "expected_revision": 1})
        self.assertEqual(status, 201)
        record = self._document()["group_session_rotations"][0]
        self.assertNotIn("identity_key", record)
        self.assertNotIn("expected_version", record)
        self.assertNotIn("signature", record)
        second = self._service(self.path)
        body, status = second.rotate_group_session(
            self.predecessor_id, {
                "rotation_id": "rot-1", "actor_device_id": "creator",
                "ephemeral_key": _x25519_b64(), "expected_revision": 1})
        self.assertEqual(status, 200)
        self.assertEqual(body, rotated)

    def test_tampered_signature_rejected(self) -> None:
        self.service.rotate_group_session_verified(
            self.predecessor_id, self._payload())
        document = self._document()
        record = document["group_session_rotations"][0]
        # A different validly-encoded 64-byte signature no longer verifies.
        record["signature"] = base64.b64encode(b"\x01" * 64).decode()
        self._assert_rejected(document)

    def test_incomplete_signed_record_rejected(self) -> None:
        self.service.rotate_group_session_verified(
            self.predecessor_id, self._payload())
        for dropped in ("identity_key", "expected_version", "signature"):
            document = self._document()
            record = document["group_session_rotations"][0]
            del record[dropped]
            with self.subTest(dropped=dropped):
                self._assert_rejected(document)

    def test_malformed_signed_fields_rejected(self) -> None:
        self.service.rotate_group_session_verified(
            self.predecessor_id, self._payload())
        bad_values = [
            {"identity_key": ""},
            {"identity_key": "not-a-key"},
            {"expected_version": 0},
            {"expected_version": True},
            {"expected_version": "1"},
            {"signature": "not-base64!"},
            {"signature": base64.b64encode(b"\x00" * 63).decode()},
            # A well-formed record signed for another rotation id.
            {"rotation_id": "rot-2"},
        ]
        for mutation in bad_values:
            document = self._document()
            document["group_session_rotations"][0].update(mutation)
            with self.subTest(mutation=mutation):
                self._assert_rejected(document)

    def test_persist_failure_rolls_back(self) -> None:
        service = DeviceService()
        store = attach_persistence(service, self.path)
        private, identity = _new_identity()
        service.store.add_device(Device("u1", "creator2", identity))
        service.create_group({"group_id": "g2",
                              "creator_device_id": "creator2",
                              "member_device_ids": ["creator2"]})
        ephemeral = _x25519_b64()
        predecessor = service.create_group_session({
            "group_id": "g2", "initiator_device_id": "creator2",
            "ephemeral_key": _x25519_b64()})

        def fail_save(_pending) -> None:
            raise OSError("simulated disk failure")

        store.save = fail_save  # type: ignore[assignment]
        payload = {"rotation_id": "rot-9", "actor_device_id": "creator2",
                   "ephemeral_key": ephemeral, "expected_revision": 1,
                   "expected_version": 1,
                   "signature": _authorization(
                       private, predecessor["session_id"],
                       rotation_id="rot-9", actor="creator2",
                       ephemeral_key=ephemeral, group_id="g2")}
        with self.assertRaises(PersistenceUnavailable):
            service.rotate_group_session_verified(
                predecessor["session_id"], payload)
        # Nothing was committed: the rotation id and predecessor are free.
        self.assertEqual(service.store.snapshot_state()[
            "group_session_rotations"], [])
        del store.save
        _, status = service.rotate_group_session_verified(
            predecessor["session_id"], payload)
        self.assertEqual(status, 201)


class GroupRotationVerifiedCLITest(unittest.TestCase):
    def setUp(self) -> None:
        self.server, self.service = create_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        self.private, self.identity = _new_identity()
        self.service.store.add_device(Device("u1", "creator", self.identity))
        self.service.store.add_device(Device("u2", "alice", _x25519_b64()))
        self.service.create_group({
            "group_id": "g1", "creator_device_id": "creator",
            "member_device_ids": ["alice"]})
        self.ephemeral_key = _x25519_b64()
        self.predecessor = self.service.create_group_session({
            "group_id": "g1", "initiator_device_id": "creator",
            "ephemeral_key": _x25519_b64()})
        self.predecessor_id = self.predecessor["session_id"]

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

    def _args(self, signature: str) -> tuple:
        return ("rotate-group-session-verified", self.predecessor_id,
                "--rotation-id", "rot-1",
                "--actor-device-id", "creator",
                "--ephemeral-key", self.ephemeral_key,
                "--expected-revision", "1",
                "--expected-version", "1",
                "--signature", signature)

    def test_success_stdout_single_line_json(self) -> None:
        signature = _authorization(
            self.private, self.predecessor_id,
            ephemeral_key=self.ephemeral_key)
        result = self._run(*self._args(signature))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(result.stderr.strip())
        body = self._json(result.stdout)
        self.assertEqual(set(body), _ROTATION_FIELDS)
        self.assertEqual(body["rotation_id"], "rot-1")
        self.assertEqual(body["predecessor_session_id"], self.predecessor_id)
        # An exact replay is also a success on stdout.
        replay = self._run(*self._args(signature))
        self.assertEqual(replay.returncode, 0, replay.stderr)
        self.assertEqual(self._json(replay.stdout), body)

    def test_failure_stderr_single_line_json_nonzero(self) -> None:
        bad = base64.b64encode(b"\x00" * 64).decode()
        result = self._run(*self._args(bad))
        self.assertEqual(result.returncode, 1)
        self.assertFalse(result.stdout.strip())
        self.assertEqual(self._json(result.stderr)["field"], "signature")


if __name__ == "__main__":
    unittest.main()
