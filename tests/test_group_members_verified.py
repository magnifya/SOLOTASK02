"""Tests for signature-authorized group membership changes.

Covers the service, HTTP (real loopback socket) and persistence layers for:

* POST /v1/groups/{group_id}/members/verified

The change applies only when the standard-base64 64-byte Ed25519 signature
verifies over the domain-separated canonical membership message
(``E2EE-GROUP-MEMBERSHIP-V1``) against the creator device's *current*
identity key, and ``expected_revision`` equals the group's current
``revision``. An ``add`` of a new member is 201 and advances the revision,
a duplicate add is 200; a ``remove`` of a current member is 200 and
advances the revision, removing a non-member or the creator is a 200
no-op. The unsigned member entries, frozen group sessions and sync
delivery keep their old behavior.
"""
import base64
import json
import os
import tempfile
import threading
import unittest
from http.client import HTTPConnection

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, x25519

from e2ee_backend.crypto import (
    group_membership_proof_message,
    identity_rotation_proof_message,
)
from e2ee_backend.http_app import create_server
from e2ee_backend.persistence import PersistenceUnavailable, attach_persistence
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


def _authorization(private, operation, actor, target, revision,
                   group_id="g1") -> str:
    message = group_membership_proof_message(
        group_id, operation, actor, target, revision)
    return base64.b64encode(private.sign(message)).decode()


def _register_payload(device_id, user_id="u1", identity_key=None) -> dict:
    return {
        "user_id": user_id,
        "device_id": device_id,
        "identity_key": identity_key or _x25519_b64(),
        "signed_prekeys": [{"key_id": "k1", "public_key": _x25519_b64()}],
    }


class _GroupServiceBase(unittest.TestCase):
    """Shared setup: a creator with an Ed25519 identity and two devices."""

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

    def _payload(self, operation="add", actor="creator", target="bob",
                 revision=1, signature=None, private=None, group_id="g1"):
        if signature is None:
            signature = _authorization(
                private or self.private, operation, actor, target, revision,
                group_id=group_id)
        return {"operation": operation, "actor_device_id": actor,
                "device_id": target, "expected_revision": revision,
                "signature": signature}

    def _group(self):
        return self.service.get_group("g1")


class GroupMembersVerifiedServiceTest(_GroupServiceBase):
    def test_add_new_member_201_advances_revision(self) -> None:
        body, status = self.service.change_group_member_verified(
            "g1", self._payload())
        self.assertEqual(status, 201)
        self.assertEqual(set(body),
                         {"group_id", "revision", "members", "created_at"})
        self.assertEqual(body["members"], ["creator", "alice", "bob"])
        self.assertEqual(body["revision"], 2)

    def test_add_duplicate_member_200_no_revision_change(self) -> None:
        body, status = self.service.change_group_member_verified(
            "g1", self._payload(target="alice", revision=1))
        self.assertEqual(status, 200)
        self.assertEqual(body["members"], ["creator", "alice"])
        self.assertEqual(body["revision"], 1)

    def test_remove_member_200_advances_revision(self) -> None:
        body, status = self.service.change_group_member_verified(
            "g1", self._payload(operation="remove", target="alice",
                                revision=1))
        self.assertEqual(status, 200)
        self.assertEqual(body["members"], ["creator"])
        self.assertEqual(body["revision"], 2)

    def test_remove_non_member_200_noop(self) -> None:
        body, status = self.service.change_group_member_verified(
            "g1", self._payload(operation="remove", target="bob",
                                revision=1))
        self.assertEqual(status, 200)
        self.assertEqual(body["members"], ["creator", "alice"])
        self.assertEqual(body["revision"], 1)

    def test_remove_creator_200_noop(self) -> None:
        body, status = self.service.change_group_member_verified(
            "g1", self._payload(operation="remove", target="creator",
                                revision=1))
        self.assertEqual(status, 200)
        self.assertEqual(body["members"], ["creator", "alice"])
        self.assertEqual(body["revision"], 1)

    def test_remove_revoked_member_allowed(self) -> None:
        self.service.revoke_device("alice")
        body, status = self.service.change_group_member_verified(
            "g1", self._payload(operation="remove", target="alice",
                                revision=1))
        self.assertEqual(status, 200)
        self.assertEqual(body["members"], ["creator"])
        self.assertEqual(body["revision"], 2)

    def test_add_revoked_non_member_409(self) -> None:
        self.service.revoke_device("bob")
        with self.assertRaises(ServiceError) as ctx:
            self.service.change_group_member_verified("g1", self._payload())
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "device_id")

    def test_add_revoked_existing_member_200(self) -> None:
        self.service.revoke_device("alice")
        body, status = self.service.change_group_member_verified(
            "g1", self._payload(target="alice", revision=1))
        self.assertEqual(status, 200)
        self.assertEqual(body["revision"], 1)

    # -- request-body validation -------------------------------------------

    def test_body_not_object_400_request_body(self) -> None:
        for payload in (None, [], "x", 42):
            with self.subTest(payload=payload):
                with self.assertRaises(ServiceError) as ctx:
                    self.service.change_group_member_verified("g1", payload)
                self.assertEqual(ctx.exception.status_code, 400)
                self.assertEqual(ctx.exception.field, "request_body")

    def test_missing_fields_400(self) -> None:
        good = self._payload()
        for name in ("operation", "actor_device_id", "device_id",
                     "expected_revision", "signature"):
            payload = dict(good)
            del payload[name]
            with self.subTest(missing=name):
                with self.assertRaises(ServiceError) as ctx:
                    self.service.change_group_member_verified("g1", payload)
                self.assertEqual(ctx.exception.status_code, 400)
                self.assertEqual(ctx.exception.field, name)

    def test_bad_operation_400(self) -> None:
        for bad in (None, "", "kick", "ADD", 1, True, [], {}):
            payload = self._payload()
            payload["operation"] = bad
            with self.subTest(bad=bad):
                with self.assertRaises(ServiceError) as ctx:
                    self.service.change_group_member_verified("g1", payload)
                self.assertEqual(ctx.exception.status_code, 400)
                self.assertEqual(ctx.exception.field, "operation")

    def test_bad_actor_or_device_id_400(self) -> None:
        for name in ("actor_device_id", "device_id"):
            for bad in (None, "", 7, [], {}):
                payload = self._payload()
                payload[name] = bad
                with self.subTest(name=name, bad=bad):
                    with self.assertRaises(ServiceError) as ctx:
                        self.service.change_group_member_verified(
                            "g1", payload)
                    self.assertEqual(ctx.exception.status_code, 400)
                    self.assertEqual(ctx.exception.field, name)

    def test_bad_expected_revision_400(self) -> None:
        for bad in (None, True, False, 0, -1, "1", 1.5, [], {}):
            payload = self._payload()
            payload["expected_revision"] = bad
            with self.subTest(bad=bad):
                with self.assertRaises(ServiceError) as ctx:
                    self.service.change_group_member_verified("g1", payload)
                self.assertEqual(ctx.exception.status_code, 400)
                self.assertEqual(ctx.exception.field, "expected_revision")

    def test_bad_signature_encoding_400(self) -> None:
        for bad in (None, "", 8, [], "@@@@", "abc", "a" * 88,
                    base64.b64encode(b"\x00" * 63).decode(),
                    base64.b64encode(b"\x00" * 65).decode()):
            payload = self._payload()
            payload["signature"] = bad
            with self.subTest(bad=bad):
                with self.assertRaises(ServiceError) as ctx:
                    self.service.change_group_member_verified("g1", payload)
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
                    self.service.change_group_member_verified("g1", payload)
                self.assertEqual(ctx.exception.field, "signature")

    def test_extra_fields_are_ignored(self) -> None:
        payload = self._payload()
        payload["extra"] = "ignored"
        payload["identity_key"] = self.identity
        body, status = self.service.change_group_member_verified(
            "g1", payload)
        self.assertEqual(status, 201)
        self.assertEqual(set(body),
                         {"group_id", "revision", "members", "created_at"})

    # -- locked checks ------------------------------------------------------

    def test_unknown_group_404(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.change_group_member_verified(
                "ghost", self._payload(group_id="ghost"))
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "group_id")

    def test_unknown_actor_404(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.change_group_member_verified(
                "g1", self._payload(actor="ghost"))
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "actor_device_id")

    def test_revoked_actor_409(self) -> None:
        self.service.revoke_device("creator")
        with self.assertRaises(ServiceError) as ctx:
            self.service.change_group_member_verified("g1", self._payload())
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "actor_device_id")

    def test_non_creator_actor_409(self) -> None:
        alice_private, alice_key = _new_identity()
        self.service.register(_register_payload(
            "carol", user_id="u4", identity_key=alice_key))
        with self.assertRaises(ServiceError) as ctx:
            self.service.change_group_member_verified(
                "g1", self._payload(actor="carol", private=alice_private))
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "actor_device_id")

    def test_actor_key_not_ed25519_400_identity_key(self) -> None:
        service = DeviceService()
        service.register(_register_payload("creator",
                                           identity_key=_x25519_der_b64()))
        service.register(_register_payload("bob", user_id="u3"))
        service.create_group({"group_id": "g1",
                              "creator_device_id": "creator",
                              "member_device_ids": ["creator"]})
        payload = self._payload(revision=1)
        with self.assertRaises(ServiceError) as ctx:
            service.change_group_member_verified("g1", payload)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.field, "identity_key")

    def test_revision_mismatch_409(self) -> None:
        for bad_revision in (2, 3, 99):
            payload = self._payload(revision=bad_revision)
            with self.subTest(bad_revision=bad_revision):
                with self.assertRaises(ServiceError) as ctx:
                    self.service.change_group_member_verified("g1", payload)
                self.assertEqual(ctx.exception.status_code, 409)
                self.assertEqual(ctx.exception.field, "expected_revision")

    def test_unknown_target_device_404(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.change_group_member_verified(
                "g1", self._payload(target="ghost"))
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "device_id")

    def test_bad_signature_400(self) -> None:
        other, _ = _new_identity()
        cases = [
            # Signed by a different key.
            self._payload(private=other),
            # Signed for a different group id.
            self._payload(group_id="g2"),
            # Signed for a different target.
            self._payload(signature=_authorization(
                self.private, "add", "creator", "alice", 2)),
            # Signed for a different operation.
            self._payload(signature=_authorization(
                self.private, "remove", "creator", "bob", 2)),
            # Signed for a different revision.
            self._payload(signature=_authorization(
                self.private, "add", "creator", "bob", 3)),
            # Random 64 bytes.
            self._payload(signature=base64.b64encode(b"\x00" * 64).decode()),
        ]
        for payload in cases:
            with self.subTest(payload=payload):
                with self.assertRaises(ServiceError) as ctx:
                    self.service.change_group_member_verified("g1", payload)
                self.assertEqual(ctx.exception.status_code, 400)
                self.assertEqual(ctx.exception.field, "signature")

    def test_check_order(self) -> None:
        # Unknown group beats a bad revision: 404/group_id.
        with self.assertRaises(ServiceError) as ctx:
            self.service.change_group_member_verified(
                "ghost", self._payload(revision=9, group_id="ghost"))
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "group_id")
        # Unknown actor beats a revision mismatch: 404/actor_device_id.
        with self.assertRaises(ServiceError) as ctx:
            self.service.change_group_member_verified(
                "g1", self._payload(actor="ghost", revision=9))
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "actor_device_id")
        # Revision mismatch beats an unknown target: 409/expected_revision.
        with self.assertRaises(ServiceError) as ctx:
            self.service.change_group_member_verified(
                "g1", self._payload(target="ghost", revision=9))
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "expected_revision")
        # Unknown target beats a bad signature: 404/device_id.
        payload = self._payload(target="ghost",
                                signature=base64.b64encode(
                                    b"\x00" * 64).decode())
        with self.assertRaises(ServiceError) as ctx:
            self.service.change_group_member_verified("g1", payload)
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "device_id")

    def test_failure_leaves_members_and_revision_untouched(self) -> None:
        before = self._group()
        with self.assertRaises(ServiceError):
            self.service.change_group_member_verified(
                "g1", self._payload(
                    signature=base64.b64encode(b"\x00" * 64).decode()))
        self.assertEqual(self._group(), before)

    def test_signature_uses_current_identity_key(self) -> None:
        # Rotate the creator's identity key: authorizations signed by the
        # old key stop verifying; the new current key signs instead.
        new_private, new_key = _new_identity()
        rotation = identity_rotation_proof_message(
            "u1", "creator", new_key, 1)
        self.service.rotate_identity_key_verified("creator", {
            "identity_key": new_key, "expected_version": 1,
            "signature": base64.b64encode(
                self.private.sign(rotation)).decode()})
        with self.assertRaises(ServiceError) as ctx:
            self.service.change_group_member_verified("g1", self._payload())
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.field, "signature")
        body, status = self.service.change_group_member_verified(
            "g1", self._payload(private=new_private))
        self.assertEqual(status, 201)
        self.assertEqual(body["revision"], 2)


class GroupMembersVerifiedHTTPTest(unittest.TestCase):
    def setUp(self) -> None:
        self.server, self.service = create_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        self.private, self.identity = _new_identity()
        self.service.register(_register_payload(
            "creator", identity_key=self.identity))
        self.service.register(_register_payload("alice", user_id="u2"))
        self.service.register(_register_payload("bob", user_id="u3"))
        self.service.create_group({
            "group_id": "g1", "creator_device_id": "creator",
            "member_device_ids": ["alice"]})

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

    def _payload(self, operation="add", target="bob", revision=1,
                 group_id="g1", private=None) -> dict:
        return {"operation": operation, "actor_device_id": "creator",
                "device_id": target, "expected_revision": revision,
                "signature": _authorization(
                    private or self.private, operation, "creator", target,
                    revision, group_id=group_id)}

    def test_add_201_then_duplicate_200(self) -> None:
        status, body = self._request(
            "POST", "/v1/groups/g1/members/verified", self._payload())
        self.assertEqual(status, 201)
        self.assertEqual(body["members"], ["creator", "alice", "bob"])
        self.assertEqual(body["revision"], 2)
        status, body = self._request(
            "POST", "/v1/groups/g1/members/verified", self._payload(
                revision=2))
        self.assertEqual(status, 200)
        self.assertEqual(body["revision"], 2)

    def test_remove_200_advances_revision(self) -> None:
        status, body = self._request(
            "POST", "/v1/groups/g1/members/verified",
            self._payload(operation="remove", target="alice"))
        self.assertEqual(status, 200)
        self.assertEqual(body["members"], ["creator"])
        self.assertEqual(body["revision"], 2)

    def test_invalid_json_400_request_body(self) -> None:
        status, body = self._request(
            "POST", "/v1/groups/g1/members/verified", raw="{not json")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "request_body")

    def test_non_object_400_request_body(self) -> None:
        status, body = self._request(
            "POST", "/v1/groups/g1/members/verified", [1, 2])
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "request_body")

    def test_missing_operation_400(self) -> None:
        payload = self._payload()
        del payload["operation"]
        status, body = self._request(
            "POST", "/v1/groups/g1/members/verified", payload)
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "operation")

    def test_bad_operation_400(self) -> None:
        payload = self._payload()
        payload["operation"] = "kick"
        status, body = self._request(
            "POST", "/v1/groups/g1/members/verified", payload)
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "operation")

    def test_bool_revision_400(self) -> None:
        payload = self._payload()
        payload["expected_revision"] = True
        status, body = self._request(
            "POST", "/v1/groups/g1/members/verified", payload)
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "expected_revision")

    def test_bad_signature_encoding_400(self) -> None:
        payload = self._payload()
        payload["signature"] = "not-base64!"
        status, body = self._request(
            "POST", "/v1/groups/g1/members/verified", payload)
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "signature")

    def test_unknown_group_404(self) -> None:
        status, body = self._request(
            "POST", "/v1/groups/ghost/members/verified",
            self._payload(group_id="ghost"))
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "group_id")

    def test_revision_mismatch_409(self) -> None:
        status, body = self._request(
            "POST", "/v1/groups/g1/members/verified", self._payload(
                revision=7))
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "expected_revision")

    def test_bad_signature_400(self) -> None:
        payload = self._payload()
        payload["signature"] = base64.b64encode(b"\x00" * 64).decode()
        status, body = self._request(
            "POST", "/v1/groups/g1/members/verified", payload)
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "signature")
        # The failed attempt changed neither the roster nor the revision.
        _, group = self._request("GET", "/v1/groups/g1")
        self.assertEqual(group["members"], ["creator", "alice"])
        self.assertEqual(group["revision"], 1)

    def test_unknown_actor_404(self) -> None:
        payload = self._payload()
        payload["actor_device_id"] = "ghost"
        status, body = self._request(
            "POST", "/v1/groups/g1/members/verified", payload)
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "actor_device_id")

    def test_unknown_target_404(self) -> None:
        status, body = self._request(
            "POST", "/v1/groups/g1/members/verified",
            self._payload(target="ghost"))
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "device_id")

    def test_path_with_percent_encoded_slash(self) -> None:
        # A group whose id contains an encoded slash: the signed message
        # uses the path-decoded id (an ordinary '/'), and a slash that is
        # not encoded makes the path a different shape (404).
        slash_id = "a/b"
        self.service.create_group({
            "group_id": slash_id, "creator_device_id": "creator",
            "member_device_ids": ["creator"]})
        status, body = self._request(
            "POST", "/v1/groups/a%2Fb/members/verified",
            self._payload(group_id=slash_id))
        self.assertEqual(status, 201)
        self.assertEqual(body["group_id"], slash_id)
        status, body = self._request(
            "POST", "/v1/groups/a/b/members/verified",
            self._payload(group_id=slash_id))
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "group_id")

    def test_plain_member_routes_still_work(self) -> None:
        status, body = self._request("POST", "/v1/groups/g1/members", {
            "actor_device_id": "creator", "device_id": "bob"})
        self.assertEqual(status, 201)
        self.assertEqual(body["revision"], 2)
        status, body = self._request(
            "POST", "/v1/groups/g1/members/remove", {
                "actor_device_id": "creator", "device_id": "bob"})
        self.assertEqual(status, 200)
        self.assertEqual(body["revision"], 3)


class GroupMembersVerifiedPersistenceTest(_GroupServiceBase):
    def _service(self, path):
        service = DeviceService()
        store = attach_persistence(service, path)
        return service, store

    def setUp(self) -> None:
        # Bypass the in-memory base setup; each test builds its own
        # file-backed service.
        self.path = tempfile.mktemp(suffix=".json")
        self.addCleanup(
            lambda: os.path.exists(self.path) and os.unlink(self.path))
        self.service, self.state_store = self._service(self.path)
        self.private, self.identity = _new_identity()
        self.service.register(_register_payload(
            "creator", identity_key=self.identity))
        self.service.register(_register_payload("alice", user_id="u2"))
        self.service.register(_register_payload("bob", user_id="u3"))
        self.service.create_group({
            "group_id": "g1", "creator_device_id": "creator",
            "member_device_ids": ["alice"]})

    def test_change_survives_restart_in_order(self) -> None:
        self.service.change_group_member_verified("g1", self._payload())
        self.service.change_group_member_verified(
            "g1", self._payload(operation="remove", target="alice",
                                revision=2))

        second, _ = self._service(self.path)
        group = second.get_group("g1")
        self.assertEqual(group["members"], ["creator", "bob"])
        self.assertEqual(group["revision"], 3)
        # The next change must name the recovered revision.
        body, status = second.change_group_member_verified(
            "g1", {"operation": "remove", "actor_device_id": "creator",
                   "device_id": "bob", "expected_revision": 3,
                   "signature": _authorization(
                       self.private, "remove", "creator", "bob", 3)})
        self.assertEqual(status, 200)
        self.assertEqual(body["revision"], 4)

    def test_noop_replay_advances_no_generation(self) -> None:
        self.service.change_group_member_verified("g1", self._payload())
        generation = self.state_store.commit_seq
        body, status = self.service.change_group_member_verified(
            "g1", self._payload(revision=2))
        self.assertEqual(status, 200)
        self.assertEqual(self.state_store.commit_seq, generation)

    def test_failure_advances_no_generation(self) -> None:
        generation = self.state_store.commit_seq
        with self.assertRaises(ServiceError):
            self.service.change_group_member_verified(
                "g1", self._payload(
                    signature=base64.b64encode(b"\x00" * 64).decode()))
        self.assertEqual(self.state_store.commit_seq, generation)
        group = self._group()
        self.assertEqual(group["members"], ["creator", "alice"])
        self.assertEqual(group["revision"], 1)

    def test_persist_failure_rolls_back_the_change(self) -> None:
        def fail_save(_pending) -> None:
            raise OSError("simulated disk failure")

        self.state_store.save = fail_save  # type: ignore[assignment]
        with self.assertRaises(PersistenceUnavailable):
            self.service.change_group_member_verified("g1", self._payload())
        group = self._group()
        self.assertEqual(group["members"], ["creator", "alice"])
        self.assertEqual(group["revision"], 1)
        # The next request after the rolled-back 503 still works.
        del self.state_store.save
        body, status = self.service.change_group_member_verified(
            "g1", self._payload())
        self.assertEqual(status, 201)
        self.assertEqual(body["revision"], 2)


if __name__ == "__main__":
    unittest.main()
