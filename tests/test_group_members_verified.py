"""Tests for signature-authorized group membership changes.

Covers the service, HTTP (real loopback socket) and persistence layers for:

* POST /v1/groups/{group_id}/members/verified

The change applies only when the standard-base64 64-byte Ed25519 signature
verifies over the domain-separated canonical membership message
(``E2EE-GROUP-MEMBERSHIP-V1``) against the creator device's *current*
identity key, and ``expected_revision`` equals the group's current
revision. ``add`` of a new member is 201 and advances the revision, an
existing member 200; ``remove`` is always 200, advancing the revision only
when a member actually leaves. The unsigned member entries, group sessions
and the public-key-only server posture are unchanged.
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

from e2ee_backend.crypto import group_membership_proof_message
from e2ee_backend.http_app import create_server
from e2ee_backend.persistence import attach_persistence
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


def _authorization(private, group_id, operation, actor_device_id, device_id,
                   expected_revision) -> str:
    message = group_membership_proof_message(
        group_id, operation, actor_device_id, device_id, expected_revision)
    return base64.b64encode(private.sign(message)).decode()


def _register_payload(device_id, user_id="u1", identity_key=None) -> dict:
    return {
        "user_id": user_id,
        "device_id": device_id,
        "identity_key": identity_key or _x25519_b64(),
        "signed_prekeys": [{"key_id": "k1", "public_key": _x25519_b64()}],
    }


class GroupMembersVerifiedServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = DeviceService()
        self.private, self.identity = _new_identity()
        self.service.register(_register_payload("creator", "u1", self.identity))
        for device_id in ("alice", "bob", "carol"):
            self.service.register(_register_payload(device_id))
        self.service.create_group({
            "group_id": "g1", "creator_device_id": "creator",
            "member_device_ids": ["alice", "bob"]})

    def _payload(self, operation="add", actor="creator", device_id="carol",
                 expected_revision=1, signature=None, private=None,
                 group_id="g1"):
        if signature is None:
            signer = private or self.private
            signature = _authorization(
                signer, group_id, operation, actor, device_id,
                expected_revision)
        return {"operation": operation,
                "actor_device_id": actor,
                "device_id": device_id,
                "expected_revision": expected_revision,
                "signature": signature}

    def _group(self):
        return self.service.get_group("g1")

    # -- happy paths --------------------------------------------------------

    def test_add_new_member_is_201_and_bumps_revision(self) -> None:
        body, status = self.service.change_group_member_verified(
            "g1", self._payload())
        self.assertEqual(status, 201)
        self.assertEqual(set(body),
                         {"group_id", "revision", "members", "created_at"})
        self.assertEqual(body["members"], ["creator", "alice", "bob", "carol"])
        self.assertEqual(body["revision"], 2)

    def test_add_existing_member_is_200_without_revision_change(self) -> None:
        body, status = self.service.change_group_member_verified(
            "g1", self._payload(device_id="alice"))
        self.assertEqual(status, 200)
        self.assertEqual(body["revision"], 1)
        self.assertEqual(body["members"], ["creator", "alice", "bob"])

    def test_remove_member_is_200_and_bumps_revision(self) -> None:
        body, status = self.service.change_group_member_verified(
            "g1", self._payload(operation="remove", device_id="alice"))
        self.assertEqual(status, 200)
        self.assertEqual(body["members"], ["creator", "bob"])
        self.assertEqual(body["revision"], 2)

    def test_remove_absent_member_is_idempotent_200(self) -> None:
        body, status = self.service.change_group_member_verified(
            "g1", self._payload(operation="remove", device_id="carol"))
        self.assertEqual(status, 200)
        self.assertEqual(body["revision"], 1)
        self.assertEqual(body["members"], ["creator", "alice", "bob"])

    def test_remove_creator_is_noop_200(self) -> None:
        body, status = self.service.change_group_member_verified(
            "g1", self._payload(operation="remove", device_id="creator"))
        self.assertEqual(status, 200)
        self.assertEqual(body["members"][0], "creator")
        self.assertEqual(body["revision"], 1)

    def test_revoked_member_can_still_be_removed(self) -> None:
        self.service.revoke_device("alice")
        body, status = self.service.change_group_member_verified(
            "g1", self._payload(operation="remove", device_id="alice"))
        self.assertEqual(status, 200)
        self.assertNotIn("alice", body["members"])
        self.assertEqual(body["revision"], 2)

    def test_extra_fields_are_ignored(self) -> None:
        payload = self._payload()
        payload["extra"] = "ignored"
        payload["group_id"] = "spoofed"
        body, status = self.service.change_group_member_verified("g1", payload)
        self.assertEqual(status, 201)
        self.assertEqual(body["group_id"], "g1")

    # -- request validation ---------------------------------------------------

    def test_body_not_object_400_request_body(self) -> None:
        for payload in (None, [], "x", 42):
            with self.subTest(payload=payload):
                with self.assertRaises(ServiceError) as ctx:
                    self.service.change_group_member_verified("g1", payload)
                self.assertEqual(ctx.exception.status_code, 400)
                self.assertEqual(ctx.exception.field, "request_body")

    def test_missing_or_bad_operation_400(self) -> None:
        good = self._payload()
        for bad in (None, "", "ADD", "delete", 42, [], {}):
            payload = dict(good, operation=bad)
            with self.subTest(bad=bad):
                with self.assertRaises(ServiceError) as ctx:
                    self.service.change_group_member_verified("g1", payload)
                self.assertEqual(ctx.exception.status_code, 400)
                self.assertEqual(ctx.exception.field, "operation")
        with self.assertRaises(ServiceError) as ctx:
            self.service.change_group_member_verified(
                "g1", {k: v for k, v in good.items() if k != "operation"})
        self.assertEqual(ctx.exception.field, "operation")

    def test_missing_or_bad_ids_400(self) -> None:
        good = self._payload()
        for name in ("actor_device_id", "device_id"):
            for bad in (None, "", 42, []):
                payload = dict(good, **{name: bad})
                with self.subTest(field=name, bad=bad):
                    with self.assertRaises(ServiceError) as ctx:
                        self.service.change_group_member_verified(
                            "g1", payload)
                    self.assertEqual(ctx.exception.status_code, 400)
                    self.assertEqual(ctx.exception.field, name)
            with self.assertRaises(ServiceError) as ctx:
                self.service.change_group_member_verified(
                    "g1", {k: v for k, v in good.items() if k != name})
            self.assertEqual(ctx.exception.field, name)

    def test_missing_or_bad_expected_revision_400(self) -> None:
        good = self._payload()
        for bad in (None, True, False, 0, -1, "1", 1.5, [], {}):
            payload = dict(good, expected_revision=bad)
            with self.subTest(bad=bad):
                with self.assertRaises(ServiceError) as ctx:
                    self.service.change_group_member_verified("g1", payload)
                self.assertEqual(ctx.exception.status_code, 400)
                self.assertEqual(ctx.exception.field, "expected_revision")
        with self.assertRaises(ServiceError) as ctx:
            self.service.change_group_member_verified(
                "g1", {k: v for k, v in good.items()
                       if k != "expected_revision"})
        self.assertEqual(ctx.exception.field, "expected_revision")

    def test_missing_or_bad_signature_400(self) -> None:
        good = self._payload()
        for bad in (None, "", 8, [], "@@@@", "abc", "a" * 88,
                    base64.b64encode(b"\x00" * 63).decode(),
                    base64.b64encode(b"\x00" * 65).decode()):
            payload = dict(good, signature=bad)
            with self.subTest(bad=bad):
                with self.assertRaises(ServiceError) as ctx:
                    self.service.change_group_member_verified("g1", payload)
                self.assertEqual(ctx.exception.status_code, 400)
                self.assertEqual(ctx.exception.field, "signature")
        with self.assertRaises(ServiceError) as ctx:
            self.service.change_group_member_verified(
                "g1", {k: v for k, v in good.items() if k != "signature"})
        self.assertEqual(ctx.exception.field, "signature")

    def test_signature_must_be_canonical_base64(self) -> None:
        good = self._payload()["signature"]
        bad_variants = [good.rstrip("=")]
        url_safe = good.replace("+", "-").replace("/", "_")
        if url_safe != good:  # only a real variant when + or / occurs
            bad_variants.append(url_safe)
        for bad in bad_variants:
            with self.subTest(bad=bad):
                with self.assertRaises(ServiceError) as ctx:
                    self.service.change_group_member_verified(
                        "g1", self._payload(signature=bad))
                self.assertEqual(ctx.exception.field, "signature")

    # -- state checks ---------------------------------------------------------

    def test_unknown_group_404(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.change_group_member_verified(
                "ghost", self._payload(group_id="ghost"))
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "group_id")

    def test_unknown_actor_404(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.change_group_member_verified(
                "g1", self._payload(actor="nobody"))
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "actor_device_id")

    def test_revoked_actor_409(self) -> None:
        self.service.revoke_device("creator")
        with self.assertRaises(ServiceError) as ctx:
            self.service.change_group_member_verified("g1", self._payload())
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "actor_device_id")

    def test_non_creator_actor_409(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.change_group_member_verified(
                "g1", self._payload(actor="alice"))
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "actor_device_id")

    def test_non_ed25519_identity_400(self) -> None:
        service = DeviceService()
        service.register(_register_payload("creator", identity_key=None))
        service.store.find_by_device_id("creator").identity_key = (
            _x25519_der_b64())
        service.create_group({
            "group_id": "g1", "creator_device_id": "creator",
            "member_device_ids": ["seed"]})
        with self.assertRaises(ServiceError) as ctx:
            service.change_group_member_verified(
                "g1", self._payload(expected_revision=1))
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.field, "identity_key")

    def test_revision_mismatch_409(self) -> None:
        for bad_revision in (2, 99):
            with self.subTest(bad_revision=bad_revision):
                with self.assertRaises(ServiceError) as ctx:
                    self.service.change_group_member_verified(
                        "g1", self._payload(expected_revision=bad_revision))
                self.assertEqual(ctx.exception.status_code, 409)
                self.assertEqual(ctx.exception.field, "expected_revision")

    def test_revision_checked_before_signature(self) -> None:
        # A stale revision is 409 even when the signature matches the body.
        with self.assertRaises(ServiceError) as ctx:
            self.service.change_group_member_verified(
                "g1", self._payload(expected_revision=7))
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "expected_revision")

    def test_unknown_target_device_404(self) -> None:
        for operation in ("add", "remove"):
            with self.subTest(operation=operation):
                with self.assertRaises(ServiceError) as ctx:
                    self.service.change_group_member_verified(
                        "g1", self._payload(operation=operation,
                                            device_id="nobody"))
                self.assertEqual(ctx.exception.status_code, 404)
                self.assertEqual(ctx.exception.field, "device_id")

    def test_add_revoked_target_409(self) -> None:
        self.service.revoke_device("carol")
        with self.assertRaises(ServiceError) as ctx:
            self.service.change_group_member_verified("g1", self._payload())
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "device_id")

    def test_bad_signature_400(self) -> None:
        other, _ = _new_identity()
        cases = [
            # Signed by a different key.
            self._payload(private=other),
            # Signed for a different group.
            self._payload(group_id="g2"),
            # Signed for a different operation.
            self._payload(signature=_authorization(
                self.private, "g1", "remove", "creator", "carol", 1)),
            # Signed for a different target device.
            self._payload(signature=_authorization(
                self.private, "g1", "add", "creator", "alice", 1)),
            # Signed for a different revision.
            self._payload(signature=_authorization(
                self.private, "g1", "add", "creator", "carol", 2)),
            # A 64-byte non-signature.
            self._payload(signature=base64.b64encode(b"\x00" * 64).decode()),
        ]
        for payload in cases:
            with self.subTest(payload=payload):
                with self.assertRaises(ServiceError) as ctx:
                    self.service.change_group_member_verified("g1", payload)
                self.assertEqual(ctx.exception.status_code, 400)
                self.assertEqual(ctx.exception.field, "signature")

    def test_noop_still_requires_valid_signature(self) -> None:
        other, _ = _new_identity()
        # Re-adding an existing member with a bad signature is refused.
        with self.assertRaises(ServiceError) as ctx:
            self.service.change_group_member_verified(
                "g1", self._payload(device_id="alice", private=other))
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.field, "signature")
        # Removing the creator (a no-op) with a bad signature is refused.
        with self.assertRaises(ServiceError) as ctx:
            self.service.change_group_member_verified(
                "g1", self._payload(operation="remove", device_id="creator",
                                    private=other))
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.field, "signature")

    def test_check_order_group_actor_revision_target(self) -> None:
        # Unknown group beats a bad revision: 404/group_id.
        with self.assertRaises(ServiceError) as ctx:
            self.service.change_group_member_verified(
                "ghost", self._payload(group_id="ghost", expected_revision=9))
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "group_id")
        # Non-creator actor beats a bad revision: 409/actor_device_id.
        with self.assertRaises(ServiceError) as ctx:
            self.service.change_group_member_verified(
                "g1", self._payload(actor="alice", expected_revision=9))
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "actor_device_id")
        # A bad revision beats an unknown target: 409/expected_revision.
        with self.assertRaises(ServiceError) as ctx:
            self.service.change_group_member_verified(
                "g1", self._payload(device_id="nobody", expected_revision=9))
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "expected_revision")

    def test_failure_leaves_members_and_revision_untouched(self) -> None:
        before = self._group()
        for payload in (self._payload(expected_revision=7),
                        self._payload(private=ed25519.Ed25519PrivateKey
                                      .generate()),
                        self._payload(device_id="nobody")):
            with self.assertRaises(ServiceError):
                self.service.change_group_member_verified("g1", payload)
        after = self._group()
        self.assertEqual(after["members"], before["members"])
        self.assertEqual(after["revision"], before["revision"])

    def test_signed_with_rotated_identity_uses_current_key(self) -> None:
        from e2ee_backend.crypto import identity_rotation_proof_message
        new_private, new_key = _new_identity()
        rot_message = identity_rotation_proof_message(
            "u1", "creator", new_key, 1)
        self.service.rotate_identity_key_verified(
            "creator", {"identity_key": new_key, "expected_version": 1,
                        "signature": base64.b64encode(
                            self.private.sign(rot_message)).decode()})
        # The old key no longer authorizes a membership change.
        with self.assertRaises(ServiceError) as ctx:
            self.service.change_group_member_verified("g1", self._payload())
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.field, "signature")
        # The current key does.
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
        self.service.register(_register_payload("creator", "u1", self.identity))
        for device_id in ("alice", "bob", "carol"):
            self.service.register(_register_payload(device_id))
        self.service.create_group({
            "group_id": "g1", "creator_device_id": "creator",
            "member_device_ids": ["alice", "bob"]})

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _request(self, method: str, path: str, body: object = None,
                 raw: "str | None" = None):
        connection = HTTPConnection("127.0.0.1", self.port, timeout=5)
        payload = raw if raw is not None else (
            json.dumps(body) if body is not None else None)
        headers = {"Content-Type": "application/json"} if payload else {}
        connection.request(method, path, body=payload, headers=headers)
        response = connection.getresponse()
        data = response.read().decode("utf-8")
        connection.close()
        return response.status, json.loads(data)

    def _payload(self, operation="add", device_id="carol",
                 expected_revision=1, group_id="g1") -> dict:
        return {"operation": operation,
                "actor_device_id": "creator",
                "device_id": device_id,
                "expected_revision": expected_revision,
                "signature": _authorization(
                    self.private, group_id, operation, "creator", device_id,
                    expected_revision)}

    def test_add_verified_201(self) -> None:
        status, body = self._request(
            "POST", "/v1/groups/g1/members/verified", self._payload())
        self.assertEqual(status, 201)
        self.assertEqual(body["members"],
                         ["creator", "alice", "bob", "carol"])
        self.assertEqual(body["revision"], 2)

    def test_remove_verified_200(self) -> None:
        status, body = self._request(
            "POST", "/v1/groups/g1/members/verified",
            self._payload(operation="remove", device_id="alice"))
        self.assertEqual(status, 200)
        self.assertEqual(body["members"], ["creator", "bob"])
        self.assertEqual(body["revision"], 2)

    def test_add_existing_member_200(self) -> None:
        status, body = self._request(
            "POST", "/v1/groups/g1/members/verified",
            self._payload(device_id="alice"))
        self.assertEqual(status, 200)
        self.assertEqual(body["revision"], 1)

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

    def test_bad_operation_400(self) -> None:
        payload = self._payload()
        payload["operation"] = "delete"
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
            "POST", "/v1/groups/g1/members/verified",
            self._payload(expected_revision=5))
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "expected_revision")

    def test_bad_signature_400(self) -> None:
        payload = self._payload()
        payload["signature"] = base64.b64encode(b"\x00" * 64).decode()
        status, body = self._request(
            "POST", "/v1/groups/g1/members/verified", payload)
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "signature")

    def test_group_id_with_percent_encoded_slash(self) -> None:
        # The signed message uses the path-decoded group id.
        slash_id = "a/b"
        self.service.create_group({
            "group_id": slash_id, "creator_device_id": "creator",
            "member_device_ids": ["alice"]})
        status, body = self._request(
            "POST", "/v1/groups/a%2Fb/members/verified",
            self._payload(group_id=slash_id))
        self.assertEqual(status, 201)
        self.assertEqual(body["group_id"], slash_id)

    def test_unsigned_member_routes_still_work(self) -> None:
        status, body = self._request(
            "POST", "/v1/groups/g1/members/add",
            {"actor_device_id": "creator", "device_id": "carol"})
        self.assertEqual(status, 201)
        self.assertEqual(body["revision"], 2)
        status, body = self._request(
            "POST", "/v1/groups/g1/members/remove",
            {"actor_device_id": "creator", "device_id": "carol"})
        self.assertEqual(status, 200)
        self.assertEqual(body["revision"], 3)


class GroupMembersVerifiedPersistenceTest(unittest.TestCase):
    def _service(self, path):
        service = DeviceService()
        store = attach_persistence(service, path)
        return service, store

    def _setup(self, path):
        private, identity = _new_identity()
        service, store = self._service(path)
        service.register(_register_payload("creator", "u1", identity))
        for device_id in ("alice", "carol"):
            service.register(_register_payload(device_id))
        service.create_group({
            "group_id": "g1", "creator_device_id": "creator",
            "member_device_ids": ["alice"]})
        return service, store, private

    def test_changes_survive_restart_in_order(self) -> None:
        path = tempfile.mktemp(suffix=".json")
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        service, _store, private = self._setup(path)
        service.change_group_member_verified(
            "g1", {"operation": "add", "actor_device_id": "creator",
                   "device_id": "carol", "expected_revision": 1,
                   "signature": _authorization(
                       private, "g1", "add", "creator", "carol", 1)})
        service.change_group_member_verified(
            "g1", {"operation": "remove", "actor_device_id": "creator",
                   "device_id": "alice", "expected_revision": 2,
                   "signature": _authorization(
                       private, "g1", "remove", "creator", "alice", 2)})

        restarted, _ = self._service(path)
        group = restarted.get_group("g1")
        self.assertEqual(group["members"], ["creator", "carol"])
        self.assertEqual(group["revision"], 3)

    def test_failure_advances_no_generation(self) -> None:
        path = tempfile.mktemp(suffix=".json")
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        service, store, private = self._setup(path)
        generation = store.commit_seq
        other, _ = _new_identity()
        with self.assertRaises(ServiceError):
            service.change_group_member_verified(
                "g1", {"operation": "add", "actor_device_id": "creator",
                       "device_id": "carol", "expected_revision": 1,
                       "signature": _authorization(
                           other, "g1", "add", "creator", "carol", 1)})
        self.assertEqual(store.commit_seq, generation)
        group = service.get_group("g1")
        self.assertEqual(group["members"], ["creator", "alice"])
        self.assertEqual(group["revision"], 1)

    def test_idempotent_change_advances_no_generation(self) -> None:
        path = tempfile.mktemp(suffix=".json")
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        service, store, private = self._setup(path)
        generation = store.commit_seq
        # Re-adding an existing member is a state-free 200.
        _body, status = service.change_group_member_verified(
            "g1", {"operation": "add", "actor_device_id": "creator",
                   "device_id": "alice", "expected_revision": 1,
                   "signature": _authorization(
                       private, "g1", "add", "creator", "alice", 1)})
        self.assertEqual(status, 200)
        self.assertEqual(store.commit_seq, generation)

    def test_persist_failure_rolls_back_the_change(self) -> None:
        from e2ee_backend.persistence import PersistenceUnavailable

        path = tempfile.mktemp(suffix=".json")
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        service, state_store, private = self._setup(path)

        def fail_save(_pending) -> None:
            raise OSError("simulated disk failure")

        state_store.save = fail_save  # type: ignore[assignment]
        with self.assertRaises(PersistenceUnavailable):
            service.change_group_member_verified(
                "g1", {"operation": "add", "actor_device_id": "creator",
                       "device_id": "carol", "expected_revision": 1,
                       "signature": _authorization(
                           private, "g1", "add", "creator", "carol", 1)})
        group = service.get_group("g1")
        self.assertEqual(group["members"], ["creator", "alice"])
        self.assertEqual(group["revision"], 1)
        # The next request after the rolled-back 503 still works.
        del state_store.save
        body, status = service.change_group_member_verified(
            "g1", {"operation": "add", "actor_device_id": "creator",
                   "device_id": "carol", "expected_revision": 1,
                   "signature": _authorization(
                       private, "g1", "add", "creator", "carol", 1)})
        self.assertEqual(status, 201)
        self.assertEqual(body["revision"], 2)


class GroupMembersVerifiedConcurrencyTest(unittest.TestCase):
    """Linearization against a concurrent identity rotation / revocation."""

    def test_concurrent_rotation_and_membership_change(self) -> None:
        # A verified identity rotation A->B on the creator and a membership
        # change signed by A commit under the same store lock; the stale
        # authorization must lose when the rotation commits first.
        from e2ee_backend.crypto import identity_rotation_proof_message
        for _ in range(40):
            service = DeviceService()
            private_a, key_a = _new_identity()
            private_b, key_b = _new_identity()
            service.register(_register_payload("creator", "u1", key_a))
            service.register(_register_payload("carol"))
            service.create_group({
                "group_id": "g1", "creator_device_id": "creator",
                "member_device_ids": ["seed"]})

            rot_sig = base64.b64encode(private_a.sign(
                identity_rotation_proof_message(
                    "u1", "creator", key_b, 1))).decode()
            member_sig = _authorization(
                private_a, "g1", "add", "creator", "carol", 1)
            outcomes = []

            def rotate() -> None:
                try:
                    service.rotate_identity_key_verified(
                        "creator", {"identity_key": key_b,
                                    "expected_version": 1,
                                    "signature": rot_sig})
                    outcomes.append(("rotate", None))
                except ServiceError as error:
                    outcomes.append(("rotate", error.field))

            def change() -> None:
                try:
                    service.change_group_member_verified(
                        "g1", {"operation": "add",
                               "actor_device_id": "creator",
                               "device_id": "carol",
                               "expected_revision": 1,
                               "signature": member_sig})
                    outcomes.append(("change", None))
                except ServiceError as error:
                    outcomes.append(("change", error.field))

            t1 = threading.Thread(target=rotate)
            t2 = threading.Thread(target=change)
            t1.start()
            t2.start()
            t1.join()
            t2.join()

            self.assertEqual(len(outcomes), 2)
            results = {op: field for op, field in outcomes}
            group = service.get_group("g1")
            self.assertIsNone(results["rotate"])
            if results["change"] is None:
                # The membership change committed first (revision 1 -> 2).
                self.assertEqual(group["members"],
                                 ["creator", "seed", "carol"])
                self.assertEqual(group["revision"], 2)
            else:
                # The rotation committed first: the old-key signature fails.
                self.assertEqual(results["change"], "signature")
                self.assertEqual(group["members"], ["creator", "seed"])
                self.assertEqual(group["revision"], 1)
                # A change signed by the current key still applies.
                body, status = service.change_group_member_verified(
                    "g1", {"operation": "add", "actor_device_id": "creator",
                           "device_id": "carol", "expected_revision": 1,
                           "signature": _authorization(
                               private_b, "g1", "add", "creator", "carol",
                               1)})
                self.assertEqual(status, 201)
                self.assertEqual(body["revision"], 2)


if __name__ == "__main__":
    unittest.main()
