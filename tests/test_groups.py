"""Tests for groups and frozen-membership group sessions.

Covers the service/storage contract (validation, status codes, fields,
revision linearization, frozen membership and message-read restriction),
persistence round-trips, the HTTP routes over a real socket and the CLI
subcommands as real subprocesses.
"""
from __future__ import annotations

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
from cryptography.hazmat.primitives.asymmetric import x25519

from e2ee_backend.http_app import create_server
from e2ee_backend.persistence import StateFileError, attach_persistence
from e2ee_backend.service import DeviceService, ServiceError


def _raw_key_b64() -> str:
    key = x25519.X25519PrivateKey.generate().public_key()
    raw = key.public_bytes(serialization.Encoding.Raw,
                          serialization.PublicFormat.Raw)
    return base64.b64encode(raw).decode()


class GroupFixture:
    """Registers creator/member devices on a fresh service."""

    def __init__(self) -> None:
        self.service = DeviceService()
        for device_id in ("creator", "m1", "m2", "m3", "outsider"):
            self.service.register({
                "user_id": "u", "device_id": device_id,
                "identity_key": _raw_key_b64(), "signed_prekeys": []})

    def create(self, group_id: str = "g1", creator: str = "creator",
               members=("m1", "m2")) -> dict:
        return self.service.create_group({
            "group_id": group_id, "creator_device_id": creator,
            "member_device_ids": list(members)})


# --------------------------------------------------------------------------
# service / storage
# --------------------------------------------------------------------------

class GroupCreateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fx = GroupFixture()
        self.service = self.fx.service

    def test_success_returns_five_fields_with_creator_and_revision_one(self) -> None:
        body = self.fx.create()
        self.assertEqual(
            set(body),
            {"group_id", "creator_device_id", "members", "revision",
             "created_at"})
        self.assertEqual(body["group_id"], "g1")
        self.assertEqual(body["creator_device_id"], "creator")
        # Creator not in the member list is prepended exactly once.
        self.assertEqual(body["members"], ["creator", "m1", "m2"])
        self.assertEqual(body["revision"], 1)
        self.assertTrue(body["created_at"].endswith("+00:00"))

    def test_creator_listed_in_members_keeps_given_order_once(self) -> None:
        body = self.fx.create(members=("m1", "creator", "m2"))
        self.assertEqual(body["members"], ["m1", "creator", "m2"])

    def test_duplicate_group_id_is_409(self) -> None:
        self.fx.create()
        with self.assertRaises(ServiceError) as ctx:
            self.fx.create()
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "group_id")

    def test_unknown_creator_is_404(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.create_group({
                "group_id": "gx", "creator_device_id": "ghost",
                "member_device_ids": ["m1"]})
        self.assertEqual((ctx.exception.status_code, ctx.exception.field),
                         (404, "creator_device_id"))

    def test_revoked_creator_is_409(self) -> None:
        self.service.revoke_device("creator")
        with self.assertRaises(ServiceError) as ctx:
            self.fx.create()
        self.assertEqual((ctx.exception.status_code, ctx.exception.field),
                         (409, "creator_device_id"))

    def test_unknown_member_is_404_on_member_field(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.create_group({
                "group_id": "gx", "creator_device_id": "creator",
                "member_device_ids": ["ghost"]})
        self.assertEqual((ctx.exception.status_code, ctx.exception.field),
                         (404, "member_device_ids"))

    def test_revoked_member_is_409_on_member_field(self) -> None:
        self.service.revoke_device("m1")
        with self.assertRaises(ServiceError) as ctx:
            self.fx.create()
        self.assertEqual((ctx.exception.status_code, ctx.exception.field),
                         (409, "member_device_ids"))

    def test_failure_writes_nothing_so_id_can_be_reused(self) -> None:
        with self.assertRaises(ServiceError):
            self.service.create_group({
                "group_id": "gx", "creator_device_id": "creator",
                "member_device_ids": ["ghost"]})
        # The duplicate-id check must not have recorded "gx".
        body = self.service.create_group({
            "group_id": "gx", "creator_device_id": "creator",
            "member_device_ids": ["m1"]})
        self.assertEqual(body["group_id"], "gx")

    def _assert_400(self, payload: object, field: str) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.create_group(payload)
        self.assertEqual((ctx.exception.status_code, ctx.exception.field),
                         (400, field))

    def test_validation_errors_name_fields(self) -> None:
        base = {"group_id": "gx", "creator_device_id": "creator",
                "member_device_ids": ["m1"]}

        missing_group = dict(base); del missing_group["group_id"]
        self._assert_400(missing_group, "group_id")
        missing_creator = dict(base); del missing_creator["creator_device_id"]
        self._assert_400(missing_creator, "creator_device_id")
        missing_members = dict(base); del missing_members["member_device_ids"]
        self._assert_400(missing_members, "member_device_ids")

        self._assert_400({**base, "group_id": ""}, "group_id")
        self._assert_400({**base, "group_id": 7}, "group_id")
        self._assert_400({**base, "creator_device_id": ""},
                         "creator_device_id")
        self._assert_400({**base, "member_device_ids": "m1"},
                         "member_device_ids")
        self._assert_400({**base, "member_device_ids": []},
                         "member_device_ids")
        self._assert_400({**base, "member_device_ids": [""]},
                         "member_device_ids[0]")
        self._assert_400({**base, "member_device_ids": [9]},
                         "member_device_ids[0]")
        self._assert_400({**base, "member_device_ids": ["m1", "m1"]},
                         "member_device_ids")
        self._assert_400(["not", "an", "object"], "request_body")


class GroupShowAndMembershipTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fx = GroupFixture()
        self.service = self.fx.service
        self.fx.create()

    def test_get_group_200_and_404(self) -> None:
        body = self.service.get_group("g1")
        self.assertEqual(body["members"], ["creator", "m1", "m2"])
        with self.assertRaises(ServiceError) as ctx:
            self.service.get_group("ghost")
        self.assertEqual((ctx.exception.status_code, ctx.exception.field),
                         (404, "group_id"))

    def test_add_new_member_is_201_and_increments_revision(self) -> None:
        body, status = self.service.add_group_member(
            "g1", {"actor_device_id": "creator", "device_id": "m3"})
        self.assertEqual(status, 201)
        self.assertEqual(body["members"], ["creator", "m1", "m2", "m3"])
        self.assertEqual(body["revision"], 2)

    def test_add_existing_member_is_200_without_revision_change(self) -> None:
        body, status = self.service.add_group_member(
            "g1", {"actor_device_id": "creator", "device_id": "m1"})
        self.assertEqual(status, 200)
        self.assertEqual(body["members"], ["creator", "m1", "m2"])
        self.assertEqual(body["revision"], 1)

    def test_add_to_unknown_group_is_404_group_id(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.add_group_member(
                "ghost", {"actor_device_id": "creator", "device_id": "m3"})
        self.assertEqual((ctx.exception.status_code, ctx.exception.field),
                         (404, "group_id"))

    def test_add_unknown_device_is_404_device_id(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.add_group_member(
                "g1", {"actor_device_id": "creator", "device_id": "ghost"})
        self.assertEqual((ctx.exception.status_code, ctx.exception.field),
                         (404, "device_id"))

    def test_revoked_device_cannot_be_added(self) -> None:
        self.service.revoke_device("m3")
        with self.assertRaises(ServiceError) as ctx:
            self.service.add_group_member(
                "g1", {"actor_device_id": "creator", "device_id": "m3"})
        self.assertEqual((ctx.exception.status_code, ctx.exception.field),
                         (409, "device_id"))
        self.assertEqual(self.service.get_group("g1")["revision"], 1)

    def test_unknown_actor_is_404_actor_device_id(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.add_group_member(
                "g1", {"actor_device_id": "ghost", "device_id": "m3"})
        self.assertEqual((ctx.exception.status_code, ctx.exception.field),
                         (404, "actor_device_id"))

    def test_non_creator_actor_is_409_actor_device_id(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.add_group_member(
                "g1", {"actor_device_id": "m1", "device_id": "m3"})
        self.assertEqual((ctx.exception.status_code, ctx.exception.field),
                         (409, "actor_device_id"))
        # A revoked non-creator is still just unauthorized.
        self.service.revoke_device("m1")
        with self.assertRaises(ServiceError) as ctx:
            self.service.add_group_member(
                "g1", {"actor_device_id": "m1", "device_id": "m3"})
        self.assertEqual((ctx.exception.status_code, ctx.exception.field),
                         (409, "actor_device_id"))

    def test_remove_member_is_200_and_increments_revision_once(self) -> None:
        body = self.service.remove_group_member(
            "g1", {"actor_device_id": "creator", "device_id": "m1"})
        self.assertEqual(body["members"], ["creator", "m2"])
        self.assertEqual(body["revision"], 2)
        # Repeated removal stays 200 and does not bump the revision.
        again = self.service.remove_group_member(
            "g1", {"actor_device_id": "creator", "device_id": "m1"})
        self.assertEqual(again["members"], ["creator", "m2"])
        self.assertEqual(again["revision"], 2)

    def test_remove_errors_map_like_add(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.remove_group_member(
                "ghost", {"actor_device_id": "creator", "device_id": "m1"})
        self.assertEqual((ctx.exception.status_code, ctx.exception.field),
                         (404, "group_id"))
        with self.assertRaises(ServiceError) as ctx:
            self.service.remove_group_member(
                "g1", {"actor_device_id": "m1", "device_id": "m2"})
        self.assertEqual((ctx.exception.status_code, ctx.exception.field),
                         (409, "actor_device_id"))
        with self.assertRaises(ServiceError) as ctx:
            self.service.remove_group_member(
                "g1", {"actor_device_id": "creator", "device_id": "ghost"})
        self.assertEqual((ctx.exception.status_code, ctx.exception.field),
                         (404, "device_id"))
        with self.assertRaises(ServiceError) as ctx:
            self.service.remove_group_member(
                "g1", {"actor_device_id": "ghost", "device_id": "m1"})
        self.assertEqual((ctx.exception.status_code, ctx.exception.field),
                         (404, "actor_device_id"))

    def test_member_change_validation(self) -> None:
        for payload, field in (
            (None, "request_body"),
            ({}, "actor_device_id"),
            ({"actor_device_id": "creator"}, "device_id"),
            ({"actor_device_id": "", "device_id": "m3"}, "actor_device_id"),
            ({"actor_device_id": "creator", "device_id": ""}, "device_id"),
        ):
            with self.assertRaises(ServiceError) as ctx:
                self.service.add_group_member("g1", payload)
            self.assertEqual((ctx.exception.status_code, ctx.exception.field),
                             (400, field), payload)


class GroupSessionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fx = GroupFixture()
        self.service = self.fx.service
        self.group = self.fx.create()

    def _create(self, group_id: str = "g1",
                initiator: str = "creator") -> dict:
        return self.service.create_group_session({
            "group_id": group_id, "initiator_device_id": initiator,
            "ephemeral_key": _raw_key_b64()})

    def test_success_freezes_members_and_revision(self) -> None:
        body = self._create()
        self.assertEqual(set(body), {
            "session_id", "group_id", "initiator_device_id", "ephemeral_key",
            "members", "revision", "created_at"})
        self.assertEqual(body["group_id"], "g1")
        self.assertEqual(body["initiator_device_id"], "creator")
        self.assertEqual(body["members"], ["creator", "m1", "m2"])
        self.assertEqual(body["revision"], 1)
        self.assertTrue(body["created_at"].endswith("+00:00"))
        self.assertEqual(len(body["session_id"]), 32)

    def test_repeated_post_creates_a_new_session(self) -> None:
        first = self._create()
        second = self._create()
        self.assertNotEqual(first["session_id"], second["session_id"])

    def test_get_returns_identical_frozen_snapshot(self) -> None:
        created = self._create()
        shown = self.service.get_group_session(created["session_id"])
        self.assertEqual(shown, created)

    def test_unknown_session_is_404(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.get_group_session("ghost")
        self.assertEqual((ctx.exception.status_code, ctx.exception.field),
                         (404, "session_id"))

    def test_unknown_group_is_404(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self._create(group_id="ghost")
        self.assertEqual((ctx.exception.status_code, ctx.exception.field),
                         (404, "group_id"))

    def test_unknown_or_revoked_or_non_member_initiator(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self._create(initiator="ghost")
        self.assertEqual((ctx.exception.status_code, ctx.exception.field),
                         (404, "initiator_device_id"))
        with self.assertRaises(ServiceError) as ctx:
            self._create(initiator="outsider")
        self.assertEqual((ctx.exception.status_code, ctx.exception.field),
                         (409, "initiator_device_id"))
        self.service.revoke_device("m1")
        with self.assertRaises(ServiceError) as ctx:
            self._create(initiator="m1")
        self.assertEqual((ctx.exception.status_code, ctx.exception.field),
                         (409, "initiator_device_id"))

    def test_membership_changes_after_creation_do_not_alter_freeze(self) -> None:
        frozen = self._create()
        self.service.add_group_member(
            "g1", {"actor_device_id": "creator", "device_id": "m3"})
        self.service.remove_group_member(
            "g1", {"actor_device_id": "creator", "device_id": "m1"})
        shown = self.service.get_group_session(frozen["session_id"])
        self.assertEqual(shown["members"], ["creator", "m1", "m2"])
        self.assertEqual(shown["revision"], 1)
        # A freshly created session sees the new membership and revision.
        later = self._create()
        self.assertEqual(later["members"], ["creator", "m2", "m3"])
        self.assertEqual(later["revision"], 3)

    def test_validation(self) -> None:
        good = {"group_id": "g1", "initiator_device_id": "creator",
                "ephemeral_key": _raw_key_b64()}

        def assert_400(payload: object, field: str) -> None:
            with self.assertRaises(ServiceError) as ctx:
                self.service.create_group_session(payload)
            self.assertEqual((ctx.exception.status_code, ctx.exception.field),
                             (400, field))

        for name in ("group_id", "initiator_device_id", "ephemeral_key"):
            missing = dict(good); del missing[name]
            assert_400(missing, name)
        assert_400({**good, "ephemeral_key": "garbage"}, "ephemeral_key")
        assert_400(["nope"], "request_body")

    def _post(self, session_id: str, sender: str, message_id: str,
              sequence: int, nonce: str = "nonce") -> None:
        self.service.post_message({
            "session_id": session_id, "sender_device_id": sender,
            "message_id": message_id, "sequence": sequence,
            "nonce": nonce, "ciphertext": "ciphertext"})

    def test_messages_are_restricted_to_frozen_members(self) -> None:
        frozen = self._create()
        sid = frozen["session_id"]
        # A frozen member can post.
        self._post(sid, "creator", "msg-1", 1)
        # m3 is an active device but not on the frozen list.
        with self.assertRaises(ServiceError) as ctx:
            self._post(sid, "m3", "msg-2", 2)
        self.assertEqual((ctx.exception.status_code, ctx.exception.field),
                         (409, "sender_device_id"))
        # Adding m3 to the group afterwards still does not grant access to
        # the previously frozen session.
        self.service.add_group_member(
            "g1", {"actor_device_id": "creator", "device_id": "m3"})
        with self.assertRaises(ServiceError) as ctx:
            self._post(sid, "m3", "msg-2", 2)
        self.assertEqual(ctx.exception.field, "sender_device_id")
        # The rejected write neither appended nor advanced the sequence.
        self._post(sid, "m1", "msg-2", 2, nonce="nonce2")

    def test_message_reads_are_restricted_to_frozen_members(self) -> None:
        frozen = self._create()
        sid = frozen["session_id"]
        self._post(sid, "creator", "msg-1", 1)
        page = self.service.list_messages(sid, "m2", 0, 100)
        self.assertEqual([m["message_id"] for m in page["messages"]], ["msg-1"])
        with self.assertRaises(ServiceError) as ctx:
            self.service.list_messages(sid, "outsider", 0, 100)
        self.assertEqual((ctx.exception.status_code, ctx.exception.field),
                         (409, "device_id"))
        # Unknown group-session id on the message routes is a 404.
        with self.assertRaises(ServiceError) as ctx:
            self.service.list_messages("ghost", "m2", 0, 100)
        self.assertEqual((ctx.exception.status_code, ctx.exception.field),
                         (404, "session_id"))


# --------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------

class GroupPersistenceTest(unittest.TestCase):
    def setUp(self) -> None:
        handle = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        self.path = handle.name
        handle.close()
        os.unlink(self.path)

    def tearDown(self) -> None:
        if os.path.exists(self.path):
            os.unlink(self.path)

    def _populated_service(self) -> DeviceService:
        service = DeviceService()
        attach_persistence(service, self.path)
        for device_id in ("creator", "m1", "m2", "m3"):
            service.register({
                "user_id": "u", "device_id": device_id,
                "identity_key": _raw_key_b64(), "signed_prekeys": []})
        service.create_group({
            "group_id": "g1", "creator_device_id": "creator",
            "member_device_ids": ["m1", "m2"]})
        service.add_group_member(
            "g1", {"actor_device_id": "creator", "device_id": "m3"})
        session = service.create_group_session({
            "group_id": "g1", "initiator_device_id": "creator",
            "ephemeral_key": _raw_key_b64()})
        service.post_message({
            "session_id": session["session_id"],
            "sender_device_id": "creator", "message_id": "msg-1",
            "sequence": 1, "nonce": "n-1", "ciphertext": "ct"})
        return service

    def test_state_file_is_version_one_with_group_sections(self) -> None:
        self._populated_service()
        with open(self.path, encoding="utf-8") as handle:
            document = json.load(handle)
        self.assertEqual(document["version"], 1)
        self.assertEqual(len(document["groups"]), 1)
        self.assertEqual(len(document["group_sessions"]), 1)

    def test_restart_restores_groups_sessions_and_frozen_messages(self) -> None:
        self._populated_service()
        with open(self.path, encoding="utf-8") as handle:
            session_id = json.load(handle)["group_sessions"][0]["session_id"]

        restored = DeviceService()
        attach_persistence(restored, self.path)
        group = restored.get_group("g1")
        self.assertEqual(group["members"], ["creator", "m1", "m2", "m3"])
        self.assertEqual(group["revision"], 2)
        frozen = restored.get_group_session(session_id)
        self.assertEqual(frozen["members"], ["creator", "m1", "m2", "m3"])
        page = restored.list_messages(session_id, "m1", 0, 100)
        self.assertEqual([m["message_id"] for m in page["messages"]], ["msg-1"])
        # Session-scoped nonce replay protection survived the restart.
        with self.assertRaises(ServiceError) as ctx:
            restored.post_message({
                "session_id": session_id, "sender_device_id": "creator",
                "message_id": "msg-2", "sequence": 2, "nonce": "n-1",
                "ciphertext": "ct"})
        self.assertEqual(ctx.exception.field, "nonce")

    def test_membership_change_after_restore_persists_but_keeps_freeze(self) -> None:
        self._populated_service()
        with open(self.path, encoding="utf-8") as handle:
            session_id = json.load(handle)["group_sessions"][0]["session_id"]
        first = DeviceService()
        attach_persistence(first, self.path)
        first.remove_group_member(
            "g1", {"actor_device_id": "creator", "device_id": "m3"})

        second = DeviceService()
        attach_persistence(second, self.path)
        self.assertEqual(second.get_group("g1")["revision"], 3)
        self.assertEqual(second.get_group_session(session_id)["members"],
                         ["creator", "m1", "m2", "m3"])

    def test_malformed_group_sections_refuse_start(self) -> None:
        self._populated_service()
        for mutated in ("groups", "group_sessions"):
            bad_path = self.path + ".bad"
            with open(self.path, encoding="utf-8") as handle:
                document = json.load(handle)
            document[mutated] = "not-a-list"
            with open(bad_path, "w", encoding="utf-8") as handle:
                json.dump(document, handle)
            with self.assertRaises(StateFileError):
                attach_persistence(DeviceService(), bad_path)
            os.unlink(bad_path)

    def test_group_session_referencing_unknown_group_refuses_start(self) -> None:
        self._populated_service()
        bad_path = self.path + ".bad"
        with open(self.path, encoding="utf-8") as handle:
            document = json.load(handle)
        document["groups"] = []
        with open(bad_path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        with self.assertRaises(StateFileError):
            attach_persistence(DeviceService(), bad_path)
        os.unlink(bad_path)


# --------------------------------------------------------------------------
# HTTP over a real socket
# --------------------------------------------------------------------------

class GroupHTTPTest(unittest.TestCase):
    def setUp(self) -> None:
        self.server, _ = create_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        for device_id in ("creator", "m1", "m2", "m3"):
            status, _ = self._request("POST", "/v1/devices", {
                "user_id": "u", "device_id": device_id,
                "identity_key": _raw_key_b64(), "signed_prekeys": []})
            self.assertEqual(status, 201)

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _request(self, method: str, path: str, body: object = None):
        connection = HTTPConnection("127.0.0.1", self.port, timeout=5)
        payload = json.dumps(body) if body is not None else None
        headers = {"Content-Type": "application/json"} if payload is not None else {}
        connection.request(method, path, body=payload, headers=headers)
        response = connection.getresponse()
        data = response.read().decode("utf-8")
        connection.close()
        return response.status, json.loads(data)

    def test_group_lifecycle_over_http(self) -> None:
        status, body = self._request("POST", "/v1/groups", {
            "group_id": "g1", "creator_device_id": "creator",
            "member_device_ids": ["m1", "m2"]})
        self.assertEqual(status, 201)
        self.assertEqual(body["members"], ["creator", "m1", "m2"])
        self.assertEqual(body["revision"], 1)

        status, body = self._request("POST", "/v1/groups", {
            "group_id": "g1", "creator_device_id": "creator",
            "member_device_ids": ["m1"]})
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "group_id")

        status, body = self._request("GET", "/v1/groups/g1")
        self.assertEqual(status, 200)
        self.assertEqual(body["group_id"], "g1")

        status, body = self._request("POST", "/v1/groups/g1/members",
                                     {"actor_device_id": "creator",
                                      "device_id": "m3"})
        self.assertEqual(status, 201)
        self.assertEqual(body["revision"], 2)
        status, body = self._request("POST", "/v1/groups/g1/members",
                                     {"actor_device_id": "creator",
                                      "device_id": "m3"})
        self.assertEqual(status, 200)
        self.assertEqual(body["revision"], 2)

        status, body = self._request(
            "POST", "/v1/groups/g1/members/remove",
            {"actor_device_id": "creator", "device_id": "m3"})
        self.assertEqual(status, 200)
        self.assertEqual(body["members"], ["creator", "m1", "m2"])
        self.assertEqual(body["revision"], 3)

    def test_group_http_errors(self) -> None:
        self._request("POST", "/v1/groups", {
            "group_id": "g1", "creator_device_id": "creator",
            "member_device_ids": ["m1"]})
        cases = [
            ("GET", "/v1/groups/ghost", None, 404, "group_id"),
            ("POST", "/v1/groups/ghost/members",
             {"actor_device_id": "creator", "device_id": "m2"},
             404, "group_id"),
            ("POST", "/v1/groups/g1/members",
             {"actor_device_id": "m1", "device_id": "m2"},
             409, "actor_device_id"),
            ("POST", "/v1/groups/g1/members",
             {"actor_device_id": "creator", "device_id": "ghost"},
             404, "device_id"),
        ]
        for method, path, body, expected_status, expected_field in cases:
            status, received = self._request(method, path, body)
            self.assertEqual(status, expected_status, (method, path, received))
            self.assertEqual(received["field"], expected_field,
                             (method, path, received))

    def test_group_session_freeze_and_message_routes_over_http(self) -> None:
        self._request("POST", "/v1/groups", {
            "group_id": "g1", "creator_device_id": "creator",
            "member_device_ids": ["m1", "m2"]})
        status, frozen = self._request("POST", "/v1/group-sessions", {
            "group_id": "g1", "initiator_device_id": "creator",
            "ephemeral_key": _raw_key_b64()})
        self.assertEqual(status, 201)
        sid = frozen["session_id"]

        status, body = self._request(
            "GET", f"/v1/group-sessions/{sid}")
        self.assertEqual(status, 200)
        self.assertEqual(body, frozen)

        # m3 joins after the freeze; it cannot post or read the old session.
        self._request("POST", "/v1/groups/g1/members",
                      {"actor_device_id": "creator", "device_id": "m3"})
        status, body = self._request("POST", "/v1/messages", {
            "session_id": sid, "sender_device_id": "m3", "message_id": "x",
            "sequence": 1, "nonce": "n", "ciphertext": "c"})
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "sender_device_id")
        status, body = self._request(
            "GET", f"/v1/messages/{sid}?device_id=m3")
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "device_id")

        status, _ = self._request("GET", "/v1/group-sessions/ghost")
        self.assertEqual(status, 404)

    def test_url_encoded_group_id(self) -> None:
        group_id = "team a/b"
        status, body = self._request("POST", "/v1/groups", {
            "group_id": group_id, "creator_device_id": "creator",
            "member_device_ids": ["m1"]})
        self.assertEqual(status, 201)
        from urllib.parse import quote
        status, body = self._request(
            "GET", f"/v1/groups/{quote(group_id, safe='')}")
        self.assertEqual(status, 200)
        self.assertEqual(body["group_id"], group_id)


# --------------------------------------------------------------------------
# CLI (real subprocesses)
# --------------------------------------------------------------------------

class GroupCLITest(unittest.TestCase):
    def setUp(self) -> None:
        self.server, _ = create_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        for device_id in ("creator", "m1", "m2", "m3"):
            result = self._run(
                "register", "--user-id", "u", "--device-id", device_id,
                "--identity-key", _raw_key_b64())
            self.assertEqual(result.returncode, 0, result.stderr)

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _run(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "e2ee_backend",
             "--base-url", self.base_url, *arguments],
            capture_output=True, text=True, timeout=15)

    def test_group_create_show_add_remove(self) -> None:
        result = self._run(
            "group-create", "--group-id", "g1",
            "--creator-device-id", "creator",
            "--member-device-id", "m1", "--member-device-id", "m2")
        self.assertEqual(result.returncode, 0, result.stderr)
        line = result.stdout.strip()
        self.assertEqual(line.count("\n"), 0)
        created = json.loads(line)
        self.assertEqual(created["members"], ["creator", "m1", "m2"])
        self.assertEqual(created["revision"], 1)

        # Duplicate group id -> stderr JSON, exit 1.
        conflict = self._run(
            "group-create", "--group-id", "g1",
            "--creator-device-id", "creator", "--member-device-id", "m1")
        self.assertEqual(conflict.returncode, 1)
        self.assertEqual(conflict.stdout.strip(), "")
        self.assertEqual(json.loads(conflict.stderr.strip())["field"],
                         "group_id")

        shown = self._run("group-show", "g1")
        self.assertEqual(shown.returncode, 0, shown.stderr)
        self.assertEqual(json.loads(shown.stdout.strip()), created)
        missing = self._run("group-show", "ghost")
        self.assertEqual(missing.returncode, 1)
        self.assertEqual(json.loads(missing.stderr.strip())["field"],
                         "group_id")

        added = self._run(
            "group-add-member", "g1", "--actor-device-id", "creator",
            "--device-id", "m3")
        self.assertEqual(added.returncode, 0, added.stderr)
        self.assertEqual(json.loads(added.stdout.strip())["revision"], 2)
        again = self._run(
            "group-add-member", "g1", "--actor-device-id", "creator",
            "--device-id", "m3")
        self.assertEqual(again.returncode, 0)

        forbidden = self._run(
            "group-add-member", "g1", "--actor-device-id", "m1",
            "--device-id", "m2")
        self.assertEqual(forbidden.returncode, 1)
        self.assertEqual(json.loads(forbidden.stderr.strip())["field"],
                         "actor_device_id")

        removed = self._run(
            "group-remove-member", "g1", "--actor-device-id", "creator",
            "--device-id", "m3")
        self.assertEqual(removed.returncode, 0, removed.stderr)
        body = json.loads(removed.stdout.strip())
        self.assertEqual(body["members"], ["creator", "m1", "m2"])
        self.assertEqual(body["revision"], 3)

    def test_group_create_requires_at_least_one_member_locally(self) -> None:
        result = self._run(
            "group-create", "--group-id", "g2",
            "--creator-device-id", "creator")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stderr.strip())["field"],
                         "member_device_ids")

    def test_group_session_commands(self) -> None:
        self._run("group-create", "--group-id", "g1",
                  "--creator-device-id", "creator",
                  "--member-device-id", "m1", "--member-device-id", "m2")
        result = self._run(
            "create-group-session", "--group-id", "g1",
            "--initiator-device-id", "creator",
            "--ephemeral-key", _raw_key_b64())
        self.assertEqual(result.returncode, 0, result.stderr)
        created = json.loads(result.stdout.strip())
        self.assertEqual(created["members"], ["creator", "m1", "m2"])
        self.assertEqual(set(created), {
            "session_id", "group_id", "initiator_device_id", "ephemeral_key",
            "members", "revision", "created_at"})

        shown = self._run("show-group-session", created["session_id"])
        self.assertEqual(shown.returncode, 0, shown.stderr)
        self.assertEqual(json.loads(shown.stdout.strip()), created)

        missing = self._run("show-group-session", "ghost")
        self.assertEqual(missing.returncode, 1)
        self.assertEqual(json.loads(missing.stderr.strip())["field"],
                         "session_id")

        bad_key = self._run(
            "create-group-session", "--group-id", "g1",
            "--initiator-device-id", "creator", "--ephemeral-key", "garbage")
        self.assertEqual(bad_key.returncode, 1)
        self.assertEqual(json.loads(bad_key.stderr.strip())["field"],
                         "ephemeral_key")


if __name__ == "__main__":
    unittest.main()
