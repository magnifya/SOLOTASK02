"""Tests for groups and group sessions: validation, membership, freezing."""
import base64
import json
import os
import tempfile
import threading
import unittest

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import x25519

from e2ee_backend.models import Device
from e2ee_backend.persistence import attach_persistence
from e2ee_backend.service import DeviceService, ServiceError
from e2ee_backend.storage import DeviceStore, GroupError


def _raw_key_b64() -> str:
    key = x25519.X25519PrivateKey.generate().public_key()
    raw = key.public_bytes(serialization.Encoding.Raw,
                           serialization.PublicFormat.Raw)
    return base64.b64encode(raw).decode()


class GroupServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = DeviceService()
        # Bypass public-key validation for devices; group logic only cares
        # about identifiers and the active/revoked flag.
        for device_id in ("creator", "alice", "bob", "carol"):
            self.service.store.add_device(
                Device("u", device_id, "identity-key"))

    def _create(self, group_id: str = "g1",
                creator: str = "creator",
                members=None) -> dict:
        return self.service.create_group({
            "group_id": group_id,
            "creator_device_id": creator,
            "member_device_ids": list(members if members is not None
                                      else ["alice", "bob"]),
        })

    # -- creation ----------------------------------------------------------

    def test_create_group_returns_contract_fields(self) -> None:
        body = self._create()
        self.assertEqual(set(body),
                         {"group_id", "revision", "members", "created_at"})
        self.assertEqual(body["group_id"], "g1")
        self.assertEqual(body["revision"], 1)
        # Creator is always the first member; the others keep request order.
        self.assertEqual(body["members"], ["creator", "alice", "bob"])
        self.assertTrue(body["created_at"].endswith("+00:00"))

    def test_create_deduplicates_repeated_member_ids(self) -> None:
        body = self._create(members=["alice", "alice", "creator", "bob"])
        self.assertEqual(body["members"], ["creator", "alice", "bob"])

    def test_duplicate_group_id_is_conflict(self) -> None:
        self._create()
        with self.assertRaises(ServiceError) as ctx:
            self._create()
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "group_id")

    def test_unknown_creator_is_404(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self._create(creator="nobody")
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "creator_device_id")

    def test_revoked_creator_is_409(self) -> None:
        self.service.store.revoke_device("creator")
        with self.assertRaises(ServiceError) as ctx:
            self._create()
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "creator_device_id")

    def test_member_ids_need_not_be_registered_devices(self) -> None:
        body = self._create(members=["alice", "future-device"])
        self.assertIn("future-device", body["members"])

    def test_create_validation_errors(self) -> None:
        cases = [
            ({}, "group_id"),
            ({"group_id": "g2", "creator_device_id": "creator"},
             "member_device_ids"),
            ({"group_id": "g2", "creator_device_id": "creator",
              "member_device_ids": []}, "member_device_ids"),
            ({"group_id": "g2", "creator_device_id": "creator",
              "member_device_ids": ["ok", ""]}, "member_device_ids"),
            ({"group_id": "g2", "creator_device_id": "",
              "member_device_ids": ["alice"]}, "creator_device_id"),
            ({"group_id": "", "creator_device_id": "creator",
              "member_device_ids": ["alice"]}, "group_id"),
        ]
        for payload, field in cases:
            with self.subTest(payload=payload):
                with self.assertRaises(ServiceError) as ctx:
                    self.service.create_group(payload)
                self.assertEqual(ctx.exception.status_code, 400)
                self.assertTrue(ctx.exception.field.startswith(field),
                                ctx.exception.field)

    def test_non_object_body_is_400(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.create_group(["nope"])
        self.assertEqual(ctx.exception.field, "request_body")

    # -- lookup ------------------------------------------------------------

    def test_get_group(self) -> None:
        self._create()
        body = self.service.get_group("g1")
        self.assertEqual(body["members"], ["creator", "alice", "bob"])

    def test_get_unknown_group_is_404(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.get_group("missing")
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "group_id")

    # -- add member --------------------------------------------------------

    def test_add_new_member_is_201_and_bumps_revision(self) -> None:
        self._create()
        body, status = self.service.add_group_member(
            "g1", {"actor_device_id": "creator", "device_id": "carol"})
        self.assertEqual(status, 201)
        self.assertEqual(body["members"], ["creator", "alice", "bob", "carol"])
        self.assertEqual(body["revision"], 2)

    def test_add_existing_member_is_200_without_revision_change(self) -> None:
        self._create()
        body, status = self.service.add_group_member(
            "g1", {"actor_device_id": "creator", "device_id": "alice"})
        self.assertEqual(status, 200)
        self.assertEqual(body["revision"], 1)

    def test_add_unknown_group_is_404(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.add_group_member(
                "missing", {"actor_device_id": "creator", "device_id": "carol"})
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "group_id")

    def test_add_unknown_target_device_is_404(self) -> None:
        self._create()
        with self.assertRaises(ServiceError) as ctx:
            self.service.add_group_member(
                "g1", {"actor_device_id": "creator", "device_id": "nobody"})
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "device_id")

    def test_revoked_device_cannot_be_added(self) -> None:
        self._create()
        self.service.store.revoke_device("carol")
        with self.assertRaises(ServiceError) as ctx:
            self.service.add_group_member(
                "g1", {"actor_device_id": "creator", "device_id": "carol"})
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "device_id")

    def test_non_creator_actor_is_409(self) -> None:
        self._create()
        with self.assertRaises(ServiceError) as ctx:
            self.service.add_group_member(
                "g1", {"actor_device_id": "alice", "device_id": "carol"})
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "actor_device_id")

    def test_unknown_actor_is_404(self) -> None:
        self._create()
        with self.assertRaises(ServiceError) as ctx:
            self.service.add_group_member(
                "g1", {"actor_device_id": "nobody", "device_id": "carol"})
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "actor_device_id")

    def test_revoked_actor_is_409(self) -> None:
        self._create()
        self.service.store.revoke_device("creator")
        with self.assertRaises(ServiceError) as ctx:
            self.service.add_group_member(
                "g1", {"actor_device_id": "creator", "device_id": "carol"})
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "actor_device_id")

    def test_member_change_validation(self) -> None:
        for payload in ({}, {"actor_device_id": "creator"},
                        {"actor_device_id": "creator", "device_id": ""},
                        {"actor_device_id": 42, "device_id": "carol"}):
            with self.subTest(payload=payload):
                with self.assertRaises(ServiceError) as ctx:
                    self.service.add_group_member("g1", payload)
                self.assertEqual(ctx.exception.status_code, 400)

    # -- remove member -----------------------------------------------------

    def test_remove_member_is_200_and_bumps_revision(self) -> None:
        self._create()
        body = self.service.remove_group_member(
            "g1", {"actor_device_id": "creator", "device_id": "alice"})
        self.assertEqual(body["members"], ["creator", "bob"])
        self.assertEqual(body["revision"], 2)

    def test_remove_absent_member_is_idempotent_200(self) -> None:
        self._create()
        body = self.service.remove_group_member(
            "g1", {"actor_device_id": "creator", "device_id": "carol"})
        self.assertEqual(body["revision"], 1)
        self.assertEqual(body["members"], ["creator", "alice", "bob"])

    def test_remove_unknown_device_id_is_404(self) -> None:
        self._create()
        with self.assertRaises(ServiceError) as ctx:
            self.service.remove_group_member(
                "g1", {"actor_device_id": "creator", "device_id": "nobody"})
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "device_id")

    def test_revoked_member_can_still_be_removed(self) -> None:
        self._create()
        self.service.store.revoke_device("alice")
        body = self.service.remove_group_member(
            "g1", {"actor_device_id": "creator", "device_id": "alice"})
        self.assertNotIn("alice", body["members"])

    def test_remove_by_non_creator_is_409(self) -> None:
        self._create()
        with self.assertRaises(ServiceError) as ctx:
            self.service.remove_group_member(
                "g1", {"actor_device_id": "bob", "device_id": "alice"})
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "actor_device_id")

    def test_creator_cannot_be_removed(self) -> None:
        self._create()
        body = self.service.remove_group_member(
            "g1", {"actor_device_id": "creator", "device_id": "creator"})
        # Still 200; the creator remains the first member, revision untouched.
        self.assertEqual(body["members"][0], "creator")
        self.assertEqual(body["revision"], 1)


class GroupSessionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = DeviceService()
        for device_id in ("creator", "alice", "bob", "carol", "outsider"):
            self.service.store.add_device(
                Device("u", device_id, "identity-key"))
        self.service.create_group({
            "group_id": "g1", "creator_device_id": "creator",
            "member_device_ids": ["alice", "bob"]})

    def _create_session(self, group_id="g1", initiator="creator",
                        ephemeral_key="epk"):
        return self.service.create_group_session({
            "group_id": group_id,
            "initiator_device_id": initiator,
            "ephemeral_key": ephemeral_key})

    def test_create_returns_contract_fields(self) -> None:
        body = self._create_session()
        self.assertEqual(set(body), {"session_id", "group_id",
                                     "initiator_device_id", "ephemeral_key",
                                     "revision", "members", "created_at"})
        self.assertEqual(body["group_id"], "g1")
        self.assertEqual(body["revision"], 1)
        self.assertEqual(body["members"], ["creator", "alice", "bob"])
        self.assertTrue(body["created_at"].endswith("+00:00"))

    def test_every_post_creates_a_new_session(self) -> None:
        first = self._create_session()
        second = self._create_session()
        self.assertNotEqual(first["session_id"], second["session_id"])
        # Both snapshots remain independently queryable.
        self.assertEqual(
            self.service.get_group_session(first["session_id"])["session_id"],
            first["session_id"])

    def test_membership_is_frozen_at_creation(self) -> None:
        frozen = self._create_session()
        # Change the group after the freeze.
        self.service.add_group_member(
            "g1", {"actor_device_id": "creator", "device_id": "carol"})
        self.service.remove_group_member(
            "g1", {"actor_device_id": "creator", "device_id": "alice"})
        view = self.service.get_group_session(frozen["session_id"])
        self.assertEqual(view["members"], ["creator", "alice", "bob"])
        self.assertEqual(view["revision"], 1)
        # The group itself did advance.
        self.assertEqual(self.service.get_group("g1")["revision"], 3)

    def test_new_session_reflects_current_roster(self) -> None:
        self.service.add_group_member(
            "g1", {"actor_device_id": "creator", "device_id": "carol"})
        body = self._create_session()
        self.assertEqual(body["members"],
                         ["creator", "alice", "bob", "carol"])
        self.assertEqual(body["revision"], 2)

    def test_initiator_must_be_a_current_member(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self._create_session(initiator="outsider")
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "initiator_device_id")

    def test_removed_member_can_no_longer_initiate(self) -> None:
        self.service.remove_group_member(
            "g1", {"actor_device_id": "creator", "device_id": "alice"})
        with self.assertRaises(ServiceError) as ctx:
            self._create_session(initiator="alice")
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "initiator_device_id")

    def test_unknown_group_initiator_are_404(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self._create_session(group_id="missing")
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "group_id")
        with self.assertRaises(ServiceError) as ctx:
            self._create_session(initiator="nobody")
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "initiator_device_id")

    def test_revoked_initiator_is_409(self) -> None:
        self.service.store.revoke_device("alice")
        with self.assertRaises(ServiceError) as ctx:
            self._create_session(initiator="alice")
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "initiator_device_id")

    def test_create_validation(self) -> None:
        for payload, field in (
            ({}, "group_id"),
            ({"group_id": "g1", "initiator_device_id": "creator"},
             "ephemeral_key"),
            ({"group_id": "g1", "initiator_device_id": "",
              "ephemeral_key": "ek"}, "initiator_device_id"),
        ):
            with self.subTest(payload=payload):
                with self.assertRaises(ServiceError) as ctx:
                    self.service.create_group_session(payload)
                self.assertEqual(ctx.exception.status_code, 400)
                self.assertEqual(ctx.exception.field, field)

    def test_get_unknown_session_is_404(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.get_group_session("missing")
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "session_id")


class GroupMessageAccessTest(unittest.TestCase):
    """Messages posted into a group session are bounded to frozen members."""

    def setUp(self) -> None:
        self.service = DeviceService()
        for device_id in ("creator", "alice", "bob", "carol"):
            self.service.store.add_device(
                Device("u", device_id, "identity-key"))
        self.service.create_group({
            "group_id": "g1", "creator_device_id": "creator",
            "member_device_ids": ["alice", "bob"]})
        self.session = self.service.create_group_session({
            "group_id": "g1", "initiator_device_id": "creator",
            "ephemeral_key": "epk"})
        self.sid = self.session["session_id"]

    def _post(self, sender: str, message_id: str, sequence: int) -> None:
        self.service.post_message({
            "session_id": self.sid, "sender_device_id": sender,
            "message_id": message_id, "sequence": sequence,
            "nonce": f"nonce-{message_id}", "ciphertext": "ct"})

    def test_frozen_members_post_and_read(self) -> None:
        self._post("creator", "m1", 1)
        self._post("alice", "m2", 2)
        for reader in ("creator", "alice", "bob"):
            body = self.service.list_messages(self.sid, reader, 0, 100)
            self.assertEqual([m["message_id"] for m in body["messages"]],
                             ["m1", "m2"])

    def test_non_frozen_member_cannot_post(self) -> None:
        # carol joins the group only after the freeze.
        self.service.add_group_member(
            "g1", {"actor_device_id": "creator", "device_id": "carol"})
        with self.assertRaises(ServiceError) as ctx:
            self._post("carol", "m1", 1)
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "sender_device_id")

    def test_non_frozen_member_cannot_read(self) -> None:
        self._post("creator", "m1", 1)
        self.service.add_group_member(
            "g1", {"actor_device_id": "creator", "device_id": "carol"})
        with self.assertRaises(ServiceError) as ctx:
            self.service.list_messages(self.sid, "carol", 0, 100)
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "device_id")

    def test_member_removed_after_freeze_keeps_frozen_access(self) -> None:
        # Membership is read against the frozen snapshot: removing alice from
        # the live group after the freeze neither revokes her device nor
        # changes the snapshot, so she can still read the frozen session.
        self._post("creator", "m1", 1)
        self.service.remove_group_member(
            "g1", {"actor_device_id": "creator", "device_id": "alice"})
        body = self.service.list_messages(self.sid, "alice", 0, 100)
        self.assertEqual(len(body["messages"]), 1)

    def test_revoked_frozen_member_cannot_read(self) -> None:
        # Device revocation is independent of group membership and still gates
        # message reads, even for a device frozen into the session.
        self._post("creator", "m1", 1)
        self.service.store.revoke_device("alice")
        with self.assertRaises(ServiceError) as ctx:
            self.service.list_messages(self.sid, "alice", 0, 100)
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "device_id")

    def test_messages_survive_group_removal_of_sender(self) -> None:
        # alice is frozen in, posts, then is removed from the live group.
        # The frozen snapshot still lists her, so she keeps access.
        self._post("alice", "m1", 1)
        self.service.remove_group_member(
            "g1", {"actor_device_id": "creator", "device_id": "alice"})
        body = self.service.list_messages(self.sid, "alice", 0, 100)
        self.assertEqual(len(body["messages"]), 1)


class GroupLinearizationTest(unittest.TestCase):
    """Group creation is linearized against device revocation under one lock."""

    def test_revoke_first_then_create_loses(self) -> None:
        store = DeviceStore()
        store.add_device(Device("u", "creator", "ik"))
        store.revoke_device("creator")
        with self.assertRaises(GroupError):
            store.create_group("g1", "creator", ["alice"])
        self.assertIsNone(store.get_group("g1"))

    def test_create_first_then_revoke_wins(self) -> None:
        store = DeviceStore()
        store.add_device(Device("u", "creator", "ik"))
        group = store.create_group("g1", "creator", ["alice"])
        store.revoke_device("creator")
        # The group survives; its already-frozen data does not change.
        self.assertEqual(store.get_group("g1"), group)

    def test_concurrent_create_and_revoke_have_only_valid_outcomes(self) -> None:
        # Repeat the race: the result is always one of the two legal
        # linearizations — either the group exists or it does not, never a
        # half-applied state.
        created = rejected = 0
        for iteration in range(100):
            store = DeviceStore()
            store.add_device(Device("u", "creator", "ik"))
            start = threading.Barrier(2)
            outcome = []

            def race(create: bool) -> None:
                start.wait()
                if create:
                    try:
                        store.create_group(f"g-{iteration}", "creator",
                                           ["alice"])
                        outcome.append("created")
                    except GroupError:
                        outcome.append("rejected")
                else:
                    store.revoke_device("creator")

            threads = [threading.Thread(target=race, args=(True,)),
                       threading.Thread(target=race, args=(False,))]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.assertIn(outcome[0], ("created", "rejected"))
            group = store.get_group(f"g-{iteration}")
            if outcome[0] == "created":
                created += 1
                self.assertIsNotNone(group)
            else:
                rejected += 1
                self.assertIsNone(group)
        # Across 100 races both orders are realistically observed; the test
        # tolerates a one-sided scheduler but asserts full state validity.
        self.assertGreaterEqual(created + rejected, 100)


class GroupPersistenceTest(unittest.TestCase):
    def test_groups_and_group_sessions_survive_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "state.json")
            service = DeviceService()
            attach_persistence(service, path)
            for device_id in ("creator", "alice", "bob"):
                service.store.add_device(Device("u", device_id, "ik"))
            service.create_group({
                "group_id": "g1", "creator_device_id": "creator",
                "member_device_ids": ["alice"]})
            frozen = service.create_group_session({
                "group_id": "g1", "initiator_device_id": "creator",
                "ephemeral_key": "epk"})
            # A post-freeze change must not alter the persisted snapshot.
            service.add_group_member(
                "g1", {"actor_device_id": "creator", "device_id": "bob"})

            restarted = DeviceService()
            attach_persistence(restarted, path)

            group = restarted.get_group("g1")
            self.assertEqual(group["revision"], 2)
            self.assertEqual(group["members"], ["creator", "alice", "bob"])
            snapshot = restarted.get_group_session(frozen["session_id"])
            self.assertEqual(snapshot["members"], ["creator", "alice"])
            self.assertEqual(snapshot["revision"], 1)
            self.assertEqual(snapshot["ephemeral_key"], "epk")

            with open(path, encoding="utf-8") as handle:
                document = json.load(handle)
            self.assertEqual(document["version"], 1)
            self.assertIn("groups", document)
            self.assertIn("group_sessions", document)


if __name__ == "__main__":
    unittest.main()
