"""Startup consistency checks for the persisted version-1 state document.

A ``--data-file`` document is only trusted when every section is internally
consistent: message streams must belong to saved sessions with well-typed,
consecutively sequenced envelopes, unique ids/nonces and known senders
(frozen members for group sessions); ``used_nonces`` must equal the nonces
in the message history exactly; delivery records must reference stored
messages with matching attempt counts and a coherent ack flag/cursor pair.
Any violation refuses startup with ``field=data_file`` and leaves the file
(bytes and inode) untouched.
"""
import copy
import json
import os
import shutil
import tempfile
import unittest
from typing import Any, Dict, Tuple

from e2ee_backend.models import Device, SignedPreKey
from e2ee_backend.persistence import StateFileError, attach_persistence
from e2ee_backend.service import DeviceService, ServiceError


def build_fixture(directory: str) -> Tuple[DeviceService, str, str, str]:
    """Persist a service with a direct session and a group session.

    Returns ``(service, path, direct_session_id, group_session_id)``:

    * devices alice/bob/carol, each with one signed pre-key;
    * a direct session alice->bob holding messages m1/m2 (seq 1/2), with
      m1 retried twice (attempt ids a1/a2) and acked;
    * a group (alice creator, bob member; carol registered but outside)
      whose frozen session holds one message from bob.
    """
    path = os.path.join(directory, "state.json")
    service = DeviceService()
    attach_persistence(service, path)
    for device_id in ("alice", "bob", "carol"):
        service.store.add_device(Device(
            "u", device_id, "ik",
            prekeys=[SignedPreKey(f"pk-{device_id}", f"pub-{device_id}")]))
    direct = service.store.create_session("alice", "bob", "pk-bob", "epk")
    sid = direct.session_id
    for sequence in (1, 2):
        service.post_message({
            "session_id": sid, "sender_device_id": "alice",
            "message_id": f"m{sequence}", "sequence": sequence,
            "nonce": f"n{sequence}", "ciphertext": "ct"})
    service.retry_message(sid, "m1", {"device_id": "bob", "attempt_id": "a1"})
    service.retry_message(sid, "m1", {"device_id": "bob", "attempt_id": "a2"})
    service.ack_message(sid, {"device_id": "bob", "message_id": "m1",
                              "sequence": 1})
    service.create_group({"group_id": "g1", "creator_device_id": "alice",
                          "member_device_ids": ["bob"]})
    group_session = service.create_group_session({
        "group_id": "g1", "initiator_device_id": "alice",
        "ephemeral_key": "gepk"})
    gsid = group_session["session_id"]
    service.post_message({
        "session_id": gsid, "sender_device_id": "bob",
        "message_id": "gm1", "sequence": 1, "nonce": "gn1",
        "ciphertext": "gct"})
    return service, path, sid, gsid


class ConsistencyFixtureMixin(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        _service, self.path, self.sid, self.gsid = build_fixture(
            self.directory)
        with open(self.path, encoding="utf-8") as handle:
            self.document = json.load(handle)

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def assert_rejected(self, document: Dict[str, Any]) -> None:
        """A mutated document refuses startup without touching the file."""
        path = os.path.join(self.directory, "bad.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        before_bytes = open(path, "rb").read()
        before_inode = os.stat(path).st_ino
        with self.assertRaises(StateFileError):
            attach_persistence(DeviceService(), path)
        self.assertEqual(open(path, "rb").read(), before_bytes)
        self.assertEqual(os.stat(path).st_ino, before_inode)

    def mutate_messages(self, sid: str, index: int, **changes: Any
                        ) -> Dict[str, Any]:
        document = copy.deepcopy(self.document)
        envelope = document["messages"][sid][index]
        envelope.update(changes)
        return document


class WellFormedDocumentTest(ConsistencyFixtureMixin):
    def test_fixture_document_loads_and_preserves_state(self) -> None:
        service = DeviceService()
        attach_persistence(service, self.path)
        # Direct-session history and delivery state survive the restart.
        body = service.list_messages(self.sid, "bob", 0, 100)
        self.assertEqual([m["message_id"] for m in body["messages"]],
                         ["m1", "m2"])
        status = service.message_status(self.sid, "m1", "bob")
        self.assertEqual((status["status"], status["attempts"]),
                         ("acked", 2))
        # Replay protection was restored: the old nonce is rejected.
        with self.assertRaises(ServiceError) as caught:
            service.post_message({
                "session_id": self.sid, "sender_device_id": "alice",
                "message_id": "m3", "sequence": 3, "nonce": "n1",
                "ciphertext": "ct"})
        self.assertEqual((caught.exception.status_code,
                          caught.exception.field), (409, "nonce"))

    def test_missing_file_creates_version1_document(self) -> None:
        path = os.path.join(self.directory, "fresh.json")
        service = DeviceService()
        attach_persistence(service, path)
        with open(path, encoding="utf-8") as handle:
            document = json.load(handle)
        self.assertEqual(document["version"], 1)

    def test_legacy_file_without_used_nonces_rebuilds_replay_guard(
            self) -> None:
        document = copy.deepcopy(self.document)
        del document["used_nonces"]
        path = os.path.join(self.directory, "legacy.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        service = DeviceService()
        attach_persistence(service, path)
        self.assertEqual(service.store._used_nonces[self.sid], {"n1", "n2"})
        with self.assertRaises(ServiceError) as caught:
            service.post_message({
                "session_id": self.sid, "sender_device_id": "alice",
                "message_id": "m3", "sequence": 3, "nonce": "n2",
                "ciphertext": "ct"})
        self.assertEqual(caught.exception.field, "nonce")

    def test_revoked_sender_history_still_loads(self) -> None:
        service = DeviceService()
        attach_persistence(service, self.path)
        service.revoke_device("alice")
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        body = restarted.list_messages(self.sid, "bob", 0, 100)
        self.assertEqual([m["sender_device_id"] for m in body["messages"]],
                         ["alice", "alice"])


class MessageSectionConsistencyTest(ConsistencyFixtureMixin):
    def test_messages_must_be_an_object(self) -> None:
        document = copy.deepcopy(self.document)
        document["messages"] = [self.sid]
        self.assert_rejected(document)

    def test_stream_key_must_be_a_saved_session(self) -> None:
        document = copy.deepcopy(self.document)
        document["messages"]["ghost-session"] = []
        self.assert_rejected(document)

    def test_envelope_session_id_must_match_the_key(self) -> None:
        self.assert_rejected(
            self.mutate_messages(self.sid, 0, session_id=self.gsid))

    def test_envelope_field_types_are_checked(self) -> None:
        bad_changes = [
            {"sequence": "1"},
            {"sequence": True},
            {"nonce": 7},
            {"message_id": 3},
            {"ciphertext": None},
            {"sender_device_id": 42},
            {"created_at": 0},
        ]
        for changes in bad_changes:
            self.assert_rejected(self.mutate_messages(self.sid, 0, **changes))

    def test_sequence_must_start_at_one_and_run_consecutively(self) -> None:
        # Starts at 2.
        document = copy.deepcopy(self.document)
        document["messages"][self.sid][0]["sequence"] = 2
        document["messages"][self.sid][1]["sequence"] = 3
        self.assert_rejected(document)
        # Gap between 1 and 3.
        self.assert_rejected(self.mutate_messages(self.sid, 1, sequence=3))
        # Repeated sequence.
        self.assert_rejected(self.mutate_messages(self.sid, 1, sequence=1))

    def test_message_id_must_be_unique_within_session(self) -> None:
        self.assert_rejected(self.mutate_messages(self.sid, 1,
                                                  message_id="m1"))

    def test_nonce_must_be_unique_within_session(self) -> None:
        document = self.mutate_messages(self.sid, 1, nonce="n1")
        # Keep used_nonces consistent with the mutated history so only the
        # in-stream duplicate is under test.
        document["used_nonces"][self.sid] = ["n1"]
        self.assert_rejected(document)

    def test_sender_device_must_exist(self) -> None:
        self.assert_rejected(
            self.mutate_messages(self.sid, 0, sender_device_id="ghost"))

    def test_group_message_sender_must_be_a_frozen_member(self) -> None:
        # carol is registered but was never frozen into the group session.
        self.assert_rejected(
            self.mutate_messages(self.gsid, 0, sender_device_id="carol"))


class UsedNoncesConsistencyTest(ConsistencyFixtureMixin):
    def test_section_type_is_checked(self) -> None:
        document = copy.deepcopy(self.document)
        document["used_nonces"] = ["n1"]
        self.assert_rejected(document)
        document = copy.deepcopy(self.document)
        document["used_nonces"][self.sid] = "n1"
        self.assert_rejected(document)
        document = copy.deepcopy(self.document)
        document["used_nonces"][self.sid] = ["n1", 2]
        self.assert_rejected(document)

    def test_extra_nonce_is_rejected(self) -> None:
        document = copy.deepcopy(self.document)
        document["used_nonces"][self.sid].append("n-extra")
        self.assert_rejected(document)

    def test_missing_nonce_is_rejected(self) -> None:
        document = copy.deepcopy(self.document)
        document["used_nonces"][self.sid] = ["n1"]
        self.assert_rejected(document)

    def test_unknown_session_key_is_rejected(self) -> None:
        document = copy.deepcopy(self.document)
        document["used_nonces"]["ghost-session"] = ["n9"]
        self.assert_rejected(document)

    def test_duplicate_nonce_in_list_is_rejected(self) -> None:
        document = copy.deepcopy(self.document)
        document["used_nonces"][self.sid] = ["n1", "n2", "n1"]
        self.assert_rejected(document)


class DeliveryConsistencyTest(ConsistencyFixtureMixin):
    def _mutate_delivery(self, **changes: Any) -> Dict[str, Any]:
        document = copy.deepcopy(self.document)
        record = next(r for r in document["delivery"]
                      if r["message_id"] == "m1")
        record.update(changes)
        return document

    def test_delivery_must_reference_an_existing_message(self) -> None:
        self.assert_rejected(self._mutate_delivery(message_id="ghost"))
        self.assert_rejected(self._mutate_delivery(session_id="ghost"))

    def test_attempt_ids_must_be_distinct_strings(self) -> None:
        self.assert_rejected(self._mutate_delivery(
            attempt_ids=["a1", "a1"], attempts=2))
        self.assert_rejected(self._mutate_delivery(
            attempt_ids=["a1", 2], attempts=2))
        self.assert_rejected(self._mutate_delivery(attempt_ids="a1"))

    def test_attempts_must_equal_the_attempt_id_count(self) -> None:
        self.assert_rejected(self._mutate_delivery(attempts=1))
        self.assert_rejected(self._mutate_delivery(attempts=3))

    def test_field_types_are_checked(self) -> None:
        self.assert_rejected(self._mutate_delivery(attempts="2"))
        self.assert_rejected(self._mutate_delivery(attempts=True))
        self.assert_rejected(self._mutate_delivery(acked=1))
        self.assert_rejected(self._mutate_delivery(ack_sequence="1"))
        self.assert_rejected(self._mutate_delivery(ack_sequence=True))

    def test_acked_record_must_pin_the_message_sequence(self) -> None:
        self.assert_rejected(self._mutate_delivery(ack_sequence=2))
        self.assert_rejected(self._mutate_delivery(ack_sequence=0))

    def test_unacked_record_must_have_zero_ack_sequence(self) -> None:
        self.assert_rejected(self._mutate_delivery(acked=False,
                                                   ack_sequence=1))

    def test_unacked_record_with_zero_ack_sequence_loads(self) -> None:
        document = self._mutate_delivery(acked=False, ack_sequence=0)
        path = os.path.join(self.directory, "unacked.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        service = DeviceService()
        attach_persistence(service, path)
        status = service.message_status(self.sid, "m1", "bob")
        self.assertEqual((status["status"], status["attempts"]),
                         ("pending", 2))

    def test_duplicate_delivery_record_is_rejected(self) -> None:
        document = copy.deepcopy(self.document)
        record = next(r for r in document["delivery"]
                      if r["message_id"] == "m1")
        document["delivery"].append(dict(record))
        self.assert_rejected(document)


if __name__ == "__main__":
    unittest.main()
