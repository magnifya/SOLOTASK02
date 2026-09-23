"""Idempotent replay requests must be state-free transactions.

A replay that changes nothing — a delivery retry carrying an already-recorded
``attempt_id`` (even after an ack), a device's repeated ack, or the second and
later calls to either revocation endpoint — returns the same 200 response with
the same fields, but must be invisible to the durability layer:

* no persistence hook fires, so nothing is written (the formal file's bytes
  and inode are identical) and ``commit_seq`` does not advance;
* no ``key_events`` entry is appended and no lazy anchor migration is run for
  a legacy file that lacks the ``key_events`` section;
* a failing persistence layer does not make such a replay fail;
* concurrent identical requests linearize to exactly one state change and one
  durable commit.

Requests that *do* change ``attempts``, ``acked`` or a revocation marker keep
the existing atomic, fully-rollback 503 semantics (covered elsewhere).
"""
from __future__ import annotations

import base64
import json
import os
import shutil
import tempfile
import threading
import unittest
from typing import Any, Dict, Tuple

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import x25519

from e2ee_backend.persistence import (
    PersistenceUnavailable,
    attach_persistence,
)
from e2ee_backend.service import DeviceService


def _raw_key_b64() -> str:
    key = x25519.X25519PrivateKey.generate().public_key()
    raw = key.public_bytes(serialization.Encoding.Raw,
                           serialization.PublicFormat.Raw)
    return base64.b64encode(raw).decode()


def _message_payload(session_id: str, message_id: str = "m1",
                     sequence: int = 1, sender: str = "d1") -> dict:
    return {
        "session_id": session_id,
        "sender_device_id": sender,
        "message_id": message_id,
        "sequence": sequence,
        "nonce": base64.b64encode(f"nonce-{message_id}".encode()).decode(),
        "ciphertext": base64.b64encode(b"ciphertext-and-tag").decode(),
    }


class PersistedIdempotentReplayTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.path = os.path.join(self.directory, "state.json")
        self.service = DeviceService()
        self.state_store = attach_persistence(self.service, self.path)
        for device_id in ("d1", "d2"):
            self.service.register({
                "user_id": "u1", "device_id": device_id,
                "identity_key": _raw_key_b64(),
                "signed_prekeys": [{"key_id": "k1",
                                    "public_key": _raw_key_b64()}]})
        self.sid = self.service.create_session({
            "initiator_device_id": "d1", "recipient_device_id": "d2",
            "prekey_id": "k1", "ephemeral_key": _raw_key_b64(),
        })["session_id"]
        self.service.post_message(_message_payload(self.sid))

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def _file_state(self) -> Tuple[bytes, int, int]:
        with open(self.path, "rb") as handle:
            data = handle.read()
        return data, os.stat(self.path).st_ino, json.loads(data)["commit_seq"]

    def test_same_attempt_id_replay_is_200_original_and_writes_nothing(self):
        body, status = self.service.retry_message(
            self.sid, "m1", {"device_id": "d2", "attempt_id": "a1"})
        self.assertEqual(status, 201)
        self.assertEqual(list(body),
                         ["session_id", "message_id", "status", "attempts",
                          "sequence"])
        before = self._file_state()
        replay, replay_status = self.service.retry_message(
            self.sid, "m1", {"device_id": "d2", "attempt_id": "a1"})
        self.assertEqual(replay_status, 200)
        self.assertEqual(replay, body)
        self.assertEqual(self._file_state(), before)

    def test_attempt_replay_after_ack_still_writes_nothing(self):
        self.service.retry_message(
            self.sid, "m1", {"device_id": "d2", "attempt_id": "a1"})
        self.service.ack_message(
            self.sid, {"device_id": "d2", "message_id": "m1", "sequence": 1})
        before = self._file_state()
        body, status = self.service.retry_message(
            self.sid, "m1", {"device_id": "d2", "attempt_id": "a1"})
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "acked")
        self.assertEqual(body["attempts"], 1)
        self.assertEqual(self._file_state(), before)

    def test_new_attempt_id_still_commits_exactly_once(self):
        self.service.retry_message(
            self.sid, "m1", {"device_id": "d2", "attempt_id": "a1"})
        before = self._file_state()
        body, status = self.service.retry_message(
            self.sid, "m1", {"device_id": "d2", "attempt_id": "a2"})
        self.assertEqual(status, 200)
        self.assertEqual(body["attempts"], 2)
        _bytes, _ino, seq = self._file_state()
        self.assertEqual(seq, before[2] + 1)

    def test_duplicate_ack_is_200_original_and_writes_nothing(self):
        payload = {"device_id": "d2", "message_id": "m1", "sequence": 1}
        first, first_status = self.service.ack_message(self.sid, payload)
        self.assertEqual(first_status, 201)
        self.assertEqual(list(first),
                         ["session_id", "message_id", "status", "attempts",
                          "sequence"])
        before = self._file_state()
        again, again_status = self.service.ack_message(self.sid, dict(payload))
        self.assertEqual(again_status, 200)
        self.assertEqual(again, first)
        self.assertEqual(self._file_state(), before)

    def test_duplicate_prekey_revoke_writes_nothing(self):
        first = self.service.revoke_prekey("d1", "k1")
        self.assertEqual(first,
                         {"device_id": "d1", "key_id": "k1", "revoked": True})
        self.assertEqual(list(first), ["device_id", "key_id", "revoked"])
        before = self._file_state()
        again = self.service.revoke_prekey("d1", "k1")
        self.assertEqual(again, first)
        self.assertEqual(list(again), ["device_id", "key_id", "revoked"])
        self.assertEqual(self._file_state(), before)

    def test_duplicate_device_revoke_writes_nothing(self):
        first = self.service.revoke_device("d1")
        self.assertEqual(first, {"device_id": "d1", "revoked": True})
        self.assertEqual(list(first), ["device_id", "revoked"])
        before = self._file_state()
        again = self.service.revoke_device("d1")
        self.assertEqual(again, first)
        self.assertEqual(list(again), ["device_id", "revoked"])
        self.assertEqual(self._file_state(), before)

    def test_revocation_replays_append_no_audit_events(self):
        self.service.revoke_prekey("d1", "k1")
        self.service.revoke_prekey("d1", "k1")
        self.service.revoke_device("d1")
        self.service.revoke_device("d1")
        with open(self.path, encoding="utf-8") as handle:
            document = json.load(handle)
        d1_events = [event for event in document["key_events"]
                     if event["device_id"] == "d1"]
        self.assertEqual([event["type"] for event in d1_events],
                         ["registered", "prekey_revoked", "device_revoked"])

    def test_replays_succeed_even_when_persistence_is_broken(self):
        # Establish committed state, then break every durable write. A replay
        # never opens a transaction, so it still returns 200 instead of 503.
        self.service.retry_message(
            self.sid, "m1", {"device_id": "d2", "attempt_id": "a1"})
        self.service.ack_message(
            self.sid, {"device_id": "d2", "message_id": "m1", "sequence": 1})
        self.service.revoke_prekey("d1", "k1")
        self.service.revoke_device("d1")
        with open(self.path, "rb") as handle:
            good = handle.read()

        def fail_save(state: Dict[str, Any]) -> None:
            raise OSError("simulated disk failure")

        self.state_store.save = fail_save  # type: ignore[assignment]
        body, status = self.service.retry_message(
            self.sid, "m1", {"device_id": "d2", "attempt_id": "a1"})
        self.assertEqual((status, body["status"], body["attempts"]),
                         (200, "acked", 1))
        body, status = self.service.ack_message(
            self.sid, {"device_id": "d2", "message_id": "m1", "sequence": 1})
        self.assertEqual((status, body["status"]), (200, "acked"))
        self.assertEqual(
            self.service.revoke_prekey("d1", "k1"),
            {"device_id": "d1", "key_id": "k1", "revoked": True})
        self.assertEqual(self.service.revoke_device("d1"),
                         {"device_id": "d1", "revoked": True})
        with open(self.path, "rb") as handle:
            self.assertEqual(handle.read(), good)

    def test_changing_request_under_broken_persistence_rolls_back(self):
        def fail_save(state: Dict[str, Any]) -> None:
            raise OSError("simulated disk failure")

        self.state_store.save = fail_save  # type: ignore[assignment]
        with self.assertRaises(PersistenceUnavailable):
            self.service.retry_message(
                self.sid, "m1", {"device_id": "d2", "attempt_id": "a1"})
        with self.assertRaises(PersistenceUnavailable):
            self.service.ack_message(
                self.sid,
                {"device_id": "d2", "message_id": "m1", "sequence": 1})
        with self.assertRaises(PersistenceUnavailable):
            self.service.revoke_device("d1")
        self.assertFalse(self.service.store.find_by_device_id("d1").revoked)
        self.assertNotIn((self.sid, "m1"), self.service.store._delivery)


class ConcurrentIdempotentReplayTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.path = os.path.join(self.directory, "state.json")
        self.service = DeviceService()
        attach_persistence(self.service, self.path)
        for device_id in ("g1", "g2"):
            self.service.register({
                "user_id": "u1", "device_id": device_id,
                "identity_key": _raw_key_b64(),
                "signed_prekeys": [{"key_id": "k1",
                                    "public_key": _raw_key_b64()}]})
        self.service.create_group({
            "group_id": "team", "creator_device_id": "g1",
            "member_device_ids": ["g2"]})
        self.gsid = self.service.create_group_session({
            "group_id": "team", "initiator_device_id": "g1",
            "ephemeral_key": _raw_key_b64(),
        })["session_id"]
        self.service.post_message(
            _message_payload(self.gsid, "gm1", sender="g1"))

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def _commit_seq(self) -> int:
        with open(self.path, encoding="utf-8") as handle:
            return json.load(handle)["commit_seq"]

    def _race(self, target) -> None:
        errors = []

        def worker() -> None:
            try:
                target()
            except Exception as error:  # pragma: no cover - failure surface
                errors.append(error)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(errors, [])

    def test_concurrent_same_attempt_id_changes_state_once(self):
        before = self._commit_seq()
        self._race(lambda: self.service.retry_message(
            self.gsid, "gm1",
            {"device_id": "g2", "attempt_id": "shared"}))
        self.assertEqual(self._commit_seq(), before + 1)
        state = self.service.store._group_delivery[
            (self.gsid, "gm1", "g2")]
        self.assertEqual(state.attempts, 1)
        self.assertEqual(state.attempt_ids, {"shared"})

    def test_concurrent_duplicate_acks_commit_once(self):
        self.service.retry_message(
            self.gsid, "gm1", {"device_id": "g2", "attempt_id": "a1"})
        before = self._commit_seq()
        self._race(lambda: self.service.ack_message(
            self.gsid,
            {"device_id": "g2", "message_id": "gm1", "sequence": 1}))
        self.assertEqual(self._commit_seq(), before + 1)
        self.assertTrue(self.service.store._group_delivery[
            (self.gsid, "gm1", "g2")].acked)

    def test_concurrent_revokes_commit_once(self):
        before = self._commit_seq()
        self._race(lambda: self.service.revoke_device("g2"))
        self.assertEqual(self._commit_seq(), before + 1)
        self.assertTrue(self.service.store.find_by_device_id("g2").revoked)


class LegacyAnchorReplayTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.path = os.path.join(self.directory, "state.json")
        service = DeviceService()
        attach_persistence(service, self.path)
        for device_id in ("L1", "L2", "L3"):
            service.register({
                "user_id": "u1", "device_id": device_id,
                "identity_key": _raw_key_b64(),
                "signed_prekeys": [{"key_id": "k1",
                                    "public_key": _raw_key_b64()}]})
        sid = service.create_session({
            "initiator_device_id": "L2", "recipient_device_id": "L1",
            "prekey_id": "k1", "ephemeral_key": _raw_key_b64(),
        })["session_id"]
        service.post_message(_message_payload(sid, sender="L2"))
        service.retry_message(
            sid, "m1", {"device_id": "L1", "attempt_id": "a1"})
        service.ack_message(
            sid, {"device_id": "L1", "message_id": "m1", "sequence": 1})
        service.revoke_device("L3")
        service.revoke_prekey("L2", "k1")
        self.sid = sid
        # Rewrite as a legacy version-1 document without the audit section:
        # a replay against it must not trigger the lazy anchor migration.
        with open(self.path, encoding="utf-8") as handle:
            document = json.load(handle)
        del document["key_events"]
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_stateless_replays_do_not_anchor_or_write_legacy_file(self):
        service = DeviceService()
        attach_persistence(service, self.path)
        before_bytes = open(self.path, "rb").read()
        before_ino = os.stat(self.path).st_ino

        service.retry_message(
            self.sid, "m1", {"device_id": "L1", "attempt_id": "a1"})
        service.ack_message(
            self.sid,
            {"device_id": "L1", "message_id": "m1", "sequence": 1})
        service.revoke_device("L3")
        service.revoke_prekey("L2", "k1")
        service.retry_message(
            self.sid, "m1", {"device_id": "L1", "attempt_id": "a1"})

        with open(self.path, "rb") as handle:
            self.assertEqual(handle.read(), before_bytes)
        self.assertEqual(os.stat(self.path).st_ino, before_ino)
        with open(self.path, encoding="utf-8") as handle:
            self.assertNotIn("key_events", json.load(handle))


if __name__ == "__main__":
    unittest.main()
