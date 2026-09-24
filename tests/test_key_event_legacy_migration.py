"""Lazy migration of old (section-less) version-1 files onto the key-audit chain.

An older version-1 state file predates the optional ``key_events`` section.
It must still load, and the first change that is persisted (a message, group,
sync, delivery, revocation, new-device registration, ...) first rebuilds, in
the same storage-lock transaction, an anchor chain for every registered
device that has none — in registration order:

* ``registered`` with the *current* identity key and the ordered pre-keys,
  ``prev_hash`` empty and ``created_at`` fixed to ``registered_at``;
* then one ``prekey_revoked`` per already-revoked pre-key, in pre-key order;
* then ``device_revoked`` for a revoked device;
* then the real event of the triggering change.

The migration and the business change are persisted together in one atomic
version-1 write. Any write/fsync/replace failure answers
503/field=data_file and rolls memory, file bytes *and* inode, business state
and audit chain all back — leaving the legacy file untouched. With no change
the section stays absent.
"""
import json
import os
import tempfile
import unittest

from e2ee_backend.models import Device, SignedPreKey
from e2ee_backend.persistence import (
    JsonStateStore,
    PersistenceUnavailable,
    attach_persistence,
)
from e2ee_backend.service import DeviceService
from e2ee_backend.storage import key_event_hash

from tests.test_key_events import _raw_key_b64, _register_payload


def _device_record(device_id: str, *, user_id: str = "u1",
                   identity_key: str | None = None,
                   registered_at: str = "2026-03-01T00:00:00+00:00",
                   revoked: bool = False,
                   prekeys: list | None = None) -> dict:
    return {
        "user_id": user_id,
        "device_id": device_id,
        "identity_key": identity_key or _raw_key_b64(),
        "registered_at": registered_at,
        "rotated_at": registered_at,
        "revoked": revoked,
        "prekeys": (prekeys if prekeys is not None else [
            {"key_id": "k1", "public_key": _raw_key_b64(),
             "revoked": False, "consumed": False}]),
    }


def _legacy_document(devices: list) -> dict:
    return {
        "version": 1,
        "devices": devices,
        "sessions": [],
        "prekey_claims": [],
        "prekey_batch_claims": [],
        "claim_session_bindings": [],
        "batch_claim_session_bindings": [],
        "groups": [],
        "group_sessions": [],
        "group_session_rotations": [],
        "messages": {},
        "delivery": [],
        "group_delivery": [],
        "used_nonces": {},
        "group_sync_cursors": [],
        "message_sync_cursors": [],
        "message_submissions": [],
    }


class LegacyAnchorMigrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.path = os.path.join(self.directory, "state.json")

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self.directory, ignore_errors=True)

    def _write_legacy(self, devices: list) -> None:
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(_legacy_document(devices), handle)

    def _restart(self) -> DeviceService:
        service = DeviceService()
        attach_persistence(service, self.path)
        return service

    def _chains(self, service: DeviceService) -> dict:
        chains: dict = {}
        for event in service.store.snapshot_state()["key_events"]:
            chains.setdefault(event["device_id"], []).append(event)
        return chains

    def _assert_chain_links(self, chain: list) -> None:
        prev = ""
        for position, event in enumerate(chain):
            self.assertEqual(event["seq"], position + 1)
            self.assertEqual(event["prev_hash"], prev)
            self.assertEqual(
                event["hash"],
                key_event_hash(event["device_id"], event["seq"],
                               event["type"], event["payload"],
                               event["prev_hash"], event["created_at"]))
            prev = event["hash"]

    def test_section_stays_absent_without_a_change(self) -> None:
        self._write_legacy([_device_record("d1")])
        service = self._restart()
        # Read-only access (incl. the key-events query) must not migrate.
        page = service.store.key_events_page("d1", 0, 100)
        assert page is not None
        self.assertEqual(page[0], [])
        self.assertNotIn("key_events", service.store.snapshot_state())
        with open(self.path, "r", encoding="utf-8") as handle:
            self.assertNotIn("key_events", json.load(handle))

    def test_anchor_shapes_in_registration_order(self) -> None:
        # d1: one revoked pre-key; d2: fully revoked (every key revoked).
        d1 = _device_record(
            "d1", registered_at="2026-03-01T00:00:00+00:00",
            prekeys=[{"key_id": "k1", "public_key": _raw_key_b64(),
                      "revoked": True, "consumed": False},
                     {"key_id": "k2", "public_key": _raw_key_b64(),
                      "revoked": False, "consumed": False}])
        d2 = _device_record(
            "d2", registered_at="2026-03-02T00:00:00+00:00", revoked=True,
            prekeys=[{"key_id": "a1", "public_key": _raw_key_b64(),
                      "revoked": True, "consumed": False}])
        self._write_legacy([d1, d2])
        service = self._restart()

        # Trigger with a new registration: every legacy device is anchored
        # before the new device's own registered event.
        self.assertTrue(service.store.add_device(Device(
            user_id="u3", device_id="d3", identity_key=_raw_key_b64(),
            prekeys=[])))

        chains = self._chains(service)
        self.assertEqual(
            [e["type"] for e in chains["d1"]],
            ["registered", "prekey_revoked"])
        self.assertEqual(
            [e["type"] for e in chains["d2"]],
            ["registered", "prekey_revoked", "device_revoked"])
        self.assertEqual([e["type"] for e in chains["d3"]], ["registered"])

        d1_registered = chains["d1"][0]
        self.assertEqual(d1_registered["seq"], 1)
        self.assertEqual(d1_registered["prev_hash"], "")
        self.assertEqual(d1_registered["created_at"],
                         "2026-03-01T00:00:00+00:00")
        self.assertEqual(d1_registered["payload"]["identity_key"],
                         d1["identity_key"])
        self.assertEqual(d1_registered["payload"]["signed_prekeys"],
                         [{"key_id": "k1", "public_key": d1["prekeys"][0]["public_key"]},
                          {"key_id": "k2", "public_key": d1["prekeys"][1]["public_key"]}])
        self.assertEqual(chains["d1"][1]["payload"],
                         {"key_id": "k1", "public_key": d1["prekeys"][0]["public_key"]})
        self.assertEqual(chains["d2"][0]["created_at"],
                         "2026-03-02T00:00:00+00:00")
        self.assertEqual(chains["d2"][2]["payload"], {})
        for chain in chains.values():
            for position, event in enumerate(chain):
                self.assertEqual(event["seq"], position + 1)
            self._assert_chain_links(chain)

        # The anchors were appended in registration order: d1 and d2 chains
        # both close before d3 opens (d3 is the new registration).
        flattened = service.store.snapshot_state()["key_events"]
        self.assertEqual([e["device_id"] for e in flattened],
                         ["d1", "d1", "d2", "d2", "d2", "d3"])

    def test_migrated_file_restarts_and_validates(self) -> None:
        d1 = _device_record(
            "d1",
            prekeys=[{"key_id": "k1", "public_key": _raw_key_b64(),
                      "revoked": True, "consumed": False}])
        self._write_legacy([d1])
        service = self._restart()
        # A non-key business change is enough; use add_device here as a
        # generic persisting mutation.
        service.store.add_device(Device(
            user_id="u2", device_id="d2", identity_key=_raw_key_b64(),
            prekeys=[]))
        with open(self.path, "r", encoding="utf-8") as handle:
            on_disk = json.load(handle)
        self.assertIn("key_events", on_disk)

        restarted = self._restart()  # must not raise StateFileError
        page = restarted.store.key_events_page("d1", 0, 100)
        assert page is not None
        self.assertEqual([e["type"] for e in page[0]],
                         ["registered", "prekey_revoked"])

    def test_key_change_anchors_then_appends_real_event(self) -> None:
        identity = _raw_key_b64()
        prekey = _raw_key_b64()
        self._write_legacy([_device_record(
            "d1", identity_key=identity,
            prekeys=[{"key_id": "k1", "public_key": prekey,
                      "revoked": False, "consumed": False}])])
        service = self._restart()
        new_identity = _raw_key_b64()
        service.rotate_identity_key("d1", {"identity_key": new_identity})
        chain = self._chains(service)["d1"]
        self.assertEqual([e["type"] for e in chain],
                         ["registered", "identity_rotated"])
        self.assertEqual(chain[0]["payload"]["identity_key"], identity)
        self.assertEqual(chain[1]["payload"],
                         {"old_identity_key": identity,
                          "new_identity_key": new_identity})
        # A subsequent pre-key revoke continues the same chain (no second
        # registered anchor, no duplicated revocation).
        service.revoke_prekey("d1", "k1")
        chain = self._chains(service)["d1"]
        self.assertEqual(
            [e["type"] for e in chain],
            ["registered", "identity_rotated", "prekey_revoked"])

    def test_failed_write_rolls_back_and_keeps_legacy_file(self) -> None:
        self._write_legacy([_device_record("d1")])
        service = self._restart()
        with open(self.path, "rb") as handle:
            original_bytes = handle.read()
        original_inode = os.stat(self.path).st_ino

        original_save = JsonStateStore.save
        JsonStateStore.save = lambda self, state: (_ for _ in ()).throw(
            OSError("simulated disk full"))
        try:
            with self.assertRaises(PersistenceUnavailable):
                service.store.add_device(Device(
                    user_id="u2", device_id="d2",
                    identity_key=_raw_key_b64(), prekeys=[]))
        finally:
            JsonStateStore.save = original_save

        # File bytes and inode preserved, section still absent.
        with open(self.path, "rb") as handle:
            self.assertEqual(handle.read(), original_bytes)
        self.assertEqual(os.stat(self.path).st_ino, original_inode)
        with open(self.path, "r", encoding="utf-8") as handle:
            self.assertNotIn("key_events", json.load(handle))
        # Memory rolled back: no d2, legacy still pending.
        self.assertIsNone(service.store.find_by_device_id("d2"))
        self.assertNotIn("key_events", service.store.snapshot_state())

        # A later successful change migrates and commits atomically.
        self.assertTrue(service.store.add_device(Device(
            user_id="u2", device_id="d2", identity_key=_raw_key_b64(),
            prekeys=[])))
        with open(self.path, "r", encoding="utf-8") as handle:
            on_disk = json.load(handle)
        self.assertIn("key_events", on_disk)
        restarted = self._restart()
        chains = self._chains(restarted)
        self.assertIn("d1", chains)
        self.assertIn("d2", chains)

    def test_revoked_device_without_key_flags_is_normalized(self) -> None:
        # The lenient legacy loader accepts device.revoked=true while a
        # pre-key flag is still false; anchoring normalizes flags inside the
        # transaction so the chain replays exactly to the stored section.
        record = _device_record(
            "d1", revoked=True,
            prekeys=[{"key_id": "k1", "public_key": _raw_key_b64(),
                      "revoked": False, "consumed": False}])
        self._write_legacy([record])
        service = self._restart()
        service.store.add_device(Device(
            user_id="u2", device_id="d2", identity_key=_raw_key_b64(),
            prekeys=[]))
        chain = self._chains(service)["d1"]
        self.assertEqual(
            [e["type"] for e in chain],
            ["registered", "prekey_revoked", "device_revoked"])
        # Restart must accept the normalized, migrated document.
        restarted = self._restart()
        device = restarted.store.find_by_device_id("d1")
        self.assertTrue(device.revoked)
        self.assertTrue(all(pk.revoked for pk in device.prekeys))

    def test_in_memory_mode_never_carries_pending_migration(self) -> None:
        # No data file: a fresh store is modern, the section is always present.
        service = DeviceService()
        service.register(_register_payload("d1"))
        self.assertIn("key_events", service.store.snapshot_state())

    def test_message_write_on_legacy_file_migrates_in_one_commit(self) -> None:
        # Build a fully valid fixture with a modern service (two devices and a
        # session), then strip key_events to emulate an old version-1 file.
        builder = DeviceService()
        attach_persistence(builder, self.path)
        identity = _raw_key_b64()
        builder.register(_register_payload(
            "d1", identity_key=identity,
            prekeys=[{"key_id": "k1", "public_key": _raw_key_b64()}]))
        builder.register(_register_payload("d2"))
        session = builder.create_session({
            "initiator_device_id": "d1", "recipient_device_id": "d2",
            "prekey_id": "k1", "ephemeral_key": _raw_key_b64()})
        document = self._read_document()
        self.assertIn("key_events", document)
        del document["key_events"]
        # Emulate a genuine pre-integrity-log old file: drop the marker and
        # its sidecar as well.
        document.pop("integrity_log_version", None)
        self._write_document(document)
        sidecar = self.path + ".integrity"
        if os.path.exists(sidecar):
            os.remove(sidecar)

        service = self._restart()
        # A message (a persisting non-key business change) anchors d1/d2 in
        # the same locked transaction and then appends the message.
        response = service.post_message({
            "session_id": session["session_id"], "sender_device_id": "d1",
            "message_id": "m1", "sequence": 1, "nonce": "nonce-1",
            "ciphertext": "ciphertext"})
        self.assertEqual(response["message_id"], "m1")

        on_disk = self._read_document()
        chains: dict = {}
        for event in on_disk["key_events"]:
            chains.setdefault(event["device_id"], []).append(event)
        self.assertEqual([e["type"] for e in chains["d1"]], ["registered"])
        self.assertEqual([e["type"] for e in chains["d2"]], ["registered"])
        # Only the registered anchors — the message appends no key event.
        self.assertEqual(
            sorted((e["device_id"], e["type"]) for e in on_disk["key_events"]),
            [("d1", "registered"), ("d2", "registered")])
        # The business change committed together with the migration.
        self.assertEqual(on_disk["messages"][session["session_id"]][0]
                         ["message_id"], "m1")
        self._restart()  # migrated document restarts cleanly

    def _read_document(self) -> dict:
        with open(self.path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _write_document(self, document: dict) -> None:
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)


if __name__ == "__main__":
    unittest.main()
