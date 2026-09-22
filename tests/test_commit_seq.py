"""Tests for the top-level commit generation ``commit_seq``.

Covers:
* the first (empty) durable state is written with ``commit_seq = 0`` and
  every successful persisted transaction — device, group, message, sync,
  delivery and audit-chain changes — advances it exactly once, strictly
  consecutively; idempotent replays do not advance it;
* a failed durable write rolls the generation back together with the rest
  of the in-memory state (the old file bytes/inode stand), and the retry
  reuses the same generation;
* a restart continues the sequence — cursors, messages, delivery,
  revocation and the audit chain never move backwards;
* an old version-1 file without the field loads as generation 0, while a
  present field that is a boolean, negative, float or string makes startup
  refuse with the file untouched (``serve``: stderr single-line JSON,
  field=data_file, exit 1);
* crash-leftover recovery ranks candidates by the highest commit_seq,
  falling back to newest mtime at equal generations, degenerates to the old
  mtime rule when every candidate lacks the field, and skips candidates
  whose generation is malformed; the next commit continues the generation.
"""
import contextlib
import io
import json
import os
import shutil
import tempfile
import unittest
from typing import Any, Dict, Optional

from e2ee_backend import cli as cli_mod
from e2ee_backend.models import Device
from e2ee_backend.persistence import (
    JsonStateStore,
    PersistenceUnavailable,
    StateFileError,
    attach_persistence,
)
from e2ee_backend.service import DeviceService


def _read(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


class CommitSequencingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.path = os.path.join(self.directory, "state.json")
        self.service = DeviceService()
        attach_persistence(self.service, self.path)

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_empty_state_written_at_generation_zero(self) -> None:
        self.assertEqual(self.service.store.commit_seq, 0)
        self.assertEqual(_read(self.path)["commit_seq"], 0)

    def test_each_successful_transaction_advances_once(self) -> None:
        # Device registration.
        self.service.store.add_device(Device("u", "creator", "ik"))
        self.assertEqual(self.service.store.commit_seq, 1)
        self.assertEqual(_read(self.path)["commit_seq"], 1)
        self.service.store.add_device(Device("u", "alice", "ik"))
        self.service.store.add_device(Device("u", "bob", "ik"))
        self.assertEqual(self.service.store.commit_seq, 3)

        # Group + group session (one persisted transaction each).
        self.service.create_group({
            "group_id": "g1", "creator_device_id": "creator",
            "member_device_ids": ["alice", "bob"]})
        self.assertEqual(self.service.store.commit_seq, 4)
        session = self.service.create_group_session({
            "group_id": "g1", "initiator_device_id": "creator",
            "ephemeral_key": "epk"})
        sid = session["session_id"]
        self.assertEqual(self.service.store.commit_seq, 5)

        # Messages (and the audit chain stays in the same transaction).
        for sequence in range(1, 4):
            self.service.post_message({
                "session_id": sid, "sender_device_id": "creator",
                "message_id": f"m{sequence}", "sequence": sequence,
                "nonce": f"n{sequence}", "ciphertext": "ct"})
        self.assertEqual(self.service.store.commit_seq, 8)

        # Sync-cursor advance and delivery ack each persist once.
        self.service.sync_group_messages(sid, "alice", None, 100)
        self.assertEqual(self.service.store.commit_seq, 9)
        self.service.retry_message(sid, "m1", {
            "device_id": "alice", "attempt_id": "a1"})
        self.assertEqual(self.service.store.commit_seq, 10)
        self.service.ack_message(sid, {
            "message_id": "m1", "device_id": "alice", "sequence": 1})
        self.assertEqual(self.service.store.commit_seq, 11)

    def test_idempotent_replay_does_not_advance(self) -> None:
        self.service.store.add_device(Device("u", "creator", "ik"))
        self.service.store.add_device(Device("u", "alice", "ik"))
        self.service.create_group({
            "group_id": "g1", "creator_device_id": "creator",
            "member_device_ids": ["alice"]})
        session = self.service.create_group_session({
            "group_id": "g1", "initiator_device_id": "creator",
            "ephemeral_key": "epk"})
        sid = session["session_id"]
        envelope = {
            "request_id": "r1", "session_id": sid,
            "sender_device_id": "creator", "message_id": "m1",
            "sequence": 1, "nonce": "n1", "ciphertext": "ct"}
        _body, status = self.service.submit_message(envelope)
        self.assertEqual(status, 201)
        before = self.service.store.commit_seq
        # Same request_id/envelope replays as 200 without a new transaction.
        _again, again_status = self.service.submit_message(dict(envelope))
        self.assertEqual(again_status, 200)
        self.assertEqual(self.service.store.commit_seq, before)
        self.assertEqual(_read(self.path)["commit_seq"], before)

    def test_failed_write_rolls_generation_back(self) -> None:
        self.service.store.add_device(Device("u", "d1", "ik"))
        self.assertEqual(self.service.store.commit_seq, 1)
        good_bytes = open(self.path, "rb").read()
        good_ino = os.stat(self.path).st_ino

        real_save = JsonStateStore.save

        def fail_save(store_self, state):  # noqa: ANN001
            raise OSError("simulated disk failure")

        JsonStateStore.save = fail_save  # type: ignore[assignment]
        try:
            with self.assertRaises(PersistenceUnavailable):
                self.service.store.add_device(Device("u", "d2", "ik"))
        finally:
            JsonStateStore.save = real_save  # type: ignore[assignment]

        # Memory, generation, file bytes and inode all stand at the last good
        # transaction; the retry reuses the same, strictly consecutive number.
        self.assertEqual(self.service.store.commit_seq, 1)
        self.assertEqual(open(self.path, "rb").read(), good_bytes)
        self.assertEqual(os.stat(self.path).st_ino, good_ino)
        self.service.store.add_device(Device("u", "d2", "ik"))
        self.assertEqual(self.service.store.commit_seq, 2)
        self.assertEqual(_read(self.path)["commit_seq"], 2)

    def test_restart_continues_the_generation_without_regression(self) -> None:
        self.service.store.add_device(Device("u", "creator", "ik"))
        self.service.store.add_device(Device("u", "alice", "ik"))
        self.service.create_group({
            "group_id": "g1", "creator_device_id": "creator",
            "member_device_ids": ["alice"]})
        session = self.service.create_group_session({
            "group_id": "g1", "initiator_device_id": "creator",
            "ephemeral_key": "epk"})
        sid = session["session_id"]
        self.service.post_message({
            "session_id": sid, "sender_device_id": "creator",
            "message_id": "m1", "sequence": 1, "nonce": "n1",
            "ciphertext": "ct"})
        self.service.sync_group_messages(sid, "alice", None, 100)
        expected = self.service.store.commit_seq
        self.assertEqual(expected, 6)

        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        self.assertEqual(restarted.store.commit_seq, expected)
        # The synced cursor survived the restart.
        self.assertEqual(
            restarted.store._group_sync_cursors[(sid, "alice")].cursor, 1)
        restarted.store.add_device(Device("u", "bob", "ik"))
        self.assertEqual(restarted.store.commit_seq, expected + 1)
        self.assertEqual(_read(self.path)["commit_seq"], expected + 1)


class LegacyAndMalformedCommitSeqTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.path = os.path.join(self.directory, "state.json")
        service = DeviceService()
        attach_persistence(service, self.path)
        service.store.add_device(Device("u", "d1", "ik"))

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def _rewrite(self, value: Any, present: bool = True) -> bytes:
        document = _read(self.path)
        if present:
            document["commit_seq"] = value
        else:
            document.pop("commit_seq", None)
        payload = json.dumps(document, separators=(",", ":")).encode("utf-8")
        with open(self.path, "wb") as handle:
            handle.write(payload)
        return payload

    def test_legacy_missing_field_loads_as_zero(self) -> None:
        self._rewrite(None, present=False)
        service = DeviceService()
        attach_persistence(service, self.path)
        self.assertEqual(service.store.commit_seq, 0)
        service.store.add_device(Device("u", "d2", "ik"))
        self.assertEqual(service.store.commit_seq, 1)
        self.assertEqual(_read(self.path)["commit_seq"], 1)

    def test_zero_is_accepted(self) -> None:
        self._rewrite(0)
        service = DeviceService()
        attach_persistence(service, self.path)
        self.assertEqual(service.store.commit_seq, 0)

    def test_malformed_values_refuse_startup_and_leave_file(self) -> None:
        for value in (True, False, -1, -100, 1.5, 2.0, "3", "0", None):
            # None with present=True means an explicit JSON null.
            payload = self._rewrite(value)
            with self.subTest(value=value):
                with self.assertRaises(StateFileError):
                    attach_persistence(DeviceService(), self.path)
                # The startup refusal never touches the formal file.
                self.assertEqual(open(self.path, "rb").read(), payload)

    def test_serve_refuses_with_single_stderr_json_line_exit_1(self) -> None:
        self._rewrite(-1)
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            status = cli_mod.main(
                ["serve", "--data-file", self.path, "--port", "0"])
        self.assertEqual(status, 1)
        line = stderr.getvalue().strip()
        self.assertEqual(len(line.splitlines()), 1)
        self.assertEqual(json.loads(line)["field"], "data_file")
        self.assertEqual(_read(self.path)["commit_seq"], -1)


def _build_candidate(directory: str, name: str, device: str,
                     mtime_ns: Optional[int],
                     commit_seq: Optional[int]) -> str:
    """Build one section-complete verifiable snapshot as a leftover.

    Built through a real attach_persistence in an isolated subdirectory so
    the build's own tmp/bak files never mix into *directory*'s scan; the
    finished document is copied in under *name*. ``commit_seq=None`` removes
    the field (a pre-commit_seq snapshot).
    """
    build_dir = tempfile.mkdtemp(dir=directory)
    try:
        build_path = os.path.join(build_dir, "build.json")
        service = DeviceService()
        attach_persistence(service, build_path)
        service.store.add_device(Device("u", device, "ik"))
        document = _read(build_path)
        if commit_seq is None:
            document.pop("commit_seq", None)
        else:
            document["commit_seq"] = commit_seq
        target = os.path.join(directory, name)
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(document, handle, separators=(",", ":"))
    finally:
        shutil.rmtree(build_dir, ignore_errors=True)
    if mtime_ns is not None:
        os.utime(target, ns=(mtime_ns, mtime_ns))
    return target


class CrashLeftoverCommitSeqTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.path = os.path.join(self.directory, "state.json")

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def _devices(self, service: DeviceService):
        return {device.device_id
                for device in service.store._devices.values()}

    def test_higher_generation_beats_newer_mtime(self) -> None:
        older = _build_candidate(
            self.directory, ".state-high.tmp", "highdev",
            1_000_000_000, 5)
        newer = _build_candidate(
            self.directory, ".state-low.tmp", "lowdev",
            9_000_000_000, 2)
        service = DeviceService()
        attach_persistence(service, self.path)
        self.assertEqual(service.store.commit_seq, 5)
        self.assertIn("highdev", self._devices(service))
        self.assertNotIn("lowdev", self._devices(service))
        self.assertFalse(os.path.exists(older))
        self.assertFalse(os.path.exists(newer))
        # The next commit continues on the recovered generation.
        service.store.add_device(Device("u", "next", "ik"))
        self.assertEqual(_read(self.path)["commit_seq"], 6)

    def test_equal_generation_prefers_newest_mtime(self) -> None:
        _build_candidate(self.directory, ".state-old.tmp", "olddev",
                         1_000_000_000, 3)
        _build_candidate(self.directory, ".state-new.tmp", "newdev",
                         2_000_000_000, 3)
        service = DeviceService()
        attach_persistence(service, self.path)
        self.assertEqual(service.store.commit_seq, 3)
        self.assertIn("newdev", self._devices(service))
        self.assertNotIn("olddev", self._devices(service))

    def test_all_candidates_missing_field_keep_mtime_rule(self) -> None:
        _build_candidate(self.directory, ".state-old.tmp", "olddev",
                         1_000_000_000, None)
        _build_candidate(self.directory, ".state-new.tmp", "newdev",
                         2_000_000_000, None)
        service = DeviceService()
        attach_persistence(service, self.path)
        self.assertIn("newdev", self._devices(service))
        self.assertEqual(service.store.commit_seq, 0)
        service.store.add_device(Device("u", "next", "ik"))
        self.assertEqual(_read(self.path)["commit_seq"], 1)

    def test_candidate_with_field_present_beats_fieldless(self) -> None:
        # A generation-0 snapshot that carries the field explicitly is at the
        # same numeric generation as a fieldless one, but the two share mtime
        # tie-breaking; the important guarantee is that a generation-1
        # snapshot wins over any number of fieldless/newer-mtime ones.
        _build_candidate(self.directory, ".state-legacy.tmp", "legacydev",
                         9_000_000_000, None)
        _build_candidate(self.directory, ".state-gen1.tmp", "gen1dev",
                         1_000_000_000, 1)
        service = DeviceService()
        attach_persistence(service, self.path)
        self.assertEqual(service.store.commit_seq, 1)
        self.assertIn("gen1dev", self._devices(service))
        self.assertNotIn("legacydev", self._devices(service))

    def test_malformed_generation_candidate_is_skipped(self) -> None:
        # Negative generation: parses as version=1 and carries the sections,
        # but the full semantic restore (restore_state) rejects it; the
        # candidate is skipped and the next verifiable one is recovered.
        _build_candidate(self.directory, ".state-bad.tmp", "baddev",
                         9_000_000_000, -7)
        _build_candidate(self.directory, ".state-good.tmp", "gooddev",
                         1_000_000_000, 4)
        service = DeviceService()
        attach_persistence(service, self.path)
        self.assertEqual(service.store.commit_seq, 4)
        self.assertIn("gooddev", self._devices(service))
        self.assertNotIn("baddev", self._devices(service))

    def test_valid_formal_file_wins_and_clears_leftovers(self) -> None:
        service = DeviceService()
        attach_persistence(service, self.path)
        service.store.add_device(Device("u", "formaldev", "ik"))
        formal_seq = service.store.commit_seq
        formal_bytes = open(self.path, "rb").read()
        # A higher-generation leftover must not displace a valid formal file.
        _build_candidate(self.directory, ".state-stray.tmp", "straydev",
                         None, formal_seq + 100)
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        self.assertEqual(restarted.store.commit_seq, formal_seq)
        self.assertIn("formaldev", self._devices(restarted))
        self.assertEqual(open(self.path, "rb").read(), formal_bytes)
        leftovers = [name for name in os.listdir(self.directory)
                     if name.endswith((".tmp", ".bak"))]
        self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()
