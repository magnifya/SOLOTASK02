"""Top-level commit generation (``commit_seq``).

Every durable document carries a strictly-consecutive ``commit_seq``: the
first empty state is created at 0 and each successful persistence transaction
advances it by exactly one. A failed durable transaction consumes no
generation and rewrites the same one after repair; a restart resumes exactly
where the file left off. A legacy version-1 file without the field is treated
as generation 0; a present-but-invalid field (bool, negative number, float,
string) makes startup refuse with field=data_file without touching the file.

When the formal file is missing, crash-leftover snapshots rank by commit
generation first (highest ``commit_seq`` wins even with an older mtime); when
two or more verifiable candidates tie at that highest generation the choice is
ambiguous and startup refuses (nothing is promoted or removed) rather than
ever breaking the tie by name or mtime. Only when every candidate lacks
the field does the legacy newest-mtime rule stand.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from typing import Any, Dict, List, Tuple

from e2ee_backend.models import Device
from e2ee_backend.persistence import (
    PersistenceUnavailable,
    StateFileError,
    attach_persistence,
)
from e2ee_backend.service import DeviceService

from tests.test_crash_recovery import build_fixture, tmp_names, write_tmp


def _read_doc(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _write_doc(path: str, document: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(document, handle)


class CommitSeqLifecycleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.path = os.path.join(self.directory, "state.json")

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def _service(self) -> DeviceService:
        service = DeviceService()
        self.state_store = attach_persistence(service, self.path)
        return service

    def test_empty_state_is_created_at_generation_zero(self) -> None:
        service = self._service()
        self.assertEqual(_read_doc(self.path)["commit_seq"], 0)
        # No further writes happen while the store stays untouched.
        self.assertEqual(_read_doc(self.path)["commit_seq"], 0)

    def test_each_successful_transaction_advances_exactly_once(self) -> None:
        service = self._service()
        # One device registration = one durable transaction.
        service.store.add_device(Device("u", "d1", "ik"))
        self.assertEqual(_read_doc(self.path)["commit_seq"], 1)
        service.store.add_device(Device("u", "d2", "ik"))
        self.assertEqual(_read_doc(self.path)["commit_seq"], 2)
        # A group, a session and three messages each add one generation; the
        # audit-chain mutations are part of the same transaction, never extra.
        service.create_group({
            "group_id": "g1", "creator_device_id": "d1",
            "member_device_ids": ["d2"]})
        self.assertEqual(_read_doc(self.path)["commit_seq"], 3)
        session = service.create_group_session({
            "group_id": "g1", "initiator_device_id": "d1",
            "ephemeral_key": "epk"})
        sid = session["session_id"]
        self.assertEqual(_read_doc(self.path)["commit_seq"], 4)
        for seq in range(1, 4):
            service.post_message({
                "session_id": sid, "sender_device_id": "d1",
                "message_id": f"m{seq}", "sequence": seq,
                "nonce": f"n{seq}", "ciphertext": "ct"})
        self.assertEqual(_read_doc(self.path)["commit_seq"], 7)
        # A sync checkpoint (cursor change) and a delivery ack each commit.
        service.sync_group_checkpoint(
            sid, {"device_id": "d2", "cursor": 2})
        self.assertEqual(_read_doc(self.path)["commit_seq"], 8)
        service.ack_message(sid, {
            "device_id": "d2", "message_id": "m2",
            "sequence": 2})
        self.assertEqual(_read_doc(self.path)["commit_seq"], 9)
        # Key-audit change (identity rotation) is one generation too.
        service.store.rotate_identity_key("d1", "ik2")
        self.assertEqual(_read_doc(self.path)["commit_seq"], 10)

    def test_idempotent_replays_do_not_advance_generation(self) -> None:
        service = self._service()
        service.store.add_device(Device("u", "d1", "ik"))
        before = _read_doc(self.path)["commit_seq"]
        # Rotating to the same identity key is a no-op: changed=False and the
        # store performs no persistence transaction, so no generation is used.
        service.store.rotate_identity_key("d1", "ik")
        service.store.rotate_identity_key("d1", "ik")
        self.assertEqual(_read_doc(self.path)["commit_seq"], before)
        # A real rotation still commits exactly one generation.
        service.store.rotate_identity_key("d1", "ik2")
        self.assertEqual(_read_doc(self.path)["commit_seq"], before + 1)

    def test_restart_resumes_without_repeating_or_skipping(self) -> None:
        service = self._service()
        for index in range(5):
            service.store.add_device(Device("u", f"d{index}", "ik"))
        self.assertEqual(_read_doc(self.path)["commit_seq"], 5)
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        restarted.store.add_device(Device("u", "d5", "ik"))
        self.assertEqual(_read_doc(self.path)["commit_seq"], 6)
        again = DeviceService()
        attach_persistence(again, self.path)
        again.store.add_device(Device("u", "d6", "ik"))
        self.assertEqual(_read_doc(self.path)["commit_seq"], 7)

    def test_failed_transaction_consumes_no_generation(self) -> None:
        service = self._service()
        state_store = self.state_store
        service.store.add_device(Device("u", "d1", "ik"))
        self.assertEqual(_read_doc(self.path)["commit_seq"], 1)
        good_bytes = open(self.path, "rb").read()
        good_ino = os.stat(self.path).st_ino

        def fail_save(state: Dict[str, Any]) -> None:
            raise OSError("simulated disk failure")

        state_store.save = fail_save  # type: ignore[assignment]
        with self.assertRaises(PersistenceUnavailable):
            service.store.add_device(Device("u", "d2", "ik"))
        # Memory rolled back, the file keeps generation 1, its bytes and inode.
        self.assertIsNone(service.store.find_by_device_id("d2"))
        self.assertEqual(open(self.path, "rb").read(), good_bytes)
        self.assertEqual(os.stat(self.path).st_ino, good_ino)
        del state_store.save
        service.store.add_device(Device("u", "d2", "ik"))
        # The repaired transaction commits the generation the failed one
        # tried to consume: no gap, no duplicate.
        self.assertEqual(_read_doc(self.path)["commit_seq"], 2)


class LegacyCommitSeqCompatibilityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        _service, self.path, _sid, formal_bytes = build_fixture(
            self.directory, cursor=2)
        self.document = json.loads(formal_bytes.decode("utf-8"))
        del self.document["commit_seq"]
        # The recreated legacy file lives at its own path with no sidecar;
        # drop the integrity marker so it is a genuine pre-feature document.
        self.document.pop("integrity_log_version", None)

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_legacy_file_loads_at_generation_zero(self) -> None:
        legacy_path = os.path.join(self.directory, "legacy.json")
        _write_doc(legacy_path, self.document)
        service = DeviceService()
        state_store = attach_persistence(service, legacy_path)
        self.assertEqual(state_store.commit_seq, 1)
        service.store.add_device(Device("u", "extra", "ik"))
        self.assertEqual(_read_doc(legacy_path)["commit_seq"], 1)
        # Cursors and the audit chain survived the legacy load unchanged.
        self.assertTrue(_read_doc(legacy_path)["group_sync_cursors"])
        self.assertTrue(_read_doc(legacy_path)["key_events"])


class InvalidCommitSeqStartupTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        _service, self.path, _sid, formal_bytes = build_fixture(self.directory)
        self.document = json.loads(formal_bytes.decode("utf-8"))
        # Hand-built bad-*.json/good-*.json files at their own paths carry no
        # sidecar; drop the marker so they are genuine pre-feature documents.
        self.document.pop("integrity_log_version", None)

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def _assert_rejected_untouched(self, value: Any) -> None:
        path = os.path.join(self.directory, f"bad-{type(value).__name__}.json")
        document = dict(self.document)
        document["commit_seq"] = value
        _write_doc(path, document)
        before = open(path, "rb").read()
        before_ino = os.stat(path).st_ino
        with self.assertRaises(StateFileError):
            attach_persistence(DeviceService(), path)
        self.assertEqual(open(path, "rb").read(), before)
        self.assertEqual(os.stat(path).st_ino, before_ino)

    def test_bool_negative_float_and_string_are_rejected(self) -> None:
        for bad in (True, False, -1, -10 ** 9, 1.5, "3", "0", None, [], {}):
            self._assert_rejected_untouched(bad)

    def test_zero_and_positive_integers_still_load(self) -> None:
        for good in (0, 1, 10 ** 12):
            path = os.path.join(self.directory, f"good-{good}.json")
            document = dict(self.document)
            document["commit_seq"] = good
            _write_doc(path, document)
            service = DeviceService()
            state_store = attach_persistence(service, path)
            self.assertEqual(state_store.commit_seq, good + 1)

    def test_serve_refuses_with_stderr_json_field_data_file_exit_1(self) -> None:
        path = os.path.join(self.directory, "bad-serve.json")
        document = dict(self.document)
        document["commit_seq"] = -1
        _write_doc(path, document)
        result = subprocess.run(
            [sys.executable, "-m", "e2ee_backend", "serve",
             "--host", "127.0.0.1", "--port", "0",
             "--data-file", path],
            capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 1)
        lines = result.stderr.strip().splitlines()
        self.assertEqual(len(lines), 1)
        body = json.loads(lines[0])
        self.assertEqual(body["field"], "data_file")
        # The rejected file is untouched.
        self.assertEqual(
            json.loads(open(path, encoding="utf-8").read())["commit_seq"], -1)


class MissingFormalGenerationRankingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        _service, self.path, self.sid, formal_bytes = build_fixture(
            self.directory, cursor=2)
        self.formal = json.loads(formal_bytes.decode("utf-8"))
        # These tests carve pre-integrity-log crash leftovers out of the
        # fixture: strip the marker and remove the modern sidecar so a
        # recovered leftover loads as a genuine legacy document (the ranking
        # logic under test predates the integrity history).
        self.formal.pop("integrity_log_version", None)
        sidecar = self.path + ".integrity"
        if os.path.exists(sidecar):
            os.remove(sidecar)

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def _leftover(self, name: str, document: Dict[str, Any],
                  mtime_ns: int) -> str:
        return write_tmp(self.directory, name,
                         json.dumps(document).encode("utf-8"),
                         mtime_ns=mtime_ns)

    def _remove_formal(self) -> None:
        os.unlink(self.path)

    def _recover(self) -> Tuple[DeviceService, Any]:
        service = DeviceService()
        state_store = attach_persistence(service, self.path)
        return service, state_store

    def _doc_at(self, seq: int) -> Dict[str, Any]:
        document = json.loads(json.dumps(self.formal))
        document["commit_seq"] = seq
        return document

    def test_higher_generation_wins_over_newer_mtime(self) -> None:
        self._remove_formal()
        # Stale low-generation snapshot with a newer mtime must not win.
        self._leftover(".state-low-new.tmp", self._doc_at(2), 9000)
        self._leftover(".state-high-old.tmp", self._doc_at(10), 1000)
        service, state_store = self._recover()
        self.assertEqual(_read_doc(self.path)["commit_seq"], 10)
        # The recovered generation continues: next commit is 11.
        self.assertEqual(state_store.commit_seq, 11)
        service.store.add_device(Device("u", "next", "ik"))
        self.assertEqual(_read_doc(self.path)["commit_seq"], 11)
        self.assertEqual(tmp_names(self.directory), [])
        # The higher-generation cursor survives.
        self.assertEqual(
            service.store._group_sync_cursors[(self.sid, "alice")].cursor, 2)

    def test_equal_generation_tie_is_refused_not_mtime_picked(self) -> None:
        from e2ee_backend.persistence import StateFileError
        self._remove_formal()
        # Two verifiable candidates at the same generation; one has a strictly
        # newer mtime and the other the earlier-sorting name. Neither may win.
        older_path = self._leftover(
            ".state-000-eq-old.tmp", self._doc_at(7), 1000)
        newer_path = self._leftover(
            ".state-eq-new.tmp", self._doc_at(7), 9000)
        older_bytes = open(older_path, "rb").read()
        newer_bytes = open(newer_path, "rb").read()
        # Recovery aborts: startup refuses, the formal path stays missing and both
        # candidates are preserved byte-for-byte for inspection/resolution.
        with self.assertRaises(StateFileError):
            self._recover()
        self.assertFalse(os.path.exists(self.path))
        self.assertEqual(open(older_path, "rb").read(), older_bytes)
        self.assertEqual(open(newer_path, "rb").read(), newer_bytes)
        self.assertEqual(
            tmp_names(self.directory),
            [".state-000-eq-old.tmp", ".state-eq-new.tmp"])
        # Resolving the ambiguity (removing one twin) lets the unique survivor
        # promote normally; the retained generation then continues consecutively.
        os.unlink(newer_path)
        service, state_store = self._recover()
        self.assertEqual(_read_doc(self.path)["commit_seq"], 7)
        self.assertEqual(state_store.commit_seq, 8)
        self.assertEqual(tmp_names(self.directory), [])
        service.store.add_device(Device("u", "next", "ik"))
        self.assertEqual(_read_doc(self.path)["commit_seq"], 8)

    def test_equal_generation_tie_behind_unique_higher_one_still_recovers(
            self) -> None:
        # A tied pair at a lower generation must not block promotion of a unique,
        # strictly-higher verifiable candidate.
        self._remove_formal()
        self._leftover(".state-tie-a.tmp", self._doc_at(2), 1000)
        self._leftover(".state-tie-b.tmp", self._doc_at(2), 9000)
        winner = self._leftover(
            ".state-win.tmp", self._doc_at(3), 5000)
        winner_ino = os.stat(winner).st_ino
        self._recover()
        self.assertEqual(os.stat(self.path).st_ino, winner_ino)
        self.assertEqual(_read_doc(self.path)["commit_seq"], 3)
        self.assertEqual(tmp_names(self.directory), [])

    def test_fieldless_candidates_keep_legacy_mtime_rule(self) -> None:
        self._remove_formal()
        old = json.loads(json.dumps(self.formal))
        del old["commit_seq"]
        newer = json.loads(json.dumps(old))
        self._leftover(".state-seqless-old.tmp", old, 1000)
        self._leftover(".state-seqless-new.tmp", newer, 9000)
        self._recover()
        self.assertNotIn("commit_seq", _read_doc(self.path))
        self.assertEqual(tmp_names(self.directory), [])

    def test_generation_candidate_beats_all_fieldless_ones(self) -> None:
        self._remove_formal()
        fieldless = json.loads(json.dumps(self.formal))
        del fieldless["commit_seq"]
        self._leftover(".state-seqless-new.tmp", fieldless, 9000)
        self._leftover(".state-seq-old.tmp", self._doc_at(1), 1000)
        _service, _store = self._recover()
        self.assertEqual(_read_doc(self.path)["commit_seq"], 1)

    def test_malformed_generation_leftover_is_skipped(self) -> None:
        self._remove_formal()
        bad = self._doc_at(99)
        bad["commit_seq"] = -5
        self._leftover(".state-bad-new.tmp", bad, 9000)
        self._leftover(".state-good-old.tmp", self._doc_at(3), 1000)
        _service, _store = self._recover()
        self.assertEqual(_read_doc(self.path)["commit_seq"], 3)
        self.assertEqual(tmp_names(self.directory), [])

    def test_section_incomplete_candidate_skipped_regardless_of_generation(
            self) -> None:
        self._remove_formal()
        incomplete = self._doc_at(100)
        del incomplete["key_events"]
        self._leftover(".state-incomplete.tmp", incomplete, 9000)
        self._leftover(".state-complete.tmp", self._doc_at(1), 1000)
        _service, _store = self._recover()
        self.assertEqual(_read_doc(self.path)["commit_seq"], 1)

    def test_no_valid_candidate_includes_generation_zero_empty_state(
            self) -> None:
        self._remove_formal()
        bad = self._doc_at(50)
        bad["messages"] = {"ghost-session": []}
        self._leftover(".state-bad.tmp", bad, 9000)
        _service, _store = self._recover()
        self.assertTrue(os.path.exists(self.path))
        self.assertEqual(_read_doc(self.path)["commit_seq"], 0)
        self.assertEqual(tmp_names(self.directory), [])


class ValidFormalGenerationCleanupTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        _service, self.path, _sid, formal_bytes = build_fixture(self.directory)
        self.formal = json.loads(formal_bytes.decode("utf-8"))
        self.formal_ino = os.stat(self.path).st_ino

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_valid_formal_wins_and_cleans_higher_generation_leftovers(
            self) -> None:
        leftover_doc = dict(self.formal)
        leftover_doc["commit_seq"] = self.formal["commit_seq"] + 1000
        write_tmp(self.directory, ".state-future.tmp",
                  json.dumps(leftover_doc).encode("utf-8"))
        service = DeviceService()
        state_store = attach_persistence(service, self.path)
        # The formal file's bytes and inode are untouched.
        with open(self.path, "rb") as handle:
            formal_bytes = handle.read()
        self.assertEqual(json.loads(formal_bytes)["commit_seq"],
                         self.formal["commit_seq"])
        self.assertEqual(os.stat(self.path).st_ino, self.formal_ino)
        self.assertEqual(tmp_names(self.directory), [])
        self.assertEqual(state_store.commit_seq,
                         self.formal["commit_seq"] + 1)

    def test_corrupt_formal_refuses_and_keeps_leftovers(self) -> None:
        document = dict(self.formal)
        document["commit_seq"] = "nope"
        _write_doc(self.path, document)
        before = open(self.path, "rb").read()
        leftover = write_tmp(
            self.directory, ".state-orphan.tmp",
            json.dumps(self.formal).encode("utf-8"))
        with self.assertRaises(StateFileError):
            attach_persistence(DeviceService(), self.path)
        self.assertEqual(open(self.path, "rb").read(), before)
        self.assertTrue(os.path.exists(leftover))


if __name__ == "__main__":
    unittest.main()
