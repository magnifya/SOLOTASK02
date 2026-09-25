"""Two-file (state + ``<state>.integrity``) crash recovery.

Every durable commit is one transaction over two formal files: the version=1
state document and its append-only integrity sidecar. Temporaries are
``.state-*.tmp`` / ``.integrity-*.tmp`` and pre-replace inode backups are
``.state-*.bak`` / ``.integrity-*.bak``. A crash or a failed directory fsync
can therefore leave matched pairs beside the targets, and recovery must
treat them only as *pairs*:

* a commit lands both files at the same generation with a hash chain whose
  tail binds that state's generation and canonical ``state_hash``;
* with the formal state missing, a state leftover is recovered only together
  with the sidecar whose chain tail binds it (a marker-less legacy state
  with no sidecar at all is the legacy pair); staged ``.tmp`` and pinned
  ``.bak`` sidecars both count;
* the unique highest-generation pair is recovered; verifiable candidates
  without a unique pair make startup refuse with :class:`StateFileError`,
  deleting and overwriting nothing (``serve`` prints one stderr JSON line
  with field=data_file and exits 1);
* only when no candidate is verifiable are every leftover (of either file)
  removed and an empty, sidecar-less state created;
* a valid formal pair wins and every leftover, including stray sidecar
  backups, is swept; a present-but-corrupt formal state is never overwritten;
* a failed directory fsync leaves the paired old inodes as two ``.bak``
  files with both formal paths missing (degraded); the next write verifies
  the pair inside the store lock, hard-links both back, and commits only
  that current request. A corrupt/un-pairable sidecar backup keeps the heal
  at 503/field=data_file until it is repaired.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from typing import Any, List, Tuple

from e2ee_backend.models import Device
from e2ee_backend.persistence import (
    PersistenceUnavailable,
    StateFileError,
    attach_persistence,
)
from e2ee_backend.service import DeviceService

from tests.test_crash_recovery import build_fixture, write_tmp


def _state_bak_names(directory: str) -> List[str]:
    return sorted(name for name in os.listdir(directory)
                  if name.startswith(".state-") and name.endswith(".bak"))


def _sidecar_names(directory: str) -> List[str]:
    return sorted(name for name in os.listdir(directory)
                  if name.startswith(".integrity-"))


def _sidecar_bak_names(directory: str) -> List[str]:
    return sorted(name for name in os.listdir(directory)
                  if name.startswith(".integrity-")
                  and name.endswith(".bak"))


class PairedCommitShapeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.path = os.path.join(self.directory, "state.json")
        self.sidecar = self.path + ".integrity"
        self.service = DeviceService()
        attach_persistence(self.service, self.path)

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_committed_pair_is_same_generation_and_chain_bound(self) -> None:
        for index in range(1, 4):
            self.service.store.add_device(Device("u", f"d{index}", "ik"))
        state = json.load(open(self.path, encoding="utf-8"))
        sidecar = json.load(open(self.sidecar, encoding="utf-8"))
        self.assertEqual(list(sidecar), ["version", "entries"])
        seqs = [entry["commit_seq"] for entry in sidecar["entries"]]
        self.assertEqual(seqs, [1, 2, 3])
        # The two files are of one generation.
        self.assertEqual(state["commit_seq"], seqs[-1])
        self.assertEqual(state["integrity_log_version"], 1)
        # The tail entry binds exactly this state document.
        from e2ee_backend.storage import (
            canonical_integrity_snapshot, integrity_state_hash)
        payload = {key: value for key, value in state.items()
                   if key not in ("version", "commit_seq",
                                  "integrity_log_version")}
        expected_hash = integrity_state_hash(
            canonical_integrity_snapshot(payload))
        self.assertEqual(sidecar["entries"][-1]["state_hash"], expected_hash)
        prev = ""
        for entry in sidecar["entries"]:
            self.assertEqual(list(entry),
                             ["commit_seq", "state_hash", "prev_hash",
                              "hash"])
            self.assertEqual(entry["prev_hash"], prev)
            import hashlib
            basis = {"commit_seq": entry["commit_seq"],
                     "state_hash": entry["state_hash"],
                     "prev_hash": entry["prev_hash"]}
            digest = hashlib.sha256(json.dumps(
                basis, sort_keys=True, separators=(",", ":"),
                ensure_ascii=False).encode("utf-8")).hexdigest()
            self.assertEqual(entry["hash"], digest)
            prev = entry["hash"]

    def test_state_document_is_compact_utf8_without_ascii_escaping(
            self) -> None:
        # Non-ASCII content must be written literally (ensure_ascii=False),
        # compactly and with no trailing newline, like the sidecar.
        self.service.store.add_device(Device("用户", "设备✓", "ik-日本語"))
        raw = open(self.path, "rb").read()
        self.assertNotIn(b"\n", raw)
        self.assertNotIn(b", ", raw)
        self.assertNotIn(b": ", raw)
        self.assertIn("设备✓".encode("utf-8"), raw)
        self.assertNotIn(b"\\u", raw)
        json.loads(raw.decode("utf-8"))


class PairedCrashRecoveryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        _service, self.path, _sid, self.state_bytes = build_fixture(
            self.directory, cursor=2)
        self.sidecar = self.path + ".integrity"
        with open(self.sidecar, "rb") as handle:
            self.sidecar_bytes = handle.read()
        self.generation = json.loads(
            self.state_bytes.decode("utf-8"))["commit_seq"]

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def _remove_formal_pair(self) -> None:
        os.unlink(self.path)
        os.unlink(self.sidecar)

    def test_staged_state_and_sidecar_tmps_recover_as_a_pair(self) -> None:
        self._remove_formal_pair()
        state_tmp = write_tmp(self.directory, ".state-staged.tmp",
                              self.state_bytes)
        log_tmp = write_tmp(self.directory, ".integrity-staged.tmp",
                            self.sidecar_bytes)
        state_ino = os.stat(state_tmp).st_ino
        log_ino = os.stat(log_tmp).st_ino
        service = DeviceService()
        store = attach_persistence(service, self.path)
        self.assertEqual(open(self.path, "rb").read(), self.state_bytes)
        self.assertEqual(open(self.sidecar, "rb").read(), self.sidecar_bytes)
        self.assertEqual(os.stat(self.path).st_ino, state_ino)
        self.assertEqual(os.stat(self.sidecar).st_ino, log_ino)
        self.assertEqual(store.commit_seq, self.generation + 1)
        self.assertEqual(_sidecar_names(self.directory), [])
        self.assertEqual(_state_bak_names(self.directory), [])

    def test_pinned_state_and_sidecar_baks_recover_as_a_pair(self) -> None:
        self._remove_formal_pair()
        write_tmp(self.directory, ".state-pinned.bak", self.state_bytes)
        write_tmp(self.directory, ".integrity-pinned.bak", self.sidecar_bytes)
        service = DeviceService()
        attach_persistence(service, self.path)
        self.assertEqual(open(self.path, "rb").read(), self.state_bytes)
        self.assertEqual(open(self.sidecar, "rb").read(), self.sidecar_bytes)
        self.assertEqual(_sidecar_names(self.directory), [])
        self.assertEqual(_state_bak_names(self.directory), [])

    def test_mixed_tmp_bak_pair_recovers_and_sweeps_other_pair(self) -> None:
        # A complete lower-generation pair (.bak) plus the higher
        # generation pair staged as .tmp; the higher one wins and every
        # lower-generation leftover of either file is swept.
        low_dir = tempfile.mkdtemp()
        try:
            _svc, low_path, _sid, low_state = build_fixture(low_dir, cursor=0)
            low_sidecar = low_path + ".integrity"
            with open(low_sidecar, "rb") as handle:
                low_log = handle.read()
            self._remove_formal_pair()
            write_tmp(self.directory, ".state-low.bak", low_state)
            write_tmp(self.directory, ".integrity-low.bak", low_log)
            write_tmp(self.directory, ".state-high.tmp", self.state_bytes)
            write_tmp(self.directory, ".integrity-high.tmp",
                      self.sidecar_bytes)
            service = DeviceService()
            attach_persistence(service, self.path)
            self.assertEqual(open(self.path, "rb").read(), self.state_bytes)
            self.assertEqual(open(self.sidecar, "rb").read(),
                             self.sidecar_bytes)
            self.assertEqual(_sidecar_names(self.directory), [])
            self.assertEqual(_state_bak_names(self.directory), [])
        finally:
            shutil.rmtree(low_dir, ignore_errors=True)

    def test_unpaired_modern_state_candidate_is_not_recovered(self) -> None:
        # A marker-stamped state leftover with NO sidecar candidate cannot be
        # verified as a committed pair, so it is not promoted: no verifiable
        # candidate exists, the leftovers are swept and an empty state is
        # created (no sidecar until its first real commit).
        self._remove_formal_pair()
        write_tmp(self.directory, ".state-orphan.tmp", self.state_bytes)
        service = DeviceService()
        attach_persistence(service, self.path)
        document = json.load(open(self.path, encoding="utf-8"))
        self.assertEqual(document["commit_seq"], 0)
        self.assertEqual(document["devices"], [])
        self.assertFalse(os.path.exists(self.sidecar))
        self.assertEqual(_sidecar_names(self.directory), [])
        self.assertEqual(
            [n for n in os.listdir(self.directory)
             if n.startswith(".state-")], [])

    def test_unmatched_sidecar_candidate_does_not_recover_state(self) -> None:
        # The state leftover's tail does not bind the only sidecar leftover:
        # no verifiable pair -> empty state, both leftovers swept.
        self._remove_formal_pair()
        other_dir = tempfile.mkdtemp()
        try:
            _svc, other_path, _sid, other_state = build_fixture(
                other_dir, cursor=0)
            # Make the other fixture a strictly different generation by one
            # more commit, so its sidecar tail cannot bind self.state_bytes.
            _svc.store.add_device(Device("u", "extra", "ik"))
            with open(other_path + ".integrity", "rb") as handle:
                other_log = handle.read()
            write_tmp(self.directory, ".state-a.tmp", self.state_bytes)
            write_tmp(self.directory, ".integrity-a.tmp", other_log)
            service = DeviceService()
            attach_persistence(service, self.path)
            self.assertEqual(
                json.load(open(self.path, encoding="utf-8"))["commit_seq"], 0)
            self.assertFalse(os.path.exists(self.sidecar))
            self.assertEqual(_sidecar_names(self.directory), [])
        finally:
            shutil.rmtree(other_dir, ignore_errors=True)

    def test_two_state_candidates_one_sidecar_is_ambiguous_and_untouched(
            self) -> None:
        # Two verifiable state backups of the committed generation both bind
        # the single sidecar backup: two verifiable pairs, no unique choice.
        # Startup must refuse and delete/overwrite nothing.
        self._remove_formal_pair()
        first = write_tmp(self.directory, ".state-a.bak", self.state_bytes)
        second = write_tmp(self.directory, ".state-b.tmp", self.state_bytes)
        log = write_tmp(self.directory, ".integrity-a.bak",
                        self.sidecar_bytes)
        with self.assertRaises(StateFileError):
            attach_persistence(DeviceService(), self.path)
        self.assertFalse(os.path.exists(self.path))
        self.assertFalse(os.path.exists(self.sidecar))
        # Nothing was deleted or overwritten.
        self.assertEqual(open(first, "rb").read(), self.state_bytes)
        self.assertEqual(open(second, "rb").read(), self.state_bytes)
        self.assertEqual(open(log, "rb").read(), self.sidecar_bytes)

    def test_serve_ambiguous_pairs_exit_1_single_data_file_line(
            self) -> None:
        self._remove_formal_pair()
        write_tmp(self.directory, ".state-a.bak", self.state_bytes)
        write_tmp(self.directory, ".state-b.tmp", self.state_bytes)
        write_tmp(self.directory, ".integrity-a.bak", self.sidecar_bytes)
        result = subprocess.run(
            [sys.executable, "-m", "e2ee_backend", "serve",
             "--host", "127.0.0.1", "--port", "0",
             "--data-file", self.path],
            capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        lines = result.stderr.strip().splitlines()
        self.assertEqual(len(lines), 1)
        self.assertEqual(json.loads(lines[0])["field"], "data_file")
        # Still nothing promoted or deleted.
        self.assertFalse(os.path.exists(self.path))

    def test_no_verifiable_pair_sweeps_both_files_and_creates_empty(
            self) -> None:
        self._remove_formal_pair()
        write_tmp(self.directory, ".state-bad.tmp", b"{not json")
        write_tmp(self.directory, ".integrity-bad.tmp", b"{broken")
        write_tmp(self.directory, ".state-old.bak", b'{"version": 1}')
        service = DeviceService()
        attach_persistence(service, self.path)
        self.assertTrue(os.path.exists(self.path))
        self.assertFalse(os.path.exists(self.sidecar))
        self.assertEqual(
            json.load(open(self.path, encoding="utf-8"))["commit_seq"], 0)
        self.assertEqual(_sidecar_names(self.directory), [])
        self.assertEqual(
            [n for n in os.listdir(self.directory)
             if n.startswith(".state-") and
             (n.endswith(".tmp") or n.endswith(".bak"))], [])

    def test_corrupt_formal_state_refuses_without_touching_the_pair(
            self) -> None:
        with open(self.path, "wb") as handle:
            handle.write(b"{corrupt")
        state_before = open(self.path, "rb").read()
        state_ino = os.stat(self.path).st_ino
        log_before = open(self.sidecar, "rb").read()
        log_ino = os.stat(self.sidecar).st_ino
        leftover = write_tmp(self.directory, ".state-orphan.tmp",
                             self.state_bytes)
        with self.assertRaises(StateFileError):
            attach_persistence(DeviceService(), self.path)
        self.assertEqual(open(self.path, "rb").read(), state_before)
        self.assertEqual(os.stat(self.path).st_ino, state_ino)
        self.assertEqual(open(self.sidecar, "rb").read(), log_before)
        self.assertEqual(os.stat(self.sidecar).st_ino, log_ino)
        self.assertTrue(os.path.exists(leftover))

    def test_valid_formal_pair_sweeps_stray_sidecar_leftovers(self) -> None:
        write_tmp(self.directory, ".integrity-stray.bak", self.sidecar_bytes)
        write_tmp(self.directory, ".integrity-staged.tmp", b"{garbage")
        write_tmp(self.directory, ".state-stray.bak", self.state_bytes)
        service = DeviceService()
        attach_persistence(service, self.path)
        self.assertEqual(open(self.path, "rb").read(), self.state_bytes)
        self.assertEqual(open(self.sidecar, "rb").read(), self.sidecar_bytes)
        self.assertEqual(_sidecar_names(self.directory), [])
        self.assertEqual(_state_bak_names(self.directory), [])

    def test_highest_generation_pair_wins_regardless_of_suffix(self) -> None:
        # A complete older pair staged as .tmp and a newer pinned pair as
        # .bak: suffix never decides, generation does.
        older_dir = tempfile.mkdtemp()
        try:
            _svc, older_path, _sid, older_state = build_fixture(
                older_dir, cursor=0)
            with open(older_path + ".integrity", "rb") as handle:
                older_log = handle.read()
            self._remove_formal_pair()
            write_tmp(self.directory, ".state-newer.bak", self.state_bytes)
            write_tmp(self.directory, ".integrity-newer.bak",
                      self.sidecar_bytes)
            write_tmp(self.directory, ".state-older.tmp", older_state)
            write_tmp(self.directory, ".integrity-older.tmp", older_log)
            service = DeviceService()
            attach_persistence(service, self.path)
            self.assertEqual(
                json.load(open(self.path, encoding="utf-8"))["commit_seq"],
                self.generation)
            self.assertEqual(open(self.sidecar, "rb").read(),
                             self.sidecar_bytes)
        finally:
            shutil.rmtree(older_dir, ignore_errors=True)


class PairedDegradedHealTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.service, self.path, _sid, _bytes = build_fixture(self.directory)
        self.sidecar = self.path + ".integrity"
        self.good_state = open(self.path, "rb").read()
        self.good_log = open(self.sidecar, "rb").read()
        self.good_seq = json.loads(
            self.good_state.decode("utf-8"))["commit_seq"]

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def _fail_directory_fsync(self) -> None:
        import e2ee_backend.persistence as persistence_mod
        real_fsync = persistence_mod.os.fsync

        def fail_on_directory_fd(fd: int) -> None:  # noqa: ANN001
            if os.fstat(fd).st_mode & 0o170000 == 0o040000:
                raise OSError("simulated directory fsync failure")
            real_fsync(fd)

        persistence_mod.os.fsync = fail_on_directory_fd

    def _restore_fsync(self) -> None:
        import e2ee_backend.persistence as persistence_mod
        persistence_mod.os.fsync = self._real_fsync

    def test_directory_fsync_failure_leaves_paired_backups(self) -> None:
        import e2ee_backend.persistence as persistence_mod
        self._real_fsync = persistence_mod.os.fsync
        self._fail_directory_fsync()
        try:
            with self.assertRaises(PersistenceUnavailable):
                self.service.revoke_device("alice")
        finally:
            self._restore_fsync()
        # Both formal paths are missing, each predecessor pinned exactly
        # once, and no un-committed snapshot is parked for promotion.
        self.assertFalse(os.path.exists(self.path))
        self.assertFalse(os.path.exists(self.sidecar))
        state_baks = _state_bak_names(self.directory)
        log_baks = _sidecar_bak_names(self.directory)
        self.assertEqual(len(state_baks), 1)
        self.assertEqual(len(log_baks), 1)
        self.assertEqual(
            open(os.path.join(self.directory, state_baks[0]), "rb").read(),
            self.good_state)
        self.assertEqual(
            open(os.path.join(self.directory, log_baks[0]), "rb").read(),
            self.good_log)
        self.assertEqual(
            [n for n in os.listdir(self.directory)
             if n.endswith(".quarantine")], [])
        self.assertFalse(
            self.service.store.find_by_device_id("alice").revoked)

    def test_next_write_heals_the_pair_and_commits_current_request(
            self) -> None:
        import e2ee_backend.persistence as persistence_mod
        self._real_fsync = persistence_mod.os.fsync
        self._fail_directory_fsync()
        try:
            with self.assertRaises(PersistenceUnavailable):
                self.service.revoke_device("alice")
        finally:
            self._restore_fsync()
        # The current request (bob), not a replay of alice, heals the paired
        # backups and advances exactly one generation.
        self.service.revoke_device("bob")
        self.assertTrue(os.path.exists(self.path))
        self.assertTrue(os.path.exists(self.sidecar))
        self.assertEqual(_sidecar_names(self.directory), [])
        self.assertEqual(_state_bak_names(self.directory), [])
        state = json.load(open(self.path, encoding="utf-8"))
        sidecar = json.load(open(self.sidecar, encoding="utf-8"))
        self.assertEqual(state["commit_seq"], self.good_seq + 1)
        self.assertEqual(
            [e["commit_seq"] for e in sidecar["entries"]][-1],
            self.good_seq + 1)
        self.assertTrue(self.service.store.find_by_device_id("bob").revoked)
        self.assertFalse(
            self.service.store.find_by_device_id("alice").revoked)
        # A restart accepts the healed pair unchanged.
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        self.assertTrue(restarted.store.find_by_device_id("bob").revoked)
        self.assertFalse(
            restarted.store.find_by_device_id("alice").revoked)

    def test_corrupt_sidecar_backup_keeps_heal_at_503_until_repaired(
            self) -> None:
        import e2ee_backend.persistence as persistence_mod
        self._real_fsync = persistence_mod.os.fsync
        self._fail_directory_fsync()
        try:
            with self.assertRaises(PersistenceUnavailable):
                self.service.revoke_device("alice")
        finally:
            self._restore_fsync()
        log_bak = os.path.join(self.directory,
                               _sidecar_bak_names(self.directory)[0])
        with open(log_bak, "wb") as handle:
            handle.write(b"{not a sidecar")
        # No verifiable pair: the heal fails 503 and moves nothing.
        with self.assertRaises(PersistenceUnavailable):
            self.service.revoke_device("bob")
        self.assertFalse(os.path.exists(self.path))
        self.assertFalse(os.path.exists(self.sidecar))
        self.assertTrue(
            self.service.store.find_by_device_id("bob") is None
            or not self.service.store.find_by_device_id("bob").revoked)
        # Repair the sidecar pin: the next write heals and commits once.
        with open(log_bak, "wb") as handle:
            handle.write(self.good_log)
        self.service.revoke_device("bob")
        self.assertTrue(os.path.exists(self.path))
        self.assertTrue(os.path.exists(self.sidecar))
        self.assertEqual(
            json.load(open(self.path, encoding="utf-8"))["commit_seq"],
            self.good_seq + 1)
        self.assertTrue(self.service.store.find_by_device_id("bob").revoked)
        self.assertFalse(
            self.service.store.find_by_device_id("alice").revoked)
        self.assertEqual(_sidecar_names(self.directory), [])


class LegacyPairCompatibilityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        _service, self.path, _sid, formal_bytes = build_fixture(self.directory)
        self.sidecar = self.path + ".integrity"
        document = json.loads(formal_bytes.decode("utf-8"))
        # A genuine pre-integrity-log file: no marker, no sidecar.
        document.pop("integrity_log_version", None)
        with open(self.path, "wb") as handle:
            handle.write(json.dumps(document).encode("utf-8"))
        os.remove(self.sidecar)

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_legacy_file_loads_and_stray_sidecar_leftover_is_swept(
            self) -> None:
        # A sidecar crash leftover beside a marker-less legacy formal file is
        # crash garbage; the valid formal file wins and it is swept, while a
        # marker-less file with no sidecar still starts.
        write_tmp(self.directory, ".integrity-stray.tmp", b"{garbage")
        service = DeviceService()
        attach_persistence(service, self.path)
        self.assertFalse(os.path.exists(self.sidecar))
        self.assertEqual(_sidecar_names(self.directory), [])
        # The first real commit re-anchors the chain in one transaction.
        service.store.add_device(Device("u", "brand-new", "ik"))
        state = json.load(open(self.path, encoding="utf-8"))
        sidecar = json.load(open(self.sidecar, encoding="utf-8"))
        self.assertEqual(state["integrity_log_version"], 1)
        self.assertEqual(len(sidecar["entries"]), 1)
        self.assertEqual(sidecar["entries"][0]["commit_seq"],
                         state["commit_seq"])


if __name__ == "__main__":
    unittest.main()
