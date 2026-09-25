"""Two-file (state + ``<state>.integrity`` sidecar) crash recovery.

A normal commit publishes the version=1 state document and its integrity
sidecar in one locked transaction with a pair of hard-link backups. These
tests pin the paired failure/recovery contract:

* a post-replace directory-fsync failure that cannot be rolled back durably
  vacates both formal paths and leaves one matched ``.state-*.bak`` /
  ``.integrity-*.bak`` pair holding the previous inodes' exact bytes;
* the next write in the same process (heal) and a later restart recover that
  unique *pair* — generation and tail ``state_hash`` must match and the
  sidecar chain must verify — and never replay the 503ed request;
* a verifiable marked state with no sidecar mate, a mismatched mate, or two
  distinct matching pairs is refused with :class:`StateFileError`, nothing is
  deleted or overwritten, and the CLI exits 1 with one stderr JSON line
  (``field=data_file``);
* a sidecar-less legacy state still recovers without a sidecar;
* a valid formal pair takes precedence and sweeps every leftover, and with
  no verifiable candidate the leftovers (state and sidecar) are cleaned and
  an empty state is created.
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
    _integrity_entry_hash,
    attach_persistence,
)
from e2ee_backend.service import DeviceService


def build_fixture(directory: str) -> Tuple[DeviceService, Any, str, str,
                                           bytes, bytes]:
    """Persist a few commits; return service, store, paths and good bytes."""
    path = os.path.join(directory, "state.json")
    sidecar = path + ".integrity"
    service = DeviceService()
    store = attach_persistence(service, path)
    for device_id in ("creator", "alice", "bob"):
        service.store.add_device(Device("u", device_id, "ik"))
    service.revoke_device("alice")
    with open(path, "rb") as handle:
        good_state = handle.read()
    with open(sidecar, "rb") as handle:
        good_side = handle.read()
    return service, store, path, sidecar, good_state, good_side


def _fail_directory_fsync(persistence_mod: Any) -> None:
    """Make every directory-fsync raise (file fsyncs still succeed)."""
    real_fsync = persistence_mod.os.fsync

    def fail_on_directory_fd(fd: int) -> None:  # noqa: ANN001
        if os.fstat(fd).st_mode & 0o170000 == 0o040000:
            raise OSError("simulated directory fsync failure")
        real_fsync(fd)

    persistence_mod.os.fsync = fail_on_directory_fd


def state_baks(directory: str) -> List[str]:
    return sorted(n for n in os.listdir(directory)
                  if n.startswith(".state-") and n.endswith(".bak"))


def side_baks(directory: str) -> List[str]:
    return sorted(n for n in os.listdir(directory)
                  if n.startswith(".integrity-") and n.endswith(".bak"))


def all_pair_leftovers(directory: str) -> List[str]:
    return sorted(
        n for n in os.listdir(directory)
        if (n.startswith(".state-") or n.startswith(".integrity-"))
        and (n.endswith(".tmp") or n.endswith(".bak")
             or n.endswith(".quarantine") or n.endswith(".block")))


class PairedFailureTriageTest(unittest.TestCase):
    def setUp(self) -> None:
        import e2ee_backend.persistence as persistence_mod
        self.persistence_mod = persistence_mod
        self._real_fsync = persistence_mod.os.fsync
        self.directory = tempfile.mkdtemp()
        (self.service, self.store, self.path, self.sidecar,
         self.good_state, self.good_side) = build_fixture(self.directory)

    def tearDown(self) -> None:
        self.persistence_mod.os.fsync = self._real_fsync
        shutil.rmtree(self.directory, ignore_errors=True)

    def _induce_paired_degraded(self) -> None:
        _fail_directory_fsync(self.persistence_mod)
        try:
            with self.assertRaises(PersistenceUnavailable):
                self.service.revoke_device("bob")
        finally:
            self.persistence_mod.os.fsync = self._real_fsync

    def test_failure_leaves_a_matched_pair_of_backups(self) -> None:
        self._induce_paired_degraded()
        # Both formal paths are vacated; exactly one matched pin pair keeps
        # the previous inodes' exact bytes.
        self.assertFalse(os.path.exists(self.path))
        self.assertFalse(os.path.exists(self.sidecar))
        self.assertEqual(len(state_baks(self.directory)), 1)
        self.assertEqual(len(side_baks(self.directory)), 1)
        with open(os.path.join(self.directory,
                               state_baks(self.directory)[0]), "rb") as h:
            self.assertEqual(h.read(), self.good_state)
        with open(os.path.join(self.directory,
                               side_baks(self.directory)[0]), "rb") as h:
            self.assertEqual(h.read(), self.good_side)
        self.assertTrue(self.store.degraded)
        self.assertFalse(self.store.blocked)
        # Memory rolled back: the 503ed revoke never happened.
        self.assertFalse(
            self.service.store.find_by_device_id("bob").revoked)

    def test_next_write_heals_pair_and_commits_only_current_request(
            self) -> None:
        self._induce_paired_degraded()
        # A different current request heals the pair then commits once.
        self.service.revoke_device("creator")
        # The heal restored the pair, then this one request committed a new
        # generation: the formal state is the post-request snapshot (creator
        # revoked), the sidecar tail binds to it, and no pin survives.
        self._verify_committed_sidecar_matches_state()
        self.assertEqual(all_pair_leftovers(self.directory), [])
        doc = json.loads(open(self.path, encoding="utf-8").read())
        self.assertEqual(
            doc["commit_seq"], json.loads(self.good_state)["commit_seq"] + 1)
        self.assertTrue(
            self.service.store.find_by_device_id("creator").revoked)
        self.assertFalse(
            self.service.store.find_by_device_id("bob").revoked)

    def test_restart_recovers_the_matched_pair(self) -> None:
        self._induce_paired_degraded()
        service = DeviceService()
        attach_persistence(service, self.path)
        self.assertEqual(open(self.path, "rb").read(), self.good_state)
        self.assertEqual(open(self.sidecar, "rb").read(), self.good_side)
        self.assertEqual(all_pair_leftovers(self.directory), [])
        self.assertFalse(service.store.find_by_device_id("bob").revoked)
        # The recovered pair passes the strict history gate.
        report = service.persistence_integrity_history()
        self.assertEqual(
            report["commit_seq"],
            json.loads(self.good_state)["commit_seq"])

    def _verify_committed_sidecar_matches_state(self) -> None:
        state_doc = json.loads(open(self.path, encoding="utf-8").read())
        side_doc = json.loads(open(self.sidecar, encoding="utf-8").read())
        last = side_doc["entries"][-1]
        self.assertEqual(last["commit_seq"], state_doc["commit_seq"])

    def test_transient_dir_fsync_rolls_both_inodes_back_in_place(self) -> None:
        # First directory fsync fails (rollback); the second (flushing the
        # rollback) succeeds: both previous inodes return to the formal paths
        # and no pin survives.
        real_fsync = self._real_fsync
        calls = {"n": 0}

        def fail_first(fd: int) -> None:  # noqa: ANN001
            if os.fstat(fd).st_mode & 0o170000 == 0o040000:
                calls["n"] += 1
                if calls["n"] == 1:
                    raise OSError("transient directory fsync failure")
            real_fsync(fd)

        self.persistence_mod.os.fsync = fail_first
        try:
            with self.assertRaises(PersistenceUnavailable):
                self.service.revoke_device("bob")
        finally:
            self.persistence_mod.os.fsync = real_fsync
        self.assertEqual(open(self.path, "rb").read(), self.good_state)
        self.assertEqual(open(self.sidecar, "rb").read(), self.good_side)
        self.assertEqual(all_pair_leftovers(self.directory), [])
        self.assertFalse(self.store.degraded)


class PairRecoverySelectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        (_s, _st, self.path, self.sidecar,
         self.good_state, self.good_side) = build_fixture(self.directory)

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def _reset_to_leftovers(self, *writes: Tuple[str, bytes]) -> None:
        os.unlink(self.path)
        os.unlink(self.sidecar)
        for name, content in writes:
            with open(os.path.join(self.directory, name), "wb") as handle:
                handle.write(content)

    def test_formal_state_deleted_but_formal_sidecar_survives_recovers(self) -> None:
        os.unlink(self.path)
        with open(os.path.join(self.directory, ".state-crash.tmp"),
                  "wb") as handle:
            handle.write(self.good_state)
        # The formal sidecar is still on its path; the state binds to it.
        service = DeviceService()
        attach_persistence(service, self.path)
        self.assertEqual(open(self.path, "rb").read(), self.good_state)
        self.assertEqual(open(self.sidecar, "rb").read(), self.good_side)
        self.assertEqual(all_pair_leftovers(self.directory), [])

    def test_marked_state_without_any_sidecar_is_hard_refusal(self) -> None:
        self._reset_to_leftovers((".state-a.tmp", self.good_state))
        with self.assertRaises(StateFileError):
            attach_persistence(DeviceService(), self.path)
        # Nothing is promoted, deleted or overwritten.
        self.assertFalse(os.path.exists(self.path))
        self.assertTrue(
            os.path.exists(os.path.join(self.directory, ".state-a.tmp")))
        self.assertEqual(side_baks(self.directory), [])

    def test_two_distinct_matching_sidecar_pairs_are_refused(self) -> None:
        self._reset_to_leftovers(
            (".state-a.tmp", self.good_state),
            (".integrity-a.bak", self.good_side),
            (".integrity-b.bak", self.good_side))
        before = {n: open(os.path.join(self.directory, n), "rb").read()
                  for n in (".state-a.tmp", ".integrity-a.bak",
                            ".integrity-b.bak")}
        with self.assertRaises(StateFileError):
            attach_persistence(DeviceService(), self.path)
        self.assertFalse(os.path.exists(self.path))
        for name, content in before.items():
            self.assertEqual(
                open(os.path.join(self.directory, name), "rb").read(),
                content)

    def test_mismatched_sidecar_tail_hash_is_refused(self) -> None:
        side_doc = json.loads(self.good_side.decode("utf-8"))
        last = side_doc["entries"][-1]
        last["state_hash"] = "9" * 64
        last["hash"] = _integrity_entry_hash(
            last["commit_seq"], last["state_hash"], last["prev_hash"])
        tampered_side = json.dumps(
            side_doc, separators=(",", ":"), ensure_ascii=False).encode()
        self._reset_to_leftovers(
            (".state-a.tmp", self.good_state),
            (".integrity-a.bak", tampered_side))
        with self.assertRaises(StateFileError):
            attach_persistence(DeviceService(), self.path)
        self.assertFalse(os.path.exists(self.path))

    def test_unpaired_verifiable_state_blocks_recovery_and_keeps_files(
            self) -> None:
        # An older complete pair AND a newer verifiable state whose sidecar is
        # missing: recovery cannot prove whether the newer state committed,
        # so it must refuse rather than recover the older pair and delete the
        # orphan. Nothing is promoted, deleted or overwritten.
        older_state = self.good_state
        older_side = self.good_side
        other_dir = tempfile.mkdtemp()
        try:
            o_service = DeviceService()
            attach_persistence(o_service,
                               os.path.join(other_dir, "state.json"))
            for device_id in ("creator", "alice", "bob"):
                o_service.store.add_device(Device("u", device_id, "ik"))
            o_service.revoke_device("alice")
            o_service.revoke_device("bob")  # one commit newer
            with open(os.path.join(other_dir, "state.json"), "rb") as h:
                newer_state = h.read()
        finally:
            shutil.rmtree(other_dir, ignore_errors=True)
        self._reset_to_leftovers(
            (".state-old.tmp", older_state),
            (".integrity-old.bak", older_side),
            (".state-newer.tmp", newer_state))
        with self.assertRaises(StateFileError):
            attach_persistence(DeviceService(), self.path)
        self.assertFalse(os.path.exists(self.path))
        # Every leftover is kept for an operator to resolve.
        for name in (".state-old.tmp", ".integrity-old.bak",
                     ".state-newer.tmp"):
            self.assertTrue(os.path.exists(os.path.join(self.directory, name)),
                            name)

    def test_legacy_sidecarless_state_is_recovered(self) -> None:
        doc = json.loads(self.good_state.decode("utf-8"))
        doc.pop("integrity_log_version")
        legacy = json.dumps(doc, separators=(",", ":")).encode()
        self._reset_to_leftovers((".state-old.tmp", legacy))
        service = DeviceService()
        attach_persistence(service, self.path)
        self.assertEqual(open(self.path, "rb").read(), legacy)
        self.assertFalse(os.path.exists(self.sidecar))
        self.assertEqual(all_pair_leftovers(self.directory), [])

    def test_no_verifiable_candidate_cleans_both_and_creates_empty(
            self) -> None:
        self._reset_to_leftovers(
            (".state-bad.tmp", b"{nope"),
            (".integrity-bad.bak", b"{nope"))
        service = DeviceService()
        attach_persistence(service, self.path)
        self.assertTrue(os.path.exists(self.path))
        self.assertEqual(all_pair_leftovers(self.directory), [])
        self.assertEqual(service.store.snapshot_state()["devices"], [])
        # The fresh empty store starts sidecar-less until its first commit.
        self.assertFalse(os.path.exists(self.sidecar))

    def test_valid_formal_pair_sweeps_every_leftover(self) -> None:
        # Both formals valid; leftover pins (state and sidecar) are stale.
        for name, content in ((".state-old.bak", self.good_state),
                              (".integrity-old.bak", self.good_side)):
            with open(os.path.join(self.directory, name), "wb") as handle:
                handle.write(content)
        state_ino = os.stat(self.path).st_ino
        side_ino = os.stat(self.sidecar).st_ino
        service = DeviceService()
        attach_persistence(service, self.path)
        self.assertEqual(os.stat(self.path).st_ino, state_ino)
        self.assertEqual(os.stat(self.sidecar).st_ino, side_ino)
        self.assertEqual(all_pair_leftovers(self.directory), [])


class PairedBlockingRestartTest(unittest.TestCase):
    """The fully blocked triage keeps paired pins and refuses at restart."""

    def setUp(self) -> None:
        import e2ee_backend.persistence as persistence_mod
        self.persistence_mod = persistence_mod
        self._real_fsync = persistence_mod.os.fsync
        self._real_replace = persistence_mod.os.replace
        self._real_unlink = persistence_mod.os.unlink
        self.directory = tempfile.mkdtemp()
        (_s, self.store, self.path, self.sidecar,
         self.good_state, self.good_side) = build_fixture(self.directory)
        self.service = _s

    def tearDown(self) -> None:
        self.persistence_mod.os.fsync = self._real_fsync
        self.persistence_mod.os.replace = self._real_replace
        self.persistence_mod.os.unlink = self._real_unlink
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_blocked_restart_refuses_and_promotes_after_path_vacated(
            self) -> None:
        real_fsync = self._real_fsync
        real_replace = self._real_replace
        real_unlink = self._real_unlink
        target = os.path.abspath(self.path)

        def fail_fd(fd: int) -> None:  # noqa: ANN001
            if os.fstat(fd).st_mode & 0o170000 == 0o040000:
                raise OSError("simulated post-replace fsync failure")
            real_fsync(fd)

        def repl(src: str, dst: str) -> None:
            s = os.path.abspath(src)
            dd = os.path.abspath(dst)
            if (s.endswith(".bak") and dd == target) or (
                    s == target and dd.endswith(".quarantine")):
                raise OSError("simulated blocked rename")
            return real_replace(src, dst)

        def unlink(path: str) -> None:
            if os.path.abspath(path) == target:
                raise OSError("simulated blocked formal unlink")
            return real_unlink(path)

        self.persistence_mod.os.fsync = fail_fd
        self.persistence_mod.os.replace = repl
        self.persistence_mod.os.unlink = unlink
        try:
            with self.assertRaises(PersistenceUnavailable):
                self.service.revoke_device("bob")
        finally:
            self.persistence_mod.os.fsync = real_fsync
            self.persistence_mod.os.replace = real_replace
            self.persistence_mod.os.unlink = real_unlink

        self.assertTrue(self.store.blocked)
        # The un-committed residual occupies the state formal; the old chain
        # is re-pinned as a sidecar backup; a block marker survives.
        self.assertTrue(os.path.exists(self.path))
        self.assertEqual(len(side_baks(self.directory)), 1)
        self.assertEqual(
            len([n for n in os.listdir(self.directory)
                 if n.endswith(".block")]), 1)
        residual = open(self.path, "rb").read()
        with self.assertRaises(StateFileError):
            attach_persistence(DeviceService(), self.path)
        self.assertEqual(open(self.path, "rb").read(), residual)

        # Operator vacates the residual; the unique paired pins recover.
        os.unlink(self.path)
        service = DeviceService()
        attach_persistence(service, self.path)
        self.assertEqual(open(self.path, "rb").read(), self.good_state)
        self.assertEqual(open(self.sidecar, "rb").read(), self.good_side)
        self.assertEqual(all_pair_leftovers(self.directory), [])
        self.assertFalse(service.store.find_by_device_id("bob").revoked)


class PairAmbiguityCliTest(unittest.TestCase):
    def test_serve_exits_1_single_stderr_json_line(self) -> None:
        directory = tempfile.mkdtemp()
        try:
            _s, _st, path, _side, good_state, good_side = build_fixture(
                directory)
            os.unlink(path)
            os.unlink(path + ".integrity")
            for name, content in ((".state-a.tmp", good_state),
                                  (".integrity-a.bak", good_side),
                                  (".integrity-b.bak", good_side)):
                with open(os.path.join(directory, name), "wb") as handle:
                    handle.write(content)
            result = subprocess.run(
                [sys.executable, "-m", "e2ee_backend", "serve",
                 "--host", "127.0.0.1", "--port", "0",
                 "--data-file", path],
                capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 1)
            self.assertEqual(result.stdout, "")
            lines = result.stderr.strip().splitlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(json.loads(lines[0])["field"], "data_file")
        finally:
            shutil.rmtree(directory, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
