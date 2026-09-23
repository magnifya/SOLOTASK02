"""Same-process self-healing after an undecidable durable-write failure.

When a new snapshot has replaced the formal file but the following directory
fsync fails *and* the rollback (rename or its fsync) also fails, the failure
is made decidable: the formal path is left missing and the old committed
inode survives only as a single ``.state-*.bak`` (the un-committed snapshot
is parked under ``.quarantine`` or deleted). Memory is rolled back and the
triggering request is answered 503/field=data_file.

These tests cover the next persistable write in the *same* process: while
still holding the storage lock it must fully verify the pinned backup
(version=1, the exact commit generation, the complete cross-entity/restore
semantics and equality with the in-memory last-good state), atomically
promote it and remove the failed transaction's leftovers, and only then
commit that *later* request — never replaying the request that already got a
503. A failed healing keeps everything at the last committed state and keeps
answering 503 so it can be retried; exactly one of concurrent writers
performs the recovery and each successful commit adds one generation.
"""
import json
import os
import shutil
import tempfile
import threading
import unittest
from http.client import HTTPConnection
from typing import Any, List

import e2ee_backend.persistence as persistence_mod
from e2ee_backend.models import Device
from e2ee_backend.persistence import (
    PersistenceUnavailable,
    attach_persistence,
)
from e2ee_backend.service import DeviceService


def build_fixture(directory: str):
    """Persist one group session with seq 1..3; return service, store, path."""
    path = os.path.join(directory, "state.json")
    service = DeviceService()
    state_store = attach_persistence(service, path)
    for device_id in ("creator", "alice", "bob", "carol"):
        service.store.add_device(Device("u", device_id, "ik"))
    service.create_group({
        "group_id": "g1", "creator_device_id": "creator",
        "member_device_ids": ["alice", "bob", "carol"]})
    session = service.create_group_session({
        "group_id": "g1", "initiator_device_id": "creator",
        "ephemeral_key": "epk"})
    sid = session["session_id"]
    for sequence in range(1, 4):
        service.post_message({
            "session_id": sid, "sender_device_id": "creator",
            "message_id": f"m{sequence}", "sequence": sequence,
            "nonce": f"n{sequence}", "ciphertext": "ct"})
    service.sync_group_checkpoint(
        sid, {"device_id": "alice", "cursor": 2})
    return service, state_store, path, sid


def names_suffix(directory: str, suffix: str) -> List[str]:
    return sorted(name for name in os.listdir(directory)
                  if name.startswith(".state-") and name.endswith(suffix))


def break_directory_fsync() -> None:
    """Make every fsync of a directory fd raise; file fsyncs still work."""
    real_fsync = persistence_mod.os.fsync

    def fail_on_directory_fd(fd: int) -> None:  # noqa: ANN001
        if os.fstat(fd).st_mode & 0o170000 == 0o040000:
            raise OSError("simulated directory fsync failure")
        real_fsync(fd)

    persistence_mod.os.fsync = fail_on_directory_fd


class SameProcessSelfHealTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.service, self.state_store, self.path, self.sid = build_fixture(
            self.directory)
        with open(self.path, "rb") as handle:
            self.good_bytes = handle.read()
        self.good_ino = os.stat(self.path).st_ino
        self.good_seq = json.loads(self.good_bytes.decode("utf-8"))[
            "commit_seq"]
        self._real_fsync = persistence_mod.os.fsync

    def tearDown(self) -> None:
        persistence_mod.os.fsync = self._real_fsync
        persistence_mod.os.replace = os.replace
        shutil.rmtree(self.directory, ignore_errors=True)

    # -- inducing the undecidable failure ---------------------------------

    def _induce_degraded_failure(self, device_id: str = "alice") -> None:
        break_directory_fsync()
        try:
            with self.assertRaises(PersistenceUnavailable):
                self.service.revoke_device(device_id)
        finally:
            persistence_mod.os.fsync = self._real_fsync

    def test_failure_leaves_decidable_triage(self) -> None:
        self._induce_degraded_failure()
        self.assertTrue(self.state_store.degraded)
        self.assertFalse(os.path.exists(self.path))
        baks = names_suffix(self.directory, ".bak")
        self.assertEqual(len(baks), 1)
        backup = os.path.join(self.directory, baks[0])
        self.assertEqual(open(backup, "rb").read(), self.good_bytes)
        self.assertEqual(os.stat(backup).st_ino, self.good_ino)
        # The un-committed new snapshot is never at the formal path nor
        # offered as a recovery candidate.
        self.assertFalse(
            any(name.endswith(".tmp")
                for name in os.listdir(self.directory)))
        self.assertFalse(
            self.service.store.find_by_device_id("alice").revoked)

    # -- the next write heals and commits exactly once ---------------------

    def test_next_write_promotes_backup_and_commits_current_request(self) -> None:
        self._induce_degraded_failure()
        # The later request (revoke bob), not a replay of the 503 request
        # (revoke alice), is what the healed commit carries.
        self.service.revoke_device("bob")
        self.assertFalse(self.state_store.degraded)
        self.assertTrue(os.path.exists(self.path))
        self.assertEqual(names_suffix(self.directory, ".bak"), [])
        self.assertEqual(names_suffix(self.directory, ".tmp"), [])
        self.assertEqual(names_suffix(self.directory, ".quarantine"), [])
        document = json.loads(open(self.path, encoding="utf-8").read())
        self.assertEqual(document["commit_seq"], self.good_seq + 1)
        self.assertFalse(
            self.service.store.find_by_device_id("alice").revoked)
        self.assertTrue(
            self.service.store.find_by_device_id("bob").revoked)

    def test_memory_file_bytes_inode_and_key_events_all_agree(self) -> None:
        self._induce_degraded_failure()
        self.service.revoke_device("bob")
        # A fresh process sees exactly the committed state.
        reloaded = DeviceService()
        attach_persistence(reloaded, self.path)
        self.assertEqual(reloaded.store.snapshot_state(),
                         self.service.store.snapshot_state())
        # The revoked device's audit chain links through, one event more than
        # before (device_revoked) and the file carries the same chains.
        good_doc = json.loads(self.good_bytes.decode("utf-8"))
        bob_chain_before = len(
            [e for e in good_doc["key_events"] if e["device_id"] == "bob"])
        reloaded_events = [
            e for e in reloaded.store.snapshot_state()["key_events"]
            if e["device_id"] == "bob"]
        self.assertEqual(reloaded_events[-1]["type"], "device_revoked")
        self.assertEqual(len(reloaded_events), bob_chain_before + 1)
        # The saved cursor survived the heal unchanged.
        self.assertEqual(
            reloaded.store._group_sync_cursors[(self.sid, "alice")].cursor, 2)

    def test_recovered_generation_then_generations_stay_consecutive(self) -> None:
        self._induce_degraded_failure()
        self.service.revoke_device("bob")
        self.service.revoke_device("carol")
        document = json.loads(open(self.path, encoding="utf-8").read())
        self.assertEqual(document["commit_seq"], self.good_seq + 2)

    # -- the 503 request is never replayed ---------------------------------

    def test_replay_does_not_resurrect_failed_request(self) -> None:
        self._induce_degraded_failure("alice")
        # The next request is a completely different operation (a message).
        self.service.post_message({
            "session_id": self.sid, "sender_device_id": "creator",
            "message_id": "m4", "sequence": 4,
            "nonce": "n4", "ciphertext": "ct"})
        self.assertFalse(
            self.service.store.find_by_device_id("alice").revoked)
        stream = self.service.store.snapshot_state()["messages"][self.sid]
        self.assertEqual([m["sequence"] for m in stream], [1, 2, 3, 4])
        reloaded = DeviceService()
        attach_persistence(reloaded, self.path)
        self.assertFalse(
            reloaded.store.find_by_device_id("alice").revoked)

    # -- healing failures keep returning 503 and advance nothing -----------

    def test_corrupt_pinned_backup_is_refused_and_retryable(self) -> None:
        self._induce_degraded_failure()
        baks = names_suffix(self.directory, ".bak")
        self.assertEqual(len(baks), 1)
        # Corrupt the only backup: healing must refuse to promote it.
        with open(os.path.join(self.directory, baks[0]), "wb") as handle:
            handle.write(b"{not the committed state")
        with self.assertRaises(PersistenceUnavailable):
            self.service.revoke_device("bob")
        # Nothing advanced: the formal path is still missing, the corrupt
        # backup stays pinned, memory is still at the last-good state.
        self.assertTrue(self.state_store.degraded)
        self.assertFalse(os.path.exists(self.path))
        self.assertEqual(names_suffix(self.directory, ".bak"), baks)
        self.assertFalse(
            self.service.store.find_by_device_id("bob").revoked)
        # Restore a valid backup with the exact last-good bytes: the next
        # attempt now heals and commits.
        with open(os.path.join(self.directory, baks[0]), "wb") as handle:
            handle.write(self.good_bytes)
        self.service.revoke_device("bob")
        self.assertFalse(self.state_store.degraded)
        self.assertTrue(
            self.service.store.find_by_device_id("bob").revoked)
        self.assertEqual(
            json.loads(open(self.path, encoding="utf-8").read())[
                "commit_seq"],
            self.good_seq + 1)

    def test_semantically_invalid_backup_is_refused(self) -> None:
        self._induce_degraded_failure()
        baks = names_suffix(self.directory, ".bak")
        # Version=1 parses but is semantically dangling: a cursor past the
        # session's max sequence. Such a backup must never be promoted.
        document = json.loads(self.good_bytes.decode("utf-8"))
        document["group_sync_cursors"] = [{
            "session_id": self.sid, "device_id": "alice", "cursor": 99,
            "updated_at": "2026-01-01T00:00:00+00:00"}]
        with open(os.path.join(self.directory, baks[0]), "w",
                  encoding="utf-8") as handle:
            json.dump(document, handle)
        with self.assertRaises(PersistenceUnavailable):
            self.service.revoke_device("bob")
        self.assertTrue(self.state_store.degraded)
        self.assertFalse(os.path.exists(self.path))

    def test_foreign_backup_is_refused(self) -> None:
        self._induce_degraded_failure()
        baks = names_suffix(self.directory, ".bak")
        # A structurally and semantically valid *different* state (an empty
        # store) at the expected generation: it must not replace the process's
        # actual last-good state.
        other_dir = tempfile.mkdtemp()
        try:
            other = DeviceService()
            attach_persistence(other, os.path.join(other_dir, "state.json"))
            foreign_path = os.path.join(other_dir, "state.json")
            # Stamp the foreign document with the expected generation.
            foreign = json.loads(
                open(foreign_path, encoding="utf-8").read())
            foreign["commit_seq"] = self.good_seq
            with open(os.path.join(self.directory, baks[0]), "w",
                      encoding="utf-8") as handle:
                json.dump(foreign, handle)
        finally:
            shutil.rmtree(other_dir, ignore_errors=True)
        with self.assertRaises(PersistenceUnavailable):
            self.service.revoke_device("bob")
        self.assertTrue(self.state_store.degraded)
        self.assertFalse(os.path.exists(self.path))
        # Memory was not clobbered by the foreign snapshot.
        self.assertIsNotNone(self.service.store.find_by_device_id("creator"))

    def test_two_matching_backups_is_ambiguous_and_refused(self) -> None:
        self._induce_degraded_failure()
        # A second, byte-identical pinned backup is an ambiguity the
        # same-process heal must not resolve by guessing.
        baks = names_suffix(self.directory, ".bak")
        second = os.path.join(self.directory, ".state-dup.bak")
        shutil.copyfile(os.path.join(self.directory, baks[0]), second)
        with self.assertRaises(PersistenceUnavailable):
            self.service.revoke_device("bob")
        self.assertTrue(self.state_store.degraded)
        self.assertFalse(os.path.exists(self.path))
        # Remove the ambiguity; the next attempt heals.
        os.unlink(second)
        self.service.revoke_device("bob")
        self.assertFalse(self.state_store.degraded)

    def test_promotion_rename_failure_keeps_503_and_is_retryable(self) -> None:
        self._induce_degraded_failure()
        real_replace = persistence_mod.os.replace
        target = os.path.abspath(self.path)
        calls = {"n": 0}

        def fail_promotion(src: str, dst: str) -> None:
            # Fail only the .bak -> formal promotion of the heal, once.
            if (os.path.abspath(dst) == target and src.endswith(".bak")):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise OSError("simulated promotion failure")
            return real_replace(src, dst)

        persistence_mod.os.replace = fail_promotion
        try:
            with self.assertRaises(PersistenceUnavailable):
                self.service.revoke_device("bob")
        finally:
            persistence_mod.os.replace = real_replace
        # The failed promotion moved nothing: formal missing, backup pinned.
        self.assertTrue(self.state_store.degraded)
        self.assertFalse(os.path.exists(self.path))
        self.assertEqual(len(names_suffix(self.directory, ".bak")), 1)
        self.assertFalse(
            self.service.store.find_by_device_id("bob").revoked)
        # Next attempt heals and commits normally.
        self.service.revoke_device("bob")
        self.assertFalse(self.state_store.degraded)
        self.assertTrue(
            self.service.store.find_by_device_id("bob").revoked)

    # -- concurrency: exactly one recovery ---------------------------------

    def test_concurrent_writers_have_single_recovery_and_commit(self) -> None:
        self._induce_degraded_failure()
        real_recover = persistence_mod.recover_same_process
        recover_calls = {"started": 0, "healed": 0}
        call_lock = threading.Lock()

        def counting_recover(store, expected):  # noqa: ANN001
            with call_lock:
                recover_calls["started"] += 1
            result = real_recover(store, expected)
            if result:
                with call_lock:
                    recover_calls["healed"] += 1
            return result

        persistence_mod.recover_same_process = counting_recover
        errors: List[str] = []
        errors_lock = threading.Lock()
        device_ids = ["bob", "carol"]
        barrier = threading.Barrier(len(device_ids))

        def revoke(device_id: str) -> None:
            barrier.wait()
            try:
                self.service.revoke_device(device_id)
            except PersistenceUnavailable:
                with errors_lock:
                    errors.append(device_id)

        threads = [threading.Thread(target=revoke, args=(d,))
                   for d in device_ids]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        persistence_mod.recover_same_process = real_recover

        self.assertEqual(errors, [])
        # Exactly one writer performed the heal; both distinct requests
        # committed afterwards, each adding one generation.
        self.assertEqual(recover_calls["healed"], 1)
        self.assertEqual(recover_calls["started"], 1)
        self.assertTrue(
            self.service.store.find_by_device_id("bob").revoked)
        self.assertTrue(
            self.service.store.find_by_device_id("carol").revoked)
        document = json.loads(open(self.path, encoding="utf-8").read())
        self.assertEqual(document["commit_seq"], self.good_seq + 2)
        self.assertEqual(names_suffix(self.directory, ".bak"), [])
        self.assertEqual(names_suffix(self.directory, ".tmp"), [])
        self.assertEqual(names_suffix(self.directory, ".quarantine"), [])

    # -- quarantine leftovers of the failed transaction are cleaned --------

    def test_quarantined_snapshot_is_removed_on_heal(self) -> None:
        # Induce the triage via the rollback-rename failure path, which parks
        # the un-committed new snapshot under a .quarantine name.
        real_replace = persistence_mod.os.replace
        target = os.path.abspath(self.path)

        def fail_backup_rename(src: str, dst: str) -> None:
            if os.path.abspath(dst) == target and src.endswith(".bak"):
                raise OSError("simulated rollback rename failure")
            return real_replace(src, dst)

        real_fsync = self._real_fsync

        def fail_on_directory_fd(fd: int) -> None:  # noqa: ANN001
            if os.fstat(fd).st_mode & 0o170000 == 0o040000:
                raise OSError("simulated post-replace fsync failure")
            real_fsync(fd)

        persistence_mod.os.fsync = fail_on_directory_fd
        persistence_mod.os.replace = fail_backup_rename
        try:
            with self.assertRaises(PersistenceUnavailable):
                self.service.revoke_device("alice")
        finally:
            persistence_mod.os.fsync = real_fsync
            persistence_mod.os.replace = real_replace
        self.assertEqual(len(names_suffix(self.directory, ".quarantine")), 1)
        self.service.revoke_device("bob")
        self.assertEqual(names_suffix(self.directory, ".quarantine"), [])
        self.assertEqual(names_suffix(self.directory, ".bak"), [])
        self.assertTrue(
            self.service.store.find_by_device_id("bob").revoked)
        self.assertFalse(
            self.service.store.find_by_device_id("alice").revoked)


class SameProcessSelfHealHTTPTest(unittest.TestCase):
    """The 503 then next-write-200 boundary over a real loopback socket."""

    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.service = DeviceService()
        self.state_store = attach_persistence(
            self.service, os.path.join(self.directory, "state.json"))
        self.service.store.add_device(Device("u", "d1", "ik"))
        self.service.store.add_device(Device("u", "d2", "ik"))
        from e2ee_backend.http_app import create_server
        self.server, _ = create_server("127.0.0.1", 0, service=self.service)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(
            target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self._real_fsync = persistence_mod.os.fsync

    def tearDown(self) -> None:
        persistence_mod.os.fsync = self._real_fsync
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        shutil.rmtree(self.directory, ignore_errors=True)

    def _post(self, path: str, body: Any):
        connection = HTTPConnection("127.0.0.1", self.port, timeout=5)
        connection.request("POST", path, body=json.dumps(body),
                           headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        data = response.read().decode("utf-8")
        connection.close()
        return response.status, json.loads(data)

    def test_503_then_next_write_heals_with_200(self) -> None:
        break_directory_fsync()
        try:
            status, body = self._post("/v1/devices/d1/revoke", {})
        finally:
            persistence_mod.os.fsync = self._real_fsync
        self.assertEqual(status, 503)
        self.assertEqual(body["field"], "data_file")
        self.assertFalse(self.service.store.find_by_device_id("d1").revoked)
        # The next request heals in-process and succeeds.
        status, body = self._post("/v1/devices/d2/revoke", {})
        self.assertEqual(status, 200)
        self.assertEqual(body, {"device_id": "d2", "revoked": True})
        self.assertFalse(self.service.store.find_by_device_id("d1").revoked)
        self.assertTrue(self.service.store.find_by_device_id("d2").revoked)
        # And a third request is a plain one-generation commit.
        seq_after_heal = json.loads(open(
            os.path.join(self.directory, "state.json"),
            encoding="utf-8").read())["commit_seq"]
        status, _body = self._post("/v1/devices/d1/revoke", {})
        self.assertEqual(status, 200)
        document = json.loads(open(
            os.path.join(self.directory, "state.json"),
            encoding="utf-8").read())
        self.assertEqual(document["commit_seq"], seq_after_heal + 1)


class LegacyBackupSelfHealTest(unittest.TestCase):
    """A pinned backup written by an older build still heals in-process."""

    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.path = os.path.join(self.directory, "state.json")

    def tearDown(self) -> None:
        persistence_mod.os.fsync = os.fsync
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_legacy_generation_zero_backup_is_promoted(self) -> None:
        # Hand-write a legacy version=1 document: no commit_seq and none of
        # the later optional sections.
        legacy = {
            "version": 1,
            "devices": [{
                "user_id": "u", "device_id": "d1", "identity_key": "ik",
                "registered_at": "2026-01-01T00:00:00+00:00",
                "rotated_at": "2026-01-01T00:00:00+00:00",
                "revoked": False, "prekeys": []}],
            "sessions": [], "prekey_claims": [], "prekey_batch_claims": [],
            "claim_session_bindings": [],
            "batch_claim_session_bindings": [], "groups": [],
            "group_sessions": [], "group_session_rotations": [],
            "messages": {}, "delivery": [], "group_delivery": [],
            "used_nonces": {}, "message_submissions": []}
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(legacy, handle)
        service = DeviceService()
        state_store = attach_persistence(service, self.path)
        self.assertEqual(state_store.commit_seq, 1)  # next save writes gen 1
        # Induce the undecidable failure.
        real_fsync = persistence_mod.os.fsync

        def fail_on_directory_fd(fd: int) -> None:  # noqa: ANN001
            if os.fstat(fd).st_mode & 0o170000 == 0o040000:
                raise OSError("simulated directory fsync failure")
            real_fsync(fd)

        persistence_mod.os.fsync = fail_on_directory_fd
        try:
            with self.assertRaises(PersistenceUnavailable):
                service.revoke_device("d1")
        finally:
            persistence_mod.os.fsync = real_fsync
        self.assertTrue(state_store.degraded)
        # The next write promotes the generation-0 legacy backup and then
        # commits as generation 1 (strictly consecutive, no skip/repeat).
        service.store.add_device(Device("u", "d2", "ik"))
        self.assertFalse(state_store.degraded)
        document = json.loads(open(self.path, encoding="utf-8").read())
        self.assertEqual(document["commit_seq"], 1)
        self.assertTrue(service.store.find_by_device_id("d2") is not None)
        # The 503 request was not replayed: d1 is still active until a real
        # later revocation commits.
        self.assertFalse(service.store.find_by_device_id("d1").revoked)


if __name__ == "__main__":
    unittest.main()
