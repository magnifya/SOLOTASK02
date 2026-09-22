"""Crash-recovery candidates must be one complete durable transaction.

When the formal state file is missing, a ``.state-*.tmp``/``.bak`` leftover is
only a recovery candidate when it is a *full* atomic-commit snapshot: beyond
parsing as version=1 and passing the whole semantic restore validation it must
also explicitly carry the per-device sync cursor sections
(``group_sync_cursors`` and ``message_sync_cursors``) and the ``key_events``
audit-chain section. A semantically-valid but section-less legacy/partial
snapshot is never promoted over a complete one (cursors and the audit chain
must not be silently dropped); with only incomplete candidates the leftovers
are removed and an empty state is created. The section gate applies only to
leftover recovery — a section-less *formal* file still loads leniently.
"""
import json
import os
import shutil
import tempfile
import unittest
from typing import Any, Dict

from e2ee_backend.persistence import attach_persistence
from e2ee_backend.service import DeviceService

from tests.test_crash_recovery import build_fixture, tmp_names, write_tmp


class SectionCompleteCandidateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        _service, self.path, self.sid, self.formal_bytes = build_fixture(
            self.directory, cursor=2)

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def _remove_formal(self) -> None:
        os.unlink(self.path)

    def _strip(self, *sections: str) -> bytes:
        document = json.loads(self.formal_bytes.decode("utf-8"))
        for section in sections:
            del document[section]
        return json.dumps(document).encode("utf-8")

    def _assert_recovered_formal_bytes(self) -> DeviceService:
        service = DeviceService()
        attach_persistence(service, self.path)
        self.assertEqual(open(self.path, "rb").read(), self.formal_bytes)
        self.assertEqual(tmp_names(self.directory), [])
        return service

    def test_newer_keyless_snapshot_falls_back_to_complete_older(self) -> None:
        self._remove_formal()
        # Newer leftover: a valid legacy document whose key_events section was
        # stripped. Older leftover: the complete modern snapshot.
        write_tmp(self.directory, ".state-complete-old.tmp",
                  self.formal_bytes, mtime_ns=1000)
        write_tmp(self.directory, ".state-keyless-new.tmp",
                  self._strip("key_events"), mtime_ns=2000)
        service = self._assert_recovered_formal_bytes()
        # The recovered cursor (only present in the complete snapshot) is back.
        self.assertEqual(
            service.store._group_sync_cursors[(self.sid, "alice")].cursor, 2)
        # And the audit chain survived with it.
        self.assertTrue(
            service.store.snapshot_state().get("key_events"))

    def test_newer_cursorless_snapshot_falls_back_to_complete_older(
            self) -> None:
        self._remove_formal()
        write_tmp(self.directory, ".state-complete-old.bak",
                  self.formal_bytes, mtime_ns=1000)
        write_tmp(self.directory, ".state-cursorless-new.tmp",
                  self._strip("group_sync_cursors"), mtime_ns=2000)
        service = self._assert_recovered_formal_bytes()
        self.assertEqual(
            service.store._group_sync_cursors[(self.sid, "alice")].cursor, 2)

    def test_missing_message_sync_cursors_section_also_disqualifies(
            self) -> None:
        self._remove_formal()
        write_tmp(self.directory, ".state-complete-old.tmp",
                  self.formal_bytes, mtime_ns=1000)
        write_tmp(self.directory, ".state-incomplete-new.tmp",
                  self._strip("message_sync_cursors"), mtime_ns=2000)
        self._assert_recovered_formal_bytes()

    def test_only_incomplete_candidates_create_empty_state(self) -> None:
        self._remove_formal()
        # Every leftover parses and restores, but each is missing one of the
        # durable-transaction sections; none may be promoted.
        write_tmp(self.directory, ".state-a.tmp",
                  self._strip("key_events"), mtime_ns=3000)
        write_tmp(self.directory, ".state-b.bak",
                  self._strip("group_sync_cursors"), mtime_ns=2000)
        write_tmp(self.directory, ".state-c.tmp",
                  self._strip("message_sync_cursors"), mtime_ns=1000)
        service = DeviceService()
        attach_persistence(service, self.path)
        self.assertTrue(os.path.exists(self.path))
        self.assertEqual(tmp_names(self.directory), [])
        snapshot = service.store.snapshot_state()
        self.assertEqual(snapshot["devices"], [])
        self.assertEqual(snapshot["group_sync_cursors"], [])
        with open(self.path, encoding="utf-8") as handle:
            document = json.load(handle)
        self.assertEqual(document["version"], 1)
        # The freshly created empty state is itself section-complete.
        self.assertEqual(document["group_sync_cursors"], [])
        self.assertEqual(document["message_sync_cursors"], [])
        self.assertEqual(document["key_events"], [])

    def test_complete_modern_snapshot_still_recovered(self) -> None:
        self._remove_formal()
        write_tmp(self.directory, ".state-crash.tmp", self.formal_bytes)
        service = self._assert_recovered_formal_bytes()
        body = service.sync_group_messages(self.sid, "alice", None, 100)
        self.assertEqual([m["sequence"] for m in body["messages"]], [3])


class SectionGateDoesNotAffectFormalFileTest(unittest.TestCase):
    """A section-less *formal* file keeps loading leniently (legacy rule)."""

    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        _service, self.path, _sid, formal_bytes = build_fixture(
            self.directory, cursor=2)
        document = json.loads(formal_bytes.decode("utf-8"))
        del document["key_events"]
        del document["group_sync_cursors"]
        with open(self.path, "wb") as handle:
            handle.write(json.dumps(document).encode("utf-8"))

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_sectionless_formal_file_loads_without_recovery_gate(self) -> None:
        service = DeviceService()
        attach_persistence(service, self.path)
        self.assertEqual(service.store._group_sync_cursors, {})
        # Legacy file: chains are pending until the first persisted change.
        self.assertNotIn("key_events", service.store.snapshot_state())
        self.assertEqual(tmp_names(self.directory), [])


if __name__ == "__main__":
    unittest.main()
