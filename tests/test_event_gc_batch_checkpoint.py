"""Tests for the batch-cleanup audit consumer checkpoint endpoint.

``POST /v1/event-gc-batch/checkpoint`` reads or advances one consumer's
checkpoint on the committed batch-cleanup audit chain (the
``event_gc_batch_cleanup_requests`` records). The call takes no query
parameters (any -> 400/query) and the body carries exactly
``consumer_id`` (a non-empty string), ``expected`` and ``after`` (both
null, or both non-boolean integers in 0..2**63-1); a bad/non-object body
is 400/request_body, a missing/wrongly typed/extra field is 400 with that
field, and exactly one of ``expected``/``after`` being null is
400/expected. A double-null body is a read-only query (200): a consumer
that never advanced reads as ``after`` 0 with a null ``updated_at``.
Otherwise ``expected`` must equal the current checkpoint (0 with no
record, 409/expected), ``after`` must not move backwards or exceed the
number of committed audit records (409/after), an equal ``after`` is an
idempotent no-op (200) and a greater one advances (201). Checkpoints
persist in the ``cleanup_checkpoints`` section (right after
``event_gc_batch_cleanup_requests``, part of the integrity snapshot);
older version-1 files without the section load as empty.
"""
import json
import os
import shutil
import tempfile
import threading
import unittest
from http.client import HTTPConnection

from e2ee_backend.models import Device
from e2ee_backend.service import DeviceService, ServiceError
from e2ee_backend.http_app import create_server
from e2ee_backend.persistence import (
    PersistenceUnavailable,
    StateFileError,
    attach_persistence,
)

PATH = "/v1/event-gc-batch/checkpoint"


class CheckpointMixin:
    def _build(self) -> None:
        self.service = DeviceService()
        self.service.store.add_device(Device("u", "alice", "ik"))

    def _commit(self, request_id):
        # A commit carrying request_id always appends one audit record,
        # even when nothing was deleted.
        return self.service.event_gc_cleanup_expired_batch({
            "mode": "commit", "device_ids": ["alice"],
            "after": 0, "limit": 100, "request_id": request_id})

    def _checkpoint(self, consumer="c1", expected=None, after=None):
        return self.service.event_gc_batch_checkpoint(
            {"consumer_id": consumer, "expected": expected,
             "after": after})


class CheckpointServiceTest(CheckpointMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def test_read_without_record_returns_zero_and_null(self) -> None:
        body, status = self._checkpoint()
        self.assertEqual(status, 200)
        self.assertEqual(list(body), ["consumer_id", "after", "updated_at"])
        self.assertEqual(body, {"consumer_id": "c1", "after": 0,
                                "updated_at": None})

    def test_advance_equal_and_read_back(self) -> None:
        self._commit("r1")
        self._commit("r2")
        body, status = self._checkpoint(expected=0, after=2)
        self.assertEqual(status, 201)
        self.assertEqual(body["after"], 2)
        self.assertIsNotNone(body["updated_at"])
        self.assertTrue(body["updated_at"].endswith("+00:00"))
        # An equal after is an idempotent no-op: 200, timestamp untouched.
        again, status = self._checkpoint(expected=2, after=2)
        self.assertEqual(status, 200)
        self.assertEqual(again, body)
        # A double-null body reads the stored values and writes nothing.
        read, status = self._checkpoint()
        self.assertEqual(status, 200)
        self.assertEqual(read, body)

    def test_advance_refreshes_updated_at(self) -> None:
        self._commit("r1")
        self._commit("r2")
        first, _ = self._checkpoint(expected=0, after=1)
        second, status = self._checkpoint(expected=1, after=2)
        self.assertEqual(status, 201)
        self.assertEqual(second["after"], 2)
        self.assertGreaterEqual(second["updated_at"], first["updated_at"])

    def test_zero_after_with_no_record_is_noop(self) -> None:
        body, status = self._checkpoint(expected=0, after=0)
        self.assertEqual(status, 200)
        self.assertEqual(body["after"], 0)
        self.assertIsNone(body["updated_at"])
        # The no-op created no record: a later read still reports null.
        read, _ = self._checkpoint()
        self.assertIsNone(read["updated_at"])

    def test_expected_mismatch_is_409(self) -> None:
        self._commit("r1")
        with self.assertRaises(ServiceError) as caught:
            self._checkpoint(expected=1, after=1)
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "expected")
        self._checkpoint(expected=0, after=1)
        with self.assertRaises(ServiceError) as caught:
            self._checkpoint(expected=0, after=1)
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "expected")

    def test_after_beyond_audit_records_is_409(self) -> None:
        self._commit("r1")
        with self.assertRaises(ServiceError) as caught:
            self._checkpoint(expected=0, after=2)
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "after")

    def test_after_below_stored_value_is_409(self) -> None:
        self._commit("r1")
        self._commit("r2")
        self._checkpoint(expected=0, after=2)
        with self.assertRaises(ServiceError) as caught:
            self._checkpoint(expected=2, after=1)
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "after")

    def test_checkpoints_are_independent_per_consumer(self) -> None:
        self._commit("r1")
        self._checkpoint(consumer="c1", expected=0, after=1)
        body, status = self._checkpoint(consumer="c2")
        self.assertEqual(status, 200)
        self.assertEqual((body["after"], body["updated_at"]), (0, None))
        with self.assertRaises(ServiceError) as caught:
            self._checkpoint(consumer="c2", expected=1, after=1)
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "expected")

    def test_payload_validation(self) -> None:
        for payload, field in (
                (None, "request_body"),
                ([], "request_body"),
                ("x", "request_body"),
                (42, "request_body"),
                ({}, "consumer_id"),
                ({"consumer_id": "c1"}, "expected"),
                ({"consumer_id": "c1", "expected": None}, "after"),
                ({"consumer_id": "", "expected": None, "after": None},
                 "consumer_id"),
                ({"consumer_id": 1, "expected": None, "after": None},
                 "consumer_id"),
                ({"consumer_id": None, "expected": None, "after": None},
                 "consumer_id"),
                ({"consumer_id": "c1", "expected": "0", "after": 0},
                 "expected"),
                ({"consumer_id": "c1", "expected": True, "after": 0},
                 "expected"),
                ({"consumer_id": "c1", "expected": -1, "after": 0},
                 "expected"),
                ({"consumer_id": "c1", "expected": 2**63, "after": 0},
                 "expected"),
                ({"consumer_id": "c1", "expected": 0.0, "after": 0},
                 "expected"),
                ({"consumer_id": "c1", "expected": 0, "after": "0"},
                 "after"),
                ({"consumer_id": "c1", "expected": 0, "after": False},
                 "after"),
                ({"consumer_id": "c1", "expected": 0, "after": -1},
                 "after"),
                ({"consumer_id": "c1", "expected": 0, "after": 2**63},
                 "after"),
                # exactly one null -> 400/expected
                ({"consumer_id": "c1", "expected": None, "after": 0},
                 "expected"),
                ({"consumer_id": "c1", "expected": 0, "after": None},
                 "expected"),
                ({"consumer_id": "c1", "expected": None, "after": None,
                  "extra": 1}, "extra"),
                ({"consumer_id": "c1", "expected": None, "after": None,
                  "seq": 0}, "seq"),
        ):
            with self.subTest(payload=payload):
                with self.assertRaises(ServiceError) as caught:
                    self.service.event_gc_batch_checkpoint(payload)
                self.assertEqual(caught.exception.status_code, 400)
                self.assertEqual(caught.exception.field, field)
                self.assertEqual(list(caught.exception.to_body()),
                                 ["message", "field"])

    def test_max_integer_after_accepted_when_bounded(self) -> None:
        # The type bound is 0..2**63-1; the audit-chain bound still
        # applies afterwards (no records here, so any advance is 409).
        with self.assertRaises(ServiceError) as caught:
            self._checkpoint(expected=0, after=2**63 - 1)
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "after")


class CheckpointHTTPTest(CheckpointMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()
        self.server, self.service = create_server(
            "127.0.0.1", 0, self.service)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _request(self, path=PATH, raw=None):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        headers = {"Content-Type": "application/json"} if raw is not None \
            else {}
        conn.request("POST", path, body=raw, headers=headers)
        response = conn.getresponse()
        data = response.read().decode("utf-8")
        conn.close()
        return response.status, (json.loads(data) if data else None), data

    def test_read_advance_read_over_http_with_key_order(self) -> None:
        self._commit("r1")
        status, body, raw = self._request(
            raw='{"consumer_id":"c1","expected":null,"after":null}')
        self.assertEqual(status, 200)
        self.assertEqual(body, {"consumer_id": "c1", "after": 0,
                                "updated_at": None})
        status, body, raw = self._request(
            raw='{"consumer_id":"c1","expected":0,"after":1}')
        self.assertEqual(status, 201)
        self.assertEqual(list(body), ["consumer_id", "after", "updated_at"])
        for earlier, later in (
                ('"consumer_id"', '"after"'), ('"after"', '"updated_at"')):
            self.assertLess(raw.index(earlier), raw.index(later))
        status, again, _ = self._request(
            raw='{"consumer_id":"c1","expected":null,"after":null}')
        self.assertEqual(status, 200)
        self.assertEqual(again, body)

    def test_trailing_question_mark_accepted(self) -> None:
        status, _, _ = self._request(
            path=PATH + "?",
            raw='{"consumer_id":"c1","expected":null,"after":null}')
        self.assertEqual(status, 200)

    def test_query_rejected(self) -> None:
        for query in ("?after=0", "?foo", "?x=1", "?expected=0"):
            with self.subTest(query=query):
                status, body, _ = self._request(
                    path=PATH + query,
                    raw='{"consumer_id":"c1","expected":null,"after":null}')
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], "query")
                self.assertEqual(list(body), ["message", "field"])

    def test_bad_json_and_non_object_are_400_request_body(self) -> None:
        for raw in ("{", '"x"', "[1]", "null", ""):
            with self.subTest(raw=raw):
                status, body, _ = self._request(raw=raw)
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], "request_body")
                self.assertEqual(list(body), ["message", "field"])

    def test_field_errors_over_http(self) -> None:
        cases = [
            ('{}', "consumer_id"),
            ('{"consumer_id":"c1"}', "expected"),
            ('{"consumer_id":"c1","expected":null}', "after"),
            ('{"consumer_id":"c1","expected":null,"after":null,"x":1}', "x"),
            ('{"consumer_id":"c1","expected":0,"after":null}', "expected"),
            ('{"consumer_id":"c1","expected":null,"after":0}', "expected"),
            ('{"consumer_id":"c1","expected":true,"after":0}', "expected"),
            ('{"consumer_id":"c1","expected":0,"after":-1}', "after"),
        ]
        for raw, field in cases:
            with self.subTest(raw=raw):
                status, body, _ = self._request(raw=raw)
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], field)
                self.assertEqual(list(body), ["message", "field"])

    def test_conflicts_over_http(self) -> None:
        self._commit("r1")
        status, body, _ = self._request(
            raw='{"consumer_id":"c1","expected":1,"after":1}')
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "expected")
        self.assertEqual(list(body), ["message", "field"])
        status, body, _ = self._request(
            raw='{"consumer_id":"c1","expected":0,"after":2}')
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "after")

    def test_concurrent_same_expected_at_most_one_201(self) -> None:
        self._commit("r1")
        barrier = threading.Barrier(2)
        outcomes = []

        def advance():
            barrier.wait(timeout=5)
            try:
                _, status = self._checkpoint(expected=0, after=1)
                outcomes.append(status)
            except ServiceError as error:
                outcomes.append(error.status_code)

        threads = [threading.Thread(target=advance) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
        self.assertEqual(sorted(outcomes), [201, 409])
        body, status = self._checkpoint()
        self.assertEqual(status, 200)
        self.assertEqual(body["after"], 1)


class CheckpointPersistenceTest(CheckpointMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self._build()
        self.path = os.path.join(self.directory, "state.json")
        self.state_store = attach_persistence(self.service, self.path)

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def _document(self):
        with open(self.path, encoding="utf-8") as handle:
            return json.load(handle)

    def test_section_follows_batch_cleanup_requests(self) -> None:
        self._commit("r1")
        self._checkpoint(expected=0, after=1)
        document = self._document()
        keys = list(document)
        self.assertEqual(keys.index("cleanup_checkpoints"),
                         keys.index("event_gc_batch_cleanup_requests") + 1)
        section = document["cleanup_checkpoints"]
        self.assertEqual(len(section), 1)
        item = section[0]
        self.assertEqual(list(item), ["consumer_id", "after", "updated_at"])
        self.assertEqual(item["consumer_id"], "c1")
        self.assertEqual(item["after"], 1)
        self.assertTrue(item["updated_at"].endswith("+00:00"))

    def test_section_keeps_creation_order(self) -> None:
        self._commit("r1")
        self._checkpoint(consumer="c2", expected=0, after=1)
        self._checkpoint(consumer="c1", expected=0, after=1)
        # A later advance keeps the record's original position.
        self._commit("r2")
        self._checkpoint(consumer="c2", expected=1, after=2)
        document = self._document()
        self.assertEqual(
            [item["consumer_id"] for item in document["cleanup_checkpoints"]],
            ["c2", "c1"])

    def test_advance_consumes_a_generation_reads_do_not(self) -> None:
        self._commit("r1")
        generation = self.state_store.commit_seq
        self._checkpoint()
        self._checkpoint(expected=0, after=1)
        self.assertEqual(self.state_store.commit_seq, generation + 1)
        self._checkpoint(expected=1, after=1)  # equal: no write
        self._checkpoint()                     # read-only
        self.assertEqual(self.state_store.commit_seq, generation + 1)

    def test_restart_restores_checkpoint_and_integrity_passes(self) -> None:
        self._commit("r1")
        self._commit("r2")
        _, status = self._checkpoint(expected=0, after=2)
        self.assertEqual(status, 201)
        before, _ = self._checkpoint()
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        after, status = restarted.event_gc_batch_checkpoint(
            {"consumer_id": "c1", "expected": None, "after": None})
        self.assertEqual(status, 200)
        self.assertEqual(after, before)
        # The stored value still bounds later moves after the restart.
        with self.assertRaises(ServiceError) as caught:
            restarted.event_gc_batch_checkpoint(
                {"consumer_id": "c1", "expected": 2, "after": 1})
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "after")
        self.assertTrue(restarted.persistence_integrity()["consistent"])

    def test_save_failure_rolls_checkpoint_back(self) -> None:
        import e2ee_backend.persistence as persistence_mod

        self._commit("r1")
        real_fsync = persistence_mod.os.fsync
        calls = {"n": 0}

        def fail_first_directory_fsync(fd: int) -> None:  # noqa: ANN001
            if os.fstat(fd).st_mode & 0o170000 == 0o040000:
                calls["n"] += 1
                if calls["n"] == 1:
                    raise OSError("transient directory fsync failure")
            real_fsync(fd)

        generation = self.state_store.commit_seq
        persistence_mod.os.fsync = fail_first_directory_fsync
        try:
            with self.assertRaises(PersistenceUnavailable):
                self._checkpoint(expected=0, after=1)
        finally:
            persistence_mod.os.fsync = real_fsync
        # Rolled back: no generation consumed and no checkpoint recorded.
        self.assertEqual(self.state_store.commit_seq, generation)
        body, status = self._checkpoint()
        self.assertEqual((body["after"], body["updated_at"]), (0, None))
        self.assertEqual(self._document()["cleanup_checkpoints"], [])
        # The advance can be retried and now commits.
        _, status = self._checkpoint(expected=0, after=1)
        self.assertEqual(status, 201)

    def test_legacy_document_without_section_loads(self) -> None:
        self._commit("r1")
        self._checkpoint(expected=0, after=1)
        document = self._document()
        document.pop("cleanup_checkpoints")
        document.pop("integrity_log_version", None)
        legacy_path = os.path.join(self.directory, "legacy.json")
        with open(legacy_path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        restarted = DeviceService()
        attach_persistence(restarted, legacy_path)  # must not raise
        # The dropped checkpoints are simply gone: the consumer reads as 0.
        body, status = restarted.event_gc_batch_checkpoint(
            {"consumer_id": "c1", "expected": None, "after": None})
        self.assertEqual(status, 200)
        self.assertEqual((body["after"], body["updated_at"]), (0, None))

    def _document_with_checkpoint(self, mutate):
        # Seed a fresh deterministic state per call (one audit record, one
        # checkpoint at after=1) so every mutation is rejected by the
        # payload's own semantic validation.
        self._seed_calls = getattr(self, "_seed_calls", 0) + 1
        seed_path = os.path.join(self.directory,
                                 f"seed{self._seed_calls}.json")
        service = DeviceService()
        service.store.add_device(Device("u", "alice", "ik"))
        attach_persistence(service, seed_path)
        service.event_gc_cleanup_expired_batch({
            "mode": "commit", "device_ids": ["alice"],
            "after": 0, "limit": 100, "request_id": "r1"})
        service.event_gc_batch_checkpoint(
            {"consumer_id": "c1", "expected": 0, "after": 1})
        with open(seed_path, encoding="utf-8") as handle:
            document = json.load(handle)
        mutate(document)
        # Write to a fresh path as a marker-less/sidecar-less document, so
        # rejection comes from the payload's own semantic validation rather
        # than the integrity hash gate; the original state file is
        # untouched.
        document.pop("integrity_log_version", None)
        bad_path = os.path.join(self.directory,
                                f"bad{self._seed_calls}.json")
        with open(bad_path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        return bad_path

    def _assert_refuses_startup(self, bad_path) -> None:
        with open(bad_path, "rb") as handle:
            original = handle.read()
        restarted = DeviceService()
        with self.assertRaises(StateFileError):
            attach_persistence(restarted, bad_path)
        # The rejected file is never overwritten.
        with open(bad_path, "rb") as handle:
            self.assertEqual(handle.read(), original)

    def test_restore_rejects_malformed_section(self) -> None:
        def mutate(document):
            document["cleanup_checkpoints"] = {}
        self._assert_refuses_startup(self._document_with_checkpoint(mutate))

    def test_restore_rejects_bad_item_shape(self) -> None:
        good_time = "2026-01-01T00:00:00.000000+00:00"

        def replace(item):
            return lambda d: d.__setitem__("cleanup_checkpoints", [item])

        def append(item):
            return lambda d: d["cleanup_checkpoints"].append(item)

        for mutate in (
                replace("x"),
                # missing updated_at
                replace({"consumer_id": "c2", "after": 1}),
                # extra key
                replace({"consumer_id": "c2", "after": 1,
                         "updated_at": good_time, "x": 1}),
                # wrong key order
                replace({"after": 1, "consumer_id": "c2",
                         "updated_at": good_time}),
                # empty / wrongly typed consumer_id
                replace({"consumer_id": "", "after": 1,
                         "updated_at": good_time}),
                replace({"consumer_id": 1, "after": 1,
                         "updated_at": good_time}),
                # boolean / negative / too-large after
                replace({"consumer_id": "c2", "after": True,
                         "updated_at": good_time}),
                replace({"consumer_id": "c2", "after": -1,
                         "updated_at": good_time}),
                replace({"consumer_id": "c2", "after": 2**63,
                         "updated_at": good_time}),
                # after beyond the single committed audit record
                replace({"consumer_id": "c2", "after": 2,
                         "updated_at": good_time}),
                # empty / null / non-canonical updated_at
                replace({"consumer_id": "c2", "after": 1,
                         "updated_at": ""}),
                replace({"consumer_id": "c2", "after": 1,
                         "updated_at": None}),
                replace({"consumer_id": "c2", "after": 1,
                         "updated_at": "2026-01-01T00:00:00Z"}),
                # duplicate consumer
                append({"consumer_id": "c1", "after": 1,
                        "updated_at": good_time}),
        ):
            with self.subTest(mutate=mutate):
                self._assert_refuses_startup(
                    self._document_with_checkpoint(mutate))


if __name__ == "__main__":
    unittest.main()
