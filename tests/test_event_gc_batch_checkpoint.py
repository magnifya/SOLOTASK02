"""Tests for the batch-cleanup audit consumer checkpoint endpoint.

POST /v1/event-gc-batch/checkpoint reads or advances one consumer's
checkpoint on the batch-cleanup audit chain (the committed
``event_gc_batch_cleanup_requests`` records, in commit order). The call
takes no query parameters (any -> 400/query); the body carries exactly
``consumer_id`` (a non-empty string), ``expected`` and ``after`` — the
latter two both null (read-only query) or both non-boolean integers in
0..2^63-1 (compare-and-advance); exactly one null is 400/expected. A
bad/non-object body is 400/request_body, a missing/wrongly typed/extra
field is 400 with that field. A read-only query answers 200 (a consumer
that never advanced reads as ``after`` 0, ``updated_at`` null). An
``expected`` differing from the current checkpoint (0 when none is
stored) is 409/expected; an ``after`` below the current checkpoint or
beyond the number of committed audit records is 409/after; an equal
``after`` is an idempotent no-op (200, timestamp untouched); a strictly
greater one advances and refreshes ``updated_at`` (201). Success keys
are ``consumer_id``, ``after``, ``updated_at`` in that order.
Checkpoints persist in the ``cleanup_checkpoints`` section (right after
``event_gc_batch_cleanup_requests``), which is part of the integrity
snapshot; older version-1 files without the section load as empty.
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
        self.service.store.add_device(Device("u", "bob", "ik"))

    def _commit(self, request_id, device_ids=None):
        # Every idempotent commit appends exactly one audit record.
        return self.service.event_gc_cleanup_expired_batch({
            "mode": "commit", "device_ids": device_ids or ["bob"],
            "after": 0, "limit": 100, "request_id": request_id})

    def _checkpoint(self, consumer="c1", expected=None, after=None):
        return self.service.event_gc_batch_checkpoint({
            "consumer_id": consumer, "expected": expected, "after": after})


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
        # A null pair reads the stored values and writes nothing.
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

    def test_expected_mismatch_is_409_expected(self) -> None:
        self._commit("r1")
        # No record: the current value is 0.
        with self.assertRaises(ServiceError) as caught:
            self._checkpoint(expected=1, after=1)
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "expected")
        self._checkpoint(expected=0, after=1)
        with self.assertRaises(ServiceError) as caught:
            self._checkpoint(expected=0, after=1)
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "expected")

    def test_after_below_current_is_409_after(self) -> None:
        self._commit("r1")
        self._commit("r2")
        self._checkpoint(expected=0, after=2)
        with self.assertRaises(ServiceError) as caught:
            self._checkpoint(expected=2, after=1)
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "after")

    def test_after_beyond_audit_count_is_409_after(self) -> None:
        # With no audit records at all, any positive after is out of range.
        with self.assertRaises(ServiceError) as caught:
            self._checkpoint(expected=0, after=1)
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "after")
        self._commit("r1")
        with self.assertRaises(ServiceError) as caught:
            self._checkpoint(expected=0, after=2)
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "after")

    def test_checkpoints_are_independent_per_consumer(self) -> None:
        self._commit("r1")
        self._checkpoint(consumer="c1", expected=0, after=1)
        body, status = self._checkpoint(consumer="c2")
        self.assertEqual(status, 200)
        self.assertEqual((body["after"], body["updated_at"]), (0, None))
        body, status = self._checkpoint(consumer="c2", expected=0, after=1)
        self.assertEqual(status, 201)
        body, _ = self._checkpoint(consumer="c1")
        self.assertEqual(body["after"], 1)

    def test_payload_validation(self) -> None:
        for payload, field in (
                (None, "request_body"),
                ([], "request_body"),
                ("x", "request_body"),
                ({"expected": None, "after": None}, "consumer_id"),
                ({"consumer_id": "", "expected": None, "after": None},
                 "consumer_id"),
                ({"consumer_id": 1, "expected": None, "after": None},
                 "consumer_id"),
                ({"consumer_id": None, "expected": None, "after": None},
                 "consumer_id"),
                ({"consumer_id": "c1", "after": None}, "expected"),
                ({"consumer_id": "c1", "expected": None}, "after"),
                # Exactly one of expected/after null is 400/expected.
                ({"consumer_id": "c1", "expected": None, "after": 0},
                 "expected"),
                ({"consumer_id": "c1", "expected": 0, "after": None},
                 "expected"),
                ({"consumer_id": "c1", "expected": True, "after": 0},
                 "expected"),
                ({"consumer_id": "c1", "expected": -1, "after": 0},
                 "expected"),
                ({"consumer_id": "c1", "expected": 1.0, "after": 0},
                 "expected"),
                ({"consumer_id": "c1", "expected": "0", "after": 0},
                 "expected"),
                ({"consumer_id": "c1", "expected": 2**63, "after": 0},
                 "expected"),
                ({"consumer_id": "c1", "expected": 0, "after": True},
                 "after"),
                ({"consumer_id": "c1", "expected": 0, "after": -1},
                 "after"),
                ({"consumer_id": "c1", "expected": 0, "after": 1.0},
                 "after"),
                ({"consumer_id": "c1", "expected": 0, "after": "0"},
                 "after"),
                ({"consumer_id": "c1", "expected": 0, "after": 2**63},
                 "after"),
                ({"consumer_id": "c1", "expected": None, "after": None,
                  "x": 1}, "x"),
                ({"consumer_id": "c1", "expected": None, "after": None,
                  "y": 1, "z": 2}, "y")):
            with self.subTest(payload=payload):
                with self.assertRaises(ServiceError) as caught:
                    self.service.event_gc_batch_checkpoint(payload)
                self.assertEqual(caught.exception.status_code, 400)
                self.assertEqual(caught.exception.field, field)

    def test_boundary_integers_pass_validation(self) -> None:
        self._commit("r1")
        # 2^63-1 is a valid integer shape: it fails only the value checks.
        with self.assertRaises(ServiceError) as caught:
            self._checkpoint(expected=2**63 - 1, after=0)
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "expected")
        with self.assertRaises(ServiceError) as caught:
            self._checkpoint(expected=0, after=2**63 - 1)
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "after")


class CheckpointConcurrencyTest(CheckpointMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()
        self._commit("r1")

    def test_concurrent_same_expected_at_most_one_201(self) -> None:
        results = []
        barrier = threading.Barrier(8)

        def worker() -> None:
            barrier.wait()
            try:
                _body, status = self._checkpoint(expected=0, after=1)
            except ServiceError as error:
                status = error.status_code
            results.append(status)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
        self.assertEqual(results.count(201), 1)
        self.assertEqual(results.count(409), 7)
        body, _ = self._checkpoint()
        self.assertEqual(body["after"], 1)


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

    def _request(self, path=PATH, raw=None, method="POST"):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        headers = {"Content-Type": "application/json"} if raw else {}
        conn.request(method, path, body=raw, headers=headers)
        response = conn.getresponse()
        data = response.read().decode("utf-8")
        conn.close()
        return response.status, (json.loads(data) if data else None), data

    def _checkpoint_http(self, consumer="c1", expected=None, after=None,
                         path=PATH):
        return self._request(
            path, json.dumps({"consumer_id": consumer, "expected": expected,
                              "after": after}))

    def test_read_advance_read_over_http_with_key_order(self) -> None:
        self._commit("r1")
        status, body, _ = self._checkpoint_http()
        self.assertEqual(status, 200)
        self.assertEqual(list(body), ["consumer_id", "after", "updated_at"])
        self.assertEqual(body, {"consumer_id": "c1", "after": 0,
                                "updated_at": None})
        status, body, raw = self._checkpoint_http(expected=0, after=1)
        self.assertEqual(status, 201)
        self.assertEqual(body["after"], 1)
        self.assertLess(raw.index('"consumer_id"'), raw.index('"after"'))
        self.assertLess(raw.index('"after"'), raw.index('"updated_at"'))
        status, body, _ = self._checkpoint_http()
        self.assertEqual(status, 200)
        self.assertEqual(body["after"], 1)

    def test_query_rejected(self) -> None:
        for query in ("?after=0", "?foo", "?x=1"):
            with self.subTest(query=query):
                status, body, _ = self._checkpoint_http(path=PATH + query)
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], "query")
                self.assertEqual(list(body), ["message", "field"])
        # A bare trailing '?' carries no parameter and is accepted.
        status, _, _ = self._checkpoint_http(path=PATH + "?")
        self.assertEqual(status, 200)

    def test_bad_json_and_non_object_are_400_request_body(self) -> None:
        for raw in ("{", "[1]", "null", '"s"'):
            with self.subTest(raw=raw):
                status, body, _ = self._request(raw=raw)
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], "request_body")
                self.assertEqual(list(body), ["message", "field"])

    def test_field_errors_over_http(self) -> None:
        for payload, field in (
                ({"expected": None, "after": None}, "consumer_id"),
                ({"consumer_id": "", "expected": None, "after": None},
                 "consumer_id"),
                ({"consumer_id": "c1", "after": None}, "expected"),
                ({"consumer_id": "c1", "expected": None}, "after"),
                ({"consumer_id": "c1", "expected": None, "after": 0},
                 "expected"),
                ({"consumer_id": "c1", "expected": 0, "after": True},
                 "after"),
                ({"consumer_id": "c1", "expected": None, "after": None,
                  "extra": 1}, "extra")):
            with self.subTest(payload=payload):
                status, body, _ = self._request(raw=json.dumps(payload))
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], field)
                self.assertEqual(list(body), ["message", "field"])

    def test_conflicts_over_http(self) -> None:
        self._commit("r1")
        status, body, _ = self._checkpoint_http(expected=1, after=1)
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "expected")
        self.assertEqual(list(body), ["message", "field"])
        status, body, _ = self._checkpoint_http(expected=0, after=2)
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "after")
        self._checkpoint_http(expected=0, after=1)
        status, body, _ = self._checkpoint_http(expected=1, after=0)
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "after")


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

    def test_section_follows_cleanup_requests_in_creation_order(self) -> None:
        self._commit("r1")
        self._commit("r2")
        self._checkpoint(consumer="c2", expected=0, after=1)
        self._checkpoint(consumer="c1", expected=0, after=2)
        document = self._document()
        keys = list(document)
        self.assertEqual(keys.index("cleanup_checkpoints"),
                         keys.index("event_gc_batch_cleanup_requests") + 1)
        section = document["cleanup_checkpoints"]
        self.assertEqual([item["consumer_id"] for item in section],
                         ["c2", "c1"])
        self.assertEqual([item["after"] for item in section], [1, 2])
        for item in section:
            self.assertEqual(list(item),
                             ["consumer_id", "after", "updated_at"])
            self.assertTrue(item["updated_at"].endswith("+00:00"))

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
        # Rolled back: no generation consumed and no checkpoint recorded,
        # in memory or on disk.
        self.assertEqual(self.state_store.commit_seq, generation)
        body, status = self._checkpoint()
        self.assertEqual((body["after"], body["updated_at"]), (0, None))
        self.assertEqual(self._document()["cleanup_checkpoints"], [])
        # The advance can be retried and now commits.
        _, status = self._checkpoint(expected=0, after=1)
        self.assertEqual(status, 201)
        self.assertEqual(self.state_store.commit_seq, generation + 1)

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
        # A fresh service per call: one committed audit record and one
        # checkpoint at after=1, so every mutation starts from the same
        # document regardless of subTest order.
        service = DeviceService()
        service.store.add_device(Device("u", "bob", "ik"))
        self._seed_n = getattr(self, "_seed_n", 0) + 1
        seed_path = os.path.join(self.directory,
                                 f"seed{self._seed_n}.json")
        attach_persistence(service, seed_path)
        service.event_gc_cleanup_expired_batch({
            "mode": "commit", "device_ids": ["bob"], "after": 0,
            "limit": 100, "request_id": "r1"})
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
        bad_path = os.path.join(self.directory, f"bad{self._seed_n}.json")
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
        stamp = "2026-01-01T00:00:00.000000+00:00"

        def append(item):
            return lambda d: d["cleanup_checkpoints"].append(item)
        for mutate in (
                append("x"),
                # missing updated_at
                append({"consumer_id": "c2", "after": 1}),
                # extra key
                append({"consumer_id": "c2", "after": 1,
                        "updated_at": stamp, "x": 1}),
                # wrong key order
                append({"after": 1, "consumer_id": "c2",
                        "updated_at": stamp}),
                append({"consumer_id": "", "after": 1,
                        "updated_at": stamp}),
                append({"consumer_id": "c2", "after": True,
                        "updated_at": stamp}),
                append({"consumer_id": "c2", "after": -1,
                        "updated_at": stamp}),
                append({"consumer_id": "c2", "after": 2**63,
                        "updated_at": stamp}),
                # after beyond the audit record count (1)
                append({"consumer_id": "c2", "after": 2,
                        "updated_at": stamp}),
                # updated_at empty / not a string
                append({"consumer_id": "c2", "after": 1,
                        "updated_at": None}),
                append({"consumer_id": "c2", "after": 1,
                        "updated_at": ""}),
                # updated_at not the canonical UTC microsecond form
                append({"consumer_id": "c2", "after": 1,
                        "updated_at": "2026-01-01T00:00:00Z"})):
            with self.subTest(mutate=mutate):
                self._assert_refuses_startup(
                    self._document_with_checkpoint(mutate))

    def test_restore_rejects_duplicate_consumer(self) -> None:
        def mutate(document):
            document["cleanup_checkpoints"].append(
                dict(document["cleanup_checkpoints"][0]))
        self._assert_refuses_startup(self._document_with_checkpoint(mutate))


if __name__ == "__main__":
    unittest.main()
