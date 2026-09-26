"""Tests for the per-consumer redelivery-job event checkpoint endpoint.

``POST /v1/devices/{device_id}/inbox-job-events/checkpoint`` stores, per
``(device_id, consumer_id)`` pair, the greatest job-event ``seq`` the
consumer has acknowledged. The body carries exactly ``consumer_id`` (a
non-empty string) and ``seq`` (``null`` for a read-only query or a
non-boolean non-negative integer): bad JSON / a non-object body is
400/request_body, a missing/mistyped field or an extra key is 400 with
that key. An unknown device is 404/device_id; a revoked device keeps
checkpointing. ``seq=null`` only reads (200): no record returns 0 with a
null timestamp, otherwise the saved values. An integer past the device's
last event seq, or below the consumer's stored checkpoint, is 409/seq;
an equal seq is a 200 no-op that writes nothing, a greater one advances
the record (201) with a refreshed timestamp. Success keys are
``device_id``, ``consumer_id``, ``seq`` and ``updated_at`` in that order;
the timestamp is null or canonical UTC ISO-8601 (six microsecond digits,
``+00:00``). Pairs are independent per (device_id, consumer_id); a
checkpoint advance shares the store lock with event appends and a
durable-write failure is 503/data_file with the in-memory state,
commit_seq and both files rolled back. The persisted section
``redelivery_job_event_checkpoints`` follows ``redelivery_job_events``;
its items carry ``device_id``/``consumer_id``/``seq``/``updated_at`` in
order and startup rejects a duplicate pair, unknown device, out-of-range
seq, or any field/type/key-order error without overwriting the file.
"""
import json
import os
import shutil
import tempfile
import threading
import unittest
from http.client import HTTPConnection

from e2ee_backend.models import Device, SignedPreKey
from e2ee_backend.service import DeviceService, ServiceError
from e2ee_backend.http_app import create_server
from e2ee_backend.persistence import (
    PersistenceUnavailable,
    StateFileError,
    attach_persistence,
)


class CheckpointMixin:
    def _build(self) -> None:
        self.service = DeviceService()
        self.service.store.add_device(Device("u", "alice", "ik"))
        self.service.store.add_device(Device(
            "u", "bob", "ik",
            prekeys=[SignedPreKey("pk1", "pubk1")]))
        self.sid = self.service.store.create_session(
            "alice", "bob", "pk1", "ek1").session_id

    def _post_message(self, message_id="m1", sequence=1) -> None:
        self.service.post_message({
            "session_id": self.sid, "sender_device_id": "alice",
            "message_id": message_id, "sequence": sequence,
            "nonce": f"n{message_id}", "ciphertext": "ct"})

    def _op(self, job_id, op, device="bob", **extra):
        payload = {"device_id": device, "job_id": job_id, "op": op}
        payload.update(extra)
        return self.service.inbox_job(payload)

    def _checkpoint(self, consumer="c1", seq=None, device="bob"):
        return self.service.inbox_job_event_checkpoint(
            device, {"consumer_id": consumer, "seq": seq})


class CheckpointServiceTest(CheckpointMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def test_null_read_with_no_record(self) -> None:
        body, status = self._checkpoint()
        self.assertEqual(status, 200)
        self.assertEqual(list(body),
                         ["device_id", "consumer_id", "seq", "updated_at"])
        self.assertEqual(body, {"device_id": "bob", "consumer_id": "c1",
                                "seq": 0, "updated_at": None})

    def test_advance_equal_and_null_reread(self) -> None:
        self._op("J1", "queue")
        body, status = self._checkpoint(seq=1)
        self.assertEqual(status, 201)
        self.assertEqual(body["seq"], 1)
        timestamp = body["updated_at"]
        self.assertIsNotNone(timestamp)
        # The stored timestamp is canonical UTC ISO-8601 with six digits.
        self.assertTrue(timestamp.endswith("+00:00"))
        self.assertEqual(timestamp[19:27][0], ".")
        # An equal seq is a 200 no-op that leaves the timestamp untouched.
        same, status = self._checkpoint(seq=1)
        self.assertEqual(status, 200)
        self.assertEqual(same, body)
        # A null read reports exactly the saved values.
        read, status = self._checkpoint()
        self.assertEqual(status, 200)
        self.assertEqual(read, body)

    def test_equal_zero_creates_no_record(self) -> None:
        # No events yet: seq 0 equals the implicit zero checkpoint, so the
        # call is a 200 no-op and no record (hence still a null timestamp).
        body, status = self._checkpoint(seq=0)
        self.assertEqual(status, 200)
        self.assertEqual(body["seq"], 0)
        self.assertIsNone(body["updated_at"])
        with self.service.store._lock:
            self.assertEqual(
                self.service.store._redelivery_job_event_checkpoints, {})
        # A null read still reports the empty checkpoint.
        read, _ = self._checkpoint()
        self.assertEqual(read, body)

    def test_forward_move_refreshes_timestamp(self) -> None:
        self._op("J1", "queue")
        first, status = self._checkpoint(seq=1)
        self.assertEqual(status, 201)
        self._op("J1", "cancel", cancellation_id="X1")
        second, status = self._checkpoint(seq=2)
        self.assertEqual(status, 201)
        self.assertEqual(second["seq"], 2)
        self.assertGreaterEqual(second["updated_at"], first["updated_at"])

    def test_seq_past_last_event_conflicts(self) -> None:
        # An empty chain ends at 0; seq 1 already conflicts.
        with self.assertRaises(ServiceError) as caught:
            self._checkpoint(seq=1)
        self.assertEqual((caught.exception.status_code,
                          caught.exception.field), (409, "seq"))
        # One event committed: seq 2 is now past the end.
        self._op("J1", "queue")
        self._checkpoint(seq=1)
        with self.assertRaises(ServiceError) as caught:
            self._checkpoint(seq=2)
        self.assertEqual((caught.exception.status_code,
                          caught.exception.field), (409, "seq"))

    def test_backward_move_conflicts(self) -> None:
        self._op("J1", "queue")
        self._checkpoint(seq=1)
        with self.assertRaises(ServiceError) as caught:
            self._checkpoint(seq=0)
        self.assertEqual((caught.exception.status_code,
                          caught.exception.field), (409, "seq"))

    def test_pairs_are_independent(self) -> None:
        self._op("J1", "queue")
        first, _ = self._checkpoint(consumer="c1", seq=1)
        # A different consumer on the same device starts at zero.
        other, status = self._checkpoint(consumer="c2")
        self.assertEqual(status, 200)
        self.assertEqual(other["seq"], 0)
        self.assertIsNone(other["updated_at"])
        # The same consumer id on a different device is a separate pair.
        third, status = self._checkpoint(consumer="c1", seq=0,
                                         device="alice")
        self.assertEqual(status, 200)
        self.assertIsNone(third["updated_at"])
        # Advancing one leaves the others untouched.
        self._checkpoint(consumer="c2", seq=1)
        read, _ = self._checkpoint(consumer="c1")
        self.assertEqual(read, first)

    def test_unknown_device_is_404(self) -> None:
        for seq in (None, 0, 1):
            with self.subTest(seq=seq):
                with self.assertRaises(ServiceError) as caught:
                    self._checkpoint(device="ghost", seq=seq)
                self.assertEqual((caught.exception.status_code,
                                  caught.exception.field),
                                 (404, "device_id"))

    def test_revoked_device_keeps_checkpointing(self) -> None:
        self._op("J1", "queue")
        self.service.revoke_device("bob")
        body, status = self._checkpoint(seq=1)
        self.assertEqual(status, 201)
        read, status = self._checkpoint()
        self.assertEqual(status, 200)
        self.assertEqual(read, body)

    def test_body_validation(self) -> None:
        cases = [
            (None, "request_body"),
            ([], "request_body"),
            ("x", "request_body"),
            (42, "request_body"),
            ({}, "consumer_id"),
            ({"seq": 1}, "consumer_id"),
            ({"consumer_id": "", "seq": 1}, "consumer_id"),
            ({"consumer_id": 7, "seq": 1}, "consumer_id"),
            ({"consumer_id": None, "seq": 1}, "consumer_id"),
            ({"consumer_id": "c1"}, "seq"),
            ({"consumer_id": "c1", "seq": -1}, "seq"),
            ({"consumer_id": "c1", "seq": True}, "seq"),
            ({"consumer_id": "c1", "seq": False}, "seq"),
            ({"consumer_id": "c1", "seq": 1.0}, "seq"),
            ({"consumer_id": "c1", "seq": "1"}, "seq"),
            ({"consumer_id": "c1", "seq": [1]}, "seq"),
            ({"consumer_id": "c1", "seq": 1, "extra": 2}, "extra"),
            ({"x": 1, "consumer_id": "c1", "seq": 1}, "x"),
        ]
        for payload, field in cases:
            with self.subTest(payload=payload):
                with self.assertRaises(ServiceError) as caught:
                    self.service.inbox_job_event_checkpoint("bob", payload)
                self.assertEqual(caught.exception.status_code, 400)
                self.assertEqual(caught.exception.field, field)


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

    def _request(self, device="bob", raw=None, method="POST"):
        path = f"/v1/devices/{device}/inbox-job-events/checkpoint"
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        headers = {"Content-Type": "application/json"} if raw else {}
        conn.request(method, path, body=raw, headers=headers)
        response = conn.getresponse()
        data = response.read().decode("utf-8")
        conn.close()
        return response.status, (json.loads(data) if data else None), data

    def _queue_http(self, job_id):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("POST", "/v1/inbox-jobs",
                     json.dumps({"device_id": "bob", "job_id": job_id,
                                 "op": "queue"}),
                     headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        self.assertEqual(response.status, 201)
        response.read()
        conn.close()

    def test_null_read_and_advance_key_order(self) -> None:
        status, body, raw = self._request(
            raw=json.dumps({"consumer_id": "c1", "seq": None}))
        self.assertEqual(status, 200)
        self.assertEqual(body, {"device_id": "bob", "consumer_id": "c1",
                                "seq": 0, "updated_at": None})
        self.assertLess(raw.index('"device_id"'),
                        raw.index('"consumer_id"'))
        self.assertLess(raw.index('"consumer_id"'), raw.index('"seq"'))
        self.assertLess(raw.index('"seq"'), raw.index('"updated_at"'))
        self._queue_http("J1")
        status, body, _ = self._request(
            raw=json.dumps({"consumer_id": "c1", "seq": 1}))
        self.assertEqual(status, 201)
        self.assertEqual(list(body),
                         ["device_id", "consumer_id", "seq", "updated_at"])
        # Equal seq is 200.
        status, _, _ = self._request(
            raw=json.dumps({"consumer_id": "c1", "seq": 1}))
        self.assertEqual(status, 200)

    def test_bad_json_and_non_object_are_400_request_body(self) -> None:
        for raw in ("{", "[1, 2]", '"x"', "42"):
            with self.subTest(raw=raw):
                status, body, _ = self._request(raw=raw)
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], "request_body")

    def test_empty_body_is_400_request_body(self) -> None:
        status, body, _ = self._request()
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "request_body")

    def test_field_errors(self) -> None:
        cases = [
            ({"seq": 1}, "consumer_id"),
            ({"consumer_id": "", "seq": 1}, "consumer_id"),
            ({"consumer_id": "c1"}, "seq"),
            ({"consumer_id": "c1", "seq": -1}, "seq"),
            ({"consumer_id": "c1", "seq": True}, "seq"),
            ({"consumer_id": "c1", "seq": 1, "x": 2}, "x"),
        ]
        for payload, field in cases:
            with self.subTest(payload=payload):
                status, body, _ = self._request(raw=json.dumps(payload))
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], field)

    def test_unknown_device_is_404(self) -> None:
        status, body, _ = self._request(
            device="ghost",
            raw=json.dumps({"consumer_id": "c1", "seq": None}))
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "device_id")

    def test_out_of_range_is_409_seq(self) -> None:
        status, body, _ = self._request(
            raw=json.dumps({"consumer_id": "c1", "seq": 5}))
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "seq")

    def test_revoked_device_still_checkpoints(self) -> None:
        self._queue_http("J1")
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("POST", "/v1/devices/bob/revoke",
                     json.dumps({}),
                     headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        self.assertEqual(response.status, 200)
        response.read()
        conn.close()
        status, _, _ = self._request(
            raw=json.dumps({"consumer_id": "c1", "seq": 1}))
        self.assertEqual(status, 201)


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

    def _advance(self, consumer="c1", seq=1, device="bob"):
        body, status = self._checkpoint(consumer=consumer, seq=seq,
                                        device=device)
        self.assertEqual(status, 201)
        return body

    def test_section_follows_events_with_prefixed_items(self) -> None:
        self._op("J1", "queue")
        body = self._advance()
        document = self._document()
        keys = list(document)
        self.assertEqual(
            keys.index("redelivery_job_event_checkpoints"),
            keys.index("redelivery_job_events") + 1)
        self.assertEqual(document["redelivery_job_event_checkpoints"], [
            {"device_id": "bob", "consumer_id": "c1", "seq": 1,
             "updated_at": body["updated_at"]}])
        # Compact, UTF-8, no newline; a non-ASCII consumer id is written
        # literally (ensure_ascii=False).
        self._advance(consumer="消费c")
        raw = open(self.path, "rb").read()
        self.assertNotIn(b"\n", raw)
        self.assertIn("消费c".encode("utf-8"), raw)
        self.assertNotIn(b"\\u", raw)

    def test_restart_restores_checkpoint_and_integrity_passes(self) -> None:
        self._op("J1", "queue")
        self._advance()
        before, _ = self._checkpoint()
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        read, status = restarted.inbox_job_event_checkpoint(
            "bob", {"consumer_id": "c1", "seq": None})
        self.assertEqual(status, 200)
        self.assertEqual(read, before)
        # The restored checkpoint still enforces the range rules.
        with self.assertRaises(ServiceError) as caught:
            restarted.inbox_job_event_checkpoint(
                "bob", {"consumer_id": "c1", "seq": 0})
        self.assertEqual(caught.exception.field, "seq")
        self.assertTrue(restarted.persistence_integrity()["consistent"])

    def test_null_read_and_equal_consume_no_generation(self) -> None:
        self._op("J1", "queue")
        self._advance(seq=1)
        generation = self.state_store.commit_seq
        self._checkpoint(seq=None)
        self._checkpoint(seq=1)
        # A second consumer's equal-zero no-op also writes nothing.
        self._checkpoint(consumer="c2", seq=0)
        self.assertEqual(self.state_store.commit_seq, generation)
        document = self._document()
        self.assertEqual(len(document["redelivery_job_event_checkpoints"]),
                         1)

    def test_save_failure_rolls_checkpoint_back(self) -> None:
        import e2ee_backend.persistence as persistence_mod

        real_fsync = persistence_mod.os.fsync
        calls = {"n": 0}

        def fail_first_directory_fsync(fd: int) -> None:  # noqa: ANN001
            if os.fstat(fd).st_mode & 0o170000 == 0o040000:
                calls["n"] += 1
                if calls["n"] == 1:
                    raise OSError("transient directory fsync failure")
            real_fsync(fd)

        self._op("J1", "queue")
        generation = self.state_store.commit_seq
        persistence_mod.os.fsync = fail_first_directory_fsync
        try:
            with self.assertRaises(PersistenceUnavailable):
                self._checkpoint(seq=1)
        finally:
            persistence_mod.os.fsync = real_fsync
        # Rolled back: no generation consumed and no checkpoint stored.
        self.assertEqual(self.state_store.commit_seq, generation)
        read, status = self._checkpoint()
        self.assertEqual(status, 200)
        self.assertEqual(read["seq"], 0)
        self.assertIsNone(read["updated_at"])
        # The advance can be retried and now commits.
        body = self._advance(seq=1)
        self.assertEqual(body["seq"], 1)
        self.assertEqual(self.state_store.commit_seq, generation + 1)

    def test_legacy_document_without_section_loads(self) -> None:
        self._op("J1", "queue")
        self._advance(seq=1)
        document = self._document()
        document.pop("redelivery_job_event_checkpoints")
        document.pop("integrity_log_version", None)
        legacy_path = os.path.join(self.directory, "legacy.json")
        with open(legacy_path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        restarted = DeviceService()
        attach_persistence(restarted, legacy_path)  # must not raise
        # The missing section reads as empty and a fresh advance commits.
        read, status = restarted.inbox_job_event_checkpoint(
            "bob", {"consumer_id": "c1", "seq": None})
        self.assertEqual(status, 200)
        self.assertEqual(read["seq"], 0)
        self.assertIsNone(read["updated_at"])
        body, status = restarted.inbox_job_event_checkpoint(
            "bob", {"consumer_id": "c1", "seq": 1})
        self.assertEqual(status, 201)
        self.assertEqual(body["seq"], 1)

    def _document_with_checkpoint(self, mutate):
        # One queued job gives the device one event, so a seq-1 checkpoint
        # is in range. Repeated invocations replay the queue/no-op the
        # checkpoint idempotently (the live store accumulates across
        # subtests); the document on disk always carries the one c1 record.
        self._op("J1", "queue")
        _, status = self._checkpoint(seq=1)
        self.assertIn(status, (200, 201))
        document = self._document()
        mutate(document)
        # Write to a fresh path as a marker-less/sidecar-less document, so
        # rejection comes from the payload's own semantic validation rather
        # than the integrity hash gate; the original state file is
        # untouched.
        document.pop("integrity_log_version", None)
        bad_path = os.path.join(self.directory, "bad.json")
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
            document["redelivery_job_event_checkpoints"] = {}
        self._assert_refuses_startup(
            self._document_with_checkpoint(mutate))

    def test_restore_rejects_bad_item_shape(self) -> None:
        valid = {"device_id": "bob", "consumer_id": "nx", "seq": 1,
                 "updated_at": "2026-01-01T00:00:00.000000+00:00"}

        def replace(item):
            def mutate(document):
                document["redelivery_job_event_checkpoints"][0] = item
            return mutate

        def append(item):
            return lambda d: d["redelivery_job_event_checkpoints"].append(
                item)

        for mutate in (
                append("x"),
                append({"consumer_id": "nx", "seq": 1,
                        "updated_at": valid["updated_at"]}),
                append({"device_id": "bob", "consumer_id": "nx",
                        "seq": 1}),
                append({"device_id": "bob", "consumer_id": "nx",
                        "seq": 1,
                        "updated_at": valid["updated_at"], "x": 1}),
                append({"device_id": "", "consumer_id": "nx", "seq": 1,
                        "updated_at": valid["updated_at"]}),
                append({"device_id": 7, "consumer_id": "nx", "seq": 1,
                        "updated_at": valid["updated_at"]}),
                append({"device_id": "bob", "consumer_id": "", "seq": 1,
                        "updated_at": valid["updated_at"]}),
                append({"device_id": "bob", "consumer_id": None,
                        "seq": 1, "updated_at": valid["updated_at"]}),
                append({"device_id": "bob", "consumer_id": "nx",
                        "seq": -1, "updated_at": valid["updated_at"]}),
                append({"device_id": "bob", "consumer_id": "nx",
                        "seq": True, "updated_at": valid["updated_at"]}),
                append({"device_id": "bob", "consumer_id": "nx",
                        "seq": 1.0, "updated_at": valid["updated_at"]}),
                append({"device_id": "bob", "consumer_id": "nx", "seq": 1,
                        "updated_at": "2026-01-01T00:00:00Z"}),
                append({"device_id": "bob", "consumer_id": "nx", "seq": 1,
                        "updated_at": "2026-01-01T00:00:00+00:00"}),
                append({"device_id": "bob", "consumer_id": "nx", "seq": 1,
                        "updated_at": "not-a-timestamp"}),
                append({"device_id": "ghost", "consumer_id": "nx",
                        "seq": 1, "updated_at": valid["updated_at"]}),
                append({"device_id": "bob", "consumer_id": "nx", "seq": 2,
                        "updated_at": valid["updated_at"]}),
                # Keys present but in the wrong order.
                replace({"consumer_id": "nx", "device_id": "bob",
                         "seq": 1, "updated_at": valid["updated_at"]})):
            with self.subTest(mutate=mutate):
                self._assert_refuses_startup(
                    self._document_with_checkpoint(mutate))

    def test_restore_rejects_duplicate_pair(self) -> None:
        def mutate(document):
            document["redelivery_job_event_checkpoints"].append(
                {"device_id": "bob", "consumer_id": "c1", "seq": 1,
                 "updated_at": "2026-01-02T00:00:00.000000+00:00"})
        self._assert_refuses_startup(
            self._document_with_checkpoint(mutate))

    def test_restore_accepts_distinct_consumers(self) -> None:
        def mutate(document):
            document["redelivery_job_event_checkpoints"].append(
                {"device_id": "bob", "consumer_id": "c2", "seq": 0,
                 "updated_at": "2026-01-02T00:00:00.000000+00:00"})
        bad_path = self._document_with_checkpoint(mutate)
        restarted = DeviceService()
        attach_persistence(restarted, bad_path)  # must not raise
        read, status = restarted.inbox_job_event_checkpoint(
            "bob", {"consumer_id": "c2", "seq": None})
        self.assertEqual(status, 200)
        self.assertEqual(read["seq"], 0)


if __name__ == "__main__":
    unittest.main()
