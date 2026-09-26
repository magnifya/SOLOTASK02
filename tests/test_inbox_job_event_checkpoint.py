"""Tests for the 1:1 inbox redelivery-job event checkpoint endpoint.

POST /v1/devices/{device_id}/inbox-job-events/checkpoint reads or advances
one consumer's checkpoint on the device's redelivery-job lifecycle event
chain. The body carries exactly ``consumer_id`` (a non-empty string) and
``seq`` (null or a non-negative, non-boolean integer); a bad/non-object
body is 400/request_body, a missing/empty/wrongly typed field is 400 with
that field, as is the first extra key. An unknown device is
404/device_id; a revoked device stays checkpointable. ``seq=null`` is a
read-only query (200): a pair that never checkpointed reports ``seq`` 0
and ``updated_at`` null, otherwise the stored values. An integer ``seq``
above the device's last event seq or below this consumer's stored value
is 409/seq; an equal seq is an idempotent no-op (200, timestamp
untouched); a strictly greater one advances and refreshes ``updated_at``
(201). Checkpoints are independent per (device_id, consumer_id) and
persisted in the ``redelivery_job_event_checkpoints`` section (right
after ``redelivery_job_events``), which is part of the integrity
snapshot; older version-1 files without the section load as empty.
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

    def _op(self, job_id, op, device="bob", **extra):
        payload = {"device_id": device, "job_id": job_id, "op": op}
        payload.update(extra)
        return self.service.inbox_job(payload)

    def _checkpoint(self, device="bob", consumer="c1", seq=None):
        return self.service.inbox_job_event_checkpoint(
            device, {"consumer_id": consumer, "seq": seq})


class CheckpointServiceTest(CheckpointMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def test_read_without_record_returns_zero_and_null(self) -> None:
        body, status = self._checkpoint()
        self.assertEqual(status, 200)
        self.assertEqual(list(body),
                         ["device_id", "consumer_id", "seq", "updated_at"])
        self.assertEqual(body, {"device_id": "bob", "consumer_id": "c1",
                                "seq": 0, "updated_at": None})

    def test_advance_equal_and_read_back(self) -> None:
        self._op("J1", "queue")
        self._op("J2", "queue")
        body, status = self._checkpoint(seq=2)
        self.assertEqual(status, 201)
        self.assertEqual(body["seq"], 2)
        self.assertIsNotNone(body["updated_at"])
        self.assertTrue(body["updated_at"].endswith("+00:00"))
        # An equal seq is an idempotent no-op: 200, timestamp untouched.
        again, status = self._checkpoint(seq=2)
        self.assertEqual(status, 200)
        self.assertEqual(again, body)
        # A null seq reads the stored values and writes nothing.
        read, status = self._checkpoint()
        self.assertEqual(status, 200)
        self.assertEqual(read, body)

    def test_advance_refreshes_updated_at(self) -> None:
        self._op("J1", "queue")
        self._op("J2", "queue")
        first, _ = self._checkpoint(seq=1)
        second, status = self._checkpoint(seq=2)
        self.assertEqual(status, 201)
        self.assertEqual(second["seq"], 2)
        self.assertGreaterEqual(second["updated_at"], first["updated_at"])

    def test_zero_seq_with_no_record_is_noop(self) -> None:
        self._op("J1", "queue")
        body, status = self._checkpoint(seq=0)
        self.assertEqual(status, 200)
        self.assertEqual(body["seq"], 0)
        self.assertIsNone(body["updated_at"])

    def test_seq_beyond_last_event_is_409(self) -> None:
        self._op("J1", "queue")
        with self.assertRaises(ServiceError) as caught:
            self._checkpoint(seq=2)
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "seq")

    def test_seq_below_stored_value_is_409(self) -> None:
        self._op("J1", "queue")
        self._op("J2", "queue")
        self._checkpoint(seq=2)
        with self.assertRaises(ServiceError) as caught:
            self._checkpoint(seq=1)
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "seq")

    def test_checkpoints_are_independent_per_pair(self) -> None:
        self._op("J1", "queue")
        self._op("J2", "queue")
        self._checkpoint(consumer="c1", seq=2)
        # Another consumer of the same device starts at 0.
        body, status = self._checkpoint(consumer="c2")
        self.assertEqual((body["seq"], body["updated_at"]), (0, None))
        # Another device with the same consumer id starts at 0 too.
        body, status = self._checkpoint(device="alice", consumer="c1")
        self.assertEqual((body["seq"], body["updated_at"]), (0, None))
        # And c2 can only advance to the device's last event seq.
        with self.assertRaises(ServiceError) as caught:
            self._checkpoint(device="alice", consumer="c1", seq=1)
        self.assertEqual(caught.exception.status_code, 409)

    def test_unknown_device_is_404_device_id(self) -> None:
        with self.assertRaises(ServiceError) as caught:
            self._checkpoint(device="ghost")
        self.assertEqual(caught.exception.status_code, 404)
        self.assertEqual(caught.exception.field, "device_id")

    def test_revoked_device_stays_checkpointable(self) -> None:
        self._op("J1", "queue")
        self.service.revoke_device("bob")
        body, status = self._checkpoint(seq=1)
        self.assertEqual(status, 201)
        self.assertEqual(body["seq"], 1)

    def test_payload_validation(self) -> None:
        for payload, field in (
                (None, "request_body"),
                ([], "request_body"),
                ("x", "request_body"),
                ({"seq": 1}, "consumer_id"),
                ({"consumer_id": "", "seq": 1}, "consumer_id"),
                ({"consumer_id": 1, "seq": 1}, "consumer_id"),
                ({"consumer_id": None, "seq": 1}, "consumer_id"),
                ({"consumer_id": "c1"}, "seq"),
                ({"consumer_id": "c1", "seq": True}, "seq"),
                ({"consumer_id": "c1", "seq": -1}, "seq"),
                ({"consumer_id": "c1", "seq": 1.0}, "seq"),
                ({"consumer_id": "c1", "seq": "1"}, "seq"),
                ({"consumer_id": "c1", "seq": 0, "x": 1}, "x"),
                ({"consumer_id": "c1", "seq": 0, "y": 1, "z": 2}, "y")):
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

    def _request(self, path, raw=None, method="POST"):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        headers = {"Content-Type": "application/json"} if raw else {}
        conn.request(method, path, body=raw, headers=headers)
        response = conn.getresponse()
        data = response.read().decode("utf-8")
        conn.close()
        return response.status, (json.loads(data) if data else None), data

    def _checkpoint_http(self, device="bob", consumer="c1", seq=None):
        return self._request(
            f"/v1/devices/{device}/inbox-job-events/checkpoint",
            json.dumps({"consumer_id": consumer, "seq": seq}))

    def _queue_http(self, job_id):
        status, _, _ = self._request(
            "/v1/inbox-jobs",
            json.dumps({"device_id": "bob", "job_id": job_id,
                        "op": "queue"}))
        self.assertEqual(status, 201)

    def test_read_advance_read_over_http_with_key_order(self) -> None:
        self._queue_http("J1")
        status, body, raw = self._checkpoint_http()
        self.assertEqual(status, 200)
        self.assertEqual(list(body),
                         ["device_id", "consumer_id", "seq", "updated_at"])
        self.assertEqual(body, {"device_id": "bob", "consumer_id": "c1",
                                "seq": 0, "updated_at": None})
        status, body, raw = self._checkpoint_http(seq=1)
        self.assertEqual(status, 201)
        self.assertEqual(body["seq"], 1)
        self.assertLess(raw.index('"device_id"'),
                        raw.index('"consumer_id"'))
        self.assertLess(raw.index('"consumer_id"'), raw.index('"seq"'))
        self.assertLess(raw.index('"seq"'), raw.index('"updated_at"'))
        status, body, _ = self._checkpoint_http()
        self.assertEqual(status, 200)
        self.assertEqual(body["seq"], 1)

    def test_bad_json_and_non_object_are_400_request_body(self) -> None:
        for raw in ("{", "[1]", "null", '"s"'):
            with self.subTest(raw=raw):
                status, body, _ = self._request(
                    "/v1/devices/bob/inbox-job-events/checkpoint", raw=raw)
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], "request_body")
                self.assertEqual(list(body), ["message", "field"])

    def test_field_errors_over_http(self) -> None:
        for payload, field in (
                ({"seq": 1}, "consumer_id"),
                ({"consumer_id": "", "seq": 1}, "consumer_id"),
                ({"consumer_id": "c1"}, "seq"),
                ({"consumer_id": "c1", "seq": -1}, "seq"),
                ({"consumer_id": "c1", "seq": True}, "seq"),
                ({"consumer_id": "c1", "seq": 0, "extra": 1}, "extra")):
            with self.subTest(payload=payload):
                status, body, _ = self._request(
                    "/v1/devices/bob/inbox-job-events/checkpoint",
                    raw=json.dumps(payload))
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], field)

    def test_unknown_device_is_404(self) -> None:
        status, body, _ = self._checkpoint_http(device="ghost")
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "device_id")

    def test_conflicts_over_http(self) -> None:
        self._queue_http("J1")
        status, body, _ = self._checkpoint_http(seq=2)
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "seq")
        self._checkpoint_http(seq=1)
        status, body, _ = self._checkpoint_http(seq=0)
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "seq")

    def test_malformed_route_is_404_device_id(self) -> None:
        status, body, _ = self._request(
            "/v1/devices/bob/extra/inbox-job-events/checkpoint",
            raw=json.dumps({"consumer_id": "c1", "seq": None}))
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "device_id")


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

    def test_section_follows_redelivery_job_events(self) -> None:
        self._op("J1", "queue")
        self._checkpoint(seq=1)
        document = self._document()
        keys = list(document)
        self.assertEqual(keys.index("redelivery_job_event_checkpoints"),
                         keys.index("redelivery_job_events") + 1)
        self.assertEqual(document["redelivery_job_event_checkpoints"], [
            {"device_id": "bob", "consumer_id": "c1", "seq": 1,
             "updated_at": document["redelivery_job_event_checkpoints"]
             [0]["updated_at"]}])
        item = document["redelivery_job_event_checkpoints"][0]
        self.assertEqual(list(item),
                         ["device_id", "consumer_id", "seq", "updated_at"])

    def test_advance_consumes_a_generation_reads_do_not(self) -> None:
        self._op("J1", "queue")
        generation = self.state_store.commit_seq
        self._checkpoint()
        self._checkpoint(seq=1)  # equal? no: advances to 1
        self.assertEqual(self.state_store.commit_seq, generation + 1)
        self._checkpoint(seq=1)  # equal: no write
        self._checkpoint()       # read-only
        self.assertEqual(self.state_store.commit_seq, generation + 1)

    def test_restart_restores_checkpoint_and_integrity_passes(self) -> None:
        self._op("J1", "queue")
        self._op("J2", "queue")
        _, status = self._checkpoint(seq=2)
        self.assertEqual(status, 201)
        before, _ = self._checkpoint()
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        after, status = restarted.inbox_job_event_checkpoint(
            "bob", {"consumer_id": "c1", "seq": None})
        self.assertEqual(status, 200)
        self.assertEqual(after, before)
        # The stored value still bounds later moves after the restart.
        with self.assertRaises(ServiceError) as caught:
            restarted.inbox_job_event_checkpoint(
                "bob", {"consumer_id": "c1", "seq": 1})
        self.assertEqual(caught.exception.status_code, 409)
        self.assertTrue(restarted.persistence_integrity()["consistent"])

    def test_save_failure_rolls_checkpoint_back(self) -> None:
        import e2ee_backend.persistence as persistence_mod

        self._op("J1", "queue")
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
                self._checkpoint(seq=1)
        finally:
            persistence_mod.os.fsync = real_fsync
        # Rolled back: no generation consumed and no checkpoint recorded.
        self.assertEqual(self.state_store.commit_seq, generation)
        body, status = self._checkpoint()
        self.assertEqual((body["seq"], body["updated_at"]), (0, None))
        # The advance can be retried and now commits.
        _, status = self._checkpoint(seq=1)
        self.assertEqual(status, 201)

    def test_legacy_document_without_section_loads(self) -> None:
        self._op("J1", "queue")
        self._checkpoint(seq=1)
        document = self._document()
        document.pop("redelivery_job_event_checkpoints")
        document.pop("integrity_log_version", None)
        legacy_path = os.path.join(self.directory, "legacy.json")
        with open(legacy_path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        restarted = DeviceService()
        attach_persistence(restarted, legacy_path)  # must not raise
        # The dropped checkpoints are simply gone: the pair reads as 0.
        body, status = restarted.inbox_job_event_checkpoint(
            "bob", {"consumer_id": "c1", "seq": None})
        self.assertEqual(status, 200)
        self.assertEqual((body["seq"], body["updated_at"]), (0, None))

    def _document_with_checkpoint(self, mutate):
        self._op("J1", "queue")
        self._op("J2", "queue")
        self._checkpoint(seq=2)
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
        self._assert_refuses_startup(self._document_with_checkpoint(mutate))

    def test_restore_rejects_bad_item_shape(self) -> None:
        def append(item):
            return lambda d: d["redelivery_job_event_checkpoints"].append(
                item)
        for mutate in (
                append("x"),
                # missing updated_at
                append({"device_id": "bob", "consumer_id": "c2",
                        "seq": 1}),
                # extra key
                append({"device_id": "bob", "consumer_id": "c2", "seq": 1,
                        "updated_at": "2026-01-01T00:00:00.000000+00:00",
                        "x": 1}),
                # wrong key order
                append({"consumer_id": "c2", "device_id": "bob", "seq": 1,
                        "updated_at":
                            "2026-01-01T00:00:00.000000+00:00"}),
                append({"device_id": "", "consumer_id": "c2", "seq": 1,
                        "updated_at":
                            "2026-01-01T00:00:00.000000+00:00"}),
                append({"device_id": "bob", "consumer_id": "", "seq": 1,
                        "updated_at":
                            "2026-01-01T00:00:00.000000+00:00"}),
                append({"device_id": "bob", "consumer_id": "c2",
                        "seq": True,
                        "updated_at":
                            "2026-01-01T00:00:00.000000+00:00"}),
                append({"device_id": "bob", "consumer_id": "c2", "seq": -1,
                        "updated_at":
                            "2026-01-01T00:00:00.000000+00:00"}),
                # updated_at not a string
                append({"device_id": "bob", "consumer_id": "c2", "seq": 1,
                        "updated_at": None}),
                # updated_at not the canonical UTC microsecond form
                append({"device_id": "bob", "consumer_id": "c2", "seq": 1,
                        "updated_at": "2026-01-01T00:00:00Z"}),
                # unknown device
                append({"device_id": "ghost", "consumer_id": "c2",
                        "seq": 1,
                        "updated_at":
                            "2026-01-01T00:00:00.000000+00:00"}),
                # seq beyond the device's last event seq (2)
                append({"device_id": "bob", "consumer_id": "c2", "seq": 3,
                        "updated_at":
                            "2026-01-01T00:00:00.000000+00:00"})):
            with self.subTest(mutate=mutate):
                self._assert_refuses_startup(
                    self._document_with_checkpoint(mutate))

    def test_restore_rejects_duplicate_pair(self) -> None:
        def mutate(document):
            document["redelivery_job_event_checkpoints"].append(
                dict(document["redelivery_job_event_checkpoints"][0]))
        self._assert_refuses_startup(self._document_with_checkpoint(mutate))


if __name__ == "__main__":
    unittest.main()
