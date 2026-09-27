"""Tests for the redelivery-job event retention (event-gc) endpoint.

POST /v1/event-gc/{device_id} manages the retention of a device's
redelivery-job lifecycle event chain with three ops in one entry point.
The body carries exactly ``consumer``, ``op`` and ``seq``; a bad or
non-object body is 400/request_body, a missing/invalid/extra field is
400 with that field. An unknown device is 404/device_id; a revoked
device stays usable. ``touch`` (consumer non-empty, seq null) registers
the consumer at the current watermark with a thirty-day lease (201);
``revoke`` (same parameters) deactivates the registration (unknown
consumer 404/consumer, first 201, replay 200); ``prune`` (consumer
null, seq a non-boolean non-negative integer) deletes the device's
events with ``seq <= seq`` — no valid consumer is 409/consumer, a seq
outside [watermark, minimum valid checkpoint] is 409/seq, an equal seq
is an idempotent 200, a greater one advances the watermark (201).
Touch/revoke responses carry device_id/consumer/seq/expires; prune
carries device_id/seq/removed. Pruning never renumbers the surviving
events: the events GET rejects an ``after`` below the watermark with
409/after, and a checkpoint advance naming an invalid (revoked or
lease-expired) consumer is 409/consumer_id. Checkpoint records gain
``expires``/``active`` after the legacy four keys; the watermark is
persisted in the new ``event_gc`` section.
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
from e2ee_backend.storage import DeviceStore
from e2ee_backend.http_app import create_server
from e2ee_backend.persistence import attach_persistence


class EventGcMixin:
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

    def _queue(self, *job_ids):
        for job_id in job_ids:
            self._op(job_id, "queue")

    def _gc(self, device, payload):
        return self.service.event_gc(device, payload)

    def _touch(self, consumer="c1", device="bob"):
        return self._gc(device, {"consumer": consumer, "op": "touch",
                                 "seq": None})

    def _revoke(self, consumer="c1", device="bob"):
        return self._gc(device, {"consumer": consumer, "op": "revoke",
                                 "seq": None})

    def _prune(self, seq, device="bob"):
        return self._gc(device, {"consumer": None, "op": "prune",
                                 "seq": seq})

    def _checkpoint(self, consumer="c1", seq=None, device="bob"):
        return self.service.inbox_job_event_checkpoint(
            device, {"consumer_id": consumer, "seq": seq})

    def _events(self, after=0, device="bob"):
        return self.service.inbox_job_events_page(device, after, 100)

    def _assert_error(self, status, field, fn, *args):
        with self.assertRaises(ServiceError) as caught:
            fn(*args)
        self.assertEqual(caught.exception.status_code, status)
        self.assertEqual(caught.exception.field, field)
        return caught.exception


class EventGcValidationTest(EventGcMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def test_non_object_body(self) -> None:
        for bad in (None, [], "x", 1, True):
            self._assert_error(400, "request_body", self._gc, "bob", bad)

    def test_extra_key_rejected_in_payload_order(self) -> None:
        self._assert_error(400, "zzz", self._gc, "bob",
                           {"consumer": "c", "op": "touch", "seq": None,
                            "zzz": 1, "aaa": 2})

    def test_missing_fields(self) -> None:
        self._assert_error(400, "consumer", self._gc, "bob",
                           {"op": "touch", "seq": None})
        self._assert_error(400, "op", self._gc, "bob",
                           {"consumer": "c", "seq": None})
        self._assert_error(400, "seq", self._gc, "bob",
                           {"consumer": "c", "op": "touch"})

    def test_bad_op(self) -> None:
        for bad in ("peek", "", 1, None, True):
            self._assert_error(400, "op", self._gc, "bob",
                               {"consumer": "c", "op": bad, "seq": None})

    def test_bad_consumer(self) -> None:
        for bad in ("", 1, True, []):
            self._assert_error(400, "consumer", self._gc, "bob",
                               {"consumer": bad, "op": "touch",
                                "seq": None})

    def test_bad_seq(self) -> None:
        for bad in (True, 1.5, "1", -1):
            self._assert_error(400, "seq", self._gc, "bob",
                               {"consumer": None, "op": "prune",
                                "seq": bad})

    def test_op_specific_field_shapes(self) -> None:
        # touch/revoke need a non-empty consumer and a null seq.
        self._assert_error(400, "consumer", self._gc, "bob",
                           {"consumer": None, "op": "touch", "seq": None})
        self._assert_error(400, "seq", self._gc, "bob",
                           {"consumer": "c", "op": "touch", "seq": 0})
        self._assert_error(400, "consumer", self._gc, "bob",
                           {"consumer": None, "op": "revoke", "seq": None})
        self._assert_error(400, "seq", self._gc, "bob",
                           {"consumer": "c", "op": "revoke", "seq": 0})
        # prune needs a null consumer and an integer seq.
        self._assert_error(400, "consumer", self._gc, "bob",
                           {"consumer": "c", "op": "prune", "seq": 0})
        self._assert_error(400, "seq", self._gc, "bob",
                           {"consumer": None, "op": "prune", "seq": None})

    def test_unknown_device(self) -> None:
        for op_payload in ({"consumer": "c", "op": "touch", "seq": None},
                           {"consumer": "c", "op": "revoke", "seq": None},
                           {"consumer": None, "op": "prune", "seq": 0}):
            self._assert_error(404, "device_id", self._gc, "ghost",
                               op_payload)

    def test_revoked_device_stays_usable(self) -> None:
        self.service.store.revoke_device("bob")
        body, status = self._touch()
        self.assertEqual(status, 201)
        self.assertEqual(body["device_id"], "bob")


class EventGcFlowTest(EventGcMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()
        self._queue("J1", "J2", "J3")

    def test_touch_registers_at_watermark_with_thirty_day_lease(self) -> None:
        body, status = self._touch()
        self.assertEqual(status, 201)
        self.assertEqual(list(body),
                         ["device_id", "consumer", "seq", "expires"])
        self.assertEqual(body["device_id"], "bob")
        self.assertEqual(body["consumer"], "c1")
        self.assertEqual(body["seq"], 0)
        self.assertTrue(body["expires"].endswith("+00:00"))
        self.assertEqual(len(body["expires"].split(".")[1]),
                         len("000000+00:00"))
        # The lease deadline is roughly thirty days out.
        from datetime import datetime
        expires = datetime.fromisoformat(body["expires"])
        delta = expires - datetime.now(expires.tzinfo)
        self.assertTrue(
            29 * 86400 < delta.total_seconds() <= 30 * 86400)

    def test_touch_refreshes_and_replays_as_201(self) -> None:
        first, _ = self._touch()
        again, status = self._touch()
        self.assertEqual(status, 201)
        self.assertEqual(again["seq"], 0)
        self.assertGreaterEqual(again["expires"], first["expires"])

    def test_touch_after_prune_registers_at_watermark(self) -> None:
        self._touch("c1")
        self._checkpoint("c1", seq=2)
        body, _ = self._prune(2)
        self.assertEqual(body["removed"], 2)
        body, status = self._touch("c2")
        self.assertEqual(status, 201)
        self.assertEqual(body["seq"], 2)

    def test_revoke_lifecycle(self) -> None:
        self._assert_error(404, "consumer", self._revoke)
        self._touch()
        self._checkpoint("c1", seq=1)
        body, status = self._revoke()
        self.assertEqual(status, 201)
        self.assertEqual(list(body),
                         ["device_id", "consumer", "seq", "expires"])
        self.assertEqual(body["seq"], 1)
        self.assertIsNone(body["expires"])
        # A replay is an idempotent 200 and writes nothing.
        body, status = self._revoke()
        self.assertEqual(status, 200)
        self.assertIsNone(body["expires"])

    def test_prune_requires_a_valid_consumer(self) -> None:
        self._assert_error(409, "consumer", self._prune, 0)
        self._touch()
        self._revoke()
        self._assert_error(409, "consumer", self._prune, 0)

    def test_prune_seq_window(self) -> None:
        self._touch("c1")  # checkpoint 0
        self._assert_error(409, "seq", self._prune, 1)
        self._checkpoint("c1", seq=2)
        self._assert_error(409, "seq", self._prune, 3)
        body, status = self._prune(2)
        self.assertEqual(status, 201)
        self.assertEqual(list(body), ["device_id", "seq", "removed"])
        self.assertEqual(body, {"device_id": "bob", "seq": 2,
                                "removed": 2})
        # Below the watermark is a conflict; equal is an idempotent 200.
        self._assert_error(409, "seq", self._prune, 1)
        body, status = self._prune(2)
        self.assertEqual(status, 200)
        self.assertEqual(body["removed"], 0)

    def test_prune_deletes_events_without_renumbering(self) -> None:
        self._touch("c1")
        self._checkpoint("c1", seq=2)
        self._prune(2)
        page = self._events(after=2)
        self.assertEqual([e["seq"] for e in page["events"]], [3])
        # New events continue the numbering past the pruned prefix.
        self._queue("J4")
        page = self._events(after=2)
        self.assertEqual([e["seq"] for e in page["events"]], [3, 4])

    def test_events_after_below_watermark(self) -> None:
        self._touch("c1")
        self._checkpoint("c1", seq=1)
        self._prune(1)
        self._assert_error(409, "after", self._events, 0)
        page = self._events(after=1)
        self.assertEqual([e["seq"] for e in page["events"]], [2, 3])

    def test_checkpoint_advance_with_invalid_consumer(self) -> None:
        self._touch()
        self._revoke()
        self._assert_error(409, "consumer_id", self._checkpoint, "c1", 1)
        # The read-only query still reports the stored checkpoint.
        body, status = self._checkpoint("c1", None)
        self.assertEqual(status, 200)
        # An unregistered consumer still checkpoints lazily as before.
        body, status = self._checkpoint("c2", 1)
        self.assertEqual(status, 201)

    def test_prune_window_uses_minimum_valid_checkpoint(self) -> None:
        self._touch("c1")
        self._touch("c2")
        self._checkpoint("c1", seq=3)
        self._checkpoint("c2", seq=1)
        # The minimum valid checkpoint bounds the prune.
        self._assert_error(409, "seq", self._prune, 2)
        body, status = self._prune(1)
        self.assertEqual(status, 201)
        self.assertEqual(body["removed"], 1)


class EventGcPersistenceTest(EventGcMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self._build()
        self.path = os.path.join(self.directory, "state.json")
        self.state_file = attach_persistence(self.service, self.path)

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def _document(self):
        with open(self.path, encoding="utf-8") as handle:
            return json.load(handle)

    def test_sections_and_checkpoint_keys(self) -> None:
        self._queue("J1")
        self._touch()
        document = self._document()
        keys = list(document)
        self.assertEqual(keys.index("event_gc"),
                         keys.index("redelivery_job_event_checkpoints") + 1)
        self.assertEqual(document["event_gc"], [])
        item = document["redelivery_job_event_checkpoints"][0]
        self.assertEqual(list(item),
                         ["device_id", "consumer_id", "seq", "updated_at",
                          "expires", "active"])
        self.assertEqual(item["active"], True)
        self.assertIsNotNone(item["expires"])

    def test_prune_persists_watermark_and_survives_restart(self) -> None:
        self._queue("J1", "J2", "J3")
        self._touch()
        self._checkpoint("c1", seq=2)
        self._prune(2)
        document = self._document()
        self.assertEqual(document["event_gc"],
                         [{"device_id": "bob", "seq": 2}])
        self.assertEqual([e["seq"] for e in
                          document["redelivery_job_events"]], [3])
        # A fresh store recovers the watermark, the pruned chain and the
        # registration; the events GET still rejects a stale after.
        restored = DeviceService()
        attach_persistence(restored, self.path)
        self.assertEqual(
            restored.store._redelivery_job_event_gc, {"bob": 2})
        with self.assertRaises(ServiceError) as caught:
            restored.inbox_job_events_page("bob", 1, 100)
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "after")
        page = restored.inbox_job_events_page("bob", 2, 100)
        self.assertEqual([e["seq"] for e in page["events"]], [3])
        # New events keep the numbering after a restart.
        restored.inbox_job({"device_id": "bob", "job_id": "J9",
                            "op": "queue"})
        page = restored.inbox_job_events_page("bob", 2, 100)
        self.assertEqual([e["seq"] for e in page["events"]], [3, 4])

    def test_equal_prune_and_replay_revoke_do_not_consume_generation(
            self) -> None:
        self._queue("J1")
        self._touch()
        self._checkpoint("c1", seq=1)
        self._prune(1)
        generation = self.state_file.commit_seq
        self._prune(1)          # equal: no-op
        self._revoke()          # first: writes
        self.assertEqual(self.state_file.commit_seq, generation + 1)
        self._revoke()          # replay: no-op
        self.assertEqual(self.state_file.commit_seq, generation + 1)

    def test_legacy_four_key_checkpoint_loads(self) -> None:
        self._queue("J1")
        self._touch()
        document = self._document()
        item = document["redelivery_job_event_checkpoints"][0]
        legacy = {key: item[key]
                  for key in ("device_id", "consumer_id", "seq",
                              "updated_at")}
        document["redelivery_job_event_checkpoints"] = [legacy]
        payload = {key: value for key, value in document.items()
                   if key not in ("version", "commit_seq",
                                  "integrity_log_version")}
        restored = DeviceStore()
        restored.restore_state(payload)
        record = restored._redelivery_job_event_checkpoints[("bob", "c1")]
        self.assertIsNone(record.expires)
        self.assertTrue(record.active)

    def test_malformed_event_gc_section_refuses_startup(self) -> None:
        self._queue("J1")
        document = self._document()
        for bad in ([{"device_id": "bob"}],
                    [{"device_id": "bob", "seq": -1}],
                    [{"device_id": "bob", "seq": True}],
                    [{"device_id": "ghost", "seq": 0}],
                    [{"device_id": "bob", "seq": 0, "x": 1}],
                    [{"device_id": "bob", "seq": 0},
                     {"device_id": "bob", "seq": 1}],
                    ["oops"]):
            broken = dict(document)
            broken["event_gc"] = bad
            with self.assertRaises(ValueError):
                DeviceStore().restore_state(broken)

    def test_chain_must_start_above_watermark(self) -> None:
        self._queue("J1")
        document = self._document()
        document["event_gc"] = [{"device_id": "bob", "seq": 1}]
        # The stored chain still starts at seq 1: below the watermark.
        with self.assertRaises(ValueError):
            DeviceStore().restore_state(document)


class EventGcHttpTest(EventGcMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server, cls.service = create_server("127.0.0.1", 0)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(
            target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self) -> None:
        self.service.store.add_device(Device("u", "alice", "ik"))
        self.service.store.add_device(Device(
            "u", "bob", "ik",
            prekeys=[SignedPreKey("pk1", "pubk1")]))

    def _request(self, method, path, body=None, raw=None):
        connection = HTTPConnection("127.0.0.1", self.port)
        payload = raw if raw is not None else (
            json.dumps(body) if body is not None else None)
        connection.request(
            method, path, body=payload,
            headers={"Content-Type": "application/json"}
            if payload is not None else {})
        response = connection.getresponse()
        data = json.loads(response.read() or b"null")
        connection.close()
        return response.status, data

    def test_http_flow(self) -> None:
        status, body = self._request(
            "POST", "/v1/event-gc/bob",
            {"consumer": "c1", "op": "touch", "seq": None})
        self.assertEqual(status, 201)
        self.assertEqual(list(body),
                         ["device_id", "consumer", "seq", "expires"])
        status, body = self._request(
            "POST", "/v1/event-gc/bob",
            {"consumer": None, "op": "prune", "seq": 0})
        self.assertEqual(status, 200)
        self.assertEqual(list(body), ["device_id", "seq", "removed"])
        status, body = self._request(
            "POST", "/v1/event-gc/bob",
            {"consumer": "c1", "op": "revoke", "seq": None})
        self.assertEqual(status, 201)
        self.assertIsNone(body["expires"])

    def test_http_bad_json_and_unknown_device(self) -> None:
        status, body = self._request("POST", "/v1/event-gc/bob",
                                     raw="{not json")
        self.assertEqual((status, body["field"]), (400, "request_body"))
        status, body = self._request("POST", "/v1/event-gc/bob",
                                     raw="[1]")
        self.assertEqual((status, body["field"]), (400, "request_body"))
        status, body = self._request(
            "POST", "/v1/event-gc/ghost",
            {"consumer": "c", "op": "touch", "seq": None})
        self.assertEqual((status, body["field"]), (404, "device_id"))
        status, body = self._request(
            "POST", "/v1/event-gc/bob",
            {"consumer": "c", "op": "touch", "seq": None, "x": 1})
        self.assertEqual((status, body["field"]), (400, "x"))


if __name__ == "__main__":
    unittest.main()
