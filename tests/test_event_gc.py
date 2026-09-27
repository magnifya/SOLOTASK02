"""Tests for the redelivery-event retention (event GC) endpoint.

POST /v1/event-gc/{device_id} carries exactly ``consumer``, ``op`` and
``seq``. ``op=touch`` (non-empty consumer, null seq) registers the
consumer at the current retention watermark with a 30-day lease (201);
``op=revoke`` (same parameters) drops the lease (unknown consumer
404/consumer, first 201, replay 200); ``op=prune`` (null consumer,
non-negative integer seq) deletes the device's events up to ``seq`` and
advances the watermark (no valid consumer 409/consumer, seq outside
[watermark, minimum valid checkpoint] 409/seq, equal 200, advance 201).
Pruning never renumbers the surviving events; an events-page ``after``
below the watermark is 409/after and an invalid consumer's checkpoint is
409/consumer_id. The checkpoint records persist with ``expires`` and
``active`` appended to the legacy four keys, and the per-device
watermarks persist in the new ``event_gc`` section.
"""
import json
import os
import shutil
import tempfile
import threading
import unittest
from datetime import datetime, timedelta, timezone
from http.client import HTTPConnection

from e2ee_backend.models import Device, SignedPreKey
from e2ee_backend.service import DeviceService, ServiceError
from e2ee_backend.http_app import create_server
from e2ee_backend.persistence import (
    PersistenceUnavailable,
    StateFileError,
    attach_persistence,
)


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

    def _gc(self, device="bob", consumer=None, op="touch", seq=None):
        return self.service.event_gc(
            device, {"consumer": consumer, "op": op, "seq": seq})

    def _touch(self, consumer="c1", device="bob"):
        return self._gc(device=device, consumer=consumer, op="touch")

    def _revoke(self, consumer="c1", device="bob"):
        return self._gc(device=device, consumer=consumer, op="revoke")

    def _prune(self, seq, device="bob"):
        return self._gc(device=device, consumer=None, op="prune", seq=seq)

    def _checkpoint(self, device="bob", consumer="c1", seq=None):
        return self.service.inbox_job_event_checkpoint(
            device, {"consumer_id": consumer, "seq": seq})

    def _events(self, device="bob", after=0):
        return self.service.inbox_job_events_page(device, after, 100)

    def _expire_lease(self, consumer="c1", device="bob"):
        record = self.service.store._redelivery_job_event_checkpoints[
            (device, consumer)]
        record.expires = "2020-01-01T00:00:00.000000+00:00"


class EventGcValidationTest(EventGcMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def test_payload_validation(self) -> None:
        for payload, field in (
                (None, "request_body"),
                ([], "request_body"),
                ("x", "request_body"),
                ({"op": "touch", "seq": None}, "consumer"),
                ({"consumer": "c1", "seq": None}, "op"),
                ({"consumer": "c1", "op": "touch"}, "seq"),
                ({"consumer": "c1", "op": "", "seq": None}, "op"),
                ({"consumer": "c1", "op": 1, "seq": None}, "op"),
                ({"consumer": "c1", "op": "bogus", "seq": None}, "op"),
                ({"consumer": "", "op": "touch", "seq": None}, "consumer"),
                ({"consumer": None, "op": "touch", "seq": None}, "consumer"),
                ({"consumer": 1, "op": "touch", "seq": None}, "consumer"),
                ({"consumer": "c1", "op": "touch", "seq": 0}, "seq"),
                ({"consumer": "", "op": "revoke", "seq": None}, "consumer"),
                ({"consumer": "c1", "op": "revoke", "seq": 1}, "seq"),
                ({"consumer": "c1", "op": "prune", "seq": 0}, "consumer"),
                ({"consumer": None, "op": "prune", "seq": None}, "seq"),
                ({"consumer": None, "op": "prune", "seq": -1}, "seq"),
                ({"consumer": None, "op": "prune", "seq": True}, "seq"),
                ({"consumer": None, "op": "prune", "seq": 1.0}, "seq"),
                ({"consumer": None, "op": "prune", "seq": "1"}, "seq"),
                ({"consumer": "c1", "op": "touch", "seq": None,
                  "x": 1}, "x"),
                ({"consumer": "c1", "op": "touch", "seq": None,
                  "y": 1, "z": 2}, "y")):
            with self.subTest(payload=payload):
                with self.assertRaises(ServiceError) as caught:
                    self.service.event_gc("bob", payload)
                self.assertEqual(caught.exception.status_code, 400)
                self.assertEqual(caught.exception.field, field)

    def test_unknown_device_is_404_device_id(self) -> None:
        for op_payload in (
                {"consumer": "c1", "op": "touch", "seq": None},
                {"consumer": "c1", "op": "revoke", "seq": None},
                {"consumer": None, "op": "prune", "seq": 0}):
            with self.subTest(payload=op_payload):
                with self.assertRaises(ServiceError) as caught:
                    self.service.event_gc("ghost", op_payload)
                self.assertEqual(caught.exception.status_code, 404)
                self.assertEqual(caught.exception.field, "device_id")

    def test_revoked_device_stays_usable(self) -> None:
        self._op("J1", "queue")
        self._touch()
        self.service.revoke_device("bob")
        body, status = self._touch()
        self.assertEqual(status, 201)
        body, status = self._prune(0)
        self.assertEqual(status, 200)
        body, status = self._revoke()
        self.assertEqual(status, 201)


class EventGcTouchRevokeTest(EventGcMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def test_touch_registers_at_watermark_with_lease(self) -> None:
        self._op("J1", "queue")
        self._op("J2", "queue")
        before = datetime.now(timezone.utc)
        body, status = self._touch()
        self.assertEqual(status, 201)
        self.assertEqual(list(body),
                         ["device_id", "consumer", "seq", "expires"])
        self.assertEqual(body["device_id"], "bob")
        self.assertEqual(body["consumer"], "c1")
        self.assertEqual(body["seq"], 0)
        expires = datetime.fromisoformat(body["expires"])
        self.assertEqual(body["expires"],
                         expires.isoformat(timespec="microseconds"))
        self.assertTrue(body["expires"].endswith("+00:00"))
        delta = expires - before
        self.assertGreater(delta, timedelta(days=29))
        self.assertLessEqual(delta, timedelta(days=30, seconds=5))

    def test_touch_renews_lease_and_keeps_checkpoint(self) -> None:
        self._op("J1", "queue")
        self._op("J2", "queue")
        self._touch()
        self._checkpoint(seq=2)
        first = self.service.store._redelivery_job_event_checkpoints[
            ("bob", "c1")].expires
        body, status = self._touch()
        self.assertEqual(status, 201)
        self.assertEqual(body["seq"], 2)
        self.assertGreaterEqual(body["expires"], first)
        # The checkpoint itself was not advanced by the touch.
        read, _ = self._checkpoint()
        self.assertEqual(read["seq"], 2)

    def test_revoke_unknown_consumer_is_404(self) -> None:
        with self.assertRaises(ServiceError) as caught:
            self._revoke()
        self.assertEqual(caught.exception.status_code, 404)
        self.assertEqual(caught.exception.field, "consumer")

    def test_revoke_first_and_replay(self) -> None:
        self._op("J1", "queue")
        self._touch()
        self._checkpoint(seq=1)
        body, status = self._revoke()
        self.assertEqual(status, 201)
        self.assertEqual(list(body),
                         ["device_id", "consumer", "seq", "expires"])
        self.assertEqual(body, {"device_id": "bob", "consumer": "c1",
                                "seq": 1, "expires": None})
        again, status = self._revoke()
        self.assertEqual(status, 200)
        self.assertEqual(again, body)

    def test_revoked_consumer_checkpoint_is_409(self) -> None:
        self._op("J1", "queue")
        self._touch()
        self._revoke()
        for seq in (None, 1):
            with self.subTest(seq=seq):
                with self.assertRaises(ServiceError) as caught:
                    self._checkpoint(seq=seq)
                self.assertEqual(caught.exception.status_code, 409)
                self.assertEqual(caught.exception.field, "consumer_id")

    def test_expired_consumer_checkpoint_is_409(self) -> None:
        self._op("J1", "queue")
        self._touch()
        self._expire_lease()
        for seq in (None, 1):
            with self.subTest(seq=seq):
                with self.assertRaises(ServiceError) as caught:
                    self._checkpoint(seq=seq)
                self.assertEqual(caught.exception.status_code, 409)
                self.assertEqual(caught.exception.field, "consumer_id")

    def test_touch_reactivates_after_revoke(self) -> None:
        self._op("J1", "queue")
        self._touch()
        self._revoke()
        body, status = self._touch()
        self.assertEqual(status, 201)
        read, status = self._checkpoint(seq=1)
        self.assertEqual(status, 201)

    def test_lease_less_checkpoint_stays_valid(self) -> None:
        # A checkpoint written without any touch (expires null) never
        # expires and never turns invalid.
        self._op("J1", "queue")
        self._checkpoint(seq=1)
        read, status = self._checkpoint()
        self.assertEqual(status, 200)
        self.assertEqual(read["seq"], 1)


class EventGcPruneTest(EventGcMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def test_prune_without_any_consumer_is_409(self) -> None:
        self._op("J1", "queue")
        with self.assertRaises(ServiceError) as caught:
            self._prune(0)
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "consumer")

    def test_prune_with_only_expired_consumers_is_409(self) -> None:
        self._op("J1", "queue")
        self._touch()
        self._expire_lease()
        with self.assertRaises(ServiceError) as caught:
            self._prune(0)
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "consumer")

    def test_prune_with_only_revoked_consumers_is_409(self) -> None:
        self._op("J1", "queue")
        self._touch()
        self._revoke()
        with self.assertRaises(ServiceError) as caught:
            self._prune(0)
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "consumer")

    def test_prune_equal_watermark_is_200_noop(self) -> None:
        self._op("J1", "queue")
        self._touch()
        body, status = self._prune(0)
        self.assertEqual(status, 200)
        self.assertEqual(list(body), ["device_id", "seq", "removed"])
        self.assertEqual(body, {"device_id": "bob", "seq": 0,
                                "removed": 0})

    def test_prune_beyond_min_valid_checkpoint_is_409(self) -> None:
        self._op("J1", "queue")
        self._op("J2", "queue")
        self._touch()
        with self.assertRaises(ServiceError) as caught:
            self._prune(1)
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "seq")

    def test_prune_below_watermark_is_409(self) -> None:
        self._op("J1", "queue")
        self._op("J2", "queue")
        self._touch()
        self._checkpoint(seq=2)
        body, status = self._prune(2)
        self.assertEqual(status, 201)
        with self.assertRaises(ServiceError) as caught:
            self._prune(1)
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "seq")

    def test_prune_deletes_events_and_keeps_seqs(self) -> None:
        self._op("J1", "queue")
        self._op("J2", "queue")
        self._op("J3", "queue")
        self._touch()
        self._checkpoint(seq=2)
        body, status = self._prune(2)
        self.assertEqual(status, 201)
        self.assertEqual(body, {"device_id": "bob", "seq": 2,
                                "removed": 2})
        page = self._events(after=2)
        self.assertEqual([event["seq"] for event in page["events"]], [3])
        # New events continue past the last seq; nothing is renumbered.
        self._op("J4", "queue")
        page = self._events(after=2)
        self.assertEqual([event["seq"] for event in page["events"]],
                         [3, 4])

    def test_prune_up_to_min_valid_checkpoint(self) -> None:
        self._op("J1", "queue")
        self._op("J2", "queue")
        self._touch("c1")
        self._touch("c2")
        self._checkpoint(consumer="c1", seq=2)
        self._checkpoint(consumer="c2", seq=1)
        # The minimum valid checkpoint bounds the prune.
        with self.assertRaises(ServiceError) as caught:
            self._prune(2)
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "seq")
        body, status = self._prune(1)
        self.assertEqual(status, 201)
        self.assertEqual(body["removed"], 1)

    def test_lease_less_checkpoint_blocks_prune(self) -> None:
        self._op("J1", "queue")
        self._checkpoint(seq=1)  # no lease, still a valid consumer
        body, status = self._prune(1)
        self.assertEqual(status, 201)
        self.assertEqual(body["removed"], 1)

    def test_events_after_below_watermark_is_409(self) -> None:
        self._op("J1", "queue")
        self._op("J2", "queue")
        self._touch()
        self._checkpoint(seq=2)
        self._prune(2)
        with self.assertRaises(ServiceError) as caught:
            self._events(after=1)
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "after")
        # The default after (0) is below the watermark too.
        with self.assertRaises(ServiceError) as caught:
            self._events()
        self.assertEqual(caught.exception.status_code, 409)
        # An after equal to the watermark pages normally.
        page = self._events(after=2)
        self.assertEqual(page["events"], [])
        self.assertEqual(page["next_after"], 2)
        self.assertFalse(page["has_more"])

    def test_checkpoint_below_watermark_consumer_is_409(self) -> None:
        self._op("J1", "queue")
        self._op("J2", "queue")
        self._touch("c1")
        self._touch("c2")
        self._checkpoint(consumer="c1", seq=2)
        # c2 never advanced; once its lease is gone its checkpoint (0)
        # falls below the watermark a prune can reach via c1... but c2 is
        # still valid, so it bounds the prune. Revoke it first.
        self._revoke("c2")
        body, status = self._prune(2)
        self.assertEqual(status, 201)
        with self.assertRaises(ServiceError) as caught:
            self._checkpoint(consumer="c2")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "consumer_id")
        # A fresh touch re-registers c2 at the watermark.
        body, status = self._touch("c2")
        self.assertEqual(status, 201)
        self.assertEqual(body["seq"], 2)
        read, _ = self._checkpoint(consumer="c2")
        self.assertEqual(read["seq"], 2)

    def test_checkpoint_advance_after_full_prune_uses_watermark(self) -> None:
        self._op("J1", "queue")
        self._touch()
        self._checkpoint(seq=1)
        self._prune(1)
        # The chain is empty; the last seq is the watermark itself, so an
        # advance beyond it conflicts.
        with self.assertRaises(ServiceError) as caught:
            self._checkpoint(seq=2)
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "seq")
        # A new event continues at seq 2 and becomes checkpointable.
        self._op("J2", "queue")
        _, status = self._checkpoint(seq=2)
        self.assertEqual(status, 201)


class EventGcHTTPTest(EventGcMixin, unittest.TestCase):
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

    def _gc_http(self, device="bob", consumer="c1", op="touch", seq=None):
        return self._request(
            f"/v1/event-gc/{device}",
            json.dumps({"consumer": consumer, "op": op, "seq": seq}))

    def _queue_http(self, job_id):
        status, _, _ = self._request(
            "/v1/inbox-jobs",
            json.dumps({"device_id": "bob", "job_id": job_id,
                        "op": "queue"}))
        self.assertEqual(status, 201)

    def test_touch_revoke_prune_flow_with_key_order(self) -> None:
        self._queue_http("J1")
        self._queue_http("J2")
        status, body, raw = self._gc_http()
        self.assertEqual(status, 201)
        self.assertEqual(list(body),
                         ["device_id", "consumer", "seq", "expires"])
        self.assertLess(raw.index('"device_id"'), raw.index('"consumer"'))
        self.assertLess(raw.index('"consumer"'), raw.index('"seq"'))
        self.assertLess(raw.index('"seq"'), raw.index('"expires"'))
        # Advance the checkpoint over HTTP, then prune up to it.
        status, _, _ = self._request(
            "/v1/devices/bob/inbox-job-events/checkpoint",
            json.dumps({"consumer_id": "c1", "seq": 2}))
        self.assertEqual(status, 201)
        status, body, raw = self._gc_http(
            consumer=None, op="prune", seq=2)
        self.assertEqual(status, 201)
        self.assertEqual(list(body), ["device_id", "seq", "removed"])
        self.assertEqual(body, {"device_id": "bob", "seq": 2,
                                "removed": 2})
        # The events below the watermark are gone for good.
        status, body, _ = self._request(
            "/v1/devices/bob/inbox-job-events?after=1", method="GET")
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "after")
        status, body, _ = self._gc_http()
        self.assertEqual(status, 201)
        status, body, _ = self._gc_http(op="revoke")
        self.assertEqual(status, 201)
        self.assertEqual(body["expires"], None)
        status, body, _ = self._gc_http(op="revoke")
        self.assertEqual(status, 200)

    def test_bad_json_and_non_object_are_400_request_body(self) -> None:
        for raw in ("{", "[1]", "null", '"s"'):
            with self.subTest(raw=raw):
                status, body, _ = self._request("/v1/event-gc/bob", raw=raw)
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], "request_body")
                self.assertEqual(list(body), ["message", "field"])

    def test_field_errors_over_http(self) -> None:
        for payload, field in (
                ({"op": "touch", "seq": None}, "consumer"),
                ({"consumer": "c1", "seq": None}, "op"),
                ({"consumer": "c1", "op": "touch"}, "seq"),
                ({"consumer": "c1", "op": "bogus", "seq": None}, "op"),
                ({"consumer": "c1", "op": "touch", "seq": 0}, "seq"),
                ({"consumer": None, "op": "prune", "seq": -1}, "seq"),
                ({"consumer": "c1", "op": "touch", "seq": None,
                  "extra": 1}, "extra")):
            with self.subTest(payload=payload):
                status, body, _ = self._request(
                    "/v1/event-gc/bob", raw=json.dumps(payload))
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], field)

    def test_unknown_device_and_route_errors(self) -> None:
        status, body, _ = self._gc_http(device="ghost")
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "device_id")
        status, body, _ = self._request(
            "/v1/event-gc/bob/extra",
            raw=json.dumps({"consumer": "c1", "op": "touch", "seq": None}))
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "device_id")

    def test_conflicts_over_http(self) -> None:
        self._queue_http("J1")
        status, body, _ = self._gc_http(consumer=None, op="prune", seq=0)
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "consumer")
        self._gc_http()
        status, body, _ = self._gc_http(consumer=None, op="prune", seq=1)
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "seq")
        status, body, _ = self._gc_http(op="revoke", consumer="ghost-c")
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "consumer")


class EventGcPersistenceTest(EventGcMixin, unittest.TestCase):
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

    def test_sections_and_key_order(self) -> None:
        self._op("J1", "queue")
        self._touch()
        self._checkpoint(seq=1)
        self._prune(1)
        document = self._document()
        keys = list(document)
        self.assertEqual(keys.index("event_gc"),
                         keys.index("redelivery_job_event_checkpoints") + 1)
        self.assertEqual(document["event_gc"],
                         [{"device_id": "bob", "seq": 1}])
        item = document["redelivery_job_event_checkpoints"][0]
        self.assertEqual(list(item),
                         ["device_id", "consumer_id", "seq", "updated_at",
                          "expires", "active"])
        self.assertEqual(item["consumer_id"], "c1")
        self.assertIsNotNone(item["expires"])
        self.assertTrue(item["active"])

    def test_restart_restores_watermark_and_leases(self) -> None:
        self._op("J1", "queue")
        self._op("J2", "queue")
        self._touch()
        self._checkpoint(seq=2)
        self._prune(2)
        self._op("J3", "queue")
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        page = restarted.inbox_job_events_page("bob", 2, 100)
        self.assertEqual([event["seq"] for event in page["events"]], [3])
        with self.assertRaises(ServiceError) as caught:
            restarted.inbox_job_events_page("bob", 1, 100)
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "after")
        read, _ = restarted.inbox_job_event_checkpoint(
            "bob", {"consumer_id": "c1", "seq": None})
        self.assertEqual(read["seq"], 2)
        self.assertTrue(restarted.persistence_integrity()["consistent"])

    def test_legacy_document_without_section_loads(self) -> None:
        self._op("J1", "queue")
        self._touch()
        document = self._document()
        document.pop("event_gc")
        document.pop("integrity_log_version", None)
        legacy_path = os.path.join(self.directory, "legacy.json")
        with open(legacy_path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        restarted = DeviceService()
        attach_persistence(restarted, legacy_path)
        # The watermark loads as 0: the full chain is still there.
        page = restarted.inbox_job_events_page("bob", 0, 100)
        self.assertEqual([event["seq"] for event in page["events"]], [1])

    def test_legacy_four_key_checkpoint_loads(self) -> None:
        self._op("J1", "queue")
        self._touch()
        document = self._document()
        item = document["redelivery_job_event_checkpoints"][0]
        for key in ("expires", "active"):
            item.pop(key)
        document.pop("integrity_log_version", None)
        legacy_path = os.path.join(self.directory, "legacy.json")
        with open(legacy_path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        restarted = DeviceService()
        attach_persistence(restarted, legacy_path)
        read, status = restarted.inbox_job_event_checkpoint(
            "bob", {"consumer_id": "c1", "seq": None})
        self.assertEqual(status, 200)
        # The lease-less record is a valid consumer and blocks pruning.
        with self.assertRaises(ServiceError) as caught:
            restarted.event_gc(
                "bob", {"consumer": None, "op": "prune", "seq": 1})
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "seq")

    def test_malformed_event_gc_refuses_startup(self) -> None:
        self._op("J1", "queue")
        self._touch()
        self._checkpoint(seq=1)
        self._prune(1)
        document = self._document()
        # Write each candidate marker-less/sidecar-less, so rejection
        # comes from the payload's own semantic validation rather than
        # the integrity hash gate.
        document.pop("integrity_log_version", None)
        cases = []
        bad = json.loads(json.dumps(document))
        bad["event_gc"] = {}
        cases.append(bad)
        bad = json.loads(json.dumps(document))
        bad["event_gc"] = [{"seq": 1, "device_id": "bob"}]
        cases.append(bad)
        bad = json.loads(json.dumps(document))
        bad["event_gc"] = [{"device_id": "bob", "seq": True}]
        cases.append(bad)
        bad = json.loads(json.dumps(document))
        bad["event_gc"] = [{"device_id": "ghost", "seq": 1}]
        cases.append(bad)
        bad = json.loads(json.dumps(document))
        bad["event_gc"] = [{"device_id": "bob", "seq": 1},
                           {"device_id": "bob", "seq": 1}]
        cases.append(bad)
        bad = json.loads(json.dumps(document))
        # The surviving chain must start at the watermark + 1.
        bad["event_gc"] = [{"device_id": "bob", "seq": 0}]
        cases.append(bad)
        for index, candidate in enumerate(cases):
            with self.subTest(case=index):
                candidate_path = os.path.join(
                    self.directory, f"bad{index}.json")
                with open(candidate_path, "w", encoding="utf-8") as handle:
                    json.dump(candidate, handle)
                with open(candidate_path, "rb") as handle:
                    before = handle.read()
                with self.assertRaises(StateFileError):
                    attach_persistence(DeviceService(), candidate_path)
                with open(candidate_path, "rb") as handle:
                    self.assertEqual(handle.read(), before)

    def test_malformed_checkpoint_lease_keys_refuse_startup(self) -> None:
        self._op("J1", "queue")
        self._touch()
        document = self._document()
        document.pop("integrity_log_version", None)
        cases = []
        bad = json.loads(json.dumps(document))
        bad["redelivery_job_event_checkpoints"][0]["expires"] = "soon"
        cases.append(bad)
        bad = json.loads(json.dumps(document))
        bad["redelivery_job_event_checkpoints"][0]["active"] = "yes"
        cases.append(bad)
        bad = json.loads(json.dumps(document))
        bad["redelivery_job_event_checkpoints"][0]["active"] = False
        cases.append(bad)  # inactive but still carrying a lease
        bad = json.loads(json.dumps(document))
        item = bad["redelivery_job_event_checkpoints"][0]
        item["expires"] = item.pop("expires")  # right keys, wrong order
        cases.append(bad)
        for index, candidate in enumerate(cases):
            with self.subTest(case=index):
                candidate_path = os.path.join(
                    self.directory, f"badcp{index}.json")
                with open(candidate_path, "w", encoding="utf-8") as handle:
                    json.dump(candidate, handle)
                with open(candidate_path, "rb") as handle:
                    before = handle.read()
                with self.assertRaises(StateFileError):
                    attach_persistence(DeviceService(), candidate_path)
                with open(candidate_path, "rb") as handle:
                    self.assertEqual(handle.read(), before)

    def test_save_failure_rolls_prune_back(self) -> None:
        import e2ee_backend.persistence as persistence_mod

        self._op("J1", "queue")
        self._touch()
        self._checkpoint(seq=1)
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
                self._prune(1)
        finally:
            persistence_mod.os.fsync = real_fsync
        # Rolled back: no generation consumed, the event and the
        # watermark are unchanged.
        self.assertEqual(self.state_store.commit_seq, generation)
        page = self._events()
        self.assertEqual([event["seq"] for event in page["events"]], [1])
        self.assertEqual(self.service.store._redelivery_job_event_gc, {})
        # The prune can be retried and now commits.
        body, status = self._prune(1)
        self.assertEqual(status, 201)
        self.assertEqual(body["removed"], 1)

    def test_save_failure_rolls_touch_back(self) -> None:
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
                self._touch()
        finally:
            persistence_mod.os.fsync = real_fsync
        self.assertEqual(self.state_store.commit_seq, generation)
        with self.assertRaises(ServiceError) as caught:
            self._revoke()
        self.assertEqual(caught.exception.status_code, 404)
        _, status = self._touch()
        self.assertEqual(status, 201)

    def test_noop_operations_do_not_consume_generations(self) -> None:
        self._op("J1", "queue")
        self._touch()
        self._checkpoint(seq=1)
        self._prune(1)
        generation = self.state_store.commit_seq
        self._prune(1)      # equal watermark: 200 no-op
        self._revoke()
        self._revoke()      # replay: 200 no-op
        self.assertEqual(self.state_store.commit_seq, generation + 1)


if __name__ == "__main__":
    unittest.main()
