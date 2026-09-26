"""Tests for ``POST /v1/inbox-jobs/status-batch``.

A read-only batch query of 1:1-inbox redelivery-job states: the body
carries exactly a non-empty string ``device_id`` and a non-empty
``items`` array of objects carrying exactly a non-empty string
``job_id`` (no ``job_id`` repeats across items). An unknown device is
404/device_id while a revoked device stays queryable; items are then
prechecked in array order (unknown job 404/items[i].job_id; a job owned
by another device 409/items[i].job_id) and the first error aborts the
whole batch. On success the answer is always 200 with keys ``device_id``
then ``results``; each result is ``job_id``, ``state``, ``lease_id``,
``recoveries``, ``cancellation_id``, ``cancelled_at`` in that order, the
recovery chain in commit order. The query shares the store lock with the
mutating operations but writes nothing, advances no ``commit_seq`` and
changes no state.
"""
import json
import os
import shutil
import tempfile
import threading
import unittest
from http.client import HTTPConnection

from tests.test_device_inbox import InboxMixin
from e2ee_backend.models import Device
from e2ee_backend.persistence import attach_persistence
from e2ee_backend.service import DeviceService, ServiceError
from e2ee_backend.http_app import create_server

_PAST = "2000-01-01T00:00:00.000000+00:00"


class _BatchMixin(InboxMixin):
    def _error(self, callable_):
        with self.assertRaises(ServiceError) as caught:
            callable_()
        return caught.exception

    def _job(self, job_id, op="queue", device="bob", **extra):
        payload = {"device_id": device, "job_id": job_id, "op": op}
        payload.update(extra)
        return self.service.inbox_job(payload)

    def _queue(self, job_id, device="bob"):
        body, status = self._job(job_id, device=device)
        self.assertEqual(status, 201)
        return body

    def _dispatch(self, job_id):
        self._queue(job_id)
        body, status = self._job(job_id, "dispatch")
        self.assertEqual(status, 201)
        return body

    def _expire(self, lease_id) -> None:
        with self.service.store._lock:
            for state in self.service.store._delivery.values():
                for lease in state.leases:
                    if lease.lease_id == lease_id:
                        lease.leased_until = _PAST

    def _recover(self, job_id, recovery_id):
        return self._job(job_id, "recover", recovery_id=recovery_id)

    def _cancel(self, job_id, cancellation_id):
        return self._job(job_id, "cancel",
                         cancellation_id=cancellation_id)

    def _ack_all_inbox(self) -> None:
        """Ack every message seeded into bob's inbox (a1..a3, b1..b2)."""
        for session_id, message_ids in (
                (self.sid1, ("a1", "a2", "a3")),
                (self.sid2, ("b1", "b2"))):
            for sequence, message_id in enumerate(message_ids, start=1):
                self.service.ack_message(session_id, {
                    "device_id": "bob", "message_id": message_id,
                    "sequence": sequence})

    def _batch(self, device_id="bob", items=None, **extra):
        payload = {"device_id": device_id,
                   "items": items if items is not None else []}
        payload.update(extra)
        return self.service.inbox_job_status_batch(payload)

    def _item(self, job_id):
        return {"job_id": job_id}


class InboxJobStatusBatchServiceTest(_BatchMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def test_bad_body_and_device_id_errors(self) -> None:
        good = [{"job_id": "J1"}]
        for payload in (None, [], "x", 3, True):
            error = self._error(
                lambda payload=payload:
                self.service.inbox_job_status_batch(payload))
            self.assertEqual((error.status_code, error.field),
                             (400, "request_body"), payload)
        for payload in (
                {"items": good},
                {"device_id": "", "items": good},
                {"device_id": 4, "items": good},
                {"device_id": None, "items": good},
                {"device_id": True, "items": good}):
            error = self._error(
                lambda payload=payload:
                self.service.inbox_job_status_batch(payload))
            self.assertEqual((error.status_code, error.field),
                             (400, "device_id"), payload)

    def test_items_shape_errors(self) -> None:
        for items in (None, {}, "x", 3, []):
            error = self._error(lambda items=items: self._batch(items=items))
            self.assertEqual((error.status_code, error.field),
                             (400, "items"), items)
        error = self._error(lambda: self._batch(items=["x"]))
        self.assertEqual((error.status_code, error.field),
                         (400, "items[0]"))
        error = self._error(lambda: self._batch(items=[[]]))
        self.assertEqual((error.status_code, error.field),
                         (400, "items[0]"))

    def test_extra_top_level_key_rejected_with_that_field(self) -> None:
        item = {"job_id": "J1"}
        error = self._error(lambda: self._batch(items=[item], op="status"))
        self.assertEqual((error.status_code, error.field), (400, "op"))
        error = self._error(lambda: self._batch(items=[item], bogus=1))
        self.assertEqual((error.status_code, error.field),
                         (400, "bogus"))

    def test_item_field_errors_carry_the_index(self) -> None:
        cases = (
            ([{}], "items[0].job_id"),
            ([{"job_id": ""}], "items[0].job_id"),
            ([{"job_id": 4}], "items[0].job_id"),
            ([{"job_id": None}], "items[0].job_id"),
            ([{"job_id": True}], "items[0].job_id"),
        )
        for items, field in cases:
            error = self._error(lambda items=items: self._batch(items=items))
            self.assertEqual((error.status_code, error.field),
                             (400, field), items)

    def test_item_extra_keys_rejected_at_item_level(self) -> None:
        cases = (
            [{"job_id": "J1", "extra": 1}],
            [{"job_id": "J1", "state": "pending"}],
            [{"job_id": "J1"}, {"job_id": "J2", "bogus": None}],
        )
        for items in cases:
            index = 0 if len(items) == 1 else 1
            error = self._error(lambda items=items: self._batch(items=items))
            self.assertEqual((error.status_code, error.field),
                             (400, f"items[{index}]"), items)

    def test_duplicate_job_ids_rejected_at_item_level(self) -> None:
        error = self._error(lambda: self._batch(items=[
            {"job_id": "J1"}, {"job_id": "J1"}]))
        self.assertEqual((error.status_code, error.field),
                         (400, "items[1]"))

    def test_unknown_device_is_404_device_id(self) -> None:
        self._queue("J1")
        error = self._error(lambda: self._batch(
            device_id="ghost", items=[self._item("J1")]))
        self.assertEqual((error.status_code, error.field),
                         (404, "device_id"))

    def test_revoked_device_stays_queryable(self) -> None:
        self._queue("J1")
        self.service.store.revoke_device("bob")
        body, status = self._batch(items=[self._item("J1")])
        self.assertEqual(status, 200)
        self.assertEqual(body["results"][0]["state"], "pending")

    def test_device_gate_precedes_item_checks(self) -> None:
        # An unknown device is 404/device_id even when every item is
        # itself unknown or ill-formed-by-state (a non-object item still
        # fails at shape validation first, so use a merely unknown job).
        error = self._error(lambda: self._batch(
            device_id="ghost", items=[self._item("NOPE")]))
        self.assertEqual((error.status_code, error.field),
                         (404, "device_id"))

    def test_pending_shape_and_key_order(self) -> None:
        self._queue("J1")
        body, status = self._batch(items=[self._item("J1")])
        self.assertEqual(status, 200)
        self.assertEqual(list(body), ["device_id", "results"])
        self.assertEqual(body["device_id"], "bob")
        item = body["results"][0]
        self.assertEqual(list(item),
                         ["job_id", "state", "lease_id", "recoveries",
                          "cancellation_id", "cancelled_at"])
        self.assertEqual(item["job_id"], "J1")
        self.assertEqual(item["state"], "pending")
        self.assertIsNone(item["lease_id"])
        self.assertEqual(item["recoveries"], [])
        self.assertIsNone(item["cancellation_id"])
        self.assertIsNone(item["cancelled_at"])

    def test_running_names_dispatch_lease(self) -> None:
        self._dispatch("J1")
        body, _ = self._batch(items=[self._item("J1")])
        item = body["results"][0]
        self.assertEqual(item["state"], "running")
        self.assertEqual(item["lease_id"], "J1")

    def test_succeeded_empty_dispatch_keeps_null_lease(self) -> None:
        # A device with no inbox messages at all: its first dispatch picks
        # the empty set and goes straight to succeeded with lease_id null.
        self.service.store.add_device(Device("u", "dave", "ik"))
        self._queue("JE", device="dave")
        body, status = self._job("JE", "dispatch", device="dave")
        self.assertEqual(status, 201)
        result, status = self._batch(device_id="dave",
                                     items=[self._item("JE")])
        self.assertEqual(status, 200)
        item = result["results"][0]
        self.assertEqual(item["state"], "succeeded")
        self.assertIsNone(item["lease_id"])

    def test_failed_after_failed_completion(self) -> None:
        self._dispatch("J1")
        self.service.inbox_lease_complete(
            "bob", "J1", {"completion_id": "C1", "outcome": "failed"})
        body, _ = self._batch(items=[self._item("J1")])
        item = body["results"][0]
        self.assertEqual(item["state"], "failed")
        self.assertEqual(item["lease_id"], "J1")

    def test_succeeded_after_delivered_completion(self) -> None:
        self._dispatch("J1")
        self.service.inbox_lease_complete(
            "bob", "J1", {"completion_id": "C1", "outcome": "delivered"})
        body, _ = self._batch(items=[self._item("J1")])
        self.assertEqual(body["results"][0]["state"], "succeeded")

    def test_recoveries_listed_in_commit_order(self) -> None:
        self._dispatch("J1")
        self._expire("J1")
        body, status = self._recover("J1", "R1")
        self.assertEqual(status, 201)
        # Ack every seeded message so the next recovery finds nothing
        # left to redeliver and terminates the job succeeded.
        self._ack_all_inbox()
        self._expire("R1")
        body, status = self._recover("J1", "R2")
        self.assertEqual(status, 201)
        self.assertEqual(body["state"], "succeeded")

        result, _ = self._batch(items=[self._item("J1")])
        item = result["results"][0]
        self.assertEqual(item["state"], "succeeded")
        self.assertIsNone(item["lease_id"])
        recoveries = item["recoveries"]
        self.assertEqual(len(recoveries), 2)
        self.assertEqual([list(record) for record in recoveries],
                         [["recovery_id", "lease_id"],
                          ["recovery_id", "lease_id"]])
        self.assertEqual(recoveries[0],
                         {"recovery_id": "R1", "lease_id": "R1"})
        self.assertEqual(recoveries[1],
                         {"recovery_id": "R2", "lease_id": None})

    def test_cancelled_pending_carries_cancellation_fields(self) -> None:
        self._queue("J1")
        _, status = self._cancel("J1", "X1")
        self.assertEqual(status, 201)
        body, _ = self._batch(items=[self._item("J1")])
        item = body["results"][0]
        self.assertEqual(item["state"], "cancelled")
        self.assertIsNone(item["lease_id"])
        self.assertEqual(item["cancellation_id"], "X1")
        cancelled_at = item["cancelled_at"]
        self.assertIsInstance(cancelled_at, str)
        self.assertTrue(cancelled_at.endswith("+00:00"))
        fractional = cancelled_at[:-len("+00:00")].split(".", 1)[1]
        self.assertEqual(len(fractional), 6)  # six microsecond digits

    def test_cancelled_running_keeps_current_lease_id(self) -> None:
        self._dispatch("J1")
        self._cancel("J1", "X1")
        body, _ = self._batch(items=[self._item("J1")])
        item = body["results"][0]
        self.assertEqual(item["state"], "cancelled")
        self.assertEqual(item["lease_id"], "J1")
        self.assertEqual(item["cancellation_id"], "X1")

    def test_results_keep_input_order(self) -> None:
        self._queue("J1")
        self._queue("J2")
        self._queue("J3")
        body, status = self._batch(items=[self._item("J3"),
                                          self._item("J1"),
                                          self._item("J2")])
        self.assertEqual(status, 200)
        self.assertEqual([r["job_id"] for r in body["results"]],
                         ["J3", "J1", "J2"])

    def test_mixed_states_in_one_batch(self) -> None:
        self._queue("JP")                 # pending
        self._dispatch("JR")              # claims the five inbox messages
        self.assertEqual(
            self._batch(items=[self._item("JR")])[0]["results"][0]["state"],
            "running")
        self._queue("JC")
        self._cancel("JC", "X1")          # cancelled while pending
        body, status = self._batch(items=[self._item("JC"),
                                          self._item("JR"),
                                          self._item("JP")])
        self.assertEqual(status, 200)
        self.assertEqual([r["state"] for r in body["results"]],
                         ["cancelled", "running", "pending"])

    def test_unknown_job_is_404_indexed_field(self) -> None:
        error = self._error(
            lambda: self._batch(items=[self._item("ghost")]))
        self.assertEqual((error.status_code, error.field),
                         (404, "items[0].job_id"))

    def test_other_devices_job_is_409_indexed_field(self) -> None:
        self._queue("J1")
        error = self._error(lambda: self._batch(
            device_id="carol", items=[self._item("J1")]))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[0].job_id"))

    def test_first_error_in_array_order_wins(self) -> None:
        self._queue("J1")
        error = self._error(lambda: self._batch(items=[
            self._item("J1"), self._item("ghost")]))
        self.assertEqual((error.status_code, error.field),
                         (404, "items[1].job_id"))
        error = self._error(lambda: self._batch(items=[
            self._item("ghost"), self._item("J1")]))
        self.assertEqual((error.status_code, error.field),
                         (404, "items[0].job_id"))
        # A cross-device job at index 1 wins over a valid index 0.
        self._queue("JX", device="carol")
        error = self._error(lambda: self._batch(items=[
            self._item("J1"), self._item("JX")]))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[1].job_id"))

    def test_query_is_read_only_and_repeatable(self) -> None:
        self._dispatch("J1")
        self._queue("J2")
        store = self.service.store
        with store._lock:
            before = {
                job_id: (job.state, job.lease_id,
                         [(r.recovery_id, r.lease_id)
                          for r in job.recoveries],
                         job.cancellation_id, job.cancelled_at)
                for job_id, job in store._redelivery_jobs.items()}
        items = [self._item("J2"), self._item("J1")]
        first, status = self._batch(items=items)
        self.assertEqual(status, 200)
        second, status = self._batch(items=items)
        self.assertEqual(status, 200)
        self.assertEqual(first, second)
        with store._lock:
            after = {
                job_id: (job.state, job.lease_id,
                         [(r.recovery_id, r.lease_id)
                          for r in job.recoveries],
                         job.cancellation_id, job.cancelled_at)
                for job_id, job in store._redelivery_jobs.items()}
        self.assertEqual(before, after)


class InboxJobStatusBatchPersistenceTest(_BatchMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self._build()
        self.path = os.path.join(self.directory, "state.json")
        self.state_store = attach_persistence(self.service, self.path)

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_query_advances_no_generation_on_success_or_failure(
            self) -> None:
        self._queue("J1")
        self._queue("J2")
        before = self.state_store.commit_seq
        body, status = self._batch(
            items=[self._item("J1"), self._item("J2")])
        self.assertEqual(status, 200)
        self.assertEqual(self.state_store.commit_seq, before)
        with self.assertRaises(ServiceError):
            self._batch(items=[self._item("ghost")])
        self.assertEqual(self.state_store.commit_seq, before)
        # The state file was not rewritten by the read-only query.
        with open(self.path, encoding="utf-8") as handle:
            document = json.load(handle)
        self.assertIn("redelivery_jobs", document)

    def test_restart_yields_the_same_result(self) -> None:
        self._dispatch("J1")
        self._cancel("J1", "X1")
        self._queue("J2")
        items = [self._item("J2"), self._item("J1")]
        first, status = self._batch(items=items)
        self.assertEqual(status, 200)
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        replay, status = restarted.inbox_job_status_batch(
            {"device_id": "bob", "items": items})
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)
        self.assertTrue(restarted.persistence_integrity()["consistent"])

    def test_restart_unknown_device_still_404(self) -> None:
        self._queue("J1")
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        with self.assertRaises(ServiceError) as caught:
            restarted.inbox_job_status_batch(
                {"device_id": "ghost", "items": [self._item("J1")]})
        self.assertEqual((caught.exception.status_code,
                          caught.exception.field), (404, "device_id"))


class InboxJobStatusBatchHTTPTest(_BatchMixin, unittest.TestCase):
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

    def _request(self, payload, raw_body=None):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        body = raw_body if raw_body is not None else json.dumps(payload)
        conn.request("POST", "/v1/inbox-jobs/status-batch",
                     body=body,
                     headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        data = response.read().decode("utf-8")
        conn.close()
        return response.status, (json.loads(data) if data else None), data

    def _cancel_http(self, job_id, cancellation_id) -> None:
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("POST", "/v1/inbox-jobs",
                     body=json.dumps({"device_id": "bob", "job_id": job_id,
                                      "op": "cancel",
                                      "cancellation_id": cancellation_id}),
                     headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        self.assertEqual(response.status, 201)
        response.read()
        conn.close()

    def test_status_batch_over_http(self) -> None:
        self._queue("J1")
        self._queue("J2")
        status, body, raw = self._request(
            {"device_id": "bob",
             "items": [{"job_id": "J2"}, {"job_id": "J1"}]})
        self.assertEqual(status, 200)
        self.assertEqual(list(body), ["device_id", "results"])
        self.assertEqual([r["job_id"] for r in body["results"]],
                         ["J2", "J1"])
        for result in body["results"]:
            self.assertEqual(list(result),
                             ["job_id", "state", "lease_id", "recoveries",
                              "cancellation_id", "cancelled_at"])
        # Key order is the serialization order too.
        self.assertLess(raw.index('"device_id"'), raw.index('"results"'))
        self.assertLess(raw.index('"job_id"'), raw.index('"state"'))
        self.assertLess(raw.index('"state"'), raw.index('"lease_id"'))
        self.assertLess(raw.index('"lease_id"'),
                        raw.index('"recoveries"'))
        self.assertLess(raw.index('"recoveries"'),
                        raw.index('"cancellation_id"'))
        self.assertLess(raw.index('"cancellation_id"'),
                        raw.index('"cancelled_at"'))
        # A repeated query answers byte-identically.
        status, body, raw2 = self._request(
            {"device_id": "bob",
             "items": [{"job_id": "J2"}, {"job_id": "J1"}]})
        self.assertEqual(status, 200)
        self.assertEqual(raw2, raw)

    def test_errors_over_http(self) -> None:
        self._queue("J1")
        status, body, _ = self._request(None, raw_body="{")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "request_body")
        self.assertEqual(list(body), ["message", "field"])
        status, body, _ = self._request([1, 2])
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "request_body")
        status, body, _ = self._request(
            {"device_id": "bob", "items": []})
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "items")
        status, body, _ = self._request(
            {"device_id": "bob", "items": [{}]})
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "items[0].job_id")
        status, body, _ = self._request(
            {"device_id": "bob",
             "items": [{"job_id": "J1", "x": 1}]})
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "items[0]")
        status, body, _ = self._request(
            {"device_id": "bob",
             "items": [{"job_id": "J1"}, {"job_id": "J1"}]})
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "items[1]")
        status, body, _ = self._request(
            {"device_id": "bob", "items": [{"job_id": "J1"}],
             "bogus": 1})
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "bogus")
        # Unknown device -> 404/device_id.
        status, body, _ = self._request(
            {"device_id": "ghost", "items": [{"job_id": "J1"}]})
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "device_id")
        # Unknown job -> 404 at the indexed field.
        status, body, _ = self._request(
            {"device_id": "bob", "items": [{"job_id": "ghost"}]})
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "items[0].job_id")
        self.assertEqual(list(body), ["message", "field"])
        # A job owned by another device conflicts at the item field.
        status, body, _ = self._request(
            {"device_id": "carol", "items": [{"job_id": "J1"}]})
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "items[0].job_id")

    def test_revoked_device_still_200_over_http(self) -> None:
        self._queue("J1")
        self.service.store.revoke_device("bob")
        status, body, _ = self._request(
            {"device_id": "bob", "items": [{"job_id": "J1"}]})
        self.assertEqual(status, 200)
        self.assertEqual(body["results"][0]["state"], "pending")

    def test_cancelled_fields_over_http(self) -> None:
        self._queue("J1")
        self._cancel_http("J1", "X1")
        status, body, _ = self._request(
            {"device_id": "bob", "items": [{"job_id": "J1"}]})
        self.assertEqual(status, 200)
        item = body["results"][0]
        self.assertEqual(item["state"], "cancelled")
        self.assertEqual(item["cancellation_id"], "X1")
        self.assertTrue(item["cancelled_at"].endswith("+00:00"))
        self.assertEqual(item["recoveries"], [])


if __name__ == "__main__":
    unittest.main()
