"""Tests for ``POST /v1/inbox-jobs/status-batch``.

A read-only batch query of 1:1-inbox redelivery jobs: the body carries
exactly a non-empty string ``device_id`` and a non-empty ``items`` array
of objects carrying exactly a unique non-empty string ``job_id``. The
device gate runs first (unknown device -> 404/device_id; a revoked
device's jobs stay queryable, like the single-job GET); items are then
prechecked in array order (never-queued job 404/items[i].job_id; a job
owned by another device 409/items[i].job_id) and the first error aborts
the whole batch. On success the answer is always 200 with keys
``device_id`` then ``results``; results keep input order and each item is
``job_id``, ``state``, ``lease_id``, ``recoveries``,
``cancellation_id``, ``cancelled_at`` in that order. The query shares
the store lock with the mutating operations but writes nothing, advances
no ``commit_seq`` and changes no state.
"""
import json
import os
import shutil
import tempfile
import threading
import unittest
from http.client import HTTPConnection

from tests.test_device_inbox import InboxMixin
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

    def _ack_everything(self) -> None:
        """Ack every inbox message of sid1/sid2 for bob."""
        for sid, messages in ((self.sid1, ("a1", "a2", "a3")),
                              (self.sid2, ("b1", "b2"))):
            for sequence, message_id in enumerate(messages, start=1):
                self.service.ack_message(sid, {
                    "device_id": "bob", "message_id": message_id,
                    "sequence": sequence})

    def _build_all_states(self):
        """Create one job in every state plus a recovery chain.

        Returns a mapping ``job_id -> expected six-key view`` (without the
        variable cancelled_at/timestamp fields).
        """
        # pending
        self._queue("JP")
        # running: dispatch leases all five inbox messages.
        self._dispatch("JR")
        # succeeded with an empty dispatch (JR holds every message).
        self._dispatch("JE")
        # failed: let JR's lease lapse, dispatch JF and fail its lease.
        self._expire("JR")
        self._dispatch("JF")
        self.service.inbox_lease_complete(
            "bob", "JF", {"completion_id": "CF", "outcome": "failed"})
        # cancelled running: a failed completion no longer withholds the
        # messages, so JCR leases them and is then cancelled (its lease is
        # released inside the cancel transaction).
        self._dispatch("JCR")
        _, cancel_status = self._job(
            "JCR", "cancel", cancellation_id="CCR")
        self.assertEqual(cancel_status, 201)
        # A running job recovered twice: first recovery re-leases the
        # messages under R1; after the messages are acked and R1 lapses,
        # the second recovery finds nothing and ends the job succeeded
        # with lease_id null.
        self._dispatch("JRR")
        self._expire("JRR")
        body, status = self._job("JRR", "recover", recovery_id="R1")
        self.assertEqual(status, 201)
        self.assertEqual(body["lease_id"], "R1")
        self._ack_everything()
        self._expire("R1")
        body, status = self._job("JRR", "recover", recovery_id="R2")
        self.assertEqual(status, 201)
        self.assertEqual(body["state"], "succeeded")
        self.assertIsNone(body["lease_id"])
        # cancelled pending (no lease at all).
        self._queue("JCP")
        _, cancel_status = self._job(
            "JCP", "cancel", cancellation_id="CCP")
        self.assertEqual(cancel_status, 201)

    def _batch(self, device_id="bob", job_ids=None, **extra):
        payload = {"device_id": device_id,
                   "items": [{"job_id": job_id}
                             for job_id in (job_ids if job_ids is not None
                                            else [])]}
        payload.update(extra)
        return self.service.inbox_job_status_batch(payload)


class InboxJobStatusBatchServiceTest(_BatchMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def test_bad_body_and_device_id_errors(self) -> None:
        good = [{"job_id": "j1"}]
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
            error = self._error(
                lambda items=items:
                self.service.inbox_job_status_batch(
                    {"device_id": "bob", "items": items}))
            self.assertEqual((error.status_code, error.field),
                             (400, "items"), items)
        error = self._error(
            lambda: self.service.inbox_job_status_batch(
                {"device_id": "bob", "items": ["x"]}))
        self.assertEqual((error.status_code, error.field), (400, "items[0]"))
        error = self._error(
            lambda: self.service.inbox_job_status_batch(
                {"device_id": "bob", "items": [[]]}))
        self.assertEqual((error.status_code, error.field), (400, "items[0]"))

    def test_extra_top_level_key_rejected_with_that_field(self) -> None:
        error = self._error(lambda: self._batch(job_ids=["JP"], op="status"))
        self.assertEqual((error.status_code, error.field), (400, "op"))
        error = self._error(lambda: self._batch(job_ids=["JP"], bogus=1))
        self.assertEqual((error.status_code, error.field), (400, "bogus"))

    def test_item_field_errors_carry_the_index(self) -> None:
        cases = (
            ([{}], "items[0].job_id"),
            ([{"job_id": ""}], "items[0].job_id"),
            ([{"job_id": 4}], "items[0].job_id"),
            ([{"job_id": None}], "items[0].job_id"),
            ([{"job_id": True}], "items[0].job_id"),
            ([{"job_id": "JP"}, {}], "items[1].job_id"),
        )
        for items, field in cases:
            error = self._error(
                lambda items=items:
                self.service.inbox_job_status_batch(
                    {"device_id": "bob", "items": items}))
            self.assertEqual((error.status_code, error.field),
                             (400, field), items)

    def test_item_extra_keys_rejected_at_item_level(self) -> None:
        cases = (
            [{"job_id": "JP", "state": "pending"}],
            [{"job_id": "JP", "recovery_id": "r"}],
            [{"job_id": "JP", "lease_id": "L"}],
            [{"job_id": "JP", "extra": 1}],
            [{"job_id": "JP"}, {"job_id": "JE", "bogus": None}],
        )
        for items in cases:
            index = 0 if len(items) == 1 else 1
            error = self._error(
                lambda items=items:
                self.service.inbox_job_status_batch(
                    {"device_id": "bob", "items": items}))
            self.assertEqual((error.status_code, error.field),
                             (400, f"items[{index}]"), items)

    def test_duplicate_job_id_rejected_at_item_level(self) -> None:
        self._queue("JP")
        error = self._error(lambda: self._batch(job_ids=["JP", "JP"]))
        self.assertEqual((error.status_code, error.field), (400, "items[1]"))

    def test_unknown_device_is_404_and_revoked_still_readable(self) -> None:
        self._queue("JP")
        error = self._error(
            lambda: self._batch(device_id="ghost", job_ids=["JP"]))
        self.assertEqual((error.status_code, error.field),
                         (404, "device_id"))
        # A revoked device's jobs stay queryable (cancelled included).
        self.service.store.revoke_device("bob")
        body, status = self._batch(job_ids=["JP"])
        self.assertEqual(status, 200)
        self.assertEqual(body["results"][0]["state"], "pending")

    def test_per_item_errors_use_indexed_fields(self) -> None:
        self._queue("JP")
        self._queue("OTHER", device="bob2")
        error = self._error(lambda: self._batch(job_ids=["ghost"]))
        self.assertEqual((error.status_code, error.field),
                         (404, "items[0].job_id"))
        # A job owned by another device conflicts at the item field.
        error = self._error(lambda: self._batch(job_ids=["OTHER"]))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[0].job_id"))
        # And looking up bob's job as bob2 conflicts the same way.
        error = self._error(
            lambda: self._batch(device_id="bob2", job_ids=["JP"]))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[0].job_id"))

    def test_first_error_in_array_order_wins(self) -> None:
        self._queue("OTHER", device="bob2")
        # Cross-device first, unknown second -> 409 at index 0.
        error = self._error(
            lambda: self._batch(job_ids=["OTHER", "ghost"]))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[0].job_id"))
        # Unknown first, cross-device second -> 404 at index 0.
        error = self._error(
            lambda: self._batch(job_ids=["ghost", "OTHER"]))
        self.assertEqual((error.status_code, error.field),
                         (404, "items[0].job_id"))
        # A valid first item does not mask a later offending item.
        self._queue("JP")
        error = self._error(
            lambda: self._batch(job_ids=["JP", "ghost"]))
        self.assertEqual((error.status_code, error.field),
                         (404, "items[1].job_id"))

    def test_all_five_states_and_views(self) -> None:
        self._build_all_states()
        body, status = self._batch(job_ids=[
            "JP", "JR", "JE", "JF", "JCR", "JRR", "JCP"])
        self.assertEqual(status, 200)
        self.assertEqual(list(body), ["device_id", "results"])
        self.assertEqual(body["device_id"], "bob")
        item_keys = ["job_id", "state", "lease_id", "recoveries",
                     "cancellation_id", "cancelled_at"]
        for item in body["results"]:
            self.assertEqual(list(item), item_keys)
            self.assertNotIn("device_id", item)
        by_id = {item["job_id"]: item for item in body["results"]}
        self.assertEqual(by_id["JP"]["state"], "pending")
        self.assertIsNone(by_id["JP"]["lease_id"])
        self.assertEqual(by_id["JR"]["state"], "running")
        self.assertEqual(by_id["JR"]["lease_id"], "JR")
        self.assertEqual(by_id["JE"]["state"], "succeeded")
        self.assertIsNone(by_id["JE"]["lease_id"])
        self.assertEqual(by_id["JF"]["state"], "failed")
        self.assertEqual(by_id["JF"]["lease_id"], "JF")
        self.assertEqual(by_id["JCR"]["state"], "cancelled")
        self.assertEqual(by_id["JCR"]["lease_id"], "JCR")
        self.assertEqual(by_id["JCP"]["state"], "cancelled")
        self.assertIsNone(by_id["JCP"]["lease_id"])
        # Recovered job ended succeeded with a null lease but keeps history.
        self.assertEqual(by_id["JRR"]["state"], "succeeded")
        self.assertIsNone(by_id["JRR"]["lease_id"])
        recoveries = by_id["JRR"]["recoveries"]
        self.assertEqual([list(record) for record in recoveries],
                         [["recovery_id", "lease_id"],
                          ["recovery_id", "lease_id"]])
        self.assertEqual(recoveries, [
            {"recovery_id": "R1", "lease_id": "R1"},
            {"recovery_id": "R2", "lease_id": None}])
        # Cancellation fields: non-null on cancelled jobs, null elsewhere.
        self.assertEqual(by_id["JCR"]["cancellation_id"], "CCR")
        self.assertEqual(by_id["JCP"]["cancellation_id"], "CCP")
        stamp = by_id["JCR"]["cancelled_at"]
        self.assertIsInstance(stamp, str)
        self.assertTrue(stamp.endswith("+00:00"))
        self.assertEqual(len(stamp[:-len("+00:00")].split(".", 1)[1]), 6)
        for job_id in ("JP", "JR", "JE", "JF", "JRR"):
            self.assertIsNone(by_id[job_id]["cancellation_id"])
            self.assertIsNone(by_id[job_id]["cancelled_at"])
            self.assertEqual(by_id[job_id]["recoveries"],
                             [] if job_id != "JRR" else recoveries)
        # Every state value is one of the five allowed strings.
        for item in body["results"]:
            self.assertIn(item["state"],
                          ("pending", "running", "succeeded", "failed",
                           "cancelled"))

    def test_results_keep_input_order_not_queue_order(self) -> None:
        self._build_all_states()
        ordered = ["JCP", "JP", "JRR", "JF", "JE", "JCR", "JR"]
        body, status = self._batch(job_ids=ordered)
        self.assertEqual(status, 200)
        self.assertEqual([item["job_id"] for item in body["results"]],
                         ordered)

    def test_matches_single_job_detail_minus_device_id(self) -> None:
        self._build_all_states()
        body, status = self._batch(job_ids=["JRR", "JCR"])
        self.assertEqual(status, 200)
        for item in body["results"]:
            detail = self.service.inbox_job_get("bob", item["job_id"])
            expected = {key: value for key, value in detail.items()
                        if key != "device_id"}
            self.assertEqual(item, expected)

    def test_query_is_read_only_and_repeatable(self) -> None:
        self._build_all_states()
        store = self.service.store
        with store._lock:
            before = {
                job_id: (job.state, job.lease_id,
                         [(r.recovery_id, r.lease_id) for r in job.recoveries],
                         job.cancellation_id, job.cancelled_at)
                for job_id, job in store._redelivery_jobs.items()}
        items = ["JP", "JR", "JE", "JF", "JCR", "JRR", "JCP"]
        first, status = self._batch(job_ids=items)
        self.assertEqual(status, 200)
        second, status = self._batch(job_ids=items)
        self.assertEqual(status, 200)
        self.assertEqual(first, second)
        # A failing query is just as side-effect free.
        with self.assertRaises(ServiceError):
            self._batch(job_ids=["ghost"])
        with self.assertRaises(ServiceError):
            self._batch(job_ids=["JP", "OTHER"])
        with store._lock:
            after = {
                job_id: (job.state, job.lease_id,
                         [(r.recovery_id, r.lease_id) for r in job.recoveries],
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

    def test_query_advances_no_generation_or_file_bytes(self) -> None:
        self._build_all_states()
        before = self.state_store.commit_seq
        with open(self.path, "rb") as handle:
            bytes_before = handle.read()
        items = ["JP", "JR", "JE", "JF", "JCR", "JRR", "JCP"]
        body, status = self._batch(job_ids=items)
        self.assertEqual(status, 200)
        self.assertEqual(self.state_store.commit_seq, before)
        with open(self.path, "rb") as handle:
            self.assertEqual(handle.read(), bytes_before)
        # Failing queries advance nothing either.
        with self.assertRaises(ServiceError):
            self._batch(device_id="ghost", job_ids=["JP"])
        with self.assertRaises(ServiceError):
            self._batch(job_ids=["ghost"])
        self.assertEqual(self.state_store.commit_seq, before)
        with open(self.path, "rb") as handle:
            self.assertEqual(handle.read(), bytes_before)
        return body

    def test_restart_yields_the_same_result(self) -> None:
        self._build_all_states()
        items = ["JCP", "JP", "JRR", "JF", "JE", "JCR", "JR"]
        first, status = self._batch(job_ids=items)
        self.assertEqual(status, 200)
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        replay, status = restarted.inbox_job_status_batch(
            {"device_id": "bob",
             "items": [{"job_id": job_id} for job_id in items]})
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)
        self.assertTrue(restarted.persistence_integrity()["consistent"])


class InboxJobStatusBatchHTTPTest(_BatchMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()
        self.server, self.service = create_server(
            "127.0.0.1", 0, self.service)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        self._build_all_states()
        self._queue("OTHER", device="bob2")

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

    def test_status_batch_over_http(self) -> None:
        status, body, raw = self._request(
            {"device_id": "bob",
             "items": [{"job_id": "JRR"}, {"job_id": "JCR"}]})
        self.assertEqual(status, 200)
        self.assertEqual(list(body), ["device_id", "results"])
        six = ["job_id", "state", "lease_id", "recoveries",
               "cancellation_id", "cancelled_at"]
        self.assertEqual([list(item) for item in body["results"]],
                         [six, six])
        self.assertEqual([item["job_id"] for item in body["results"]],
                         ["JRR", "JCR"])
        # Serialization key order on the wire as well.
        top = ['"device_id"', '"results"']
        positions = [raw.index(name) for name in top]
        self.assertEqual(positions, sorted(positions))
        for name in ('"job_id"', '"state"', '"lease_id"', '"recoveries"',
                     '"cancellation_id"', '"cancelled_at"'):
            self.assertIn(name, raw)
        # Recovery item key order, compact serialization.
        self.assertIn(
            '"recoveries":[{"recovery_id":"R1","lease_id":"R1"},'
            '{"recovery_id":"R2","lease_id":null}]', raw)
        # A repeated query answers byte-identically.
        status, _, raw2 = self._request(
            {"device_id": "bob",
             "items": [{"job_id": "JRR"}, {"job_id": "JCR"}]})
        self.assertEqual(status, 200)
        self.assertEqual(raw2, raw)

    def test_shape_errors_over_http(self) -> None:
        status, body, _ = self._request(None, raw_body="{")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "request_body")
        self.assertEqual(list(body), ["message", "field"])
        status, body, _ = self._request([1, 2])
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "request_body")
        status, body, _ = self._request("string")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "request_body")
        status, body, _ = self._request(
            {"items": [{"job_id": "JP"}]})
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "device_id")
        status, body, _ = self._request({"device_id": "bob"})
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "items")
        status, body, _ = self._request({"device_id": "bob", "items": []})
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "items")
        status, body, _ = self._request(
            {"device_id": "bob", "items": [{}]})
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "items[0].job_id")
        status, body, _ = self._request(
            {"device_id": "bob", "items": [{"job_id": ""}]})
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "items[0].job_id")
        status, body, _ = self._request(
            {"device_id": "bob", "items": ["nope"]})
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "items[0]")
        status, body, _ = self._request(
            {"device_id": "bob",
             "items": [{"job_id": "JP", "extra": 1}]})
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "items[0]")
        status, body, _ = self._request(
            {"device_id": "bob",
             "items": [{"job_id": "JP"}, {"job_id": "JP"}]})
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "items[1]")
        status, body, _ = self._request(
            {"device_id": "bob", "items": [{"job_id": "JP"}], "bogus": 1})
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "bogus")

    def test_semantic_errors_over_http(self) -> None:
        status, body, _ = self._request(
            {"device_id": "ghost", "items": [{"job_id": "JP"}]})
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "device_id")
        self.assertEqual(list(body), ["message", "field"])
        status, body, _ = self._request(
            {"device_id": "bob", "items": [{"job_id": "ghost"}]})
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "items[0].job_id")
        status, body, _ = self._request(
            {"device_id": "bob", "items": [{"job_id": "OTHER"}]})
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "items[0].job_id")
        # First error in array order.
        status, body, _ = self._request(
            {"device_id": "bob",
             "items": [{"job_id": "OTHER"}, {"job_id": "ghost"}]})
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "items[0].job_id")
        status, body, _ = self._request(
            {"device_id": "bob",
             "items": [{"job_id": "ghost"}, {"job_id": "OTHER"}]})
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "items[0].job_id")

    def test_revoked_device_remains_readable_over_http(self) -> None:
        self.service.store.revoke_device("bob")
        status, body, _ = self._request(
            {"device_id": "bob",
             "items": [{"job_id": "JCR"}, {"job_id": "JP"}]})
        self.assertEqual(status, 200)
        self.assertEqual([item["state"] for item in body["results"]],
                         ["cancelled", "pending"])

    def test_cancelled_at_wire_format(self) -> None:
        status, body, _ = self._request(
            {"device_id": "bob", "items": [{"job_id": "JCP"}]})
        self.assertEqual(status, 200)
        stamp = body["results"][0]["cancelled_at"]
        self.assertTrue(stamp.endswith("+00:00"))
        self.assertEqual(len(stamp[:-len("+00:00")].split(".", 1)[1]), 6)


if __name__ == "__main__":
    unittest.main()
