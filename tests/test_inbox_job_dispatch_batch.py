"""Tests for ``POST /v1/inbox-jobs/dispatch-batch``.

An atomic batch dispatch of 1:1-inbox redelivery jobs: the body carries
exactly a ``device_id`` and a non-empty ``items`` array of objects each
carrying a unique non-empty string ``job_id``. Every item is prechecked in
array order with the single-job ``op=dispatch`` rules (first error aborts
the batch, its ``field`` prefixed to ``items[i].``); only then does the
batch dispatch in input order and commit once. A batch whose items all
replay an already applied dispatch answers 200 and writes nothing; a
partial replay conflicts 409 with the first replayed item's
``items[i].job_id``.
"""
import json
import os
import shutil
import tempfile
import threading
import unittest
from http.client import HTTPConnection

from tests.test_device_inbox import InboxMixin
from e2ee_backend.persistence import (
    PersistenceUnavailable, attach_persistence)
from e2ee_backend.service import DeviceService, ServiceError
from e2ee_backend.http_app import create_server


class _BatchMixin(InboxMixin):
    def _error(self, callable_):
        with self.assertRaises(ServiceError) as caught:
            callable_()
        return caught.exception

    def _job(self, device_id="bob", job_id="j1", op="queue", **extra):
        payload = {"device_id": device_id, "job_id": job_id, "op": op}
        payload.update(extra)
        return self.service.inbox_job(payload)

    def _queue(self, *job_ids, device_id="bob"):
        for job_id in job_ids:
            self._job(device_id=device_id, job_id=job_id)

    def _batch(self, device_id="bob", job_ids=None, **extra):
        payload = {"device_id": device_id,
                   "items": [{"job_id": job_id}
                             for job_id in (job_ids if job_ids is not None
                                            else [])]}
        payload.update(extra)
        return self.service.inbox_job_dispatch_batch(payload)


class InboxJobDispatchBatchServiceTest(_BatchMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def test_bad_body_and_device_id_errors(self) -> None:
        for payload in (None, [], "x", 3):
            error = self._error(
                lambda payload=payload:
                self.service.inbox_job_dispatch_batch(payload))
            self.assertEqual((error.status_code, error.field),
                             (400, "request_body"), payload)
        for payload in (
                {"items": [{"job_id": "j1"}]},
                {"device_id": "", "items": [{"job_id": "j1"}]},
                {"device_id": 4, "items": [{"job_id": "j1"}]},
                {"device_id": None, "items": [{"job_id": "j1"}]}):
            error = self._error(
                lambda payload=payload:
                self.service.inbox_job_dispatch_batch(payload))
            self.assertEqual((error.status_code, error.field),
                             (400, "device_id"), payload)

    def test_items_shape_errors(self) -> None:
        for items in (None, {}, "x", 3, []):
            error = self._error(
                lambda: self.service.inbox_job_dispatch_batch(
                    {"device_id": "bob", "items": items}))
            self.assertEqual((error.status_code, error.field),
                             (400, "items"), items)
        error = self._error(
            lambda: self.service.inbox_job_dispatch_batch(
                {"device_id": "bob", "items": ["x"]}))
        self.assertEqual((error.status_code, error.field), (400, "items[0]"))
        error = self._error(
            lambda: self.service.inbox_job_dispatch_batch(
                {"device_id": "bob", "items": [[]]}))
        self.assertEqual((error.status_code, error.field), (400, "items[0]"))

    def test_extra_top_level_key_rejected_with_that_field(self) -> None:
        error = self._error(lambda: self._batch(
            job_ids=["j1"], op="dispatch"))
        self.assertEqual((error.status_code, error.field), (400, "op"))
        error = self._error(lambda: self._batch(job_ids=["j1"], bogus=1))
        self.assertEqual((error.status_code, error.field), (400, "bogus"))

    def test_item_field_errors_carry_the_index(self) -> None:
        cases = (
            ([{}], "items[0].job_id"),
            ([{"job_id": ""}], "items[0].job_id"),
            ([{"job_id": 4}], "items[0].job_id"),
            ([{"job_id": None}], "items[0].job_id"),
            ([{"job_id": "j1"}, {}], "items[1].job_id"),
        )
        for items, field in cases:
            error = self._error(
                lambda items=items:
                self.service.inbox_job_dispatch_batch(
                    {"device_id": "bob", "items": items}))
            self.assertEqual((error.status_code, error.field),
                             (400, field), items)

    def test_item_extra_keys_rejected_at_item_level(self) -> None:
        cases = (
            [{"job_id": "j1", "recovery_id": "r"}],
            [{"job_id": "j1", "cancellation_id": "c"}],
            [{"job_id": "j1", "lease_id": "L"}],
            [{"job_id": "j1", "extra": 1}],
            [{"job_id": "j1"}, {"job_id": "j2", "bogus": None}],
        )
        for items in cases:
            index = 0 if len(items) == 1 else 1
            error = self._error(
                lambda items=items:
                self.service.inbox_job_dispatch_batch(
                    {"device_id": "bob", "items": items}))
            self.assertEqual((error.status_code, error.field),
                             (400, f"items[{index}]"), items)

    def test_duplicate_job_id_rejected(self) -> None:
        items = [{"job_id": "j1"}, {"job_id": "j1"}]
        error = self._error(
            lambda: self.service.inbox_job_dispatch_batch(
                {"device_id": "bob", "items": items}))
        self.assertEqual((error.status_code, error.field),
                         (400, "items[1]"))

    def test_device_gate_precedes_item_checks(self) -> None:
        self._queue("j1")
        error = self._error(lambda: self._batch(device_id="ghost",
                                                job_ids=["j1"]))
        self.assertEqual((error.status_code, error.field),
                         (409, "device_id"))
        self.service.store.revoke_device("bob")
        error = self._error(lambda: self._batch(job_ids=["j1"]))
        self.assertEqual((error.status_code, error.field),
                         (409, "device_id"))

    def test_first_batch_dispatches_in_input_order_earlier_item_wins(self):
        # bob's inbox holds five unacked messages (a1..a3, b1..b2). The
        # first job leases all five; the second finds nothing and finishes
        # succeeded with lease_id null within the same transaction.
        self._queue("j1", "j2")
        body, status = self._batch(job_ids=["j1", "j2"])
        self.assertEqual(status, 201)
        self.assertEqual(list(body), ["device_id", "results"])
        self.assertEqual(body["device_id"], "bob")
        self.assertEqual(body["results"], [
            {"job_id": "j1", "state": "running", "lease_id": "j1"},
            {"job_id": "j2", "state": "succeeded", "lease_id": None},
        ])
        for item in body["results"]:
            self.assertEqual(list(item), ["job_id", "state", "lease_id"])
        # The first job's lease is the five inbox messages and is active.
        lease = self.service.inbox_lease_get("bob", "j1")
        self.assertEqual(lease["state"], "active")
        self.assertEqual(lease["limit"], 100)
        self.assertEqual(
            [(m["session_id"], m["message_id"]) for m in lease["messages"]],
            [(self.sid1, "a1"), (self.sid1, "a2"), (self.sid1, "a3"),
             (self.sid2, "b1"), (self.sid2, "b2")])

    def test_empty_inbox_succeeds_every_job(self) -> None:
        self._queue("j1")
        self._job(job_id="j1", op="dispatch")  # leases the whole inbox
        self._queue("empty")
        body, status = self._batch(job_ids=["empty"])
        self.assertEqual(status, 201)
        self.assertEqual(body["results"], [
            {"job_id": "empty", "state": "succeeded", "lease_id": None}])

    def test_per_item_errors_use_indexed_fields(self) -> None:
        self._queue("j1")
        # Unknown job.
        error = self._error(lambda: self._batch(job_ids=["ghost"]))
        self.assertEqual((error.status_code, error.field),
                         (404, "items[0].job_id"))
        # A job owned by another device.
        self._queue("jb", device_id="bob2")
        error = self._error(lambda: self._batch(job_ids=["jb"]))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[0].job_id"))
        # A job_id already occupied by an unrelated inbox lease.
        self.service.inbox_claim(
            "bob", {"lease_id": "occ", "limit": 100})
        self._queue("occ")
        error = self._error(lambda: self._batch(job_ids=["occ"]))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[0].job_id"))

    def test_first_error_in_array_order_wins_and_nothing_is_written(
            self) -> None:
        self._queue("j1")
        # Item 0 is fine, item 1 unknown: items[1] is reported and item
        # 0's dispatch is not applied.
        error = self._error(lambda: self._batch(job_ids=["j1", "ghost"]))
        self.assertEqual((error.status_code, error.field),
                         (404, "items[1].job_id"))
        body, _ = self._job(job_id="j1", op="status")
        self.assertEqual((body["state"], body["lease_id"]),
                         ("pending", None))
        # An earlier bad item wins over a later one.
        error = self._error(lambda: self._batch(job_ids=["ghost", "j1"]))
        self.assertEqual((error.status_code, error.field),
                         (404, "items[0].job_id"))
        body, _ = self._job(job_id="j1", op="status")
        self.assertEqual((body["state"], body["lease_id"]),
                         ("pending", None))

    def test_all_replays_answer_200_without_writing(self) -> None:
        self._queue("j1", "j2")
        first, status = self._batch(job_ids=["j1", "j2"])
        self.assertEqual(status, 201)
        replay, status = self._batch(job_ids=["j1", "j2"])
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)
        # A batch of terminal jobs (succeeded + cancelled) is all replay.
        self._queue("jc")
        self.service.inbox_job({"device_id": "bob", "job_id": "jc",
                                "op": "cancel", "cancellation_id": "c"})
        replay, status = self._batch(job_ids=["j2", "jc"])
        self.assertEqual(status, 200)
        self.assertEqual(replay["results"], [
            {"job_id": "j2", "state": "succeeded", "lease_id": None},
            {"job_id": "jc", "state": "cancelled", "lease_id": None}])

    def test_partial_replay_conflicts_with_first_replayed_item(self) -> None:
        self._queue("j1", "j2", "j3")
        self._job(job_id="j1", op="dispatch")  # j1 now running
        # The replay sits at index 0.
        error = self._error(lambda: self._batch(job_ids=["j1", "j2"]))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[0].job_id"))
        # ... and at index 1.
        error = self._error(lambda: self._batch(job_ids=["j2", "j1"]))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[1].job_id"))
        # Nothing was dispatched for the fresh item.
        body, _ = self._job(job_id="j2", op="status")
        self.assertEqual((body["state"], body["lease_id"]),
                         ("pending", None))

    def test_earlier_items_fresh_lease_withholds_messages_from_later(self):
        # Three pending jobs; the first leases the five messages, so the
        # next two both end succeeded in input order.
        self._queue("j1", "j2", "j3")
        body, status = self._batch(job_ids=["j1", "j2", "j3"])
        self.assertEqual(status, 201)
        self.assertEqual([item["state"] for item in body["results"]],
                         ["running", "succeeded", "succeeded"])
        self.assertEqual([item["lease_id"] for item in body["results"]],
                         ["j1", None, None])


class InboxJobDispatchBatchPersistenceTest(_BatchMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self._build()
        self.path = os.path.join(self.directory, "state.json")
        self.state_store = attach_persistence(self.service, self.path)

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_batch_advances_one_generation_replays_and_failures_none(
            self) -> None:
        self._queue("j1", "j2")
        before = self.state_store.commit_seq
        _, status = self._batch(job_ids=["j1", "j2"])
        self.assertEqual(status, 201)
        self.assertEqual(self.state_store.commit_seq, before + 1)
        _, status = self._batch(job_ids=["j1", "j2"])
        self.assertEqual(status, 200)
        self.assertEqual(self.state_store.commit_seq, before + 1)
        with self.assertRaises(ServiceError):
            self._batch(job_ids=["ghost"])
        self.assertEqual(self.state_store.commit_seq, before + 1)
        # A partial replay (j1 already dispatched, j3 still pending) does
        # not advance either (queueing j3 itself advances once).
        self._queue("j3")
        queued = self.state_store.commit_seq
        with self.assertRaises(ServiceError):
            self._batch(job_ids=["j1", "j3"])
        self.assertEqual(self.state_store.commit_seq, queued)

    def test_failed_batch_writes_nothing(self) -> None:
        self._queue("j1")
        before = self.state_store.commit_seq
        with self.assertRaises(ServiceError) as caught:
            self._batch(job_ids=["j1", "ghost"])
        self.assertEqual(caught.exception.status_code, 404)
        self.assertEqual(self.state_store.commit_seq, before)
        body, _ = self._job(job_id="j1", op="status")
        self.assertEqual((body["state"], body["lease_id"]),
                         ("pending", None))
        with open(self.path, encoding="utf-8") as handle:
            document = json.load(handle)
        jobs = {item["job_id"]: item
                for item in document["redelivery_jobs"]}
        self.assertEqual(jobs["j1"]["state"], "pending")
        self.assertIsNone(jobs["j1"]["lease_id"])

    def test_state_file_shape(self) -> None:
        self._queue("j1", "j2")
        self._batch(job_ids=["j1", "j2"])
        with open(self.path, encoding="utf-8") as handle:
            raw = handle.read()
        document = json.loads(raw)
        # Compact, ASCII-free-form JSON without newlines.
        self.assertNotIn("\n", raw)
        jobs = {item["job_id"]: item
                for item in document["redelivery_jobs"]}
        for item in jobs.values():
            self.assertEqual(list(item), ["job_id", "device_id", "state",
                                          "lease_id", "recoveries",
                                          "cancellation_id", "cancelled_at"])
        self.assertEqual(jobs["j1"]["state"], "running")
        self.assertEqual(jobs["j1"]["lease_id"], "j1")
        self.assertEqual(jobs["j2"]["state"], "succeeded")
        self.assertIsNone(jobs["j2"]["lease_id"])
        for item in jobs.values():
            self.assertEqual(item["recoveries"], [])
            self.assertIsNone(item["cancellation_id"])
            self.assertIsNone(item["cancelled_at"])
        # The dispatch lease landed on delivery records with limit 100.
        leases = {lease["lease_id"]: lease
                  for record in document["delivery"]
                  for lease in record["leases"]}
        self.assertIn("j1", leases)
        self.assertEqual(leases["j1"]["limit"], 100)

    def test_restart_restores_batch_and_replays(self) -> None:
        self._queue("j1", "j2")
        self._batch(job_ids=["j1", "j2"])
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        body, status = restarted.inbox_job(
            {"device_id": "bob", "job_id": "j1", "op": "status"})
        self.assertEqual(status, 200)
        self.assertEqual(body, {"job_id": "j1", "device_id": "bob",
                                "state": "running", "lease_id": "j1"})
        body, _ = restarted.inbox_job(
            {"device_id": "bob", "job_id": "j2", "op": "status"})
        self.assertEqual((body["state"], body["lease_id"]),
                         ("succeeded", None))
        # The dispatch lease survived the restart.
        lease = restarted.inbox_lease_get("bob", "j1")
        self.assertEqual(lease["state"], "active")
        # The committed batch still replays as 200 without writing.
        body, status = restarted.inbox_job_dispatch_batch(
            {"device_id": "bob",
             "items": [{"job_id": "j1"}, {"job_id": "j2"}]})
        self.assertEqual(status, 200)
        self.assertEqual(body["results"], [
            {"job_id": "j1", "state": "running", "lease_id": "j1"},
            {"job_id": "j2", "state": "succeeded", "lease_id": None}])
        self.assertTrue(restarted.persistence_integrity()["consistent"])

    def test_save_failure_rolls_back_the_whole_batch(self) -> None:
        import e2ee_backend.persistence as persistence_mod

        real_fsync = persistence_mod.os.fsync
        calls = {"n": 0}

        def fail_first_directory_fsync(fd: int) -> None:  # noqa: ANN001
            if os.fstat(fd).st_mode & 0o170000 == 0o040000:
                calls["n"] += 1
                if calls["n"] == 1:
                    raise OSError("transient directory fsync failure")
            real_fsync(fd)

        self._queue("j1", "j2")
        generation = self.state_store.commit_seq
        persistence_mod.os.fsync = fail_first_directory_fsync
        try:
            with self.assertRaises(PersistenceUnavailable):
                self._batch(job_ids=["j1", "j2"])
        finally:
            persistence_mod.os.fsync = real_fsync
        self.assertEqual(self.state_store.commit_seq, generation)
        # Neither job moved (memory was rolled back from the on-disk state).
        for job_id in ("j1", "j2"):
            body, _ = self._job(job_id=job_id, op="status")
            self.assertEqual((body["state"], body["lease_id"]),
                             ("pending", None))
        body, status = self._batch(job_ids=["j1", "j2"])
        self.assertEqual(status, 201)
        self.assertEqual(body["results"], [
            {"job_id": "j1", "state": "running", "lease_id": "j1"},
            {"job_id": "j2", "state": "succeeded", "lease_id": None}])
        self.assertEqual(self.state_store.commit_seq, generation + 1)


class InboxJobDispatchBatchHTTPTest(_BatchMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()
        self.server, self.service = create_server(
            "127.0.0.1", 0, self.service)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        self._queue("j1", "j2")

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _request(self, payload, raw_body=None):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        body = raw_body if raw_body is not None else json.dumps(payload)
        conn.request("POST", "/v1/inbox-jobs/dispatch-batch", body=body,
                     headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        data = response.read().decode("utf-8")
        conn.close()
        return response.status, (json.loads(data) if data else None), data

    def test_dispatch_batch_over_http(self) -> None:
        status, body, raw = self._request(
            {"device_id": "bob",
             "items": [{"job_id": "j1"}, {"job_id": "j2"}]})
        self.assertEqual(status, 201)
        self.assertEqual(body, {"device_id": "bob", "results": [
            {"job_id": "j1", "state": "running", "lease_id": "j1"},
            {"job_id": "j2", "state": "succeeded", "lease_id": None}]})
        self.assertLess(raw.index('"device_id"'), raw.index('"results"'))
        first = raw.index('"job_id"')
        self.assertLess(first, raw.index('"state"'))
        self.assertLess(raw.index('"state"'), raw.index('"lease_id"'))
        # The full replay answers 200 byte-identically.
        status, body, raw2 = self._request(
            {"device_id": "bob",
             "items": [{"job_id": "j1"}, {"job_id": "j2"}]})
        self.assertEqual(status, 200)
        self.assertEqual(raw2, raw)

    def test_errors_over_http(self) -> None:
        status, body, _ = self._request(None, raw_body="{")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "request_body")
        self.assertEqual(list(body), ["message", "field"])
        status, body, _ = self._request([1, 2])
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "request_body")
        status, body, _ = self._request({"device_id": "bob", "items": []})
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "items")
        status, body, _ = self._request(
            {"device_id": "bob", "items": [{}]})
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "items[0].job_id")
        status, body, _ = self._request(
            {"device_id": "bob",
             "items": [{"job_id": "j1", "op": "x"}]})
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "items[0]")
        status, body, _ = self._request(
            {"device_id": "ghost", "items": [{"job_id": "j1"}]})
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "device_id")
        status, body, _ = self._request(
            {"device_id": "bob", "items": [{"job_id": "ghost"}]})
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "items[0].job_id")
        # Partial replay: dispatch j1 alone, then mix it with the fresh j2.
        status, _, _ = self._request(
            {"device_id": "bob", "items": [{"job_id": "j1"}]})
        self.assertEqual(status, 201)
        status, body, _ = self._request(
            {"device_id": "bob",
             "items": [{"job_id": "j1"}, {"job_id": "j2"}]})
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "items[0].job_id")
        # The failed batch wrote nothing: j2 is still pending and now
        # dispatches as the first item of an all-first-time batch.
        status, body, _ = self._request(
            {"device_id": "bob", "items": [{"job_id": "j2"}]})
        self.assertEqual(status, 201)
        self.assertEqual(body["results"], [
            {"job_id": "j2", "state": "succeeded", "lease_id": None}])


if __name__ == "__main__":
    unittest.main()
