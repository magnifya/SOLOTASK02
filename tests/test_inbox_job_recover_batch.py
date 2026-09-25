"""Tests for ``POST /v1/inbox-jobs/recover-batch``.

An atomic batch recovery of 1:1-inbox redelivery jobs: the body carries a
``device_id`` and a non-empty ``items`` array of ``job_id``/``recovery_id``
pairs (both unique across items). Every item is prechecked in array order
with the single-job ``op=recover`` rules (first error aborts the batch,
its ``field`` prefixed to ``items[i].``); only then does the batch claim
in input order and commit once. A batch whose items all replay their
committed recoveries answers 200 and writes nothing; a partial replay
conflicts 409 with the first replayed item's ``items[i].recovery_id``.
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
    PersistenceUnavailable, StateFileError, attach_persistence)
from e2ee_backend.service import DeviceService, ServiceError
from e2ee_backend.http_app import create_server


_PAST = "2000-01-01T00:00:00.000000+00:00"


class _BatchMixin(InboxMixin):
    def _error(self, callable_):
        with self.assertRaises(ServiceError) as caught:
            callable_()
        return caught.exception

    def _job(self, device_id="bob", job_id="j1", op="queue", **extra):
        payload = {"device_id": device_id, "job_id": job_id, "op": op}
        payload.update(extra)
        return self.service.inbox_job(payload)

    def _dispatch(self, job_id="j1", device_id="bob"):
        self._job(device_id=device_id, job_id=job_id)
        return self._job(device_id=device_id, job_id=job_id, op="dispatch")

    def _expire(self, lease_id: str) -> None:
        with self.service.store._lock:
            for state in self.service.store._delivery.values():
                for lease in state.leases:
                    if lease.lease_id == lease_id:
                        lease.leased_until = _PAST

    def _batch(self, device_id="bob", items=None, **extra):
        payload = {"device_id": device_id,
                   "items": items if items is not None else []}
        payload.update(extra)
        return self.service.inbox_job_recover_batch(payload)

    def _two_running_jobs(self):
        # j1 leases the whole inbox, its lease expires, then j2 leases it
        # again: two running jobs whose current leases are both expired.
        self._dispatch(job_id="j1")
        self._expire("j1")
        self._dispatch(job_id="j2")
        self._expire("j2")


class InboxJobRecoverBatchServiceTest(_BatchMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def test_bad_body_and_device_id_errors(self) -> None:
        for payload in (None, [], "x", 3):
            error = self._error(
                lambda payload=payload:
                self.service.inbox_job_recover_batch(payload))
            self.assertEqual((error.status_code, error.field),
                             (400, "request_body"), payload)
        for payload in (
                {"items": [{"job_id": "j1", "recovery_id": "r1"}]},
                {"device_id": "",
                 "items": [{"job_id": "j1", "recovery_id": "r1"}]},
                {"device_id": 4,
                 "items": [{"job_id": "j1", "recovery_id": "r1"}]},
                {"device_id": None,
                 "items": [{"job_id": "j1", "recovery_id": "r1"}]}):
            error = self._error(
                lambda payload=payload:
                self.service.inbox_job_recover_batch(payload))
            self.assertEqual((error.status_code, error.field),
                             (400, "device_id"), payload)

    def test_items_shape_errors(self) -> None:
        for items in (None, {}, "x", 3, []):
            error = self._error(lambda items=items: self._batch(items=items))
            self.assertEqual((error.status_code, error.field),
                             (400, "items"), items)
        error = self._error(lambda: self._batch(items=["x"]))
        self.assertEqual((error.status_code, error.field), (400, "items[0]"))
        error = self._error(lambda: self._batch(items=[[]]))
        self.assertEqual((error.status_code, error.field), (400, "items[0]"))

    def test_item_field_errors_carry_the_index(self) -> None:
        good = {"job_id": "j1", "recovery_id": "r1"}
        cases = (
            ([{"recovery_id": "r1"}], "items[0].job_id"),
            ([{"job_id": "", "recovery_id": "r1"}], "items[0].job_id"),
            ([{"job_id": 4, "recovery_id": "r1"}], "items[0].job_id"),
            ([{"job_id": "j1"}], "items[0].recovery_id"),
            ([{"job_id": "j1", "recovery_id": ""}], "items[0].recovery_id"),
            ([{"job_id": "j1", "recovery_id": None}], "items[0].recovery_id"),
            ([good, {"recovery_id": "r2"}], "items[1].job_id"),
            ([good, {"job_id": "j2"}], "items[1].recovery_id"),
        )
        for items, field in cases:
            error = self._error(lambda items=items: self._batch(items=items))
            self.assertEqual((error.status_code, error.field),
                             (400, field), items)

    def test_item_with_extra_keys_rejected(self) -> None:
        # An item may only carry job_id and recovery_id; any other key is
        # 400/items[i] and aborts the batch before anything is written.
        self._two_running_jobs()
        cases = (
            [{"job_id": "j1", "recovery_id": "r1", "extra": 1}],
            [{"job_id": "j1", "recovery_id": "r1", "op": "recover"}],
            [{"job_id": "j1", "recovery_id": "r1",
              "cancellation_id": "c1"}],
            [{"job_id": "j1", "recovery_id": "r1"},
             {"job_id": "j2", "recovery_id": "r2", "extra": None}],
        )
        for items in cases:
            index = 1 if len(items) > 1 else 0
            error = self._error(lambda items=items: self._batch(items=items))
            self.assertEqual((error.status_code, error.field),
                             (400, f"items[{index}]"), items)
        # Nothing was recovered: both jobs are still running on their
        # dispatch leases.
        for job_id in ("j1", "j2"):
            body, _ = self._job(job_id=job_id, op="status")
            self.assertEqual((body["state"], body["lease_id"]),
                             ("running", job_id))

    def test_duplicate_job_id_or_recovery_id_rejected(self) -> None:
        items = [{"job_id": "j1", "recovery_id": "r1"},
                 {"job_id": "j1", "recovery_id": "r2"}]
        error = self._error(lambda: self._batch(items=items))
        self.assertEqual((error.status_code, error.field), (400, "items[1]"))
        items = [{"job_id": "j1", "recovery_id": "r1"},
                 {"job_id": "j2", "recovery_id": "r1"}]
        error = self._error(lambda: self._batch(items=items))
        self.assertEqual((error.status_code, error.field), (400, "items[1]"))
        # The same pair on the first item is fine shape-wise (it fails
        # later, in the store, as an unknown job).
        error = self._error(lambda: self._batch(
            items=[{"job_id": "ghost", "recovery_id": "r1"}]))
        self.assertEqual((error.status_code, error.field),
                         (404, "items[0].job_id"))

    def test_device_gate_precedes_item_checks(self) -> None:
        self._two_running_jobs()
        items = [{"job_id": "j1", "recovery_id": "r1"}]
        error = self._error(lambda: self._batch(device_id="ghost",
                                                items=items))
        self.assertEqual((error.status_code, error.field),
                         (409, "device_id"))
        self.service.store.revoke_device("bob")
        error = self._error(lambda: self._batch(items=items))
        self.assertEqual((error.status_code, error.field),
                         (409, "device_id"))

    def test_first_batch_recovers_in_input_order(self) -> None:
        self._two_running_jobs()
        body, status = self._batch(items=[
            {"job_id": "j1", "recovery_id": "r1"},
            {"job_id": "j2", "recovery_id": "r2"}])
        self.assertEqual(status, 201)
        self.assertEqual(list(body), ["device_id", "results"])
        self.assertEqual(body["device_id"], "bob")
        # j1 claims the whole inbox under r1; j2 then finds nothing left
        # and finishes succeeded without a lease.
        self.assertEqual(body["results"], [
            {"job_id": "j1", "state": "running", "lease_id": "r1"},
            {"job_id": "j2", "state": "succeeded", "lease_id": None},
        ])
        for item in body["results"]:
            self.assertEqual(list(item), ["job_id", "state", "lease_id"])
        lease = self.service.inbox_lease_get("bob", "r1")
        self.assertEqual(lease["state"], "active")
        self.assertEqual(lease["limit"], 100)
        self.assertEqual([m["message_id"] for m in lease["messages"]],
                         ["a1", "a2", "a3", "b1", "b2"])
        # The reversed input order gives j2 the messages instead.
        self._build()
        self._two_running_jobs()
        body, status = self._batch(items=[
            {"job_id": "j2", "recovery_id": "r2"},
            {"job_id": "j1", "recovery_id": "r1"}])
        self.assertEqual(status, 201)
        self.assertEqual(body["results"], [
            {"job_id": "j2", "state": "running", "lease_id": "r2"},
            {"job_id": "j1", "state": "succeeded", "lease_id": None},
        ])

    def test_per_item_errors_use_indexed_fields(self) -> None:
        self._two_running_jobs()
        # A still-valid lease cannot be recovered (j3 claims the inbox
        # while j1's and j2's expired leases no longer withhold it).
        self._dispatch(job_id="j3")
        error = self._error(lambda: self._batch(items=[
            {"job_id": "j3", "recovery_id": "r3"}]))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[0].lease_id"))
        self._expire("j3")
        # Unknown job.
        error = self._error(lambda: self._batch(items=[
            {"job_id": "j1", "recovery_id": "rx"},
            {"job_id": "ghost", "recovery_id": "r2"}]))
        self.assertEqual((error.status_code, error.field),
                         (404, "items[1].job_id"))
        # A job owned by another device.
        self._dispatch(job_id="jb", device_id="bob2")
        error = self._error(lambda: self._batch(items=[
            {"job_id": "jb", "recovery_id": "rb"}]))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[0].job_id"))
        # recovery_id equals an existing job_id (its own included).
        error = self._error(lambda: self._batch(items=[
            {"job_id": "j1", "recovery_id": "j2"}]))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[0].recovery_id"))
        error = self._error(lambda: self._batch(items=[
            {"job_id": "j1", "recovery_id": "j1"}]))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[0].recovery_id"))
        # recovery_id occupied by an unrelated inbox lease.
        self.service.inbox_claim("bob", {"lease_id": "L-used", "limit": 1})
        error = self._error(lambda: self._batch(items=[
            {"job_id": "j1", "recovery_id": "L-used"}]))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[0].recovery_id"))
        # recovery_id already committed on another job.
        self._job(op="recover", job_id="j1", recovery_id="r1")
        error = self._error(lambda: self._batch(items=[
            {"job_id": "j2", "recovery_id": "r1"}]))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[0].recovery_id"))
        # A pending job cannot be recovered.
        self._job(job_id="jp")
        error = self._error(lambda: self._batch(items=[
            {"job_id": "jp", "recovery_id": "rp"}]))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[0].job_id"))

    def test_first_error_in_array_order_wins_and_nothing_is_written(
            self) -> None:
        self._two_running_jobs()
        # Item 0 is fine, item 1 unknown, item 2 also bad: items[1] is
        # reported and item 0's recovery is not applied.
        error = self._error(lambda: self._batch(items=[
            {"job_id": "j1", "recovery_id": "r1"},
            {"job_id": "ghost", "recovery_id": "r2"},
            {"job_id": "j2", "recovery_id": "j1"}]))
        self.assertEqual((error.status_code, error.field),
                         (404, "items[1].job_id"))
        body, _ = self._job(op="status")
        self.assertEqual((body["state"], body["lease_id"]),
                         ("running", "j1"))
        error = self._error(lambda: self.service.inbox_lease_get("bob", "r1"))
        self.assertEqual((error.status_code, error.field),
                         (404, "lease_id"))
        # An earlier bad item wins over a later one.
        error = self._error(lambda: self._batch(items=[
            {"job_id": "ghost", "recovery_id": "r1"},
            {"job_id": "j2", "recovery_id": "j1"}]))
        self.assertEqual((error.status_code, error.field),
                         (404, "items[0].job_id"))

    def test_all_replays_answer_200_without_writing(self) -> None:
        self._two_running_jobs()
        first, status = self._batch(items=[
            {"job_id": "j1", "recovery_id": "r1"},
            {"job_id": "j2", "recovery_id": "r2"}])
        self.assertEqual(status, 201)
        replay, status = self._batch(items=[
            {"job_id": "j1", "recovery_id": "r1"},
            {"job_id": "j2", "recovery_id": "r2"}])
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)
        # A single-item replay also answers the current view with 200, even
        # after the job terminated.
        self.service.inbox_lease_complete(
            "bob", "r1", {"completion_id": "c1", "outcome": "failed"})
        replay, status = self._batch(items=[
            {"job_id": "j1", "recovery_id": "r1"}])
        self.assertEqual(status, 200)
        self.assertEqual(replay["results"], [
            {"job_id": "j1", "state": "failed", "lease_id": "r1"}])

    def test_partial_replay_conflicts_with_first_replayed_item(self) -> None:
        self._two_running_jobs()
        _, status = self._batch(items=[{"job_id": "j1", "recovery_id": "r1"}])
        self.assertEqual(status, 201)
        self._expire("r1")
        self._dispatch(job_id="j3")
        self._expire("j3")
        # The replay sits at index 0.
        error = self._error(lambda: self._batch(items=[
            {"job_id": "j1", "recovery_id": "r1"},
            {"job_id": "j3", "recovery_id": "r3"}]))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[0].recovery_id"))
        # ... and at index 1.
        error = self._error(lambda: self._batch(items=[
            {"job_id": "j3", "recovery_id": "r3"},
            {"job_id": "j1", "recovery_id": "r1"}]))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[1].recovery_id"))
        # Nothing was written for the fresh item.
        body, _ = self._job(job_id="j3", op="status")
        self.assertEqual((body["state"], body["lease_id"]),
                         ("running", "j3"))

    def test_empty_selection_succeeds_the_job(self) -> None:
        self._dispatch()
        self._expire("j1")
        self.service.sync_session_ack(
            self.sid1, {"device_id": "bob", "cursor": 3})
        self.service.sync_session_ack(
            self.sid2, {"device_id": "bob", "cursor": 2})
        body, status = self._batch(items=[
            {"job_id": "j1", "recovery_id": "empty"}])
        self.assertEqual(status, 201)
        self.assertEqual(body["results"], [
            {"job_id": "j1", "state": "succeeded", "lease_id": None}])
        error = self._error(
            lambda: self.service.inbox_lease_get("bob", "empty"))
        self.assertEqual((error.status_code, error.field),
                         (404, "lease_id"))

    def test_recovery_lease_completion_terminates_the_job(self) -> None:
        self._two_running_jobs()
        self._batch(items=[{"job_id": "j1", "recovery_id": "r1"},
                           {"job_id": "j2", "recovery_id": "r2"}])
        self.service.inbox_lease_complete(
            "bob", "r1", {"completion_id": "c1", "outcome": "delivered"})
        body, _ = self._job(op="status")
        self.assertEqual((body["state"], body["lease_id"]),
                         ("succeeded", "r1"))


class InboxJobRecoverBatchPersistenceTest(_BatchMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self._build()
        self.path = os.path.join(self.directory, "state.json")
        self.state_store = attach_persistence(self.service, self.path)

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_batch_advances_one_generation_replays_and_failures_none(
            self) -> None:
        self._two_running_jobs()
        before = self.state_store.commit_seq
        _, status = self._batch(items=[{"job_id": "j1", "recovery_id": "r1"},
                                       {"job_id": "j2", "recovery_id": "r2"}])
        self.assertEqual(status, 201)
        self.assertEqual(self.state_store.commit_seq, before + 1)
        _, status = self._batch(items=[{"job_id": "j1", "recovery_id": "r1"},
                                       {"job_id": "j2", "recovery_id": "r2"}])
        self.assertEqual(status, 200)
        self.assertEqual(self.state_store.commit_seq, before + 1)
        with self.assertRaises(ServiceError):
            self._batch(items=[{"job_id": "ghost", "recovery_id": "r9"}])
        self.assertEqual(self.state_store.commit_seq, before + 1)

    def test_failed_batch_writes_nothing(self) -> None:
        self._two_running_jobs()
        before = self.state_store.commit_seq
        with self.assertRaises(ServiceError) as caught:
            self._batch(items=[{"job_id": "j1", "recovery_id": "r1"},
                               {"job_id": "ghost", "recovery_id": "r2"}])
        self.assertEqual(caught.exception.status_code, 404)
        self.assertEqual(self.state_store.commit_seq, before)
        body, _ = self._job(op="status")
        self.assertEqual((body["state"], body["lease_id"]),
                         ("running", "j1"))
        with open(self.path, encoding="utf-8") as handle:
            document = json.load(handle)
        jobs = {item["job_id"]: item
                for item in document["redelivery_jobs"]}
        self.assertEqual(jobs["j1"]["recoveries"], [])

    def test_state_file_shape(self) -> None:
        self._two_running_jobs()
        self._job(job_id="pending")
        self._batch(items=[{"job_id": "j1", "recovery_id": "r1"},
                           {"job_id": "j2", "recovery_id": "r2"}])
        with open(self.path, encoding="utf-8") as handle:
            raw = handle.read()
        document = json.loads(raw)
        # Compact, ASCII-free-form JSON without newlines.
        self.assertNotIn("\n", raw)
        items = {item["job_id"]: item
                 for item in document["redelivery_jobs"]}
        for item in items.values():
            self.assertEqual(list(item), ["job_id", "device_id", "state",
                                          "lease_id", "recoveries",
                                          "cancellation_id",
                                          "cancelled_at"])
        self.assertEqual(items["pending"]["recoveries"], [])
        self.assertEqual(items["j1"]["recoveries"], [
            {"recovery_id": "r1", "lease_id": "r1"}])
        self.assertEqual(items["j2"]["recoveries"], [
            {"recovery_id": "r2", "lease_id": None}])
        for item in items.values():
            for record in item["recoveries"]:
                self.assertEqual(list(record), ["recovery_id", "lease_id"])

    def test_restart_restores_batch_and_replays(self) -> None:
        self._two_running_jobs()
        self._batch(items=[{"job_id": "j1", "recovery_id": "r1"},
                           {"job_id": "j2", "recovery_id": "r2"}])
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        body, status = restarted.inbox_job(
            {"device_id": "bob", "job_id": "j1", "op": "status"})
        self.assertEqual(status, 200)
        self.assertEqual(body, {"job_id": "j1", "device_id": "bob",
                                "state": "running", "lease_id": "r1"})
        body, _ = restarted.inbox_job(
            {"device_id": "bob", "job_id": "j2", "op": "status"})
        self.assertEqual((body["state"], body["lease_id"]),
                         ("succeeded", None))
        self.assertEqual(
            restarted.inbox_lease_get("bob", "r1")["state"], "active")
        # The committed batch still replays as 200 without writing.
        body, status = restarted.inbox_job_recover_batch(
            {"device_id": "bob", "items": [
                {"job_id": "j1", "recovery_id": "r1"},
                {"job_id": "j2", "recovery_id": "r2"}]})
        self.assertEqual(status, 200)
        self.assertEqual(body["results"], [
            {"job_id": "j1", "state": "running", "lease_id": "r1"},
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

        self._two_running_jobs()
        generation = self.state_store.commit_seq
        persistence_mod.os.fsync = fail_first_directory_fsync
        try:
            with self.assertRaises(PersistenceUnavailable):
                self._batch(items=[{"job_id": "j1", "recovery_id": "r1"},
                                   {"job_id": "j2", "recovery_id": "r2"}])
        finally:
            persistence_mod.os.fsync = real_fsync
        self.assertEqual(self.state_store.commit_seq, generation)
        # Neither job moved; memory was rolled back from the on-disk state
        # (which also discards the in-memory expiry hack), so expire the
        # leases again before the ids can be reused.
        for job_id in ("j1", "j2"):
            body, _ = self._job(job_id=job_id, op="status")
            self.assertEqual((body["state"], body["lease_id"]),
                             ("running", job_id))
        self._expire("j1")
        self._expire("j2")
        body, status = self._batch(items=[
            {"job_id": "j1", "recovery_id": "r1"},
            {"job_id": "j2", "recovery_id": "r2"}])
        self.assertEqual(status, 201)
        self.assertEqual(body["results"], [
            {"job_id": "j1", "state": "running", "lease_id": "r1"},
            {"job_id": "j2", "state": "succeeded", "lease_id": None}])
        self.assertEqual(self.state_store.commit_seq, generation + 1)

    def test_restore_rejects_contradictory_batch_written_history(
            self) -> None:
        self._two_running_jobs()
        self._batch(items=[{"job_id": "j1", "recovery_id": "r1"},
                           {"job_id": "j2", "recovery_id": "r2"}])
        with open(self.path, encoding="utf-8") as handle:
            document = json.load(handle)
        # The second job's empty recovery is not its final state.
        document["redelivery_jobs"][1]["state"] = "running"
        document.pop("integrity_log_version", None)
        bad_path = os.path.join(self.directory, "bad.json")
        with open(bad_path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        original = open(bad_path, "rb").read()
        restarted = DeviceService()
        with self.assertRaises(StateFileError):
            attach_persistence(restarted, bad_path)
        with open(bad_path, "rb") as handle:
            self.assertEqual(handle.read(), original)


class InboxJobRecoverBatchHTTPTest(_BatchMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()
        self.server, self.service = create_server(
            "127.0.0.1", 0, self.service)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        self._two_running_jobs()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _request(self, payload, raw_body=None):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        body = raw_body if raw_body is not None else json.dumps(payload)
        conn.request("POST", "/v1/inbox-jobs/recover-batch", body=body,
                     headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        data = response.read().decode("utf-8")
        conn.close()
        return response.status, (json.loads(data) if data else None), data

    def test_recover_batch_over_http(self) -> None:
        status, body, raw = self._request(
            {"device_id": "bob", "items": [
                {"job_id": "j1", "recovery_id": "r1"},
                {"job_id": "j2", "recovery_id": "r2"}]})
        self.assertEqual(status, 201)
        self.assertEqual(body, {"device_id": "bob", "results": [
            {"job_id": "j1", "state": "running", "lease_id": "r1"},
            {"job_id": "j2", "state": "succeeded", "lease_id": None}]})
        self.assertLess(raw.index('"device_id"'), raw.index('"results"'))
        first = raw.index('"job_id"')
        self.assertLess(first, raw.index('"state"'))
        self.assertLess(raw.index('"state"'), raw.index('"lease_id"'))
        # The full replay answers 200 byte-identically.
        status, body, raw2 = self._request(
            {"device_id": "bob", "items": [
                {"job_id": "j1", "recovery_id": "r1"},
                {"job_id": "j2", "recovery_id": "r2"}]})
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
            {"device_id": "bob", "items": [{"job_id": "j1"}]})
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "items[0].recovery_id")
        status, body, _ = self._request(
            {"device_id": "ghost",
             "items": [{"job_id": "j1", "recovery_id": "r1"}]})
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "device_id")
        status, body, _ = self._request(
            {"device_id": "bob",
             "items": [{"job_id": "ghost", "recovery_id": "r1"}]})
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "items[0].job_id")
        status, body, _ = self._request(
            {"device_id": "bob",
             "items": [{"job_id": "j1", "recovery_id": "r1"},
                       {"job_id": "j2", "recovery_id": "j1"}]})
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "items[1].recovery_id")
        # The failed batch wrote nothing: both jobs are still recoverable.
        status, body, _ = self._request(
            {"device_id": "bob", "items": [
                {"job_id": "j1", "recovery_id": "r1"},
                {"job_id": "j2", "recovery_id": "r2"}]})
        self.assertEqual(status, 201)


if __name__ == "__main__":
    unittest.main()
