"""Tests for ``POST /v1/inbox-jobs/complete-batch``.

An atomic batch completion of 1:1-inbox redelivery leases: the body
carries a ``device_id`` and a non-empty ``items`` array of
``lease_id``/``completion_id``/``outcome`` triples (neither id repeats
across items). Every item is prechecked in array order with the
single-lease completion rules (first error aborts the batch, its
``field`` prefixed to ``items[i].``); only then does the batch complete
in input order with one shared ``completed_at`` and commit once. A batch
whose items all replay their committed completions answers 200 and
writes nothing; a partial replay conflicts 409 with the first replayed
item's ``items[i].completion_id``.
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

    def _claim(self, lease_id, device_id="bob", limit=100):
        return self.service.inbox_claim(
            device_id, {"lease_id": lease_id, "limit": limit})

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
        return self.service.inbox_job_complete_batch(payload)


class InboxJobCompleteBatchServiceTest(_BatchMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def test_bad_body_and_device_id_errors(self) -> None:
        good = [{"lease_id": "j1", "completion_id": "c1",
                 "outcome": "delivered"}]
        for payload in (None, [], "x", 3):
            error = self._error(
                lambda payload=payload:
                self.service.inbox_job_complete_batch(payload))
            self.assertEqual((error.status_code, error.field),
                             (400, "request_body"), payload)
        for payload in (
                {"items": good},
                {"device_id": "", "items": good},
                {"device_id": 4, "items": good},
                {"device_id": None, "items": good}):
            error = self._error(
                lambda payload=payload:
                self.service.inbox_job_complete_batch(payload))
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

    def test_extra_top_level_key_rejected_with_that_field(self) -> None:
        item = {"lease_id": "j1", "completion_id": "c1",
                "outcome": "delivered"}
        error = self._error(lambda: self._batch(items=[item], op="done"))
        self.assertEqual((error.status_code, error.field), (400, "op"))
        error = self._error(lambda: self._batch(items=[item], bogus=1))
        self.assertEqual((error.status_code, error.field), (400, "bogus"))

    def test_item_field_errors_carry_the_index(self) -> None:
        good = {"lease_id": "j1", "completion_id": "c1",
                "outcome": "delivered"}
        cases = (
            ([{"completion_id": "c1", "outcome": "delivered"}],
             "items[0].lease_id"),
            ([{"lease_id": "", "completion_id": "c1",
               "outcome": "delivered"}], "items[0].lease_id"),
            ([{"lease_id": 4, "completion_id": "c1",
               "outcome": "delivered"}], "items[0].lease_id"),
            ([{"lease_id": "j1", "outcome": "delivered"}],
             "items[0].completion_id"),
            ([{"lease_id": "j1", "completion_id": "",
               "outcome": "delivered"}], "items[0].completion_id"),
            ([{"lease_id": "j1", "completion_id": None,
               "outcome": "delivered"}], "items[0].completion_id"),
            ([{"lease_id": "j1", "completion_id": "c1"}],
             "items[0].outcome"),
            ([{"lease_id": "j1", "completion_id": "c1", "outcome": None}],
             "items[0].outcome"),
            ([{"lease_id": "j1", "completion_id": "c1", "outcome": ""}],
             "items[0].outcome"),
            ([{"lease_id": "j1", "completion_id": "c1",
               "outcome": "Delivered"}], "items[0].outcome"),
            ([good, {"completion_id": "c2", "outcome": "failed"}],
             "items[1].lease_id"),
            ([good, {"lease_id": "j2", "completion_id": "c2"}],
             "items[1].outcome"),
        )
        for items, field in cases:
            error = self._error(lambda items=items: self._batch(items=items))
            self.assertEqual((error.status_code, error.field),
                             (400, field), items)

    def test_item_extra_keys_rejected_at_item_level(self) -> None:
        base = {"lease_id": "j1", "completion_id": "c1",
                "outcome": "delivered"}
        cases = (
            [dict(base, extra=1)],
            [dict(base, job_id="j1")],
            [dict(base), {"lease_id": "j2", "completion_id": "c2",
                          "outcome": "failed", "bogus": None}],
        )
        for items in cases:
            index = 0 if len(items) == 1 else 1
            error = self._error(lambda items=items: self._batch(items=items))
            self.assertEqual((error.status_code, error.field),
                             (400, f"items[{index}]"), items)

    def test_duplicate_ids_rejected_at_item_level(self) -> None:
        items = [{"lease_id": "j1", "completion_id": "c1",
                  "outcome": "delivered"},
                 {"lease_id": "j1", "completion_id": "c2",
                  "outcome": "failed"}]
        error = self._error(lambda: self._batch(items=items))
        self.assertEqual((error.status_code, error.field), (400, "items[1]"))
        items = [{"lease_id": "j1", "completion_id": "c1",
                  "outcome": "delivered"},
                 {"lease_id": "j2", "completion_id": "c1",
                  "outcome": "failed"}]
        error = self._error(lambda: self._batch(items=items))
        self.assertEqual((error.status_code, error.field), (400, "items[1]"))

    def test_device_gate_precedes_item_checks(self) -> None:
        self._dispatch(job_id="j1")
        items = [{"lease_id": "j1", "completion_id": "c1",
                  "outcome": "delivered"}]
        error = self._error(lambda: self._batch(device_id="ghost",
                                                items=items))
        self.assertEqual((error.status_code, error.field),
                         (409, "device_id"))
        self.service.store.revoke_device("bob")
        error = self._error(lambda: self._batch(items=items))
        self.assertEqual((error.status_code, error.field),
                         (409, "device_id"))

    def test_first_batch_completes_one_lease(self) -> None:
        self._dispatch(job_id="j1")
        body, status = self._batch(items=[
            {"lease_id": "j1", "completion_id": "c1",
             "outcome": "delivered"}])
        self.assertEqual(status, 201)
        self.assertEqual(list(body), ["device_id", "results"])
        self.assertEqual(body["device_id"], "bob")
        self.assertEqual(len(body["results"]), 1)
        item = body["results"][0]
        self.assertEqual(list(item), ["lease_id", "completion_id",
                                      "outcome", "completed_at"])
        self.assertEqual(item["lease_id"], "j1")
        self.assertEqual(item["completion_id"], "c1")
        self.assertEqual(item["outcome"], "delivered")
        self.assertTrue(item["completed_at"].endswith("+00:00"))
        # The running job moved to succeeded in the same transaction.
        job, _ = self._job(job_id="j1", op="status")
        self.assertEqual(job["state"], "succeeded")
        # The lease is now completed history.
        lease = self.service.inbox_lease_get("bob", "j1")
        self.assertEqual(lease["state"], "completed")
        self.assertEqual(lease["completion"]["completion_id"], "c1")

    def test_batch_two_distinct_active_leases_share_completed_at(self) -> None:
        # j1's dispatch leases all five messages. Expire it and recover so
        # the inbox is re-leased under r1; then take a second ordinary
        # lease L1 is impossible while r1 holds everything. Instead build
        # two leases on disjoint message sets by limiting j1 is not
        # available for dispatch (fixed 100). Claim L1 first with limit 2,
        # then dispatch j1: j1 claims the remaining three.
        self._claim("L1", limit=2)
        self._dispatch(job_id="j1")
        body, status = self._batch(items=[
            {"lease_id": "L1", "completion_id": "cL", "outcome": "failed"},
            {"lease_id": "j1", "completion_id": "cJ",
             "outcome": "delivered"}])
        self.assertEqual(status, 201)
        self.assertEqual([r["lease_id"] for r in body["results"]],
                         ["L1", "j1"])
        self.assertEqual(body["results"][0]["completed_at"],
                         body["results"][1]["completed_at"])
        self.assertEqual([r["outcome"] for r in body["results"]],
                         ["failed", "delivered"])
        # Only j1 is a redelivery job; it follows its delivered outcome.
        job, _ = self._job(job_id="j1", op="status")
        self.assertEqual(job["state"], "succeeded")

    def test_failed_outcome_moves_job_to_failed(self) -> None:
        self._dispatch(job_id="j1")
        _, status = self._batch(items=[
            {"lease_id": "j1", "completion_id": "c1", "outcome": "failed"}])
        self.assertEqual(status, 201)
        job, _ = self._job(job_id="j1", op="status")
        self.assertEqual(job["state"], "failed")

    def test_per_item_errors_use_indexed_fields(self) -> None:
        self._dispatch(job_id="j1")
        # Unknown lease.
        error = self._error(lambda: self._batch(items=[
            {"lease_id": "ghost", "completion_id": "c1",
             "outcome": "delivered"}]))
        self.assertEqual((error.status_code, error.field),
                         (404, "items[0].lease_id"))
        # A lease owned by another device (bob2's dispatch lease).
        self._dispatch(job_id="jb", device_id="bob2")
        error = self._error(lambda: self._batch(items=[
            {"lease_id": "jb", "completion_id": "cb",
             "outcome": "delivered"}]))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[0].lease_id"))
        # An unfinished but expired lease conflicts at lease_id.
        self._expire("j1")
        error = self._error(lambda: self._batch(items=[
            {"lease_id": "j1", "completion_id": "c1",
             "outcome": "delivered"}]))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[0].lease_id"))
        # An unfinished but released lease conflicts at lease_id.
        self._dispatch(job_id="j2")
        self.service.inbox_release("bob", "j2")
        error = self._error(lambda: self._batch(items=[
            {"lease_id": "j2", "completion_id": "c2",
             "outcome": "delivered"}]))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[0].lease_id"))

    def test_completed_lease_under_new_id_conflicts_at_completion_id(
            self) -> None:
        self._dispatch(job_id="j1")
        self._batch(items=[{"lease_id": "j1", "completion_id": "c1",
                            "outcome": "delivered"}])
        error = self._error(lambda: self._batch(items=[
            {"lease_id": "j1", "completion_id": "other",
             "outcome": "delivered"}]))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[0].completion_id"))

    def test_same_id_different_outcome_conflicts_at_completion_id(
            self) -> None:
        self._dispatch(job_id="j1")
        self._batch(items=[{"lease_id": "j1", "completion_id": "c1",
                            "outcome": "delivered"}])
        error = self._error(lambda: self._batch(items=[
            {"lease_id": "j1", "completion_id": "c1", "outcome": "failed"}]))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[0].completion_id"))

    def test_first_error_in_array_order_wins_and_nothing_is_written(
            self) -> None:
        self._claim("L1", limit=2)
        self._dispatch(job_id="j1")
        # Item 0 fine, item 1 unknown: items[1] reported and L1 not
        # completed.
        error = self._error(lambda: self._batch(items=[
            {"lease_id": "L1", "completion_id": "cL", "outcome": "failed"},
            {"lease_id": "ghost", "completion_id": "cX",
             "outcome": "delivered"}]))
        self.assertEqual((error.status_code, error.field),
                         (404, "items[1].lease_id"))
        self.assertEqual(
            self.service.inbox_lease_get("bob", "L1")["state"], "active")
        # An earlier bad item wins over a later good one.
        error = self._error(lambda: self._batch(items=[
            {"lease_id": "ghost", "completion_id": "cX",
             "outcome": "delivered"},
            {"lease_id": "j1", "completion_id": "cJ",
             "outcome": "delivered"}]))
        self.assertEqual((error.status_code, error.field),
                         (404, "items[0].lease_id"))
        job, _ = self._job(job_id="j1", op="status")
        self.assertEqual(job["state"], "running")

    def test_all_replays_answer_200_without_writing(self) -> None:
        self._claim("L1", limit=2)
        self._dispatch(job_id="j1")
        first, status = self._batch(items=[
            {"lease_id": "L1", "completion_id": "cL", "outcome": "failed"},
            {"lease_id": "j1", "completion_id": "cJ",
             "outcome": "delivered"}])
        self.assertEqual(status, 201)
        replay, status = self._batch(items=[
            {"lease_id": "L1", "completion_id": "cL", "outcome": "failed"},
            {"lease_id": "j1", "completion_id": "cJ",
             "outcome": "delivered"}])
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)
        # A single-item replay also answers 200, even after the device has
        # been revoked (the device gate still runs first, though).
        replay, status = self._batch(items=[
            {"lease_id": "j1", "completion_id": "cJ",
             "outcome": "delivered"}])
        self.assertEqual(status, 200)
        self.assertEqual(replay["results"][0]["completion_id"], "cJ")

    def test_partial_replay_conflicts_with_first_replayed_item(self) -> None:
        self._claim("L1", limit=2)
        self._dispatch(job_id="j1")
        self._batch(items=[{"lease_id": "j1", "completion_id": "cJ",
                            "outcome": "delivered"}])
        # The replay sits at index 0.
        error = self._error(lambda: self._batch(items=[
            {"lease_id": "j1", "completion_id": "cJ",
             "outcome": "delivered"},
            {"lease_id": "L1", "completion_id": "cL",
             "outcome": "failed"}]))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[0].completion_id"))
        # ... and at index 1.
        error = self._error(lambda: self._batch(items=[
            {"lease_id": "L1", "completion_id": "cL", "outcome": "failed"},
            {"lease_id": "j1", "completion_id": "cJ",
             "outcome": "delivered"}]))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[1].completion_id"))
        # The fresh item was not written.
        self.assertEqual(
            self.service.inbox_lease_get("bob", "L1")["state"], "active")


class InboxJobCompleteBatchPersistenceTest(_BatchMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self._build()
        self.path = os.path.join(self.directory, "state.json")
        self.state_store = attach_persistence(self.service, self.path)

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_batch_advances_one_generation_replays_and_failures_none(
            self) -> None:
        self._claim("L1", limit=2)
        self._dispatch(job_id="j1")
        items = [
            {"lease_id": "L1", "completion_id": "cL", "outcome": "failed"},
            {"lease_id": "j1", "completion_id": "cJ",
             "outcome": "delivered"}]
        before = self.state_store.commit_seq
        _, status = self._batch(items=items)
        self.assertEqual(status, 201)
        self.assertEqual(self.state_store.commit_seq, before + 1)
        _, status = self._batch(items=items)
        self.assertEqual(status, 200)
        self.assertEqual(self.state_store.commit_seq, before + 1)
        with self.assertRaises(ServiceError):
            self._batch(items=[{"lease_id": "ghost",
                                "completion_id": "c9",
                                "outcome": "delivered"}])
        self.assertEqual(self.state_store.commit_seq, before + 1)

    def test_failed_batch_writes_nothing(self) -> None:
        self._claim("L1", limit=2)
        self._dispatch(job_id="j1")
        before = self.state_store.commit_seq
        with self.assertRaises(ServiceError) as caught:
            self._batch(items=[
                {"lease_id": "L1", "completion_id": "cL",
                 "outcome": "failed"},
                {"lease_id": "ghost", "completion_id": "cX",
                 "outcome": "delivered"}])
        self.assertEqual(caught.exception.status_code, 404)
        self.assertEqual(self.state_store.commit_seq, before)
        self.assertEqual(
            self.service.inbox_lease_get("bob", "L1")["state"], "active")
        job, _ = self._job(job_id="j1", op="status")
        self.assertEqual(job["state"], "running")

    def test_restart_restores_completions_and_replays(self) -> None:
        self._dispatch(job_id="j1")
        first, status = self._batch(items=[
            {"lease_id": "j1", "completion_id": "c1",
             "outcome": "delivered"}])
        self.assertEqual(status, 201)
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        job, _ = restarted.inbox_job(
            {"device_id": "bob", "job_id": "j1", "op": "status"})
        self.assertEqual(job["state"], "succeeded")
        lease = restarted.inbox_lease_get("bob", "j1")
        self.assertEqual(lease["state"], "completed")
        # The committed batch still replays as 200 with the frozen values.
        body, status = restarted.inbox_job_complete_batch(
            {"device_id": "bob", "items": [
                {"lease_id": "j1", "completion_id": "c1",
                 "outcome": "delivered"}]})
        self.assertEqual(status, 200)
        self.assertEqual(body["results"], first["results"])
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

        self._dispatch(job_id="j1")
        generation = self.state_store.commit_seq
        persistence_mod.os.fsync = fail_first_directory_fsync
        try:
            with self.assertRaises(PersistenceUnavailable):
                self._batch(items=[{"lease_id": "j1",
                                    "completion_id": "c1",
                                    "outcome": "delivered"}])
        finally:
            persistence_mod.os.fsync = real_fsync
        self.assertEqual(self.state_store.commit_seq, generation)
        job, _ = self._job(job_id="j1", op="status")
        self.assertEqual(job["state"], "running")
        body, status = self._batch(items=[
            {"lease_id": "j1", "completion_id": "c1",
             "outcome": "delivered"}])
        self.assertEqual(status, 201)
        self.assertEqual(body["results"][0]["completion_id"], "c1")
        self.assertEqual(self.state_store.commit_seq, generation + 1)


class InboxJobCompleteBatchHTTPTest(_BatchMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()
        self.server, self.service = create_server(
            "127.0.0.1", 0, self.service)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        self._dispatch(job_id="j1")

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _request(self, payload, raw_body=None):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        body = raw_body if raw_body is not None else json.dumps(payload)
        conn.request("POST", "/v1/inbox-jobs/complete-batch", body=body,
                     headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        data = response.read().decode("utf-8")
        conn.close()
        return response.status, (json.loads(data) if data else None), data

    def test_complete_batch_over_http(self) -> None:
        status, body, raw = self._request(
            {"device_id": "bob", "items": [
                {"lease_id": "j1", "completion_id": "c1",
                 "outcome": "delivered"}]})
        self.assertEqual(status, 201)
        self.assertEqual(list(body), ["device_id", "results"])
        self.assertEqual(body["device_id"], "bob")
        item = body["results"][0]
        self.assertEqual(list(item), ["lease_id", "completion_id",
                                      "outcome", "completed_at"])
        self.assertEqual(item["lease_id"], "j1")
        self.assertLess(raw.index('"device_id"'), raw.index('"results"'))
        self.assertLess(raw.index('"lease_id"'),
                        raw.index('"completion_id"'))
        self.assertLess(raw.index('"completion_id"'),
                        raw.index('"outcome"'))
        self.assertLess(raw.index('"outcome"'),
                        raw.index('"completed_at"'))
        # The full replay answers 200 byte-identically.
        status, body, raw2 = self._request(
            {"device_id": "bob", "items": [
                {"lease_id": "j1", "completion_id": "c1",
                 "outcome": "delivered"}]})
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
            {"device_id": "bob", "items": [{"completion_id": "c1",
                                            "outcome": "delivered"}]})
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "items[0].lease_id")
        status, body, _ = self._request(
            {"device_id": "bob", "items": [{"lease_id": "j1",
                                            "completion_id": "c1"}]})
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "items[0].outcome")
        status, body, _ = self._request(
            {"device_id": "ghost", "items": [
                {"lease_id": "j1", "completion_id": "c1",
                 "outcome": "delivered"}]})
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "device_id")
        status, body, _ = self._request(
            {"device_id": "bob", "items": [
                {"lease_id": "ghost", "completion_id": "c1",
                 "outcome": "delivered"}]})
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "items[0].lease_id")
        # Partial replay: complete j1, then replay it mixed with nothing
        # else valid — use a released second lease to force a fresh item.
        self._request({"device_id": "bob", "items": [
            {"lease_id": "j1", "completion_id": "c1",
             "outcome": "delivered"}]})
        self._dispatch(job_id="j2")
        status, body, _ = self._request(
            {"device_id": "bob", "items": [
                {"lease_id": "j1", "completion_id": "c1",
                 "outcome": "delivered"},
                {"lease_id": "j2", "completion_id": "c2",
                 "outcome": "delivered"}]})
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "items[0].completion_id")
        # The failed batch wrote nothing: j2 is still running.
        job, _ = self.service.inbox_job(
            {"device_id": "bob", "job_id": "j2", "op": "status"})
        self.assertEqual(job["state"], "running")


if __name__ == "__main__":
    unittest.main()
