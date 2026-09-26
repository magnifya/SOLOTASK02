"""Tests for ``POST /v1/inbox-jobs/complete-batch``.

An atomic batch completion of 1:1-inbox leases: the body carries a
``device_id`` and a non-empty ``items`` array of
``lease_id``/``completion_id``/``outcome`` triples (the two ids unique
across items). Every item is prechecked in array order with the
single-lease completion rules (first error aborts the batch, its
``field`` prefixed to ``items[i].``); only then does the batch complete
in input order and commit once, moving any running redelivery job
holding a lease to ``succeeded``/``failed`` with the item's outcome. A
batch whose items all replay their committed completions answers 200
with the frozen first responses and writes nothing; a partial replay
conflicts 409 with the first replayed item's
``items[i].completion_id``.
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

    def _claim(self, lease_id, device_id="bob", limit=2):
        body, status = self.service.inbox_claim(
            device_id, {"lease_id": lease_id, "limit": limit})
        assert status == 201, (lease_id, body)
        return body

    def _job(self, device_id="bob", job_id="j1", op="queue", **extra):
        payload = {"device_id": device_id, "job_id": job_id, "op": op}
        payload.update(extra)
        return self.service.inbox_job(payload)

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
        return self.service.inbox_lease_complete_batch(payload)

    def _two_leases(self):
        # L1 leases the first two inbox messages, L2 the next two.
        self._claim("L1")
        self._claim("L2")


class InboxJobCompleteBatchServiceTest(_BatchMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def test_bad_body_and_device_id_errors(self) -> None:
        for payload in (None, [], "x", 3):
            error = self._error(
                lambda payload=payload:
                self.service.inbox_lease_complete_batch(payload))
            self.assertEqual((error.status_code, error.field),
                             (400, "request_body"), payload)
        item = {"lease_id": "L1", "completion_id": "c1",
                "outcome": "delivered"}
        for payload in ({"items": [item]},
                        {"device_id": "", "items": [item]},
                        {"device_id": 4, "items": [item]},
                        {"device_id": None, "items": [item]}):
            error = self._error(
                lambda payload=payload:
                self.service.inbox_lease_complete_batch(payload))
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
        item = {"lease_id": "L1", "completion_id": "c1",
                "outcome": "delivered"}
        error = self._error(lambda: self._batch(items=[item], op="complete"))
        self.assertEqual((error.status_code, error.field), (400, "op"))
        error = self._error(lambda: self._batch(items=[item], bogus=1))
        self.assertEqual((error.status_code, error.field), (400, "bogus"))

    def test_item_field_errors_carry_the_index(self) -> None:
        good = {"lease_id": "L1", "completion_id": "c1",
                "outcome": "delivered"}
        cases = (
            ([{"completion_id": "c1", "outcome": "delivered"}],
             "items[0].lease_id"),
            ([{"lease_id": "", "completion_id": "c1",
               "outcome": "delivered"}], "items[0].lease_id"),
            ([{"lease_id": 4, "completion_id": "c1",
               "outcome": "delivered"}], "items[0].lease_id"),
            ([{"lease_id": "L1", "outcome": "delivered"}],
             "items[0].completion_id"),
            ([{"lease_id": "L1", "completion_id": "",
               "outcome": "delivered"}], "items[0].completion_id"),
            ([{"lease_id": "L1", "completion_id": None,
               "outcome": "delivered"}], "items[0].completion_id"),
            ([{"lease_id": "L1", "completion_id": "c1"}],
             "items[0].outcome"),
            ([{"lease_id": "L1", "completion_id": "c1", "outcome": ""}],
             "items[0].outcome"),
            ([{"lease_id": "L1", "completion_id": "c1", "outcome": "x"}],
             "items[0].outcome"),
            ([{"lease_id": "L1", "completion_id": "c1", "outcome": 4}],
             "items[0].outcome"),
            ([good, {"completion_id": "c2", "outcome": "failed"}],
             "items[1].lease_id"),
            ([good, {"lease_id": "L2", "outcome": "failed"}],
             "items[1].completion_id"),
            ([good, {"lease_id": "L2", "completion_id": "c2"}],
             "items[1].outcome"),
        )
        for items, field in cases:
            error = self._error(lambda items=items: self._batch(items=items))
            self.assertEqual((error.status_code, error.field),
                             (400, field), items)

    def test_item_extra_keys_rejected_at_item_level(self) -> None:
        good = {"lease_id": "L1", "completion_id": "c1",
                "outcome": "delivered"}
        cases = (
            [{"lease_id": "L1", "completion_id": "c1",
              "outcome": "delivered", "job_id": "j"}],
            [{"lease_id": "L1", "completion_id": "c1",
              "outcome": "delivered", "extra": 1}],
            [good, {"lease_id": "L2", "completion_id": "c2",
                    "outcome": "failed", "bogus": None}],
        )
        for items in cases:
            index = 0 if len(items) == 1 else 1
            error = self._error(lambda items=items: self._batch(items=items))
            self.assertEqual((error.status_code, error.field),
                             (400, f"items[{index}]"), items)

    def test_duplicate_lease_id_or_completion_id_rejected(self) -> None:
        items = [{"lease_id": "L1", "completion_id": "c1",
                  "outcome": "delivered"},
                 {"lease_id": "L1", "completion_id": "c2",
                  "outcome": "failed"}]
        error = self._error(lambda: self._batch(items=items))
        self.assertEqual((error.status_code, error.field), (400, "items[1]"))
        items = [{"lease_id": "L1", "completion_id": "c1",
                  "outcome": "delivered"},
                 {"lease_id": "L2", "completion_id": "c1",
                  "outcome": "failed"}]
        error = self._error(lambda: self._batch(items=items))
        self.assertEqual((error.status_code, error.field), (400, "items[1]"))
        # The same outcome on both items is fine shape-wise (it fails
        # later, in the store, as an unknown lease).
        error = self._error(lambda: self._batch(items=[
            {"lease_id": "ghost", "completion_id": "c1",
             "outcome": "delivered"}]))
        self.assertEqual((error.status_code, error.field),
                         (404, "items[0].lease_id"))

    def test_device_gate_precedes_item_checks(self) -> None:
        self._two_leases()
        items = [{"lease_id": "L1", "completion_id": "c1",
                  "outcome": "delivered"}]
        error = self._error(lambda: self._batch(device_id="ghost",
                                                items=items))
        self.assertEqual((error.status_code, error.field),
                         (409, "device_id"))
        self.service.store.revoke_device("bob")
        error = self._error(lambda: self._batch(items=items))
        self.assertEqual((error.status_code, error.field),
                         (409, "device_id"))

    def test_first_batch_completes_in_input_order_with_shared_timestamp(
            self) -> None:
        self._two_leases()
        body, status = self._batch(items=[
            {"lease_id": "L2", "completion_id": "c2", "outcome": "failed"},
            {"lease_id": "L1", "completion_id": "c1",
             "outcome": "delivered"}])
        self.assertEqual(status, 201)
        self.assertEqual(list(body), ["device_id", "results"])
        self.assertEqual(body["device_id"], "bob")
        results = body["results"]
        self.assertEqual([r["lease_id"] for r in results], ["L2", "L1"])
        for result, completion_id, outcome in (
                (results[0], "c2", "failed"),
                (results[1], "c1", "delivered")):
            self.assertEqual(list(result),
                             ["lease_id", "completion_id", "outcome",
                              "completed_at"])
            self.assertEqual(result["completion_id"], completion_id)
            self.assertEqual(result["outcome"], outcome)
        # One shared UTC timestamp, six microsecond digits, +00:00.
        stamps = {r["completed_at"] for r in results}
        self.assertEqual(len(stamps), 1)
        stamp = stamps.pop()
        self.assertTrue(stamp.endswith("+00:00"))
        self.assertEqual(len(stamp.split(".")[1]), len("000000+00:00"))
        # Both leases are completed history now.
        for lease_id, outcome in (("L1", "delivered"), ("L2", "failed")):
            lease = self.service.inbox_lease_get("bob", lease_id)
            self.assertEqual(lease["state"], "completed")
            self.assertEqual(lease["completion"]["outcome"], outcome)
            self.assertEqual(lease["completion"]["completed_at"], stamp)
        # Completion is not an acknowledgement: the messages are
        # claimable again.
        claim, status = self.service.inbox_claim(
            "bob", {"lease_id": "L-after", "limit": 100})
        self.assertEqual(status, 201)
        self.assertEqual([m["message_id"] for m in claim["messages"]],
                         ["a1", "a2", "a3", "b1", "b2"])

    def test_running_jobs_follow_the_item_outcome(self) -> None:
        # A running job holds its dispatch lease (the job_id itself);
        # completing it moves the job to the outcome's terminal state.
        self._job(job_id="j1")
        self._job(job_id="j1", op="dispatch")
        body, status = self._batch(items=[
            {"lease_id": "j1", "completion_id": "c1",
             "outcome": "delivered"}])
        self.assertEqual(status, 201)
        self.assertEqual([r["lease_id"] for r in body["results"]], ["j1"])
        view, _ = self._job(job_id="j1", op="status")
        self.assertEqual(view["state"], "succeeded")
        # A failed outcome moves a fresh running job to failed instead.
        self._build()
        self._job(job_id="j2")
        self._job(job_id="j2", op="dispatch")
        _, status = self._batch(items=[
            {"lease_id": "j2", "completion_id": "c2", "outcome": "failed"}])
        self.assertEqual(status, 201)
        view, _ = self._job(job_id="j2", op="status")
        self.assertEqual(view["state"], "failed")

    def test_per_item_errors_use_indexed_fields(self) -> None:
        self._two_leases()
        # Unknown lease.
        error = self._error(lambda: self._batch(items=[
            {"lease_id": "ghost", "completion_id": "c1",
             "outcome": "delivered"}]))
        self.assertEqual((error.status_code, error.field),
                         (404, "items[0].lease_id"))
        # A lease owned by another device.
        self._claim("LB", device_id="bob2")
        error = self._error(lambda: self._batch(items=[
            {"lease_id": "LB", "completion_id": "c1",
             "outcome": "delivered"}]))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[0].lease_id"))
        # A released lease cannot complete.
        self.service.inbox_release("bob", "L1")
        error = self._error(lambda: self._batch(items=[
            {"lease_id": "L1", "completion_id": "c1",
             "outcome": "delivered"}]))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[0].lease_id"))
        # An expired lease cannot complete.
        self._expire("L2")
        error = self._error(lambda: self._batch(items=[
            {"lease_id": "L2", "completion_id": "c2",
             "outcome": "delivered"}]))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[0].lease_id"))

    def test_completed_lease_conflicts_at_completion_id(self) -> None:
        self._two_leases()
        _, status = self._batch(items=[
            {"lease_id": "L1", "completion_id": "c1",
             "outcome": "delivered"}])
        self.assertEqual(status, 201)
        # The same lease under another completion_id conflicts.
        error = self._error(lambda: self._batch(items=[
            {"lease_id": "L1", "completion_id": "other",
             "outcome": "delivered"}]))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[0].completion_id"))

    def test_first_error_in_array_order_wins_and_nothing_is_written(
            self) -> None:
        self._two_leases()
        # Item 0 is fine, item 1 unknown: items[1] is reported and item
        # 0's completion is not applied.
        error = self._error(lambda: self._batch(items=[
            {"lease_id": "L1", "completion_id": "c1",
             "outcome": "delivered"},
            {"lease_id": "ghost", "completion_id": "c2",
             "outcome": "failed"}]))
        self.assertEqual((error.status_code, error.field),
                         (404, "items[1].lease_id"))
        self.assertEqual(self.service.inbox_lease_get("bob", "L1")["state"],
                         "active")
        # An earlier bad item wins over a later one.
        error = self._error(lambda: self._batch(items=[
            {"lease_id": "ghost", "completion_id": "c1",
             "outcome": "delivered"},
            {"lease_id": "L2", "completion_id": "c2",
             "outcome": "failed"}]))
        self.assertEqual((error.status_code, error.field),
                         (404, "items[0].lease_id"))
        self.assertEqual(self.service.inbox_lease_get("bob", "L2")["state"],
                         "active")

    def test_all_replays_answer_200_frozen_without_writing(self) -> None:
        self._two_leases()
        first, status = self._batch(items=[
            {"lease_id": "L1", "completion_id": "c1",
             "outcome": "delivered"},
            {"lease_id": "L2", "completion_id": "c2", "outcome": "failed"}])
        self.assertEqual(status, 201)
        replay, status = self._batch(items=[
            {"lease_id": "L1", "completion_id": "c1",
             "outcome": "delivered"},
            {"lease_id": "L2", "completion_id": "c2", "outcome": "failed"}])
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)
        # A single-item replay also answers 200 with the frozen values.
        replay, status = self._batch(items=[
            {"lease_id": "L1", "completion_id": "c1",
             "outcome": "delivered"}])
        self.assertEqual(status, 200)
        self.assertEqual(replay["results"], [first["results"][0]])

    def test_partial_replay_conflicts_with_first_replayed_item(self) -> None:
        self._two_leases()
        self._claim("L3")
        _, status = self._batch(items=[
            {"lease_id": "L1", "completion_id": "c1",
             "outcome": "delivered"}])
        self.assertEqual(status, 201)
        # The replay sits at index 0.
        error = self._error(lambda: self._batch(items=[
            {"lease_id": "L1", "completion_id": "c1",
             "outcome": "delivered"},
            {"lease_id": "L3", "completion_id": "c3",
             "outcome": "delivered"}]))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[0].completion_id"))
        # ... and at index 1.
        error = self._error(lambda: self._batch(items=[
            {"lease_id": "L3", "completion_id": "c3",
             "outcome": "delivered"},
            {"lease_id": "L1", "completion_id": "c1",
             "outcome": "delivered"}]))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[1].completion_id"))
        # Nothing was written for the fresh item.
        self.assertEqual(self.service.inbox_lease_get("bob", "L3")["state"],
                         "active")


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
        self._two_leases()
        before = self.state_store.commit_seq
        _, status = self._batch(items=[
            {"lease_id": "L1", "completion_id": "c1",
             "outcome": "delivered"},
            {"lease_id": "L2", "completion_id": "c2", "outcome": "failed"}])
        self.assertEqual(status, 201)
        self.assertEqual(self.state_store.commit_seq, before + 1)
        _, status = self._batch(items=[
            {"lease_id": "L1", "completion_id": "c1",
             "outcome": "delivered"},
            {"lease_id": "L2", "completion_id": "c2", "outcome": "failed"}])
        self.assertEqual(status, 200)
        self.assertEqual(self.state_store.commit_seq, before + 1)
        with self.assertRaises(ServiceError):
            self._batch(items=[
                {"lease_id": "ghost", "completion_id": "c9",
                 "outcome": "failed"}])
        self.assertEqual(self.state_store.commit_seq, before + 1)

    def test_failed_batch_writes_nothing(self) -> None:
        self._two_leases()
        before = self.state_store.commit_seq
        with self.assertRaises(ServiceError) as caught:
            self._batch(items=[
                {"lease_id": "L1", "completion_id": "c1",
                 "outcome": "delivered"},
                {"lease_id": "ghost", "completion_id": "c2",
                 "outcome": "failed"}])
        self.assertEqual(caught.exception.status_code, 404)
        self.assertEqual(self.state_store.commit_seq, before)
        self.assertEqual(self.service.inbox_lease_get("bob", "L1")["state"],
                         "active")
        with open(self.path, encoding="utf-8") as handle:
            document = json.load(handle)
        leases = [lease for record in document["delivery"]
                  for lease in record["leases"]]
        self.assertTrue(all(lease["completion"] is None
                            for lease in leases))

    def test_restart_restores_batch_and_replays(self) -> None:
        self._two_leases()
        first, status = self._batch(items=[
            {"lease_id": "L1", "completion_id": "c1",
             "outcome": "delivered"},
            {"lease_id": "L2", "completion_id": "c2", "outcome": "failed"}])
        self.assertEqual(status, 201)
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        for lease_id, outcome in (("L1", "delivered"), ("L2", "failed")):
            lease = restarted.inbox_lease_get("bob", lease_id)
            self.assertEqual(lease["state"], "completed")
            self.assertEqual(lease["completion"]["outcome"], outcome)
        # The committed batch still replays as 200 without writing.
        body, status = restarted.inbox_lease_complete_batch(
            {"device_id": "bob", "items": [
                {"lease_id": "L1", "completion_id": "c1",
                 "outcome": "delivered"},
                {"lease_id": "L2", "completion_id": "c2",
                 "outcome": "failed"}]})
        self.assertEqual(status, 200)
        self.assertEqual(body, first)
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

        self._two_leases()
        generation = self.state_store.commit_seq
        persistence_mod.os.fsync = fail_first_directory_fsync
        try:
            with self.assertRaises(PersistenceUnavailable):
                self._batch(items=[
                    {"lease_id": "L1", "completion_id": "c1",
                     "outcome": "delivered"},
                    {"lease_id": "L2", "completion_id": "c2",
                     "outcome": "failed"}])
        finally:
            persistence_mod.os.fsync = real_fsync
        self.assertEqual(self.state_store.commit_seq, generation)
        # Neither lease moved (memory was rolled back from the on-disk
        # state).
        self.assertEqual(self.service.inbox_lease_get("bob", "L1")["state"],
                         "active")
        self.assertEqual(self.service.inbox_lease_get("bob", "L2")["state"],
                         "active")
        body, status = self._batch(items=[
            {"lease_id": "L1", "completion_id": "c1",
             "outcome": "delivered"},
            {"lease_id": "L2", "completion_id": "c2", "outcome": "failed"}])
        self.assertEqual(status, 201)
        self.assertEqual([r["lease_id"] for r in body["results"]],
                         ["L1", "L2"])
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
        self._two_leases()

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
                {"lease_id": "L1", "completion_id": "c1",
                 "outcome": "delivered"},
                {"lease_id": "L2", "completion_id": "c2",
                 "outcome": "failed"}]})
        self.assertEqual(status, 201)
        self.assertEqual(list(body), ["device_id", "results"])
        self.assertEqual(body["device_id"], "bob")
        self.assertEqual([r["lease_id"] for r in body["results"]],
                         ["L1", "L2"])
        for result in body["results"]:
            self.assertEqual(list(result),
                             ["lease_id", "completion_id", "outcome",
                              "completed_at"])
        self.assertLess(raw.index('"device_id"'), raw.index('"results"'))
        first = raw.index('"lease_id"')
        self.assertLess(first, raw.index('"completion_id"'))
        self.assertLess(raw.index('"completion_id"'), raw.index('"outcome"'))
        self.assertLess(raw.index('"outcome"'), raw.index('"completed_at"'))
        # The full replay answers 200 byte-identically.
        status, body, raw2 = self._request(
            {"device_id": "bob", "items": [
                {"lease_id": "L1", "completion_id": "c1",
                 "outcome": "delivered"},
                {"lease_id": "L2", "completion_id": "c2",
                 "outcome": "failed"}]})
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
            {"device_id": "bob", "items": [{"lease_id": "L1"}]})
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "items[0].completion_id")
        status, body, _ = self._request(
            {"device_id": "ghost",
             "items": [{"lease_id": "L1", "completion_id": "c1",
                        "outcome": "delivered"}]})
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "device_id")
        status, body, _ = self._request(
            {"device_id": "bob",
             "items": [{"lease_id": "ghost", "completion_id": "c1",
                        "outcome": "delivered"}]})
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "items[0].lease_id")
        # Partial replay: complete L1, then mix it with a fresh item.
        self._request({"device_id": "bob", "items": [
            {"lease_id": "L1", "completion_id": "c1",
             "outcome": "delivered"}]})
        status, body, _ = self._request(
            {"device_id": "bob", "items": [
                {"lease_id": "L1", "completion_id": "c1",
                 "outcome": "delivered"},
                {"lease_id": "L2", "completion_id": "c2",
                 "outcome": "failed"}]})
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "items[0].completion_id")
        # The failed batch wrote nothing: L2 still completes.
        status, body, _ = self._request(
            {"device_id": "bob", "items": [
                {"lease_id": "L2", "completion_id": "c2",
                 "outcome": "failed"}]})
        self.assertEqual(status, 201)
        self.assertEqual([r["lease_id"] for r in body["results"]], ["L2"])


if __name__ == "__main__":
    unittest.main()
