"""Tests for ``POST /v1/inbox-jobs/ack-batch``.

An atomic batch acknowledgement of delivered 1:1-inbox leases: the body
carries exactly a ``device_id`` and a non-empty ``items`` array of
``lease_id``/``ack_id`` string pairs (neither id repeats across items).
The device gate (unknown/revoked -> 409/device_id) runs first; every item
is then prechecked in array order with the single-lease ack rules (first
error aborts the batch, its ``field`` prefixed to ``items[i].``); only
then does the batch acknowledge every leased message in one transaction
(commit_seq + 1, 503/data_file rolls everything back). A batch whose
items all replay their recorded ack_id answers 200 and writes nothing; a
partial replay conflicts 409 with the first replayed item's
``items[i].ack_id``. The lease records the ack_id (``delivery[].leases[]``
gets an ``ack_id`` key after ``completion``, absent in v1 files means
null) and a file contradicting that (different value, or an ack_id on a
lease whose completion is missing/not delivered, or on an unacked
delivery record) refuses to start.
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

    def _complete(self, lease_id, completion_id="c1", outcome="delivered",
                  device_id="bob"):
        return self.service.inbox_lease_complete(
            device_id, lease_id,
            {"completion_id": completion_id, "outcome": outcome})

    def _delivered_lease(self, lease_id="L1", device_id="bob", limit=100):
        self._claim(lease_id, device_id=device_id, limit=limit)
        self._complete(lease_id, completion_id=f"c-{lease_id}",
                       device_id=device_id)
        return lease_id

    def _batch(self, device_id="bob", items=None, **extra):
        payload = {"device_id": device_id,
                   "items": items if items is not None else []}
        payload.update(extra)
        return self.service.inbox_job_ack_batch(payload)

    def _delivery_acked(self, lease_id):
        with self.service.store._lock:
            rows = []
            for state in self.service.store._delivery.values():
                for lease in state.leases:
                    if lease.lease_id == lease_id:
                        rows.append((state.acked, state.ack_sequence,
                                     lease.ack_id))
            return rows


class InboxJobAckBatchServiceTest(_BatchMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def test_bad_body_and_device_id_errors(self) -> None:
        good = [{"lease_id": "L1", "ack_id": "a1"}]
        for payload in (None, [], "x", 3):
            error = self._error(
                lambda payload=payload:
                self.service.inbox_job_ack_batch(payload))
            self.assertEqual((error.status_code, error.field),
                             (400, "request_body"), payload)
        for payload in (
                {"items": good},
                {"device_id": "", "items": good},
                {"device_id": 4, "items": good},
                {"device_id": None, "items": good}):
            error = self._error(
                lambda payload=payload:
                self.service.inbox_job_ack_batch(payload))
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
        item = {"lease_id": "L1", "ack_id": "a1"}
        error = self._error(lambda: self._batch(items=[item], op="ack"))
        self.assertEqual((error.status_code, error.field), (400, "op"))
        error = self._error(lambda: self._batch(items=[item], bogus=1))
        self.assertEqual((error.status_code, error.field), (400, "bogus"))

    def test_item_field_errors_carry_the_index(self) -> None:
        good = {"lease_id": "L1", "ack_id": "a1"}
        cases = (
            ([{"ack_id": "a1"}], "items[0].lease_id"),
            ([{"lease_id": "", "ack_id": "a1"}], "items[0].lease_id"),
            ([{"lease_id": 4, "ack_id": "a1"}], "items[0].lease_id"),
            ([{"lease_id": None, "ack_id": "a1"}], "items[0].lease_id"),
            ([{"lease_id": "L1"}], "items[0].ack_id"),
            ([{"lease_id": "L1", "ack_id": ""}], "items[0].ack_id"),
            ([{"lease_id": "L1", "ack_id": 4}], "items[0].ack_id"),
            ([{"lease_id": "L1", "ack_id": None}], "items[0].ack_id"),
            ([good, {"ack_id": "a2"}], "items[1].lease_id"),
            ([good, {"lease_id": "L2"}], "items[1].ack_id"),
        )
        for items, field in cases:
            error = self._error(lambda items=items: self._batch(items=items))
            self.assertEqual((error.status_code, error.field),
                             (400, field), items)

    def test_item_extra_keys_rejected_at_item_level(self) -> None:
        base = {"lease_id": "L1", "ack_id": "a1"}
        cases = (
            [dict(base, extra=1)],
            [dict(base, completion_id="c1")],
            [dict(base), {"lease_id": "L2", "ack_id": "a2", "bogus": None}],
        )
        for items in cases:
            index = 0 if len(items) == 1 else 1
            error = self._error(lambda items=items: self._batch(items=items))
            self.assertEqual((error.status_code, error.field),
                             (400, f"items[{index}]"), items)

    def test_duplicate_ids_rejected_at_item_level(self) -> None:
        error = self._error(lambda: self._batch(items=[
            {"lease_id": "L1", "ack_id": "a1"},
            {"lease_id": "L1", "ack_id": "a2"}]))
        self.assertEqual((error.status_code, error.field), (400, "items[1]"))
        error = self._error(lambda: self._batch(items=[
            {"lease_id": "L1", "ack_id": "a1"},
            {"lease_id": "L2", "ack_id": "a1"}]))
        self.assertEqual((error.status_code, error.field), (400, "items[1]"))

    def test_device_gate_precedes_item_checks(self) -> None:
        self._delivered_lease("L1")
        items = [{"lease_id": "L1", "ack_id": "a1"}]
        error = self._error(lambda: self._batch(device_id="ghost",
                                                items=items))
        self.assertEqual((error.status_code, error.field),
                         (409, "device_id"))
        self.service.store.revoke_device("bob")
        error = self._error(lambda: self._batch(items=items))
        self.assertEqual((error.status_code, error.field),
                         (409, "device_id"))
        # Nothing was written.
        self.assertEqual(self._delivery_acked("L1")[0][2], None)

    def test_first_batch_acks_one_lease(self) -> None:
        self._delivered_lease("L1")
        body, status = self._batch(items=[{"lease_id": "L1",
                                          "ack_id": "a1"}])
        self.assertEqual(status, 201)
        self.assertEqual(list(body), ["device_id", "results"])
        self.assertEqual(body["device_id"], "bob")
        self.assertEqual(len(body["results"]), 1)
        item = body["results"][0]
        self.assertEqual(list(item),
                         ["lease_id", "ack_id", "message_count"])
        self.assertEqual(item["lease_id"], "L1")
        self.assertEqual(item["ack_id"], "a1")
        # L1 claimed all five of bob's messages.
        self.assertEqual(item["message_count"], 5)
        rows = self._delivery_acked("L1")
        self.assertEqual(len(rows), 5)
        self.assertEqual({(acked, ack_id) for acked, _seq, ack_id in rows},
                         {(True, "a1")})
        self.assertEqual(sorted(seq for _acked, seq, _ack_id in rows),
                         [1, 1, 2, 2, 3])

    def test_batch_two_leases_one_transaction(self) -> None:
        self._claim("L1", limit=2)
        self._complete("L1")
        self._claim("L2", limit=100)
        self._complete("L2", completion_id="c-L2")
        body, status = self._batch(items=[
            {"lease_id": "L1", "ack_id": "a1"},
            {"lease_id": "L2", "ack_id": "a2"}])
        self.assertEqual(status, 201)
        self.assertEqual([r["lease_id"] for r in body["results"]],
                         ["L1", "L2"])
        self.assertEqual([r["ack_id"] for r in body["results"]],
                         ["a1", "a2"])
        self.assertEqual([r["message_count"] for r in body["results"]],
                         [2, 5])
        self.assertEqual({row[2] for row in self._delivery_acked("L1")},
                         {"a1"})
        self.assertEqual({row[2] for row in self._delivery_acked("L2")},
                         {"a2"})

    def test_per_item_errors_use_indexed_fields(self) -> None:
        self._delivered_lease("L1")
        self._delivered_lease("Lb", device_id="bob2")
        # Unknown lease.
        error = self._error(lambda: self._batch(items=[
            {"lease_id": "ghost", "ack_id": "a1"}]))
        self.assertEqual((error.status_code, error.field),
                         (404, "items[0].lease_id"))
        # A lease owned by another device.
        error = self._error(lambda: self._batch(items=[
            {"lease_id": "Lb", "ack_id": "a1"}]))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[0].lease_id"))
        # A lease that is not yet completed (active): L1's delivered
        # completion freed its messages, so L2 takes two of them.
        self._claim("L2", limit=2)
        error = self._error(lambda: self._batch(items=[
            {"lease_id": "L2", "ack_id": "a2"}]))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[0].lease_id"))
        # A completed-but-failed lease: expire L2 so its messages can be
        # leased again under L3, then fail L3.
        self._expire("L2")
        self._claim("L3", limit=2)
        self._complete("L3", completion_id="c-L3", outcome="failed")
        error = self._error(lambda: self._batch(items=[
            {"lease_id": "L3", "ack_id": "a3"}]))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[0].lease_id"))

    def test_recorded_ack_under_new_id_conflicts_at_ack_id(self) -> None:
        self._delivered_lease("L1")
        self._batch(items=[{"lease_id": "L1", "ack_id": "a1"}])
        error = self._error(lambda: self._batch(items=[
            {"lease_id": "L1", "ack_id": "other"}]))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[0].ack_id"))

    def test_same_ack_id_is_reusable_on_another_lease(self) -> None:
        # An ack_id is scoped to one lease, unlike completion-batch ids;
        # reusing one on a different lease is a first-time ack there.
        self._claim("L1", limit=2)
        self._complete("L1")
        self._claim("L2", limit=100)
        self._complete("L2", completion_id="c-L2")
        body, status = self._batch(items=[{"lease_id": "L1",
                                          "ack_id": "same"}])
        self.assertEqual(status, 201)
        body, status = self._batch(items=[{"lease_id": "L2",
                                          "ack_id": "same"}])
        self.assertEqual(status, 201)
        self.assertEqual(body["results"][0]["ack_id"], "same")

    def test_first_error_in_array_order_wins_and_nothing_is_written(
            self) -> None:
        self._delivered_lease("L1")
        error = self._error(lambda: self._batch(items=[
            {"lease_id": "L1", "ack_id": "a1"},
            {"lease_id": "ghost", "ack_id": "a2"}]))
        self.assertEqual((error.status_code, error.field),
                         (404, "items[1].lease_id"))
        self.assertEqual(self._delivery_acked("L1")[0][2], None)
        error = self._error(lambda: self._batch(items=[
            {"lease_id": "ghost", "ack_id": "a2"},
            {"lease_id": "L1", "ack_id": "a1"}]))
        self.assertEqual((error.status_code, error.field),
                         (404, "items[0].lease_id"))
        self.assertEqual(self._delivery_acked("L1")[0][2], None)

    def test_all_replays_answer_200_without_writing(self) -> None:
        self._claim("L1", limit=2)
        self._complete("L1")
        self._claim("L2", limit=100)
        self._complete("L2", completion_id="c-L2")
        first, status = self._batch(items=[
            {"lease_id": "L1", "ack_id": "a1"},
            {"lease_id": "L2", "ack_id": "a2"}])
        self.assertEqual(status, 201)
        replay, status = self._batch(items=[
            {"lease_id": "L1", "ack_id": "a1"},
            {"lease_id": "L2", "ack_id": "a2"}])
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)
        # A single-item full replay also answers 200.
        replay, status = self._batch(items=[{"lease_id": "L1",
                                            "ack_id": "a1"}])
        self.assertEqual(status, 200)
        self.assertEqual(replay["results"][0], first["results"][0])

    def test_replay_rejected_after_device_revoked(self) -> None:
        # The device gate precedes replay detection for the batch route.
        self._delivered_lease("L1")
        self._batch(items=[{"lease_id": "L1", "ack_id": "a1"}])
        self.service.store.revoke_device("bob")
        error = self._error(lambda: self._batch(items=[
            {"lease_id": "L1", "ack_id": "a1"}]))
        self.assertEqual((error.status_code, error.field),
                         (409, "device_id"))

    def test_partial_replay_conflicts_with_first_replayed_item(self) -> None:
        self._claim("L1", limit=2)
        self._complete("L1")
        self._claim("L2", limit=100)
        self._complete("L2", completion_id="c-L2")
        self._batch(items=[{"lease_id": "L1", "ack_id": "a1"}])
        # The replay sits at index 0.
        error = self._error(lambda: self._batch(items=[
            {"lease_id": "L1", "ack_id": "a1"},
            {"lease_id": "L2", "ack_id": "a2"}]))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[0].ack_id"))
        # ... and at index 1.
        error = self._error(lambda: self._batch(items=[
            {"lease_id": "L2", "ack_id": "a2"},
            {"lease_id": "L1", "ack_id": "a1"}]))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[1].ack_id"))
        # The fresh item was not written.
        self.assertEqual(self._delivery_acked("L2")[0][2], None)

    def test_attempts_unchanged_by_ack(self) -> None:
        self._delivered_lease("L1")
        # A retry beforehand leaves an attempt the ack must not touch.
        self.service.retry_message(
            self.sid1, "a1",
            {"device_id": "bob", "attempt_id": "t1"})
        self._batch(items=[{"lease_id": "L1", "ack_id": "a1"}])
        with self.service.store._lock:
            state = self.service.store._delivery[(self.sid1, "a1")]
            self.assertEqual(state.attempts, 1)
            self.assertEqual(state.attempt_ids, {"t1"})
            self.assertTrue(state.acked)


class InboxJobAckBatchPersistenceTest(_BatchMixin, unittest.TestCase):
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

    def test_batch_advances_one_generation_replays_and_failures_none(
            self) -> None:
        self._delivered_lease("L1")
        items = [{"lease_id": "L1", "ack_id": "a1"}]
        before = self.state_store.commit_seq
        _, status = self._batch(items=items)
        self.assertEqual(status, 201)
        self.assertEqual(self.state_store.commit_seq, before + 1)
        _, status = self._batch(items=items)
        self.assertEqual(status, 200)
        self.assertEqual(self.state_store.commit_seq, before + 1)
        with self.assertRaises(ServiceError):
            self._batch(items=[{"lease_id": "ghost", "ack_id": "a9"}])
        self.assertEqual(self.state_store.commit_seq, before + 1)

    def test_failed_batch_writes_nothing(self) -> None:
        self._delivered_lease("L1")
        before = self.state_store.commit_seq
        with self.assertRaises(ServiceError) as caught:
            self._batch(items=[
                {"lease_id": "L1", "ack_id": "a1"},
                {"lease_id": "ghost", "ack_id": "a2"}])
        self.assertEqual(caught.exception.status_code, 404)
        self.assertEqual(self.state_store.commit_seq, before)
        self.assertEqual(self._delivery_acked("L1")[0][2], None)

    def test_lease_serializes_ack_id_after_completion(self) -> None:
        self._claim("L1", limit=1)
        self._complete("L1")
        before_ack = [record for record in self._document()["delivery"]
                      if record.get("leases")]
        # Pre-ack leases already carry an explicit null ack_id (the key is
        # appended to every serialized lease).
        self.assertEqual(list(before_ack[0]["leases"][0]),
                         ["lease_id", "limit", "leased_until", "released_at",
                          "renewals", "completion", "ack_id"])
        self.assertIsNone(before_ack[0]["leases"][0]["ack_id"])
        self._batch(items=[{"lease_id": "L1", "ack_id": "a1"}])
        leased = [record for record in self._document()["delivery"]
                  if any(lease["lease_id"] == "L1"
                         for lease in record.get("leases", []))]
        self.assertTrue(leased)
        for record in leased:
            lease = next(lease for lease in record["leases"]
                         if lease["lease_id"] == "L1")
            self.assertEqual(lease["ack_id"], "a1")

    def test_restart_restores_acks_and_replays(self) -> None:
        self._delivered_lease("L1")
        first, status = self._batch(items=[{"lease_id": "L1",
                                           "ack_id": "a1"}])
        self.assertEqual(status, 201)
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        lease = restarted.inbox_lease_get("bob", "L1")
        self.assertTrue(all(item["acked"] for item in lease["messages"]))
        # The committed batch still replays as 200 with the frozen values.
        body, status = restarted.inbox_job_ack_batch(
            {"device_id": "bob", "items": [
                {"lease_id": "L1", "ack_id": "a1"}]})
        self.assertEqual(status, 200)
        self.assertEqual(body["results"], first["results"])
        self.assertTrue(restarted.persistence_integrity()["consistent"])

    def _corrupt_lease(self, mutate):
        doc = self._document()
        mutate(doc)
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(doc, handle)

    def test_ack_id_without_delivered_completion_refuses_start(self) -> None:
        self._claim("L1", limit=1)
        self._complete("L1")
        self._batch(items=[{"lease_id": "L1", "ack_id": "a1"}])

        def break_completion(doc):
            for record in doc["delivery"]:
                for lease in record.get("leases", []):
                    if lease["lease_id"] == "L1":
                        lease["completion"]["outcome"] = "failed"

        self._corrupt_lease(break_completion)
        restarted = DeviceService()
        with self.assertRaises(Exception):
            attach_persistence(restarted, self.path)

    def test_divergent_ack_id_copy_refuses_start(self) -> None:
        self._claim("L1", limit=2)
        self._complete("L1")
        self._batch(items=[{"lease_id": "L1", "ack_id": "a1"}])

        def break_copy(doc):
            touched = False
            for record in doc["delivery"]:
                for lease in record.get("leases", []):
                    if lease["lease_id"] == "L1" and not touched:
                        lease["ack_id"] = "different"
                        touched = True

        self._corrupt_lease(break_copy)
        restarted = DeviceService()
        with self.assertRaises(Exception):
            attach_persistence(restarted, self.path)

    def test_bad_ack_id_type_refuses_start(self) -> None:
        self._claim("L1", limit=1)
        self._complete("L1")
        self._batch(items=[{"lease_id": "L1", "ack_id": "a1"}])

        def break_type(doc):
            for record in doc["delivery"]:
                for lease in record.get("leases", []):
                    if lease["lease_id"] == "L1":
                        lease["ack_id"] = 4

        self._corrupt_lease(break_type)
        restarted = DeviceService()
        with self.assertRaises(Exception):
            attach_persistence(restarted, self.path)

    def test_legacy_lease_without_ack_id_key_loads_as_null(self) -> None:
        self._delivered_lease("L1")
        # Strip the ack_id key like a v1 file written before this feature,
        # onto a fresh path with the integrity marker removed and no
        # sidecar: a genuine marker-less legacy document.
        doc = self._document()
        for record in doc["delivery"]:
            for lease in record.get("leases", []):
                lease.pop("ack_id", None)
        doc.pop("integrity_log_version", None)
        legacy_path = os.path.join(self.directory, "legacy.json")
        with open(legacy_path, "w", encoding="utf-8") as handle:
            json.dump(doc, handle)
        restarted = DeviceService()
        attach_persistence(restarted, legacy_path)
        self.assertTrue(restarted.persistence_integrity()["consistent"])
        # A first ack still works (absent key means null/unrecorded).
        body, status = restarted.inbox_job_ack_batch(
            {"device_id": "bob", "items": [
                {"lease_id": "L1", "ack_id": "a1"}]})
        self.assertEqual(status, 201)
        self.assertEqual(body["results"][0]["ack_id"], "a1")

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

        self._delivered_lease("L1")
        generation = self.state_store.commit_seq
        persistence_mod.os.fsync = fail_first_directory_fsync
        try:
            with self.assertRaises(PersistenceUnavailable):
                self._batch(items=[{"lease_id": "L1", "ack_id": "a1"}])
        finally:
            persistence_mod.os.fsync = real_fsync
        self.assertEqual(self.state_store.commit_seq, generation)
        self.assertEqual(self._delivery_acked("L1")[0][2], None)
        body, status = self._batch(items=[{"lease_id": "L1",
                                          "ack_id": "a1"}])
        self.assertEqual(status, 201)
        self.assertEqual(body["results"][0]["ack_id"], "a1")
        self.assertEqual(self.state_store.commit_seq, generation + 1)


class InboxJobAckBatchHTTPTest(_BatchMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()
        self.server, self.service = create_server(
            "127.0.0.1", 0, self.service)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        self._delivered_lease("L1")

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _request(self, payload, raw_body=None):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        body = raw_body if raw_body is not None else json.dumps(payload)
        conn.request("POST", "/v1/inbox-jobs/ack-batch", body=body,
                     headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        data = response.read().decode("utf-8")
        conn.close()
        return response.status, (json.loads(data) if data else None), data

    def test_ack_batch_over_http(self) -> None:
        status, body, raw = self._request(
            {"device_id": "bob", "items": [
                {"lease_id": "L1", "ack_id": "a1"}]})
        self.assertEqual(status, 201)
        self.assertEqual(list(body), ["device_id", "results"])
        self.assertEqual(body["device_id"], "bob")
        item = body["results"][0]
        self.assertEqual(list(item),
                         ["lease_id", "ack_id", "message_count"])
        self.assertLess(raw.index('"device_id"'), raw.index('"results"'))
        self.assertLess(raw.index('"lease_id"'), raw.index('"ack_id"'))
        self.assertLess(raw.index('"ack_id"'),
                        raw.index('"message_count"'))
        # The full replay answers 200 byte-identically.
        status, body, raw2 = self._request(
            {"device_id": "bob", "items": [
                {"lease_id": "L1", "ack_id": "a1"}]})
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
            {"device_id": "bob", "items": [{"ack_id": "a1"}]})
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "items[0].lease_id")
        status, body, _ = self._request(
            {"device_id": "bob", "items": [{"lease_id": "L1"}]})
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "items[0].ack_id")
        status, body, _ = self._request(
            {"device_id": "ghost", "items": [
                {"lease_id": "L1", "ack_id": "a1"}]})
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "device_id")
        status, body, _ = self._request(
            {"device_id": "bob", "items": [
                {"lease_id": "ghost", "ack_id": "a1"}]})
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "items[0].lease_id")
        # Partial replay: set up a second delivered lease L2 while L1's
        # messages are still unacked, then ack only L1.
        self._delivered_lease("L2")
        self._request({"device_id": "bob", "items": [
            {"lease_id": "L1", "ack_id": "a1"}]})
        status, body, _ = self._request(
            {"device_id": "bob", "items": [
                {"lease_id": "L1", "ack_id": "a1"},
                {"lease_id": "L2", "ack_id": "a2"}]})
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "items[0].ack_id")
        # The failed batch wrote nothing: L2 carries no ack_id.
        with self.service.store._lock:
            ack_ids = {
                lease.ack_id
                for state in self.service.store._delivery.values()
                for lease in state.leases if lease.lease_id == "L2"}
        self.assertEqual(ack_ids, {None})


if __name__ == "__main__":
    unittest.main()
