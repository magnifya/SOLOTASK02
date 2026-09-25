"""Tests for ``POST /v1/inbox-jobs/recover-batch``.

The batch entry atomically recovers many running redelivery jobs of one
device: every item follows the single-job ``op=recover`` rules (its error
status codes, with the field prefixed to ``items[i].``), all items are
prechecked in array order before anything is claimed, and the whole batch
commits as one generation. A fully replayed batch answers 200 and writes
nothing; a partially replayed one conflicts 409 at its first replayed
item's ``items[i].recovery_id``.
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


class RecoverBatchMixin(InboxMixin):
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

    def _post_extra(self) -> None:
        # One more unacked message to bob (sid1's stream continues at 4),
        # so a further dispatch has something to lease while earlier
        # leases are still valid.
        sequence = getattr(self, "_extra_sequence", 4)
        self._extra_sequence = sequence + 1
        self.service.post_message({
            "session_id": self.sid1, "sender_device_id": "alice",
            "message_id": f"x{sequence}", "sequence": sequence,
            "nonce": f"nx{sequence}", "ciphertext": "ct"})

    def _expire(self, *lease_ids: str) -> None:
        with self.service.store._lock:
            for state in self.service.store._delivery.values():
                for lease in state.leases:
                    if lease.lease_id in lease_ids:
                        lease.leased_until = _PAST

    def _batch(self, items, device_id="bob"):
        return self.service.inbox_job_recover_batch(
            {"device_id": device_id, "items": items})

    @staticmethod
    def _item(job_id, recovery_id):
        return {"job_id": job_id, "recovery_id": recovery_id}


class InboxJobRecoverBatchShapeTest(RecoverBatchMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def test_body_and_device_id_shape(self) -> None:
        for bad in (None, [], "x", 1, True):
            error = self._error(
                lambda: self.service.inbox_job_recover_batch(bad))
            self.assertEqual((error.status_code, error.field),
                             (400, "request_body"), bad)
        for bad in ({}, {"device_id": ""}, {"device_id": 1},
                    {"device_id": None}, {"device_id": True}):
            error = self._error(
                lambda: self.service.inbox_job_recover_batch(bad))
            self.assertEqual((error.status_code, error.field),
                             (400, "device_id"), bad)

    def test_items_shape(self) -> None:
        for bad in ({}, {"items": None}, {"items": {}}, {"items": "x"},
                    {"items": []}):
            error = self._error(lambda: self.service.inbox_job_recover_batch(
                dict({"device_id": "bob"}, **bad)))
            self.assertEqual((error.status_code, error.field),
                             (400, "items"), bad)
        for bad in ("x", 1, None, [], True):
            error = self._error(lambda: self._batch([bad]))
            self.assertEqual((error.status_code, error.field),
                             (400, "items[0]"), bad)

    def test_item_field_shape(self) -> None:
        cases = [
            ({}, "items[0].job_id"),
            ({"recovery_id": "r1"}, "items[0].job_id"),
            ({"job_id": ""}, "items[0].job_id"),
            ({"job_id": 1}, "items[0].job_id"),
            ({"job_id": None}, "items[0].job_id"),
            ({"job_id": True}, "items[0].job_id"),
            ({"job_id": "j1"}, "items[0].recovery_id"),
            ({"job_id": "j1", "recovery_id": ""}, "items[0].recovery_id"),
            ({"job_id": "j1", "recovery_id": 1}, "items[0].recovery_id"),
            ({"job_id": "j1", "recovery_id": None}, "items[0].recovery_id"),
            ({"job_id": "j1", "recovery_id": True}, "items[0].recovery_id"),
        ]
        for item, field in cases:
            error = self._error(lambda: self._batch([item]))
            self.assertEqual((error.status_code, error.field),
                             (400, field), item)

    def test_item_field_shape_reports_the_first_bad_item(self) -> None:
        error = self._error(lambda: self._batch(
            [self._item("j1", "r1"), {"job_id": "j2"},
             {"job_id": "j3", "recovery_id": 4}]))
        self.assertEqual((error.status_code, error.field),
                         (400, "items[1].recovery_id"))

    def test_duplicate_fields_are_rejected_per_field(self) -> None:
        error = self._error(lambda: self._batch(
            [self._item("j1", "r1"), self._item("j1", "r2")]))
        self.assertEqual((error.status_code, error.field),
                         (400, "items[1].job_id"))
        error = self._error(lambda: self._batch(
            [self._item("j1", "r1"), self._item("j2", "r1")]))
        self.assertEqual((error.status_code, error.field),
                         (400, "items[1].recovery_id"))
        # One item's job_id may equal another item's recovery_id at the
        # shape level; the store's own rules reject the unknown job.
        self._dispatch("j1")
        self._expire("j1")
        error = self._error(lambda: self._batch(
            [self._item("j1", "r1"), self._item("r1", "r9")]))
        self.assertEqual((error.status_code, error.field),
                         (404, "items[1].job_id"))


class InboxJobRecoverBatchServiceTest(RecoverBatchMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def test_device_state_wins_over_item_checks(self) -> None:
        self._dispatch()
        self._expire("j1")
        error = self._error(lambda: self._batch(
            [self._item("j1", "r1")], device_id="ghost"))
        self.assertEqual((error.status_code, error.field),
                         (409, "device_id"))
        self.service.store.revoke_device("bob")
        error = self._error(lambda: self._batch([self._item("j1", "r1")]))
        self.assertEqual((error.status_code, error.field),
                         (409, "device_id"))

    def test_unknown_and_cross_device_jobs(self) -> None:
        error = self._error(lambda: self._batch([self._item("ghost", "r1")]))
        self.assertEqual((error.status_code, error.field),
                         (404, "items[0].job_id"))
        self._dispatch()
        self._expire("j1")
        error = self._error(lambda: self._batch(
            [self._item("j1", "r1")], device_id="bob2"))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[0].job_id"))

    def test_first_error_in_array_order_is_reported(self) -> None:
        self._dispatch("j1")
        self._expire("j1")
        self._dispatch("j3")  # takes over the expired selection
        # Item 0 is valid; item 1's job is unknown; item 2's lease is
        # still valid — the array's first error wins.
        error = self._error(lambda: self._batch(
            [self._item("j1", "r1"), self._item("ghost", "r2"),
             self._item("j3", "r3")]))
        self.assertEqual((error.status_code, error.field),
                         (404, "items[1].job_id"))
        # Nothing was written: j1 keeps its dispatch lease and no recovery
        # history, so the same batch fails identically again.
        error = self._error(lambda: self._batch(
            [self._item("j1", "r1"), self._item("ghost", "r2")]))
        self.assertEqual((error.status_code, error.field),
                         (404, "items[1].job_id"))
        body, _ = self._job(op="status")
        self.assertEqual(body["lease_id"], "j1")

    def test_recovery_id_conflicts_carry_the_item_path(self) -> None:
        self._dispatch("j1")
        self._expire("j1")
        # The id equals another job's job_id.
        self._job(job_id="j2")
        error = self._error(lambda: self._batch([self._item("j1", "j2")]))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[0].recovery_id"))
        # The id is occupied by an unrelated inbox claim lease.
        self.service.inbox_claim("bob", {"lease_id": "L-used", "limit": 1})
        error = self._error(lambda: self._batch([self._item("j1", "L-used")]))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[0].recovery_id"))
        # The id is committed on another job's recovery history.
        self._batch([self._item("j1", "r1")])
        self._post_extra()
        self._dispatch("j3")
        self._expire("j3")
        error = self._error(lambda: self._batch([self._item("j3", "r1")]))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[0].recovery_id"))

    def test_unrecoverable_states_carry_the_item_path(self) -> None:
        self._dispatch("j1")
        # The dispatch lease is still valid.
        error = self._error(lambda: self._batch([self._item("j1", "r1")]))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[0].lease_id"))
        # A pending job cannot be recovered.
        self._job(job_id="jp")
        error = self._error(lambda: self._batch([self._item("jp", "rp")]))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[0].job_id"))
        # A terminal job cannot be recovered.
        self._expire("j1")
        self._batch([self._item("j1", "r1")])
        self.service.inbox_lease_complete(
            "bob", "r1", {"completion_id": "c1", "outcome": "delivered"})
        error = self._error(lambda: self._batch([self._item("j1", "r2")]))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[0].job_id"))

    def test_batch_recovers_in_input_order_as_one_commit(self) -> None:
        self._dispatch("j1")          # leases a1..a3, b1..b2
        self._post_extra()            # x4 stays unleased
        self._dispatch("j2")          # leases x4
        self._expire("j1", "j2")
        body, status = self._batch(
            [self._item("j2", "r2"), self._item("j1", "r1")])
        self.assertEqual(status, 201)
        self.assertEqual(list(body), ["device_id", "results"])
        self.assertEqual(body["device_id"], "bob")
        self.assertEqual([list(item) for item in body["results"]],
                         [["job_id", "state", "lease_id"]] * 2)
        # Results keep input order: the first item's recovery claimed the
        # whole (all-expired) inbox, so the second found an empty
        # selection and succeeded without a lease.
        self.assertEqual(body["results"], [
            {"job_id": "j2", "state": "running", "lease_id": "r2"},
            {"job_id": "j1", "state": "succeeded", "lease_id": None},
        ])
        lease = self.service.inbox_lease_get("bob", "r2")
        self.assertEqual(lease["state"], "active")
        self.assertEqual(lease["limit"], 100)
        self.assertEqual([m["message_id"] for m in lease["messages"]],
                         ["a1", "a2", "a3", "x4", "b1", "b2"])

    def test_empty_selection_succeeds_without_lease(self) -> None:
        self._dispatch("j1")
        self._expire("j1")
        self.service.sync_session_ack(
            self.sid1, {"device_id": "bob", "cursor": 3})
        self.service.sync_session_ack(
            self.sid2, {"device_id": "bob", "cursor": 2})
        body, status = self._batch([self._item("j1", "empty")])
        self.assertEqual(status, 201)
        self.assertEqual(body["results"], [
            {"job_id": "j1", "state": "succeeded", "lease_id": None}])
        error = self._error(
            lambda: self.service.inbox_lease_get("bob", "empty"))
        self.assertEqual((error.status_code, error.field), (404, "lease_id"))

    def test_full_replay_returns_current_views_without_writing(self) -> None:
        self._dispatch("j1")
        self._post_extra()
        self._dispatch("j2")
        self._expire("j1", "j2")
        first, status = self._batch(
            [self._item("j1", "r1"), self._item("j2", "r2")])
        self.assertEqual(status, 201)
        # j1's recovery claimed everything; j2's found an empty inbox.
        self.assertEqual(first["results"], [
            {"job_id": "j1", "state": "running", "lease_id": "r1"},
            {"job_id": "j2", "state": "succeeded", "lease_id": None},
        ])
        # The identical batch replays byte-identically with 200.
        replay, status = self._batch(
            [self._item("j1", "r1"), self._item("j2", "r2")])
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)
        # A replay after the job terminated returns the current views.
        self.service.inbox_lease_complete(
            "bob", "r1", {"completion_id": "c1", "outcome": "failed"})
        replay, status = self._batch(
            [self._item("j1", "r1"), self._item("j2", "r2")])
        self.assertEqual(status, 200)
        self.assertEqual(replay["results"][0],
                         {"job_id": "j1", "state": "failed",
                          "lease_id": "r1"})
        # The device gate still precedes the replay decision.
        self.service.store.revoke_device("bob")
        error = self._error(lambda: self._batch(
            [self._item("j1", "r1"), self._item("j2", "r2")]))
        self.assertEqual((error.status_code, error.field), (409, "device_id"))

    def test_partial_replay_conflicts_at_the_first_replayed_item(self) -> None:
        self._dispatch("j1")
        self._post_extra()
        self._dispatch("j2")
        self._expire("j1", "j2")
        _, status = self._batch([self._item("j1", "r1")])
        self.assertEqual(status, 201)
        # Item 0 is a replay, item 1 is fresh and recoverable: the batch
        # was partially applied before and cannot be reproduced
        # atomically, so it conflicts at the replayed item.
        error = self._error(lambda: self._batch(
            [self._item("j1", "r1"), self._item("j2", "r2")]))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[0].recovery_id"))
        # The first replayed item is reported, not a later one.
        self._expire("r1")
        _, status = self._batch([self._item("j2", "r2")])
        self.assertEqual(status, 201)
        error = self._error(lambda: self._batch(
            [self._item("j1", "r9"), self._item("j2", "r2")]))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[1].recovery_id"))
        # The conflict wrote nothing: j1 is still on its r1 lease.
        body, _ = self._job(op="status")
        self.assertEqual(body["lease_id"], "r1")

    def test_recovery_lease_completion_terminates_its_job(self) -> None:
        self._dispatch("j1")
        self._post_extra()
        self._dispatch("j2")
        self._expire("j1", "j2")
        self._batch([self._item("j1", "r1")])
        self._post_extra()
        self._batch([self._item("j2", "r2")])
        self.service.inbox_lease_complete(
            "bob", "r2", {"completion_id": "c2", "outcome": "failed"})
        body, _ = self._job(job_id="j2", op="status")
        self.assertEqual(body["state"], "failed")
        self.assertEqual(body["lease_id"], "r2")
        body, _ = self._job(job_id="j1", op="status")
        self.assertEqual(body["state"], "running")
        self.assertEqual(body["lease_id"], "r1")


class InboxJobRecoverBatchPersistenceTest(RecoverBatchMixin,
                                          unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self._build()
        self.path = os.path.join(self.directory, "state.json")
        self.state_store = attach_persistence(self.service, self.path)

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def _two_running_jobs(self) -> None:
        self._dispatch("j1")          # leases a1..a3, b1..b2
        self._post_extra()            # x4 stays unleased
        self._dispatch("j2")          # leases x4
        self._expire("j1", "j2")

    def test_batch_advances_one_generation_replay_none(self) -> None:
        self._two_running_jobs()
        before = self.state_store.commit_seq
        _, status = self._batch([self._item("j1", "r1"),
                                 self._item("j2", "r2")])
        self.assertEqual(status, 201)
        self.assertEqual(self.state_store.commit_seq, before + 1)
        _, status = self._batch([self._item("j1", "r1"),
                                 self._item("j2", "r2")])
        self.assertEqual(status, 200)
        self.assertEqual(self.state_store.commit_seq, before + 1)

    def test_state_file_shape(self) -> None:
        self._two_running_jobs()
        self._batch([self._item("j1", "r1"), self._item("j2", "r2")])
        with open(self.path, encoding="utf-8") as handle:
            raw = handle.read()
        self.assertNotIn("\n", raw)
        document = json.loads(raw)
        items = {item["job_id"]: item
                 for item in document["redelivery_jobs"]}
        # j1's recovery leased the inbox under r1; j2's found an empty
        # selection and froze a null lease id.
        expected = {"j1": {"recovery_id": "r1", "lease_id": "r1"},
                    "j2": {"recovery_id": "r2", "lease_id": None}}
        for job_id, record in expected.items():
            item = items[job_id]
            self.assertEqual(list(item), ["job_id", "device_id", "state",
                                          "lease_id", "recoveries"])
            self.assertEqual(item["recoveries"], [record])
            self.assertEqual(list(item["recoveries"][0]),
                             ["recovery_id", "lease_id"])

    def test_restart_restores_and_replays_the_batch(self) -> None:
        self._two_running_jobs()
        self._batch([self._item("j1", "r1"), self._item("j2", "r2")])
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        body, status = restarted.inbox_job_recover_batch(
            {"device_id": "bob",
             "items": [self._item("j1", "r1"), self._item("j2", "r2")]})
        self.assertEqual(status, 200)
        self.assertEqual(body["results"], [
            {"job_id": "j1", "state": "running", "lease_id": "r1"},
            {"job_id": "j2", "state": "succeeded", "lease_id": None},
        ])
        self.assertTrue(restarted.persistence_integrity()["consistent"])

    def test_save_failure_rolls_back_the_whole_batch(self) -> None:
        import e2ee_backend.persistence as persistence_mod
        from e2ee_backend.persistence import PersistenceUnavailable

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
                self._batch([self._item("j1", "r1"), self._item("j2", "r2")])
        finally:
            persistence_mod.os.fsync = real_fsync
        self.assertEqual(self.state_store.commit_seq, generation)
        # Neither job recovered: both are still on their dispatch leases
        # and no recovery history or lease was committed. Memory was
        # rolled back from the on-disk state, which also discards the
        # in-memory expiry hack, so expire again before retrying.
        for job_id in ("j1", "j2"):
            body, status = self._job(job_id=job_id, op="status")
            self.assertEqual((body["state"], body["lease_id"]),
                             ("running", job_id))
        self._expire("j1", "j2")
        body, status = self._batch([self._item("j1", "r1"),
                                    self._item("j2", "r2")])
        self.assertEqual(status, 201)
        self.assertEqual(self.state_store.commit_seq, generation + 1)


class InboxJobRecoverBatchHTTPTest(RecoverBatchMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()
        self.server, self.service = create_server(
            "127.0.0.1", 0, self.service)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        self._dispatch("j1")          # leases a1..a3, b1..b2
        self._post_extra()            # x4 stays unleased
        self._dispatch("j2")          # leases x4
        self._expire("j1", "j2")

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _request(self, payload):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("POST", "/v1/inbox-jobs/recover-batch",
                     body=json.dumps(payload),
                     headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        data = response.read().decode("utf-8")
        conn.close()
        return response.status, (json.loads(data) if data else None), data

    def test_recover_batch_over_http(self) -> None:
        payload = {"device_id": "bob", "items": [
            {"job_id": "j1", "recovery_id": "r1"},
            {"job_id": "j2", "recovery_id": "r2"}]}
        status, body, raw = self._request(payload)
        self.assertEqual(status, 201)
        self.assertEqual(body, {"device_id": "bob", "results": [
            {"job_id": "j1", "state": "running", "lease_id": "r1"},
            {"job_id": "j2", "state": "succeeded", "lease_id": None}]})
        self.assertLess(raw.index('"device_id"'), raw.index('"results"'))
        item = raw[raw.index('"results"'):]
        self.assertLess(item.index('"job_id"'), item.index('"state"'))
        self.assertLess(item.index('"state"'), item.index('"lease_id"'))
        status, body, _ = self._request(payload)
        self.assertEqual(status, 200)
        self.assertEqual(body["results"][0]["lease_id"], "r1")

    def test_errors_over_http(self) -> None:
        status, body, raw = self._request({"device_id": "bob", "items": [
            {"job_id": "ghost", "recovery_id": "r9"}]})
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "items[0].job_id")
        self.assertLess(raw.index('"message"'), raw.index('"field"'))
        status, body, _ = self._request({"device_id": "ghost", "items": [
            {"job_id": "j1", "recovery_id": "r9"}]})
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "device_id")
        status, body, _ = self._request({"device_id": "bob", "items": [
            {"job_id": "j1", "recovery_id": "r1"},
            {"job_id": "j2", "recovery_id": "r1"}]})
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "items[1].recovery_id")
        # A first batch commits; replaying it with one item changed is a
        # partial replay and conflicts at the replayed item.
        status, _, _ = self._request({"device_id": "bob", "items": [
            {"job_id": "j1", "recovery_id": "r1"}]})
        self.assertEqual(status, 201)
        status, body, _ = self._request({"device_id": "bob", "items": [
            {"job_id": "j1", "recovery_id": "r1"},
            {"job_id": "j2", "recovery_id": "r2"}]})
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "items[0].recovery_id")


if __name__ == "__main__":
    unittest.main()
