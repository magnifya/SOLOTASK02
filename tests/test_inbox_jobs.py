"""Tests for the 1:1-inbox redelivery job endpoint.

``POST /v1/inbox-jobs`` queues/dispatches/inspects a redelivery job. A
``queue`` creates a ``pending`` job; a first ``dispatch`` claims up to 100
unacked messages without a valid lease (using the ``job_id`` as the
lease_id), turning the job into ``running`` (non-empty claim) or
``succeeded`` (empty claim); the job only settles afterwards when that
lease is completed (``delivered`` -> ``succeeded``, ``failed`` ->
``failed``). The job record, its claim lease, acknowledgements and device
revocation all commit under the one store lock.
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


class InboxJobServiceTest(InboxMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def _job(self, device_id="bob", job_id="j1", op="queue"):
        return self.service.inbox_job(
            {"device_id": device_id, "job_id": job_id, "op": op})

    def _error(self, payload):
        with self.assertRaises(ServiceError) as caught:
            self.service.inbox_job(payload)
        return caught.exception

    def test_body_validation(self) -> None:
        for payload, field in (
                (None, "request_body"),
                ("x", "request_body"),
                ([1], "request_body"),
                ({}, "device_id"),
                ({"device_id": "bob"}, "job_id"),
                ({"device_id": "bob", "job_id": "j1"}, "op"),
                ({"device_id": "", "job_id": "j1", "op": "queue"},
                 "device_id"),
                ({"device_id": 7, "job_id": "j1", "op": "queue"}, "device_id"),
                ({"device_id": "bob", "job_id": "", "op": "queue"}, "job_id"),
                ({"device_id": "bob", "job_id": 1, "op": "queue"}, "job_id"),
                ({"device_id": "bob", "job_id": "j1", "op": ""}, "op"),
                ({"device_id": "bob", "job_id": "j1", "op": "run"}, "op"),
                ({"device_id": "bob", "job_id": "j1", "op": 3}, "op")):
            error = self._error(payload)
            self.assertEqual((error.status_code, error.field),
                             (400, field), payload)

    def test_queue_creates_pending_and_replays(self) -> None:
        body, status = self._job()
        self.assertEqual(status, 201)
        self.assertEqual(list(body),
                         ["job_id", "device_id", "state", "lease_id"])
        self.assertEqual(body, {"job_id": "j1", "device_id": "bob",
                                "state": "pending", "lease_id": None})
        body2, status2 = self._job()
        self.assertEqual(status2, 200)
        self.assertEqual(body2, body)

    def test_queue_unknown_or_revoked_device(self) -> None:
        error = self._error(
            {"device_id": "ghost", "job_id": "g", "op": "queue"})
        self.assertEqual((error.status_code, error.field),
                         (409, "device_id"))
        self.service.store.revoke_device("bob")
        error = self._error(
            {"device_id": "bob", "job_id": "r", "op": "queue"})
        self.assertEqual((error.status_code, error.field),
                         (409, "device_id"))

    def test_job_id_is_unique_per_device(self) -> None:
        self._job("bob", "j1")
        error = self._error(
            {"device_id": "alice", "job_id": "j1", "op": "queue"})
        self.assertEqual((error.status_code, error.field),
                         (409, "job_id"))

    def test_dispatch_unknown_job_is_404(self) -> None:
        for op in ("dispatch", "status"):
            error = self._error(
                {"device_id": "bob", "job_id": "nope", "op": op})
            self.assertEqual((error.status_code, error.field),
                             (404, "job_id"), op)

    def test_cross_device_dispatch_and_status_conflict(self) -> None:
        self._job("bob", "j1")
        for op in ("dispatch", "status"):
            error = self._error(
                {"device_id": "alice", "job_id": "j1", "op": op})
            self.assertEqual((error.status_code, error.field),
                             (409, "job_id"), op)

    def test_dispatch_claims_and_running_job_replays(self) -> None:
        self._job()
        body, status = self._job(op="dispatch")
        self.assertEqual(status, 201)
        self.assertEqual(body["state"], "running")
        self.assertEqual(body["lease_id"], "j1")
        # The job_id is an ordinary lease on the picked messages.
        lease = self.service.store.inbox_lease_get("bob", "j1")
        self.assertEqual(lease["limit"], 100)
        self.assertEqual(lease["state"], "active")
        self.assertEqual([m["message_id"] for m in lease["messages"]],
                         ["a1", "a2", "a3", "b1", "b2"])
        # Repeated dispatch is an idempotent 200 with the same record.
        body2, status2 = self._job(op="dispatch")
        self.assertEqual(status2, 200)
        self.assertEqual(body2, body)

    def test_status_is_read_only(self) -> None:
        self._job()
        self._job(op="dispatch")
        delivery_before = dict(self.service.store._delivery)
        body, status = self._job(op="status")
        self.assertEqual(status, 200)
        self.assertEqual(body["state"], "running")
        self.assertEqual(dict(self.service.store._delivery),
                         delivery_before)

    def test_dispatch_revoked_device_conflicts_but_status_stays(self) -> None:
        self._job()
        self.service.store.revoke_device("bob")
        error = self._error(
            {"device_id": "bob", "job_id": "j1", "op": "dispatch"})
        self.assertEqual((error.status_code, error.field),
                         (409, "device_id"))
        body, status = self._job(op="status")
        self.assertEqual(status, 200)
        self.assertEqual(body["state"], "pending")

    def test_empty_dispatch_settles_succeeded_without_lease(self) -> None:
        # Bob's inbox is fully leased by another active lease.
        self.service.inbox_claim("bob", {"lease_id": "other", "limit": 100})
        self._job()
        body, status = self._job(op="dispatch")
        self.assertEqual(status, 201)
        self.assertEqual(body["state"], "succeeded")
        self.assertIsNone(body["lease_id"])
        self.assertIsNone(
            self.service.store._find_inbox_lease_locked("j1"))

    def test_lease_completion_settles_the_job(self) -> None:
        self._job()
        self._job(op="dispatch")
        _body, code = self.service.store.inbox_lease_complete(
            "bob", "j1", "c1", "delivered")
        self.assertEqual(code, 201)
        body, status = self._job(op="status")
        self.assertEqual(status, 200)
        self.assertEqual(body["state"], "succeeded")
        self.assertEqual(body["lease_id"], "j1")

    def test_failed_completion_settles_job_failed(self) -> None:
        self._job("carol", "jf")
        body, status = self._job("carol", "jf", "dispatch")
        self.assertEqual(status, 201)
        self.assertEqual(body["state"], "running")
        self.service.store.inbox_lease_complete(
            "carol", "jf", "c1", "failed")
        body, _status = self._job("carol", "jf", "status")
        self.assertEqual(body["state"], "failed")
        self.assertEqual(body["lease_id"], "jf")

    def test_claim_namespace_is_disjoint_from_job_ids(self) -> None:
        self._job("bob", "j1")
        with self.assertRaises(ServiceError) as caught:
            self.service.inbox_claim(
                "bob", {"lease_id": "j1", "limit": 10})
        self.assertEqual((caught.exception.status_code,
                          caught.exception.field), (409, "lease_id"))
        # Claim first (non-empty), then queue must refuse the id.
        self.service.inbox_claim("bob", {"lease_id": "L9", "limit": 5})
        with self.assertRaises(ServiceError) as caught2:
            self._job("bob", "L9")
        self.assertEqual((caught2.exception.status_code,
                          caught2.exception.field), (409, "job_id"))


class InboxJobPersistenceTest(InboxMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self._build()
        self.path = os.path.join(self.directory, "state.json")
        self.state_store = attach_persistence(self.service, self.path)

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def _job(self, device_id="bob", job_id="j1", op="queue"):
        return self.service.inbox_job(
            {"device_id": device_id, "job_id": job_id, "op": op})

    def test_queue_dispatch_completion_advance_one_generation_each(self) -> None:
        generation = self.state_store.commit_seq
        self._job()
        self.assertEqual(self.state_store.commit_seq, generation + 1)
        self._job()
        self.assertEqual(self.state_store.commit_seq, generation + 1)
        self._job(op="dispatch")
        self.assertEqual(self.state_store.commit_seq, generation + 2)
        self._job(op="status")
        self.assertEqual(self.state_store.commit_seq, generation + 2)
        self.service.store.inbox_lease_complete("bob", "j1", "c1",
                                                 "delivered")
        self.assertEqual(self.state_store.commit_seq, generation + 3)

    def test_failed_write_rolls_the_job_back(self) -> None:
        generation = self.state_store.commit_seq
        original = self.state_store.save

        def fail(*_args, **_kwargs):
            raise OSError("disk full")

        self.state_store.save = fail
        with self.assertRaises(Exception):
            self._job()
        self.state_store.save = original
        self.assertNotIn("j1", self.service.store._redelivery_jobs)
        self.assertEqual(self.state_store.commit_seq, generation)
        # The failed request did not consume the id: it queues afterwards.
        body, status = self._job()
        self.assertEqual(status, 201)
        self.assertEqual(body["state"], "pending")

    def test_restart_rebuilds_jobs(self) -> None:
        self._job()
        self._job(op="dispatch")
        self.service.store.inbox_lease_complete(
            "bob", "j1", "c1", "delivered")
        # A delivered completion is not itself an ack: bulk-ack the lease so
        # the next dispatch finds an actually empty inbox.
        self.service.store.inbox_lease_ack("bob", "j1")
        self._job("bob", "empty", "queue")
        self._job("bob", "empty", "dispatch")
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        body, status = restarted.inbox_job(
            {"device_id": "bob", "job_id": "j1", "op": "status"})
        self.assertEqual(status, 200)
        self.assertEqual(body["state"], "succeeded")
        self.assertEqual(body["lease_id"], "j1")
        body, _status = restarted.inbox_job(
            {"device_id": "bob", "job_id": "empty", "op": "status"})
        self.assertEqual(body["state"], "succeeded")
        self.assertIsNone(body["lease_id"])

    def test_document_is_compact_utf8(self) -> None:
        self._job()
        self._job(op="dispatch")
        with open(self.path, "rb") as handle:
            raw = handle.read()
        self.assertNotIn(b"\n", raw.strip())
        self.assertIn(b"redelivery_jobs", raw)
        json.loads(raw.decode("utf-8"))


class InboxJobHTTPTest(InboxMixin, unittest.TestCase):
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

    def _post(self, body):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.request("POST", "/v1/inbox-jobs",
                     body=json.dumps(body),
                     headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        return response.status, json.loads(response.read())

    def test_queue_dispatch_status_over_http(self) -> None:
        status, body = self._post(
            {"device_id": "bob", "job_id": "j1", "op": "queue"})
        self.assertEqual(status, 201)
        self.assertEqual(list(body),
                         ["job_id", "device_id", "state", "lease_id"])
        self.assertEqual(body["lease_id"], None)
        status, body = self._post(
            {"device_id": "bob", "job_id": "j1", "op": "queue"})
        self.assertEqual(status, 200)
        status, body = self._post(
            {"device_id": "bob", "job_id": "j1", "op": "dispatch"})
        self.assertEqual(status, 201)
        self.assertEqual(body["state"], "running")
        self.assertEqual(body["lease_id"], "j1")
        status, body = self._post(
            {"device_id": "bob", "job_id": "j1", "op": "status"})
        self.assertEqual(status, 200)
        self.assertEqual(body["state"], "running")

    def test_http_errors(self) -> None:
        status, body = self._post(
            {"device_id": "alice", "job_id": "j1", "op": "dispatch"})
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "job_id")
        status, body = self._post(
            {"device_id": "ghost", "job_id": "g", "op": "queue"})
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "device_id")
        status, body = self._post(
            {"device_id": "bob", "job_id": "j", "op": "wrong"})
        self.assertEqual(status, 400)
        self.assertEqual(list(body), ["message", "field"])
        self.assertEqual(body["field"], "op")
        status, body = self._post({})
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "device_id")


if __name__ == "__main__":
    unittest.main()
