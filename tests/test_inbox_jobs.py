"""Tests for the 1:1 inbox redelivery job endpoint.

POST /v1/inbox-jobs queues, dispatches and inspects redelivery jobs. A
queued job is ``pending``; dispatching it leases up to 100 unacked,
currently unleased inbox messages under a lease_id equal to the job_id
(non-empty selection -> ``running``, empty -> ``succeeded``); completing
that lease ``delivered``/``failed`` moves the job to
``succeeded``/``failed`` in the same locked transaction.
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
    StateFileError, attach_persistence)
from e2ee_backend.service import DeviceService, ServiceError
from e2ee_backend.http_app import create_server


class InboxJobServiceTest(InboxMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def _error(self, callable_):
        with self.assertRaises(ServiceError) as caught:
            callable_()
        return caught.exception

    def _job(self, device_id="bob", job_id="j1", op="queue"):
        return self.service.inbox_job(
            {"device_id": device_id, "job_id": job_id, "op": op})

    def test_queue_creates_pending_job(self) -> None:
        body, status = self._job()
        self.assertEqual(status, 201)
        self.assertEqual(list(body),
                         ["job_id", "device_id", "state", "lease_id"])
        self.assertEqual(body, {"job_id": "j1", "device_id": "bob",
                                "state": "pending", "lease_id": None})

    def test_queue_replay_is_idempotent(self) -> None:
        first, _ = self._job()
        replay, status = self._job()
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)

    def test_queue_cross_device_conflict(self) -> None:
        self._job()
        error = self._error(lambda: self._job(device_id="bob2"))
        self.assertEqual((error.status_code, error.field),
                         (409, "job_id"))

    def test_unknown_and_revoked_device(self) -> None:
        error = self._error(lambda: self._job(device_id="ghost"))
        self.assertEqual((error.status_code, error.field),
                         (409, "device_id"))
        self.service.store.revoke_device("bob")
        error = self._error(lambda: self._job())
        self.assertEqual((error.status_code, error.field),
                         (409, "device_id"))
        # The device check wins over the job lookup.
        error = self._error(lambda: self._job(op="status"))
        self.assertEqual((error.status_code, error.field),
                         (409, "device_id"))

    def test_dispatch_and_status_on_unknown_job(self) -> None:
        for op in ("dispatch", "status"):
            error = self._error(lambda: self._job(op=op))
            self.assertEqual((error.status_code, error.field),
                             (404, "job_id"), op)

    def test_dispatch_leases_inbox_messages(self) -> None:
        self._job()
        body, status = self._job(op="dispatch")
        self.assertEqual(status, 201)
        self.assertEqual(body, {"job_id": "j1", "device_id": "bob",
                                "state": "running", "lease_id": "j1"})
        # The dispatch lease is an ordinary inbox lease named by the job_id.
        lease = self.service.inbox_lease_get("bob", "j1")
        self.assertEqual(lease["state"], "active")
        self.assertEqual(lease["limit"], 100)
        self.assertEqual([m["message_id"] for m in lease["messages"]],
                         ["a1", "a2", "a3", "b1", "b2"])
        # The leased messages are withheld from a fresh claim.
        empty, status = self.service.inbox_claim(
            "bob", {"lease_id": "L-other", "limit": 10})
        self.assertEqual(status, 200)
        self.assertEqual(empty["messages"], [])

    def test_dispatch_replay_on_non_pending(self) -> None:
        self._job()
        first, _ = self._job(op="dispatch")
        replay, status = self._job(op="dispatch")
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)

    def test_dispatch_empty_inbox_succeeds_without_lease(self) -> None:
        # bob2's only message is acked, so its inbox is empty.
        self.service.ack_message(self.sid_other, {
            "device_id": "bob2", "message_id": "o1", "sequence": 1})
        self._job(device_id="bob2", job_id="j9")
        body, status = self._job(device_id="bob2", job_id="j9", op="dispatch")
        self.assertEqual(status, 201)
        self.assertEqual(body, {"job_id": "j9", "device_id": "bob2",
                                "state": "succeeded", "lease_id": None})
        # No lease was ever taken under the job id.
        error = self._error(
            lambda: self.service.inbox_lease_get("bob2", "j9"))
        self.assertEqual((error.status_code, error.field),
                         (404, "lease_id"))

    def test_dispatch_skips_actively_leased_messages(self) -> None:
        self.service.inbox_claim("bob", {"lease_id": "L1", "limit": 2})
        self._job()
        body, status = self._job(op="dispatch")
        self.assertEqual(status, 201)
        self.assertEqual(body["state"], "running")
        lease = self.service.inbox_lease_get("bob", "j1")
        self.assertEqual([m["message_id"] for m in lease["messages"]],
                         ["a3", "b1", "b2"])

    def test_dispatch_leases_at_most_100_messages(self) -> None:
        for sequence in range(4, 109):
            self.service.post_message({
                "session_id": self.sid1, "sender_device_id": "alice",
                "message_id": f"x{sequence}", "sequence": sequence,
                "nonce": f"nx{sequence}", "ciphertext": "ct"})
        self._job()
        body, status = self._job(op="dispatch")
        self.assertEqual(status, 201)
        self.assertEqual(body["state"], "running")
        lease = self.service.inbox_lease_get("bob", "j1")
        self.assertEqual(len(lease["messages"]), 100)
        # The ten remaining messages are claimable under another id.
        rest, _ = self.service.inbox_claim(
            "bob", {"lease_id": "L-rest", "limit": 100})
        self.assertEqual(len(rest["messages"]), 10)

    def test_status_is_read_only_current_view(self) -> None:
        self._job()
        body, status = self._job(op="status")
        self.assertEqual(status, 200)
        self.assertEqual(body["state"], "pending")
        self._job(op="dispatch")
        body, status = self._job(op="status")
        self.assertEqual(status, 200)
        self.assertEqual(body, {"job_id": "j1", "device_id": "bob",
                                "state": "running", "lease_id": "j1"})

    def test_lease_completion_moves_job_to_terminal_state(self) -> None:
        self._job()
        self._job(op="dispatch")
        self.service.inbox_lease_complete(
            "bob", "j1", {"completion_id": "c1", "outcome": "delivered"})
        body, status = self._job(op="status")
        self.assertEqual(status, 200)
        self.assertEqual(body["state"], "succeeded")
        self.assertEqual(body["lease_id"], "j1")

    def test_lease_failure_moves_job_to_failed(self) -> None:
        self._job()
        self._job(op="dispatch")
        self.service.inbox_lease_complete(
            "bob", "j1", {"completion_id": "c1", "outcome": "failed"})
        body, _ = self._job(op="status")
        self.assertEqual(body["state"], "failed")
        self.assertEqual(body["lease_id"], "j1")

    def test_completion_of_unrelated_lease_leaves_job_running(self) -> None:
        self.service.inbox_claim("bob", {"lease_id": "L1", "limit": 1})
        self._job()
        self._job(op="dispatch")
        self.service.inbox_lease_complete(
            "bob", "L1", {"completion_id": "c1", "outcome": "delivered"})
        body, _ = self._job(op="status")
        self.assertEqual(body["state"], "running")

    def test_dispatch_conflict_when_lease_id_occupied(self) -> None:
        # An unrelated claim already holds the id the dispatch would take.
        self.service.inbox_claim("bob", {"lease_id": "j1", "limit": 1})
        self._job()
        error = self._error(lambda: self._job(op="dispatch"))
        self.assertEqual((error.status_code, error.field),
                         (409, "job_id"))

    def test_body_and_field_validation(self) -> None:
        for bad in (None, [], "x", 1):
            error = self._error(lambda: self.service.inbox_job(bad))
            self.assertEqual((error.status_code, error.field),
                             (400, "request_body"), bad)
        for field in ("device_id", "job_id", "op"):
            payload = {"device_id": "bob", "job_id": "j1", "op": "queue"}
            del payload[field]
            error = self._error(lambda: self.service.inbox_job(payload))
            self.assertEqual((error.status_code, error.field),
                             (400, field), field)
            for bad in ("", 1, None, True):
                payload = {"device_id": "bob", "job_id": "j1", "op": "queue",
                           field: bad}
                error = self._error(lambda: self.service.inbox_job(payload))
                self.assertEqual((error.status_code, error.field),
                                 (400, field), (field, bad))
        for bad_op in ("Queue", "run", "queued", ""):
            error = self._error(lambda: self.service.inbox_job(
                {"device_id": "bob", "job_id": "j1", "op": bad_op}))
            self.assertEqual((error.status_code, error.field),
                             (400, "op"), bad_op)


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

    def test_queue_and_dispatch_each_advance_one_generation(self) -> None:
        before = self.state_store.commit_seq
        self._job()
        self.assertEqual(self.state_store.commit_seq, before + 1)
        self._job(op="dispatch")
        self.assertEqual(self.state_store.commit_seq, before + 2)
        with open(self.path, encoding="utf-8") as handle:
            document = json.load(handle)
        self.assertEqual(document["redelivery_jobs"], [
            {"job_id": "j1", "device_id": "bob", "state": "running",
             "lease_id": "j1", "recoveries": [],
             "cancellation_id": None, "cancelled_at": None}])

    def test_replays_and_status_consume_no_generation(self) -> None:
        self._job()
        self._job(op="dispatch")
        generation = self.state_store.commit_seq
        self._job()                      # queue replay
        self._job(op="dispatch")         # dispatch replay
        self._job(op="status")           # read-only
        self.assertEqual(self.state_store.commit_seq, generation)

    def test_restart_restores_job_states(self) -> None:
        self._job(job_id="j-pending")
        self._job(job_id="j1")
        self._job(job_id="j1", op="dispatch")
        self.service.inbox_lease_complete(
            "bob", "j1", {"completion_id": "c1", "outcome": "delivered"})
        self._job(job_id="j2")
        self._job(job_id="j2", op="dispatch")
        self.service.inbox_lease_complete(
            "bob", "j2", {"completion_id": "c2", "outcome": "failed"})
        # bob2's inbox is emptied, so j3 succeeds without a lease.
        self.service.ack_message(self.sid_other, {
            "device_id": "bob2", "message_id": "o1", "sequence": 1})
        self._job(device_id="bob2", job_id="j3")
        self._job(device_id="bob2", job_id="j3", op="dispatch")

        restarted = DeviceService()
        attach_persistence(restarted, self.path)

        def state(job_id, device_id="bob"):
            body, status = restarted.inbox_job(
                {"device_id": device_id, "job_id": job_id, "op": "status"})
            self.assertEqual(status, 200)
            return body["state"], body["lease_id"]

        self.assertEqual(state("j-pending"), ("pending", None))
        self.assertEqual(state("j1"), ("succeeded", "j1"))
        self.assertEqual(state("j2"), ("failed", "j2"))
        self.assertEqual(state("j3", "bob2"), ("succeeded", None))
        # The integrity probe still passes with jobs in the state.
        self.assertTrue(restarted.persistence_integrity()["consistent"])

    def test_save_failure_rolls_back_and_surfaces(self) -> None:
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

        generation = self.state_store.commit_seq
        persistence_mod.os.fsync = fail_first_directory_fsync
        try:
            with self.assertRaises(PersistenceUnavailable):
                self._job()
        finally:
            persistence_mod.os.fsync = real_fsync
        # Rolled back: no generation consumed and the job was never queued.
        self.assertEqual(self.state_store.commit_seq, generation)
        error = None
        try:
            self._job(op="status")
        except ServiceError as caught:
            error = caught
        self.assertEqual((error.status_code, error.field), (404, "job_id"))
        # The queue can be retried and now commits.
        _, status = self._job()
        self.assertEqual(status, 201)
        self.assertEqual(self.state_store.commit_seq, generation + 1)

    def test_legacy_document_without_section_loads(self) -> None:
        self._job()
        with open(self.path, encoding="utf-8") as handle:
            document = json.load(handle)
        document.pop("redelivery_jobs")
        document.pop("redelivery_job_events", None)
        document.pop("integrity_log_version", None)
        legacy_path = os.path.join(self.directory, "legacy.json")
        with open(legacy_path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        restarted = DeviceService()
        attach_persistence(restarted, legacy_path)  # must not raise
        # The dropped job is simply gone; its id can be queued afresh.
        body, status = restarted.inbox_job(
            {"device_id": "bob", "job_id": "j1", "op": "queue"})
        self.assertEqual(status, 201)
        self.assertEqual(body["state"], "pending")

    def _document_with_jobs(self, mutate):
        self._job(job_id="j1")
        self._job(job_id="j1", op="dispatch")
        self._job(job_id="j2")
        with open(self.path, encoding="utf-8") as handle:
            document = json.load(handle)
        mutate(document)
        # Write to a fresh path as a marker-less/sidecar-less document, so
        # rejection comes from the payload's own semantic validation rather
        # than the integrity hash gate; the original state file is untouched.
        document.pop("integrity_log_version", None)
        bad_path = os.path.join(self.directory, "bad.json")
        with open(bad_path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        return bad_path

    def _assert_refuses_startup(self, bad_path) -> None:
        with open(bad_path, "rb") as handle:
            original = handle.read()
        restarted = DeviceService()
        with self.assertRaises(StateFileError):
            attach_persistence(restarted, bad_path)
        # The rejected file is never overwritten.
        with open(bad_path, "rb") as handle:
            self.assertEqual(handle.read(), original)

    def test_restore_rejects_malformed_section(self) -> None:
        def mutate(document):
            document["redelivery_jobs"] = {}
        self._assert_refuses_startup(self._document_with_jobs(mutate))

    def test_restore_rejects_bad_item_shape(self) -> None:
        for mutate in (
                lambda d: d["redelivery_jobs"].append("x"),
                lambda d: d["redelivery_jobs"].append(
                    {"job_id": "j9", "device_id": "bob", "state": "pending"}),
                lambda d: d["redelivery_jobs"].append(
                    {"job_id": "", "device_id": "bob", "state": "pending",
                     "lease_id": None}),
                lambda d: d["redelivery_jobs"].append(
                    {"job_id": "j9", "device_id": "bob", "state": "queued",
                     "lease_id": None}),
                lambda d: d["redelivery_jobs"].append(
                    {"job_id": "j9", "device_id": "ghost",
                     "state": "pending", "lease_id": None}),
                lambda d: d["redelivery_jobs"].append(
                    {"job_id": "j9", "device_id": "bob", "state": "pending",
                     "lease_id": "other"}),
                lambda d: d["redelivery_jobs"].append(
                    {"job_id": "j2", "device_id": "bob", "state": "pending",
                     "lease_id": None}),
        ):
            with self.subTest(mutate=mutate):
                self._assert_refuses_startup(
                    self._document_with_jobs(mutate))

    def test_restore_rejects_state_lease_contradictions(self) -> None:
        # j1 is running with lease j1; j2 is pending.
        def pending_with_lease(document):
            document["redelivery_jobs"][1]["lease_id"] = "j2"
        self._assert_refuses_startup(
            self._document_with_jobs(pending_with_lease))

        def running_without_lease(document):
            document["redelivery_jobs"][0]["lease_id"] = None
        self._assert_refuses_startup(
            self._document_with_jobs(running_without_lease))

        def failed_without_completion(document):
            document["redelivery_jobs"][0]["state"] = "failed"
        self._assert_refuses_startup(
            self._document_with_jobs(failed_without_completion))

        def succeeded_with_uncompleted_lease(document):
            document["redelivery_jobs"][0]["state"] = "succeeded"
        self._assert_refuses_startup(
            self._document_with_jobs(succeeded_with_uncompleted_lease))


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

    def _request(self, payload=None, raw=None):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        body = raw if raw is not None else json.dumps(payload)
        conn.request("POST", "/v1/inbox-jobs", body=body,
                     headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        data = response.read().decode("utf-8")
        conn.close()
        return response.status, (json.loads(data) if data else None), data

    def test_full_flow_and_key_order(self) -> None:
        status, body, raw = self._request(
            {"device_id": "bob", "job_id": "j1", "op": "queue"})
        self.assertEqual(status, 201)
        self.assertEqual(list(body),
                         ["job_id", "device_id", "state", "lease_id"])
        # Key order is also correct in the serialized bytes.
        self.assertLess(raw.index('"job_id"'), raw.index('"device_id"'))
        self.assertLess(raw.index('"device_id"'), raw.index('"state"'))
        self.assertLess(raw.index('"state"'), raw.index('"lease_id"'))
        status, body, _ = self._request(
            {"device_id": "bob", "job_id": "j1", "op": "dispatch"})
        self.assertEqual(status, 201)
        self.assertEqual(body["state"], "running")
        self.assertEqual(body["lease_id"], "j1")
        status, body, _ = self._request(
            {"device_id": "bob", "job_id": "j1", "op": "status"})
        self.assertEqual(status, 200)
        self.assertEqual(body["state"], "running")

    def test_bad_body_and_fields(self) -> None:
        status, body, _ = self._request(raw="not json")
        self.assertEqual(status, 400)
        self.assertEqual(list(body), ["message", "field"])
        self.assertEqual(body["field"], "request_body")
        status, body, _ = self._request(raw="[]")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "request_body")
        status, body, _ = self._request({"device_id": "bob", "job_id": "j1"})
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "op")
        status, body, _ = self._request(
            {"device_id": "bob", "job_id": "j1", "op": "run"})
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "op")

    def test_error_statuses(self) -> None:
        status, body, _ = self._request(
            {"device_id": "ghost", "job_id": "j1", "op": "queue"})
        self.assertEqual(status, 409)
        self.assertEqual(list(body), ["message", "field"])
        self.assertEqual(body["field"], "device_id")
        status, body, _ = self._request(
            {"device_id": "bob", "job_id": "j1", "op": "dispatch"})
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "job_id")
        self._request({"device_id": "bob", "job_id": "j1", "op": "queue"})
        status, body, _ = self._request(
            {"device_id": "bob2", "job_id": "j1", "op": "queue"})
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "job_id")


if __name__ == "__main__":
    unittest.main()
