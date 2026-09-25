"""Tests for ``POST /v1/inbox-jobs`` with ``op=cancel``.

A cancel moves a ``pending`` or ``running`` redelivery job to the terminal
``cancelled`` state: a running job's current lease is released in the same
locked transaction so its messages can be claimed again. The body carries
exactly the three common keys plus a non-empty string ``cancellation_id``
(the job-scoped idempotency key): the same id replays 200 without writing,
a different id on the cancelled job conflicts 409/cancellation_id, and a
terminal succeeded/failed job rejects 409/job_id.
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


class InboxJobCancelServiceTest(InboxMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def _error(self, callable_):
        with self.assertRaises(ServiceError) as caught:
            callable_()
        return caught.exception

    def _job(self, device_id="bob", job_id="j1", op="queue", **extra):
        payload = {"device_id": device_id, "job_id": job_id, "op": op}
        payload.update(extra)
        return self.service.inbox_job(payload)

    def _cancel(self, device_id="bob", job_id="j1", cancellation_id="c1"):
        return self._job(device_id=device_id, job_id=job_id, op="cancel",
                         cancellation_id=cancellation_id)

    def _dispatch(self, job_id="j1", device_id="bob"):
        self._job(device_id=device_id, job_id=job_id)
        return self._job(device_id=device_id, job_id=job_id, op="dispatch")

    def test_cancel_pending_job(self) -> None:
        self._job()
        body, status = self._cancel()
        self.assertEqual(status, 201)
        self.assertEqual(list(body),
                         ["job_id", "device_id", "state", "lease_id"])
        self.assertEqual(body, {"job_id": "j1", "device_id": "bob",
                                "state": "cancelled", "lease_id": None})

    def test_cancel_running_job_releases_its_lease(self) -> None:
        self._dispatch()
        body, status = self._cancel()
        self.assertEqual(status, 201)
        self.assertEqual(body, {"job_id": "j1", "device_id": "bob",
                                "state": "cancelled", "lease_id": "j1"})
        # The dispatch lease is released: it no longer withholds the
        # messages and reads back as released history.
        lease = self.service.inbox_lease_get("bob", "j1")
        self.assertEqual(lease["state"], "released")
        self.assertIsNotNone(lease["released_at"])
        # The messages can be claimed again under a fresh lease id.
        claim, status = self.service.inbox_claim(
            "bob", {"lease_id": "L-next", "limit": 10})
        self.assertEqual(status, 201)
        self.assertEqual([m["message_id"] for m in claim["messages"]],
                         ["a1", "a2", "a3", "b1", "b2"])

    def test_cancel_replay_same_id_is_idempotent(self) -> None:
        self._dispatch()
        first, _ = self._cancel()
        replay, status = self._cancel()
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)

    def test_cancelled_job_with_another_id_conflicts(self) -> None:
        self._job()
        self._cancel()
        error = self._error(lambda: self._cancel(cancellation_id="c2"))
        self.assertEqual((error.status_code, error.field),
                         (409, "cancellation_id"))

    def test_terminal_jobs_reject_cancel(self) -> None:
        self._dispatch()
        self.service.inbox_lease_complete(
            "bob", "j1", {"completion_id": "c1", "outcome": "delivered"})
        error = self._error(lambda: self._cancel())
        self.assertEqual((error.status_code, error.field), (409, "job_id"))
        self._dispatch(job_id="j2")
        self.service.inbox_lease_complete(
            "bob", "j2", {"completion_id": "c2", "outcome": "failed"})
        error = self._error(lambda: self._cancel(job_id="j2"))
        self.assertEqual((error.status_code, error.field), (409, "job_id"))

    def test_unknown_and_cross_device_job(self) -> None:
        error = self._error(lambda: self._cancel(job_id="ghost"))
        self.assertEqual((error.status_code, error.field), (404, "job_id"))
        self._job()
        error = self._error(lambda: self._cancel(device_id="bob2"))
        self.assertEqual((error.status_code, error.field), (409, "job_id"))

    def test_device_gate_precedes_job_and_replay(self) -> None:
        error = self._error(lambda: self._cancel(device_id="ghost"))
        self.assertEqual((error.status_code, error.field),
                         (409, "device_id"))
        self._job()
        self._cancel()
        self.service.store.revoke_device("bob")
        # Even an exact replay hits the device gate first.
        error = self._error(lambda: self._cancel())
        self.assertEqual((error.status_code, error.field),
                         (409, "device_id"))

    def test_cancelled_job_other_ops(self) -> None:
        self._job()
        self._cancel()
        # A dispatch on the cancelled job is a replay of the current view.
        body, status = self._job(op="dispatch")
        self.assertEqual(status, 200)
        self.assertEqual(body["state"], "cancelled")
        # Status is read-only and reports the cancelled state.
        body, status = self._job(op="status")
        self.assertEqual(status, 200)
        self.assertEqual(body["state"], "cancelled")
        # A cancelled job cannot be recovered.
        error = self._error(lambda: self._job(op="recover",
                                              recovery_id="r1"))
        self.assertEqual((error.status_code, error.field), (409, "job_id"))
        # A queue replay on the same id answers the current view.
        body, status = self._job()
        self.assertEqual(status, 200)
        self.assertEqual(body["state"], "cancelled")

    def test_cancel_body_validation(self) -> None:
        self._job()
        for payload in (
                {"device_id": "bob", "job_id": "j1", "op": "cancel"},
                {"device_id": "bob", "job_id": "j1", "op": "cancel",
                 "cancellation_id": ""},
                {"device_id": "bob", "job_id": "j1", "op": "cancel",
                 "cancellation_id": 4},
                {"device_id": "bob", "job_id": "j1", "op": "cancel",
                 "cancellation_id": None},
                {"device_id": "bob", "job_id": "j1", "op": "cancel",
                 "cancellation_id": True}):
            error = self._error(
                lambda payload=payload: self.service.inbox_job(payload))
            self.assertEqual((error.status_code, error.field),
                             (400, "cancellation_id"), payload)
        # Extra keys beyond the three common ones and cancellation_id are
        # rejected with the offending key as the field.
        for extra in ("recovery_id", "foo", "limit"):
            payload = {"device_id": "bob", "job_id": "j1", "op": "cancel",
                       "cancellation_id": "c1", extra: "x"}
            error = self._error(
                lambda payload=payload: self.service.inbox_job(payload))
            self.assertEqual((error.status_code, error.field),
                             (400, extra), payload)
        # The job was never cancelled by the rejected bodies.
        body, status = self._job(op="status")
        self.assertEqual(body["state"], "pending")
        # "cancel" is accepted alongside the other verbs; near-misses are
        # still 400/op.
        for bad_op in ("cancel ", "Cancel", "cancelled"):
            error = self._error(lambda: self.service.inbox_job(
                {"device_id": "bob", "job_id": "j1", "op": bad_op,
                 "cancellation_id": "c1"}))
            self.assertEqual((error.status_code, error.field), (400, "op"))


class InboxJobCancelPersistenceTest(InboxMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self._build()
        self.path = os.path.join(self.directory, "state.json")
        self.state_store = attach_persistence(self.service, self.path)

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def _job(self, device_id="bob", job_id="j1", op="queue", **extra):
        payload = {"device_id": device_id, "job_id": job_id, "op": op}
        payload.update(extra)
        return self.service.inbox_job(payload)

    def _cancel(self, device_id="bob", job_id="j1", cancellation_id="c1"):
        return self._job(device_id=device_id, job_id=job_id, op="cancel",
                         cancellation_id=cancellation_id)

    def _dispatch(self, job_id="j1", device_id="bob"):
        self._job(device_id=device_id, job_id=job_id)
        return self._job(device_id=device_id, job_id=job_id, op="dispatch")

    def test_cancel_advances_one_generation_and_writes_seven_keys(
            self) -> None:
        self._dispatch()
        before = self.state_store.commit_seq
        self._cancel()
        self.assertEqual(self.state_store.commit_seq, before + 1)
        with open(self.path, encoding="utf-8") as handle:
            document = json.load(handle)
        (item,) = document["redelivery_jobs"]
        self.assertEqual(list(item),
                         ["job_id", "device_id", "state", "lease_id",
                          "recoveries", "cancellation_id", "cancelled_at"])
        self.assertEqual(item["state"], "cancelled")
        self.assertEqual(item["lease_id"], "j1")
        self.assertEqual(item["recoveries"], [])
        self.assertEqual(item["cancellation_id"], "c1")
        self.assertIsInstance(item["cancelled_at"], str)
        self.assertTrue(item["cancelled_at"].endswith("+00:00"))
        # Six microsecond digits, matching the server's canonical form.
        fraction = item["cancelled_at"].split("T")[1].split("+")[0]
        self.assertEqual(len(fraction.split(".")[1]), 6)
        # The released dispatch lease carries the same timestamp.
        leases = [lease for record in document["delivery"]
                  for lease in record.get("leases", [])
                  if lease["lease_id"] == "j1"]
        self.assertTrue(leases)
        for lease in leases:
            self.assertEqual(lease["released_at"], item["cancelled_at"])

    def test_replay_consumes_no_generation(self) -> None:
        self._job()
        self._cancel()
        generation = self.state_store.commit_seq
        self._cancel()
        self.assertEqual(self.state_store.commit_seq, generation)

    def test_restart_restores_cancellation(self) -> None:
        self._dispatch()
        self._cancel()
        self._job(job_id="jp")
        self._cancel(job_id="jp", cancellation_id="cp")

        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        body, status = restarted.inbox_job(
            {"device_id": "bob", "job_id": "j1", "op": "status"})
        self.assertEqual(status, 200)
        self.assertEqual(body, {"job_id": "j1", "device_id": "bob",
                                "state": "cancelled", "lease_id": "j1"})
        body, _ = restarted.inbox_job(
            {"device_id": "bob", "job_id": "jp", "op": "status"})
        self.assertEqual((body["state"], body["lease_id"]),
                         ("cancelled", None))
        # The committed cancel still replays as 200 and a different id
        # still conflicts after the restart.
        body, status = restarted.inbox_job(
            {"device_id": "bob", "job_id": "j1", "op": "cancel",
             "cancellation_id": "c1"})
        self.assertEqual(status, 200)
        with self.assertRaises(ServiceError) as caught:
            restarted.inbox_job(
                {"device_id": "bob", "job_id": "j1", "op": "cancel",
                 "cancellation_id": "other"})
        self.assertEqual((caught.exception.status_code,
                          caught.exception.field),
                         (409, "cancellation_id"))
        # The released lease still no longer withholds the messages.
        claim, status = restarted.inbox_claim(
            "bob", {"lease_id": "L-next", "limit": 10})
        self.assertEqual(status, 201)
        self.assertEqual(len(claim["messages"]), 5)
        self.assertTrue(restarted.persistence_integrity()["consistent"])

    def test_save_failure_rolls_back_cancel(self) -> None:
        import e2ee_backend.persistence as persistence_mod

        real_fsync = persistence_mod.os.fsync
        calls = {"n": 0}

        def fail_first_directory_fsync(fd: int) -> None:  # noqa: ANN001
            if os.fstat(fd).st_mode & 0o170000 == 0o040000:
                calls["n"] += 1
                if calls["n"] == 1:
                    raise OSError("transient directory fsync failure")
            real_fsync(fd)

        self._dispatch()
        generation = self.state_store.commit_seq
        persistence_mod.os.fsync = fail_first_directory_fsync
        try:
            with self.assertRaises(PersistenceUnavailable):
                self._cancel()
        finally:
            persistence_mod.os.fsync = real_fsync
        # Rolled back: no generation consumed, the job is still running
        # and its lease was not released.
        self.assertEqual(self.state_store.commit_seq, generation)
        body, _ = self._job(op="status")
        self.assertEqual((body["state"], body["lease_id"]),
                         ("running", "j1"))
        lease = self.service.inbox_lease_get("bob", "j1")
        self.assertEqual(lease["state"], "active")
        # The cancel can be retried and now commits.
        _, status = self._cancel()
        self.assertEqual(status, 201)
        self.assertEqual(self.state_store.commit_seq, generation + 1)

    def _document_with_cancel(self, mutate):
        self._dispatch()
        self._cancel()
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
        with open(bad_path, "rb") as handle:
            self.assertEqual(handle.read(), original)

    def test_restore_rejects_cancellation_contradictions(self) -> None:
        item = lambda d: d["redelivery_jobs"][0]  # noqa: E731
        # A cancelled job without its cancellation pair.
        self._assert_refuses_startup(self._document_with_cancel(
            lambda d: item(d).__setitem__("cancellation_id", None)))
        self._assert_refuses_startup(self._document_with_cancel(
            lambda d: item(d).__setitem__("cancelled_at", None)))
        self._assert_refuses_startup(self._document_with_cancel(
            lambda d: item(d).pop("cancellation_id")))
        # Malformed cancellation fields.
        self._assert_refuses_startup(self._document_with_cancel(
            lambda d: item(d).__setitem__("cancellation_id", "")))
        self._assert_refuses_startup(self._document_with_cancel(
            lambda d: item(d).__setitem__("cancellation_id", 4)))
        self._assert_refuses_startup(self._document_with_cancel(
            lambda d: item(d).__setitem__("cancelled_at", "not-a-time")))
        self._assert_refuses_startup(self._document_with_cancel(
            lambda d: item(d).__setitem__(
                "cancelled_at", "2026-09-26T10:00:00+00:00")))
        # A non-cancelled state carrying a cancellation.
        self._assert_refuses_startup(self._document_with_cancel(
            lambda d: item(d).__setitem__("state", "running")))
        # A cancelled job whose dispatch lease was never released.
        def unrelease(document):
            for record in document["delivery"]:
                for lease in record.get("leases", []):
                    if lease["lease_id"] == "j1":
                        lease["released_at"] = None
        self._assert_refuses_startup(
            self._document_with_cancel(unrelease))
        # A cancelled job whose lease_id does not name its dispatch lease.
        self._assert_refuses_startup(self._document_with_cancel(
            lambda d: item(d).__setitem__("lease_id", "ghost")))

    def test_legacy_five_key_item_loads(self) -> None:
        self._dispatch()
        # A legacy item predates the cancellation pair: strip the two keys
        # (written to a fresh path without the integrity marker). It loads
        # with no cancellation and stays consistent.
        with open(self.path, encoding="utf-8") as handle:
            document = json.load(handle)
        for item in document["redelivery_jobs"]:
            item.pop("cancellation_id", None)
            item.pop("cancelled_at", None)
        self.assertNotIn("cancellation_id", document["redelivery_jobs"][0])
        document.pop("integrity_log_version", None)
        legacy_path = os.path.join(self.directory, "legacy.json")
        with open(legacy_path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        restarted = DeviceService()
        attach_persistence(restarted, legacy_path)  # must not raise
        body, status = restarted.inbox_job(
            {"device_id": "bob", "job_id": "j1", "op": "status"})
        self.assertEqual(status, 200)
        self.assertEqual(body["state"], "running")
        # The restored job can still be cancelled.
        body, status = restarted.inbox_job(
            {"device_id": "bob", "job_id": "j1", "op": "cancel",
             "cancellation_id": "c1"})
        self.assertEqual(status, 201)
        self.assertEqual(body["state"], "cancelled")


class InboxJobCancelHTTPTest(InboxMixin, unittest.TestCase):
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

    def test_cancel_flow_and_key_order(self) -> None:
        self._request({"device_id": "bob", "job_id": "j1", "op": "queue"})
        self._request({"device_id": "bob", "job_id": "j1", "op": "dispatch"})
        status, body, raw = self._request(
            {"device_id": "bob", "job_id": "j1", "op": "cancel",
             "cancellation_id": "c1"})
        self.assertEqual(status, 201)
        self.assertEqual(list(body),
                         ["job_id", "device_id", "state", "lease_id"])
        self.assertEqual(body["state"], "cancelled")
        self.assertEqual(body["lease_id"], "j1")
        self.assertLess(raw.index('"job_id"'), raw.index('"device_id"'))
        self.assertLess(raw.index('"device_id"'), raw.index('"state"'))
        self.assertLess(raw.index('"state"'), raw.index('"lease_id"'))
        # The same cancellation_id replays as 200 with the same view.
        status, replay, _ = self._request(
            {"device_id": "bob", "job_id": "j1", "op": "cancel",
             "cancellation_id": "c1"})
        self.assertEqual(status, 200)
        self.assertEqual(replay, body)
        # A different id conflicts; a terminal-state error names job_id.
        status, error, _ = self._request(
            {"device_id": "bob", "job_id": "j1", "op": "cancel",
             "cancellation_id": "c2"})
        self.assertEqual(status, 409)
        self.assertEqual(list(error), ["message", "field"])
        self.assertEqual(error["field"], "cancellation_id")

    def test_cancel_body_errors(self) -> None:
        status, body, _ = self._request(
            {"device_id": "bob", "job_id": "j1", "op": "cancel"})
        self.assertEqual(status, 400)
        self.assertEqual(list(body), ["message", "field"])
        self.assertEqual(body["field"], "cancellation_id")
        status, body, _ = self._request(
            {"device_id": "bob", "job_id": "j1", "op": "cancel",
             "cancellation_id": "c1", "extra": 1})
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "extra")
        status, body, _ = self._request(
            {"device_id": "ghost", "job_id": "j1", "op": "cancel",
             "cancellation_id": "c1"})
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "device_id")
        status, body, _ = self._request(
            {"device_id": "bob", "job_id": "ghost", "op": "cancel",
             "cancellation_id": "c1"})
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "job_id")


if __name__ == "__main__":
    unittest.main()
