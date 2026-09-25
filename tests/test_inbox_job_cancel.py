"""Tests for ``POST /v1/inbox-jobs`` with ``op=cancel``.

A cancel ends a ``pending`` or ``running`` redelivery job as ``cancelled``:
a pending cancel keeps ``lease_id`` null and touches no lease; a running
cancel releases the job's current lease so its messages can be claimed
again (the lease stays on the records as released history) while the job
keeps that lease id. The same ``cancellation_id`` on the same job replays
the frozen first response (200, no write); a different id on an already
cancelled job conflicts 409/cancellation_id; a succeeded/failed job
conflicts 409/job_id. The device gate precedes the job gate (unknown job
404/job_id, cross-device 409/job_id, unknown/revoked device
409/device_id).
"""
import json
import os
import re
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
_UTC = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}\+00:00$")


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

    def _queue(self, job_id="j1", device_id="bob"):
        return self._job(device_id=device_id, job_id=job_id)

    def _dispatch(self, job_id="j1", device_id="bob"):
        self._job(device_id=device_id, job_id=job_id)
        return self._job(device_id=device_id, job_id=job_id, op="dispatch")

    def _cancel(self, job_id="j1", device_id="bob",
                cancellation_id="c1", **extra):
        return self._job(device_id=device_id, job_id=job_id, op="cancel",
                         cancellation_id=cancellation_id, **extra)

    def test_cancel_pending_job(self) -> None:
        self._queue()
        body, status = self._cancel()
        self.assertEqual(status, 201)
        self.assertEqual(list(body),
                         ["job_id", "device_id", "state", "lease_id"])
        self.assertEqual(body, {"job_id": "j1", "device_id": "bob",
                                "state": "cancelled", "lease_id": None})

    def test_cancel_running_job_releases_its_lease(self) -> None:
        self._dispatch()
        lease = self.service.inbox_lease_get("bob", "j1")
        self.assertEqual(lease["state"], "active")
        body, status = self._cancel()
        self.assertEqual(status, 201)
        self.assertEqual(body, {"job_id": "j1", "device_id": "bob",
                                "state": "cancelled", "lease_id": "j1"})
        # The lease is released, so its messages are claimable again.
        lease = self.service.inbox_lease_get("bob", "j1")
        self.assertEqual(lease["state"], "released")
        self.assertIsNotNone(lease["released_at"])
        rest, claim_status = self.service.inbox_claim(
            "bob", {"lease_id": "L-after-cancel", "limit": 10})
        self.assertEqual(claim_status, 201)
        self.assertEqual([m["message_id"] for m in rest["messages"]],
                         ["a1", "a2", "a3", "b1", "b2"])

    def test_running_cancel_leases_messages_but_job_stays_cancelled(self) -> None:
        self._dispatch()
        self._cancel()
        # The released dispatch lease never moves the job out of cancelled.
        body, _ = self._job(op="status")
        self.assertEqual(body["state"], "cancelled")
        self.assertEqual(body["lease_id"], "j1")

    def test_cancel_replay_is_idempotent(self) -> None:
        self._queue()
        first, status = self._cancel()
        self.assertEqual(status, 201)
        replay, status = self._cancel()
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)

    def test_cancel_running_replay_returns_frozen_response(self) -> None:
        self._dispatch()
        first, _ = self._cancel()
        replay, status = self._cancel()
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)
        self.assertEqual(replay["state"], "cancelled")

    def test_different_cancellation_id_conflicts(self) -> None:
        self._queue()
        self._cancel(cancellation_id="c1")
        error = self._error(lambda: self._cancel(cancellation_id="c2"))
        self.assertEqual((error.status_code, error.field),
                         (409, "cancellation_id"))

    def test_terminal_succeeded_and_failed_conflict(self) -> None:
        self._dispatch()
        self.service.inbox_lease_complete(
            "bob", "j1", {"completion_id": "ok", "outcome": "delivered"})
        error = self._error(lambda: self._cancel())
        self.assertEqual((error.status_code, error.field),
                         (409, "job_id"))
        # A different cancellation_id on a succeeded job is still a job
        # terminal-state conflict, not a cancellation_id conflict.
        error = self._error(
            lambda: self._cancel(cancellation_id="other"))
        self.assertEqual((error.status_code, error.field),
                         (409, "job_id"))

        self._dispatch(job_id="j2")
        self.service.inbox_lease_complete(
            "bob", "j2", {"completion_id": "bad", "outcome": "failed"})
        error = self._error(
            lambda: self._cancel(job_id="j2"))
        self.assertEqual((error.status_code, error.field),
                         (409, "job_id"))

    def test_unknown_job_404_and_cross_device_409(self) -> None:
        error = self._error(lambda: self._cancel(job_id="ghost"))
        self.assertEqual((error.status_code, error.field),
                         (404, "job_id"))
        self._queue()
        error = self._error(
            lambda: self._cancel(device_id="bob2"))
        self.assertEqual((error.status_code, error.field),
                         (409, "job_id"))

    def test_device_gate_precedes_job_and_replay(self) -> None:
        self._queue()
        error = self._error(
            lambda: self._cancel(device_id="ghost"))
        self.assertEqual((error.status_code, error.field),
                         (409, "device_id"))
        self._cancel()
        # Even an exact replay after revocation loses to the device gate.
        self.service.store.revoke_device("bob")
        error = self._error(lambda: self._cancel())
        self.assertEqual((error.status_code, error.field),
                         (409, "device_id"))

    def test_cancel_after_recover_releases_recovery_lease(self) -> None:
        self._dispatch()
        with self.service.store._lock:
            for state in self.service.store._delivery.values():
                for lease in state.leases:
                    if lease.lease_id == "j1":
                        lease.leased_until = _PAST
        self._job(op="recover", recovery_id="r1")
        body, status = self._cancel(cancellation_id="c1")
        self.assertEqual(status, 201)
        self.assertEqual(body, {"job_id": "j1", "device_id": "bob",
                                "state": "cancelled", "lease_id": "r1"})
        self.assertEqual(
            self.service.inbox_lease_get("bob", "r1")["state"], "released")
        # Messages are claimable under a fresh lease.
        rest, claim_status = self.service.inbox_claim(
            "bob", {"lease_id": "L-again", "limit": 10})
        self.assertEqual(claim_status, 201)
        self.assertEqual(len(rest["messages"]), 5)

    def test_cancel_running_with_already_released_lease_keeps_stamp(self) -> None:
        self._dispatch()
        released, _ = self.service.inbox_release("bob", "j1")
        body, status = self._cancel()
        self.assertEqual(status, 201)
        self.assertEqual(body["state"], "cancelled")
        lease = self.service.inbox_lease_get("bob", "j1")
        self.assertEqual(lease["state"], "released")
        # The cancel does not overwrite the earlier release timestamp.
        self.assertEqual(lease["released_at"], released["released_at"])

    # -- validation -------------------------------------------------------

    def test_cancel_requires_cancellation_id(self) -> None:
        self._queue()
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

    def test_cancel_rejects_extra_keys(self) -> None:
        self._queue()
        for payload in (
                {"device_id": "bob", "job_id": "j1", "op": "cancel",
                 "cancellation_id": "c1", "recovery_id": "x"},
                {"device_id": "bob", "job_id": "j1", "op": "cancel",
                 "cancellation_id": "c1", "bogus": 1},
                {"device_id": "bob", "job_id": "j1", "op": "cancel",
                 "cancellation_id": "c1", "extra": None}):
            error = self._error(
                lambda payload=payload: self.service.inbox_job(payload))
            self.assertEqual(error.status_code, 400, payload)
            self.assertIn(error.field, ("recovery_id", "bogus", "extra"))

    def test_non_cancel_ops_still_ignore_extra_keys(self) -> None:
        # The strict key set applies only to cancel; the other verbs keep
        # their historical tolerance for unknown body keys.
        body, status = self.service.inbox_job(
            {"device_id": "bob", "job_id": "j1", "op": "queue",
             "cancellation_id": "ignored"})
        self.assertEqual(status, 201)
        self.assertEqual(body["state"], "pending")

    def test_cancel_op_string_validated_like_other_ops(self) -> None:
        for bad_op in ("Cancel", "cancel ", "cancelled"):
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

    def _dispatch(self, job_id="j1", device_id="bob"):
        self._job(device_id=device_id, job_id=job_id)
        return self._job(device_id=device_id, job_id=job_id, op="dispatch")

    def _cancel(self, job_id="j1", device_id="bob",
                cancellation_id="c1"):
        return self._job(device_id=device_id, job_id=job_id, op="cancel",
                         cancellation_id=cancellation_id)

    def _expire(self, lease_id: str) -> None:
        with self.service.store._lock:
            for state in self.service.store._delivery.values():
                for lease in state.leases:
                    if lease.lease_id == lease_id:
                        lease.leased_until = _PAST

    def _read(self):
        with open(self.path, encoding="utf-8") as handle:
            return json.load(handle)

    def test_pending_and_running_cancel_each_advance_one_generation(self) -> None:
        before = self.state_store.commit_seq
        self._job(job_id="jp")
        self._cancel(job_id="jp")
        self.assertEqual(self.state_store.commit_seq, before + 2)
        self._dispatch(job_id="j1")
        generation = self.state_store.commit_seq
        self._cancel(job_id="j1")
        self.assertEqual(self.state_store.commit_seq, generation + 1)
        document = self._read()
        jobs = {item["job_id"]: item
                for item in document["redelivery_jobs"]}
        self.assertEqual(jobs["jp"], {
            "job_id": "jp", "device_id": "bob", "state": "cancelled",
            "lease_id": None, "recoveries": [],
            "cancellation_id": "c1", "cancelled_at": jobs["jp"][
                "cancelled_at"]})
        self.assertTrue(_UTC.match(jobs["jp"]["cancelled_at"]))
        self.assertEqual(jobs["j1"]["state"], "cancelled")
        self.assertEqual(jobs["j1"]["lease_id"], "j1")
        self.assertEqual(jobs["j1"]["cancellation_id"], "c1")
        self.assertTrue(_UTC.match(jobs["j1"]["cancelled_at"]))
        self.assertEqual(list(jobs["j1"]),
                         ["job_id", "device_id", "state", "lease_id",
                          "recoveries", "cancellation_id", "cancelled_at"])

    def test_cancel_replay_consumes_no_generation(self) -> None:
        self._dispatch()
        self._cancel()
        generation = self.state_store.commit_seq
        self._cancel()
        self._cancel()
        self.assertEqual(self.state_store.commit_seq, generation)

    def test_restart_restores_cancelled_jobs(self) -> None:
        self._job(job_id="jp")
        self._cancel(job_id="jp")
        self._dispatch(job_id="j1")
        self._cancel(job_id="j1")
        # A cancelled job whose lease had been recovered before cancel.
        self._dispatch(job_id="j2")
        self._expire("j2")
        self._job(job_id="j2", op="recover", recovery_id="r2")
        self._cancel(job_id="j2", cancellation_id="c2")

        restarted = DeviceService()
        attach_persistence(restarted, self.path)

        def view(job_id):
            body, status = restarted.inbox_job(
                {"device_id": "bob", "job_id": job_id, "op": "status"})
            self.assertEqual(status, 200)
            return body

        self.assertEqual(view("jp"),
                         {"job_id": "jp", "device_id": "bob",
                          "state": "cancelled", "lease_id": None})
        self.assertEqual(view("j1"),
                         {"job_id": "j1", "device_id": "bob",
                          "state": "cancelled", "lease_id": "j1"})
        self.assertEqual(view("j2"),
                         {"job_id": "j2", "device_id": "bob",
                          "state": "cancelled", "lease_id": "r2"})
        # Replay after restart returns the frozen view with 200.
        replay, status = restarted.inbox_job(
            {"device_id": "bob", "job_id": "j1", "op": "cancel",
             "cancellation_id": "c1"})
        self.assertEqual(status, 200)
        self.assertEqual(replay["state"], "cancelled")
        # A different cancellation id conflicts after restart.
        with self.assertRaises(ServiceError) as caught:
            restarted.inbox_job(
                {"device_id": "bob", "job_id": "j1", "op": "cancel",
                 "cancellation_id": "other"})
        self.assertEqual((caught.exception.status_code,
                          caught.exception.field),
                         (409, "cancellation_id"))
        # The released leases stay released; messages are claimable.
        lease = restarted.inbox_lease_get("bob", "j1")
        self.assertEqual(lease["state"], "released")
        self.assertTrue(restarted.persistence_integrity()["consistent"])

    def test_cancelled_jobs_have_utc_cancelled_at(self) -> None:
        self._dispatch()
        self._cancel()
        jobs = {item["job_id"]: item
                for item in self._read()["redelivery_jobs"]}
        stamp = jobs["j1"]["cancelled_at"]
        self.assertTrue(_UTC.match(stamp), stamp)

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
        # Rolled back: the job is still running and the lease still active.
        self.assertEqual(self.state_store.commit_seq, generation)
        body, _ = self._job(op="status")
        self.assertEqual(body["state"], "running")
        self.assertEqual(body["lease_id"], "j1")
        lease = self.service.inbox_lease_get("bob", "j1")
        self.assertEqual(lease["state"], "active")
        # The cancel can be retried and now commits (same id still fresh).
        body, status = self._cancel()
        self.assertEqual(status, 201)
        self.assertEqual(body["state"], "cancelled")
        self.assertEqual(self.state_store.commit_seq, generation + 1)

    def _bad_document(self, mutate):
        self._dispatch()
        self._cancel()
        with open(self.path, encoding="utf-8") as handle:
            document = json.load(handle)
        mutate(document)
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

    def test_restore_rejects_malformed_cancellation_fields(self) -> None:
        item = lambda d: d["redelivery_jobs"][0]
        self._bad_document(lambda d: item(d).__setitem__(
            "cancellation_id", ""))
        self._bad_document(lambda d: item(d).__setitem__(
            "cancellation_id", 7))
        self._bad_document(lambda d: item(d).__setitem__(
            "cancelled_at", "2026-01-01"))
        self._bad_document(lambda d: item(d).__setitem__(
            "cancelled_at", None))
        # A cancelled job missing either field is contradictory.
        self._bad_document(lambda d: item(d).pop("cancellation_id"))
        self._bad_document(lambda d: item(d).pop("cancelled_at"))

    def test_restore_rejects_cancellation_state_contradictions(self) -> None:
        item = lambda d: d["redelivery_jobs"][0]
        # Cancelled fields on a non-cancelled job.
        def not_cancelled_with_fields(document):
            item(document)["state"] = "running"
        self._bad_document(not_cancelled_with_fields)
        # A cancelled job whose dispatch lease is not released.
        def cancelled_with_active_lease(document):
            for record in document["delivery"]:
                for lease in record.get("leases", []):
                    if lease["lease_id"] == "j1":
                        lease.pop("released_at", None)
        self._bad_document(cancelled_with_active_lease)

    def test_legacy_items_load_with_null_cancellation_fields(self) -> None:
        self._dispatch()
        with open(self.path, encoding="utf-8") as handle:
            document = json.load(handle)
        # Strip both new keys (and the recoveries key) to emulate the
        # oldest four-key items, then load from a marker-less path.
        for job in document["redelivery_jobs"]:
            job.pop("recoveries", None)
            job.pop("cancellation_id", None)
            job.pop("cancelled_at", None)
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
        # After the first commit the item is rewritten with seven keys and
        # null cancellation fields.
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

    def test_cancel_full_flow_and_key_order(self) -> None:
        self._queue = self.service.inbox_job(
            {"device_id": "bob", "job_id": "jp", "op": "queue"})
        status, body, raw = self._request(
            {"device_id": "bob", "job_id": "jp", "op": "cancel",
             "cancellation_id": "c1"})
        self.assertEqual(status, 201)
        self.assertEqual(list(body),
                         ["job_id", "device_id", "state", "lease_id"])
        self.assertLess(raw.index('"job_id"'), raw.index('"device_id"'))
        self.assertLess(raw.index('"device_id"'), raw.index('"state"'))
        self.assertLess(raw.index('"state"'), raw.index('"lease_id"'))
        self.assertEqual(body, {"job_id": "jp", "device_id": "bob",
                                "state": "cancelled", "lease_id": None})

    def test_cancel_error_bodies(self) -> None:
        status, body, _ = self._request(
            {"device_id": "bob", "job_id": "ghost", "op": "cancel",
             "cancellation_id": "c1"})
        self.assertEqual(status, 404)
        self.assertEqual(list(body), ["message", "field"])
        self.assertEqual(body["field"], "job_id")
        self.service.inbox_job(
            {"device_id": "bob", "job_id": "j1", "op": "queue"})
        status, body, _ = self._request(
            {"device_id": "bob", "job_id": "j1", "op": "cancel"})
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "cancellation_id")
        status, body, _ = self._request(
            {"device_id": "bob", "job_id": "j1", "op": "cancel",
             "cancellation_id": "c1", "recovery_id": "x"})
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "recovery_id")
        self.service.inbox_job(
            {"device_id": "bob", "job_id": "j1", "op": "cancel",
             "cancellation_id": "c1"})
        status, body, _ = self._request(
            {"device_id": "bob", "job_id": "j1", "op": "cancel",
             "cancellation_id": "c2"})
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "cancellation_id")


if __name__ == "__main__":
    unittest.main()
