"""Tests for ``POST /v1/inbox-jobs`` with ``op=recover``.

A recover re-establishes the expired (or released) lease of a ``running``
redelivery job: a non-empty selection is leased again under an ordinary
inbox lease named by the client ``recovery_id`` and the job stays
``running`` with that new ``lease_id``; an empty selection ends the job
``succeeded`` with ``lease_id`` null. The same ``recovery_id`` replays
(200, no write); cross-job/cross-lease id reuse conflicts 409; only an
expired/released running lease is recoverable.
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


_PAST = "2000-01-01T00:00:00.000000+00:00"


class InboxJobRecoverServiceTest(InboxMixin, unittest.TestCase):
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

    def _dispatch(self, job_id="j1", device_id="bob"):
        self._job(device_id=device_id, job_id=job_id)
        return self._job(device_id=device_id, job_id=job_id, op="dispatch")

    def _expire(self, lease_id: str) -> None:
        with self.service.store._lock:
            for state in self.service.store._delivery.values():
                for lease in state.leases:
                    if lease.lease_id == lease_id:
                        lease.leased_until = _PAST

    def test_recover_requires_recovery_id(self) -> None:
        self._dispatch()
        for payload in (
                {"device_id": "bob", "job_id": "j1", "op": "recover"},
                {"device_id": "bob", "job_id": "j1", "op": "recover",
                 "recovery_id": ""},
                {"device_id": "bob", "job_id": "j1", "op": "recover",
                 "recovery_id": 4},
                {"device_id": "bob", "job_id": "j1", "op": "recover",
                 "recovery_id": None},
                {"device_id": "bob", "job_id": "j1", "op": "recover",
                 "recovery_id": True}):
            error = self._error(
                lambda payload=payload: self.service.inbox_job(payload))
            self.assertEqual((error.status_code, error.field),
                             (400, "recovery_id"), payload)

    def test_recover_op_is_accepted_alongside_the_other_three(self) -> None:
        for bad_op in ("recover ", "Recover", "recovered"):
            error = self._error(lambda: self.service.inbox_job(
                {"device_id": "bob", "job_id": "j1", "op": bad_op,
                 "recovery_id": "r1"}))
            self.assertEqual((error.status_code, error.field), (400, "op"))

    def test_device_state_wins_over_job_and_recovery_checks(self) -> None:
        self._dispatch()
        error = self._error(lambda: self._job(
            device_id="ghost", op="recover", recovery_id="r1"))
        self.assertEqual((error.status_code, error.field),
                         (409, "device_id"))
        self.service.store.revoke_device("bob")
        error = self._error(lambda: self._job(
            op="recover", recovery_id="r1"))
        self.assertEqual((error.status_code, error.field),
                         (409, "device_id"))

    def test_unknown_job_404_and_cross_device_409(self) -> None:
        error = self._error(lambda: self._job(
            job_id="ghost", op="recover", recovery_id="r1"))
        self.assertEqual((error.status_code, error.field),
                         (404, "job_id"))
        self._dispatch()
        error = self._error(lambda: self._job(
            device_id="bob2", op="recover", recovery_id="r1"))
        self.assertEqual((error.status_code, error.field),
                         (409, "job_id"))

    def test_only_expired_or_released_running_lease_is_recoverable(self) -> None:
        self._dispatch()
        # The dispatch lease is still valid.
        error = self._error(lambda: self._job(
            op="recover", recovery_id="r1"))
        self.assertEqual((error.status_code, error.field),
                         (409, "lease_id"))
        # pending jobs cannot be recovered.
        self._job(job_id="jp")
        error = self._error(lambda: self._job(
            job_id="jp", op="recover", recovery_id="rp"))
        self.assertEqual((error.status_code, error.field),
                         (409, "job_id"))
        # terminal jobs cannot be recovered.
        self._expire("j1")
        self._job(op="recover", recovery_id="r1")
        self.service.inbox_lease_complete(
            "bob", "r1", {"completion_id": "c1", "outcome": "delivered"})
        error = self._error(lambda: self._job(
            op="recover", recovery_id="r2"))
        self.assertEqual((error.status_code, error.field),
                         (409, "job_id"))

    def test_recover_after_expiry_leases_under_ordinary_lease(self) -> None:
        body, status = self._dispatch()
        self.assertEqual(body["lease_id"], "j1")
        self._expire("j1")
        body, status = self._job(op="recover", recovery_id="r1")
        self.assertEqual(status, 201)
        self.assertEqual(list(body),
                         ["job_id", "device_id", "state", "lease_id"])
        self.assertEqual(body, {"job_id": "j1", "device_id": "bob",
                                "state": "running", "lease_id": "r1"})
        # The new lease is an ordinary inbox lease named by the recovery id.
        lease = self.service.inbox_lease_get("bob", "r1")
        self.assertEqual(lease["state"], "active")
        self.assertEqual(lease["limit"], 100)
        self.assertEqual([m["message_id"] for m in lease["messages"]],
                         ["a1", "a2", "a3", "b1", "b2"])
        # The old dispatch lease remains history but is expired.
        old = self.service.inbox_lease_get("bob", "j1")
        self.assertEqual(old["state"], "expired")
        # The recovered messages are withheld from an unrelated claim.
        empty, status = self.service.inbox_claim(
            "bob", {"lease_id": "L-other", "limit": 10})
        self.assertEqual(status, 200)
        self.assertEqual(empty["messages"], [])

    def test_recover_after_release(self) -> None:
        self._dispatch()
        released, status = self.service.inbox_release("bob", "j1")
        self.assertEqual(status, 201)
        body, status = self._job(op="recover", recovery_id="r1")
        self.assertEqual(status, 201)
        self.assertEqual(body["lease_id"], "r1")
        self.assertEqual(
            self.service.inbox_lease_get("bob", "r1")["state"], "active")

    def test_replay_is_idempotent_and_returns_current_view(self) -> None:
        self._dispatch()
        self._expire("j1")
        first, _ = self._job(op="recover", recovery_id="r1")
        replay, status = self._job(op="recover", recovery_id="r1")
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)
        # After the job terminates, the same id still replays the current
        # view (200) without writing.
        self.service.inbox_lease_complete(
            "bob", "r1", {"completion_id": "c1", "outcome": "failed"})
        replay, status = self._job(op="recover", recovery_id="r1")
        self.assertEqual(status, 200)
        self.assertEqual(replay["state"], "failed")
        self.assertEqual(replay["lease_id"], "r1")
        # The device gate precedes every job/replay decision for this
        # entry (mirroring queue/dispatch/status), so a replay after the
        # device is revoked is 409/device_id rather than an idempotent 200.
        self.service.store.revoke_device("bob")
        error = self._error(lambda: self._job(
            op="recover", recovery_id="r1"))
        self.assertEqual((error.status_code, error.field),
                         (409, "device_id"))

    def test_recovery_id_conflicts(self) -> None:
        # The id equals another job's job_id.
        self._dispatch(job_id="j1")
        self._expire("j1")
        self._job(job_id="j2")
        error = self._error(lambda: self._job(
            op="recover", recovery_id="j2"))
        self.assertEqual((error.status_code, error.field),
                         (409, "recovery_id"))
        # The id is already occupied by an unrelated inbox claim lease.
        self.service.inbox_claim("bob", {"lease_id": "L-used", "limit": 1})
        error = self._error(lambda: self._job(
            op="recover", recovery_id="L-used"))
        self.assertEqual((error.status_code, error.field),
                         (409, "recovery_id"))
        # The same id on another job conflicts.
        self._job(op="recover", recovery_id="r1")
        self._dispatch(job_id="j3")
        self._expire("j3")
        error = self._error(lambda: self._job(
            job_id="j3", op="recover", recovery_id="r1"))
        self.assertEqual((error.status_code, error.field),
                         (409, "recovery_id"))

    def test_sequential_recoveries(self) -> None:
        self._dispatch()
        self._expire("j1")
        self._job(op="recover", recovery_id="r1")
        self._expire("r1")
        body, status = self._job(op="recover", recovery_id="r2")
        self.assertEqual(status, 201)
        self.assertEqual(body["lease_id"], "r2")
        # r1 is expired history and no longer found by the job.
        status_view = self.service.inbox_lease_get("bob", "j1")
        self.assertEqual(status_view["state"], "expired")
        self.assertEqual(
            self.service.inbox_lease_get("bob", "r1")["state"], "expired")
        self.assertEqual(
            self.service.inbox_lease_get("bob", "r2")["state"], "active")

    def test_empty_recovery_succeeds_without_lease(self) -> None:
        self._dispatch()
        self._expire("j1")
        # Ack every bob message so the inbox is empty when recovering.
        self.service.sync_session_ack(
            self.sid1, {"device_id": "bob", "cursor": 3})
        self.service.sync_session_ack(
            self.sid2, {"device_id": "bob", "cursor": 2})
        body, status = self._job(op="recover", recovery_id="empty")
        self.assertEqual(status, 201)
        self.assertEqual(body, {"job_id": "j1", "device_id": "bob",
                                "state": "succeeded", "lease_id": None})
        # No lease was taken under the recovery id.
        error = self._error(
            lambda: self.service.inbox_lease_get("bob", "empty"))
        self.assertEqual((error.status_code, error.field),
                         (404, "lease_id"))
        # Replay returns the terminal view with 200.
        replay, status = self._job(op="recover", recovery_id="empty")
        self.assertEqual(status, 200)
        self.assertEqual(replay, body)

    def test_recovery_lease_completion_terminates_job(self) -> None:
        self._dispatch()
        self._expire("j1")
        self._job(op="recover", recovery_id="r1")
        self.service.inbox_lease_complete(
            "bob", "r1", {"completion_id": "c1", "outcome": "delivered"})
        body, _ = self._job(op="status")
        self.assertEqual(body["state"], "succeeded")
        self.assertEqual(body["lease_id"], "r1")
        # A failed outcome on a second recovery drives a failed job.
        self._dispatch(job_id="j2")
        self._expire("j2")
        self._job(job_id="j2", op="recover", recovery_id="e1")
        self._expire("e1")
        self._job(job_id="j2", op="recover", recovery_id="e2")
        self.service.inbox_lease_complete(
            "bob", "e2", {"completion_id": "c2", "outcome": "failed"})
        body, _ = self._job(job_id="j2", op="status")
        self.assertEqual(body["state"], "failed")
        self.assertEqual(body["lease_id"], "e2")


class InboxJobRecoverPersistenceTest(InboxMixin, unittest.TestCase):
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

    def _expire(self, lease_id: str) -> None:
        with self.service.store._lock:
            for state in self.service.store._delivery.values():
                for lease in state.leases:
                    if lease.lease_id == lease_id:
                        lease.leased_until = _PAST

    def test_recover_advances_one_generation_replay_none(self) -> None:
        self._dispatch()
        self._expire("j1")
        before = self.state_store.commit_seq
        _, status = self._job(op="recover", recovery_id="r1")
        self.assertEqual(status, 201)
        self.assertEqual(self.state_store.commit_seq, before + 1)
        _, status = self._job(op="recover", recovery_id="r1")
        self.assertEqual(status, 200)
        self.assertEqual(self.state_store.commit_seq, before + 1)

    def test_state_file_shape(self) -> None:
        # A job without recoveries keeps the four-key item (the fifth key
        # is omitted while empty).
        self._job(job_id="pending")
        self._dispatch(job_id="j1")
        self._expire("j1")
        self._job(op="recover", recovery_id="r1")
        self._expire("r1")
        self._job(op="recover", recovery_id="r2")
        with open(self.path, encoding="utf-8") as handle:
            document = json.load(handle)
        items = {item["job_id"]: item
                 for item in document["redelivery_jobs"]}
        self.assertEqual(list(items["pending"]),
                         ["job_id", "device_id", "state", "lease_id"])
        j1 = items["j1"]
        self.assertEqual(list(j1),
                         ["job_id", "device_id", "state", "lease_id",
                          "recoveries"])
        self.assertEqual(j1["recoveries"], [
            {"recovery_id": "r1", "lease_id": "r1"},
            {"recovery_id": "r2", "lease_id": "r2"},
        ])
        for record in j1["recoveries"]:
            self.assertEqual(list(record), ["recovery_id", "lease_id"])

    def test_restart_restores_recoveries_and_replays(self) -> None:
        self._dispatch()
        self._expire("j1")
        self._job(op="recover", recovery_id="r1")
        # A job on bob2 whose dispatch leased o1; acking it afterwards lets
        # the recovery find an empty inbox and terminate the job.
        self._job(device_id="bob2", job_id="j2")
        self._job(device_id="bob2", job_id="j2", op="dispatch")
        self._expire("j2")
        self.service.sync_session_ack(
            self.sid_other, {"device_id": "bob2", "cursor": 1})
        self._job(device_id="bob2", job_id="j2", op="recover",
                  recovery_id="empty")

        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        body, status = restarted.inbox_job(
            {"device_id": "bob", "job_id": "j1", "op": "status"})
        self.assertEqual(status, 200)
        self.assertEqual(body, {"job_id": "j1", "device_id": "bob",
                                "state": "running", "lease_id": "r1"})
        self.assertEqual(
            restarted.inbox_lease_get("bob", "r1")["state"], "active")
        body, status = restarted.inbox_job(
            {"device_id": "bob2", "job_id": "j2", "op": "recover",
             "recovery_id": "empty"})
        self.assertEqual(status, 200)
        self.assertEqual(body["state"], "succeeded")
        self.assertIsNone(body["lease_id"])
        # A recovery id committed on another job is still rejected after
        # restart (j2 owns "empty").
        with self.assertRaises(ServiceError) as caught:
            restarted.inbox_job(
                {"device_id": "bob", "job_id": "j1", "op": "recover",
                 "recovery_id": "empty"})
        self.assertEqual((caught.exception.status_code,
                          caught.exception.field), (409, "recovery_id"))
        self.assertTrue(restarted.persistence_integrity()["consistent"])

    def test_save_failure_rolls_back_recover(self) -> None:
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

        self._dispatch()
        self._expire("j1")
        generation = self.state_store.commit_seq
        persistence_mod.os.fsync = fail_first_directory_fsync
        try:
            with self.assertRaises(PersistenceUnavailable):
                self._job(op="recover", recovery_id="r1")
        finally:
            persistence_mod.os.fsync = real_fsync
        self.assertEqual(self.state_store.commit_seq, generation)
        # The recovery never committed: the job is still on its dispatch
        # lease. Memory was rolled back from the on-disk state, which also
        # discards the in-memory expiry hack, so expire it again before
        # the id can be reused.
        body, status = self._job(op="status")
        self.assertEqual(body["state"], "running")
        self.assertEqual(body["lease_id"], "j1")
        self._expire("j1")
        body, status = self._job(op="recover", recovery_id="r1")
        self.assertEqual(status, 201)
        self.assertEqual(body["lease_id"], "r1")
        self.assertEqual(self.state_store.commit_seq, generation + 1)

    def _bad_document(self, mutate):
        self._dispatch()
        self._expire("j1")
        self._job(op="recover", recovery_id="r1")
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

    def test_restore_rejects_malformed_recoveries(self) -> None:
        item = lambda d: d["redelivery_jobs"][0]
        self._bad_document(lambda d: item(d).__setitem__(
            "recoveries", {}))
        self._bad_document(lambda d: item(d)["recoveries"].append("x"))
        self._bad_document(lambda d: item(d)["recoveries"].append(
            {"recovery_id": "z", "lease_id": None, "extra": 1}))
        self._bad_document(lambda d: item(d)["recoveries"].append(
            {"recovery_id": "", "lease_id": None}))
        self._bad_document(lambda d: item(d)["recoveries"].append(
            {"recovery_id": "z", "lease_id": 3}))
        self._bad_document(lambda d: item(d)["recoveries"].append(
            {"recovery_id": "z", "lease_id": "other"}))
        self._bad_document(lambda d: item(d)["recoveries"].append(
            {"recovery_id": "r1", "lease_id": "r1"}))
        self._bad_document(lambda d: item(d).__setitem__(
            "state", "pending"))
        self._bad_document(lambda d: item(d).__setitem__(
            "lease_id", "ghost"))
        self._bad_document(lambda d: item(d).__setitem__(
            "recoveries", [{"recovery_id": "q", "lease_id": "q"}]))
        self._bad_document(lambda d: item(d).__setitem__(
            "state", "failed"))

    def test_legacy_item_without_recoveries_key_loads(self) -> None:
        self._dispatch()
        with open(self.path, encoding="utf-8") as handle:
            document = json.load(handle)
        self.assertNotIn("recoveries", document["redelivery_jobs"][0])
        restarted = DeviceService()
        attach_persistence(restarted, self.path)  # must not raise
        body, status = restarted.inbox_job(
            {"device_id": "bob", "job_id": "j1", "op": "status"})
        self.assertEqual(status, 200)
        self.assertEqual(body["state"], "running")


class InboxJobRecoverHTTPTest(InboxMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()
        self.server, self.service = create_server(
            "127.0.0.1", 0, self.service)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        self._dispatch()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _dispatch(self) -> None:
        self._request({"device_id": "bob", "job_id": "j1", "op": "queue"})
        self._request(
            {"device_id": "bob", "job_id": "j1", "op": "dispatch"})

    def _request(self, payload):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("POST", "/v1/inbox-jobs", body=json.dumps(payload),
                     headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        data = response.read().decode("utf-8")
        conn.close()
        return response.status, (json.loads(data) if data else None), data

    def test_recover_over_http(self) -> None:
        with self.service.store._lock:
            for state in self.service.store._delivery.values():
                for lease in state.leases:
                    if lease.lease_id == "j1":
                        lease.leased_until = _PAST
        status, body, raw = self._request(
            {"device_id": "bob", "job_id": "j1", "op": "recover",
             "recovery_id": "r1"})
        self.assertEqual(status, 201)
        self.assertEqual(body, {"job_id": "j1", "device_id": "bob",
                                "state": "running", "lease_id": "r1"})
        self.assertLess(raw.index('"job_id"'), raw.index('"device_id"'))
        self.assertLess(raw.index('"device_id"'), raw.index('"state"'))
        self.assertLess(raw.index('"state"'), raw.index('"lease_id"'))
        status, body, _ = self._request(
            {"device_id": "bob", "job_id": "j1", "op": "recover",
             "recovery_id": "r1"})
        self.assertEqual(status, 200)
        self.assertEqual(body["lease_id"], "r1")

    def test_recover_field_errors_over_http(self) -> None:
        status, body, _ = self._request(
            {"device_id": "bob", "job_id": "j1", "op": "recover"})
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "recovery_id")
        status, body, _ = self._request(
            {"device_id": "bob", "job_id": "j1", "op": "recover",
             "recovery_id": "r1"})
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "lease_id")
        status, body, _ = self._request(
            {"device_id": "ghost", "job_id": "j1", "op": "recover",
             "recovery_id": "r1"})
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "device_id")
        status, body, _ = self._request(
            {"device_id": "bob", "job_id": "ghost", "op": "recover",
             "recovery_id": "r1"})
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "job_id")


if __name__ == "__main__":
    unittest.main()
