"""Tests for the idempotent batch expired-registration cleanup.

``POST /v1/event-gc-batch/cleanup-expired`` in ``commit`` mode accepts an
optional ``request_id`` (a non-empty string, otherwise 400/request_id; its
presence on a ``preview`` is 400/request_id too; any other extra key stays
400/that key). A first-seen id commits the deletions, the frozen response
and the idempotency record in the one locked transaction (a delete-nothing
commit included — exactly one ``commit_seq`` advance). A replay with the
identical paged input (``device_ids`` in order, ``after``, ``limit``)
returns the first status code and the byte-identical first response
without writing or advancing ``commit_seq``, even if the state has since
changed; the same id with any differing input field is 409/request_id.
The records persist in the ``event_gc_batch_cleanup_requests`` section
(right after ``event_gc``), so the guarantee survives a restart; a
duplicate id, an illegal key order/type or a paging/device-order
contradiction in that section refuses startup without touching the file.
"""
import json
import os
import shutil
import tempfile
import threading
import unittest
from http.client import HTTPConnection

from e2ee_backend.models import Device, SignedPreKey
from e2ee_backend.service import DeviceService, ServiceError
from e2ee_backend.http_app import create_server
from e2ee_backend.persistence import (
    PersistenceUnavailable,
    StateFileError,
    attach_persistence,
)

PATH = "/v1/event-gc-batch/cleanup-expired"


class IdempotentBatchMixin:
    def _build(self) -> None:
        self.service = DeviceService()
        self.service.store.add_device(Device("u", "alice", "ik"))
        self.service.store.add_device(Device(
            "u", "bob", "ik",
            prekeys=[SignedPreKey("pk1", "pubk1")]))
        self.service.store.add_device(Device("u", "dave", "ik"))
        self.sid = self.service.store.create_session(
            "alice", "bob", "pk1", "ek1").session_id

    def _op(self, job_id, device="bob"):
        return self.service.inbox_job(
            {"device_id": device, "job_id": job_id, "op": "queue"})

    def _touch(self, consumer, device="bob"):
        return self.service.event_gc(
            device, {"consumer": consumer, "op": "touch", "seq": None})

    def _expire_lease(self, consumer, device="bob"):
        record = self.service.store._redelivery_job_event_checkpoints[
            (device, consumer)]
        record.expires = "2020-01-01T00:00:00.000000+00:00"

    def _seed_expired(self, consumer="c1", device="bob"):
        self._touch(consumer, device=device)
        self._expire_lease(consumer, device=device)

    def _batch(self, mode, device_ids, after=0, limit=100, request_id=None):
        payload = {"mode": mode, "device_ids": device_ids,
                   "after": after, "limit": limit}
        if request_id is not None:
            payload["request_id"] = request_id
        return self.service.event_gc_cleanup_expired_batch(payload)


class IdempotentBatchServiceTest(IdempotentBatchMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def test_replay_returns_frozen_response_and_status(self) -> None:
        self._op("J1")
        self._seed_expired("c1")
        body, status = self._batch(
            "commit", ["bob", "ghost"], request_id="R1")
        self.assertEqual(status, 201)
        self.assertEqual(body["results"][0]["removed"], 1)
        # New expirable state appears after the commit.
        self._seed_expired("c2")
        replay, replay_status = self._batch(
            "commit", ["bob", "ghost"], request_id="R1")
        self.assertEqual(replay_status, 201)
        self.assertEqual(replay, body)
        self.assertEqual(list(replay), list(body))
        # The replay deleted nothing: c2's expired lease survives.
        self.assertIn(("bob", "c2"),
                      self.service.store._redelivery_job_event_checkpoints)

    def test_replay_of_delete_nothing_commit_keeps_200(self) -> None:
        self._op("J1")
        self._touch("c1")
        body, status = self._batch("commit", ["bob"], request_id="R1")
        self.assertEqual(status, 200)
        self.assertEqual(body["results"][0]["removed"], 0)
        self._seed_expired("c2")
        replay, replay_status = self._batch("commit", ["bob"],
                                            request_id="R1")
        self.assertEqual(replay_status, 200)
        self.assertEqual(replay, body)
        self.assertIn(("bob", "c2"),
                      self.service.store._redelivery_job_event_checkpoints)

    def test_same_id_with_different_input_is_409(self) -> None:
        self._batch("commit", ["bob"], request_id="R1")
        for payload in (
                {"mode": "commit", "device_ids": ["bob", "alice"],
                 "after": 0, "limit": 100, "request_id": "R1"},
                {"mode": "commit", "device_ids": ["alice", "bob"],
                 "after": 0, "limit": 100, "request_id": "R1"},
                {"mode": "commit", "device_ids": ["bob"], "after": 1,
                 "limit": 100, "request_id": "R1"},
                {"mode": "commit", "device_ids": ["bob"], "after": 0,
                 "limit": 1, "request_id": "R1"}):
            with self.subTest(payload=payload):
                with self.assertRaises(ServiceError) as caught:
                    self.service.event_gc_cleanup_expired_batch(payload)
                self.assertEqual(caught.exception.status_code, 409)
                self.assertEqual(caught.exception.field, "request_id")

    def test_device_ids_order_matters_for_replay(self) -> None:
        body, status = self._batch(
            "commit", ["bob", "alice"], request_id="R1")
        self.assertEqual(status, 200)
        with self.assertRaises(ServiceError) as caught:
            self._batch("commit", ["alice", "bob"], request_id="R1")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "request_id")

    def test_request_id_rejected_on_preview(self) -> None:
        with self.assertRaises(ServiceError) as caught:
            self._batch("preview", ["bob"], request_id="R1")
        self.assertEqual(caught.exception.status_code, 400)
        self.assertEqual(caught.exception.field, "request_id")

    def test_bad_request_id(self) -> None:
        for value in ("", 1, None, True, ["R1"], {"r": 1}):
            with self.subTest(value=value):
                with self.assertRaises(ServiceError) as caught:
                    self.service.event_gc_cleanup_expired_batch({
                        "mode": "commit", "device_ids": ["bob"],
                        "after": 0, "limit": 1, "request_id": value})
                self.assertEqual(caught.exception.status_code, 400)
                self.assertEqual(caught.exception.field, "request_id")

    def test_extra_key_still_rejected(self) -> None:
        with self.assertRaises(ServiceError) as caught:
            self.service.event_gc_cleanup_expired_batch({
                "mode": "commit", "device_ids": ["bob"], "after": 0,
                "limit": 1, "request_id": "R1", "extra": 1})
        self.assertEqual(caught.exception.status_code, 400)
        self.assertEqual(caught.exception.field, "extra")

    def test_commit_without_request_id_unchanged(self) -> None:
        self._op("J1")
        self._seed_expired("c1")
        body, status = self._batch("commit", ["bob"])
        self.assertEqual(status, 201)
        self.assertEqual(body["results"][0]["removed"], 1)
        # No record was kept: the same call simply re-executes.
        body, status = self._batch("commit", ["bob"])
        self.assertEqual(status, 200)
        self.assertEqual(body["results"][0]["removed"], 0)
        self.assertEqual(
            self.service.store._event_gc_batch_cleanup_requests, {})

    def test_concurrent_same_id_executes_once(self) -> None:
        self._op("J1")
        for consumer in ("c1", "c2", "c3"):
            self._seed_expired(consumer)
        outcomes = []
        barrier = threading.Barrier(4)

        def run():
            barrier.wait()
            outcomes.append(self._batch(
                "commit", ["bob"], request_id="R1"))

        threads = [threading.Thread(target=run) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        self.assertEqual(len(outcomes), 4)
        first_body, first_status = outcomes[0]
        self.assertEqual(first_status, 201)
        self.assertEqual(first_body["results"][0]["removed"], 3)
        for body, status in outcomes:
            self.assertEqual(status, 201)
            self.assertEqual(body, first_body)
        self.assertEqual(
            len(self.service.store._event_gc_batch_cleanup_requests), 1)
        self.assertEqual(
            [key for key in
             self.service.store._redelivery_job_event_checkpoints
             if key[0] == "bob"], [])


class IdempotentBatchHTTPTest(IdempotentBatchMixin, unittest.TestCase):
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

    def _request(self, payload):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("POST", PATH, body=json.dumps(payload),
                     headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        data = response.read().decode("utf-8")
        conn.close()
        return response.status, json.loads(data), data

    def test_replay_is_byte_identical_over_http(self) -> None:
        self._op("J1")
        self._seed_expired("c1")
        payload = {"mode": "commit", "device_ids": ["bob", "ghost"],
                   "after": 0, "limit": 100, "request_id": "R1"}
        status, body, raw = self._request(payload)
        self.assertEqual(status, 201)
        self._seed_expired("c2")
        replay_status, replay_body, replay_raw = self._request(payload)
        self.assertEqual(replay_status, 201)
        self.assertEqual(replay_raw, raw)
        self.assertEqual(replay_body, body)

    def test_conflict_over_http(self) -> None:
        self._request({"mode": "commit", "device_ids": ["bob"], "after": 0,
                       "limit": 100, "request_id": "R1"})
        status, body, _ = self._request({
            "mode": "commit", "device_ids": ["bob", "alice"], "after": 0,
            "limit": 100, "request_id": "R1"})
        self.assertEqual(status, 409)
        self.assertEqual(list(body), ["message", "field"])
        self.assertEqual(body["field"], "request_id")

    def test_request_id_validation_over_http(self) -> None:
        cases = [
            ({"mode": "preview", "device_ids": ["bob"], "after": 0,
              "limit": 1, "request_id": "R1"}, 400, "request_id"),
            ({"mode": "commit", "device_ids": ["bob"], "after": 0,
              "limit": 1, "request_id": ""}, 400, "request_id"),
            ({"mode": "commit", "device_ids": ["bob"], "after": 0,
              "limit": 1, "request_id": None}, 400, "request_id"),
            ({"mode": "commit", "device_ids": ["bob"], "after": 0,
              "limit": 1, "request_id": "R1", "x": 0}, 400, "x"),
        ]
        for payload, expected_status, field in cases:
            with self.subTest(field=field):
                status, body, _ = self._request(payload)
                self.assertEqual(status, expected_status)
                self.assertEqual(body["field"], field)
                self.assertEqual(list(body), ["message", "field"])


class IdempotentBatchPersistenceTest(IdempotentBatchMixin,
                                     unittest.TestCase):
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

    def test_record_persisted_with_key_order_after_event_gc(self) -> None:
        self._op("J1")
        self._seed_expired("c1")
        body, status = self._batch(
            "commit", ["bob", "ghost"], after=0, limit=100, request_id="R1")
        self.assertEqual(status, 201)
        document = self._document()
        keys = list(document)
        self.assertEqual(keys.index("event_gc_batch_cleanup_requests"),
                         keys.index("event_gc") + 1)
        records = document["event_gc_batch_cleanup_requests"]
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(list(record), ["request_id", "device_ids", "after",
                                        "limit", "status", "response"])
        self.assertEqual(record["request_id"], "R1")
        self.assertEqual(record["device_ids"], ["bob", "ghost"])
        self.assertEqual(record["after"], 0)
        self.assertEqual(record["limit"], 100)
        self.assertEqual(record["status"], 201)
        self.assertEqual(record["response"], body)
        self.assertEqual(list(record["response"]),
                         ["mode", "results", "next_after", "has_more"])
        self.assertEqual(list(record["response"]["results"][0]),
                         ["device_id", "watermark", "expired", "removed",
                          "error"])
        self.assertTrue(self.service.persistence_integrity()["consistent"])

    def test_delete_nothing_commit_consumes_one_generation(self) -> None:
        generation = self.state_store.commit_seq
        body, status = self._batch("commit", ["dave", "ghost"],
                                   request_id="R1")
        self.assertEqual(status, 200)
        self.assertEqual(self.state_store.commit_seq, generation + 1)
        # The replay consumes nothing.
        replay, replay_status = self._batch("commit", ["dave", "ghost"],
                                            request_id="R1")
        self.assertEqual(replay_status, 200)
        self.assertEqual(replay, body)
        self.assertEqual(self.state_store.commit_seq, generation + 1)

    def test_replay_consumes_no_generation(self) -> None:
        self._op("J1")
        self._seed_expired("c1")
        body, status = self._batch("commit", ["bob"], request_id="R1")
        self.assertEqual(status, 201)
        generation = self.state_store.commit_seq
        for _ in range(3):
            replay, replay_status = self._batch("commit", ["bob"],
                                                request_id="R1")
            self.assertEqual(replay_status, 201)
            self.assertEqual(replay, body)
        self.assertEqual(self.state_store.commit_seq, generation)
        self.assertTrue(self.service.persistence_integrity()["consistent"])

    def test_replay_survives_restart(self) -> None:
        self._op("J1")
        self._seed_expired("c1")
        body, status = self._batch(
            "commit", ["bob", "dave"], request_id="R1")
        self.assertEqual(status, 201)
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        replay, replay_status = restarted.event_gc_cleanup_expired_batch({
            "mode": "commit", "device_ids": ["bob", "dave"], "after": 0,
            "limit": 100, "request_id": "R1"})
        self.assertEqual(replay_status, 201)
        self.assertEqual(replay, body)
        # A fresh id re-executes against the current (cleaned) state.
        fresh, fresh_status = restarted.event_gc_cleanup_expired_batch({
            "mode": "commit", "device_ids": ["bob", "dave"], "after": 0,
            "limit": 100, "request_id": "R2"})
        self.assertEqual(fresh_status, 200)
        self.assertEqual([i["removed"] for i in fresh["results"]], [0, 0])

    def test_legacy_document_without_section_loads(self) -> None:
        self._op("J1")
        self._touch("c1")
        document = self._document()
        self.assertIn("event_gc_batch_cleanup_requests", document)
        document.pop("event_gc_batch_cleanup_requests")
        document.pop("integrity_log_version", None)
        legacy_path = os.path.join(self.directory, "legacy.json")
        with open(legacy_path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        restarted = DeviceService()
        attach_persistence(restarted, legacy_path)
        body, status = restarted.event_gc_cleanup_expired_batch({
            "mode": "preview", "device_ids": ["bob"], "after": 0,
            "limit": 100})
        self.assertEqual(status, 200)
        self.assertEqual(
            restarted.store._event_gc_batch_cleanup_requests, {})

    def test_save_failure_rolls_record_back(self) -> None:
        import e2ee_backend.persistence as persistence_mod

        self._op("J1")
        self._seed_expired("c1")
        # One unrelated real mutation commits the in-memory expiry edit,
        # so last-good already carries the expired record (the anchor
        # lease lives on alice, which is not part of the batch).
        self._touch("anchor-lease", device="alice")
        real_fsync = persistence_mod.os.fsync
        calls = {"n": 0}

        def fail_first_directory_fsync(fd):  # noqa: ANN001
            if os.fstat(fd).st_mode & 0o170000 == 0o040000:
                calls["n"] += 1
                if calls["n"] == 1:
                    raise OSError("transient directory fsync failure")
            real_fsync(fd)

        generation = self.state_store.commit_seq
        persistence_mod.os.fsync = fail_first_directory_fsync
        try:
            with self.assertRaises(PersistenceUnavailable):
                self._batch("commit", ["bob"], request_id="R1")
        finally:
            persistence_mod.os.fsync = real_fsync
        # Nothing survived: no generation, no record, the lease is back.
        self.assertEqual(self.state_store.commit_seq, generation)
        self.assertEqual(
            self.service.store._event_gc_batch_cleanup_requests, {})
        self.assertIn(("bob", "c1"),
                      self.service.store._redelivery_job_event_checkpoints)
        self.assertEqual(
            self._document()["event_gc_batch_cleanup_requests"], [])
        # The failed id is not consumed: the retry commits once.
        body, status = self._batch("commit", ["bob"], request_id="R1")
        self.assertEqual(status, 201)
        self.assertEqual(body["results"][0]["removed"], 1)
        self.assertEqual(self.state_store.commit_seq, generation + 1)

    def test_malformed_section_refuses_startup(self) -> None:
        self._op("J1")
        self._seed_expired("c1")
        self._batch("commit", ["bob", "ghost"], after=0, limit=100,
                    request_id="R1")
        document = self._document()
        # Write each candidate marker-less/sidecar-less, so rejection
        # comes from the payload's own semantic validation rather than
        # the integrity hash gate.
        document.pop("integrity_log_version", None)
        record = document["event_gc_batch_cleanup_requests"][0]
        cases = []
        bad = json.loads(json.dumps(document))
        bad["event_gc_batch_cleanup_requests"] = {}
        cases.append(bad)
        bad = json.loads(json.dumps(document))
        bad["event_gc_batch_cleanup_requests"][0]["status"] = 200
        cases.append(bad)  # status contradicts the frozen 201 response
        bad = json.loads(json.dumps(document))
        item = bad["event_gc_batch_cleanup_requests"][0]
        item["status"] = item.pop("status")  # right keys, wrong order
        cases.append(bad)
        bad = json.loads(json.dumps(document))
        bad["event_gc_batch_cleanup_requests"].append(
            json.loads(json.dumps(record)))
        cases.append(bad)  # duplicate request_id
        bad = json.loads(json.dumps(document))
        bad["event_gc_batch_cleanup_requests"][0]["device_ids"] = \
            ["ghost", "bob"]
        cases.append(bad)  # device order contradicts the frozen results
        bad = json.loads(json.dumps(document))
        bad["event_gc_batch_cleanup_requests"][0]["after"] = 1
        cases.append(bad)  # paging contradicts the frozen results
        bad = json.loads(json.dumps(document))
        bad["event_gc_batch_cleanup_requests"][0]["response"]["next_after"] \
            = 1
        cases.append(bad)  # next_after contradicts the page
        bad = json.loads(json.dumps(document))
        bad["event_gc_batch_cleanup_requests"][0]["response"]["has_more"] = \
            True
        cases.append(bad)  # has_more contradicts the page
        bad = json.loads(json.dumps(document))
        bad["event_gc_batch_cleanup_requests"][0]["limit"] = 0
        cases.append(bad)
        bad = json.loads(json.dumps(document))
        bad["event_gc_batch_cleanup_requests"][0]["request_id"] = ""
        cases.append(bad)
        bad = json.loads(json.dumps(document))
        bad["event_gc_batch_cleanup_requests"][0]["response"]["mode"] = \
            "preview"
        cases.append(bad)
        for index, candidate in enumerate(cases):
            with self.subTest(case=index):
                candidate_path = os.path.join(
                    self.directory, f"bad{index}.json")
                with open(candidate_path, "w", encoding="utf-8") as handle:
                    json.dump(candidate, handle)
                with open(candidate_path, "rb") as handle:
                    before = handle.read()
                with self.assertRaises(StateFileError):
                    attach_persistence(DeviceService(), candidate_path)
                with open(candidate_path, "rb") as handle:
                    self.assertEqual(handle.read(), before)


if __name__ == "__main__":
    unittest.main()
