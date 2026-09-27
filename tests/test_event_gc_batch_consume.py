"""Tests for the batch-cleanup audit atomic consume endpoint.

POST /v1/event-gc-batch/consume atomically pulls one page of the
batch-cleanup audit chain (the committed
``event_gc_batch_cleanup_requests`` records, in commit order) and
advances the consumer's checkpoint past the page. The call takes no
query parameters (any -> 400/query); the body carries exactly
``consumer_id`` (a non-empty string), ``expected`` (a non-boolean
integer in 0..2^63-1) and ``limit`` (a non-boolean integer in 1..100);
a bad/non-object body is 400/request_body and a missing/wrongly
typed/extra field is 400 with that field. The checkpoint starts at 0;
an ``expected`` differing from the current value is 409/expected. On a
match the records after the checkpoint are read in commit order, at
most ``limit`` of them, and the checkpoint advances to the offset just
past the last record: a non-empty page answers 201 with a fresh
``updated_at``; an empty page answers 200 with the offset and the
timestamp untouched and writes nothing. Success keys are
``consumer_id``, ``records``, ``next_after``, ``has_more`` and
``updated_at`` in that order; each record carries the audit chain's six
keys with the nested key order untouched. Concurrent consumes with the
same ``expected`` linearize to at most one 201; a data-file failure is
503/data_file and rolls the checkpoint, the commit generation and both
files back.
"""
import json
import os
import shutil
import tempfile
import threading
import unittest
from http.client import HTTPConnection

from e2ee_backend.models import Device
from e2ee_backend.service import DeviceService, ServiceError
from e2ee_backend.http_app import create_server
from e2ee_backend.persistence import (
    PersistenceUnavailable,
    attach_persistence,
)

PATH = "/v1/event-gc-batch/consume"


class ConsumeMixin:
    def _build(self) -> None:
        self.service = DeviceService()
        self.service.store.add_device(Device("u", "alice", "ik"))
        self.service.store.add_device(Device("u", "bob", "ik"))

    def _commit(self, request_id, device_ids=None):
        # Every idempotent commit appends exactly one audit record.
        return self.service.event_gc_cleanup_expired_batch({
            "mode": "commit", "device_ids": device_ids or ["bob"],
            "after": 0, "limit": 100, "request_id": request_id})

    def _consume(self, consumer="c1", expected=0, limit=100):
        return self.service.event_gc_batch_consume({
            "consumer_id": consumer, "expected": expected, "limit": limit})


class ConsumeServiceTest(ConsumeMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def test_empty_chain_is_200_and_writes_nothing(self) -> None:
        body, status = self._consume()
        self.assertEqual(status, 200)
        self.assertEqual(list(body), ["consumer_id", "records",
                                      "next_after", "has_more",
                                      "updated_at"])
        self.assertEqual(body, {"consumer_id": "c1", "records": [],
                                "next_after": 0, "has_more": False,
                                "updated_at": None})
        # The empty page created no checkpoint record.
        page = self.service.event_gc_batch_checkpoints(0, 100)
        self.assertEqual(page["checkpoints"], [])

    def test_nonempty_page_is_201_with_six_key_records(self) -> None:
        committed, _ = self._commit("r1")
        body, status = self._consume()
        self.assertEqual(status, 201)
        self.assertEqual(list(body), ["consumer_id", "records",
                                      "next_after", "has_more",
                                      "updated_at"])
        self.assertEqual(body["consumer_id"], "c1")
        self.assertEqual(body["next_after"], 1)
        self.assertFalse(body["has_more"])
        self.assertTrue(body["updated_at"].endswith("+00:00"))
        self.assertEqual(len(body["records"]), 1)
        record = body["records"][0]
        self.assertEqual(list(record), ["request_id", "device_ids",
                                        "after", "limit", "status",
                                        "response"])
        self.assertEqual(record["request_id"], "r1")
        self.assertEqual(record["device_ids"], ["bob"])
        self.assertEqual(record["status"], 200)
        self.assertEqual(list(record["response"]),
                         ["mode", "results", "next_after", "has_more"])
        self.assertEqual(record["response"], committed)

    def test_paging_and_has_more(self) -> None:
        for request_id in ("r1", "r2", "r3"):
            self._commit(request_id)
        body, status = self._consume(limit=2)
        self.assertEqual(status, 201)
        self.assertEqual([r["request_id"] for r in body["records"]],
                         ["r1", "r2"])
        self.assertEqual(body["next_after"], 2)
        self.assertTrue(body["has_more"])
        first_stamp = body["updated_at"]
        # The next page continues from the advanced checkpoint.
        body, status = self._consume(expected=2, limit=2)
        self.assertEqual(status, 201)
        self.assertEqual([r["request_id"] for r in body["records"]],
                         ["r3"])
        self.assertEqual(body["next_after"], 3)
        self.assertFalse(body["has_more"])
        last_stamp = body["updated_at"]
        self.assertGreaterEqual(last_stamp, first_stamp)
        # The chain is drained: an empty page keeps offset and time.
        body, status = self._consume(expected=3, limit=2)
        self.assertEqual(status, 200)
        self.assertEqual(body["records"], [])
        self.assertEqual(body["next_after"], 3)
        self.assertFalse(body["has_more"])
        self.assertEqual(body["updated_at"], last_stamp)

    def test_empty_page_keeps_stored_timestamp(self) -> None:
        self._commit("r1")
        first, _ = self._consume()
        again, status = self._consume(expected=1)
        self.assertEqual(status, 200)
        self.assertEqual(again["updated_at"], first["updated_at"])
        self.assertEqual(again["next_after"], 1)

    def test_expected_mismatch_is_409(self) -> None:
        self._commit("r1")
        with self.assertRaises(ServiceError) as caught:
            self._consume(expected=1)
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "expected")
        self._consume(expected=0)
        with self.assertRaises(ServiceError) as caught:
            self._consume(expected=0)
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "expected")

    def test_consumers_are_independent(self) -> None:
        self._commit("r1")
        body, status = self._consume(consumer="c1")
        self.assertEqual(status, 201)
        body, status = self._consume(consumer="c2", expected=0)
        self.assertEqual(status, 201)
        self.assertEqual(body["next_after"], 1)
        page = self.service.event_gc_batch_checkpoints(0, 100)
        self.assertEqual([i["consumer_id"] for i in page["checkpoints"]],
                         ["c1", "c2"])

    def test_payload_validation(self) -> None:
        for payload, field in (
                (None, "request_body"),
                ([], "request_body"),
                ("x", "request_body"),
                ({"expected": 0, "limit": 1}, "consumer_id"),
                ({"consumer_id": "", "expected": 0, "limit": 1},
                 "consumer_id"),
                ({"consumer_id": 1, "expected": 0, "limit": 1},
                 "consumer_id"),
                ({"consumer_id": None, "expected": 0, "limit": 1},
                 "consumer_id"),
                ({"consumer_id": "c1", "limit": 1}, "expected"),
                ({"consumer_id": "c1", "expected": 0}, "limit"),
                ({"consumer_id": "c1", "expected": True, "limit": 1},
                 "expected"),
                ({"consumer_id": "c1", "expected": -1, "limit": 1},
                 "expected"),
                ({"consumer_id": "c1", "expected": 1.0, "limit": 1},
                 "expected"),
                ({"consumer_id": "c1", "expected": "0", "limit": 1},
                 "expected"),
                ({"consumer_id": "c1", "expected": None, "limit": 1},
                 "expected"),
                ({"consumer_id": "c1", "expected": 2**63, "limit": 1},
                 "expected"),
                ({"consumer_id": "c1", "expected": 0, "limit": True},
                 "limit"),
                ({"consumer_id": "c1", "expected": 0, "limit": 0},
                 "limit"),
                ({"consumer_id": "c1", "expected": 0, "limit": 101},
                 "limit"),
                ({"consumer_id": "c1", "expected": 0, "limit": -1},
                 "limit"),
                ({"consumer_id": "c1", "expected": 0, "limit": 1.0},
                 "limit"),
                ({"consumer_id": "c1", "expected": 0, "limit": "1"},
                 "limit"),
                ({"consumer_id": "c1", "expected": 0, "limit": None},
                 "limit"),
                ({"consumer_id": "c1", "expected": 0, "limit": 1,
                  "x": 1}, "x"),
                ({"consumer_id": "c1", "expected": 0, "limit": 1,
                  "y": 1, "z": 2}, "y")):
            with self.subTest(payload=payload):
                with self.assertRaises(ServiceError) as caught:
                    self.service.event_gc_batch_consume(payload)
                self.assertEqual(caught.exception.status_code, 400)
                self.assertEqual(caught.exception.field, field)

    def test_boundary_values_pass_validation(self) -> None:
        self._commit("r1")
        # 2^63-1 is a valid expected shape: it fails only the value check.
        with self.assertRaises(ServiceError) as caught:
            self._consume(expected=2**63 - 1)
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "expected")
        # limit 100 and limit 1 are both accepted.
        body, status = self._consume(limit=100)
        self.assertEqual(status, 201)
        body, status = self._consume(expected=1, limit=1)
        self.assertEqual(status, 200)


class ConsumeConcurrencyTest(ConsumeMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()
        self._commit("r1")

    def test_concurrent_same_expected_at_most_one_201(self) -> None:
        results = []
        barrier = threading.Barrier(8)

        def worker() -> None:
            barrier.wait()
            try:
                _body, status = self._consume(expected=0, limit=1)
            except ServiceError as error:
                status = error.status_code
            results.append(status)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
        self.assertEqual(results.count(201), 1)
        self.assertEqual(results.count(409), 7)
        page = self.service.event_gc_batch_checkpoints(0, 100)
        self.assertEqual(page["checkpoints"][0]["after"], 1)


class ConsumeHTTPTest(ConsumeMixin, unittest.TestCase):
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

    def _request(self, path=PATH, raw=None, method="POST"):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        headers = {"Content-Type": "application/json"} if raw else {}
        conn.request(method, path, body=raw, headers=headers)
        response = conn.getresponse()
        data = response.read().decode("utf-8")
        conn.close()
        return response.status, (json.loads(data) if data else None), data

    def _consume_http(self, consumer="c1", expected=0, limit=100,
                      path=PATH):
        return self._request(
            path, json.dumps({"consumer_id": consumer,
                              "expected": expected, "limit": limit}))

    def test_consume_over_http_with_key_order(self) -> None:
        self._commit("r1")
        status, body, raw = self._consume_http()
        self.assertEqual(status, 201)
        self.assertEqual(list(body), ["consumer_id", "records",
                                      "next_after", "has_more",
                                      "updated_at"])
        for earlier, later in (
                ('"consumer_id"', '"records"'),
                ('"records"', '"next_after"'),
                ('"next_after"', '"has_more"'),
                ('"has_more"', '"updated_at"'),
                ('"request_id"', '"device_ids"'),
                ('"device_ids"', '"after"'),
                ('"after"', '"limit"'),
                ('"limit"', '"status"'),
                ('"status"', '"response"')):
            self.assertLess(raw.index(earlier), raw.index(later))
        # The drained chain answers an empty 200 page.
        status, body, _ = self._consume_http(expected=1)
        self.assertEqual(status, 200)
        self.assertEqual(body["records"], [])
        self.assertEqual(body["next_after"], 1)

    def test_query_rejected(self) -> None:
        for query in ("?after=0", "?foo", "?x=1"):
            with self.subTest(query=query):
                status, body, _ = self._consume_http(path=PATH + query)
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], "query")
                self.assertEqual(list(body), ["message", "field"])
        # A bare trailing '?' carries no parameter and is accepted.
        status, _, _ = self._consume_http(path=PATH + "?")
        self.assertEqual(status, 200)

    def test_bad_json_and_non_object_are_400_request_body(self) -> None:
        for raw in ("{", "[1]", "null", '"s"'):
            with self.subTest(raw=raw):
                status, body, _ = self._request(raw=raw)
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], "request_body")
                self.assertEqual(list(body), ["message", "field"])

    def test_field_errors_over_http(self) -> None:
        for payload, field in (
                ({"expected": 0, "limit": 1}, "consumer_id"),
                ({"consumer_id": "", "expected": 0, "limit": 1},
                 "consumer_id"),
                ({"consumer_id": "c1", "limit": 1}, "expected"),
                ({"consumer_id": "c1", "expected": 0}, "limit"),
                ({"consumer_id": "c1", "expected": True, "limit": 1},
                 "expected"),
                ({"consumer_id": "c1", "expected": 0, "limit": 0},
                 "limit"),
                ({"consumer_id": "c1", "expected": 0, "limit": 101},
                 "limit"),
                ({"consumer_id": "c1", "expected": 0, "limit": 1,
                  "extra": 1}, "extra")):
            with self.subTest(payload=payload):
                status, body, _ = self._request(raw=json.dumps(payload))
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], field)
                self.assertEqual(list(body), ["message", "field"])

    def test_expected_conflict_over_http(self) -> None:
        self._commit("r1")
        status, body, _ = self._consume_http(expected=1)
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "expected")
        self.assertEqual(list(body), ["message", "field"])

    def test_get_is_404(self) -> None:
        status, _, _ = self._request(method="GET")
        self.assertEqual(status, 404)


class ConsumePersistenceTest(ConsumeMixin, unittest.TestCase):
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

    def test_nonempty_page_consumes_a_generation_empty_does_not(self) -> None:
        self._commit("r1")
        generation = self.state_store.commit_seq
        _, status = self._consume()
        self.assertEqual(status, 201)
        self.assertEqual(self.state_store.commit_seq, generation + 1)
        _, status = self._consume(expected=1)
        self.assertEqual(status, 200)
        self.assertEqual(self.state_store.commit_seq, generation + 1)

    def test_restart_restores_checkpoint_and_consume_continues(self) -> None:
        self._commit("r1")
        self._commit("r2")
        _, status = self._consume(limit=1)
        self.assertEqual(status, 201)
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        # The restored checkpoint bounds the next consume.
        with self.assertRaises(ServiceError) as caught:
            restarted.event_gc_batch_consume(
                {"consumer_id": "c1", "expected": 0, "limit": 1})
        self.assertEqual(caught.exception.status_code, 409)
        body, status = restarted.event_gc_batch_consume(
            {"consumer_id": "c1", "expected": 1, "limit": 1})
        self.assertEqual(status, 201)
        self.assertEqual([r["request_id"] for r in body["records"]], ["r2"])
        self.assertEqual(body["next_after"], 2)
        self.assertTrue(restarted.persistence_integrity()["consistent"])

    def test_save_failure_rolls_checkpoint_back(self) -> None:
        import e2ee_backend.persistence as persistence_mod

        self._commit("r1")
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
                self._consume(expected=0, limit=1)
        finally:
            persistence_mod.os.fsync = real_fsync
        # Rolled back: no generation consumed and no checkpoint recorded,
        # in memory or on disk.
        self.assertEqual(self.state_store.commit_seq, generation)
        self.assertEqual(self._document()["cleanup_checkpoints"], [])
        page = self.service.event_gc_batch_checkpoints(0, 100)
        self.assertEqual(page["checkpoints"], [])
        # The consume can be retried and now commits.
        body, status = self._consume(expected=0, limit=1)
        self.assertEqual(status, 201)
        self.assertEqual(body["next_after"], 1)
        self.assertEqual(self.state_store.commit_seq, generation + 1)


if __name__ == "__main__":
    unittest.main()
