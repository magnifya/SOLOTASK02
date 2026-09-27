"""Tests for the batch expired-registration cleanup.

``POST /v1/event-gc-batch/cleanup-expired`` takes no query (400/query) and
a JSON object carrying exactly ``mode`` (``preview``/``commit``),
``device_ids`` (a non-empty array of unique non-empty strings), ``after``
(a non-negative, non-boolean integer) and ``limit`` (1..100, non-boolean).
A bad/non-object body is 400/request_body, a missing or illegal field is
400/that field, an extra key is 400/that key, and an illegal or repeated
element is 400/``device_ids[i]``.

The input list is paged from the zero-based offset ``after`` for at most
``limit`` entries and the whole page is handled under the store lock at
one instant (a revoked device stays cleanable). Each known device reports
non-negative integer ``watermark``/``expired``/``removed`` and
``error=null``; an unknown device reports the three values null and
``error`` ``{"status": 404, "field": "device_id"}``. ``preview`` counts
registrations whose ``expires`` is non-null and due, keeps ``removed`` 0
and writes nothing (always 200); ``commit`` deletes that set across the
page in one commit (201 when anything was deleted, 200 otherwise). The
body keys are ``mode``, ``results``, ``next_after`` and ``has_more``;
each item ``device_id``, ``watermark``, ``expired``, ``removed`` and
``error``. A data-file failure rolls the whole batch back with
503/data_file.
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
    attach_persistence,
)

PATH = "/v1/event-gc-batch/cleanup-expired"


class EventGcBatchMixin:
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

    def _observe(self, device="bob", after=0, limit=100):
        return self.service.event_gc_observe(device, after, limit)

    def _batch(self, mode, device_ids, after=0, limit=100):
        return self.service.event_gc_cleanup_expired_batch({
            "mode": mode, "device_ids": device_ids,
            "after": after, "limit": limit})


class EventGcBatchServiceTest(EventGcBatchMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    # -- response shape ---------------------------------------------------

    def test_preview_shape_and_key_order(self) -> None:
        self._op("J1")
        self._touch("c1")
        self._expire_lease("c1")
        body, status = self._batch("preview", ["bob", "ghost"])
        self.assertEqual(status, 200)
        self.assertEqual(list(body),
                         ["mode", "results", "next_after", "has_more"])
        self.assertEqual(body["mode"], "preview")
        self.assertEqual(body["next_after"], 2)
        self.assertFalse(body["has_more"])
        known, unknown = body["results"]
        self.assertEqual(list(known),
                         ["device_id", "watermark", "expired", "removed",
                          "error"])
        self.assertEqual(known, {"device_id": "bob", "watermark": 0,
                                 "expired": 1, "removed": 0, "error": None})
        self.assertEqual(list(unknown["error"]), ["status", "field"])
        self.assertEqual(unknown, {
            "device_id": "ghost", "watermark": None, "expired": None,
            "removed": None,
            "error": {"status": 404, "field": "device_id"}})

    def test_results_keep_input_order(self) -> None:
        body, _ = self._batch("preview",
                              ["ghost2", "bob", "ghost1", "alice"])
        self.assertEqual([item["device_id"] for item in body["results"]],
                         ["ghost2", "bob", "ghost1", "alice"])
        self.assertEqual([item["error"] for item in body["results"]],
                         [{"status": 404, "field": "device_id"}, None,
                          {"status": 404, "field": "device_id"}, None])

    def test_preview_counts_expired_across_consumers(self) -> None:
        self._op("J1")
        self._touch("c1")
        self._touch("c2")
        self._touch("c3")
        self._expire_lease("c1")
        self._expire_lease("c3")
        body, status = self._batch("preview", ["bob"])
        self.assertEqual(status, 200)
        self.assertEqual(body["results"][0]["expired"], 2)
        self.assertEqual(body["results"][0]["removed"], 0)

    def test_preview_writes_nothing(self) -> None:
        self._op("J1")
        self._touch("c1")
        self._expire_lease("c1")
        snapshot = self.service.store.snapshot_state()
        body, _ = self._batch("preview", ["bob", "ghost"])
        self.assertEqual(body["results"][0]["expired"], 1)
        self.assertEqual(self.service.store.snapshot_state(), snapshot)
        self.assertIn(("bob", "c1"),
                      self.service.store._redelivery_job_event_checkpoints)

    def test_commit_deletes_expired_and_reports_removed(self) -> None:
        self._op("J1")
        self._touch("c1")
        self._touch("c2")
        self._expire_lease("c1")
        body, status = self._batch(
            "commit", ["bob", "ghost"])
        self.assertEqual(status, 201)
        known, unknown = body["results"]
        self.assertEqual(known["removed"], 1)
        self.assertEqual(known["expired"], 1)
        self.assertEqual(unknown["removed"], None)
        remaining = {key[1] for key
                     in self.service.store._redelivery_job_event_checkpoints
                     if key[0] == "bob"}
        self.assertEqual(remaining, {"c2"})

    def test_commit_with_unknown_device_still_cleans_known_one(self) -> None:
        self._op("J1")
        self._touch("c1")
        self._expire_lease("c1")
        body, status = self._batch("commit", ["ghost", "bob"])
        self.assertEqual(status, 201)
        self.assertEqual(body["results"][0]["error"],
                         {"status": 404, "field": "device_id"})
        self.assertEqual(body["results"][1]["removed"], 1)

    def test_commit_with_nothing_expired_is_200_and_removes_nothing(self) \
            -> None:
        self._op("J1")
        self._touch("c1")
        body, status = self._batch("commit", ["bob", "ghost"])
        self.assertEqual(status, 200)
        self.assertEqual(body["results"][0], {
            "device_id": "bob", "watermark": 0, "expired": 0,
            "removed": 0, "error": None})
        self.assertIn(("bob", "c1"),
                      self.service.store._redelivery_job_event_checkpoints)

    def test_revoked_device_stays_cleanable(self) -> None:
        self._op("J1")
        self._touch("c1")
        self._expire_lease("c1")
        self.service.revoke_device("bob")
        body, status = self._batch("commit", ["bob"])
        self.assertEqual(status, 201)
        self.assertEqual(body["results"][0]["removed"], 1)

    # -- paging -----------------------------------------------------------

    def test_paging_by_after_and_limit(self) -> None:
        ids = ["a", "bob", "c", "dave", "e"]
        body, _ = self._batch("preview", ids, after=1, limit=2)
        self.assertEqual([item["device_id"] for item in body["results"]],
                         ["bob", "c"])
        self.assertEqual(body["next_after"], 3)
        self.assertTrue(body["has_more"])
        body, _ = self._batch("preview", ids, after=3, limit=2)
        self.assertEqual([item["device_id"] for item in body["results"]],
                         ["dave", "e"])
        self.assertEqual(body["next_after"], 5)
        self.assertFalse(body["has_more"])

    def test_empty_page(self) -> None:
        body, _ = self._batch("preview", ["bob"], after=5, limit=2)
        self.assertEqual(body["results"], [])
        self.assertEqual(body["next_after"], 5)
        self.assertFalse(body["has_more"])

    def test_paged_commit_cleans_each_page_once(self) -> None:
        self._op("J1")
        for consumer in ("c1", "c2", "c3"):
            self._touch(consumer)
            self._expire_lease(consumer)
        ids = ["ghost", "bob", "alice"]
        body, first = self._batch("commit", ids, after=0, limit=2)
        self.assertEqual(first, 201)
        self.assertEqual([i["device_id"] for i in body["results"]],
                         ["ghost", "bob"])
        self.assertEqual(body["results"][1]["removed"], 3)
        body, second = self._batch("commit", ids, after=2, limit=2)
        self.assertEqual(second, 200)
        self.assertEqual(body["results"][0]["removed"], 0)

    # -- validation -------------------------------------------------------

    def _assert_400(self, payload, field):
        with self.assertRaises(ServiceError) as caught:
            self.service.event_gc_cleanup_expired_batch(payload)
        self.assertEqual(caught.exception.status_code, 400)
        self.assertEqual(caught.exception.field, field)
        self.assertEqual(list(caught.exception.to_body()),
                         ["message", "field"])

    def test_bad_or_non_object_body(self) -> None:
        for payload in (None, "x", 42, ["preview"], False, True):
            with self.subTest(payload=payload):
                self._assert_400(payload, "request_body")

    def test_missing_fields(self) -> None:
        base = {"mode": "preview", "device_ids": ["bob"],
                "after": 0, "limit": 1}
        for name in ("mode", "device_ids", "after", "limit"):
            payload = dict(base)
            del payload[name]
            with self.subTest(name=name):
                self._assert_400(payload, name)

    def test_bad_mode(self) -> None:
        for mode in ("", "PREVIEW", "touch", 1, None, True, ["preview"]):
            with self.subTest(mode=mode):
                self._assert_400(
                    {"mode": mode, "device_ids": ["bob"],
                     "after": 0, "limit": 1}, "mode")

    def test_bad_device_ids(self) -> None:
        for value in (None, [], "bob", 1, True, {"bob": 1}):
            with self.subTest(value=value):
                self._assert_400(
                    {"mode": "preview", "device_ids": value,
                     "after": 0, "limit": 1}, "device_ids")

    def test_bad_or_duplicate_device_id_element(self) -> None:
        for value, index in (
                ([""], 0),
                (["a", ""], 1),
                (["a", 1], 1),
                (["a", None], 1),
                (["a", True], 1),
                (["a", "b", "a"], 2)):
            with self.subTest(value=value):
                self._assert_400(
                    {"mode": "preview", "device_ids": value,
                     "after": 0, "limit": 10}, f"device_ids[{index}]")

    def test_bad_after(self) -> None:
        for value in (-1, 0.0, 1.5, "0", None, True, False, [0]):
            with self.subTest(value=value):
                self._assert_400(
                    {"mode": "preview", "device_ids": ["bob"],
                     "after": value, "limit": 1}, "after")

    def test_bad_limit(self) -> None:
        for value in (0, 101, -1, 1.0, 100.0, "1", None, True, False):
            with self.subTest(value=value):
                self._assert_400(
                    {"mode": "preview", "device_ids": ["bob"],
                     "after": 0, "limit": value}, "limit")

    def test_extra_top_level_key(self) -> None:
        self._assert_400(
            {"mode": "preview", "device_ids": ["bob"], "after": 0,
             "limit": 1, "extra": 1}, "extra")
        # The first extra key in payload order is reported.
        self._assert_400(
            {"mode": "preview", "zzz": 1, "device_ids": ["bob"],
             "after": 0, "limit": 1}, "zzz")


class EventGcBatchHTTPTest(EventGcBatchMixin, unittest.TestCase):
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
        headers = {"Content-Type": "application/json"} if raw is not None \
            else {}
        conn.request(method, path, body=raw, headers=headers)
        response = conn.getresponse()
        data = response.read().decode("utf-8")
        conn.close()
        return response.status, (json.loads(data) if data else None), data

    def _batch_http(self, mode, device_ids, after=0, limit=100):
        return self._request(raw=json.dumps({
            "mode": mode, "device_ids": device_ids,
            "after": after, "limit": limit}))

    def _seed_expired(self, device="bob"):
        if device == "bob":
            self._op("J1")
        self._touch("c1", device=device)
        self._expire_lease("c1", device=device)

    def test_query_rejected_but_bare_question_mark_allowed(self) -> None:
        for query in ("?x=1", "?after=0", "?mode=preview", "?foo"):
            with self.subTest(query=query):
                status, body, _ = self._request(path=PATH + query, raw="{}")
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], "query")
        status, _, _ = self._request(path=PATH + "?", raw="{}")
        self.assertEqual(status, 400)  # {} fails on missing mode, not query

    def test_bad_json_and_non_object_body(self) -> None:
        for raw in ("junk", "[]", "null", "42", '"preview"', ""):
            with self.subTest(raw=raw):
                status, body, _ = self._request(raw=raw)
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], "request_body")

    def test_preview_over_http_with_key_order(self) -> None:
        self._seed_expired()
        status, body, raw = self._batch_http(
            "preview", ["bob", "ghost"])
        self.assertEqual(status, 200)
        self.assertEqual(list(body),
                         ["mode", "results", "next_after", "has_more"])
        self.assertEqual(list(body["results"][0]),
                         ["device_id", "watermark", "expired", "removed",
                          "error"])
        self.assertEqual(body["results"][1]["error"],
                         {"status": 404, "field": "device_id"})
        for earlier, later in (
                ('"mode"', '"results"'),
                ('"results"', '"next_after"'),
                ('"next_after"', '"has_more"'),
                ('"device_id"', '"watermark"'),
                ('"watermark"', '"expired"'),
                ('"expired"', '"removed"'),
                ('"removed"', '"error"'),
                ('"status"', '"field"')):
            self.assertLess(raw.index(earlier), raw.index(later))

    def test_commit_over_http(self) -> None:
        self._seed_expired()
        status, body, _ = self._batch_http("commit", ["bob"])
        self.assertEqual(status, 201)
        self.assertEqual(body["results"][0]["removed"], 1)
        status, body, _ = self._batch_http("commit", ["bob"])
        self.assertEqual(status, 200)
        self.assertEqual(body["results"][0]["removed"], 0)

    def test_validation_field_names_over_http(self) -> None:
        cases = [
            ({"mode": "nope", "device_ids": ["bob"], "after": 0,
              "limit": 1}, "mode"),
            ({"mode": "preview", "device_ids": [], "after": 0,
              "limit": 1}, "device_ids"),
            ({"mode": "preview", "device_ids": ["a", "a"], "after": 0,
              "limit": 1}, "device_ids[1]"),
            ({"mode": "preview", "device_ids": ["a"], "after": -1,
              "limit": 1}, "after"),
            ({"mode": "preview", "device_ids": ["a"], "after": 0,
              "limit": 0}, "limit"),
            ({"mode": "preview", "device_ids": ["a"], "after": 0,
              "limit": 1, "x": 0}, "x"),
        ]
        for payload, field in cases:
            with self.subTest(field=field):
                status, body, _ = self._request(raw=json.dumps(payload))
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], field)
                self.assertEqual(list(body), ["message", "field"])

    def test_get_is_404(self) -> None:
        status, body, _ = self._request(method="GET")
        self.assertEqual(status, 404)
        self.assertEqual(body["message"], f"not found: {PATH}")


class EventGcBatchPersistenceTest(EventGcBatchMixin, unittest.TestCase):
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

    def _seed_expired(self, device, consumer):
        self._touch(consumer, device=device)
        self._expire_lease(consumer, device=device)

    def _anchor_durable(self):
        # One unrelated real mutation commits the in-memory expiry edits,
        # so last-good already carries the expired records (the anchor
        # lease lives on alice, which is never part of these batches).
        self._touch("anchor-lease", device="alice")

    def test_preview_consumes_no_generation(self) -> None:
        self._seed_expired("bob", "c1")
        self._anchor_durable()
        generation = self.state_store.commit_seq
        for _ in range(3):
            body, status = self._batch(
                "preview", ["bob", "ghost", "dave"])
        self.assertEqual(status, 200)
        self.assertEqual(body["results"][0]["expired"], 1)
        self.assertEqual(self.state_store.commit_seq, generation)
        self.assertTrue(self.service.persistence_integrity()["consistent"])

    def test_noop_commit_consumes_no_generation(self) -> None:
        self._touch("c1", device="bob")
        generation = self.state_store.commit_seq
        body, status = self._batch("commit", ["dave", "ghost"])
        self.assertEqual(status, 200)
        self.assertEqual(body["results"][0]["removed"], 0)
        self.assertEqual(self.state_store.commit_seq, generation)

    def test_whole_batch_commits_one_generation(self) -> None:
        self._op("J1", device="bob")
        self._seed_expired("bob", "c1")
        self._seed_expired("bob", "c2")
        self._seed_expired("dave", "d1")
        self._anchor_durable()
        before = self._document()
        generation = self.state_store.commit_seq
        body, status = self._batch(
            "commit", ["bob", "ghost", "dave"], after=0, limit=100)
        self.assertEqual(status, 201)
        self.assertEqual([item["removed"] for item in body["results"]],
                         [2, None, 1])
        self.assertEqual(self.state_store.commit_seq, generation + 1)
        after = self._document()
        # Only the anchor lease (on alice, outside the batch) survives.
        self.assertEqual(
            sorted((item["device_id"], item["consumer_id"])
                   for item in after["redelivery_job_event_checkpoints"]),
            [("alice", "anchor-lease")])
        # The event chain and watermark sections are untouched.
        self.assertEqual(after["redelivery_job_events"],
                         before["redelivery_job_events"])
        self.assertEqual(after["event_gc"], before["event_gc"])
        self.assertTrue(self.service.persistence_integrity()["consistent"])

    def test_batch_result_survives_restart(self) -> None:
        self._seed_expired("bob", "c1")
        self._seed_expired("dave", "d1")
        self._anchor_durable()
        body, status = self._batch("commit", ["bob", "dave"])
        self.assertEqual(status, 201)
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        body, status = restarted.event_gc_cleanup_expired_batch({
            "mode": "commit", "device_ids": ["bob", "dave"],
            "after": 0, "limit": 100})
        self.assertEqual(status, 200)
        self.assertEqual([i["removed"] for i in body["results"]], [0, 0])

    def test_save_failure_rolls_whole_batch_back(self) -> None:
        import e2ee_backend.persistence as persistence_mod

        self._seed_expired("bob", "c1")
        self._seed_expired("dave", "d1")
        self._anchor_durable()
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
                self._batch("commit", ["bob", "dave"])
        finally:
            persistence_mod.os.fsync = real_fsync
        # The whole batch rolled back: no generation consumed and every
        # expired record is back in memory and on disk.
        self.assertEqual(self.state_store.commit_seq, generation)
        checkpoints = self.service.store._redelivery_job_event_checkpoints
        self.assertIn(("bob", "c1"), checkpoints)
        self.assertIn(("dave", "d1"), checkpoints)
        document = self._document()
        self.assertEqual(
            sorted((item["device_id"], item["consumer_id"])
                   for item in document["redelivery_job_event_checkpoints"]),
            [("alice", "anchor-lease"), ("bob", "c1"), ("dave", "d1")])
        # The batch retries and now commits once.
        body, status = self._batch("commit", ["bob", "dave"])
        self.assertEqual(status, 201)
        self.assertEqual([i["removed"] for i in body["results"]], [1, 1])
        self.assertEqual(self.state_store.commit_seq, generation + 1)


if __name__ == "__main__":
    unittest.main()
