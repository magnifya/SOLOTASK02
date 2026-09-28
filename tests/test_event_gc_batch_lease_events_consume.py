"""Tests for the cleanup audit lease event subscribe-and-pull endpoint.

``POST /v1/event-gc-batch/lease-events/consume`` takes no query
parameters (any -> 400/query) and a JSON body carrying exactly
``subscriber_id`` (a non-empty string), ``expected`` (a non-boolean
integer in 0..2**63-1) and ``limit`` (a non-boolean integer in
1..100); a bad/non-object body is 400/request_body and a
missing/wrongly typed/extra field is 400 with that field. A brand new
subscriber must start at ``expected == 0`` (otherwise 409/expected);
an existing subscriber must name its stored cursor exactly. On a match
the store takes the global cleanup-lease lifecycle event chain past the
cursor, at most ``limit`` events, and advances the cursor: a non-empty
page answers 201 whether the subscriber was just created or advanced,
an existing subscriber's empty page answers 200 and writes nothing (a
new subscriber is never created on an empty page). Success keys are
``subscriber_id``, ``events``, ``next_after`` and ``has_more`` in that
order; each event keeps ``seq``/``lease_id``/``consumer_id``/``type``
order; an empty page echoes ``next_after=expected`` and ``has_more``
says whether a later event exists. Concurrent pulls with the same
``expected`` linearize to at most one 201; a data-file failure is
503/data_file with cursor, commit generation and both files rolled
back. Cursors persist in the new ``lease_event_cursors`` section right
after ``cleanup_lease_events``; duplicate ids, type/key-order errors
and cursors past the stream refuse startup.
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
    StateFileError,
    attach_persistence,
)

PATH = "/v1/event-gc-batch/lease-events/consume"


class LeaseEventsConsumeMixin:
    def _build(self) -> None:
        self.service = DeviceService()
        self.service.store.add_device(Device("u", "bob", "ik"))

    def _commit(self, request_id):
        return self.service.event_gc_cleanup_expired_batch({
            "mode": "commit", "device_ids": ["bob"],
            "after": 0, "limit": 100, "request_id": request_id})

    def _claim(self, consumer, lease_id, expected=0, limit=1):
        return self.service.event_gc_batch_claim({
            "consumer_id": consumer, "lease_id": lease_id,
            "expected": expected, "limit": limit})

    def _renew(self, consumer, lease_id, renewal_id):
        return self.service.event_gc_batch_lease_renew({
            "consumer_id": consumer, "lease_id": lease_id,
            "renewal_id": renewal_id})

    def _release(self, consumer, lease_id):
        return self.service.event_gc_batch_lease_op({
            "consumer_id": consumer, "lease_id": lease_id,
            "expected": 0, "op": "release"})

    def _seed(self):
        # claim L1 (seq 1), claim L2 (seq 2), renew L1 (seq 3),
        # release L1 (seq 4).
        self._commit("r1")
        self._commit("r2")
        self._claim("c1", "L1", limit=1)
        self._claim("c2", "L2", limit=100)
        self._renew("c1", "L1", "rn1")
        self._release("c1", "L1")

    def _consume(self, subscriber="s1", expected=0, limit=100):
        return self.service.event_gc_batch_lease_events_consume({
            "subscriber_id": subscriber, "expected": expected,
            "limit": limit})


class LeaseEventsConsumeServiceTest(LeaseEventsConsumeMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def test_empty_stream_new_subscriber_empty_page_200_no_cursor(
            self) -> None:
        body, status = self._consume(expected=0)
        self.assertEqual(status, 200)
        self.assertEqual(body, {"subscriber_id": "s1", "events": [],
                                "next_after": 0, "has_more": False})
        # No cursor was created: a non-zero expected still conflicts.
        with self.assertRaises(ServiceError) as caught:
            self._consume(expected=1)
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "expected")
        # Events arriving later are still consumable from the head.
        self._seed()
        body, status = self._consume(expected=0, limit=1)
        self.assertEqual(status, 201)
        self.assertEqual(body["events"][0]["seq"], 1)

    def test_nonempty_page_creates_subscriber(self) -> None:
        self._seed()
        body, status = self._consume(expected=0, limit=2)
        self.assertEqual(status, 201)
        self.assertEqual(list(body),
                         ["subscriber_id", "events", "next_after",
                          "has_more"])
        self.assertEqual([event["seq"] for event in body["events"]], [1, 2])
        self.assertEqual(body["next_after"], 2)
        self.assertTrue(body["has_more"])
        for event in body["events"]:
            self.assertEqual(
                list(event), ["seq", "lease_id", "consumer_id", "type"])

    def test_paging_advances_then_drains(self) -> None:
        self._seed()
        first, status = self._consume(expected=0, limit=2)
        self.assertEqual(status, 201)
        self.assertEqual(
            [(e["lease_id"], e["type"]) for e in first["events"]],
            [("L1", "claim"), ("L2", "claim")])
        self.assertEqual(first["next_after"], 2)
        self.assertTrue(first["has_more"])
        second, status = self._consume(expected=2, limit=2)
        self.assertEqual(status, 201)
        self.assertEqual(
            [(e["lease_id"], e["type"]) for e in second["events"]],
            [("L1", "renew"), ("L1", "release")])
        self.assertEqual(second["next_after"], 4)
        self.assertFalse(second["has_more"])
        # Drained: an empty page for the existing subscriber is 200 and
        # echoes expected without advancing anything.
        drained, status = self._consume(expected=4, limit=2)
        self.assertEqual(status, 200)
        self.assertEqual(drained, {"subscriber_id": "s1", "events": [],
                                   "next_after": 4, "has_more": False})

    def test_new_events_after_drain_are_consumable(self) -> None:
        self._commit("r1")
        self._claim("c1", "L1", limit=100)
        body, status = self._consume(expected=0)
        self.assertEqual(status, 201)
        self.assertEqual(body["next_after"], 1)
        drained, status = self._consume(expected=1)
        self.assertEqual(status, 200)
        self.assertEqual(drained["events"], [])
        # A fresh lease appends a new event past the cursor.
        self._claim("c2", "L2", limit=100)
        body, status = self._consume(expected=1)
        self.assertEqual(status, 201)
        self.assertEqual([(e["seq"], e["lease_id"]) for e in body["events"]],
                         [(2, "L2")])
        self.assertEqual(body["next_after"], 2)

    def test_new_subscriber_must_start_at_zero(self) -> None:
        self._seed()
        with self.assertRaises(ServiceError) as caught:
            self._consume(expected=1)
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "expected")
        # The rejected create left no cursor behind: starting at 0 works.
        body, status = self._consume(expected=0, limit=100)
        self.assertEqual(status, 201)
        self.assertEqual(body["next_after"], 4)

    def test_existing_subscriber_expected_mismatch(self) -> None:
        self._seed()
        self._consume(expected=0, limit=2)
        for wrong in (0, 1, 3):
            with self.subTest(expected=wrong):
                with self.assertRaises(ServiceError) as caught:
                    self._consume(expected=wrong)
                self.assertEqual(caught.exception.status_code, 409)
                self.assertEqual(caught.exception.field, "expected")
        # The cursor is untouched: the correct expected still advances.
        body, status = self._consume(expected=2)
        self.assertEqual(status, 201)
        self.assertEqual(body["next_after"], 4)

    def test_subscribers_are_independent(self) -> None:
        self._seed()
        first, status = self._consume(subscriber="s1", expected=0, limit=2)
        self.assertEqual(status, 201)
        self.assertEqual(first["next_after"], 2)
        second, status = self._consume(subscriber="s2", expected=0, limit=100)
        self.assertEqual(status, 201)
        self.assertEqual(second["next_after"], 4)
        # s1 continues from its own cursor.
        body, status = self._consume(subscriber="s1", expected=2)
        self.assertEqual(status, 201)
        self.assertEqual(body["next_after"], 4)
        # s2's repeat at 4 is an existing-subscriber empty page (200),
        # not a new-subscriber conflict.
        body, status = self._consume(subscriber="s2", expected=4)
        self.assertEqual(status, 200)
        self.assertEqual(body["next_after"], 4)

    def test_payload_validation(self) -> None:
        for payload, field in (
                (None, "request_body"),
                ([], "request_body"),
                ("x", "request_body"),
                (42, "request_body"),
                ({"expected": 0, "limit": 1}, "subscriber_id"),
                ({"subscriber_id": "", "expected": 0, "limit": 1},
                 "subscriber_id"),
                ({"subscriber_id": 1, "expected": 0, "limit": 1},
                 "subscriber_id"),
                ({"subscriber_id": True, "expected": 0, "limit": 1},
                 "subscriber_id"),
                ({"subscriber_id": None, "expected": 0, "limit": 1},
                 "subscriber_id"),
                ({"subscriber_id": "s1", "limit": 1}, "expected"),
                ({"subscriber_id": "s1", "expected": 0}, "limit"),
                ({"subscriber_id": "s1", "expected": None, "limit": 1},
                 "expected"),
                ({"subscriber_id": "s1", "expected": True, "limit": 1},
                 "expected"),
                ({"subscriber_id": "s1", "expected": False, "limit": 1},
                 "expected"),
                ({"subscriber_id": "s1", "expected": -1, "limit": 1},
                 "expected"),
                ({"subscriber_id": "s1", "expected": 1.0, "limit": 1},
                 "expected"),
                ({"subscriber_id": "s1", "expected": "0", "limit": 1},
                 "expected"),
                ({"subscriber_id": "s1", "expected": 2**63, "limit": 1},
                 "expected"),
                ({"subscriber_id": "s1", "expected": 0, "limit": None},
                 "limit"),
                ({"subscriber_id": "s1", "expected": 0, "limit": True},
                 "limit"),
                ({"subscriber_id": "s1", "expected": 0, "limit": 0},
                 "limit"),
                ({"subscriber_id": "s1", "expected": 0, "limit": 101},
                 "limit"),
                ({"subscriber_id": "s1", "expected": 0, "limit": -1},
                 "limit"),
                ({"subscriber_id": "s1", "expected": 0, "limit": 1.0},
                 "limit"),
                ({"subscriber_id": "s1", "expected": 0, "limit": "1"},
                 "limit"),
                ({"subscriber_id": "s1", "expected": 0, "limit": 1,
                  "x": 1}, "x"),
                ({"subscriber_id": "s1", "expected": 0, "limit": 1,
                  "y": 1, "z": 2}, "y")):
            with self.subTest(payload=payload):
                with self.assertRaises(ServiceError) as caught:
                    self.service.event_gc_batch_lease_events_consume(payload)
                self.assertEqual(caught.exception.status_code, 400)
                self.assertEqual(caught.exception.field, field)

    def test_boundary_integers_pass_shape_validation(self) -> None:
        self._seed()
        # 2^63-1 is a valid shape: it fails only the expected comparison.
        with self.assertRaises(ServiceError) as caught:
            self._consume(expected=2**63 - 1)
        self.assertEqual(caught.exception.status_code, 409)
        # limit 1 and 100 are both accepted shapes.
        _, status = self._consume(expected=0, limit=1)
        self.assertEqual(status, 201)
        _, status = self._consume(expected=1, limit=100)
        self.assertEqual(status, 201)


class LeaseEventsConsumeConcurrencyTest(LeaseEventsConsumeMixin,
                                       unittest.TestCase):
    def setUp(self) -> None:
        self._build()
        self._seed()

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
        # The single winner advanced exactly to seq 1; the stale cursor
        # now conflicts and the correct one drains the rest.
        with self.assertRaises(ServiceError):
            self._consume(expected=0, limit=1)
        body, status = self._consume(expected=1, limit=100)
        self.assertEqual(status, 201)
        self.assertEqual(body["next_after"], 4)


class LeaseEventsConsumeHTTPTest(LeaseEventsConsumeMixin, unittest.TestCase):
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

    def _consume_http(self, subscriber="s1", expected=0, limit=100,
                      path=PATH):
        return self._request(
            path, json.dumps({"subscriber_id": subscriber,
                              "expected": expected, "limit": limit}))

    def test_consume_over_http_with_key_order(self) -> None:
        self._seed()
        status, body, raw = self._consume_http(expected=0, limit=2)
        self.assertEqual(status, 201)
        self.assertEqual(list(body),
                         ["subscriber_id", "events", "next_after",
                          "has_more"])
        for earlier, later in (
                ('"subscriber_id"', '"events"'),
                ('"events"', '"next_after"'),
                ('"next_after"', '"has_more"'),
                ('"seq"', '"lease_id"'),
                ('"lease_id"', '"consumer_id"'),
                ('"consumer_id"', '"type"')):
            self.assertLess(raw.index(earlier), raw.index(later))
        self.assertEqual([e["seq"] for e in body["events"]], [1, 2])
        self.assertEqual(body["next_after"], 2)
        self.assertTrue(body["has_more"])
        status, body, _ = self._consume_http(expected=2, limit=100)
        self.assertEqual(status, 201)
        self.assertEqual([e["seq"] for e in body["events"]], [3, 4])
        self.assertFalse(body["has_more"])
        status, body, _ = self._consume_http(expected=4, limit=100)
        self.assertEqual(status, 200)
        self.assertEqual(body["events"], [])
        self.assertEqual(body["next_after"], 4)

    def test_empty_stream_over_http(self) -> None:
        status, body, _ = self._consume_http()
        self.assertEqual(status, 200)
        self.assertEqual(body, {"subscriber_id": "s1", "events": [],
                                "next_after": 0, "has_more": False})

    def test_query_rejected(self) -> None:
        for query in ("?after=0", "?foo", "?x=1", "?limit=1"):
            with self.subTest(query=query):
                status, body, _ = self._consume_http(path=PATH + query)
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], "query")
                self.assertEqual(list(body), ["message", "field"])
        # A bare trailing '?' carries no parameter and is accepted.
        status, _, _ = self._consume_http(path=PATH + "?")
        self.assertEqual(status, 200)

    def test_bad_json_and_non_object_are_400_request_body(self) -> None:
        for raw in ("{", "[1]", "null", '"s"', ""):
            with self.subTest(raw=raw):
                status, body, _ = self._request(raw=raw)
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], "request_body")
                self.assertEqual(list(body), ["message", "field"])

    def test_field_errors_over_http(self) -> None:
        for payload, field in (
                ({"expected": 0, "limit": 1}, "subscriber_id"),
                ({"subscriber_id": "", "expected": 0, "limit": 1},
                 "subscriber_id"),
                ({"subscriber_id": 1, "expected": 0, "limit": 1},
                 "subscriber_id"),
                ({"subscriber_id": "s1", "limit": 1}, "expected"),
                ({"subscriber_id": "s1", "expected": 0}, "limit"),
                ({"subscriber_id": "s1", "expected": True, "limit": 1},
                 "expected"),
                ({"subscriber_id": "s1", "expected": -1, "limit": 1},
                 "expected"),
                ({"subscriber_id": "s1", "expected": 0, "limit": 0},
                 "limit"),
                ({"subscriber_id": "s1", "expected": 0, "limit": 101},
                 "limit"),
                ({"subscriber_id": "s1", "expected": 0, "limit": "1"},
                 "limit"),
                ({"subscriber_id": "s1", "expected": 0, "limit": 1,
                  "extra": 1}, "extra")):
            with self.subTest(payload=payload):
                status, body, _ = self._request(raw=json.dumps(payload))
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], field)
                self.assertEqual(list(body), ["message", "field"])

    def test_conflict_over_http(self) -> None:
        self._seed()
        status, body, _ = self._consume_http(expected=2)
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "expected")
        self.assertEqual(list(body), ["message", "field"])

    def test_get_is_404(self) -> None:
        status, _, _ = self._request(method="GET")
        self.assertEqual(status, 404)


class LeaseEventsConsumePersistenceTest(LeaseEventsConsumeMixin,
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

    def test_section_follows_cleanup_lease_events(self) -> None:
        self._seed()
        self._consume(expected=0, limit=100)
        keys = list(self._document())
        self.assertIn("lease_event_cursors", keys)
        self.assertEqual(
            keys.index("lease_event_cursors"),
            keys.index("cleanup_lease_events") + 1)
        cursors = self._document()["lease_event_cursors"]
        self.assertEqual(cursors, [{"subscriber_id": "s1", "after": 4}])
        for record in cursors:
            self.assertEqual(list(record), ["subscriber_id", "after"])

    def test_nonempty_page_consumes_a_generation_empty_page_does_not(
            self) -> None:
        self._seed()
        generation = self.state_store.commit_seq
        _, status = self._consume(expected=0, limit=2)
        self.assertEqual(status, 201)
        self.assertEqual(self.state_store.commit_seq, generation + 1)
        # Existing subscriber empty page: 200, no write, no generation.
        _, status = self._consume(expected=2, limit=2)
        self.assertEqual(status, 201)
        self.assertEqual(self.state_store.commit_seq, generation + 2)
        before = self._document()
        _, status = self._consume(expected=4, limit=2)
        self.assertEqual(status, 200)
        self.assertEqual(self.state_store.commit_seq, generation + 2)
        self.assertEqual(self._document(), before)

    def test_empty_page_on_empty_stream_writes_nothing(self) -> None:
        generation = self.state_store.commit_seq
        before = self._document()
        _, status = self._consume(expected=0, limit=100)
        self.assertEqual(status, 200)
        self.assertEqual(self.state_store.commit_seq, generation)
        self.assertEqual(self._document(), before)
        self.assertEqual(self._document()["lease_event_cursors"], [])

    def test_restart_continues_from_saved_cursor(self) -> None:
        self._seed()
        self._consume(expected=0, limit=2)
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        with self.assertRaises(ServiceError) as caught:
            restarted.event_gc_batch_lease_events_consume(
                {"subscriber_id": "s1", "expected": 0, "limit": 100})
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "expected")
        body, status = restarted.event_gc_batch_lease_events_consume(
            {"subscriber_id": "s1", "expected": 2, "limit": 100})
        self.assertEqual(status, 201)
        self.assertEqual([event["seq"] for event in body["events"]], [3, 4])
        self.assertTrue(restarted.persistence_integrity()["consistent"])

    def test_legacy_document_without_section_loads_empty(self) -> None:
        self._seed()
        self._consume(expected=0, limit=100)
        document = self._document()
        document.pop("lease_event_cursors")
        document.pop("integrity_log_version", None)
        legacy_path = os.path.join(self.directory, "legacy.json")
        with open(legacy_path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        restarted = DeviceService()
        attach_persistence(restarted, legacy_path)  # must not raise
        body, status = restarted.event_gc_batch_lease_events_consume(
            {"subscriber_id": "s1", "expected": 0, "limit": 100})
        self.assertEqual(status, 201)
        self.assertEqual(len(body["events"]), 4)
        self.assertTrue(restarted.persistence_integrity()["consistent"])

    def test_save_failure_rolls_cursor_back(self) -> None:
        import e2ee_backend.persistence as persistence_mod

        self._seed()
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
                self._consume(expected=0, limit=100)
        finally:
            persistence_mod.os.fsync = real_fsync
        # Rolled back in memory: the cursor was not created.
        self.assertEqual(self.state_store.commit_seq, generation)
        with self.assertRaises(ServiceError) as caught:
            self._consume(expected=1, limit=100)
        self.assertEqual(caught.exception.status_code, 409)
        # And on disk: the section stays empty.
        self.assertEqual(self._document()["lease_event_cursors"], [])
        # The pull retries and now commits.
        body, status = self._consume(expected=0, limit=100)
        self.assertEqual(status, 201)
        self.assertEqual(len(body["events"]), 4)
        self.assertEqual(self.state_store.commit_seq, generation + 1)
        self.assertTrue(self.service.persistence_integrity()["consistent"])

    def _assert_refuses_startup(self, mutate):
        document = self._document()
        mutate(document)
        document.pop("integrity_log_version", None)
        bad_path = os.path.join(
            self.directory,
            f"bad-{threading.get_ident()}-"
            f"{len(os.listdir(self.directory))}.json")
        with open(bad_path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        with open(bad_path, "rb") as handle:
            original = handle.read()
        service = DeviceService()
        with self.assertRaises(StateFileError):
            attach_persistence(service, bad_path)
        with open(bad_path, "rb") as handle:
            self.assertEqual(handle.read(), original)

    def test_duplicate_subscriber_refuses_startup(self) -> None:
        self._seed()
        self._consume(expected=0, limit=100)

        def mutate(document):
            document["lease_event_cursors"].append(
                {"subscriber_id": "s1", "after": 4})
        self._assert_refuses_startup(mutate)

    def test_bad_cursor_type_refuses_startup(self) -> None:
        self._seed()
        self._consume(expected=0, limit=100)
        for value in (True, "4", 1.0, -1, None):
            with self.subTest(after=value):
                def mutate(document, value=value):
                    document["lease_event_cursors"][0]["after"] = value
                self._assert_refuses_startup(mutate)
        for value in ("", 1, True, None):
            with self.subTest(subscriber_id=value):
                def mutate(document, value=value):
                    document["lease_event_cursors"][0]["subscriber_id"] = value
                self._assert_refuses_startup(mutate)

    def test_bad_key_order_refuses_startup(self) -> None:
        self._seed()
        self._consume(expected=0, limit=100)

        def mutate(document):
            raw = document["lease_event_cursors"][0]
            document["lease_event_cursors"][0] = {
                "after": raw["after"], "subscriber_id": raw["subscriber_id"]}
        self._assert_refuses_startup(mutate)

    def test_cursor_past_stream_refuses_startup(self) -> None:
        self._seed()
        self._consume(expected=0, limit=100)

        def mutate(document):
            document["lease_event_cursors"][0]["after"] = 5
        self._assert_refuses_startup(mutate)

    def test_non_list_section_refuses_startup(self) -> None:
        self._seed()
        self._consume(expected=0, limit=100)

        def mutate(document):
            document["lease_event_cursors"] = {}
        self._assert_refuses_startup(mutate)


if __name__ == "__main__":
    unittest.main()
