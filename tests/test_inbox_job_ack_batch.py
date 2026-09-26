"""Tests for ``POST /v1/inbox-jobs/ack-batch``.

An atomic batch acknowledgement of delivered 1:1-inbox redelivery leases:
the body carries a ``device_id`` and a non-empty ``items`` array of
``lease_id``/``ack_id`` pairs (neither id repeats across items). Every
item is prechecked in array order after the device gate, the first error
aborting the whole batch with its ``field`` prefixed to ``items[i].``;
only then does the batch acknowledge every lease in input order and
commit once. A lease already carrying a different ``ack_id`` conflicts at
``items[i].ack_id``; an exact replay of the same ``ack_id`` skips the
remaining checks. A batch whose items all replay their committed acks
answers 200 and writes nothing; a partial replay conflicts 409 with the
first replayed item's ``items[i].ack_id``.
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


class _BatchMixin(InboxMixin):
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

    def _claim(self, lease_id, device_id="bob", limit=100):
        return self.service.inbox_claim(
            device_id, {"lease_id": lease_id, "limit": limit})

    def _complete(self, lease_id, outcome="delivered", completion_id=None,
                  device_id="bob"):
        return self.service.inbox_lease_complete(
            device_id, lease_id,
            {"completion_id": completion_id or f"c-{lease_id}",
             "outcome": outcome})

    def _expire(self, lease_id: str) -> None:
        with self.service.store._lock:
            for state in self.service.store._delivery.values():
                for lease in state.leases:
                    if lease.lease_id == lease_id:
                        lease.leased_until = "2000-01-01T00:00:00.000000+00:00"

    def _ack_batch(self, device_id="bob", items=None, **extra):
        payload = {"device_id": device_id,
                   "items": items if items is not None else []}
        payload.update(extra)
        return self.service.inbox_job_ack_batch(payload)

    def _ack_pair(self, lease_id, ack_id, device_id="bob"):
        return self._ack_batch(
            device_id=device_id, items=[{"lease_id": lease_id,
                                         "ack_id": ack_id}])


class InboxJobAckBatchServiceTest(_BatchMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    # -- shape validation -------------------------------------------------

    def test_bad_body_and_device_id_errors(self) -> None:
        good = [{"lease_id": "j1", "ack_id": "a1"}]
        for payload in (None, [], "x", 3):
            error = self._error(
                lambda payload=payload:
                self.service.inbox_job_ack_batch(payload))
            self.assertEqual((error.status_code, error.field),
                             (400, "request_body"), payload)
        for payload in (
                {"items": good},
                {"device_id": "", "items": good},
                {"device_id": 4, "items": good},
                {"device_id": None, "items": good}):
            error = self._error(
                lambda payload=payload:
                self.service.inbox_job_ack_batch(payload))
            self.assertEqual((error.status_code, error.field),
                             (400, "device_id"), payload)

    def test_items_shape_errors(self) -> None:
        for items in (None, {}, "x", 3, []):
            error = self._error(lambda items=items:
                                self._ack_batch(items=items))
            self.assertEqual((error.status_code, error.field),
                             (400, "items"), items)
        error = self._error(lambda: self._ack_batch(items=["x"]))
        self.assertEqual((error.status_code, error.field), (400, "items[0]"))
        error = self._error(lambda: self._ack_batch(items=[[]]))
        self.assertEqual((error.status_code, error.field), (400, "items[0]"))

    def test_extra_top_level_key_rejected_with_that_field(self) -> None:
        item = {"lease_id": "j1", "ack_id": "a1"}
        error = self._error(lambda: self._ack_batch(items=[item], op="done"))
        self.assertEqual((error.status_code, error.field), (400, "op"))
        error = self._error(lambda: self._ack_batch(items=[item], bogus=1))
        self.assertEqual((error.status_code, error.field), (400, "bogus"))

    def test_item_field_errors_carry_the_index(self) -> None:
        good = {"lease_id": "j1", "ack_id": "a1"}
        cases = (
            # lease_id
            ([{"ack_id": "a1"}], "items[0].lease_id"),
            ([{"lease_id": "", "ack_id": "a1"}], "items[0].lease_id"),
            ([{"lease_id": 4, "ack_id": "a1"}], "items[0].lease_id"),
            ([{"lease_id": None, "ack_id": "a1"}], "items[0].lease_id"),
            # ack_id
            ([{"lease_id": "j1"}], "items[0].ack_id"),
            ([{"lease_id": "j1", "ack_id": ""}], "items[0].ack_id"),
            ([{"lease_id": "j1", "ack_id": 4}], "items[0].ack_id"),
            ([{"lease_id": "j1", "ack_id": None}], "items[0].ack_id"),
            # later items keep their index
            ([good, {"ack_id": "a2"}], "items[1].lease_id"),
            ([good, {"lease_id": "j2"}], "items[1].ack_id"),
        )
        for items, field in cases:
            error = self._error(lambda items=items:
                                self._ack_batch(items=items))
            self.assertEqual((error.status_code, error.field),
                             (400, field), items)

    def test_item_extra_keys_rejected_at_item_level(self) -> None:
        base = {"lease_id": "j1", "ack_id": "a1"}
        cases = (
            [dict(base, extra=1)],
            [dict(base, outcome="delivered")],
            [dict(base), dict(base, lease_id="j2", bogus=None)],
        )
        for items in cases:
            index = 0 if len(items) == 1 else 1
            error = self._error(lambda items=items:
                                self._ack_batch(items=items))
            self.assertEqual((error.status_code, error.field),
                             (400, f"items[{index}]"), items)

    def test_duplicate_ids_rejected_at_item_level(self) -> None:
        items = [{"lease_id": "j1", "ack_id": "a1"},
                 {"lease_id": "j1", "ack_id": "a2"}]
        error = self._error(lambda: self._ack_batch(items=items))
        self.assertEqual((error.status_code, error.field), (400, "items[1]"))
        items = [{"lease_id": "j1", "ack_id": "a1"},
                 {"lease_id": "j2", "ack_id": "a1"}]
        error = self._error(lambda: self._ack_batch(items=items))
        self.assertEqual((error.status_code, error.field), (400, "items[1]"))

    # -- device gate ------------------------------------------------------

    def test_device_gate_precedes_item_checks(self) -> None:
        self._dispatch(job_id="j1")
        self._complete("j1")
        items = [{"lease_id": "j1", "ack_id": "a1"}]
        error = self._error(lambda: self._ack_batch(device_id="ghost",
                                                    items=items))
        self.assertEqual((error.status_code, error.field),
                         (409, "device_id"))
        self.service.store.revoke_device("bob")
        error = self._error(lambda: self._ack_batch(items=items))
        self.assertEqual((error.status_code, error.field),
                         (409, "device_id"))

    # -- first-time ack ---------------------------------------------------

    def _setup_two_leases(self):
        # L1 takes the first two messages, j1's dispatch the rest.
        self._claim("L1", limit=2)
        self._dispatch(job_id="j1")
        self._complete("L1")
        self._complete("j1")

    def test_first_batch_ack_is_201_with_ordered_keys(self) -> None:
        self._setup_two_leases()
        body, status = self._ack_batch(items=[
            {"lease_id": "L1", "ack_id": "aL"},
            {"lease_id": "j1", "ack_id": "aJ"}])
        self.assertEqual(status, 201)
        self.assertEqual(list(body), ["device_id", "results"])
        self.assertEqual(body["device_id"], "bob")
        self.assertEqual([r["lease_id"] for r in body["results"]],
                         ["L1", "j1"])
        for result in body["results"]:
            self.assertEqual(list(result),
                             ["lease_id", "ack_id", "message_count"])
        self.assertEqual(body["results"][0],
                         {"lease_id": "L1", "ack_id": "aL",
                          "message_count": 2})
        self.assertEqual(body["results"][1],
                         {"lease_id": "j1", "ack_id": "aJ",
                          "message_count": 3})

    def test_ack_sets_acked_and_ack_sequence_and_freezes_ack_id(self) -> None:
        self._setup_two_leases()
        # Retries before the ack must keep their attempts counts.
        self.service.retry_message(
            self.sid1, "a1", {"device_id": "bob", "attempt_id": "at1"})
        self._ack_batch(items=[{"lease_id": "L1", "ack_id": "aL"}])
        delivery = self.service.store._delivery
        # L1 claimed a1 and a2 in inbox order.
        for message_id, sequence in (("a1", 1), ("a2", 2)):
            state = delivery[(self.sid1, message_id)]
            self.assertIs(state.acked, True)
            self.assertEqual(state.ack_sequence, sequence)
            self.assertEqual(state.attempts, 1 if message_id == "a1" else 0)
            self.assertEqual([l.ack_id for l in state.leases if
                              l.lease_id == "L1"], ["aL"])
        # j1's messages stay unacked.
        self.assertFalse(delivery[(self.sid1, "a3")].acked)

    def test_batch_does_not_move_an_already_terminal_job(self) -> None:
        self._dispatch(job_id="j1")
        self._complete("j1", outcome="delivered")
        job, _ = self._job(job_id="j1", op="status")
        self.assertEqual(job["state"], "succeeded")
        self._ack_pair("j1", "a1")
        job, _ = self._job(job_id="j1", op="status")
        self.assertEqual(job["state"], "succeeded")

    # -- precheck errors --------------------------------------------------

    def test_unknown_lease_is_404_at_items_index(self) -> None:
        error = self._error(lambda: self._ack_batch(items=[
            {"lease_id": "ghost", "ack_id": "a1"}]))
        self.assertEqual((error.status_code, error.field),
                         (404, "items[0].lease_id"))

    def test_cross_device_lease_is_409_at_lease_id(self) -> None:
        self._dispatch(job_id="jb", device_id="bob2")
        error = self._error(lambda: self._ack_batch(items=[
            {"lease_id": "jb", "ack_id": "a1"}]))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[0].lease_id"))

    def test_lease_without_delivered_completion_is_409(self) -> None:
        # An active lease (no completion at all).
        self._claim("L1", limit=1)
        error = self._error(lambda: self._ack_batch(items=[
            {"lease_id": "L1", "ack_id": "a1"}]))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[0].lease_id"))
        # A lease completed failed.
        self._dispatch(job_id="j1")
        self._complete("j1", outcome="failed")
        error = self._error(lambda: self._ack_batch(items=[
            {"lease_id": "j1", "ack_id": "a1"}]))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[0].lease_id"))

    def test_different_ack_id_is_409_at_ack_id(self) -> None:
        self._dispatch(job_id="j1")
        self._complete("j1")
        self._ack_pair("j1", "a1")
        error = self._error(lambda: self._ack_pair("j1", "other"))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[0].ack_id"))

    def test_first_error_in_array_order_wins_and_nothing_is_written(
            self) -> None:
        self._setup_two_leases()
        # Item 0 fine, item 1 unknown -> items[1] reported, nothing acked.
        error = self._error(lambda: self._ack_batch(items=[
            {"lease_id": "L1", "ack_id": "aL"},
            {"lease_id": "ghost", "ack_id": "aX"}]))
        self.assertEqual((error.status_code, error.field),
                         (404, "items[1].lease_id"))
        for state in self.service.store._delivery.values():
            self.assertFalse(state.acked)
            self.assertTrue(all(l.ack_id is None for l in state.leases))
        # An earlier bad item wins over a later good one.
        error = self._error(lambda: self._ack_batch(items=[
            {"lease_id": "ghost", "ack_id": "aX"},
            {"lease_id": "j1", "ack_id": "aJ"}]))
        self.assertEqual((error.status_code, error.field),
                         (404, "items[0].lease_id"))

    # -- replay semantics -------------------------------------------------

    def test_all_replays_answer_200_byte_identically(self) -> None:
        self._setup_two_leases()
        items = [{"lease_id": "L1", "ack_id": "aL"},
                 {"lease_id": "j1", "ack_id": "aJ"}]
        first, status = self._ack_batch(items=items)
        self.assertEqual(status, 201)
        replay, status = self._ack_batch(items=items)
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)
        # A single-item replay is also 200.
        one, status = self._ack_pair("j1", "aJ")
        self.assertEqual(status, 200)
        self.assertEqual(one["results"][0], first["results"][1])

    def test_partial_replay_conflicts_with_first_replayed_item(self) -> None:
        self._setup_two_leases()
        self._ack_pair("L1", "aL")
        # Replay at index 0.
        error = self._error(lambda: self._ack_batch(items=[
            {"lease_id": "L1", "ack_id": "aL"},
            {"lease_id": "j1", "ack_id": "aJ"}]))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[0].ack_id"))
        # Replay at index 1.
        error = self._error(lambda: self._ack_batch(items=[
            {"lease_id": "j1", "ack_id": "aJ"},
            {"lease_id": "L1", "ack_id": "aL"}]))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[1].ack_id"))
        # The fresh item was not written.
        self.assertTrue(all(
            l.ack_id is None
            for state in self.service.store._delivery.values()
            for l in state.leases if l.lease_id == "j1"))
        self.assertFalse(
            self.service.store._delivery[(self.sid1, "a3")].acked)

    def test_interop_with_single_lease_ack(self) -> None:
        # Single ack first (records no ack_id); the batch then treats the
        # lease as a first-time ack, 201, recording only the ack_id.
        self._claim("L1", limit=2)
        self._dispatch(job_id="j1")
        self._complete("L1")
        self._complete("j1")
        single = self.service.inbox_lease_ack("bob", "j1")
        self.assertEqual(single[1], 201)
        body, status = self._ack_pair("j1", "aJ")
        self.assertEqual(status, 201)
        self.assertEqual(body["results"][0]["message_count"], 3)
        # The same batch id now replays 200.
        _, status = self._ack_pair("j1", "aJ")
        self.assertEqual(status, 200)
        # Batch first, single ack afterwards answers its idempotent 200.
        _, status = self._ack_pair("L1", "aL")
        self.assertEqual(status, 201)
        _, status = self.service.inbox_lease_ack("bob", "L1")
        self.assertEqual(status, 200)


class InboxJobAckBatchPersistenceTest(_BatchMixin, unittest.TestCase):
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

    def test_first_ack_persists_ack_id_after_completion(self) -> None:
        self._claim("L1", limit=2)
        self._complete("L1")
        before = self.state_store.commit_seq
        _, status = self._ack_pair("L1", "a1")
        self.assertEqual(status, 201)
        self.assertEqual(self.state_store.commit_seq, before + 1)
        document = self._document()
        leased = [r for r in document["delivery"] if r.get("leases")]
        self.assertEqual(len(leased), 2)
        sequences = {m["message_id"]: m["sequence"]
                     for stream in document["messages"].values()
                     for m in stream}
        for record in leased:
            lease = record["leases"][0]
            self.assertEqual(list(lease),
                             ["lease_id", "limit", "leased_until",
                              "released_at", "renewals", "completion",
                              "ack_id"])
            self.assertEqual(lease["ack_id"], "a1")
            self.assertIs(record["acked"], True)
            self.assertEqual(record["ack_sequence"],
                             sequences[record["message_id"]])

    def test_replay_persists_nothing_and_advances_no_generation(self) -> None:
        self._claim("L1", limit=2)
        self._complete("L1")
        items = [{"lease_id": "L1", "ack_id": "a1"}]
        first, status = self._ack_batch(items=items)
        self.assertEqual(status, 201)
        generation = self.state_store.commit_seq
        replay, status = self._ack_batch(items=items)
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)
        self.assertEqual(self.state_store.commit_seq, generation)

    def test_restart_restores_ack_ids_and_replays(self) -> None:
        self._claim("L1", limit=2)
        self._dispatch(job_id="j1")
        self._complete("L1")
        self._complete("j1")
        first, status = self._ack_batch(items=[
            {"lease_id": "L1", "ack_id": "aL"},
            {"lease_id": "j1", "ack_id": "aJ"}])
        self.assertEqual(status, 201)
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        body, status = restarted.inbox_job_ack_batch(
            {"device_id": "bob", "items": [
                {"lease_id": "L1", "ack_id": "aL"},
                {"lease_id": "j1", "ack_id": "aJ"}]})
        self.assertEqual(status, 200)
        self.assertEqual(body["results"], first["results"])
        # A different ack_id still conflicts after restart.
        with self.assertRaises(ServiceError) as caught:
            restarted.inbox_job_ack_batch(
                {"device_id": "bob",
                 "items": [{"lease_id": "L1", "ack_id": "other"}]})
        self.assertEqual((caught.exception.status_code,
                          caught.exception.field),
                         (409, "items[0].ack_id"))
        self.assertTrue(restarted.persistence_integrity()["consistent"])

    def test_legacy_lease_without_ack_id_key_loads_as_null(self) -> None:
        self._claim("L1", limit=2)
        self._complete("L1")
        document = self._document()
        for record in document["delivery"]:
            for lease in record.get("leases", []):
                lease.pop("ack_id", None)
        document.pop("integrity_log_version", None)
        legacy_path = os.path.join(self.directory, "legacy.json")
        with open(legacy_path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        restarted = DeviceService()
        attach_persistence(restarted, legacy_path)  # must not raise
        body, status = restarted.inbox_job_ack_batch(
            {"device_id": "bob",
             "items": [{"lease_id": "L1", "ack_id": "a1"}]})
        self.assertEqual(status, 201)
        self.assertEqual(body["results"][0]["message_count"], 2)

    def _malformed_document(self, mutate):
        self._claim("L1", limit=2)
        self._complete("L1")
        self._ack_pair("L1", "a1")
        document = self._document()
        mutate(document)
        document.pop("integrity_log_version", None)
        bad_path = os.path.join(self.directory, f"bad-{id(mutate)}.json")
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

    def test_restore_rejects_bad_ack_id_types(self) -> None:
        def as_value(value):
            def mutate(document):
                document["delivery"][0]["leases"][0]["ack_id"] = value
            return mutate
        for value in (5, "", True, ["a1"], {"ack_id": "a1"}):
            with self.subTest(value=value):
                self._assert_refuses_startup(
                    self._malformed_document(as_value(value)))

    def test_restore_rejects_ack_id_without_delivered_completion(self) -> None:
        def no_completion(document):
            for record in document["delivery"]:
                for lease in record.get("leases", []):
                    lease["completion"] = None

        def failed_outcome(document):
            for record in document["delivery"]:
                for lease in record.get("leases", []):
                    lease["completion"]["outcome"] = "failed"
        self._assert_refuses_startup(
            self._malformed_document(no_completion))
        self._assert_refuses_startup(
            self._malformed_document(failed_outcome))

    def test_restore_rejects_ack_id_on_unacked_record(self) -> None:
        def mutate(document):
            document["delivery"][0]["acked"] = False
            document["delivery"][0]["ack_sequence"] = 0
        self._assert_refuses_startup(self._malformed_document(mutate))

    def test_restore_rejects_inconsistent_ack_id_across_records(self) -> None:
        def mutate(document):
            leased = [r for r in document["delivery"] if r.get("leases")]
            self.assertEqual(len(leased), 2)
            leased[1]["leases"][0]["ack_id"] = "OTHER"
        self._assert_refuses_startup(self._malformed_document(mutate))

    def test_save_failure_rolls_back_the_whole_batch(self) -> None:
        import e2ee_backend.persistence as persistence_mod

        real_fsync = persistence_mod.os.fsync
        calls = {"n": 0}

        def fail_first_directory_fsync(fd):  # noqa: ANN001
            if os.fstat(fd).st_mode & 0o170000 == 0o040000:
                calls["n"] += 1
                if calls["n"] == 1:
                    raise OSError("transient directory fsync failure")
            real_fsync(fd)

        self._claim("L1", limit=2)
        self._complete("L1")
        generation = self.state_store.commit_seq
        persistence_mod.os.fsync = fail_first_directory_fsync
        try:
            with self.assertRaises(PersistenceUnavailable):
                self._ack_pair("L1", "a1")
        finally:
            persistence_mod.os.fsync = real_fsync
        self.assertEqual(self.state_store.commit_seq, generation)
        # Nothing landed in memory: the lease is still first-time.
        self.assertTrue(all(
            l.ack_id is None
            for state in self.service.store._delivery.values()
            for l in state.leases))
        body, status = self._ack_pair("L1", "a1")
        self.assertEqual(status, 201)
        self.assertEqual(body["results"][0]["ack_id"], "a1")
        self.assertEqual(self.state_store.commit_seq, generation + 1)


class InboxJobAckBatchHTTPTest(_BatchMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()
        self.server, self.service = create_server(
            "127.0.0.1", 0, self.service)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        self._claim("L1", limit=2)
        self._dispatch(job_id="j1")
        self._complete("L1")
        self._complete("j1")

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _request(self, payload, raw_body=None):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        body = raw_body if raw_body is not None else json.dumps(payload)
        conn.request("POST", "/v1/inbox-jobs/ack-batch", body=body,
                     headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        data = response.read().decode("utf-8")
        conn.close()
        return response.status, (json.loads(data) if data else None), data

    def _items(self):
        return [{"lease_id": "L1", "ack_id": "aL"},
                {"lease_id": "j1", "ack_id": "aJ"}]

    def test_ack_batch_over_http(self) -> None:
        status, body, raw = self._request(
            {"device_id": "bob", "items": self._items()})
        self.assertEqual(status, 201)
        self.assertEqual(list(body), ["device_id", "results"])
        self.assertEqual([r["message_count"] for r in body["results"]],
                         [2, 3])
        for item in body["results"]:
            self.assertEqual(list(item),
                             ["lease_id", "ack_id", "message_count"])
        self.assertLess(raw.index('"device_id"'), raw.index('"results"'))
        self.assertLess(raw.index('"lease_id"'), raw.index('"ack_id"'))
        self.assertLess(raw.index('"ack_id"'), raw.index('"message_count"'))
        # The full replay answers 200 byte-identically.
        status, body, raw2 = self._request(
            {"device_id": "bob", "items": self._items()})
        self.assertEqual(status, 200)
        self.assertEqual(raw2, raw)

    def test_errors_over_http(self) -> None:
        status, body, _ = self._request(None, raw_body="{")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "request_body")
        self.assertEqual(list(body), ["message", "field"])
        status, body, _ = self._request([1, 2])
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "request_body")
        status, body, _ = self._request({"device_id": "bob", "items": []})
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "items")
        status, body, _ = self._request(
            {"device_id": "bob", "items": [{"ack_id": "a1"}]})
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "items[0].lease_id")
        status, body, _ = self._request(
            {"device_id": "bob", "items": [{"lease_id": "j1"}]})
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "items[0].ack_id")
        status, body, _ = self._request(
            {"device_id": "bob", "items": [{"lease_id": "j1",
                                            "ack_id": "a1",
                                            "bogus": 1}]})
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "items[0]")
        status, body, _ = self._request(
            {"device_id": "ghost",
             "items": [{"lease_id": "j1", "ack_id": "a1"}]})
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "device_id")
        status, body, _ = self._request(
            {"device_id": "bob",
             "items": [{"lease_id": "ghost", "ack_id": "a1"}]})
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "items[0].lease_id")
        # Partial replay: ack L1, then replay it alongside a fresh j1.
        self._request({"device_id": "bob",
                       "items": [{"lease_id": "L1", "ack_id": "aL"}]})
        status, body, _ = self._request(
            {"device_id": "bob", "items": [
                {"lease_id": "L1", "ack_id": "aL"},
                {"lease_id": "j1", "ack_id": "aJ"}]})
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "items[0].ack_id")
        # The failed batch wrote nothing: j1 is still first-time.
        self.assertTrue(all(
            l.ack_id is None
            for state in self.service.store._delivery.values()
            for l in state.leases if l.lease_id == "j1"))


if __name__ == "__main__":
    unittest.main()
