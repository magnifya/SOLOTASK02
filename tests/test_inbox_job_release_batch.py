"""Tests for ``POST /v1/inbox-jobs/release-batch``.

An atomic batch release of 1:1-inbox redelivery leases: the body carries a
non-empty string ``device_id`` and a non-empty ``items`` array of objects
carrying exactly a non-empty string ``lease_id`` (no ``lease_id`` repeats
across items). The device gate (unknown/revoked -> 409/device_id) runs
first; items are then prechecked in array order with the single-lease
release rules (unknown lease 404/items[i].lease_id; cross-device or
already completed lease 409/items[i].lease_id) and the first error aborts
the whole batch with nothing written. An already-released lease is an
exact replay: a batch whose items all replay committed releases answers
200 with the frozen values and writes nothing; a partial replay conflicts
409 with the first replayed item's ``items[i].lease_id``. Otherwise every
first-time lease is released in input order — an expired but uncompleted
lease may still be released — stamping one shared UTC ``released_at``
onto every delivery record, and the batch commits once
(201, commit_seq + 1).
"""
import json
import os
import shutil
import tempfile
import threading
import unittest
from datetime import datetime
from http.client import HTTPConnection

from tests.test_device_inbox import InboxMixin
from e2ee_backend.persistence import (
    PersistenceUnavailable, attach_persistence)
from e2ee_backend.service import DeviceService, ServiceError
from e2ee_backend.http_app import create_server

_PAST = "2000-01-01T00:00:00.000000+00:00"


def _parse(stamp: str) -> datetime:
    return datetime.fromisoformat(stamp)


class _BatchMixin(InboxMixin):
    def _error(self, callable_):
        with self.assertRaises(ServiceError) as caught:
            callable_()
        return caught.exception

    def _claim(self, lease_id, device_id="bob", limit=100):
        body, status = self.service.inbox_claim(
            device_id, {"lease_id": lease_id, "limit": limit})
        self.assertEqual(status, 201)
        return body

    def _expire_claim(self, lease_id: str) -> None:
        """Rewrite a lease with no renewals to a past claim deadline."""
        with self.service.store._lock:
            for state in self.service.store._delivery.values():
                for lease in state.leases:
                    if lease.lease_id == lease_id:
                        lease.leased_until = _PAST

    def _batch(self, device_id="bob", items=None, **extra):
        payload = {"device_id": device_id,
                   "items": items if items is not None else []}
        payload.update(extra)
        return self.service.inbox_job_release_batch(payload)

    def _two_leases(self):
        """L1 claims a1/a2 (limit 2); L2 claims the remaining three."""
        first = self._claim("L1", limit=2)
        second = self._claim("L2", limit=100)
        return first, second


class InboxJobReleaseBatchServiceTest(_BatchMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def test_bad_body_and_device_id_errors(self) -> None:
        good = [{"lease_id": "L1"}]
        for payload in (None, [], "x", 3, True):
            error = self._error(
                lambda payload=payload:
                self.service.inbox_job_release_batch(payload))
            self.assertEqual((error.status_code, error.field),
                             (400, "request_body"), payload)
        for payload in (
                {"items": good},
                {"device_id": "", "items": good},
                {"device_id": 4, "items": good},
                {"device_id": None, "items": good},
                {"device_id": True, "items": good}):
            error = self._error(
                lambda payload=payload:
                self.service.inbox_job_release_batch(payload))
            self.assertEqual((error.status_code, error.field),
                             (400, "device_id"), payload)

    def test_items_shape_errors(self) -> None:
        for items in (None, {}, "x", 3, []):
            error = self._error(lambda items=items: self._batch(items=items))
            self.assertEqual((error.status_code, error.field),
                             (400, "items"), items)
        error = self._error(lambda: self._batch(items=["x"]))
        self.assertEqual((error.status_code, error.field), (400, "items[0]"))
        error = self._error(lambda: self._batch(items=[[]]))
        self.assertEqual((error.status_code, error.field), (400, "items[0]"))

    def test_extra_top_level_key_rejected_with_that_field(self) -> None:
        item = {"lease_id": "L1"}
        error = self._error(lambda: self._batch(items=[item], op="done"))
        self.assertEqual((error.status_code, error.field), (400, "op"))
        error = self._error(lambda: self._batch(items=[item], bogus=1))
        self.assertEqual((error.status_code, error.field), (400, "bogus"))

    def test_item_field_errors_carry_the_index(self) -> None:
        cases = (
            ([{}], "items[0].lease_id"),
            ([{"lease_id": ""}], "items[0].lease_id"),
            ([{"lease_id": 4}], "items[0].lease_id"),
            ([{"lease_id": None}], "items[0].lease_id"),
            ([{"lease_id": True}], "items[0].lease_id"),
        )
        for items, field in cases:
            error = self._error(lambda items=items: self._batch(items=items))
            self.assertEqual((error.status_code, error.field),
                             (400, field), items)

    def test_item_extra_keys_rejected_at_item_level(self) -> None:
        cases = (
            [{"lease_id": "L1", "extra": 1}],
            [{"lease_id": "L1", "renewal_id": "r1"}],
            [{"lease_id": "L1"}, {"lease_id": "L2", "bogus": None}],
        )
        for items in cases:
            index = 0 if len(items) == 1 else 1
            error = self._error(lambda items=items: self._batch(items=items))
            self.assertEqual((error.status_code, error.field),
                             (400, f"items[{index}]"), items)

    def test_duplicate_lease_ids_rejected_at_item_level(self) -> None:
        error = self._error(lambda: self._batch(items=[
            {"lease_id": "L1"}, {"lease_id": "L1"}]))
        self.assertEqual((error.status_code, error.field), (400, "items[1]"))

    def test_device_gate_precedes_item_checks(self) -> None:
        self._two_leases()
        items = [{"lease_id": "NOPE"}]
        error = self._error(lambda: self._batch(device_id="ghost",
                                                items=items))
        self.assertEqual((error.status_code, error.field),
                         (409, "device_id"))
        self.service.store.revoke_device("bob")
        error = self._error(lambda: self._batch(items=items))
        self.assertEqual((error.status_code, error.field),
                         (409, "device_id"))

    def test_first_batch_releases_each_lease_with_one_shared_stamp(
            self) -> None:
        self._two_leases()
        body, status = self._batch(items=[{"lease_id": "L1"},
                                          {"lease_id": "L2"}])
        self.assertEqual(status, 201)
        self.assertEqual(list(body), ["device_id", "results"])
        self.assertEqual(body["device_id"], "bob")
        self.assertEqual(len(body["results"]), 2)
        for item in body["results"]:
            self.assertEqual(list(item),
                             ["lease_id", "released_at", "released_count"])
            self.assertRegex(item["released_at"], r"\.\d{6}\+00:00$")
        self.assertEqual(body["results"][0]["lease_id"], "L1")
        self.assertEqual(body["results"][0]["released_count"], 2)
        self.assertEqual(body["results"][1]["lease_id"], "L2")
        self.assertEqual(body["results"][1]["released_count"], 3)
        # Every item of one batch shares the same release timestamp.
        self.assertEqual(body["results"][0]["released_at"],
                         body["results"][1]["released_at"])

    def test_release_is_stamped_on_every_copy_of_the_lease(self) -> None:
        self._two_leases()
        body, _ = self._batch(items=[{"lease_id": "L1"},
                                     {"lease_id": "L2"}])
        stamp = body["results"][0]["released_at"]
        copies = {"L1": 0, "L2": 0}
        for state in self.service.store._delivery.values():
            for lease in state.leases:
                if lease.lease_id in copies:
                    copies[lease.lease_id] += 1
                    self.assertEqual(lease.released_at, stamp)
        self.assertEqual(copies, {"L1": 2, "L2": 3})
        # The released leases show up as released in the single-lease view.
        self.assertEqual(
            self.service.inbox_lease_get("bob", "L1")["state"], "released")
        self.assertEqual(
            self.service.inbox_lease_get("bob", "L2")["released_at"], stamp)

    def test_expired_uncompleted_lease_can_still_be_released(self) -> None:
        self._two_leases()
        self._expire_claim("L1")
        body, status = self._batch(items=[{"lease_id": "L1"}])
        self.assertEqual(status, 201)
        self.assertEqual(body["results"][0]["released_count"], 2)
        self.assertEqual(
            self.service.inbox_lease_get("bob", "L1")["state"], "released")

    def test_all_replays_answer_200_with_frozen_values(self) -> None:
        self._two_leases()
        items = [{"lease_id": "L1"}, {"lease_id": "L2"}]
        first, status = self._batch(items=items)
        self.assertEqual(status, 201)
        replay, status = self._batch(items=items)
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)

    def test_replay_wins_after_expiry_but_not_after_revocation(self) -> None:
        self._two_leases()
        items = [{"lease_id": "L1"}]
        first, status = self._batch(items=items)
        self.assertEqual(status, 201)
        # The frozen release is answered however stale the request.
        self._expire_claim("L1")
        replay, status = self._batch(items=items)
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)
        # Unlike the single-lease route, the batch's device gate runs
        # ahead of every item check, so a revoked device rejects even an
        # otherwise-exact replay (409/device_id).
        self.service.store.revoke_device("bob")
        error = self._error(lambda: self._batch(items=items))
        self.assertEqual((error.status_code, error.field),
                         (409, "device_id"))

    def test_partial_replay_conflicts_with_first_replayed_item(self) -> None:
        self._two_leases()
        self._batch(items=[{"lease_id": "L1"}, {"lease_id": "L2"}])
        self._claim("L3", limit=100)
        # Replay at index 0.
        error = self._error(lambda: self._batch(items=[
            {"lease_id": "L1"}, {"lease_id": "L3"}]))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[0].lease_id"))
        # Replay at index 1.
        error = self._error(lambda: self._batch(items=[
            {"lease_id": "L3"}, {"lease_id": "L2"}]))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[1].lease_id"))
        # The fresh release of L3 was not written: L3 is still active.
        self.assertEqual(
            self.service.inbox_lease_get("bob", "L3")["state"], "active")

    def test_per_item_errors_use_indexed_fields(self) -> None:
        self._two_leases()
        # Unknown lease.
        error = self._error(lambda: self._batch(items=[
            {"lease_id": "ghost"}]))
        self.assertEqual((error.status_code, error.field),
                         (404, "items[0].lease_id"))
        # A lease owned by another device (bob2's claim on o1).
        self._claim("lb", device_id="bob2", limit=10)
        error = self._error(lambda: self._batch(items=[
            {"lease_id": "lb"}]))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[0].lease_id"))
        # A completed lease cannot be released.
        self.service.inbox_lease_complete(
            "bob", "L1", {"completion_id": "c1", "outcome": "delivered"})
        error = self._error(lambda: self._batch(items=[
            {"lease_id": "L1"}]))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[0].lease_id"))

    def test_first_error_in_array_order_wins_and_nothing_is_written(
            self) -> None:
        self._two_leases()
        # Item 0 fine, item 1 unknown: items[1] reported and L1 not
        # released.
        error = self._error(lambda: self._batch(items=[
            {"lease_id": "L1"}, {"lease_id": "ghost"}]))
        self.assertEqual((error.status_code, error.field),
                         (404, "items[1].lease_id"))
        self.assertEqual(
            self.service.inbox_lease_get("bob", "L1")["state"], "active")
        # An earlier bad item wins over a later good one.
        error = self._error(lambda: self._batch(items=[
            {"lease_id": "ghost"}, {"lease_id": "L2"}]))
        self.assertEqual((error.status_code, error.field),
                         (404, "items[0].lease_id"))
        self.assertEqual(
            self.service.inbox_lease_get("bob", "L2")["state"], "active")


class InboxJobReleaseBatchPersistenceTest(_BatchMixin, unittest.TestCase):
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

    def test_batch_advances_one_generation_replays_and_failures_none(
            self) -> None:
        self._two_leases()
        items = [{"lease_id": "L1"}, {"lease_id": "L2"}]
        before = self.state_store.commit_seq
        first, status = self._batch(items=items)
        self.assertEqual(status, 201)
        self.assertEqual(self.state_store.commit_seq, before + 1)
        # The release landed on every delivery copy, keyed in order.
        released = [record for record in self._document()["delivery"]
                    if record.get("leases")]
        self.assertEqual(len(released), 5)
        stamp = first["results"][0]["released_at"]
        for record in released:
            lease = record["leases"][0]
            self.assertEqual(list(lease),
                             ["lease_id", "limit", "leased_until",
                              "released_at", "renewals", "completion",
                              "ack_id"])
            self.assertEqual(lease["released_at"], stamp)
        # A full replay consumes no generation.
        _, status = self._batch(items=items)
        self.assertEqual(status, 200)
        self.assertEqual(self.state_store.commit_seq, before + 1)
        # A failed batch consumes no generation either.
        with self.assertRaises(ServiceError):
            self._batch(items=[{"lease_id": "ghost"}])
        self.assertEqual(self.state_store.commit_seq, before + 1)

    def test_failed_batch_writes_nothing(self) -> None:
        self._two_leases()
        before = self.state_store.commit_seq
        with self.assertRaises(ServiceError) as caught:
            self._batch(items=[{"lease_id": "L1"}, {"lease_id": "ghost"}])
        self.assertEqual(caught.exception.status_code, 404)
        self.assertEqual(self.state_store.commit_seq, before)
        self.assertEqual(
            self.service.inbox_lease_get("bob", "L1")["state"], "active")

    def test_restart_restores_releases_and_replays(self) -> None:
        self._two_leases()
        items = [{"lease_id": "L1"}, {"lease_id": "L2"}]
        first, status = self._batch(items=items)
        self.assertEqual(status, 201)
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        replay, status = restarted.inbox_job_release_batch(
            {"device_id": "bob", "items": items})
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)
        self.assertTrue(restarted.persistence_integrity()["consistent"])

    def test_save_failure_rolls_back_the_whole_batch(self) -> None:
        import e2ee_backend.persistence as persistence_mod

        real_fsync = persistence_mod.os.fsync
        calls = {"n": 0}

        def fail_first_directory_fsync(fd: int) -> None:  # noqa: ANN001
            if os.fstat(fd).st_mode & 0o170000 == 0o040000:
                calls["n"] += 1
                if calls["n"] == 1:
                    raise OSError("transient directory fsync failure")
            real_fsync(fd)

        self._two_leases()
        generation = self.state_store.commit_seq
        persistence_mod.os.fsync = fail_first_directory_fsync
        try:
            with self.assertRaises(PersistenceUnavailable):
                self._batch(items=[{"lease_id": "L1"}, {"lease_id": "L2"}])
        finally:
            persistence_mod.os.fsync = real_fsync
        self.assertEqual(self.state_store.commit_seq, generation)
        # Neither release landed in memory: both are fresh 201s.
        body, status = self._batch(items=[{"lease_id": "L1"},
                                          {"lease_id": "L2"}])
        self.assertEqual(status, 201)
        self.assertEqual(self.state_store.commit_seq, generation + 1)
        self.assertEqual(len(body["results"]), 2)


class InboxJobReleaseBatchHTTPTest(_BatchMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()
        self.server, self.service = create_server(
            "127.0.0.1", 0, self.service)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        self._two_leases()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _request(self, payload, raw_body=None):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        body = raw_body if raw_body is not None else json.dumps(payload)
        conn.request("POST", "/v1/inbox-jobs/release-batch", body=body,
                     headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        data = response.read().decode("utf-8")
        conn.close()
        return response.status, (json.loads(data) if data else None), data

    def test_release_batch_over_http(self) -> None:
        status, body, raw = self._request(
            {"device_id": "bob", "items": [{"lease_id": "L1"},
                                           {"lease_id": "L2"}]})
        self.assertEqual(status, 201)
        self.assertEqual(list(body), ["device_id", "results"])
        self.assertEqual([list(r) for r in body["results"]],
                         [["lease_id", "released_at", "released_count"],
                          ["lease_id", "released_at", "released_count"]])
        self.assertLess(raw.index('"device_id"'), raw.index('"results"'))
        self.assertLess(raw.index('"lease_id"'),
                        raw.index('"released_at"'))
        self.assertLess(raw.index('"released_at"'),
                        raw.index('"released_count"'))
        # Full replay answers 200 byte-identically.
        status, body, raw2 = self._request(
            {"device_id": "bob", "items": [{"lease_id": "L1"},
                                           {"lease_id": "L2"}]})
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
            {"device_id": "bob", "items": [{}]})
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "items[0].lease_id")
        status, body, _ = self._request(
            {"device_id": "bob", "items": [{"lease_id": "L1", "x": 1}]})
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "items[0]")
        status, body, _ = self._request(
            {"device_id": "bob", "items": [{"lease_id": "L1"}], "bogus": 1})
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "bogus")
        status, body, _ = self._request(
            {"device_id": "ghost", "items": [{"lease_id": "L1"}]})
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "device_id")
        status, body, _ = self._request(
            {"device_id": "bob", "items": [{"lease_id": "ghost"}]})
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "items[0].lease_id")
        status, body, _ = self._request(
            {"device_id": "bob", "items": [{"lease_id": "L1"}]})
        self.assertEqual(status, 201)
        # Partial replay conflicts at the first replayed item's lease_id
        # and leaves the second item's lease untouched.
        status, body, _ = self._request(
            {"device_id": "bob", "items": [{"lease_id": "L1"},
                                           {"lease_id": "L2"}]})
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "items[0].lease_id")
        self.assertEqual(list(body), ["message", "field"])
        # The fresh release of L2 was not written: L2 is still active.
        self.assertEqual(
            self.service.inbox_lease_get("bob", "L2")["state"], "active")


if __name__ == "__main__":
    unittest.main()
