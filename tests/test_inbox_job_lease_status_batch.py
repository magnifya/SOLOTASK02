"""Tests for ``POST /v1/inbox-jobs/lease-status-batch``.

A read-only batch query of 1:1-inbox lease states: the body carries
exactly a non-empty string ``device_id`` and a non-empty ``items`` array
of objects carrying exactly a non-empty string ``lease_id`` (no
``lease_id`` repeats across items). The device gate (unknown/revoked ->
409/device_id) runs first; items are then prechecked in array order
(unknown lease 404/items[i].lease_id; cross-device lease
409/items[i].lease_id) and the first error aborts the whole batch. On
success the answer is always 200 with keys ``device_id`` then
``results``; each result is ``lease_id``, ``state``, ``leased_until``,
``released_at``, ``completion``, ``message_count`` in that order, every
state decided against one uniform batch instant (completion wins, then
release, then expiry). The query shares the store lock with the
mutating operations but writes nothing, advances no ``commit_seq`` and
changes no state.
"""
import json
import os
import shutil
import tempfile
import threading
import unittest
from http.client import HTTPConnection

from tests.test_device_inbox import InboxMixin
from e2ee_backend.persistence import attach_persistence
from e2ee_backend.service import DeviceService, ServiceError
from e2ee_backend.http_app import create_server

_PAST = "2000-01-01T00:00:00.000000+00:00"


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
        return self.service.inbox_job_lease_status_batch(payload)

    def _two_leases(self):
        """L1 claims a1/a2 (limit 2); L2 claims the remaining three."""
        first = self._claim("L1", limit=2)
        second = self._claim("L2", limit=100)
        return first, second


class InboxJobLeaseStatusBatchServiceTest(_BatchMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def test_bad_body_and_device_id_errors(self) -> None:
        good = [{"lease_id": "L1"}]
        for payload in (None, [], "x", 3, True):
            error = self._error(
                lambda payload=payload:
                self.service.inbox_job_lease_status_batch(payload))
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
                self.service.inbox_job_lease_status_batch(payload))
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
            [{"lease_id": "L1", "state": "active"}],
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
        # Unlike the single-lease GET, the batch query rejects a revoked
        # device (409/device_id) ahead of every item check.
        self.service.store.revoke_device("bob")
        error = self._error(lambda: self._batch(items=items))
        self.assertEqual((error.status_code, error.field),
                         (409, "device_id"))
        error = self._error(lambda: self._batch(items=[{"lease_id": "L1"}]))
        self.assertEqual((error.status_code, error.field),
                         (409, "device_id"))

    def test_active_leases_200_shape_and_input_order(self) -> None:
        first, second = self._two_leases()
        body, status = self._batch(items=[{"lease_id": "L2"},
                                          {"lease_id": "L1"}])
        self.assertEqual(status, 200)
        self.assertEqual(list(body), ["device_id", "results"])
        self.assertEqual(body["device_id"], "bob")
        self.assertEqual(len(body["results"]), 2)
        # Results keep the input order, not the claim order.
        self.assertEqual([r["lease_id"] for r in body["results"]],
                         ["L2", "L1"])
        for item in body["results"]:
            self.assertEqual(list(item),
                             ["lease_id", "state", "leased_until",
                              "released_at", "completion", "message_count"])
            self.assertEqual(item["state"], "active")
            self.assertIsNone(item["released_at"])
            self.assertIsNone(item["completion"])
        self.assertEqual(body["results"][0]["leased_until"],
                         second["leased_until"])
        self.assertEqual(body["results"][0]["message_count"], 3)
        self.assertEqual(body["results"][1]["leased_until"],
                         first["leased_until"])
        self.assertEqual(body["results"][1]["message_count"], 2)

    def test_renewal_moves_leased_until_to_the_last_renewal(self) -> None:
        self._claim("L1", limit=2)
        renewed, status = self.service.inbox_lease_renew(
            "bob", "L1", {"renewal_id": "R1"})
        self.assertEqual(status, 201)
        renewed2, _ = self.service.inbox_lease_renew(
            "bob", "L1", {"renewal_id": "R2"})
        body, status = self._batch(items=[{"lease_id": "L1"}])
        self.assertEqual(status, 200)
        item = body["results"][0]
        self.assertEqual(item["state"], "active")
        self.assertEqual(item["leased_until"], renewed2["leased_until"])
        self.assertNotEqual(item["leased_until"], renewed["leased_until"])

    def test_released_lease_reports_released_with_stamp(self) -> None:
        self._claim("L1", limit=2)
        released, status = self.service.inbox_release("bob", "L1")
        self.assertEqual(status, 201)
        body, status = self._batch(items=[{"lease_id": "L1"}])
        self.assertEqual(status, 200)
        item = body["results"][0]
        self.assertEqual(item["state"], "released")
        self.assertEqual(item["released_at"], released["released_at"])
        self.assertIsNone(item["completion"])
        self.assertEqual(item["message_count"], 2)

    def test_completed_lease_reports_completion_object(self) -> None:
        self._claim("L1", limit=2)
        completed, status = self.service.inbox_lease_complete(
            "bob", "L1", {"completion_id": "C1", "outcome": "failed"})
        self.assertEqual(status, 201)
        body, status = self._batch(items=[{"lease_id": "L1"}])
        self.assertEqual(status, 200)
        item = body["results"][0]
        self.assertEqual(item["state"], "completed")
        self.assertIsNone(item["released_at"])
        self.assertEqual(item["completion"], {
            "completion_id": "C1", "outcome": "failed",
            "completed_at": completed["completed_at"]})
        self.assertEqual(list(item["completion"]),
                         ["completion_id", "outcome", "completed_at"])

    def test_expired_lease_reports_expired(self) -> None:
        self._claim("L1", limit=1)
        self._expire_claim("L1")
        body, status = self._batch(items=[{"lease_id": "L1"}])
        self.assertEqual(status, 200)
        item = body["results"][0]
        self.assertEqual(item["state"], "expired")
        self.assertEqual(item["leased_until"], _PAST)

    def test_mixed_states_in_one_batch(self) -> None:
        self._two_leases()
        # Releasing L1 makes its two messages claimable again by L3.
        self.service.inbox_release("bob", "L1")
        self._claim("L3", limit=100)
        self._expire_claim("L2")
        body, status = self._batch(items=[{"lease_id": "L1"},
                                          {"lease_id": "L2"},
                                          {"lease_id": "L3"}])
        self.assertEqual(status, 200)
        states = [r["state"] for r in body["results"]]
        self.assertEqual(states, ["released", "expired", "active"])
        self.assertEqual([r["message_count"] for r in body["results"]],
                         [2, 3, 2])

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

    def test_first_error_in_array_order_wins(self) -> None:
        self._two_leases()
        error = self._error(lambda: self._batch(items=[
            {"lease_id": "L1"}, {"lease_id": "ghost"}]))
        self.assertEqual((error.status_code, error.field),
                         (404, "items[1].lease_id"))
        error = self._error(lambda: self._batch(items=[
            {"lease_id": "ghost"}, {"lease_id": "L2"}]))
        self.assertEqual((error.status_code, error.field),
                         (404, "items[0].lease_id"))

    def test_query_is_read_only_and_repeatable(self) -> None:
        self._two_leases()
        store = self.service.store
        with store._lock:
            before = {key: (value.attempts, value.acked,
                            [(lease.leased_until, lease.released_at)
                             for lease in value.leases])
                      for key, value in store._delivery.items()}
        items = [{"lease_id": "L1"}, {"lease_id": "L2"}]
        first, status = self._batch(items=items)
        self.assertEqual(status, 200)
        second, status = self._batch(items=items)
        self.assertEqual(status, 200)
        self.assertEqual(first, second)
        with store._lock:
            after = {key: (value.attempts, value.acked,
                           [(lease.leased_until, lease.released_at)
                            for lease in value.leases])
                     for key, value in store._delivery.items()}
        self.assertEqual(before, after)


class InboxJobLeaseStatusBatchPersistenceTest(_BatchMixin,
                                              unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self._build()
        self.path = os.path.join(self.directory, "state.json")
        self.state_store = attach_persistence(self.service, self.path)

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_query_advances_no_generation_on_success_or_failure(
            self) -> None:
        self._two_leases()
        before = self.state_store.commit_seq
        body, status = self._batch(items=[{"lease_id": "L1"},
                                          {"lease_id": "L2"}])
        self.assertEqual(status, 200)
        self.assertEqual(self.state_store.commit_seq, before)
        with self.assertRaises(ServiceError):
            self._batch(items=[{"lease_id": "ghost"}])
        self.assertEqual(self.state_store.commit_seq, before)
        # The state file was not rewritten by the read-only query.
        with open(self.path, encoding="utf-8") as handle:
            document = json.load(handle)
        self.assertIn("delivery", document)

    def test_restart_yields_the_same_result(self) -> None:
        self._two_leases()
        self.service.inbox_release("bob", "L1")
        items = [{"lease_id": "L1"}, {"lease_id": "L2"}]
        first, status = self._batch(items=items)
        self.assertEqual(status, 200)
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        replay, status = restarted.inbox_job_lease_status_batch(
            {"device_id": "bob", "items": items})
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)
        self.assertTrue(restarted.persistence_integrity()["consistent"])


class InboxJobLeaseStatusBatchHTTPTest(_BatchMixin, unittest.TestCase):
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
        conn.request("POST", "/v1/inbox-jobs/lease-status-batch",
                     body=body,
                     headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        data = response.read().decode("utf-8")
        conn.close()
        return response.status, (json.loads(data) if data else None), data

    def test_status_batch_over_http(self) -> None:
        status, body, raw = self._request(
            {"device_id": "bob", "items": [{"lease_id": "L1"},
                                           {"lease_id": "L2"}]})
        self.assertEqual(status, 200)
        self.assertEqual(list(body), ["device_id", "results"])
        self.assertEqual([list(r) for r in body["results"]],
                         [["lease_id", "state", "leased_until",
                           "released_at", "completion", "message_count"],
                          ["lease_id", "state", "leased_until",
                           "released_at", "completion", "message_count"]])
        self.assertLess(raw.index('"device_id"'), raw.index('"results"'))
        self.assertLess(raw.index('"lease_id"'), raw.index('"state"'))
        self.assertLess(raw.index('"state"'), raw.index('"leased_until"'))
        self.assertLess(raw.index('"leased_until"'),
                        raw.index('"released_at"'))
        self.assertLess(raw.index('"released_at"'),
                        raw.index('"completion"'))
        self.assertLess(raw.index('"completion"'),
                        raw.index('"message_count"'))
        # A repeated query answers byte-identically.
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
            {"device_id": "bob", "items": [{"lease_id": "L1"},
                                           {"lease_id": "L1"}]})
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "items[1]")
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
        self.assertEqual(list(body), ["message", "field"])
        # A lease owned by another device conflicts at the item field.
        status, body, _ = self._request(
            {"device_id": "bob2", "items": [{"lease_id": "L1"}]})
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "items[0].lease_id")

    def test_revoked_device_is_409_over_http(self) -> None:
        self.service.store.revoke_device("bob")
        status, body, _ = self._request(
            {"device_id": "bob", "items": [{"lease_id": "L1"}]})
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "device_id")


if __name__ == "__main__":
    unittest.main()
