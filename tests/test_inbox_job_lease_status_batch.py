"""Tests for ``POST /v1/inbox-jobs/lease-status-batch``.

A read-only batch query of 1:1-inbox redelivery lease states: the body
carries a non-empty string ``device_id`` and a non-empty ``items`` array of
objects carrying exactly a non-empty string ``lease_id`` (no repeats across
items), and no other top-level key. The device gate (unknown/revoked ->
409/device_id) runs first; items are then prechecked in array order with
the single-lease lookup rules (unknown lease 404/items[i].lease_id;
cross-device lease 409/items[i].lease_id) and the first error aborts the
whole query. The success body keys are ``device_id`` then ``results``;
results keep input order and each item is ``lease_id``, ``state``,
``leased_until``, ``released_at``, ``completion`` and ``message_count``.
Every lease is judged at the same locked instant — completion wins over
release, which wins over expiry; the deadline is the last renewal's value.
The query writes nothing, never persists and advances no ``commit_seq``.
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

    def _renew(self, lease_id, renewal_id, device_id="bob"):
        return self.service.inbox_lease_renew(
            device_id, lease_id, {"renewal_id": renewal_id})

    def _release(self, lease_id, device_id="bob"):
        return self.service.inbox_release(device_id, lease_id)

    def _complete(self, lease_id, completion_id, outcome="delivered",
                  device_id="bob"):
        return self.service.inbox_lease_complete(
            device_id, lease_id,
            {"completion_id": completion_id, "outcome": outcome})

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
        error = self._error(lambda: self._batch(items=[item], op="status"))
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
        self.service.store.revoke_device("bob")
        error = self._error(lambda: self._batch(items=items))
        self.assertEqual((error.status_code, error.field),
                         (409, "device_id"))

    def test_active_lease_status_shape_and_order(self) -> None:
        self._two_leases()
        body = self._batch(items=[{"lease_id": "L1"},
                                  {"lease_id": "L2"}])
        self.assertEqual(list(body), ["device_id", "results"])
        self.assertEqual(body["device_id"], "bob")
        self.assertEqual(len(body["results"]), 2)
        for item in body["results"]:
            self.assertEqual(list(item), [
                "lease_id", "state", "leased_until", "released_at",
                "completion", "message_count"])
            self.assertEqual(item["state"], "active")
            self.assertIsInstance(item["leased_until"], str)
            self.assertIsNone(item["released_at"])
            self.assertIsNone(item["completion"])
            self.assertIsInstance(item["message_count"], int)
            self.assertGreaterEqual(item["message_count"], 0)
        self.assertEqual(body["results"][0]["lease_id"], "L1")
        self.assertEqual(body["results"][0]["message_count"], 2)
        self.assertEqual(body["results"][1]["lease_id"], "L2")
        self.assertEqual(body["results"][1]["message_count"], 3)

    def test_results_keep_input_order(self) -> None:
        self._two_leases()
        body = self._batch(items=[{"lease_id": "L2"},
                                  {"lease_id": "L1"}])
        self.assertEqual([r["lease_id"] for r in body["results"]],
                         ["L2", "L1"])

    def test_expired_lease_state_and_renewed_deadline(self) -> None:
        claim = self._claim("L1")
        claim_deadline = claim["leased_until"]
        # Expired without renewal.
        self._expire_claim("L1")
        body = self._batch(items=[{"lease_id": "L1"}])
        self.assertEqual(body["results"][0]["state"], "expired")
        self.assertEqual(body["results"][0]["leased_until"], _PAST)
        # A renewal (with a fresh deadline) makes it active again and the
        # reported deadline is the last renewal's value.
        with self.service.store._lock:
            for state in self.service.store._delivery.values():
                for lease in state.leases:
                    if lease.lease_id == "L1":
                        lease.leased_until = claim_deadline
        renewed, status = self._renew("L1", "r1")
        self.assertEqual(status, 201)
        body = self._batch(items=[{"lease_id": "L1"}])
        item = body["results"][0]
        self.assertEqual(item["state"], "active")
        self.assertEqual(item["leased_until"], renewed["leased_until"])

    def test_released_state_precedence(self) -> None:
        self._claim("L1")
        released, status = self._release("L1")
        self.assertEqual(status, 201)
        body = self._batch(items=[{"lease_id": "L1"}])
        item = body["results"][0]
        self.assertEqual(item["state"], "released")
        self.assertEqual(item["released_at"], released["released_at"])
        self.assertIsNone(item["completion"])
        # A released lease stays released even once its deadline passes.
        self._expire_claim("L1")
        body = self._batch(items=[{"lease_id": "L1"}])
        self.assertEqual(body["results"][0]["state"], "released")

    def test_completion_wins_over_release_and_expiry(self) -> None:
        self._claim("L1")
        completed, status = self._complete("L1", "c1", "delivered")
        self.assertEqual(status, 201)
        self._expire_claim("L1")
        body = self._batch(items=[{"lease_id": "L1"}])
        item = body["results"][0]
        self.assertEqual(item["state"], "completed")
        self.assertEqual(item["completion"], {
            "completion_id": "c1",
            "outcome": "delivered",
            "completed_at": completed["completed_at"]})
        self.assertEqual(list(item["completion"]),
                         ["completion_id", "outcome", "completed_at"])
        self.assertEqual(item["message_count"], 5)

    def test_failed_completion_state(self) -> None:
        self._claim("L1")
        self._complete("L1", "c1", "failed")
        body = self._batch(items=[{"lease_id": "L1"}])
        item = body["results"][0]
        self.assertEqual(item["state"], "completed")
        self.assertEqual(item["completion"]["outcome"], "failed")

    def test_same_batch_same_instant_for_all_items(self) -> None:
        # Both leases judged together at one instant share one decision.
        self._two_leases()
        near_future = "2099-01-01T00:00:00.000000+00:00"
        with self.service.store._lock:
            for state in self.service.store._delivery.values():
                for lease in state.leases:
                    if lease.lease_id in ("L1", "L2"):
                        lease.leased_until = near_future
        body = self._batch(items=[{"lease_id": "L1"},
                                  {"lease_id": "L2"}])
        self.assertEqual([r["state"] for r in body["results"]],
                         ["active", "active"])
        # The same two leases, moved to the past, are both expired.
        with self.service.store._lock:
            for state in self.service.store._delivery.values():
                for lease in state.leases:
                    if lease.lease_id in ("L1", "L2"):
                        lease.leased_until = _PAST
        body = self._batch(items=[{"lease_id": "L1"},
                                  {"lease_id": "L2"}])
        self.assertEqual([r["state"] for r in body["results"]],
                         ["expired", "expired"])

    def test_per_item_errors_use_indexed_fields_in_array_order(self) -> None:
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
        # Cross-device at index 1 after a good item 0.
        self._claim("lb", device_id="bob2", limit=10)
        error = self._error(lambda: self._batch(items=[
            {"lease_id": "L1"}, {"lease_id": "lb"}]))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[1].lease_id"))

    def test_repeated_identical_query_is_byte_stable(self) -> None:
        self._two_leases()
        payload = {"device_id": "bob",
                   "items": [{"lease_id": "L1"}, {"lease_id": "L2"}]}
        first = self.service.inbox_job_lease_status_batch(payload)
        second = self.service.inbox_job_lease_status_batch(payload)
        self.assertEqual(first, second)

    def test_query_is_read_only(self) -> None:
        self._two_leases()
        snapshot = [
            (l.lease_id, l.leased_until, l.released_at, l.completion,
             list(l.renewals))
            for state in self.service.store._delivery.values()
            for l in state.leases]
        self._batch(items=[{"lease_id": "L1"}, {"lease_id": "L2"}])
        self.assertEqual(snapshot, [
            (l.lease_id, l.leased_until, l.released_at, l.completion,
             list(l.renewals))
            for state in self.service.store._delivery.values()
            for l in state.leases])


class InboxJobLeaseStatusBatchPersistenceTest(_BatchMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self._build()
        self.path = os.path.join(self.directory, "state.json")
        self.state_store = attach_persistence(self.service, self.path)

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_query_consumes_no_generation(self) -> None:
        self._two_leases()
        generation = self.state_store.commit_seq
        body = self._batch(items=[{"lease_id": "L1"},
                                  {"lease_id": "L2"}])
        self.assertEqual(body["device_id"], "bob")
        self.assertEqual(self.state_store.commit_seq, generation)
        # Repeated queries still consume no generation.
        self._batch(items=[{"lease_id": "L1"}])
        self.assertEqual(self.state_store.commit_seq, generation)
        # A failing query consumes no generation either.
        with self.assertRaises(ServiceError):
            self._batch(items=[{"lease_id": "ghost"}])
        self.assertEqual(self.state_store.commit_seq, generation)

    def test_query_does_not_modify_the_state_file(self) -> None:
        self._two_leases()
        with open(self.path, "rb") as handle:
            before = handle.read()
        self._batch(items=[{"lease_id": "L1"}, {"lease_id": "L2"}])
        with open(self.path, "rb") as handle:
            after = handle.read()
        self.assertEqual(before, after)

    def test_restart_yields_the_same_results(self) -> None:
        self._claim("L1")
        self._renew("L1", "r1")
        self._release("L1")
        self._claim("L2", limit=100)
        self._complete("L2", "c2", "failed")
        payload = {"device_id": "bob",
                   "items": [{"lease_id": "L2"}, {"lease_id": "L1"}]}
        before = self.service.inbox_job_lease_status_batch(payload)
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        after = restarted.inbox_job_lease_status_batch(payload)
        self.assertEqual(after, before)
        self.assertTrue(restarted.persistence_integrity()["consistent"])
        self.assertEqual([r["state"] for r in after["results"]],
                         ["completed", "released"])

    def test_revoked_device_rejected_but_state_file_intact(self) -> None:
        self._two_leases()
        self.service.store.revoke_device("bob")
        with self.assertRaises(ServiceError) as caught:
            self._batch(items=[{"lease_id": "L1"}])
        self.assertEqual((caught.exception.status_code,
                          caught.exception.field), (409, "device_id"))


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
        conn.request("POST", "/v1/inbox-jobs/lease-status-batch", body=body,
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
        for result in body["results"]:
            self.assertEqual(result["state"], "active")
            self.assertLess(raw.index('"lease_id"'),
                            raw.index('"state"'))
            self.assertLess(raw.index('"state"'),
                            raw.index('"leased_until"'))
            self.assertLess(raw.index('"leased_until"'),
                            raw.index('"released_at"'))
            self.assertLess(raw.index('"released_at"'),
                            raw.index('"completion"'))
            self.assertLess(raw.index('"completion"'),
                            raw.index('"message_count"'))
        # A repeated query is byte-identical (purely read-only).
        status, _body, raw2 = self._request(
            {"device_id": "bob", "items": [{"lease_id": "L1"},
                                           {"lease_id": "L2"}]})
        self.assertEqual(status, 200)
        self.assertEqual(raw2, raw)

    def test_completed_lease_shape_over_http(self) -> None:
        self.service.inbox_lease_complete(
            "bob", "L1", {"completion_id": "c1", "outcome": "delivered"})
        status, body, _ = self._request(
            {"device_id": "bob", "items": [{"lease_id": "L1"}]})
        self.assertEqual(status, 200)
        item = body["results"][0]
        self.assertEqual(item["state"], "completed")
        self.assertEqual(list(item["completion"]),
                         ["completion_id", "outcome", "completed_at"])
        self.assertEqual(item["completion"]["outcome"], "delivered")
        self.assertEqual(item["message_count"], 2)

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
            {"device_id": "bob", "items": [4]})
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "items[0]")
        status, body, _ = self._request(
            {"device_id": "bob",
             "items": [{"lease_id": "L1", "x": 1}]})
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "items[0]")
        status, body, _ = self._request(
            {"device_id": "bob", "items": [{"lease_id": "L1"}],
             "bogus": 1})
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "bogus")
        status, body, _ = self._request(
            {"device_id": "bob",
             "items": [{"lease_id": "L1"}, {"lease_id": "L1"}]})
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "items[1]")
        status, body, _ = self._request(
            {"device_id": "ghost", "items": [{"lease_id": "L1"}]})
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "device_id")
        status, body, _ = self._request(
            {"device_id": "bob", "items": [{"lease_id": "ghost"}]})
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "items[0].lease_id")
        # Cross-device: bob2's claim on the o1 session.
        self.service.inbox_claim(
            "bob2", {"lease_id": "lb", "limit": 10})
        status, body, _ = self._request(
            {"device_id": "bob", "items": [{"lease_id": "lb"}]})
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "items[0].lease_id")
        self.assertEqual(list(body), ["message", "field"])


if __name__ == "__main__":
    unittest.main()
