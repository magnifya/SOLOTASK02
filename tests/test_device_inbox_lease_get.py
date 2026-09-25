"""Tests for the 1:1 inbox lease query endpoint.

GET /v1/devices/{device_id}/inbox/leases/{lease_id} returns one occupied
lease's current state. It takes no request body (non-empty -> 400/
request_body) and no query parameters (any -> 400/query). An unknown
lease id is 404/lease_id and one owned by another device is 409/lease_id,
both ahead of the path device's state; a matching lease stays readable
after its device is revoked. The 200 body keys are device_id, lease_id,
limit, state, leased_until, released_at, completion, messages; the query
is purely read-only (no write, no commit_seq change) and linearized under
the store lock with the mutating lease operations.
"""
import json
import threading
import unittest
from http.client import HTTPConnection

from e2ee_backend.models import Device, SignedPreKey
from e2ee_backend.service import DeviceService, ServiceError
from e2ee_backend.http_app import create_server


class GetMixin:
    def _build(self) -> None:
        self.service = DeviceService()
        self.service.store.add_device(Device("u", "alice", "ik"))
        self.service.store.add_device(Device(
            "u", "bob", "ik",
            prekeys=[SignedPreKey("pk1", "pubk1"),
                     SignedPreKey("pk2", "pubk2")]))
        self.service.store.add_device(
            Device("u", "carol", "ik",
                   prekeys=[SignedPreKey("pkC", "pubkC")]))
        # Session alice -> bob with three messages; another carol -> bob.
        self.sid1 = self.service.store.create_session(
            "alice", "bob", "pk1", "ek1").session_id
        self.sid2 = self.service.store.create_session(
            "carol", "bob", "pk2", "ek2").session_id
        for sequence in range(1, 4):
            self.service.post_message({
                "session_id": self.sid1, "sender_device_id": "alice",
                "message_id": f"a{sequence}", "sequence": sequence,
                "nonce": f"na{sequence}", "ciphertext": "ct"})
        for sequence in range(1, 3):
            self.service.post_message({
                "session_id": self.sid2, "sender_device_id": "carol",
                "message_id": f"b{sequence}", "sequence": sequence,
                "nonce": f"nb{sequence}", "ciphertext": "ct"})

    def _claim(self, lease_id="L1", limit=2):
        body, status = self.service.inbox_claim(
            "bob", {"lease_id": lease_id, "limit": limit})
        self.assertEqual(status, 201)
        return body


class GetServiceTest(GetMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def test_active_lease_200_shape(self) -> None:
        claim = self._claim("L1", 2)
        body = self.service.inbox_lease_get("bob", "L1")
        self.assertEqual(list(body),
                         ["device_id", "lease_id", "limit", "state",
                          "leased_until", "released_at", "completion",
                          "messages"])
        self.assertEqual(body["device_id"], "bob")
        self.assertEqual(body["lease_id"], "L1")
        self.assertEqual(body["limit"], 2)
        self.assertEqual(body["state"], "active")
        self.assertEqual(body["leased_until"], claim["leased_until"])
        self.assertIsNone(body["released_at"])
        self.assertIsNone(body["completion"])

    def test_messages_in_claim_order_with_current_delivery_values(self) -> None:
        claim = self._claim("L1", 3)
        claimed_ids = [m["message_id"] for m in claim["messages"]]
        body = self.service.inbox_lease_get("bob", "L1")
        self.assertEqual(len(body["messages"]), 3)
        self.assertEqual([m["message_id"] for m in body["messages"]],
                         claimed_ids)
        for message in body["messages"]:
            self.assertEqual(
                list(message),
                ["session_id", "message_id", "sequence", "acked",
                 "attempts"])
            self.assertFalse(message["acked"])
            self.assertEqual(message["attempts"], 0)

    def test_ack_marks_item_without_removing_it(self) -> None:
        self._claim("L1", 5)
        self.service.sync_session_ack(self.sid1, {
            "device_id": "bob", "cursor": 2})
        body = self.service.inbox_lease_get("bob", "L1")
        self.assertEqual(len(body["messages"]), 5)
        by_id = {m["message_id"]: m for m in body["messages"]}
        self.assertTrue(by_id["a1"]["acked"])
        self.assertTrue(by_id["a2"]["acked"])
        self.assertFalse(by_id["a3"]["acked"])
        self.assertFalse(by_id["b1"]["acked"])

    def test_attempts_reflect_retry_batches(self) -> None:
        self._claim("L1", 3)
        self.service.inbox_retry_batch("bob", {
            "attempt_id": "t1",
            "items": [{"session_id": self.sid1, "message_id": "a1"}]})
        body = self.service.inbox_lease_get("bob", "L1")
        by_id = {m["message_id"]: m for m in body["messages"]}
        self.assertEqual(by_id["a1"]["attempts"], 1)
        self.assertEqual(by_id["a2"]["attempts"], 0)
        # A replayed attempt id does not move the counter.
        self.service.inbox_retry_batch("bob", {
            "attempt_id": "t1",
            "items": [{"session_id": self.sid1, "message_id": "a1"}]})
        body = self.service.inbox_lease_get("bob", "L1")
        by_id = {m["message_id"]: m for m in body["messages"]}
        self.assertEqual(by_id["a1"]["attempts"], 1)

    def test_renew_updates_effective_deadline(self) -> None:
        self._claim("L1", 1)
        renewed, status = self.service.inbox_lease_renew(
            "bob", "L1", {"renewal_id": "R1"})
        self.assertEqual(status, 201)
        body = self.service.inbox_lease_get("bob", "L1")
        self.assertEqual(body["state"], "active")
        self.assertEqual(body["leased_until"], renewed["leased_until"])
        renewed2, _ = self.service.inbox_lease_renew(
            "bob", "L1", {"renewal_id": "R2"})
        body = self.service.inbox_lease_get("bob", "L1")
        self.assertEqual(body["leased_until"], renewed2["leased_until"])

    def test_released_state(self) -> None:
        self._claim("L1", 2)
        released, status = self.service.inbox_release("bob", "L1")
        self.assertEqual(status, 201)
        body = self.service.inbox_lease_get("bob", "L1")
        self.assertEqual(body["state"], "released")
        self.assertEqual(body["released_at"], released["released_at"])
        self.assertIsNone(body["completion"])
        self.assertEqual(len(body["messages"]), 2)

    def test_completed_state_wins_over_release(self) -> None:
        self._claim("L1", 2)
        completed, status = self.service.inbox_lease_complete(
            "bob", "L1", {"completion_id": "C1", "outcome": "failed"})
        self.assertEqual(status, 201)
        body = self.service.inbox_lease_get("bob", "L1")
        self.assertEqual(body["state"], "completed")
        self.assertIsNone(body["released_at"])
        self.assertEqual(body["completion"], {
            "completion_id": "C1", "outcome": "failed",
            "completed_at": completed["completed_at"]})
        self.assertEqual(list(body["completion"]),
                         ["completion_id", "outcome", "completed_at"])

    def test_expired_state_after_deadline_passes(self) -> None:
        claim = self._claim("L1", 1)
        # Rewrite the claim deadline into the past on every copy.
        past = "2000-01-01T00:00:00.000000+00:00"
        for state in self.service.store._delivery.values():
            for lease in state.leases:
                if lease.lease_id == "L1":
                    lease.leased_until = past
        body = self.service.inbox_lease_get("bob", "L1")
        self.assertEqual(body["state"], "expired")
        self.assertEqual(body["leased_until"], past)

    def test_unknown_lease_is_404_lease_id(self) -> None:
        with self.assertRaises(ServiceError) as caught:
            self.service.inbox_lease_get("bob", "NOPE")
        self.assertEqual(caught.exception.status_code, 404)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_unknown_lease_404_precedes_unknown_device(self) -> None:
        with self.assertRaises(ServiceError) as caught:
            self.service.inbox_lease_get("ghost", "NOPE")
        self.assertEqual(caught.exception.status_code, 404)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_cross_device_lookup_is_409_lease_id(self) -> None:
        self._claim("L1", 2)
        with self.assertRaises(ServiceError) as caught:
            self.service.inbox_lease_get("carol", "L1")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_matching_lease_readable_after_device_revoked(self) -> None:
        self._claim("L1", 2)
        self.service.revoke_device("bob")
        body = self.service.inbox_lease_get("bob", "L1")
        self.assertEqual(body["state"], "active")
        self.assertEqual(body["lease_id"], "L1")

    def test_query_is_read_only(self) -> None:
        self._claim("L1", 2)
        store = self.service.store
        with store._lock:
            before = {key: (value.attempts, value.acked,
                            [lease.leased_until for lease in value.leases])
                      for key, value in store._delivery.items()}
        first = self.service.inbox_lease_get("bob", "L1")
        second = self.service.inbox_lease_get("bob", "L1")
        self.assertEqual(first, second)
        with store._lock:
            after = {key: (value.attempts, value.acked,
                           [lease.leased_until for lease in value.leases])
                     for key, value in store._delivery.items()}
        self.assertEqual(before, after)


class GetHTTPTest(GetMixin, unittest.TestCase):
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

    def _request(self, path, raw=None, method="GET"):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        headers = {"Content-Type": "application/json"} if raw else {}
        conn.request(method, path, body=raw, headers=headers)
        response = conn.getresponse()
        data = response.read().decode("utf-8")
        conn.close()
        return response.status, (json.loads(data) if data else None), data

    def _claim_http(self, lease_id="L1", limit=2):
        status, _, _ = self._request(
            "/v1/devices/bob/inbox/claim",
            json.dumps({"lease_id": lease_id, "limit": limit}),
            method="POST")
        self.assertEqual(status, 201)

    def test_get_active_lease_over_http(self) -> None:
        self._claim_http("L1", 2)
        status, body, raw = self._request(
            "/v1/devices/bob/inbox/leases/L1")
        self.assertEqual(status, 200)
        self.assertEqual(list(body),
                         ["device_id", "lease_id", "limit", "state",
                          "leased_until", "released_at", "completion",
                          "messages"])
        self.assertLess(raw.index('"device_id"'), raw.index('"lease_id"'))
        self.assertLess(raw.index('"leased_until"'),
                        raw.index('"released_at"'))
        self.assertLess(raw.index('"completion"'),
                        raw.index('"messages"'))
        self.assertEqual(body["state"], "active")
        self.assertIsNone(body["released_at"])
        self.assertIsNone(body["completion"])

    def test_repeated_gets_are_byte_identical(self) -> None:
        self._claim_http("L1", 2)
        _, _, first = self._request("/v1/devices/bob/inbox/leases/L1")
        _, _, second = self._request("/v1/devices/bob/inbox/leases/L1")
        self.assertEqual(first, second)

    def test_query_parameter_is_400_query(self) -> None:
        self._claim_http("L1", 2)
        for path in ("/v1/devices/bob/inbox/leases/L1?x=1",
                     "/v1/devices/bob/inbox/leases/L1?foo",
                     "/v1/devices/bob/inbox/leases/L1?limit=1"):
            with self.subTest(path=path):
                status, body, _ = self._request(path)
                self.assertEqual(status, 400)
                self.assertEqual(list(body), ["message", "field"])
                self.assertEqual(body["field"], "query")

    def test_empty_query_string_is_allowed(self) -> None:
        self._claim_http("L1", 2)
        status, _, _ = self._request(
            "/v1/devices/bob/inbox/leases/L1?")
        self.assertEqual(status, 200)

    def test_nonempty_body_is_400_request_body(self) -> None:
        self._claim_http("L1", 2)
        status, body, _ = self._request(
            "/v1/devices/bob/inbox/leases/L1", raw="{}")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "request_body")

    def test_errors_over_http(self) -> None:
        self._claim_http("L1", 2)
        status, body, _ = self._request(
            "/v1/devices/bob/inbox/leases/NOPE")
        self.assertEqual((status, body["field"]), (404, "lease_id"))
        status, body, _ = self._request(
            "/v1/devices/carol/inbox/leases/L1")
        self.assertEqual((status, body["field"]), (409, "lease_id"))

    def test_released_lease_is_200_released_over_http(self) -> None:
        self._claim_http("L1", 2)
        status, _, _ = self._request(
            "/v1/devices/bob/inbox/leases/L1/release", method="POST")
        self.assertEqual(status, 201)
        status, body, _ = self._request(
            "/v1/devices/bob/inbox/leases/L1")
        self.assertEqual(status, 200)
        self.assertEqual(body["state"], "released")

    def test_get_sub_paths_are_not_the_lease_route(self) -> None:
        self._claim_http("L1", 2)
        # GET .../release is not a thing (release is POST-only).
        status, _, _ = self._request(
            "/v1/devices/bob/inbox/leases/L1/release")
        self.assertEqual(status, 404)

    def test_percent_encoded_lease_id_over_http(self) -> None:
        self._claim_http("a/b", 1)
        status, body, _ = self._request(
            "/v1/devices/bob/inbox/leases/a%2Fb")
        self.assertEqual(status, 200)
        self.assertEqual(body["lease_id"], "a/b")


if __name__ == "__main__":
    unittest.main()
