"""Tests for the group-inbox single lease query endpoint.

GET /v1/devices/{device_id}/group-inbox/leases/{lease_id} observes one
occupied group lease and its covered messages. It takes no request body
(non-empty -> 400/request_body) and no query parameters (any ->
400/query). Both path segments are strictly percent-decoded as UTF-8,
device_id first and lease_id second; a deeper path is 404/lease_id. An
unknown lease id is 404/lease_id and one owned by another device (or
committed in the 1:1 namespace) is 409/lease_id, both ahead of the
path device's state; a matching lease stays readable after its device
is revoked. The 200 body keys are device_id, lease_id, limit, state,
leased_until, released_at, completion, messages; the query is purely
read-only (no
write, no commit_seq change) and linearized under the store lock with
the mutating group lease operations.
"""
import json
import threading
import unittest
from datetime import datetime, timezone
from http.client import HTTPConnection

from e2ee_backend.models import Device, MessageLeaseRenewal, SignedPreKey
from e2ee_backend.service import DeviceService, ServiceError
from e2ee_backend.http_app import create_server


def _message(session_id, message_id, sequence, sender):
    return {
        "session_id": session_id,
        "sender_device_id": sender,
        "message_id": message_id,
        "sequence": sequence,
        "nonce": f"nonce-{message_id}",
        "ciphertext": "ciphertext",
    }


class GetMixin:
    def _build(self) -> None:
        self.service = DeviceService()
        self.store = self.service.store
        self.store.add_device(Device("u", "d1", "ik"))
        self.store.add_device(Device("u", "d3", "ik"))
        self.store.add_device(Device("u", "d4", "ik"))
        self.store.add_device(Device(
            "u", "d2", "ik",
            prekeys=[SignedPreKey("pk1", "pubk1")]))
        # d1 creates g1(d1,d2,d3); gs1 freezes that roster.
        self.service.create_group({
            "group_id": "g1", "creator_device_id": "d1",
            "member_device_ids": ["d2", "d3"]})
        self.gs1 = self.service.create_group_session({
            "group_id": "g1", "initiator_device_id": "d1",
            "ephemeral_key": "epk1"})["session_id"]
        # m1/m2 from d1, m3 from d2: d2 may claim m1/m2 only, d3 all 3.
        for message_id, sequence, sender in (
                ("m1", 1, "d1"), ("m2", 2, "d1"), ("m3", 3, "d2")):
            self.service.post_message(
                _message(self.gs1, message_id, sequence, sender))
        # A 1:1 session d1 -> d2 never contributes to a group lease.
        self.one2one = self.store.create_session(
            "d1", "d2", "pk1", "ek").session_id
        self.service.post_message(
            _message(self.one2one, "p1", 1, "d1"))

    def _claim(self, lease_id="L1", limit=2, device_id="d3"):
        body, status = self.service.group_inbox_claim(
            device_id, {"lease_id": lease_id, "limit": limit})
        self.assertEqual(status, 201)
        return body


class GetServiceTest(GetMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def test_active_lease_200_shape(self) -> None:
        claim = self._claim("L1", 2)
        body = self.service.group_inbox_lease_get("d3", "L1")
        self.assertEqual(list(body),
                         ["device_id", "lease_id", "limit", "state",
                          "leased_until", "released_at", "completion",
                          "messages"])
        self.assertEqual(body["device_id"], "d3")
        self.assertEqual(body["lease_id"], "L1")
        self.assertEqual(body["limit"], 2)
        self.assertEqual(body["state"], "active")
        self.assertEqual(body["leased_until"], claim["leased_until"])
        self.assertIsNone(body["released_at"])

    def test_messages_in_claim_order_with_current_delivery_values(
            self) -> None:
        claim = self._claim("L1", 3)
        claimed = [(m["session_id"], m["message_id"], m["sequence"])
                   for m in claim["messages"]]
        body = self.service.group_inbox_lease_get("d3", "L1")
        self.assertEqual(len(body["messages"]), 3)
        self.assertEqual(
            [(m["session_id"], m["message_id"], m["sequence"])
             for m in body["messages"]],
            claimed)
        for message in body["messages"]:
            self.assertEqual(
                list(message),
                ["session_id", "message_id", "sequence", "acked",
                 "attempts"])
            self.assertFalse(message["acked"])
            self.assertEqual(message["attempts"], 0)

    def test_ack_marks_item_without_removing_it(self) -> None:
        self._claim("L1", 3)
        self.service.sync_group_ack_messages("d3", {"items": [
            {"session_id": self.gs1, "message_id": "m1",
             "sequence": 1}]})
        body = self.service.group_inbox_lease_get("d3", "L1")
        self.assertEqual(len(body["messages"]), 3)
        by_id = {m["message_id"]: m for m in body["messages"]}
        self.assertTrue(by_id["m1"]["acked"])
        self.assertFalse(by_id["m2"]["acked"])
        self.assertFalse(by_id["m3"]["acked"])

    def test_attempts_reflect_group_retry_batches(self) -> None:
        self._claim("L1", 3)
        self.service.group_inbox_retry_batch("d3", {
            "attempt_id": "t1",
            "items": [{"session_id": self.gs1, "message_id": "m1"}]})
        body = self.service.group_inbox_lease_get("d3", "L1")
        by_id = {m["message_id"]: m for m in body["messages"]}
        self.assertEqual(by_id["m1"]["attempts"], 1)
        self.assertEqual(by_id["m2"]["attempts"], 0)
        # A replayed attempt id does not move the counter.
        self.service.group_inbox_retry_batch("d3", {
            "attempt_id": "t1",
            "items": [{"session_id": self.gs1, "message_id": "m1"}]})
        body = self.service.group_inbox_lease_get("d3", "L1")
        by_id = {m["message_id"]: m for m in body["messages"]}
        self.assertEqual(by_id["m1"]["attempts"], 1)

    def test_renew_updates_effective_deadline(self) -> None:
        self._claim("L1", 1)
        renewed, status = self.service.group_inbox_lease_renew(
            "d3", "L1", {"renewal_id": "R1"})
        self.assertEqual(status, 201)
        body = self.service.group_inbox_lease_get("d3", "L1")
        self.assertEqual(body["state"], "active")
        self.assertEqual(body["leased_until"], renewed["leased_until"])
        renewed2, _ = self.service.group_inbox_lease_renew(
            "d3", "L1", {"renewal_id": "R2"})
        body = self.service.group_inbox_lease_get("d3", "L1")
        self.assertEqual(body["leased_until"], renewed2["leased_until"])

    def test_released_state(self) -> None:
        self._claim("L1", 2)
        released, status = self.service.group_inbox_release("d3", "L1")
        self.assertEqual(status, 201)
        body = self.service.group_inbox_lease_get("d3", "L1")
        self.assertEqual(body["state"], "released")
        self.assertEqual(body["released_at"], released["released_at"])
        self.assertEqual(len(body["messages"]), 2)

    def test_expired_state_after_claim_deadline_passes(self) -> None:
        self._claim("L1", 1)
        past = "2000-01-01T00:00:00.000000+00:00"
        for state in self.store._group_delivery.values():
            for lease in state.leases:
                if lease.lease_id == "L1":
                    lease.leased_until = past
        body = self.service.group_inbox_lease_get("d3", "L1")
        self.assertEqual(body["state"], "expired")
        self.assertEqual(body["leased_until"], past)

    def test_expired_after_renewal_deadline_passes(self) -> None:
        self._claim("L1", 1)
        self.service.group_inbox_lease_renew(
            "d3", "L1", {"renewal_id": "R1"})
        past = "2000-01-01T00:00:00.000000+00:00"
        for state in self.store._group_delivery.values():
            for lease in state.leases:
                if lease.lease_id == "L1":
                    lease.renewals = [MessageLeaseRenewal("R1", past)]
        body = self.service.group_inbox_lease_get("d3", "L1")
        self.assertEqual(body["state"], "expired")
        self.assertEqual(body["leased_until"], past)

    def test_deadline_equal_to_query_instant_is_not_active(self) -> None:
        self._claim("L1", 1)
        # Active requires the deadline to be strictly in the future.
        now = datetime.now(timezone.utc).isoformat(timespec="microseconds")
        for state in self.store._group_delivery.values():
            for lease in state.leases:
                if lease.lease_id == "L1":
                    lease.leased_until = now
        body = self.service.group_inbox_lease_get("d3", "L1")
        self.assertEqual(body["state"], "expired")

    def test_unknown_lease_is_404_lease_id(self) -> None:
        with self.assertRaises(ServiceError) as caught:
            self.service.group_inbox_lease_get("d3", "NOPE")
        self.assertEqual(caught.exception.status_code, 404)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_unknown_lease_404_precedes_unknown_device(self) -> None:
        with self.assertRaises(ServiceError) as caught:
            self.service.group_inbox_lease_get("ghost", "NOPE")
        self.assertEqual(caught.exception.status_code, 404)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_cross_device_lookup_is_409_lease_id(self) -> None:
        self._claim("L1", 2)
        with self.assertRaises(ServiceError) as caught:
            self.service.group_inbox_lease_get("d2", "L1")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_one_to_one_namespace_id_is_409_lease_id(self) -> None:
        _body, status = self.service.inbox_claim(
            "d2", {"lease_id": "P1", "limit": 1})
        self.assertEqual(status, 201)
        for device_id in ("d2", "ghost"):
            with self.assertRaises(ServiceError) as caught:
                self.service.group_inbox_lease_get(device_id, "P1")
            self.assertEqual(caught.exception.status_code, 409)
            self.assertEqual(caught.exception.field, "lease_id")

    def test_matching_lease_readable_after_device_revoked(self) -> None:
        self._claim("L1", 2)
        self.service.revoke_device("d3")
        body = self.service.group_inbox_lease_get("d3", "L1")
        self.assertEqual(body["state"], "active")
        self.assertEqual(body["lease_id"], "L1")

    def test_released_lease_readable_after_device_revoked(self) -> None:
        self._claim("L1", 2)
        self.service.group_inbox_release("d3", "L1")
        self.service.revoke_device("d3")
        body = self.service.group_inbox_lease_get("d3", "L1")
        self.assertEqual(body["state"], "released")
        self.assertIsNotNone(body["released_at"])

    def test_query_is_read_only(self) -> None:
        self._claim("L1", 2)

        def snapshot():
            with self.store._lock:
                return {
                    key: (value.attempts, value.acked,
                          [(lease.leased_until, lease.released_at)
                           for lease in value.leases])
                    for key, value
                    in self.store._group_delivery.items()}

        before = snapshot()
        first = self.service.group_inbox_lease_get("d3", "L1")
        second = self.service.group_inbox_lease_get("d3", "L1")
        self.assertEqual(first, second)
        self.assertEqual(before, snapshot())

    def test_frozen_limit_is_reported(self) -> None:
        self._claim("L1", 1)
        body = self.service.group_inbox_lease_get("d3", "L1")
        self.assertEqual(body["limit"], 1)
        self.assertEqual(len(body["messages"]), 1)


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

    def _request(self, path, raw=None, method="GET", headers=None):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        req_headers = {"Content-Type": "application/json"} if raw else {}
        if headers:
            req_headers.update(headers)
        conn.request(method, path, body=raw, headers=req_headers)
        response = conn.getresponse()
        data = response.read().decode("utf-8")
        conn.close()
        return response.status, (json.loads(data) if data else None), data

    def _claim_http(self, lease_id="L1", limit=2, device_id="d3"):
        status, _, _ = self._request(
            f"/v1/devices/{device_id}/group-inbox/claim",
            json.dumps({"lease_id": lease_id, "limit": limit}),
            method="POST")
        self.assertEqual(status, 201)

    def _release_http(self, device_id="d3", lease_id="L1"):
        status, _, _ = self._request(
            f"/v1/devices/{device_id}/group-inbox/leases/"
            f"{lease_id}/release",
            method="POST")
        self.assertEqual(status, 201)

    def test_get_active_lease_over_http(self) -> None:
        self._claim_http("L1", 2)
        status, body, raw = self._request(
            "/v1/devices/d3/group-inbox/leases/L1")
        self.assertEqual(status, 200)
        self.assertEqual(list(body),
                         ["device_id", "lease_id", "limit", "state",
                          "leased_until", "released_at", "completion",
                          "messages"])
        self.assertLess(raw.index('"device_id"'), raw.index('"lease_id"'))
        self.assertLess(raw.index('"limit"'), raw.index('"state"'))
        self.assertLess(raw.index('"leased_until"'),
                        raw.index('"released_at"'))
        self.assertLess(raw.index('"released_at"'),
                        raw.index('"messages"'))
        self.assertEqual(body["device_id"], "d3")
        self.assertEqual(body["state"], "active")
        self.assertIsNone(body["released_at"])

    def test_repeated_gets_are_byte_identical(self) -> None:
        self._claim_http("L1", 2)
        path = "/v1/devices/d3/group-inbox/leases/L1"
        _, _, first = self._request(path)
        _, _, second = self._request(path)
        self.assertEqual(first, second)

    def test_query_parameter_is_400_query(self) -> None:
        self._claim_http("L1", 2)
        for path in (
                "/v1/devices/d3/group-inbox/leases/L1?x=1",
                "/v1/devices/d3/group-inbox/leases/L1?foo",
                "/v1/devices/d3/group-inbox/leases/L1?limit=1"):
            with self.subTest(path=path):
                status, body, _ = self._request(path)
                self.assertEqual(status, 400)
                self.assertEqual(list(body), ["message", "field"])
                self.assertEqual(body["field"], "query")

    def test_empty_query_string_is_allowed(self) -> None:
        self._claim_http("L1", 2)
        status, _, _ = self._request(
            "/v1/devices/d3/group-inbox/leases/L1?")
        self.assertEqual(status, 200)

    def test_nonempty_body_is_400_request_body(self) -> None:
        self._claim_http("L1", 2)
        status, body, _ = self._request(
            "/v1/devices/d3/group-inbox/leases/L1", raw="{}")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "request_body")

    def test_chunked_nonempty_body_is_400_request_body(self) -> None:
        self._claim_http("L1", 2)
        import socket
        sock = socket.create_connection(
            ("127.0.0.1", self.port), timeout=5)
        try:
            sock.sendall(
                b"GET /v1/devices/d3/group-inbox/leases/L1 HTTP/1.1\r\n"
                b"Host: 127.0.0.1\r\n"
                b"Transfer-Encoding: chunked\r\n"
                b"\r\n"
                b"2\r\n{}\r\n0\r\n\r\n")
            chunks = []
            while True:
                data = sock.recv(4096)
                if not data:
                    break
                chunks.append(data)
        finally:
            sock.close()
        head, _, raw = b"".join(chunks).partition(b"\r\n\r\n")
        status = int(head.split(b" ", 2)[1])
        body = json.loads(raw.decode("utf-8"))
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "request_body")

    def test_errors_over_http(self) -> None:
        self._claim_http("L1", 2)
        status, body, _ = self._request(
            "/v1/devices/d3/group-inbox/leases/NOPE")
        self.assertEqual((status, body["field"]), (404, "lease_id"))
        status, body, _ = self._request(
            "/v1/devices/d2/group-inbox/leases/L1")
        self.assertEqual((status, body["field"]), (409, "lease_id"))

    def test_unknown_lease_404_precedes_unknown_device_over_http(
            self) -> None:
        status, body, _ = self._request(
            "/v1/devices/ghost/group-inbox/leases/NOPE")
        self.assertEqual((status, body["field"]), (404, "lease_id"))

    def test_revoked_device_still_reads_matching_lease(self) -> None:
        self._claim_http("L1", 2)
        self.service.revoke_device("d3")
        status, body, _ = self._request(
            "/v1/devices/d3/group-inbox/leases/L1")
        self.assertEqual(status, 200)
        self.assertEqual(body["state"], "active")

    def test_released_lease_is_200_released_over_http(self) -> None:
        self._claim_http("L1", 2)
        self._release_http()
        status, body, _ = self._request(
            "/v1/devices/d3/group-inbox/leases/L1")
        self.assertEqual(status, 200)
        self.assertEqual(body["state"], "released")
        self.assertIsNotNone(body["released_at"])

    def test_deeper_path_is_404_lease_id(self) -> None:
        self._claim_http("L1", 2)
        for path in (
                "/v1/devices/d3/group-inbox/leases/L1/release",
                "/v1/devices/d3/group-inbox/leases/L1/renew",
                "/v1/devices/d3/group-inbox/leases/L1/extra",
                "/v1/devices/d3/group-inbox/leases/L1/"):
            with self.subTest(path=path):
                status, body, _ = self._request(path)
                self.assertEqual(status, 404)
                self.assertEqual(body["field"], "lease_id")

    def test_bad_path_escapes_are_400_by_segment(self) -> None:
        self._claim_http("L1", 2)
        # Bad device_id escape is reported first.
        status, body, _ = self._request(
            "/v1/devices/d%zz/group-inbox/leases/L1")
        self.assertEqual((status, body["field"]), (400, "device_id"))
        # Invalid UTF-8 in the device segment.
        status, body, _ = self._request(
            "/v1/devices/d%ff/group-inbox/leases/L1")
        self.assertEqual((status, body["field"]), (400, "device_id"))
        # Both bad: device_id still wins.
        status, body, _ = self._request(
            "/v1/devices/d%ff/group-inbox/leases/L%zz")
        self.assertEqual((status, body["field"]), (400, "device_id"))
        # A valid device segment reaches the lease-segment check.
        status, body, _ = self._request(
            "/v1/devices/d3/group-inbox/leases/L%zz")
        self.assertEqual((status, body["field"]), (400, "lease_id"))
        status, body, _ = self._request(
            "/v1/devices/d3/group-inbox/leases/L%ff")
        self.assertEqual((status, body["field"]), (400, "lease_id"))

    def test_percent_encoded_segments_over_http(self) -> None:
        self._claim_http("a/b", 1)
        status, body, _ = self._request(
            "/v1/devices/d3/group-inbox/leases/a%2Fb")
        self.assertEqual(status, 200)
        self.assertEqual(body["lease_id"], "a/b")
        # A percent-encoded '3' decodes into the device id; routing never
        # splits on an encoded slash inside the device segment.
        self._claim_http("L2", 1)
        status, body, _ = self._request(
            "/v1/devices/d%33/group-inbox/leases/L2")
        self.assertEqual(status, 200)
        self.assertEqual(body["device_id"], "d3")

    def test_one_to_one_namespace_id_is_409_over_http(self) -> None:
        status, _, _ = self._request(
            "/v1/devices/d2/inbox/claim",
            json.dumps({"lease_id": "P1", "limit": 1}),
            method="POST")
        self.assertEqual(status, 201)
        status, body, _ = self._request(
            "/v1/devices/d2/group-inbox/leases/P1")
        self.assertEqual((status, body["field"]), (409, "lease_id"))


if __name__ == "__main__":
    unittest.main()
