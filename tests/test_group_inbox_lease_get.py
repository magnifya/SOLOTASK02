"""Tests for the group-inbox lease query endpoint.

GET /v1/devices/{device_id}/group-inbox/leases/{lease_id} returns one
occupied group lease's current state. It takes no request body
(non-empty -> 400/request_body) and no query parameters (any ->
400/query). Path segments are strictly percent-decoded as UTF-8,
device_id first and lease_id second; a deeper path under the lease is
404/lease_id. An unknown lease id is 404/lease_id and one owned by
another device (or committed in the 1:1 namespace) is 409/lease_id,
both ahead of the path device's state; a matching lease stays readable
after its device is revoked. The 200 body keys are device_id, lease_id,
limit, state, leased_until, released_at, messages; the query is purely
read-only (no write, no commit_seq change) and linearized under the
store lock with the mutating lease operations.
"""
import json
import os
import shutil
import tempfile
import threading
import unittest
from datetime import datetime, timedelta, timezone
from http.client import HTTPConnection

from e2ee_backend.models import Device, MessageLeaseRenewal, SignedPreKey
from e2ee_backend.persistence import attach_persistence
from e2ee_backend.service import DeviceService, ServiceError
from e2ee_backend.http_app import create_server

LEASE_SECONDS = 30


def _message(session_id, message_id, sequence, sender):
    return {
        "session_id": session_id,
        "sender_device_id": sender,
        "message_id": message_id,
        "sequence": sequence,
        "nonce": f"nonce-{message_id}",
        "ciphertext": "ciphertext",
    }


class GroupGetMixin:
    def _build(self) -> None:
        self.service = DeviceService()
        self.store = self.service.store
        self.store.add_device(Device("u", "d1", "ik"))
        self.store.add_device(Device("u", "d3", "ik"))
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
        # m1/m2 from d1, m3 from d2: d3 may claim all three.
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

    def _claim_deadline(self, lease_id) -> str:
        for state in self.store._group_delivery.values():
            for lease in state.leases:
                if lease.lease_id == lease_id:
                    return lease.leased_until
        raise AssertionError(lease_id)

    def _expire_group_lease(self, lease_id, *, with_renewal=False,
                            seconds_ago=5) -> None:
        """Rewrite the lease's stored deadlines to the past in memory."""
        if with_renewal:
            seconds_ago = max(seconds_ago, LEASE_SECONDS + 5)
        claim = datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)
        claim_s = claim.isoformat(timespec="microseconds")
        renewal_s = (claim + timedelta(seconds=LEASE_SECONDS)) \
            .isoformat(timespec="microseconds")
        for state in self.store._group_delivery.values():
            for lease in state.leases:
                if lease.lease_id != lease_id:
                    continue
                lease.leased_until = claim_s
                if with_renewal:
                    lease.renewals = [
                        MessageLeaseRenewal("R1", renewal_s)]


class GroupGetServiceTest(GroupGetMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def test_active_lease_200_shape(self) -> None:
        claim = self._claim("L1", 2)
        body = self.service.group_inbox_lease_get("d3", "L1")
        self.assertEqual(list(body),
                         ["device_id", "lease_id", "limit", "state",
                          "leased_until", "released_at", "messages"])
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
             for m in body["messages"]], claimed)
        for message in body["messages"]:
            self.assertEqual(
                list(message),
                ["session_id", "message_id", "sequence", "acked",
                 "attempts"])
            self.assertFalse(message["acked"])
            self.assertEqual(message["attempts"], 0)

    def test_ack_marks_item_without_removing_it(self) -> None:
        self._claim("L1", 5)
        self.service.sync_group_ack_messages("d3", {"items": [
            {"session_id": self.gs1, "message_id": "m1", "sequence": 1},
            {"session_id": self.gs1, "message_id": "m2",
             "sequence": 2}]})
        body = self.service.group_inbox_lease_get("d3", "L1")
        self.assertEqual(len(body["messages"]), 3)
        by_id = {m["message_id"]: m for m in body["messages"]}
        self.assertTrue(by_id["m1"]["acked"])
        self.assertTrue(by_id["m2"]["acked"])
        self.assertFalse(by_id["m3"]["acked"])

    def test_attempts_reflect_retry_batches(self) -> None:
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
        claim = self._claim("L1", 2)
        first, status = self.service.group_inbox_lease_renew(
            "d3", "L1", {"renewal_id": "R1"})
        self.assertEqual(status, 201)
        second, status = self.service.group_inbox_lease_renew(
            "d3", "L1", {"renewal_id": "R2"})
        self.assertEqual(status, 201)
        body = self.service.group_inbox_lease_get("d3", "L1")
        self.assertEqual(body["state"], "active")
        self.assertNotEqual(body["leased_until"], claim["leased_until"])
        self.assertEqual(body["leased_until"], second["leased_until"])
        self.assertNotEqual(body["leased_until"], first["leased_until"])

    def test_released_lease_reports_released_state_and_stamp(self) -> None:
        self._claim("L1", 2)
        released, status = self.service.group_inbox_release("d3", "L1")
        self.assertEqual(status, 201)
        body = self.service.group_inbox_lease_get("d3", "L1")
        self.assertEqual(body["state"], "released")
        self.assertEqual(body["released_at"], released["released_at"])
        self.assertEqual(body["leased_until"], self._claim_deadline("L1"))

    def test_expired_lease_reports_expired(self) -> None:
        self._claim("L1", 2)
        self._expire_group_lease("L1")
        body = self.service.group_inbox_lease_get("d3", "L1")
        self.assertEqual(body["state"], "expired")
        self.assertIsNone(body["released_at"])

    def test_expired_after_renewal_uses_renewal_deadline(self) -> None:
        self._claim("L1", 2)
        self.service.group_inbox_lease_renew(
            "d3", "L1", {"renewal_id": "R1"})
        self._expire_group_lease("L1", with_renewal=True)
        body = self.service.group_inbox_lease_get("d3", "L1")
        self.assertEqual(body["state"], "expired")
        deadlines = [
            lease.renewals[-1].leased_until
            for state in self.store._group_delivery.values()
            for lease in state.leases if lease.lease_id == "L1"]
        self.assertEqual(body["leased_until"], deadlines[0])

    def test_unknown_lease_is_404_lease_id(self) -> None:
        with self.assertRaises(ServiceError) as caught:
            self.service.group_inbox_lease_get("d3", "NOPE")
        self.assertEqual((caught.exception.status_code,
                          caught.exception.to_body()["field"]),
                         (404, "lease_id"))

    def test_other_device_lease_is_409_lease_id(self) -> None:
        self._claim("L1", 2, device_id="d3")
        with self.assertRaises(ServiceError) as caught:
            self.service.group_inbox_lease_get("d2", "L1")
        self.assertEqual((caught.exception.status_code,
                          caught.exception.to_body()["field"]),
                         (409, "lease_id"))

    def test_one_to_one_namespace_lease_is_409_lease_id(self) -> None:
        body, status = self.service.inbox_claim(
            "d2", {"lease_id": "ONE", "limit": 1})
        self.assertEqual(status, 201)
        self.assertTrue(body["messages"])
        with self.assertRaises(ServiceError) as caught:
            self.service.group_inbox_lease_get("d2", "ONE")
        self.assertEqual((caught.exception.status_code,
                          caught.exception.to_body()["field"]),
                         (409, "lease_id"))

    def test_conflicts_take_priority_over_unknown_device(self) -> None:
        self._claim("L1", 2, device_id="d3")
        # An unregistered path device still gets the ownership conflict.
        with self.assertRaises(ServiceError) as caught:
            self.service.group_inbox_lease_get("ghost", "L1")
        self.assertEqual(caught.exception.status_code, 409)
        with self.assertRaises(ServiceError) as caught:
            self.service.group_inbox_lease_get("ghost", "NOPE")
        self.assertEqual(caught.exception.status_code, 404)

    def test_matching_lease_readable_after_revocation(self) -> None:
        self._claim("L1", 2)
        self.store.revoke_device("d3")
        body = self.service.group_inbox_lease_get("d3", "L1")
        self.assertEqual(body["device_id"], "d3")
        self.assertEqual(body["lease_id"], "L1")
        self.assertEqual(len(body["messages"]), 2)

    def test_unchanged_state_answers_byte_identically(self) -> None:
        self._claim("L1", 2)
        first = json.dumps(
            self.service.group_inbox_lease_get("d3", "L1"),
            ensure_ascii=False, separators=(",", ":"))
        second = json.dumps(
            self.service.group_inbox_lease_get("d3", "L1"),
            ensure_ascii=False, separators=(",", ":"))
        self.assertEqual(second, first)


class GroupGetPersistenceTest(GroupGetMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self._build()
        self.path = os.path.join(self.directory, "state.json")
        self.state_store = attach_persistence(self.service, self.path)

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_query_writes_nothing_and_advances_no_generation(self) -> None:
        self._claim("L1", 2)
        self.service.group_inbox_lease_renew(
            "d3", "L1", {"renewal_id": "R1"})
        generation = self.state_store.commit_seq
        body = self.service.group_inbox_lease_get("d3", "L1")
        self.assertEqual(body["state"], "active")
        self.assertEqual(self.state_store.commit_seq, generation)

    def test_restart_preserves_query_result(self) -> None:
        self._claim("L1", 2)
        self.service.group_inbox_lease_renew(
            "d3", "L1", {"renewal_id": "R1"})
        self.service.sync_group_ack_messages("d3", {"items": [
            {"session_id": self.gs1, "message_id": "m1", "sequence": 1}]})
        before = self.service.group_inbox_lease_get("d3", "L1")
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        after = restarted.group_inbox_lease_get("d3", "L1")
        self.assertEqual(after, before)


class GroupGetHTTPTest(GroupGetMixin, unittest.TestCase):
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

    def _get(self, device_id, lease_id, query="", body=None):
        target = f"/v1/devices/{device_id}/group-inbox/leases/" \
                 f"{lease_id}{query}"
        headers = {}
        if body is not None:
            headers["Content-Type"] = "application/json"
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("GET", target, body=body, headers=headers)
        response = conn.getresponse()
        data = response.read().decode("utf-8")
        conn.close()
        return response.status, (json.loads(data) if data else None), data

    def _claim_http(self, lease_id="L1", limit=2, device_id="d3"):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request(
            "POST", f"/v1/devices/{device_id}/group-inbox/claim",
            body=json.dumps({"lease_id": lease_id, "limit": limit}),
            headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        response.read()
        conn.close()
        self.assertEqual(response.status, 201)

    def test_get_active_lease_over_http(self) -> None:
        self._claim_http("L1", 2)
        status, body, raw = self._get("d3", "L1")
        self.assertEqual(status, 200)
        self.assertEqual(list(body),
                         ["device_id", "lease_id", "limit", "state",
                          "leased_until", "released_at", "messages"])
        self.assertLess(raw.index('"device_id"'), raw.index('"lease_id"'))
        self.assertLess(raw.index('"lease_id"'), raw.index('"limit"'))
        self.assertLess(raw.index('"limit"'), raw.index('"state"'))
        self.assertLess(raw.index('"state"'),
                        raw.index('"leased_until"'))
        self.assertLess(raw.index('"leased_until"'),
                        raw.index('"released_at"'))
        self.assertLess(raw.index('"released_at"'),
                        raw.index('"messages"'))
        self.assertEqual(body["state"], "active")
        self.assertEqual(len(body["messages"]), 2)
        for message in body["messages"]:
            self.assertEqual(
                list(message),
                ["session_id", "message_id", "sequence", "acked",
                 "attempts"])

    def test_repeated_gets_are_byte_identical(self) -> None:
        self._claim_http("L1", 2)
        _status, _body, first = self._get("d3", "L1")
        _status, _body, second = self._get("d3", "L1")
        self.assertEqual(second, first)

    def test_non_empty_body_is_400_request_body(self) -> None:
        self._claim_http("L1", 2)
        for payload in ("{}", "x", "[]"):
            with self.subTest(payload=payload):
                status, body, _ = self._get("d3", "L1", body=payload)
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], "request_body")
        # A follow-up request still works: the rejected body was drained.
        status, body, _ = self._get("d3", "L1")
        self.assertEqual(status, 200)
        self.assertEqual(body["lease_id"], "L1")

    def test_query_parameters_are_400_query(self) -> None:
        self._claim_http("L1", 2)
        for query in ("?foo", "?x=1", "?x="):
            with self.subTest(query=query):
                status, body, _ = self._get("d3", "L1", query)
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], "query")
        # A bare trailing '?' carries no parameter and is accepted.
        status, _, _ = self._get("d3", "L1", "?")
        self.assertEqual(status, 200)

    def test_errors_over_http(self) -> None:
        self._claim_http("L1", 2)
        status, body, _ = self._get("d3", "NOPE")
        self.assertEqual((status, body["field"]), (404, "lease_id"))
        status, body, _ = self._get("d2", "L1")
        self.assertEqual((status, body["field"]), (409, "lease_id"))
        # A matching lease stays readable after its device is revoked.
        self.service.store.revoke_device("d3")
        status, body, _ = self._get("d3", "L1")
        self.assertEqual(status, 200)
        self.assertEqual(body["device_id"], "d3")

    def test_one_to_one_namespace_is_409_over_http(self) -> None:
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request(
            "POST", "/v1/devices/d2/inbox/claim",
            body=json.dumps({"lease_id": "ONE", "limit": 1}),
            headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        self.assertEqual(response.status, 201)
        response.read()
        conn.close()
        status, body, _ = self._get("d2", "ONE")
        self.assertEqual((status, body["field"]), (409, "lease_id"))

    def test_bad_path_escapes_are_400_by_segment(self) -> None:
        self._claim_http("L1", 2)
        # Bad device_id escape is reported first.
        status, body, _ = self._get("d%zz", "L1")
        self.assertEqual((status, body["field"]), (400, "device_id"))
        status, body, _ = self._get("d%ff", "L1")
        self.assertEqual((status, body["field"]), (400, "device_id"))
        # A valid device segment reaches the lease-segment check.
        status, body, _ = self._get("d3", "L%zz")
        self.assertEqual((status, body["field"]), (400, "lease_id"))
        status, body, _ = self._get("d3", "L%ff")
        self.assertEqual((status, body["field"]), (400, "lease_id"))

    def test_deeper_path_is_404_lease_id(self) -> None:
        self._claim_http("L1", 2)
        status, body, _ = self._get("d3", "L1/extra")
        self.assertEqual((status, body["field"]), (404, "lease_id"))
        status, body, _ = self._get("d3", "L1/release")
        self.assertEqual((status, body["field"]), (404, "lease_id"))

    def test_percent_encoded_segments_over_http(self) -> None:
        self._claim_http("a/b", 1, device_id="d3")
        self._claim_http("L1", 1, device_id="d3")
        status, body, _ = self._get("d3", "a%2Fb")
        self.assertEqual(status, 200)
        self.assertEqual(body["lease_id"], "a/b")
        # A percent-encoded '3' in the device segment decodes normally.
        status, body, _ = self._get("d%33", "L1")
        self.assertEqual(status, 200)
        self.assertEqual(body["device_id"], "d3")


if __name__ == "__main__":
    unittest.main()
