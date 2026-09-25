"""Tests for the 1:1 inbox lease query endpoint.

GET /v1/devices/{device_id}/inbox/leases/{lease_id} returns one occupied
lease's current snapshot. It takes no body (400/request_body) and no query
parameters (400/query); an unknown lease is 404/lease_id and one owned by
another device is 409/lease_id, both ahead of the path device's state.
The 200 body keys are device_id, lease_id, limit, state, leased_until,
released_at, completion, messages: limit is the claim value, leased_until
is the last renewal's deadline (or the claim deadline), state is only
active|expired|released|completed (completion/release win, an
unterminated lease is active only before its deadline), completion is
null or completion_id/outcome/completed_at, and messages stay in claim
order with acked/attempts read live (an ack never drops an item). The
query shares the store lock but is a pure read: no persistence, no
commit_seq advance, and a restart answers consistently.
"""
import json
import os
import shutil
import tempfile
import threading
import unittest
from datetime import datetime, timedelta, timezone
from http.client import HTTPConnection

from e2ee_backend.models import (
    Device, MessageLease, MessageLeaseRenewal, SignedPreKey)
from e2ee_backend.persistence import attach_persistence
from e2ee_backend.service import DeviceService, ServiceError
from e2ee_backend.http_app import create_server

LEASE_SECONDS = 30


def _parse(stamp: str) -> datetime:
    return datetime.fromisoformat(stamp)


class LeaseQueryMixin:
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

    def _get(self, device_id="bob", lease_id="L1"):
        return self.service.inbox_lease_get(device_id, lease_id)

    def _expire_lease(self, lease_id, renewals=False,
                      seconds_ago=5) -> None:
        """Rewrite the lease's stored deadlines to the past in memory."""
        if renewals:
            seconds_ago = max(seconds_ago, LEASE_SECONDS + 5)
        claim = datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)
        claim_s = claim.isoformat(timespec="microseconds")
        renewal_s = (claim + timedelta(seconds=LEASE_SECONDS)) \
            .isoformat(timespec="microseconds")
        for state in self.service.store._delivery.values():
            for lease in state.leases:
                if lease.lease_id != lease_id:
                    continue
                lease.leased_until = claim_s
                if renewals:
                    lease.renewals = [
                        MessageLeaseRenewal("R1", renewal_s)]


class LeaseQueryServiceTest(LeaseQueryMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def test_active_lease_snapshot_key_order_and_values(self) -> None:
        claim = self._claim("L1", 2)
        body = self._get()
        self.assertEqual(list(body),
                         ["device_id", "lease_id", "limit", "state",
                          "leased_until", "released_at", "completion",
                          "messages"])
        self.assertEqual(body["device_id"], "bob")
        self.assertEqual(body["lease_id"], "L1")
        self.assertEqual(body["limit"], 2)
        self.assertEqual(body["state"], "active")
        self.assertEqual(body["leased_until"], claim["leased_until"])
        self.assertRegex(body["leased_until"], r"\.\d{6}\+00:00$")
        self.assertIsNone(body["released_at"])
        self.assertIsNone(body["completion"])

    def test_messages_keep_claim_order_and_item_key_order(self) -> None:
        self._claim("L1", 10)
        body = self._get()
        self.assertEqual(
            [(m["session_id"], m["message_id"]) for m in body["messages"]],
            [(self.sid1, "a1"), (self.sid1, "a2"), (self.sid1, "a3"),
             (self.sid2, "b1"), (self.sid2, "b2")])
        first = body["messages"][0]
        self.assertEqual(list(first),
                         ["session_id", "message_id", "sequence",
                          "acked", "attempts"])
        self.assertEqual(first["sequence"], 1)
        self.assertIs(first["acked"], False)
        self.assertEqual(first["attempts"], 0)

    def test_limit_is_the_claim_value_not_message_count(self) -> None:
        claim = self._claim("L1", 10)
        self.assertEqual(len(claim["messages"]), 5)
        body = self._get()
        self.assertEqual(body["limit"], 10)
        self.assertEqual(len(body["messages"]), 5)

    def test_leased_until_uses_last_renewal_deadline(self) -> None:
        claim = self._claim("L1", 2)
        first, _ = self.service.inbox_lease_renew(
            "bob", "L1", {"renewal_id": "R1"})
        second, _ = self.service.inbox_lease_renew(
            "bob", "L1", {"renewal_id": "R2"})
        body = self._get()
        self.assertEqual(body["state"], "active")
        self.assertEqual(body["leased_until"], second["leased_until"])
        self.assertNotEqual(body["leased_until"], claim["leased_until"])
        self.assertNotEqual(body["leased_until"], first["leased_until"])

    def test_expired_lease_state_is_expired(self) -> None:
        self._claim("L1", 2)
        self._expire_lease("L1")
        body = self._get()
        self.assertEqual(body["state"], "expired")
        self.assertIsNone(body["released_at"])
        self.assertIsNone(body["completion"])

    def test_expired_after_renewal_still_reports_last_renewal_deadline(
            self) -> None:
        self._claim("L1", 2)
        self.service.inbox_lease_renew(
            "bob", "L1", {"renewal_id": "R1"})
        self._expire_lease("L1", renewals=True)
        # The expiry fabrication rewrote the renewal deadline; the view
        # still surfaces the (fabricated) last renewal value, not the claim.
        fabricated = next(
            lease for state in self.service.store._delivery.values()
            for lease in state.leases if lease.lease_id == "L1")
        body = self._get()
        self.assertEqual(body["state"], "expired")
        self.assertEqual(body["leased_until"],
                         fabricated.renewals[-1].leased_until)
        self.assertNotEqual(body["leased_until"],
                            fabricated.leased_until)

    def test_released_lease_state_is_released(self) -> None:
        self._claim("L1", 2)
        release, status = self.service.inbox_release("bob", "L1")
        self.assertEqual(status, 201)
        body = self._get()
        self.assertEqual(body["state"], "released")
        self.assertEqual(body["released_at"], release["released_at"])
        self.assertIsNone(body["completion"])
        # State after the deadline would otherwise be expired: release wins.
        self._expire_lease("L1")
        self.assertEqual(self._get()["state"], "released")

    def test_completed_lease_state_is_completed_with_record(self) -> None:
        self._claim("L1", 2)
        complete, status = self.service.inbox_lease_complete(
            "bob", "L1", {"completion_id": "C1", "outcome": "delivered"})
        self.assertEqual(status, 201)
        body = self._get()
        self.assertEqual(body["state"], "completed")
        self.assertEqual(list(body["completion"]),
                         ["completion_id", "outcome", "completed_at"])
        self.assertEqual(body["completion"], {
            "completion_id": "C1", "outcome": "delivered",
            "completed_at": complete["completed_at"]})
        # A completed lease past its deadline is still "completed".
        self._expire_lease("L1")
        self.assertEqual(self._get()["state"], "completed")

    def test_failed_completion_outcome_is_reported(self) -> None:
        self._claim("L1", 2)
        self.service.inbox_lease_complete(
            "bob", "L1", {"completion_id": "C1", "outcome": "failed"})
        body = self._get()
        self.assertEqual(body["state"], "completed")
        self.assertEqual(body["completion"]["outcome"], "failed")

    def test_acked_message_stays_listed_with_live_acked_and_attempts(
            self) -> None:
        self._claim("L1", 10)
        # A retry batch bumps attempts on a1; then acking through sequence 2
        # marks a1 and a2 acknowledged without removing either item.
        self.service.inbox_retry_batch("bob", {
            "attempt_id": "AT1",
            "items": [{"session_id": self.sid1, "message_id": "a1"}]})
        self.service.sync_session_ack(self.sid1, {
            "device_id": "bob", "cursor": 2})
        body = self._get()
        items = {(m["session_id"], m["message_id"]): m
                 for m in body["messages"]}
        self.assertEqual(len(items), 5)
        self.assertIs(items[(self.sid1, "a1")]["acked"], True)
        self.assertEqual(items[(self.sid1, "a1")]["attempts"], 1)
        self.assertIs(items[(self.sid1, "a2")]["acked"], True)
        self.assertEqual(items[(self.sid1, "a2")]["attempts"], 0)
        self.assertIs(items[(self.sid1, "a3")]["acked"], False)
        self.assertIs(items[(self.sid2, "b1")]["acked"], False)

    def test_ack_does_not_change_order_or_other_items(self) -> None:
        self._claim("L1", 10)
        self.service.sync_session_ack(self.sid1, {
            "device_id": "bob", "cursor": 3})
        body = self._get()
        self.assertEqual(
            [m["message_id"] for m in body["messages"]],
            ["a1", "a2", "a3", "b1", "b2"])

    def test_unknown_lease_is_404_lease_id(self) -> None:
        with self.assertRaises(ServiceError) as caught:
            self._get("bob", "NOPE")
        self.assertEqual(caught.exception.status_code, 404)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_unknown_lease_404_precedes_unknown_device(self) -> None:
        with self.assertRaises(ServiceError) as caught:
            self._get("ghost", "NOPE")
        self.assertEqual(caught.exception.status_code, 404)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_cross_device_query_is_409_lease_id(self) -> None:
        self._claim("L1", 2)
        with self.assertRaises(ServiceError) as caught:
            self._get("carol", "L1")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_cross_device_conflict_precedes_path_device_revocation(
            self) -> None:
        self._claim("L1", 2)
        self.service.revoke_device("carol")
        with self.assertRaises(ServiceError) as caught:
            self._get("carol", "L1")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_matching_lease_remains_queryable_after_owner_revoked(self) -> None:
        self._claim("L1", 2)
        self.service.revoke_device("bob")
        body = self._get("bob", "L1")
        self.assertEqual(body["device_id"], "bob")
        self.assertEqual(body["lease_id"], "L1")
        self.assertEqual(len(body["messages"]), 2)

    def test_repeat_queries_are_byte_identical_while_state_unchanged(
            self) -> None:
        self._claim("L1", 2)
        first = json.dumps(self._get(), separators=(",", ":"),
                           ensure_ascii=False)
        second = json.dumps(self._get(), separators=(",", ":"),
                            ensure_ascii=False)
        self.assertEqual(second, first)


class LeaseQueryPersistenceTest(LeaseQueryMixin, unittest.TestCase):
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

    def test_query_persists_nothing_and_advances_no_generation(self) -> None:
        self._claim("L1", 2)
        generation = self.state_store.commit_seq
        before = json.dumps(self._document(), sort_keys=True)
        body = self._get()
        self.assertEqual(body["state"], "active")
        self.assertEqual(self.state_store.commit_seq, generation)
        after = json.dumps(self._document(), sort_keys=True)
        self.assertEqual(after, before)
        # A second query still does not write.
        self._get()
        self.assertEqual(self.state_store.commit_seq, generation)

    def test_restart_rebuilds_the_same_view(self) -> None:
        claim = self._claim("L1", 10)
        self.service.inbox_lease_renew(
            "bob", "L1", {"renewal_id": "R1"})
        self.service.inbox_retry_batch("bob", {
            "attempt_id": "AT1",
            "items": [{"session_id": self.sid1, "message_id": "a1"}]})
        self.service.sync_session_ack(self.sid1, {
            "device_id": "bob", "cursor": 1})
        before = self._get()

        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        after = restarted.inbox_lease_get("bob", "L1")
        self.assertEqual(after, before)
        self.assertEqual(after["limit"], 10)
        self.assertEqual(after["leased_until"],
                         before["leased_until"])
        self.assertNotEqual(after["leased_until"], claim["leased_until"])
        self.assertEqual(
            [(m["message_id"], m["acked"], m["attempts"])
             for m in after["messages"]],
            [("a1", True, 1), ("a2", False, 0), ("a3", False, 0),
             ("b1", False, 0), ("b2", False, 0)])

    def test_restart_view_of_released_and_completed_leases(self) -> None:
        self._claim("L1", 2)
        release, _ = self.service.inbox_release("bob", "L1")
        self._claim("L2", 2)
        complete, _ = self.service.inbox_lease_complete(
            "bob", "L2", {"completion_id": "C1", "outcome": "failed"})
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        released = restarted.inbox_lease_get("bob", "L1")
        self.assertEqual(released["state"], "released")
        self.assertEqual(released["released_at"],
                         release["released_at"])
        completed = restarted.inbox_lease_get("bob", "L2")
        self.assertEqual(completed["state"], "completed")
        self.assertEqual(completed["completion"]["completion_id"], "C1")
        self.assertEqual(completed["completion"]["completed_at"],
                         complete["completed_at"])


class LeaseQueryHTTPTest(LeaseQueryMixin, unittest.TestCase):
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

    def _get(self, device_id="bob", lease_id="L1", query="", body=None):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request(
            "GET",
            f"/v1/devices/{device_id}/inbox/leases/{lease_id}{query}",
            body=body,
            headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        data = response.read().decode("utf-8")
        conn.close()
        return response.status, (json.loads(data) if data else None), data

    def _claim(self, lease_id="L1", limit=2):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("POST", "/v1/devices/bob/inbox/claim",
                     body=json.dumps({"lease_id": lease_id, "limit": limit}),
                     headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        response.read()
        conn.close()
        self.assertEqual(response.status, 201)

    def test_get_over_http_key_order(self) -> None:
        self._claim("L1", 2)
        status, body, raw = self._get()
        self.assertEqual(status, 200)
        self.assertEqual(list(body),
                         ["device_id", "lease_id", "limit", "state",
                          "leased_until", "released_at", "completion",
                          "messages"])
        positions = [raw.index(f'"{key}"') for key in (
            "device_id", "lease_id", "limit", "state", "leased_until",
            "released_at", "completion", "messages")]
        self.assertEqual(positions, sorted(positions))
        self.assertEqual(list(body["messages"][0]),
                         ["session_id", "message_id", "sequence",
                          "acked", "attempts"])
        self.assertIn("null", raw)

    def test_nonempty_body_is_400_request_body(self) -> None:
        self._claim("L1", 2)
        for raw_body in ("{}", "x", json.dumps({"limit": 1})):
            with self.subTest(raw_body=raw_body):
                status, body, _ = self._get(body=raw_body)
                self.assertEqual(status, 400)
                self.assertEqual(list(body), ["message", "field"])
                self.assertEqual(body["field"], "request_body")

    def test_query_parameters_are_400_query(self) -> None:
        self._claim("L1", 2)
        for query in ("?limit=1", "?x=1", "?foo", "?a=b&c=d"):
            with self.subTest(query=query):
                status, body, _ = self._get(query=query)
                self.assertEqual(status, 400)
                self.assertEqual(list(body), ["message", "field"])
                self.assertEqual(body["field"], "query")

    def test_errors_over_http(self) -> None:
        self._claim("L1", 2)
        status, body, _ = self._get(lease_id="NOPE")
        self.assertEqual((status, body["field"]), (404, "lease_id"))
        status, body, _ = self._get(device_id="carol")
        self.assertEqual((status, body["field"]), (409, "lease_id"))
        # A revoked owner still serves its leases.
        self.service.store.revoke_device("bob")
        status, body, _ = self._get()
        self.assertEqual(status, 200)
        self.assertEqual(body["state"], "active")

    def test_states_over_http(self) -> None:
        self._claim("L1", 2)
        status, body, _ = self._get()
        self.assertEqual(body["state"], "active")
        self.service.inbox_release("bob", "L1")
        status, body, _ = self._get()
        self.assertEqual(body["state"], "released")
        self._claim("L2", 2)
        self.service.inbox_lease_complete(
            "bob", "L2", {"completion_id": "C1", "outcome": "delivered"})
        status, body, _ = self._get(lease_id="L2")
        self.assertEqual(body["state"], "completed")
        self.assertEqual(body["completion"]["outcome"], "delivered")

    def test_percent_encoded_lease_id_over_http(self) -> None:
        self._claim("a/b", 1)
        status, body, _ = self._get(lease_id="a%2Fb")
        self.assertEqual(status, 200)
        self.assertEqual(body["lease_id"], "a/b")


if __name__ == "__main__":
    unittest.main()
