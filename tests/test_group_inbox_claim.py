"""Tests for group-inbox lease claim and release endpoints.

POST /v1/devices/{device_id}/group-inbox/claim and
POST /v1/devices/{device_id}/group-inbox/leases/{lease_id}/release add a
lease placeholder on top of the existing aggregated group inbox without
touching acks, attempts or the attempt-id dedup sets.
"""
import json
import os
import re
import shutil
import tempfile
import threading
import unittest
from http.client import HTTPConnection

from e2ee_backend.http_app import create_server
from e2ee_backend.models import Device, SignedPreKey
from e2ee_backend.persistence import PersistenceUnavailable, attach_persistence
from e2ee_backend.service import DeviceService, ServiceError

_ISO_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}\+00:00$")


class GroupClaimFixture:
    def __init__(self) -> None:
        self.service = DeviceService()
        self.store = self.service.store
        for device_id in ("d1", "d2", "d3", "d4"):
            self.store.add_device(Device("u", device_id, "ik"))
        self.store.add_device(Device(
            "u", "d5", "ik",
            prekeys=[SignedPreKey("pk1", "pubk1")]))
        self.service.create_group({
            "group_id": "g1", "creator_device_id": "d1",
            "member_device_ids": ["d2", "d3"]})
        self.gs1 = self.service.create_group_session({
            "group_id": "g1", "initiator_device_id": "d1",
            "ephemeral_key": "epk1"})["session_id"]
        for message_id, sequence, sender in (
                ("m1", 1, "d1"), ("m2", 2, "d1"), ("m3", 3, "d2")):
            self.service.post_message({
                "session_id": self.gs1,
                "sender_device_id": sender,
                "message_id": message_id, "sequence": sequence,
                "nonce": f"n-{message_id}", "ciphertext": "ct"})
        self.one2one = self.store.create_session(
            "d1", "d5", "pk1", "ek").session_id
        self.service.post_message({
            "session_id": self.one2one, "sender_device_id": "d1",
            "message_id": "p1", "sequence": 1,
            "nonce": "np", "ciphertext": "ct"})


class GroupClaimServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = GroupClaimFixture()
        self.service = self.fixture.service
        self.store = self.fixture.store

    def test_claim_shape_and_order(self) -> None:
        body, status = self.service.group_inbox_claim(
            "d3", {"lease_id": "L1", "limit": 2})
        self.assertEqual(status, 201)
        self.assertEqual(list(body),
                         ["device_id", "lease_id", "leased_until",
                          "messages"])
        self.assertEqual(body["device_id"], "d3")
        self.assertEqual(body["lease_id"], "L1")
        self.assertRegex(body["leased_until"], _ISO_RE)
        self.assertEqual([m["message_id"] for m in body["messages"]],
                         ["m1", "m2"])

    def test_empty_claim_occupies_nothing(self) -> None:
        body, status = self.service.group_inbox_claim(
            "d4", {"lease_id": "E1", "limit": 10})
        self.assertEqual(status, 200)
        self.assertIsNone(body["leased_until"])
        self.assertEqual(body["messages"], [])
        self.assertEqual(self.store._group_delivery, {})
        # The id stayed free: a later non-empty claim under the same id is
        # a fresh 201, not a frozen replay.
        self.service.add_group_member("g1", {
            "actor_device_id": "d1", "device_id": "d4"})
        gs2 = self.service.create_group_session({
            "group_id": "g1", "initiator_device_id": "d1",
            "ephemeral_key": "epk2"})["session_id"]
        self.service.post_message({
            "session_id": gs2, "sender_device_id": "d1",
            "message_id": "n1", "sequence": 1,
            "nonce": "x", "ciphertext": "ct"})
        body, status = self.service.group_inbox_claim(
            "d4", {"lease_id": "E1", "limit": 10})
        self.assertEqual(status, 201)
        self.assertEqual([m["message_id"] for m in body["messages"]],
                         ["n1"])

    def test_exact_replay_is_frozen(self) -> None:
        first, status = self.service.group_inbox_claim(
            "d3", {"lease_id": "L1", "limit": 2})
        self.assertEqual(status, 201)
        second, status = self.service.group_inbox_claim(
            "d3", {"lease_id": "L1", "limit": 2})
        self.assertEqual(status, 200)
        self.assertEqual(second, first)

    def test_replay_after_release_stays_frozen(self) -> None:
        first, _ = self.service.group_inbox_claim(
            "d3", {"lease_id": "L1", "limit": 2})
        released, status = self.service.group_inbox_release("d3", "L1")
        self.assertEqual(status, 201)
        self.assertEqual(released["released_count"], 2)
        replay, status = self.service.group_inbox_claim(
            "d3", {"lease_id": "L1", "limit": 2})
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)
        # Released messages are claimable by a new lease.
        body, status = self.service.group_inbox_claim(
            "d3", {"lease_id": "L2", "limit": 10})
        self.assertEqual(status, 201)
        self.assertEqual([m["message_id"] for m in body["messages"]],
                         ["m1", "m2", "m3"])

    def test_changed_device_or_limit_conflicts_first(self) -> None:
        self.service.group_inbox_claim(
            "d3", {"lease_id": "L1", "limit": 2})
        self.store.revoke_device("d4")
        with self.assertRaises(ServiceError) as caught:
            self.service.group_inbox_claim(
                "d4", {"lease_id": "L1", "limit": 2})
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")
        with self.assertRaises(ServiceError) as caught:
            self.service.group_inbox_claim(
                "d3", {"lease_id": "L1", "limit": 3})
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_unknown_or_revoked_device(self) -> None:
        with self.assertRaises(ServiceError) as caught:
            self.service.group_inbox_claim(
                "ghost", {"lease_id": "L", "limit": 1})
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "device_id")
        self.store.revoke_device("d3")
        with self.assertRaises(ServiceError) as caught:
            self.service.group_inbox_claim(
                "d3", {"lease_id": "L2", "limit": 1})
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "device_id")

    def test_lease_id_is_global_across_inboxes(self) -> None:
        self.service.inbox_claim(
            "d5", {"lease_id": "SHARED", "limit": 1})
        with self.assertRaises(ServiceError) as caught:
            self.service.group_inbox_claim(
                "d3", {"lease_id": "SHARED", "limit": 1})
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_claim_ignores_attempts_but_respects_acks(self) -> None:
        # A retry-batch write must not withhold messages from a claim.
        self.service.group_inbox_retry_batch("d3", {
            "attempt_id": "a1",
            "items": [{"session_id": self.fixture.gs1, "message_id": "m1"}]})
        body, _ = self.service.group_inbox_claim(
            "d3", {"lease_id": "L1", "limit": 1})
        self.assertEqual([m["message_id"] for m in body["messages"]],
                         ["m1"])
        # An ack during the lease changes nothing for the frozen lease but
        # excludes the message from a later new lease after release.
        self.service.ack_message(self.fixture.gs1, {
            "session_id": self.fixture.gs1, "message_id": "m1",
            "device_id": "d3", "sequence": 1})
        self.service.group_inbox_release("d3", "L1")
        body, _ = self.service.group_inbox_claim(
            "d3", {"lease_id": "L2", "limit": 10})
        self.assertEqual([m["message_id"] for m in body["messages"]],
                         ["m2", "m3"])

    def test_release_lifecycle_and_replay(self) -> None:
        self.service.group_inbox_claim(
            "d3", {"lease_id": "L1", "limit": 2})
        body, status = self.service.group_inbox_release("d3", "L1")
        self.assertEqual(status, 201)
        self.assertEqual(list(body),
                         ["device_id", "lease_id", "released_at",
                          "released_count"])
        self.assertEqual(body["device_id"], "d3")
        self.assertEqual(body["lease_id"], "L1")
        self.assertEqual(body["released_count"], 2)
        self.assertRegex(body["released_at"], _ISO_RE)
        replay, status = self.service.group_inbox_release("d3", "L1")
        self.assertEqual(status, 200)
        self.assertEqual(replay, body)

    def test_release_errors(self) -> None:
        with self.assertRaises(ServiceError) as caught:
            self.service.group_inbox_release("d3", "nope")
        self.assertEqual(caught.exception.status_code, 404)
        self.assertEqual(caught.exception.field, "lease_id")
        self.service.group_inbox_claim(
            "d3", {"lease_id": "L1", "limit": 2})
        with self.assertRaises(ServiceError) as caught:
            self.service.group_inbox_release("d2", "L1")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")
        # Ownership beats device state: a revoked path device still gets
        # the 409/lease_id ownership error.
        self.store.revoke_device("d2")
        with self.assertRaises(ServiceError) as caught:
            self.service.group_inbox_release("d2", "L1")
        self.assertEqual(caught.exception.field, "lease_id")
        # A first release on a revoked owner is 409/device_id.
        self.store.revoke_device("d3")
        with self.assertRaises(ServiceError) as caught:
            self.service.group_inbox_release("d3", "L1")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "device_id")

    def test_body_validation(self) -> None:
        def claim(payload):
            return self.service.group_inbox_claim("d3", payload)

        cases = (
            ([1, 2], "request_body"),
            ({"lease_id": "L1"}, "limit"),
            ({"limit": 2}, "lease_id"),
            ({"lease_id": "", "limit": 2}, "lease_id"),
            ({"lease_id": 5, "limit": 2}, "lease_id"),
            ({"lease_id": None, "limit": 2}, "lease_id"),
            ({"lease_id": "L1", "limit": None}, "limit"),
            ({"lease_id": "L1", "limit": True}, "limit"),
            ({"lease_id": "L1", "limit": 1.5}, "limit"),
            ({"lease_id": "L1", "limit": 0}, "limit"),
            ({"lease_id": "L1", "limit": 101}, "limit"),
            ({"lease_id": "L1", "limit": "2"}, "limit"),
            ({"lease_id": "L1", "limit": 2, "extra": 1}, "extra"))
        for payload, field in cases:
            with self.assertRaises(ServiceError) as caught:
                claim(payload)
            self.assertEqual(caught.exception.field, field, payload)
            self.assertEqual(caught.exception.status_code, 400)


class GroupClaimPersistenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.fixture = GroupClaimFixture()
        self.service = self.fixture.service
        self.state_store = attach_persistence(
            self.service, os.path.join(self.directory, "state.json"))

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_claim_and_release_commit_one_generation_each(self) -> None:
        generation = self.state_store.commit_seq
        self.service.group_inbox_claim(
            "d3", {"lease_id": "L1", "limit": 2})
        self.assertEqual(self.state_store.commit_seq, generation + 1)
        self.service.group_inbox_release("d3", "L1")
        self.assertEqual(self.state_store.commit_seq, generation + 2)
        # Replays and empty claims consume no generation.
        self.service.group_inbox_claim(
            "d3", {"lease_id": "L1", "limit": 2})
        self.service.group_inbox_release("d3", "L1")
        self.service.group_inbox_claim(
            "d4", {"lease_id": "E1", "limit": 5})
        self.assertEqual(self.state_store.commit_seq, generation + 2)

    def test_failed_claim_write_rolls_back_every_lease(self) -> None:
        def raise_oserror(state):
            raise OSError("simulated disk failure")

        self.state_store.save = raise_oserror
        with self.assertRaises(PersistenceUnavailable):
            self.service.group_inbox_claim(
                "d3", {"lease_id": "L1", "limit": 10})
        self.assertFalse(any(
            state.leases
            for state in self.service.store._group_delivery.values()))

    def test_restart_recovers_leases_and_release(self) -> None:
        first, _ = self.service.group_inbox_claim(
            "d3", {"lease_id": "L1", "limit": 2})
        self.service.group_inbox_release("d3", "L1")
        self.service.group_inbox_claim(
            "d3", {"lease_id": "L2", "limit": 2})
        self.service.group_inbox_release("d3", "L2")
        restarted = DeviceService()
        attach_persistence(restarted, self.state_store.path)
        replay, status = restarted.group_inbox_claim(
            "d3", {"lease_id": "L1", "limit": 2})
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)
        released, status = restarted.group_inbox_release("d3", "L1")
        self.assertEqual(status, 200)
        self.assertEqual(released["released_count"], 2)
        with self.assertRaises(ServiceError) as caught:
            restarted.group_inbox_claim(
                "d3", {"lease_id": "L2", "limit": 3})
        self.assertEqual(caught.exception.field, "lease_id")
        released2, status = restarted.group_inbox_release("d3", "L2")
        self.assertEqual(status, 200)
        self.assertEqual(released2["released_count"], 2)
        # After recovery the released L1 messages are claimable again.
        body, status = restarted.group_inbox_claim(
            "d3", {"lease_id": "L3", "limit": 10})
        self.assertEqual(status, 201)
        self.assertEqual([m["message_id"] for m in body["messages"]],
                         ["m1", "m2", "m3"])


class GroupClaimHTTPTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = GroupClaimFixture()
        self.server, _service = create_server(
            "127.0.0.1", 0, self.fixture.service)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _post(self, path, body=b"", content_type="application/json"):
        headers = {}
        if body:
            headers["Content-Type"] = content_type
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("POST", path, body=body, headers=headers)
        response = conn.getresponse()
        status = response.status
        raw = response.read()
        conn.close()
        return status, raw

    def test_claim_and_release_over_http(self) -> None:
        status, raw = self._post(
            "/v1/devices/d3/group-inbox/claim",
            json.dumps({"lease_id": "L1", "limit": 2}))
        self.assertEqual(status, 201)
        body = json.loads(raw)
        self.assertEqual([m["message_id"] for m in body["messages"]],
                         ["m1", "m2"])
        status, raw = self._post(
            "/v1/devices/d3/group-inbox/claim",
            json.dumps({"lease_id": "L1", "limit": 2}))
        self.assertEqual(status, 200)
        status, raw = self._post(
            "/v1/devices/d3/group-inbox/leases/L1/release")
        self.assertEqual(status, 201)
        released = json.loads(raw)
        self.assertEqual(list(released),
                         ["device_id", "lease_id", "released_at",
                          "released_count"])
        status, _raw = self._post(
            "/v1/devices/d3/group-inbox/leases/L1/release")
        self.assertEqual(status, 200)

    def test_empty_claim_http(self) -> None:
        status, raw = self._post(
            "/v1/devices/d4/group-inbox/claim",
            json.dumps({"lease_id": "E1", "limit": 3}))
        self.assertEqual(status, 200)
        body = json.loads(raw)
        self.assertIsNone(body["leased_until"])
        self.assertEqual(body["messages"], [])

    def test_claim_body_errors(self) -> None:
        for raw_body, field in (
                (b"{bad json", "request_body"),
                (b"[1,2]", "request_body"),
                (b'{"lease_id":"L1"}', "limit"),
                (b'{"limit":2}', "lease_id"),
                (b'{"lease_id":"L1","limit":0}', "limit"),
                (b'{"lease_id":"L1","limit":true}', "limit"),
                (b'{"lease_id":"L1","limit":1.5}', "limit"),
                (b'{"lease_id":"L1","limit":101}', "limit"),
                (b'{"lease_id":"","limit":2}', "lease_id"),
                (b'{"lease_id":"L1","limit":2,"x":1}', "x")):
            status, raw = self._post(
                "/v1/devices/d3/group-inbox/claim", raw_body)
            self.assertEqual(status, 400, raw_body)
            self.assertEqual(json.loads(raw)["field"], field, raw_body)

    def test_claim_device_errors(self) -> None:
        status, raw = self._post(
            "/v1/devices/ghost/group-inbox/claim",
            json.dumps({"lease_id": "L", "limit": 1}))
        self.assertEqual(status, 409)
        self.assertEqual(json.loads(raw)["field"], "device_id")
        status, raw = self._post(
            "/v1/devices/d3%FF/group-inbox/claim",
            json.dumps({"lease_id": "L", "limit": 1}))
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(raw)["field"], "device_id")

    def test_release_rejects_body_query_and_bad_encoding(self) -> None:
        self._post("/v1/devices/d3/group-inbox/claim",
                   json.dumps({"lease_id": "L1", "limit": 1}))
        status, raw = self._post(
            "/v1/devices/d3/group-inbox/leases/L1/release",
            body=b"x")
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(raw)["field"], "request_body")
        status, raw = self._post(
            "/v1/devices/d3/group-inbox/leases/L1/release?foo=1")
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(raw)["field"], "query")
        # A trailing '?' with no parameters is accepted.
        status, _raw = self._post(
            "/v1/devices/d3/group-inbox/leases/L1/release?")
        self.assertEqual(status, 201)
        status, raw = self._post(
            "/v1/devices/d3/group-inbox/leases/L%FF/release")
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(raw)["field"], "lease_id")
        status, raw = self._post(
            "/v1/devices/d3%FF/group-inbox/leases/L1/release")
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(raw)["field"], "device_id")

    def test_release_not_found_and_owner_conflict(self) -> None:
        status, raw = self._post(
            "/v1/devices/d3/group-inbox/leases/missing/release")
        self.assertEqual(status, 404)
        self.assertEqual(json.loads(raw)["field"], "lease_id")
        self._post("/v1/devices/d3/group-inbox/claim",
                   json.dumps({"lease_id": "L1", "limit": 1}))
        status, raw = self._post(
            "/v1/devices/d2/group-inbox/leases/L1/release")
        self.assertEqual(status, 409)
        self.assertEqual(json.loads(raw)["field"], "lease_id")


if __name__ == "__main__":
    unittest.main()
