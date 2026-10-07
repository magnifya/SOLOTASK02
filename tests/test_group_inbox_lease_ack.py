"""Tests for the group-inbox lease bulk-acknowledgement endpoint.

POST /v1/devices/{device_id}/group-inbox/leases/{lease_id}/ack takes no
request body (a non-empty one is 400/request_body) and no query string
(any parameter is 400/query). An unknown group lease is 404/lease_id,
one owned by another device (or committed in the 1:1 namespace) is
409/lease_id; both precede the path device's state. Only a lease
completed with outcome ``delivered`` may be acked: an active, expired,
released or ``failed`` lease is 409/lease_id. A lease whose messages are
already all acked answers 200 and writes nothing, even after the device
is revoked. A first ack on an unknown/revoked device is 409/device_id.
A first ack returns 201 with device_id/lease_id/acked/message_count and
sets every covered group delivery record acked with ack_sequence the
message sequence (attempts and attempt ids unchanged), persisting one
generation.
"""
import json
import os
import shutil
import tempfile
import threading
import unittest
from datetime import datetime, timedelta, timezone
from http.client import HTTPConnection
from urllib.parse import quote

from e2ee_backend.models import Device, SignedPreKey
from e2ee_backend.persistence import (
    PersistenceUnavailable, attach_persistence)
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


class GroupAckMixin:
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
        # A 1:1 session d1 -> d2 never contributes to a group claim.
        self.one2one = self.store.create_session(
            "d1", "d2", "pk1", "ek").session_id
        self.service.post_message(
            _message(self.one2one, "p1", 1, "d1"))

    def _claim(self, lease_id="L1", limit=2, device_id="d3"):
        body, status = self.service.group_inbox_claim(
            device_id, {"lease_id": lease_id, "limit": limit})
        self.assertEqual(status, 201)
        return body

    def _complete(self, lease_id="L1", outcome="delivered",
                  device_id="d3"):
        body, status = self.service.group_inbox_lease_complete(
            device_id, lease_id,
            {"completion_id": "C1", "outcome": outcome})
        self.assertEqual(status, 201)
        return body

    def _ack(self, device_id="d3", lease_id="L1"):
        return self.service.group_inbox_lease_ack(device_id, lease_id)

    def _error(self, call):
        with self.assertRaises(ServiceError) as caught:
            call()
        return caught.exception

    def _expire_group_lease(self, lease_id, seconds_ago=5) -> None:
        """Rewrite the lease's stored claim deadline to the past."""
        claim = datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)
        claim_s = claim.isoformat(timespec="microseconds")
        for state in self.store._group_delivery.values():
            for lease in state.leases:
                if lease.lease_id == lease_id:
                    lease.leased_until = claim_s


class GroupAckServiceTest(GroupAckMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def test_first_ack_after_delivered_is_201(self) -> None:
        self._claim("L1", 2)
        self._complete("L1", "delivered")
        body, status = self._ack()
        self.assertEqual(status, 201)
        self.assertEqual(list(body),
                         ["device_id", "lease_id", "acked",
                          "message_count"])
        self.assertEqual(body, {"device_id": "d3", "lease_id": "L1",
                                "acked": True, "message_count": 2})

    def test_ack_sets_acked_and_ack_sequence_per_message(self) -> None:
        self._claim("L1", 2)
        self._complete("L1", "delivered")
        self._ack()
        for message_id, sequence in (("m1", 1), ("m2", 2)):
            state = self.store._group_delivery[
                (self.gs1, message_id, "d3")]
            self.assertTrue(state.acked)
            self.assertEqual(state.ack_sequence, sequence)

    def test_ack_leaves_attempts_and_attempt_ids_untouched(self) -> None:
        self._claim("L1", 2)
        # One redelivery attempt on m1 before the ack.
        self.service.group_inbox_retry_batch(
            "d3", {"attempt_id": "att-1",
                   "items": [{"session_id": self.gs1,
                              "message_id": "m1"}]})
        self._complete("L1", "delivered")
        self._ack()
        state = self.store._group_delivery[(self.gs1, "m1", "d3")]
        self.assertEqual(state.attempts, 1)
        self.assertEqual(state.attempt_ids, {"att-1"})
        self.assertTrue(state.acked)

    def test_message_count_covers_all_claimed_messages(self) -> None:
        self._claim("L1", 3)
        self._complete("L1", "delivered")
        body, status = self._ack()
        self.assertEqual(status, 201)
        self.assertEqual(body["message_count"], 3)

    def test_replay_returns_200_same_body(self) -> None:
        self._claim("L1", 2)
        self._complete("L1", "delivered")
        first, status = self._ack()
        self.assertEqual(status, 201)
        replay, status = self._ack()
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)

    def test_replay_succeeds_after_device_revoked(self) -> None:
        self._claim("L1", 2)
        self._complete("L1", "delivered")
        first, _ = self._ack()
        self.service.revoke_device("d3")
        replay, status = self._ack()
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)

    def test_already_fully_acked_answers_200_without_completion(
            self) -> None:
        # Every message acked individually before any completion: the
        # bulk ack is an idempotent 200 even though the lease was never
        # completed.
        self._claim("L1", 2)
        for message_id, sequence in (("m1", 1), ("m2", 2)):
            self.service.ack_message(
                self.gs1, {"device_id": "d3", "message_id": message_id,
                           "sequence": sequence})
        body, status = self._ack()
        self.assertEqual(status, 200)
        self.assertEqual(body["message_count"], 2)

    def test_partial_ack_without_completion_is_409(self) -> None:
        self._claim("L1", 2)
        self.service.ack_message(
            self.gs1, {"device_id": "d3", "message_id": "m1",
                       "sequence": 1})
        error = self._error(lambda: self._ack())
        self.assertEqual((error.status_code, error.field),
                         (409, "lease_id"))

    def test_partial_ack_then_delivered_completes_rest(self) -> None:
        self._claim("L1", 2)
        self.service.ack_message(
            self.gs1, {"device_id": "d3", "message_id": "m1",
                       "sequence": 1})
        self._complete("L1", "delivered")
        body, status = self._ack()
        self.assertEqual(status, 201)
        self.assertEqual(body["message_count"], 2)
        self.assertTrue(self.store._group_delivery[
            (self.gs1, "m2", "d3")].acked)

    def test_active_lease_without_completion_is_409(self) -> None:
        self._claim("L1", 2)
        error = self._error(lambda: self._ack())
        self.assertEqual((error.status_code, error.field),
                         (409, "lease_id"))

    def test_failed_completion_is_409_lease_id(self) -> None:
        self._claim("L1", 2)
        self._complete("L1", "failed")
        error = self._error(lambda: self._ack())
        self.assertEqual((error.status_code, error.field),
                         (409, "lease_id"))

    def test_released_lease_is_409_lease_id(self) -> None:
        self._claim("L1", 2)
        self.service.group_inbox_release("d3", "L1")
        error = self._error(lambda: self._ack())
        self.assertEqual((error.status_code, error.field),
                         (409, "lease_id"))

    def test_expired_uncompleted_lease_is_409_lease_id(self) -> None:
        self._claim("L1", 2)
        self._expire_group_lease("L1")
        error = self._error(lambda: self._ack())
        self.assertEqual((error.status_code, error.field),
                         (409, "lease_id"))

    def test_unknown_lease_is_404_lease_id(self) -> None:
        error = self._error(lambda: self._ack(lease_id="NOPE"))
        self.assertEqual((error.status_code, error.field),
                         (404, "lease_id"))

    def test_one2one_lease_id_is_409_lease_id(self) -> None:
        self.service.inbox_claim("d2", {"lease_id": "L1", "limit": 1})
        error = self._error(lambda: self._ack(lease_id="L1"))
        self.assertEqual((error.status_code, error.field),
                         (409, "lease_id"))

    def test_cross_device_is_409_lease_id(self) -> None:
        self._claim("L1", 2)
        self._complete("L1", "delivered")
        error = self._error(lambda: self._ack(device_id="d4"))
        self.assertEqual((error.status_code, error.field),
                         (409, "lease_id"))

    def test_delivered_lease_on_revoked_device_is_409_device_id(
            self) -> None:
        self._claim("L1", 2)
        self._complete("L1", "delivered")
        self.service.revoke_device("d3")
        error = self._error(lambda: self._ack())
        self.assertEqual((error.status_code, error.field),
                         (409, "device_id"))
        # The failed ack wrote nothing.
        for message_id in ("m1", "m2"):
            self.assertFalse(self.store._group_delivery[
                (self.gs1, message_id, "d3")].acked)

    def test_delivered_lease_on_unknown_device_is_409_device_id(
            self) -> None:
        # A lease owner that vanishes from the device index cannot happen
        # through the API, but the store reason still maps cleanly.
        self._claim("L1", 2)
        self._complete("L1", "delivered")
        index = self.service.store._device_index
        user_key = index.pop("d3")
        self.service.store._devices.pop(user_key)
        error = self._error(lambda: self._ack())
        self.assertEqual((error.status_code, error.field),
                         (409, "device_id"))

    def test_ack_removes_messages_from_group_inbox(self) -> None:
        self._claim("L1", 2)
        self._complete("L1", "delivered")
        self._ack()
        inbox = self.service.device_group_inbox("d3", 10)
        self.assertEqual([m["message_id"] for m in inbox["messages"]],
                         ["m3"])

    def test_delivered_unacked_messages_are_claimable(self) -> None:
        # A delivered completion is not an acknowledgement: until the
        # bulk ack lands, a new lease picks the messages up.
        self._claim("L1", 2)
        self._complete("L1", "delivered")
        body, status = self.service.group_inbox_claim(
            "d3", {"lease_id": "L2", "limit": 10})
        self.assertEqual(status, 201)
        self.assertEqual([m["message_id"] for m in body["messages"]],
                         ["m1", "m2", "m3"])
        self._ack(lease_id="L1")
        self.assertTrue(self.store._group_delivery[
            (self.gs1, "m1", "d3")].acked)


class GroupAckPersistenceTest(GroupAckMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self._build()
        self.path = os.path.join(self.directory, "state.json")
        self.state_store = attach_persistence(self.service, self.path)

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_first_ack_commits_one_generation(self) -> None:
        self._claim("L1", 2)
        self._complete("L1", "delivered")
        generation = self.state_store.commit_seq
        body, status = self._ack()
        self.assertEqual(status, 201)
        self.assertEqual(self.state_store.commit_seq, generation + 1)
        self.assertEqual(body["message_count"], 2)

    def test_replay_commits_nothing(self) -> None:
        self._claim("L1", 2)
        self._complete("L1", "delivered")
        self._ack()
        generation = self.state_store.commit_seq
        body, status = self._ack()
        self.assertEqual(status, 200)
        self.assertEqual(self.state_store.commit_seq, generation)

    def test_restart_restores_acks(self) -> None:
        self._claim("L1", 2)
        self._complete("L1", "delivered")
        first, _ = self._ack()
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        for message_id, sequence in (("m1", 1), ("m2", 2)):
            state = restarted.store._group_delivery[
                (self.gs1, message_id, "d3")]
            self.assertTrue(state.acked)
            self.assertEqual(state.ack_sequence, sequence)
        replay, status = restarted.group_inbox_lease_ack("d3", "L1")
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)

    def test_save_failure_rolls_back(self) -> None:
        import e2ee_backend.persistence as persistence_mod

        self._claim("L1", 2)
        self._complete("L1", "delivered")
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
                self._ack()
        finally:
            persistence_mod.os.fsync = real_fsync
        self.assertEqual(self.state_store.commit_seq, generation)
        # Nothing landed in memory: the records are still unacked and the
        # retry is a fresh 201.
        for message_id in ("m1", "m2"):
            self.assertFalse(self.store._group_delivery[
                (self.gs1, message_id, "d3")].acked)
        body, status = self._ack()
        self.assertEqual(status, 201)
        self.assertEqual(self.state_store.commit_seq, generation + 1)


class GroupAckHTTPTest(GroupAckMixin, unittest.TestCase):
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

    def _post(self, target, raw=None):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        if raw is None:
            conn.request("POST", target)
        else:
            conn.request("POST", target, body=raw,
                         headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        data = response.read().decode("utf-8")
        conn.close()
        return response.status, (json.loads(data) if data else None), data

    def _claim_and_complete(self, lease_id="L1", limit=2,
                            outcome="delivered"):
        status, _, _ = self._post(
            "/v1/devices/d3/group-inbox/claim",
            json.dumps({"lease_id": lease_id, "limit": limit}))
        self.assertEqual(status, 201)
        status, _, _ = self._post(
            f"/v1/devices/d3/group-inbox/leases/{lease_id}/complete",
            json.dumps({"completion_id": "C1", "outcome": outcome}))
        self.assertEqual(status, 201)

    def test_ack_and_replay_over_http(self) -> None:
        self._claim_and_complete()
        status, body, raw = self._post(
            "/v1/devices/d3/group-inbox/leases/L1/ack")
        self.assertEqual(status, 201)
        self.assertEqual(list(body),
                         ["device_id", "lease_id", "acked",
                          "message_count"])
        self.assertLess(raw.index('"device_id"'), raw.index('"lease_id"'))
        self.assertLess(raw.index('"acked"'),
                        raw.index('"message_count"'))
        status, replay, _ = self._post(
            "/v1/devices/d3/group-inbox/leases/L1/ack")
        self.assertEqual(status, 200)
        self.assertEqual(replay, body)

    def test_non_empty_body_is_400_request_body(self) -> None:
        self._claim_and_complete()
        status, body, _ = self._post(
            "/v1/devices/d3/group-inbox/leases/L1/ack", "{}")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "request_body")

    def test_query_parameter_is_400_query(self) -> None:
        self._claim_and_complete()
        status, body, _ = self._post(
            "/v1/devices/d3/group-inbox/leases/L1/ack?x=1")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "query")

    def test_unknown_lease_over_http(self) -> None:
        status, body, _ = self._post(
            "/v1/devices/d3/group-inbox/leases/NOPE/ack")
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "lease_id")

    def test_undelivered_lease_over_http(self) -> None:
        self._claim_and_complete(outcome="failed")
        status, body, _ = self._post(
            "/v1/devices/d3/group-inbox/leases/L1/ack")
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "lease_id")

    def test_percent_encoded_ids_over_http(self) -> None:
        status, _, _ = self._post(
            "/v1/devices/d3/group-inbox/claim",
            json.dumps({"lease_id": "a/b", "limit": 2}))
        self.assertEqual(status, 201)
        status, _, _ = self._post(
            f"/v1/devices/d3/group-inbox/leases/"
            f"{quote('a/b', safe='')}/complete",
            json.dumps({"completion_id": "C1", "outcome": "delivered"}))
        self.assertEqual(status, 201)
        status, body, _ = self._post(
            f"/v1/devices/d3/group-inbox/leases/"
            f"{quote('a/b', safe='')}/ack")
        self.assertEqual(status, 201)
        self.assertEqual(body["lease_id"], "a/b")


if __name__ == "__main__":
    unittest.main()
