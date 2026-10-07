"""Tests for the group-inbox lease bulk-ack endpoint.

POST /v1/devices/{device_id}/group-inbox/leases/{lease_id}/ack
acknowledges every group message of one lease in one locked
transaction. The endpoint takes no request body (a non-empty one is
400/request_body) and no query parameters (any -> 400/query). An
unknown lease is 404/lease_id and a lease owned by another device (or
committed in the 1:1 namespace) is 409/lease_id, both ahead of the
path device's state. Only a lease completed with
``completion.outcome == "delivered"`` may be acked (active, expired,
released or ``failed`` -> 409/lease_id); a first ack on an unknown or
revoked device is 409/device_id. A lease already fully acked answers
200 and writes nothing even when the device was later revoked. A first
ack returns 201, sets every leased message's group delivery record
acked with ack_sequence the message sequence (attempts/attempt ids
untouched) and commits one generation; a replay returns 200.
"""
import json
import os
import shutil
import tempfile
import threading
import unittest
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
                  completion_id="C1", device_id="d3"):
        return self.service.group_inbox_lease_complete(
            device_id, lease_id,
            {"completion_id": completion_id, "outcome": outcome})

    def _ack(self, device_id="d3", lease_id="L1"):
        return self.service.group_inbox_lease_ack(device_id, lease_id)

    def _expire(self, lease_id) -> None:
        for state in self.store._group_delivery.values():
            for lease in state.leases:
                if lease.lease_id == lease_id:
                    lease.leased_until = "2000-01-01T00:00:00.000000+00:00"

    def _delivery(self, message_id, device_id="d3"):
        return self.store._group_delivery[(self.gs1, message_id, device_id)]


class GroupAckServiceTest(GroupAckMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def test_first_ack_after_delivered_is_201(self) -> None:
        self._claim("L1", 2)
        self._complete("L1", "delivered")
        body, status = self._ack()
        self.assertEqual(status, 201)
        self.assertEqual(list(body),
                         ["device_id", "lease_id", "acked", "message_count"])
        self.assertEqual(body, {"device_id": "d3", "lease_id": "L1",
                                "acked": True, "message_count": 2})

    def test_message_count_covers_all_claimed_messages(self) -> None:
        self._claim("L1", 10)  # all three group messages
        self._complete("L1", "delivered")
        body, status = self._ack()
        self.assertEqual(status, 201)
        self.assertEqual(body["message_count"], 3)
        self.assertIs(body["acked"], True)

    def test_ack_sets_acked_and_ack_sequence_per_message(self) -> None:
        self._claim("L1", 10)
        self._complete("L1", "delivered")
        self._ack()
        for message_id, sequence in (("m1", 1), ("m2", 2), ("m3", 3)):
            delivery = self._delivery(message_id)
            self.assertIs(delivery.acked, True)
            self.assertEqual(delivery.ack_sequence, sequence)

    def test_ack_leaves_attempts_and_attempt_ids_untouched(self) -> None:
        self._claim("L1", 2)
        self.service.group_inbox_retry_batch("d3", {
            "attempt_id": "at1",
            "items": [{"session_id": self.gs1, "message_id": "m1"}]})
        self.service.group_inbox_retry_batch("d3", {
            "attempt_id": "at2",
            "items": [{"session_id": self.gs1, "message_id": "m2"}]})
        self._complete("L1", "delivered")
        self._ack()
        first = self._delivery("m1")
        second = self._delivery("m2")
        self.assertEqual(first.attempts, 1)
        self.assertEqual(set(first.attempt_ids), {"at1"})
        self.assertEqual(second.attempts, 1)
        self.assertEqual(set(second.attempt_ids), {"at2"})

    def test_replay_returns_200_same_body(self) -> None:
        self._claim("L1", 2)
        self._complete("L1", "delivered")
        first, first_status = self._ack()
        self.assertEqual(first_status, 201)
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
        # Every leased message acked through the ordinary group sync ack
        # before any completion: the lease ack is an idempotent 200.
        self._claim("L1", 2)
        self.service.sync_group_ack_messages("d3", {"items": [
            {"session_id": self.gs1, "message_id": "m1", "sequence": 1},
            {"session_id": self.gs1, "message_id": "m2", "sequence": 2}]})
        body, status = self._ack()
        self.assertEqual(status, 200)
        self.assertEqual(body, {"device_id": "d3", "lease_id": "L1",
                                "acked": True, "message_count": 2})

    def test_fully_acked_200_precedes_revocation_without_completion(
            self) -> None:
        self._claim("L1", 2)
        self.service.sync_group_ack_messages("d3", {"items": [
            {"session_id": self.gs1, "message_id": "m1", "sequence": 1},
            {"session_id": self.gs1, "message_id": "m2", "sequence": 2}]})
        self.service.revoke_device("d3")
        body, status = self._ack()
        self.assertEqual(status, 200)
        self.assertEqual(body["message_count"], 2)

    def test_partial_ack_then_delivered_completes_rest(self) -> None:
        self._claim("L1", 3)
        self.service.sync_group_ack_messages("d3", {"items": [
            {"session_id": self.gs1, "message_id": "m1", "sequence": 1}]})
        self._complete("L1", "delivered")
        body, status = self._ack()
        self.assertEqual(status, 201)
        self.assertEqual(body["message_count"], 3)
        for message_id in ("m1", "m2", "m3"):
            self.assertTrue(self._delivery(message_id).acked)

    def test_partial_ack_without_completion_is_409(self) -> None:
        self._claim("L1", 3)
        self.service.sync_group_ack_messages("d3", {"items": [
            {"session_id": self.gs1, "message_id": "m1", "sequence": 1}]})
        with self.assertRaises(ServiceError) as caught:
            self._ack()
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_active_lease_without_completion_is_409(self) -> None:
        self._claim("L1", 2)
        with self.assertRaises(ServiceError) as caught:
            self._ack()
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_failed_completion_is_409_lease_id(self) -> None:
        self._claim("L1", 2)
        self._complete("L1", "failed")
        with self.assertRaises(ServiceError) as caught:
            self._ack()
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")
        for message_id in ("m1", "m2"):
            self.assertFalse(self._delivery(message_id).acked)

    def test_released_lease_is_409_lease_id(self) -> None:
        self._claim("L1", 2)
        self.service.group_inbox_release("d3", "L1")
        with self.assertRaises(ServiceError) as caught:
            self._ack()
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_expired_uncompleted_lease_is_409_lease_id(self) -> None:
        self._claim("L1", 2)
        self._expire("L1")
        with self.assertRaises(ServiceError) as caught:
            self._ack()
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_unknown_lease_is_404_lease_id(self) -> None:
        with self.assertRaises(ServiceError) as caught:
            self._ack(lease_id="NOPE")
        self.assertEqual(caught.exception.status_code, 404)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_unknown_lease_404_precedes_unknown_device(self) -> None:
        with self.assertRaises(ServiceError) as caught:
            self._ack(device_id="ghost", lease_id="NOPE")
        self.assertEqual(caught.exception.status_code, 404)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_cross_device_is_409_lease_id(self) -> None:
        self._claim("L1", 2)
        self._complete("L1", "delivered")
        with self.assertRaises(ServiceError) as caught:
            self._ack(device_id="d2")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_cross_device_conflict_precedes_path_revocation(self) -> None:
        self._claim("L1", 2)
        self._complete("L1", "delivered")
        self.service.revoke_device("d2")
        with self.assertRaises(ServiceError) as caught:
            self._ack(device_id="d2")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_one_to_one_lease_id_is_409_lease_id(self) -> None:
        # An id committed in the 1:1 namespace conflicts, regardless of
        # the path device's state.
        self.service.inbox_claim("d2", {"lease_id": "X1", "limit": 5})
        with self.assertRaises(ServiceError) as caught:
            self._ack(device_id="d2", lease_id="X1")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_delivered_lease_on_revoked_device_is_409_device_id(
            self) -> None:
        self._claim("L1", 2)
        self._complete("L1", "delivered")
        self.service.revoke_device("d3")
        with self.assertRaises(ServiceError) as caught:
            self._ack()
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "device_id")
        # The failed ack wrote nothing: replaying completion is still 200
        # and the messages stay unacked.
        for message_id in ("m1", "m2"):
            self.assertFalse(self._delivery(message_id).acked)

    def test_delivered_lease_on_unknown_device_is_409_device_id(
            self) -> None:
        # A lease owner that vanishes from the device index cannot happen
        # through the API, but the store reason still maps cleanly.
        self._claim("L1", 2)
        self._complete("L1", "delivered")
        index = self.store._device_index
        user_key = index.pop("d3")
        self.store._devices.pop(user_key)
        with self.assertRaises(ServiceError) as caught:
            self._ack()
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "device_id")

    def test_ack_removes_messages_from_group_inbox(self) -> None:
        self._claim("L1", 2)
        self._complete("L1", "delivered")
        self._ack()
        inbox = self.service.device_group_inbox("d3", 100)
        self.assertEqual([m["message_id"] for m in inbox["messages"]],
                         ["m3"])


class GroupAckConcurrencyTest(GroupAckMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()
        self._claim("L1", 10)
        self._complete("L1", "delivered")

    def test_concurrent_acks_at_most_one_201(self) -> None:
        results = []
        barrier = threading.Barrier(8)

        def worker() -> None:
            barrier.wait()
            _body, status = self.service.group_inbox_lease_ack("d3", "L1")
            results.append(status)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
        self.assertEqual(sorted(results).count(201), 1)
        self.assertEqual(sorted(results).count(200), 7)
        for message_id, sequence in (("m1", 1), ("m2", 2), ("m3", 3)):
            delivery = self._delivery(message_id)
            self.assertTrue(delivery.acked)
            self.assertEqual(delivery.ack_sequence, sequence)


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
        before = self.state_store.commit_seq
        body, status = self._ack()
        self.assertEqual(status, 201)
        self.assertEqual(self.state_store.commit_seq, before + 1)
        with open(self.path, encoding="utf-8") as handle:
            document = json.load(handle)
        acked = [record for record in document["group_delivery"]
                 if record["session_id"] == self.gs1
                 and record["device_id"] == "d3"
                 and record["message_id"] in ("m1", "m2")]
        self.assertEqual(len(acked), 2)
        sequences = {record["message_id"]: record["ack_sequence"]
                     for record in acked}
        self.assertEqual(sequences, {"m1": 1, "m2": 2})
        self.assertTrue(all(record["acked"] for record in acked))

    def test_replay_commits_nothing(self) -> None:
        self._claim("L1", 2)
        self._complete("L1", "delivered")
        first, _ = self._ack()
        generation = self.state_store.commit_seq
        replay, status = self._ack()
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)
        self.assertEqual(self.state_store.commit_seq, generation)

    def test_restart_restores_acks(self) -> None:
        self._claim("L1", 2)
        self._complete("L1", "delivered")
        first, _ = self._ack()
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        replay, status = restarted.group_inbox_lease_ack("d3", "L1")
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)
        # The acked messages leave the rebuilt group inbox.
        inbox = restarted.device_group_inbox("d3", 100)
        self.assertNotIn("m1",
                         [m["message_id"] for m in inbox["messages"]])
        self.assertNotIn("m2",
                         [m["message_id"] for m in inbox["messages"]])

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
        for message_id in ("m1", "m2"):
            delivery = self._delivery(message_id)
            self.assertFalse(delivery.acked)
            self.assertEqual(delivery.ack_sequence, 0)
        # The lease/state files are still coherent: retry succeeds once.
        body, status = self._ack()
        self.assertEqual(status, 201)
        self.assertEqual(self.state_store.commit_seq, generation + 1)
        self.assertEqual(body["message_count"], 2)


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

    def _request(self, device_id, lease_id, raw=None, query=""):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request(
            "POST",
            f"/v1/devices/{device_id}/group-inbox/leases/"
            f"{lease_id}/ack{query}",
            body=raw)
        response = conn.getresponse()
        data = response.read().decode("utf-8")
        conn.close()
        return response.status, (json.loads(data) if data else None), data

    def _claim(self, lease_id="L1", limit=2, device_id="d3"):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request(
            "POST", f"/v1/devices/{device_id}/group-inbox/claim",
            body=json.dumps({"lease_id": lease_id, "limit": limit}),
            headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        response.read()
        conn.close()
        self.assertEqual(response.status, 201)

    def _complete(self, lease_id, outcome="delivered", device_id="d3"):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request(
            "POST",
            f"/v1/devices/{device_id}/group-inbox/leases/"
            f"{lease_id}/complete",
            body=json.dumps({"completion_id": "C1", "outcome": outcome}),
            headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        response.read()
        conn.close()
        self.assertEqual(response.status, 201)

    def test_ack_and_replay_over_http(self) -> None:
        self._claim("L1", 2)
        self._complete("L1", "delivered")
        status, body, raw = self._request("d3", "L1")
        self.assertEqual(status, 201)
        self.assertEqual(list(body),
                         ["device_id", "lease_id", "acked", "message_count"])
        self.assertLess(raw.index('"device_id"'), raw.index('"lease_id"'))
        self.assertLess(raw.index('"lease_id"'), raw.index('"acked"'))
        self.assertLess(raw.index('"acked"'),
                        raw.index('"message_count"'))
        self.assertEqual(body["acked"], True)
        self.assertEqual(body["message_count"], 2)
        status, replay, replay_raw = self._request("d3", "L1")
        self.assertEqual(status, 200)
        self.assertEqual(replay_raw, raw)

    def test_non_empty_body_is_400_request_body(self) -> None:
        self._claim("L1", 2)
        self._complete("L1", "delivered")
        for raw in ("{}", "x", json.dumps({"lease_id": "L1"})):
            status, body, _ = self._request("d3", "L1", raw=raw)
            self.assertEqual(status, 400, raw)
            self.assertEqual(list(body), ["message", "field"])
            self.assertEqual(body["field"], "request_body")
        # The rejections consumed nothing: the ack still goes through.
        status, _body, _ = self._request("d3", "L1")
        self.assertEqual(status, 201)

    def test_query_parameters_are_400_query(self) -> None:
        self._claim("L1", 2)
        self._complete("L1", "delivered")
        for query in ("?foo", "?lease_id=L1", "?x="):
            with self.subTest(query=query):
                status, body, _ = self._request("d3", "L1", query=query)
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], "query")
        # A bare trailing '?' carries no parameter and is accepted.
        status, _, _ = self._request("d3", "L1", query="?")
        self.assertEqual(status, 201)

    def test_unknown_lease_over_http(self) -> None:
        status, body, _ = self._request("d3", "NOPE")
        self.assertEqual(status, 404)
        self.assertEqual(list(body), ["message", "field"])
        self.assertEqual(body["field"], "lease_id")

    def test_cross_device_over_http(self) -> None:
        self._claim("L1", 2)
        self._complete("L1", "delivered")
        status, body, _ = self._request("d2", "L1")
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "lease_id")

    def test_undelivered_lease_over_http(self) -> None:
        self._claim("L1", 2)
        status, body, _ = self._request("d3", "L1")
        self.assertEqual(status, 409)
        self.assertEqual(list(body), ["message", "field"])
        self.assertEqual(body["field"], "lease_id")

    def test_failed_completion_over_http(self) -> None:
        self._claim("L1", 2)
        self._complete("L1", "failed")
        status, body, _ = self._request("d3", "L1")
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "lease_id")

    def test_revoked_device_over_http(self) -> None:
        self._claim("L1", 2)
        self._complete("L1", "delivered")
        self.service.store.revoke_device("d3")
        status, body, _ = self._request("d3", "L1")
        self.assertEqual(status, 409)
        self.assertEqual(list(body), ["message", "field"])
        self.assertEqual(body["field"], "device_id")

    def test_bad_path_escapes_are_400_by_segment(self) -> None:
        self._claim("L1", 2)
        self._complete("L1", "delivered")
        # Bad device_id escape is reported first.
        status, body, _ = self._request("d%zz", "L1")
        self.assertEqual((status, body["field"]), (400, "device_id"))
        # Invalid UTF-8 in the device segment.
        status, body, _ = self._request("d%ff", "L1")
        self.assertEqual((status, body["field"]), (400, "device_id"))
        # A valid device segment reaches the lease-segment check.
        status, body, _ = self._request("d3", "L%zz")
        self.assertEqual((status, body["field"]), (400, "lease_id"))
        status, body, _ = self._request("d3", "L%ff")
        self.assertEqual((status, body["field"]), (400, "lease_id"))

    def test_percent_encoded_ids_over_http(self) -> None:
        self._claim("a/b", 1)
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request(
            "POST",
            f"/v1/devices/d3/group-inbox/leases/"
            f"{quote('a/b', safe='')}/complete",
            body=json.dumps({"completion_id": "C1", "outcome": "delivered"}),
            headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        response.read()
        conn.close()
        self.assertEqual(response.status, 201)
        status, body, _ = self._request("d3", "a%2Fb")
        self.assertEqual(status, 201)
        self.assertEqual(body["lease_id"], "a/b")
        self.assertEqual(body["message_count"], 1)


if __name__ == "__main__":
    unittest.main()
