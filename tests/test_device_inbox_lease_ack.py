"""Tests for the 1:1 inbox lease bulk-ack endpoint.

POST /v1/devices/{device_id}/inbox/leases/{lease_id}/ack acknowledges
every message of one lease in one locked transaction. The endpoint takes
no request body (a non-empty one is 400/request_body). An unknown lease
is 404/lease_id and a lease owned by another device is 409/lease_id,
both ahead of the path device's state. Only a lease completed with
``completion.outcome == "delivered"`` may be acked (active, expired,
released or ``failed`` -> 409/lease_id); a first ack on an unknown or
revoked device is 409/device_id. A lease already fully acked answers 200
and writes nothing even when the device was later revoked. A first ack
returns 201, sets every leased message's delivery record acked with
ack_sequence the message sequence (attempts/attempt ids untouched) and
commits one generation; a replay returns 200.
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


class AckMixin:
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

    def _complete(self, lease_id, outcome="delivered",
                  completion_id="C1", device_id="bob"):
        return self.service.inbox_lease_complete(
            device_id, lease_id,
            {"completion_id": completion_id, "outcome": outcome})

    def _ack(self, device_id="bob", lease_id="L1"):
        return self.service.inbox_lease_ack(device_id, lease_id)

    def _expire(self, lease_id) -> None:
        for state in self.service.store._delivery.values():
            for lease in state.leases:
                if lease.lease_id == lease_id:
                    lease.leased_until = "2000-01-01T00:00:00.000000+00:00"


class AckServiceTest(AckMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def test_first_ack_after_delivered_is_201(self) -> None:
        self._claim("L1", 2)
        self._complete("L1", "delivered")
        body, status = self._ack()
        self.assertEqual(status, 201)
        self.assertEqual(list(body),
                         ["device_id", "lease_id", "acked", "message_count"])
        self.assertEqual(body, {"device_id": "bob", "lease_id": "L1",
                                "acked": True, "message_count": 2})

    def test_message_count_covers_all_claimed_messages(self) -> None:
        self._claim("L1", 10)  # all five messages across two sessions
        self._complete("L1", "delivered")
        body, status = self._ack()
        self.assertEqual(status, 201)
        self.assertEqual(body["message_count"], 5)
        self.assertIs(body["acked"], True)

    def test_ack_sets_acked_and_ack_sequence_per_message(self) -> None:
        self._claim("L1", 10)
        self._complete("L1", "delivered")
        self._ack()
        expected = {(self.sid1, "a1"): 1, (self.sid1, "a2"): 2,
                    (self.sid1, "a3"): 3, (self.sid2, "b1"): 1,
                    (self.sid2, "b2"): 2}
        for (session_id, message_id), sequence in expected.items():
            delivery = self.service.store._delivery[(session_id, message_id)]
            self.assertIs(delivery.acked, True)
            self.assertEqual(delivery.ack_sequence, sequence)

    def test_ack_leaves_attempts_and_attempt_ids_untouched(self) -> None:
        self._claim("L1", 2)
        self.service.retry_message(
            self.sid1, "a1", {"device_id": "bob", "attempt_id": "at1"})
        self.service.retry_message(
            self.sid1, "a2", {"device_id": "bob", "attempt_id": "at2"})
        self._complete("L1", "delivered")
        self._ack()
        first = self.service.store._delivery[(self.sid1, "a1")]
        second = self.service.store._delivery[(self.sid1, "a2")]
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
        self.service.revoke_device("bob")
        replay, status = self._ack()
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)

    def test_already_fully_acked_answers_200_without_completion(self) -> None:
        # Every message acked through the sync path before any completion:
        # the bulk ack is idempotent even though the lease is still active.
        self._claim("L1", 2)
        self.service.sync_session_ack(
            self.sid1, {"device_id": "bob", "cursor": 2})
        body, status = self._ack()
        self.assertEqual(status, 200)
        self.assertEqual(body["message_count"], 2)

    def test_fully_acked_200_preceded_revocation_without_completion(
            self) -> None:
        self._claim("L1", 2)
        self.service.sync_session_ack(
            self.sid1, {"device_id": "bob", "cursor": 2})
        self.service.revoke_device("bob")
        body, status = self._ack()
        self.assertEqual(status, 200)
        self.assertEqual(body["acked"], True)

    def test_partial_ack_then_delivered_completes_rest(self) -> None:
        # Claim a1/a2; only a1 is acked through sync. A delivered
        # completion followed by bulk ack acks a2 too and reports both.
        self._claim("L1", 2)
        self.service.sync_session_ack(
            self.sid1, {"device_id": "bob", "cursor": 1})
        self._complete("L1", "delivered")
        body, status = self._ack()
        self.assertEqual(status, 201)
        self.assertEqual(body["message_count"], 2)
        second = self.service.store._delivery[(self.sid1, "a2")]
        self.assertTrue(second.acked)
        self.assertEqual(second.ack_sequence, 2)

    def test_partial_ack_without_completion_is_409(self) -> None:
        self._claim("L1", 2)
        self.service.sync_session_ack(
            self.sid1, {"device_id": "bob", "cursor": 1})
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
        # Nothing was acked.
        for message_id in ("a1", "a2"):
            self.assertFalse(
                self.service.store._delivery[(self.sid1, message_id)].acked)

    def test_released_lease_is_409_lease_id(self) -> None:
        self._claim("L1", 2)
        self.service.inbox_release("bob", "L1")
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
            self._ack("bob", "NOPE")
        self.assertEqual(caught.exception.status_code, 404)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_unknown_lease_404_precedes_unknown_device(self) -> None:
        with self.assertRaises(ServiceError) as caught:
            self._ack("ghost", "NOPE")
        self.assertEqual(caught.exception.status_code, 404)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_cross_device_is_409_lease_id(self) -> None:
        self._claim("L1", 2)
        with self.assertRaises(ServiceError) as caught:
            self._ack("carol", "L1")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_cross_device_conflict_precedes_path_revocation(self) -> None:
        self._claim("L1", 2)
        self._complete("L1", "delivered")
        self.service.revoke_device("carol")
        with self.assertRaises(ServiceError) as caught:
            self._ack("carol", "L1")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_delivered_lease_on_revoked_device_is_409_device_id(self) -> None:
        self._claim("L1", 2)
        self._complete("L1", "delivered")
        self.service.revoke_device("bob")
        with self.assertRaises(ServiceError) as caught:
            self._ack()
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "device_id")
        # The failed ack wrote nothing: replaying completion is still 200
        # and the messages stay unacked.
        for message_id in ("a1", "a2"):
            self.assertFalse(
                self.service.store._delivery[(self.sid1, message_id)].acked)

    def test_delivered_lease_on_unknown_device_is_409_device_id(self) -> None:
        # A lease owner that vanishes from the device index cannot happen
        # through the API, but the store reason still maps cleanly.
        self._claim("L1", 2)
        self._complete("L1", "delivered")
        index = self.service.store._device_index
        user_key = index.pop("bob")
        self.service.store._devices.pop(user_key)
        with self.assertRaises(ServiceError) as caught:
            self._ack()
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "device_id")

    def test_ack_removes_messages_from_inbox(self) -> None:
        self._claim("L1", 2)
        self._complete("L1", "delivered")
        self._ack()
        inbox = self.service.device_inbox("bob", 100)
        self.assertEqual([m["message_id"] for m in inbox["messages"]],
                         ["a3", "b1", "b2"])


class AckConcurrencyTest(AckMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()
        self._claim("L1", 10)
        self._complete("L1", "delivered")

    def test_concurrent_acks_at_most_one_201(self) -> None:
        results = []
        barrier = threading.Barrier(8)

        def worker() -> None:
            barrier.wait()
            _body, status = self.service.inbox_lease_ack("bob", "L1")
            results.append(status)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
        self.assertEqual(sorted(results).count(201), 1)
        self.assertEqual(sorted(results).count(200), 7)
        for message_id in ("a1", "a2", "a3"):
            delivery = self.service.store._delivery[(self.sid1, message_id)]
            self.assertTrue(delivery.acked)
            self.assertEqual(delivery.ack_sequence, int(message_id[1]))


class AckPersistenceTest(AckMixin, unittest.TestCase):
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
        acked = [record for record in document["delivery"]
                 if record["session_id"] == self.sid1
                 and record["message_id"] in ("a1", "a2")]
        self.assertEqual(len(acked), 2)
        sequences = {record["message_id"]: record["ack_sequence"]
                     for record in acked}
        self.assertEqual(sequences, {"a1": 1, "a2": 2})
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
        replay, status = restarted.inbox_lease_ack("bob", "L1")
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)
        # The acked messages leave the rebuilt inbox.
        inbox = restarted.device_inbox("bob", 100)
        self.assertNotIn("a1",
                         [m["message_id"] for m in inbox["messages"]])
        self.assertNotIn("a2",
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
        for message_id in ("a1", "a2"):
            delivery = self.service.store._delivery[
                (self.sid1, message_id)]
            self.assertFalse(delivery.acked)
            self.assertEqual(delivery.ack_sequence, 0)
        # The lease/state files are still coherent: retry succeeds once.
        body, status = self._ack()
        self.assertEqual(status, 201)
        self.assertEqual(self.state_store.commit_seq, generation + 1)
        self.assertEqual(body["message_count"], 2)


class AckHTTPTest(AckMixin, unittest.TestCase):
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

    def _request(self, device_id, lease_id, raw=None):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request(
            "POST",
            f"/v1/devices/{device_id}/inbox/leases/{lease_id}/ack",
            body=raw)
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

    def _complete(self, lease_id, outcome="delivered"):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request(
            "POST",
            f"/v1/devices/bob/inbox/leases/{lease_id}/complete",
            body=json.dumps({"completion_id": "C1", "outcome": outcome}),
            headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        response.read()
        conn.close()
        self.assertEqual(response.status, 201)

    def test_ack_and_replay_over_http(self) -> None:
        self._claim("L1", 2)
        self._complete("L1", "delivered")
        status, body, raw = self._request("bob", "L1")
        self.assertEqual(status, 201)
        self.assertEqual(list(body),
                         ["device_id", "lease_id", "acked", "message_count"])
        self.assertLess(raw.index('"device_id"'), raw.index('"lease_id"'))
        self.assertLess(raw.index('"lease_id"'), raw.index('"acked"'))
        self.assertLess(raw.index('"acked"'),
                        raw.index('"message_count"'))
        self.assertEqual(body["acked"], True)
        self.assertEqual(body["message_count"], 2)
        status, replay, replay_raw = self._request("bob", "L1")
        self.assertEqual(status, 200)
        self.assertEqual(replay_raw, raw)

    def test_non_empty_body_is_400_request_body(self) -> None:
        self._claim("L1", 2)
        self._complete("L1", "delivered")
        for raw in ("{}", "x", json.dumps({"lease_id": "L1"})):
            status, body, _ = self._request("bob", "L1", raw=raw)
            self.assertEqual(status, 400, raw)
            self.assertEqual(list(body), ["message", "field"])
            self.assertEqual(body["field"], "request_body")
        # The rejections consumed nothing: the ack still goes through.
        status, _body, _ = self._request("bob", "L1")
        self.assertEqual(status, 201)

    def test_unknown_lease_over_http(self) -> None:
        status, body, _ = self._request("bob", "NOPE")
        self.assertEqual(status, 404)
        self.assertEqual(list(body), ["message", "field"])
        self.assertEqual(body["field"], "lease_id")

    def test_cross_device_over_http(self) -> None:
        self._claim("L1", 2)
        self._complete("L1", "delivered")
        status, body, _ = self._request("carol", "L1")
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "lease_id")

    def test_undelivered_lease_over_http(self) -> None:
        self._claim("L1", 2)
        status, body, _ = self._request("bob", "L1")
        self.assertEqual(status, 409)
        self.assertEqual(list(body), ["message", "field"])
        self.assertEqual(body["field"], "lease_id")

    def test_revoked_device_over_http(self) -> None:
        self._claim("L1", 2)
        self._complete("L1", "delivered")
        self.service.store.revoke_device("bob")
        status, body, _ = self._request("bob", "L1")
        self.assertEqual(status, 409)
        self.assertEqual(list(body), ["message", "field"])
        self.assertEqual(body["field"], "device_id")

    def test_percent_encoded_ids_over_http(self) -> None:
        self._claim("a/b", 1)
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request(
            "POST",
            f"/v1/devices/bob/inbox/leases/{quote('a/b', safe='')}/complete",
            body=json.dumps({"completion_id": "C1", "outcome": "delivered"}),
            headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        response.read()
        conn.close()
        self.assertEqual(response.status, 201)
        status, body, _ = self._request("bob", "a%2Fb")
        self.assertEqual(status, 201)
        self.assertEqual(body["lease_id"], "a/b")
        self.assertEqual(body["message_count"], 1)


if __name__ == "__main__":
    unittest.main()
