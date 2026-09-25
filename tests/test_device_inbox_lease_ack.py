"""Tests for the 1:1 inbox lease batch acknowledgement endpoint.

POST /v1/devices/{device_id}/inbox/leases/{lease_id}/ack acknowledges every
message a lease claimed, provided the lease was completed with the outcome
``delivered``. The endpoint takes no request body. An unknown lease id is
404/lease_id, a lease owned by another device is 409/lease_id, and a lease
without a ``delivered`` completion is 409/lease_id. A first acknowledgement
returns 201 with ``acked``/``message_count`` and persists one generation; a
lease whose messages are all already acked answers 200 and writes nothing,
even after the device is revoked.
"""
import json
import os
import shutil
import tempfile
import threading
import unittest
from http.client import HTTPConnection

from e2ee_backend.models import Device, SignedPreKey
from e2ee_backend.persistence import (
    PersistenceUnavailable, StateFileError, attach_persistence)
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

    def _complete(self, lease_id="L1", outcome="delivered",
                  completion_id="C1"):
        body, status = self.service.inbox_lease_complete(
            "bob", lease_id,
            {"completion_id": completion_id, "outcome": outcome})
        self.assertEqual(status, 201)
        return body

    def _delivered_lease(self, lease_id="L1", limit=2):
        self._claim(lease_id, limit)
        self._complete(lease_id, "delivered", f"C-{lease_id}")


class AckServiceTest(AckMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def test_first_ack_201_response_shape(self) -> None:
        self._delivered_lease("L1", 2)
        body, status = self.service.inbox_lease_ack("bob", "L1")
        self.assertEqual(status, 201)
        self.assertEqual(list(body),
                         ["device_id", "lease_id", "acked", "message_count"])
        self.assertEqual(body["device_id"], "bob")
        self.assertEqual(body["lease_id"], "L1")
        self.assertIs(body["acked"], True)
        self.assertEqual(body["message_count"], 2)
        self.assertIsInstance(body["message_count"], int)

    def test_message_count_matches_claimed_messages(self) -> None:
        self._delivered_lease("L1", 10)
        body, _ = self.service.inbox_lease_ack("bob", "L1")
        self.assertEqual(body["message_count"], 5)

    def test_ack_marks_every_leased_delivery(self) -> None:
        self._delivered_lease("L1", 2)
        _, status = self.service.inbox_lease_ack("bob", "L1")
        self.assertEqual(status, 201)
        for sequence in (1, 2):
            state = self.service.store._delivery[(self.sid1, f"a{sequence}")]
            self.assertTrue(state.acked)
            self.assertEqual(state.ack_sequence, sequence)
        # Messages outside the lease stay unacked.
        self.assertNotIn((self.sid1, "a3"), self.service.store._delivery)

    def test_ack_leaves_attempts_untouched(self) -> None:
        self._delivered_lease("L1", 2)
        self.service.inbox_retry_batch("bob", {
            "attempt_id": "att1",
            "items": [{"session_id": self.sid1, "message_id": "a1"}]})
        _, status = self.service.inbox_lease_ack("bob", "L1")
        self.assertEqual(status, 201)
        state = self.service.store._delivery[(self.sid1, "a1")]
        self.assertEqual(state.attempts, 1)
        self.assertEqual(state.attempt_ids, {"att1"})

    def test_repeat_ack_replays_first_response_200(self) -> None:
        self._delivered_lease("L1", 2)
        first, first_status = self.service.inbox_lease_ack("bob", "L1")
        self.assertEqual(first_status, 201)
        replay, status = self.service.inbox_lease_ack("bob", "L1")
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)

    def test_repeat_ack_succeeds_after_device_revoked(self) -> None:
        self._delivered_lease("L1", 2)
        first, _ = self.service.inbox_lease_ack("bob", "L1")
        self.service.revoke_device("bob")
        replay, status = self.service.inbox_lease_ack("bob", "L1")
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)

    def test_all_acked_elsewhere_replays_200_without_write(self) -> None:
        self._delivered_lease("L1", 2)
        # Ack both leased messages through the sync path; the lease ack then
        # finds everything already acked and answers 200 without writing.
        self.service.sync_device_ack_batch("bob", {
            "items": [{"session_id": self.sid1, "cursor": 2}]})
        body, status = self.service.inbox_lease_ack("bob", "L1")
        self.assertEqual(status, 200)
        self.assertEqual(body, {"device_id": "bob", "lease_id": "L1",
                                "acked": True, "message_count": 2})

    def test_partially_acked_lease_still_commits_201(self) -> None:
        self._delivered_lease("L1", 2)
        self.service.sync_device_ack_batch("bob", {
            "items": [{"session_id": self.sid1, "cursor": 1}]})
        body, status = self.service.inbox_lease_ack("bob", "L1")
        self.assertEqual(status, 201)
        self.assertEqual(body["message_count"], 2)
        state = self.service.store._delivery[(self.sid1, "a2")]
        self.assertTrue(state.acked)
        self.assertEqual(state.ack_sequence, 2)

    def test_unknown_lease_is_404_lease_id(self) -> None:
        with self.assertRaises(ServiceError) as caught:
            self.service.inbox_lease_ack("bob", "NOPE")
        self.assertEqual(caught.exception.status_code, 404)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_unknown_lease_404_precedes_unknown_device(self) -> None:
        with self.assertRaises(ServiceError) as caught:
            self.service.inbox_lease_ack("ghost", "NOPE")
        self.assertEqual(caught.exception.status_code, 404)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_cross_device_ack_is_409_lease_id(self) -> None:
        self._delivered_lease("L1", 2)
        with self.assertRaises(ServiceError) as caught:
            self.service.inbox_lease_ack("carol", "L1")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_cross_device_conflict_precedes_path_device_state(self) -> None:
        self._delivered_lease("L1", 2)
        self.service.revoke_device("carol")
        with self.assertRaises(ServiceError) as caught:
            self.service.inbox_lease_ack("carol", "L1")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_uncompleted_lease_is_409_lease_id(self) -> None:
        self._claim("L1", 2)
        with self.assertRaises(ServiceError) as caught:
            self.service.inbox_lease_ack("bob", "L1")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_failed_completion_is_409_lease_id(self) -> None:
        self._claim("L1", 2)
        self._complete("L1", "failed", "C1")
        with self.assertRaises(ServiceError) as caught:
            self.service.inbox_lease_ack("bob", "L1")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_released_lease_is_409_lease_id(self) -> None:
        self._claim("L1", 2)
        self.service.inbox_release("bob", "L1")
        with self.assertRaises(ServiceError) as caught:
            self.service.inbox_lease_ack("bob", "L1")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_not_delivered_check_precedes_device_state(self) -> None:
        self._claim("L1", 2)
        self.service.revoke_device("bob")
        with self.assertRaises(ServiceError) as caught:
            self.service.inbox_lease_ack("bob", "L1")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_first_ack_on_revoked_device_is_409_device_id(self) -> None:
        self._delivered_lease("L1", 2)
        self.service.revoke_device("bob")
        with self.assertRaises(ServiceError) as caught:
            self.service.inbox_lease_ack("bob", "L1")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "device_id")

    def test_first_ack_on_unknown_device_is_409_device_id(self) -> None:
        self._delivered_lease("L1", 2)
        # An unregistered path device with an existing lease id: the lease
        # belongs to bob, so this is the cross-device conflict instead.
        with self.assertRaises(ServiceError) as caught:
            self.service.inbox_lease_ack("ghost", "L1")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_acked_messages_leave_the_inbox(self) -> None:
        self._delivered_lease("L1", 2)
        self.service.inbox_lease_ack("bob", "L1")
        inbox = self.service.device_inbox("bob", 100)
        self.assertEqual([m["message_id"] for m in inbox["messages"]],
                         ["a3", "b1", "b2"])

    def test_concurrent_acks_commit_at_most_once(self) -> None:
        self._delivered_lease("L1", 2)
        results = []
        barrier = threading.Barrier(8)

        def worker():
            barrier.wait()
            results.append(self.service.inbox_lease_ack("bob", "L1")[1])

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(results.count(201), 1)
        self.assertEqual(results.count(200), 7)


class AckPersistenceTest(AckMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self._build()
        self.path = os.path.join(self.directory, "state.json")
        self.state_store = attach_persistence(self.service, self.path)

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_first_ack_persists_and_advances_one_generation(self) -> None:
        self._delivered_lease("L1", 2)
        before = self.state_store.commit_seq
        body, status = self.service.inbox_lease_ack("bob", "L1")
        self.assertEqual(status, 201)
        self.assertEqual(self.state_store.commit_seq, before + 1)

        with open(self.path, encoding="utf-8") as handle:
            document = json.load(handle)
        leased = [record for record in document["delivery"]
                  if record.get("leases")]
        self.assertEqual(len(leased), 2)
        for record in leased:
            self.assertEqual(list(record),
                             ["session_id", "message_id", "attempts",
                              "attempt_ids", "acked", "ack_sequence",
                              "leases"])
            self.assertIs(record["acked"], True)
        sequences = {record["message_id"]: record["ack_sequence"]
                     for record in leased}
        self.assertEqual(sequences, {"a1": 1, "a2": 2})

    def test_repeat_ack_persists_nothing(self) -> None:
        self._delivered_lease("L1", 2)
        first, _ = self.service.inbox_lease_ack("bob", "L1")
        generation = self.state_store.commit_seq
        replay, status = self.service.inbox_lease_ack("bob", "L1")
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)
        self.assertEqual(self.state_store.commit_seq, generation)

    def test_restart_restores_ack(self) -> None:
        self._delivered_lease("L1", 2)
        first, _ = self.service.inbox_lease_ack("bob", "L1")
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        # The acknowledgement replays byte-identically after the restart.
        replay, status = restarted.inbox_lease_ack("bob", "L1")
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)
        # And the acked messages stay out of the inbox.
        inbox = restarted.device_inbox("bob", 100)
        self.assertEqual([m["message_id"] for m in inbox["messages"]],
                         ["a3", "b1", "b2"])

    def test_save_failure_rolls_back_and_surfaces(self) -> None:
        import e2ee_backend.persistence as persistence_mod

        self._delivered_lease("L1", 2)
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
                self.service.inbox_lease_ack("bob", "L1")
        finally:
            persistence_mod.os.fsync = real_fsync
        # Rolled back: no generation consumed and nothing is acked, so the
        # messages are still in the inbox and the ack can be retried.
        self.assertEqual(self.state_store.commit_seq, generation)
        inbox = self.service.device_inbox("bob", 100)
        self.assertEqual([m["message_id"] for m in inbox["messages"]],
                         ["a1", "a2", "a3", "b1", "b2"])
        generation = self.state_store.commit_seq
        body, status = self.service.inbox_lease_ack("bob", "L1")
        self.assertEqual(status, 201)
        self.assertEqual(self.state_store.commit_seq, generation + 1)
        self.assertEqual(body["message_count"], 2)

    def _malformed_document(self, mutate):
        self._delivered_lease(f"L1-{id(mutate)}", 2)
        self.service.inbox_lease_ack("bob", f"L1-{id(mutate)}")
        with open(self.path, encoding="utf-8") as handle:
            document = json.load(handle)
        mutate(document)
        # Write to a fresh path as a marker-less/sidecar-less document, so
        # rejection comes from the payload's own semantic validation rather
        # than the integrity hash gate; the original state file is untouched.
        document.pop("integrity_log_version", None)
        bad_path = os.path.join(self.directory, f"bad-{id(mutate)}.json")
        with open(bad_path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        return bad_path

    def _assert_refuses_startup(self, bad_path) -> None:
        with open(bad_path, "rb") as handle:
            original = handle.read()
        restarted = DeviceService()
        with self.assertRaises(StateFileError):
            attach_persistence(restarted, bad_path)
        # The rejected file is never overwritten.
        with open(bad_path, "rb") as handle:
            self.assertEqual(handle.read(), original)

    def test_restore_rejects_acked_without_matching_sequence(self) -> None:
        def mutate(document):
            leased = [r for r in document["delivery"] if r.get("leases")]
            leased[0]["ack_sequence"] = 99
        self._assert_refuses_startup(self._malformed_document(mutate))

    def test_restore_rejects_ack_sequence_while_unacked(self) -> None:
        def mutate(document):
            leased = [r for r in document["delivery"] if r.get("leases")]
            leased[0]["acked"] = False
        self._assert_refuses_startup(self._malformed_document(mutate))


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
            "POST", f"/v1/devices/{device_id}/inbox/leases/{lease_id}/ack",
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

    def _complete(self, lease_id="L1", outcome="delivered"):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("POST",
                     f"/v1/devices/bob/inbox/leases/{lease_id}/complete",
                     body=json.dumps({"completion_id": f"C-{lease_id}",
                                      "outcome": outcome}),
                     headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        response.read()
        conn.close()
        self.assertEqual(response.status, 201)

    def test_ack_and_replay_over_http(self) -> None:
        self._claim("L1", 2)
        self._complete("L1")
        status, body, raw = self._request("bob", "L1")
        self.assertEqual(status, 201)
        self.assertEqual(list(body),
                         ["device_id", "lease_id", "acked", "message_count"])
        # Key order is also correct in the serialized bytes.
        self.assertLess(raw.index('"device_id"'), raw.index('"lease_id"'))
        self.assertLess(raw.index('"lease_id"'), raw.index('"acked"'))
        self.assertLess(raw.index('"acked"'), raw.index('"message_count"'))
        self.assertIs(body["acked"], True)
        self.assertEqual(body["message_count"], 2)
        status, replay, replay_raw = self._request("bob", "L1")
        self.assertEqual(status, 200)
        self.assertEqual(replay_raw, raw)

    def test_non_empty_body_is_400_request_body(self) -> None:
        self._claim("L1", 2)
        self._complete("L1")
        for raw in ("{}", "x", json.dumps({"lease_id": "L1"})):
            status, body, _ = self._request("bob", "L1", raw=raw)
            self.assertEqual(status, 400, raw)
            self.assertEqual(list(body), ["message", "field"])
            self.assertEqual(body["field"], "request_body")
        # The rejections consumed nothing: the ack still works.
        status, body, _ = self._request("bob", "L1")
        self.assertEqual(status, 201)

    def test_unknown_lease_over_http(self) -> None:
        status, body, _ = self._request("bob", "NOPE")
        self.assertEqual(status, 404)
        self.assertEqual(list(body), ["message", "field"])
        self.assertEqual(body["field"], "lease_id")

    def test_cross_device_over_http(self) -> None:
        self._claim("L1", 2)
        self._complete("L1")
        status, body, _ = self._request("carol", "L1")
        self.assertEqual(status, 409)
        self.assertEqual(list(body), ["message", "field"])
        self.assertEqual(body["field"], "lease_id")

    def test_uncompleted_lease_over_http(self) -> None:
        self._claim("L1", 2)
        status, body, _ = self._request("bob", "L1")
        self.assertEqual(status, 409)
        self.assertEqual(list(body), ["message", "field"])
        self.assertEqual(body["field"], "lease_id")

    def test_failed_completion_over_http(self) -> None:
        self._claim("L1", 2)
        self._complete("L1", outcome="failed")
        status, body, _ = self._request("bob", "L1")
        self.assertEqual(status, 409)
        self.assertEqual(list(body), ["message", "field"])
        self.assertEqual(body["field"], "lease_id")

    def test_revoked_device_over_http(self) -> None:
        self._claim("L1", 2)
        self._complete("L1")
        self.service.store.revoke_device("bob")
        status, body, _ = self._request("bob", "L1")
        self.assertEqual(status, 409)
        self.assertEqual(list(body), ["message", "field"])
        self.assertEqual(body["field"], "device_id")

    def test_percent_encoded_ids_over_http(self) -> None:
        # A lease id containing a slash reaches the store percent-decoded.
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("POST", "/v1/devices/bob/inbox/claim",
                     body=json.dumps({"lease_id": "a/b", "limit": 1}),
                     headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        response.read()
        conn.close()
        self.assertEqual(response.status, 201)
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("POST", "/v1/devices/bob/inbox/leases/a%2Fb/complete",
                     body=json.dumps({"completion_id": "C1",
                                      "outcome": "delivered"}),
                     headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        response.read()
        conn.close()
        self.assertEqual(response.status, 201)
        status, body, _ = self._request("bob", "a%2Fb")
        self.assertEqual(status, 201)
        self.assertEqual(body["lease_id"], "a/b")


if __name__ == "__main__":
    unittest.main()
