"""Tests for the group-inbox lease claim and release endpoints.

POST /v1/devices/{device_id}/group-inbox/claim leases up to ``limit``
unacked group-inbox messages (those with no lease or an expired/released
one) for 30 seconds, under the store lock shared with submission, group
ack, retry and revocation. A non-empty claim returns 201 with a UTC
ISO-8601 deadline; an empty selection returns 200 with ``leased_until``
null and writes nothing. The client-chosen ``lease_id`` is globally bound
and idempotent only for the same device and limit.

POST /v1/devices/{device_id}/group-inbox/leases/{lease_id}/release
releases an occupied group lease so its messages can be claimed again.
"""
import json
import os
import shutil
import tempfile
import threading
import unittest
from datetime import datetime
from http.client import HTTPConnection

from e2ee_backend.models import Device, SignedPreKey
from e2ee_backend.persistence import (
    PersistenceUnavailable, StateFileError, attach_persistence)
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


class GroupLeaseMixin:
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
        # m1/m2 from d1, m3 from d2: d2 may claim m1/m2 only, d3 all three.
        for message_id, sequence, sender in (
                ("m1", 1, "d1"), ("m2", 2, "d1"), ("m3", 3, "d2")):
            self.service.post_message(
                _message(self.gs1, message_id, sequence, sender))
        # A 1:1 session d1 -> d2 never contributes to a group claim.
        self.one2one = self.store.create_session(
            "d1", "d2", "pk1", "ek").session_id
        self.service.post_message(
            _message(self.one2one, "p1", 1, "d1"))

    def _expire_all_group_leases(self, service=None) -> None:
        store = (service or self.service).store
        for state in store._group_delivery.values():
            for lease in state.leases:
                lease.leased_until = "2000-01-01T00:00:00.000000+00:00"


class GroupClaimServiceTest(GroupLeaseMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def test_claims_in_inbox_order_up_to_limit(self) -> None:
        body, status = self.service.group_inbox_claim(
            "d3", {"lease_id": "L1", "limit": 2})
        self.assertEqual(status, 201)
        self.assertEqual(list(body),
                         ["device_id", "lease_id", "leased_until",
                          "messages"])
        self.assertEqual(body["device_id"], "d3")
        self.assertEqual(body["lease_id"], "L1")
        self.assertEqual([m["message_id"] for m in body["messages"]],
                         ["m1", "m2"])
        for message in body["messages"]:
            self.assertEqual(list(message), [
                "session_id", "sender_device_id", "message_id", "sequence",
                "nonce", "ciphertext", "created_at"])
        deadline = body["leased_until"]
        self.assertRegex(deadline, r"\.\d{6}\+00:00$")
        parsed = datetime.fromisoformat(deadline)
        self.assertEqual(parsed.utcoffset().total_seconds(), 0)
        delta = parsed - datetime.now(parsed.tzinfo)
        self.assertAlmostEqual(delta.total_seconds(), 30, delta=2)

    def test_own_messages_and_one_to_one_never_contribute(self) -> None:
        body, status = self.service.group_inbox_claim(
            "d2", {"lease_id": "L1", "limit": 10})
        self.assertEqual(status, 201)
        self.assertEqual([m["message_id"] for m in body["messages"]],
                         ["m1", "m2"])

    def test_second_claim_skips_active_leased_messages(self) -> None:
        first, _ = self.service.group_inbox_claim(
            "d3", {"lease_id": "L1", "limit": 2})
        self.assertEqual([m["message_id"] for m in first["messages"]],
                         ["m1", "m2"])
        second, status = self.service.group_inbox_claim(
            "d3", {"lease_id": "L2", "limit": 10})
        self.assertEqual(status, 201)
        self.assertEqual([m["message_id"] for m in second["messages"]],
                         ["m3"])
        third, status = self.service.group_inbox_claim(
            "d3", {"lease_id": "L3", "limit": 10})
        self.assertEqual(status, 200)
        self.assertEqual(third["messages"], [])
        self.assertIsNone(third["leased_until"])

    def test_empty_claim_does_not_occupy_id_or_write_record(self) -> None:
        self.service.group_inbox_claim(
            "d3", {"lease_id": "L1", "limit": 10})
        empty, status = self.service.group_inbox_claim(
            "d3", {"lease_id": "L2", "limit": 10})
        self.assertEqual(status, 200)
        self.assertIsNone(empty["leased_until"])
        # L2 stayed free: releasing it is a not-found, and after L1 is
        # released a later non-empty claim binds L2 anew.
        with self.assertRaises(ServiceError) as caught:
            self.service.group_inbox_release("d3", "L2")
        self.assertEqual(caught.exception.status_code, 404)
        self.service.group_inbox_release("d3", "L1")
        again, status = self.service.group_inbox_claim(
            "d3", {"lease_id": "L2", "limit": 10})
        self.assertEqual(status, 201)
        self.assertEqual([m["message_id"] for m in again["messages"]],
                         ["m1", "m2", "m3"])

    def test_exact_replay_is_frozen_200(self) -> None:
        first, _ = self.service.group_inbox_claim(
            "d3", {"lease_id": "L1", "limit": 2})
        replay, status = self.service.group_inbox_claim(
            "d3", {"lease_id": "L1", "limit": 2})
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)
        # Still frozen after the lease was released and the messages leased
        # again under a different id.
        self.service.group_inbox_release("d3", "L1")
        self.service.group_inbox_claim(
            "d3", {"lease_id": "L9", "limit": 10})
        replay, status = self.service.group_inbox_claim(
            "d3", {"lease_id": "L1", "limit": 2})
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)

    def test_changed_device_or_limit_conflicts(self) -> None:
        self.service.group_inbox_claim(
            "d3", {"lease_id": "L1", "limit": 2})
        with self.assertRaises(ServiceError) as caught:
            self.service.group_inbox_claim(
                "d3", {"lease_id": "L1", "limit": 3})
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")
        with self.assertRaises(ServiceError) as caught:
            self.service.group_inbox_claim(
                "d2", {"lease_id": "L1", "limit": 2})
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")
        # The conflict takes priority over an unknown path device.
        with self.assertRaises(ServiceError) as caught:
            self.service.group_inbox_claim(
                "ghost", {"lease_id": "L1", "limit": 9})
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_lease_id_is_global_across_namespaces(self) -> None:
        self.service.inbox_claim(
            "d2", {"lease_id": "SHARED", "limit": 1})
        with self.assertRaises(ServiceError) as caught:
            self.service.group_inbox_claim(
                "d2", {"lease_id": "SHARED", "limit": 1})
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")
        # Releasing a 1:1 lease id through the group endpoint is a conflict,
        # not a not-found.
        with self.assertRaises(ServiceError) as caught:
            self.service.group_inbox_release("d2", "SHARED")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")
        # And the reverse direction: a group id blocks a 1:1 claim/release.
        self.service.group_inbox_claim(
            "d3", {"lease_id": "GONLY", "limit": 1})
        with self.assertRaises(ServiceError) as caught:
            self.service.inbox_claim(
                "d2", {"lease_id": "GONLY", "limit": 1})
        self.assertEqual(caught.exception.status_code, 409)
        with self.assertRaises(ServiceError) as caught:
            self.service.inbox_release("d2", "GONLY")
        self.assertEqual(caught.exception.status_code, 409)

    def test_unknown_or_revoked_device(self) -> None:
        with self.assertRaises(ServiceError) as caught:
            self.service.group_inbox_claim(
                "ghost", {"lease_id": "Z", "limit": 1})
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "device_id")
        self.store.revoke_device("d3")
        with self.assertRaises(ServiceError) as caught:
            self.service.group_inbox_claim(
                "d3", {"lease_id": "Z", "limit": 1})
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "device_id")

    def test_expired_lease_is_reclaimable(self) -> None:
        first, _ = self.service.group_inbox_claim(
            "d3", {"lease_id": "L1", "limit": 1})
        self.assertEqual([m["message_id"] for m in first["messages"]],
                         ["m1"])
        self._expire_all_group_leases()
        second, status = self.service.group_inbox_claim(
            "d3", {"lease_id": "L2", "limit": 10})
        self.assertEqual(status, 201)
        self.assertEqual([m["message_id"] for m in second["messages"]],
                         ["m1", "m2", "m3"])

    def test_claim_does_not_touch_acks_or_attempts(self) -> None:
        self.service.group_inbox_claim(
            "d3", {"lease_id": "L1", "limit": 10})
        # The read-only query still lists every unacked message; leases
        # never alter attempts/dedup state.
        page = self.service.device_group_inbox("d3", 100)
        self.assertEqual([m["message_id"] for m in page["messages"]],
                         ["m1", "m2", "m3"])
        for state in self.store._group_delivery.values():
            self.assertEqual(state.attempts, 0)
            self.assertEqual(state.attempt_ids, set())
            self.assertFalse(state.acked)
        # A group ack of a leased message removes it from a later claim
        # without disturbing the lease history.
        self.service.sync_group_ack_messages("d3", {"items": [
            {"session_id": self.gs1, "message_id": "m1", "sequence": 1}]})
        self._expire_all_group_leases()
        body, status = self.service.group_inbox_claim(
            "d3", {"lease_id": "L2", "limit": 10})
        self.assertEqual(status, 201)
        self.assertEqual([m["message_id"] for m in body["messages"]],
                         ["m2", "m3"])

    def test_body_validation(self) -> None:
        def expect(payload, field, status=400):
            with self.assertRaises(ServiceError) as caught:
                self.service.group_inbox_claim("d3", payload)
            self.assertEqual(caught.exception.status_code, status, payload)
            self.assertEqual(caught.exception.field, field, payload)

        expect(None, "request_body")
        expect([1, 2], "request_body")
        expect("text", "request_body")
        expect({"lease_id": "L", "limit": 1, "extra": 2}, "extra")
        expect({"limit": 1}, "lease_id")
        expect({"lease_id": "", "limit": 1}, "lease_id")
        expect({"lease_id": 5, "limit": 1}, "lease_id")
        expect({"lease_id": None, "limit": 1}, "lease_id")
        expect({"lease_id": "L"}, "limit")
        expect({"lease_id": "L", "limit": None}, "limit")
        expect({"lease_id": "L", "limit": True}, "limit")
        expect({"lease_id": "L", "limit": 1.5}, "limit")
        expect({"lease_id": "L", "limit": 0}, "limit")
        expect({"lease_id": "L", "limit": 101}, "limit")
        expect({"lease_id": "L", "limit": "2"}, "limit")


class GroupReleaseServiceTest(GroupLeaseMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def _claim(self, lease_id, limit, device="d3"):
        return self.service.group_inbox_claim(
            device, {"lease_id": lease_id, "limit": limit})

    def test_first_release_201_and_repeat_frozen_200(self) -> None:
        first_claim, _ = self._claim("L1", 2)
        body, status = self.service.group_inbox_release("d3", "L1")
        self.assertEqual(status, 201)
        self.assertEqual(list(body),
                         ["device_id", "lease_id", "released_at",
                          "released_count"])
        self.assertEqual(body["device_id"], "d3")
        self.assertEqual(body["lease_id"], "L1")
        self.assertEqual(body["released_count"], 2)
        self.assertRegex(body["released_at"], r"\.\d{6}\+00:00$")
        repeat, status = self.service.group_inbox_release("d3", "L1")
        self.assertEqual(status, 200)
        self.assertEqual(repeat, body)

    def test_release_frees_messages_for_new_claim(self) -> None:
        self._claim("L1", 2)
        self.service.group_inbox_release("d3", "L1")
        again, status = self.service.group_inbox_claim(
            "d3", {"lease_id": "L2", "limit": 10})
        self.assertEqual(status, 201)
        self.assertEqual([m["message_id"] for m in again["messages"]],
                         ["m1", "m2", "m3"])

    def test_not_found_conflict_and_revocation_rules(self) -> None:
        self._claim("L1", 2)
        with self.assertRaises(ServiceError) as caught:
            self.service.group_inbox_release("d3", "missing")
        self.assertEqual(caught.exception.status_code, 404)
        self.assertEqual(caught.exception.field, "lease_id")
        with self.assertRaises(ServiceError) as caught:
            self.service.group_inbox_release("d2", "L1")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")
        # A fresh id on an unknown device is the device error, but an
        # existing lease keeps its not-found/conflict precedence.
        with self.assertRaises(ServiceError) as caught:
            self.service.group_inbox_release("ghost", "fresh")
        self.assertEqual(caught.exception.status_code, 404)
        # First release on a revoked device is 409/device_id, but a repeat
        # still answers the frozen first response with 200.
        self._claim("L2", 1)
        self.service.group_inbox_release("d3", "L2")
        self.store.revoke_device("d3")
        with self.assertRaises(ServiceError) as caught:
            self.service.group_inbox_release("d3", "L1")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "device_id")
        repeat, status = self.service.group_inbox_release("d3", "L2")
        self.assertEqual(status, 200)
        self.assertEqual(repeat["released_count"], 1)


class GroupLeasePersistenceTest(GroupLeaseMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self._build()
        self.path = os.path.join(self.directory, "state.json")
        self.state_store = attach_persistence(self.service, self.path)

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_nonempty_claim_persists_one_generation(self) -> None:
        before = self.state_store.commit_seq
        _, status = self.service.group_inbox_claim(
            "d3", {"lease_id": "L1", "limit": 2})
        self.assertEqual(status, 201)
        self.assertEqual(self.state_store.commit_seq, before + 1)
        with open(self.path, encoding="utf-8") as handle:
            document = json.load(handle)
        leased = [record for record in document["group_delivery"]
                  if record.get("leases")]
        self.assertEqual(len(leased), 2)
        for record in leased:
            self.assertEqual(list(record), [
                "session_id", "message_id", "device_id", "attempts",
                "attempt_ids", "acked", "ack_sequence", "leases"])
            self.assertEqual(len(record["leases"]), 1)
            self.assertEqual(list(record["leases"][0]),
                             ["lease_id", "limit", "leased_until",
                              "released_at"])
            self.assertEqual(record["leases"][0]["lease_id"], "L1")
            self.assertEqual(record["leases"][0]["limit"], 2)
            self.assertIsNone(record["leases"][0]["released_at"])

    def test_empty_claim_and_replay_consume_no_generation(self) -> None:
        self.service.group_inbox_claim(
            "d3", {"lease_id": "L1", "limit": 10})
        generation = self.state_store.commit_seq
        empty, status = self.service.group_inbox_claim(
            "d3", {"lease_id": "L2", "limit": 10})
        self.assertEqual(status, 200)
        self.assertEqual(empty["messages"], [])
        self.assertEqual(self.state_store.commit_seq, generation)
        replay, status = self.service.group_inbox_claim(
            "d3", {"lease_id": "L1", "limit": 10})
        self.assertEqual(status, 200)
        self.assertEqual(self.state_store.commit_seq, generation)

    def test_restart_restores_claim_and_release(self) -> None:
        first, _ = self.service.group_inbox_claim(
            "d3", {"lease_id": "L1", "limit": 2})
        release, _ = self.service.group_inbox_release("d3", "L1")
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        replay, status = restarted.group_inbox_claim(
            "d3", {"lease_id": "L1", "limit": 2})
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)
        release_replay, status = restarted.group_inbox_release("d3", "L1")
        self.assertEqual(status, 200)
        self.assertEqual(release_replay, release)
        # Freed by the restored release: a fresh claim takes all three.
        again, status = restarted.group_inbox_claim(
            "d3", {"lease_id": "L2", "limit": 10})
        self.assertEqual(status, 201)
        self.assertEqual([m["message_id"] for m in again["messages"]],
                         ["m1", "m2", "m3"])

    def test_legacy_group_record_without_leases_loads(self) -> None:
        self.service.group_inbox_claim(
            "d3", {"lease_id": "L1", "limit": 1})
        with open(self.path, encoding="utf-8") as handle:
            document = json.load(handle)
        for record in document["group_delivery"]:
            record.pop("leases", None)
        document.pop("integrity_log_version", None)
        legacy_path = os.path.join(self.directory, "legacy.json")
        with open(legacy_path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        restarted = DeviceService()
        attach_persistence(restarted, legacy_path)  # must not raise
        body, status = restarted.group_inbox_claim(
            "d3", {"lease_id": "L2", "limit": 1})
        self.assertEqual(status, 201)
        self.assertEqual([m["message_id"] for m in body["messages"]],
                         ["m1"])

    def _malformed_group_document(self, mutate):
        self.service.group_inbox_claim(
            "d3", {"lease_id": "L1", "limit": 2})
        with open(self.path, encoding="utf-8") as handle:
            document = json.load(handle)
        mutate(document)
        document.pop("integrity_log_version", None)
        bad_path = os.path.join(self.directory, "bad.json")
        with open(bad_path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        return bad_path

    def _assert_refuses_startup(self, bad_path) -> None:
        with open(bad_path, "rb") as handle:
            original = handle.read()
        restarted = DeviceService()
        with self.assertRaises(StateFileError):
            attach_persistence(restarted, bad_path)
        with open(bad_path, "rb") as handle:
            self.assertEqual(handle.read(), original)

    def test_restore_rejects_bad_group_lease_shape(self) -> None:
        def mutate(document):
            document["group_delivery"][0]["leases"][0]["limit"] = 0
        self._assert_refuses_startup(self._malformed_group_document(mutate))

        def mutate(document):
            document["group_delivery"][0]["leases"][0]["renewals"] = []
        self._assert_refuses_startup(self._malformed_group_document(mutate))

    def test_restore_rejects_inconsistent_binding(self) -> None:
        def mutate(document):
            first = next(record for record in document["group_delivery"]
                         if record.get("leases"))
            first["leases"][0]["limit"] = 3
        self._assert_refuses_startup(self._malformed_group_document(mutate))

    def test_restore_rejects_lease_id_shared_with_delivery(self) -> None:
        self.service.inbox_claim(
            "d2", {"lease_id": "X1", "limit": 1})

        def mutate(document):
            # Force the clash by renaming the 1:1 lease onto the group id.
            for record in document["delivery"]:
                for lease in record.get("leases", []):
                    lease["lease_id"] = "L1"
        self._assert_refuses_startup(self._malformed_group_document(mutate))

    def test_save_failure_rolls_back_claim(self) -> None:
        import e2ee_backend.persistence as persistence_mod

        real_fsync = persistence_mod.os.fsync
        calls = {"n": 0}

        def fail_first_directory_fsync(fd):  # noqa: ANN001
            if os.fstat(fd).st_mode & 0o170000 == 0o040000:
                calls["n"] += 1
                if calls["n"] == 1:
                    raise OSError("transient directory fsync failure")
            real_fsync(fd)

        generation = self.state_store.commit_seq
        persistence_mod.os.fsync = fail_first_directory_fsync
        try:
            with self.assertRaises(PersistenceUnavailable):
                self.service.group_inbox_claim(
                    "d3", {"lease_id": "L1", "limit": 2})
        finally:
            persistence_mod.os.fsync = real_fsync
        self.assertEqual(self.state_store.commit_seq, generation)
        # Rolled back: the id is free and the messages are still claimable.
        body, status = self.service.group_inbox_claim(
            "d3", {"lease_id": "L1", "limit": 2})
        self.assertEqual(status, 201)
        self.assertEqual([m["message_id"] for m in body["messages"]],
                         ["m1", "m2"])


class GroupLeaseHTTPTest(GroupLeaseMixin, unittest.TestCase):
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

    def _request(self, method, target, raw=None):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        headers = {"Content-Type": "application/json"} if raw is not None \
            else {}
        conn.request(method, target, body=raw, headers=headers)
        response = conn.getresponse()
        data = response.read().decode("utf-8")
        conn.close()
        return response.status, (json.loads(data) if data else None), data

    def test_claim_replay_and_empty_over_http(self) -> None:
        status, body, raw = self._request(
            "POST", "/v1/devices/d3/group-inbox/claim",
            json.dumps({"lease_id": "L1", "limit": 2}))
        self.assertEqual(status, 201)
        self.assertEqual(list(body),
                         ["device_id", "lease_id", "leased_until",
                          "messages"])
        self.assertLess(raw.index('"device_id"'), raw.index('"lease_id"'))
        self.assertLess(raw.index('"lease_id"'),
                        raw.index('"leased_until"'))
        status, replay, _ = self._request(
            "POST", "/v1/devices/d3/group-inbox/claim",
            json.dumps({"lease_id": "L1", "limit": 2}))
        self.assertEqual(status, 200)
        self.assertEqual(replay, body)
        status, body, raw = self._request(
            "POST", "/v1/devices/d3/group-inbox/claim",
            json.dumps({"lease_id": "L2", "limit": 10}))
        # m1/m2 held by L1, m3 still free, so this is non-empty; exhaust it.
        self.assertEqual(status, 201)
        status, body, raw = self._request(
            "POST", "/v1/devices/d3/group-inbox/claim",
            json.dumps({"lease_id": "L3", "limit": 10}))
        self.assertEqual(status, 200)
        self.assertIsNone(body["leased_until"])
        self.assertEqual(body["messages"], [])
        self.assertIn('"leased_until":null', raw.replace(" ", ""))

    def test_claim_validation_over_http(self) -> None:
        cases = [
            ("{bad", "request_body"),
            (json.dumps([1]), "request_body"),
            ("", "request_body"),
            (json.dumps({"lease_id": "L", "limit": 1, "x": 1}), "x"),
            (json.dumps({"limit": 1}), "lease_id"),
            (json.dumps({"lease_id": "L", "limit": 2.5}), "limit"),
        ]
        for raw, field in cases:
            status, body, _ = self._request(
                "POST", "/v1/devices/d3/group-inbox/claim", raw)
            self.assertEqual(status, 400, raw)
            self.assertEqual(body["field"], field, raw)
        status, body, _ = self._request(
            "POST", "/v1/devices/d3/group-inbox/claim?foo",
            json.dumps({"lease_id": "L", "limit": 1}))
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "query")

    def test_claim_device_and_conflict_over_http(self) -> None:
        status, body, _ = self._request(
            "POST", "/v1/devices/ghost/group-inbox/claim",
            json.dumps({"lease_id": "Z", "limit": 1}))
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "device_id")
        self._request("POST", "/v1/devices/d3/group-inbox/claim",
                      json.dumps({"lease_id": "L1", "limit": 2}))
        status, body, _ = self._request(
            "POST", "/v1/devices/d2/group-inbox/claim",
            json.dumps({"lease_id": "L1", "limit": 2}))
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "lease_id")

    def test_bad_path_encoding_is_400(self) -> None:
        status, body, _ = self._request(
            "POST", "/v1/devices/d%zz/group-inbox/claim",
            json.dumps({"lease_id": "L", "limit": 1}))
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "device_id")
        status, body, _ = self._request(
            "POST",
            "/v1/devices/d3/group-inbox/leases/L%zz/release")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "lease_id")

    def test_release_flow_over_http(self) -> None:
        target = "/v1/devices/d3/group-inbox/leases/L1/release"
        # Unknown lease first: 404.
        status, body, _ = self._request("POST", target)
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "lease_id")
        self._request("POST", "/v1/devices/d3/group-inbox/claim",
                      json.dumps({"lease_id": "L1", "limit": 2}))
        status, body, raw = self._request("POST", target)
        self.assertEqual(status, 201)
        self.assertEqual(list(body),
                         ["device_id", "lease_id", "released_at",
                          "released_count"])
        self.assertEqual(body["released_count"], 2)
        self.assertRegex(body["released_at"], r"\.\d{6}\+00:00$")
        self.assertLess(raw.index('"released_at"'),
                        raw.index('"released_count"'))
        status, repeat, _ = self._request("POST", target)
        self.assertEqual(status, 200)
        self.assertEqual(repeat, body)
        # Owned by another device.
        status, body, _ = self._request(
            "POST", "/v1/devices/d2/group-inbox/leases/L1/release")
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "lease_id")

    def test_release_rejects_body_and_query(self) -> None:
        self._request("POST", "/v1/devices/d3/group-inbox/claim",
                      json.dumps({"lease_id": "L1", "limit": 2}))
        target = "/v1/devices/d3/group-inbox/leases/L1/release"
        status, body, _ = self._request("POST", target, "{}")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "request_body")
        status, body, _ = self._request("POST", target + "?x=1")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "query")


if __name__ == "__main__":
    unittest.main()
