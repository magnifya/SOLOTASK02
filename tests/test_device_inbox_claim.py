"""Tests for the 1:1 inbox claim-lease endpoint.

POST /v1/devices/{device_id}/inbox/claim leases up to ``limit`` unacked 1:1
inbox messages for 30 seconds under the store lock. Non-empty claims answer
201 and persist (commit_seq + 1); empty selections answer 200 with
``leased_until`` null and write nothing; an occupied lease_id replays for the
same device/limit with the byte-identical first response, while cross-device
or changed-limit reuse conflicts with 409/lease_id.
"""
import json
import os
import re
import shutil
import tempfile
import threading
import unittest
from datetime import datetime, timedelta, timezone
from http.client import HTTPConnection

from e2ee_backend.models import Device, SignedPreKey
from e2ee_backend.persistence import (
    PersistenceUnavailable, StateFileError, attach_persistence)
from e2ee_backend.service import DeviceService, ServiceError
from e2ee_backend.http_app import create_server

ISO_MICROS = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}\+00:00$")


class ClaimMixin:
    def _build(self) -> None:
        self.service = DeviceService()
        self.service.store.add_device(Device("u", "alice", "ik"))
        self.service.store.add_device(Device(
            "u", "bob", "ik",
            prekeys=[SignedPreKey("pk1", "pubk1"),
                     SignedPreKey("pk2", "pubk2"),
                     SignedPreKey("pk3", "pubk3")]))
        self.service.store.add_device(
            Device("u", "bob2", "ik",
                   prekeys=[SignedPreKey("pkB", "pubkB")]))
        self.service.store.add_device(
            Device("u", "carol", "ik",
                   prekeys=[SignedPreKey("pkC", "pubkC")]))
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
        # Session addressed to bob's other device (must not contribute).
        self.sid_other = self.service.store.create_session(
            "alice", "bob2", "pkB", "ekB").session_id
        self.service.post_message({
            "session_id": self.sid_other, "sender_device_id": "alice",
            "message_id": "o1", "sequence": 1,
            "nonce": "no1", "ciphertext": "ct"})

    def _claim(self, device_id="bob", lease_id="L", limit=100):
        return self.service.inbox_claim(
            device_id, {"lease_id": lease_id, "limit": limit})

    def _ids(self, body):
        return [m["message_id"] for m in body["messages"]]

    def _advance_clock(self, seconds):
        base = datetime.now(timezone.utc)
        self.service.store._lease_clock = \
            lambda: base + timedelta(seconds=seconds)


class ClaimServiceTest(ClaimMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def _error(self, payload, device_id="bob"):
        with self.assertRaises(ServiceError) as caught:
            self.service.inbox_claim(device_id, payload)
        return caught.exception

    def test_non_empty_claim_is_201_with_key_order(self) -> None:
        body, code = self._claim(lease_id="L1", limit=2)
        self.assertEqual(code, 201)
        self.assertEqual(list(body),
                         ["device_id", "lease_id", "leased_until", "messages"])
        self.assertEqual(body["device_id"], "bob")
        self.assertEqual(body["lease_id"], "L1")
        self.assertRegex(body["leased_until"], ISO_MICROS)
        self.assertEqual(self._ids(body), ["a1", "a2"])
        for message in body["messages"]:
            self.assertEqual(list(message), [
                "session_id", "sender_device_id", "message_id", "sequence",
                "nonce", "ciphertext", "created_at"])

    def test_lease_is_thirty_seconds(self) -> None:
        fixed = datetime(2026, 9, 25, 12, 0, 0, tzinfo=timezone.utc)
        self.service.store._lease_clock = lambda: fixed
        body, _ = self._claim(lease_id="L1", limit=1)
        self.assertEqual(body["leased_until"],
                         "2026-09-25T12:00:30.000000+00:00")

    def test_active_lease_excludes_messages_from_later_claims(self) -> None:
        first, code = self._claim(lease_id="L1", limit=2)
        self.assertEqual(code, 201)
        second, code = self._claim(lease_id="L2", limit=100)
        self.assertEqual(code, 201)
        self.assertEqual(self._ids(second), ["a3", "b1", "b2"])

    def test_empty_selection_is_200_null_time_and_does_not_occupy_id(self) -> None:
        self._claim(lease_id="L1", limit=100)  # leases all five
        empty, code = self._claim(lease_id="EMPTY", limit=100)
        self.assertEqual(code, 200)
        self.assertIsNone(empty["leased_until"])
        self.assertEqual(empty["messages"], [])
        # The id was not occupied: after the first lease expires it leases.
        self._advance_clock(31)
        body, code = self._claim(lease_id="EMPTY", limit=1)
        self.assertEqual(code, 201)
        self.assertEqual(self._ids(body), ["a1"])

    def test_replay_same_device_same_limit_returns_first_response(self) -> None:
        first, code1 = self._claim(lease_id="L1", limit=2)
        replay, code2 = self._claim(lease_id="L1", limit=2)
        self.assertEqual(code1, 201)
        self.assertEqual(code2, 200)
        self.assertEqual(replay, first)
        self.assertEqual(json.dumps(replay, sort_keys=True),
                         json.dumps(first, sort_keys=True))

    def test_replay_after_expiry_still_returns_first_response(self) -> None:
        first, _ = self._claim(lease_id="L1", limit=2)
        self._advance_clock(120)
        replay, code = self._claim(lease_id="L1", limit=2)
        self.assertEqual(code, 200)
        self.assertEqual(replay, first)
        # Expiry made the same messages leasable by a *new* id meanwhile.
        other, code = self._claim(lease_id="L9", limit=2)
        self.assertEqual(code, 201)
        self.assertEqual(self._ids(other), ["a1", "a2"])

    def test_replay_after_device_revocation_still_200(self) -> None:
        first, _ = self._claim(lease_id="L1", limit=2)
        self.service.store.revoke_device("bob")
        replay, code = self._claim(lease_id="L1", limit=2)
        self.assertEqual(code, 200)
        self.assertEqual(replay, first)

    def test_replay_is_frozen_after_acked_messages(self) -> None:
        first, _ = self._claim(lease_id="L1", limit=2)
        self.service.ack_message(self.sid1, {
            "device_id": "bob", "message_id": "a1", "sequence": 1})
        replay, code = self._claim(lease_id="L1", limit=2)
        self.assertEqual(code, 200)
        self.assertEqual(replay, first)
        self.assertEqual(self._ids(replay), ["a1", "a2"])

    def test_changed_limit_conflicts(self) -> None:
        self._claim(lease_id="L1", limit=2)
        error = self._error({"lease_id": "L1", "limit": 3})
        self.assertEqual((error.status_code, error.field), (409, "lease_id"))
        error = self._error({"lease_id": "L1", "limit": 1})
        self.assertEqual((error.status_code, error.field), (409, "lease_id"))

    def test_lease_id_across_devices_conflicts(self) -> None:
        self._claim(device_id="bob", lease_id="L1", limit=10)
        error = self._error({"lease_id": "L1", "limit": 10},
                            device_id="bob2")
        self.assertEqual((error.status_code, error.field), (409, "lease_id"))

    def test_unknown_or_revoked_device(self) -> None:
        error = self._error({"lease_id": "L", "limit": 2}, device_id="ghost")
        self.assertEqual((error.status_code, error.field),
                         (409, "device_id"))
        self.service.store.revoke_device("bob")
        error = self._error({"lease_id": "L", "limit": 2})
        self.assertEqual((error.status_code, error.field),
                         (409, "device_id"))

    def test_unknown_path_device_beats_cross_device_conflict(self) -> None:
        self._claim(lease_id="L1", limit=2)
        error = self._error({"lease_id": "L1", "limit": 2},
                            device_id="ghost")
        self.assertEqual((error.status_code, error.field),
                         (409, "device_id"))
        self.service.store.revoke_device("bob2")
        error = self._error({"lease_id": "L1", "limit": 2},
                            device_id="bob2")
        self.assertEqual((error.status_code, error.field),
                         (409, "device_id"))

    def test_expired_lease_allows_new_claim_in_inbox_order(self) -> None:
        self._claim(lease_id="L1", limit=2)
        self._claim(lease_id="L2", limit=100)  # a3,b1,b2
        self._advance_clock(31)
        body, code = self._claim(lease_id="L3", limit=3)
        self.assertEqual(code, 201)
        self.assertEqual(self._ids(body), ["a1", "a2", "a3"])

    def test_acked_messages_are_never_leased_and_drop_from_replay_set(self) -> None:
        self._claim(lease_id="L1", limit=100)  # all five active
        self._advance_clock(31)
        self.service.ack_message(self.sid1, {
            "device_id": "bob", "message_id": "a1", "sequence": 1})
        body, code = self._claim(lease_id="L3", limit=100)
        self.assertEqual(code, 201)
        self.assertEqual(self._ids(body), ["a2", "a3", "b1", "b2"])

    def test_excludes_other_device_sessions(self) -> None:
        body, _ = self._claim(device_id="bob2", lease_id="LB", limit=100)
        self.assertEqual(self._ids(body), ["o1"])

    def test_body_validation(self) -> None:
        cases = {
            "[]": "request_body",
            "null": "request_body",
            '"x"': "request_body",
            "{}": "lease_id",
            '{"lease_id":"L"}': "limit",
            '{"limit":2}': "lease_id",
            '{"lease_id":"","limit":2}': "lease_id",
            '{"lease_id":5,"limit":2}': "lease_id",
            '{"lease_id":null,"limit":2}': "lease_id",
            '{"lease_id":"L","limit":0}': "limit",
            '{"lease_id":"L","limit":101}': "limit",
            '{"lease_id":"L","limit":-1}': "limit",
            '{"lease_id":"L","limit":true}': "limit",
            '{"lease_id":"L","limit":"2"}': "limit",
            '{"lease_id":"L","limit":1.5}': "limit",
        }
        for text, field in cases.items():
            error = self._error(json.loads(text))
            self.assertEqual((error.status_code, error.field), (400, field),
                             text)

    def test_in_memory_empty_claim_writes_nothing(self) -> None:
        before = dict(self.service.store._delivery)
        self._claim(lease_id="L1", limit=100)
        self._claim(lease_id="EMPTY", limit=100)
        # The empty claim created no records beyond L1's five.
        leases = [l for state in self.service.store._delivery.values()
                  for l in state.leases]
        self.assertEqual([l.lease_id for l in leases], ["L1"] * 5)
        self.assertNotIn("EMPTY",
                         self.service.store._inbox_lease_index)


class ClaimPersistenceTest(ClaimMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self._build()
        self.path = os.path.join(self.directory, "state.json")
        self.state_store = attach_persistence(self.service, self.path)

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_non_empty_claim_advances_generation_once(self) -> None:
        generation = self.state_store.commit_seq
        self._claim(lease_id="L1", limit=2)
        self.assertEqual(self.state_store.commit_seq, generation + 1)

    def test_empty_claim_advances_no_generation(self) -> None:
        self._claim(lease_id="L1", limit=100)
        generation = self.state_store.commit_seq
        self._claim(lease_id="EMPTY", limit=10)
        self.assertEqual(self.state_store.commit_seq, generation)

    def test_replay_is_byte_identical_after_restart(self) -> None:
        first, _ = self._claim(lease_id="L1", limit=2)
        second, _ = self._claim(lease_id="L2", limit=100)
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        replay1, code1 = restarted.inbox_claim(
            "bob", {"lease_id": "L1", "limit": 2})
        replay2, code2 = restarted.inbox_claim(
            "bob", {"lease_id": "L2", "limit": 100})
        self.assertEqual((code1, replay1), (200, first))
        self.assertEqual((code2, replay2), (200, second))

    def test_conflicts_survive_restart(self) -> None:
        self._claim(lease_id="L1", limit=2)
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        with self.assertRaises(ServiceError) as caught:
            restarted.inbox_claim(
                "bob2", {"lease_id": "L1", "limit": 2})
        self.assertEqual((caught.exception.status_code,
                          caught.exception.field), (409, "lease_id"))
        with self.assertRaises(ServiceError) as caught:
            restarted.inbox_claim(
                "bob", {"lease_id": "L1", "limit": 3})
        self.assertEqual((caught.exception.status_code,
                          caught.exception.field), (409, "lease_id"))

    def test_failed_write_rolls_back_memory_and_generation(self) -> None:
        generation = self.state_store.commit_seq
        delivery_before = {
            key: len(state.leases)
            for key, state in self.service.store._delivery.items()}

        def raise_oserror(*_args, **_kwargs):
            raise OSError("simulated write failure")

        self.state_store.save = raise_oserror
        with self.assertRaises(PersistenceUnavailable):
            self._claim(lease_id="L1", limit=2)
        self.assertEqual(self.state_store.commit_seq, generation)
        # No lease survives in memory.
        self.assertNotIn("L1", self.service.store._inbox_lease_index)
        self.assertEqual({
            key: len(state.leases)
            for key, state in self.service.store._delivery.items()},
            delivery_before)

    def _tamper(self, mutate, label):
        with open(self.path, encoding="utf-8") as handle:
            raw = json.load(handle)
        mutate(raw)
        path = os.path.join(self.directory, f"{label}.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(raw, handle)
        service = DeviceService()
        with self.assertRaises(StateFileError):
            attach_persistence(service, path)
        os.remove(path)

    def _record(self, doc, message_id="a1"):
        return next(record for record in doc["delivery"]
                    if record["message_id"] == message_id)

    def test_malformed_leases_reject_startup(self) -> None:
        self._claim(lease_id="L1", limit=2)
        self._tamper(
            lambda raw: self._record(raw)["leases"][0].__setitem__(
                "limit", 5), "limit-mismatch")
        self._tamper(
            lambda raw: self._record(raw)["leases"][0].__setitem__(
                "leased_until",
                "2030-01-01T00:00:00.000000+00:00"), "time-mismatch")
        self._tamper(
            lambda raw: self._record(raw)["leases"][0].__setitem__(
                "lease_id", 7), "lease-id-type")
        self._tamper(
            lambda raw: self._record(raw)["leases"][0].__setitem__(
                "limit", True), "limit-bool")
        self._tamper(
            lambda raw: self._record(raw)["leases"][0].__setitem__(
                "limit", 0), "limit-zero")
        self._tamper(
            lambda raw: self._record(raw)["leases"][0].__setitem__(
                "leased_until", "not-a-timestamp"), "bad-time")
        self._tamper(
            lambda raw: self._record(raw)["leases"][0].__setitem__(
                "leased_until",
                "2030-01-01T00:00:00.000000"), "naive-time")
        self._tamper(
            lambda raw: self._record(raw)["leases"][0].__setitem__(
                "leased_until",
                "2030-01-01T05:00:00.000000+05:00"), "non-utc-time")
        self._tamper(
            lambda raw: self._record(raw).__setitem__("leases", "x"),
            "leases-not-list")
        self._tamper(
            lambda raw: self._record(raw)["leases"].append(
                dict(self._record(raw)["leases"][0])),
            "duplicate-lease-id")

        def drop_limit(raw):
            del self._record(raw)["leases"][0]["limit"]

        self._tamper(drop_limit, "missing-limit")

    def test_cross_device_lease_id_rejects_startup(self) -> None:
        # A second recipient device (carol) with its own session/message and
        # lease; pinning bob's lease_id onto carol's delivery record is a
        # cross-device reuse and must refuse startup.
        self.service.store.add_device(
            Device("u", "dave", "ik",
                   prekeys=[SignedPreKey("pkD", "pubkD")]))
        sid_d = self.service.store.create_session(
            "bob", "dave", "pkD", "ekD").session_id
        self.service.post_message({
            "session_id": sid_d, "sender_device_id": "bob",
            "message_id": "d1", "sequence": 1, "nonce": "nd1",
            "ciphertext": "ct"})
        self._claim(lease_id="L1", limit=2)
        self.service.inbox_claim(
            "dave", {"lease_id": "LD", "limit": 10})
        with open(self.path, encoding="utf-8") as handle:
            raw = json.load(handle)
        bob_lease = self._record(raw, "a1")["leases"][0]
        dave_record = next(record for record in raw["delivery"]
                           if record["message_id"] == "d1")
        dave_record["leases"].append(dict(bob_lease))
        path = os.path.join(self.directory, "cross-device.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(raw, handle)
        service = DeviceService()
        with self.assertRaises(StateFileError):
            attach_persistence(service, path)
        os.remove(path)

    def test_legacy_delivery_without_leases_loads(self) -> None:
        self._claim(lease_id="L1", limit=2)
        with open(self.path, encoding="utf-8") as handle:
            raw = json.load(handle)
        for record in raw["delivery"]:
            record.pop("leases", None)
        raw.pop("integrity_log_version", None)
        path = os.path.join(self.directory, "legacy.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(raw, handle)
        service = DeviceService()
        attach_persistence(service, path)
        self.assertFalse(any(state.leases
                             for state in service.store._delivery.values()))


class ClaimHTTPTest(ClaimMixin, unittest.TestCase):
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

    def _post(self, device_id, raw_body):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("POST", f"/v1/devices/{device_id}/inbox/claim",
                     body=raw_body,
                     headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        return response.status, response.read()

    def test_claim_and_replay_status_codes(self) -> None:
        status, raw = self._post("bob", json.dumps(
            {"lease_id": "L1", "limit": 2}))
        self.assertEqual(status, 201)
        first = json.loads(raw.decode("utf-8"))
        self.assertEqual(list(first),
                         ["device_id", "lease_id", "leased_until",
                          "messages"])
        self.assertEqual([m["message_id"] for m in first["messages"]],
                         ["a1", "a2"])
        status, raw = self._post("bob", json.dumps(
            {"lease_id": "L1", "limit": 2}))
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(raw.decode("utf-8")), first)

    def test_empty_claim_is_200_null_time(self) -> None:
        self._post("bob", json.dumps({"lease_id": "L1", "limit": 100}))
        status, raw = self._post("bob", json.dumps(
            {"lease_id": "EMPTY", "limit": 10}))
        self.assertEqual(status, 200)
        body = json.loads(raw.decode("utf-8"))
        self.assertIsNone(body["leased_until"])
        self.assertEqual(body["messages"], [])

    def test_bad_json_is_400_request_body(self) -> None:
        status, raw = self._post("bob", "{not json")
        body = json.loads(raw.decode("utf-8"))
        self.assertEqual(status, 400)
        self.assertEqual(list(body), ["message", "field"])
        self.assertEqual(body["field"], "request_body")

    def test_field_errors(self) -> None:
        for text, field in (
                ("[]", "request_body"),
                ("{}", "lease_id"),
                ('{"lease_id":"L"}', "limit"),
                ('{"lease_id":"","limit":2}', "lease_id"),
                ('{"lease_id":"L","limit":0}', "limit"),
                ('{"lease_id":"L","limit":true}', "limit")):
            status, raw = self._post("bob", text)
            body = json.loads(raw.decode("utf-8"))
            self.assertEqual(status, 400, text)
            self.assertEqual(body["field"], field, text)

    def test_device_and_lease_conflicts(self) -> None:
        self._post("bob", json.dumps({"lease_id": "L1", "limit": 2}))
        status, raw = self._post("ghost", json.dumps(
            {"lease_id": "L", "limit": 2}))
        self.assertEqual(status, 409)
        self.assertEqual(json.loads(raw.decode("utf-8"))["field"],
                         "device_id")
        status, raw = self._post("bob2", json.dumps(
            {"lease_id": "L1", "limit": 2}))
        self.assertEqual(status, 409)
        self.assertEqual(json.loads(raw.decode("utf-8"))["field"],
                         "lease_id")
        status, raw = self._post("bob", json.dumps(
            {"lease_id": "L1", "limit": 5}))
        self.assertEqual(status, 409)
        self.assertEqual(json.loads(raw.decode("utf-8"))["field"],
                         "lease_id")

    def test_failed_durable_write_is_503_data_file(self) -> None:
        directory = tempfile.mkdtemp()
        try:
            path = os.path.join(directory, "state.json")
            persisted = DeviceService()
            # Mirror the fixture into the persisted service.
            persisted.store.add_device(Device("u", "alice", "ik"))
            persisted.store.add_device(Device(
                "u", "bob", "ik",
                prekeys=[SignedPreKey("pk1", "pubk1"),
                         SignedPreKey("pk2", "pubk2")]))
            sid = persisted.store.create_session(
                "alice", "bob", "pk1", "ek1").session_id
            for sequence in range(1, 4):
                persisted.post_message({
                    "session_id": sid, "sender_device_id": "alice",
                    "message_id": f"a{sequence}", "sequence": sequence,
                    "nonce": f"na{sequence}", "ciphertext": "ct"})
            state_store = attach_persistence(persisted, path)

            def raise_oserror(*_args, **_kwargs):
                raise OSError("simulated write failure")

            state_store.save = raise_oserror
            server, service = create_server(
                "127.0.0.1", 0, persisted)
            port = server.server_address[1]
            thread = threading.Thread(target=server.serve_forever,
                                      daemon=True)
            thread.start()
            try:
                conn = HTTPConnection("127.0.0.1", port, timeout=5)
                conn.request(
                    "POST", "/v1/devices/bob/inbox/claim",
                    body=json.dumps({"lease_id": "L1", "limit": 2}),
                    headers={"Content-Type": "application/json"})
                response = conn.getresponse()
                self.assertEqual(response.status, 503)
                body = json.loads(response.read().decode("utf-8"))
                self.assertEqual(list(body), ["message", "field"])
                self.assertEqual(body["field"], "data_file")
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)
        finally:
            shutil.rmtree(directory, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
