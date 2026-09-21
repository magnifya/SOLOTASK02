"""Tests for one-time pre-key claims (``POST /v1/prekeys/claim``).

Covers registration-order selection, atomic consumption, claim-id
idempotency, the 400/404/409/503 error contract, exclusion of consumed keys
from the public listing and from session creation, shared-lock race
linearization, durable persistence/restart recovery, and refusal to start on
a malformed ``prekey_claims`` section.
"""
import base64
import json
import os
import shutil
import tempfile
import threading
import unittest
from http.client import HTTPConnection

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import x25519

from e2ee_backend.persistence import (
    PersistenceUnavailable,
    StateFileError,
    attach_persistence,
)
from e2ee_backend.service import DeviceService, ServiceError
from e2ee_backend.http_app import create_server


def _raw_key_b64() -> str:
    key = x25519.X25519PrivateKey.generate().public_key()
    raw = key.public_bytes(serialization.Encoding.Raw,
                           serialization.PublicFormat.Raw)
    return base64.b64encode(raw).decode()


def _register_payload(device_id: str = "d1", key_ids=("k1", "k2")) -> dict:
    return {
        "user_id": "u1",
        "device_id": device_id,
        "identity_key": _raw_key_b64(),
        "signed_prekeys": [{"key_id": kid, "public_key": _raw_key_b64()}
                           for kid in key_ids],
    }


_CLAIM_FIELDS = {"claim_id", "recipient_device_id", "identity_key", "key_id",
                 "public_key", "claimed_at"}


class ClaimServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = DeviceService()
        self.payload = _register_payload()
        self.keys = {pk["key_id"]: pk["public_key"]
                     for pk in self.payload["signed_prekeys"]}
        self.identity = self.payload["identity_key"]
        self.service.register(self.payload)

    def _claim(self, claim_id="c1", device="d1"):
        return self.service.claim_prekey(
            {"recipient_device_id": device, "claim_id": claim_id})

    def test_first_claim_returns_first_key_with_six_fields_201(self) -> None:
        body, status = self._claim("c1")
        self.assertEqual(status, 201)
        self.assertEqual(set(body), _CLAIM_FIELDS)
        self.assertEqual(body["claim_id"], "c1")
        self.assertEqual(body["recipient_device_id"], "d1")
        self.assertEqual(body["key_id"], "k1")
        self.assertEqual(body["public_key"], self.keys["k1"])
        self.assertEqual(body["identity_key"], self.identity)
        self.assertTrue(body["claimed_at"].endswith("+00:00"))

    def test_repeated_claim_id_is_200_and_identical_without_reconsuming(self
                                                                        ) -> None:
        first, status1 = self._claim("c1")
        second, status2 = self._claim("c1")
        self.assertEqual(status1, 201)
        self.assertEqual(status2, 200)
        self.assertEqual(first, second)
        # Only k1 was consumed; k2 stays available.
        self.assertEqual(self.service.get_device("d1")["prekey_ids"], ["k2"])

    def test_different_claim_id_takes_the_next_key(self) -> None:
        first, _ = self._claim("c1")
        second, status = self._claim("c2")
        self.assertEqual(status, 201)
        self.assertEqual(first["key_id"], "k1")
        self.assertEqual(second["key_id"], "k2")

    def test_claimed_keys_disappear_from_public_listing_in_order(self) -> None:
        self.assertEqual(self.service.get_device("d1")["prekey_ids"],
                         ["k1", "k2"])
        self._claim("c1")
        self.assertEqual(self.service.get_device("d1")["prekey_ids"], ["k2"])
        self._claim("c2")
        self.assertEqual(self.service.get_device("d1")["prekey_ids"], [])

    def test_no_available_key_is_409_prekey_id(self) -> None:
        self._claim("c1")
        self._claim("c2")
        with self.assertRaises(ServiceError) as ctx:
            self._claim("c3")
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "prekey_id")

    def test_revoked_key_is_skipped_and_exhaustion_conflicts(self) -> None:
        self.service.revoke_prekey("d1", "k1")
        body, status = self._claim("c1")
        self.assertEqual(status, 201)
        self.assertEqual(body["key_id"], "k2")
        with self.assertRaises(ServiceError) as ctx:
            self._claim("c2")
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "prekey_id")

    def test_unknown_recipient_is_404_field(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self._claim("c1", device="ghost")
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "recipient_device_id")

    def test_revoked_recipient_is_409_field(self) -> None:
        self.service.revoke_device("d1")
        with self.assertRaises(ServiceError) as ctx:
            self._claim("c9")
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "recipient_device_id")

    def test_replay_succeeds_after_recipient_revocation(self) -> None:
        first, _ = self._claim("c1")
        self.service.revoke_device("d1")
        replay, status = self._claim("c1")
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)

    def test_session_creation_with_consumed_key_is_409_prekey_id(self) -> None:
        # An initiator device is required for session creation.
        self.service.register(_register_payload("init", ("i1",)))
        self._claim("c1")  # consumes k1 of d1
        payload = {
            "initiator_device_id": "init",
            "recipient_device_id": "d1",
            "prekey_id": "k1",
            "ephemeral_key": _raw_key_b64(),
        }
        with self.assertRaises(ServiceError) as ctx:
            self.service.create_session(payload)
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "prekey_id")
        # The still-available k2 works.
        payload["prekey_id"] = "k2"
        session = self.service.create_session(payload)
        self.assertEqual(session["prekey_id"], "k2")


class ClaimValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = DeviceService()
        self.service.register(_register_payload())

    def _assert_400(self, payload: object, field: str) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.claim_prekey(payload)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.field, field)

    def test_body_must_be_object(self) -> None:
        self._assert_400(["not", "an", "object"], "request_body")

    def test_missing_fields_are_400_with_field_name(self) -> None:
        self._assert_400({}, "recipient_device_id")
        self._assert_400({"recipient_device_id": "d1"}, "claim_id")

    def test_empty_or_typed_wrong_fields_are_400(self) -> None:
        self._assert_400({"recipient_device_id": "", "claim_id": "c"},
                         "recipient_device_id")
        self._assert_400({"recipient_device_id": "d1", "claim_id": ""},
                         "claim_id")
        self._assert_400({"recipient_device_id": 9, "claim_id": "c"},
                         "recipient_device_id")
        self._assert_400({"recipient_device_id": "d1", "claim_id": 7},
                         "claim_id")
        self._assert_400({"recipient_device_id": None, "claim_id": "c"},
                         "recipient_device_id")

    def test_failed_validation_consumes_no_key(self) -> None:
        for bad in ({}, {"recipient_device_id": "d1"},
                    {"recipient_device_id": "d1", "claim_id": ""}):
            with self.assertRaises(ServiceError):
                self.service.claim_prekey(bad)
        self.assertEqual(self.service.get_device("d1")["prekey_ids"],
                         ["k1", "k2"])


class ClaimConcurrencyTest(unittest.TestCase):
    def _build(self, n_keys: int) -> DeviceService:
        service = DeviceService()
        service.register(_register_payload("d1", [f"k{i}" for i in range(n_keys)]))
        return service

    def test_parallel_distinct_claims_linearize_to_one_per_key(self) -> None:
        n_keys = 30
        service = self._build(n_keys)
        results = []
        lock = threading.Lock()

        def worker(index: int) -> None:
            try:
                body, created = service.claim_prekey(
                    {"recipient_device_id": "d1", "claim_id": f"c{index}"})
                with lock:
                    results.append(("ok", body["key_id"]))
            except ServiceError as error:
                with lock:
                    results.append(("conflict", error.field))

        threads = [threading.Thread(target=worker, args=(i,))
                   for i in range(n_keys * 2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        claimed = [key for outcome, key in results if outcome == "ok"]
        conflict_fields = [field for outcome, field in results
                           if outcome == "conflict"]
        self.assertEqual(len(claimed), n_keys)
        self.assertEqual(len(set(claimed)), n_keys)
        self.assertEqual(len(conflict_fields), n_keys)
        self.assertTrue(all(field == "prekey_id" for field in conflict_fields))
        self.assertEqual(service.get_device("d1")["prekey_ids"], [])

    def test_parallel_same_claim_id_consumes_exactly_one_key(self) -> None:
        service = self._build(5)
        results = []
        lock = threading.Lock()
        barrier = threading.Barrier(8)

        def worker() -> None:
            barrier.wait()
            body, status = service.claim_prekey(
                {"recipient_device_id": "d1", "claim_id": "dup"})
            with lock:
                results.append((status, body["key_id"]))

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(results), 8)
        self.assertEqual({key for _, key in results}, {"k0"})
        self.assertEqual(sum(1 for status, _ in results if status == 201), 1)
        self.assertEqual(sum(1 for status, _ in results if status == 200), 7)
        # Exactly one key consumed even though eight clients raced.
        self.assertEqual(service.get_device("d1")["prekey_ids"],
                         ["k1", "k2", "k3", "k4"])


class ClaimPersistenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.path = os.path.join(self.directory, "state.json")

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def _fresh_service(self) -> DeviceService:
        service = DeviceService()
        attach_persistence(service, self.path)
        return service

    def test_claim_survives_restart_and_replay_stays_idempotent(self) -> None:
        service = self._fresh_service()
        service.register(_register_payload())
        first, status = service.claim_prekey(
            {"recipient_device_id": "d1", "claim_id": "c1"})
        self.assertEqual(status, 201)
        service.claim_prekey({"recipient_device_id": "d1", "claim_id": "c2"})

        restarted = self._fresh_service()
        # The consumed keys stay excluded after recovery.
        self.assertEqual(restarted.get_device("d1")["prekey_ids"], [])
        # A replay of c1 returns the frozen record with 200, no new key.
        replay, status = restarted.claim_prekey(
            {"recipient_device_id": "d1", "claim_id": "c1"})
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)
        # No keys remain: a fresh claim conflicts.
        with self.assertRaises(ServiceError) as ctx:
            restarted.claim_prekey(
                {"recipient_device_id": "d1", "claim_id": "c3"})
        self.assertEqual(ctx.exception.field, "prekey_id")

    def test_persisted_record_has_device_id_and_contract_fields(self) -> None:
        service = self._fresh_service()
        service.register(_register_payload())
        service.claim_prekey({"recipient_device_id": "d1", "claim_id": "c1"})
        with open(self.path, encoding="utf-8") as handle:
            document = json.load(handle)
        (record,) = document["prekey_claims"]
        self.assertEqual(set(record),
                         {"claim_id", "device_id", "key_id", "identity_key",
                          "public_key", "claimed_at"})
        self.assertEqual(record["device_id"], "d1")
        # The consumed flag is persisted on the pre-key.
        prekeys = document["devices"][0]["prekeys"]
        self.assertTrue(prekeys[0]["consumed"])
        self.assertFalse(prekeys[1]["consumed"])

    def _base_document(self) -> dict:
        service = self._fresh_service()
        service.register(_register_payload())
        service.claim_prekey({"recipient_device_id": "d1", "claim_id": "c1"})
        with open(self.path, encoding="utf-8") as handle:
            return json.load(handle)

    def _assert_refused(self) -> None:
        with self.assertRaises(StateFileError):
            attach_persistence(DeviceService(), self.path)

    def test_duplicate_claim_id_section_is_refused(self) -> None:
        doc = self._base_document()
        doc["prekey_claims"].append(dict(doc["prekey_claims"][0]))
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(doc, handle)
        self._assert_refused()

    def test_claim_with_wrong_field_type_is_refused(self) -> None:
        doc = self._base_document()
        doc["prekey_claims"][0]["key_id"] = 5
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(doc, handle)
        self._assert_refused()

    def test_claim_referencing_unknown_device_is_refused(self) -> None:
        doc = self._base_document()
        doc["prekey_claims"][0]["device_id"] = "ghost"
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(doc, handle)
        self._assert_refused()

    def test_claim_referencing_foreign_key_is_refused(self) -> None:
        doc = self._base_document()
        doc["prekey_claims"][0]["key_id"] = "k2"
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(doc, handle)
        self._assert_refused()

    def test_consumed_key_without_claim_record_is_refused(self) -> None:
        doc = self._base_document()
        doc["prekey_claims"] = []
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(doc, handle)
        self._assert_refused()

    def test_claim_naming_unconsumed_key_is_refused(self) -> None:
        # Add a claim for k2 but mark only k1 consumed (the fixture state).
        doc = self._base_document()
        doc["prekey_claims"][0]["key_id"] = "k2"
        doc["prekey_claims"][0]["public_key"] = next(
            pk["public_key"] for pk in doc["devices"][0]["prekeys"]
            if pk["key_id"] == "k2")
        doc["prekey_claims"][0]["claim_id"] = "c2"
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(doc, handle)
        self._assert_refused()

    def test_failed_durable_write_rolls_the_claim_back(self) -> None:
        service = self._fresh_service()
        service.register(_register_payload())
        # Force the next durable write to fail; the persist transaction must
        # roll the in-memory store back to the last good state.
        from e2ee_backend.persistence import JsonStateStore
        original_save = JsonStateStore.save

        def fail_save(self, state):  # noqa: ANN001
            raise OSError("simulated disk failure")

        JsonStateStore.save = fail_save
        try:
            with self.assertRaises(PersistenceUnavailable):
                service.claim_prekey(
                    {"recipient_device_id": "d1", "claim_id": "c1"})
        finally:
            JsonStateStore.save = original_save
        # Neither the claim nor the consumption survived in memory.
        self.assertNotIn("c1", service.store._prekey_claims)
        self.assertEqual(service.get_device("d1")["prekey_ids"], ["k1", "k2"])
        # A retry after the disk recovers succeeds.
        body, status = service.claim_prekey(
            {"recipient_device_id": "d1", "claim_id": "c1"})
        self.assertEqual(status, 201)
        self.assertEqual(body["key_id"], "k1")


class ClaimHTTPTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = DeviceService()
        self.service.register(_register_payload())
        self.server, _ = create_server("127.0.0.1", 0, service=self.service)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _request(self, body: object):
        connection = HTTPConnection("127.0.0.1", self.port, timeout=5)
        payload = json.dumps(body)
        connection.request("POST", "/v1/prekeys/claim", body=payload,
                           headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        data = response.read().decode("utf-8")
        connection.close()
        return response.status, json.loads(data)

    def test_claim_201_then_replay_200(self) -> None:
        status, body = self._request(
            {"recipient_device_id": "d1", "claim_id": "c1"})
        self.assertEqual(status, 201)
        self.assertEqual(set(body), _CLAIM_FIELDS)
        status2, body2 = self._request(
            {"recipient_device_id": "d1", "claim_id": "c1"})
        self.assertEqual(status2, 200)
        self.assertEqual(body, body2)

    def test_claim_error_status_codes(self) -> None:
        status, body = self._request({"claim_id": "c"})
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "recipient_device_id")
        status, body = self._request(
            {"recipient_device_id": "ghost", "claim_id": "c"})
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "recipient_device_id")
        self._request({"recipient_device_id": "d1", "claim_id": "a"})
        self._request({"recipient_device_id": "d1", "claim_id": "b"})
        status, body = self._request(
            {"recipient_device_id": "d1", "claim_id": "c"})
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "prekey_id")


if __name__ == "__main__":
    unittest.main()
