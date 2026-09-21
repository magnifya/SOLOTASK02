"""Tests for claim-based session establishment (``POST /v1/sessions/from-claim``).

Covers frozen-value sessions, the 400/404/409/503 error contract, one-session
per claim_id idempotency, failed attempts not consuming the claim, shared-lock
race linearization, durable persistence/restart recovery (including old files
lacking the bindings section) and refusal on a malformed bindings section.
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
    JsonStateStore,
    attach_persistence,
)
from e2ee_backend.service import DeviceService, ServiceError
from e2ee_backend.http_app import create_server


def _raw_key_b64() -> str:
    key = x25519.X25519PrivateKey.generate().public_key()
    raw = key.public_bytes(serialization.Encoding.Raw,
                           serialization.PublicFormat.Raw)
    return base64.b64encode(raw).decode()


def _register_payload(device_id: str, key_ids=("k1", "k2")) -> dict:
    return {
        "user_id": "u1",
        "device_id": device_id,
        "identity_key": _raw_key_b64(),
        "signed_prekeys": [{"key_id": kid, "public_key": _raw_key_b64()}
                           for kid in key_ids],
    }


_SESSION_FIELDS = {"session_id", "initiator_device_id",
                   "recipient_device_id", "prekey_id", "ephemeral_key",
                   "identity_key", "public_key", "created_at"}


class SessionFromClaimServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = DeviceService()
        recipient = _register_payload("r1")
        self.keys = {pk["key_id"]: pk["public_key"]
                     for pk in recipient["signed_prekeys"]}
        self.identity = recipient["identity_key"]
        self.service.register(recipient)
        self.service.register(_register_payload("i1", key_ids=()))
        self.claim, status = self.service.claim_prekey(
            {"recipient_device_id": "r1", "claim_id": "c1"})
        self.assertEqual(status, 201)
        self.ephemeral_key = _raw_key_b64()

    def _create(self, claim_id="c1", initiator="i1", ephemeral_key=None):
        return self.service.create_session_from_claim({
            "claim_id": claim_id,
            "initiator_device_id": initiator,
            "ephemeral_key": self.ephemeral_key if ephemeral_key is None
            else ephemeral_key,
        })

    def test_success_returns_eight_fields_frozen_from_claim(self) -> None:
        body = self._create()
        self.assertEqual(set(body), _SESSION_FIELDS)
        self.assertEqual(body["recipient_device_id"], "r1")
        self.assertEqual(body["prekey_id"], "k1")
        self.assertEqual(body["identity_key"], self.identity)
        self.assertEqual(body["public_key"], self.keys["k1"])
        self.assertEqual(body["initiator_device_id"], "i1")
        self.assertEqual(body["ephemeral_key"], self.ephemeral_key)
        self.assertTrue(body["created_at"].endswith("+00:00"))
        self.assertTrue(body["session_id"])

    def test_get_returns_the_same_frozen_snapshot(self) -> None:
        created = self._create()
        self.assertEqual(self.service.get_session(created["session_id"]),
                         created)

    def test_frozen_identity_survives_recipient_rotation(self) -> None:
        created = self._create()
        self.service.rotate_identity_key("r1", {"identity_key": _raw_key_b64()})
        self.assertEqual(
            self.service.get_session(created["session_id"])["identity_key"],
            self.identity)

    def test_duplicate_claim_is_409_and_creates_no_session(self) -> None:
        first = self._create()
        with self.assertRaises(ServiceError) as ctx:
            self._create(ephemeral_key=_raw_key_b64())
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "claim_id")
        # Even a different initiator cannot reuse the claim.
        self.service.register(_register_payload("i2", key_ids=()))
        with self.assertRaises(ServiceError) as ctx:
            self._create(initiator="i2", ephemeral_key=_raw_key_b64())
        self.assertEqual(ctx.exception.field, "claim_id")
        self.assertEqual(len(self.service.store._sessions), 1)
        self.assertEqual(set(self.service.store._claim_session_bindings),
                         {"c1"})
        self.assertEqual(
            self.service.store._claim_session_bindings["c1"].session_id,
            first["session_id"])

    def test_unknown_claim_is_404_claim_id(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self._create(claim_id="nope")
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "claim_id")

    def test_unknown_initiator_is_404(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self._create(initiator="ghost")
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "initiator_device_id")

    def test_revoked_initiator_is_409(self) -> None:
        self.service.revoke_device("i1")
        with self.assertRaises(ServiceError) as ctx:
            self._create()
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "initiator_device_id")

    def test_revoked_recipient_is_409(self) -> None:
        self.service.revoke_device("r1")
        with self.assertRaises(ServiceError) as ctx:
            self._create()
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "recipient_device_id")

    def test_revoked_claimed_prekey_is_409(self) -> None:
        self.service.revoke_prekey("r1", "k1")
        with self.assertRaises(ServiceError) as ctx:
            self._create()
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "prekey_id")

    def test_failed_attempt_does_not_consume_the_claim(self) -> None:
        with self.assertRaises(ServiceError):
            self._create(initiator="ghost")
        self.service.revoke_prekey("r1", "k1")
        with self.assertRaises(ServiceError):
            self._create()
        # No binding after the failures; the claim is still occupied by the
        # pre-key consumption, but has not established a session.
        self.assertEqual(self.service.store._claim_session_bindings, {})
        self.assertEqual(len(self.service.store._sessions), 0)

    def test_claim_becomes_usable_once_condition_is_fixed(self) -> None:
        # Unknown initiator now, registered later: the same claim then succeeds.
        self.service.register(_register_payload("late", key_ids=()))
        with self.assertRaises(ServiceError) as ctx:
            self._create(initiator="not-yet")
        self.assertEqual(ctx.exception.field, "initiator_device_id")
        self.service.register(_register_payload("not-yet", key_ids=()))
        body = self._create(initiator="not-yet")
        self.assertEqual(body["prekey_id"], "k1")

    def test_failure_after_success_is_still_a_duplicate(self) -> None:
        self._create()
        self.service.revoke_device("r1")
        with self.assertRaises(ServiceError) as ctx:
            self._create()
        # The binding wins: a repeat after revocation reports the duplicate,
        # not the revoked recipient.
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "claim_id")


class SessionFromClaimValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = DeviceService()
        self.service.register(_register_payload("r1"))
        self.service.claim_prekey(
            {"recipient_device_id": "r1", "claim_id": "c1"})

    def _assert_400(self, payload: object, field: str) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.create_session_from_claim(payload)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.field, field)

    def test_body_must_be_object(self) -> None:
        self._assert_400(["not", "an", "object"], "request_body")

    def test_missing_fields_are_400_with_field_name(self) -> None:
        self._assert_400({}, "claim_id")
        self._assert_400({"claim_id": "c1"}, "initiator_device_id")
        self._assert_400({"claim_id": "c1", "initiator_device_id": "i1"},
                         "ephemeral_key")

    def test_empty_or_wrong_typed_fields_are_400(self) -> None:
        self._assert_400({"claim_id": "", "initiator_device_id": "i1",
                          "ephemeral_key": _raw_key_b64()}, "claim_id")
        self._assert_400({"claim_id": 9, "initiator_device_id": "i1",
                          "ephemeral_key": _raw_key_b64()}, "claim_id")
        self._assert_400({"claim_id": "c1", "initiator_device_id": None,
                          "ephemeral_key": _raw_key_b64()},
                         "initiator_device_id")
        self._assert_400({"claim_id": "c1", "initiator_device_id": "i1",
                          "ephemeral_key": "not-a-key"}, "ephemeral_key")
        self._assert_400({"claim_id": "c1", "initiator_device_id": "i1",
                          "ephemeral_key": 7}, "ephemeral_key")


class SessionFromClaimConcurrencyTest(unittest.TestCase):
    def test_parallel_same_claim_creates_exactly_one_session(self) -> None:
        service = DeviceService()
        service.register(_register_payload("r1"))
        for index in range(8):
            service.register(_register_payload(f"i{index}", key_ids=()))
        service.claim_prekey({"recipient_device_id": "r1", "claim_id": "dup"})

        results = []
        lock = threading.Lock()
        barrier = threading.Barrier(8)

        def worker(index: int) -> None:
            barrier.wait()
            try:
                body = service.create_session_from_claim({
                    "claim_id": "dup",
                    "initiator_device_id": f"i{index}",
                    "ephemeral_key": _raw_key_b64(),
                })
                with lock:
                    results.append(("ok", body["session_id"]))
            except ServiceError as error:
                with lock:
                    results.append(("conflict", error.field))

        threads = [threading.Thread(target=worker, args=(i,))
                   for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        created = [sid for outcome, sid in results if outcome == "ok"]
        conflicts = [field for outcome, field in results
                     if outcome == "conflict"]
        self.assertEqual(len(created), 1)
        self.assertEqual(len(conflicts), 7)
        self.assertTrue(all(field == "claim_id" for field in conflicts))
        self.assertEqual(len(service.store._sessions), 1)
        self.assertEqual(set(service.store._claim_session_bindings), {"dup"})


class SessionFromClaimPersistenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.path = os.path.join(self.directory, "state.json")

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def _fresh_service(self) -> DeviceService:
        service = DeviceService()
        attach_persistence(service, self.path)
        return service

    def _setup_bound_claim(self) -> tuple:
        service = self._fresh_service()
        service.register(_register_payload("r1"))
        service.register(_register_payload("i1", key_ids=()))
        claim, _ = service.claim_prekey(
            {"recipient_device_id": "r1", "claim_id": "c1"})
        body = service.create_session_from_claim({
            "claim_id": "c1", "initiator_device_id": "i1",
            "ephemeral_key": _raw_key_b64()})
        return service, claim, body

    def test_binding_survives_restart_and_blocks_repeat(self) -> None:
        _service, claim, body = self._setup_bound_claim()
        restarted = self._fresh_service()
        self.assertEqual(set(restarted.store._claim_session_bindings), {"c1"})
        with self.assertRaises(ServiceError) as ctx:
            restarted.create_session_from_claim({
                "claim_id": "c1", "initiator_device_id": "i1",
                "ephemeral_key": _raw_key_b64()})
        self.assertEqual(ctx.exception.field, "claim_id")
        self.assertEqual(restarted.get_session(body["session_id"]), body)
        # A different committed claim still establishes a session.
        restarted.claim_prekey({"recipient_device_id": "r1", "claim_id": "c2"})
        second = restarted.create_session_from_claim({
            "claim_id": "c2", "initiator_device_id": "i1",
            "ephemeral_key": _raw_key_b64()})
        self.assertEqual(second["prekey_id"], "k2")
        self.assertEqual(second["identity_key"], claim["identity_key"])

    def test_persisted_record_has_contract_fields(self) -> None:
        self._setup_bound_claim()
        with open(self.path, encoding="utf-8") as handle:
            document = json.load(handle)
        self.assertEqual(document["version"], 1)
        (record,) = document["claim_session_bindings"]
        self.assertEqual(set(record), {
            "claim_id", "session_id", "recipient_device_id",
            "prekey_id", "identity_key", "public_key", "created_at"})
        self.assertEqual(record["claim_id"], "c1")
        self.assertEqual(record["recipient_device_id"], "r1")

    def test_old_file_without_section_loads_empty(self) -> None:
        self._setup_bound_claim()
        with open(self.path, encoding="utf-8") as handle:
            document = json.load(handle)
        del document["claim_session_bindings"]
        old_path = os.path.join(self.directory, "old.json")
        with open(old_path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        service = DeviceService()
        attach_persistence(service, old_path)
        self.assertEqual(service.store._claim_session_bindings, {})
        # The old claim may now establish a (new) session.
        body = service.create_session_from_claim({
            "claim_id": "c1", "initiator_device_id": "i1",
            "ephemeral_key": _raw_key_b64()})
        self.assertEqual(body["prekey_id"], "k1")

    def _assert_refused(self, path: str) -> None:
        with self.assertRaises(StateFileError):
            attach_persistence(DeviceService(), path)

    def _mutated_document(self, mutate) -> str:
        with open(self.path, encoding="utf-8") as handle:
            document = json.load(handle)
        mutate(document)
        path = os.path.join(self.directory, f"mut-{id(mutate)}.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        return path

    def test_malformed_sections_are_refused(self) -> None:
        self._setup_bound_claim()

        def wrong_type(doc):
            doc["claim_session_bindings"] = "nope"

        def unknown_claim(doc):
            doc["claim_session_bindings"][0]["claim_id"] = "ghost"

        def unknown_session(doc):
            doc["claim_session_bindings"][0]["session_id"] = "deadbeef"

        def duplicate(doc):
            doc["claim_session_bindings"].append(
                dict(doc["claim_session_bindings"][0]))

        def mismatched_key(doc):
            prekeys = next(d for d in doc["devices"]
                           if d["device_id"] == "r1")["prekeys"]
            doc["claim_session_bindings"][0]["public_key"] = prekeys[1][
                "public_key"]

        for mutate in (wrong_type, unknown_claim, unknown_session,
                       duplicate, mismatched_key):
            self._assert_refused(self._mutated_document(mutate))

    def test_failed_durable_write_rolls_binding_and_session_back(self) -> None:
        service = self._fresh_service()
        service.register(_register_payload("r1"))
        service.register(_register_payload("i1", key_ids=()))
        service.claim_prekey({"recipient_device_id": "r1", "claim_id": "c1"})
        original_save = JsonStateStore.save

        def fail_save(self, state):  # noqa: ANN001
            raise OSError("simulated disk failure")

        JsonStateStore.save = fail_save
        try:
            with self.assertRaises(PersistenceUnavailable):
                service.create_session_from_claim({
                    "claim_id": "c1", "initiator_device_id": "i1",
                    "ephemeral_key": _raw_key_b64()})
        finally:
            JsonStateStore.save = original_save
        # Nothing survived in memory; the claim is still usable.
        self.assertEqual(service.store._claim_session_bindings, {})
        self.assertEqual(len(service.store._sessions), 0)
        body = service.create_session_from_claim({
            "claim_id": "c1", "initiator_device_id": "i1",
            "ephemeral_key": _raw_key_b64()})
        self.assertEqual(body["prekey_id"], "k1")


class SessionFromClaimHTTPTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = DeviceService()
        self.service.register(_register_payload("r1"))
        self.service.register(_register_payload("i1", key_ids=()))
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
        connection.request("POST", "/v1/sessions/from-claim",
                           body=json.dumps(body),
                           headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        data = response.read().decode("utf-8")
        connection.close()
        return response.status, json.loads(data)

    def _claim(self, claim_id="c1") -> None:
        status, _ = self._request_claim(claim_id)
        self.assertEqual(status, 201)

    def _request_claim(self, claim_id: str):
        connection = HTTPConnection("127.0.0.1", self.port, timeout=5)
        connection.request("POST", "/v1/prekeys/claim",
                           body=json.dumps({"recipient_device_id": "r1",
                                            "claim_id": claim_id}),
                           headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        data = response.read().decode("utf-8")
        connection.close()
        return response.status, json.loads(data)

    def test_create_201_then_repeat_409(self) -> None:
        self._claim()
        status, body = self._request(
            {"claim_id": "c1", "initiator_device_id": "i1",
             "ephemeral_key": _raw_key_b64()})
        self.assertEqual(status, 201)
        self.assertEqual(set(body), _SESSION_FIELDS)
        self.assertEqual(body["recipient_device_id"], "r1")
        status2, body2 = self._request(
            {"claim_id": "c1", "initiator_device_id": "i1",
             "ephemeral_key": _raw_key_b64()})
        self.assertEqual(status2, 409)
        self.assertEqual(body2["field"], "claim_id")

    def test_error_status_codes_and_fields(self) -> None:
        status, body = self._request({})
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "claim_id")
        status, body = self._request(
            {"claim_id": "c1", "initiator_device_id": "i1"})
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "ephemeral_key")
        status, body = self._request(
            {"claim_id": "nope", "initiator_device_id": "i1",
             "ephemeral_key": _raw_key_b64()})
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "claim_id")
        self._claim("c2")
        status, body = self._request(
            {"claim_id": "c2", "initiator_device_id": "ghost",
             "ephemeral_key": _raw_key_b64()})
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "initiator_device_id")

    def test_original_sessions_endpoint_is_unchanged(self) -> None:
        connection = HTTPConnection("127.0.0.1", self.port, timeout=5)
        connection.request("POST", "/v1/sessions", body=json.dumps({
            "initiator_device_id": "i1", "recipient_device_id": "r1",
            "prekey_id": "k1", "ephemeral_key": _raw_key_b64()}),
            headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        data = json.loads(response.read().decode("utf-8"))
        connection.close()
        self.assertEqual(response.status, 201)
        self.assertEqual(set(data), _SESSION_FIELDS)


if __name__ == "__main__":
    unittest.main()
