"""Tests for atomic multi-device session creation from a batch claim
(``POST /v1/sessions/from-batch-claim``).

Covers the 201 body (claim_id plus sessions in claim order, eight frozen
fields per session), the 400/404/409/503 status/field contract, the
device-set-must-equal-snapshot rule, the initiator-in-snapshot 400, one-batch
per claim_id idempotency, all-or-nothing failure with no sessions created,
shared-lock race linearization, durable persistence/restart recovery
(including old files lacking the bindings section, frozen-vs-rotated
identity, registration-order contradictions and malformed sections refusing
startup), the HTTP endpoint, and the ``create-batch-sessions`` CLI contract.
"""
import base64
import json
import os
import shutil
import subprocess
import sys
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


def _register_payload(user_id: str = "u1", device_id: str = "d1",
                      key_ids=("k1", "k2", "k3"), identity: str | None = None
                      ) -> dict:
    return {
        "user_id": user_id,
        "device_id": device_id,
        "identity_key": identity or _raw_key_b64(),
        "signed_prekeys": [{"key_id": kid, "public_key": _raw_key_b64()}
                           for kid in key_ids],
    }


_SESSION_FIELDS = {"session_id", "initiator_device_id",
                   "recipient_device_id", "prekey_id", "ephemeral_key",
                   "identity_key", "public_key", "created_at"}


class BatchSessionServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = DeviceService()
        self.p1 = _register_payload("u1", "d1", ("a1", "a2", "a3"))
        self.p2 = _register_payload("u1", "d2", ("b1", "b2", "b3"))
        self.pi = _register_payload("u2", "i1", ())
        for payload in (self.p1, self.p2, self.pi):
            self.service.register(payload)
        self.batch, _ = self.service.claim_prekey_batch(
            {"user_id": "u1", "claim_id": "bc1"})

    def _keys(self, batch=None):
        batch = batch or self.batch
        return [{"device_id": e["device_id"], "ephemeral_key": _raw_key_b64()}
                for e in batch["devices"]]

    def _create(self, claim_id="bc1", initiator="i1", keys=None):
        return self.service.create_sessions_from_batch_claim({
            "claim_id": claim_id,
            "initiator_device_id": initiator,
            "ephemeral_keys": keys if keys is not None else self._keys(),
        })

    def test_success_lists_eight_field_sessions_in_claim_order(self) -> None:
        body = self._create()
        self.assertEqual(set(body), {"claim_id", "sessions"})
        self.assertEqual(body["claim_id"], "bc1")
        sessions = body["sessions"]
        self.assertEqual([s["recipient_device_id"] for s in sessions],
                         ["d1", "d2"])
        for session in sessions:
            self.assertEqual(set(session), _SESSION_FIELDS)
        first, second = sessions
        self.assertEqual(first["prekey_id"], "a1")
        self.assertEqual(first["identity_key"], self.p1["identity_key"])
        self.assertEqual(first["public_key"],
                         self.p1["signed_prekeys"][0]["public_key"])
        self.assertEqual(second["prekey_id"], "b1")
        self.assertEqual(second["identity_key"], self.p2["identity_key"])
        self.assertTrue(first["created_at"].endswith("+00:00"))
        self.assertNotEqual(first["session_id"], second["session_id"])
        self.assertTrue(all(s["initiator_device_id"] == "i1"
                            for s in sessions))

    def test_ephemeral_keys_come_from_the_request_per_device(self) -> None:
        keys = self._keys()
        body = self._create(keys=keys)
        by_device = {s["recipient_device_id"]: s["ephemeral_key"]
                     for s in body["sessions"]}
        self.assertEqual(by_device, {e["device_id"]: e["ephemeral_key"]
                                     for e in keys})

    def test_get_returns_each_frozen_snapshot(self) -> None:
        body = self._create()
        for session in body["sessions"]:
            self.assertEqual(
                self.service.get_session(session["session_id"]), session)

    def test_frozen_identity_survives_rotation(self) -> None:
        body = self._create()
        frozen = body["sessions"][0]["identity_key"]
        self.service.rotate_identity_key("d1",
                                         {"identity_key": _raw_key_b64()})
        self.assertEqual(
            self.service.get_session(body["sessions"][0]["session_id"])[
                "identity_key"], frozen)

    def test_duplicate_is_409_claim_id_and_creates_nothing(self) -> None:
        first = self._create()
        self.assertEqual(len(first["sessions"]), 2)
        with self.assertRaises(ServiceError) as ctx:
            self._create(keys=self._keys())
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "claim_id")
        self.assertEqual(len(self.service.store._sessions), 2)
        self.assertEqual(
            set(self.service.store._batch_claim_session_bindings), {"bc1"})
        bound = self.service.store._batch_claim_session_bindings["bc1"]
        self.assertEqual([e.session_id for e in bound.entries],
                         [s["session_id"] for s in first["sessions"]])

    def test_unknown_claim_is_404_claim_id(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self._create(claim_id="nope")
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "claim_id")

    def test_single_claim_id_is_409_claim_id(self) -> None:
        self.service.claim_prekey(
            {"recipient_device_id": "d1", "claim_id": "single"})
        with self.assertRaises(ServiceError) as ctx:
            self._create(claim_id="single",
                         keys=[{"device_id": "d1",
                                "ephemeral_key": _raw_key_b64()}])
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "claim_id")
        self.assertEqual(len(self.service.store._sessions), 0)

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

    def test_initiator_in_snapshot_is_400_even_when_revoked(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self._create(initiator="d1")
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.field, "initiator_device_id")
        self.assertEqual(len(self.service.store._sessions), 0)
        # A revoked snapshot device still makes the request malformed (400),
        # not a 409 conflict.
        self.service.revoke_device("d2")
        with self.assertRaises(ServiceError) as ctx:
            self._create(initiator="d2")
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.field, "initiator_device_id")
        self.assertEqual(len(self.service.store._sessions), 0)

    def test_revoked_recipient_is_409_naming_device_and_creates_nothing(self
                                                                        ) -> None:
        self.service.revoke_device("d2")
        with self.assertRaises(ServiceError) as ctx:
            self._create()
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "recipient_device_id")
        self.assertEqual(len(self.service.store._sessions), 0)
        # The claim is still usable once the condition cannot recur here
        # (revocation is durable) but a fresh un-revoked claim on another
        # device set still succeeds, proving the failed attempt bound nothing.
        self.service.register(_register_payload("u3", "d3", ("c1",)))
        other, _ = self.service.claim_prekey_batch(
            {"user_id": "u3", "claim_id": "bc2"})
        body = self._create(
            claim_id="bc2",
            keys=[{"device_id": "d3", "ephemeral_key": _raw_key_b64()}])
        self.assertEqual([s["recipient_device_id"] for s in body["sessions"]],
                         ["d3"])

    def test_revoked_prekey_is_409_naming_prekey_id(self) -> None:
        self.service.revoke_prekey("d2", "b1")
        with self.assertRaises(ServiceError) as ctx:
            self._create()
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "prekey_id")
        self.assertEqual(len(self.service.store._sessions), 0)

    def test_missing_ephemeral_device_is_400_ephemeral_keys(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self._create(keys=[{"device_id": "d1",
                                "ephemeral_key": _raw_key_b64()}])
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.field, "ephemeral_keys")

    def test_extra_ephemeral_device_is_400_ephemeral_keys(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self._create(keys=self._keys() + [
                {"device_id": "d3", "ephemeral_key": _raw_key_b64()}])
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.field, "ephemeral_keys")


class BatchSessionValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = DeviceService()
        self.service.register(_register_payload("u1", "d1"))
        self.service.register(_register_payload("u2", "i1", ()))
        self.service.claim_prekey_batch({"user_id": "u1", "claim_id": "bc1"})

    def _assert_400(self, payload: object, field: str) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.create_sessions_from_batch_claim(payload)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.field, field)

    def test_body_must_be_object(self) -> None:
        self._assert_400(["not", "an", "object"], "request_body")

    def test_missing_scalars_are_400_with_field_name(self) -> None:
        self._assert_400({}, "claim_id")
        self._assert_400({"claim_id": "bc1"}, "initiator_device_id")
        self._assert_400({"claim_id": "bc1", "initiator_device_id": "i1"},
                         "ephemeral_keys")

    def test_empty_or_bad_scalars_are_400(self) -> None:
        self._assert_400({"claim_id": "", "initiator_device_id": "i1",
                          "ephemeral_keys": []}, "claim_id")
        self._assert_400({"claim_id": 7, "initiator_device_id": "i1",
                          "ephemeral_keys": []}, "claim_id")
        self._assert_400({"claim_id": "bc1", "initiator_device_id": None,
                          "ephemeral_keys": []}, "initiator_device_id")

    def test_ephemeral_keys_shape_errors(self) -> None:
        base = {"claim_id": "bc1", "initiator_device_id": "i1"}
        self._assert_400({**base, "ephemeral_keys": []}, "ephemeral_keys")
        self._assert_400({**base, "ephemeral_keys": "x"}, "ephemeral_keys")
        self._assert_400({**base, "ephemeral_keys": [5]},
                         "ephemeral_keys[0]")
        self._assert_400({**base, "ephemeral_keys": [{}]},
                         "ephemeral_keys[0].device_id")
        self._assert_400({**base, "ephemeral_keys": [
            {"device_id": "", "ephemeral_key": _raw_key_b64()}]},
            "ephemeral_keys[0].device_id")
        self._assert_400({**base, "ephemeral_keys": [
            {"device_id": "d1"}]},
            "ephemeral_keys[0].ephemeral_key")
        self._assert_400({**base, "ephemeral_keys": [
            {"device_id": "d1", "ephemeral_key": "not-a-key"}]},
            "ephemeral_keys[0].ephemeral_key")

    def test_duplicate_device_id_in_array_is_400(self) -> None:
        self._assert_400(
            {"claim_id": "bc1", "initiator_device_id": "i1",
             "ephemeral_keys": [
                 {"device_id": "d1", "ephemeral_key": _raw_key_b64()},
                 {"device_id": "d1", "ephemeral_key": _raw_key_b64()}]},
            "ephemeral_keys[1].device_id")


class BatchSessionConcurrencyTest(unittest.TestCase):
    def test_parallel_same_batch_creates_exactly_one_batch(self) -> None:
        service = DeviceService()
        service.register(_register_payload("u1", "d1", ("k1",)))
        service.register(_register_payload("u1", "d2", ("k1",)))
        for index in range(8):
            service.register(_register_payload("u2", f"i{index}", ()))
        batch, _ = service.claim_prekey_batch(
            {"user_id": "u1", "claim_id": "dup"})
        keys = [{"device_id": e["device_id"],
                 "ephemeral_key": _raw_key_b64()} for e in batch["devices"]]

        results = []
        lock = threading.Lock()
        barrier = threading.Barrier(8)

        def worker(index: int) -> None:
            barrier.wait()
            try:
                body = service.create_sessions_from_batch_claim({
                    "claim_id": "dup",
                    "initiator_device_id": f"i{index}",
                    "ephemeral_keys": keys})
                with lock:
                    results.append(("ok", len(body["sessions"])))
            except ServiceError as error:
                with lock:
                    results.append(("conflict", error.field))

        threads = [threading.Thread(target=worker, args=(i,))
                   for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        oks = [count for outcome, count in results if outcome == "ok"]
        conflicts = [field for outcome, field in results
                     if outcome == "conflict"]
        self.assertEqual(oks, [2])
        self.assertEqual(len(conflicts), 7)
        self.assertTrue(all(field == "claim_id" for field in conflicts))
        self.assertEqual(len(service.store._sessions), 2)
        self.assertEqual(set(service.store._batch_claim_session_bindings),
                         {"dup"})


class BatchSessionPersistenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.path = os.path.join(self.directory, "state.json")

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def _fresh_service(self) -> DeviceService:
        service = DeviceService()
        attach_persistence(service, self.path)
        return service

    def _setup_bound_batch(self) -> tuple:
        service = self._fresh_service()
        p1 = _register_payload("u1", "d1", ("a1", "a2"))
        p2 = _register_payload("u1", "d2", ("b1", "b2"))
        pi = _register_payload("u2", "i1", ())
        for payload in (p1, p2, pi):
            service.register(payload)
        batch, _ = service.claim_prekey_batch(
            {"user_id": "u1", "claim_id": "bc1"})
        keys = [{"device_id": e["device_id"],
                 "ephemeral_key": _raw_key_b64()} for e in batch["devices"]]
        body = service.create_sessions_from_batch_claim({
            "claim_id": "bc1", "initiator_device_id": "i1",
            "ephemeral_keys": keys})
        return service, batch, body

    def test_binding_survives_restart_and_blocks_repeat(self) -> None:
        _service, batch, body = self._setup_bound_batch()
        restarted = self._fresh_service()
        self.assertEqual(
            set(restarted.store._batch_claim_session_bindings), {"bc1"})
        keys = [{"device_id": e["device_id"],
                 "ephemeral_key": _raw_key_b64()} for e in batch["devices"]]
        with self.assertRaises(ServiceError) as ctx:
            restarted.create_sessions_from_batch_claim({
                "claim_id": "bc1", "initiator_device_id": "i1",
                "ephemeral_keys": keys})
        self.assertEqual(ctx.exception.field, "claim_id")
        for created in body["sessions"]:
            self.assertEqual(restarted.get_session(created["session_id"]),
                             created)

    def test_persisted_record_has_contract_fields(self) -> None:
        self._setup_bound_batch()
        with open(self.path, encoding="utf-8") as handle:
            document = json.load(handle)
        self.assertEqual(document["version"], 1)
        (record,) = document["batch_claim_session_bindings"]
        self.assertEqual(set(record), {"claim_id", "created_at", "entries"})
        self.assertEqual(
            [e["recipient_device_id"] for e in record["entries"]],
            ["d1", "d2"])
        self.assertEqual(set(record["entries"][0]), {
            "session_id", "recipient_device_id", "prekey_id",
            "identity_key", "public_key"})

    def test_old_file_without_section_loads_empty(self) -> None:
        self._setup_bound_batch()
        with open(self.path, encoding="utf-8") as handle:
            document = json.load(handle)
        del document["batch_claim_session_bindings"]
        old_path = os.path.join(self.directory, "old.json")
        with open(old_path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        service = DeviceService()
        attach_persistence(service, old_path)
        self.assertEqual(service.store._batch_claim_session_bindings, {})
        keys = [{"device_id": "d1", "ephemeral_key": _raw_key_b64()},
                {"device_id": "d2", "ephemeral_key": _raw_key_b64()}]
        body = service.create_sessions_from_batch_claim({
            "claim_id": "bc1", "initiator_device_id": "i1",
            "ephemeral_keys": keys})
        self.assertEqual([s["recipient_device_id"] for s in body["sessions"]],
                         ["d1", "d2"])

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
        self._setup_bound_batch()

        def wrong_type(doc):
            doc["batch_claim_session_bindings"] = "nope"

        def unknown_claim(doc):
            doc["batch_claim_session_bindings"][0]["claim_id"] = "ghost"

        def unknown_session(doc):
            doc["batch_claim_session_bindings"][0]["entries"][0][
                "session_id"] = "deadbeef"

        def duplicate_record(doc):
            doc["batch_claim_session_bindings"].append(
                dict(doc["batch_claim_session_bindings"][0]))

        def reordered_entries(doc):
            doc["batch_claim_session_bindings"][0]["entries"].reverse()

        def dropped_entry(doc):
            doc["batch_claim_session_bindings"][0]["entries"].pop()

        def wrong_frozen_identity(doc):
            other = doc["batch_claim_session_bindings"][0]["entries"][1]
            doc["batch_claim_session_bindings"][0]["entries"][0][
                "identity_key"] = other["identity_key"]

        def bound_session_material_mismatch(doc):
            sid = doc["batch_claim_session_bindings"][0]["entries"][0][
                "session_id"]
            session = next(s for s in doc["sessions"]
                           if s["session_id"] == sid)
            session["identity_key"] = next(
                s["identity_key"] for s in doc["sessions"]
                if s["session_id"] != sid)

        for mutate in (wrong_type, unknown_claim, unknown_session,
                       duplicate_record, reordered_entries, dropped_entry,
                       wrong_frozen_identity,
                       bound_session_material_mismatch):
            self._assert_refused(self._mutated_document(mutate))

    def test_registration_order_contradiction_refuses_startup(self) -> None:
        self._setup_bound_batch()
        with open(self.path, encoding="utf-8") as handle:
            document = json.load(handle)
        document["prekey_batch_claims"][0]["devices"].reverse()
        path = os.path.join(self.directory, "reordered.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        self._assert_refused(path)
        # The original formal file is untouched and still loads.
        attach_persistence(DeviceService(), self.path)

    def test_failed_durable_write_rolls_all_sessions_and_binding_back(
            self) -> None:
        service = self._fresh_service()
        service.register(_register_payload("u1", "d1", ("a1",)))
        service.register(_register_payload("u1", "d2", ("b1",)))
        service.register(_register_payload("u2", "i1", ()))
        batch, _ = service.claim_prekey_batch(
            {"user_id": "u1", "claim_id": "bc1"})
        keys = [{"device_id": e["device_id"],
                 "ephemeral_key": _raw_key_b64()} for e in batch["devices"]]
        original_save = JsonStateStore.save

        def fail_save(self, state):  # noqa: ANN001
            raise OSError("simulated disk failure")

        JsonStateStore.save = fail_save
        try:
            with self.assertRaises(PersistenceUnavailable):
                service.create_sessions_from_batch_claim({
                    "claim_id": "bc1", "initiator_device_id": "i1",
                    "ephemeral_keys": keys})
        finally:
            JsonStateStore.save = original_save
        self.assertEqual(service.store._batch_claim_session_bindings, {})
        self.assertEqual(len(service.store._sessions), 0)
        body = service.create_sessions_from_batch_claim({
            "claim_id": "bc1", "initiator_device_id": "i1",
            "ephemeral_keys": keys})
        self.assertEqual([s["recipient_device_id"] for s in body["sessions"]],
                         ["d1", "d2"])


class BatchSessionHTTPTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = DeviceService()
        self.service.register(_register_payload("u1", "d1"))
        self.service.register(_register_payload("u1", "d2"))
        self.service.register(_register_payload("u2", "i1", ()))
        self.server, _ = create_server("127.0.0.1", 0, service=self.service)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _post(self, path: str, body: object):
        connection = HTTPConnection("127.0.0.1", self.port, timeout=5)
        connection.request("POST", path, body=json.dumps(body),
                           headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        data = response.read().decode("utf-8")
        connection.close()
        return response.status, json.loads(data)

    def _claim(self, claim_id: str):
        status, body = self._post("/v1/prekeys/claim-batch",
                                  {"user_id": "u1", "claim_id": claim_id})
        self.assertEqual(status, 201)
        return body

    def _keys(self, batch):
        return [{"device_id": e["device_id"],
                 "ephemeral_key": _raw_key_b64()} for e in batch["devices"]]

    def test_create_201_then_repeat_409(self) -> None:
        batch = self._claim("c1")
        keys = self._keys(batch)
        status, body = self._post("/v1/sessions/from-batch-claim", {
            "claim_id": "c1", "initiator_device_id": "i1",
            "ephemeral_keys": keys})
        self.assertEqual(status, 201)
        self.assertEqual(body["claim_id"], "c1")
        self.assertEqual([s["recipient_device_id"] for s in body["sessions"]],
                         ["d1", "d2"])
        self.assertTrue(all(set(s) == _SESSION_FIELDS
                            for s in body["sessions"]))
        status2, body2 = self._post("/v1/sessions/from-batch-claim", {
            "claim_id": "c1", "initiator_device_id": "i1",
            "ephemeral_keys": keys})
        self.assertEqual(status2, 409)
        self.assertEqual(body2["field"], "claim_id")

    def test_error_status_codes_and_fields(self) -> None:
        status, body = self._post("/v1/sessions/from-batch-claim", {})
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "claim_id")

        batch = self._claim("c2")
        status, body = self._post("/v1/sessions/from-batch-claim", {
            "claim_id": "c2", "initiator_device_id": "i1"})
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "ephemeral_keys")

        status, body = self._post("/v1/sessions/from-batch-claim", {
            "claim_id": "missing", "initiator_device_id": "i1",
            "ephemeral_keys": self._keys(batch)})
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "claim_id")

        status, body = self._post("/v1/sessions/from-batch-claim", {
            "claim_id": "c2", "initiator_device_id": "ghost",
            "ephemeral_keys": self._keys(batch)})
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "initiator_device_id")

        status, body = self._post("/v1/sessions/from-batch-claim", {
            "claim_id": "c2", "initiator_device_id": "d1",
            "ephemeral_keys": self._keys(batch)})
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "initiator_device_id")

        status, body = self._post("/v1/sessions/from-batch-claim", {
            "claim_id": "c2", "initiator_device_id": "i1",
            "ephemeral_keys": self._keys(batch)[:1]})
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "ephemeral_keys")

    def test_single_from_claim_endpoint_is_unchanged(self) -> None:
        status, _ = self._post("/v1/prekeys/claim",
                               {"recipient_device_id": "d1", "claim_id": "s1"})
        self.assertEqual(status, 201)
        status, body = self._post("/v1/sessions/from-claim", {
            "claim_id": "s1", "initiator_device_id": "i1",
            "ephemeral_key": _raw_key_b64()})
        self.assertEqual(status, 201)
        self.assertEqual(set(body), _SESSION_FIELDS)


class BatchSessionCLITest(unittest.TestCase):
    def setUp(self) -> None:
        self.server, self.service = create_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        self.service.register(_register_payload("u1", "d1"))
        self.service.register(_register_payload("u1", "d2"))
        self.service.register(_register_payload("u2", "i1", ()))

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _run(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "e2ee_backend", "--base-url",
             self.base_url, *arguments],
            capture_output=True, text=True, timeout=15)

    def _claim(self, claim_id: str) -> list:
        result = self._run("claim-user-prekeys", "--user-id", "u1",
                           "--claim-id", claim_id)
        self.assertEqual(result.returncode, 0, result.stderr)
        body = json.loads(result.stdout.strip())
        return body["devices"]

    def test_success_single_line_stdout_exit_0(self) -> None:
        devices = self._claim("bc1")
        args = ["create-batch-sessions", "--claim-id", "bc1",
                "--initiator-device-id", "i1"]
        for device in devices:
            args += ["--ephemeral-key",
                     f"{device['device_id']}:{_raw_key_b64()}"]
        result = self._run(*args)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr.strip(), "")
        line = result.stdout.strip()
        self.assertEqual(line.count("\n"), 0)
        body = json.loads(line)
        self.assertEqual(body["claim_id"], "bc1")
        self.assertEqual([s["recipient_device_id"] for s in body["sessions"]],
                         ["d1", "d2"])

    def test_repeat_conflict_exit_1_field_claim_id(self) -> None:
        devices = self._claim("bc2")
        specs = [f"{d['device_id']}:{_raw_key_b64()}" for d in devices]
        args = ["create-batch-sessions", "--claim-id", "bc2",
                "--initiator-device-id", "i1"]
        for spec in specs:
            args += ["--ephemeral-key", spec]
        first = self._run(*args)
        self.assertEqual(first.returncode, 0, first.stderr)
        repeat = self._run(*args)
        self.assertEqual(repeat.returncode, 1)
        self.assertEqual(repeat.stdout.strip(), "")
        self.assertEqual(json.loads(repeat.stderr.strip())["field"],
                         "claim_id")

    def test_malformed_spec_fails_locally_exit_2(self) -> None:
        result = self._run("create-batch-sessions", "--claim-id", "bc2",
                           "--initiator-device-id", "i1",
                           "--ephemeral-key", "no-colon")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stderr.strip())["field"],
                         "ephemeral_keys")


if __name__ == "__main__":
    unittest.main()
