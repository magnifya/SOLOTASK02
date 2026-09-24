"""Tests for atomic multi-device sessions from a batch claim
(``POST /v1/sessions/from-batch-claim``).

Covers the 201 body (``claim_id`` plus eight-field sessions listed in the
claim's frozen registration order, identity/pre-key material frozen from
the claim, ephemeral keys taken from the request), the 400/404/409/503
error contract with the exact field names, initiator-in-snapshot rejection,
all-or-nothing atomicity, shared-lock race linearization, durable
persistence/restart recovery (including the new bindings section, old files
lacking it, frozen-value/order recovery checks and registration-order
validation of batch claims), the HTTP route and the ``create-batch-sessions``
CLI contract.
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


def _register_payload(device_id: str, key_ids=("k1", "k2"),
                      user_id="u1") -> dict:
    return {
        "user_id": user_id,
        "device_id": device_id,
        "identity_key": _raw_key_b64(),
        "signed_prekeys": [{"key_id": kid, "public_key": _raw_key_b64()}
                           for kid in key_ids],
    }


_SESSION_FIELDS = {"session_id", "initiator_device_id",
                   "recipient_device_id", "prekey_id", "ephemeral_key",
                   "identity_key", "public_key", "created_at"}


def _entries_for(claim: dict):
    """Build ephemeral_keys entries (claim order) for a batch claim body."""
    return [{"device_id": entry["device_id"], "ephemeral_key": _raw_key_b64()}
            for entry in claim["devices"]]


class BatchSessionServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = DeviceService()
        # Register recipients interleaved with another user's device: the
        # batch device order must follow global registration order.
        self.recipients = {}
        for did in ("r1", "r2", "r3"):
            payload = _register_payload(did)
            self.recipients[did] = {
                "identity": payload["identity_key"],
                "keys": {pk["key_id"]: pk["public_key"]
                         for pk in payload["signed_prekeys"]},
            }
            self.service.register(payload)
        self.service.register(_register_payload("other", key_ids=("k1",),
                                                user_id="u2"))
        self.service.register(_register_payload("i1", key_ids=(),
                                                user_id="u9"))
        self.claim, status = self.service.claim_prekey_batch(
            {"user_id": "u1", "claim_id": "b1"})
        self.assertEqual(status, 201)
        self.order = [entry["device_id"]
                      for entry in self.claim["devices"]]

    def _create(self, claim_id="b1", initiator="i1", entries=None):
        if entries is None:
            entries = _entries_for(self.claim)
        return self.service.create_sessions_from_batch_claim({
            "claim_id": claim_id,
            "initiator_device_id": initiator,
            "ephemeral_keys": entries,
        })

    def _entry_map(self, body):
        return {session["recipient_device_id"]: session
                for session in body["sessions"]}

    def test_success_body_shape_and_frozen_order(self) -> None:
        body = self._create()
        self.assertEqual(set(body), {"claim_id", "sessions"})
        self.assertEqual(body["claim_id"], "b1")
        self.assertEqual(len(body["sessions"]), 3)
        self.assertEqual(
            [s["recipient_device_id"] for s in body["sessions"]],
            self.order)
        for session in body["sessions"]:
            self.assertEqual(set(session), _SESSION_FIELDS)
            self.assertTrue(session["session_id"])
            self.assertTrue(session["created_at"].endswith("+00:00"))
            self.assertEqual(session["initiator_device_id"], "i1")

    def test_sessions_freeze_claim_material_and_use_request_ephemerals(self):
        ephemerals = {did: _raw_key_b64() for did in self.order}
        body = self._create(entries=[{"device_id": did,
                                      "ephemeral_key": ephemerals[did]}
                                     for did in self.order])
        for did, session in self._entry_map(body).items():
            frozen = self.recipients[did]
            self.assertEqual(session["identity_key"], frozen["identity"])
            self.assertEqual(session["prekey_id"], "k1")
            self.assertEqual(session["public_key"], frozen["keys"]["k1"])
            self.assertEqual(session["ephemeral_key"], ephemerals[did])

    def test_request_device_order_is_ignored_response_uses_claim_order(self):
        entries = _entries_for(self.claim)
        body = self._create(entries=list(reversed(entries)))
        self.assertEqual(
            [s["recipient_device_id"] for s in body["sessions"]],
            self.order)

    def test_sessions_are_independently_gettable(self) -> None:
        body = self._create()
        for session in body["sessions"]:
            self.assertEqual(self.service.get_session(session["session_id"]),
                             session)

    def test_frozen_identity_survives_later_rotation(self) -> None:
        body = self._create()
        for did in self.order:
            self.service.rotate_identity_key(
                did, {"identity_key": _raw_key_b64()})
        for session in body["sessions"]:
            again = self.service.get_session(session["session_id"])
            self.assertEqual(again["identity_key"], session["identity_key"])

    def test_extra_request_device_is_400_on_item_field(self) -> None:
        entries = _entries_for(self.claim)
        entries[1] = {"device_id": "ghost", "ephemeral_key": _raw_key_b64()}
        with self.assertRaises(ServiceError) as ctx:
            self._create(entries=entries)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.field,
                         "ephemeral_keys[1].device_id")
        self.assertEqual(self.service.store._sessions, {})
        self.assertEqual(
            self.service.store._batch_claim_session_bindings, {})

    def test_missing_request_device_is_400_on_the_array(self) -> None:
        entries = _entries_for(self.claim)[:2]
        with self.assertRaises(ServiceError) as ctx:
            self._create(entries=entries)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.field, "ephemeral_keys")
        self.assertEqual(self.service.store._sessions, {})

    def test_replaced_device_points_at_the_extra_item(self) -> None:
        entries = _entries_for(self.claim)
        entries[0] = {"device_id": "ghost", "ephemeral_key": _raw_key_b64()}
        with self.assertRaises(ServiceError) as ctx:
            self._create(entries=entries)
        self.assertEqual(ctx.exception.field,
                         "ephemeral_keys[0].device_id")

    def test_initiator_in_snapshot_is_400_and_creates_nothing(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self._create(initiator=self.order[0])
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.field, "initiator_device_id")
        self.assertEqual(self.service.store._sessions, {})
        self.assertEqual(
            self.service.store._batch_claim_session_bindings, {})
        # The claim is still usable by a valid initiator.
        body = self._create(initiator="i1")
        self.assertEqual(len(body["sessions"]), 3)

    def test_unknown_batch_claim_is_404_claim_id(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self._create(claim_id="nope")
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "claim_id")

    def test_single_claim_id_is_409_claim_id(self) -> None:
        # The shared claim namespace: a single claim id is known and
        # occupied by another claim kind, so it conflicts rather than 404s.
        self.service.claim_prekey(
            {"recipient_device_id": "r1", "claim_id": "single"})
        with self.assertRaises(ServiceError) as ctx:
            self._create(claim_id="single")
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "claim_id")

    def test_duplicate_batch_claim_is_409_and_creates_nothing(self) -> None:
        first = self._create()
        with self.assertRaises(ServiceError) as ctx:
            self._create()
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "claim_id")
        # Even another initiator and ephemeral keys cannot rebind it.
        self.service.register(_register_payload("i2", key_ids=(),
                                                user_id="u9"))
        with self.assertRaises(ServiceError) as ctx:
            self._create(initiator="i2")
        self.assertEqual(ctx.exception.field, "claim_id")
        self.assertEqual(len(self.service.store._sessions), 3)
        self.assertEqual(
            list(self.service.store._batch_claim_session_bindings), ["b1"])
        self.assertEqual(
            [e.session_id for e in
             self.service.store._batch_claim_session_bindings["b1"].entries],
            [s["session_id"] for s in first["sessions"]])

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

    def test_revoked_recipient_is_409_naming_that_device(self) -> None:
        self.service.revoke_device("r2")
        with self.assertRaises(ServiceError) as ctx:
            self._create()
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "recipient_device_id")
        self.assertEqual(self.service.store._sessions, {})

    def test_revoked_claimed_prekey_is_409(self) -> None:
        self.service.revoke_prekey("r3", "k1")
        with self.assertRaises(ServiceError) as ctx:
            self._create()
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "prekey_id")
        self.assertEqual(self.service.store._sessions, {})

    def test_failure_is_all_or_nothing_and_binds_nothing(self) -> None:
        # Revoke the second device's claimed key: no session for anyone.
        self.service.revoke_prekey("r2", "k1")
        with self.assertRaises(ServiceError):
            self._create()
        self.assertEqual(len(self.service.store._sessions), 0)
        self.assertEqual(
            self.service.store._batch_claim_session_bindings, {})
        # Retrying still reaches the pre-key check — the failed attempt did
        # not occupy the claim_id with a duplicate binding.
        with self.assertRaises(ServiceError) as ctx:
            self._create()
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "prekey_id")

    def test_repeat_after_success_reports_duplicate_even_if_revoked(self):
        self._create()
        self.service.revoke_device("r1")
        with self.assertRaises(ServiceError) as ctx:
            self._create()
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "claim_id")


class BatchSessionValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = DeviceService()
        for did in ("r1", "r2"):
            self.service.register(_register_payload(did))
        self.service.register(_register_payload("i1", key_ids=(),
                                                user_id="u9"))
        self.claim, _ = self.service.claim_prekey_batch(
            {"user_id": "u1", "claim_id": "b1"})
        self.order = [d["device_id"] for d in self.claim["devices"]]

    def _assert_400(self, payload: object, field: str) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.create_sessions_from_batch_claim(payload)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.field, field)

    def _valid(self, **overrides):
        payload = {"claim_id": "b1", "initiator_device_id": "i1",
                   "ephemeral_keys": _entries_for(self.claim)}
        payload.update(overrides)
        return payload

    def test_body_must_be_object(self) -> None:
        self._assert_400(["nope"], "request_body")

    def test_missing_scalar_fields(self) -> None:
        self._assert_400({}, "claim_id")
        self._assert_400({"claim_id": "b1"}, "initiator_device_id")
        self._assert_400(
            {"claim_id": "b1", "initiator_device_id": "i1"},
            "ephemeral_keys")

    def test_scalar_field_types(self) -> None:
        self._assert_400(self._valid(claim_id=""), "claim_id")
        self._assert_400(self._valid(claim_id=7), "claim_id")
        self._assert_400(self._valid(claim_id=None), "claim_id")
        self._assert_400(
            self._valid(initiator_device_id=""), "initiator_device_id")
        self._assert_400(
            self._valid(initiator_device_id=3), "initiator_device_id")

    def test_ephemeral_keys_array_shape(self) -> None:
        self._assert_400(self._valid(ephemeral_keys=[]), "ephemeral_keys")
        self._assert_400(self._valid(ephemeral_keys="x"), "ephemeral_keys")
        self._assert_400(self._valid(ephemeral_keys=None), "ephemeral_keys")
        self._assert_400(
            self._valid(ephemeral_keys=["nope"]), "ephemeral_keys[0]")

    def test_item_field_errors(self) -> None:
        self._assert_400(
            self._valid(ephemeral_keys=[{"ephemeral_key": _raw_key_b64()}]),
            "ephemeral_keys[0].device_id")
        self._assert_400(
            self._valid(ephemeral_keys=[{"device_id": "r1"}]),
            "ephemeral_keys[0].ephemeral_key")
        self._assert_400(
            self._valid(ephemeral_keys=[
                {"device_id": "", "ephemeral_key": _raw_key_b64()}]),
            "ephemeral_keys[0].device_id")
        self._assert_400(
            self._valid(ephemeral_keys=[
                {"device_id": 9, "ephemeral_key": _raw_key_b64()}]),
            "ephemeral_keys[0].device_id")
        self._assert_400(
            self._valid(ephemeral_keys=[
                {"device_id": "r1", "ephemeral_key": "not-a-key"}]),
            "ephemeral_keys[0].ephemeral_key")
        self._assert_400(
            self._valid(ephemeral_keys=[
                {"device_id": "r1", "ephemeral_key": 5}]),
            "ephemeral_keys[0].ephemeral_key")

    def test_duplicate_device_id_in_request(self) -> None:
        entries = [
            {"device_id": "r1", "ephemeral_key": _raw_key_b64()},
            {"device_id": "r1", "ephemeral_key": _raw_key_b64()},
        ]
        self._assert_400(self._valid(ephemeral_keys=entries),
                         "ephemeral_keys[1].device_id")


class BatchSessionConcurrencyTest(unittest.TestCase):
    def test_parallel_requests_create_exactly_one_batch_set(self) -> None:
        service = DeviceService()
        for did in ("r1", "r2"):
            service.register(_register_payload(did))
        for index in range(8):
            service.register(_register_payload(
                f"i{index}", key_ids=(), user_id="u9"))
        claim, _ = service.claim_prekey_batch(
            {"user_id": "u1", "claim_id": "dup"})
        entries = _entries_for(claim)

        results = []
        lock = threading.Lock()
        barrier = threading.Barrier(8)

        def worker(index: int) -> None:
            barrier.wait()
            try:
                body = service.create_sessions_from_batch_claim({
                    "claim_id": "dup",
                    "initiator_device_id": f"i{index}",
                    "ephemeral_keys": entries,
                })
                with lock:
                    results.append(("ok", body))
            except ServiceError as error:
                with lock:
                    results.append(("conflict", error.field))

        threads = [threading.Thread(target=worker, args=(i,))
                   for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        successes = [body for outcome, body in results if outcome == "ok"]
        conflicts = [field for outcome, field in results
                     if outcome == "conflict"]
        self.assertEqual(len(successes), 1)
        self.assertEqual(len(conflicts), 7)
        self.assertTrue(all(field == "claim_id" for field in conflicts))
        self.assertEqual(len(service.store._sessions), 2)
        self.assertEqual(
            list(service.store._batch_claim_session_bindings), ["dup"])


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
        for did in ("r1", "r2"):
            service.register(_register_payload(did))
        service.register(_register_payload("i1", key_ids=(), user_id="u9"))
        claim, _ = service.claim_prekey_batch(
            {"user_id": "u1", "claim_id": "b1"})
        body = service.create_sessions_from_batch_claim({
            "claim_id": "b1", "initiator_device_id": "i1",
            "ephemeral_keys": _entries_for(claim)})
        return service, claim, body

    def test_binding_survives_restart_and_blocks_repeat(self) -> None:
        _service, claim, body = self._setup_bound_batch()
        restarted = self._fresh_service()
        self.assertEqual(
            set(restarted.store._batch_claim_session_bindings), {"b1"})
        with self.assertRaises(ServiceError) as ctx:
            restarted.create_sessions_from_batch_claim({
                "claim_id": "b1", "initiator_device_id": "i1",
                "ephemeral_keys": _entries_for(claim)})
        self.assertEqual(ctx.exception.field, "claim_id")
        for session in body["sessions"]:
            self.assertEqual(restarted.get_session(session["session_id"]),
                             session)

    def test_persisted_record_has_contract_fields(self) -> None:
        self._setup_bound_batch()
        with open(self.path, encoding="utf-8") as handle:
            document = json.load(handle)
        self.assertEqual(document["version"], 1)
        (record,) = document["batch_claim_session_bindings"]
        self.assertEqual(
            set(record), {"claim_id", "initiator_device_id",
                          "created_at", "entries"})
        self.assertEqual(record["claim_id"], "b1")
        self.assertEqual(record["initiator_device_id"], "i1")
        self.assertEqual(len(record["entries"]), 2)
        self.assertEqual(
            set(record["entries"][0]),
            {"recipient_device_id", "prekey_id", "identity_key",
             "public_key", "session_id"})
        self.assertEqual(
            [entry["recipient_device_id"] for entry in record["entries"]],
            ["r1", "r2"])

    def test_old_file_without_section_loads_empty(self) -> None:
        self._setup_bound_batch()
        with open(self.path, encoding="utf-8") as handle:
            document = json.load(handle)
        del document["batch_claim_session_bindings"]
        # A legacy file also predates the integrity-log marker/sidecar.
        del document["integrity_log_version"]
        old_path = os.path.join(self.directory, "old.json")
        with open(old_path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        service = DeviceService()
        attach_persistence(service, old_path)
        self.assertEqual(service.store._batch_claim_session_bindings, {})
        # The old batch claim may now establish its sessions once.
        batch = service.store.get_batch_claim("b1")
        body = service.create_sessions_from_batch_claim({
            "claim_id": "b1", "initiator_device_id": "i1",
            "ephemeral_keys": _entries_for(
                {"devices": [
                    {"device_id": e.device_id} for e in batch.devices]})})
        self.assertEqual(len(body["sessions"]), 2)

    def test_restore_after_identity_rotation_uses_frozen_keys(self) -> None:
        _service, claim, body = self._setup_bound_batch()
        # Rotate a recipient after the sessions exist; the frozen snapshot
        # and binding must still restore.
        _service.rotate_identity_key("r1", {"identity_key": _raw_key_b64()})
        restarted = self._fresh_service()
        self.assertEqual(
            set(restarted.store._batch_claim_session_bindings), {"b1"})
        for session in body["sessions"]:
            self.assertEqual(restarted.get_session(session["session_id"]),
                             session)

    def _mutated_document(self, mutate) -> str:
        with open(self.path, encoding="utf-8") as handle:
            document = json.load(handle)
        mutate(document)
        path = os.path.join(self.directory, f"mut-{id(mutate)}.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        return path

    def _assert_refused(self, path: str) -> None:
        with self.assertRaises(StateFileError):
            attach_persistence(DeviceService(), path)

    def test_malformed_batch_binding_sections_are_refused(self) -> None:
        self._setup_bound_batch()

        def wrong_type(doc):
            doc["batch_claim_session_bindings"] = "nope"

        def unknown_claim(doc):
            doc["batch_claim_session_bindings"][0]["claim_id"] = "ghost"

        def unknown_session(doc):
            doc["batch_claim_session_bindings"][0]["entries"][0][
                "session_id"] = "deadbeef"

        def duplicate_binding(doc):
            record = dict(doc["batch_claim_session_bindings"][0])
            doc["batch_claim_session_bindings"].append(record)

        def swapped_order(doc):
            entries = doc["batch_claim_session_bindings"][0]["entries"]
            entries[0]["recipient_device_id"], entries[1][
                "recipient_device_id"] = (entries[1]["recipient_device_id"],
                                          entries[0]["recipient_device_id"])

        def missing_entry(doc):
            del doc["batch_claim_session_bindings"][0]["entries"][1]

        def wrong_material(doc):
            doc["batch_claim_session_bindings"][0]["entries"][0][
                "public_key"] = "other-frozen-key"

        def session_reused_by_single_binding(doc):
            session_id = doc["batch_claim_session_bindings"][0][
                "entries"][0]["session_id"]
            device = doc["devices"][0]
            doc["prekey_claims"].append({
                "claim_id": "c-single",
                "device_id": "r1",
                "key_id": "k2",
                "identity_key": device["identity_key"],
                "public_key": next(
                    pk for pk in device["prekeys"] if pk["key_id"] == "k2")[
                    "public_key"],
                "claimed_at": "2026-01-01T00:00:00+00:00"})
            for pk in device["prekeys"]:
                if pk["key_id"] == "k2":
                    pk["consumed"] = True
            doc["claim_session_bindings"].append({
                "claim_id": "c-single",
                "session_id": session_id,
                "recipient_device_id": "r1",
                "prekey_id": "k2",
                "identity_key": device["identity_key"],
                "public_key": next(
                    pk for pk in device["prekeys"] if pk["key_id"] == "k2")[
                    "public_key"],
                "created_at": "2026-01-01T00:00:00+00:00"})

        def initiator_unknown(doc):
            doc["batch_claim_session_bindings"][0][
                "initiator_device_id"] = "ghost"

        for mutate in (wrong_type, unknown_claim, unknown_session,
                       duplicate_binding, swapped_order, missing_entry,
                       wrong_material, session_reused_by_single_binding,
                       initiator_unknown):
            self._assert_refused(self._mutated_document(mutate))

    def test_batch_claim_wrong_registration_order_is_refused(self) -> None:
        self._setup_bound_batch()

        def scramble(doc):
            devices = doc["prekey_batch_claims"][0]["devices"]
            devices[0], devices[1] = devices[1], devices[0]

        self._assert_refused(self._mutated_document(scramble))

    def test_failed_durable_write_rolls_everything_back(self) -> None:
        service = self._fresh_service()
        for did in ("r1", "r2"):
            service.register(_register_payload(did))
        service.register(_register_payload("i1", key_ids=(), user_id="u9"))
        claim, _ = service.claim_prekey_batch(
            {"user_id": "u1", "claim_id": "b1"})
        original_save = JsonStateStore.save

        def fail_save(self, state):  # noqa: ANN001
            raise OSError("simulated disk failure")

        JsonStateStore.save = fail_save
        try:
            with self.assertRaises(PersistenceUnavailable):
                service.create_sessions_from_batch_claim({
                    "claim_id": "b1", "initiator_device_id": "i1",
                    "ephemeral_keys": _entries_for(claim)})
        finally:
            JsonStateStore.save = original_save
        # Nothing survived in memory or occupies the batch claim.
        self.assertEqual(service.store._batch_claim_session_bindings, {})
        self.assertEqual(len(service.store._sessions), 0)
        body = service.create_sessions_from_batch_claim({
            "claim_id": "b1", "initiator_device_id": "i1",
            "ephemeral_keys": _entries_for(claim)})
        self.assertEqual(len(body["sessions"]), 2)
        # The state file advanced from the previous committed state only.
        with open(self.path, encoding="utf-8") as handle:
            document = json.load(handle)
        self.assertEqual(len(document["sessions"]), 2)
        self.assertEqual(
            len(document["batch_claim_session_bindings"]), 1)


class BatchSessionHTTPTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = DeviceService()
        for did in ("r1", "r2"):
            self.service.register(_register_payload(did))
        self.service.register(_register_payload("i1", key_ids=(),
                                                user_id="u9"))
        self.server, _ = create_server("127.0.0.1", 0,
                                       service=self.service)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        self.claim = self._batch_claim()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _post(self, path: str, body: object):
        connection = HTTPConnection("127.0.0.1", self.port, timeout=5)
        connection.request("POST", path, body=json.dumps(body),
                           headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        data = json.loads(response.read().decode("utf-8"))
        connection.close()
        return response.status, data

    def _batch_claim(self, claim_id="b1", user_id="u1"):
        status, body = self._post("/v1/prekeys/claim-batch",
                                  {"user_id": user_id, "claim_id": claim_id})
        assert status == 201
        return body

    def _payload(self, **overrides):
        payload = {"claim_id": "b1", "initiator_device_id": "i1",
                   "ephemeral_keys": _entries_for(self.claim)}
        payload.update(overrides)
        return payload

    def test_create_201_then_repeat_409(self) -> None:
        status, body = self._post("/v1/sessions/from-batch-claim",
                                  self._payload())
        self.assertEqual(status, 201)
        self.assertEqual(body["claim_id"], "b1")
        self.assertEqual(len(body["sessions"]), 2)
        self.assertEqual(
            [s["recipient_device_id"] for s in body["sessions"]],
            [entry["device_id"] for entry in self.claim["devices"]])
        status, body = self._post("/v1/sessions/from-batch-claim",
                                  self._payload())
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "claim_id")

    def test_validation_status_codes_and_fields(self) -> None:
        cases = [
            ({}, 400, "claim_id"),
            ({"claim_id": "b1"}, 400, "initiator_device_id"),
            ({"claim_id": "b1", "initiator_device_id": "i1"},
             400, "ephemeral_keys"),
            (self._valid_with_entries([]), 400, "ephemeral_keys"),
            (self._payload_with_entry(
                {"device_id": "r1"}), 400,
             "ephemeral_keys[0].ephemeral_key"),
            (self._payload(claim_id="nope"), 404, "claim_id"),
            (self._payload(initiator_device_id="ghost"),
             404, "initiator_device_id"),
            (self._payload(initiator_device_id="r1"),
             400, "initiator_device_id"),
        ]
        for payload, expected_status, expected_field in cases:
            status, body = self._post("/v1/sessions/from-batch-claim",
                                      payload)
            self.assertEqual((status, body.get("field")),
                             (expected_status, expected_field),
                             payload)

    def _valid_with_entries(self, entries):
        return {"claim_id": "b1", "initiator_device_id": "i1",
                "ephemeral_keys": entries}

    def _payload_with_entry(self, entry):
        return self._valid_with_entries([entry])

    def test_revoked_recipient_and_prekey_fields(self) -> None:
        self.service.revoke_device("r2")
        status, body = self._post("/v1/sessions/from-batch-claim",
                                  self._payload())
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "recipient_device_id")

    def test_disk_failure_is_503_data_file(self) -> None:
        directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
        path = os.path.join(directory, "state.json")
        attach_persistence(self.service, path)
        original_save = JsonStateStore.save

        def fail_save(self, state):  # noqa: ANN001
            raise OSError("simulated disk failure")

        JsonStateStore.save = fail_save
        try:
            status, body = self._post("/v1/sessions/from-batch-claim",
                                      self._payload())
        finally:
            JsonStateStore.save = original_save
        self.assertEqual(status, 503)
        self.assertEqual(body["field"], "data_file")
        self.assertEqual(len(self.service.store._sessions), 0)


class BatchSessionCLITest(unittest.TestCase):
    def setUp(self) -> None:
        self.server, self.service = create_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        for did in ("r1", "r2"):
            self._cli("register", "--user-id", "u1", "--device-id", did,
                      "--identity-key", _raw_key_b64(),
                      "--prekey", f"k1:{_raw_key_b64()}",
                      "--prekey", f"k2:{_raw_key_b64()}")
        self._cli("register", "--user-id", "u9", "--device-id", "i1",
                  "--identity-key", _raw_key_b64())
        result = self._cli("claim-user-prekeys", "--user-id", "u1",
                           "--claim-id", "b1")
        self.claim = json.loads(result.stdout)

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _cli(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "e2ee_backend", "--base-url",
             self.base_url, *arguments],
            capture_output=True, text=True, timeout=15)

    def test_success_single_line_stdout_exit_zero(self) -> None:
        arguments = ["create-batch-sessions", "--claim-id", "b1",
                     "--initiator-device-id", "i1"]
        for entry in self.claim["devices"]:
            arguments += ["--ephemeral-key",
                          f"{entry['device_id']}:{_raw_key_b64()}"]
        result = self._cli(*arguments)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        line = result.stdout.strip()
        self.assertEqual(line.count("\n"), 0)
        body = json.loads(line)
        self.assertEqual(body["claim_id"], "b1")
        self.assertEqual(len(body["sessions"]), 2)
        for session in body["sessions"]:
            self.assertEqual(set(session), _SESSION_FIELDS)

    def test_failure_single_line_stderr_nonzero(self) -> None:
        result = self._cli("create-batch-sessions", "--claim-id", "missing",
                           "--initiator-device-id", "i1",
                           "--ephemeral-key", f"r1:{_raw_key_b64()}")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        line = result.stderr.strip()
        self.assertEqual(line.count("\n"), 0)
        body = json.loads(line)
        self.assertEqual(body["field"], "claim_id")

    def test_connection_failure_is_field_server(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "e2ee_backend",
             "--base-url", "http://127.0.0.1:1",
             "create-batch-sessions", "--claim-id", "b1",
             "--initiator-device-id", "i1",
             "--ephemeral-key", f"r1:{_raw_key_b64()}"],
            capture_output=True, text=True, timeout=15)
        self.assertNotEqual(result.returncode, 0)
        body = json.loads(result.stderr.strip())
        self.assertEqual(body["field"], "server")


if __name__ == "__main__":
    unittest.main()
