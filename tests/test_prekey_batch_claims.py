"""Tests for user-wide batch pre-key claims (``POST /v1/prekeys/claim-batch``).

Covers registration-order enumeration of a user's active devices, per-device
first-key selection, all-or-nothing consumption, batch claim-id idempotency,
the shared claim-id namespace with single claims, the 400/404/409/503 error
contract, shared-lock race linearization, durable persistence/restart
recovery, refusal to start on a malformed ``prekey_batch_claims`` section,
the HTTP endpoint, and the ``claim-user-prekeys`` CLI contract.
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
                      key_ids=("k1", "k2"), identity: str | None = None
                      ) -> dict:
    return {
        "user_id": user_id,
        "device_id": device_id,
        "identity_key": identity or _raw_key_b64(),
        "signed_prekeys": [{"key_id": kid, "public_key": _raw_key_b64()}
                           for kid in key_ids],
    }


_BATCH_TOP_FIELDS = {"claim_id", "user_id", "claimed_at", "devices"}
_BATCH_DEVICE_FIELDS = {"device_id", "identity_key", "key_id", "public_key"}


class BatchClaimServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = DeviceService()
        self.p1 = _register_payload("u1", "d1", ("a1", "a2"))
        self.p2 = _register_payload("u1", "d2", ("b1",))
        self.p3 = _register_payload("u2", "d3", ("c1",))
        for payload in (self.p1, self.p2, self.p3):
            self.service.register(payload)

    def _batch(self, user_id="u1", claim_id="bc1"):
        return self.service.claim_prekey_batch(
            {"user_id": user_id, "claim_id": claim_id})

    def test_first_batch_claims_one_key_per_active_device_in_order_201(self
                                                                       ) -> None:
        body, status = self._batch()
        self.assertEqual(status, 201)
        self.assertEqual(set(body), _BATCH_TOP_FIELDS)
        self.assertEqual(body["claim_id"], "bc1")
        self.assertEqual(body["user_id"], "u1")
        self.assertTrue(body["claimed_at"].endswith("+00:00"))
        self.assertEqual([d["device_id"] for d in body["devices"]],
                         ["d1", "d2"])
        d1, d2 = body["devices"]
        self.assertEqual(set(d1), _BATCH_DEVICE_FIELDS)
        self.assertEqual(d1["key_id"], "a1")
        self.assertEqual(d1["public_key"],
                         self.p1["signed_prekeys"][0]["public_key"])
        self.assertEqual(d1["identity_key"], self.p1["identity_key"])
        self.assertEqual(d2["key_id"], "b1")
        self.assertEqual(d2["identity_key"], self.p2["identity_key"])
        # Only devices of the named user are enumerated.
        self.assertEqual([d["device_id"] for d in body["devices"]],
                         ["d1", "d2"])

    def test_each_device_contributes_its_first_available_key(self) -> None:
        # Revoke d1's first key; the batch skips it and takes a2.
        self.service.revoke_prekey("d1", "a1")
        body, _ = self._batch()
        self.assertEqual([d["key_id"] for d in body["devices"]], ["a2", "b1"])

    def test_revoked_devices_are_skipped_not_enumerated(self) -> None:
        self.service.revoke_device("d1")
        body, _ = self._batch()
        self.assertEqual([d["device_id"] for d in body["devices"]], ["d2"])
        self.assertEqual(body["devices"][0]["key_id"], "b1")

    def test_repeated_batch_claim_id_is_200_and_identical(self) -> None:
        first, s1 = self._batch()
        second, s2 = self._batch()
        self.assertEqual(s1, 201)
        self.assertEqual(s2, 200)
        self.assertEqual(first, second)

    def test_replay_returns_frozen_material_after_revocation(self) -> None:
        first, _ = self._batch()
        self.service.revoke_device("d2")
        self.service.revoke_prekey("d1", "a2")
        replay, status = self._batch()
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)

    def test_one_device_without_a_key_aborts_all_consumption(self) -> None:
        body, _ = self._batch("u1", "bc1")
        self.assertEqual([d["key_id"] for d in body["devices"]], ["a1", "b1"])
        # d2 now has no key while d1 still has a2: a new batch must fail and
        # consume nothing at all.
        with self.assertRaises(ServiceError) as ctx:
            self._batch("u1", "bc2")
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "prekey_id")
        self.assertEqual(self.service.get_device("d1")["prekey_ids"], ["a2"])
        self.assertEqual(self.service.get_device("d2")["prekey_ids"], [])
        # No batch record was written for bc2.
        self.assertNotIn("bc2", self.service.store._prekey_batch_claims)

    def test_unknown_user_is_404_user_id(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self._batch("ghost", "x")
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "user_id")

    def test_all_devices_revoked_is_409_device_id(self) -> None:
        self.service.revoke_device("d3")
        with self.assertRaises(ServiceError) as ctx:
            self._batch("u2", "x")
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "device_id")

    def test_batch_and_single_claim_share_claim_id_namespace(self) -> None:
        # A single claim first; the same id as a batch is 409/claim_id.
        self.service.claim_prekey(
            {"recipient_device_id": "d3", "claim_id": "shared"})
        with self.assertRaises(ServiceError) as ctx:
            self._batch("u2", "shared")
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "claim_id")
        # A batch first; the same id as a single claim is 409/claim_id.
        self._batch("u1", "shared2")
        with self.assertRaises(ServiceError) as ctx:
            self.service.claim_prekey(
                {"recipient_device_id": "d3", "claim_id": "shared2"})
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "claim_id")

    def test_batch_consumption_excludes_keys_from_later_single_claims(self
                                                                       ) -> None:
        self._batch()
        # a1 and b1 consumed; a single claim on d1 now takes a2.
        view, status = self.service.claim_prekey(
            {"recipient_device_id": "d1", "claim_id": "s1"})
        self.assertEqual(status, 201)
        self.assertEqual(view["key_id"], "a2")
        # And d1 is exhausted.
        with self.assertRaises(ServiceError) as ctx:
            self.service.claim_prekey(
                {"recipient_device_id": "d1", "claim_id": "s2"})
        self.assertEqual(ctx.exception.field, "prekey_id")


class BatchClaimValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = DeviceService()
        self.service.register(_register_payload())

    def _assert_400(self, payload: object, field: str) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.claim_prekey_batch(payload)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.field, field)

    def test_body_must_be_object(self) -> None:
        self._assert_400(["not", "an", "object"], "request_body")

    def test_missing_fields_are_400_with_field_name(self) -> None:
        self._assert_400({}, "user_id")
        self._assert_400({"user_id": "u1"}, "claim_id")

    def test_empty_or_wrong_typed_fields_are_400(self) -> None:
        self._assert_400({"user_id": "", "claim_id": "c"}, "user_id")
        self._assert_400({"user_id": "u1", "claim_id": ""}, "claim_id")
        self._assert_400({"user_id": 7, "claim_id": "c"}, "user_id")
        self._assert_400({"user_id": "u1", "claim_id": 7}, "claim_id")
        self._assert_400({"user_id": None, "claim_id": "c"}, "user_id")

    def test_failed_validation_consumes_no_key(self) -> None:
        for bad in ({}, {"user_id": "u1"},
                    {"user_id": "u1", "claim_id": ""}):
            with self.assertRaises(ServiceError):
                self.service.claim_prekey_batch(bad)
        self.assertEqual(self.service.get_device("d1")["prekey_ids"],
                         ["k1", "k2"])


class BatchClaimConcurrencyTest(unittest.TestCase):
    def test_parallel_distinct_batches_linearize(self) -> None:
        # Two devices, each with two keys => only two full batches can win;
        # the third (and beyond) must conflict with prekey_id having consumed
        # nothing extra.
        service = DeviceService()
        service.register(_register_payload("u1", "d1", ["a1", "a2"]))
        service.register(_register_payload("u1", "d2", ["b1", "b2"]))
        results = []
        lock = threading.Lock()
        barrier = threading.Barrier(8)

        def worker(index: int) -> None:
            barrier.wait()
            try:
                body, status = service.claim_prekey_batch(
                    {"user_id": "u1", "claim_id": f"bc{index}"})
                with lock:
                    results.append(("ok", status,
                                    [d["key_id"] for d in body["devices"]]))
            except ServiceError as error:
                with lock:
                    results.append(("conflict", error.field, None))

        threads = [threading.Thread(target=worker, args=(i,))
                   for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        oks = [r for r in results if r[0] == "ok"]
        conflicts = [r for r in results if r[0] == "conflict"]
        self.assertEqual(len(oks), 2)
        self.assertEqual({r[1] for r in oks}, {201})
        self.assertEqual({tuple(r[2]) for r in oks}, {("a1", "b1"),
                                                       ("a2", "b2")})
        self.assertEqual(len(conflicts), 6)
        self.assertTrue(all(r[1] == "prekey_id" for r in conflicts))
        self.assertEqual(service.get_device("d1")["prekey_ids"], [])
        self.assertEqual(service.get_device("d2")["prekey_ids"], [])

    def test_parallel_same_batch_id_consumes_one_set(self) -> None:
        service = DeviceService()
        service.register(_register_payload("u1", "d1", ["a1", "a2"]))
        service.register(_register_payload("u1", "d2", ["b1", "b2"]))
        results = []
        lock = threading.Lock()
        barrier = threading.Barrier(8)

        def worker() -> None:
            barrier.wait()
            body, status = service.claim_prekey_batch(
                {"user_id": "u1", "claim_id": "dup"})
            with lock:
                results.append((status, tuple(d["key_id"]
                                              for d in body["devices"])))

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(results), 8)
        self.assertEqual({keys for _, keys in results}, {("a1", "b1")})
        self.assertEqual(sum(1 for status, _ in results if status == 201), 1)
        self.assertEqual(sum(1 for status, _ in results if status == 200), 7)
        self.assertEqual(service.get_device("d1")["prekey_ids"], ["a2"])
        self.assertEqual(service.get_device("d2")["prekey_ids"], ["b2"])


class BatchClaimPersistenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.path = os.path.join(self.directory, "state.json")

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def _fresh_service(self) -> DeviceService:
        service = DeviceService()
        attach_persistence(service, self.path)
        return service

    def _seed(self) -> dict:
        service = self._fresh_service()
        p1 = _register_payload("u1", "d1", ("a1", "a2"))
        p2 = _register_payload("u1", "d2", ("b1",))
        service.register(p1)
        service.register(p2)
        body, status = service.claim_prekey_batch(
            {"user_id": "u1", "claim_id": "bc1"})
        self.assertEqual(status, 201)
        return body

    def test_batch_survives_restart_and_replay_stays_idempotent(self) -> None:
        first = self._seed()
        restarted = self._fresh_service()
        # The consumed keys stay excluded after recovery.
        self.assertEqual(restarted.get_device("d1")["prekey_ids"], ["a2"])
        self.assertEqual(restarted.get_device("d2")["prekey_ids"], [])
        replay, status = restarted.claim_prekey_batch(
            {"user_id": "u1", "claim_id": "bc1"})
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)

    def test_persisted_record_shape(self) -> None:
        self._seed()
        with open(self.path, encoding="utf-8") as handle:
            document = json.load(handle)
        (record,) = document["prekey_batch_claims"]
        self.assertEqual(set(record),
                         {"claim_id", "user_id", "claimed_at", "devices"})
        self.assertEqual(record["user_id"], "u1")
        self.assertEqual([d["device_id"] for d in record["devices"]],
                         ["d1", "d2"])
        for entry in record["devices"]:
            self.assertEqual(set(entry), _BATCH_DEVICE_FIELDS)
        # The consumed flags are persisted on the pre-keys.
        by_device = {d["device_id"]: d for d in document["devices"]}
        prekeys = by_device["d1"]["prekeys"]
        self.assertTrue(prekeys[0]["consumed"])
        self.assertFalse(prekeys[1]["consumed"])
        self.assertTrue(by_device["d2"]["prekeys"][0]["consumed"])

    def _base_document(self) -> dict:
        self._seed()
        with open(self.path, encoding="utf-8") as handle:
            return json.load(handle)

    def _rewrite(self, document: dict) -> None:
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)

    def _assert_refused(self) -> None:
        with self.assertRaises(StateFileError):
            attach_persistence(DeviceService(), self.path)

    def test_missing_section_loads_as_empty(self) -> None:
        # A version-1 file written before batch claims existed has no
        # prekey_batch_claims section; it must load with an empty batch map.
        service = self._fresh_service()
        service.register(_register_payload("u1", "d1", ("a1",)))
        with open(self.path, encoding="utf-8") as handle:
            doc = json.load(handle)
        self.assertIn("prekey_batch_claims", doc)
        del doc["prekey_batch_claims"]
        self._rewrite(doc)
        restarted = self._fresh_service()
        self.assertEqual(restarted.store._prekey_batch_claims, {})
        # And a batch claim then works normally.
        body, status = restarted.claim_prekey_batch(
            {"user_id": "u1", "claim_id": "bc1"})
        self.assertEqual(status, 201)
        self.assertEqual([d["key_id"] for d in body["devices"]], ["a1"])

    def test_duplicate_batch_claim_id_is_refused(self) -> None:
        doc = self._base_document()
        doc["prekey_batch_claims"].append(
            dict(doc["prekey_batch_claims"][0]))
        self._rewrite(doc)
        self._assert_refused()

    def test_batch_claim_id_shared_with_single_claim_is_refused(self) -> None:
        service = self._fresh_service()
        service.register(_register_payload("u1", "d1", ("a1",)))
        service.claim_prekey(
            {"recipient_device_id": "d1", "claim_id": "bc1"})
        with open(self.path, encoding="utf-8") as handle:
            doc = json.load(handle)
        # Hand-craft a batch record reusing the single claim's id and key.
        doc["prekey_batch_claims"].append({
            "claim_id": "bc1", "user_id": "u1",
            "claimed_at": "2026-01-01T00:00:00+00:00",
            "devices": [{"device_id": "d1",
                         "identity_key": doc["devices"][0]["identity_key"],
                         "key_id": "a1",
                         "public_key": doc["devices"][0]["prekeys"][0][
                             "public_key"]}]})
        self._rewrite(doc)
        self._assert_refused()

    def test_device_of_other_user_is_refused(self) -> None:
        # Seed the bc1 batch (d1/d2), then add d3 under a different user and
        # hand-edit the batch record to claim d3's key, marking that key
        # consumed directly (the fabricated batch entry backs it, so the
        # check reached is batch/user ownership).
        self._seed()
        service = self._fresh_service()
        service.register(_register_payload("u9", "d3", ("z1",)))
        with open(self.path, encoding="utf-8") as handle:
            doc = json.load(handle)
        d3 = next(d for d in doc["devices"] if d["device_id"] == "d3")
        z1 = d3["prekeys"][0]
        z1["consumed"] = True
        doc["prekey_batch_claims"][0]["devices"][0] = {
            "device_id": "d3", "identity_key": d3["identity_key"],
            "key_id": z1["key_id"], "public_key": z1["public_key"]}
        self._rewrite(doc)
        self._assert_refused()

    def test_frozen_public_key_mismatch_is_refused(self) -> None:
        doc = self._base_document()
        doc["prekey_batch_claims"][0]["devices"][0]["public_key"] = _raw_key_b64()
        self._rewrite(doc)
        self._assert_refused()

    def test_entry_naming_unconsumed_key_is_refused(self) -> None:
        doc = self._base_document()
        # Point the d1 entry at a2 (unconsumed) with matching material.
        d1 = next(d for d in doc["devices"] if d["device_id"] == "d1")
        a2 = next(pk for pk in d1["prekeys"] if pk["key_id"] == "a2")
        entry = doc["prekey_batch_claims"][0]["devices"][0]
        entry["key_id"] = "a2"
        entry["public_key"] = a2["public_key"]
        self._rewrite(doc)
        self._assert_refused()

    def test_consumed_key_without_any_claim_record_is_refused(self) -> None:
        doc = self._base_document()
        doc["prekey_batch_claims"] = []
        self._rewrite(doc)
        self._assert_refused()

    def test_failed_durable_write_rolls_the_batch_back(self) -> None:
        service = self._fresh_service()
        service.register(_register_payload("u1", "d1", ("a1",)))
        service.register(_register_payload("u1", "d2", ("b1",)))
        from e2ee_backend.persistence import JsonStateStore
        original_save = JsonStateStore.save

        def fail_save(self, state):  # noqa: ANN001
            raise OSError("simulated disk failure")

        JsonStateStore.save = fail_save
        try:
            with self.assertRaises(PersistenceUnavailable):
                service.claim_prekey_batch(
                    {"user_id": "u1", "claim_id": "bc1"})
        finally:
            JsonStateStore.save = original_save
        self.assertNotIn("bc1", service.store._prekey_batch_claims)
        self.assertEqual(service.get_device("d1")["prekey_ids"], ["a1"])
        self.assertEqual(service.get_device("d2")["prekey_ids"], ["b1"])
        body, status = service.claim_prekey_batch(
            {"user_id": "u1", "claim_id": "bc1"})
        self.assertEqual(status, 201)
        self.assertEqual([d["key_id"] for d in body["devices"]], ["a1", "b1"])


class BatchClaimHTTPTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = DeviceService()
        self.service.register(_register_payload("u1", "d1", ("a1", "a2")))
        self.service.register(_register_payload("u1", "d2", ("b1",)))
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
        connection.request("POST", "/v1/prekeys/claim-batch",
                           body=json.dumps(body),
                           headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        data = response.read().decode("utf-8")
        connection.close()
        return response.status, json.loads(data)

    def test_batch_201_then_replay_200(self) -> None:
        status, body = self._request({"user_id": "u1", "claim_id": "bc1"})
        self.assertEqual(status, 201)
        self.assertEqual(set(body), _BATCH_TOP_FIELDS)
        self.assertEqual([d["device_id"] for d in body["devices"]],
                         ["d1", "d2"])
        status2, body2 = self._request({"user_id": "u1", "claim_id": "bc1"})
        self.assertEqual(status2, 200)
        self.assertEqual(body, body2)

    def test_error_status_codes(self) -> None:
        status, body = self._request({"claim_id": "c"})
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "user_id")
        status, body = self._request({"user_id": "ghost", "claim_id": "c"})
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "user_id")
        self._request({"user_id": "u1", "claim_id": "one"})
        status, body = self._request({"user_id": "u1", "claim_id": "two"})
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "prekey_id")

    def test_existing_endpoints_unchanged(self) -> None:
        # The single-claim route still answers on its own path.
        connection = HTTPConnection("127.0.0.1", self.port, timeout=5)
        connection.request("POST", "/v1/prekeys/claim",
                           body=json.dumps({"recipient_device_id": "d1",
                                            "claim_id": "s1"}),
                           headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        body = json.loads(response.read())
        connection.close()
        self.assertEqual(response.status, 201)
        self.assertEqual(body["key_id"], "a1")


class BatchClaimCLITest(unittest.TestCase):
    def setUp(self) -> None:
        self.server, _ = create_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _run(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "e2ee_backend", "--base-url",
             self.base_url, *arguments],
            capture_output=True, text=True, timeout=15)

    def _register(self, device_id: str, key_id: str) -> None:
        result = self._run(
            "register", "--user-id", "u1", "--device-id", device_id,
            "--identity-key", _raw_key_b64(),
            "--prekey", f"{key_id}:{_raw_key_b64()}")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_claim_user_prekeys_success_then_idempotent(self) -> None:
        self._register("d1", "a1")
        self._register("d2", "b1")
        result = self._run("claim-user-prekeys", "--user-id", "u1",
                           "--claim-id", "bc1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        line = result.stdout.strip()
        self.assertEqual(line.count("\n"), 0)
        body = json.loads(line)
        self.assertEqual(set(body), _BATCH_TOP_FIELDS)
        self.assertEqual([d["device_id"] for d in body["devices"]],
                         ["d1", "d2"])

        replay = self._run("claim-user-prekeys", "--user-id", "u1",
                           "--claim-id", "bc1")
        self.assertEqual(replay.returncode, 0, replay.stderr)
        self.assertEqual(replay.stdout, result.stdout)

    def test_claim_user_prekeys_failure_stderr_nonzero(self) -> None:
        result = self._run("claim-user-prekeys", "--user-id", "ghost",
                           "--claim-id", "x")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        line = result.stderr.strip()
        self.assertEqual(line.count("\n"), 0)
        self.assertEqual(json.loads(line)["field"], "user_id")


if __name__ == "__main__":
    unittest.main()
