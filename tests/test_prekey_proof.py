"""Tests for the published signed pre-key proof query.

Covers the service, HTTP (real loopback socket) and persistence layers for
``GET /v1/devices/{device_id}/prekeys/{key_id}/proof``.

Only pre-keys first created through a verified entry (verified registration
or the first-successful verified replenishment) keep a proof. The proof
freezes the six published strings, including the identity public key used
at verification time; ordinary publications, legacy data and idempotent
replays have/keep no proof. The query is a pure read: it does not consume
keys, append audit events, move cursors or advance commit generations.
"""
import base64
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from http.client import HTTPConnection

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, x25519

from e2ee_backend.crypto import signed_prekey_proof_message
from e2ee_backend.http_app import create_server
from e2ee_backend.persistence import StateFileError, attach_persistence
from e2ee_backend.service import DeviceService, ServiceError


def _raw_b64(key) -> str:
    raw = key.public_bytes(serialization.Encoding.Raw,
                           serialization.PublicFormat.Raw)
    return base64.b64encode(raw).decode()


def _new_identity():
    private = ed25519.Ed25519PrivateKey.generate()
    return private, _raw_b64(private.public_key())


def _new_prekey() -> str:
    return _raw_b64(x25519.X25519PrivateKey.generate().public_key())


def _proof(private, user_id, device_id, key_id, public_key) -> str:
    message = signed_prekey_proof_message(
        user_id, device_id, key_id, public_key)
    return base64.b64encode(private.sign(message)).decode()


class PrekeyProofServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = DeviceService()
        self.private, self.identity = _new_identity()
        self.prekey_k1 = _new_prekey()
        self.sig_k1 = _proof(self.private, "u1", "d1", "k1",
                             self.prekey_k1)
        self.service.register_verified({
            "user_id": "u1", "device_id": "d1",
            "identity_key": self.identity,
            "signed_prekeys": [{
                "key_id": "k1", "public_key": self.prekey_k1,
                "signature": self.sig_k1}],
        })

    def test_verified_registration_proof_returns_six_strings(self) -> None:
        proof = self.service.get_prekey_proof("d1", "k1")
        self.assertEqual(
            set(proof),
            {"user_id", "device_id", "key_id", "public_key",
             "identity_key", "signature"})
        for value in proof.values():
            self.assertIsInstance(value, str)
            self.assertTrue(value)
        self.assertEqual(proof["user_id"], "u1")
        self.assertEqual(proof["device_id"], "d1")
        self.assertEqual(proof["key_id"], "k1")
        self.assertEqual(proof["public_key"], self.prekey_k1)
        self.assertEqual(proof["identity_key"], self.identity)
        self.assertEqual(proof["signature"], self.sig_k1)

    def test_proof_is_independently_verifiable(self) -> None:
        proof = self.service.get_prekey_proof("d1", "k1")
        identity = ed25519.Ed25519PublicKey.from_public_bytes(
            base64.b64decode(proof["identity_key"]))
        signature = base64.b64decode(proof["signature"])
        identity.verify(signature, signed_prekey_proof_message(
            proof["user_id"], proof["device_id"], proof["key_id"],
            proof["public_key"]))

    def test_ordinary_registration_has_no_proof(self) -> None:
        public_key = _new_prekey()
        self.service.register({
            "user_id": "u2", "device_id": "d2",
            "identity_key": _new_prekey(),
            "signed_prekeys": [{"key_id": "k1",
                                "public_key": public_key}]})
        with self.assertRaises(ServiceError) as ctx:
            self.service.get_prekey_proof("d2", "k1")
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "signature")

    def test_ordinary_add_prekey_has_no_proof(self) -> None:
        public_key = _new_prekey()
        self.service.add_prekey(
            "d1", {"key_id": "plain", "public_key": public_key})
        with self.assertRaises(ServiceError) as ctx:
            self.service.get_prekey_proof("d1", "plain")
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "signature")

    def test_verified_add_first_creation_keeps_proof(self) -> None:
        public_key = _new_prekey()
        signature = _proof(self.private, "u1", "d1", "k2", public_key)
        body, status = self.service.add_prekey_verified(
            "d1", {"key_id": "k2", "public_key": public_key,
                   "signature": signature})
        self.assertEqual(status, 201)
        proof = self.service.get_prekey_proof("d1", "k2")
        self.assertEqual(proof["signature"], signature)
        self.assertEqual(proof["identity_key"], self.identity)
        self.assertEqual(proof["public_key"], public_key)

    def test_unknown_device_404_device_id(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.get_prekey_proof("ghost", "k1")
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "device_id")

    def test_unknown_key_404_key_id(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.get_prekey_proof("d1", "ghost-key")
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "key_id")

    def test_lookup_order_device_before_key(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.get_prekey_proof("ghost", "ghost-key")
        self.assertEqual(ctx.exception.field, "device_id")

    def test_revoked_device_still_serves_proof(self) -> None:
        self.service.revoke_device("d1")
        proof = self.service.get_prekey_proof("d1", "k1")
        self.assertEqual(proof["signature"], self.sig_k1)

    def test_consumed_prekey_still_serves_proof(self) -> None:
        body, status = self.service.claim_prekey({
            "recipient_device_id": "d1", "claim_id": "c1"})
        self.assertEqual(status, 201)
        self.assertEqual(body["key_id"], "k1")
        proof = self.service.get_prekey_proof("d1", "k1")
        self.assertEqual(proof["signature"], self.sig_k1)
        # The query itself did not consume another key.
        self.assertEqual(self.service.get_device("d1")["prekey_ids"], [])

    def test_revoked_prekey_still_serves_proof(self) -> None:
        self.service.revoke_prekey("d1", "k1")
        proof = self.service.get_prekey_proof("d1", "k1")
        self.assertEqual(proof["signature"], self.sig_k1)

    def test_rotation_keeps_frozen_identity(self) -> None:
        new_private, new_identity = _new_identity()
        self.service.rotate_identity_key(
            "d1", {"identity_key": new_identity})
        proof = self.service.get_prekey_proof("d1", "k1")
        self.assertEqual(proof["identity_key"], self.identity)
        # A key added after rotation freezes the new identity.
        public_key = _new_prekey()
        signature = _proof(new_private, "u1", "d1", "k2", public_key)
        self.service.add_prekey_verified(
            "d1", {"key_id": "k2", "public_key": public_key,
                   "signature": signature})
        after = self.service.get_prekey_proof("d1", "k2")
        self.assertEqual(after["identity_key"], new_identity)
        self.assertEqual(self.service.get_prekey_proof("d1", "k1")[
            "identity_key"], self.identity)

    def test_idempotent_replay_does_not_backfill_or_replace(self) -> None:
        # An ordinary key keeps no proof even when replayed through the
        # verified entry (the replay only matches the same key value).
        public_key = _new_prekey()
        self.service.add_prekey(
            "d1", {"key_id": "plain", "public_key": public_key})
        signature = _proof(self.private, "u1", "d1", "plain", public_key)
        _, status = self.service.add_prekey_verified(
            "d1", {"key_id": "plain", "public_key": public_key,
                   "signature": signature})
        self.assertEqual(status, 200)
        with self.assertRaises(ServiceError) as ctx:
            self.service.get_prekey_proof("d1", "plain")
        self.assertEqual(ctx.exception.field, "signature")

    def test_replay_keeps_first_proof_and_generation(self) -> None:
        path = tempfile.mktemp(suffix=".json")
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        service = DeviceService()
        store = attach_persistence(service, path)
        service.register_verified({
            "user_id": "u1", "device_id": "d1",
            "identity_key": self.identity,
            "signed_prekeys": [{
                "key_id": "k1", "public_key": self.prekey_k1,
                "signature": self.sig_k1}]})
        public_key = _new_prekey()
        signature = _proof(self.private, "u1", "d1", "k2", public_key)
        service.add_prekey_verified(
            "d1", {"key_id": "k2", "public_key": public_key,
                   "signature": signature})
        generation = store.commit_seq
        events_before = len(
            service.list_key_events("d1", 0, 100)["events"])

        # Replay with the same key and a valid proof: 200, no generation.
        _, status = service.add_prekey_verified(
            "d1", {"key_id": "k2", "public_key": public_key,
                   "signature": signature})
        self.assertEqual(status, 200)
        self.assertEqual(store.commit_seq, generation)
        self.assertEqual(len(service.list_key_events("d1", 0, 100)[
            "events"]), events_before)
        proof = service.get_prekey_proof("d1", "k2")
        self.assertEqual(proof["signature"], signature)
        self.assertEqual(proof["identity_key"], self.identity)

    def test_query_changes_nothing(self) -> None:
        path = tempfile.mktemp(suffix=".json")
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        service = DeviceService()
        store = attach_persistence(service, path)
        service.register_verified({
            "user_id": "u1", "device_id": "d1",
            "identity_key": self.identity,
            "signed_prekeys": [{
                "key_id": "k1", "public_key": self.prekey_k1,
                "signature": self.sig_k1}]})
        generation = store.commit_seq
        for _ in range(3):
            service.get_prekey_proof("d1", "k1")
        self.assertEqual(store.commit_seq, generation)
        events = service.list_key_events("d1", 0, 100)["events"]
        self.assertEqual([e["type"] for e in events], ["registered"])
        self.assertEqual(service.get_device("d1")["prekey_ids"], ["k1"])


class PrekeyProofHTTPTest(unittest.TestCase):
    def setUp(self) -> None:
        self.server, self.service = create_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        self.private, self.identity = _new_identity()
        self.public_key = _new_prekey()
        self.signature = _proof(self.private, "u1", "d1", "k1",
                                self.public_key)
        self.service.register_verified({
            "user_id": "u1", "device_id": "d1",
            "identity_key": self.identity,
            "signed_prekeys": [{
                "key_id": "k1", "public_key": self.public_key,
                "signature": self.signature}]})
        self.service.add_prekey(
            "d1", {"key_id": "plain", "public_key": _new_prekey()})

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _get(self, path, body=None, raw_body=None):
        connection = HTTPConnection("127.0.0.1", self.port, timeout=5)
        if raw_body is not None:
            connection.request("GET", path, body=raw_body, headers={
                "Content-Type": "application/json"})
        else:
            connection.request(
                "GET", path,
                body=json.dumps(body) if body is not None else None)
        response = connection.getresponse()
        data = response.read().decode("utf-8")
        connection.close()
        return response.status, json.loads(data) if data else None

    def test_proof_200(self) -> None:
        status, body = self._get("/v1/devices/d1/prekeys/k1/proof")
        self.assertEqual(status, 200)
        self.assertEqual(body, {
            "user_id": "u1", "device_id": "d1", "key_id": "k1",
            "public_key": self.public_key, "identity_key": self.identity,
            "signature": self.signature})

    def test_query_and_body_ignored(self) -> None:
        status, body = self._get(
            "/v1/devices/d1/prekeys/k1/proof?anything=%2F&x=1", body={"x": 1})
        self.assertEqual(status, 200)
        status, _ = self._get(
            "/v1/devices/d1/prekeys/k1/proof", raw_body="not even json")
        self.assertEqual(status, 200)

    def test_no_proof_409_signature(self) -> None:
        status, body = self._get("/v1/devices/d1/prekeys/plain/proof")
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "signature")

    def test_unknown_device_and_key(self) -> None:
        status, body = self._get("/v1/devices/ghost/prekeys/k1/proof")
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "device_id")
        status, body = self._get("/v1/devices/d1/prekeys/ghost/proof")
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "key_id")

    def test_encoded_slash_is_identifier_content(self) -> None:
        public_key = _new_prekey()
        signature = _proof(self.private, "u1", "a/b", "c/d", public_key)
        self.service.register_verified({
            "user_id": "u1", "device_id": "a/b",
            "identity_key": self.identity,
            "signed_prekeys": [{
                "key_id": "c/d", "public_key": public_key,
                "signature": signature}]})
        status, body = self._get(
            "/v1/devices/a%2Fb/prekeys/c%2Fd/proof")
        self.assertEqual(status, 200)
        self.assertEqual(body["device_id"], "a/b")
        self.assertEqual(body["key_id"], "c/d")

    def test_empty_segments_400_naming_identifier(self) -> None:
        status, body = self._get("/v1/devices//prekeys/k1/proof")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "device_id")
        status, body = self._get("/v1/devices/d1/prekeys//proof")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "key_id")

    def test_bad_percent_encoding_400(self) -> None:
        for path, field in (
                ("/v1/devices/d%ZZ/prekeys/k1/proof", "device_id"),
                ("/v1/devices/d1/prekeys/k%ZZ/proof", "key_id"),
                ("/v1/devices/%ff/prekeys/k1/proof", "device_id")):
            with self.subTest(path=path):
                status, body = self._get(path)
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], field)

    def test_raw_slash_in_key_segment_404_key_id(self) -> None:
        status, body = self._get("/v1/devices/d1/prekeys/a/b/proof")
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "key_id")

    def test_proof_path_does_not_serve_post(self) -> None:
        # POST .../prekeys/<segment>/proof is not the pre-key route shape
        # (that is exactly .../prekeys); it is an add-pre-key request for a
        # device id "d1/prekeys/k1/proof"-shaped path and misses the device
        # shape, answering 404/device_id rather than exposing the proof.
        connection = HTTPConnection("127.0.0.1", self.port, timeout=5)
        connection.request("POST", "/v1/devices/d1/prekeys/k1/proof",
                           body="{}",
                           headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        data = json.loads(response.read().decode())
        connection.close()
        self.assertEqual(response.status, 404)
        self.assertEqual(data["field"], "device_id")


class PrekeyProofPersistenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.path = os.path.join(self.directory, "state.json")
        self.private, self.identity = _new_identity()
        self.prekey_k1 = _new_prekey()
        self.sig_k1 = _proof(self.private, "u1", "d1", "k1",
                             self.prekey_k1)

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self.directory, ignore_errors=True)

    def _register(self, service: DeviceService) -> None:
        service.register_verified({
            "user_id": "u1", "device_id": "d1",
            "identity_key": self.identity,
            "signed_prekeys": [{
                "key_id": "k1", "public_key": self.prekey_k1,
                "signature": self.sig_k1}]})

    def _document(self):
        with open(self.path, encoding="utf-8") as handle:
            return json.load(handle)

    def _overwrite(self, document) -> None:
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(document, handle, ensure_ascii=False)

    def test_proof_persists_and_reloads_unchanged(self) -> None:
        first = DeviceService()
        attach_persistence(first, self.path)
        self._register(first)
        added = _new_prekey()
        added_sig = _proof(self.private, "u1", "d1", "k2", added)
        first.add_prekey_verified("d1", {
            "key_id": "k2", "public_key": added,
            "signature": added_sig})
        first.add_prekey("d1", {"key_id": "plain",
                                "public_key": _new_prekey()})

        document = self._document()
        prekeys = {pk["key_id"]: pk
                   for pk in document["devices"][0]["prekeys"]}
        self.assertEqual(prekeys["k1"]["signature"], self.sig_k1)
        self.assertEqual(prekeys["k1"]["identity_key"], self.identity)
        self.assertEqual(prekeys["k2"]["signature"], added_sig)
        # Proof-less keys keep the legacy four-field shape.
        self.assertEqual(
            set(prekeys["plain"]),
            {"key_id", "public_key", "revoked", "consumed"})

        second = DeviceService()
        attach_persistence(second, self.path)
        proof = second.get_prekey_proof("d1", "k1")
        self.assertEqual(proof["signature"], self.sig_k1)
        self.assertEqual(proof["identity_key"], self.identity)
        self.assertEqual(second.get_prekey_proof("d1", "k2")[
            "signature"], added_sig)
        with self.assertRaises(ServiceError) as ctx:
            second.get_prekey_proof("d1", "plain")
        self.assertEqual(ctx.exception.field, "signature")

    def test_legacy_file_without_proof_fields_loads(self) -> None:
        # Hand-write a legacy version=1 document (no proof fields at all):
        # it must load, the keys answer 409/signature, and the first write
        # migrates without touching key shapes beyond its own changes.
        legacy = {
            "version": 1,
            "devices": [{
                "user_id": "u1", "device_id": "d1",
                "identity_key": _new_prekey(),
                "registered_at": "2024-01-01T00:00:00+00:00",
                "rotated_at": "2024-01-01T00:00:00+00:00",
                "revoked": False, "identity_key_version": 1,
                "prekeys": [{
                    "key_id": "k1", "public_key": _new_prekey(),
                    "revoked": False, "consumed": False}]}],
        }
        self._overwrite(legacy)
        service = DeviceService()
        attach_persistence(service, self.path)
        with self.assertRaises(ServiceError) as ctx:
            service.get_prekey_proof("d1", "k1")
        self.assertEqual(ctx.exception.field, "signature")

    def test_integrity_probe_covers_proofs(self) -> None:
        service = DeviceService()
        store = attach_persistence(service, self.path)
        self._register(service)
        report = service.persistence_integrity()
        self.assertIs(report["consistent"], True)
        self.assertEqual(report["commit_seq"], store.commit_seq - 1)
        # Tampering with the on-disk proof moves the hash away from the
        # committed sidecar tail; the live store is untouched and the probe
        # reports inconsistency (503 at the HTTP boundary).
        document = self._document()
        prekeys = document["devices"][0]["prekeys"][0]
        prekeys["signature"] = prekeys["signature"][:-2] + "AA"
        self._overwrite(document)
        with self.assertRaises(ServiceError) as ctx:
            service.persistence_integrity()
        self.assertEqual(ctx.exception.status_code, 503)
        self.assertEqual(ctx.exception.field, "data_file")
        # The failed probe changed neither the live proof nor generation.
        self.assertEqual(service.get_prekey_proof("d1", "k1")[
            "signature"], self.sig_k1)

    def _assert_startup_refuses(self) -> None:
        with self.assertRaises(StateFileError):
            attach_persistence(DeviceService(), self.path)

    def test_missing_identity_key_field_refuses(self) -> None:
        first = DeviceService()
        attach_persistence(first, self.path)
        self._register(first)
        document = self._document()
        del document["devices"][0]["prekeys"][0]["identity_key"]
        self._overwrite(document)
        self._assert_startup_refuses()

    def test_wrong_typed_fields_refuse(self) -> None:
        first = DeviceService()
        attach_persistence(first, self.path)
        self._register(first)
        for mutate in (
                lambda d: d["devices"][0]["prekeys"][0].update(
                    signature=123),
                lambda d: d["devices"][0]["prekeys"][0].update(
                    signature=""),
                lambda d: d["devices"][0]["prekeys"][0].update(
                    identity_key=42),
                lambda d: d["devices"][0]["prekeys"][0].update(
                    identity_key=None),
                lambda d: d["devices"][0]["prekeys"][0].__setitem__(
                    "extra", "x")):
            document = self._document()
            mutate(document)
            self._overwrite(document)
            with self.subTest():
                # An extra field changes the canonical hash; the structural
                # edits trip the restore validation. Either way startup
                # refuses.
                self._assert_startup_refuses()

    def test_bad_signature_encoding_refuses(self) -> None:
        first = DeviceService()
        attach_persistence(first, self.path)
        self._register(first)
        for bad in ("not-base64", "abc",
                    base64.b64encode(b"\x00" * 63).decode(),
                    base64.b64encode(b"\x00" * 65).decode()):
            document = self._document()
            document["devices"][0]["prekeys"][0]["signature"] = bad
            self._overwrite(document)
            with self.subTest(bad=bad):
                self._assert_startup_refuses()

    def test_non_ed25519_identity_refuses(self) -> None:
        first = DeviceService()
        attach_persistence(first, self.path)
        self._register(first)
        document = self._document()
        # X25519 raw point: 32 bytes but not Ed25519.
        document["devices"][0]["prekeys"][0]["identity_key"] = _new_prekey()
        self._overwrite(document)
        self._assert_startup_refuses()

    def test_wrong_public_key_binding_refuses(self) -> None:
        first = DeviceService()
        attach_persistence(first, self.path)
        self._register(first)
        document = self._document()
        document["devices"][0]["prekeys"][0]["public_key"] = _new_prekey()
        self._overwrite(document)
        self._assert_startup_refuses()

    def test_wrong_owner_binding_refuses(self) -> None:
        first = DeviceService()
        attach_persistence(first, self.path)
        self._register(first)
        document = self._document()
        document["devices"][0]["device_id"] = "other-device"
        self._overwrite(document)
        self._assert_startup_refuses()

    def test_foreign_signature_refuses(self) -> None:
        first = DeviceService()
        attach_persistence(first, self.path)
        self._register(first)
        other, _ = _new_identity()
        document = self._document()
        document["devices"][0]["prekeys"][0]["signature"] = _proof(
            other, "u1", "d1", "k1", self.prekey_k1)
        self._overwrite(document)
        self._assert_startup_refuses()

    def test_file_bytes_untouched_on_refusal(self) -> None:
        first = DeviceService()
        attach_persistence(first, self.path)
        self._register(first)
        document = self._document()
        document["devices"][0]["prekeys"][0]["signature"] = \
            self.sig_k1[:-2] + "AA"
        self._overwrite(document)
        before = open(self.path, "rb").read()
        self._assert_startup_refuses()
        self.assertEqual(open(self.path, "rb").read(), before)


class PrekeyProofServeRefusalTest(unittest.TestCase):
    def test_serve_stderr_single_json_field_data_file_exit_1(self) -> None:
        directory = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(
            directory, ignore_errors=True))
        path = os.path.join(directory, "state.json")
        private, identity = _new_identity()
        public_key = _new_prekey()
        signature = _proof(private, "u1", "d1", "k1", public_key)
        service = DeviceService()
        attach_persistence(service, path)
        service.register_verified({
            "user_id": "u1", "device_id": "d1",
            "identity_key": identity,
            "signed_prekeys": [{
                "key_id": "k1", "public_key": public_key,
                "signature": signature}]})
        document = json.load(open(path))
        document["devices"][0]["prekeys"][0]["identity_key"] = \
            _raw_b64(ed25519.Ed25519PrivateKey.generate().public_key())
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        before = open(path, "rb").read()

        env = dict(os.environ, PYTHONPATH=".")
        result = subprocess.run(
            [sys.executable, "-m", "e2ee_backend", "serve",
             "--data-file", path, "--host", "127.0.0.1", "--port", "0"],
            capture_output=True, text=True, env=env, timeout=15)
        self.assertEqual(result.returncode, 1)
        lines = result.stderr.strip().splitlines()
        self.assertEqual(len(lines), 1)
        body = json.loads(lines[0])
        self.assertEqual(body["field"], "data_file")
        self.assertFalse(result.stdout.strip())
        self.assertEqual(open(path, "rb").read(), before)


if __name__ == "__main__":
    unittest.main()


class PrekeyProofTransactionTest(unittest.TestCase):
    def test_persist_failure_rolls_back_proof_and_device(self) -> None:
        from e2ee_backend.persistence import PersistenceUnavailable

        path = tempfile.mktemp(suffix=".json")
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        private, identity = _new_identity()
        public_key = _new_prekey()
        signature = _proof(private, "u1", "d1", "k1", public_key)
        service = DeviceService()
        state_store = attach_persistence(service, path)

        def fail_save(_pending) -> None:
            raise OSError("simulated disk failure")

        state_store.save = fail_save  # type: ignore[assignment]
        with self.assertRaises(PersistenceUnavailable):
            service.register_verified({
                "user_id": "u1", "device_id": "d1",
                "identity_key": identity,
                "signed_prekeys": [{
                    "key_id": "k1", "public_key": public_key,
                    "signature": signature}]})
        # Whole transaction rolled back: no device, no partial proof.
        with self.assertRaises(ServiceError) as ctx:
            service.get_prekey_proof("d1", "k1")
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "device_id")

    def test_idempotent_replay_writes_no_generation_or_proof(self) -> None:
        path = tempfile.mktemp(suffix=".json")
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        private, identity = _new_identity()
        registered_key = _new_prekey()
        service = DeviceService()
        store = attach_persistence(service, path)
        service.register_verified({
            "user_id": "u1", "device_id": "d1",
            "identity_key": identity,
            "signed_prekeys": [{
                "key_id": "k1", "public_key": registered_key,
                "signature": _proof(private, "u1", "d1", "k1",
                                    registered_key)}]})
        # An ordinary key created after registration carries no proof.
        plain_key = _new_prekey()
        service.add_prekey(
            "d1", {"key_id": "plain", "public_key": plain_key})
        generation = store.commit_seq

        def fail_save(_pending) -> None:
            raise AssertionError("an idempotent replay must never save")

        store.save = fail_save  # type: ignore[assignment]
        valid = _proof(private, "u1", "d1", "plain", plain_key)
        _, status = service.add_prekey_verified(
            "d1", {"key_id": "plain", "public_key": plain_key,
                   "signature": valid})
        self.assertEqual(status, 200)
        # No generation was consumed and no proof was backfilled.
        self.assertEqual(store.commit_seq, generation)
        with self.assertRaises(ServiceError) as ctx:
            service.get_prekey_proof("d1", "plain")
        self.assertEqual(ctx.exception.field, "signature")
