"""Tests for the signed pre-key proof query.

Covers the service, HTTP (real loopback socket) and persistence layers for:

* GET /v1/devices/{device_id}/prekeys/{key_id}/proof

Only pre-keys first created through a verifiable entry (verified
registration or the verified replenishment route) retain a proof. The
200 response carries exactly six frozen string fields
(``user_id``/``device_id``/``key_id``/``public_key``/``identity_key``/
``signature``); the identity key is the publish-time one and never changes
after a rotation. Ordinary publishes, idempotent verified replays of
ordinarily-published ids and legacy data have no proof (409/signature).
The lookup is read-only and survives key consumption, key/device
revocation and restarts.
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
from cryptography.hazmat.primitives.asymmetric import ed25519, x25519

from e2ee_backend.crypto import (
    load_ed25519_public_key,
    signed_prekey_proof_message,
    verify_signed_prekey,
)
from e2ee_backend.http_app import create_server
from e2ee_backend.persistence import (
    PersistenceUnavailable,
    StateFileError,
    attach_persistence,
)
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


PROOF_FIELDS = ("user_id", "device_id", "key_id", "public_key",
                "identity_key", "signature")


class PreKeyProofServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = DeviceService()
        self.private, self.identity = _new_identity()
        self.prekey_k1 = _new_prekey()
        self.sig_k1 = _proof(self.private, "u1", "d1",
                             "k1", self.prekey_k1)
        self.service.register_verified({
            "user_id": "u1", "device_id": "d1",
            "identity_key": self.identity,
            "signed_prekeys": [{
                "key_id": "k1", "public_key": self.prekey_k1,
                "signature": self.sig_k1}]})

    def test_registered_proof_is_frozen_verbatim(self) -> None:
        proof = self.service.get_prekey_proof("d1", "k1")
        self.assertEqual(set(proof), set(PROOF_FIELDS))
        self.assertEqual(proof, {
            "user_id": "u1", "device_id": "d1", "key_id": "k1",
            "public_key": self.prekey_k1,
            "identity_key": self.identity,
            "signature": self.sig_k1})

    def test_proof_independently_verifies_under_published_rules(self) -> None:
        proof = self.service.get_prekey_proof("d1", "k1")
        identity = load_ed25519_public_key(proof["identity_key"])
        self.assertIsNotNone(identity)
        self.assertTrue(verify_signed_prekey(
            identity, base64.b64decode(proof["signature"]),
            proof["user_id"], proof["device_id"], proof["key_id"],
            proof["public_key"]))

    def test_verified_add_retains_proof(self) -> None:
        public_key = _new_prekey()
        signature = _proof(self.private, "u1", "d1", "k2", public_key)
        _, status = self.service.add_prekey_verified(
            "d1", {"key_id": "k2", "public_key": public_key,
                   "signature": signature})
        self.assertEqual(status, 201)
        proof = self.service.get_prekey_proof("d1", "k2")
        self.assertEqual(proof, {
            "user_id": "u1", "device_id": "d1", "key_id": "k2",
            "public_key": public_key, "identity_key": self.identity,
            "signature": signature})

    def test_ordinary_publish_has_no_proof(self) -> None:
        public_key = _new_prekey()
        self.service.add_prekey(
            "d1", {"key_id": "plain", "public_key": public_key})
        with self.assertRaises(ServiceError) as ctx:
            self.service.get_prekey_proof("d1", "plain")
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "signature")

    def test_idempotent_verified_replay_does_not_backfill(self) -> None:
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
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "signature")

    def test_idempotent_replay_keeps_existing_proof_unchanged(self) -> None:
        public_key = _new_prekey()
        signature = _proof(self.private, "u1", "d1", "k2", public_key)
        self.service.add_prekey_verified(
            "d1", {"key_id": "k2", "public_key": public_key,
                   "signature": signature})
        _, status = self.service.add_prekey_verified(
            "d1", {"key_id": "k2", "public_key": public_key,
                   "signature": signature})
        self.assertEqual(status, 200)
        self.assertEqual(
            self.service.get_prekey_proof("d1", "k2")["signature"],
            signature)

    def test_unknown_device_404_device_id(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.get_prekey_proof("ghost", "k1")
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "device_id")

    def test_unknown_key_404_key_id(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.get_prekey_proof("d1", "ghost")
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "key_id")

    def test_proof_survives_prekey_revocation_unchanged(self) -> None:
        self.service.revoke_prekey("d1", "k1")
        self.assertEqual(
            self.service.get_prekey_proof("d1", "k1")["signature"],
            self.sig_k1)

    def test_proof_survives_device_revocation_unchanged(self) -> None:
        self.service.revoke_device("d1")
        proof = self.service.get_prekey_proof("d1", "k1")
        self.assertEqual(proof["signature"], self.sig_k1)
        self.assertEqual(proof["identity_key"], self.identity)

    def test_proof_survives_claim_consumption(self) -> None:
        body, status = self.service.claim_prekey(
            {"recipient_device_id": "d1", "claim_id": "c1"})
        self.assertEqual(status, 201)
        self.assertEqual(body["key_id"], "k1")
        self.assertEqual(
            self.service.get_prekey_proof("d1", "k1")["public_key"],
            self.prekey_k1)

    def test_proof_keeps_publish_identity_after_rotation(self) -> None:
        new_private, new_identity = _new_identity()
        self.service.rotate_identity_key(
            "d1", {"identity_key": new_identity})
        proof = self.service.get_prekey_proof("d1", "k1")
        self.assertEqual(proof["identity_key"], self.identity)
        identity = load_ed25519_public_key(self.identity)
        self.assertTrue(verify_signed_prekey(
            identity, base64.b64decode(proof["signature"]),
            proof["user_id"], proof["device_id"], proof["key_id"],
            proof["public_key"]))
        public_key = _new_prekey()
        signature = _proof(new_private, "u1", "d1", "k2", public_key)
        self.service.add_prekey_verified(
            "d1", {"key_id": "k2", "public_key": public_key,
                   "signature": signature})
        self.assertEqual(
            self.service.get_prekey_proof("d1", "k2")["identity_key"],
            new_identity)


class PreKeyProofHTTPTest(unittest.TestCase):
    def setUp(self) -> None:
        self.server, self.service = create_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        self.private, self.identity = _new_identity()
        self.public_key = _new_prekey()
        self.signature = _proof(self.private, "u1", "d1",
                                "k1", self.public_key)
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

    def _get(self, target):
        connection = HTTPConnection("127.0.0.1", self.port, timeout=5)
        connection.request("GET", target)
        response = connection.getresponse()
        data = response.read().decode("utf-8")
        connection.close()
        return response.status, json.loads(data)

    def test_proof_200_six_fields(self) -> None:
        status, body = self._get("/v1/devices/d1/prekeys/k1/proof")
        self.assertEqual(status, 200)
        self.assertEqual(set(body), set(PROOF_FIELDS))
        self.assertEqual(body["user_id"], "u1")
        self.assertEqual(body["device_id"], "d1")
        self.assertEqual(body["key_id"], "k1")
        self.assertEqual(body["public_key"], self.public_key)
        self.assertEqual(body["identity_key"], self.identity)
        self.assertEqual(body["signature"], self.signature)

    def test_query_string_and_body_ignored(self) -> None:
        status, body = self._get(
            "/v1/devices/d1/prekeys/k1/proof?unused=%2F&x=1")
        self.assertEqual(status, 200)
        self.assertEqual(body["signature"], self.signature)

    def test_unknown_device_404(self) -> None:
        status, body = self._get(
            "/v1/devices/ghost/prekeys/k1/proof")
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "device_id")

    def test_unknown_key_404(self) -> None:
        status, body = self._get(
            "/v1/devices/d1/prekeys/ghost/proof")
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "key_id")

    def test_existing_key_without_proof_409_signature(self) -> None:
        status, body = self._get(
            "/v1/devices/d1/prekeys/plain/proof")
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "signature")

    def test_percent_encoded_slash_is_part_of_identifiers(self) -> None:
        public_key = _new_prekey()
        signature = _proof(self.private, "u1", "d/1", "a/b", public_key)
        self.service.register_verified({
            "user_id": "u1", "device_id": "d/1",
            "identity_key": self.identity,
            "signed_prekeys": [{
                "key_id": "a/b", "public_key": public_key,
                "signature": signature}]})
        status, body = self._get(
            "/v1/devices/d%2F1/prekeys/a%2Fb/proof")
        self.assertEqual(status, 200)
        self.assertEqual(body["device_id"], "d/1")
        self.assertEqual(body["key_id"], "a/b")

    def test_empty_decoded_device_id_400(self) -> None:
        status, body = self._get(
            "/v1/devices/%20/prekeys/k/proof")
        # %20 decodes to a space (non-empty) — an unknown device, not 400.
        self.assertEqual(status, 404)

    def test_bad_escape_device_id_400(self) -> None:
        status, body = self._get(
            "/v1/devices/d%zz/prekeys/k/proof")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "device_id")

    def test_bad_escape_key_id_400(self) -> None:
        status, body = self._get(
            "/v1/devices/d1/prekeys/k%zz/proof")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "key_id")

    def test_invalid_utf8_device_id_400(self) -> None:
        status, body = self._get(
            "/v1/devices/%ff/prekeys/k/proof")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "device_id")

    def test_invalid_utf8_key_id_400(self) -> None:
        status, body = self._get(
            "/v1/devices/d1/prekeys/%ff/proof")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "key_id")

    def test_deeper_path_is_404(self) -> None:
        status, _ = self._get(
            "/v1/devices/d1/prekeys/k1/proof/extra")
        self.assertEqual(status, 404)

    def test_empty_device_segment_400(self) -> None:
        status, body = self._get("/v1/devices//prekeys/k1/proof")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "device_id")

    def test_empty_key_segment_400(self) -> None:
        status, body = self._get(
            "/v1/devices/d1/prekeys//proof")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "key_id")


class PreKeyProofPersistenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.path = os.path.join(self.directory, "state.json")
        self.private, self.identity = _new_identity()
        self.public_key = _new_prekey()
        self.signature = _proof(self.private, "u1", "persist",
                                "k1", self.public_key)

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def _register(self, service) -> None:
        service.register_verified({
            "user_id": "u1", "device_id": "persist",
            "identity_key": self.identity,
            "signed_prekeys": [{
                "key_id": "k1", "public_key": self.public_key,
                "signature": self.signature}]})

    def test_proof_persists_and_restarts_unchanged(self) -> None:
        service = DeviceService()
        attach_persistence(service, self.path)
        self._register(service)
        before = service.get_prekey_proof("persist", "k1")

        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        self.assertEqual(
            restarted.get_prekey_proof("persist", "k1"), before)

    def test_proof_is_covered_by_the_integrity_probe_and_sidecar(self) -> None:
        service = DeviceService()
        attach_persistence(service, self.path)
        self._register(service)
        report = service.persistence_integrity()
        self.assertTrue(report["consistent"])
        with open(self.path + ".integrity", encoding="utf-8") as handle:
            sidecar = json.load(handle)
        self.assertEqual(sidecar["entries"][-1]["state_hash"],
                         report["state_hash"])

    def test_legacy_proofless_file_loads_and_answers_409(self) -> None:
        service = DeviceService()
        attach_persistence(service, self.path)
        self._register(service)
        # Rewrite the document as a proof-less legacy file (no marker and
        # no sidecar, as it would have looked before this feature).
        document = json.load(open(self.path, encoding="utf-8"))
        for prekey in document["devices"][0]["prekeys"]:
            prekey.pop("proof", None)
        document.pop("integrity_log_version", None)
        os.unlink(self.path)
        os.unlink(self.path + ".integrity")
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(document, handle, separators=(",", ":"),
                      ensure_ascii=False)
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        with self.assertRaises(ServiceError) as ctx:
            restarted.get_prekey_proof("persist", "k1")
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "signature")
        self.assertTrue(restarted.persistence_integrity()["consistent"])

    def test_persist_failure_leaves_no_partial_proof(self) -> None:
        service = DeviceService()
        state_store = attach_persistence(service, self.path)
        self._register(service)

        def fail_save(_pending) -> None:
            raise OSError("simulated disk failure")

        state_store.save = fail_save  # type: ignore[assignment]
        added = _new_prekey()
        with self.assertRaises(PersistenceUnavailable):
            service.add_prekey_verified(
                "persist",
                {"key_id": "k2", "public_key": added,
                 "signature": _proof(self.private, "u1", "persist",
                                     "k2", added)})
        with self.assertRaises(ServiceError) as ctx:
            service.get_prekey_proof("persist", "k2")
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "key_id")
        self.assertEqual(service.get_device("persist")["prekey_ids"],
                         ["k1"])

    def _tamper_file(self, mutate) -> None:
        service = DeviceService()
        attach_persistence(service, self.path)
        self._register(service)
        document = json.load(open(self.path, encoding="utf-8"))
        mutate(document)
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(document, handle, separators=(",", ":"),
                      ensure_ascii=False)

    def _assert_startup_refuses(self) -> None:
        with self.assertRaises(StateFileError):
            attach_persistence(DeviceService(), self.path)

    def test_missing_proof_field_refuses_startup(self) -> None:
        self._tamper_file(
            lambda d: d["devices"][0]["prekeys"][0]["proof"]
            .pop("signature"))
        self._assert_startup_refuses()

    def test_wrong_proof_field_type_refuses_startup(self) -> None:
        self._tamper_file(
            lambda d: d["devices"][0]["prekeys"][0]["proof"]
            .__setitem__("signature", 9))
        self._assert_startup_refuses()

    def test_bad_signature_encoding_refuses_startup(self) -> None:
        self._tamper_file(
            lambda d: d["devices"][0]["prekeys"][0]["proof"]
            .__setitem__("signature", "@@@@"))
        self._assert_startup_refuses()

    def test_non_ed25519_proof_identity_refuses_startup(self) -> None:
        self._tamper_file(
            lambda d: d["devices"][0]["prekeys"][0]["proof"]
            .__setitem__("identity_key", _new_prekey()))
        self._assert_startup_refuses()

    def test_public_key_mismatch_refuses_startup(self) -> None:
        self._tamper_file(
            lambda d: d["devices"][0]["prekeys"][0]
            .__setitem__("public_key", _new_prekey()))
        self._assert_startup_refuses()

    def test_foreign_identity_refuses_startup(self) -> None:
        _, other_identity = _new_identity()
        self._tamper_file(
            lambda d: d["devices"][0]["prekeys"][0]["proof"]
            .__setitem__("identity_key", other_identity))
        self._assert_startup_refuses()

    def test_tampered_signature_refuses_startup(self) -> None:
        def flip(document) -> None:
            raw = bytearray(base64.b64decode(
                document["devices"][0]["prekeys"][0]["proof"]
                ["signature"]))
            raw[0] ^= 1
            document["devices"][0]["prekeys"][0]["proof"][
                "signature"] = base64.b64encode(bytes(raw)).decode()

        self._tamper_file(flip)
        self._assert_startup_refuses()

    def test_refused_startup_leaves_file_untouched(self) -> None:
        self._tamper_file(
            lambda d: d["devices"][0]["prekeys"][0]["proof"]
            .pop("signature"))
        before = open(self.path, "rb").read()
        with self.assertRaises(StateFileError):
            attach_persistence(DeviceService(), self.path)
        self.assertEqual(open(self.path, "rb").read(), before)
