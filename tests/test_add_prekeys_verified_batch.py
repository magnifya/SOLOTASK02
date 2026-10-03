"""Tests for identity-authorized batch pre-key replenishment.

Covers the service, HTTP (real loopback socket), CLI (real subprocess),
key-audit and persistence layers for:

* POST /v1/devices/{device_id}/prekeys/verified-batch

The batch entry publishes a non-empty ``signed_prekeys`` array
all-or-nothing: every element's standard-base64 64-byte Ed25519 signature
is verified over the same domain-separated canonical proof
(``E2EE-SIGNED-PREKEY-V1``) as the single-item verified entry, checked
against the device's *current* identity key in array order, each element's
proof before its conflict decision. A batch with at least one new id
returns 201; a pure idempotent replay returns 200. Nothing changes on
failure.
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
from e2ee_backend.persistence import attach_persistence
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


class AddPrekeysVerifiedBatchServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = DeviceService()
        self.private, self.identity = _new_identity()
        self.prekey_k1 = _new_prekey()
        self.service.register_verified({
            "user_id": "u1", "device_id": "d1",
            "identity_key": self.identity,
            "signed_prekeys": [{
                "key_id": "k1", "public_key": self.prekey_k1,
                "signature": _proof(self.private, "u1", "d1",
                                    "k1", self.prekey_k1)}],
        })

    def _element(self, key_id, public_key=None, signature=None,
                 private=None):
        if public_key is None:
            public_key = _new_prekey()
        if signature is None:
            signer = private or self.private
            signature = _proof(signer, "u1", "d1", key_id, public_key)
        return {"key_id": key_id, "public_key": public_key,
                "signature": signature}

    def _payload(self, *key_ids):
        return {"signed_prekeys": [self._element(key_id)
                                   for key_id in key_ids]}

    def test_valid_batch_appends_201_in_order(self) -> None:
        elements = [self._element("k2"), self._element("k3")]
        body, status = self.service.add_prekeys_verified_batch(
            "d1", {"signed_prekeys": elements})
        self.assertEqual(status, 201)
        self.assertEqual(body, {
            "device_id": "d1",
            "signed_prekeys": [
                {"key_id": "k2", "public_key": elements[0]["public_key"]},
                {"key_id": "k3", "public_key": elements[1]["public_key"]}],
        })
        self.assertEqual(self.service.get_device("d1")["prekey_ids"],
                         ["k1", "k2", "k3"])

    def test_mixed_new_and_idempotent_201(self) -> None:
        first = self._payload("k2")
        _, status = self.service.add_prekeys_verified_batch("d1", first)
        self.assertEqual(status, 201)
        mixed = {"signed_prekeys": [first["signed_prekeys"][0],
                                    self._element("k3")]}
        body, status = self.service.add_prekeys_verified_batch("d1", mixed)
        self.assertEqual(status, 201)
        self.assertEqual([e["key_id"] for e in body["signed_prekeys"]],
                         ["k2", "k3"])
        self.assertEqual(self.service.get_device("d1")["prekey_ids"],
                         ["k1", "k2", "k3"])

    def test_pure_replay_200(self) -> None:
        payload = self._payload("k2", "k3")
        _, first = self.service.add_prekeys_verified_batch("d1", payload)
        self.assertEqual(first, 201)
        replay = {"signed_prekeys": [dict(e) for e in
                                     payload["signed_prekeys"]]}
        body, second = self.service.add_prekeys_verified_batch("d1", replay)
        self.assertEqual(second, 200)
        self.assertEqual([e["key_id"] for e in body["signed_prekeys"]],
                         ["k2", "k3"])
        self.assertEqual(self.service.get_device("d1")["prekey_ids"],
                         ["k1", "k2", "k3"])

    def test_body_not_object_400_request_body(self) -> None:
        for payload in (None, [], "x", 42):
            with self.subTest(payload=payload):
                with self.assertRaises(ServiceError) as ctx:
                    self.service.add_prekeys_verified_batch("d1", payload)
                self.assertEqual(ctx.exception.status_code, 400)
                self.assertEqual(ctx.exception.field, "request_body")

    def test_signed_prekeys_missing_not_array_or_empty_400(self) -> None:
        for payload in ({}, {"signed_prekeys": "x"},
                        {"signed_prekeys": 42}, {"signed_prekeys": {}},
                        {"signed_prekeys": []}):
            with self.subTest(payload=payload):
                with self.assertRaises(ServiceError) as ctx:
                    self.service.add_prekeys_verified_batch("d1", payload)
                self.assertEqual(ctx.exception.status_code, 400)
                self.assertEqual(ctx.exception.field, "signed_prekeys")

    def test_non_object_element_400_indexed(self) -> None:
        for raw in (None, "x", 42, []):
            payload = {"signed_prekeys": [self._element("k2"), raw]}
            with self.subTest(raw=raw):
                with self.assertRaises(ServiceError) as ctx:
                    self.service.add_prekeys_verified_batch("d1", payload)
                self.assertEqual(ctx.exception.status_code, 400)
                self.assertEqual(ctx.exception.field, "signed_prekeys[1]")

    def test_element_field_errors_400_indexed_path(self) -> None:
        good = self._element("k2")
        cases = [
            ({}, "signed_prekeys[0].key_id"),
            ({"key_id": "k2"}, "signed_prekeys[0].public_key"),
            ({"key_id": "k2", "public_key": _new_prekey()},
             "signed_prekeys[0].signature"),
            (dict(good, key_id=""), "signed_prekeys[0].key_id"),
            (dict(good, key_id=9), "signed_prekeys[0].key_id"),
            (dict(good, public_key=7), "signed_prekeys[0].public_key"),
            (dict(good, public_key=""), "signed_prekeys[0].public_key"),
            (dict(good, signature=8), "signed_prekeys[0].signature"),
            (dict(good, signature=""), "signed_prekeys[0].signature"),
        ]
        for element, field in cases:
            with self.subTest(field=field):
                with self.assertRaises(ServiceError) as ctx:
                    self.service.add_prekeys_verified_batch(
                        "d1", {"signed_prekeys": [element]})
                self.assertEqual(ctx.exception.status_code, 400)
                self.assertEqual(ctx.exception.field, field)

    def test_bad_public_key_400_indexed_public_key(self) -> None:
        payload = {"signed_prekeys": [
            self._element("k2"),
            self._element("k3", public_key="not-a-key")]}
        with self.assertRaises(ServiceError) as ctx:
            self.service.add_prekeys_verified_batch("d1", payload)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.field, "signed_prekeys[1].public_key")

    def test_bad_signature_encoding_400_indexed_signature(self) -> None:
        for bad in ("@@@@", "abc", "a" * 88,
                    base64.b64encode(b"\x00" * 63).decode(),
                    base64.b64encode(b"\x00" * 65).decode()):
            element = self._element("k3", signature=bad)
            payload = {"signed_prekeys": [self._element("k2"), element]}
            with self.subTest(bad=bad):
                with self.assertRaises(ServiceError) as ctx:
                    self.service.add_prekeys_verified_batch("d1", payload)
                self.assertEqual(ctx.exception.status_code, 400)
                self.assertEqual(ctx.exception.field,
                                 "signed_prekeys[1].signature")

    def test_duplicate_key_id_400_at_second_occurrence(self) -> None:
        element = self._element("k2")
        payload = {"signed_prekeys": [self._element("k9"), element,
                                      self._element("k2")]}
        with self.assertRaises(ServiceError) as ctx:
            self.service.add_prekeys_verified_batch("d1", payload)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.field, "signed_prekeys[2].key_id")
        # Nothing was published.
        self.assertEqual(self.service.get_device("d1")["prekey_ids"], ["k1"])

    def test_wrong_signer_400_indexed_signature(self) -> None:
        other, _ = _new_identity()
        payload = {"signed_prekeys": [self._element("k2"),
                                      self._element("k3", private=other)]}
        with self.assertRaises(ServiceError) as ctx:
            self.service.add_prekeys_verified_batch("d1", payload)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.field, "signed_prekeys[1].signature")
        # All-or-nothing: the valid first element was not appended either.
        self.assertEqual(self.service.get_device("d1")["prekey_ids"], ["k1"])

    def test_unknown_device_404(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.add_prekeys_verified_batch(
                "ghost", self._payload("k2"))
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "device_id")

    def test_revoked_device_409(self) -> None:
        self.service.revoke_device("d1")
        with self.assertRaises(ServiceError) as ctx:
            self.service.add_prekeys_verified_batch(
                "d1", self._payload("k2"))
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "device_id")

    def test_non_ed25519_identity_400_identity_key(self) -> None:
        x25519_der = base64.b64encode(
            x25519.X25519PrivateKey.generate().public_key().public_bytes(
                serialization.Encoding.DER,
                serialization.PublicFormat.SubjectPublicKeyInfo)).decode()
        self.service.register({
            "user_id": "u2", "device_id": "d2",
            "identity_key": x25519_der,
            "signed_prekeys": [{"key_id": "k1", "public_key": _new_prekey()}],
        })
        with self.assertRaises(ServiceError) as ctx:
            self.service.add_prekeys_verified_batch(
                "d2", self._payload("k2"))
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.field, "identity_key")
        self.assertEqual(self.service.get_device("d2")["prekey_ids"], ["k1"])

    def test_same_id_changed_key_409_indexed_key_id(self) -> None:
        public_key = _new_prekey()
        self.service.add_prekeys_verified_batch(
            "d1", {"signed_prekeys": [self._element("k2",
                                                  public_key=public_key)]})
        payload = {"signed_prekeys": [
            self._element("k3"),
            self._element("k2", public_key=_new_prekey())]}
        with self.assertRaises(ServiceError) as ctx:
            self.service.add_prekeys_verified_batch("d1", payload)
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "signed_prekeys[1].key_id")
        # All-or-nothing: k3 was not appended.
        self.assertEqual(self.service.get_device("d1")["prekey_ids"],
                         ["k1", "k2"])

    def test_revoked_key_id_409_indexed_key_id(self) -> None:
        public_key = _new_prekey()
        self.service.add_prekeys_verified_batch(
            "d1", {"signed_prekeys": [self._element("k2",
                                                  public_key=public_key)]})
        self.service.revoke_prekey("d1", "k2")
        with self.assertRaises(ServiceError) as ctx:
            self.service.add_prekeys_verified_batch(
                "d1", {"signed_prekeys": [self._element("k2",
                                                      public_key=public_key)]})
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "signed_prekeys[0].key_id")

    def test_invalid_signature_precedes_conflict_of_same_element(self) -> None:
        # An existing-id conflict must not mask a bad proof on the same
        # element: 400/signature at that element's path.
        public_key = _new_prekey()
        self.service.add_prekeys_verified_batch(
            "d1", {"signed_prekeys": [self._element("k2",
                                                  public_key=public_key)]})
        other, _ = _new_identity()
        payload = {"signed_prekeys": [
            self._element("k2", public_key=public_key, private=other)]}
        with self.assertRaises(ServiceError) as ctx:
            self.service.add_prekeys_verified_batch("d1", payload)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.field, "signed_prekeys[0].signature")

    def test_elements_checked_in_array_order(self) -> None:
        # Element 0 conflicts, element 1 has a bad proof: the first failing
        # element in array order is reported.
        public_key = _new_prekey()
        self.service.add_prekeys_verified_batch(
            "d1", {"signed_prekeys": [self._element("k2",
                                                  public_key=public_key)]})
        other, _ = _new_identity()
        payload = {"signed_prekeys": [
            self._element("k2", public_key=_new_prekey()),
            self._element("k3", private=other)]}
        with self.assertRaises(ServiceError) as ctx:
            self.service.add_prekeys_verified_batch("d1", payload)
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "signed_prekeys[0].key_id")

    def test_verifies_against_current_identity_after_rotation(self) -> None:
        new_private, new_identity = _new_identity()
        self.service.rotate_identity_key(
            "d1", {"identity_key": new_identity})
        with self.assertRaises(ServiceError) as ctx:
            self.service.add_prekeys_verified_batch(
                "d1", self._payload("k2"))
        self.assertEqual(ctx.exception.field, "signed_prekeys[0].signature")
        body, status = self.service.add_prekeys_verified_batch(
            "d1", {"signed_prekeys": [self._element("k2",
                                                  private=new_private)]})
        self.assertEqual(status, 201)
        self.assertEqual(body["signed_prekeys"][0]["key_id"], "k2")

    def test_uses_stored_user_id_in_proof(self) -> None:
        public_key = _new_prekey()
        signature = _proof(self.private, "attacker", "d1", "k2", public_key)
        with self.assertRaises(ServiceError) as ctx:
            self.service.add_prekeys_verified_batch(
                "d1", {"signed_prekeys": [
                    {"key_id": "k2", "public_key": public_key,
                     "signature": signature}]})
        self.assertEqual(ctx.exception.field, "signed_prekeys[0].signature")

    def test_audit_events_per_new_item_only(self) -> None:
        page = self.service.list_key_events("d1", 0, 100)
        self.assertEqual([e["type"] for e in page["events"]],
                         ["registered"])
        payload = self._payload("k2", "k3")
        self.service.add_prekeys_verified_batch("d1", payload)
        self.service.add_prekeys_verified_batch(  # pure replay: no events
            "d1", {"signed_prekeys": [dict(e) for e in
                                      payload["signed_prekeys"]]})
        page = self.service.list_key_events("d1", 0, 100)
        self.assertEqual([e["type"] for e in page["events"]],
                         ["registered", "prekey_added", "prekey_added"])
        added = page["events"][1:]
        self.assertEqual([e["payload"]["key_id"] for e in added],
                         ["k2", "k3"])
        self.assertEqual(added[0]["prev_hash"], page["events"][0]["hash"])
        self.assertEqual(added[1]["prev_hash"], added[0]["hash"])

    def test_new_items_publish_frozen_proofs(self) -> None:
        element = self._element("k2")
        self.service.add_prekeys_verified_batch(
            "d1", {"signed_prekeys": [element]})
        proof = self.service.get_prekey_proof("d1", "k2")
        self.assertEqual(proof, {
            "user_id": "u1", "device_id": "d1", "key_id": "k2",
            "public_key": element["public_key"],
            "identity_key": self.identity,
            "signature": element["signature"]})

    def test_replay_does_not_rewrite_proof_or_restore_consumed(self) -> None:
        element = self._element("k2")
        self.service.add_prekeys_verified_batch(
            "d1", {"signed_prekeys": [element]})
        # Consume k2, then rotate the identity; a replay must neither
        # restore the consumed flag nor replace the frozen proof.
        _, status = self.service.claim_prekey(
            {"recipient_device_id": "d1", "claim_id": "c1"})
        self.assertEqual(status, 201)
        _, status = self.service.claim_prekey(
            {"recipient_device_id": "d1", "claim_id": "c2"})
        self.assertEqual(status, 201)
        new_private, new_identity = _new_identity()
        self.service.rotate_identity_key(
            "d1", {"identity_key": new_identity})
        proof_before = self.service.get_prekey_proof("d1", "k2")
        replayed = dict(element)
        replayed["signature"] = _proof(new_private, "u1", "d1", "k2",
                                       element["public_key"])
        body, status = self.service.add_prekeys_verified_batch(
            "d1", {"signed_prekeys": [replayed]})
        self.assertEqual(status, 200)
        self.assertNotIn("k2", self.service.get_device("d1")["prekey_ids"])
        self.assertEqual(self.service.get_prekey_proof("d1", "k2"),
                         proof_before)

    def test_added_prekeys_are_claimable_in_order(self) -> None:
        self.service.add_prekeys_verified_batch(
            "d1", self._payload("k2", "k3"))
        claimed = []
        for index in range(3):
            body, status = self.service.claim_prekey(
                {"recipient_device_id": "d1", "claim_id": f"c{index}"})
            self.assertEqual(status, 201)
            claimed.append(body["key_id"])
        self.assertEqual(claimed, ["k1", "k2", "k3"])


class AddPrekeysVerifiedBatchHTTPTest(unittest.TestCase):
    def setUp(self) -> None:
        self.server, self.service = create_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        self.private, self.identity = _new_identity()
        public_key = _new_prekey()
        self.service.register_verified({
            "user_id": "u1", "device_id": "d1",
            "identity_key": self.identity,
            "signed_prekeys": [{
                "key_id": "k1", "public_key": public_key,
                "signature": _proof(self.private, "u1", "d1",
                                    "k1", public_key)}],
        })

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _request(self, method, path, body=None, raw=None):
        connection = HTTPConnection("127.0.0.1", self.port, timeout=5)
        payload = raw if raw is not None else (
            json.dumps(body) if body is not None else None)
        headers = {"Content-Type": "application/json"} if payload else {}
        connection.request(method, path, body=payload, headers=headers)
        response = connection.getresponse()
        data = response.read().decode("utf-8")
        connection.close()
        return response.status, json.loads(data)

    def _element(self, key_id, private=None):
        public_key = _new_prekey()
        signer = private or self.private
        return {"key_id": key_id, "public_key": public_key,
                "signature": _proof(signer, "u1", "d1", key_id, public_key)}

    def test_201_then_200(self) -> None:
        payload = {"signed_prekeys": [self._element("k8"),
                                      self._element("k9")]}
        status, body = self._request(
            "POST", "/v1/devices/d1/prekeys/verified-batch", payload)
        self.assertEqual(status, 201)
        self.assertEqual(set(body), {"device_id", "signed_prekeys"})
        self.assertEqual([set(e) for e in body["signed_prekeys"]],
                         [{"key_id", "public_key"}] * 2)
        replay = {"signed_prekeys": [dict(e) for e in
                                     payload["signed_prekeys"]]}
        status, body = self._request(
            "POST", "/v1/devices/d1/prekeys/verified-batch", replay)
        self.assertEqual(status, 200)

    def test_invalid_json_400_request_body(self) -> None:
        status, body = self._request(
            "POST", "/v1/devices/d1/prekeys/verified-batch", raw="{not json")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "request_body")

    def test_non_object_body_400_request_body(self) -> None:
        for raw in ("[]", "null", '"x"', "42"):
            status, body = self._request(
                "POST", "/v1/devices/d1/prekeys/verified-batch", raw=raw)
            self.assertEqual(status, 400, raw)
            self.assertEqual(body["field"], "request_body")

    def test_empty_array_400_signed_prekeys(self) -> None:
        status, body = self._request(
            "POST", "/v1/devices/d1/prekeys/verified-batch",
            {"signed_prekeys": []})
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "signed_prekeys")

    def test_bad_signature_400_indexed(self) -> None:
        status, body = self._request(
            "POST", "/v1/devices/d1/prekeys/verified-batch",
            {"signed_prekeys": [
                self._element("k2"),
                {"key_id": "k3", "public_key": _new_prekey(),
                 "signature": "not-base64"}]})
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "signed_prekeys[1].signature")

    def test_proof_failure_400_indexed_signature(self) -> None:
        other, _ = _new_identity()
        status, body = self._request(
            "POST", "/v1/devices/d1/prekeys/verified-batch",
            {"signed_prekeys": [self._element("k2", private=other)]})
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "signed_prekeys[0].signature")

    def test_unknown_device_404(self) -> None:
        status, body = self._request(
            "POST", "/v1/devices/ghost/prekeys/verified-batch",
            {"signed_prekeys": [self._element("k2")]})
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "device_id")

    def test_revoked_device_409(self) -> None:
        self.service.revoke_device("d1")
        status, body = self._request(
            "POST", "/v1/devices/d1/prekeys/verified-batch",
            {"signed_prekeys": [self._element("k2")]})
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "device_id")

    def test_single_and_ordinary_routes_still_work(self) -> None:
        element = self._element("k2")
        status, body = self._request(
            "POST", "/v1/devices/d1/prekeys/verified", element)
        self.assertEqual(status, 201)
        status, body = self._request(
            "POST", "/v1/devices/d1/prekeys",
            {"key_id": "plain", "public_key": _new_prekey()})
        self.assertEqual(status, 201)


class AddPrekeysVerifiedBatchCLITest(unittest.TestCase):
    def setUp(self) -> None:
        self.server, service = create_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        self.private, self.identity = _new_identity()
        public_key = _new_prekey()
        service.register_verified({
            "user_id": "u1", "device_id": "d1",
            "identity_key": self.identity,
            "signed_prekeys": [{
                "key_id": "k1", "public_key": public_key,
                "signature": _proof(self.private, "u1", "d1",
                                    "k1", public_key)}],
        })

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _run(self, *arguments) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "e2ee_backend", "--base-url",
             self.base_url, *arguments],
            capture_output=True, text=True, timeout=15)

    def _spec(self, key_id, public_key=None):
        if public_key is None:
            public_key = _new_prekey()
        signature = _proof(self.private, "u1", "d1", key_id, public_key)
        return f"{key_id}:{public_key}:{signature}"

    def test_success_stdout_zero_and_idempotent(self) -> None:
        spec1, spec2 = self._spec("kc1"), self._spec("kc2")
        first = self._run("add-prekeys-verified", "--device-id", "d1",
                          "--prekey", spec1,
                          "--prekey", spec2)
        self.assertEqual(first.returncode, 0, first.stderr)
        body = json.loads(first.stdout.strip())
        self.assertEqual(set(body), {"device_id", "signed_prekeys"})
        self.assertEqual([e["key_id"] for e in body["signed_prekeys"]],
                         ["kc1", "kc2"])
        self.assertFalse(first.stderr.strip())

        # Replaying the exact same specs is a pure idempotent replay.
        second = self._run("add-prekeys-verified", "--device-id", "d1",
                           "--prekey", spec1,
                           "--prekey", spec2)
        self.assertEqual(second.returncode, 0, second.stderr)

    def test_at_file_input(self) -> None:
        public_key = _new_prekey()
        element = {"key_id": "kf", "public_key": public_key,
                   "signature": _proof(self.private, "u1", "d1",
                                       "kf", public_key)}
        handle, path = tempfile.mkstemp(suffix=".json")
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        with os.fdopen(handle, "w", encoding="utf-8") as file:
            json.dump(element, file)
        result = self._run("add-prekeys-verified", "--device-id", "d1",
                           "--prekey", f"@{path}")
        self.assertEqual(result.returncode, 0, result.stderr)
        body = json.loads(result.stdout.strip())
        self.assertEqual([e["key_id"] for e in body["signed_prekeys"]],
                         ["kf"])

    def test_bad_prekey_spec_exit_2(self) -> None:
        result = self._run("add-prekeys-verified", "--device-id", "d1",
                           "--prekey", "no-colons-here")
        self.assertEqual(result.returncode, 2)
        self.assertFalse(result.stdout.strip())
        body = json.loads(result.stderr.strip())
        self.assertEqual(body["field"], "signed_prekeys")

    def test_failure_stderr_field_nonzero(self) -> None:
        result = self._run("add-prekeys-verified", "--device-id", "ghost",
                           "--prekey", self._spec("kc"))
        self.assertEqual(result.returncode, 1)
        self.assertFalse(result.stdout.strip())
        body = json.loads(result.stderr.strip())
        self.assertEqual(body["field"], "device_id")


class AddPrekeysVerifiedBatchPersistenceTest(unittest.TestCase):
    def _register(self, service, private, identity):
        k1 = _new_prekey()
        service.register_verified({
            "user_id": "u1", "device_id": "d1",
            "identity_key": identity,
            "signed_prekeys": [{
                "key_id": "k1", "public_key": k1,
                "signature": _proof(private, "u1", "d1", "k1", k1)}],
        })
        return k1

    def test_batch_and_proofs_survive_restart(self) -> None:
        path = tempfile.mktemp(suffix=".json")
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        private, identity = _new_identity()

        first = DeviceService()
        attach_persistence(first, path)
        self._register(first, private, identity)
        added = [_new_prekey(), _new_prekey()]
        first.add_prekeys_verified_batch("d1", {"signed_prekeys": [
            {"key_id": "k2", "public_key": added[0],
             "signature": _proof(private, "u1", "d1", "k2", added[0])},
            {"key_id": "k3", "public_key": added[1],
             "signature": _proof(private, "u1", "d1", "k3", added[1])}]})
        # Consume k2 so the restart also proves the consumed flag survives.
        first.claim_prekey({"recipient_device_id": "d1", "claim_id": "c1"})
        first.claim_prekey({"recipient_device_id": "d1", "claim_id": "c2"})

        second = DeviceService()
        attach_persistence(second, path)
        self.assertEqual(second.get_device("d1")["prekey_ids"], ["k3"])
        events = second.list_key_events("d1", 0, 100)["events"]
        self.assertEqual([e["type"] for e in events],
                         ["registered", "prekey_added", "prekey_added"])
        self.assertEqual([e["payload"]["key_id"] for e in events[1:]],
                         ["k2", "k3"])
        proof = second.get_prekey_proof("d1", "k2")
        self.assertEqual(proof["public_key"], added[0])
        self.assertEqual(proof["identity_key"], identity)

    def test_one_commit_generation_per_batch(self) -> None:
        path = tempfile.mktemp(suffix=".json")
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        private, identity = _new_identity()
        service = DeviceService()
        store = attach_persistence(service, path)
        self._register(service, private, identity)
        generation_after_register = store.commit_seq

        service.add_prekeys_verified_batch("d1", {"signed_prekeys": [
            {"key_id": key_id, "public_key": (key := _new_prekey()),
             "signature": _proof(private, "u1", "d1", key_id, key)}
            for key_id in ("k2", "k3", "k4")]})
        # The whole batch (three keys, three events) is one generation.
        self.assertEqual(store.commit_seq, generation_after_register + 1)

    def test_pure_replay_consumes_no_generation(self) -> None:
        path = tempfile.mktemp(suffix=".json")
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        private, identity = _new_identity()
        service = DeviceService()
        store = attach_persistence(service, path)
        self._register(service, private, identity)
        key = _new_prekey()
        payload = {"signed_prekeys": [
            {"key_id": "k2", "public_key": key,
             "signature": _proof(private, "u1", "d1", "k2", key)}]}
        service.add_prekeys_verified_batch("d1", payload)
        generation = store.commit_seq
        events = len(service.list_key_events("d1", 0, 100)["events"])

        _, status = service.add_prekeys_verified_batch(
            "d1", {"signed_prekeys": [dict(payload["signed_prekeys"][0])]})
        self.assertEqual(status, 200)
        self.assertEqual(store.commit_seq, generation)
        self.assertEqual(
            len(service.list_key_events("d1", 0, 100)["events"]), events)

    def test_failure_advances_no_generation_and_leaves_no_event(self) -> None:
        path = tempfile.mktemp(suffix=".json")
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        private, identity = _new_identity()
        service = DeviceService()
        store = attach_persistence(service, path)
        self._register(service, private, identity)
        generation_after_register = store.commit_seq
        events_after_register = len(
            service.list_key_events("d1", 0, 100)["events"])

        other, _ = _new_identity()
        bad_key = _new_prekey()
        with self.assertRaises(ServiceError):
            service.add_prekeys_verified_batch("d1", {"signed_prekeys": [
                {"key_id": "k2", "public_key": (key := _new_prekey()),
                 "signature": _proof(private, "u1", "d1", "k2", key)},
                {"key_id": "k3", "public_key": bad_key,
                 "signature": _proof(other, "u1", "d1", "k3", bad_key)}]})
        # No generation consumed, no event appended, no key added.
        self.assertEqual(store.commit_seq, generation_after_register)
        self.assertEqual(
            len(service.list_key_events("d1", 0, 100)["events"]),
            events_after_register)
        self.assertEqual(service.get_device("d1")["prekey_ids"], ["k1"])

    def test_persist_failure_rolls_back_the_whole_batch(self) -> None:
        from e2ee_backend.persistence import PersistenceUnavailable

        path = tempfile.mktemp(suffix=".json")
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        private, identity = _new_identity()
        service = DeviceService()
        state_store = attach_persistence(service, path)
        self._register(service, private, identity)

        def fail_save(_pending) -> None:
            raise OSError("simulated disk failure")

        state_store.save = fail_save  # type: ignore[assignment]
        added = [_new_prekey(), _new_prekey()]
        # The durable write fails inside the locked transaction: the HTTP
        # layer answers 503/data_file and the in-memory mutation is rolled
        # back to the last committed state.
        with self.assertRaises(PersistenceUnavailable):
            service.add_prekeys_verified_batch("d1", {"signed_prekeys": [
                {"key_id": "k2", "public_key": added[0],
                 "signature": _proof(private, "u1", "d1", "k2", added[0])},
                {"key_id": "k3", "public_key": added[1],
                 "signature": _proof(private, "u1", "d1", "k3", added[1])}]})
        # The mutation was rolled back: no keys, no appended events.
        self.assertEqual(service.get_device("d1")["prekey_ids"], ["k1"])
        self.assertEqual(
            [e["type"] for e in
             service.list_key_events("d1", 0, 100)["events"]],
            ["registered"])


class AddPrekeysVerifiedBatchConcurrencyTest(unittest.TestCase):
    def test_concurrent_batch_and_revoke_linearizes(self) -> None:
        service = DeviceService()
        private, identity = _new_identity()
        k1 = _new_prekey()
        service.register_verified({
            "user_id": "u1", "device_id": "d1",
            "identity_key": identity,
            "signed_prekeys": [{
                "key_id": "k1", "public_key": k1,
                "signature": _proof(private, "u1", "d1", "k1", k1)}],
        })
        added = [_new_prekey(), _new_prekey()]
        payload = {"signed_prekeys": [
            {"key_id": "k2", "public_key": added[0],
             "signature": _proof(private, "u1", "d1", "k2", added[0])},
            {"key_id": "k3", "public_key": added[1],
             "signature": _proof(private, "u1", "d1", "k3", added[1])}]}
        outcomes = []

        def add() -> None:
            try:
                _, status = service.add_prekeys_verified_batch(
                    "d1", {"signed_prekeys": [dict(e) for e in
                                              payload["signed_prekeys"]]})
                outcomes.append(("add", status))
            except ServiceError as error:
                outcomes.append(("add", error.status_code))

        def revoke() -> None:
            service.revoke_device("d1")
            outcomes.append(("revoke", 200))

        threads = [threading.Thread(target=add),
                   threading.Thread(target=revoke)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)

        add_result = next(code for kind, code in outcomes if kind == "add")
        # Exactly one linearized result: 201 (batch first) or 409 (revoke
        # first). The batch is never half-applied.
        self.assertIn(add_result, (201, 409))
        view = service.get_device("d1")
        self.assertEqual(view["prekey_ids"], [])
        events = service.list_key_events("d1", 0, 100)["events"]
        if add_result == 201:
            self.assertEqual(
                [e["type"] for e in events],
                ["registered", "prekey_added", "prekey_added",
                 "device_revoked"])
        else:
            self.assertEqual(
                [e["type"] for e in events],
                ["registered", "device_revoked"])


if __name__ == "__main__":
    unittest.main()
