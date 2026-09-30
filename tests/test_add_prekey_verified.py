"""Tests for identity-authorized post-registration pre-key replenishment.

Covers the service, HTTP (real loopback socket), CLI (real subprocess),
key-audit and persistence layers for:

* POST /v1/devices/{device_id}/prekeys/verified

The new entry appends a pre-key only when its standard-base64 64-byte
Ed25519 signature verifies over the same domain-separated canonical proof
(``E2EE-SIGNED-PREKEY-V1``) as verified registration, checked against the
device's *current* identity key. A new id returns 201; the same id with the
same non-revoked key and a valid proof is idempotent (200); the same id with
a changed key or a revoked id is 409/key_id. Nothing changes on failure.
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


class AddPrekeyVerifiedServiceTest(unittest.TestCase):
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

    def _payload(self, key_id="k2", public_key=None, signature=None,
                 private=None):
        if public_key is None:
            public_key = _new_prekey()
        if signature is None:
            signer = private or self.private
            signature = _proof(signer, "u1", "d1", key_id, public_key)
        return {"key_id": key_id, "public_key": public_key,
                "signature": signature}

    def test_valid_proof_appends_201(self) -> None:
        public_key = _new_prekey()
        body, status = self.service.add_prekey_verified(
            "d1", self._payload(key_id="k2", public_key=public_key))
        self.assertEqual(status, 201)
        self.assertEqual(body, {"device_id": "d1", "key_id": "k2",
                                "public_key": public_key})
        self.assertEqual(self.service.get_device("d1")["prekey_ids"],
                         ["k1", "k2"])

    def test_idempotent_same_key_200(self) -> None:
        public_key = _new_prekey()
        payload = self._payload(key_id="k2", public_key=public_key)
        _, first = self.service.add_prekey_verified("d1", payload)
        self.assertEqual(first, 201)
        body, second = self.service.add_prekey_verified("d1", dict(payload))
        self.assertEqual(second, 200)
        self.assertEqual(body, {"device_id": "d1", "key_id": "k2",
                                "public_key": public_key})
        self.assertEqual(self.service.get_device("d1")["prekey_ids"],
                         ["k1", "k2"])

    def test_body_not_object_400_request_body(self) -> None:
        for payload in (None, [], "x", 42):
            with self.subTest(payload=payload):
                with self.assertRaises(ServiceError) as ctx:
                    self.service.add_prekey_verified("d1", payload)
                self.assertEqual(ctx.exception.status_code, 400)
                self.assertEqual(ctx.exception.field, "request_body")

    def test_missing_or_typed_fields_400(self) -> None:
        cases = [
            ({}, "key_id"),
            ({"key_id": "k2"}, "public_key"),
            ({"key_id": "k2", "public_key": _new_prekey()}, "signature"),
            ({"key_id": "", "public_key": "p", "signature": "s"}, "key_id"),
            ({"key_id": 9, "public_key": "p", "signature": "s"}, "key_id"),
            ({"key_id": "k2", "public_key": 7, "signature": "s"},
             "public_key"),
            ({"key_id": "k2", "public_key": "p", "signature": 8},
             "signature"),
        ]
        for payload, field in cases:
            with self.subTest(field=field):
                with self.assertRaises(ServiceError) as ctx:
                    self.service.add_prekey_verified("d1", payload)
                self.assertEqual(ctx.exception.status_code, 400)
                self.assertEqual(ctx.exception.field, field)

    def test_bad_public_key_400_public_key(self) -> None:
        payload = self._payload(public_key="not-a-key")
        # signature was built for "not-a-key" but public_key validation wins.
        with self.assertRaises(ServiceError) as ctx:
            self.service.add_prekey_verified("d1", payload)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.field, "public_key")

    def test_bad_signature_encoding_400_signature(self) -> None:
        for bad in ("@@@@", "abc", "a" * 88,
                    base64.b64encode(b"\x00" * 63).decode(),
                    base64.b64encode(b"\x00" * 65).decode()):
            payload = {"key_id": "k2", "public_key": _new_prekey(),
                       "signature": bad}
            with self.subTest(bad=bad):
                with self.assertRaises(ServiceError) as ctx:
                    self.service.add_prekey_verified("d1", payload)
                self.assertEqual(ctx.exception.status_code, 400)
                self.assertEqual(ctx.exception.field, "signature")

    def test_signature_must_be_canonical_base64(self) -> None:
        # Standard base64 only: URL-safe alphabet and missing padding refuse.
        good = base64.b64encode(b"\x00" * 64).decode()
        url_safe = good.replace("+", "-").replace("/", "_")
        unpadded = good.rstrip("=")
        for bad in (url_safe, unpadded):
            payload = {"key_id": "k2", "public_key": _new_prekey(),
                       "signature": bad}
            with self.assertRaises(ServiceError) as ctx:
                self.service.add_prekey_verified("d1", payload)
            self.assertEqual(ctx.exception.field, "signature")

    def test_wrong_signer_400_signature(self) -> None:
        other, _ = _new_identity()
        payload = self._payload(private=other)
        with self.assertRaises(ServiceError) as ctx:
            self.service.add_prekey_verified("d1", payload)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.field, "signature")
        # Nothing was added.
        self.assertEqual(self.service.get_device("d1")["prekey_ids"], ["k1"])

    def test_tampered_fields_400_signature(self) -> None:
        public_key = _new_prekey()
        good = self._payload(key_id="k2", public_key=public_key)
        # Signature over k2, request names k3.
        tampered = dict(good, key_id="k3")
        with self.assertRaises(ServiceError) as ctx:
            self.service.add_prekey_verified("d1", tampered)
        self.assertEqual(ctx.exception.field, "signature")

    def test_unknown_device_404(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.add_prekey_verified(
                "ghost", self._payload())
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "device_id")

    def test_revoked_device_409(self) -> None:
        self.service.revoke_device("d1")
        with self.assertRaises(ServiceError) as ctx:
            self.service.add_prekey_verified("d1", self._payload())
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "device_id")

    def test_same_id_changed_key_409_key_id(self) -> None:
        public_key = _new_prekey()
        self.service.add_prekey_verified(
            "d1", self._payload(key_id="k2", public_key=public_key))
        other = _new_prekey()
        with self.assertRaises(ServiceError) as ctx:
            self.service.add_prekey_verified(
                "d1", self._payload(key_id="k2", public_key=other))
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "key_id")

    def test_revoked_key_id_409_key_id(self) -> None:
        public_key = _new_prekey()
        self.service.add_prekey_verified(
            "d1", self._payload(key_id="k2", public_key=public_key))
        self.service.revoke_prekey("d1", "k2")
        # Re-adding the revoked id with the same key and a valid proof is a
        # conflict, not an idempotent replay.
        with self.assertRaises(ServiceError) as ctx:
            self.service.add_prekey_verified(
                "d1", self._payload(key_id="k2", public_key=public_key))
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "key_id")

    def test_invalid_signature_precedes_key_id_conflict(self) -> None:
        # An existing-id conflict must not mask a bad proof: 400/signature.
        public_key = _new_prekey()
        self.service.add_prekey_verified(
            "d1", self._payload(key_id="k2", public_key=public_key))
        other, _ = _new_identity()
        payload = self._payload(key_id="k2", public_key=public_key,
                                private=other)
        with self.assertRaises(ServiceError) as ctx:
            self.service.add_prekey_verified("d1", payload)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.field, "signature")

    def test_non_ed25519_identity_cannot_authorize_400_identity_key(self) -> None:
        # A device whose current identity key carries an X25519 algorithm OID
        # (DER SubjectPublicKeyInfo) cannot verify an Ed25519 proof: raw 32
        # bytes cannot distinguish the two curves, but the DER OID does.
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
            self.service.add_prekey_verified(
                "d2", self._payload())
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.field, "identity_key")
        # Nothing was added to the X25519 device.
        self.assertEqual(self.service.get_device("d2")["prekey_ids"], ["k1"])

    def test_verifies_against_current_identity_after_rotation(self) -> None:
        new_private, new_identity = _new_identity()
        self.service.rotate_identity_key(
            "d1", {"identity_key": new_identity})
        # A proof by the old identity now fails ...
        with self.assertRaises(ServiceError) as ctx:
            self.service.add_prekey_verified(
                "d1", self._payload(key_id="k2"))
        self.assertEqual(ctx.exception.field, "signature")
        # ... while a proof by the current identity succeeds.
        body, status = self.service.add_prekey_verified(
            "d1", self._payload(key_id="k2", private=new_private))
        self.assertEqual(status, 201)
        self.assertEqual(body["key_id"], "k2")

    def test_uses_stored_user_id_in_proof(self) -> None:
        # A signature that names a different user_id does not verify even
        # though the device id matches.
        public_key = _new_prekey()
        signature = _proof(self.private, "attacker", "d1", "k2", public_key)
        with self.assertRaises(ServiceError) as ctx:
            self.service.add_prekey_verified(
                "d1", {"key_id": "k2", "public_key": public_key,
                        "signature": signature})
        self.assertEqual(ctx.exception.field, "signature")

    def test_audit_event_appended_only_on_create(self) -> None:
        page = self.service.list_key_events("d1", 0, 100)
        self.assertEqual([e["type"] for e in page["events"]],
                         ["registered"])
        public_key = _new_prekey()
        payload = self._payload(key_id="k2", public_key=public_key)
        self.service.add_prekey_verified("d1", payload)
        self.service.add_prekey_verified("d1", dict(payload))  # idempotent
        page = self.service.list_key_events("d1", 0, 100)
        self.assertEqual([e["type"] for e in page["events"]],
                         ["registered", "prekey_added"])
        added = page["events"][-1]
        self.assertEqual(added["payload"],
                         {"key_id": "k2", "public_key": public_key})
        self.assertTrue(added["hash"])
        self.assertEqual(added["prev_hash"],
                         page["events"][0]["hash"])

    def test_added_prekey_is_claimable(self) -> None:
        # k1 is consumed by the first claim; the verified-added k2 by the next.
        self.service.add_prekey_verified("d1", self._payload(key_id="k2"))
        first, s1 = self.service.claim_prekey(
            {"recipient_device_id": "d1", "claim_id": "c1"})
        second, s2 = self.service.claim_prekey(
            {"recipient_device_id": "d1", "claim_id": "c2"})
        self.assertEqual(s1, 201)
        self.assertEqual(first["key_id"], "k1")
        self.assertEqual(s2, 201)
        self.assertEqual(second["key_id"], "k2")


class AddPrekeyVerifiedHTTPTest(unittest.TestCase):
    def setUp(self) -> None:
        self.server, self.service = create_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        self.private, self.identity = _new_identity()
        self._register()

    def _register(self) -> None:
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

    def _request(self, method, path, body=None):
        connection = HTTPConnection("127.0.0.1", self.port, timeout=5)
        payload = json.dumps(body) if body is not None else None
        headers = {"Content-Type": "application/json"} if payload else {}
        connection.request(method, path, body=payload, headers=headers)
        response = connection.getresponse()
        data = response.read().decode("utf-8")
        connection.close()
        return response.status, json.loads(data)

    def _payload(self, key_id="k2", public_key=None, private=None):
        if public_key is None:
            public_key = _new_prekey()
        signer = private or self.private
        return {"key_id": key_id, "public_key": public_key,
                "signature": _proof(signer, "u1", "d1", key_id, public_key)}

    def test_201_then_200(self) -> None:
        payload = self._payload(key_id="k9")
        status, body = self._request(
            "POST", "/v1/devices/d1/prekeys/verified", payload)
        self.assertEqual(status, 201)
        self.assertEqual(set(body), {"device_id", "key_id", "public_key"})
        status, body = self._request(
            "POST", "/v1/devices/d1/prekeys/verified", dict(payload))
        self.assertEqual(status, 200)

    def test_non_object_body_400_request_body(self) -> None:
        for raw in ("[]", "null", '"x"', "42"):
            connection = HTTPConnection("127.0.0.1", self.port, timeout=5)
            connection.request(
                "POST", "/v1/devices/d1/prekeys/verified",
                body=raw, headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            body = json.loads(response.read().decode())
            connection.close()
            self.assertEqual(response.status, 400, raw)
            self.assertEqual(body["field"], "request_body")

    def test_bad_signature_400(self) -> None:
        status, body = self._request(
            "POST", "/v1/devices/d1/prekeys/verified",
            {"key_id": "k2", "public_key": _new_prekey(),
             "signature": "not-base64"})
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "signature")

    def test_proof_failure_400_signature(self) -> None:
        other, _ = _new_identity()
        status, body = self._request(
            "POST", "/v1/devices/d1/prekeys/verified",
            self._payload(private=other))
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "signature")

    def test_unknown_device_404(self) -> None:
        status, body = self._request(
            "POST", "/v1/devices/ghost/prekeys/verified", self._payload())
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "device_id")

    def test_revoked_device_409(self) -> None:
        self.service.revoke_device("d1")
        status, body = self._request(
            "POST", "/v1/devices/d1/prekeys/verified", self._payload())
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "device_id")

    def test_ordinary_prekeys_route_still_works(self) -> None:
        status, body = self._request(
            "POST", "/v1/devices/d1/prekeys",
            {"key_id": "plain", "public_key": _new_prekey()})
        self.assertEqual(status, 201)
        self.assertEqual(body["key_id"], "plain")


class AddPrekeyVerifiedCLITest(unittest.TestCase):
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

    def test_success_stdout_zero_and_idempotent(self) -> None:
        public_key = _new_prekey()
        signature = _proof(self.private, "u1", "d1", "kc", public_key)
        first = self._run("add-prekey-verified", "--device-id", "d1",
                          "--key-id", "kc", "--public-key", public_key,
                          "--signature", signature)
        self.assertEqual(first.returncode, 0, first.stderr)
        body = json.loads(first.stdout.strip())
        self.assertEqual(set(body), {"device_id", "key_id", "public_key"})
        self.assertEqual(body["key_id"], "kc")
        self.assertFalse(first.stderr.strip())

        second = self._run("add-prekey-verified", "--device-id", "d1",
                           "--key-id", "kc", "--public-key", public_key,
                           "--signature", signature)
        self.assertEqual(second.returncode, 0, second.stderr)

    def test_failure_stderr_field_nonzero(self) -> None:
        result = self._run("add-prekey-verified", "--device-id", "ghost",
                          "--key-id", "kc", "--public-key", _new_prekey(),
                          "--signature",
                          base64.b64encode(b"\x00" * 64).decode())
        self.assertEqual(result.returncode, 1)
        self.assertFalse(result.stdout.strip())
        body = json.loads(result.stderr.strip())
        self.assertEqual(body["field"], "device_id")

    def test_bad_signature_stderr_signature(self) -> None:
        result = self._run("add-prekey-verified", "--device-id", "d1",
                          "--key-id", "kc", "--public-key", _new_prekey(),
                          "--signature", "not-base64")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(json.loads(result.stderr.strip())["field"],
                         "signature")


class AddPrekeyVerifiedPersistenceTest(unittest.TestCase):
    def test_verified_prekey_and_audit_survive_restart(self) -> None:
        path = tempfile.mktemp(suffix=".json")
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        private, identity = _new_identity()
        added = _new_prekey()

        first = DeviceService()
        attach_persistence(first, path)
        k1 = _new_prekey()
        first.register_verified({
            "user_id": "u1", "device_id": "d1",
            "identity_key": identity,
            "signed_prekeys": [{
                "key_id": "k1", "public_key": k1,
                "signature": _proof(private, "u1", "d1", "k1", k1)}],
        })
        first.add_prekey_verified(
            "d1", {"key_id": "k2", "public_key": added,
                   "signature": _proof(private, "u1", "d1", "k2", added)})

        second = DeviceService()
        attach_persistence(second, path)
        self.assertEqual(second.get_device("d1")["prekey_ids"],
                         ["k1", "k2"])
        events = second.list_key_events("d1", 0, 100)["events"]
        self.assertEqual([e["type"] for e in events],
                         ["registered", "prekey_added"])
        self.assertEqual(events[-1]["payload"],
                         {"key_id": "k2", "public_key": added})

    def test_failure_advances_no_generation_and_leaves_no_event(self) -> None:
        path = tempfile.mktemp(suffix=".json")
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        private, identity = _new_identity()
        k1 = _new_prekey()
        service = DeviceService()
        store = attach_persistence(service, path)
        service.register_verified({
            "user_id": "u1", "device_id": "d1",
            "identity_key": identity,
            "signed_prekeys": [{
                "key_id": "k1", "public_key": k1,
                "signature": _proof(private, "u1", "d1", "k1", k1)}],
        })
        generation_after_register = store.commit_seq
        events_after_register = len(
            service.list_key_events("d1", 0, 100)["events"])

        other, _ = _new_identity()
        with self.assertRaises(ServiceError):
            service.add_prekey_verified(
                "d1", {"key_id": "k2", "public_key": _new_prekey(),
                       "signature": _proof(other, "u1", "d1", "k2",
                                           _new_prekey())})
        # No generation consumed, no event appended, no key added.
        self.assertEqual(store.commit_seq, generation_after_register)
        self.assertEqual(
            len(service.list_key_events("d1", 0, 100)["events"]),
            events_after_register)
        self.assertEqual(service.get_device("d1")["prekey_ids"], ["k1"])

    def test_persist_failure_rolls_back_the_append(self) -> None:
        from e2ee_backend.persistence import PersistenceUnavailable

        path = tempfile.mktemp(suffix=".json")
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        private, identity = _new_identity()
        k1 = _new_prekey()
        service = DeviceService()
        state_store = attach_persistence(service, path)
        service.register_verified({
            "user_id": "u1", "device_id": "d1",
            "identity_key": identity,
            "signed_prekeys": [{
                "key_id": "k1", "public_key": k1,
                "signature": _proof(private, "u1", "d1", "k1", k1)}],
        })

        def fail_save(_pending) -> None:
            raise OSError("simulated disk failure")

        state_store.save = fail_save  # type: ignore[assignment]
        added = _new_prekey()
        # The durable write fails inside the locked transaction: the HTTP
        # layer answers 503/data_file and the in-memory mutation is rolled
        # back to the last committed state.
        with self.assertRaises(PersistenceUnavailable):
            service.add_prekey_verified(
                "d1", {"key_id": "k2", "public_key": added,
                       "signature": _proof(private, "u1", "d1", "k2", added)})
        # The mutation was rolled back: no k2, no appended event.
        self.assertEqual(service.get_device("d1")["prekey_ids"], ["k1"])
        self.assertEqual(
            [e["type"] for e in
             service.list_key_events("d1", 0, 100)["events"]],
            ["registered"])


class AddPrekeyVerifiedConcurrencyTest(unittest.TestCase):
    def test_concurrent_verified_add_and_revoke_linearizes(self) -> None:
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
        added = _new_prekey()
        payload = {"key_id": "k2", "public_key": added,
                   "signature": _proof(private, "u1", "d1", "k2", added)}
        outcomes = []

        def add() -> None:
            try:
                _, status = service.add_prekey_verified("d1", dict(payload))
                outcomes.append(("add", status))
            except ServiceError as error:
                outcomes.append(("add", error.status_code))

        def revoke() -> None:
            service.revoke_device("d1")
            outcomes.append(("revoke", 200))

        threads = [threading.Thread(target=add), threading.Thread(target=revoke)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)

        add_result = next(code for kind, code in outcomes if kind == "add")
        # Exactly one linearized result: 201 (add first) or 409 (revoke first).
        self.assertIn(add_result, (201, 409))
        view = service.get_device("d1")
        if add_result == 201:
            # The device was revoked afterwards; k2 was added then revoked too.
            self.assertNotIn("k2", view["prekey_ids"])
        else:
            self.assertEqual(view["prekey_ids"], [])


if __name__ == "__main__":
    unittest.main()
