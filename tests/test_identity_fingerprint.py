"""Tests for identity fingerprints and explicit fingerprint confirmation.

Covers the crypto normalization, service, HTTP (real loopback socket), CLI
(real subprocess) and persistence layers for:

* GET  /v1/devices/{device_id}/identity-fingerprint
* POST /v1/devices/{device_id}/identity-verifications
"""
import base64
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from http.client import HTTPConnection

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import x25519

from e2ee_backend.crypto import (
    IDENTITY_FINGERPRINT_PREFIX,
    identity_fingerprint,
    is_identity_fingerprint,
)
from e2ee_backend.http_app import create_server
from e2ee_backend.persistence import attach_persistence
from e2ee_backend.service import DeviceService, ServiceError


def _raw_key_b64() -> str:
    key = x25519.X25519PrivateKey.generate().public_key()
    raw = key.public_bytes(serialization.Encoding.Raw,
                           serialization.PublicFormat.Raw)
    return base64.b64encode(raw).decode()


def _key_encodings() -> tuple[str, str, str, str, str]:
    """Return (raw_b64, raw_hex, der_b64, der_hex, pem) of one X25519 key."""
    key = x25519.X25519PrivateKey.generate().public_key()
    raw = key.public_bytes(serialization.Encoding.Raw,
                           serialization.PublicFormat.Raw)
    der = key.public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo)
    pem = key.public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo).decode("ascii")
    return (base64.b64encode(raw).decode(), raw.hex(),
            base64.b64encode(der).decode(), der.hex(), pem)


def _register(service: DeviceService, device_id: str,
              identity_key: str | None = None) -> str:
    key = identity_key or _raw_key_b64()
    service.register({
        "user_id": "u1", "device_id": device_id,
        "identity_key": key,
        "signed_prekeys": [{"key_id": "p1", "public_key": _raw_key_b64()}]})
    return key


class FingerprintCryptoTest(unittest.TestCase):
    def test_same_key_encodings_fingerprint_equal(self) -> None:
        raw_b64, raw_hex, der_b64, der_hex, pem = _key_encodings()
        fingerprints = {
            identity_fingerprint(raw_b64),
            identity_fingerprint(raw_hex),
            identity_fingerprint(der_b64),
            identity_fingerprint(der_hex),
            identity_fingerprint(pem),
        }
        self.assertEqual(len(fingerprints), 1)
        fingerprint = fingerprints.pop()
        der = base64.b64decode(der_b64)
        expected = hashlib.sha256(
            (IDENTITY_FINGERPRINT_PREFIX + "\n").encode("utf-8") + der
        ).hexdigest()
        self.assertEqual(fingerprint, expected)

    def test_fingerprint_shape(self) -> None:
        fingerprint = identity_fingerprint(_raw_key_b64())
        self.assertIsNotNone(fingerprint)
        self.assertTrue(is_identity_fingerprint(fingerprint))
        self.assertFalse(is_identity_fingerprint(fingerprint.upper()))
        self.assertFalse(is_identity_fingerprint(fingerprint[:-1]))
        self.assertFalse(is_identity_fingerprint("g" * 64))
        self.assertFalse(is_identity_fingerprint(None))

    def test_bad_key_is_none(self) -> None:
        self.assertIsNone(identity_fingerprint("not-a-key"))
        self.assertIsNone(identity_fingerprint(""))


class FingerprintServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = DeviceService()
        self.alice_key = _register(self.service, "alice")
        _register(self.service, "bob")

    def _fingerprint(self, device_id: str = "alice",
                     verifier: str = "bob") -> dict:
        return self.service.identity_fingerprint(device_id, verifier)

    def test_initial_view_is_unverified_version_1(self) -> None:
        view = self._fingerprint()
        self.assertEqual(set(view), {
            "identity_key", "fingerprint", "identity_key_version",
            "verification_status", "verification_id"})
        self.assertEqual(view["identity_key"], self.alice_key)
        self.assertEqual(view["identity_key_version"], 1)
        self.assertEqual(view["verification_status"], "unverified")
        self.assertIsNone(view["verification_id"])
        self.assertTrue(is_identity_fingerprint(view["fingerprint"]))

    def test_same_key_rotation_keeps_version_and_fingerprint(self) -> None:
        before = self._fingerprint()
        self.service.rotate_identity_key(
            "alice", {"identity_key": self.alice_key})
        after = self._fingerprint()
        self.assertEqual(after["identity_key_version"], 1)
        self.assertEqual(after["fingerprint"], before["fingerprint"])

    def test_different_key_rotation_increments_version(self) -> None:
        new_key = _raw_key_b64()
        self.service.rotate_identity_key(
            "alice", {"identity_key": new_key})
        view = self._fingerprint()
        self.assertEqual(view["identity_key_version"], 2)
        self.service.rotate_identity_key(
            "alice", {"identity_key": self.alice_key})
        self.assertEqual(self._fingerprint()["identity_key_version"], 3)

    def test_first_confirm_201_replay_200(self) -> None:
        fp = self._fingerprint()["fingerprint"]
        payload = {"verifier_device_id": "bob", "verification_id": "v1",
                   "expected_fingerprint": fp}
        body, code = self.service.confirm_identity("alice", payload)
        self.assertEqual(code, 201)
        self.assertEqual(set(body), {
            "verification_id", "device_id", "verifier_device_id",
            "fingerprint", "verified_at", "active"})
        self.assertTrue(body["active"])
        self.assertEqual(body["device_id"], "alice")
        self.assertEqual(body["verifier_device_id"], "bob")
        self.assertEqual(body["fingerprint"], fp)

        replay, replay_code = self.service.confirm_identity("alice", payload)
        self.assertEqual(replay_code, 200)
        self.assertEqual(replay, body)

        view = self._fingerprint()
        self.assertEqual(view["verification_status"], "verified")
        self.assertEqual(view["verification_id"], "v1")

    def test_second_active_id_for_same_pair_is_409(self) -> None:
        fp = self._fingerprint()["fingerprint"]
        self.service.confirm_identity("alice", {
            "verifier_device_id": "bob", "verification_id": "v1",
            "expected_fingerprint": fp})
        with self.assertRaises(ServiceError) as caught:
            self.service.confirm_identity("alice", {
                "verifier_device_id": "bob", "verification_id": "v2",
                "expected_fingerprint": fp})
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "verification_id")

    def test_same_id_rebound_to_other_pair_is_409(self) -> None:
        _register(self.service, "carol")
        fp = self._fingerprint()["fingerprint"]
        self.service.confirm_identity("alice", {
            "verifier_device_id": "bob", "verification_id": "v1",
            "expected_fingerprint": fp})
        with self.assertRaises(ServiceError) as caught:
            self.service.confirm_identity("alice", {
                "verifier_device_id": "carol", "verification_id": "v1",
                "expected_fingerprint": fp})
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "verification_id")

    def test_distinct_verifiers_are_independent(self) -> None:
        _register(self.service, "carol")
        fp = self._fingerprint()["fingerprint"]
        body_bob, code_bob = self.service.confirm_identity("alice", {
            "verifier_device_id": "bob", "verification_id": "vb",
            "expected_fingerprint": fp})
        body_carol, code_carol = self.service.confirm_identity("alice", {
            "verifier_device_id": "carol", "verification_id": "vc",
            "expected_fingerprint": fp})
        self.assertEqual((code_bob, code_carol), (201, 201))
        self.assertNotEqual(body_bob["verification_id"],
                            body_carol["verification_id"])
        self.assertEqual(
            self._fingerprint(verifier="bob")["verification_id"], "vb")
        self.assertEqual(
            self._fingerprint(verifier="carol")["verification_id"], "vc")

    def test_rotation_marks_changed_without_auto_follow(self) -> None:
        fp1 = self._fingerprint()["fingerprint"]
        self.service.confirm_identity("alice", {
            "verifier_device_id": "bob", "verification_id": "v1",
            "expected_fingerprint": fp1})
        self.service.rotate_identity_key(
            "alice", {"identity_key": _raw_key_b64()})
        view = self._fingerprint()
        self.assertEqual(view["verification_status"], "changed")
        self.assertEqual(view["verification_id"], "v1")
        self.assertNotEqual(view["fingerprint"], fp1)

    def test_new_id_confirms_new_fingerprint_old_id_conflicts(self) -> None:
        fp1 = self._fingerprint()["fingerprint"]
        self.service.confirm_identity("alice", {
            "verifier_device_id": "bob", "verification_id": "v1",
            "expected_fingerprint": fp1})
        self.service.rotate_identity_key(
            "alice", {"identity_key": _raw_key_b64()})
        fp2 = self._fingerprint()["fingerprint"]

        # Reusing the old id even with the new fingerprint still conflicts.
        with self.assertRaises(ServiceError) as caught:
            self.service.confirm_identity("alice", {
                "verifier_device_id": "bob", "verification_id": "v1",
                "expected_fingerprint": fp2})
        self.assertEqual((caught.exception.status_code,
                          caught.exception.field), (409, "verification_id"))

        # A stale or invented fingerprint cannot be confirmed either.
        with self.assertRaises(ServiceError) as caught:
            self.service.confirm_identity("alice", {
                "verifier_device_id": "bob", "verification_id": "v2",
                "expected_fingerprint": fp1})
        self.assertEqual((caught.exception.status_code,
                          caught.exception.field),
                         (409, "expected_fingerprint"))

        body, code = self.service.confirm_identity("alice", {
            "verifier_device_id": "bob", "verification_id": "v2",
            "expected_fingerprint": fp2})
        self.assertEqual(code, 201)
        self.assertTrue(body["active"])
        view = self._fingerprint()
        self.assertEqual(view["verification_status"], "verified")
        self.assertEqual(view["verification_id"], "v2")

    def test_get_lookup_failures(self) -> None:
        with self.assertRaises(ServiceError) as caught:
            self.service.identity_fingerprint("ghost", "bob")
        self.assertEqual((caught.exception.status_code,
                          caught.exception.field), (404, "device_id"))
        with self.assertRaises(ServiceError) as caught:
            self.service.identity_fingerprint("alice", "ghost")
        self.assertEqual((caught.exception.status_code,
                          caught.exception.field),
                         (404, "verifier_device_id"))
        with self.assertRaises(ServiceError) as caught:
            self.service.identity_fingerprint("alice", "alice")
        self.assertEqual((caught.exception.status_code,
                          caught.exception.field),
                         (400, "verifier_device_id"))
        self.service.revoke_device("bob")
        with self.assertRaises(ServiceError) as caught:
            self.service.identity_fingerprint("alice", "bob")
        self.assertEqual((caught.exception.status_code,
                          caught.exception.field),
                         (404, "verifier_device_id"))

    def test_confirm_validation_failures_400(self) -> None:
        fp = self._fingerprint()["fingerprint"]
        valid = {"verifier_device_id": "bob", "verification_id": "v1",
                 "expected_fingerprint": fp}
        for name in ("verifier_device_id", "verification_id",
                     "expected_fingerprint"):
            missing = dict(valid)
            del missing[name]
            with self.assertRaises(ServiceError) as caught:
                self.service.confirm_identity("alice", missing)
            self.assertEqual((caught.exception.status_code,
                              caught.exception.field), (400, name))
            empty = dict(valid, **{name: ""})
            with self.assertRaises(ServiceError) as caught:
                self.service.confirm_identity("alice", empty)
            self.assertEqual((caught.exception.status_code,
                              caught.exception.field), (400, name))
        for bad in ("abc", "A" * 64, "g" * 64, 123, None):
            with self.assertRaises(ServiceError) as caught:
                self.service.confirm_identity(
                    "alice", dict(valid, expected_fingerprint=bad))
            self.assertEqual((caught.exception.status_code,
                              caught.exception.field),
                             (400, "expected_fingerprint"))
        with self.assertRaises(ServiceError) as caught:
            self.service.confirm_identity(
                "alice", dict(valid, verifier_device_id="alice"))
        self.assertEqual((caught.exception.status_code,
                          caught.exception.field),
                         (400, "verifier_device_id"))
        with self.assertRaises(ServiceError) as caught:
            self.service.confirm_identity("alice", "not-an-object")
        self.assertEqual((caught.exception.status_code,
                          caught.exception.field), (400, "request_body"))

    def test_confirm_unknown_and_revoked_404(self) -> None:
        fp = self._fingerprint()["fingerprint"]
        with self.assertRaises(ServiceError) as caught:
            self.service.confirm_identity("ghost", {
                "verifier_device_id": "bob", "verification_id": "v1",
                "expected_fingerprint": fp})
        self.assertEqual((caught.exception.status_code,
                          caught.exception.field), (404, "device_id"))
        with self.assertRaises(ServiceError) as caught:
            self.service.confirm_identity("alice", {
                "verifier_device_id": "ghost", "verification_id": "v1",
                "expected_fingerprint": fp})
        self.assertEqual((caught.exception.status_code,
                          caught.exception.field),
                         (404, "verifier_device_id"))


class FingerprintHTTPTest(unittest.TestCase):
    def setUp(self) -> None:
        self.server, self.service = create_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        self.alice_key = _register(self.service, "alice")
        _register(self.service, "bob")

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _request(self, method: str, target: str, body: object = None):
        connection = HTTPConnection("127.0.0.1", self.port, timeout=5)
        payload = json.dumps(body) if body is not None else None
        headers = {"Content-Type": "application/json"} if payload else {}
        connection.request(method, target, body=payload, headers=headers)
        response = connection.getresponse()
        data = response.read().decode("utf-8")
        connection.close()
        return response.status, json.loads(data)

    def _get_fp(self) -> str:
        status, body = self._request(
            "GET",
            "/v1/devices/alice/identity-fingerprint?verifier_device_id=bob")
        self.assertEqual(status, 200)
        return body["fingerprint"]

    def test_get_fingerprint_contract(self) -> None:
        status, body = self._request(
            "GET",
            "/v1/devices/alice/identity-fingerprint?verifier_device_id=bob")
        self.assertEqual(status, 200)
        self.assertEqual(set(body), {
            "identity_key", "fingerprint", "identity_key_version",
            "verification_status", "verification_id"})
        self.assertEqual(body["identity_key"], self.alice_key)
        self.assertEqual(body["identity_key_version"], 1)
        self.assertEqual(body["verification_status"], "unverified")
        self.assertIsNone(body["verification_id"])

    def test_get_query_parameter_failures(self) -> None:
        targets = [
            "/v1/devices/alice/identity-fingerprint",
            "/v1/devices/alice/identity-fingerprint?",
            "/v1/devices/alice/identity-fingerprint?verifier_device_id=",
            "/v1/devices/alice/identity-fingerprint?verifier_device_id=bob"
            "&verifier_device_id=bob",
            "/v1/devices/alice/identity-fingerprint?verifier_device_id=alice",
        ]
        for target in targets:
            status, body = self._request("GET", target)
            self.assertEqual(status, 400, target)
            self.assertEqual(body["field"], "verifier_device_id", target)

    def test_get_lookup_failures(self) -> None:
        status, body = self._request(
            "GET",
            "/v1/devices/ghost/identity-fingerprint?verifier_device_id=bob")
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "device_id")
        status, body = self._request(
            "GET",
            "/v1/devices/alice/identity-fingerprint?verifier_device_id=ghost")
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "verifier_device_id")

    def test_verify_201_200_and_pair_conflict_409(self) -> None:
        fp = self._get_fp()
        payload = {"verifier_device_id": "bob", "verification_id": "v1",
                   "expected_fingerprint": fp}
        status, body = self._request(
            "POST", "/v1/devices/alice/identity-verifications", payload)
        self.assertEqual(status, 201)
        self.assertTrue(body["active"])
        status, body = self._request(
            "POST", "/v1/devices/alice/identity-verifications", payload)
        self.assertEqual(status, 200)

        status, body = self._request(
            "POST", "/v1/devices/alice/identity-verifications",
            dict(payload, verification_id="v2"))
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "verification_id")

    def test_verify_body_validation_400(self) -> None:
        fp = self._get_fp()
        cases = [
            ({}, "verifier_device_id"),
            ({"verifier_device_id": "bob"}, "verification_id"),
            ({"verifier_device_id": "bob", "verification_id": "v1"},
             "expected_fingerprint"),
            ({"verifier_device_id": "bob", "verification_id": "v1",
              "expected_fingerprint": "aabb"}, "expected_fingerprint"),
            ({"verifier_device_id": "bob", "verification_id": "v1",
              "expected_fingerprint": "A" * 64}, "expected_fingerprint"),
            ({"verifier_device_id": "alice", "verification_id": "v1",
              "expected_fingerprint": fp}, "verifier_device_id"),
        ]
        for payload, field in cases:
            status, body = self._request(
                "POST", "/v1/devices/alice/identity-verifications", payload)
            self.assertEqual(status, 400, payload)
            self.assertEqual(body["field"], field, payload)

    def test_rotation_changed_and_supersede_flow(self) -> None:
        fp1 = self._get_fp()
        self._request("POST", "/v1/devices/alice/identity-verifications",
                      {"verifier_device_id": "bob", "verification_id": "v1",
                       "expected_fingerprint": fp1})
        new_key = _raw_key_b64()
        status, body = self._request(
            "POST", "/v1/devices/alice/identity-key/rotate",
            {"identity_key": new_key})
        self.assertEqual(status, 200)
        status, body = self._request(
            "GET",
            "/v1/devices/alice/identity-fingerprint?verifier_device_id=bob")
        self.assertEqual(status, 200)
        self.assertEqual(body["identity_key_version"], 2)
        self.assertEqual(body["verification_status"], "changed")
        self.assertEqual(body["verification_id"], "v1")
        fp2 = body["fingerprint"]

        status, body = self._request(
            "POST", "/v1/devices/alice/identity-verifications",
            {"verifier_device_id": "bob", "verification_id": "v1",
             "expected_fingerprint": fp2})
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "verification_id")

        status, body = self._request(
            "POST", "/v1/devices/alice/identity-verifications",
            {"verifier_device_id": "bob", "verification_id": "v2",
             "expected_fingerprint": fp2})
        self.assertEqual(status, 201)
        status, body = self._request(
            "GET",
            "/v1/devices/alice/identity-fingerprint?verifier_device_id=bob")
        self.assertEqual(body["verification_status"], "verified")
        self.assertEqual(body["verification_id"], "v2")

    def test_revoked_subject_404(self) -> None:
        self.service.revoke_device("alice")
        status, body = self._request(
            "GET",
            "/v1/devices/alice/identity-fingerprint?verifier_device_id=bob")
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "device_id")


class FingerprintCLITest(unittest.TestCase):
    def setUp(self) -> None:
        self.server, service = create_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        _register(service, "alice")
        _register(service, "bob")
        self.service = service

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _run(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "e2ee_backend", "--base-url",
             self.base_url, *arguments],
            capture_output=True, text=True, timeout=15)

    def test_fingerprint_command(self) -> None:
        result = self._run("fingerprint", "alice",
                           "--verifier-device-id", "bob")
        self.assertEqual(result.returncode, 0, result.stderr)
        line = result.stdout.strip()
        self.assertEqual(line.count("\n"), 0)
        body = json.loads(line)
        self.assertEqual(set(body), {
            "identity_key", "fingerprint", "identity_key_version",
            "verification_status", "verification_id"})
        self.assertEqual(body["verification_status"], "unverified")

    def test_fingerprint_command_failure(self) -> None:
        result = self._run("fingerprint", "ghost",
                           "--verifier-device-id", "bob")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(json.loads(result.stderr.strip())["field"],
                         "device_id")

    def test_verify_identity_command_201_200(self) -> None:
        fp = self.service.identity_fingerprint("alice", "bob")["fingerprint"]
        first = self._run(
            "verify-identity", "--device-id", "alice",
            "--verifier-device-id", "bob", "--verification-id", "v1",
            "--expected-fingerprint", fp)
        self.assertEqual(first.returncode, 0, first.stderr)
        body = json.loads(first.stdout.strip())
        self.assertEqual(body["verification_id"], "v1")
        self.assertTrue(body["active"])

        replay = self._run(
            "verify-identity", "--device-id", "alice",
            "--verifier-device-id", "bob", "--verification-id", "v1",
            "--expected-fingerprint", fp)
        self.assertEqual(replay.returncode, 0, replay.stderr)

        conflict = self._run(
            "verify-identity", "--device-id", "alice",
            "--verifier-device-id", "bob", "--verification-id", "v2",
            "--expected-fingerprint", fp)
        self.assertEqual(conflict.returncode, 1)
        self.assertEqual(json.loads(conflict.stderr.strip())["field"],
                         "verification_id")


class FingerprintPersistenceTest(unittest.TestCase):
    def _path(self) -> str:
        path = tempfile.mktemp(suffix=".json")
        self.addCleanup(
            lambda: os.path.exists(path) and os.unlink(path))
        sidecar = path + ".integrity"
        self.addCleanup(
            lambda: os.path.exists(sidecar) and os.unlink(sidecar))
        return path

    def test_verification_and_version_survive_restart(self) -> None:
        path = self._path()
        first = DeviceService()
        attach_persistence(first, path)
        alice_key = _register(first, "alice")
        _register(first, "bob")
        fp1 = first.identity_fingerprint("alice", "bob")["fingerprint"]
        body, code = first.confirm_identity("alice", {
            "verifier_device_id": "bob", "verification_id": "v1",
            "expected_fingerprint": fp1})
        self.assertEqual(code, 201)
        new_key = _raw_key_b64()
        first.rotate_identity_key("alice", {"identity_key": new_key})
        changed = first.identity_fingerprint("alice", "bob")
        self.assertEqual(changed["identity_key_version"], 2)
        self.assertEqual(changed["verification_status"], "changed")

        second = DeviceService()
        attach_persistence(second, path)
        device = second.store.find_by_device_id("alice")
        self.assertEqual(device.identity_key, new_key)
        self.assertEqual(device.identity_key_version, 2)
        self.assertNotEqual(device.identity_key, alice_key)
        view = second.identity_fingerprint("alice", "bob")
        self.assertEqual(view, changed)

        # Idempotent after restart: exact reply stays 200.
        _, replay_code = second.confirm_identity("alice", {
            "verifier_device_id": "bob", "verification_id": "v1",
            "expected_fingerprint": fp1})
        self.assertEqual(replay_code, 200)
        # The old id cannot be reused to confirm the new fingerprint.
        with self.assertRaises(ServiceError) as caught:
            second.confirm_identity("alice", {
                "verifier_device_id": "bob", "verification_id": "v1",
                "expected_fingerprint": changed["fingerprint"]})
        self.assertEqual((caught.exception.status_code,
                          caught.exception.field),
                         (409, "verification_id"))
        # A new id confirms the current fingerprint after restart.
        _, supersede_code = second.confirm_identity("alice", {
            "verifier_device_id": "bob", "verification_id": "v2",
            "expected_fingerprint": changed["fingerprint"]})
        self.assertEqual(supersede_code, 201)
        self.assertEqual(
            second.identity_fingerprint("alice", "bob")[
                "verification_status"],
            "verified")

    def test_legacy_document_reads_as_unverified_version_1(self) -> None:
        path = self._path()
        key = _raw_key_b64()
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({
                "version": 1,
                "devices": [{
                    "user_id": "u1", "device_id": "alice",
                    "identity_key": key,
                    "registered_at": "2026-01-01T00:00:00+00:00",
                    "revoked": False,
                    "prekeys": []}, {
                    "user_id": "u1", "device_id": "bob",
                    "identity_key": _raw_key_b64(),
                    "registered_at": "2026-01-01T00:00:00+00:00",
                    "revoked": False,
                    "prekeys": []}],
                "sessions": [], "messages": {}, "delivery": [],
            }, handle)
        service = DeviceService()
        attach_persistence(service, path)
        device = service.store.find_by_device_id("alice")
        self.assertEqual(device.identity_key_version, 1)
        view = service.identity_fingerprint("alice", "bob")
        self.assertEqual(view["identity_key_version"], 1)
        self.assertEqual(view["verification_status"], "unverified")
        self.assertIsNone(view["verification_id"])


if __name__ == "__main__":
    unittest.main()
