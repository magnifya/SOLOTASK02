"""Tests for identity fingerprints and explicit identity verification.

Covers the service, HTTP (real loopback socket), CLI (real subprocess) and
persistence layers for:

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
from cryptography.hazmat.primitives.asymmetric import ed25519, x25519

from e2ee_backend.crypto import IDENTITY_FINGERPRINT_PREFIX
from e2ee_backend.http_app import create_server
from e2ee_backend.persistence import StateFileError, attach_persistence
from e2ee_backend.service import DeviceService, ServiceError


def _x25519_key() -> str:
    key = x25519.X25519PrivateKey.generate().public_key()
    raw = key.public_bytes(serialization.Encoding.Raw,
                           serialization.PublicFormat.Raw)
    return base64.b64encode(raw).decode()


def _ed25519_key() -> str:
    key = ed25519.Ed25519PrivateKey.generate().public_key()
    raw = key.public_bytes(serialization.Encoding.Raw,
                           serialization.PublicFormat.Raw)
    return base64.b64encode(raw).decode()


def _all_encodings(raw_key_b64: str):
    """PEM/DER-base64/DER-hex/raw-base64/raw-hex spellings of one Ed25519 key."""
    raw = base64.b64decode(raw_key_b64)
    public = ed25519.Ed25519PublicKey.from_public_bytes(raw)
    der = public.public_bytes(serialization.Encoding.DER,
                              serialization.PublicFormat.SubjectPublicKeyInfo)
    pem = public.public_bytes(serialization.Encoding.PEM,
                              serialization.PublicFormat.SubjectPublicKeyInfo)
    return {
        "raw_b64": raw_key_b64,
        "raw_hex": raw.hex(),
        "der_b64": base64.b64encode(der).decode(),
        "der_hex": der.hex(),
        "pem": pem.decode("ascii"),
    }


def _expected_fingerprint(raw_key_b64: str) -> str:
    raw = base64.b64decode(raw_key_b64)
    return hashlib.sha256(
        (IDENTITY_FINGERPRINT_PREFIX + "\n").encode("utf-8") + raw).hexdigest()


def _register(service: DeviceService, device_id: str,
              identity_key: str | None = None,
              user_id: str = "u1") -> str:
    identity_key = identity_key or _x25519_key()
    service.register({
        "user_id": user_id, "device_id": device_id,
        "identity_key": identity_key,
        "signed_prekeys": [
            {"key_id": "k1", "public_key": _x25519_key()}]})
    return identity_key


class FingerprintCryptoTest(unittest.TestCase):
    def test_encodings_of_same_key_fingerprint_identically(self) -> None:
        encodings = _all_encodings(_ed25519_key())
        from e2ee_backend.crypto import identity_fingerprint
        fingerprints = {identity_fingerprint(value)
                        for value in encodings.values()}
        self.assertEqual(len(fingerprints), 1)
        fingerprint = fingerprints.pop()
        self.assertEqual(fingerprint, _expected_fingerprint(encodings["raw_b64"]))

    def test_fingerprint_is_64_lowercase_hex(self) -> None:
        from e2ee_backend.crypto import identity_fingerprint
        fingerprint = identity_fingerprint(_x25519_key())
        self.assertRegex(fingerprint, r"^[0-9a-f]{64}$")

    def test_fingerprint_distinct_keys_distinct_values(self) -> None:
        from e2ee_backend.crypto import identity_fingerprint
        self.assertNotEqual(identity_fingerprint(_x25519_key()),
                            identity_fingerprint(_x25519_key()))

    def test_bad_key_is_none(self) -> None:
        from e2ee_backend.crypto import identity_fingerprint
        self.assertIsNone(identity_fingerprint("not-a-key"))


class FingerprintServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = DeviceService()
        self.peer_key = _register(self.service, "peer")
        _register(self.service, "verifier")

    def test_initial_view_is_unverified_version_one(self) -> None:
        view = self.service.identity_fingerprint_view("peer", "verifier")
        self.assertEqual(set(view), {
            "identity_key", "fingerprint", "identity_key_version",
            "verification_status", "verification_id"})
        self.assertEqual(view["identity_key"], self.peer_key)
        self.assertEqual(view["fingerprint"],
                         _expected_fingerprint(self.peer_key))
        self.assertEqual(view["identity_key_version"], 1)
        self.assertEqual(view["verification_status"], "unverified")
        self.assertIsNone(view["verification_id"])

    def test_same_key_rotation_keeps_version(self) -> None:
        self.service.rotate_identity_key(
            "peer", {"identity_key": self.peer_key})
        view = self.service.identity_fingerprint_view("peer", "verifier")
        self.assertEqual(view["identity_key_version"], 1)

    def test_different_key_rotation_increments_version(self) -> None:
        new_key = _x25519_key()
        self.service.rotate_identity_key("peer",
                                         {"identity_key": new_key})
        view = self.service.identity_fingerprint_view("peer", "verifier")
        self.assertEqual(view["identity_key_version"], 2)
        self.assertEqual(view["fingerprint"],
                         _expected_fingerprint(new_key))
        self.service.rotate_identity_key("peer",
                                         {"identity_key": new_key})
        view = self.service.identity_fingerprint_view("peer", "verifier")
        self.assertEqual(view["identity_key_version"], 2)
        third_key = _x25519_key()
        self.service.rotate_identity_key("peer",
                                         {"identity_key": third_key})
        view = self.service.identity_fingerprint_view("peer", "verifier")
        self.assertEqual(view["identity_key_version"], 3)

    def test_unknown_or_revoked_peer_is_404_device_id(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.identity_fingerprint_view("ghost", "verifier")
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "device_id")
        self.service.revoke_device("peer")
        with self.assertRaises(ServiceError) as ctx:
            self.service.identity_fingerprint_view("peer", "verifier")
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "device_id")

    def test_unknown_or_revoked_verifier_is_404_verifier(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.identity_fingerprint_view("peer", "ghost")
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "verifier_device_id")
        self.service.revoke_device("verifier")
        with self.assertRaises(ServiceError) as ctx:
            self.service.identity_fingerprint_view("peer", "verifier")
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "verifier_device_id")


    def test_corrupt_stored_identity_key_is_400_identity_key(self) -> None:
        # Fabricate a registered device whose stored key does not parse.
        from e2ee_backend.models import Device
        device = Device(user_id="u1", device_id="broken",
                        identity_key="not-a-public-key",
                        prekeys=[])
        self.assertTrue(self.service.store.add_device(device))
        with self.assertRaises(ServiceError) as ctx:
            self.service.identity_fingerprint_view("broken", "verifier")
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.field, "identity_key")
        with self.assertRaises(ServiceError) as ctx:
            self.service.verify_identity("broken", {
                "verifier_device_id": "verifier",
                "verification_id": "v1",
                "expected_fingerprint": "0" * 64})
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.field, "identity_key")


class VerificationServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = DeviceService()
        self.peer_key = _register(self.service, "peer")
        _register(self.service, "verifier")
        self.fingerprint = _expected_fingerprint(self.peer_key)

    def _confirm(self, verification_id="v1", fingerprint=None,
                 verifier="verifier"):
        return self.service.verify_identity("peer", {
            "verifier_device_id": verifier,
            "verification_id": verification_id,
            "expected_fingerprint": fingerprint or self.fingerprint})

    def test_first_match_writes_201_verified(self) -> None:
        body, status = self._confirm()
        self.assertEqual(status, 201)
        self.assertEqual(body["verification_status"], "verified")
        self.assertEqual(body["verification_id"], "v1")
        self.assertEqual(body["fingerprint"], self.fingerprint)
        self.assertEqual(body["identity_key_version"], 1)
        view = self.service.identity_fingerprint_view("peer", "verifier")
        self.assertEqual(view["verification_status"], "verified")
        self.assertEqual(view["verification_id"], "v1")

    def test_same_id_same_subject_same_fingerprint_replay_200(self) -> None:
        first, status_one = self._confirm()
        self.assertEqual(status_one, 201)
        second, status_two = self._confirm()
        self.assertEqual(status_two, 200)
        self.assertEqual(second, first)

    def test_same_id_rebound_to_other_pair_is_409(self) -> None:
        _register(self.service, "peer2")
        _register(self.service, "verifier2")
        self._confirm(verification_id="v1")
        with self.assertRaises(ServiceError) as ctx:
            self.service.verify_identity("peer2", {
                "verifier_device_id": "verifier2",
                "verification_id": "v1",
                "expected_fingerprint": _expected_fingerprint(
                    self.service.get_device("peer2")["identity_key"])})
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "verification_id")
        with self.assertRaises(ServiceError) as ctx:
            self._confirm(verification_id="v1", verifier="verifier2")
        self.assertEqual(ctx.exception.field, "verification_id")

    def test_pair_already_active_under_other_id_is_409(self) -> None:
        self._confirm(verification_id="v1")
        with self.assertRaises(ServiceError) as ctx:
            self._confirm(verification_id="v2")
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "verification_id")

    def test_status_becomes_changed_after_rotation(self) -> None:
        self._confirm(verification_id="v1")
        new_key = _x25519_key()
        self.service.rotate_identity_key("peer",
                                         {"identity_key": new_key})
        view = self.service.identity_fingerprint_view("peer", "verifier")
        self.assertEqual(view["verification_status"], "changed")
        self.assertEqual(view["verification_id"], "v1")
        self.assertEqual(view["identity_key_version"], 2)

    def test_changed_old_id_reuse_is_409_even_matching_current(self) -> None:
        self._confirm(verification_id="v1")
        new_key = _x25519_key()
        self.service.rotate_identity_key("peer",
                                         {"identity_key": new_key})
        new_fp = _expected_fingerprint(new_key)
        # Re-confirming the current fingerprint with the OLD id is a conflict.
        with self.assertRaises(ServiceError) as ctx:
            self._confirm(verification_id="v1", fingerprint=new_fp)
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "verification_id")
        # Still changed: nothing was written.
        view = self.service.identity_fingerprint_view("peer", "verifier")
        self.assertEqual(view["verification_status"], "changed")

    def test_changed_new_id_confirms_current_fingerprint(self) -> None:
        self._confirm(verification_id="v1")
        new_key = _x25519_key()
        self.service.rotate_identity_key("peer",
                                         {"identity_key": new_key})
        new_fp = _expected_fingerprint(new_key)
        body, status = self._confirm(verification_id="v2",
                                     fingerprint=new_fp)
        self.assertEqual(status, 201)
        self.assertEqual(body["verification_status"], "verified")
        self.assertEqual(body["verification_id"], "v2")
        self.assertEqual(body["fingerprint"], new_fp)
        # The new id is replayable; the old one can never succeed again.
        _, replay_status = self._confirm(verification_id="v2",
                                         fingerprint=new_fp)
        self.assertEqual(replay_status, 200)
        with self.assertRaises(ServiceError) as ctx:
            self._confirm(verification_id="v1", fingerprint=new_fp)
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "verification_id")

    def test_expected_fingerprint_mismatch_is_409_expected_field(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self._confirm(verification_id="v9", fingerprint="0" * 64)
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "expected_fingerprint")
        # A rejected confirmation leaves the pair unverified and id free:
        # the same id succeeds once it carries the right fingerprint.
        body, status = self._confirm(verification_id="v9")
        self.assertEqual(status, 201)
        self.assertEqual(body["verification_status"], "verified")

    def test_verification_input_validation_400_fields(self) -> None:
        valid = {"verifier_device_id": "verifier",
                 "verification_id": "v1",
                 "expected_fingerprint": self.fingerprint}
        bad_payloads = [
            ({}, None),
            ({"verification_id": "v1",
              "expected_fingerprint": self.fingerprint},
             "verifier_device_id"),
            ({"verifier_device_id": "verifier",
              "expected_fingerprint": self.fingerprint},
             "verification_id"),
            ({"verifier_device_id": "verifier",
              "verification_id": "v1"},
             "expected_fingerprint"),
            ({"verifier_device_id": "",
              "verification_id": "v1",
              "expected_fingerprint": self.fingerprint},
             "verifier_device_id"),
            ({"verifier_device_id": "verifier",
              "verification_id": "",
              "expected_fingerprint": self.fingerprint},
             "verification_id"),
            ({"verifier_device_id": "verifier",
              "verification_id": "v1",
              "expected_fingerprint": ""},
             "expected_fingerprint"),
            ({"verifier_device_id": "verifier",
              "verification_id": "v1",
              "expected_fingerprint": "A" * 64},
             "expected_fingerprint"),
            ({"verifier_device_id": "verifier",
              "verification_id": "v1",
              "expected_fingerprint": "0" * 63},
             "expected_fingerprint"),
            ({"verifier_device_id": "verifier",
              "verification_id": "v1",
              "expected_fingerprint": "g" * 64},
             "expected_fingerprint"),
            ({"verifier_device_id": 7,
              "verification_id": "v1",
              "expected_fingerprint": self.fingerprint},
             "verifier_device_id"),
        ]
        for payload, field in bad_payloads:
            with self.subTest(payload=payload):
                with self.assertRaises(ServiceError) as ctx:
                    self.service.verify_identity("peer", payload)
                self.assertEqual(ctx.exception.status_code, 400)
                if field is not None:
                    self.assertEqual(ctx.exception.field, field)

    def test_verifier_same_as_peer_is_400_verifier_field(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.verify_identity("peer", {
                "verifier_device_id": "peer",
                "verification_id": "v1",
                "expected_fingerprint": self.fingerprint})
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.field, "verifier_device_id")

    def test_verification_unknown_revoked_404_fields(self) -> None:
        payload = {"verifier_device_id": "verifier",
                   "verification_id": "v1",
                   "expected_fingerprint": self.fingerprint}
        with self.assertRaises(ServiceError) as ctx:
            self.service.verify_identity("ghost", payload)
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "device_id")
        with self.assertRaises(ServiceError) as ctx:
            self.service.verify_identity(
                "peer", {**payload, "verifier_device_id": "ghost"})
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "verifier_device_id")
        self.service.revoke_device("peer")
        with self.assertRaises(ServiceError) as ctx:
            self.service.verify_identity("peer", payload)
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "device_id")

    def test_independent_verifier_pairs_do_not_conflict(self) -> None:
        _register(self.service, "verifier2")
        body1, status1 = self._confirm(verification_id="v1",
                                       verifier="verifier")
        body2, status2 = self._confirm(verification_id="v2",
                                       verifier="verifier2")
        self.assertEqual((status1, status2), (201, 201))
        self.assertEqual(body1["verification_id"], "v1")
        self.assertEqual(body2["verification_id"], "v2")
        self.assertEqual(
            self.service.identity_fingerprint_view(
                "peer", "verifier")["verification_status"], "verified")
        self.assertEqual(
            self.service.identity_fingerprint_view(
                "peer", "verifier2")["verification_status"], "verified")


class IdentityHTTPTest(unittest.TestCase):
    def setUp(self) -> None:
        self.server, self.service = create_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        self.peer_key = _register(self.service, "peer")
        _register(self.service, "verifier")
        self.fingerprint = _expected_fingerprint(self.peer_key)

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _request(self, method: str, path: str, body: object = None):
        connection = HTTPConnection("127.0.0.1", self.port, timeout=5)
        payload = json.dumps(body) if body is not None else None
        headers = {"Content-Type": "application/json"} if payload else {}
        connection.request(method, path, body=payload, headers=headers)
        response = connection.getresponse()
        data = response.read().decode("utf-8")
        connection.close()
        return response.status, json.loads(data)

    def test_get_fingerprint_ok(self) -> None:
        status, body = self._request(
            "GET", "/v1/devices/peer/identity-fingerprint"
                   "?verifier_device_id=verifier")
        self.assertEqual(status, 200)
        self.assertEqual(body["identity_key"], self.peer_key)
        self.assertEqual(body["fingerprint"], self.fingerprint)
        self.assertEqual(body["identity_key_version"], 1)
        self.assertEqual(body["verification_status"], "unverified")
        self.assertIsNone(body["verification_id"])

    def test_get_missing_param_400(self) -> None:
        status, body = self._request(
            "GET", "/v1/devices/peer/identity-fingerprint")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "verifier_device_id")

    def test_get_empty_param_400(self) -> None:
        status, body = self._request(
            "GET", "/v1/devices/peer/identity-fingerprint"
                   "?verifier_device_id=")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "verifier_device_id")

    def test_get_duplicate_param_400(self) -> None:
        status, body = self._request(
            "GET", "/v1/devices/peer/identity-fingerprint"
                   "?verifier_device_id=verifier&verifier_device_id=verifier")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "verifier_device_id")

    def test_get_same_device_400(self) -> None:
        status, body = self._request(
            "GET", "/v1/devices/peer/identity-fingerprint"
                   "?verifier_device_id=peer")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "verifier_device_id")

    def test_get_unknown_peer_404(self) -> None:
        status, body = self._request(
            "GET", "/v1/devices/ghost/identity-fingerprint"
                   "?verifier_device_id=verifier")
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "device_id")

    def test_get_unknown_verifier_404(self) -> None:
        status, body = self._request(
            "GET", "/v1/devices/peer/identity-fingerprint"
                   "?verifier_device_id=ghost")
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "verifier_device_id")

    def test_get_revoked_peer_and_verifier_404(self) -> None:
        self.service.revoke_device("verifier")
        status, body = self._request(
            "GET", "/v1/devices/peer/identity-fingerprint"
                   "?verifier_device_id=verifier")
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "verifier_device_id")
        self.service.revoke_device("peer")
        status, body = self._request(
            "GET", "/v1/devices/peer/identity-fingerprint"
                   "?verifier_device_id=verifier")
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "device_id")

    def test_post_201_200_then_changed_201_old_id_409(self) -> None:
        payload = {"verifier_device_id": "verifier",
                   "verification_id": "v1",
                   "expected_fingerprint": self.fingerprint}
        status, body = self._request(
            "POST", "/v1/devices/peer/identity-verifications", payload)
        self.assertEqual(status, 201)
        self.assertEqual(body["verification_status"], "verified")
        status, body = self._request(
            "POST", "/v1/devices/peer/identity-verifications", payload)
        self.assertEqual(status, 200)
        self.assertEqual(body["verification_id"], "v1")

        status, body = self._request(
            "GET", "/v1/devices/peer/identity-fingerprint"
                   "?verifier_device_id=verifier")
        self.assertEqual(status, 200)
        self.assertEqual(body["verification_status"], "verified")

        new_key = _x25519_key()
        self.service.rotate_identity_key("peer",
                                         {"identity_key": new_key})
        new_fp = _expected_fingerprint(new_key)
        status, body = self._request(
            "GET", "/v1/devices/peer/identity-fingerprint"
                   "?verifier_device_id=verifier")
        self.assertEqual(body["verification_status"], "changed")
        self.assertEqual(body["identity_key_version"], 2)
        self.assertEqual(body["verification_id"], "v1")

        old_replay = {**payload, "expected_fingerprint": new_fp}
        status, body = self._request(
            "POST", "/v1/devices/peer/identity-verifications", old_replay)
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "verification_id")

        new_confirm = {"verifier_device_id": "verifier",
                       "verification_id": "v2",
                       "expected_fingerprint": new_fp}
        status, body = self._request(
            "POST", "/v1/devices/peer/identity-verifications", new_confirm)
        self.assertEqual(status, 201)
        self.assertEqual(body["verification_status"], "verified")
        self.assertEqual(body["verification_id"], "v2")

    def test_post_active_under_other_id_409(self) -> None:
        payload = {"verifier_device_id": "verifier",
                   "verification_id": "v1",
                   "expected_fingerprint": self.fingerprint}
        self._request("POST",
                      "/v1/devices/peer/identity-verifications", payload)
        status, body = self._request(
            "POST", "/v1/devices/peer/identity-verifications",
            {**payload, "verification_id": "v2"})
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "verification_id")

    def test_post_mismatch_409_expected_field(self) -> None:
        status, body = self._request(
            "POST", "/v1/devices/peer/identity-verifications",
            {"verifier_device_id": "verifier",
             "verification_id": "v1",
             "expected_fingerprint": "0" * 64})
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "expected_fingerprint")

    def test_post_bad_input_400(self) -> None:
        status, body = self._request(
            "POST", "/v1/devices/peer/identity-verifications",
            {"verifier_device_id": "verifier",
             "verification_id": "v1"})
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "expected_fingerprint")

    def test_post_unknown_peer_404(self) -> None:
        status, body = self._request(
            "POST", "/v1/devices/ghost/identity-verifications",
            {"verifier_device_id": "verifier",
             "verification_id": "v1",
             "expected_fingerprint": self.fingerprint})
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "device_id")


class IdentityCLITest(unittest.TestCase):
    def setUp(self) -> None:
        self.server, service = create_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        self.peer_key = _register(service, "peer")
        _register(service, "verifier")
        self.fingerprint = _expected_fingerprint(self.peer_key)

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
        result = self._run("fingerprint", "peer",
                           "--verifier-device-id", "verifier")
        self.assertEqual(result.returncode, 0, result.stderr)
        body = json.loads(result.stdout.strip())
        self.assertEqual(body["fingerprint"], self.fingerprint)
        self.assertEqual(body["identity_key_version"], 1)
        self.assertEqual(body["verification_status"], "unverified")
        self.assertIsNone(body["verification_id"])

    def test_fingerprint_failure_stderr_nonzero(self) -> None:
        result = self._run("fingerprint", "peer")
        self.assertNotEqual(result.returncode, 0)
        result = self._run("fingerprint", "ghost",
                           "--verifier-device-id", "verifier")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(json.loads(result.stderr.strip())["field"],
                         "device_id")

    def test_verify_identity_201_replay_200(self) -> None:
        first = self._run(
            "verify-identity", "peer",
            "--verifier-device-id", "verifier",
            "--verification-id", "v1",
            "--expected-fingerprint", self.fingerprint)
        self.assertEqual(first.returncode, 0, first.stderr)
        body = json.loads(first.stdout.strip())
        self.assertEqual(body["verification_status"], "verified")
        self.assertEqual(body["verification_id"], "v1")
        second = self._run(
            "verify-identity", "peer",
            "--verifier-device-id", "verifier",
            "--verification-id", "v1",
            "--expected-fingerprint", self.fingerprint)
        self.assertEqual(second.returncode, 0, second.stderr)

    def test_verify_identity_conflict_nonzero(self) -> None:
        result = self._run(
            "verify-identity", "peer",
            "--verifier-device-id", "verifier",
            "--verification-id", "v1",
            "--expected-fingerprint", "0" * 64)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(json.loads(result.stderr.strip())["field"],
                         "expected_fingerprint")


class IdentityPersistenceTest(unittest.TestCase):
    def _path(self) -> str:
        path = tempfile.mktemp(suffix=".json")
        self.addCleanup(
            lambda: os.path.exists(path) and os.unlink(path))
        return path

    def test_verification_and_version_survive_restart(self) -> None:
        path = self._path()
        peer_key = None
        first = DeviceService()
        attach_persistence(first, path)
        peer_key = _register(first, "peer")
        _register(first, "verifier")
        fingerprint = _expected_fingerprint(peer_key)
        body, status = first.verify_identity("peer", {
            "verifier_device_id": "verifier",
            "verification_id": "v1",
            "expected_fingerprint": fingerprint})
        self.assertEqual(status, 201)
        new_key = _x25519_key()
        first.rotate_identity_key("peer", {"identity_key": new_key})
        new_fp = _expected_fingerprint(new_key)
        body, status = first.verify_identity("peer", {
            "verifier_device_id": "verifier",
            "verification_id": "v2",
            "expected_fingerprint": new_fp})
        self.assertEqual(status, 201)

        second = DeviceService()
        attach_persistence(second, path)
        device = second.store.find_by_device_id("peer")
        self.assertEqual(device.identity_key, new_key)
        self.assertEqual(device.identity_key_version, 2)
        view = second.identity_fingerprint_view("peer", "verifier")
        self.assertEqual(view["identity_key_version"], 2)
        self.assertEqual(view["verification_status"], "verified")
        self.assertEqual(view["verification_id"], "v2")
        self.assertEqual(view["fingerprint"], new_fp)

        # Idempotency is stable across restart.
        body, status = second.verify_identity("peer", {
            "verifier_device_id": "verifier",
            "verification_id": "v2",
            "expected_fingerprint": new_fp})
        self.assertEqual(status, 200)
        # The superseded id can never succeed again after restart.
        with self.assertRaises(ServiceError) as ctx:
            second.verify_identity("peer", {
                "verifier_device_id": "verifier",
                "verification_id": "v1",
                "expected_fingerprint": new_fp})
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "verification_id")

    def test_changed_status_survives_restart(self) -> None:
        path = self._path()
        first = DeviceService()
        attach_persistence(first, path)
        peer_key = _register(first, "peer")
        _register(first, "verifier")
        fingerprint = _expected_fingerprint(peer_key)
        first.verify_identity("peer", {
            "verifier_device_id": "verifier",
            "verification_id": "v1",
            "expected_fingerprint": fingerprint})
        new_key = _x25519_key()
        first.rotate_identity_key("peer", {"identity_key": new_key})

        second = DeviceService()
        attach_persistence(second, path)
        view = second.identity_fingerprint_view("peer", "verifier")
        self.assertEqual(view["verification_status"], "changed")
        self.assertEqual(view["verification_id"], "v1")
        self.assertEqual(view["identity_key_version"], 2)

    def test_legacy_file_without_fields_reads_as_unverified_v1(self) -> None:
        path = self._path()
        key = _x25519_key()
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({
                "version": 1,
                "devices": [{
                    "user_id": "u1", "device_id": "peer",
                    "identity_key": key,
                    "registered_at": "2026-01-01T00:00:00+00:00",
                    "revoked": False, "prekeys": []}, {
                    "user_id": "u1", "device_id": "verifier",
                    "identity_key": _x25519_key(),
                    "registered_at": "2026-01-01T00:00:00+00:00",
                    "revoked": False, "prekeys": []}],
                "sessions": [], "messages": {}, "delivery": [],
            }, handle)
        service = DeviceService()
        attach_persistence(service, path)
        device = service.store.find_by_device_id("peer")
        self.assertEqual(device.identity_key_version, 1)
        view = service.identity_fingerprint_view("peer", "verifier")
        self.assertEqual(view["identity_key_version"], 1)
        self.assertEqual(view["verification_status"], "unverified")
        self.assertIsNone(view["verification_id"])
        self.assertEqual(view["fingerprint"], _expected_fingerprint(key))
        # A first verification on the upgraded file persists normally.
        body, status = service.verify_identity("peer", {
            "verifier_device_id": "verifier",
            "verification_id": "v1",
            "expected_fingerprint": view["fingerprint"]})
        self.assertEqual(status, 201)

    def test_malformed_verification_section_refuses_startup(self) -> None:
        path = self._path()
        key = _x25519_key()
        base_devices = [{
            "user_id": "u1", "device_id": "peer",
            "identity_key": key,
            "registered_at": "2026-01-01T00:00:00+00:00",
            "identity_key_version": 1, "revoked": False, "prekeys": []}, {
            "user_id": "u1", "device_id": "verifier",
            "identity_key": _x25519_key(),
            "registered_at": "2026-01-01T00:00:00+00:00",
            "identity_key_version": 1, "revoked": False, "prekeys": []}]
        record = {
            "verification_id": "v1",
            "verifier_device_id": "verifier",
            "device_id": "peer",
            "fingerprint": _expected_fingerprint(key),
            "confirmed_at": "2026-01-02T00:00:00+00:00"}
        bad_cases = [
            "not-a-list",
            ["not-an-object"],
            [{**record, "verification_id": ""}],
            [{**record, "fingerprint": "Z" * 64}],
            [{**record, "fingerprint": "0" * 63}],
            [{**record, "device_id": "ghost"}],
            [{**record, "verifier_device_id": "ghost"}],
            [{**record, "verifier_device_id": "peer"}],
            [record, {**record, "confirmed_at":
                      "2026-01-03T00:00:00+00:00"}],
        ]
        for case in bad_cases:
            bad_path = self._path()
            with open(bad_path, "w", encoding="utf-8") as handle:
                json.dump({"version": 1, "devices": base_devices,
                           "identity_verifications": case,
                           "sessions": [], "messages": {},
                           "delivery": []}, handle)
            with self.subTest(case=case):
                service = DeviceService()
                with self.assertRaises(StateFileError):
                    attach_persistence(service, bad_path)

    def test_version_mismatch_with_chain_refuses_startup(self) -> None:
        path = self._path()
        first = DeviceService()
        attach_persistence(first, path)
        key1 = _register(first, "peer")
        _register(first, "verifier")
        key2 = _x25519_key()
        first.rotate_identity_key("peer", {"identity_key": key2})
        # Tamper: claim version 1 although the chain records one rotation.
        with open(path, encoding="utf-8") as handle:
            document = json.load(handle)
        document["devices"][0]["identity_key_version"] = 1
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        second = DeviceService()
        with self.assertRaises(StateFileError):
            attach_persistence(second, path)


if __name__ == "__main__":
    unittest.main()
