"""Tests for signature-authorized device revocation.

Covers the service, HTTP (real loopback socket), CLI (real subprocess),
key-audit and persistence layers for:

* POST /v1/devices/{device_id}/revoke-verified

The revocation applies only when the standard-base64 64-byte Ed25519
signature verifies over the domain-separated canonical revocation message
(``E2EE-DEVICE-REVOCATION-V1``) against the device's *current* identity
key, and ``expected_version`` equals the current
``identity_key_version``. A first-time revocation revokes the device and
all of its pre-keys and appends one ``device_revoked`` event; a valid
replay against an already revoked device changes no state, audit chain
or commit generation. The identity key, its version and timestamps are
never changed. The ordinary ``/revoke`` entry keeps its old behavior.
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

from e2ee_backend.crypto import device_revocation_proof_message
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


def _x25519_b64() -> str:
    return _raw_b64(x25519.X25519PrivateKey.generate().public_key())


def _x25519_der_b64() -> str:
    """X25519 public key as base64 DER SPKI — never parses as Ed25519."""
    der = x25519.X25519PrivateKey.generate().public_key().public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo)
    return base64.b64encode(der).decode()


def _authorization(private, user_id, device_id, expected_version) -> str:
    message = device_revocation_proof_message(
        user_id, device_id, expected_version)
    return base64.b64encode(private.sign(message)).decode()


def _register_payload(device_id="d1", user_id="u1", identity_key=None,
                      prekeys=None) -> dict:
    return {
        "user_id": user_id,
        "device_id": device_id,
        "identity_key": identity_key or _x25519_b64(),
        "signed_prekeys": (prekeys if prekeys is not None
                           else [{"key_id": "k1",
                                  "public_key": _x25519_b64()}]),
    }


class RevokeVerifiedServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = DeviceService()
        self.private, self.identity = _new_identity()
        self.service.register(
            _register_payload(identity_key=self.identity,
                              prekeys=[{"key_id": "k1",
                                        "public_key": _x25519_b64()},
                                       {"key_id": "k2",
                                        "public_key": _x25519_b64()}]))
        self.registered_at = self.service.get_device("d1")["registered_at"]

    def _payload(self, expected_version=1, signature=None, private=None,
                 user_id="u1", device_id="d1"):
        if signature is None:
            signer = private or self.private
            signature = _authorization(
                signer, user_id, device_id, expected_version)
        return {"expected_version": expected_version,
                "signature": signature}

    def _device(self):
        return self.service.store.find_by_device_id("d1")

    def test_valid_authorization_revokes_200(self) -> None:
        body = self.service.revoke_device_verified("d1", self._payload())
        self.assertEqual(body, {"device_id": "d1", "revoked": True})
        device = self._device()
        self.assertTrue(device.revoked)
        self.assertTrue(all(pk.revoked for pk in device.prekeys))
        events = self.service.list_key_events("d1", 0, 100)["events"]
        self.assertEqual([e["type"] for e in events],
                         ["registered", "device_revoked"])

    def test_revocation_does_not_change_identity_or_timestamps(self) -> None:
        body = self.service.revoke_device_verified("d1", self._payload())
        self.assertEqual(body, {"device_id": "d1", "revoked": True})
        device = self._device()
        self.assertEqual(device.identity_key, self.identity)
        self.assertEqual(device.identity_key_version, 1)
        self.assertEqual(device.rotated_at, self.registered_at)
        self.assertEqual(device.registered_at, self.registered_at)

    def test_other_device_and_public_view_unaffected_until_revoked(
            self) -> None:
        bob_private, bob_key = _new_identity()
        self.service.register(_register_payload(
            device_id="bob", identity_key=bob_key,
            prekeys=[{"key_id": "b1", "public_key": _x25519_b64()}]))
        self.service.revoke_device_verified("d1", self._payload())
        bob = self.service.store.find_by_device_id("bob")
        self.assertFalse(bob.revoked)
        self.assertEqual(bob.identity_key_version, 1)
        # d1's public view reports it revoked; bob's pre-key is still listed.
        self.assertEqual(
            self.service.get_device("bob")["prekey_ids"], ["b1"])

    def test_body_not_object_400_request_body(self) -> None:
        for payload in (None, [], "x", 42):
            with self.subTest(payload=payload):
                with self.assertRaises(ServiceError) as ctx:
                    self.service.revoke_device_verified("d1", payload)
                self.assertEqual(ctx.exception.status_code, 400)
                self.assertEqual(ctx.exception.field, "request_body")

    def test_missing_or_bad_expected_version_400(self) -> None:
        good_sig = _authorization(self.private, "u1", "d1", 1)
        for bad in (None, True, False, 0, -1, "1", 1.5, [], {}):
            payload = {"expected_version": bad, "signature": good_sig}
            with self.subTest(bad=bad):
                with self.assertRaises(ServiceError) as ctx:
                    self.service.revoke_device_verified("d1", payload)
                self.assertEqual(ctx.exception.status_code, 400)
                self.assertEqual(ctx.exception.field, "expected_version")
        with self.assertRaises(ServiceError) as ctx:
            self.service.revoke_device_verified(
                "d1", {"signature": good_sig})
        self.assertEqual(ctx.exception.field, "expected_version")

    def test_missing_or_bad_signature_400(self) -> None:
        for bad in (None, "", 8, [], "@@@@", "abc", "a" * 88,
                    base64.b64encode(b"\x00" * 63).decode(),
                    base64.b64encode(b"\x00" * 65).decode()):
            payload = {"expected_version": 1, "signature": bad}
            with self.subTest(bad=bad):
                with self.assertRaises(ServiceError) as ctx:
                    self.service.revoke_device_verified("d1", payload)
                self.assertEqual(ctx.exception.status_code, 400)
                self.assertEqual(ctx.exception.field, "signature")
        with self.assertRaises(ServiceError) as ctx:
            self.service.revoke_device_verified(
                "d1", {"expected_version": 1})
        self.assertEqual(ctx.exception.field, "signature")

    def test_signature_must_be_canonical_base64(self) -> None:
        good = _authorization(self.private, "u1", "d1", 1)
        for bad in (good.replace("+", "-").replace("/", "_"),
                    good.rstrip("=")):
            with self.subTest(bad=bad):
                with self.assertRaises(ServiceError) as ctx:
                    self.service.revoke_device_verified(
                        "d1", {"expected_version": 1, "signature": bad})
                self.assertEqual(ctx.exception.field, "signature")

    def test_extra_fields_are_ignored(self) -> None:
        payload = self._payload()
        payload["extra"] = "ignored"
        payload["identity_key"] = self.identity
        body = self.service.revoke_device_verified("d1", payload)
        self.assertEqual(body, {"device_id": "d1", "revoked": True})

    def test_unknown_device_404(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.revoke_device_verified("ghost", self._payload())
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "device_id")

    def test_current_key_not_ed25519_400_identity_key(self) -> None:
        service = DeviceService()
        service.register(_register_payload(
            identity_key=_x25519_der_b64()))
        with self.assertRaises(ServiceError) as ctx:
            service.revoke_device_verified(
                "d1", {"expected_version": 1,
                       "signature": _authorization(
                           self.private, "u1", "d1", 1)})
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.field, "identity_key")

    def test_version_mismatch_409_expected_version(self) -> None:
        for bad_version in (2, 99):
            with self.subTest(bad_version=bad_version):
                with self.assertRaises(ServiceError) as ctx:
                    self.service.revoke_device_verified(
                        "d1", self._payload(expected_version=bad_version))
                self.assertEqual(ctx.exception.status_code, 409)
                self.assertEqual(ctx.exception.field, "expected_version")

    def test_version_mismatch_checked_before_signature(self) -> None:
        payload = self._payload(expected_version=2)
        with self.assertRaises(ServiceError) as ctx:
            self.service.revoke_device_verified("d1", payload)
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "expected_version")

    def test_bad_signature_400_signature(self) -> None:
        other, _ = _new_identity()
        cases = [
            # Signed by a different key.
            self._payload(private=other),
            # Signed for a different device id.
            self._payload(device_id="d2"),
            # Signed for a different user id.
            self._payload(user_id="u2"),
            # A signature over a different version (decoded length still 64).
            {"expected_version": 1,
             "signature": _authorization(self.private, "u1", "d1", 2)},
        ]
        for payload in cases:
            with self.subTest(payload=payload):
                with self.assertRaises(ServiceError) as ctx:
                    self.service.revoke_device_verified("d1", payload)
                self.assertEqual(ctx.exception.status_code, 400)
                self.assertEqual(ctx.exception.field, "signature")

    def test_check_order_device_key_version_signature(self) -> None:
        # Unknown device beats a bad version: 404/device_id.
        with self.assertRaises(ServiceError) as ctx:
            self.service.revoke_device_verified(
                "ghost", self._payload(expected_version=9))
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "device_id")
        # Non-Ed25519 current key beats a bad version: 400/identity_key.
        service = DeviceService()
        service.register(_register_payload(
            identity_key=_x25519_der_b64()))
        with self.assertRaises(ServiceError) as ctx:
            service.revoke_device_verified(
                "d1", {"expected_version": 9,
                       "signature": _authorization(
                           self.private, "u1", "d1", 1)})
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.field, "identity_key")

    def test_failure_leaves_state_untouched(self) -> None:
        with self.assertRaises(ServiceError):
            self.service.revoke_device_verified(
                "d1", self._payload(expected_version=7))
        device = self._device()
        self.assertFalse(device.revoked)
        self.assertTrue(all(not pk.revoked for pk in device.prekeys))
        self.assertEqual(device.identity_key_version, 1)
        events = self.service.list_key_events("d1", 0, 100)["events"]
        self.assertEqual([e["type"] for e in events], ["registered"])

    # -- already revoked devices still require the authorization ----------

    def test_revoked_device_valid_replay_is_state_free(self) -> None:
        self.service.revoke_device("d1")  # ordinary entry
        events_before = self.service.list_key_events("d1", 0, 100)["events"]
        # A valid replay answers 200 with the same body...
        body = self.service.revoke_device_verified("d1", self._payload())
        self.assertEqual(body, {"device_id": "d1", "revoked": True})
        # ...but appends no event and leaves the version and timestamps.
        events_after = self.service.list_key_events("d1", 0, 100)["events"]
        self.assertEqual(events_before, events_after)
        device = self._device()
        self.assertEqual(device.identity_key_version, 1)
        self.assertEqual(device.rotated_at, self.registered_at)

    def test_revoked_device_still_checks_the_authorization(self) -> None:
        self.service.revoke_device("d1")
        other, _ = _new_identity()
        # Bad signature is refused even though the device is already revoked.
        with self.assertRaises(ServiceError) as ctx:
            self.service.revoke_device_verified(
                "d1", self._payload(private=other))
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.field, "signature")
        # A stale version is refused too.
        with self.assertRaises(ServiceError) as ctx:
            self.service.revoke_device_verified(
                "d1", self._payload(expected_version=2))
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "expected_version")

    def test_authorized_revoke_after_rotation_needs_current_key_version(
            self) -> None:
        from e2ee_backend.crypto import identity_rotation_proof_message
        new_private, new_key = _new_identity()
        rot_message = identity_rotation_proof_message(
            "u1", "d1", new_key, 1)
        self.service.rotate_identity_key_verified(
            "d1", {"identity_key": new_key, "expected_version": 1,
                   "signature": base64.b64encode(
                       self.private.sign(rot_message)).decode()})
        self.assertEqual(self._device().identity_key_version, 2)
        # An old-key, version-1 revocation is stale: version 1 != 2 would be
        # 409, but this body claims version 1 so the version check fails
        # first; either way it is refused. A valid old-key signature over
        # version 2 fails verification against the new current key.
        with self.assertRaises(ServiceError) as ctx:
            self.service.revoke_device_verified(
                "d1", {"expected_version": 1,
                       "signature": _authorization(
                           self.private, "u1", "d1", 1)})
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "expected_version")
        with self.assertRaises(ServiceError) as ctx:
            self.service.revoke_device_verified(
                "d1", {"expected_version": 2,
                       "signature": _authorization(
                           self.private, "u1", "d1", 2)})
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.field, "signature")
        # The current key at the current version authorizes the revocation.
        body = self.service.revoke_device_verified(
            "d1", {"expected_version": 2,
                   "signature": _authorization(
                       new_private, "u1", "d1", 2)})
        self.assertEqual(body, {"device_id": "d1", "revoked": True})
        self.assertTrue(self._device().revoked)
        self.assertEqual(self._device().identity_key_version, 2)
        # Identity version and timestamps survive the revocation.
        self.assertEqual(self._device().identity_key, new_key)


class RevokeVerifiedHTTPTest(unittest.TestCase):
    def setUp(self) -> None:
        self.server, self.service = create_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        self.private, self.identity = _new_identity()
        self.service.register(
            _register_payload(identity_key=self.identity))

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _request(self, method: str, path: str, body: object = None,
                 raw: str | None = None):
        connection = HTTPConnection("127.0.0.1", self.port, timeout=5)
        payload = raw if raw is not None else (
            json.dumps(body) if body is not None else None)
        headers = {"Content-Type": "application/json"} if payload else {}
        connection.request(method, path, body=payload, headers=headers)
        response = connection.getresponse()
        data = response.read().decode("utf-8")
        connection.close()
        return response.status, json.loads(data)

    def _payload(self, expected_version=1) -> dict:
        return {"expected_version": expected_version,
                "signature": _authorization(
                    self.private, "u1", "d1", expected_version)}

    def test_revoke_verified_200(self) -> None:
        status, body = self._request(
            "POST", "/v1/devices/d1/revoke-verified", self._payload())
        self.assertEqual(status, 200)
        self.assertEqual(body, {"device_id": "d1", "revoked": True})
        self.assertTrue(
            self.service.store.find_by_device_id("d1").revoked)

    def test_replay_is_200(self) -> None:
        status, _ = self._request(
            "POST", "/v1/devices/d1/revoke-verified", self._payload())
        self.assertEqual(status, 200)
        status, body = self._request(
            "POST", "/v1/devices/d1/revoke-verified", self._payload())
        self.assertEqual(status, 200)
        self.assertEqual(body, {"device_id": "d1", "revoked": True})

    def test_invalid_json_400_request_body(self) -> None:
        status, body = self._request(
            "POST", "/v1/devices/d1/revoke-verified", raw="{not json")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "request_body")

    def test_non_object_400_request_body(self) -> None:
        status, body = self._request(
            "POST", "/v1/devices/d1/revoke-verified", [1, 2])
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "request_body")

    def test_missing_version_400(self) -> None:
        status, body = self._request(
            "POST", "/v1/devices/d1/revoke-verified",
            {"signature": self._payload()["signature"]})
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "expected_version")

    def test_bool_version_400(self) -> None:
        payload = self._payload()
        payload["expected_version"] = True
        status, body = self._request(
            "POST", "/v1/devices/d1/revoke-verified", payload)
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "expected_version")

    def test_bad_signature_encoding_400(self) -> None:
        status, body = self._request(
            "POST", "/v1/devices/d1/revoke-verified",
            {"expected_version": 1, "signature": "not-base64!"})
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "signature")

    def test_unknown_device_404(self) -> None:
        status, body = self._request(
            "POST", "/v1/devices/ghost/revoke-verified", self._payload())
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "device_id")

    def test_version_mismatch_409(self) -> None:
        status, body = self._request(
            "POST", "/v1/devices/d1/revoke-verified",
            self._payload(expected_version=5))
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "expected_version")

    def test_bad_signature_400(self) -> None:
        payload = self._payload()
        payload["signature"] = base64.b64encode(b"\x00" * 64).decode()
        status, body = self._request(
            "POST", "/v1/devices/d1/revoke-verified", payload)
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "signature")

    def test_revoked_device_bad_signature_still_400(self) -> None:
        self.service.revoke_device("d1")
        payload = self._payload()
        payload["signature"] = base64.b64encode(b"\x00" * 64).decode()
        status, body = self._request(
            "POST", "/v1/devices/d1/revoke-verified", payload)
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "signature")

    def test_plain_revoke_route_still_works(self) -> None:
        status, body = self._request(
            "POST", "/v1/devices/d1/revoke", None)
        self.assertEqual(status, 200)
        self.assertEqual(body, {"device_id": "d1", "revoked": True})

    def test_path_with_percent_encoded_slash(self) -> None:
        # A device whose id contains an encoded slash: the signed message
        # uses the path-decoded id (an ordinary '/'), and a slash that is
        # not encoded makes the path a different shape (404).
        slash_id = "a/b"
        self.service.register(_register_payload(
            device_id=slash_id, identity_key=self.identity))
        signature = _authorization(self.private, "u1", slash_id, 1)
        status, body = self._request(
            "POST", "/v1/devices/a%2Fb/revoke-verified",
            {"expected_version": 1, "signature": signature})
        self.assertEqual(status, 200)
        self.assertEqual(body["device_id"], slash_id)
        # The raw-slash shape is not the single-segment revoke route.
        status, body = self._request(
            "POST", "/v1/devices/a/b/revoke-verified",
            {"expected_version": 1, "signature": signature})
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "device_id")


class RevokeVerifiedCLITest(unittest.TestCase):
    def setUp(self) -> None:
        self.server, service = create_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        self.private, self.identity = _new_identity()
        service.register(_register_payload(identity_key=self.identity))

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _run(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "e2ee_backend", "--base-url",
             self.base_url, *arguments],
            capture_output=True, text=True, timeout=15)

    def test_revoke_verified_success(self) -> None:
        signature = _authorization(self.private, "u1", "d1", 1)
        result = self._run("revoke-device-verified",
                           "--device-id", "d1",
                           "--expected-version", "1",
                           "--signature", signature)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout.strip()),
                         {"device_id": "d1", "revoked": True})
        self.assertEqual(result.stderr, "")

    def test_revoke_verified_failure_stderr_nonzero(self) -> None:
        signature = _authorization(self.private, "u1", "d1", 1)
        result = self._run("revoke-device-verified",
                           "--device-id", "d1",
                           "--expected-version", "9",
                           "--signature", signature)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        self.assertEqual(json.loads(result.stderr.strip())["field"],
                         "expected_version")

    def test_revoke_verified_bad_signature_exit_1(self) -> None:
        result = self._run("revoke-device-verified",
                           "--device-id", "d1",
                           "--expected-version", "1",
                           "--signature",
                           base64.b64encode(b"\x00" * 64).decode())
        self.assertEqual(result.returncode, 1)
        self.assertEqual(json.loads(result.stderr.strip())["field"],
                         "signature")

    def test_plain_revoke_command_still_works(self) -> None:
        result = self._run("revoke-device", "--device-id", "d1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout.strip()),
                         {"device_id": "d1", "revoked": True})


class RevokeVerifiedPersistenceTest(unittest.TestCase):
    def _service(self, path):
        service = DeviceService()
        store = attach_persistence(service, path)
        return service, store

    def test_revocation_survives_restart(self) -> None:
        path = tempfile.mktemp(suffix=".json")
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        private, identity = _new_identity()

        first, _ = self._service(path)
        first.register(_register_payload(
            identity_key=identity,
            prekeys=[{"key_id": "k1", "public_key": _x25519_b64()}]))
        first.revoke_device_verified(
            "d1", {"expected_version": 1,
                   "signature": _authorization(private, "u1", "d1", 1)})

        second, _ = self._service(path)
        device = second.store.find_by_device_id("d1")
        self.assertTrue(device.revoked)
        self.assertTrue(all(pk.revoked for pk in device.prekeys))
        self.assertEqual(device.identity_key, identity)
        self.assertEqual(device.identity_key_version, 1)
        events = second.list_key_events("d1", 0, 100)["events"]
        self.assertEqual([e["type"] for e in events],
                         ["registered", "device_revoked"])

    def test_valid_replay_advances_no_generation_or_event(self) -> None:
        path = tempfile.mktemp(suffix=".json")
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        private, identity = _new_identity()
        service, store = self._service(path)
        service.register(_register_payload(identity_key=identity))
        service.revoke_device_verified(
            "d1", {"expected_version": 1,
                   "signature": _authorization(private, "u1", "d1", 1)})
        generation = store.commit_seq
        events_before = service.list_key_events("d1", 0, 100)["events"]

        body = service.revoke_device_verified(
            "d1", {"expected_version": 1,
                   "signature": _authorization(private, "u1", "d1", 1)})
        self.assertEqual(body, {"device_id": "d1", "revoked": True})
        self.assertEqual(store.commit_seq, generation)
        self.assertEqual(service.list_key_events("d1", 0, 100)["events"],
                         events_before)

    def test_failure_advances_no_generation_and_leaves_no_event(self) -> None:
        path = tempfile.mktemp(suffix=".json")
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        private, identity = _new_identity()
        service, store = self._service(path)
        service.register(_register_payload(identity_key=identity))
        generation = store.commit_seq

        other, _ = _new_identity()
        with self.assertRaises(ServiceError):
            service.revoke_device_verified(
                "d1", {"expected_version": 1,
                       "signature": _authorization(other, "u1", "d1", 1)})
        self.assertEqual(store.commit_seq, generation)
        device = service.store.find_by_device_id("d1")
        self.assertFalse(device.revoked)
        self.assertEqual(device.identity_key_version, 1)
        events = service.list_key_events("d1", 0, 100)["events"]
        self.assertEqual([e["type"] for e in events], ["registered"])

    def test_persist_failure_rolls_back_the_revocation(self) -> None:
        from e2ee_backend.persistence import PersistenceUnavailable

        path = tempfile.mktemp(suffix=".json")
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        private, identity = _new_identity()
        service, state_store = self._service(path)
        service.register(_register_payload(
            identity_key=identity,
            prekeys=[{"key_id": "k1", "public_key": _x25519_b64()}]))

        def fail_save(_pending) -> None:
            raise OSError("simulated disk failure")

        state_store.save = fail_save  # type: ignore[assignment]
        with self.assertRaises(PersistenceUnavailable):
            service.revoke_device_verified(
                "d1", {"expected_version": 1,
                       "signature": _authorization(private, "u1", "d1", 1)})
        device = service.store.find_by_device_id("d1")
        self.assertFalse(device.revoked)
        self.assertTrue(all(not pk.revoked for pk in device.prekeys))
        events = service.list_key_events("d1", 0, 100)["events"]
        self.assertEqual([e["type"] for e in events], ["registered"])
        # The next request after the rolled-back 503 still works.
        del state_store.save
        body = service.revoke_device_verified(
            "d1", {"expected_version": 1,
                   "signature": _authorization(private, "u1", "d1", 1)})
        self.assertEqual(body, {"device_id": "d1", "revoked": True})
        self.assertTrue(service.store.find_by_device_id("d1").revoked)


class RevokeVerifiedConcurrencyTest(unittest.TestCase):
    """Linearization against a concurrent identity rotation."""

    def test_concurrent_rotation_and_revocation(self) -> None:
        # Repeat the race: a verified rotation A->B (version 1 -> 2) and a
        # verified revocation signed by A at version 1 commit under the same
        # store lock. The authorization valid only under the old state must
        # lose when the rotation commits first (400/signature, or 409/
        # device_id when the revocation commits first and the rotation loses);
        # never both succeed, and the final state is always consistent.
        for _ in range(40):
            service = DeviceService()
            private_a, key_a = _new_identity()
            private_b, key_b = _new_identity()
            service.register(_register_payload(identity_key=key_a))

            from e2ee_backend.crypto import identity_rotation_proof_message
            rot_sig = base64.b64encode(private_a.sign(
                identity_rotation_proof_message(
                    "u1", "d1", key_b, 1))).decode()
            rev_sig = _authorization(private_a, "u1", "d1", 1)
            outcomes = []

            def rotate() -> None:
                try:
                    service.rotate_identity_key_verified(
                        "d1", {"identity_key": key_b,
                               "expected_version": 1,
                               "signature": rot_sig})
                    outcomes.append(("rotate", None))
                except ServiceError as error:
                    outcomes.append(("rotate", error.field))

            def revoke() -> None:
                try:
                    service.revoke_device_verified(
                        "d1", {"expected_version": 1,
                               "signature": rev_sig})
                    outcomes.append(("revoke", None))
                except ServiceError as error:
                    outcomes.append(("revoke", error.field))

            t1 = threading.Thread(target=rotate)
            t2 = threading.Thread(target=revoke)
            t1.start()
            t2.start()
            t1.join()
            t2.join()

            self.assertEqual(len(outcomes), 2)
            results = {op: field for op, field in outcomes}
            device = service.store.find_by_device_id("d1")
            if results["rotate"] is None and results["revoke"] is None:
                self.fail("both a rotation and a stale revocation succeeded")
            if results["rotate"] is None:
                # Rotation committed first: the stale (version-1) revocation
                # fails under the new state — version 1 != 2 is 409, and even
                # naming version 2 would fail verifying against key b.
                self.assertFalse(device.revoked)
                self.assertEqual(device.identity_key, key_b)
                self.assertEqual(device.identity_key_version, 2)
                self.assertIn(results["revoke"],
                              ("expected_version", "signature"))
            else:
                # Revocation committed first: the rotation is refused on the
                # revoked device.
                self.assertTrue(device.revoked)
                self.assertEqual(device.identity_key, key_a)
                self.assertEqual(device.identity_key_version, 1)
                self.assertEqual(results["rotate"], "device_id")
                self.assertIsNone(results["revoke"])

            # A revocation valid under the post-race current state still
            # applies when the rotation won (current key b, version 2).
            if results["rotate"] is None:
                body = service.revoke_device_verified(
                    "d1", {"expected_version": 2,
                           "signature": _authorization(
                               private_b, "u1", "d1", 2)})
                self.assertEqual(body, {"device_id": "d1", "revoked": True})


if __name__ == "__main__":
    unittest.main()
