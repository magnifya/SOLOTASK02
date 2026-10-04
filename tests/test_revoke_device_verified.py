"""Tests for signature-authorized device revocation.

Covers the service, HTTP (real loopback socket), CLI (real subprocess),
key-audit and persistence layers for:

* POST /v1/devices/{device_id}/revoke-verified

The revocation applies only when the standard-base64 64-byte Ed25519
signature verifies over the domain-separated canonical revocation message
(``E2EE-DEVICE-REVOCATION-V1``) against the device's *current* identity
key, and ``expected_version`` equals the current ``identity_key_version``.
A first valid call revokes the device and every pre-key and appends the
single ``device_revoked`` audit event; a valid replay against an already
revoked device changes no state, audit chain or commit generation. An
already-revoked device still requires a valid authorization. The ordinary
``/revoke`` entry is unchanged.
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


def _rotation_authorization(private, user_id, device_id, identity_key,
                            expected_version) -> str:
    from e2ee_backend.crypto import identity_rotation_proof_message
    message = identity_rotation_proof_message(
        user_id, device_id, identity_key, expected_version)
    return base64.b64encode(private.sign(message)).decode()


def _register_payload(device_id="d1", user_id="u1", identity_key=None,
                      prekeys=None) -> dict:
    return {
        "user_id": user_id,
        "device_id": device_id,
        "identity_key": identity_key or _x25519_b64(),
        "signed_prekeys": (prekeys if prekeys is not None
                           else [{"key_id": "k1",
                                  "public_key": _x25519_b64()},
                                 {"key_id": "k2",
                                  "public_key": _x25519_b64()}]),
    }


class RevokeVerifiedServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = DeviceService()
        self.private, self.identity = _new_identity()
        self.service.register(
            _register_payload(identity_key=self.identity))
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
        self.assertEqual([pk.key_id for pk in device.prekeys], ["k1", "k2"])
        # Identity material, version and timestamps are untouched.
        self.assertEqual(device.identity_key, self.identity)
        self.assertEqual(device.identity_key_version, 1)
        self.assertEqual(device.rotated_at, self.registered_at)
        events = self.service.list_key_events("d1", 0, 100)["events"]
        self.assertEqual([e["type"] for e in events],
                         ["registered", "device_revoked"])
        self.assertEqual(events[-1]["payload"], {})
        shown = self.service.get_device("d1")
        self.assertEqual(shown["prekey_ids"], [])

    def test_valid_replay_after_verified_revoke_is_noop(self) -> None:
        payload = self._payload()
        self.service.revoke_device_verified("d1", payload)
        events_before = self.service.list_key_events("d1", 0, 100)["events"]
        body = self.service.revoke_device_verified("d1", payload)
        self.assertEqual(body, {"device_id": "d1", "revoked": True})
        events_after = self.service.list_key_events("d1", 0, 100)["events"]
        self.assertEqual(events_after, events_before)

    def test_valid_authorization_after_plain_revoke_is_noop(self) -> None:
        self.service.revoke_device("d1")
        events_before = self.service.list_key_events("d1", 0, 100)["events"]
        body = self.service.revoke_device_verified("d1", self._payload())
        self.assertEqual(body, {"device_id": "d1", "revoked": True})
        events_after = self.service.list_key_events("d1", 0, 100)["events"]
        self.assertEqual(
            [e["type"] for e in events_after],
            [e["type"] for e in events_before])

    def test_works_after_identity_rotation(self) -> None:
        new_private, new_key = _new_identity()
        self.service.rotate_identity_key_verified(
            "d1", {"identity_key": new_key, "expected_version": 1,
                   "signature": _rotation_authorization(
                       self.private, "u1", "d1", new_key, 1)})
        # A version-1 authorization is stale after the rotation.
        with self.assertRaises(ServiceError) as ctx:
            self.service.revoke_device_verified("d1", self._payload())
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "expected_version")
        # The current key at the current version authorizes the revocation.
        body = self.service.revoke_device_verified(
            "d1", self._payload(expected_version=2, private=new_private))
        self.assertEqual(body, {"device_id": "d1", "revoked": True})
        device = self._device()
        self.assertTrue(device.revoked)
        self.assertEqual(device.identity_key, new_key)
        self.assertEqual(device.identity_key_version, 2)
        events = self.service.list_key_events("d1", 0, 100)["events"]
        self.assertEqual([e["type"] for e in events],
                         ["registered", "identity_rotated",
                          "device_revoked"])

    def test_body_not_object_400_request_body(self) -> None:
        for payload in (None, [], "x", 42):
            with self.subTest(payload=payload):
                with self.assertRaises(ServiceError) as ctx:
                    self.service.revoke_device_verified("d1", payload)
                self.assertEqual(ctx.exception.status_code, 400)
                self.assertEqual(ctx.exception.field, "request_body")

    def test_missing_fields_400(self) -> None:
        cases = [
            ({}, "expected_version"),
            ({"expected_version": 1}, "signature"),
        ]
        for payload, field in cases:
            with self.subTest(field=field):
                with self.assertRaises(ServiceError) as ctx:
                    self.service.revoke_device_verified("d1", payload)
                self.assertEqual(ctx.exception.status_code, 400)
                self.assertEqual(ctx.exception.field, field)

    def test_expected_version_must_be_positive_int_no_bool(self) -> None:
        for bad in (True, False, 0, -1, "1", 1.5, None):
            with self.subTest(bad=bad):
                with self.assertRaises(ServiceError) as ctx:
                    self.service.revoke_device_verified(
                        "d1", self._payload(expected_version=bad))
                self.assertEqual(ctx.exception.status_code, 400)
                self.assertEqual(ctx.exception.field, "expected_version")

    def test_signature_field_errors_400(self) -> None:
        good = self._payload()["signature"]
        cases = [
            None, "", 8, b"x", "@@@@", "abc", "a" * 88,
            base64.b64encode(b"\x00" * 63).decode(),
            base64.b64encode(b"\x00" * 65).decode(),
        ]
        for bad in cases:
            with self.subTest(bad=bad):
                with self.assertRaises(ServiceError) as ctx:
                    self.service.revoke_device_verified(
                        "d1", {"expected_version": 1, "signature": bad})
                self.assertEqual(ctx.exception.status_code, 400)
                self.assertEqual(ctx.exception.field, "signature")
        # A well-formed signature must still be present even when the
        # version itself is malformed (version is checked first).
        with self.assertRaises(ServiceError) as ctx:
            self.service.revoke_device_verified(
                "d1", {"expected_version": True, "signature": good})
        self.assertEqual(ctx.exception.field, "expected_version")

    def test_signature_must_be_canonical_base64(self) -> None:
        good = self._payload()["signature"]
        for bad in (good.replace("+", "-").replace("/", "_"),
                    good.rstrip("=")):
            with self.subTest(bad=bad):
                with self.assertRaises(ServiceError) as ctx:
                    self.service.revoke_device_verified(
                        "d1", {"expected_version": 1, "signature": bad})
                self.assertEqual(ctx.exception.status_code, 400)
                self.assertEqual(ctx.exception.field, "signature")

    def test_extra_fields_are_ignored(self) -> None:
        payload = self._payload()
        payload["identity_key"] = "whatever"
        payload["extra"] = [1, 2]
        body = self.service.revoke_device_verified("d1", payload)
        self.assertEqual(body, {"device_id": "d1", "revoked": True})

    def test_unknown_device_404(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.revoke_device_verified("ghost", self._payload())
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "device_id")

    def test_current_key_not_ed25519_400_identity_key(self) -> None:
        service = DeviceService()
        service.register(_register_payload(identity_key=_x25519_der_b64()))
        with self.assertRaises(ServiceError) as ctx:
            service.revoke_device_verified("d1", self._payload())
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
        payload = {"expected_version": 7,
                   "signature": base64.b64encode(b"\x00" * 64).decode()}
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
            # Signed over a different version than the request carries.
            {"expected_version": 1,
             "signature": _authorization(self.private, "u1", "d1", 2)},
            # Raw 64 zero bytes.
            {"expected_version": 1,
             "signature": base64.b64encode(b"\x00" * 64).decode()},
        ]
        for payload in cases:
            with self.subTest(payload=payload):
                with self.assertRaises(ServiceError) as ctx:
                    self.service.revoke_device_verified("d1", payload)
                self.assertEqual(ctx.exception.status_code, 400)
                self.assertEqual(ctx.exception.field, "signature")

    def test_revoked_device_still_requires_authorization(self) -> None:
        self.service.revoke_device("d1")
        # A bad/encoding-valid signature is still refused.
        with self.assertRaises(ServiceError) as ctx:
            self.service.revoke_device_verified(
                "d1", {"expected_version": 1,
                       "signature": base64.b64encode(b"\x00" * 64).decode()})
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.field, "signature")
        # A stale version is still refused.
        with self.assertRaises(ServiceError) as ctx:
            self.service.revoke_device_verified(
                "d1", self._payload(expected_version=5))
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "expected_version")
        events = self.service.list_key_events("d1", 0, 100)["events"]
        self.assertEqual([e["type"] for e in events],
                         ["registered", "device_revoked"])

    def test_failure_leaves_state_untouched(self) -> None:
        with self.assertRaises(ServiceError):
            self.service.revoke_device_verified(
                "d1", self._payload(private=_new_identity()[0]))
        device = self._device()
        self.assertFalse(device.revoked)
        self.assertFalse(any(pk.revoked for pk in device.prekeys))
        events = self.service.list_key_events("d1", 0, 100)["events"]
        self.assertEqual([e["type"] for e in events], ["registered"])

    def test_does_not_touch_other_device(self) -> None:
        bob_private, bob_key = _new_identity()
        self.service.register(_register_payload(
            device_id="bob", identity_key=bob_key,
            prekeys=[{"key_id": "p1", "public_key": _x25519_b64()}]))
        self.service.revoke_device_verified("d1", self._payload())
        bob = self.service.store.find_by_device_id("bob")
        self.assertFalse(bob.revoked)
        self.assertFalse(bob.prekeys[0].revoked)
        self.assertEqual(bob.identity_key_version, 1)
        events = self.service.list_key_events("bob", 0, 100)["events"]
        self.assertEqual([e["type"] for e in events], ["registered"])

    def test_concurrent_with_rotation_only_one_commits(self) -> None:
        # A version-1 revocation authorization racing a v1->v2 rotation is
        # accepted only when it is still valid at the commit instant: either
        # the revocation lands first (rotation then sees a revoked device)
        # or the rotation lands first (the stale authorization gets 409).
        for _ in range(25):
            service = DeviceService()
            private, identity = _new_identity()
            service.register(_register_payload(identity_key=identity))
            _, new_key = _new_identity()
            rotation_payload = {
                "identity_key": new_key, "expected_version": 1,
                "signature": _rotation_authorization(
                    private, "u1", "d1", new_key, 1)}
            revocation_payload = {
                "expected_version": 1,
                "signature": _authorization(private, "u1", "d1", 1)}
            outcomes = {}
            barrier = threading.Barrier(2)

            def rotate() -> None:
                barrier.wait()
                try:
                    service.rotate_identity_key_verified(
                        "d1", rotation_payload)
                    outcomes["rotation"] = "ok"
                except ServiceError as error:
                    outcomes["rotation"] = (error.status_code, error.field)

            def revoke() -> None:
                barrier.wait()
                try:
                    service.revoke_device_verified(
                        "d1", revocation_payload)
                    outcomes["revocation"] = "ok"
                except ServiceError as error:
                    outcomes["revocation"] = (error.status_code, error.field)

            threads = [threading.Thread(target=rotate),
                       threading.Thread(target=revoke)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=5)
            self.assertEqual(set(outcomes), {"rotation", "revocation"})
            self.assertEqual(sum(1 for value in outcomes.values()
                                if value == "ok"), 1, outcomes)
            device = service.store.find_by_device_id("d1")
            types = [e["type"] for e in
                     service.list_key_events("d1", 0, 100)["events"]]
            if outcomes["revocation"] == "ok":
                self.assertTrue(device.revoked)
                self.assertEqual(outcomes["rotation"][1], "device_id")
                self.assertEqual(types,
                                 ["registered", "device_revoked"])
            else:
                self.assertFalse(device.revoked)
                self.assertEqual(device.identity_key_version, 2)
                self.assertEqual(outcomes["revocation"],
                                 (409, "expected_version"))
                self.assertEqual(types,
                                 ["registered", "identity_rotated"])


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

    def _payload(self, expected_version=1, **kwargs) -> dict:
        return {"expected_version": expected_version,
                "signature": _authorization(
                    self.private, "u1", "d1", expected_version),
                **kwargs}

    def test_revoke_verified_200(self) -> None:
        status, body = self._request(
            "POST", "/v1/devices/d1/revoke-verified", self._payload())
        self.assertEqual(status, 200)
        self.assertEqual(body, {"device_id": "d1", "revoked": True})
        status, shown = self._request("GET", "/v1/devices/d1")
        self.assertEqual(status, 200)
        self.assertEqual(shown["prekey_ids"], [])

    def test_replay_is_200(self) -> None:
        payload = self._payload()
        first = self._request("POST", "/v1/devices/d1/revoke-verified",
                              payload)
        second = self._request("POST", "/v1/devices/d1/revoke-verified",
                               payload)
        self.assertEqual(first[0], 200)
        self.assertEqual(second[0], 200)
        self.assertEqual(second[1], first[1])

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

    def test_empty_body_400_request_body(self) -> None:
        status, body = self._request(
            "POST", "/v1/devices/d1/revoke-verified")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "request_body")

    def test_bool_version_400(self) -> None:
        payload = self._payload()
        payload["expected_version"] = True
        status, body = self._request(
            "POST", "/v1/devices/d1/revoke-verified", payload)
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "expected_version")

    def test_missing_signature_400(self) -> None:
        status, body = self._request(
            "POST", "/v1/devices/d1/revoke-verified",
            {"expected_version": 1})
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

    def test_extra_fields_ignored(self) -> None:
        status, body = self._request(
            "POST", "/v1/devices/d1/revoke-verified",
            self._payload(extra="ignored"))
        self.assertEqual(status, 200)
        self.assertEqual(body, {"device_id": "d1", "revoked": True})

    def test_percent_encoded_device_id_decodes_for_signature(self) -> None:
        # A percent-encoded slash decodes to an ordinary '/' that is part of
        # the device id; the signed message carries the decoded id verbatim.
        device_id = "team/a"
        private, identity = _new_identity()
        self.service.register(_register_payload(
            device_id=device_id, identity_key=identity))
        payload = {"expected_version": 1,
                   "signature": _authorization(private, "u1", device_id, 1)}
        status, body = self._request(
            "POST", "/v1/devices/team%2Fa/revoke-verified", payload)
        self.assertEqual(status, 200, body)
        self.assertEqual(body, {"device_id": device_id, "revoked": True})

    def test_percent_encoded_id_signature_must_match_decoded_id(self) -> None:
        device_id = "team/a"
        private, identity = _new_identity()
        self.service.register(_register_payload(
            device_id=device_id, identity_key=identity))
        # Signing the still-encoded segment must fail: only the decoded id
        # is signed.
        payload = {"expected_version": 1,
                   "signature": _authorization(private, "u1", "team%2Fa", 1)}
        status, body = self._request(
            "POST", "/v1/devices/team%2Fa/revoke-verified", payload)
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "signature")

    def test_bad_path_shape_404(self) -> None:
        status, body = self._request(
            "POST", "/v1/devices/d1/extra/revoke-verified",
            self._payload())
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "device_id")

    def test_plain_revoke_route_still_works(self) -> None:
        status, body = self._request(
            "POST", "/v1/devices/d1/revoke")
        self.assertEqual(status, 200)
        self.assertEqual(body, {"device_id": "d1", "revoked": True})
        # And it needs no authorization.
        status, shown = self._request("GET", "/v1/devices/d1")
        self.assertEqual(shown["prekey_ids"], [])


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
        self.assertEqual(result.stderr, "")
        line = result.stdout.strip()
        self.assertEqual(line.count("\n"), 0)
        self.assertEqual(json.loads(line),
                         {"device_id": "d1", "revoked": True})

    def test_signature_at_path(self) -> None:
        signature = _authorization(self.private, "u1", "d1", 1)
        path = tempfile.mktemp(suffix=".b64")
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(signature + "\n")
        result = self._run("revoke-device-verified",
                           "--device-id", "d1",
                           "--expected-version", "1",
                           "--signature", f"@{path}")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout.strip()),
                         {"device_id": "d1", "revoked": True})

    def test_version_mismatch_failure_stderr_nonzero(self) -> None:
        signature = _authorization(self.private, "u1", "d1", 1)
        result = self._run("revoke-device-verified",
                           "--device-id", "d1",
                           "--expected-version", "9",
                           "--signature", signature)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        self.assertEqual(json.loads(result.stderr.strip())["field"],
                         "expected_version")

    def test_bad_signature_exit_1(self) -> None:
        result = self._run("revoke-device-verified",
                           "--device-id", "d1",
                           "--expected-version", "1",
                           "--signature",
                           base64.b64encode(b"\x00" * 64).decode())
        self.assertEqual(result.returncode, 1)
        self.assertEqual(json.loads(result.stderr.strip())["field"],
                         "signature")

    def test_unknown_device_exit_1(self) -> None:
        signature = _authorization(self.private, "u1", "d1", 1)
        result = self._run("revoke-device-verified",
                           "--device-id", "ghost",
                           "--expected-version", "1",
                           "--signature", signature)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(json.loads(result.stderr.strip())["field"],
                         "device_id")

    def test_plain_revoke_device_command_still_works(self) -> None:
        result = self._run("revoke-device", "--device-id", "d1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout.strip()),
                         {"device_id": "d1", "revoked": True})


class RevokeVerifiedPersistenceTest(unittest.TestCase):
    def _service_with_file(self):
        path = tempfile.mktemp(suffix=".json")
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        self.addCleanup(lambda: os.path.exists(path + ".integrity")
                        and os.unlink(path + ".integrity"))
        service = DeviceService()
        store = attach_persistence(service, path)
        return path, service, store

    def test_revocation_survives_restart(self) -> None:
        path, first, _store = self._service_with_file()
        private, identity = _new_identity()
        first.register(_register_payload(identity_key=identity))
        first.revoke_device_verified(
            "d1", {"expected_version": 1,
                   "signature": _authorization(private, "u1", "d1", 1)})

        second = DeviceService()
        attach_persistence(second, path)
        device = second.store.find_by_device_id("d1")
        self.assertTrue(device.revoked)
        self.assertTrue(all(pk.revoked for pk in device.prekeys))
        self.assertEqual(device.identity_key, identity)
        self.assertEqual(device.identity_key_version, 1)
        events = second.list_key_events("d1", 0, 100)["events"]
        self.assertEqual([e["type"] for e in events],
                         ["registered", "device_revoked"])
        # After restart an already-revoked device still verifies
        # authorizations; the valid replay succeeds and writes nothing.
        body = second.revoke_device_verified(
            "d1", {"expected_version": 1,
                   "signature": _authorization(private, "u1", "d1", 1)})
        self.assertEqual(body, {"device_id": "d1", "revoked": True})
        self.assertEqual(
            [e["type"] for e in
             second.list_key_events("d1", 0, 100)["events"]],
            ["registered", "device_revoked"])
        # ... and an invalid one is still refused after restart.
        with self.assertRaises(ServiceError) as ctx:
            second.revoke_device_verified(
                "d1", {"expected_version": 1,
                       "signature": base64.b64encode(b"\x00" * 64).decode()})
        self.assertEqual(ctx.exception.field, "signature")

    def test_failure_advances_no_generation_and_leaves_no_event(self) -> None:
        _path, service, store = self._service_with_file()
        private, identity = _new_identity()
        service.register(_register_payload(identity_key=identity))
        generation_after_register = store.commit_seq

        other, _ = _new_identity()
        with self.assertRaises(ServiceError):
            service.revoke_device_verified(
                "d1", {"expected_version": 1,
                       "signature": _authorization(other, "u1", "d1", 1)})
        self.assertEqual(store.commit_seq, generation_after_register)
        device = service.store.find_by_device_id("d1")
        self.assertFalse(device.revoked)
        self.assertFalse(any(pk.revoked for pk in device.prekeys))
        events = service.list_key_events("d1", 0, 100)["events"]
        self.assertEqual([e["type"] for e in events], ["registered"])

    def test_valid_replay_advances_no_generation(self) -> None:
        _path, service, store = self._service_with_file()
        private, identity = _new_identity()
        service.register(_register_payload(identity_key=identity))
        payload = {"expected_version": 1,
                   "signature": _authorization(private, "u1", "d1", 1)}
        service.revoke_device_verified("d1", payload)
        generation_after_revoke = store.commit_seq
        service.revoke_device_verified("d1", payload)
        service.revoke_device_verified("d1", payload)
        self.assertEqual(store.commit_seq, generation_after_revoke)

    def test_persist_failure_rolls_back_the_revocation(self) -> None:
        from e2ee_backend.persistence import PersistenceUnavailable

        _path, service, state_store = self._service_with_file()
        private, identity = _new_identity()
        service.register(_register_payload(identity_key=identity))

        def fail_save(_pending) -> None:
            raise OSError("simulated disk failure")

        state_store.save = fail_save  # type: ignore[assignment]
        # The durable write fails inside the locked transaction: the call
        # raises PersistenceUnavailable (the HTTP layer answers
        # 503/data_file) and the in-memory mutation is rolled back to the
        # last committed state.
        with self.assertRaises(PersistenceUnavailable):
            service.revoke_device_verified(
                "d1", {"expected_version": 1,
                       "signature": _authorization(private, "u1", "d1", 1)})
        device = service.store.find_by_device_id("d1")
        self.assertFalse(device.revoked)
        self.assertFalse(any(pk.revoked for pk in device.prekeys))
        self.assertEqual(device.identity_key_version, 1)
        events = service.list_key_events("d1", 0, 100)["events"]
        self.assertEqual([e["type"] for e in events], ["registered"])


if __name__ == "__main__":
    unittest.main()
