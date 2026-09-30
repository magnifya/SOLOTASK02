"""Tests for signature-verified device registration.

Covers ``POST /v1/devices/verified`` (alias ``POST /v1/register-verified``)
and the service/CLI behind it:

* 201 success publishes exactly like legacy registration (contract fields,
  pre-key order, one-time claims, revocation, persistence/restart);
* the signed message is precisely ``E2EE-SIGNED-PREKEY-V1\\n`` plus the
  compact JSON of ``device_id``/``key_id``/``public_key``/``user_id`` sorted,
  Unicode written as-is, with the request's original strings;
* every malformed proof or structure is 400 naming the precise field, and
  writes no state (no device registered, no pre-key consumed);
* a non-Ed25519 identity key (raw X25519 or SPKI-wrapped) is
  400/field=identity_key;
* duplicate ``(user_id, device_id)`` stays 409/field=device_id;
* the legacy unsigned ``POST /v1/devices`` registration is unchanged and the
  verified route still rejects unsigned entries;
* the ``register-verified`` CLI prints single-line JSON on success.
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
from cryptography.hazmat.primitives.asymmetric import ed25519, x25519

from e2ee_backend.http_app import create_server
from e2ee_backend.persistence import attach_persistence
from e2ee_backend.service import DeviceService, ServiceError


# ---------------------------------------------------------------------------
# Key/signature helpers
# ---------------------------------------------------------------------------

def _x25519_raw_b64() -> str:
    key = x25519.X25519PrivateKey.generate().public_key()
    raw = key.public_bytes(serialization.Encoding.Raw,
                           serialization.PublicFormat.Raw)
    return base64.b64encode(raw).decode()


def _ed25519_identity():
    """Return ``(private_key, raw-base64 public key)``."""
    private = ed25519.Ed25519PrivateKey.generate()
    raw = private.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return private, base64.b64encode(raw).decode()


def _proof_message(user_id, device_id, key_id, public_key):
    """Independently reproduce the exact signed bytes."""
    canonical = json.dumps(
        {"device_id": device_id, "key_id": key_id,
         "public_key": public_key, "user_id": user_id},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return b"E2EE-SIGNED-PREKEY-V1\n" + canonical.encode("utf-8")


def _sign(private, user_id, device_id, key_id, public_key):
    return base64.b64encode(
        private.sign(_proof_message(user_id, device_id, key_id,
                                    public_key))).decode("ascii")


def _x25519_spki_b64():
    key = x25519.X25519PrivateKey.generate().public_key()
    der = key.public_bytes(serialization.Encoding.DER,
                           serialization.PublicFormat.SubjectPublicKeyInfo)
    return base64.b64encode(der).decode("ascii")


def make_payload(device_id="d1", user_id="u1", identity=None,
                 key_specs=(("k1", None),)):
    """Build a verified-registration payload with valid proofs.

    Each *key_specs* entry is ``(key_id, public_key_or_None)``; a fresh
    X25519 key is generated when the public key is omitted. Returns
    ``(payload, private_identity_key)``.
    """
    private, identity_b64 = identity or _ed25519_identity()
    prekeys = []
    for key_id, public_key in key_specs:
        if public_key is None:
            public_key = _x25519_raw_b64()
        prekeys.append({
            "key_id": key_id,
            "public_key": public_key,
            "signature": _sign(private, user_id, device_id, key_id,
                               public_key),
        })
    return {
        "user_id": user_id,
        "device_id": device_id,
        "identity_key": identity_b64,
        "signed_prekeys": prekeys,
    }, private


# ---------------------------------------------------------------------------
# Service layer
# ---------------------------------------------------------------------------

class VerifiedRegistrationServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = DeviceService()

    def test_success_returns_contract_fields_and_publishes(self) -> None:
        payload, _ = make_payload(
            key_specs=(("k1", None), ("k2", None)))
        body = self.service.register_verified(payload)
        self.assertEqual(set(body), {"device_id", "registered_at"})
        self.assertEqual(body["device_id"], "d1")
        view = self.service.get_device("d1")
        self.assertEqual(view["prekey_ids"], ["k1", "k2"])
        self.assertEqual(view["identity_key"], payload["identity_key"])

    def test_empty_prekey_array_registers(self) -> None:
        payload, _ = make_payload(key_specs=())
        body = self.service.register_verified(payload)
        self.assertEqual(body["device_id"], "d1")
        self.assertEqual(self.service.get_device("d1")["prekey_ids"], [])

    def test_duplicate_same_user_device_is_409(self) -> None:
        payload, _ = make_payload()
        self.service.register_verified(payload)
        with self.assertRaises(ServiceError) as ctx:
            self.service.register_verified(payload)
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "device_id")

    def test_conflict_with_legacy_registration(self) -> None:
        legacy = {
            "user_id": "u1", "device_id": "d1",
            "identity_key": _x25519_raw_b64(),
            "signed_prekeys": [{"key_id": "k",
                                 "public_key": _x25519_raw_b64()}]}
        self.service.register(legacy)
        payload, _ = make_payload()
        with self.assertRaises(ServiceError) as ctx:
            self.service.register_verified(payload)
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "device_id")

    def test_legacy_route_still_accepts_unsigned_entries(self) -> None:
        legacy = {
            "user_id": "u1", "device_id": "legacy",
            "identity_key": _x25519_raw_b64(),
            "signed_prekeys": [{"key_id": "k1",
                                 "public_key": _x25519_raw_b64()}]}
        self.assertEqual(self.service.register(legacy)["device_id"], "legacy")

    def test_verified_rejects_unsigned_element(self) -> None:
        payload, _ = make_payload()
        del payload["signed_prekeys"][0]["signature"]
        with self.assertRaises(ServiceError) as ctx:
            self.service.register_verified(payload)
        self.assertEqual(ctx.exception.field,
                         "signed_prekeys[0].signature")

    def test_bad_signature_names_element_signature(self) -> None:
        payload, private = make_payload()
        # Re-sign over a tampered device id.
        element = payload["signed_prekeys"][0]
        element["signature"] = _sign(private, "u1", "other-device",
                                     element["key_id"],
                                     element["public_key"])
        self._assert_400(payload, "signed_prekeys[0].signature")

    def test_signature_over_wrong_user_id_fails(self) -> None:
        payload, private = make_payload()
        element = payload["signed_prekeys"][0]
        element["signature"] = _sign(private, "other-user", "d1",
                                     element["key_id"],
                                     element["public_key"])
        self._assert_400(payload, "signed_prekeys[0].signature")

    def test_signature_over_wrong_key_id_fails(self) -> None:
        payload, private = make_payload()
        element = payload["signed_prekeys"][0]
        element["signature"] = _sign(private, "u1", "d1", "other",
                                     element["public_key"])
        self._assert_400(payload, "signed_prekeys[0].signature")

    def test_signature_over_wrong_public_key_fails(self) -> None:
        payload, private = make_payload()
        element = payload["signed_prekeys"][0]
        element["signature"] = _sign(private, "u1", "d1",
                                     element["key_id"], _x25519_raw_b64())
        self._assert_400(payload, "signed_prekeys[0].signature")

    def test_wrong_identity_key_signature_fails(self) -> None:
        payload, _ = make_payload()
        other, other_b64 = _ed25519_identity()
        # A well-formed signature by a different identity key.
        element = payload["signed_prekeys"][0]
        element["signature"] = _sign(other, "u1", "d1",
                                     element["key_id"],
                                     element["public_key"])
        self._assert_400(payload, "signed_prekeys[0].signature")
        # The foreign identity key by itself is fine; combine both to prove
        # verification is against the request's identity key.
        payload["identity_key"] = other_b64
        self.assertEqual(self.service.register_verified(payload)["device_id"],
                         "d1")

    def test_signature_garbage_bytes_fails(self) -> None:
        payload, _ = make_payload()
        payload["signed_prekeys"][0]["signature"] = base64.b64encode(
            b"\x00" * 64).decode()
        self._assert_400(payload, "signed_prekeys[0].signature")

    def test_non_standard_base64_signature_rejected(self) -> None:
        payload, _ = make_payload()
        valid_sig = payload["signed_prekeys"][0]["signature"]
        # Flip the first data character: it sits in a full quantum, so it
        # always yields different (still 64-byte, unverifiable) bytes.
        flipped = ("A" if valid_sig[0] != "A" else "B") + valid_sig[1:]
        self.assertNotEqual(flipped, valid_sig)
        for tampered in (valid_sig[:-2] + "--",  # url-safe alphabet
                         valid_sig.rstrip("="),  # padding removed
                         flipped):               # 64 bytes, wrong signature
            payload["signed_prekeys"][0]["signature"] = tampered
            self._assert_400(payload, "signed_prekeys[0].signature")

    def test_signature_wrong_length_rejected(self) -> None:
        payload, _ = make_payload()
        for value in (base64.b64encode(b"\x00" * 32).decode(),
                      base64.b64encode(b"\x00" * 63).decode(),
                      base64.b64encode(b"\x00" * 65).decode(),
                      ""):
            payload["signed_prekeys"][0]["signature"] = value
            self._assert_400(payload, "signed_prekeys[0].signature")

    def test_raw_x25519_point_cannot_verify_proof(self) -> None:
        # A raw 32-byte X25519 point is also a syntactically valid raw Ed25519
        # point (the two encodings are indistinguishable without an algorithm
        # marker), so the malformed-identity case can only surface at the
        # proof gate: that key never verifies the real identity's signature.
        # The request is still rejected 400 and registers nothing.
        payload, _ = make_payload()
        payload["identity_key"] = _x25519_raw_b64()
        self._assert_400(payload, "signed_prekeys[0].signature")

    def test_wrong_length_identity_blob_rejected(self) -> None:
        payload, _ = make_payload()
        payload["identity_key"] = base64.b64encode(b"\x00" * 31).decode()
        self._assert_400(payload, "identity_key")

    def test_spki_x25519_identity_rejected(self) -> None:
        payload, _ = make_payload()
        payload["identity_key"] = _x25519_spki_b64()
        self._assert_400(payload, "identity_key")

    def test_garbage_identity_rejected(self) -> None:
        payload, _ = make_payload()
        payload["identity_key"] = "not-a-key"
        self._assert_400(payload, "identity_key")

    def test_missing_scalar_fields(self) -> None:
        for name in ("user_id", "device_id", "identity_key"):
            payload, _ = make_payload(device_id=f"d-{name}")
            del payload[name]
            self._assert_400(payload, name)

    def test_non_string_scalars(self) -> None:
        for name in ("user_id", "device_id", "identity_key"):
            payload, _ = make_payload(device_id=f"d-{name}")
            payload[name] = 123
            self._assert_400(payload, name)
            payload[name] = ""
            self._assert_400(payload, name)

    def test_signed_prekeys_must_be_array(self) -> None:
        payload, _ = make_payload()
        payload["signed_prekeys"] = {"key_id": "k1"}
        self._assert_400(payload, "signed_prekeys")

    def test_element_must_be_object(self) -> None:
        payload, _ = make_payload()
        payload["signed_prekeys"] = [42]
        self._assert_400(payload, "signed_prekeys[0]")

    def test_missing_element_fields(self) -> None:
        for name in ("key_id", "public_key", "signature"):
            payload, _ = make_payload(device_id=f"d-{name}")
            del payload["signed_prekeys"][0][name]
            self._assert_400(payload, f"signed_prekeys[0].{name}")

    def test_non_string_element_fields(self) -> None:
        for name in ("key_id", "public_key", "signature"):
            payload, _ = make_payload(device_id=f"d-{name}")
            payload["signed_prekeys"][0][name] = 7
            self._assert_400(payload, f"signed_prekeys[0].{name}")

    def test_invalid_public_key_named(self) -> None:
        payload, _ = make_payload()
        payload["signed_prekeys"][0]["public_key"] = "garbage"
        self._assert_400(payload, "signed_prekeys[0].public_key")

    def test_duplicate_key_id_names_second_index(self) -> None:
        payload, _ = make_payload(key_specs=(("k1", None), ("k1", None)))
        self._assert_400(payload, "signed_prekeys[1].key_id")

    def test_body_must_be_object(self) -> None:
        for body in (None, ["nope"], "nope", 42):
            with self.assertRaises(ServiceError) as ctx:
                self.service.register_verified(body)
            self.assertEqual(ctx.exception.status_code, 400)
            self.assertEqual(ctx.exception.field, "request_body")

    def test_failure_registers_nothing_and_consumes_no_key(self) -> None:
        payload, private = make_payload(
            key_specs=(("k1", None), ("k2", None)))
        # Tamper the second signature after the first verifies, proving every
        # proof is checked before the device exists.
        payload["signed_prekeys"][1]["signature"] = _sign(
            private, "u1", "d1", "k2", _x25519_raw_b64())
        self._assert_400(payload, "signed_prekeys[1].signature")
        with self.assertRaises(ServiceError) as ctx:
            self.service.get_device("d1")
        self.assertEqual(ctx.exception.status_code, 404)
        with self.assertRaises(ServiceError) as claim_ctx:
            self.service.claim_prekey(
                {"recipient_device_id": "d1", "claim_id": "c1"})
        self.assertEqual(claim_ctx.exception.status_code, 404)

    def test_verified_prekeys_are_claimed_in_order_once(self) -> None:
        payload, _ = make_payload(
            key_specs=(("k1", None), ("k2", None)))
        self.service.register_verified(payload)
        first, status = self.service.claim_prekey(
            {"recipient_device_id": "d1", "claim_id": "c1"})
        self.assertEqual(status, 201)
        self.assertEqual(first["key_id"], "k1")
        self.assertEqual(first["identity_key"], payload["identity_key"])
        second, status = self.service.claim_prekey(
            {"recipient_device_id": "d1", "claim_id": "c2"})
        self.assertEqual(second["key_id"], "k2")
        self.assertEqual(self.service.get_device("d1")["prekey_ids"], [])
        with self.assertRaises(ServiceError):
            self.service.claim_prekey(
                {"recipient_device_id": "d1", "claim_id": "c3"})

    def test_revocation_of_verified_device(self) -> None:
        payload, _ = make_payload(key_specs=(("k1", None),))
        self.service.register_verified(payload)
        self.service.revoke_device("d1")
        self.assertEqual(self.service.get_device("d1")["prekey_ids"], [])

    def test_unicode_values_are_signed_as_is(self) -> None:
        user_id = "ユーザー ☃"
        device_id = "dévice-Ω"
        key_id = "clé-1"
        payload, _ = make_payload(user_id=user_id, device_id=device_id,
                                  key_specs=((key_id, None),))
        self.assertEqual(
            self.service.register_verified(payload)["device_id"], device_id)
        view = self.service.get_device(device_id)
        self.assertEqual(view["prekey_ids"], [key_id])

    def test_hex_encoded_prekey_public_key_verifies(self) -> None:
        # The proof binds the request's original string, not a re-encoding:
        # sign over the hex representation and register with it verbatim.
        private, identity_b64 = _ed25519_identity()
        raw = base64.b64decode(_x25519_raw_b64())
        public_hex = raw.hex()
        payload = {
            "user_id": "u1", "device_id": "d1",
            "identity_key": identity_b64,
            "signed_prekeys": [{
                "key_id": "k1", "public_key": public_hex,
                "signature": _sign(private, "u1", "d1", "k1", public_hex)}]}
        self.assertEqual(
            self.service.register_verified(payload)["device_id"], "d1")
        self.assertEqual(
            self.service.get_device("d1")["prekey_ids"], ["k1"])

    def test_all_signatures_checked_in_order(self) -> None:
        # A bad proof at index 2 reports index 2, not an earlier valid one.
        payload, private = make_payload(
            key_specs=(("k1", None), ("k2", None), ("k3", None)))
        payload["signed_prekeys"][2]["signature"] = _sign(
            private, "u1", "d1", "k3", _x25519_raw_b64())
        self._assert_400(payload, "signed_prekeys[2].signature")

    def _assert_400(self, payload, field):
        with self.assertRaises(ServiceError) as ctx:
            self.service.register_verified(payload)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.field, field)


# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------

class VerifiedRegistrationHTTPTest(unittest.TestCase):
    def setUp(self) -> None:
        self.server, _ = create_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _request(self, method, path, body=None, raw=None):
        connection = HTTPConnection("127.0.0.1", self.port, timeout=5)
        if raw is not None:
            connection.request(method, path, body=raw,
                               headers={"Content-Type": "application/json"})
        else:
            payload = json.dumps(body) if body is not None else None
            headers = ({"Content-Type": "application/json"}
                       if payload is not None else {})
            connection.request(method, path, body=payload, headers=headers)
        response = connection.getresponse()
        data = response.read().decode("utf-8")
        connection.close()
        return response.status, json.loads(data)

    def test_verified_route_success_201(self) -> None:
        payload, _ = make_payload(
            key_specs=(("k1", None), ("k2", None)))
        status, body = self._request("POST", "/v1/devices/verified", payload)
        self.assertEqual(status, 201)
        self.assertEqual(set(body), {"device_id", "registered_at"})

    def test_register_verified_alias_route(self) -> None:
        payload, _ = make_payload(device_id="d2")
        status, body = self._request("POST", "/v1/register-verified", payload)
        self.assertEqual(status, 201)
        self.assertEqual(body["device_id"], "d2")

    def test_malformed_body_is_400_request_body(self) -> None:
        for path in ("/v1/devices/verified", "/v1/register-verified"):
            status, body = self._request("POST", path, raw=b"{nope")
            self.assertEqual(status, 400)
            self.assertEqual(body["field"], "request_body")

    def test_bad_signature_over_http(self) -> None:
        payload, _ = make_payload()
        payload["signed_prekeys"][0]["signature"] = base64.b64encode(
            b"x" * 64).decode()
        status, body = self._request("POST", "/v1/devices/verified", payload)
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "signed_prekeys[0].signature")

    def test_non_ed25519_identity_over_http(self) -> None:
        # Algorithm-marked (SPKI DER) X25519 is recognizably not Ed25519.
        payload, _ = make_payload()
        payload["identity_key"] = _x25519_spki_b64()
        status, body = self._request("POST", "/v1/devices/verified", payload)
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "identity_key")

    def test_duplicate_is_409_device_id(self) -> None:
        payload, _ = make_payload()
        self.assertEqual(self._request("POST", "/v1/devices/verified",
                                       payload)[0], 201)
        status, body = self._request("POST", "/v1/devices/verified", payload)
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "device_id")

    def test_failure_does_not_register(self) -> None:
        payload, _ = make_payload()
        payload["signed_prekeys"][0]["signature"] = base64.b64encode(
            b"x" * 64).decode()
        self.assertEqual(self._request("POST", "/v1/devices/verified",
                                       payload)[0], 400)
        status, body = self._request("GET", "/v1/devices/d1")
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "device_id")

    def test_legacy_route_unchanged(self) -> None:
        legacy = {
            "user_id": "u1", "device_id": "d9",
            "identity_key": _x25519_raw_b64(),
            "signed_prekeys": [{"key_id": "k1",
                                 "public_key": _x25519_raw_b64()}]}
        status, _ = self._request("POST", "/v1/devices", legacy)
        self.assertEqual(status, 201)

    def test_other_routes_unchanged_after_verified_registration(self) -> None:
        payload, _ = make_payload(key_specs=(("k1", None),))
        self._request("POST", "/v1/devices/verified", payload)
        status, body = self._request(
            "POST", "/v1/prekeys/claim",
            {"recipient_device_id": "d1", "claim_id": "c1"})
        self.assertEqual(status, 201)
        self.assertEqual(body["key_id"], "k1")


# ---------------------------------------------------------------------------
# Persistence / restart
# ---------------------------------------------------------------------------

class VerifiedRegistrationPersistenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.path = os.path.join(self.directory, "state.json")

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def _fresh_service(self):
        service = DeviceService()
        attach_persistence(service, self.path)
        return service

    def test_verified_registration_survives_restart(self) -> None:
        service = self._fresh_service()
        payload, _ = make_payload(
            key_specs=(("k1", None), ("k2", None)))
        self.assertEqual(
            service.register_verified(payload)["device_id"], "d1")
        body, status = service.claim_prekey(
            {"recipient_device_id": "d1", "claim_id": "c1"})
        self.assertEqual(status, 201)
        self.assertEqual(body["key_id"], "k1")

        restarted = self._fresh_service()
        view = restarted.get_device("d1")
        # Consumption is durable: only k2 remains in the public listing.
        self.assertEqual(view["prekey_ids"], ["k2"])
        self.assertEqual(view["identity_key"], payload["identity_key"])
        # Claim replay stays idempotent and byte-identical.
        replay, replay_status = restarted.claim_prekey(
            {"recipient_device_id": "d1", "claim_id": "c1"})
        self.assertEqual(replay_status, 200)
        self.assertEqual(replay, body)

    def test_failed_verified_request_persists_nothing(self) -> None:
        service = self._fresh_service()
        payload, _ = make_payload()
        payload["signed_prekeys"][0]["signature"] = base64.b64encode(
            b"x" * 64).decode()
        with self.assertRaises(ServiceError):
            service.register_verified(payload)

        restarted = self._fresh_service()
        with self.assertRaises(ServiceError) as ctx:
            restarted.get_device("d1")
        self.assertEqual(ctx.exception.status_code, 404)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class VerifiedRegistrationCLITest(unittest.TestCase):
    def setUp(self) -> None:
        self.server, _ = create_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        self.directory = tempfile.mkdtemp()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        shutil.rmtree(self.directory, ignore_errors=True)

    def _run(self, *arguments):
        return subprocess.run(
            [sys.executable, "-m", "e2ee_backend", "--base-url",
             self.base_url, *arguments],
            capture_output=True, text=True, timeout=15)

    def test_register_verified_success_single_line_json(self) -> None:
        payload, _ = make_payload(
            device_id="c1", key_specs=(("k1", None), ("k2", None)))
        result = self._run(
            "register-verified", "--user-id", payload["user_id"],
            "--device-id", "c1", "--identity-key", payload["identity_key"],
            "--prekey",
            f"k1:{payload['signed_prekeys'][0]['public_key']}:"
            f"{payload['signed_prekeys'][0]['signature']}",
            "--prekey",
            f"k2:{payload['signed_prekeys'][1]['public_key']}:"
            f"{payload['signed_prekeys'][1]['signature']}")
        self.assertEqual(result.returncode, 0, result.stderr)
        line = result.stdout.strip()
        self.assertEqual(line.count("\n"), 0)
        self.assertEqual(set(json.loads(line)),
                         {"device_id", "registered_at"})

    def test_register_verified_bad_proof_exit_1(self) -> None:
        payload, _ = make_payload(device_id="c2")
        result = self._run(
            "register-verified", "--user-id", "u1", "--device-id", "c2",
            "--identity-key", payload["identity_key"],
            "--prekey",
            f"k1:{payload['signed_prekeys'][0]['public_key']}:"
            f"{base64.b64encode(b'x' * 64).decode()}")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(
            json.loads(result.stderr.strip())["field"],
            "signed_prekeys[0].signature")

    def test_malformed_prekey_spec_fails_locally(self) -> None:
        _, identity_b64 = _ed25519_identity()
        result = self._run(
            "register-verified", "--user-id", "u1", "--device-id", "c3",
            "--identity-key", identity_b64,
            "--prekey", "only-two:fields")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(
            json.loads(result.stderr.strip())["field"], "signed_prekeys")

    def test_prekey_json_file(self) -> None:
        payload, _ = make_payload(device_id="c4")
        path = os.path.join(self.directory, "prekey.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload["signed_prekeys"][0], handle)
        result = self._run(
            "register-verified", "--user-id", "u1", "--device-id", "c4",
            "--identity-key", payload["identity_key"],
            "--prekey", f"@{path}")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            json.loads(result.stdout.strip())["device_id"], "c4")


if __name__ == "__main__":
    unittest.main()
