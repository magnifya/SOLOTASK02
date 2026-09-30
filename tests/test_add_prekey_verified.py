"""Tests for identity-authorized post-registration pre-key replenishment.

Covers POST /v1/devices/{device_id}/prekeys/verified at the service, HTTP
(real loopback socket), CLI (real subprocess), persistence-recovery and
concurrency layers. The entry is the post-registration counterpart of the
ordinary ``.../prekeys`` add: the three-field body additionally carries a
standard-base64 64-byte Ed25519 ``signature`` verified against the device's
*current* identity key over the same E2EE-SIGNED-PREKEY-V1 canonical proof
that verified registration publishes. A verified new id appends (201); an
identical stored entry replays idempotently (200); a changed value or revoked
id conflicts (409/key_id). Every failure names the exact field and writes
nothing.
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

from e2ee_backend.crypto import signed_prekey_proof_message
from e2ee_backend.http_app import create_server
from e2ee_backend.persistence import PersistenceUnavailable, attach_persistence
from e2ee_backend.service import DeviceService, ServiceError


def _raw_b64(key) -> str:
    raw = key.public_bytes(serialization.Encoding.Raw,
                           serialization.PublicFormat.Raw)
    return base64.b64encode(raw).decode()


def _der_b64(key) -> str:
    der = key.public_bytes(serialization.Encoding.DER,
                           serialization.PublicFormat.SubjectPublicKeyInfo)
    return base64.b64encode(der).decode()


def _new_identity():
    private = ed25519.Ed25519PrivateKey.generate()
    return private, _raw_b64(private.public_key())


def _new_prekey() -> str:
    return _raw_b64(x25519.X25519PrivateKey.generate().public_key())


def _proof(private, user_id, device_id, key_id, public_key) -> str:
    message = signed_prekey_proof_message(
        user_id, device_id, key_id, public_key)
    return base64.b64encode(private.sign(message)).decode()


class _Fixture:
    """A registered verified device plus its signing identity."""

    def __init__(self, service, user_id="u1", device_id="d1",
                 prekeys=("k1",)):
        self.service = service
        self.user_id = user_id
        self.device_id = device_id
        private, identity_key = _new_identity()
        self.private = private
        self.identity_key = identity_key
        entries = []
        for key_id in prekeys:
            public_key = _new_prekey()
            entries.append({
                "key_id": key_id, "public_key": public_key,
                "signature": _proof(private, user_id, device_id,
                                    key_id, public_key)})
        service.register_verified({
            "user_id": user_id, "device_id": device_id,
            "identity_key": self.identity_key,
            "signed_prekeys": entries})

    def add_body(self, key_id, public_key=None, signature=None):
        if public_key is None:
            public_key = _new_prekey()
        if signature is None:
            signature = _proof(self.private, self.user_id, self.device_id,
                               key_id, public_key)
        return {"key_id": key_id, "public_key": public_key,
                "signature": signature}

    def add(self, key_id, public_key=None, signature=None):
        return self.service.add_prekey_verified(
            self.device_id, self.add_body(key_id, public_key, signature))


class AddPrekeyVerifiedServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = DeviceService()
        self.fx = _Fixture(self.service)

    # -- success / idempotency / conflicts ---------------------------------

    def test_new_id_appends_in_order_201(self) -> None:
        body = self.fx.add_body("k2")
        view, status = self.service.add_prekey_verified("d1", body)
        self.assertEqual(status, 201)
        self.assertEqual(view, {"device_id": "d1", "key_id": "k2",
                                "public_key": body["public_key"]})
        self.assertEqual(self.service.get_device("d1")["prekey_ids"],
                         ["k1", "k2"])

    def test_same_id_same_key_not_revoked_is_200_idempotent(self) -> None:
        body = self.fx.add_body("k1", public_key=self.k1_public)
        view, status = self.service.add_prekey_verified("d1", body)
        self.assertEqual(status, 200)
        self.assertEqual(view, {"device_id": "d1", "key_id": "k1",
                                "public_key": body["public_key"]})
        self.assertEqual(self.service.get_device("d1")["prekey_ids"], ["k1"])

    def test_idempotent_replay_needs_no_valid_signature_after_rotation(self) -> None:
        # Once an authorized key is committed, the replay is resolved from the
        # stored tuple exactly like the ordinary entry and is not re-verified
        # against a subsequently rotated identity.
        body = self.fx.add_body("k2")
        self.assertEqual(self.service.add_prekey_verified("d1", body)[1], 201)
        other, other_ident = _new_identity()
        self.service.rotate_identity_key(
            "d1", {"identity_key": other_ident})
        # Same stored tuple, signature now signed by the *old* identity — yet
        # the committed entry replays with 200.
        _, status = self.service.add_prekey_verified("d1", body)
        self.assertEqual(status, 200)

    def test_same_id_changed_value_is_409_key_id(self) -> None:
        body = self.fx.add_body("k1")
        with self.assertRaises(ServiceError) as ctx:
            self.service.add_prekey_verified("d1", body)
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "key_id")

    def test_revoked_id_is_409_key_id_even_with_a_valid_proof(self) -> None:
        body = self.fx.add_body("k1", public_key=self.k1_public)
        self.service.revoke_prekey("d1", "k1")
        with self.assertRaises(ServiceError) as ctx:
            self.service.add_prekey_verified("d1", body)
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "key_id")

    # -- body / field validation -------------------------------------------

    def test_body_must_be_object(self) -> None:
        for payload in (None, [], "x", 42):
            with self.subTest(payload=payload):
                with self.assertRaises(ServiceError) as ctx:
                    self.service.add_prekey_verified("d1", payload)
                self.assertEqual(ctx.exception.status_code, 400)
                self.assertEqual(ctx.exception.field, "request_body")

    def test_field_validation_order_and_paths(self) -> None:
        # A well-formed signature lets the public_key-parsing step be reached.
        good_sig = base64.b64encode(b"\x00" * 64).decode()
        cases = [
            ({}, "key_id"),
            ({"key_id": "x"}, "public_key"),
            ({"key_id": "", "public_key": "p"}, "key_id"),
            ({"key_id": 7, "public_key": "p"}, "key_id"),
            ({"key_id": None, "public_key": "p"}, "key_id"),
            ({"key_id": "x", "public_key": ""}, "public_key"),
            ({"key_id": "x", "public_key": 9}, "public_key"),
            ({"key_id": "x", "public_key": ["p"]}, "public_key"),
            # Presence/types are checked before key parsing, so a missing
            # signature is reported as signature even when public_key is
            # malformed (same ordering as verified registration).
            ({"key_id": "x", "public_key": "not-a-key"}, "signature"),
            # With a present, decodable signature the illegal public_key is
            # reached first (public_key precedes signature verification).
            ({"key_id": "x", "public_key": "not-a-key",
              "signature": good_sig}, "public_key"),
            ({"key_id": "x", "public_key": self.k1_public}, "signature"),
            ({"key_id": "x", "public_key": self.k1_public,
              "signature": ""}, "signature"),
            ({"key_id": "x", "public_key": self.k1_public,
              "signature": 11}, "signature"),
            ({"key_id": "x", "public_key": self.k1_public,
              "signature": None}, "signature"),
        ]
        for payload, field in cases:
            with self.subTest(payload=payload):
                with self.assertRaises(ServiceError) as ctx:
                    self.service.add_prekey_verified("d1", payload)
                self.assertEqual(ctx.exception.status_code, 400)
                self.assertEqual(ctx.exception.field, field)

    def test_signature_bad_base64(self) -> None:
        body = self.fx.add_body("k2")
        body["signature"] = "not base64!"
        self._assert_field(body, "signature")

    def test_signature_wrong_length(self) -> None:
        body = self.fx.add_body("k2")
        body["signature"] = base64.b64encode(b"\x00" * 32).decode()
        self._assert_field(body, "signature")

    def test_signature_non_canonical_base64(self) -> None:
        body = self.fx.add_body("k2")
        self.assertTrue(body["signature"].endswith("="))
        body["signature"] = body["signature"].rstrip("=")
        self._assert_field(body, "signature")

    def test_signature_urlsafe_alphabet_rejected(self) -> None:
        body = self.fx.add_body("k2")
        # Force a '-'/'_' spelling by re-encoding URL-safe; it is canonical for
        # its own alphabet but not the required standard alphabet.
        raw = base64.b64decode(body["signature"])
        body["signature"] = base64.urlsafe_b64encode(raw).decode()
        self._assert_field(body, "signature")

    # -- proof verification -------------------------------------------------

    def test_tampered_signature_fails(self) -> None:
        body = self.fx.add_body("k2")
        raw = base64.b64decode(body["signature"])
        body["signature"] = base64.b64encode(
            bytes([raw[0] ^ 0x01]) + raw[1:]).decode()
        self._assert_field(body, "signature")

    def test_proof_over_other_device_fails(self) -> None:
        public_key = _new_prekey()
        body = {"key_id": "k2", "public_key": public_key,
                "signature": _proof(self.fx.private, "u1", "other-device",
                                    "k2", public_key)}
        self._assert_field(body, "signature")

    def test_proof_over_other_user_fails(self) -> None:
        public_key = _new_prekey()
        body = {"key_id": "k2", "public_key": public_key,
                "signature": _proof(self.fx.private, "other-user", "d1",
                                    "k2", public_key)}
        self._assert_field(body, "signature")

    def test_proof_signed_by_other_identity_fails(self) -> None:
        stranger, _ = _new_identity()
        public_key = _new_prekey()
        body = {"key_id": "k2", "public_key": public_key,
                "signature": _proof(stranger, "u1", "d1", "k2", public_key)}
        self._assert_field(body, "signature")

    def test_unicode_strings_participate_verbatim(self) -> None:
        service = DeviceService()
        private, identity_key = _new_identity()
        user_id, device_id, key_id = "用户", "dæ-1", "密钥①"
        public_key = _new_prekey()
        service.register_verified({
            "user_id": user_id, "device_id": device_id,
            "identity_key": identity_key,
            "signed_prekeys": [{"key_id": "seed", "public_key": public_key,
                                "signature": _proof(private, user_id,
                                                    device_id, "seed",
                                                    public_key)}]})
        added = _new_prekey()
        body = {"key_id": key_id, "public_key": added,
                "signature": _proof(private, user_id, device_id,
                                    key_id, added)}
        _, status = service.add_prekey_verified(device_id, body)
        self.assertEqual(status, 201)

    def test_der_ed25519_identity_accepted(self) -> None:
        service = DeviceService()
        private = ed25519.Ed25519PrivateKey.generate()
        public_key = _new_prekey()
        service.register_verified({
            "user_id": "u", "device_id": "d",
            "identity_key": _der_b64(private.public_key()),
            "signed_prekeys": [{"key_id": "k0", "public_key": public_key,
                                "signature": _proof(private, "u", "d",
                                                    "k0", public_key)}]})
        body = {"key_id": "k1", "public_key": _new_prekey(),
                "signature": None}
        body["signature"] = _proof(private, "u", "d", "k1",
                                   body["public_key"])
        _, status = service.add_prekey_verified("d", body)
        self.assertEqual(status, 201)

    # -- current identity key ----------------------------------------------

    def test_non_ed25519_current_identity_is_400_identity_key(self) -> None:
        # Rotate to an X25519 DER key (a DER SPKI carries the OID, so unlike a
        # raw 32-byte point it is unambiguously not Ed25519).
        x25519_der = _der_b64(
            x25519.X25519PrivateKey.generate().public_key())
        self.service.rotate_identity_key(
            "d1", {"identity_key": x25519_der})
        body = self.fx.add_body("k2")
        with self.assertRaises(ServiceError) as ctx:
            self.service.add_prekey_verified("d1", body)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.field, "identity_key")

    def test_proof_must_use_current_identity_after_rotation(self) -> None:
        new_private, new_identity = _new_identity()
        self.service.rotate_identity_key(
            "d1", {"identity_key": new_identity})
        # A proof signed by the pre-rotation identity now fails...
        old_proof = self.fx.add_body("k2")
        self._assert_field(old_proof, "signature")
        # ...while one signed by the current identity succeeds.
        public_key = _new_prekey()
        body = {"key_id": "k2", "public_key": public_key,
                "signature": _proof(new_private, "u1", "d1",
                                    "k2", public_key)}
        _, status = self.service.add_prekey_verified("d1", body)
        self.assertEqual(status, 201)

    def test_legacy_x25519_device_cannot_authorize(self) -> None:
        # A device registered through the ordinary entry with an X25519 DER
        # identity has no Ed25519 identity to authorize with.
        service = DeviceService()
        x25519_der = _der_b64(
            x25519.X25519PrivateKey.generate().public_key())
        service.register({
            "user_id": "u", "device_id": "legacy",
            "identity_key": x25519_der,
            "signed_prekeys": [{"key_id": "k0",
                                "public_key": _new_prekey()}]})
        private, _ = _new_identity()
        public_key = _new_prekey()
        body = {"key_id": "k1", "public_key": public_key,
                "signature": _proof(private, "u", "legacy",
                                    "k1", public_key)}
        with self.assertRaises(ServiceError) as ctx:
            service.add_prekey_verified("legacy", body)
        self.assertEqual(ctx.exception.field, "identity_key")

    # -- device lookup ------------------------------------------------------

    def test_unknown_device_is_404(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.add_prekey_verified(
                "ghost", self.fx.add_body("k2"))
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "device_id")

    def test_revoked_device_is_409(self) -> None:
        self.service.revoke_device("d1")
        with self.assertRaises(ServiceError) as ctx:
            self.service.add_prekey_verified(
                "d1", self.fx.add_body("k2"))
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "device_id")

    # -- audit chain / failure atomicity -----------------------------------

    def test_success_appends_one_prekey_added_event(self) -> None:
        before = self.service.list_key_events("d1", 0, 100)["events"]
        self.service.add_prekey_verified("d1", self.fx.add_body("k2"))
        after = self.service.list_key_events("d1", 0, 100)["events"]
        self.assertEqual(len(after), len(before) + 1)
        event = after[-1]
        self.assertEqual(event["type"], "prekey_added")
        self.assertEqual(event["device_id"], "d1")
        self.assertTrue(event["hash"])
        self.assertEqual(event["prev_hash"], before[-1]["hash"])

    def test_failure_appends_no_event_and_adds_no_key(self) -> None:
        before = self.service.list_key_events("d1", 0, 100)["events"]
        ids_before = self.service.get_device("d1")["prekey_ids"]
        bad = self.fx.add_body("k2")
        raw = base64.b64decode(bad["signature"])
        bad["signature"] = base64.b64encode(
            bytes([raw[0] ^ 1]) + raw[1:]).decode()
        self._assert_field(bad, "signature")
        self.assertEqual(
            self.service.list_key_events("d1", 0, 100)["events"], before)
        self.assertEqual(self.service.get_device("d1")["prekey_ids"],
                         ids_before)

    def _assert_field(self, payload, field) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.add_prekey_verified("d1", payload)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.field, field)

    @property
    def k1_public(self) -> str:
        return self.service.store.find_by_device_id("d1").prekeys[0].public_key


class AddPrekeyVerifiedHTTPTest(unittest.TestCase):
    def setUp(self) -> None:
        self.server, self.service = create_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        self.fx = _Fixture(self.service)

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _request(self, method: str, path: str, body: object = None,
                 raw: bytes | None = None):
        connection = HTTPConnection("127.0.0.1", self.port, timeout=5)
        if raw is not None:
            connection.request(method, path, body=raw,
                               headers={"Content-Type": "application/json"})
        else:
            payload = json.dumps(body) if body is not None else None
            connection.request(
                method, path, body=payload,
                headers={"Content-Type": "application/json"} if payload else {})
        response = connection.getresponse()
        data = response.read().decode("utf-8")
        connection.close()
        return response.status, json.loads(data)

    PATH = "/v1/devices/d1/prekeys/verified"

    def test_success_is_201_then_replay_200(self) -> None:
        body = self.fx.add_body("k2")
        status, first = self._request("POST", self.PATH, body)
        self.assertEqual(status, 201)
        self.assertEqual(set(first), {"device_id", "key_id", "public_key"})
        self.assertEqual(first["key_id"], "k2")
        status, second = self._request("POST", self.PATH, body)
        self.assertEqual(status, 200)
        self.assertEqual(second, first)

    def test_malformed_json_is_400_request_body(self) -> None:
        status, body = self._request("POST", self.PATH, raw=b"{nope")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "request_body")

    def test_non_object_body_is_400_request_body(self) -> None:
        status, body = self._request("POST", self.PATH, [1, 2])
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "request_body")

    def test_changed_value_is_409_key_id(self) -> None:
        status, body = self._request(
            "POST", self.PATH, self.fx.add_body("k1"))
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "key_id")

    def test_bad_signature_is_400_signature(self) -> None:
        body = self.fx.add_body("k2")
        body["signature"] = "garbage"
        status, response = self._request("POST", self.PATH, body)
        self.assertEqual(status, 400)
        self.assertEqual(response["field"], "signature")

    def test_failed_proof_is_400_signature(self) -> None:
        body = self.fx.add_body("k2")
        raw = base64.b64decode(body["signature"])
        body["signature"] = base64.b64encode(
            bytes([raw[0] ^ 1]) + raw[1:]).decode()
        status, response = self._request("POST", self.PATH, body)
        self.assertEqual(status, 400)
        self.assertEqual(response["field"], "signature")

    def test_unknown_device_is_404_device_id(self) -> None:
        status, body = self._request(
            "POST", "/v1/devices/ghost/prekeys/verified",
            self.fx.add_body("k2"))
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "device_id")

    def test_bad_path_shape_is_404(self) -> None:
        status, body = self._request(
            "POST", "/v1/devices/prekeys/verified",
            self.fx.add_body("k2"))
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "device_id")

    def test_percent_encoded_device_id(self) -> None:
        from urllib.parse import quote
        fx = _Fixture(self.service, device_id="d/æ")
        body = fx.add_body("k2")
        path = (f"/v1/devices/{quote(fx.device_id, safe='')}"
                f"/prekeys/verified")
        status, response = self._request("POST", path, body)
        self.assertEqual(status, 201)
        self.assertEqual(response["device_id"], "d/æ")

    def test_ordinary_entry_unchanged(self) -> None:
        status, body = self._request(
            "POST", "/v1/devices/d1/prekeys",
            {"key_id": "plain", "public_key": _new_prekey()})
        self.assertEqual(status, 201)
        self.assertEqual(body["key_id"], "plain")

    def test_verified_key_supports_claim_and_session(self) -> None:
        public_key = _new_prekey()
        body = self.fx.add_body("fresh", public_key=public_key)
        status, _ = self._request("POST", self.PATH, body)
        self.assertEqual(status, 201)
        status, claim = self._request(
            "POST", "/v1/prekeys/claim",
            {"recipient_device_id": "d1", "claim_id": "c1"})
        # k1 is first in order and gets claimed first.
        self.assertEqual(status, 201)
        self.assertEqual(claim["key_id"], "k1")
        status, claim2 = self._request(
            "POST", "/v1/prekeys/claim",
            {"recipient_device_id": "d1", "claim_id": "c2"})
        self.assertEqual(claim2["key_id"], "fresh")


class AddPrekeyVerifiedPersistenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.path = os.path.join(self.directory, "state.json")

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_authorized_prekey_persists_and_restarts(self) -> None:
        service = DeviceService()
        attach_persistence(service, self.path)
        fx = _Fixture(service, device_id="persist", prekeys=("k1",))
        body = fx.add_body("k2")
        _, status = service.add_prekey_verified("persist", body)
        self.assertEqual(status, 201)

        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        self.assertEqual(
            restarted.get_device("persist")["prekey_ids"], ["k1", "k2"])

        # The prekey_added audit event was anchored by the same write.
        events = restarted.list_key_events("persist", 0, 100)["events"]
        added = [e for e in events if e["type"] == "prekey_added"]
        self.assertEqual(len(added), 1)
        self.assertEqual(added[0]["payload"]["key_id"], "k2")
        self.assertEqual(added[0]["payload"]["public_key"],
                         body["public_key"])

        # Idempotent replay after restart is still 200 and adds no event.
        _, replay_status = restarted.add_prekey_verified("persist", body)
        self.assertEqual(replay_status, 200)
        events_after = restarted.list_key_events("persist", 0, 100)["events"]
        self.assertEqual(len(events_after), len(events))

        # Claim/consumption semantics for the restored key survive restart.
        claim, claim_status = restarted.claim_prekey(
            {"recipient_device_id": "persist", "claim_id": "c1"})
        self.assertEqual(claim_status, 201)
        self.assertEqual(claim["key_id"], "k1")

    def test_disk_failure_rolls_back_everything(self) -> None:
        service = DeviceService()
        store = attach_persistence(service, self.path)
        fx = _Fixture(service, device_id="d1")
        ids_before = service.get_device("d1")["prekey_ids"]
        events_before = service.list_key_events("d1", 0, 100)["events"]
        generation_before = store.commit_seq

        def fail_save(*_args, **_kwargs):
            raise OSError("simulated disk failure")

        store.save = fail_save  # type: ignore[assignment]
        with self.assertRaises(PersistenceUnavailable):
            service.add_prekey_verified("d1", fx.add_body("k2"))

        # The failed transaction is visible nowhere: no key, no event, and the
        # durable generation did not advance.
        self.assertEqual(service.get_device("d1")["prekey_ids"], ids_before)
        self.assertEqual(
            service.list_key_events("d1", 0, 100)["events"], events_before)
        self.assertEqual(store.commit_seq, generation_before)

        # After healing (the hook restores the last good state), a later
        # successful write commits normally.
        del store.save
        _, status = service.add_prekey_verified("d1", fx.add_body("k2"))
        self.assertEqual(status, 201)
        self.assertEqual(service.get_device("d1")["prekey_ids"],
                         ids_before + ["k2"])


class AddPrekeyVerifiedConcurrencyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = DeviceService()
        self.fx = _Fixture(self.service, prekeys=("k0",))

    def test_concurrent_distinct_ids_all_append(self) -> None:
        errors = []

        def worker(index: int) -> None:
            try:
                self.fx.add(f"n{index}")
            except ServiceError as error:  # pragma: no cover - failure path
                errors.append(error)

        threads = [threading.Thread(target=worker, args=(i,))
                   for i in range(20)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertFalse(errors)
        # Every distinct id appends exactly once; their committed order is the
        # nondeterministic lock-acquisition order, so compare as a set after
        # the pre-existing seed key.
        ids = self.service.get_device("d1")["prekey_ids"]
        self.assertEqual(ids[0], "k0")
        self.assertEqual(set(ids[1:]), {f"n{i}" for i in range(20)})
        self.assertEqual(len(ids), 21)

    def test_concurrent_same_id_one_created_rest_idempotent(self) -> None:
        body = self.fx.add_body("dup")
        outcomes = []
        lock = threading.Lock()

        def worker() -> None:
            try:
                _, status = self.service.add_prekey_verified("d1", body)
            except ServiceError:  # pragma: no cover - failure path
                status = "error"
            with lock:
                outcomes.append(status)

        threads = [threading.Thread(target=worker) for _ in range(12)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(sorted(outcomes).count(201), 1)
        self.assertEqual(sorted(outcomes).count(200), 11)
        self.assertEqual(self.service.get_device("d1")["prekey_ids"],
                         ["k0", "dup"])
        # Exactly one audit event for the single committed append.
        events = self.service.list_key_events("d1", 0, 100)["events"]
        dup_events = [e for e in events
                      if e["type"] == "prekey_added"
                      and e["payload"]["key_id"] == "dup"]
        self.assertEqual(len(dup_events), 1)

    def test_concurrent_revoke_vs_verified_add(self) -> None:
        # Whichever commits first, the result is deterministic: either the add
        # wins (201) and the revoke then revokes the device, or the revoke wins
        # and the add is 409/device_id. The key is never half-written.
        outcomes = []
        lock = threading.Lock()

        def add() -> None:
            try:
                _, status = self.service.add_prekey_verified(
                    "d1", self.fx.add_body("race"))
            except ServiceError as error:
                status = (error.status_code, error.field)
            with lock:
                outcomes.append(("add", status))

        def revoke() -> None:
            self.service.revoke_device("d1")

        t1 = threading.Thread(target=add)
        t2 = threading.Thread(target=revoke)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        kind, result = outcomes[0]
        self.assertEqual(kind, "add")
        device = self.service.store.find_by_device_id("d1")
        self.assertTrue(device.revoked)
        if result == 201:
            self.assertIn("race", [pk.key_id for pk in device.prekeys])
        else:
            self.assertEqual(result, (409, "device_id"))
            self.assertNotIn("race", [pk.key_id for pk in device.prekeys])


class AddPrekeyVerifiedCLITest(unittest.TestCase):
    def setUp(self) -> None:
        self.server, service = create_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        self.fx = _Fixture(service, device_id="cli1")
        self.public_key = _new_prekey()
        self.signature = _proof(self.fx.private, "u1", "cli1",
                                "k2", self.public_key)

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _run(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "e2ee_backend", "--base-url",
             self.base_url, *arguments],
            capture_output=True, text=True, timeout=15)

    def test_success_prints_single_line_json_stdout(self) -> None:
        result = self._run(
            "add-prekey-verified", "--device-id", "cli1",
            "--key-id", "k2", "--public-key", self.public_key,
            "--signature", self.signature)
        self.assertEqual(result.returncode, 0, result.stderr)
        line = result.stdout.strip()
        self.assertEqual(line.count("\n"), 0)
        body = json.loads(line)
        self.assertEqual(set(body), {"device_id", "key_id", "public_key"})
        self.assertEqual(body["key_id"], "k2")

    def test_replay_is_exit_0(self) -> None:
        args = ("add-prekey-verified", "--device-id", "cli1",
                "--key-id", "k2", "--public-key", self.public_key,
                "--signature", self.signature)
        first = self._run(*args)
        self.assertEqual(first.returncode, 0, first.stderr)
        second = self._run(*args)
        self.assertEqual(second.returncode, 0, second.stderr)

    def test_bad_proof_is_stderr_json_exit_1(self) -> None:
        result = self._run(
            "add-prekey-verified", "--device-id", "cli1",
            "--key-id", "k9", "--public-key", self.public_key,
            "--signature", self.signature)  # proof is for k2, not k9
        self.assertEqual(result.returncode, 1)
        self.assertEqual(json.loads(result.stderr.strip())["field"],
                         "signature")

    def test_bad_signature_encoding_is_stderr_exit_1(self) -> None:
        result = self._run(
            "add-prekey-verified", "--device-id", "cli1",
            "--key-id", "k9", "--public-key", self.public_key,
            "--signature", "AAAA")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(json.loads(result.stderr.strip())["field"],
                         "signature")

    def test_unknown_device_is_stderr_exit_1(self) -> None:
        result = self._run(
            "add-prekey-verified", "--device-id", "ghost",
            "--key-id", "k2", "--public-key", self.public_key,
            "--signature", self.signature)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(json.loads(result.stderr.strip())["field"],
                         "device_id")

    def test_signature_at_file(self) -> None:
        directory = tempfile.mkdtemp()
        try:
            public_key = _new_prekey()
            signature = _proof(self.fx.private, "u1", "cli1",
                               "k3", public_key)
            path = os.path.join(directory, "sig.txt")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(signature)
            result = self._run(
                "add-prekey-verified", "--device-id", "cli1",
                "--key-id", "k3", "--public-key", public_key,
                "--signature", f"@{path}")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout.strip())["key_id"], "k3")
        finally:
            shutil.rmtree(directory, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
