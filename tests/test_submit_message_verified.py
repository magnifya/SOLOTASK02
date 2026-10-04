"""Tests for verified idempotent message submission.

Covers POST /v1/messages/submit-verified (service layer, HTTP route and
durable recovery): the eight signed fields follow the verify_message
structure/UTF-8/encoding rules (400 naming the field), replay is answered
before every other check (200 with the first response even after
revocation/rotation, 409/request_id on any changed field, a namespace
independent of the plain submit entry), fresh ids run the usual
session/rotation/sender checks, then the identity-key match against the
sender device's current Ed25519 key (409/identity_key) and the signature
check (400/signature), and finally the message-id/sequence/nonce conflict
checks. The message joins the ordinary stream in the existing format;
recovery re-verifies with the frozen public key and refuses startup on
invalid, duplicate, dangling, inconsistent or badly-signed records.
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

from e2ee_backend.crypto import sign_message
from e2ee_backend.http_app import create_server
from e2ee_backend.persistence import StateFileError, attach_persistence
from e2ee_backend.service import DeviceService, ServiceError


def _seed_b64(private) -> str:
    return base64.b64encode(private.private_bytes(
        serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
        serialization.NoEncryption())).decode()


def _raw_b64(public) -> str:
    return base64.b64encode(public.public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw)).decode()


def _x25519_b64() -> str:
    return _raw_b64(x25519.X25519PrivateKey.generate().public_key())


class VerifiedFixture:
    """A service with an Ed25519-identified sender and a 1:1 session."""

    def __init__(self, service: DeviceService = None) -> None:
        self.service = service if service is not None else DeviceService()
        self.private = ed25519.Ed25519PrivateKey.generate()
        self.identity = _raw_b64(self.private.public_key())
        self.other_private = ed25519.Ed25519PrivateKey.generate()
        self.other_identity = _raw_b64(self.other_private.public_key())
        for device_id, identity in (
                ("d1", self.identity),
                ("d2", _raw_b64(ed25519.Ed25519PrivateKey.generate()
                                .public_key()))):
            self.service.register({
                "user_id": "u1", "device_id": device_id,
                "identity_key": identity,
                "signed_prekeys": [{"key_id": "k1",
                                    "public_key": _x25519_b64()}]})
        session = self.service.create_session({
            "initiator_device_id": "d1", "recipient_device_id": "d2",
            "prekey_id": "k1", "ephemeral_key": _x25519_b64()})
        self.session_id = session["session_id"]

    def payload(self, request_id="req-1", message_id="m1", sequence=1,
                session_id=None, sender="d1", private=None, **extra):
        envelope = {
            "session_id": session_id or self.session_id,
            "sender_device_id": sender,
            "message_id": message_id,
            "sequence": sequence,
            "nonce": base64.b64encode(os.urandom(12)).decode(),
            "ciphertext": base64.b64encode(os.urandom(32)).decode(),
        }
        signed = sign_message(envelope, _seed_b64(private or self.private))
        signed["request_id"] = request_id
        signed.update(extra)
        return signed


class ServiceSubmitVerifiedTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = VerifiedFixture()
        self.service = self.fixture.service
        self.session_id = self.fixture.session_id

    def _submit(self, payload=None, **overrides):
        if payload is None:
            payload = self.fixture.payload(**overrides)
        return self.service.submit_message_verified(payload)

    def _expect_error(self, payload, status, field):
        with self.assertRaises(ServiceError) as ctx:
            self.service.submit_message_verified(payload)
        self.assertEqual(ctx.exception.status_code, status)
        self.assertEqual(ctx.exception.field, field)

    def test_first_submit_is_201_with_ten_fields(self) -> None:
        payload = self.fixture.payload()
        body, status = self._submit(payload)
        self.assertEqual(status, 201)
        self.assertEqual(set(body), {"request_id", "session_id",
                                     "sender_device_id", "message_id",
                                     "sequence", "nonce", "ciphertext",
                                     "created_at", "identity_key",
                                     "signature"})
        self.assertEqual(body["identity_key"], payload["identity_key"])
        self.assertEqual(body["signature"], payload["signature"])
        self.assertEqual(body["identity_key"], self.fixture.identity)

    def test_message_joins_the_ordinary_stream(self) -> None:
        self._submit()
        page = self.service.list_messages(self.session_id, "d2", 0, 100)
        self.assertEqual([m["message_id"] for m in page["messages"]], ["m1"])
        self.assertEqual(set(page["messages"][0]),
                         {"session_id", "sender_device_id", "message_id",
                          "sequence", "nonce", "ciphertext", "created_at"})

    def test_identical_replay_is_200_with_original_body(self) -> None:
        payload = self.fixture.payload()
        first, _ = self._submit(payload)
        replay, status = self._submit(dict(payload))
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)

    def test_replay_after_revoke_and_rotation_still_returns_original(self):
        payload = self.fixture.payload()
        first, _ = self._submit(payload)
        self.service.revoke_device("d1")
        replay, status = self._submit(dict(payload))
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)

    def test_replay_after_identity_rotation_still_returns_original(self):
        payload = self.fixture.payload()
        first, _ = self._submit(payload)
        self.service.rotate_identity_key("d1", {"identity_key": _raw_b64(
            ed25519.Ed25519PrivateKey.generate().public_key())})
        replay, status = self._submit(dict(payload))
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)

    def test_changed_field_is_409_request_id(self) -> None:
        payload = self.fixture.payload()
        self._submit(payload)
        changed = dict(payload)
        changed["nonce"] = base64.b64encode(os.urandom(12)).decode()
        self._expect_error(changed, 409, "request_id")
        changed = dict(payload, message_id="m2")
        self._expect_error(changed, 409, "request_id")
        changed = dict(payload, sequence=2)
        self._expect_error(changed, 409, "request_id")
        changed = dict(payload, ciphertext=base64.b64encode(
            os.urandom(32)).decode())
        self._expect_error(changed, 409, "request_id")
        # Re-signed with a different identity: still a replay conflict.
        changed = self.fixture.payload(request_id="req-1",
                                       private=self.fixture.other_private)
        self._expect_error(changed, 409, "request_id")

    def test_request_id_namespace_is_independent_of_plain_submit(self):
        self.service.submit_message({
            "request_id": "shared", "session_id": self.session_id,
            "sender_device_id": "d1", "message_id": "m0", "sequence": 1,
            "nonce": base64.b64encode(os.urandom(12)).decode(),
            "ciphertext": base64.b64encode(os.urandom(32)).decode()})
        body, status = self._submit(request_id="shared", message_id="m1",
                                    sequence=2)
        self.assertEqual(status, 201)
        self.assertEqual(body["request_id"], "shared")

    def test_non_object_body_is_400_request_body(self) -> None:
        for bad in (None, [], "x", 42):
            with self.subTest(bad=bad):
                self._expect_error(bad, 400, "request_body")

    def test_bad_request_id_is_400(self) -> None:
        payload = self.fixture.payload()
        del payload["request_id"]
        self._expect_error(payload, 400, "request_id")
        for bad in ("", None, 7, []):
            with self.subTest(bad=bad):
                self._expect_error(self.fixture.payload(request_id=bad)
                                   if isinstance(bad, str) else
                                   dict(self.fixture.payload(),
                                        request_id=bad),
                                   400, "request_id")

    def test_field_validation_follows_verify_message_rules(self) -> None:
        payload = self.fixture.payload()
        payload["nonce"] = "not-base64!"
        self._expect_error(payload, 400, "nonce")
        payload = self.fixture.payload()
        payload["ciphertext"] = "@@"
        self._expect_error(payload, 400, "ciphertext")
        payload = self.fixture.payload()
        payload["sequence"] = 0
        self._expect_error(payload, 400, "sequence")
        payload = self.fixture.payload()
        payload["sequence"] = True
        self._expect_error(payload, 400, "sequence")
        payload = self.fixture.payload()
        del payload["message_id"]
        self._expect_error(payload, 400, "message_id")
        payload = self.fixture.payload()
        payload["identity_key"] = "not-a-key"
        self._expect_error(payload, 400, "identity_key")
        payload = self.fixture.payload()
        payload["signature"] = "!!!"
        self._expect_error(payload, 400, "signature")

    def test_extra_fields_ignored_and_strings_preserved(self) -> None:
        payload = self.fixture.payload(message_id="消息/ id ",
                                       extra="ignored")
        body, status = self._submit(payload)
        self.assertEqual(status, 201)
        self.assertNotIn("extra", body)
        self.assertEqual(body["message_id"], "消息/ id ")

    def test_unknown_session_is_404(self) -> None:
        self._expect_error(self.fixture.payload(session_id="nope"),
                           404, "session_id")

    def test_inactive_sender_is_409(self) -> None:
        self._expect_error(self.fixture.payload(sender="ghost"),
                           409, "sender_device_id")
        self.service.revoke_device("d1")
        self._expect_error(self.fixture.payload(request_id="req-2"),
                           409, "sender_device_id")

    def test_rotated_session_is_closed(self) -> None:
        self.service.rotate_session(self.session_id, {
            "rotation_id": "rot-1", "actor_device_id": "d1",
            "prekey_id": "k1", "ephemeral_key": _x25519_b64()})
        self._expect_error(self.fixture.payload(), 409, "session_id")

    def test_identity_key_mismatch_is_409(self) -> None:
        payload = self.fixture.payload(private=self.fixture.other_private)
        self._expect_error(payload, 409, "identity_key")

    def test_non_ed25519_current_key_is_409(self) -> None:
        self.service.register({
            "user_id": "u1", "device_id": "dx",
            "identity_key": _x25519_b64(),
            "signed_prekeys": [{"key_id": "k1",
                                "public_key": _x25519_b64()}]})
        session = self.service.create_session({
            "initiator_device_id": "dx", "recipient_device_id": "d2",
            "prekey_id": "k1", "ephemeral_key": _x25519_b64()})
        payload = self.fixture.payload(session_id=session["session_id"],
                                       sender="dx")
        self._expect_error(payload, 409, "identity_key")

    def test_signature_failure_is_400(self) -> None:
        payload = self.fixture.payload()
        other = self.fixture.payload(request_id="req-2", message_id="m2")
        payload["signature"] = other["signature"]
        self._expect_error(payload, 400, "signature")

    def test_failure_consumes_nothing(self) -> None:
        # A bad-signature attempt with a fresh id fails...
        payload = self.fixture.payload()
        payload["signature"] = self.fixture.payload(
            request_id="req-2", message_id="m2")["signature"]
        self._expect_error(payload, 400, "signature")
        # ...and neither the id nor the sequence was consumed.
        body, status = self._submit()
        self.assertEqual(status, 201)
        self.assertEqual(body["sequence"], 1)

    def test_equivalent_identity_key_encoding_matches(self) -> None:
        private = ed25519.Ed25519PrivateKey.generate()
        der = base64.b64encode(private.public_key().public_bytes(
            serialization.Encoding.DER,
            serialization.PublicFormat.SubjectPublicKeyInfo)).decode()
        self.service.register({
            "user_id": "u1", "device_id": "dd", "identity_key": der,
            "signed_prekeys": [{"key_id": "k1",
                                "public_key": _x25519_b64()}]})
        session = self.service.create_session({
            "initiator_device_id": "dd", "recipient_device_id": "d2",
            "prekey_id": "k1", "ephemeral_key": _x25519_b64()})
        # sign_message emits the canonical raw-point spelling; the stored
        # DER spelling names the same actual key.
        envelope = {
            "session_id": session["session_id"], "sender_device_id": "dd",
            "message_id": "m1", "sequence": 1,
            "nonce": base64.b64encode(os.urandom(12)).decode(),
            "ciphertext": base64.b64encode(os.urandom(32)).decode(),
        }
        payload = sign_message(envelope, _seed_b64(private))
        payload["request_id"] = "req-der"
        body, status = self.service.submit_message_verified(payload)
        self.assertEqual(status, 201)
        self.assertEqual(body["identity_key"], payload["identity_key"])


class GroupSubmitVerifiedTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = VerifiedFixture()
        self.service = self.fixture.service
        self.service.register({
            "user_id": "u1", "device_id": "d3",
            "identity_key": _raw_b64(ed25519.Ed25519PrivateKey.generate()
                                     .public_key()),
            "signed_prekeys": [{"key_id": "k1",
                                "public_key": _x25519_b64()}]})
        self.service.create_group({
            "group_id": "g1", "creator_device_id": "d1",
            "member_device_ids": ["d2", "d3"]})
        session = self.service.create_group_session({
            "group_id": "g1", "initiator_device_id": "d1",
            "ephemeral_key": _x25519_b64()})
        self.group_session_id = session["session_id"]

    def test_group_session_submit_and_replay(self) -> None:
        payload = self.fixture.payload(session_id=self.group_session_id)
        body, status = self.service.submit_message_verified(payload)
        self.assertEqual(status, 201)
        replay, status = self.service.submit_message_verified(dict(payload))
        self.assertEqual(status, 200)
        self.assertEqual(replay, body)

    def test_non_member_sender_is_409(self) -> None:
        self.service.register({
            "user_id": "u1", "device_id": "outsider",
            "identity_key": self.fixture.identity,
            "signed_prekeys": [{"key_id": "k1",
                                "public_key": _x25519_b64()}]})
        payload = self.fixture.payload(session_id=self.group_session_id,
                                       sender="outsider")
        with self.assertRaises(ServiceError) as ctx:
            self.service.submit_message_verified(payload)
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "sender_device_id")


class SubmitVerifiedHTTPTest(unittest.TestCase):
    def setUp(self) -> None:
        self.server, self.service = create_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        self.fixture = VerifiedFixture(self.service)

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _request(self, method, path, body=None, raw=None):
        connection = HTTPConnection("127.0.0.1", self.port, timeout=5)
        if raw is not None:
            connection.request(method, path, body=raw,
                               headers={"Content-Type": "application/json"})
        elif body is not None:
            connection.request(method, path, body=json.dumps(body),
                               headers={"Content-Type": "application/json"})
        else:
            connection.request(method, path)
        response = connection.getresponse()
        data = response.read().decode("utf-8")
        connection.close()
        return response.status, json.loads(data) if data else {}

    def test_submit_replay_and_pull_over_http(self) -> None:
        payload = self.fixture.payload()
        status, body = self._request("POST", "/v1/messages/submit-verified",
                                     payload)
        self.assertEqual(status, 201)
        self.assertEqual(body["identity_key"], self.fixture.identity)
        status, replay = self._request("POST", "/v1/messages/submit-verified",
                                       payload)
        self.assertEqual(status, 200)
        self.assertEqual(replay, body)
        # The message is pulled in the existing envelope format.
        status, page = self._request(
            "GET", f"/v1/messages/{self.fixture.session_id}?device_id=d2")
        self.assertEqual(status, 200)
        self.assertEqual(set(page["messages"][0]),
                         {"session_id", "sender_device_id", "message_id",
                          "sequence", "nonce", "ciphertext", "created_at"})

    def test_bad_json_and_non_object_are_400_request_body(self) -> None:
        status, body = self._request("POST", "/v1/messages/submit-verified",
                                     raw=b"{not json")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "request_body")
        status, body = self._request("POST", "/v1/messages/submit-verified",
                                     body=[1, 2])
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "request_body")

    def test_error_responses_over_http(self) -> None:
        payload = self.fixture.payload(session_id="nope")
        status, body = self._request("POST", "/v1/messages/submit-verified",
                                     payload)
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "session_id")
        payload = self.fixture.payload(private=self.fixture.other_private)
        status, body = self._request("POST", "/v1/messages/submit-verified",
                                     payload)
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "identity_key")


class SubmitVerifiedPersistenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.path = os.path.join(self.tmp, "state.json")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _attach(self, service) -> None:
        attach_persistence(service, self.path)

    def _document(self) -> dict:
        with open(self.path, encoding="utf-8") as handle:
            return json.load(handle)

    def _write_document(self, document) -> None:
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)

    def test_restart_preserves_signature_and_replay(self) -> None:
        service = DeviceService()
        self._attach(service)
        fixture = VerifiedFixture(service)
        payload = fixture.payload()
        body, status = service.submit_message_verified(payload)
        self.assertEqual(status, 201)
        record = self._document()["verified_message_submissions"][0]
        self.assertEqual(record["identity_key"], fixture.identity)
        self.assertEqual(record["signature"], payload["signature"])

        restored = DeviceService()
        self._attach(restored)
        replay, status = restored.submit_message_verified(dict(payload))
        self.assertEqual(status, 200)
        self.assertEqual(replay, body)

    def test_replay_survives_rotation_and_restart(self) -> None:
        service = DeviceService()
        self._attach(service)
        fixture = VerifiedFixture(service)
        payload = fixture.payload()
        body, _ = service.submit_message_verified(payload)
        service.rotate_identity_key("d1", {"identity_key": _raw_b64(
            ed25519.Ed25519PrivateKey.generate().public_key())})
        restored = DeviceService()
        self._attach(restored)
        replay, status = restored.submit_message_verified(dict(payload))
        self.assertEqual(status, 200)
        self.assertEqual(replay, body)

    def test_legacy_file_without_section_loads(self) -> None:
        service = DeviceService()
        self._attach(service)
        VerifiedFixture(service)
        document = self._document()
        document.pop("verified_message_submissions", None)
        self._write_document(document)
        restored = DeviceService()
        self._attach(restored)
        self.assertEqual(restored.store.snapshot_state()
                         ["verified_message_submissions"], [])

    def test_invalid_new_records_refuse_startup_untouched(self) -> None:
        service = DeviceService()
        self._attach(service)
        fixture = VerifiedFixture(service)
        payload = fixture.payload()
        service.submit_message_verified(payload)
        good = self._document()
        record = good["verified_message_submissions"][0]

        variants = []
        # An invalid signature.
        bad_signature = json.loads(json.dumps(good))
        bad_signature["verified_message_submissions"][0]["signature"] = \
            base64.b64encode(os.urandom(64)).decode()
        variants.append(bad_signature)
        # A duplicate request_id.
        duplicate = json.loads(json.dumps(good))
        duplicate["verified_message_submissions"].append(dict(record))
        variants.append(duplicate)
        # A dangling message reference.
        dangling = json.loads(json.dumps(good))
        dangling["verified_message_submissions"][0]["message_id"] = "ghost"
        variants.append(dangling)
        # An envelope inconsistent with the stored message.
        inconsistent = json.loads(json.dumps(good))
        inconsistent["verified_message_submissions"][0]["nonce"] = \
            base64.b64encode(os.urandom(12)).decode()
        variants.append(inconsistent)
        # A malformed record (missing field).
        malformed = json.loads(json.dumps(good))
        del malformed["verified_message_submissions"][0]["identity_key"]
        variants.append(malformed)

        for variant in variants:
            with self.subTest(variant=variant[
                    "verified_message_submissions"][0].get("message_id")):
                self._write_document(variant)
                before = json.dumps(variant, sort_keys=True)
                with self.assertRaises(StateFileError):
                    self._attach(DeviceService())
                with open(self.path, encoding="utf-8") as handle:
                    self.assertEqual(
                        json.dumps(json.load(handle), sort_keys=True), before)


if __name__ == "__main__":
    unittest.main()
