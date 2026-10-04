"""Tests for the signature-verified, idempotent message submission entry.

Covers ``POST /v1/messages/submit-verified`` at the service, persistence
and HTTP layers: it accepts the eight fields ``sign_message`` returns plus
a non-empty ``request_id`` (a namespace independent of the ordinary
``/v1/messages/submit`` entry), verifies the new message against the
sender device's *current* Ed25519 identity key, supports 1:1 and group
sessions, replays byte-identically (200) even after revocation/identity or
session rotation, conflicts on any changed field (409/request_id), keeps
the documented validation/error order, writes exactly one message under
concurrency, leaves pull/sync/delivery formats untouched, and persists the
frozen public key and signature with full startup validation (recovery
re-verifies with the frozen key; a tampered record refuses startup with
field=data_file and the file unchanged).
"""
import base64
import hashlib
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
from e2ee_backend.storage import DeviceStore


def _raw_b64(key) -> str:
    raw = key.public_bytes(serialization.Encoding.Raw,
                           serialization.PublicFormat.Raw)
    return base64.b64encode(raw).decode()


def _seed_b64(private) -> str:
    return base64.b64encode(private.private_bytes(
        serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
        serialization.NoEncryption())).decode()


def _x25519_b64() -> str:
    return _raw_b64(x25519.X25519PrivateKey.generate().public_key())


def _register_payload(device_id: str, user_id: str = "u1",
                      identity_key=None) -> dict:
    # Two pre-keys so a session created against k1 can be rotated against k2
    # without replenishing pre-keys mid-test.
    return {
        "user_id": user_id,
        "device_id": device_id,
        "identity_key": identity_key or _x25519_b64(),
        "signed_prekeys": [
            {"key_id": "k1", "public_key": _x25519_b64()},
            {"key_id": "k2", "public_key": _x25519_b64()},
        ],
    }


_ENVELOPE_FIELDS = ("session_id", "sender_device_id", "message_id",
                    "sequence", "nonce", "ciphertext")
_VIEW_FIELDS = {"request_id", *_ENVELOPE_FIELDS, "created_at",
                "identity_key", "signature"}


class VerifiedSubmitFixture:
    """A service with two devices and one 1:1 session between them.

    d1's identity is an Ed25519 key whose private seed the fixture keeps, so
    tests can sign envelopes with ``sign_message``.
    """

    def __init__(self) -> None:
        self.service = DeviceService()
        self.private = ed25519.Ed25519PrivateKey.generate()
        self.seed = _seed_b64(self.private)
        self.identity = _raw_b64(self.private.public_key())
        self.service.register(_register_payload("d1", identity_key=self.identity))
        self.service.register(_register_payload("d2"))
        session = self.service.create_session({
            "initiator_device_id": "d1",
            "recipient_device_id": "d2",
            "prekey_id": "k1",
            "ephemeral_key": _x25519_b64(),
        })
        self.session_id = session["session_id"]

    def envelope(self, request_id="req-1", message_id="m1", sequence=1,
                 sender="d1", session_id=None, nonce=None,
                 ciphertext=None) -> dict:
        # Deterministic per message id so a repeated call produces a
        # byte-identical replay (the nonce must still be exactly 12 bytes and
        # the ciphertext at least 16 under the signed-envelope rules). Explicit
        # nonce/ciphertext overrides are signed too, so a caller-provided
        # reused nonce still carries a valid signature over it.
        digest = hashlib.sha256(message_id.encode("utf-8")).digest()
        if nonce is None:
            nonce = base64.b64encode(digest[:12]).decode()
        if ciphertext is None:
            ciphertext = base64.b64encode(
                digest[12:] + b"\x00" * 16).decode()
        envelope = {
            "session_id": session_id or self.session_id,
            "sender_device_id": sender,
            "message_id": message_id,
            "sequence": sequence,
            "nonce": nonce,
            "ciphertext": ciphertext,
        }
        signed = sign_message(envelope, self.seed)
        signed["request_id"] = request_id
        return signed

    def sign_again(self, signed, private=None) -> dict:
        private = private or self.private
        envelope = {name: signed[name] for name in _ENVELOPE_FIELDS}
        return sign_message(envelope, _seed_b64(private))


class ServiceSubmitVerifiedTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = VerifiedSubmitFixture()
        self.service = self.fixture.service
        self.session_id = self.fixture.session_id

    def _submit(self, **overrides):
        payload = self.fixture.envelope()
        payload.update(overrides)
        return self.service.submit_verified_message(payload)

    def test_first_submit_is_201_with_ten_fields(self) -> None:
        body, status = self._submit()
        self.assertEqual(status, 201)
        self.assertEqual(set(body), _VIEW_FIELDS)
        self.assertEqual(body["request_id"], "req-1")
        self.assertEqual(body["sequence"], 1)
        self.assertEqual(body["identity_key"], self.fixture.identity)
        self.assertTrue(body["created_at"].endswith("+00:00"))
        # Only public material is returned; the private seed never appears.
        self.assertNotIn(self.fixture.seed, json.dumps(body))

    def test_extra_fields_ignored_and_strings_kept_verbatim(self) -> None:
        payload = self.fixture.envelope()
        payload["extra"] = "ignored"
        body, status = self.service.submit_verified_message(payload)
        self.assertEqual(status, 201)
        self.assertEqual(set(body), _VIEW_FIELDS)
        self.assertEqual(body["identity_key"], payload["identity_key"])
        self.assertEqual(body["signature"], payload["signature"])

    def test_identical_replay_is_200_with_original_body(self) -> None:
        payload = self.fixture.envelope()
        first, status = self.service.submit_verified_message(payload)
        self.assertEqual(status, 201)
        replay, status = self.service.submit_verified_message(dict(payload))
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)

    def test_replay_after_revocation_still_returns_original(self) -> None:
        first, _ = self._submit()
        self.service.revoke_device("d1")
        replay, status = self._submit()
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)

    def test_replay_after_identity_rotation_still_returns_original(self) -> None:
        first, _ = self._submit()
        new_identity = _raw_b64(ed25519.Ed25519PrivateKey.generate().public_key())
        self.service.rotate_identity_key("d1", {"identity_key": new_identity})
        replay, status = self._submit()
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)

    def test_replay_after_session_rotation_still_returns_original(self) -> None:
        first, _ = self._submit()
        # Rotate the 1:1 session closed (d1 initiates).
        self.service.rotate_session(self.session_id, {
            "rotation_id": "rot-1", "actor_device_id": "d1",
            "prekey_id": "k2", "ephemeral_key": _x25519_b64()})
        replay, status = self._submit()
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)

    def test_changed_any_of_eight_fields_is_409_request_id(self) -> None:
        payload = self.fixture.envelope()
        first, _ = self.service.submit_verified_message(payload)
        # Each alternative must remain structurally valid: shape validation
        # runs before replay, so a malformed value would name its own field
        # with 400 rather than reaching the replay conflict.
        others = {
            "session_id": self.session_id + "x",
            "sender_device_id": "d2",
            "message_id": "m2",
            "sequence": 2,
            "nonce": base64.b64encode(os.urandom(12)).decode(),
        }
        for name, value in others.items():
            changed = dict(payload)
            changed[name] = value
            with self.assertRaises(ServiceError) as ctx:
                self.service.submit_verified_message(changed)
            self.assertEqual(ctx.exception.status_code, 409, name)
            self.assertEqual(ctx.exception.field, "request_id", name)
        # A different identity_key spelling/value or signature conflicts too.
        other_private = ed25519.Ed25519PrivateKey.generate()
        other_key = _raw_b64(other_private.public_key())
        changed = dict(payload, identity_key=other_key)
        with self.assertRaises(ServiceError) as ctx:
            self.service.submit_verified_message(changed)
        self.assertEqual(ctx.exception.field, "request_id")
        raw_sig = bytearray(base64.b64decode(payload["signature"]))
        raw_sig[0] ^= 1
        changed = dict(payload,
                       signature=base64.b64encode(bytes(raw_sig)).decode())
        with self.assertRaises(ServiceError) as ctx:
            self.service.submit_verified_message(changed)
        self.assertEqual(ctx.exception.field, "request_id")
        # Flip a ciphertext byte while keeping it valid 32-byte canonical
        # base64 so the request reaches replay rather than shape validation.
        raw_ct = bytearray(base64.b64decode(payload["ciphertext"]))
        raw_ct[0] ^= 1
        changed = dict(payload,
                       ciphertext=base64.b64encode(bytes(raw_ct)).decode())
        with self.assertRaises(ServiceError) as ctx:
            self.service.submit_verified_message(changed)
        self.assertEqual(ctx.exception.field, "request_id")

    def test_request_id_namespace_is_independent_of_old_entry(self) -> None:
        # The same request_id can commit once in each entry.
        old_payload = {
            "request_id": "shared-id",
            "session_id": self.session_id,
            "sender_device_id": "d1",
            "message_id": "old-m1",
            "sequence": 1,
            "nonce": base64.b64encode(b"old-nonce-12xy").decode(),
            "ciphertext": base64.b64encode(b"ciphertext-and-tag").decode(),
        }
        old_body, status = self.service.submit_message(old_payload)
        self.assertEqual(status, 201)
        verified_payload = self.fixture.envelope(request_id="shared-id",
                                                 message_id="new-m1",
                                                 sequence=2)
        verified_body, status = self.service.submit_verified_message(
            verified_payload)
        self.assertEqual(status, 201)
        self.assertEqual(verified_body["message_id"], "new-m1")
        # Each namespace replays independently.
        replay, status = self.service.submit_message(dict(old_payload))
        self.assertEqual(status, 200)
        self.assertEqual(replay, old_body)
        replay, status = self.service.submit_verified_message(
            dict(verified_payload))
        self.assertEqual(status, 200)
        self.assertEqual(replay, verified_body)

    # -- structural validation --------------------------------------------

    def test_non_object_body_is_400_request_body(self) -> None:
        for bad in (None, [], "x", 7, True):
            with self.assertRaises(ServiceError) as ctx:
                self.service.submit_verified_message(bad)
            self.assertEqual(ctx.exception.status_code, 400, bad)
            self.assertEqual(ctx.exception.field, "request_body", bad)

    def test_bad_request_id_is_400_request_id(self) -> None:
        for bad in ("", 7, None, True):
            payload = self.fixture.envelope()
            payload["request_id"] = bad
            with self.assertRaises(ServiceError) as ctx:
                self.service.submit_verified_message(payload)
            self.assertEqual(ctx.exception.status_code, 400, bad)
            self.assertEqual(ctx.exception.field, "request_id", bad)
        payload = self.fixture.envelope()
        del payload["request_id"]
        with self.assertRaises(ServiceError) as ctx:
            self.service.submit_verified_message(payload)
        self.assertEqual(ctx.exception.field, "request_id")

    def test_lone_surrogate_request_id_is_400_request_id(self) -> None:
        payload = self.fixture.envelope()
        payload["request_id"] = "ud800\ud800"
        with self.assertRaises(ServiceError) as ctx:
            self.service.submit_verified_message(payload)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.field, "request_id")

    def test_envelope_field_errors_keep_their_field_names(self) -> None:
        payload = self.fixture.envelope()
        del payload["session_id"]
        with self.assertRaises(ServiceError) as ctx:
            self.service.submit_verified_message(payload)
        self.assertEqual(ctx.exception.field, "session_id")
        for name in ("sender_device_id", "message_id"):
            with self.assertRaises(ServiceError) as ctx:
                self.service.submit_verified_message(
                    dict(self.fixture.envelope(), **{name: ""}))
            self.assertEqual(ctx.exception.field, name)
        with self.assertRaises(ServiceError) as ctx:
            self.service.submit_verified_message(
                dict(self.fixture.envelope(), sequence=True))
        self.assertEqual(ctx.exception.field, "sequence")
        with self.assertRaises(ServiceError) as ctx:
            self.service.submit_verified_message(
                dict(self.fixture.envelope(), nonce="!!!"))
        self.assertEqual(ctx.exception.field, "nonce")
        with self.assertRaises(ServiceError) as ctx:
            self.service.submit_verified_message(
                dict(self.fixture.envelope(),
                     ciphertext=base64.b64encode(b"x" * 15).decode()))
        self.assertEqual(ctx.exception.field, "ciphertext")
        with self.assertRaises(ServiceError) as ctx:
            self.service.submit_verified_message(
                dict(self.fixture.envelope(), identity_key="not-a-key"))
        self.assertEqual(ctx.exception.field, "identity_key")
        with self.assertRaises(ServiceError) as ctx:
            self.service.submit_verified_message(
                dict(self.fixture.envelope(), signature="!!!"))
        self.assertEqual(ctx.exception.field, "signature")

    # -- ordered live-state checks -----------------------------------------

    def test_unknown_session_is_404_session_id(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self._submit(session_id="ghost")
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "session_id")

    def test_closed_rotated_session_is_409_session_id(self) -> None:
        self.service.rotate_session(self.session_id, {
            "rotation_id": "rot-1", "actor_device_id": "d1",
            "prekey_id": "k2", "ephemeral_key": _x25519_b64()})
        payload = self.fixture.envelope(request_id="req-2", message_id="m2",
                                       sequence=2)
        with self.assertRaises(ServiceError) as ctx:
            self.service.submit_verified_message(payload)
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "session_id")

    def test_revoked_sender_is_409_sender_device_id(self) -> None:
        self.service.revoke_device("d1")
        with self.assertRaises(ServiceError) as ctx:
            self._submit()
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "sender_device_id")

    def test_current_key_not_ed25519_is_409_identity_key(self) -> None:
        # A fresh fixture whose d1 carries an X25519 identity key.
        service = DeviceService()
        service.register(_register_payload("d1"))
        service.register(_register_payload("d2"))
        session = service.create_session({
            "initiator_device_id": "d1", "recipient_device_id": "d2",
            "prekey_id": "k1", "ephemeral_key": _x25519_b64()})
        signed = self.fixture.envelope(session_id=session["session_id"])
        with self.assertRaises(ServiceError) as ctx:
            service.submit_verified_message(signed)
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "identity_key")

    def test_submitted_other_key_is_409_identity_key(self) -> None:
        other = ed25519.Ed25519PrivateKey.generate()
        # Sign with a different key but keep d1's envelope ids.
        envelope = {name: self.fixture.envelope()[name]
                    for name in _ENVELOPE_FIELDS}
        foreign = sign_message(envelope, _seed_b64(other))
        foreign["request_id"] = "req-1"
        with self.assertRaises(ServiceError) as ctx:
            self.service.submit_verified_message(foreign)
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "identity_key")

    def test_signature_failure_is_400_signature(self) -> None:
        payload = self.fixture.envelope()
        raw = bytearray(base64.b64decode(payload["signature"]))
        raw[0] ^= 1
        payload["signature"] = base64.b64encode(bytes(raw)).decode()
        with self.assertRaises(ServiceError) as ctx:
            self.service.submit_verified_message(payload)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.field, "signature")

    def test_message_id_sequence_nonce_conflicts_keep_order(self) -> None:
        first, _ = self._submit()
        # Duplicate message_id wins over the (also wrong) sequence/nonce.
        with self.assertRaises(ServiceError) as ctx:
            self._submit(request_id="req-2")
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "message_id")
        # Fresh message_id, wrong sequence (a fully signed envelope so it
        # reaches the stream checks rather than failing signature first).
        payload = self.fixture.envelope(request_id="req-2", message_id="m2",
                                        sequence=3)
        with self.assertRaises(ServiceError) as ctx:
            self.service.submit_verified_message(payload)
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "sequence")
        # Fresh message_id/sequence but the first message's nonce, re-signed.
        payload = self.fixture.envelope(request_id="req-2", message_id="m2",
                                        sequence=2, nonce=first["nonce"])
        with self.assertRaises(ServiceError) as ctx:
            self.service.submit_verified_message(payload)
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "nonce")

    def test_failure_consumes_no_id_sequence_or_nonce(self) -> None:
        # Bad sequence fails (a fully signed envelope); the same request_id
        # then commits sequence 1.
        bad = self.fixture.envelope(sequence=9)
        with self.assertRaises(ServiceError) as ctx:
            self.service.submit_verified_message(bad)
        self.assertEqual(ctx.exception.field, "sequence")
        body, status = self._submit()
        self.assertEqual(status, 201)
        self.assertEqual(body["sequence"], 1)
        # Exactly one message landed.
        page = self.service.list_messages(self.session_id, "d2", 0, 100)
        self.assertEqual([m["message_id"] for m in page["messages"]], ["m1"])

    def test_rotation_then_new_message_needs_new_key(self) -> None:
        first, _ = self._submit()
        new_private = ed25519.Ed25519PrivateKey.generate()
        new_identity = _raw_b64(new_private.public_key())
        self.service.rotate_identity_key("d1",
                                         {"identity_key": new_identity})
        # A brand-new envelope still signed by the old key: key conflict.
        stale = self.fixture.envelope(request_id="req-2", message_id="m2",
                                     sequence=2)
        with self.assertRaises(ServiceError) as ctx:
            self.service.submit_verified_message(stale)
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "identity_key")
        # Nothing was consumed by that failure.
        fresh = self.fixture.envelope(request_id="req-2", message_id="m2",
                                     sequence=2)
        fresh.update(self.fixture.sign_again(fresh, new_private))
        body, status = self.service.submit_verified_message(fresh)
        self.assertEqual(status, 201)
        self.assertEqual(body["identity_key"], new_identity)

    # -- stream integration ------------------------------------------------

    def test_message_enters_normal_stream_and_pull_format_unchanged(self) -> None:
        body, _ = self._submit()
        page = self.service.list_messages(self.session_id, "d2", 0, 100)
        self.assertEqual(len(page["messages"]), 1)
        pulled = page["messages"][0]
        self.assertEqual(set(pulled),
                         {"session_id", "sender_device_id", "message_id",
                          "sequence", "nonce", "ciphertext", "created_at"})
        self.assertNotIn("identity_key", pulled)
        self.assertNotIn("signature", pulled)
        self.assertEqual(pulled["message_id"], body["message_id"])

    def test_concurrent_identical_submits_write_one_message(self) -> None:
        payload = self.fixture.envelope()
        results, errors = [], []

        def worker() -> None:
            try:
                results.append(
                    self.service.submit_verified_message(dict(payload)))
            except ServiceError as error:  # pragma: no cover - unexpected
                errors.append(error)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(errors, [])
        self.assertEqual(len(results), 8)
        self.assertEqual(sum(1 for _, status in results if status == 201), 1)
        self.assertEqual(sum(1 for _, status in results if status == 200), 7)
        bodies = {json.dumps(body, sort_keys=True) for body, _ in results}
        self.assertEqual(len(bodies), 1)
        page = self.service.list_messages(self.session_id, "d2", 0, 100)
        self.assertEqual(len(page["messages"]), 1)


class GroupSubmitVerifiedTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = VerifiedSubmitFixture()
        self.service = self.fixture.service
        self.service.register(_register_payload("d3"))
        self.service.create_group({
            "group_id": "g1", "creator_device_id": "d1",
            "member_device_ids": ["d2"]})
        session = self.service.create_group_session({
            "group_id": "g1", "initiator_device_id": "d1",
            "ephemeral_key": _x25519_b64()})
        self.group_session_id = session["session_id"]

    def _submit(self, **overrides):
        payload = self.fixture.envelope(session_id=self.group_session_id)
        payload.update(overrides)
        return self.service.submit_verified_message(payload)

    def test_group_first_submit_is_201(self) -> None:
        body, status = self._submit()
        self.assertEqual(status, 201)
        self.assertEqual(body["session_id"], self.group_session_id)
        page = self.service.list_messages(self.group_session_id, "d2", 0, 100)
        self.assertEqual(len(page["messages"]), 1)

    def test_non_member_sender_is_409_sender_device_id(self) -> None:
        # d3 is not in the frozen member set; an envelope claiming d3 fails
        # the sender-eligibility gate before the key is even consulted.
        payload = self.fixture.envelope(session_id=self.group_session_id,
                                       sender="d3")
        with self.assertRaises(ServiceError) as ctx:
            self.service.submit_verified_message(payload)
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "sender_device_id")


class SubmitVerifiedPersistenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.path = os.path.join(self.directory, "state.json")

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def _persisted_service(self) -> DeviceService:
        service = DeviceService()
        attach_persistence(service, self.path)
        return service

    def _fixture(self, service: DeviceService) -> VerifiedSubmitFixture:
        fixture = VerifiedSubmitFixture.__new__(VerifiedSubmitFixture)
        fixture.service = service
        fixture.private = ed25519.Ed25519PrivateKey.generate()
        fixture.seed = _seed_b64(fixture.private)
        fixture.identity = _raw_b64(fixture.private.public_key())
        service.register(_register_payload(
            "d1", identity_key=fixture.identity))
        service.register(_register_payload("d2"))
        session = service.create_session({
            "initiator_device_id": "d1", "recipient_device_id": "d2",
            "prekey_id": "k1", "ephemeral_key": _x25519_b64()})
        fixture.session_id = session["session_id"]
        return fixture

    def _document(self) -> dict:
        with open(self.path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _restore(self, document: dict) -> None:
        payload = {key: value for key, value in document.items()
                   if key != "version"}
        DeviceStore().restore_state(payload)

    def test_replay_and_signature_survive_restart(self) -> None:
        service = self._persisted_service()
        fixture = self._fixture(service)
        first, status = service.submit_verified_message(fixture.envelope())
        self.assertEqual(status, 201)

        restarted = self._persisted_service()
        fixture.service = restarted
        replay, status = restarted.submit_verified_message(
            fixture.envelope())
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)

    def test_recovery_uses_frozen_key_after_rotation(self) -> None:
        service = self._persisted_service()
        fixture = self._fixture(service)
        signed = fixture.envelope()
        first, _ = service.submit_verified_message(signed)
        # Rotate d1 to a new identity, then restart: the stored signature
        # must still validate against the frozen key inside the record.
        service.rotate_identity_key(
            "d1",
            {"identity_key": _raw_b64(
                ed25519.Ed25519PrivateKey.generate().public_key())})
        restarted = self._persisted_service()
        fixture.service = restarted
        replay, status = restarted.submit_verified_message(dict(signed))
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)

    def test_old_file_without_section_loads_empty(self) -> None:
        service = self._persisted_service()
        self._fixture(service)
        document = self._document()
        del document["message_submissions"]
        self._restore(document)  # must not raise

    def _submitted_document(self) -> dict:
        if os.path.exists(self.path):
            os.unlink(self.path)
        service = self._persisted_service()
        fixture = self._fixture(service)
        service.submit_verified_message(fixture.envelope())
        return self._document()

    def test_section_record_carries_key_and_signature(self) -> None:
        document = self._submitted_document()
        self.assertEqual(len(document["message_submissions"]), 1)
        record = document["message_submissions"][0]
        self.assertEqual(set(record),
                         {"request_id", "session_id", "sender_device_id",
                          "message_id", "sequence", "nonce", "ciphertext",
                          "created_at", "identity_key", "signature"})

    def test_section_must_be_a_list(self) -> None:
        document = self._submitted_document()
        document["message_submissions"] = {"request_id": "req-1"}
        with self.assertRaises(ValueError):
            self._restore(document)

    def test_duplicate_verified_request_id_rejected(self) -> None:
        document = self._submitted_document()
        document["message_submissions"].append(
            dict(document["message_submissions"][0]))
        with self.assertRaises(ValueError):
            self._restore(document)

    def test_same_id_in_both_namespaces_is_valid(self) -> None:
        if os.path.exists(self.path):
            os.unlink(self.path)
        service = self._persisted_service()
        fixture = self._fixture(service)
        service.submit_message({
            "request_id": "shared-id",
            "session_id": fixture.session_id,
            "sender_device_id": "d1",
            "message_id": "plain-m1",
            "sequence": 1,
            "nonce": base64.b64encode(b"plain-nonce-xy").decode(),
            "ciphertext": base64.b64encode(b"ciphertext-and-tag").decode(),
        })
        verified = fixture.envelope(request_id="shared-id",
                                    message_id="signed-m1", sequence=2)
        verified_body, status = service.submit_verified_message(verified)
        self.assertEqual(status, 201)
        # The on-disk pair round-trips through the strict semantic restore.
        self._restore(self._document())

    def test_dangling_references_rejected(self) -> None:
        document = self._submitted_document()
        record = document["message_submissions"][0]
        record["session_id"] = "ghost"
        with self.assertRaises(ValueError):
            self._restore(document)
        document = self._submitted_document()
        document["message_submissions"][0]["message_id"] = "ghost"
        with self.assertRaises(ValueError):
            self._restore(document)

    def test_envelope_mismatch_rejected(self) -> None:
        for field, value in (("sender_device_id", "d2"),
                             ("sequence", 7),
                             ("nonce", "other"),
                             ("ciphertext", "other"),
                             ("created_at", "other")):
            document = self._submitted_document()
            document["message_submissions"][0][field] = value
            with self.assertRaises(ValueError, msg=field):
                self._restore(document)

    def test_half_frozen_signature_pair_rejected(self) -> None:
        document = self._submitted_document()
        del document["message_submissions"][0]["signature"]
        with self.assertRaises(ValueError):
            self._restore(document)

    def test_bad_frozen_identity_key_rejected(self) -> None:
        document = self._submitted_document()
        document["message_submissions"][0]["identity_key"] = "not-a-key"
        with self.assertRaises(ValueError):
            self._restore(document)

    def test_frozen_signature_mismatch_rejected(self) -> None:
        document = self._submitted_document()
        record = document["message_submissions"][0]
        record["identity_key"] = _raw_b64(
            ed25519.Ed25519PrivateKey.generate().public_key())
        with self.assertRaises(ValueError):
            self._restore(document)
        document = self._submitted_document()
        raw = bytearray(base64.b64decode(
            document["message_submissions"][0]["signature"]))
        raw[0] ^= 1
        document["message_submissions"][0]["signature"] = \
            base64.b64encode(bytes(raw)).decode()
        with self.assertRaises(ValueError):
            self._restore(document)

    def test_malformed_record_refuses_startup_without_overwriting(self) -> None:
        document = self._submitted_document()
        raw = bytearray(base64.b64decode(
            document["message_submissions"][0]["signature"]))
        raw[0] ^= 1
        document["message_submissions"][0]["signature"] = \
            base64.b64encode(bytes(raw)).decode()
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        with open(self.path, "rb") as handle:
            original = handle.read()
        with self.assertRaises(StateFileError):
            self._persisted_service()
        with open(self.path, "rb") as handle:
            self.assertEqual(handle.read(), original)


class HTTPSubmitVerifiedTest(unittest.TestCase):
    def setUp(self) -> None:
        self.server, _ = create_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        self.private = ed25519.Ed25519PrivateKey.generate()
        identity = _raw_b64(self.private.public_key())
        for device_id, key in (("d1", identity), ("d2", None)):
            status, _ = self._request("POST", "/v1/devices",
                                      _register_payload(device_id,
                                                        identity_key=key))
            self.assertEqual(status, 201)
        status, session = self._request("POST", "/v1/sessions", {
            "initiator_device_id": "d1",
            "recipient_device_id": "d2",
            "prekey_id": "k1",
            "ephemeral_key": _x25519_b64(),
        })
        self.assertEqual(status, 201)
        self.session_id = session["session_id"]

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _request(self, method: str, path: str, body=None, raw=None):
        connection = HTTPConnection("127.0.0.1", self.port, timeout=5)
        if raw is not None:
            payload = raw
        else:
            payload = json.dumps(body) if body is not None else None
        headers = {"Content-Type": "application/json"} if payload is not None else {}
        connection.request(method, path, body=payload, headers=headers)
        response = connection.getresponse()
        data = response.read().decode("utf-8")
        connection.close()
        return response.status, (json.loads(data) if data else None)

    def _payload(self) -> dict:
        envelope = {
            "session_id": self.session_id,
            "sender_device_id": "d1",
            "message_id": "m1",
            "sequence": 1,
            "nonce": base64.b64encode(os.urandom(12)).decode(),
            "ciphertext": base64.b64encode(os.urandom(32)).decode(),
        }
        signed = sign_message(envelope, _seed_b64(self.private))
        signed["request_id"] = "req-1"
        return signed

    def test_submit_replay_and_conflict(self) -> None:
        payload = self._payload()
        status, first = self._request(
            "POST", "/v1/messages/submit-verified", payload)
        self.assertEqual(status, 201)
        self.assertEqual(set(first), _VIEW_FIELDS)
        status, replay = self._request(
            "POST", "/v1/messages/submit-verified", payload)
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)
        changed = dict(payload, message_id="m2", sequence=2)
        status, body = self._request(
            "POST", "/v1/messages/submit-verified", changed)
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "request_id")

    def test_bad_json_is_400_request_body(self) -> None:
        status, body = self._request(
            "POST", "/v1/messages/submit-verified", raw=b"{not json")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "request_body")

    def test_non_object_is_400_request_body(self) -> None:
        for raw in (b"[1, 2]", b"5", b'"x"'):
            status, body = self._request(
                "POST", "/v1/messages/submit-verified", raw=raw)
            self.assertEqual(status, 400, raw)
            self.assertEqual(body["field"], "request_body", raw)

    def test_bad_request_id_is_400_request_id(self) -> None:
        payload = self._payload()
        del payload["request_id"]
        status, body = self._request(
            "POST", "/v1/messages/submit-verified", payload)
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "request_id")

    def test_bad_signature_is_400_signature(self) -> None:
        payload = self._payload()
        raw = bytearray(base64.b64decode(payload["signature"]))
        raw[0] ^= 1
        payload["signature"] = base64.b64encode(bytes(raw)).decode()
        status, body = self._request(
            "POST", "/v1/messages/submit-verified", payload)
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "signature")

    def test_ordinary_submit_route_unchanged(self) -> None:
        payload = {
            "request_id": "req-old",
            "session_id": self.session_id,
            "sender_device_id": "d1",
            "message_id": "m0",
            "sequence": 1,
            "nonce": base64.b64encode(b"nonce-old-12xy").decode(),
            "ciphertext": base64.b64encode(b"ciphertext-and-tag").decode(),
        }
        status, body = self._request("POST", "/v1/messages/submit", payload)
        self.assertEqual(status, 201)
        self.assertNotIn("identity_key", body)
        self.assertNotIn("signature", body)


if __name__ == "__main__":
    unittest.main()
