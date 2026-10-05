"""Tests for the historical signed-message proof query.

Covers ``GET /v1/messages/{session_id}/proof/{message_id}?device_id=…``
at the service, HTTP and CLI layers. Only messages committed through
``POST /v1/messages/submit-verified`` carry a proof; an ordinary
idempotent submission, a direct send and legacy data have none, and a
proof query never back-fills a signature. The successful response is
exactly the six ``sign_message`` envelope fields plus the frozen
``identity_key`` and ``signature`` — the first-submission original
values, directly consumable by ``verify_message`` with the fingerprint
trusted at that time. The query is read-only: it advances no cursor,
delivery state, audit chain or commit generation, and it is linearized
under the store lock against submissions and revocations.
"""
import base64
import hashlib
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

from e2ee_backend.crypto import (identity_fingerprint, sign_message,
                                 verify_message)
from e2ee_backend.http_app import create_server
from e2ee_backend.persistence import attach_persistence
from e2ee_backend.service import DeviceService, ServiceError


def _raw_b64(key) -> str:
    return base64.b64encode(key.public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw)).decode()


def _seed_b64(private) -> str:
    return base64.b64encode(private.private_bytes(
        serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
        serialization.NoEncryption())).decode()


def _x25519_b64() -> str:
    return _raw_b64(x25519.X25519PrivateKey.generate().public_key())


_ENVELOPE_FIELDS = ("session_id", "sender_device_id", "message_id",
                    "sequence", "nonce", "ciphertext")
_PROOF_FIELDS = (*_ENVELOPE_FIELDS, "identity_key", "signature")


def _register_payload(device_id: str, identity_key=None) -> dict:
    return {
        "user_id": f"user-{device_id}",
        "device_id": device_id,
        "identity_key": identity_key or _x25519_b64(),
        "signed_prekeys": [
            {"key_id": "k1", "public_key": _x25519_b64()},
            {"key_id": "k2", "public_key": _x25519_b64()},
        ],
    }


class ProofFixture:
    """A service with d1 (Ed25519 identity), d2 and one 1:1 session."""

    def __init__(self, service=None) -> None:
        self.service = service or DeviceService()
        self.private = ed25519.Ed25519PrivateKey.generate()
        self.seed = _seed_b64(self.private)
        self.identity = _raw_b64(self.private.public_key())
        self.fingerprint = identity_fingerprint(self.identity)
        self.service.register(
            _register_payload("d1", identity_key=self.identity))
        self.service.register(_register_payload("d2"))
        session = self.service.create_session({
            "initiator_device_id": "d1",
            "recipient_device_id": "d2",
            "prekey_id": "k1",
            "ephemeral_key": _x25519_b64(),
        })
        self.session_id = session["session_id"]

    def verified_payload(self, *, request_id="req-1", message_id="m1",
                         sequence=1, session_id=None, sender="d1",
                         private=None) -> dict:
        digest = hashlib.sha256(
            f"{session_id or self.session_id}:{message_id}".encode()
        ).digest()
        nonce = base64.b64encode(digest[:12]).decode()
        ciphertext = base64.b64encode(digest[12:] + b"\x00" * 16).decode()
        envelope = {
            "session_id": session_id or self.session_id,
            "sender_device_id": sender,
            "message_id": message_id,
            "sequence": sequence,
            "nonce": nonce,
            "ciphertext": ciphertext,
        }
        signed = sign_message(envelope, _seed_b64(private or self.private))
        signed["request_id"] = request_id
        return signed

    def submit_verified(self, **kwargs) -> dict:
        payload = self.verified_payload(**kwargs)
        body, status = self.service.submit_verified_message(payload)
        assert status in (200, 201)
        return body


class ServiceMessageProofTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = ProofFixture()
        self.service = self.fixture.service
        self.session_id = self.fixture.session_id
        self.body = self.fixture.submit_verified()

    # -- success -----------------------------------------------------------

    def test_success_has_exactly_eight_fields(self) -> None:
        proof = self.service.message_proof(self.session_id, "m1", "d2")
        self.assertEqual(set(proof), set(_PROOF_FIELDS))
        self.assertNotIn("request_id", proof)
        self.assertNotIn("created_at", proof)

    def test_proof_keeps_first_submission_values(self) -> None:
        proof = self.service.message_proof(self.session_id, "m1", "d2")
        for name in _ENVELOPE_FIELDS:
            self.assertEqual(proof[name], self.body[name])
        self.assertEqual(proof["identity_key"], self.body["identity_key"])
        self.assertEqual(proof["signature"], self.body["signature"])

    def test_proof_directly_verifies_with_trusted_fingerprint(self) -> None:
        proof = self.service.message_proof(self.session_id, "m1", "d2")
        result = verify_message(
            proof, self.session_id, "d1", self.fixture.fingerprint)
        self.assertEqual(set(result), set(_PROOF_FIELDS) | {"fingerprint"})
        self.assertEqual(result["fingerprint"], self.fixture.fingerprint)

    def test_sender_and_recipient_may_both_query(self) -> None:
        by_sender = self.service.message_proof(self.session_id, "m1", "d1")
        by_recipient = self.service.message_proof(
            self.session_id, "m1", "d2")
        self.assertEqual(by_sender, by_recipient)

    def test_unicode_and_slash_identifiers_preserved(self) -> None:
        payload = self.fixture.verified_payload(
            request_id="req-uni", message_id=" 消息/一 ", sequence=2)
        self.service.submit_verified_message(payload)
        proof = self.service.message_proof(
            self.session_id, " 消息/一 ", "d2")
        self.assertEqual(proof["message_id"], " 消息/一 ")
        verify_message(proof, self.session_id, "d1",
                       self.fixture.fingerprint)

    def test_repeated_queries_are_identical_and_read_only(self) -> None:
        changes = []
        self.service.store.on_change = lambda: changes.append(1)
        first = self.service.message_proof(self.session_id, "m1", "d2")
        second = self.service.message_proof(self.session_id, "m1", "d1")
        self.assertEqual(second, first)
        self.assertEqual(changes, [])  # no persistence/change notification

    # -- ordered error cases ----------------------------------------------

    def test_unknown_session_is_404_session_id(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.message_proof("ghost", "m1", "d2")
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "session_id")

    def test_unknown_device_is_409_device_id(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.message_proof(self.session_id, "m1", "ghost")
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "device_id")

    def test_outsider_device_is_409_device_id(self) -> None:
        self.service.register(_register_payload("d3"))
        with self.assertRaises(ServiceError) as ctx:
            self.service.message_proof(self.session_id, "m1", "d3")
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "device_id")

    def test_session_check_precedes_device_check(self) -> None:
        # An unknown device against an unknown session still reports session.
        with self.assertRaises(ServiceError) as ctx:
            self.service.message_proof("ghost", "m1", "ghost")
        self.assertEqual(ctx.exception.field, "session_id")

    def test_device_check_precedes_message_check(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.message_proof(self.session_id, "ghost", "ghost")
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "device_id")

    def test_unknown_message_is_404_message_id(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.message_proof(self.session_id, "ghost", "d2")
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "message_id")

    def test_ordinary_submission_has_no_proof(self) -> None:
        self.service.submit_message({
            "request_id": "plain-1",
            "session_id": self.session_id,
            "sender_device_id": "d1",
            "message_id": "plain-m1",
            "sequence": 2,
            "nonce": base64.b64encode(b"plain-nonce-x").decode(),
            "ciphertext": base64.b64encode(b"plain-ciphertext!!").decode(),
        })
        with self.assertRaises(ServiceError) as ctx:
            self.service.message_proof(
                self.session_id, "plain-m1", "d2")
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "signature")
        # The failed query did not back-fill anything.
        with self.assertRaises(ServiceError) as ctx:
            self.service.message_proof(
                self.session_id, "plain-m1", "d2")
        self.assertEqual(ctx.exception.field, "signature")

    def test_direct_send_has_no_proof(self) -> None:
        self.service.post_message({
            "session_id": self.session_id,
            "sender_device_id": "d1",
            "message_id": "direct-m1",
            "sequence": 2,
            "nonce": base64.b64encode(b"direct-nonce-x").decode(),
            "ciphertext": base64.b64encode(b"direct-ciphertext!").decode(),
        })
        with self.assertRaises(ServiceError) as ctx:
            self.service.message_proof(
                self.session_id, "direct-m1", "d2")
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "signature")

    # -- history is immutable ---------------------------------------------

    def test_proof_survives_identity_rotation(self) -> None:
        before = self.service.message_proof(self.session_id, "m1", "d2")
        new_identity = _raw_b64(
            ed25519.Ed25519PrivateKey.generate().public_key())
        self.service.rotate_identity_key("d1",
                                         {"identity_key": new_identity})
        after = self.service.message_proof(self.session_id, "m1", "d2")
        self.assertEqual(after, before)
        # The frozen key still verifies against the old fingerprint.
        verify_message(after, self.session_id, "d1",
                       self.fixture.fingerprint)

    def test_proof_survives_sender_revocation(self) -> None:
        before = self.service.message_proof(self.session_id, "m1", "d2")
        self.service.revoke_device("d1")
        # The recipient is still active: history stays readable.
        after = self.service.message_proof(self.session_id, "m1", "d2")
        self.assertEqual(after, before)

    def test_revoked_reader_is_rejected(self) -> None:
        self.service.revoke_device("d2")
        with self.assertRaises(ServiceError) as ctx:
            self.service.message_proof(self.session_id, "m1", "d2")
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "device_id")

    def test_proof_survives_session_rotation(self) -> None:
        before = self.service.message_proof(self.session_id, "m1", "d2")
        self.service.rotate_session(self.session_id, {
            "rotation_id": "rot-1", "actor_device_id": "d1",
            "prekey_id": "k2", "ephemeral_key": _x25519_b64()})
        after = self.service.message_proof(self.session_id, "m1", "d2")
        self.assertEqual(after, before)

    def test_pull_format_is_unchanged(self) -> None:
        page = self.service.list_messages(self.session_id, "d2", 0, 100)
        self.assertEqual(
            set(page["messages"][0]),
            {"session_id", "sender_device_id", "message_id", "sequence",
             "nonce", "ciphertext", "created_at"})


class GroupMessageProofTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = ProofFixture()
        self.service = self.fixture.service
        for device_id in ("d3", "d4"):
            self.service.register(_register_payload(device_id))
        self.service.create_group({
            "group_id": "g1", "creator_device_id": "d1",
            "member_device_ids": ["d2"]})
        session = self.service.create_group_session({
            "group_id": "g1", "initiator_device_id": "d1",
            "ephemeral_key": _x25519_b64()})
        self.group_session_id = session["session_id"]
        self.fixture.submit_verified(
            request_id="g-req-1", message_id="gm1",
            session_id=self.group_session_id)

    def test_frozen_members_and_sender_may_query(self) -> None:
        for device_id in ("d1", "d2"):
            proof = self.service.message_proof(
                self.group_session_id, "gm1", device_id)
            self.assertEqual(set(proof), set(_PROOF_FIELDS))

    def test_outsider_and_late_added_member_rejected(self) -> None:
        # d3 never belonged to the frozen snapshot; d4 gets added after the
        # freeze and must still be refused (the snapshot is immutable).
        self.service.add_group_member("g1", {
            "device_id": "d4", "actor_device_id": "d1"})
        for device_id in ("d3", "d4"):
            with self.assertRaises(ServiceError) as ctx:
                self.service.message_proof(
                    self.group_session_id, "gm1", device_id)
            self.assertEqual(ctx.exception.status_code, 409, device_id)
            self.assertEqual(ctx.exception.field, "device_id", device_id)

    def test_removed_then_active_frozen_member_still_reads(self) -> None:
        # A frozen member removed from the live group remains a frozen
        # member; an active device keeps its read access.
        self.service.remove_group_member("g1", {
            "device_id": "d2", "actor_device_id": "d1"})
        proof = self.service.message_proof(
            self.group_session_id, "gm1", "d2")
        self.assertEqual(proof["message_id"], "gm1")


class MessageProofPersistenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.path = os.path.join(self.directory, "state.json")

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def _service(self) -> DeviceService:
        service = DeviceService()
        attach_persistence(service, self.path)
        return service

    def test_proof_identical_after_restart(self) -> None:
        service = self._service()
        fixture = ProofFixture(service)
        fixture.submit_verified()
        before = service.message_proof(
            fixture.session_id, "m1", "d2")

        restarted = self._service()
        after = restarted.message_proof(
            fixture.session_id, "m1", "d2")
        self.assertEqual(after, before)

    def test_proof_after_restart_and_rotation(self) -> None:
        service = self._service()
        fixture = ProofFixture(service)
        fixture.submit_verified()
        before = service.message_proof(
            fixture.session_id, "m1", "d2")
        service.rotate_identity_key("d1", {
            "identity_key": _raw_b64(
                ed25519.Ed25519PrivateKey.generate().public_key())})
        service.revoke_device("d2")
        restarted = self._service()
        # d2 is now revoked: d1 (still active) reads the frozen proof.
        after = restarted.message_proof(
            fixture.session_id, "m1", "d1")
        self.assertEqual(after, before)

    def test_legacy_message_has_no_proof_after_restart(self) -> None:
        service = self._service()
        fixture = ProofFixture(service)
        service.submit_message({
            "request_id": "legacy-1",
            "session_id": fixture.session_id,
            "sender_device_id": "d1",
            "message_id": "lm1",
            "sequence": 1,
            "nonce": base64.b64encode(b"legacy-nonce-x").decode(),
            "ciphertext": base64.b64encode(b"legacy-ciphertext!").decode(),
        })
        restarted = self._service()
        with self.assertRaises(ServiceError) as ctx:
            restarted.message_proof(fixture.session_id, "lm1", "d2")
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "signature")


class HTTPMessageProofTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = ProofFixture()
        self.fixture.submit_verified()
        # An ordinary (unverified) message exists but has no proof.
        self.fixture.service.submit_message({
            "request_id": "plain-1",
            "session_id": self.fixture.session_id,
            "sender_device_id": "d1",
            "message_id": "plain-m1",
            "sequence": 2,
            "nonce": base64.b64encode(b"plain-nonce-x").decode(),
            "ciphertext": base64.b64encode(b"plain-ciphertext!!").decode(),
        })
        self.server, _ = create_server(
            "127.0.0.1", 0, self.fixture.service)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _request(self, path: str, method="GET", body=None):
        connection = HTTPConnection("127.0.0.1", self.port, timeout=5)
        payload = json.dumps(body) if body is not None else None
        connection.request(
            method, path, body=payload,
            headers={"Content-Type": "application/json"} if payload else {})
        response = connection.getresponse()
        data = response.read().decode("utf-8")
        connection.close()
        return response.status, (json.loads(data) if data else None)

    def test_success_is_200_with_eight_fields(self) -> None:
        status, body = self._request(
            f"/v1/messages/{self.fixture.session_id}/proof/m1"
            f"?device_id=d2")
        self.assertEqual(status, 200)
        self.assertEqual(set(body), set(_PROOF_FIELDS))
        result = verify_message(
            body, self.fixture.session_id, "d1",
            self.fixture.fingerprint)
        self.assertEqual(result["fingerprint"], self.fixture.fingerprint)

    def test_extra_query_params_and_body_ignored(self) -> None:
        status, body = self._request(
            f"/v1/messages/{self.fixture.session_id}/proof/m1"
            f"?device_id=d2&other=kept", body={"anything": 1})
        self.assertEqual(status, 200)
        self.assertEqual(set(body), set(_PROOF_FIELDS))

    def test_device_id_param_must_appear_once_nonempty(self) -> None:
        base = f"/v1/messages/{self.fixture.session_id}/proof/m1"
        for suffix in ("", "?other=x", "?device_id=",
                       "?device_id=d2&device_id=d1",
                       "?device_id=&device_id=d2"):
            with self.subTest(suffix=suffix):
                status, body = self._request(base + suffix)
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], "device_id")

    def test_unknown_session_and_message_and_no_proof(self) -> None:
        status, body = self._request(
            "/v1/messages/ghost/proof/m1?device_id=d2")
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "session_id")
        status, body = self._request(
            f"/v1/messages/{self.fixture.session_id}/proof/ghost"
            f"?device_id=d2")
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "message_id")
        status, body = self._request(
            f"/v1/messages/{self.fixture.session_id}/proof/plain-m1"
            f"?device_id=d2")
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "signature")

    def test_device_eligibility_errors(self) -> None:
        self.fixture.service.register(_register_payload("d3"))
        base = f"/v1/messages/{self.fixture.session_id}/proof/m1"
        for device_id in ("ghost", "d3"):
            status, body = self._request(f"{base}?device_id={device_id}")
            self.assertEqual(status, 409, device_id)
            self.assertEqual(body["field"], "device_id", device_id)

    def test_path_segments_strictly_percent_decoded(self) -> None:
        base = "/v1/messages"
        # A percent-encoded slash is content, not a routing separator: the
        # session id decodes to "a/b" and is simply unknown.
        status, body = self._request(
            f"{base}/a%2Fb/proof/m1?device_id=d2")
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "session_id")
        self.assertEqual(body["message"], "session not found: a/b")
        # Malformed escapes and invalid UTF-8 are 400 naming the segment.
        status, body = self._request(f"{base}/%zz/proof/m1?device_id=d2")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "session_id")
        status, body = self._request(
            f"{base}/{self.fixture.session_id}/proof/%ff?device_id=d2")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "message_id")
        # Empty decoded segments are 400 naming that identifier.
        status, body = self._request(f"{base}//proof/m1?device_id=d2")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "session_id")
        status, body = self._request(
            f"{base}/{self.fixture.session_id}/proof/?device_id=d2")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "message_id")

    def test_unicode_identifiers_round_trip(self) -> None:
        payload = self.fixture.verified_payload(
            request_id="req-uni", message_id="消息一", sequence=3)
        self.fixture.service.submit_verified_message(payload)
        encoded = "%E6%B6%88%E6%81%AF%E4%B8%80"
        status, body = self._request(
            f"/v1/messages/{self.fixture.session_id}/proof/"
            f"{encoded}?device_id=d2")
        self.assertEqual(status, 200)
        self.assertEqual(body["message_id"], "消息一")

    def test_wrong_shape_falls_through_to_404(self) -> None:
        status, body = self._request(
            f"/v1/messages/{self.fixture.session_id}/proof/m1/extra"
            f"?device_id=d2")
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "session_id")


class CLIMessageProofTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = ProofFixture()
        self.fixture.submit_verified()
        self.server, _ = create_server(
            "127.0.0.1", 0, self.fixture.service)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _run(self, *arguments):
        return subprocess.run(
            [sys.executable, "-m", "e2ee_backend",
             f"--base-url=http://127.0.0.1:{self.port}",
             "message-proof", *arguments],
            capture_output=True, text=True, timeout=20)

    def test_success_prints_single_line_on_stdout(self) -> None:
        result = self._run(self.fixture.session_id, "m1",
                           "--device-id", "d2")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        self.assertEqual(result.stdout.count("\n"), 1)
        body = json.loads(result.stdout)
        self.assertEqual(set(body), set(_PROOF_FIELDS))
        verify_message(body, self.fixture.session_id, "d1",
                       self.fixture.fingerprint)

    def test_http_failure_prints_field_json_on_stderr(self) -> None:
        result = self._run("ghost", "m1", "--device-id", "d2")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        body = json.loads(result.stderr.strip())
        self.assertEqual(body["field"], "session_id")
        self.assertIn("message", body)
        self.assertEqual(set(body), {"message", "field"})

    def test_missing_arguments_name_the_input(self) -> None:
        cases = (
            ((), "session_id"),
            (("s1",), "message_id"),
            (("s1", "m1"), "device_id"),
            (("s1", "m1", "--device-id="), "device_id"),
        )
        for arguments, field in cases:
            with self.subTest(field=field):
                result = self._run(*arguments)
                self.assertEqual(result.returncode, 1, arguments)
                self.assertEqual(result.stdout, "")
                body = json.loads(result.stderr.strip())
                self.assertEqual(body["field"], field)
                self.assertEqual(set(body), {"message", "field"})

    def test_connection_failure_names_server(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "e2ee_backend",
             "--base-url=http://127.0.0.1:1",
             "message-proof", "s1", "m1", "--device-id", "d1"],
            capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        body = json.loads(result.stderr.strip())
        self.assertEqual(body["field"], "server")
        self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
