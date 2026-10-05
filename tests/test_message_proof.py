"""Tests for the historical message signature proof query.

Covers the service, HTTP (real loopback socket), CLI and persistence
layers for ``GET /v1/messages/{session_id}/proof/{message_id}`` and the
``message-proof`` CLI command.

Only messages committed through ``POST /v1/messages/submit-verified``
carry a proof; ordinary submissions, direct sends and legacy data do not,
and the query never backfills one. The proof freezes the first verified
submission's eight values (the six ``sign_message`` envelope fields plus
``identity_key`` and ``signature``), so it is directly acceptable by
``verify_message`` with the sender's then-current trusted fingerprint and
survives identity rotation, sender revocation, group roster changes and
session rotation. The query is a pure read: it moves no cursor, delivery
state, audit chain or commit generation.
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

from e2ee_backend.crypto import identity_fingerprint, sign_message, \
    verify_message
from e2ee_backend.http_app import create_server
from e2ee_backend.persistence import attach_persistence
from e2ee_backend.service import DeviceService, ServiceError


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
    return {
        "user_id": user_id,
        "device_id": device_id,
        "identity_key": identity_key or _x25519_b64(),
        "signed_prekeys": [
            {"key_id": "k1", "public_key": _x25519_b64()},
            {"key_id": "k2", "public_key": _x25519_b64()},
        ],
    }


_PROOF_FIELDS = {"session_id", "sender_device_id", "message_id",
                 "sequence", "nonce", "ciphertext",
                 "identity_key", "signature"}


class ProofFixture:
    """A service with devices d1 (Ed25519 identity) and d2, one 1:1 session.

    d1's private seed is kept so tests can sign envelopes with
    ``sign_message`` and submit them through the verified entry.
    """

    def __init__(self, service=None) -> None:
        self.service = service if service is not None else DeviceService()
        self.private = ed25519.Ed25519PrivateKey.generate()
        self.seed = _seed_b64(self.private)
        self.identity = _raw_b64(self.private.public_key())
        self.service.register(_register_payload(
            "d1", identity_key=self.identity))
        self.service.register(_register_payload("d2", user_id="u2"))
        session = self.service.create_session({
            "initiator_device_id": "d1",
            "recipient_device_id": "d2",
            "prekey_id": "k1",
            "ephemeral_key": _x25519_b64(),
        })
        self.session_id = session["session_id"]

    def envelope(self, request_id="req-1", message_id="m1", sequence=1,
                 sender="d1", session_id=None) -> dict:
        digest = hashlib.sha256(message_id.encode("utf-8")).digest()
        envelope = {
            "session_id": session_id or self.session_id,
            "sender_device_id": sender,
            "message_id": message_id,
            "sequence": sequence,
            "nonce": base64.b64encode(digest[:12]).decode(),
            "ciphertext": base64.b64encode(
                digest[12:] + b"\x00" * 16).decode(),
        }
        signed = sign_message(envelope, self.seed)
        signed["request_id"] = request_id
        return signed

    def submit_verified(self, **overrides):
        payload = self.envelope()
        payload.update(overrides)
        return self.service.submit_verified_message(payload)


class MessageProofServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = ProofFixture()
        self.service = self.fixture.service
        self.fixture.submit_verified()

    def test_proof_returns_exactly_the_eight_fields(self) -> None:
        proof = self.service.message_proof(
            self.fixture.session_id, "m1", "d2")
        self.assertEqual(set(proof), _PROOF_FIELDS)
        self.assertEqual(proof["session_id"], self.fixture.session_id)
        self.assertEqual(proof["sender_device_id"], "d1")
        self.assertEqual(proof["message_id"], "m1")
        self.assertEqual(proof["sequence"], 1)
        self.assertEqual(proof["identity_key"], self.fixture.identity)

    def test_proof_verifies_with_then_current_fingerprint(self) -> None:
        proof = self.service.message_proof(
            self.fixture.session_id, "m1", "d2")
        verified = verify_message(
            proof, proof["session_id"], proof["sender_device_id"],
            identity_fingerprint(proof["identity_key"]))
        self.assertEqual(verified["signature"], proof["signature"])

    def test_ordinary_submit_has_no_proof(self) -> None:
        body, status = self.service.submit_message({
            "request_id": "plain-1",
            "session_id": self.fixture.session_id,
            "sender_device_id": "d1",
            "message_id": "m-plain",
            "sequence": 2,
            "nonce": base64.b64encode(b"p" * 12).decode(),
            "ciphertext": base64.b64encode(b"c" * 16).decode(),
        })
        self.assertEqual(status, 201)
        with self.assertRaises(ServiceError) as ctx:
            self.service.message_proof(
                self.fixture.session_id, "m-plain", "d2")
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "signature")

    def test_direct_send_has_no_proof(self) -> None:
        self.service.post_message({
            "session_id": self.fixture.session_id,
            "sender_device_id": "d1",
            "message_id": "m-direct",
            "sequence": 2,
            "nonce": base64.b64encode(b"d" * 12).decode(),
            "ciphertext": base64.b64encode(b"c" * 16).decode(),
        })
        with self.assertRaises(ServiceError) as ctx:
            self.service.message_proof(
                self.fixture.session_id, "m-direct", "d2")
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "signature")

    def test_unknown_session_404_session_id(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.message_proof("ghost", "m1", "d2")
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "session_id")

    def test_unknown_device_409_device_id(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.message_proof(self.fixture.session_id, "m1", "ghost")
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "device_id")

    def test_revoked_reader_409_device_id(self) -> None:
        self.service.revoke_device("d2")
        with self.assertRaises(ServiceError) as ctx:
            self.service.message_proof(self.fixture.session_id, "m1", "d2")
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "device_id")

    def test_one_to_one_readable_by_any_active_device(self) -> None:
        self.service.register(_register_payload("d3", user_id="u3"))
        proof = self.service.message_proof(
            self.fixture.session_id, "m1", "d3")
        self.assertEqual(proof["message_id"], "m1")

    def test_unknown_message_404_message_id(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.message_proof(
                self.fixture.session_id, "ghost", "d2")
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "message_id")

    def test_identity_rotation_keeps_frozen_proof(self) -> None:
        self.service.rotate_identity_key(
            "d1", {"identity_key": _x25519_b64()})
        proof = self.service.message_proof(
            self.fixture.session_id, "m1", "d2")
        self.assertEqual(proof["identity_key"], self.fixture.identity)
        verified = verify_message(
            proof, proof["session_id"], "d1",
            identity_fingerprint(self.fixture.identity))
        self.assertEqual(verified["message_id"], "m1")

    def test_sender_revocation_keeps_proof(self) -> None:
        self.service.revoke_device("d1")
        proof = self.service.message_proof(
            self.fixture.session_id, "m1", "d2")
        self.assertEqual(proof["sender_device_id"], "d1")
        self.assertEqual(proof["identity_key"], self.fixture.identity)

    def test_session_rotation_keeps_proof(self) -> None:
        _, status = self.service.rotate_session(
            self.fixture.session_id, {
                "rotation_id": "rot-1",
                "actor_device_id": "d1",
                "prekey_id": "k2",
                "ephemeral_key": _x25519_b64(),
            })
        self.assertEqual(status, 201)
        proof = self.service.message_proof(
            self.fixture.session_id, "m1", "d2")
        self.assertEqual(proof["session_id"], self.fixture.session_id)

    def test_query_changes_nothing(self) -> None:
        directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, directory, True)
        path = os.path.join(directory, "state.json")
        fixture = ProofFixture()
        store = attach_persistence(fixture.service, path)
        fixture.submit_verified()
        generation = store.commit_seq
        for _ in range(3):
            fixture.service.message_proof(fixture.session_id, "m1", "d2")
        self.assertEqual(store.commit_seq, generation)
        events = fixture.service.list_key_events("d1", 0, 100)["events"]
        self.assertEqual([e["type"] for e in events], ["registered"])


class MessageProofGroupTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = ProofFixture()
        self.service = self.fixture.service
        self.service.register(_register_payload("d3", user_id="u3"))
        self.service.create_group({
            "group_id": "g1", "creator_device_id": "d1",
            "member_device_ids": ["d2"]})
        session = self.service.create_group_session({
            "group_id": "g1", "initiator_device_id": "d1",
            "ephemeral_key": _x25519_b64()})
        self.group_session_id = session["session_id"]
        self.service.submit_verified_message(
            self.fixture.envelope(session_id=self.group_session_id))

    def test_frozen_member_reads_proof(self) -> None:
        proof = self.service.message_proof(
            self.group_session_id, "m1", "d2")
        self.assertEqual(proof["session_id"], self.group_session_id)

    def test_non_member_active_device_409_device_id(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.message_proof(self.group_session_id, "m1", "d3")
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "device_id")

    def test_roster_change_after_freeze_does_not_change_proof(self) -> None:
        # d3 joins the group after the session froze its member set: still
        # not a reader, and the saved proof is untouched.
        self.service.add_group_member("g1", {
            "actor_device_id": "d1", "device_id": "d3"})
        with self.assertRaises(ServiceError) as ctx:
            self.service.message_proof(self.group_session_id, "m1", "d3")
        self.assertEqual(ctx.exception.field, "device_id")
        proof = self.service.message_proof(
            self.group_session_id, "m1", "d2")
        self.assertEqual(proof["identity_key"], self.fixture.identity)
        # A frozen member removed from the group afterwards keeps reading.
        self.service.remove_group_member("g1", {
            "actor_device_id": "d1", "device_id": "d2"})
        proof = self.service.message_proof(
            self.group_session_id, "m1", "d2")
        self.assertEqual(proof["message_id"], "m1")


class MessageProofPersistenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.path = os.path.join(self.directory, "state.json")

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_restart_serves_identical_proof(self) -> None:
        fixture = ProofFixture()
        attach_persistence(fixture.service, self.path)
        fixture.submit_verified()
        before = fixture.service.message_proof(
            fixture.session_id, "m1", "d2")

        restored = DeviceService()
        attach_persistence(restored, self.path)
        after = restored.message_proof(fixture.session_id, "m1", "d2")
        self.assertEqual(after, before)
        self.assertEqual(set(after), _PROOF_FIELDS)

    def test_restart_keeps_no_proof_for_ordinary_submit(self) -> None:
        fixture = ProofFixture()
        attach_persistence(fixture.service, self.path)
        fixture.service.submit_message({
            "request_id": "plain-1",
            "session_id": fixture.session_id,
            "sender_device_id": "d1",
            "message_id": "m-plain",
            "sequence": 1,
            "nonce": base64.b64encode(b"p" * 12).decode(),
            "ciphertext": base64.b64encode(b"c" * 16).decode(),
        })
        restored = DeviceService()
        attach_persistence(restored, self.path)
        with self.assertRaises(ServiceError) as ctx:
            restored.message_proof(fixture.session_id, "m-plain", "d2")
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "signature")


class MessageProofHTTPTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = ProofFixture()
        self.service = self.fixture.service
        self.server, _ = create_server("127.0.0.1", 0, self.service)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        self.fixture.submit_verified()
        self.session_id = self.fixture.session_id

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _get(self, path, body=None):
        connection = HTTPConnection("127.0.0.1", self.port, timeout=5)
        connection.request(
            "GET", path,
            body=json.dumps(body) if body is not None else None)
        response = connection.getresponse()
        data = response.read().decode("utf-8")
        connection.close()
        return response.status, json.loads(data) if data else None

    def _proof_path(self, session_id=None, message_id="m1", query=""):
        return (f"/v1/messages/{session_id or self.session_id}"
                f"/proof/{message_id}{query}")

    def test_proof_200_exact_fields(self) -> None:
        status, body = self._get(self._proof_path(query="?device_id=d2"))
        self.assertEqual(status, 200)
        self.assertEqual(set(body), _PROOF_FIELDS)
        self.assertEqual(body["identity_key"], self.fixture.identity)
        verified = verify_message(
            body, body["session_id"], "d1",
            identity_fingerprint(self.fixture.identity))
        self.assertEqual(verified["message_id"], "m1")

    def test_other_params_and_body_ignored(self) -> None:
        status, _ = self._get(
            self._proof_path(query="?device_id=d2&anything=%2F&x=1"),
            body={"x": 1})
        self.assertEqual(status, 200)

    def test_device_id_param_contract(self) -> None:
        for query in ("", "?device_id=", "?device_id=d2&device_id=d2",
                      "?device_id=&device_id=d2"):
            status, body = self._get(self._proof_path(query=query))
            self.assertEqual(status, 400, query)
            self.assertEqual(body["field"], "device_id", query)

    def test_unknown_session_404(self) -> None:
        status, body = self._get(
            self._proof_path(session_id="ghost", query="?device_id=d2"))
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "session_id")

    def test_unknown_message_404(self) -> None:
        status, body = self._get(
            self._proof_path(message_id="ghost", query="?device_id=d2"))
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "message_id")

    def test_ineligible_device_409(self) -> None:
        status, body = self._get(self._proof_path(query="?device_id=ghost"))
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "device_id")

    def test_strict_percent_decoding(self) -> None:
        # A bad escape or invalid UTF-8 in either identifier is 400 naming
        # that identifier; the session segment is checked first.
        for path, field in (
                ("/v1/messages/%ZZ/proof/m1?device_id=d2", "session_id"),
                ("/v1/messages/%FF/proof/m1?device_id=d2", "session_id"),
                ("/v1/messages//proof/m1?device_id=d2", "session_id"),
                (f"/v1/messages/{self.session_id}/proof/%ZZ?device_id=d2",
                 "message_id"),
                (f"/v1/messages/{self.session_id}/proof/?device_id=d2",
                 "message_id"),
                ("/v1/messages/%ZZ/proof/%ZZ?device_id=d2", "session_id")):
            status, body = self._get(path)
            self.assertEqual(status, 400, path)
            self.assertEqual(body["field"], field, path)

    def test_encoded_slash_is_identifier_content(self) -> None:
        payload = self.fixture.envelope(request_id="req-slash",
                                        message_id="a/b", sequence=2)
        self.service.submit_verified_message(payload)
        status, body = self._get(
            f"/v1/messages/{self.session_id}/proof/a%2Fb?device_id=d2")
        self.assertEqual(status, 200)
        self.assertEqual(body["message_id"], "a/b")

    def test_deeper_or_shallower_path_404(self) -> None:
        status, body = self._get(
            f"/v1/messages/{self.session_id}/proof/m1/extra?device_id=d2")
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "session_id")
        status, body = self._get(
            f"/v1/messages/{self.session_id}/proof?device_id=d2")
        self.assertEqual(status, 404)


class MessageProofCLITest(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = ProofFixture()
        self.server, _ = create_server("127.0.0.1", 0, self.fixture.service)
        self.port = self.server.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        self.fixture.submit_verified()
        self.session_id = self.fixture.session_id

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _run(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "e2ee_backend", "--base-url", self.base_url,
             *arguments],
            capture_output=True, text=True, timeout=15)

    def test_success_prints_proof_and_exits_zero(self) -> None:
        result = self._run("message-proof", self.session_id, "m1",
                           "--device-id", "d2")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        line = result.stdout.strip()
        self.assertEqual(line.count("\n"), 0)
        body = json.loads(line)
        self.assertEqual(set(body), _PROOF_FIELDS)
        self.assertEqual(body["identity_key"], self.fixture.identity)

    def test_http_failure_prints_error_on_stderr(self) -> None:
        result = self._run("message-proof", self.session_id, "ghost",
                           "--device-id", "d2")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        body = json.loads(result.stderr.strip())
        self.assertEqual(body["field"], "message_id")
        self.assertIn("message", body)

    def test_missing_arguments_name_their_field(self) -> None:
        result = self._run("message-proof", self.session_id, "m1")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        self.assertEqual(json.loads(result.stderr.strip())["field"],
                         "device_id")
        result = self._run("message-proof", self.session_id)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(json.loads(result.stderr.strip())["field"],
                         "message_id")
        result = self._run("message-proof")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(json.loads(result.stderr.strip())["field"],
                         "session_id")

    def test_connection_failure_field_server(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "e2ee_backend",
             "--base-url", "http://127.0.0.1:1",
             "message-proof", self.session_id, "m1", "--device-id", "d2"],
            capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        self.assertEqual(json.loads(result.stderr.strip())["field"],
                         "server")


if __name__ == "__main__":
    unittest.main()
