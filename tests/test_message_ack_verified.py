"""Tests for the signature-authorized single-message ack endpoint.

POST /v1/messages/{session_id}/acks-verified behaves like the plain
per-message ack, but the acknowledgement is authorized by an Ed25519
signature from the device's *current* identity key over the canonical
``E2EE-MESSAGE-ACK-V1`` message (device_id / expected_version /
message_id / sequence / session_id / user_id), with ``expected_version``
pinned to the current ``identity_key_version``. The authorization and the
ack commit atomically with revocation and identity rotation; a durable
write failure rolls everything back.
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

from e2ee_backend.crypto import message_ack_proof_message
from e2ee_backend.models import Device, SignedPreKey
from e2ee_backend.persistence import PersistenceUnavailable, attach_persistence
from e2ee_backend.service import DeviceService, ServiceError
from e2ee_backend.http_app import create_server


def _raw_b64(key) -> str:
    raw = key.public_bytes(serialization.Encoding.Raw,
                           serialization.PublicFormat.Raw)
    return base64.b64encode(raw).decode()


def _new_identity():
    private = ed25519.Ed25519PrivateKey.generate()
    return private, _raw_b64(private.public_key())


def _x25519_b64() -> str:
    return _raw_b64(x25519.X25519PrivateKey.generate().public_key())


def _authorization(private, user_id, device_id, session_id, message_id,
                   sequence, expected_version):
    message = message_ack_proof_message(
        user_id, device_id, session_id, message_id, sequence,
        expected_version)
    return base64.b64encode(private.sign(message)).decode()


class AckVerifiedMixin:
    def _build(self) -> None:
        self.service = DeviceService()
        self.bob_private, self.bob_identity = _new_identity()
        self.service.store.add_device(Device("u-a", "alice", "ik"))
        self.service.store.add_device(Device(
            "u", "bob", self.bob_identity,
            prekeys=[SignedPreKey("pk1", "pubk1")]))
        # carol's identity key is not an Ed25519 public key.
        self.service.store.add_device(Device(
            "u-c", "carol", "ik", prekeys=[SignedPreKey("pkc", "pubkc")]))
        self.sid = self.service.store.create_session(
            "alice", "bob", "pk1", "ek1").session_id
        self.service.post_message({
            "session_id": self.sid, "sender_device_id": "alice",
            "message_id": "m1", "sequence": 1, "nonce": "n1",
            "ciphertext": "ct"})
        self.carol_sid = self.service.store.create_session(
            "alice", "carol", "pkc", "ek2").session_id
        self.service.post_message({
            "session_id": self.carol_sid, "sender_device_id": "alice",
            "message_id": "m1", "sequence": 1, "nonce": "n1",
            "ciphertext": "ct"})

    def _payload(self, session_id=None, device_id="bob", message_id="m1",
                 sequence=1, expected_version=1, private=None, user_id="u",
                 signature=None):
        if session_id is None:
            session_id = self.sid
        if signature is None:
            signer = private if private is not None else self.bob_private
            signature = _authorization(
                signer, user_id, device_id, session_id, message_id,
                sequence, expected_version)
        return {"device_id": device_id, "message_id": message_id,
                "sequence": sequence, "expected_version": expected_version,
                "signature": signature}

    def _group_session(self) -> str:
        self.service.create_group({
            "group_id": "g1", "creator_device_id": "alice",
            "member_device_ids": ["bob"]})
        gs = self.service.create_group_session({
            "group_id": "g1", "initiator_device_id": "alice",
            "ephemeral_key": "epk"})
        sid = gs["session_id"]
        self.service.post_message({
            "session_id": sid, "sender_device_id": "alice",
            "message_id": "g1", "sequence": 1, "nonce": "gn1",
            "ciphertext": "ct"})
        return sid


class AckVerifiedServiceTest(AckVerifiedMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def _error(self, callable_):
        with self.assertRaises(ServiceError) as caught:
            callable_()
        return caught.exception

    def _call(self, session_id, payload):
        return self.service.ack_message_verified(session_id, payload)

    def test_first_ack_is_201_and_replay_is_200(self) -> None:
        body, status = self._call(self.sid, self._payload())
        self.assertEqual(status, 201)
        self.assertEqual(list(body),
                         ["session_id", "message_id", "status", "attempts",
                          "sequence"])
        self.assertEqual(body, {"session_id": self.sid, "message_id": "m1",
                                "status": "acked", "attempts": 0,
                                "sequence": 1})
        body, status = self._call(self.sid, self._payload())
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "acked")
        state = self.service.store._delivery[(self.sid, "m1")]
        self.assertTrue(state.acked)
        self.assertEqual(state.ack_sequence, 1)

    def test_group_session_ack_is_per_device(self) -> None:
        sid = self._group_session()
        body, status = self._call(sid, self._payload(
            session_id=sid, message_id="g1"))
        self.assertEqual(status, 201)
        self.assertEqual(body["status"], "acked")
        key = (sid, "g1", "bob")
        self.assertTrue(self.service.store._group_delivery[key].acked)
        body, status = self._call(sid, self._payload(
            session_id=sid, message_id="g1"))
        self.assertEqual(status, 200)

    def test_replay_still_requires_valid_authorization(self) -> None:
        self._call(self.sid, self._payload())
        other_private, _ = _new_identity()
        error = self._error(lambda: self._call(self.sid, self._payload(
            private=other_private)))
        self.assertEqual((error.status_code, error.field),
                         (400, "signature"))

    def test_extra_fields_are_ignored(self) -> None:
        payload = self._payload()
        payload["unexpected"] = "ignored"
        body, status = self._call(self.sid, payload)
        self.assertEqual(status, 201)
        self.assertEqual(body["status"], "acked")

    def test_unicode_ids_sign_verbatim(self) -> None:
        private, identity = _new_identity()
        self.service.store.add_device(Device(
            "用户", "设备-甲", identity,
            prekeys=[SignedPreKey("pk", "pubk")]))
        sid = self.service.store.create_session(
            "alice", "设备-甲", "pk", "ek").session_id
        self.service.post_message({
            "session_id": sid, "sender_device_id": "alice",
            "message_id": "消息-1", "sequence": 1, "nonce": "n1",
            "ciphertext": "ct"})
        body, status = self._call(sid, self._payload(
            session_id=sid, device_id="设备-甲", message_id="消息-1",
            private=private, user_id="用户"))
        self.assertEqual(status, 201)
        self.assertEqual(body["message_id"], "消息-1")

    # -- validation ----------------------------------------------------------

    def test_body_validation(self) -> None:
        cases = [
            (None, "request_body"),
            ([], "request_body"),
            ("x", "request_body"),
            ({}, "device_id"),
            ({"device_id": ""}, "device_id"),
            ({"device_id": 5}, "device_id"),
            ({"device_id": "bob"}, "message_id"),
            ({"device_id": "bob", "message_id": ""}, "message_id"),
            ({"device_id": "bob", "message_id": "m1"}, "sequence"),
            ({"device_id": "bob", "message_id": "m1", "sequence": "1"},
             "sequence"),
            ({"device_id": "bob", "message_id": "m1", "sequence": True},
             "sequence"),
            ({"device_id": "bob", "message_id": "m1", "sequence": 0},
             "sequence"),
            ({"device_id": "bob", "message_id": "m1", "sequence": -1},
             "sequence"),
            ({"device_id": "bob", "message_id": "m1", "sequence": 1.5},
             "sequence"),
            ({"device_id": "bob", "message_id": "m1", "sequence": 1},
             "expected_version"),
            ({"device_id": "bob", "message_id": "m1", "sequence": 1,
              "expected_version": 0}, "expected_version"),
            ({"device_id": "bob", "message_id": "m1", "sequence": 1,
              "expected_version": True}, "expected_version"),
            ({"device_id": "bob", "message_id": "m1", "sequence": 1,
              "expected_version": "1"}, "expected_version"),
            ({"device_id": "bob", "message_id": "m1", "sequence": 1,
              "expected_version": 1}, "signature"),
            ({"device_id": "bob", "message_id": "m1", "sequence": 1,
              "expected_version": 1, "signature": ""}, "signature"),
            ({"device_id": "bob", "message_id": "m1", "sequence": 1,
              "expected_version": 1, "signature": 5}, "signature"),
            ({"device_id": "bob", "message_id": "m1", "sequence": 1,
              "expected_version": 1, "signature": "!!!"}, "signature"),
            # Standard base64 of the wrong length (32 bytes, not 64).
            ({"device_id": "bob", "message_id": "m1", "sequence": 1,
              "expected_version": 1,
              "signature": base64.b64encode(b"\x00" * 32).decode()},
             "signature"),
            # 64 bytes but non-canonical (trailing whitespace) encoding.
            ({"device_id": "bob", "message_id": "m1", "sequence": 1,
              "expected_version": 1,
              "signature": base64.b64encode(b"\x00" * 64).decode() + " "},
             "signature"),
        ]
        for payload, field in cases:
            error = self._error(lambda: self._call(self.sid, payload))
            self.assertEqual((error.status_code, error.field), (400, field),
                             payload)
        self.assertEqual(self.service.store._delivery, {})

    # -- store checks, in their fixed order ----------------------------------

    def test_session_unknown_is_404(self) -> None:
        error = self._error(lambda: self._call("ghost", self._payload(
            session_id="ghost")))
        self.assertEqual((error.status_code, error.field),
                         (404, "session_id"))

    def test_message_unknown_is_404(self) -> None:
        error = self._error(lambda: self._call(self.sid, self._payload(
            message_id="ghost")))
        self.assertEqual((error.status_code, error.field),
                         (404, "message_id"))

    def test_device_unknown_is_409(self) -> None:
        error = self._error(lambda: self._call(self.sid, self._payload(
            device_id="ghost")))
        self.assertEqual((error.status_code, error.field),
                         (409, "device_id"))

    def test_device_revoked_is_409(self) -> None:
        self.service.store.revoke_device("bob")
        error = self._error(lambda: self._call(self.sid, self._payload()))
        self.assertEqual((error.status_code, error.field),
                         (409, "device_id"))

    def test_non_recipient_is_409(self) -> None:
        # alice is the initiator, not the recipient; give her an Ed25519
        # identity so only the device-eligibility check can fail.
        alice_private, alice_identity = _new_identity()
        self.service.store.rotate_identity_key("alice", alice_identity)
        error = self._error(lambda: self._call(self.sid, self._payload(
            device_id="alice", expected_version=2, private=alice_private,
            user_id="u-a")))
        self.assertEqual((error.status_code, error.field),
                         (409, "device_id"))

    def test_group_sender_and_non_member_are_409(self) -> None:
        sid = self._group_session()
        alice_private, alice_identity = _new_identity()
        self.service.store.rotate_identity_key("alice", alice_identity)
        # The sender cannot acknowledge its own message.
        error = self._error(lambda: self._call(sid, self._payload(
            session_id=sid, device_id="alice", message_id="g1",
            expected_version=2, private=alice_private, user_id="u-a")))
        self.assertEqual((error.status_code, error.field),
                         (409, "device_id"))
        # carol was never frozen into the group session's member snapshot.
        carol_private, carol_identity = _new_identity()
        self.service.store.rotate_identity_key("carol", carol_identity)
        error = self._error(lambda: self._call(sid, self._payload(
            session_id=sid, device_id="carol", message_id="g1",
            expected_version=2, private=carol_private, user_id="u-c")))
        self.assertEqual((error.status_code, error.field),
                         (409, "device_id"))

    def test_non_ed25519_identity_key_is_400(self) -> None:
        # carol's identity key ("ik") is not Ed25519; the signature encoding
        # is valid but the algorithm check fires first — even with a wrong
        # version, which is checked afterwards.
        signature = base64.b64encode(b"\x00" * 64).decode()
        error = self._error(lambda: self._call(self.carol_sid, {
            "device_id": "carol", "message_id": "m1", "sequence": 1,
            "expected_version": 9, "signature": signature}))
        self.assertEqual((error.status_code, error.field),
                         (400, "identity_key"))

    def test_version_mismatch_is_409(self) -> None:
        error = self._error(lambda: self._call(self.sid, self._payload(
            expected_version=2)))
        self.assertEqual((error.status_code, error.field),
                         (409, "expected_version"))
        self.assertEqual(self.service.store._delivery, {})

    def test_version_checked_before_signature(self) -> None:
        # A well-formed but invalid signature does not mask the version
        # mismatch.
        error = self._error(lambda: self._call(self.sid, self._payload(
            expected_version=2,
            signature=base64.b64encode(b"\x00" * 64).decode())))
        self.assertEqual((error.status_code, error.field),
                         (409, "expected_version"))

    def test_wrong_signing_key_is_400(self) -> None:
        other_private, _ = _new_identity()
        error = self._error(lambda: self._call(self.sid, self._payload(
            private=other_private)))
        self.assertEqual((error.status_code, error.field),
                         (400, "signature"))

    def test_wrong_user_id_or_session_id_in_proof_is_400(self) -> None:
        error = self._error(lambda: self._call(self.sid, self._payload(
            user_id="other")))
        self.assertEqual((error.status_code, error.field),
                         (400, "signature"))
        # Signed over a different session id than the path's.
        error = self._error(lambda: self._call(self.sid, self._payload(
            signature=_authorization(
                self.bob_private, "u", "bob", self.carol_sid, "m1", 1, 1))))
        self.assertEqual((error.status_code, error.field),
                         (400, "signature"))

    def test_rotation_invalidates_old_authorization(self) -> None:
        _, new_identity = _new_identity()
        self.service.store.rotate_identity_key("bob", new_identity)
        error = self._error(lambda: self._call(self.sid, self._payload(
            expected_version=1)))
        self.assertEqual((error.status_code, error.field),
                         (409, "expected_version"))
        error = self._error(lambda: self._call(self.sid, self._payload(
            expected_version=2)))
        self.assertEqual((error.status_code, error.field),
                         (400, "signature"))
        self.assertEqual(self.service.store._delivery, {})

    def test_sequence_mismatch_is_409(self) -> None:
        # The signature verifies over the request's own (wrong) sequence;
        # the stored sequence check fires afterwards.
        error = self._error(lambda: self._call(self.sid, self._payload(
            sequence=5)))
        self.assertEqual((error.status_code, error.field),
                         (409, "sequence"))
        self.assertEqual(self.service.store._delivery, {})

    def test_signature_checked_before_sequence(self) -> None:
        # Signed over sequence 1 but the request says 5: the authorization
        # fails before the sequence check is reached.
        error = self._error(lambda: self._call(self.sid, self._payload(
            sequence=5,
            signature=_authorization(
                self.bob_private, "u", "bob", self.sid, "m1", 1, 1))))
        self.assertEqual((error.status_code, error.field),
                         (400, "signature"))


class AckVerifiedPersistenceTest(AckVerifiedMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self._build()
        self.state_store = attach_persistence(
            self.service, os.path.join(self.directory, "state.json"))

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_first_ack_persists_with_one_generation(self) -> None:
        generation = self.state_store.commit_seq
        self.service.ack_message_verified(self.sid, self._payload())
        self.assertEqual(self.state_store.commit_seq, generation + 1)

    def test_replay_consumes_no_generation(self) -> None:
        self.service.ack_message_verified(self.sid, self._payload())
        generation = self.state_store.commit_seq
        _, status = self.service.ack_message_verified(
            self.sid, self._payload())
        self.assertEqual(status, 200)
        self.assertEqual(self.state_store.commit_seq, generation)

    def test_failed_write_rolls_back_the_ack(self) -> None:
        def raise_oserror(state):
            raise OSError("simulated disk failure")

        self.state_store.save = raise_oserror
        with self.assertRaises(PersistenceUnavailable):
            self.service.ack_message_verified(self.sid, self._payload())
        self.assertEqual(self.service.store._delivery, {})

    def test_restart_recovers_the_ack(self) -> None:
        self.service.ack_message_verified(self.sid, self._payload())
        path = os.path.join(self.directory, "state.json")
        restored = DeviceService()
        attach_persistence(restored, path)
        state = restored.store._delivery[(self.sid, "m1")]
        self.assertTrue(state.acked)
        self.assertEqual(state.ack_sequence, 1)
        # A replay against the restored store is the idempotent 200.
        _, status = restored.ack_message_verified(self.sid, self._payload())
        self.assertEqual(status, 200)


class AckVerifiedHTTPTest(AckVerifiedMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()
        self.server, self.service = create_server(
            "127.0.0.1", 0, self.service)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _request(self, method: str, path: str, body=None, raw=None):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        if raw is not None:
            conn.request(method, path, raw,
                         {"Content-Type": "application/json"})
        else:
            data = json.dumps(body).encode("utf-8") if body is not None \
                else b""
            conn.request(method, path, data,
                         {"Content-Type": "application/json"})
        response = conn.getresponse()
        return response.status, json.loads(response.read().decode("utf-8"))

    def test_first_ack_201_then_replay_200(self) -> None:
        path = f"/v1/messages/{self.sid}/acks-verified"
        status, body = self._request("POST", path, self._payload())
        self.assertEqual(status, 201)
        self.assertEqual(list(body),
                         ["session_id", "message_id", "status", "attempts",
                          "sequence"])
        self.assertEqual(body["status"], "acked")
        status, body = self._request("POST", path, self._payload())
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "acked")

    def test_error_statuses_and_fields(self) -> None:
        path = f"/v1/messages/{self.sid}/acks-verified"
        status, body = self._request("POST", path, raw="{bad")
        self.assertEqual((status, body["field"]), (400, "request_body"))
        status, body = self._request("POST", path, {})
        self.assertEqual((status, body["field"]), (400, "device_id"))
        status, body = self._request(
            "POST", path, {"device_id": "bob", "message_id": "m1",
                           "sequence": 1, "expected_version": 1})
        self.assertEqual((status, body["field"]), (400, "signature"))
        status, body = self._request(
            "POST", path, self._payload(expected_version=7))
        self.assertEqual((status, body["field"]), (409, "expected_version"))
        status, body = self._request(
            "POST", path, self._payload(sequence=9))
        self.assertEqual((status, body["field"]), (409, "sequence"))
        status, body = self._request(
            "POST", f"/v1/messages/ghost/acks-verified",
            self._payload(session_id="ghost"))
        self.assertEqual((status, body["field"]), (404, "session_id"))

    def test_bad_path_is_404(self) -> None:
        status, body = self._request(
            "POST", f"/v1/messages/{self.sid}/extra/acks-verified",
            self._payload())
        self.assertEqual((status, body["field"]), (404, "session_id"))

    def test_plain_ack_entry_is_unaffected(self) -> None:
        # The unsigned entry keeps working with no authorization fields.
        status, body = self._request(
            "POST", f"/v1/messages/{self.sid}/acks",
            {"device_id": "bob", "message_id": "m1", "sequence": 1})
        self.assertEqual(status, 201)
        self.assertEqual(body["status"], "acked")


class AckVerifiedHTTPPersistenceTest(AckVerifiedMixin, unittest.TestCase):
    """A durable-write failure answers 503/data_file and rolls back."""

    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self._build()
        self.state_store = attach_persistence(
            self.service, os.path.join(self.directory, "state.json"))
        self.server, self.service = create_server(
            "127.0.0.1", 0, self.service)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_failed_write_is_503_and_rolls_back(self) -> None:
        def raise_oserror(state):
            raise OSError("simulated disk failure")

        self.state_store.save = raise_oserror
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("POST", f"/v1/messages/{self.sid}/acks-verified",
                     json.dumps(self._payload()).encode(),
                     {"Content-Type": "application/json"})
        response = conn.getresponse()
        body = json.loads(response.read().decode("utf-8"))
        self.assertEqual((response.status, body["field"]),
                         (503, "data_file"))
        self.assertEqual(self.service.store._delivery, {})


class AckVerifiedCLITest(unittest.TestCase):
    """The ack-message-verified CLI prints single-line JSON contract."""

    def setUp(self) -> None:
        self.server, _ = create_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        self.bob_private, self.bob_identity = _new_identity()
        self._run("register", "--user-id", "u-a", "--device-id", "alice",
                  "--identity-key", _x25519_b64())
        self._run("register", "--user-id", "u", "--device-id", "bob",
                  "--identity-key", self.bob_identity,
                  "--prekey", f"pk:{_x25519_b64()}")
        created = self._run(
            "create-session", "--initiator-device-id", "alice",
            "--recipient-device-id", "bob", "--prekey-id", "pk",
            "--ephemeral-key", _x25519_b64())
        self.sid = json.loads(created.stdout.strip())["session_id"]
        self._run("send-message", "--session-id", self.sid,
                  "--sender-device-id", "alice", "--message-id", "m1",
                  "--sequence", "1", "--nonce", "n1", "--ciphertext", "ct")

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _run(self, *arguments: str) -> subprocess.CompletedProcess:
        result = subprocess.run(
            [sys.executable, "-m", "e2ee_backend", "--base-url",
             self.base_url, *arguments],
            capture_output=True, text=True, timeout=15)
        return result

    def _signature(self, sequence=1, expected_version=1) -> str:
        return _authorization(self.bob_private, "u", "bob", self.sid, "m1",
                              sequence, expected_version)

    def _ack(self, signature=None, sequence="1", expected_version="1"):
        if signature is None:
            signature = self._signature()
        return self._run(
            "ack-message-verified", self.sid, "--device-id", "bob",
            "--message-id", "m1", "--sequence", sequence,
            "--expected-version", expected_version,
            "--signature", signature)

    def test_success_prints_single_line_json_and_replay_succeeds(self) -> None:
        result = self._ack()
        self.assertEqual(result.returncode, 0, result.stderr)
        line = result.stdout.strip()
        self.assertEqual(line.count("\n"), 0)
        body = json.loads(line)
        self.assertEqual(body["status"], "acked")
        self.assertEqual(body["session_id"], self.sid)
        again = self._ack()
        self.assertEqual(again.returncode, 0, again.stderr)
        self.assertEqual(json.loads(again.stdout.strip())["status"], "acked")

    def test_failure_prints_stderr_json_and_exits_nonzero(self) -> None:
        result = self._ack(signature=self._signature(expected_version=2),
                           expected_version="2")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout.strip(), "")
        line = result.stderr.strip()
        self.assertEqual(line.count("\n"), 0)
        self.assertEqual(json.loads(line)["field"], "expected_version")
        result = self._ack(signature=self._signature(sequence=9),
                           sequence="9")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(json.loads(result.stderr.strip())["field"],
                         "sequence")


if __name__ == "__main__":
    unittest.main()
