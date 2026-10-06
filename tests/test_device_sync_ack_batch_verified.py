"""Tests for the signature-authorized device 1:1 sync-ack batch endpoint.

POST /v1/devices/{device_id}/sync/ack-batch-verified commits the same
delivery acknowledgements and unified sync-cursor advances as the unsigned
``/sync/ack-batch`` entry, but only after a 64-byte Ed25519 signature
(canonical standard base64) verifies over the domain-separated
``E2EE-SYNC-ACK-V1`` message for the device's registered ``user_id`` and
the request's path-decoded ``device_id`` / ``expected_version`` /
``items``, against the device's *current* identity key at its current
``identity_key_version``.
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
from cryptography.hazmat.primitives.asymmetric import ed25519

from e2ee_backend.crypto import sync_ack_proof_message
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


def _sign(private, payload) -> bytes:
    items = [(item["session_id"], item["cursor"]) for item in payload["items"]]
    message = sync_ack_proof_message(
        payload.get("user_id", "u"), payload["device_id"],
        payload["expected_version"], items)
    return private.sign(message)


def _authorization(private, payload) -> str:
    return base64.b64encode(_sign(private, payload)).decode("ascii")


class VerifiedAckBatchMixin:
    def _build(self) -> None:
        self.service = DeviceService()
        # alice keeps a non-Ed25519 legacy-style identity ("ik"); bob gets a
        # real Ed25519 identity whose private key signs the authorizations.
        self.alice_private, alice_identity = _new_identity()
        self.bob_private, self.bob_identity = _new_identity()
        self.service.store.add_device(Device("u", "alice", alice_identity))
        self.service.store.add_device(Device(
            "u", "bob", self.bob_identity,
            prekeys=[SignedPreKey("pk1", "pubk1"),
                     SignedPreKey("pk2", "pubk2")]))
        self.service.store.add_device(Device("u", "carol", "ik"))
        self.sid1 = self.service.store.create_session(
            "alice", "bob", "pk1", "ek1").session_id
        self.sid2 = self.service.store.create_session(
            "alice", "bob", "pk2", "ek2").session_id
        for sequence in range(1, 4):
            self.service.post_message({
                "session_id": self.sid1, "sender_device_id": "alice",
                "message_id": f"a{sequence}", "sequence": sequence,
                "nonce": f"na{sequence}", "ciphertext": "ct"})
        for sequence in range(1, 3):
            self.service.post_message({
                "session_id": self.sid2, "sender_device_id": "alice",
                "message_id": f"b{sequence}", "sequence": sequence,
                "nonce": f"nb{sequence}", "ciphertext": "ct"})

    def _payload(self, items=None, expected_version=1, *, device_id="bob",
                 user_id="u"):
        return {
            "user_id": user_id,
            "device_id": device_id,
            "expected_version": expected_version,
            "items": items if items is not None else [
                {"session_id": self.sid1, "cursor": 2},
                {"session_id": self.sid2, "cursor": 1}],
        }

    def _authorized(self, payload=None, private=None, *, signature_raw=None):
        payload = payload or self._payload()
        private = private or self.bob_private
        body = dict(payload)
        body.pop("user_id", None)
        body.pop("device_id", None)
        if signature_raw is None:
            signature_raw = _sign(private, payload)
        body["signature"] = base64.b64encode(signature_raw).decode("ascii")
        return body

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


class VerifiedAckBatchServiceTest(VerifiedAckBatchMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def _error(self, callable_):
        with self.assertRaises(ServiceError) as caught:
            callable_()
        return caught.exception

    # -- success -----------------------------------------------------------

    def test_forward_batch_commits_like_unsigned_entry(self) -> None:
        body, status = self.service.sync_device_ack_batch_verified(
            "bob", self._authorized())
        self.assertEqual(status, 201)
        self.assertEqual(list(body), ["device_id", "results"])
        self.assertEqual(body["device_id"], "bob")
        results = body["results"]
        self.assertEqual([r["session_id"] for r in results],
                         [self.sid1, self.sid2])
        self.assertEqual([r["cursor"] for r in results], [2, 1])
        self.assertEqual(results[0]["updated_at"], results[1]["updated_at"])
        self.assertIn("+00:00", results[0]["updated_at"])
        delivery = self.service.store._delivery
        self.assertTrue(delivery[(self.sid1, "a1")].acked)
        self.assertTrue(delivery[(self.sid1, "a2")].acked)
        self.assertNotIn((self.sid1, "a3"), delivery)
        cursors = self.service.store._message_sync_cursors
        self.assertEqual(cursors[(self.sid1, "bob")].cursor, 2)
        self.assertEqual(cursors[(self.sid2, "bob")].cursor, 1)

    def test_all_equal_still_checks_authorization_and_writes_nothing(self) -> None:
        # A bad signature is refused even though no cursor would move.
        payload = self._payload(items=[{"session_id": self.sid1, "cursor": 0}])
        body = dict(payload)
        body.pop("user_id", None)
        body.pop("device_id", None)
        body["signature"] = base64.b64encode(b"x" * 64).decode("ascii")
        error = self._error(lambda: self.service.
                            sync_device_ack_batch_verified("bob", body))
        self.assertEqual((error.status_code, error.field),
                         (400, "signature"))
        self.assertEqual(self.service.store._message_sync_cursors, {})
        # A valid authorization for an all-equal batch is 200 and writes
        # nothing (no cursor record created).
        body, status = self.service.sync_device_ack_batch_verified(
            "bob", self._authorized(payload))
        self.assertEqual(status, 200)
        self.assertEqual(body["results"][0]["cursor"], 0)
        self.assertNotIn((self.sid1, "bob"),
                         self.service.store._message_sync_cursors)

    def test_extra_fields_are_ignored(self) -> None:
        payload = self._payload(items=[{"session_id": self.sid1, "cursor": 1}])
        body = self._authorized(payload)
        body["unexpected"] = {"nested": [1, 2]}
        body["items"][0]["extra"] = "kept-as-in-the-unsigned-entry"
        _, status = self.service.sync_device_ack_batch_verified("bob", body)
        self.assertEqual(status, 201)

    def test_replay_after_advance_is_200(self) -> None:
        self.service.sync_device_ack_batch_verified(
            "bob", self._authorized(self._payload(
                items=[{"session_id": self.sid1, "cursor": 2}])))
        _, status = self.service.sync_device_ack_batch_verified(
            "bob", self._authorized(self._payload(
                items=[{"session_id": self.sid1, "cursor": 2}])))
        self.assertEqual(status, 200)

    # -- canonical message -------------------------------------------------

    def test_canonical_authorization_message(self) -> None:
        items = [("sém1", 7), ("sid2", 0)]
        message = sync_ack_proof_message("üser", "böb", 3, items)
        expected = (
            'E2EE-SYNC-ACK-V1\n'
            '{"device_id":"böb","expected_version":3,'
            '"items":[{"cursor":7,"session_id":"sém1"},'
            '{"cursor":0,"session_id":"sid2"}],"user_id":"üser"}'
        ).encode("utf-8")
        self.assertEqual(message, expected)

    def test_signature_binds_items_version_device_and_user(self) -> None:
        # Sign the original, then change a cursor after signing.
        body = self._authorized()
        body["items"][1]["cursor"] = 2
        error = self._error(lambda: self.service.
                            sync_device_ack_batch_verified("bob", body))
        self.assertEqual((error.status_code, error.field),
                         (400, "signature"))
        # Sign the original order, then reorder items after signing.
        original = self._payload()
        body = self._authorized(original)
        body["items"] = [body["items"][1], body["items"][0]]
        error = self._error(lambda: self.service.
                            sync_device_ack_batch_verified("bob", body))
        self.assertEqual((error.status_code, error.field),
                         (400, "signature"))
        # Sign with the wrong registered user_id.
        wrong_user = self._payload(user_id="someone-else")
        error = self._error(lambda: self.service.
                            sync_device_ack_batch_verified(
                                "bob", self._authorized(wrong_user)))
        self.assertEqual((error.status_code, error.field),
                         (400, "signature"))
        # Signed by another device's key.
        error = self._error(lambda: self.service.
                            sync_device_ack_batch_verified(
                                "bob", self._authorized(
                                    private=self.alice_private)))
        self.assertEqual((error.status_code, error.field),
                         (400, "signature"))

    def test_path_decoded_device_id_is_signed(self) -> None:
        # A device whose id needs percent-encoding on the wire.
        weird_id = "bob/日本"
        self.service.store.add_device(
            Device("u", weird_id, self.bob_identity,
                   prekeys=[SignedPreKey("pk-w", "pubkw")]))
        sid = self.service.store.create_session(
            "alice", weird_id, "pk-w", "ekw").session_id
        self.service.post_message({
            "session_id": sid, "sender_device_id": "alice",
            "message_id": "w1", "sequence": 1, "nonce": "nw1",
            "ciphertext": "ct"})
        payload = self._payload(
            items=[{"session_id": sid, "cursor": 1}], device_id=weird_id)
        body = self._authorized(payload)
        result, status = self.service.sync_device_ack_batch_verified(
            weird_id, body)
        self.assertEqual(status, 201)
        self.assertEqual(result["device_id"], weird_id)

    # -- request validation -------------------------------------------------

    def test_body_and_items_validation_matches_unsigned_entry(self) -> None:
        valid = {"session_id": self.sid1, "cursor": 1}
        cases = [
            (None, "request_body"),
            ([], "request_body"),
            ("x", "request_body"),
            ({}, "items"),
            ({"items": []}, "items"),
            ({"items": "x"}, "items"),
            ({"items": [5]}, "items[0]"),
            ({"items": [{"cursor": 1}]}, "items[0].session_id"),
            ({"items": [{"session_id": "", "cursor": 1}]},
             "items[0].session_id"),
            ({"items": [{"session_id": 5, "cursor": 1}]},
             "items[0].session_id"),
            ({"items": [{"session_id": valid["session_id"]}]},
             "items[0].cursor"),
            ({"items": [{"session_id": self.sid1, "cursor": "1"}]},
             "items[0].cursor"),
            ({"items": [{"session_id": self.sid1, "cursor": True}]},
             "items[0].cursor"),
            ({"items": [{"session_id": self.sid1, "cursor": -1}]},
             "items[0].cursor"),
        ]
        for raw_payload, field in cases:
            error = self._error(lambda: self.service.
                                sync_device_ack_batch_verified(
                                    "bob", raw_payload))
            self.assertEqual((error.status_code, error.field), (400, field),
                             raw_payload)

    def test_duplicate_session_is_400_on_the_item(self) -> None:
        body = self._authorized(self._payload(items=[
            {"session_id": self.sid1, "cursor": 1},
            {"session_id": self.sid1, "cursor": 2}]))
        error = self._error(lambda: self.service.
                            sync_device_ack_batch_verified("bob", body))
        self.assertEqual((error.status_code, error.field),
                         (400, "items[1]"))
        self.assertEqual(self.service.store._delivery, {})

    def test_expected_version_validation(self) -> None:
        for value in (None, 0, -1, 1.0, True, False, "1", [1]):
            body = self._authorized()
            body["expected_version"] = value
            error = self._error(lambda: self.service.
                                sync_device_ack_batch_verified("bob", body))
            self.assertEqual((error.status_code, error.field),
                             (400, "expected_version"), value)
        body = self._authorized()
        del body["expected_version"]
        error = self._error(lambda: self.service.
                            sync_device_ack_batch_verified("bob", body))
        self.assertEqual((error.status_code, error.field),
                         (400, "expected_version"))

    def test_signature_validation(self) -> None:
        raw64 = base64.b64encode(b"s" * 64).decode("ascii")
        short63 = base64.b64encode(b"s" * 63).decode("ascii")
        for value in (None, "", 123, True, b"raw", raw64 + " ", short63,
                      "not-base64!!", raw64.rstrip("=")):
            body = self._authorized()
            body["signature"] = value
            error = self._error(lambda: self.service.
                                sync_device_ack_batch_verified("bob", body))
            self.assertEqual((error.status_code, error.field),
                             (400, "signature"), value)
        body = self._authorized()
        del body["signature"]
        error = self._error(lambda: self.service.
                            sync_device_ack_batch_verified("bob", body))
        self.assertEqual((error.status_code, error.field),
                         (400, "signature"))

    # -- atomic authorization checks ---------------------------------------

    def test_unknown_device_is_404(self) -> None:
        error = self._error(lambda: self.service.
                            sync_device_ack_batch_verified(
                                "ghost", self._authorized(
                                    self._payload(device_id="ghost"))))
        self.assertEqual((error.status_code, error.field),
                         (404, "device_id"))

    def test_revoked_device_is_409_and_authorization_still_checked(self) -> None:
        self.service.store.revoke_device("bob")
        # Even a valid signature is refused on a revoked device.
        error = self._error(lambda: self.service.
                            sync_device_ack_batch_verified(
                                "bob", self._authorized()))
        self.assertEqual((error.status_code, error.field),
                         (409, "device_id"))
        # The revocation check precedes the signature check: a revoked
        # device with a bad signature still reports the revocation.
        body = self._authorized()
        body["signature"] = base64.b64encode(b"x" * 64).decode("ascii")
        error = self._error(lambda: self.service.
                            sync_device_ack_batch_verified("bob", body))
        self.assertEqual((error.status_code, error.field),
                         (409, "device_id"))

    def test_non_ed25519_identity_is_400(self) -> None:
        # carol's "ik" identity is not an Ed25519 public key.
        payload = {
            "user_id": "u", "device_id": "carol", "expected_version": 1,
            "items": [{"session_id": self.sid1, "cursor": 1}]}
        body = self._authorized(payload)
        error = self._error(lambda: self.service.
                            sync_device_ack_batch_verified("carol", body))
        self.assertEqual((error.status_code, error.field),
                         (400, "identity_key"))

    def test_version_mismatch_is_409(self) -> None:
        body = self._authorized(self._payload(expected_version=2))
        error = self._error(lambda: self.service.
                            sync_device_ack_batch_verified("bob", body))
        self.assertEqual((error.status_code, error.field),
                         (409, "expected_version"))

    def test_check_order_unknown_revoked_ed25519_version_signature(self) -> None:
        # A wrong version on a non-Ed25519 device reports identity_key.
        payload = {
            "user_id": "u", "device_id": "carol", "expected_version": 9,
            "items": [{"session_id": self.sid1, "cursor": 1}]}
        error = self._error(lambda: self.service.
                            sync_device_ack_batch_verified(
                                "carol", self._authorized(payload)))
        self.assertEqual((error.status_code, error.field),
                         (400, "identity_key"))
        # A bad signature with a correct version reports signature, not the
        # later item checks.
        body = self._authorized()
        body["signature"] = base64.b64encode(b"z" * 64).decode("ascii")
        error = self._error(lambda: self.service.
                            sync_device_ack_batch_verified("bob", body))
        self.assertEqual((error.status_code, error.field),
                         (400, "signature"))

    # -- post-authorization item rules --------------------------------------

    def test_item_checks_run_only_after_authorization(self) -> None:
        # A bad signature masks a cursor/session problem.
        body = self._authorized()
        body["signature"] = base64.b64encode(b"z" * 64).decode("ascii")
        body["items"][0]["cursor"] = 99
        error = self._error(lambda: self.service.
                            sync_device_ack_batch_verified("bob", body))
        self.assertEqual((error.status_code, error.field),
                         (400, "signature"))

    def test_unknown_session_is_404_on_item(self) -> None:
        body = self._authorized(self._payload(items=[
            {"session_id": self.sid1, "cursor": 1},
            {"session_id": "missing", "cursor": 0}]))
        error = self._error(lambda: self.service.
                            sync_device_ack_batch_verified("bob", body))
        self.assertEqual((error.status_code, error.field),
                         (404, "items[1].session_id"))
        self.assertEqual(self.service.store._delivery, {})

    def test_group_session_is_409_on_item(self) -> None:
        sid = self._group_session()
        body = self._authorized(self._payload(
            items=[{"session_id": sid, "cursor": 1}]))
        error = self._error(lambda: self.service.
                            sync_device_ack_batch_verified("bob", body))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[0].session_id"))

    def test_non_recipient_is_409_on_item(self) -> None:
        payload = self._payload(device_id="alice")
        body = self._authorized(payload, private=self.alice_private)
        error = self._error(lambda: self.service.
                            sync_device_ack_batch_verified("alice", body))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[0].session_id"))

    def test_cursor_range_conflict_is_409_on_item(self) -> None:
        self.service.sync_device_ack_batch_verified(
            "bob", self._authorized(self._payload(
                items=[{"session_id": self.sid1, "cursor": 2}])))
        body = self._authorized(self._payload(
            items=[{"session_id": self.sid1, "cursor": 1}]))
        error = self._error(lambda: self.service.
                            sync_device_ack_batch_verified("bob", body))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[0].cursor"))
        body = self._authorized(self._payload(
            items=[{"session_id": self.sid2, "cursor": 9}]))
        error = self._error(lambda: self.service.
                            sync_device_ack_batch_verified("bob", body))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[0].cursor"))

    def test_first_item_error_writes_nothing(self) -> None:
        body = self._authorized(self._payload(items=[
            {"session_id": self.sid1, "cursor": 2},
            {"session_id": self.sid2, "cursor": 9}]))
        error = self._error(lambda: self.service.
                            sync_device_ack_batch_verified("bob", body))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[1].cursor"))
        self.assertEqual(self.service.store._delivery, {})
        self.assertEqual(self.service.store._message_sync_cursors, {})

    # -- linearization against revocation / rotation ------------------------

    def test_identity_rotation_changes_version_and_key(self) -> None:
        new_private, new_identity = _new_identity()
        view = self.service.rotate_identity_key("bob", {
            "identity_key": new_identity})
        self.assertEqual(view["identity_key"], new_identity)
        # The old authorization (version 1, old key) fails the version
        # check.
        error = self._error(lambda: self.service.
                            sync_device_ack_batch_verified(
                                "bob", self._authorized()))
        self.assertEqual((error.status_code, error.field),
                         (409, "expected_version"))
        # The new authorization (version 2, new key) succeeds.
        body = self._authorized(self._payload(
            items=[{"session_id": self.sid1, "cursor": 1}],
            expected_version=2), private=new_private)
        _, status = self.service.sync_device_ack_batch_verified("bob", body)
        self.assertEqual(status, 201)

    def test_concurrent_revocation_races_the_ack_transaction(self) -> None:
        # Hold the store lock from a revocation mid-flight: the ack blocks
        # until the revocation commits, then fails 409 with nothing written.
        started = threading.Event()

        def revoke():
            with self.service.store._lock:
                started.set()
                self.service.store.revoke_device("bob")

        thread = threading.Thread(target=revoke)
        thread.start()
        started.wait(2)
        body = self._authorized()
        error = self._error(lambda: self.service.
                            sync_device_ack_batch_verified("bob", body))
        thread.join(2)
        self.assertEqual((error.status_code, error.field),
                         (409, "device_id"))
        self.assertEqual(self.service.store._delivery, {})

    def test_ack_then_revoke_keeps_the_ack(self) -> None:
        _, status = self.service.sync_device_ack_batch_verified(
            "bob", self._authorized(self._payload(
                items=[{"session_id": self.sid1, "cursor": 2}])))
        self.assertEqual(status, 201)
        self.service.store.revoke_device("bob")
        self.assertTrue(self.service.store._delivery[
            (self.sid1, "a1")].acked)
        self.assertEqual(self.service.store._message_sync_cursors[
            (self.sid1, "bob")].cursor, 2)


class VerifiedAckBatchPersistenceTest(VerifiedAckBatchMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self._build()
        self.state_store = attach_persistence(
            self.service, os.path.join(self.directory, "state.json"))

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_forward_batch_persists_with_one_generation(self) -> None:
        generation = self.state_store.commit_seq
        self.service.sync_device_ack_batch_verified(
            "bob", self._authorized())
        self.assertEqual(self.state_store.commit_seq, generation + 1)

    def test_all_equal_consumes_no_generation(self) -> None:
        generation = self.state_store.commit_seq
        _, status = self.service.sync_device_ack_batch_verified(
            "bob", self._authorized(self._payload(items=[
                {"session_id": self.sid1, "cursor": 0},
                {"session_id": self.sid2, "cursor": 0}])))
        self.assertEqual(status, 200)
        self.assertEqual(self.state_store.commit_seq, generation)

    def test_failed_write_rolls_back_the_whole_batch(self) -> None:
        def raise_oserror(state):
            raise OSError("simulated disk failure")

        self.state_store.save = raise_oserror
        with self.assertRaises(PersistenceUnavailable):
            self.service.sync_device_ack_batch_verified(
                "bob", self._authorized())
        self.assertEqual(self.service.store._delivery, {})
        self.assertEqual(self.service.store._message_sync_cursors, {})

    def test_restart_recovers_acks_and_cursors(self) -> None:
        self.service.sync_device_ack_batch_verified(
            "bob", self._authorized(self._payload(
                items=[{"session_id": self.sid1, "cursor": 2}])))
        restarted = DeviceService()
        attach_persistence(restarted,
                           os.path.join(self.directory, "state.json"))
        self.assertEqual(restarted.store._message_sync_cursors[
            (self.sid1, "bob")].cursor, 2)
        self.assertTrue(restarted.store._delivery[
            (self.sid1, "a1")].acked)


class VerifiedAckBatchHTTPTest(VerifiedAckBatchMixin, unittest.TestCase):
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

    PATH = "/v1/devices/bob/sync/ack-batch-verified"

    def test_forward_and_equal(self) -> None:
        status, body = self._request("POST", self.PATH, self._authorized())
        self.assertEqual(status, 201)
        self.assertEqual(list(body), ["device_id", "results"])
        self.assertEqual([r["session_id"] for r in body["results"]],
                         [self.sid1, self.sid2])
        status, _ = self._request("POST", self.PATH, self._authorized())
        self.assertEqual(status, 200)

    def test_error_statuses_and_fields(self) -> None:
        status, body = self._request("POST", self.PATH, raw="{bad")
        self.assertEqual((status, body["field"]), (400, "request_body"))
        status, body = self._request("POST", self.PATH, {"items": []})
        self.assertEqual((status, body["field"]), (400, "items"))
        # expected_version missing.
        malformed = self._authorized()
        del malformed["expected_version"]
        status, body = self._request("POST", self.PATH, malformed)
        self.assertEqual((status, body["field"]), (400, "expected_version"))
        # bad signature encoding.
        malformed = self._authorized()
        malformed["signature"] = "not base64!"
        status, body = self._request("POST", self.PATH, malformed)
        self.assertEqual((status, body["field"]), (400, "signature"))
        # unknown device.
        status, body = self._request(
            "POST", "/v1/devices/ghost/sync/ack-batch-verified",
            self._authorized(self._payload(device_id="ghost")))
        self.assertEqual((status, body["field"]), (404, "device_id"))
        # version mismatch.
        malformed = self._authorized(self._payload(expected_version=5))
        status, body = self._request("POST", self.PATH, malformed)
        self.assertEqual((status, body["field"]), (409, "expected_version"))
        # bad signature value.
        malformed = self._authorized()
        malformed["signature"] = base64.b64encode(b"q" * 64).decode("ascii")
        status, body = self._request("POST", self.PATH, malformed)
        self.assertEqual((status, body["field"]), (400, "signature"))
        # item-level 404/409 after a valid authorization.
        malformed = self._authorized(self._payload(items=[
            {"session_id": "missing", "cursor": 0}]))
        status, body = self._request("POST", self.PATH, malformed)
        self.assertEqual((status, body["field"]),
                         (404, "items[0].session_id"))
        malformed = self._authorized(self._payload(items=[
            {"session_id": self.sid1, "cursor": 9}]))
        status, body = self._request("POST", self.PATH, malformed)
        self.assertEqual((status, body["field"]),
                         (409, "items[0].cursor"))

    def test_unsigned_entry_still_available(self) -> None:
        status, body = self._request(
            "POST", "/v1/devices/bob/sync/ack-batch",
            {"items": [{"session_id": self.sid1, "cursor": 1}]})
        self.assertEqual(status, 201)
        self.assertEqual(body["results"][0]["cursor"], 1)

    def test_persist_failure_is_503_data_file(self) -> None:
        def raise_oserror(state):
            raise OSError("simulated disk failure")

        self.state_store.save = raise_oserror
        status, body = self._request("POST", self.PATH, self._authorized())
        self.assertEqual((status, body["field"]), (503, "data_file"))

    def test_unicode_device_id_is_path_decoded(self) -> None:
        weird_id = "böb"
        self.service.store.add_device(
            Device("u", weird_id, self.bob_identity,
                   prekeys=[SignedPreKey("pk-u", "pubku")]))
        sid = self.service.store.create_session(
            "alice", weird_id, "pk-u", "eku").session_id
        self.service.post_message({
            "session_id": sid, "sender_device_id": "alice",
            "message_id": "u1", "sequence": 1, "nonce": "nu1",
            "ciphertext": "ct"})
        payload = self._payload(
            items=[{"session_id": sid, "cursor": 1}], device_id=weird_id)
        body = self._authorized(payload)
        encoded = "/v1/devices/" + \
            "b%C3%B6b/sync/ack-batch-verified"
        status, response = self._request("POST", encoded, body)
        self.assertEqual(status, 201)
        self.assertEqual(response["device_id"], weird_id)


if __name__ == "__main__":
    unittest.main()
