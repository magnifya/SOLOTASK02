"""Tests for the signature-authorized group sync-ack batch endpoint.

POST /v1/devices/{device_id}/group-sync/ack-batch-verified behaves like
the plain multi-group-session sync-ack batch, but the batch is authorized
by an Ed25519 signature from the device's *current* identity key over the
canonical ``E2EE-GROUP-SYNC-ACK-V1`` message (device_id /
expected_version / items / user_id), with ``expected_version`` pinned to
the current ``identity_key_version``. The authorization and the whole
batch commit atomically with revocation and identity rotation; a durable
write failure rolls everything back.
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

from e2ee_backend.crypto import group_sync_ack_proof_message
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


def _authorization(private, user_id, device_id, expected_version, items):
    message = group_sync_ack_proof_message(
        user_id, device_id, expected_version, items)
    return base64.b64encode(private.sign(message)).decode()


class GroupAckBatchVerifiedMixin:
    """d1/d2/d3/d4 registered; two frozen group sessions.

    d2 and d4 hold Ed25519 identities (d4 is in neither group); d1 and d3
    keep the non-Ed25519 placeholder ``ik``. s1 (group g1, members
    d1/d2/d3) holds m1@1..m3@3 from d1; s2 (group g2, members d1/d2)
    holds n1@1..n2@2 from d1.
    """

    def _build(self) -> None:
        self.service = DeviceService()
        self.d2_private, d2_identity = _new_identity()
        self.d4_private, d4_identity = _new_identity()
        store = self.service.store
        store.add_device(Device("u", "d1", "ik"))
        store.add_device(Device(
            "u", "d2", d2_identity,
            prekeys=[SignedPreKey("pk1", "pubk1")]))
        store.add_device(Device("u", "d3", "ik"))
        store.add_device(Device("u", "d4", d4_identity))
        self.service.create_group({
            "group_id": "g1", "creator_device_id": "d1",
            "member_device_ids": ["d1", "d2", "d3"]})
        self.service.create_group({
            "group_id": "g2", "creator_device_id": "d1",
            "member_device_ids": ["d1", "d2"]})
        self.s1 = self.service.create_group_session({
            "group_id": "g1", "initiator_device_id": "d1",
            "ephemeral_key": "epk1"})["session_id"]
        s2_view = self.service.create_group_session({
            "group_id": "g2", "initiator_device_id": "d1",
            "ephemeral_key": "epk2"})
        self.s2 = s2_view["session_id"]
        self.s2_created_at = s2_view["created_at"]
        for sequence in range(1, 4):
            self.service.post_message({
                "session_id": self.s1, "sender_device_id": "d1",
                "message_id": f"m{sequence}", "sequence": sequence,
                "nonce": f"nm{sequence}", "ciphertext": "ct"})
        for sequence in range(1, 3):
            self.service.post_message({
                "session_id": self.s2, "sender_device_id": "d1",
                "message_id": f"n{sequence}", "sequence": sequence,
                "nonce": f"nn{sequence}", "ciphertext": "ct"})

    def _payload(self, items, expected_version=1, private=None,
                 device_id="d2", user_id="u", signature=None):
        pairs = [(item["session_id"], item["cursor"]) for item in items]
        if signature is None:
            signer = private if private is not None else self.d2_private
            signature = _authorization(
                signer, user_id, device_id, expected_version, pairs)
        return {"items": items,
                "expected_version": expected_version,
                "signature": signature}

    def _one_to_one_session(self) -> str:
        sid = self.service.store.create_session(
            "d1", "d2", "pk1", "ek1").session_id
        self.service.post_message({
            "session_id": sid, "sender_device_id": "d1",
            "message_id": "p1", "sequence": 1, "nonce": "np1",
            "ciphertext": "ct"})
        return sid


class GroupAckBatchVerifiedServiceTest(GroupAckBatchVerifiedMixin,
                                       unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def _error(self, callable_):
        with self.assertRaises(ServiceError) as caught:
            callable_()
        return caught.exception

    def _call(self, device_id, payload):
        return self.service.sync_group_ack_batch_verified(
            device_id, payload)

    def test_forward_batch_acks_ranges_and_shares_timestamp(self) -> None:
        body, status = self._call("d2", self._payload(
            [{"session_id": self.s1, "cursor": 2},
             {"session_id": self.s2, "cursor": 1}]))
        self.assertEqual(status, 201)
        self.assertEqual(list(body), ["device_id", "results"])
        self.assertEqual(body["device_id"], "d2")
        results = body["results"]
        self.assertEqual([r["session_id"] for r in results],
                         [self.s1, self.s2])
        self.assertEqual([r["cursor"] for r in results], [2, 1])
        for result in results:
            self.assertEqual(list(result),
                             ["session_id", "cursor", "updated_at"])
        self.assertEqual(results[0]["updated_at"], results[1]["updated_at"])
        self.assertIn("+00:00", results[0]["updated_at"])
        delivery = self.service.store._group_delivery
        self.assertTrue(delivery[(self.s1, "m1", "d2")].acked)
        self.assertTrue(delivery[(self.s1, "m2", "d2")].acked)
        self.assertNotIn((self.s1, "m3", "d2"), delivery)
        self.assertTrue(delivery[(self.s2, "n1", "d2")].acked)
        self.assertNotIn((self.s2, "n2", "d2"), delivery)
        cursors = self.service.store._group_sync_cursors
        self.assertEqual(cursors[(self.s1, "d2")].cursor, 2)
        self.assertEqual(cursors[(self.s2, "d2")].cursor, 1)

    def test_forward_batch_skips_own_messages_and_keeps_attempts(self) -> None:
        self.service.post_message({
            "session_id": self.s1, "sender_device_id": "d2",
            "message_id": "m4", "sequence": 4, "nonce": "nm4",
            "ciphertext": "ct"})
        view, _ = self.service.retry_message(
            self.s1, "m2", {"device_id": "d2", "attempt_id": "a1"})
        self.assertEqual(view["attempts"], 1)
        body, status = self._call("d2", self._payload(
            [{"session_id": self.s1, "cursor": 4}]))
        self.assertEqual(status, 201)
        self.assertEqual(body["results"][0]["cursor"], 4)
        delivery = self.service.store._group_delivery
        # A device never acknowledges its own outgoing group message.
        self.assertNotIn((self.s1, "m4", "d2"), delivery)
        record = delivery[(self.s1, "m2", "d2")]
        self.assertTrue(record.acked)
        self.assertEqual(record.ack_sequence, 2)
        self.assertEqual(record.attempts, 1)
        self.assertEqual(record.attempt_ids, {"a1"})

    def test_all_equal_is_200_and_writes_nothing(self) -> None:
        self._call("d2", self._payload(
            [{"session_id": self.s1, "cursor": 2}]))
        body, status = self._call("d2", self._payload(
            [{"session_id": self.s1, "cursor": 2},
             {"session_id": self.s2, "cursor": 0}]))
        self.assertEqual(status, 200)
        self.assertEqual(body["results"][0]["cursor"], 2)
        # An equal item whose cursor was never written reports the group
        # session's created_at.
        self.assertEqual(body["results"][1]["updated_at"],
                         self.s2_created_at)
        self.assertNotIn((self.s2, "d2"),
                         self.service.store._group_sync_cursors)

    def test_all_equal_still_requires_valid_authorization(self) -> None:
        self._call("d2", self._payload(
            [{"session_id": self.s1, "cursor": 2}]))
        # Every cursor equals the stored one, but a bad signature is still
        # refused.
        error = self._error(lambda: self._call("d2", self._payload(
            [{"session_id": self.s1, "cursor": 2}],
            signature=_authorization(self.d2_private, "u", "d2", 1,
                                     [(self.s1, 1)]))))
        self.assertEqual((error.status_code, error.field),
                         (400, "signature"))

    def test_extra_fields_are_ignored(self) -> None:
        payload = self._payload([{"session_id": self.s1, "cursor": 1,
                                  "note": "kept out of the proof"}])
        payload["unexpected"] = "ignored"
        body, status = self._call("d2", payload)
        self.assertEqual(status, 201)
        self.assertEqual(body["results"][0]["cursor"], 1)

    def test_unicode_user_and_device_ids_sign_verbatim(self) -> None:
        private, identity = _new_identity()
        self.service.store.add_device(Device("用户", "设备-甲", identity))
        self.service.create_group({
            "group_id": "g3", "creator_device_id": "d1",
            "member_device_ids": ["d1", "设备-甲"]})
        sid = self.service.create_group_session({
            "group_id": "g3", "initiator_device_id": "d1",
            "ephemeral_key": "epk3"})["session_id"]
        self.service.post_message({
            "session_id": sid, "sender_device_id": "d1",
            "message_id": "z1", "sequence": 1, "nonce": "nz1",
            "ciphertext": "ct"})
        body, status = self._call("设备-甲", self._payload(
            [{"session_id": sid, "cursor": 1}], private=private,
            device_id="设备-甲", user_id="用户"))
        self.assertEqual(status, 201)
        self.assertEqual(body["device_id"], "设备-甲")

    def test_item_order_is_part_of_the_proof(self) -> None:
        items = [{"session_id": self.s1, "cursor": 1},
                 {"session_id": self.s2, "cursor": 1}]
        # Signed over the reversed array: verification must fail.
        error = self._error(lambda: self._call("d2", self._payload(
            items, signature=_authorization(
                self.d2_private, "u", "d2", 1,
                [(self.s2, 1), (self.s1, 1)]))))
        self.assertEqual((error.status_code, error.field),
                         (400, "signature"))

    # -- validation --------------------------------------------------------

    def test_body_validation(self) -> None:
        cases = [
            (None, "request_body"),
            ([], "request_body"),
            ("x", "request_body"),
            ({}, "items"),
            ({"items": []}, "items"),
            ({"items": "x"}, "items"),
            ({"items": ["x"]}, "items[0]"),
            ({"items": [{"cursor": 1}]}, "items[0].session_id"),
            ({"items": [{"session_id": "", "cursor": 1}]},
             "items[0].session_id"),
            ({"items": [{"session_id": self.s1}]}, "items[0].cursor"),
            ({"items": [{"session_id": self.s1, "cursor": "1"}]},
             "items[0].cursor"),
            ({"items": [{"session_id": self.s1, "cursor": True}]},
             "items[0].cursor"),
            ({"items": [{"session_id": self.s1, "cursor": 1.5}]},
             "items[0].cursor"),
            ({"items": [{"session_id": self.s1, "cursor": -1}]},
             "items[0].cursor"),
        ]
        for payload, field in cases:
            error = self._error(lambda: self._call("d2", payload))
            self.assertEqual((error.status_code, error.field), (400, field),
                             payload)

    def test_duplicate_session_is_400_on_the_item(self) -> None:
        error = self._error(lambda: self._call("d2", self._payload(
            [{"session_id": self.s1, "cursor": 1},
             {"session_id": self.s1, "cursor": 2}])))
        self.assertEqual((error.status_code, error.field),
                         (400, "items[1]"))
        self.assertEqual(self.service.store._group_delivery, {})
        self.assertEqual(self.service.store._group_sync_cursors, {})

    def test_expected_version_validation(self) -> None:
        items = [{"session_id": self.s1, "cursor": 1}]
        cases = [
            ({"items": items}, "expected_version"),
            ({"items": items, "expected_version": 0}, "expected_version"),
            ({"items": items, "expected_version": -2}, "expected_version"),
            ({"items": items, "expected_version": True}, "expected_version"),
            ({"items": items, "expected_version": "1"}, "expected_version"),
            ({"items": items, "expected_version": 1.5}, "expected_version"),
        ]
        for payload, field in cases:
            error = self._error(lambda: self._call("d2", payload))
            self.assertEqual((error.status_code, error.field), (400, field),
                             payload)

    def test_signature_validation(self) -> None:
        items = [{"session_id": self.s1, "cursor": 1}]
        cases = [
            ({"items": items, "expected_version": 1}, "signature"),
            ({"items": items, "expected_version": 1, "signature": ""},
             "signature"),
            ({"items": items, "expected_version": 1, "signature": 5},
             "signature"),
            ({"items": items, "expected_version": 1, "signature": "!!!"},
             "signature"),
            # Standard base64 of the wrong length (32 bytes, not 64).
            ({"items": items, "expected_version": 1,
              "signature": base64.b64encode(b"\x00" * 32).decode()},
             "signature"),
            # 64 bytes but non-canonical (padded) encoding.
            ({"items": items, "expected_version": 1,
              "signature": base64.b64encode(b"\x00" * 64).decode()[:-1]
              + " "},
             "signature"),
        ]
        for payload, field in cases:
            error = self._error(lambda: self._call("d2", payload))
            self.assertEqual((error.status_code, error.field), (400, field),
                             payload)

    # -- device / identity / authorization checks --------------------------

    def test_device_unknown_is_404(self) -> None:
        error = self._error(lambda: self._call("ghost", self._payload(
            [{"session_id": self.s1, "cursor": 1}], device_id="ghost")))
        self.assertEqual((error.status_code, error.field),
                         (404, "device_id"))

    def test_device_revoked_is_409(self) -> None:
        self.service.store.revoke_device("d2")
        error = self._error(lambda: self._call("d2", self._payload(
            [{"session_id": self.s1, "cursor": 1}])))
        self.assertEqual((error.status_code, error.field),
                         (409, "device_id"))

    def test_non_ed25519_identity_key_is_400(self) -> None:
        # d3's identity key ("ik") is not an Ed25519 public key; the
        # signature encoding is valid but the algorithm check fires first.
        signature = base64.b64encode(b"\x00" * 64).decode()
        error = self._error(lambda: self._call("d3", {
            "items": [{"session_id": self.s1, "cursor": 1}],
            "expected_version": 1, "signature": signature}))
        self.assertEqual((error.status_code, error.field),
                         (400, "identity_key"))

    def test_version_mismatch_is_409(self) -> None:
        error = self._error(lambda: self._call("d2", self._payload(
            [{"session_id": self.s1, "cursor": 1}], expected_version=2)))
        self.assertEqual((error.status_code, error.field),
                         (409, "expected_version"))
        self.assertEqual(self.service.store._group_delivery, {})

    def test_rotation_invalidates_old_authorization(self) -> None:
        # Rotate d2's identity: the version rises and the old key's
        # authorization no longer verifies.
        _, new_identity = _new_identity()
        self.service.store.rotate_identity_key("d2", new_identity)
        error = self._error(lambda: self._call("d2", self._payload(
            [{"session_id": self.s1, "cursor": 1}], expected_version=1)))
        self.assertEqual((error.status_code, error.field),
                         (409, "expected_version"))
        error = self._error(lambda: self._call("d2", self._payload(
            [{"session_id": self.s1, "cursor": 1}], expected_version=2)))
        self.assertEqual((error.status_code, error.field),
                         (400, "signature"))
        self.assertEqual(self.service.store._group_delivery, {})

    def test_wrong_signing_key_is_400(self) -> None:
        other_private, _ = _new_identity()
        error = self._error(lambda: self._call("d2", self._payload(
            [{"session_id": self.s1, "cursor": 1}], private=other_private)))
        self.assertEqual((error.status_code, error.field),
                         (400, "signature"))

    def test_wrong_user_id_in_proof_is_400(self) -> None:
        error = self._error(lambda: self._call("d2", self._payload(
            [{"session_id": self.s1, "cursor": 1}], user_id="other")))
        self.assertEqual((error.status_code, error.field),
                         (400, "signature"))

    # -- item checks after authorization -----------------------------------

    def test_session_unknown_is_404_on_item(self) -> None:
        error = self._error(lambda: self._call("d2", self._payload(
            [{"session_id": self.s1, "cursor": 1},
             {"session_id": "missing", "cursor": 0}])))
        self.assertEqual((error.status_code, error.field),
                         (404, "items[1].session_id"))
        self.assertEqual(self.service.store._group_delivery, {})

    def test_one_to_one_session_is_409_on_item(self) -> None:
        one_to_one = self._one_to_one_session()
        error = self._error(lambda: self._call("d2", self._payload(
            [{"session_id": one_to_one, "cursor": 1}])))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[0].session_id"))

    def test_non_frozen_member_is_409_on_item(self) -> None:
        # d4 holds a valid Ed25519 authorization but was never frozen into
        # either group session.
        error = self._error(lambda: self._call("d4", self._payload(
            [{"session_id": self.s1, "cursor": 1}], private=self.d4_private,
            device_id="d4")))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[0].session_id"))

    def test_backward_and_over_max_are_409_on_cursor(self) -> None:
        self._call("d2", self._payload(
            [{"session_id": self.s1, "cursor": 2}]))
        error = self._error(lambda: self._call("d2", self._payload(
            [{"session_id": self.s1, "cursor": 1}])))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[0].cursor"))
        error = self._error(lambda: self._call("d2", self._payload(
            [{"session_id": self.s2, "cursor": 9}])))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[0].cursor"))

    def test_first_error_in_array_order_writes_nothing(self) -> None:
        error = self._error(lambda: self._call("d2", self._payload(
            [{"session_id": self.s1, "cursor": 2},
             {"session_id": self.s2, "cursor": 9}])))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[1].cursor"))
        self.assertEqual(self.service.store._group_delivery, {})
        self.assertEqual(self.service.store._group_sync_cursors, {})


class GroupAckBatchVerifiedPersistenceTest(GroupAckBatchVerifiedMixin,
                                           unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self._build()
        self.state_store = attach_persistence(
            self.service, os.path.join(self.directory, "state.json"))

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_whole_batch_persists_with_one_generation(self) -> None:
        generation = self.state_store.commit_seq
        self.service.sync_group_ack_batch_verified("d2", self._payload(
            [{"session_id": self.s1, "cursor": 2},
             {"session_id": self.s2, "cursor": 1}]))
        self.assertEqual(self.state_store.commit_seq, generation + 1)

    def test_all_equal_consumes_no_generation(self) -> None:
        generation = self.state_store.commit_seq
        _, status = self.service.sync_group_ack_batch_verified(
            "d2", self._payload([{"session_id": self.s1, "cursor": 0},
                                 {"session_id": self.s2, "cursor": 0}]))
        self.assertEqual(status, 200)
        self.assertEqual(self.state_store.commit_seq, generation)

    def test_failed_write_rolls_back_the_whole_batch(self) -> None:
        def raise_oserror(state):
            raise OSError("simulated disk failure")

        self.state_store.save = raise_oserror
        with self.assertRaises(PersistenceUnavailable):
            self.service.sync_group_ack_batch_verified(
                "d2", self._payload([{"session_id": self.s1, "cursor": 2},
                                     {"session_id": self.s2, "cursor": 1}]))
        self.assertEqual(self.service.store._group_delivery, {})
        self.assertEqual(self.service.store._group_sync_cursors, {})

    def test_restart_recovers_cursors_and_acks(self) -> None:
        self.service.sync_group_ack_batch_verified("d2", self._payload(
            [{"session_id": self.s1, "cursor": 2}]))
        path = os.path.join(self.directory, "state.json")
        restored = DeviceService()
        attach_persistence(restored, path)
        cursors = restored.store._group_sync_cursors
        self.assertEqual(cursors[(self.s1, "d2")].cursor, 2)
        self.assertTrue(restored.store._group_delivery[(self.s1, "m1", "d2")]
                        .acked)
        self.assertTrue(restored.store._group_delivery[(self.s1, "m2", "d2")]
                        .acked)
        self.assertNotIn((self.s1, "m3", "d2"), restored.store._group_delivery)


class GroupAckBatchVerifiedHTTPTest(GroupAckBatchVerifiedMixin,
                                    unittest.TestCase):
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

    def test_forward_and_equal(self) -> None:
        path = "/v1/devices/d2/group-sync/ack-batch-verified"
        status, body = self._request("POST", path, self._payload(
            [{"session_id": self.s1, "cursor": 2},
             {"session_id": self.s2, "cursor": 1}]))
        self.assertEqual(status, 201)
        self.assertEqual(list(body), ["device_id", "results"])
        self.assertEqual([r["session_id"] for r in body["results"]],
                         [self.s1, self.s2])
        status, body = self._request("POST", path,
                                     self._payload(
            [{"session_id": self.s1, "cursor": 2},
             {"session_id": self.s2, "cursor": 1}]))
        self.assertEqual(status, 200)

    def test_error_statuses_and_fields(self) -> None:
        path = "/v1/devices/d2/group-sync/ack-batch-verified"
        status, body = self._request("POST", path, raw="{bad")
        self.assertEqual((status, body["field"]), (400, "request_body"))
        status, body = self._request("POST", path, {"items": []})
        self.assertEqual((status, body["field"]), (400, "items"))
        status, body = self._request(
            "POST", path, {"items": [{"session_id": self.s1, "cursor": 1}]})
        self.assertEqual((status, body["field"]), (400, "expected_version"))
        status, body = self._request(
            "POST", path, {"items": [{"session_id": self.s1, "cursor": 1}],
                           "expected_version": 1})
        self.assertEqual((status, body["field"]), (400, "signature"))
        status, body = self._request(
            "POST", path, self._payload(
                [{"session_id": self.s1, "cursor": 1}],
                expected_version=7))
        self.assertEqual((status, body["field"]), (409, "expected_version"))
        status, body = self._request(
            "POST", path, self._payload(
                [{"session_id": self.s1, "cursor": 9}]))
        self.assertEqual((status, body["field"]), (409, "items[0].cursor"))

    def test_unknown_device_is_404_and_bad_path_is_404(self) -> None:
        status, body = self._request(
            "POST", "/v1/devices/ghost/group-sync/ack-batch-verified",
            self._payload([{"session_id": self.s1, "cursor": 1}],
                          device_id="ghost"))
        self.assertEqual((status, body["field"]), (404, "device_id"))
        status, body = self._request(
            "POST", "/v1/devices/d2/extra/group-sync/ack-batch-verified",
            self._payload([{"session_id": self.s1, "cursor": 1}]))
        self.assertEqual((status, body["field"]), (404, "device_id"))

    def test_percent_decoded_device_id_signs_the_decoded_value(self) -> None:
        private, identity = _new_identity()
        self.service.store.add_device(Device("u", "dev ice", identity))
        self.service.create_group({
            "group_id": "g4", "creator_device_id": "d1",
            "member_device_ids": ["d1", "dev ice"]})
        sid = self.service.create_group_session({
            "group_id": "g4", "initiator_device_id": "d1",
            "ephemeral_key": "epk4"})["session_id"]
        self.service.post_message({
            "session_id": sid, "sender_device_id": "d1",
            "message_id": "w1", "sequence": 1, "nonce": "nw1",
            "ciphertext": "ct"})
        status, body = self._request(
            "POST", "/v1/devices/dev%20ice/group-sync/ack-batch-verified",
            self._payload([{"session_id": sid, "cursor": 1}],
                          private=private, device_id="dev ice"))
        self.assertEqual(status, 201)
        self.assertEqual(body["device_id"], "dev ice")

    def test_plain_batch_entry_is_unaffected(self) -> None:
        # The unsigned group entry keeps working with no authorization
        # fields.
        status, body = self._request(
            "POST", "/v1/devices/d2/group-sync/ack-batch",
            {"items": [{"session_id": self.s1, "cursor": 1}]})
        self.assertEqual(status, 201)
        self.assertEqual(body["device_id"], "d2")


class GroupAckBatchVerifiedHTTPPersistenceTest(GroupAckBatchVerifiedMixin,
                                               unittest.TestCase):
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
        conn.request("POST", "/v1/devices/d2/group-sync/ack-batch-verified",
                     json.dumps(self._payload(
                         [{"session_id": self.s1, "cursor": 2}])).encode(),
                     {"Content-Type": "application/json"})
        response = conn.getresponse()
        body = json.loads(response.read().decode("utf-8"))
        self.assertEqual((response.status, body["field"]),
                         (503, "data_file"))
        self.assertEqual(self.service.store._group_delivery, {})
        self.assertEqual(self.service.store._group_sync_cursors, {})


if __name__ == "__main__":
    unittest.main()
