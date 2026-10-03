"""Tests for batch identity-authorized pre-key replenishment.

Covers the service, HTTP (real loopback socket), CLI (real subprocess),
key-audit and persistence layers for:

* POST /v1/devices/{device_id}/prekeys/verified-batch

The request object's non-empty ``signed_prekeys`` array carries the same
three fields/encodings as the single-item entry; every proof is verified
under the E2EE-SIGNED-PREKEY-V1 protocol against the device's *current*
Ed25519 identity key. The batch is all-or-nothing: new items append in
order (one ``prekey_added`` event and one frozen proof each, a single
durable generation), identical non-revoked items are idempotent (200), and
any failure writes nothing.
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

from e2ee_backend.crypto import signed_prekey_proof_message
from e2ee_backend.http_app import create_server
from e2ee_backend.persistence import (
    PersistenceUnavailable, attach_persistence)
from e2ee_backend.service import DeviceService, ServiceError


def _raw_b64(key) -> str:
    raw = key.public_bytes(serialization.Encoding.Raw,
                           serialization.PublicFormat.Raw)
    return base64.b64encode(raw).decode()


def _new_identity():
    private = ed25519.Ed25519PrivateKey.generate()
    return private, _raw_b64(private.public_key())


def _new_prekey() -> str:
    return _raw_b64(x25519.X25519PrivateKey.generate().public_key())


def _proof(private, user_id, device_id, key_id, public_key) -> str:
    message = signed_prekey_proof_message(
        user_id, device_id, key_id, public_key)
    return base64.b64encode(private.sign(message)).decode()


class AddPrekeysVerifiedBatchServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = DeviceService()
        self.private, self.identity = _new_identity()
        self.prekey_k1 = _new_prekey()
        self.service.register_verified({
            "user_id": "u1", "device_id": "d1",
            "identity_key": self.identity,
            "signed_prekeys": [{
                "key_id": "k1", "public_key": self.prekey_k1,
                "signature": _proof(self.private, "u1", "d1",
                                    "k1", self.prekey_k1)}],
        })

    def _entry(self, key_id, public_key=None, private=None):
        if public_key is None:
            public_key = _new_prekey()
        signer = private or self.private
        return {"key_id": key_id, "public_key": public_key,
                "signature": _proof(signer, "u1", "d1", key_id, public_key)}

    def _batch(self, *entries):
        return {"signed_prekeys": list(entries)}

    def test_valid_batch_appends_in_order_201(self) -> None:
        k2, k3 = _new_prekey(), _new_prekey()
        body, status = self.service.add_prekeys_verified_batch(
            "d1", self._batch(self._entry("k2", k2),
                              self._entry("k3", k3)))
        self.assertEqual(status, 201)
        self.assertEqual(set(body), {"device_id", "signed_prekeys"})
        self.assertEqual(body["device_id"], "d1")
        self.assertEqual(body["signed_prekeys"], [
            {"key_id": "k2", "public_key": k2},
            {"key_id": "k3", "public_key": k3}])
        self.assertEqual(self.service.get_device("d1")["prekey_ids"],
                         ["k1", "k2", "k3"])

    def test_response_preserves_request_order_not_storage_order(self) -> None:
        k2, k3 = _new_prekey(), _new_prekey()
        self.service.add_prekeys_verified_batch(
            "d1", self._batch(self._entry("k2", k2),
                              self._entry("k3", k3)))
        # Replay reversed: the response follows the request order even
        # though storage order is k2,k3.
        body, status = self.service.add_prekeys_verified_batch(
            "d1", self._batch(self._entry("k3", k3),
                              self._entry("k2", k2)))
        self.assertEqual(status, 200)
        self.assertEqual([v["key_id"] for v in body["signed_prekeys"]],
                         ["k3", "k2"])

    def test_pure_idempotent_replay_200(self) -> None:
        k2 = _new_prekey()
        payload = self._batch(self._entry("k2", k2))
        _, first = self.service.add_prekeys_verified_batch("d1", payload)
        self.assertEqual(first, 201)
        body, second = self.service.add_prekeys_verified_batch(
            "d1", self._batch(self._entry("k2", k2)))
        self.assertEqual(second, 200)
        self.assertEqual(body["signed_prekeys"],
                         [{"key_id": "k2", "public_key": k2}])
        self.assertEqual(self.service.get_device("d1")["prekey_ids"],
                         ["k1", "k2"])

    def test_mixed_replay_and_new_201(self) -> None:
        k2, k3 = _new_prekey(), _new_prekey()
        self.service.add_prekeys_verified_batch(
            "d1", self._batch(self._entry("k2", k2)))
        body, status = self.service.add_prekeys_verified_batch(
            "d1", self._batch(self._entry("k2", k2),
                              self._entry("k3", k3)))
        self.assertEqual(status, 201)
        self.assertEqual([v["key_id"] for v in body["signed_prekeys"]],
                         ["k2", "k3"])
        self.assertEqual(self.service.get_device("d1")["prekey_ids"],
                         ["k1", "k2", "k3"])

    def test_body_not_object_400_request_body(self) -> None:
        for payload in (None, [], "x", 42):
            with self.subTest(payload=payload):
                with self.assertRaises(ServiceError) as ctx:
                    self.service.add_prekeys_verified_batch("d1", payload)
                self.assertEqual(ctx.exception.status_code, 400)
                self.assertEqual(ctx.exception.field, "request_body")

    def test_array_missing_non_array_or_empty_400(self) -> None:
        for payload in ({}, {"signed_prekeys": None},
                        {"signed_prekeys": "x"}, {"signed_prekeys": []},
                        {"signed_prekeys": {}}):
            with self.subTest(payload=payload):
                with self.assertRaises(ServiceError) as ctx:
                    self.service.add_prekeys_verified_batch("d1", payload)
                self.assertEqual(ctx.exception.status_code, 400)
                self.assertEqual(ctx.exception.field, "signed_prekeys")

    def test_non_object_element_400(self) -> None:
        for bad, payload, index in (
                (None, self._batch(None), 0),
                (42, self._batch(42), 0),
                ("x", self._batch("x"), 0),
                ([], self._batch(self._entry("k2"), []), 1)):
            with self.subTest(bad=bad):
                with self.assertRaises(ServiceError) as ctx:
                    self.service.add_prekeys_verified_batch("d1", payload)
                self.assertEqual(ctx.exception.status_code, 400)
                self.assertEqual(ctx.exception.field,
                                 f"signed_prekeys[{index}]")

    def test_element_field_errors_400(self) -> None:
        cases = [
            ({}, "signed_prekeys[0].key_id"),
            ({"key_id": "k2"}, "signed_prekeys[0].public_key"),
            ({"key_id": "k2", "public_key": _new_prekey()},
             "signed_prekeys[0].signature"),
            ({"key_id": "", "public_key": "p", "signature": "s"},
             "signed_prekeys[0].key_id"),
            ({"key_id": 9, "public_key": "p", "signature": "s"},
             "signed_prekeys[0].key_id"),
            ({"key_id": "k2", "public_key": 7, "signature": "s"},
             "signed_prekeys[0].public_key"),
            ({"key_id": "k2", "public_key": "p", "signature": 8},
             "signed_prekeys[0].signature"),
        ]
        for element, field in cases:
            with self.subTest(field=field):
                with self.assertRaises(ServiceError) as ctx:
                    self.service.add_prekeys_verified_batch(
                        "d1", {"signed_prekeys": [element]})
                self.assertEqual(ctx.exception.status_code, 400)
                self.assertEqual(ctx.exception.field, field)

    def test_bad_public_key_400_at_element_path(self) -> None:
        element = {"key_id": "k2", "public_key": "not-a-key",
                   "signature": "s"}
        with self.assertRaises(ServiceError) as ctx:
            self.service.add_prekeys_verified_batch(
                "d1", {"signed_prekeys": [element]})
        self.assertEqual(ctx.exception.field,
                         "signed_prekeys[0].public_key")

    def test_bad_signature_encoding_400_at_element_path(self) -> None:
        for bad in ("@@@@", "abc",
                    base64.b64encode(b"\x00" * 63).decode(),
                    base64.b64encode(b"\x00" * 65).decode()):
            element = {"key_id": "k2", "public_key": _new_prekey(),
                       "signature": bad}
            with self.subTest(bad=bad):
                with self.assertRaises(ServiceError) as ctx:
                    self.service.add_prekeys_verified_batch(
                        "d1", {"signed_prekeys": [element]})
                self.assertEqual(ctx.exception.field,
                                 "signed_prekeys[0].signature")

    def test_error_locates_first_checked_index(self) -> None:
        # A valid element 0 followed by a typed-wrong element 1 names [1].
        payload = self._batch(self._entry("k2"),
                              {"key_id": "k3", "public_key": _new_prekey()})
        with self.assertRaises(ServiceError) as ctx:
            self.service.add_prekeys_verified_batch("d1", payload)
        self.assertEqual(ctx.exception.field,
                         "signed_prekeys[1].signature")

    def test_duplicate_key_id_in_request_400_at_second_occurrence(self) -> None:
        k2, k3 = _new_prekey(), _new_prekey()
        payload = self._batch(self._entry("k2", k2),
                              self._entry("k2", k3))
        with self.assertRaises(ServiceError) as ctx:
            self.service.add_prekeys_verified_batch("d1", payload)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.field, "signed_prekeys[1].key_id")
        # Nothing was appended.
        self.assertEqual(self.service.get_device("d1")["prekey_ids"],
                         ["k1"])

    def test_wrong_signer_400_at_item_signature(self) -> None:
        other, _ = _new_identity()
        payload = self._batch(self._entry("k2"),
                              self._entry("k9", private=other))
        with self.assertRaises(ServiceError) as ctx:
            self.service.add_prekeys_verified_batch("d1", payload)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.field, "signed_prekeys[1].signature")
        self.assertEqual(self.service.get_device("d1")["prekey_ids"],
                         ["k1"])

    def test_tampered_fields_400_at_item_signature(self) -> None:
        public_key = _new_prekey()
        good = self._entry("k2", public_key)
        tampered = dict(good, key_id="k3")
        with self.assertRaises(ServiceError) as ctx:
            self.service.add_prekeys_verified_batch(
                "d1", self._batch(tampered))
        self.assertEqual(ctx.exception.field, "signed_prekeys[0].signature")

    def test_invalid_signature_precedes_key_id_conflict(self) -> None:
        # Index 0 replays fine; index 1 is an existing id (k1) with a
        # changed key AND a bad proof: signature wins, located at [1].
        other, _ = _new_identity()
        changed = _new_prekey()
        payload = self._batch(
            self._entry("k2", _new_prekey()),
            {"key_id": "k1", "public_key": changed,
             "signature": _proof(other, "u1", "d1", "k1", changed)})
        with self.assertRaises(ServiceError) as ctx:
            self.service.add_prekeys_verified_batch("d1", payload)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.field, "signed_prekeys[1].signature")

    def test_same_id_changed_key_409_at_item_key_id(self) -> None:
        public_key = _new_prekey()
        self.service.add_prekeys_verified_batch(
            "d1", self._batch(self._entry("k2", public_key)))
        other = _new_prekey()
        with self.assertRaises(ServiceError) as ctx:
            self.service.add_prekeys_verified_batch(
                "d1", self._batch(self._entry("k2", other)))
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "signed_prekeys[0].key_id")

    def test_revoked_key_id_409_at_item_key_id(self) -> None:
        public_key = _new_prekey()
        self.service.add_prekeys_verified_batch(
            "d1", self._batch(self._entry("k2", public_key)))
        self.service.revoke_prekey("d1", "k2")
        with self.assertRaises(ServiceError) as ctx:
            self.service.add_prekeys_verified_batch(
                "d1", self._batch(self._entry("k2", public_key)))
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "signed_prekeys[0].key_id")

    def test_conflict_on_later_item_rolls_back_entire_batch(self) -> None:
        # k2 would be new; k1 exists with a different public_key, so the
        # batch fails and k2 must not be visible afterwards.
        changed = _new_prekey()
        with self.assertRaises(ServiceError) as ctx:
            self.service.add_prekeys_verified_batch(
                "d1", self._batch(self._entry("k2"),
                                  self._entry("k1", changed)))
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "signed_prekeys[1].key_id")
        self.assertEqual(self.service.get_device("d1")["prekey_ids"],
                         ["k1"])
        self.assertEqual(
            [e["type"] for e in
             self.service.list_key_events("d1", 0, 100)["events"]],
            ["registered"])

    def test_unknown_device_404(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.add_prekeys_verified_batch(
                "ghost", self._batch(self._entry("k2")))
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "device_id")

    def test_revoked_device_409(self) -> None:
        self.service.revoke_device("d1")
        with self.assertRaises(ServiceError) as ctx:
            self.service.add_prekeys_verified_batch(
                "d1", self._batch(self._entry("k2")))
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "device_id")

    def test_non_ed25519_identity_400_identity_key(self) -> None:
        x25519_der = base64.b64encode(
            x25519.X25519PrivateKey.generate().public_key().public_bytes(
                serialization.Encoding.DER,
                serialization.PublicFormat.SubjectPublicKeyInfo)).decode()
        self.service.register({
            "user_id": "u2", "device_id": "d2",
            "identity_key": x25519_der,
            "signed_prekeys": [{"key_id": "k1",
                                 "public_key": _new_prekey()}],
        })
        with self.assertRaises(ServiceError) as ctx:
            self.service.add_prekeys_verified_batch(
                "d2", self._batch(self._entry("k2")))
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.field, "identity_key")
        self.assertEqual(self.service.get_device("d2")["prekey_ids"],
                         ["k1"])

    def test_verifies_against_current_identity_after_rotation(self) -> None:
        new_private, new_identity = _new_identity()
        self.service.rotate_identity_key(
            "d1", {"identity_key": new_identity})
        with self.assertRaises(ServiceError) as ctx:
            self.service.add_prekeys_verified_batch(
                "d1", self._batch(self._entry("k2")))
        self.assertEqual(ctx.exception.field, "signed_prekeys[0].signature")
        body, status = self.service.add_prekeys_verified_batch(
            "d1", self._batch(self._entry("k2", private=new_private)))
        self.assertEqual(status, 201)
        self.assertEqual(body["signed_prekeys"][0]["key_id"], "k2")

    def test_audit_events_one_per_new_key_in_order(self) -> None:
        k2, k3 = _new_prekey(), _new_prekey()
        self.service.add_prekeys_verified_batch(
            "d1", self._batch(self._entry("k2", k2),
                              self._entry("k3", k3)))
        # A mixed replay adds no event for the replayed item.
        self.service.add_prekeys_verified_batch(
            "d1", self._batch(self._entry("k2", k2)))
        events = self.service.list_key_events("d1", 0, 100)["events"]
        self.assertEqual([e["type"] for e in events],
                         ["registered", "prekey_added", "prekey_added"])
        self.assertEqual([e["payload"]["key_id"] for e in events[1:]],
                         ["k2", "k3"])
        self.assertTrue(events[1]["hash"])
        self.assertEqual(events[2]["prev_hash"], events[1]["hash"])

    def test_frozen_proof_saved_and_replay_does_not_replace_it(self) -> None:
        k2 = _new_prekey()
        self.service.add_prekeys_verified_batch(
            "d1", self._batch(self._entry("k2", k2)))
        proof = self.service.get_prekey_proof("d1", "k2")
        self.assertEqual(proof, {
            "user_id": "u1", "device_id": "d1", "key_id": "k2",
            "public_key": k2, "identity_key": self.identity,
            "signature": _proof(self.private, "u1", "d1", "k2", k2)})
        # Pure replay answers 200 and leaves the frozen proof untouched.
        _, status = self.service.add_prekeys_verified_batch(
            "d1", self._batch(self._entry("k2", k2)))
        self.assertEqual(status, 200)
        self.assertEqual(self.service.get_prekey_proof("d1", "k2"), proof)

    def test_idempotent_replay_does_not_restore_consumed_state(self) -> None:
        k2 = _new_prekey()
        self.service.add_prekeys_verified_batch(
            "d1", self._batch(self._entry("k2", k2)))
        # Consume k1 then k2 via two claims.
        first, s1 = self.service.claim_prekey(
            {"recipient_device_id": "d1", "claim_id": "c1"})
        second, s2 = self.service.claim_prekey(
            {"recipient_device_id": "d1", "claim_id": "c2"})
        self.assertEqual((s1, first["key_id"]), (201, "k1"))
        self.assertEqual((s2, second["key_id"]), (201, "k2"))
        # The idempotent replay must not un-consume k2.
        _, status = self.service.add_prekeys_verified_batch(
            "d1", self._batch(self._entry("k2", k2)))
        self.assertEqual(status, 200)
        self.assertEqual(self.service.get_device("d1")["prekey_ids"], [])
        with self.assertRaises(ServiceError) as ctx:
            self.service.claim_prekey(
                {"recipient_device_id": "d1", "claim_id": "c3"})
        self.assertEqual(ctx.exception.field, "prekey_id")

    def test_added_batch_prekeys_are_claimable_in_order(self) -> None:
        self.service.add_prekeys_verified_batch(
            "d1", self._batch(self._entry("k2"), self._entry("k3")))
        claimed = []
        for claim_id in ("c1", "c2", "c3"):
            view, status = self.service.claim_prekey(
                {"recipient_device_id": "d1", "claim_id": claim_id})
            self.assertEqual(status, 201)
            claimed.append(view["key_id"])
        self.assertEqual(claimed, ["k1", "k2", "k3"])


class AddPrekeysVerifiedBatchPersistenceTest(unittest.TestCase):
    def _service_with_k1(self, path):
        private, identity = _new_identity()
        service = DeviceService()
        store = attach_persistence(service, path)
        k1 = _new_prekey()
        service.register_verified({
            "user_id": "u1", "device_id": "d1",
            "identity_key": identity,
            "signed_prekeys": [{
                "key_id": "k1", "public_key": k1,
                "signature": _proof(private, "u1", "d1", "k1", k1)}],
        })
        return service, store, private, identity, k1

    def test_batch_commits_one_generation(self) -> None:
        path = tempfile.mktemp(suffix=".json")
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        service, store, private, _, _ = self._service_with_k1(path)
        k2, k3 = _new_prekey(), _new_prekey()
        generation_before = store.commit_seq
        service.add_prekeys_verified_batch(
            "d1", {"signed_prekeys": [
                {"key_id": "k2", "public_key": k2,
                 "signature": _proof(private, "u1", "d1", "k2", k2)},
                {"key_id": "k3", "public_key": k3,
                 "signature": _proof(private, "u1", "d1", "k3", k3)}]})
        self.assertEqual(store.commit_seq, generation_before + 1)

    def test_pure_replay_consumes_no_generation(self) -> None:
        path = tempfile.mktemp(suffix=".json")
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        service, store, private, _, _ = self._service_with_k1(path)
        k2 = _new_prekey()
        entry = {"key_id": "k2", "public_key": k2,
                 "signature": _proof(private, "u1", "d1", "k2", k2)}
        service.add_prekeys_verified_batch(
            "d1", {"signed_prekeys": [dict(entry)]})
        generation_after_add = store.commit_seq
        events_after_add = len(
            service.list_key_events("d1", 0, 100)["events"])
        _, status = service.add_prekeys_verified_batch(
            "d1", {"signed_prekeys": [dict(entry)]})
        self.assertEqual(status, 200)
        self.assertEqual(store.commit_seq, generation_after_add)
        self.assertEqual(
            len(service.list_key_events("d1", 0, 100)["events"]),
            events_after_add)

    def test_persist_failure_rolls_back_whole_batch(self) -> None:
        path = tempfile.mktemp(suffix=".json")
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        service, state_store, private, _, _ = self._service_with_k1(path)

        def fail_save(_pending) -> None:
            raise OSError("simulated disk failure")

        state_store.save = fail_save  # type: ignore[assignment]
        k2, k3 = _new_prekey(), _new_prekey()
        with self.assertRaises(PersistenceUnavailable):
            service.add_prekeys_verified_batch(
                "d1", {"signed_prekeys": [
                    {"key_id": "k2", "public_key": k2,
                     "signature": _proof(private, "u1", "d1", "k2", k2)},
                    {"key_id": "k3", "public_key": k3,
                     "signature": _proof(private, "u1", "d1", "k3", k3)}]})
        self.assertEqual(service.get_device("d1")["prekey_ids"], ["k1"])
        self.assertEqual(
            [e["type"] for e in
             service.list_key_events("d1", 0, 100)["events"]],
            ["registered"])

    def test_order_proofs_and_consumed_marks_survive_restart(self) -> None:
        path = tempfile.mktemp(suffix=".json")
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        service, _, private, identity, _ = self._service_with_k1(path)
        k2, k3 = _new_prekey(), _new_prekey()
        service.add_prekeys_verified_batch(
            "d1", {"signed_prekeys": [
                {"key_id": "k2", "public_key": k2,
                 "signature": _proof(private, "u1", "d1", "k2", k2)},
                {"key_id": "k3", "public_key": k3,
                 "signature": _proof(private, "u1", "d1", "k3", k3)}]})
        # Consume k1 so its consumed flag survives the restart too.
        service.claim_prekey({"recipient_device_id": "d1", "claim_id": "c1"})

        second = DeviceService()
        attach_persistence(second, path)
        self.assertEqual(second.get_device("d1")["prekey_ids"],
                         ["k2", "k3"])
        events = second.list_key_events("d1", 0, 100)["events"]
        added = [e for e in events if e["type"] == "prekey_added"]
        self.assertEqual([e["payload"]["key_id"] for e in added],
                         ["k2", "k3"])
        for key_id, public_key in (("k2", k2), ("k3", k3)):
            self.assertEqual(second.get_prekey_proof("d1", key_id), {
                "user_id": "u1", "device_id": "d1", "key_id": key_id,
                "public_key": public_key, "identity_key": identity,
                "signature": _proof(private, "u1", "d1", key_id,
                                    public_key)})
        # k1 stays consumed after restart: the next claim serves k2.
        view, status = second.claim_prekey(
            {"recipient_device_id": "d1", "claim_id": "c2"})
        self.assertEqual(status, 201)
        self.assertEqual(view["key_id"], "k2")


class AddPrekeysVerifiedBatchHTTPTest(unittest.TestCase):
    def setUp(self) -> None:
        self.server, self.service = create_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        self.private, self.identity = _new_identity()
        public_key = _new_prekey()
        self.service.register_verified({
            "user_id": "u1", "device_id": "d1",
            "identity_key": self.identity,
            "signed_prekeys": [{
                "key_id": "k1", "public_key": public_key,
                "signature": _proof(self.private, "u1", "d1",
                                    "k1", public_key)}],
        })

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _request(self, method, path, body=None, raw=False):
        connection = HTTPConnection("127.0.0.1", self.port, timeout=5)
        if raw:
            payload = body
        else:
            payload = json.dumps(body) if body is not None else None
        headers = {"Content-Type": "application/json"} if payload else {}
        connection.request(method, path, body=payload, headers=headers)
        response = connection.getresponse()
        data = response.read().decode("utf-8")
        connection.close()
        return response.status, json.loads(data)

    def _path(self) -> str:
        return "/v1/devices/d1/prekeys/verified-batch"

    def _entry(self, key_id, public_key=None):
        if public_key is None:
            public_key = _new_prekey()
        return {"key_id": key_id, "public_key": public_key,
                "signature": _proof(self.private, "u1", "d1",
                                    key_id, public_key)}

    def test_201_then_200(self) -> None:
        k2 = _new_prekey()
        status, body = self._request(
            "POST", self._path(),
            {"signed_prekeys": [self._entry("k9", k2)]})
        self.assertEqual(status, 201)
        self.assertEqual(set(body), {"device_id", "signed_prekeys"})
        self.assertEqual(body["signed_prekeys"],
                         [{"key_id": "k9", "public_key": k2}])
        status, body = self._request(
            "POST", self._path(),
            {"signed_prekeys": [self._entry("k9", k2)]})
        self.assertEqual(status, 200)

    def test_invalid_json_400_request_body(self) -> None:
        status, body = self._request(
            "POST", self._path(), body="{not json", raw=True)
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "request_body")

    def test_non_object_body_400_request_body(self) -> None:
        for raw in ("[]", "null", '"x"', "42"):
            status, body = self._request(
                "POST", self._path(), body=raw, raw=True)
            self.assertEqual(status, 400, raw)
            self.assertEqual(body["field"], "request_body")

    def test_empty_array_400_signed_prekeys(self) -> None:
        status, body = self._request(
            "POST", self._path(), {"signed_prekeys": []})
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "signed_prekeys")

    def test_item_field_error_path(self) -> None:
        status, body = self._request(
            "POST", self._path(),
            {"signed_prekeys": [
                self._entry("k2"),
                {"key_id": "k3", "public_key": _new_prekey(),
                 "signature": "not-base64"}]})
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "signed_prekeys[1].signature")

    def test_duplicate_key_id_second_occurrence(self) -> None:
        k2, k3 = _new_prekey(), _new_prekey()
        status, body = self._request(
            "POST", self._path(), {"signed_prekeys": [
                self._entry("k2", k2), self._entry("k2", k3)]})
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "signed_prekeys[1].key_id")

    def test_unknown_device_404(self) -> None:
        status, body = self._request(
            "POST", "/v1/devices/ghost/prekeys/verified-batch",
            {"signed_prekeys": [self._entry("k2")]})
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "device_id")

    def test_revoked_device_409(self) -> None:
        self.service.revoke_device("d1")
        status, body = self._request(
            "POST", self._path(),
            {"signed_prekeys": [self._entry("k2")]})
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "device_id")

    def test_single_item_verified_route_still_works(self) -> None:
        status, body = self._request(
            "POST", "/v1/devices/d1/prekeys/verified",
            {"key_id": "plain", "public_key": _new_prekey(),
             "signature": "not-base64"})
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "signature")
        k2 = _new_prekey()
        status, body = self._request(
            "POST", "/v1/devices/d1/prekeys/verified",
            self._entry("single", k2))
        self.assertEqual(status, 201)
        self.assertEqual(set(body), {"device_id", "key_id", "public_key"})


class AddPrekeysVerifiedBatchCLITest(unittest.TestCase):
    def setUp(self) -> None:
        self.server, service = create_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        self.private, self.identity = _new_identity()
        k1 = _new_prekey()
        service.register_verified({
            "user_id": "u1", "device_id": "d1",
            "identity_key": self.identity,
            "signed_prekeys": [{
                "key_id": "k1", "public_key": k1,
                "signature": _proof(self.private, "u1", "d1", "k1", k1)}],
        })

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _run(self, *arguments) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "e2ee_backend", "--base-url",
             self.base_url, *arguments],
            capture_output=True, text=True, timeout=15)

    def _spec(self, key_id, public_key=None) -> str:
        if public_key is None:
            public_key = _new_prekey()
        signature = _proof(self.private, "u1", "d1", key_id, public_key)
        return f"{key_id}:{public_key}:{signature}", public_key

    def test_success_stdout_zero_and_idempotent(self) -> None:
        spec_a, k_a = self._spec("ka")
        spec_b, k_b = self._spec("kb")
        first = self._run("add-prekeys-verified", "--device-id", "d1",
                          "--prekey", spec_a, "--prekey", spec_b)
        self.assertEqual(first.returncode, 0, first.stderr)
        body = json.loads(first.stdout.strip())
        self.assertEqual(set(body), {"device_id", "signed_prekeys"})
        self.assertEqual(body["signed_prekeys"], [
            {"key_id": "ka", "public_key": k_a},
            {"key_id": "kb", "public_key": k_b}])
        self.assertFalse(first.stderr.strip())

        second = self._run("add-prekeys-verified", "--device-id", "d1",
                           "--prekey", spec_a, "--prekey", spec_b)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(json.loads(second.stdout.strip())["signed_prekeys"],
                         [{"key_id": "ka", "public_key": k_a},
                          {"key_id": "kb", "public_key": k_b}])

    def test_at_file_input(self) -> None:
        key_id, public_key = "kc", _new_prekey()
        obj = {"key_id": key_id, "public_key": public_key,
               "signature": _proof(self.private, "u1", "d1",
                                   key_id, public_key)}
        path = tempfile.mktemp(suffix=".json")
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(obj, handle)
        result = self._run("add-prekeys-verified", "--device-id", "d1",
                           "--prekey", f"@{path}")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            json.loads(result.stdout.strip())["signed_prekeys"],
            [{"key_id": "kc", "public_key": public_key}])

    def test_failure_stderr_field_nonzero(self) -> None:
        spec, _ = self._spec("kd")
        result = self._run("add-prekeys-verified", "--device-id", "ghost",
                           "--prekey", spec)
        self.assertEqual(result.returncode, 1)
        self.assertFalse(result.stdout.strip())
        self.assertEqual(json.loads(result.stderr.strip())["field"],
                         "device_id")

    def test_bad_proof_stderr_item_signature(self) -> None:
        result = self._run(
            "add-prekeys-verified", "--device-id", "d1",
            "--prekey", f"ke:{_new_prekey()}:not-base64")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(json.loads(result.stderr.strip())["field"],
                         "signed_prekeys[0].signature")

    def test_local_parse_error_exit_2_signed_prekeys(self) -> None:
        result = self._run("add-prekeys-verified", "--device-id", "d1",
                           "--prekey", "no-colons-here")
        self.assertEqual(result.returncode, 2)
        self.assertFalse(result.stdout.strip())
        self.assertEqual(json.loads(result.stderr.strip())["field"],
                         "signed_prekeys")

    def test_at_least_one_prekey_required(self) -> None:
        result = self._run("add-prekeys-verified", "--device-id", "d1")
        self.assertNotEqual(result.returncode, 0)


class AddPrekeysVerifiedBatchConcurrencyTest(unittest.TestCase):
    def _register(self, service):
        private, identity = _new_identity()
        k1 = _new_prekey()
        service.register_verified({
            "user_id": "u1", "device_id": "d1",
            "identity_key": identity,
            "signed_prekeys": [{
                "key_id": "k1", "public_key": k1,
                "signature": _proof(private, "u1", "d1", "k1", k1)}],
        })
        return private, k1

    def test_batch_linearizes_against_device_revoke(self) -> None:
        service = DeviceService()
        private, _ = self._register(service)
        k2, k3 = _new_prekey(), _new_prekey()
        outcomes = []

        def batch() -> None:
            try:
                _, status = service.add_prekeys_verified_batch(
                    "d1", {"signed_prekeys": [
                        {"key_id": "k2", "public_key": k2,
                         "signature": _proof(private, "u1", "d1",
                                             "k2", k2)},
                        {"key_id": "k3", "public_key": k3,
                         "signature": _proof(private, "u1", "d1",
                                             "k3", k3)}]})
                outcomes.append(("batch", status))
            except ServiceError as error:
                outcomes.append(("batch", error.status_code))

        def revoke() -> None:
            service.revoke_device("d1")
            outcomes.append(("revoke", 200))

        threads = [threading.Thread(target=batch),
                   threading.Thread(target=revoke)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)

        batch_result = next(code for kind, code in outcomes
                            if kind == "batch")
        self.assertIn(batch_result, (201, 409))
        view = service.get_device("d1")
        if batch_result == 201:
            # The batch committed first; revoke afterwards hides every key,
            # but both batch keys landed (never a partial k2-only append).
            self.assertEqual(view["prekey_ids"], [])
            events = service.list_key_events("d1", 0, 100)["events"]
            self.assertEqual(
                [e["payload"].get("key_id")
                 for e in events if e["type"] == "prekey_added"],
                ["k2", "k3"])
        else:
            # Revoke landed first: the batch failed as a whole (the
            # device_revoked event exists but no prekey_added ones).
            self.assertEqual(view["prekey_ids"], [])
            events = service.list_key_events("d1", 0, 100)["events"]
            self.assertNotIn("prekey_added", [e["type"] for e in events])

    def test_batch_linearizes_against_identity_rotation(self) -> None:
        service = DeviceService()
        private, _ = self._register(service)
        k2 = _new_prekey()
        new_private, new_identity = _new_identity()
        outcomes = []

        def batch() -> None:
            try:
                _, status = service.add_prekeys_verified_batch(
                    "d1", {"signed_prekeys": [{
                        "key_id": "k2", "public_key": k2,
                        "signature": _proof(private, "u1", "d1",
                                            "k2", k2)}]})
                outcomes.append(("batch", status))
            except ServiceError as error:
                outcomes.append(("batch", error.status_code))

        def rotate() -> None:
            service.rotate_identity_key(
                "d1", {"identity_key": new_identity})
            outcomes.append(("rotate", 200))

        threads = [threading.Thread(target=batch),
                   threading.Thread(target=rotate)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)

        batch_result = next(code for kind, code in outcomes
                            if kind == "batch")
        view = service.get_device("d1")
        if batch_result == 201:
            self.assertIn("k2", view["prekey_ids"])
        else:
            # The rotation committed first: the old-identity proof fails
            # and the batch wrote nothing. A new-identity proof then lands.
            self.assertEqual(batch_result, 400)
            self.assertNotIn("k2", view["prekey_ids"])
            _, status = service.add_prekeys_verified_batch(
                "d1", {"signed_prekeys": [{
                    "key_id": "k2", "public_key": k2,
                    "signature": _proof(new_private, "u1", "d1",
                                        "k2", k2)}]})
            self.assertEqual(status, 201)

    def test_batch_linearizes_against_claim(self) -> None:
        service = DeviceService()
        private, _ = self._register(service)
        k2 = _new_prekey()
        barrier = threading.Barrier(2)

        def batch() -> None:
            barrier.wait()
            service.add_prekeys_verified_batch(
                "d1", {"signed_prekeys": [{
                    "key_id": "k2", "public_key": k2,
                    "signature": _proof(private, "u1", "d1", "k2", k2)}]})

        def claim() -> None:
            barrier.wait()
            service.claim_prekey(
                {"recipient_device_id": "d1", "claim_id": "c1"})

        threads = [threading.Thread(target=batch),
                   threading.Thread(target=claim)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)

        # Exactly one claim exists; whether it consumed k1 or (never k2
        # partially) the final claim serves a consistent next key.
        view, status = service.claim_prekey(
            {"recipient_device_id": "d1", "claim_id": "c2"})
        self.assertEqual(status, 201)
        self.assertIn(view["key_id"], ("k1", "k2"))
        self.assertEqual(service.get_device("d1")["prekey_ids"], [])


if __name__ == "__main__":
    unittest.main()
