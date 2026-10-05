"""Tests for the atomic batch signature-verified message submission entry.

Covers ``POST /v1/messages/submit-verified-batch`` and the
``submit-verified-batch`` CLI subcommand: the body is an object with a
non-empty ``items`` array whose elements reuse the single verified entry's
``request_id`` plus the eight ``sign_message`` fields, processed in input
order inside one locked transaction (each item's sequence/nonce checks see
the earlier items of the same batch). Success returns ``items`` in input
order with the single entry's ten fields each (201 when at least one item
committed, 200 when every item was a replay — replays survive revocation
and rotation). The first failure names ``items[i].<field>`` and rolls the
whole batch back: no message, idempotency record, sequence or nonce
survives, a durable-write failure is 503/field=data_file, and a restart
still replays the committed values.
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

from e2ee_backend.crypto import sign_message
from e2ee_backend.http_app import create_server
from e2ee_backend.persistence import PersistenceUnavailable, attach_persistence
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


_ENVELOPE_FIELDS = ("session_id", "sender_device_id", "message_id",
                    "sequence", "nonce", "ciphertext")
_VIEW_FIELDS = {"request_id", *_ENVELOPE_FIELDS, "created_at",
                "identity_key", "signature"}

#: Repository root, prepended onto PYTHONPATH so CLI subprocesses resolve
#: the package regardless of their working directory.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class BatchFixture:
    """A service with two devices and one 1:1 session between them.

    d1's identity is an Ed25519 key whose private seed the fixture keeps, so
    tests can sign envelopes with ``sign_message``.
    """

    def __init__(self, service=None) -> None:
        self.service = service or DeviceService()
        self.private = ed25519.Ed25519PrivateKey.generate()
        self.seed = _seed_b64(self.private)
        self.identity = _raw_b64(self.private.public_key())
        self.service.register(_register_payload("d1",
                                                identity_key=self.identity))
        self.service.register(_register_payload("d2"))
        session = self.service.create_session({
            "initiator_device_id": "d1",
            "recipient_device_id": "d2",
            "prekey_id": "k1",
            "ephemeral_key": _x25519_b64(),
        })
        self.session_id = session["session_id"]

    def item(self, request_id, message_id, sequence, sender="d1",
             session_id=None, nonce=None, ciphertext=None) -> dict:
        # Deterministic per message id so a repeated call produces a
        # byte-identical replay (the nonce must still be exactly 12 bytes and
        # the ciphertext at least 16 under the signed-envelope rules).
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

    def batch(self, *items) -> dict:
        return {"items": list(items)}


class ServiceSubmitVerifiedBatchTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = BatchFixture()
        self.service = self.fixture.service
        self.session_id = self.fixture.session_id

    def _items(self, count=3):
        return [self.fixture.item(f"req-{i}", f"m{i}", i)
                for i in range(1, count + 1)]

    def _messages(self):
        page = self.service.list_messages(self.session_id, "d2", 0, 100)
        return page["messages"]

    # -- success shape -----------------------------------------------------

    def test_batch_commits_in_input_order_with_virtual_sequence(self) -> None:
        body, status = self.service.submit_verified_message_batch(
            self.fixture.batch(*self._items(3)))
        self.assertEqual(status, 201)
        self.assertEqual([item["message_id"] for item in body["items"]],
                         ["m1", "m2", "m3"])
        self.assertEqual([item["sequence"] for item in body["items"]],
                         [1, 2, 3])
        for item in body["items"]:
            self.assertEqual(set(item), _VIEW_FIELDS)
            self.assertTrue(item["created_at"].endswith("+00:00"))
            self.assertEqual(item["identity_key"], self.fixture.identity)
        self.assertEqual([m["message_id"] for m in self._messages()],
                         ["m1", "m2", "m3"])

    def test_single_item_batch_behaves_like_single_entry(self) -> None:
        body, status = self.service.submit_verified_message_batch(
            self.fixture.batch(self.fixture.item("req-1", "m1", 1)))
        self.assertEqual(status, 201)
        self.assertEqual(len(body["items"]), 1)
        # The committed message is pullable in the ordinary stream format.
        pulled = self._messages()[0]
        self.assertNotIn("identity_key", pulled)
        self.assertNotIn("signature", pulled)

    def test_all_replay_is_200_with_frozen_bodies(self) -> None:
        items = self._items(2)
        first, status = self.service.submit_verified_message_batch(
            self.fixture.batch(*items))
        self.assertEqual(status, 201)
        replay, status = self.service.submit_verified_message_batch(
            self.fixture.batch(*self._items(2)))
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)

    def test_replay_survives_revocation_and_rotation(self) -> None:
        items = self._items(2)
        first, _ = self.service.submit_verified_message_batch(
            self.fixture.batch(*items))
        self.service.rotate_identity_key(
            "d1", {"identity_key": _raw_b64(
                ed25519.Ed25519PrivateKey.generate().public_key())})
        self.service.revoke_device("d1")
        replay, status = self.service.submit_verified_message_batch(
            self.fixture.batch(*self._items(2)))
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)

    def test_mixed_replay_and_new_is_201(self) -> None:
        self.service.submit_verified_message_batch(
            self.fixture.batch(self.fixture.item("req-1", "m1", 1)))
        body, status = self.service.submit_verified_message_batch(
            self.fixture.batch(self.fixture.item("req-1", "m1", 1),
                               self.fixture.item("req-2", "m2", 2)))
        self.assertEqual(status, 201)
        self.assertEqual([item["message_id"] for item in body["items"]],
                         ["m1", "m2"])
        self.assertEqual(len(self._messages()), 2)

    # -- shape errors ------------------------------------------------------

    def _shape_error(self, payload):
        with self.assertRaises(ServiceError) as ctx:
            self.service.submit_verified_message_batch(payload)
        return ctx.exception

    def test_non_object_body_is_400_request_body(self) -> None:
        for payload in ([], "x", 5, None):
            error = self._shape_error(payload)
            self.assertEqual(error.status_code, 400)
            self.assertEqual(error.field, "request_body")

    def test_missing_or_empty_items_is_400_items(self) -> None:
        for payload in ({}, {"items": []}, {"items": "x"}, {"items": 5}):
            error = self._shape_error(payload)
            self.assertEqual(error.status_code, 400)
            self.assertEqual(error.field, "items")

    def test_non_object_element_is_400_items_index(self) -> None:
        error = self._shape_error({"items": [self.fixture.item("r", "m1", 1),
                                             "not-an-object"]})
        self.assertEqual(error.status_code, 400)
        self.assertEqual(error.field, "items[1]")

    def test_bad_request_id_is_400_items_index_request_id(self) -> None:
        item = self.fixture.item("req-1", "m1", 1)
        del item["request_id"]
        error = self._shape_error({"items": [item]})
        self.assertEqual(error.status_code, 400)
        self.assertEqual(error.field, "items[0].request_id")
        item = self.fixture.item("req-1", "m1", 1)
        item["request_id"] = ""
        error = self._shape_error({"items": [item]})
        self.assertEqual(error.field, "items[0].request_id")

    def test_duplicate_request_id_in_batch_is_400_at_second_item(self) -> None:
        first = self.fixture.item("req-dup", "m1", 1)
        second = self.fixture.item("req-dup", "m2", 2)
        error = self._shape_error({"items": [first, second]})
        self.assertEqual(error.status_code, 400)
        self.assertEqual(error.field, "items[1].request_id")

    def test_structural_field_error_names_items_index_field(self) -> None:
        good = self.fixture.item("req-1", "m1", 1)
        bad = self.fixture.item("req-2", "m2", 2)
        bad["nonce"] = "not base64!"
        error = self._shape_error({"items": [good, bad]})
        self.assertEqual(error.status_code, 400)
        self.assertEqual(error.field, "items[1].nonce")
        # The structural failure aborts before anything is written.
        self.assertEqual(self._messages(), [])

    # -- live-check errors, in input order ---------------------------------

    def _batch_error(self, items):
        with self.assertRaises(ServiceError) as ctx:
            self.service.submit_verified_message_batch({"items": items})
        return ctx.exception

    def test_unknown_session_is_404_items_session_id(self) -> None:
        items = [self.fixture.item("req-1", "m1", 1),
                 self.fixture.item("req-2", "m2", 1, session_id="ghost")]
        error = self._batch_error(items)
        self.assertEqual(error.status_code, 404)
        self.assertEqual(error.field, "items[1].session_id")
        self.assertEqual(self._messages(), [])

    def test_rotated_session_is_409_items_session_id(self) -> None:
        self.service.rotate_session(self.session_id, {
            "rotation_id": "rot-1",
            "actor_device_id": "d1",
            "prekey_id": "k2",
            "ephemeral_key": _x25519_b64(),
        })
        error = self._batch_error([self.fixture.item("req-1", "m1", 1)])
        self.assertEqual(error.status_code, 409)
        self.assertEqual(error.field, "items[0].session_id")

    def test_revoked_sender_is_409_items_sender_device_id(self) -> None:
        self.service.revoke_device("d1")
        error = self._batch_error([self.fixture.item("req-1", "m1", 1)])
        self.assertEqual(error.status_code, 409)
        self.assertEqual(error.field, "items[0].sender_device_id")

    def test_unknown_sender_is_409_items_sender_device_id(self) -> None:
        error = self._batch_error(
            [self.fixture.item("req-1", "m1", 1, sender="ghost")])
        self.assertEqual(error.status_code, 409)
        self.assertEqual(error.field, "items[0].sender_device_id")

    def test_identity_key_mismatch_is_409_items_identity_key(self) -> None:
        item = self.fixture.item("req-1", "m1", 1)
        item["identity_key"] = _raw_b64(
            ed25519.Ed25519PrivateKey.generate().public_key())
        error = self._batch_error([item])
        self.assertEqual(error.status_code, 409)
        self.assertEqual(error.field, "items[0].identity_key")

    def test_bad_signature_is_400_items_signature(self) -> None:
        items = self._items(2)
        raw = bytearray(base64.b64decode(items[1]["signature"]))
        raw[0] ^= 1
        items[1]["signature"] = base64.b64encode(bytes(raw)).decode()
        error = self._batch_error(items)
        self.assertEqual(error.status_code, 400)
        self.assertEqual(error.field, "items[1].signature")
        self.assertEqual(self._messages(), [])

    def test_duplicate_message_id_is_409_items_message_id(self) -> None:
        self.service.submit_verified_message_batch(
            self.fixture.batch(self.fixture.item("req-1", "m1", 1)))
        # Against an already-committed message id.
        error = self._batch_error([self.fixture.item("req-2", "m1", 2)])
        self.assertEqual(error.status_code, 409)
        self.assertEqual(error.field, "items[0].message_id")
        # Inside the same batch (the second item repeats the first's id).
        error = self._batch_error([self.fixture.item("req-3", "m2", 2),
                                   self.fixture.item("req-4", "m2", 3)])
        self.assertEqual(error.status_code, 409)
        self.assertEqual(error.field, "items[1].message_id")

    def test_bad_sequence_is_409_items_sequence(self) -> None:
        items = self._items(2)
        items[1] = self.fixture.item("req-2", "m2", 5)
        error = self._batch_error(items)
        self.assertEqual(error.status_code, 409)
        self.assertEqual(error.field, "items[1].sequence")

    def test_duplicate_nonce_is_409_items_nonce(self) -> None:
        # Against an already-committed nonce.
        first = self.fixture.item("req-1", "m1", 1)
        self.service.submit_verified_message_batch(
            self.fixture.batch(first))
        reuse = self.fixture.item("req-2", "m2", 2, nonce=first["nonce"])
        error = self._batch_error([reuse])
        self.assertEqual(error.status_code, 409)
        self.assertEqual(error.field, "items[0].nonce")
        # Inside the same batch.
        one = self.fixture.item("req-3", "m3", 2)
        two = self.fixture.item("req-4", "m4", 3, nonce=one["nonce"])
        error = self._batch_error([one, two])
        self.assertEqual(error.status_code, 409)
        self.assertEqual(error.field, "items[1].nonce")

    def test_existing_request_id_with_different_fields_is_409(self) -> None:
        self.service.submit_verified_message_batch(
            self.fixture.batch(self.fixture.item("req-1", "m1", 1)))
        changed = self.fixture.item("req-1", "m2", 2)
        error = self._batch_error([changed])
        self.assertEqual(error.status_code, 409)
        self.assertEqual(error.field, "items[0].request_id")

    # -- atomicity ----------------------------------------------------------

    def test_failed_batch_writes_nothing(self) -> None:
        good = self.fixture.item("req-1", "m1", 1)
        bad = self.fixture.item("req-2", "m2", 9)
        error = self._batch_error([good, bad])
        self.assertEqual(error.field, "items[1].sequence")
        # No message, no idempotency record, no sequence, no nonce: the
        # same items then commit as a fresh batch.
        self.assertEqual(self._messages(), [])
        body, status = self.service.submit_verified_message_batch(
            self.fixture.batch(self.fixture.item("req-1", "m1", 1),
                               self.fixture.item("req-2", "m2", 2)))
        self.assertEqual(status, 201)
        self.assertEqual(len(self._messages()), 2)

    def test_failed_batch_releases_message_id_and_nonce(self) -> None:
        one = self.fixture.item("req-1", "m1", 1)
        two = self.fixture.item("req-2", "m2", 2)
        two["signature"] = base64.b64encode(b"\x00" * 64).decode()
        error = self._batch_error([one, two])
        self.assertEqual(error.field, "items[1].signature")
        # m1's id, sequence 1 and its nonce were all rolled back: a fresh
        # single-entry submission with the same values commits.
        again = self.fixture.item("req-9", "m1", 1, nonce=one["nonce"])
        body, status = self.service.submit_verified_message(again)
        self.assertEqual(status, 201)
        self.assertEqual(body["sequence"], 1)

    def test_later_item_sees_earlier_batch_items(self) -> None:
        # A gap inside the batch fails at the item that breaks continuity.
        items = [self.fixture.item("req-1", "m1", 1),
                 self.fixture.item("req-2", "m2", 3)]
        error = self._batch_error(items)
        self.assertEqual(error.field, "items[1].sequence")
        # And a contiguous batch commits every item in one go.
        body, status = self.service.submit_verified_message_batch(
            self.fixture.batch(*self._items(4)))
        self.assertEqual(status, 201)
        self.assertEqual([m["sequence"] for m in self._messages()],
                         [1, 2, 3, 4])


class SubmitVerifiedBatchPersistenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.path = os.path.join(self.directory, "state.json")

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def _persisted_service(self) -> DeviceService:
        service = DeviceService()
        attach_persistence(service, self.path)
        return service

    def test_batch_replays_identically_after_restart(self) -> None:
        service = self._persisted_service()
        fixture = BatchFixture(service)
        items = [fixture.item("req-1", "m1", 1),
                 fixture.item("req-2", "m2", 2)]
        first, status = service.submit_verified_message_batch(
            {"items": items})
        self.assertEqual(status, 201)

        restarted = self._persisted_service()
        replay, status = restarted.submit_verified_message_batch(
            {"items": items})
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)

    def test_persist_failure_rolls_back_the_whole_batch(self) -> None:
        service = DeviceService()
        state_store = attach_persistence(service, self.path)
        fixture = BatchFixture(service)
        committed = state_store.commit_seq

        def fail_save(_pending) -> None:
            raise OSError("simulated disk failure")

        state_store.save = fail_save  # type: ignore[assignment]
        items = [fixture.item("req-1", "m1", 1),
                 fixture.item("req-2", "m2", 2)]
        with self.assertRaises(PersistenceUnavailable):
            service.submit_verified_message_batch({"items": items})
        # The whole batch rolled back: no message, no record, no generation.
        self.assertEqual(state_store.commit_seq, committed)
        page = service.list_messages(fixture.session_id, "d2", 0, 100)
        self.assertEqual(page["messages"], [])
        # Once the file works again the same batch commits as new.
        del state_store.save
        body, status = service.submit_verified_message_batch(
            {"items": items})
        self.assertEqual(status, 201)
        self.assertEqual(len(body["items"]), 2)


class HTTPSubmitVerifiedBatchTest(unittest.TestCase):
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

    def _item(self, request_id: str, message_id: str, sequence: int) -> dict:
        envelope = {
            "session_id": self.session_id,
            "sender_device_id": "d1",
            "message_id": message_id,
            "sequence": sequence,
            "nonce": base64.b64encode(
                hashlib.sha256(message_id.encode()).digest()[:12]).decode(),
            "ciphertext": base64.b64encode(os.urandom(32)).decode(),
        }
        signed = sign_message(envelope, _seed_b64(self.private))
        signed["request_id"] = request_id
        return signed

    _PATH = "/v1/messages/submit-verified-batch"

    def test_batch_201_then_replay_200(self) -> None:
        payload = {"items": [self._item("req-1", "m1", 1),
                             self._item("req-2", "m2", 2)]}
        status, body = self._request("POST", self._PATH, payload)
        self.assertEqual(status, 201)
        self.assertEqual(set(body), {"items"})
        self.assertEqual([item["message_id"] for item in body["items"]],
                         ["m1", "m2"])
        for item in body["items"]:
            self.assertEqual(set(item), _VIEW_FIELDS)
        status, replay = self._request("POST", self._PATH, payload)
        self.assertEqual(status, 200)
        self.assertEqual(replay, body)

    def test_error_field_is_items_index_qualified(self) -> None:
        items = [self._item("req-1", "m1", 1), self._item("req-2", "m2", 2)]
        raw = bytearray(base64.b64decode(items[1]["signature"]))
        raw[0] ^= 1
        items[1]["signature"] = base64.b64encode(bytes(raw)).decode()
        status, body = self._request("POST", self._PATH, {"items": items})
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "items[1].signature")
        # Nothing committed: the good first item commits on retry.
        status, body = self._request(
            "POST", self._PATH, {"items": [self._item("req-1", "m1", 1)]})
        self.assertEqual(status, 201)

    def test_shape_errors_over_http(self) -> None:
        status, body = self._request("POST", self._PATH, raw=b"{not json")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "request_body")
        status, body = self._request("POST", self._PATH, {"items": []})
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "items")
        status, body = self._request("POST", self._PATH, {"items": [5]})
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "items[0]")

    def test_single_entry_route_unchanged(self) -> None:
        # The ordinary unsigned submit entry gains no signature checks.
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


class CLISubmitVerifiedBatchTest(unittest.TestCase):
    def setUp(self) -> None:
        self.server, _ = create_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        self.private = ed25519.Ed25519PrivateKey.generate()
        identity = _raw_b64(self.private.public_key())
        for device_id, key in (("d1", identity), ("d2", None)):
            status, _ = self._http("POST", "/v1/devices",
                                   _register_payload(device_id,
                                                     identity_key=key))
            self.assertEqual(status, 201)
        status, session = self._http("POST", "/v1/sessions", {
            "initiator_device_id": "d1",
            "recipient_device_id": "d2",
            "prekey_id": "k1",
            "ephemeral_key": _x25519_b64(),
        })
        self.assertEqual(status, 201)
        self.session_id = session["session_id"]
        self.env = dict(os.environ, PYTHONIOENCODING="utf-8")
        existing_pythonpath = os.environ.get("PYTHONPATH", "")
        self.env["PYTHONPATH"] = (
            _REPO_ROOT + (os.pathsep + existing_pythonpath
                          if existing_pythonpath else ""))

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _http(self, method: str, path: str, body=None):
        connection = HTTPConnection("127.0.0.1", self.port, timeout=5)
        payload = json.dumps(body) if body is not None else None
        headers = {"Content-Type": "application/json"} if payload else {}
        connection.request(method, path, body=payload, headers=headers)
        response = connection.getresponse()
        data = response.read().decode("utf-8")
        connection.close()
        return response.status, (json.loads(data) if data else None)

    def _run(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "e2ee_backend", "--base-url",
             self.base_url, *arguments],
            capture_output=True, text=True, timeout=15, env=self.env)

    def _json_line(self, stream: str) -> dict:
        line = stream.strip()
        self.assertEqual(line.count("\n"), 0)
        return json.loads(line)

    def _item(self, request_id: str, message_id: str, sequence: int) -> dict:
        envelope = {
            "session_id": self.session_id,
            "sender_device_id": "d1",
            "message_id": message_id,
            "sequence": sequence,
            "nonce": base64.b64encode(
                hashlib.sha256(message_id.encode()).digest()[:12]).decode(),
            "ciphertext": base64.b64encode(os.urandom(32)).decode(),
        }
        signed = sign_message(envelope, _seed_b64(self.private))
        signed["request_id"] = request_id
        return signed

    def test_success_prints_single_line_items_json(self) -> None:
        items = [self._item("req-1", "m1", 1), self._item("req-2", "m2", 2)]
        result = self._run("submit-verified-batch",
                           "--items", json.dumps(items))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        body = self._json_line(result.stdout)
        self.assertEqual([item["message_id"] for item in body["items"]],
                         ["m1", "m2"])
        # A full replay still succeeds on stdout.
        result = self._run("submit-verified-batch",
                           "--items", json.dumps(items))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self._json_line(result.stdout), body)

    def test_items_from_at_file(self) -> None:
        items = [self._item("req-1", "m1", 1)]
        with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", suffix=".json", delete=False) as handle:
            json.dump(items, handle)
            path = handle.name
        try:
            result = self._run("submit-verified-batch", "--items",
                               "@" + path)
        finally:
            os.unlink(path)
        self.assertEqual(result.returncode, 0, result.stderr)
        body = self._json_line(result.stdout)
        self.assertEqual(body["items"][0]["request_id"], "req-1")

    def test_server_error_goes_to_stderr_nonzero_with_field(self) -> None:
        item = self._item("req-1", "m1", 1)
        item["session_id"] = "ghost"
        item = sign_message(
            {name: item[name] for name in _ENVELOPE_FIELDS},
            _seed_b64(self.private))
        item["request_id"] = "req-1"
        result = self._run("submit-verified-batch",
                           "--items", json.dumps([item]))
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        body = self._json_line(result.stderr)
        self.assertEqual(body["field"], "items[0].session_id")

    def test_invalid_items_json_is_nonzero_with_field(self) -> None:
        result = self._run("submit-verified-batch", "--items", "not json")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertEqual(self._json_line(result.stderr)["field"], "items")
        result = self._run("submit-verified-batch", "--items", "{}")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self._json_line(result.stderr)["field"], "items")


if __name__ == "__main__":
    unittest.main()
