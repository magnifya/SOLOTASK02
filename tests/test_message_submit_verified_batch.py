"""Tests for the batch signature-verified message submission entry.

Covers ``POST /v1/messages/submit-verified-batch`` at the service,
persistence, HTTP and CLI layers: the body is an object whose ``items`` is
a non-empty array of the single verified entry's ``request_id`` plus the
eight ``sign_message`` fields, processed in input order against the
session's virtual sequence. The batch validates and commits in one atomic
transaction (201 with any new item, 200 for a pure replay, replays
unaffected by later revocation/rotation), every failure names
``items[i].<field>`` and writes nothing, concurrent batches linearize to
complete success or complete failure, and the committed records replay
identically after a restart. The single-item verified entry, the ordinary
unsigned submit entry and the read paths are unchanged.
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


class VerifiedBatchFixture:
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
        # the ciphertext at least 16 under the signed-envelope rules).
        digest = hashlib.sha256(
            f"{request_id}:{message_id}".encode("utf-8")).digest()
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

    def three_items(self, start=1, tag=""):
        return [self.envelope(request_id=f"req{tag}-{i}",
                              message_id=f"m{tag}{i}", sequence=i)
                for i in range(start, start + 3)]


class ServiceSubmitVerifiedBatchTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = VerifiedBatchFixture()
        self.service = self.fixture.service
        self.session_id = self.fixture.session_id

    def _submit(self, *items):
        return self.service.submit_verified_message_batch(
            self.fixture.batch(*items))

    def test_first_batch_is_201_with_items_in_order(self) -> None:
        items = self.fixture.three_items()
        body, status = self._submit(*items)
        self.assertEqual(status, 201)
        self.assertEqual(set(body), {"items"})
        self.assertEqual(len(body["items"]), 3)
        for view, item in zip(body["items"], items):
            self.assertEqual(set(view), _VIEW_FIELDS)
            self.assertEqual(view["request_id"], item["request_id"])
            self.assertEqual(view["message_id"], item["message_id"])
            self.assertEqual(view["identity_key"], self.fixture.identity)
            self.assertEqual(view["signature"], item["signature"])
            self.assertTrue(view["created_at"].endswith("+00:00"))
        self.assertEqual([v["sequence"] for v in body["items"]], [1, 2, 3])
        # Only public material is returned; the private seed never appears.
        self.assertNotIn(self.fixture.seed, json.dumps(body))

    def test_virtual_sequence_spans_the_batch(self) -> None:
        # Sequences must continue the session stream across batch items.
        body, status = self._submit(*self.fixture.three_items())
        self.assertEqual(status, 201)
        page = self.service.list_messages(self.session_id, "d2", 0, 100)
        self.assertEqual([m["sequence"] for m in page["messages"]], [1, 2, 3])
        # The next batch continues where the previous one ended.
        more = [self.fixture.envelope(request_id="req-4", message_id="m4",
                                      sequence=4)]
        body, status = self._submit(*more)
        self.assertEqual(status, 201)

    def test_pure_replay_is_200_with_original_body(self) -> None:
        items = self.fixture.three_items()
        first, status = self._submit(*items)
        self.assertEqual(status, 201)
        replay, status = self._submit(*[dict(item) for item in items])
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)

    def test_mixed_replay_and_new_is_201(self) -> None:
        items = self.fixture.three_items()
        first, _ = self._submit(*items)
        new_item = self.fixture.envelope(request_id="req-4", message_id="m4",
                                         sequence=4)
        body, status = self._submit(items[1], new_item, items[0])
        self.assertEqual(status, 201)
        # Replay elements keep their original frozen views, in input order.
        self.assertEqual(body["items"][0], first["items"][1])
        self.assertEqual(body["items"][2], first["items"][0])
        self.assertEqual(body["items"][1]["message_id"], "m4")

    def test_replay_after_revocation_and_rotation_still_returns(self) -> None:
        items = self.fixture.three_items()
        first, _ = self._submit(*items)
        self.service.revoke_device("d1")
        replay, status = self._submit(*[dict(item) for item in items])
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)

    def test_replay_after_identity_and_session_rotation(self) -> None:
        items = self.fixture.three_items()
        first, _ = self._submit(*items)
        new_identity = _raw_b64(ed25519.Ed25519PrivateKey.generate().public_key())
        self.service.rotate_identity_key("d1", {"identity_key": new_identity})
        self.service.rotate_session(self.session_id, {
            "rotation_id": "rot-1", "actor_device_id": "d1",
            "prekey_id": "k2", "ephemeral_key": _x25519_b64()})
        replay, status = self._submit(*[dict(item) for item in items])
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)

    def test_single_entry_replays_against_batch_records(self) -> None:
        # The batch shares the single-item entry's request_id namespace.
        item = self.fixture.envelope()
        body, status = self._submit(item)
        self.assertEqual(status, 201)
        replay, status = self.service.submit_verified_message(dict(item))
        self.assertEqual(status, 200)
        self.assertEqual(replay, body["items"][0])
        # ... and the other way around.
        single = self.fixture.envelope(request_id="req-s", message_id="ms",
                                       sequence=2)
        first, status = self.service.submit_verified_message(single)
        self.assertEqual(status, 201)
        replay, status = self._submit(dict(single))
        self.assertEqual(status, 200)
        self.assertEqual(replay["items"][0], first)

    # -- structural validation --------------------------------------------

    def test_non_object_body_is_400_request_body(self) -> None:
        for bad in (None, [], "x", 7, True):
            with self.assertRaises(ServiceError) as ctx:
                self.service.submit_verified_message_batch(bad)
            self.assertEqual(ctx.exception.status_code, 400, bad)
            self.assertEqual(ctx.exception.field, "request_body", bad)

    def test_missing_empty_or_nonarray_items_is_400_items(self) -> None:
        for bad in ({}, {"items": []}, {"items": "x"}, {"items": 7},
                    {"items": None}, {"items": {}}):
            with self.assertRaises(ServiceError) as ctx:
                self.service.submit_verified_message_batch(bad)
            self.assertEqual(ctx.exception.status_code, 400, bad)
            self.assertEqual(ctx.exception.field, "items", bad)

    def test_non_object_element_is_400_items_i(self) -> None:
        good = self.fixture.envelope()
        for bad in (None, [], "x", 7, True):
            with self.assertRaises(ServiceError) as ctx:
                self._submit(good, bad)
            self.assertEqual(ctx.exception.status_code, 400, bad)
            self.assertEqual(ctx.exception.field, "items[1]", bad)

    def test_bad_request_id_is_400_items_i_request_id(self) -> None:
        good = self.fixture.envelope()
        for bad in ("", 7, None, True):
            item = self.fixture.envelope(request_id="req-2", message_id="m2",
                                         sequence=2)
            item["request_id"] = bad
            with self.assertRaises(ServiceError) as ctx:
                self._submit(good, item)
            self.assertEqual(ctx.exception.status_code, 400, bad)
            self.assertEqual(ctx.exception.field, "items[1].request_id", bad)
        item = self.fixture.envelope(request_id="req-2", message_id="m2",
                                     sequence=2)
        del item["request_id"]
        with self.assertRaises(ServiceError) as ctx:
            self._submit(good, item)
        self.assertEqual(ctx.exception.field, "items[1].request_id")
        item = self.fixture.envelope(request_id="req-2", message_id="m2",
                                     sequence=2)
        item["request_id"] = "ud800\ud800"
        with self.assertRaises(ServiceError) as ctx:
            self._submit(good, item)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.field, "items[1].request_id")

    def test_duplicate_request_id_in_batch_is_400_at_second(self) -> None:
        first = self.fixture.envelope(request_id="req-dup", message_id="m1",
                                      sequence=1)
        second = self.fixture.envelope(request_id="req-dup", message_id="m2",
                                       sequence=2)
        with self.assertRaises(ServiceError) as ctx:
            self._submit(first, second)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.field, "items[1].request_id")
        # Nothing was consumed: both ids are still usable (separately).
        body, status = self._submit(first)
        self.assertEqual(status, 201)

    def test_envelope_field_errors_are_prefixed_with_items_i(self) -> None:
        good = self.fixture.envelope()

        def bad_item(**override):
            item = self.fixture.envelope(request_id="req-2", message_id="m2",
                                         sequence=2)
            item.update(override)
            return item

        missing = bad_item()
        del missing["session_id"]
        cases = [
            (missing, "session_id"),
            (bad_item(sender_device_id=""), "sender_device_id"),
            (bad_item(message_id=""), "message_id"),
            (bad_item(sequence=True), "sequence"),
            (bad_item(sequence=0), "sequence"),
            (bad_item(nonce="!!!"), "nonce"),
            (bad_item(ciphertext=base64.b64encode(b"x" * 15).decode()),
             "ciphertext"),
            (bad_item(identity_key="not-a-key"), "identity_key"),
            (bad_item(signature="!!!"), "signature"),
        ]
        for item, name in cases:
            with self.assertRaises(ServiceError) as ctx:
                self._submit(good, item)
            self.assertEqual(ctx.exception.status_code, 400, name)
            self.assertEqual(ctx.exception.field, f"items[1].{name}", name)

    # -- ordered live-state checks -----------------------------------------

    def test_unknown_session_is_404_items_i_session_id(self) -> None:
        good = self.fixture.envelope()
        ghost = self.fixture.envelope(request_id="req-2", message_id="m2",
                                      sequence=2, session_id="ghost")
        with self.assertRaises(ServiceError) as ctx:
            self._submit(good, ghost)
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "items[1].session_id")

    def test_closed_rotated_session_is_409_items_i_session_id(self) -> None:
        self.service.rotate_session(self.session_id, {
            "rotation_id": "rot-1", "actor_device_id": "d1",
            "prekey_id": "k2", "ephemeral_key": _x25519_b64()})
        with self.assertRaises(ServiceError) as ctx:
            self._submit(self.fixture.envelope())
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "items[0].session_id")

    def test_revoked_sender_is_409_items_i_sender_device_id(self) -> None:
        self.service.revoke_device("d1")
        with self.assertRaises(ServiceError) as ctx:
            self._submit(self.fixture.envelope())
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "items[0].sender_device_id")

    def test_non_member_group_sender_is_409_items_i_sender_device_id(
            self) -> None:
        self.service.register(_register_payload("d3"))
        self.service.create_group({
            "group_id": "g1", "creator_device_id": "d1",
            "member_device_ids": ["d2"]})
        session = self.service.create_group_session({
            "group_id": "g1", "initiator_device_id": "d1",
            "ephemeral_key": _x25519_b64()})
        group_session_id = session["session_id"]
        # A frozen member commits fine through the batch entry.
        member_item = self.fixture.envelope(session_id=group_session_id)
        body, status = self._submit(member_item)
        self.assertEqual(status, 201)
        # d3 is not in the frozen member set.
        outsider = self.fixture.envelope(request_id="req-2", message_id="m2",
                                         sequence=2, sender="d3",
                                         session_id=group_session_id)
        with self.assertRaises(ServiceError) as ctx:
            self._submit(outsider)
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "items[0].sender_device_id")

    def test_submitted_other_key_is_409_items_i_identity_key(self) -> None:
        other = ed25519.Ed25519PrivateKey.generate()
        envelope = {name: self.fixture.envelope()[name]
                    for name in _ENVELOPE_FIELDS}
        foreign = sign_message(envelope, _seed_b64(other))
        foreign["request_id"] = "req-1"
        with self.assertRaises(ServiceError) as ctx:
            self._submit(foreign)
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "items[0].identity_key")

    def test_current_key_not_ed25519_is_409_items_i_identity_key(self) -> None:
        service = DeviceService()
        service.register(_register_payload("d1"))
        service.register(_register_payload("d2"))
        session = service.create_session({
            "initiator_device_id": "d1", "recipient_device_id": "d2",
            "prekey_id": "k1", "ephemeral_key": _x25519_b64()})
        signed = self.fixture.envelope(session_id=session["session_id"])
        with self.assertRaises(ServiceError) as ctx:
            service.submit_verified_message_batch({"items": [signed]})
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "items[0].identity_key")

    def test_signature_failure_is_400_items_i_signature(self) -> None:
        good = self.fixture.envelope()
        bad = self.fixture.envelope(request_id="req-2", message_id="m2",
                                    sequence=2)
        raw = bytearray(base64.b64decode(bad["signature"]))
        raw[0] ^= 1
        bad["signature"] = base64.b64encode(bytes(raw)).decode()
        with self.assertRaises(ServiceError) as ctx:
            self._submit(good, bad)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.field, "items[1].signature")

    def test_message_id_sequence_nonce_conflicts_keep_order(self) -> None:
        first, _ = self._submit(self.fixture.envelope())
        # Duplicate message_id wins over the (also wrong) sequence/nonce.
        with self.assertRaises(ServiceError) as ctx:
            self._submit(self.fixture.envelope(request_id="req-2"))
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "items[0].message_id")
        # Fresh message_id, wrong sequence.
        payload = self.fixture.envelope(request_id="req-2", message_id="m2",
                                        sequence=3)
        with self.assertRaises(ServiceError) as ctx:
            self._submit(payload)
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "items[0].sequence")
        # Fresh message_id/sequence but the first message's nonce, re-signed.
        payload = self.fixture.envelope(
            request_id="req-2", message_id="m2", sequence=2,
            nonce=first["items"][0]["nonce"])
        with self.assertRaises(ServiceError) as ctx:
            self._submit(payload)
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "items[0].nonce")

    def test_in_batch_duplicate_message_id_and_nonce_conflict(self) -> None:
        # The virtual stream counts earlier fresh items of the same batch.
        first = self.fixture.envelope(request_id="req-1", message_id="m1",
                                      sequence=1)
        same_id = self.fixture.envelope(request_id="req-2", message_id="m1",
                                        sequence=2)
        with self.assertRaises(ServiceError) as ctx:
            self._submit(first, same_id)
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "items[1].message_id")
        same_nonce = self.fixture.envelope(
            request_id="req-2", message_id="m2", sequence=2,
            nonce=first["nonce"])
        with self.assertRaises(ServiceError) as ctx:
            self._submit(first, same_nonce)
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "items[1].nonce")
        # A sequence gap inside the batch fails at the gapped item.
        gapped = self.fixture.envelope(request_id="req-2", message_id="m2",
                                       sequence=3)
        with self.assertRaises(ServiceError) as ctx:
            self._submit(first, gapped)
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "items[1].sequence")

    def test_existing_id_with_changed_fields_is_409_items_i_request_id(
            self) -> None:
        item = self.fixture.envelope()
        self._submit(item)
        changed = self.fixture.envelope(request_id="req-1", message_id="m2",
                                        sequence=2)
        with self.assertRaises(ServiceError) as ctx:
            self._submit(changed)
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "items[0].request_id")

    # -- atomicity -----------------------------------------------------------

    def test_failure_writes_nothing_and_consumes_nothing(self) -> None:
        good = self.fixture.envelope(request_id="req-1", message_id="m1",
                                     sequence=1)
        bad = self.fixture.envelope(request_id="req-2", message_id="m2",
                                    sequence=9)
        with self.assertRaises(ServiceError) as ctx:
            self._submit(good, bad)
        self.assertEqual(ctx.exception.field, "items[1].sequence")
        # No message, idempotency record, sequence or nonce was consumed:
        # the exact same batch body, fixed, now commits from sequence 1.
        fixed = self.fixture.envelope(request_id="req-2", message_id="m2",
                                      sequence=2)
        body, status = self._submit(good, fixed)
        self.assertEqual(status, 201)
        page = self.service.list_messages(self.session_id, "d2", 0, 100)
        self.assertEqual([m["message_id"] for m in page["messages"]],
                         ["m1", "m2"])

    def test_failed_batch_leaves_single_entry_usable(self) -> None:
        good = self.fixture.envelope()
        bad = self.fixture.envelope(request_id="req-2", message_id="m2",
                                    sequence=2)
        raw = bytearray(base64.b64decode(bad["signature"]))
        raw[0] ^= 1
        bad["signature"] = base64.b64encode(bytes(raw)).decode()
        with self.assertRaises(ServiceError):
            self._submit(good, bad)
        # The first item's request_id was not consumed by the failed batch.
        body, status = self.service.submit_verified_message(good)
        self.assertEqual(status, 201)
        page = self.service.list_messages(self.session_id, "d2", 0, 100)
        self.assertEqual(len(page["messages"]), 1)

    def test_concurrent_identical_batches_write_one_batch(self) -> None:
        items = self.fixture.three_items()
        results, errors = [], []

        def worker() -> None:
            try:
                results.append(self._submit(*[dict(i) for i in items]))
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
        self.assertEqual(len(page["messages"]), 3)

    def test_concurrent_overlapping_batches_are_all_or_nothing(self) -> None:
        # Two different batches racing for the same sequences: exactly one
        # commits in full; the other fails in full and consumes nothing.
        batch_a = self.fixture.three_items(tag="a")
        batch_b = self.fixture.three_items(tag="b")
        results, errors = [], []

        def run(items) -> None:
            try:
                results.append(self._submit(*items))
            except ServiceError as error:
                errors.append(error)

        threads = [threading.Thread(target=run, args=(batch_a,)),
                   threading.Thread(target=run, args=(batch_b,))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][1], 201)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].status_code, 409)
        self.assertTrue(errors[0].field.startswith("items["))
        page = self.service.list_messages(self.session_id, "d2", 0, 100)
        self.assertEqual(len(page["messages"]), 3)
        # The loser's ids were not consumed: its batch commits after
        # re-sequencing to continue the stream.
        winner_ids = {m["message_id"] for m in page["messages"]}
        loser = batch_b if winner_ids == {"ma1", "ma2", "ma3"} else batch_a
        resequenced = [
            self.fixture.envelope(request_id=item["request_id"],
                                  message_id=item["message_id"],
                                  sequence=4 + offset)
            for offset, item in enumerate(loser)
        ]
        body, status = self._submit(*resequenced)
        self.assertEqual(status, 201)
        self.assertEqual(len(body["items"]), 3)


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

    def _fixture(self, service: DeviceService) -> VerifiedBatchFixture:
        fixture = VerifiedBatchFixture.__new__(VerifiedBatchFixture)
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

    def test_batch_replay_values_survive_restart(self) -> None:
        service = self._persisted_service()
        fixture = self._fixture(service)
        items = fixture.three_items()
        first, status = service.submit_verified_message_batch(
            {"items": items})
        self.assertEqual(status, 201)

        restarted = self._persisted_service()
        fixture.service = restarted
        replay, status = restarted.submit_verified_message_batch(
            {"items": [dict(item) for item in items]})
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)
        # The stream continues after the restarted batch's sequences.
        follow = fixture.envelope(request_id="req-4", message_id="m4",
                                  sequence=4)
        body, status = restarted.submit_verified_message_batch(
            {"items": [follow]})
        self.assertEqual(status, 201)

    def test_failed_batch_persists_nothing(self) -> None:
        service = self._persisted_service()
        fixture = self._fixture(service)
        good = fixture.envelope()
        bad = fixture.envelope(request_id="req-2", message_id="m2",
                               sequence=9)
        with self.assertRaises(ServiceError):
            service.submit_verified_message_batch({"items": [good, bad]})
        restarted = self._persisted_service()
        fixture.service = restarted
        # Neither the good item's id nor its nonce/sequence was consumed.
        body, status = restarted.submit_verified_message_batch(
            {"items": [good]})
        self.assertEqual(status, 201)


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

    def _item(self, request_id, message_id, sequence) -> dict:
        envelope = {
            "session_id": self.session_id,
            "sender_device_id": "d1",
            "message_id": message_id,
            "sequence": sequence,
            "nonce": base64.b64encode(os.urandom(12)).decode(),
            "ciphertext": base64.b64encode(os.urandom(32)).decode(),
        }
        signed = sign_message(envelope, _seed_b64(self.private))
        signed["request_id"] = request_id
        return signed

    def test_submit_replay_and_conflict(self) -> None:
        items = [self._item("req-1", "m1", 1), self._item("req-2", "m2", 2)]
        status, first = self._request(
            "POST", "/v1/messages/submit-verified-batch", {"items": items})
        self.assertEqual(status, 201)
        self.assertEqual(set(first), {"items"})
        self.assertEqual([set(view) for view in first["items"]],
                         [_VIEW_FIELDS, _VIEW_FIELDS])
        status, replay = self._request(
            "POST", "/v1/messages/submit-verified-batch", {"items": items})
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)
        changed = dict(items[0], message_id="m9", sequence=9)
        status, body = self._request(
            "POST", "/v1/messages/submit-verified-batch",
            {"items": [changed]})
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "items[0].request_id")

    def test_bad_json_is_400_request_body(self) -> None:
        status, body = self._request(
            "POST", "/v1/messages/submit-verified-batch", raw=b"{not json")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "request_body")

    def test_non_object_is_400_request_body(self) -> None:
        for raw in (b"[1, 2]", b"5", b'"x"'):
            status, body = self._request(
                "POST", "/v1/messages/submit-verified-batch", raw=raw)
            self.assertEqual(status, 400, raw)
            self.assertEqual(body["field"], "request_body", raw)

    def test_empty_items_is_400_items(self) -> None:
        status, body = self._request(
            "POST", "/v1/messages/submit-verified-batch", {"items": []})
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "items")

    def test_element_error_names_items_i_field(self) -> None:
        good = self._item("req-1", "m1", 1)
        bad = self._item("req-2", "m2", 2)
        raw = bytearray(base64.b64decode(bad["signature"]))
        raw[0] ^= 1
        bad["signature"] = base64.b64encode(bytes(raw)).decode()
        status, body = self._request(
            "POST", "/v1/messages/submit-verified-batch",
            {"items": [good, bad]})
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "items[1].signature")
        # The failed batch wrote nothing: the good item still commits.
        status, body = self._request(
            "POST", "/v1/messages/submit-verified-batch", {"items": [good]})
        self.assertEqual(status, 201)

    def test_single_and_ordinary_routes_unchanged(self) -> None:
        # The ordinary unsigned submit gains no signature checking.
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
        # The single verified entry still works and shares the namespace.
        item = self._item("req-v", "mv", 2)
        status, body = self._request(
            "POST", "/v1/messages/submit-verified", item)
        self.assertEqual(status, 201)
        status, body = self._request(
            "POST", "/v1/messages/submit-verified-batch", {"items": [item]})
        self.assertEqual(status, 200)


class SubmitVerifiedBatchCLITest(unittest.TestCase):
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

    def _request(self, method: str, path: str, body=None):
        connection = HTTPConnection("127.0.0.1", self.port, timeout=5)
        payload = json.dumps(body) if body is not None else None
        headers = {"Content-Type": "application/json"} if payload is not None else {}
        connection.request(method, path, body=payload, headers=headers)
        response = connection.getresponse()
        data = response.read().decode("utf-8")
        connection.close()
        return response.status, (json.loads(data) if data else None)

    def _run(self, *arguments) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "e2ee_backend", "--base-url",
             self.base_url, *arguments],
            capture_output=True, text=True, timeout=15)

    def _item(self, request_id, message_id, sequence) -> dict:
        envelope = {
            "session_id": self.session_id,
            "sender_device_id": "d1",
            "message_id": message_id,
            "sequence": sequence,
            "nonce": base64.b64encode(os.urandom(12)).decode(),
            "ciphertext": base64.b64encode(os.urandom(32)).decode(),
        }
        signed = sign_message(envelope, _seed_b64(self.private))
        signed["request_id"] = request_id
        return signed

    def test_success_stdout_zero_and_idempotent_replay(self) -> None:
        items = [self._item("req-1", "m1", 1), self._item("req-2", "m2", 2)]
        first = self._run("submit-verified-batch", json.dumps(items))
        self.assertEqual(first.returncode, 0, first.stderr)
        body = json.loads(first.stdout.strip())
        self.assertEqual(set(body), {"items"})
        self.assertEqual([v["message_id"] for v in body["items"]],
                         ["m1", "m2"])
        self.assertFalse(first.stderr.strip())
        # Replaying the exact same items is a pure idempotent replay.
        second = self._run("submit-verified-batch", json.dumps(items))
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(json.loads(second.stdout.strip()), body)

    def test_at_file_input(self) -> None:
        items = [self._item("req-f", "mf", 1)]
        handle, path = tempfile.mkstemp(suffix=".json")
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        with os.fdopen(handle, "w", encoding="utf-8") as file:
            json.dump(items, file)
        result = self._run("submit-verified-batch", f"@{path}")
        self.assertEqual(result.returncode, 0, result.stderr)
        body = json.loads(result.stdout.strip())
        self.assertEqual([v["request_id"] for v in body["items"]], ["req-f"])

    def test_invalid_json_text_is_exit_2_items(self) -> None:
        result = self._run("submit-verified-batch", "{not json")
        self.assertEqual(result.returncode, 2)
        self.assertFalse(result.stdout.strip())
        body = json.loads(result.stderr.strip())
        self.assertEqual(body["field"], "items")

    def test_missing_file_is_exit_2_items(self) -> None:
        result = self._run("submit-verified-batch", "@/no/such/file.json")
        self.assertEqual(result.returncode, 2)
        self.assertFalse(result.stdout.strip())
        body = json.loads(result.stderr.strip())
        self.assertEqual(body["field"], "items")

    def test_server_failure_stderr_field_nonzero(self) -> None:
        # A sequence gap is a server-side 409 naming the element's field.
        items = [self._item("req-1", "m1", 3)]
        result = self._run("submit-verified-batch", json.dumps(items))
        self.assertEqual(result.returncode, 1)
        self.assertFalse(result.stdout.strip())
        body = json.loads(result.stderr.strip())
        self.assertEqual(body["field"], "items[0].sequence")

    def test_non_array_items_json_gets_server_400(self) -> None:
        result = self._run("submit-verified-batch", '{"items": []}')
        self.assertEqual(result.returncode, 1)
        body = json.loads(result.stderr.strip())
        self.assertEqual(body["field"], "items")


if __name__ == "__main__":
    unittest.main()
