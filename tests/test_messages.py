"""Tests for the message service: validation, status codes and ordering."""
import base64
import threading
import unittest

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import x25519

from e2ee_backend.service import DeviceService, ServiceError


def _raw_key_b64() -> str:
    key = x25519.X25519PrivateKey.generate().public_key()
    raw = key.public_bytes(serialization.Encoding.Raw,
                           serialization.PublicFormat.Raw)
    return base64.b64encode(raw).decode()


class _MessageTestBase(unittest.TestCase):
    """Registers da/db, negotiates a session; subclasses send into it."""

    INITIATOR = "da"
    RECIPIENT = "db"

    def setUp(self) -> None:
        self.service = DeviceService()
        self.service.register({
            "user_id": "u1", "device_id": self.INITIATOR,
            "identity_key": _raw_key_b64(), "signed_prekeys": []})
        self.service.register({
            "user_id": "u2", "device_id": self.RECIPIENT,
            "identity_key": _raw_key_b64(),
            "signed_prekeys": [{"key_id": "pk",
                                "public_key": _raw_key_b64()}]})
        self.session_id = self.service.create_session({
            "initiator_device_id": self.INITIATOR,
            "recipient_device_id": self.RECIPIENT,
            "prekey_id": "pk", "ephemeral_key": _raw_key_b64()})["session_id"]

    def _message(self, *, sequence: int, message_id: str = "m",
                 sender: str = "da", **overrides: object) -> dict:
        payload = {
            "session_id": self.session_id,
            "sender_device_id": sender,
            "message_id": message_id,
            "sequence": sequence,
            "nonce": f"nonce-{sequence}",
            "ciphertext": f"ciphertext-{sequence}",
        }
        payload.update(overrides)
        return payload


class SendMessageTest(_MessageTestBase):
    def test_first_message_is_201_with_created_at(self) -> None:
        body = self.service.send_message(self._message(sequence=1))
        self.assertEqual(set(body), {
            "session_id", "sender_device_id", "message_id", "sequence",
            "nonce", "ciphertext", "created_at"})
        self.assertEqual(body["sequence"], 1)
        self.assertTrue(body["created_at"].endswith("+00:00"))

    def test_sequences_contiguous_from_one(self) -> None:
        for sequence in range(1, 6):
            body = self.service.send_message(
                self._message(sequence=sequence, message_id=f"m{sequence}"))
            self.assertEqual(body["sequence"], sequence)

    def test_unknown_session_is_404_field_session_id(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.send_message(
                self._message(sequence=1, session_id="ghost"))
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "session_id")

    def test_unknown_sender_is_409_field_sender(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.send_message(
                self._message(sequence=1, sender="ghost"))
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "sender_device_id")

    def test_revoked_sender_is_409_field_sender(self) -> None:
        self.service.revoke_device(self.RECIPIENT)
        with self.assertRaises(ServiceError) as ctx:
            self.service.send_message(
                self._message(sequence=1, sender=self.RECIPIENT))
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "sender_device_id")

    def test_duplicate_message_id_is_409_field_message_id(self) -> None:
        self.service.send_message(self._message(sequence=1, message_id="dup"))
        with self.assertRaises(ServiceError) as ctx:
            self.service.send_message(self._message(sequence=2,
                                                    message_id="dup"))
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "message_id")

    def test_out_of_order_sequence_is_409_field_sequence(self) -> None:
        self.service.send_message(self._message(sequence=1, message_id="m1"))
        # Gap (expects 2, sends 3).
        with self.assertRaises(ServiceError) as ctx:
            self.service.send_message(self._message(sequence=3,
                                                    message_id="m3"))
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "sequence")
        # Replay of an already-used sequence.
        with self.assertRaises(ServiceError) as ctx:
            self.service.send_message(self._message(sequence=1,
                                                    message_id="again"))
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "sequence")
        # The rejected appends wrote nothing: sequence 2 still succeeds.
        body = self.service.send_message(self._message(sequence=2,
                                                       message_id="m2"))
        self.assertEqual(body["sequence"], 2)

    def test_failed_send_writes_nothing(self) -> None:
        self.service.send_message(self._message(sequence=1, message_id="m1"))
        for payload in (
                self._message(sequence=2, message_id="m1"),  # duplicate
                self._message(sequence=9, message_id="m9"),   # bad sequence
                self._message(sequence=2, sender="ghost")):   # inactive
            with self.assertRaises(ServiceError):
                self.service.send_message(payload)
        messages = self.service.store._messages[self.session_id]
        self.assertEqual(len(messages), 1)


class SendMessageValidationTest(_MessageTestBase):
    def _assert_400(self, payload: object, field: str) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.send_message(payload)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.field, field)

    def test_body_must_be_object(self) -> None:
        self._assert_400(["nope"], "request_body")
        self._assert_400("nope", "request_body")

    def test_missing_fields_400_in_declared_order(self) -> None:
        for name in ("session_id", "sender_device_id", "message_id",
                     "sequence", "nonce", "ciphertext"):
            payload = self._message(sequence=1)
            del payload[name]
            self._assert_400(payload, name)

    def test_string_fields_must_be_nonempty_strings(self) -> None:
        self._assert_400(self._message(sequence=1, session_id=""),
                         "session_id")
        self._assert_400(self._message(sequence=1, sender=7),
                         "sender_device_id")
        self._assert_400(self._message(sequence=1, message_id=None),
                         "message_id")
        self._assert_400(self._message(sequence=1, nonce=123), "nonce")
        self._assert_400(self._message(sequence=1, ciphertext=[]),
                         "ciphertext")

    def test_sequence_must_be_positive_integer(self) -> None:
        for bad in ("1", 1.0, True, False, None, 0, -1):
            self._assert_400(self._message(sequence=bad), "sequence")


class PullMessagesTest(_MessageTestBase):
    def _seed(self, count: int) -> None:
        for sequence in range(1, count + 1):
            self.service.send_message(self._message(
                sequence=sequence, message_id=f"m{sequence}"))

    def test_empty_session_returns_empty_page_next_after_zero(self) -> None:
        body = self.service.pull_messages(
            self.session_id, {"device_id": ["da"]})
        self.assertEqual(body, {"messages": [], "next_after": 0})

    def test_default_paging_returns_all_ascending(self) -> None:
        self._seed(3)
        body = self.service.pull_messages(
            self.session_id, {"device_id": [self.RECIPIENT]})
        self.assertEqual([m["sequence"] for m in body["messages"]], [1, 2, 3])
        self.assertEqual(body["next_after"], 3)
        self.assertEqual(set(body["messages"][0]), {
            "session_id", "sender_device_id", "message_id", "sequence",
            "nonce", "ciphertext", "created_at"})

    def test_after_and_limit_filter(self) -> None:
        self._seed(5)
        body = self.service.pull_messages(
            self.session_id,
            {"device_id": ["da"], "after": ["2"], "limit": ["2"]})
        self.assertEqual([m["sequence"] for m in body["messages"]], [3, 4])
        self.assertEqual(body["next_after"], 4)

    def test_empty_page_keeps_next_after_equal_to_after(self) -> None:
        self._seed(2)
        body = self.service.pull_messages(
            self.session_id, {"device_id": ["da"], "after": ["9"]})
        self.assertEqual(body, {"messages": [], "next_after": 9})

    def test_walk_pages_with_next_after(self) -> None:
        self._seed(5)
        seen = []
        after = 0
        while True:
            body = self.service.pull_messages(
                self.session_id,
                {"device_id": ["da"], "after": [str(after)], "limit": ["2"]})
            seen.extend(m["sequence"] for m in body["messages"])
            after = body["next_after"]
            if not body["messages"]:
                break
        self.assertEqual(seen, [1, 2, 3, 4, 5])

    def test_unknown_session_is_404(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.pull_messages("ghost", {"device_id": ["da"]})
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "session_id")

    def test_inactive_device_is_409(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.pull_messages(
                self.session_id, {"device_id": ["ghost"]})
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "device_id")

    def test_revoked_device_is_409(self) -> None:
        self.service.revoke_device(self.RECIPIENT)
        with self.assertRaises(ServiceError) as ctx:
            self.service.pull_messages(
                self.session_id, {"device_id": [self.RECIPIENT]})
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.field, "device_id")

    def test_query_validation_400(self) -> None:
        cases = (
            ({}, "device_id"),
            ({"device_id": [""]}, "device_id"),
            ({"device_id": ["da"], "after": ["x"]}, "after"),
            ({"device_id": ["da"], "after": ["-1"]}, "after"),
            ({"device_id": ["da"], "limit": ["x"]}, "limit"),
            ({"device_id": ["da"], "limit": ["0"]}, "limit"),
            ({"device_id": ["da"], "limit": ["101"]}, "limit"),
        )
        for params, field in cases:
            with self.assertRaises(ServiceError) as ctx:
                self.service.pull_messages(self.session_id, params)
            self.assertEqual(ctx.exception.status_code, 400, field)
            self.assertEqual(ctx.exception.field, field, field)

    def test_defaults_are_after_zero_limit_100(self) -> None:
        self._seed(3)
        body = self.service.pull_messages(
            self.session_id, {"device_id": ["da"]})
        self.assertEqual(len(body["messages"]), 3)
        self.assertEqual(body["next_after"], 3)


class MessageConcurrencyTest(unittest.TestCase):
    def test_concurrent_appends_get_contiguous_sequences(self) -> None:
        service = DeviceService()
        service.register({"user_id": "u1", "device_id": "da",
                          "identity_key": _raw_key_b64(),
                          "signed_prekeys": []})
        service.register({"user_id": "u2", "device_id": "db",
                          "identity_key": _raw_key_b64(),
                          "signed_prekeys": [{"key_id": "pk",
                                              "public_key": _raw_key_b64()}]})
        session_id = service.create_session({
            "initiator_device_id": "da", "recipient_device_id": "db",
            "prekey_id": "pk", "ephemeral_key": _raw_key_b64()})["session_id"]

        n_threads = 16
        per_thread = 10
        barrier = threading.Barrier(n_threads)

        def worker(worker_id: int) -> None:
            barrier.wait()
            for index in range(per_thread):
                message_id = f"w{worker_id}-{index}"
                sequence = 1
                while True:
                    try:
                        service.send_message({
                            "session_id": session_id,
                            "sender_device_id": "da",
                            "message_id": message_id,
                            "sequence": sequence,
                            "nonce": "n", "ciphertext": "c"})
                        break
                    except ServiceError as error:
                        # Only a sequence collision should happen; retry with
                        # the next number. Any other error is a real failure.
                        self.assertEqual(error.field, "sequence")
                        sequence += 1

        threads = [threading.Thread(target=worker, args=(i,))
                   for i in range(n_threads)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        sequences = []
        after = 0
        while True:
            page = service.pull_messages(
                session_id,
                {"device_id": ["db"], "after": [str(after)],
                 "limit": ["100"]})
            sequences.extend(m["sequence"] for m in page["messages"])
            after = page["next_after"]
            if not page["messages"]:
                break
        self.assertEqual(sequences, list(range(1, n_threads * per_thread + 1)))


if __name__ == "__main__":
    unittest.main()
