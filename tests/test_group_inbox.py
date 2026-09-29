"""Tests for the per-device aggregated group offline inbox.

GET /v1/devices/{device_id}/group-inbox and its long-polling sibling
``/group-inbox/wait`` aggregate, read-only and under the store lock, the
group-session messages the device is a frozen recipient of but has not yet
acknowledged per device, ordered by (session.created_at, session_id,
sequence). 1:1 messages, own sends, sessions frozen before the device
joined the group, and already-acked messages never contribute; a member
removed after the freeze still sees the frozen session.
"""
import json
import threading
import time
import unittest
from http.client import HTTPConnection

from e2ee_backend.http_app import create_server
from e2ee_backend.models import Device, SignedPreKey
from e2ee_backend.service import DeviceService, ServiceError


def _payload(session_id, message_id, sequence, sender):
    return {
        "session_id": session_id,
        "sender_device_id": sender,
        "message_id": message_id,
        "sequence": sequence,
        "nonce": f"nonce-{message_id}",
        "ciphertext": "ciphertext",
    }


class GroupInboxFixture:
    """d1 creates g1 with d1/d2/d3 and freezes gs1; d4 is an outsider.

    gs1 holds m1/m2 from d1 and m3 from d2. A 1:1 session d1->d2 carries a
    message that must never appear in d2's group inbox.
    """

    def __init__(self) -> None:
        self.service = DeviceService()
        self.store = self.service.store
        for device_id in ("d1", "d3", "d4"):
            self.store.add_device(Device("u", device_id, "ik"))
        self.store.add_device(Device(
            "u", "d2", "ik",
            prekeys=[SignedPreKey("pk1", "pubk1")]))
        self.service.create_group({
            "group_id": "g1", "creator_device_id": "d1",
            "member_device_ids": ["d2", "d3"]})
        self.gs1 = self.service.create_group_session({
            "group_id": "g1", "initiator_device_id": "d1",
            "ephemeral_key": "epk1"})["session_id"]
        for message_id, sequence, sender in (
                ("m1", 1, "d1"), ("m2", 2, "d1"), ("m3", 3, "d2")):
            self.service.post_message(
                _payload(self.gs1, message_id, sequence, sender))
        self.one2one = self.store.create_session(
            "d1", "d2", "pk1", "ek").session_id
        self.service.post_message(
            _payload(self.one2one, "p1", 1, "d1"))


class GroupInboxServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = GroupInboxFixture()
        self.service = self.fixture.service
        self.store = self.fixture.store

    def test_shape_and_frozen_recipient_filter(self) -> None:
        body = self.service.device_group_inbox("d2", 100)
        self.assertEqual(list(body), ["device_id", "messages", "has_more"])
        self.assertEqual(body["device_id"], "d2")
        self.assertEqual([m["message_id"] for m in body["messages"]],
                         ["m1", "m2"])
        self.assertFalse(body["has_more"])
        self.assertEqual(list(body["messages"][0]), [
            "session_id", "sender_device_id", "message_id", "sequence",
            "nonce", "ciphertext", "created_at"])
        self.assertEqual(
            [m["message_id"]
             for m in self.service.device_group_inbox("d3", 100)
             ["messages"]],
            ["m1", "m2", "m3"])
        # d1 is a frozen recipient of d2's m3 (its own m1/m2 are excluded).
        self.assertEqual(
            [m["message_id"]
             for m in self.service.device_group_inbox("d1", 100)
             ["messages"]],
            ["m3"])

    def test_outsider_and_late_joiner_see_nothing_of_old_freeze(self) -> None:
        self.assertEqual(
            self.service.device_group_inbox("d4", 100)["messages"], [])
        self.service.add_group_member("g1", {
            "actor_device_id": "d1", "device_id": "d4"})
        self.assertEqual(
            self.service.device_group_inbox("d4", 100)["messages"], [])
        gs2 = self.service.create_group_session({
            "group_id": "g1", "initiator_device_id": "d1",
            "ephemeral_key": "epk2"})["session_id"]
        self.service.post_message(_payload(gs2, "n1", 1, "d1"))
        self.assertEqual(
            [m["message_id"]
             for m in self.service.device_group_inbox("d4", 100)
             ["messages"]],
            ["n1"])
        d2_view = self.service.device_group_inbox("d2", 100)
        self.assertEqual({m["session_id"] for m in d2_view["messages"]},
                         {self.fixture.gs1, gs2})

    def test_removed_member_keeps_frozen_session(self) -> None:
        self.service.remove_group_member(
            "g1", {"actor_device_id": "d1", "device_id": "d3"})
        self.assertEqual(
            [m["message_id"]
             for m in self.service.device_group_inbox("d3", 100)
             ["messages"]],
            ["m1", "m2", "m3"])

    def test_per_device_ack_hides_only_that_devices_copy(self) -> None:
        self.service.ack_message(self.fixture.gs1, {
            "device_id": "d2", "message_id": "m1", "sequence": 1})
        self.assertEqual(
            [m["message_id"]
             for m in self.service.device_group_inbox("d2", 100)
             ["messages"]],
            ["m2"])
        self.assertEqual(
            [m["message_id"]
             for m in self.service.device_group_inbox("d3", 100)
             ["messages"]],
            ["m1", "m2", "m3"])

    def test_limit_page_and_has_more(self) -> None:
        body = self.service.device_group_inbox("d3", 2)
        self.assertEqual([m["message_id"] for m in body["messages"]],
                         ["m1", "m2"])
        self.assertTrue(body["has_more"])
        body = self.service.device_group_inbox("d3", 3)
        self.assertEqual(len(body["messages"]), 3)
        self.assertFalse(body["has_more"])

    def test_ordering_by_created_at_session_id_sequence(self) -> None:
        ga = self.service.create_group_session({
            "group_id": "g1", "initiator_device_id": "d1",
            "ephemeral_key": "eka"})["session_id"]
        gb = self.service.create_group_session({
            "group_id": "g1", "initiator_device_id": "d1",
            "ephemeral_key": "ekb"})["session_id"]
        self.service.post_message(_payload(ga, "a1", 1, "d1"))
        self.service.post_message(_payload(gb, "b1", 1, "d1"))
        stamp = "2000-01-01T00:00:00.000000+00:00"
        self.store._group_sessions[ga].created_at = stamp
        self.store._group_sessions[gb].created_at = stamp
        ids = [(m["session_id"], m["message_id"])
               for m in self.service.device_group_inbox("d3", 100)
               ["messages"]]
        expected_head = sorted([(ga, "a1"), (gb, "b1")])
        self.assertEqual(ids[:2], expected_head)
        self.assertEqual([m[1] for m in ids[2:]], ["m1", "m2", "m3"])

    def test_unknown_or_revoked_device_conflicts(self) -> None:
        with self.assertRaises(ServiceError) as caught:
            self.service.device_group_inbox("ghost", 100)
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "device_id")
        self.store.revoke_device("d2")
        with self.assertRaises(ServiceError) as caught:
            self.service.device_group_inbox("d2", 100)
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "device_id")

    def test_invalid_arguments_rejected(self) -> None:
        for limit in (0, 101, True, False, "3", 1.0):
            with self.assertRaises(ServiceError) as caught:
                self.service.device_group_inbox("d2", limit)
            self.assertEqual(caught.exception.field, "limit")
        with self.assertRaises(ServiceError) as caught:
            self.service.device_group_inbox_wait("d2", 100, 30001)
        self.assertEqual(caught.exception.field, "timeout_ms")

    def test_read_only_unchanged_state_is_stable(self) -> None:
        first = self.service.device_group_inbox("d3", 2)
        second = self.service.device_group_inbox("d3", 2)
        self.assertEqual(first, second)


class GroupInboxWaitServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = GroupInboxFixture()
        self.service = self.fixture.service

    def test_populated_returns_immediately(self) -> None:
        start = time.monotonic()
        body = self.service.device_group_inbox_wait("d2", 1, 30000)
        self.assertLess(time.monotonic() - start, 1.0)
        self.assertEqual([m["message_id"] for m in body["messages"]], ["m1"])
        self.assertTrue(body["has_more"])

    def test_zero_timeout_empty_inbox(self) -> None:
        # d4 is registered but frozen into no group session.
        body = self.service.device_group_inbox_wait("d4", 100, 0)
        self.assertEqual(body, {"device_id": "d4", "messages": [],
                                "has_more": False})

    def test_wakes_on_new_group_message(self) -> None:
        result = {}

        def wait():
            result["body"] = self.service.device_group_inbox_wait(
                "d4", 100, 3000)

        thread = threading.Thread(target=wait)
        thread.start()
        time.sleep(0.1)
        self.service.add_group_member("g1", {
            "actor_device_id": "d1", "device_id": "d4"})
        gs2 = self.service.create_group_session({
            "group_id": "g1", "initiator_device_id": "d1",
            "ephemeral_key": "epk2"})["session_id"]
        self.service.post_message(_payload(gs2, "n1", 1, "d1"))
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(
            [m["message_id"] for m in result["body"]["messages"]], ["n1"])

    def test_wakes_on_revocation_with_priority(self) -> None:
        result = {}

        def wait():
            try:
                self.service.device_group_inbox_wait("d4", 100, 3000)
            except ServiceError as error:
                result["error"] = error

        thread = threading.Thread(target=wait)
        thread.start()
        time.sleep(0.1)
        self.fixture.store.revoke_device("d4")
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result["error"].status_code, 409)
        self.assertEqual(result["error"].field, "device_id")


class GroupInboxHTTPTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = GroupInboxFixture()
        self.server, _service = create_server(
            "127.0.0.1", 0, self.fixture.service)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _request(self, path, body=None):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("GET", path, body=body)
        response = conn.getresponse()
        status = response.status
        raw = response.read()
        conn.close()
        return status, raw

    def test_success_and_default_limit(self) -> None:
        status, raw = self._request("/v1/devices/d3/group-inbox")
        self.assertEqual(status, 200)
        body = json.loads(raw)
        self.assertEqual([m["message_id"] for m in body["messages"]],
                         ["m1", "m2", "m3"])
        self.assertFalse(body["has_more"])

    def test_wait_immediate_and_timeout(self) -> None:
        status, raw = self._request(
            "/v1/devices/d3/group-inbox/wait?limit=2&timeout_ms=0")
        self.assertEqual(status, 200)
        body = json.loads(raw)
        self.assertEqual([m["message_id"] for m in body["messages"]],
                         ["m1", "m2"])
        self.assertTrue(body["has_more"])
        status, raw = self._request(
            "/v1/devices/d4/group-inbox/wait?timeout_ms=0")
        body = json.loads(raw)
        self.assertEqual(status, 200)
        self.assertEqual(body["messages"], [])
        self.assertFalse(body["has_more"])

    def test_query_validation(self) -> None:
        for query, field in (
                ("limit=", "limit"), ("limit=abc", "limit"),
                ("limit=0", "limit"), ("limit=101", "limit"),
                ("limit=1&limit=2", "limit"), ("foo=1", "query"),
                ("limit=1&foo=2", "query")):
            status, raw = self._request(
                f"/v1/devices/d3/group-inbox?{query}")
            body = json.loads(raw)
            self.assertEqual(status, 400, query)
            self.assertEqual(body["field"], field, query)

    def test_wait_query_validation(self) -> None:
        for query, field in (
                ("timeout_ms=", "timeout_ms"),
                ("timeout_ms=abc", "timeout_ms"),
                ("timeout_ms=-1", "timeout_ms"),
                ("timeout_ms=30001", "timeout_ms"),
                ("timeout_ms=1&timeout_ms=2", "timeout_ms"),
                ("limit=0", "limit"), ("foo=1", "query")):
            status, raw = self._request(
                f"/v1/devices/d3/group-inbox/wait?{query}")
            body = json.loads(raw)
            self.assertEqual(status, 400, query)
            self.assertEqual(body["field"], field, query)

    def test_body_rejected(self) -> None:
        for path in ("/v1/devices/d3/group-inbox",
                     "/v1/devices/d3/group-inbox/wait"):
            status, raw = self._request(path, body=b"{}")
            body = json.loads(raw)
            self.assertEqual(status, 400, path)
            self.assertEqual(body["field"], "request_body", path)

    def test_bad_path_escape_and_unknown_device(self) -> None:
        status, raw = self._request("/v1/devices/d%zz/group-inbox")
        body = json.loads(raw)
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "device_id")
        status, raw = self._request(
            "/v1/devices/ghost/group-inbox/wait?timeout_ms=0")
        body = json.loads(raw)
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "device_id")

    def test_revoked_device_conflicts_over_http(self) -> None:
        self.fixture.store.revoke_device("d3")
        status, _raw = self._request("/v1/devices/d3/group-inbox")
        self.assertEqual(status, 409)

    def test_unchanged_state_is_byte_identical(self) -> None:
        first = self._request("/v1/devices/d3/group-inbox?limit=2")
        second = self._request("/v1/devices/d3/group-inbox?limit=2")
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
