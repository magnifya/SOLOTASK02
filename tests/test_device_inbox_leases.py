"""Tests for the paginated 1:1 inbox lease history endpoint.

GET /v1/devices/{device_id}/inbox/leases returns one page of the device's
lease history. It takes no request body (non-empty -> 400/request_body)
and only single-valued ``state`` (default ``all``; one of
all|active|expired|released|completed), ``after`` (default 0; unsigned
decimal) and ``limit`` (default 100; 1..100) query parameters; anything
else is 400/query and a malformed value is 400 with the parameter name.
An unknown device is 404/device_id; a revoked device's history stays
readable. Under the store lock the per-message lease copies are
deduplicated by lease_id, ordered by the initial claim leased_until and
then lease_id code points, filtered by current state, then paged by
after/limit with next_after/has_more. The query is purely read-only (no
write, no commit_seq change).
"""
import json
import threading
import unittest
from http.client import HTTPConnection

from e2ee_backend.models import Device, SignedPreKey
from e2ee_backend.service import DeviceService, ServiceError
from e2ee_backend.http_app import create_server


class HistoryMixin:
    def _build(self) -> None:
        self.service = DeviceService()
        self.service.store.add_device(Device("u", "alice", "ik"))
        self.service.store.add_device(Device(
            "u", "bob", "ik",
            prekeys=[SignedPreKey("pk1", "pubk1"),
                     SignedPreKey("pk2", "pubk2")]))
        self.service.store.add_device(
            Device("u", "carol", "ik",
                   prekeys=[SignedPreKey("pkC", "pubkC")]))
        self.sid1 = self.service.store.create_session(
            "alice", "bob", "pk1", "ek1").session_id
        self.sid2 = self.service.store.create_session(
            "carol", "bob", "pk2", "ek2").session_id
        for sequence in range(1, 4):
            self.service.post_message({
                "session_id": self.sid1, "sender_device_id": "alice",
                "message_id": f"a{sequence}", "sequence": sequence,
                "nonce": f"na{sequence}", "ciphertext": "ct"})
        for sequence in range(1, 3):
            self.service.post_message({
                "session_id": self.sid2, "sender_device_id": "carol",
                "message_id": f"b{sequence}", "sequence": sequence,
                "nonce": f"nb{sequence}", "ciphertext": "ct"})

    def _claim(self, lease_id="L1", limit=2):
        body, status = self.service.inbox_claim(
            "bob", {"lease_id": lease_id, "limit": limit})
        self.assertEqual(status, 201)
        return body

    def _expire(self, lease_id) -> None:
        past = "2000-01-01T00:00:00.000000+00:00"
        for state in self.service.store._delivery.values():
            for lease in state.leases:
                if lease.lease_id == lease_id:
                    lease.leased_until = past


class HistoryServiceTest(HistoryMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def test_empty_history_shape(self) -> None:
        body = self.service.inbox_leases("bob", "all", 0, 100)
        self.assertEqual(list(body),
                         ["device_id", "leases", "next_after", "has_more"])
        self.assertEqual(body, {"device_id": "bob", "leases": [],
                                "next_after": 0, "has_more": False})

    def test_active_leases_sorted_by_initial_deadline(self) -> None:
        first = self._claim("L1", 2)
        second = self._claim("L2", 3)
        body = self.service.inbox_leases("bob", "all", 0, 100)
        self.assertEqual([item["lease_id"] for item in body["leases"]],
                         ["L1", "L2"])
        for item in body["leases"]:
            self.assertEqual(list(item),
                             ["lease_id", "state", "leased_until",
                              "message_count"])
        self.assertEqual(body["leases"][0]["leased_until"],
                         first["leased_until"])
        self.assertEqual(body["leases"][1]["leased_until"],
                         second["leased_until"])
        self.assertEqual(body["leases"][0]["state"], "active")
        self.assertEqual(body["leases"][0]["message_count"], 2)
        self.assertEqual(body["leases"][1]["message_count"], 3)
        self.assertEqual(body["next_after"], 2)
        self.assertFalse(body["has_more"])

    def test_tiebreak_is_lease_id_codepoint_order(self) -> None:
        # Claims happen id-deadline order; rewrite equal deadlines so the
        # lease_id code-point tiebreak is the only ordering signal.
        for lease_id in ("Z", "a", "A", "m"):
            self._claim(lease_id, 1)
        tie = "2030-01-01T00:00:00.000000+00:00"
        for state in self.service.store._delivery.values():
            for lease in state.leases:
                lease.leased_until = tie
        body = self.service.inbox_leases("bob", "all", 0, 100)
        self.assertEqual([item["lease_id"] for item in body["leases"]],
                         ["A", "Z", "a", "m"])

    def test_state_filters_match_single_lease_query(self) -> None:
        # Only five messages exist, so each lease's terminal transition
        # frees its messages for the next claim (release, completion and
        # expiry all return still-unacked messages to the pool).
        self._claim("L1", 2)
        self.service.inbox_release("bob", "L1")
        self._claim("L2", 2)
        self.service.inbox_lease_complete(
            "bob", "L2", {"completion_id": "C1", "outcome": "failed"})
        self._claim("L3", 2)
        self._expire("L3")
        self._claim("L4", 2)
        for state_name, lease_id in (("released", "L1"),
                                     ("completed", "L2"),
                                     ("expired", "L3"),
                                     ("active", "L4")):
            with self.subTest(state=state_name):
                page = self.service.inbox_leases("bob", state_name, 0, 100)
                self.assertEqual(
                    [item["lease_id"] for item in page["leases"]],
                    [lease_id])
                single = self.service.inbox_lease_get("bob", lease_id)
                item = page["leases"][0]
                self.assertEqual(item["state"], single["state"])
                self.assertEqual(item["leased_until"],
                                 single["leased_until"])
        page = self.service.inbox_leases("bob", "all", 0, 100)
        self.assertEqual(len(page["leases"]), 4)

    def test_renewed_deadline_is_the_effective_one(self) -> None:
        claim = self._claim("L1", 1)
        renewed, status = self.service.inbox_lease_renew(
            "bob", "L1", {"renewal_id": "R1"})
        self.assertEqual(status, 201)
        body = self.service.inbox_leases("bob", "active", 0, 100)
        item = body["leases"][0]
        self.assertEqual(item["leased_until"], renewed["leased_until"])
        self.assertNotEqual(item["leased_until"], claim["leased_until"])

    def test_message_count_counts_claimed_messages(self) -> None:
        self._claim("L1", 2)
        # One of the claimed messages gets acked: it remains a lease item
        # and the count is unchanged.
        self.service.sync_session_ack(self.sid1, {
            "device_id": "bob", "cursor": 1})
        body = self.service.inbox_leases("bob", "all", 0, 100)
        self.assertEqual(body["leases"][0]["message_count"], 2)

    def test_other_devices_leases_never_contribute(self) -> None:
        # carol holds a lease on a session addressed to carol; bob's
        # history must not include it.
        sid = self.service.store.create_session(
            "alice", "carol", "pkC", "ekC").session_id
        self.service.post_message({
            "session_id": sid, "sender_device_id": "alice",
            "message_id": "c1", "sequence": 1,
            "nonce": "nc1", "ciphertext": "ct"})
        self.service.inbox_claim("carol", {"lease_id": "CX", "limit": 1})
        self._claim("L1", 1)
        body = self.service.inbox_leases("bob", "all", 0, 100)
        self.assertEqual([item["lease_id"] for item in body["leases"]],
                         ["L1"])
        body = self.service.inbox_leases("carol", "all", 0, 100)
        self.assertEqual([item["lease_id"] for item in body["leases"]],
                         ["CX"])

    def test_pagination_after_limit_next_after_and_has_more(self) -> None:
        for lease_id in ("L1", "L2", "L3", "L4"):
            self._claim(lease_id, 1)
        first = self.service.inbox_leases("bob", "all", 0, 2)
        self.assertEqual([i["lease_id"] for i in first["leases"]],
                         ["L1", "L2"])
        self.assertEqual(first["next_after"], 2)
        self.assertTrue(first["has_more"])
        second = self.service.inbox_leases("bob", "all", 2, 2)
        self.assertEqual([i["lease_id"] for i in second["leases"]],
                         ["L3", "L4"])
        self.assertEqual(second["next_after"], 4)
        self.assertFalse(second["has_more"])
        third = self.service.inbox_leases("bob", "all", 4, 2)
        self.assertEqual(third["leases"], [])
        # Empty page: next_after echoes after.
        self.assertEqual(third["next_after"], 4)

    def test_after_pages_the_filtered_list(self) -> None:
        for lease_id in ("L1", "L2", "L3"):
            self._claim(lease_id, 1)
        self.service.inbox_release("bob", "L2")
        page = self.service.inbox_leases("bob", "active", 1, 100)
        self.assertEqual([i["lease_id"] for i in page["leases"]], ["L3"])
        self.assertEqual(page["next_after"], 2)
        self.assertFalse(page["has_more"])

    def test_unknown_device_is_404_device_id(self) -> None:
        with self.assertRaises(ServiceError) as caught:
            self.service.inbox_leases("ghost", "all", 0, 100)
        self.assertEqual(caught.exception.status_code, 404)
        self.assertEqual(caught.exception.field, "device_id")

    def test_revoked_device_history_stays_readable(self) -> None:
        self._claim("L1", 2)
        self.service.revoke_device("bob")
        body = self.service.inbox_leases("bob", "all", 0, 100)
        self.assertEqual([i["lease_id"] for i in body["leases"]], ["L1"])

    def test_invalid_state_after_limit_at_service_layer(self) -> None:
        for state, after, limit, field in (
                ("bogus", 0, 100, "state"),
                ("", 0, 100, "state"),
                ("all", -1, 100, "after"),
                ("all", 1.0, 100, "after"),
                ("all", True, 100, "after"),
                ("all", 0, 0, "limit"),
                ("all", 0, 101, "limit"),
                ("all", 0, True, "limit")):
            with self.subTest(field=field):
                with self.assertRaises(ServiceError) as caught:
                    self.service.inbox_leases("bob", state, after, limit)
                self.assertEqual(caught.exception.field, field)
                self.assertEqual(caught.exception.status_code, 400)

    def test_query_is_read_only(self) -> None:
        self._claim("L1", 2)
        store = self.service.store
        with store._lock:
            before = {key: (value.attempts, value.acked,
                            [(lease.lease_id, lease.leased_until,
                              lease.released_at)
                             for lease in value.leases])
                      for key, value in store._delivery.items()}
        first = self.service.inbox_leases("bob", "all", 0, 100)
        second = self.service.inbox_leases("bob", "all", 0, 100)
        self.assertEqual(first, second)
        with store._lock:
            after = {key: (value.attempts, value.acked,
                           [(lease.lease_id, lease.leased_until,
                             lease.released_at)
                            for lease in value.leases])
                     for key, value in store._delivery.items()}
        self.assertEqual(before, after)


class HistoryHTTPTest(HistoryMixin, unittest.TestCase):
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

    def _request(self, path, raw=None, method="GET"):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        headers = {"Content-Type": "application/json"} if raw else {}
        conn.request(method, path, body=raw, headers=headers)
        response = conn.getresponse()
        data = response.read().decode("utf-8")
        conn.close()
        return response.status, (json.loads(data) if data else None), data

    def _claim_http(self, lease_id="L1", limit=2):
        status, _, _ = self._request(
            "/v1/devices/bob/inbox/claim",
            json.dumps({"lease_id": lease_id, "limit": limit}),
            method="POST")
        self.assertEqual(status, 201)

    def test_default_query_returns_200_with_key_order(self) -> None:
        self._claim_http("L1", 2)
        status, body, raw = self._request(
            "/v1/devices/bob/inbox/leases")
        self.assertEqual(status, 200)
        self.assertEqual(list(body),
                         ["device_id", "leases", "next_after", "has_more"])
        self.assertEqual(list(body["leases"][0]),
                         ["lease_id", "state", "leased_until",
                          "message_count"])
        self.assertLess(raw.index('"device_id"'), raw.index('"leases"'))
        self.assertLess(raw.index('"leases"'), raw.index('"next_after"'))
        self.assertLess(raw.index('"next_after"'), raw.index('"has_more"'))

    def test_repeated_gets_are_byte_identical(self) -> None:
        self._claim_http("L1", 2)
        _, _, first = self._request("/v1/devices/bob/inbox/leases")
        _, _, second = self._request("/v1/devices/bob/inbox/leases")
        self.assertEqual(first, second)

    def test_error_body_key_order(self) -> None:
        status, body, _ = self._request(
            "/v1/devices/bob/inbox/leases?bogus=1")
        self.assertEqual(status, 400)
        self.assertEqual(list(body), ["message", "field"])

    def test_unknown_query_parameter_is_400_query(self) -> None:
        for path in ("/v1/devices/bob/inbox/leases?x=1",
                     "/v1/devices/bob/inbox/leases?foo",
                     "/v1/devices/bob/inbox/leases?state=all&x="):
            with self.subTest(path=path):
                status, body, _ = self._request(path)
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], "query")

    def test_state_parameter_validation(self) -> None:
        for value in ("bogus", "", "ALL", "active%20"):
            with self.subTest(value=value):
                status, body, _ = self._request(
                    f"/v1/devices/bob/inbox/leases?state={value}")
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], "state")
        status, _, _ = self._request(
            "/v1/devices/bob/inbox/leases?state=active&state=expired")
        self.assertEqual(status, 400)
        for value in ("all", "active", "expired", "released", "completed"):
            status, _, _ = self._request(
                f"/v1/devices/bob/inbox/leases?state={value}")
            self.assertEqual(status, 200, value)

    def test_after_parameter_validation(self) -> None:
        for value in ("-1", "1.5", "%2B1", "x", "1a", "%201", "1%20",
                      ""):
            with self.subTest(value=value):
                status, body, _ = self._request(
                    f"/v1/devices/bob/inbox/leases?after={value}")
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], "after")
        status, body, _ = self._request(
            "/v1/devices/bob/inbox/leases?after=1&after=2")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "after")
        # A leading-zero decimal is still an unsigned decimal integer.
        status, _, _ = self._request(
            "/v1/devices/bob/inbox/leases?after=01")
        self.assertEqual(status, 200)

    def test_limit_parameter_validation(self) -> None:
        for value in ("0", "101", "-1", "x", "1.0", "%2B1", ""):
            with self.subTest(value=value):
                status, body, _ = self._request(
                    f"/v1/devices/bob/inbox/leases?limit={value}")
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], "limit")
        status, body, _ = self._request(
            "/v1/devices/bob/inbox/leases?limit=1&limit=2")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "limit")
        status, _, _ = self._request(
            "/v1/devices/bob/inbox/leases?limit=100")
        self.assertEqual(status, 200)

    def test_pagination_over_http(self) -> None:
        for lease_id in ("L1", "L2", "L3"):
            self._claim_http(lease_id, 1)
        status, body, _ = self._request(
            "/v1/devices/bob/inbox/leases?after=0&limit=2")
        self.assertEqual(status, 200)
        self.assertEqual([i["lease_id"] for i in body["leases"]],
                         ["L1", "L2"])
        self.assertEqual(body["next_after"], 2)
        self.assertTrue(body["has_more"])
        status, body, _ = self._request(
            "/v1/devices/bob/inbox/leases?after=2&limit=2")
        self.assertEqual([i["lease_id"] for i in body["leases"]], ["L3"])
        self.assertEqual(body["next_after"], 3)
        self.assertFalse(body["has_more"])

    def test_nonempty_body_is_400_request_body(self) -> None:
        status, body, _ = self._request(
            "/v1/devices/bob/inbox/leases", raw="{}")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "request_body")

    def test_empty_query_string_is_allowed(self) -> None:
        status, _, _ = self._request("/v1/devices/bob/inbox/leases?")
        self.assertEqual(status, 200)

    def test_unknown_device_is_404(self) -> None:
        status, body, _ = self._request(
            "/v1/devices/ghost/inbox/leases")
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "device_id")

    def test_leases_collection_distinct_from_single_lease_route(self) -> None:
        self._claim_http("L1", 1)
        status, body, _ = self._request(
            "/v1/devices/bob/inbox/leases/L1")
        self.assertEqual(status, 200)
        self.assertIn("messages", body)
        status, body, _ = self._request(
            "/v1/devices/bob/inbox/leases")
        self.assertEqual(status, 200)
        self.assertNotIn("messages", body)


if __name__ == "__main__":
    unittest.main()
