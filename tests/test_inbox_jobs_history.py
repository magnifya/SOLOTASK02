"""Tests for the paginated redelivery-job history endpoint.

GET /v1/devices/{device_id}/inbox-jobs returns one page of the device's
1:1-inbox redelivery job history. It takes no request body (non-empty ->
400/request_body) and only single-valued ``state`` (default ``all``; one
of all|pending|running|succeeded|failed|cancelled), ``after`` (default
0; unsigned decimal in 0..2^63-1) and ``limit`` (default 100; 1..100)
query parameters; anything else is 400/query and a malformed value is
400 with the parameter name. An unknown device is 404/device_id; a
revoked device's history stays readable. Under the store lock the jobs
are listed in first-successful-queue commit order, filtered by current
state, then paged by after/limit with next_after/has_more. The query is
purely read-only (no write, no commit_seq change).
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
        self.sid1 = self.service.store.create_session(
            "alice", "bob", "pk1", "ek1").session_id
        for sequence in range(1, 4):
            self.service.post_message({
                "session_id": self.sid1, "sender_device_id": "alice",
                "message_id": f"a{sequence}", "sequence": sequence,
                "nonce": f"na{sequence}", "ciphertext": "ct"})

    def _queue(self, job_id):
        body, status = self.service.inbox_job(
            {"device_id": "bob", "job_id": job_id, "op": "queue"})
        self.assertEqual(status, 201)
        return body

    def _job(self, job_id, op, **extra):
        payload = {"device_id": "bob", "job_id": job_id, "op": op}
        payload.update(extra)
        return self.service.inbox_job(payload)


class HistoryServiceTest(HistoryMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def test_empty_history_shape(self) -> None:
        body = self.service.inbox_jobs("bob", "all", 0, 100)
        self.assertEqual(list(body),
                         ["device_id", "jobs", "next_after", "has_more"])
        self.assertEqual(body, {"device_id": "bob", "jobs": [],
                                "next_after": 0, "has_more": False})

    def test_jobs_in_first_queue_order_with_item_key_order(self) -> None:
        for job_id in ("j2", "j1", "j3"):
            self._queue(job_id)
        body = self.service.inbox_jobs("bob", "all", 0, 100)
        self.assertEqual([item["job_id"] for item in body["jobs"]],
                         ["j2", "j1", "j3"])
        for item in body["jobs"]:
            self.assertEqual(list(item),
                             ["job_id", "state", "lease_id",
                              "cancellation_id", "cancelled_at"])
            self.assertEqual(item["state"], "pending")
            self.assertIsNone(item["lease_id"])
            self.assertIsNone(item["cancellation_id"])
            self.assertIsNone(item["cancelled_at"])
        self.assertEqual(body["next_after"], 3)
        self.assertFalse(body["has_more"])

    def test_state_filter_matches_current_state(self) -> None:
        self._queue("j1")
        self._queue("j2")
        self._queue("j3")
        self._queue("j4")
        body, status = self._job("j1", "dispatch")
        self.assertEqual((status, body["state"]), (201, "running"))
        body, status = self._job("j2", "cancel", cancellation_id="cx2")
        self.assertEqual((status, body["state"]), (201, "cancelled"))
        # j3 stays pending; dispatch j4 after acking everything so the
        # empty selection ends it succeeded.
        self.service.sync_session_ack(self.sid1, {
            "device_id": "bob", "cursor": 3})
        body, status = self._job("j4", "dispatch")
        self.assertEqual((status, body["state"]), (201, "succeeded"))
        for state_name, job_id in (("pending", "j3"),
                                   ("running", "j1"),
                                   ("cancelled", "j2"),
                                   ("succeeded", "j4")):
            with self.subTest(state=state_name):
                page = self.service.inbox_jobs("bob", state_name, 0, 100)
                self.assertEqual([item["job_id"] for item in page["jobs"]],
                                 [job_id])
        page = self.service.inbox_jobs("bob", "failed", 0, 100)
        self.assertEqual(page["jobs"], [])
        page = self.service.inbox_jobs("bob", "all", 0, 100)
        self.assertEqual([item["job_id"] for item in page["jobs"]],
                         ["j1", "j2", "j3", "j4"])

    def test_running_job_reports_lease_id(self) -> None:
        self._queue("j1")
        self._job("j1", "dispatch")
        item = self.service.inbox_jobs("bob", "all", 0, 100)["jobs"][0]
        self.assertEqual(item["state"], "running")
        self.assertEqual(item["lease_id"], "j1")

    def test_cancelled_job_reports_cancellation_fields(self) -> None:
        self._queue("j1")
        body, _ = self._job("j1", "cancel", cancellation_id="cx1")
        item = self.service.inbox_jobs("bob", "all", 0, 100)["jobs"][0]
        self.assertEqual(item["state"], "cancelled")
        self.assertIsNone(item["lease_id"])
        self.assertEqual(item["cancellation_id"], "cx1")
        self.assertIsInstance(item["cancelled_at"], str)
        self.assertTrue(item["cancelled_at"].endswith("+00:00"))

    def test_running_cancel_keeps_lease_id_and_releases_lease(self) -> None:
        self._queue("j1")
        self._job("j1", "dispatch")
        self._job("j1", "cancel", cancellation_id="cx1")
        item = self.service.inbox_jobs("bob", "all", 0, 100)["jobs"][0]
        self.assertEqual(item["state"], "cancelled")
        self.assertEqual(item["lease_id"], "j1")
        self.assertEqual(item["cancellation_id"], "cx1")

    def test_other_devices_jobs_never_contribute(self) -> None:
        self.service.store.add_device(
            Device("u", "carol", "ik",
                   prekeys=[SignedPreKey("pkC", "pubkC")]))
        self._queue("j1")
        body, status = self.service.inbox_job(
            {"device_id": "carol", "job_id": "jC", "op": "queue"})
        self.assertEqual(status, 201)
        page = self.service.inbox_jobs("bob", "all", 0, 100)
        self.assertEqual([item["job_id"] for item in page["jobs"]], ["j1"])
        page = self.service.inbox_jobs("carol", "all", 0, 100)
        self.assertEqual([item["job_id"] for item in page["jobs"]], ["jC"])

    def test_pagination_after_limit_next_after_and_has_more(self) -> None:
        for job_id in ("j1", "j2", "j3", "j4"):
            self._queue(job_id)
        first = self.service.inbox_jobs("bob", "all", 0, 2)
        self.assertEqual([i["job_id"] for i in first["jobs"]],
                         ["j1", "j2"])
        self.assertEqual(first["next_after"], 2)
        self.assertTrue(first["has_more"])
        second = self.service.inbox_jobs("bob", "all", 2, 2)
        self.assertEqual([i["job_id"] for i in second["jobs"]],
                         ["j3", "j4"])
        self.assertEqual(second["next_after"], 4)
        self.assertFalse(second["has_more"])
        third = self.service.inbox_jobs("bob", "all", 4, 2)
        self.assertEqual(third["jobs"], [])
        # Empty page: next_after echoes after.
        self.assertEqual(third["next_after"], 4)

    def test_after_pages_the_filtered_list(self) -> None:
        for job_id in ("j1", "j2", "j3"):
            self._queue(job_id)
        self._job("j2", "cancel", cancellation_id="cx2")
        page = self.service.inbox_jobs("bob", "pending", 1, 100)
        self.assertEqual([i["job_id"] for i in page["jobs"]], ["j3"])
        self.assertEqual(page["next_after"], 2)
        self.assertFalse(page["has_more"])

    def test_unknown_device_is_404_device_id(self) -> None:
        with self.assertRaises(ServiceError) as caught:
            self.service.inbox_jobs("ghost", "all", 0, 100)
        self.assertEqual(caught.exception.status_code, 404)
        self.assertEqual(caught.exception.field, "device_id")

    def test_revoked_device_history_stays_readable(self) -> None:
        self._queue("j1")
        self.service.revoke_device("bob")
        body = self.service.inbox_jobs("bob", "all", 0, 100)
        self.assertEqual([i["job_id"] for i in body["jobs"]], ["j1"])

    def test_invalid_state_after_limit_at_service_layer(self) -> None:
        for state, after, limit, field in (
                ("bogus", 0, 100, "state"),
                ("", 0, 100, "state"),
                ("active", 0, 100, "state"),
                ("all", -1, 100, "after"),
                ("all", 1.0, 100, "after"),
                ("all", True, 100, "after"),
                ("all", (1 << 63), 100, "after"),
                ("all", 0, 0, "limit"),
                ("all", 0, 101, "limit"),
                ("all", 0, True, "limit")):
            with self.subTest(field=field):
                with self.assertRaises(ServiceError) as caught:
                    self.service.inbox_jobs("bob", state, after, limit)
                self.assertEqual(caught.exception.field, field)
                self.assertEqual(caught.exception.status_code, 400)
        # The inclusive upper bound itself is accepted.
        body = self.service.inbox_jobs("bob", "all", (1 << 63) - 1, 100)
        self.assertEqual(body["jobs"], [])
        self.assertEqual(body["next_after"], (1 << 63) - 1)

    def test_query_is_read_only(self) -> None:
        self._queue("j1")
        self._job("j1", "dispatch")
        store = self.service.store
        with store._lock:
            before = {job_id: (job.state, job.lease_id,
                               job.cancellation_id, job.cancelled_at)
                      for job_id, job in store._redelivery_jobs.items()}
        first = self.service.inbox_jobs("bob", "all", 0, 100)
        second = self.service.inbox_jobs("bob", "all", 0, 100)
        self.assertEqual(first, second)
        with store._lock:
            after = {job_id: (job.state, job.lease_id,
                              job.cancellation_id, job.cancelled_at)
                     for job_id, job in store._redelivery_jobs.items()}
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

    def _queue_http(self, job_id):
        status, _, _ = self._request(
            "/v1/inbox-jobs",
            json.dumps({"device_id": "bob", "job_id": job_id,
                        "op": "queue"}),
            method="POST")
        self.assertEqual(status, 201)

    def test_default_query_returns_200_with_key_order(self) -> None:
        self._queue_http("j1")
        status, body, raw = self._request("/v1/devices/bob/inbox-jobs")
        self.assertEqual(status, 200)
        self.assertEqual(list(body),
                         ["device_id", "jobs", "next_after", "has_more"])
        self.assertEqual(list(body["jobs"][0]),
                         ["job_id", "state", "lease_id",
                          "cancellation_id", "cancelled_at"])
        self.assertLess(raw.index('"device_id"'), raw.index('"jobs"'))
        self.assertLess(raw.index('"jobs"'), raw.index('"next_after"'))
        self.assertLess(raw.index('"next_after"'), raw.index('"has_more"'))

    def test_repeated_gets_are_byte_identical(self) -> None:
        self._queue_http("j1")
        _, _, first = self._request("/v1/devices/bob/inbox-jobs")
        _, _, second = self._request("/v1/devices/bob/inbox-jobs")
        self.assertEqual(first, second)

    def test_error_body_key_order(self) -> None:
        status, body, _ = self._request(
            "/v1/devices/bob/inbox-jobs?bogus=1")
        self.assertEqual(status, 400)
        self.assertEqual(list(body), ["message", "field"])

    def test_unknown_query_parameter_is_400_query(self) -> None:
        for path in ("/v1/devices/bob/inbox-jobs?x=1",
                     "/v1/devices/bob/inbox-jobs?foo",
                     "/v1/devices/bob/inbox-jobs?state=all&x="):
            with self.subTest(path=path):
                status, body, _ = self._request(path)
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], "query")

    def test_state_parameter_validation(self) -> None:
        for value in ("bogus", "", "ALL", "pending%20", "active"):
            with self.subTest(value=value):
                status, body, _ = self._request(
                    f"/v1/devices/bob/inbox-jobs?state={value}")
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], "state")
        status, _, _ = self._request(
            "/v1/devices/bob/inbox-jobs?state=all&state=pending")
        self.assertEqual(status, 400)
        for value in ("all", "pending", "running", "succeeded",
                      "failed", "cancelled"):
            status, _, _ = self._request(
                f"/v1/devices/bob/inbox-jobs?state={value}")
            self.assertEqual(status, 200, value)

    def test_after_parameter_validation(self) -> None:
        for value in ("-1", "1.5", "%2B1", "x", "1a", "%201", "1%20",
                      "", "9223372036854775808",
                      "99999999999999999999999999"):
            with self.subTest(value=value):
                status, body, _ = self._request(
                    f"/v1/devices/bob/inbox-jobs?after={value}")
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], "after")
        status, body, _ = self._request(
            "/v1/devices/bob/inbox-jobs?after=1&after=2")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "after")
        # A leading-zero decimal is still an unsigned decimal integer.
        status, _, _ = self._request(
            "/v1/devices/bob/inbox-jobs?after=01")
        self.assertEqual(status, 200)
        # The inclusive upper bound 2^63-1 is accepted.
        status, body, _ = self._request(
            "/v1/devices/bob/inbox-jobs?after=9223372036854775807")
        self.assertEqual(status, 200)
        self.assertEqual(body["next_after"], 9223372036854775807)

    def test_limit_parameter_validation(self) -> None:
        for value in ("0", "101", "-1", "x", "1.0", "%2B1", "",
                      "99999999999999999999999999"):
            with self.subTest(value=value):
                status, body, _ = self._request(
                    f"/v1/devices/bob/inbox-jobs?limit={value}")
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], "limit")
        status, body, _ = self._request(
            "/v1/devices/bob/inbox-jobs?limit=1&limit=2")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "limit")
        status, _, _ = self._request(
            "/v1/devices/bob/inbox-jobs?limit=100")
        self.assertEqual(status, 200)

    def test_pagination_over_http(self) -> None:
        for job_id in ("j1", "j2", "j3"):
            self._queue_http(job_id)
        status, body, _ = self._request(
            "/v1/devices/bob/inbox-jobs?after=0&limit=2")
        self.assertEqual(status, 200)
        self.assertEqual([i["job_id"] for i in body["jobs"]],
                         ["j1", "j2"])
        self.assertEqual(body["next_after"], 2)
        self.assertTrue(body["has_more"])
        status, body, _ = self._request(
            "/v1/devices/bob/inbox-jobs?after=2&limit=2")
        self.assertEqual([i["job_id"] for i in body["jobs"]], ["j3"])
        self.assertEqual(body["next_after"], 3)
        self.assertFalse(body["has_more"])

    def test_nonempty_body_is_400_request_body(self) -> None:
        status, body, _ = self._request(
            "/v1/devices/bob/inbox-jobs", raw="{}")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "request_body")

    def test_empty_query_string_is_allowed(self) -> None:
        status, _, _ = self._request("/v1/devices/bob/inbox-jobs?")
        self.assertEqual(status, 200)

    def test_unknown_device_is_404(self) -> None:
        status, body, _ = self._request("/v1/devices/ghost/inbox-jobs")
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "device_id")

    def test_jobs_collection_distinct_from_inbox_route(self) -> None:
        self._queue_http("j1")
        status, body, _ = self._request("/v1/devices/bob/inbox-jobs")
        self.assertEqual(status, 200)
        self.assertIn("jobs", body)
        status, body, _ = self._request("/v1/devices/bob/inbox")
        self.assertEqual(status, 200)
        self.assertIn("messages", body)


if __name__ == "__main__":
    unittest.main()
