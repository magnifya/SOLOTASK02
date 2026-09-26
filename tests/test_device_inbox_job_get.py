"""Tests for the 1:1 inbox redelivery-job detail endpoint.

GET /v1/devices/{device_id}/inbox-jobs/{job_id} returns one job's current
state together with its recovery chain. It takes no request body
(non-empty -> 400/request_body) and no query parameters (any ->
400/query). Both path identifiers must be non-empty single segments,
strictly percent-decoded as UTF-8 (a bad escape or invalid encoding is
400 with the segment's field; a percent-encoded slash is part of the
identifier). An unknown device is 404/device_id (a revoked device's jobs
stay readable), a never-queued job is 404/job_id and a job of another
device is 409/job_id. The 200 body keys are job_id, device_id, state,
lease_id, recoveries, cancellation_id, cancelled_at; recoveries keep
commit order with recovery_id/lease_id items. The query is purely
read-only (no write, no commit_seq change) and linearized under the
store lock with the mutating job operations.
"""
import json
import threading
import unittest
from http.client import HTTPConnection

from e2ee_backend.models import Device, SignedPreKey
from e2ee_backend.service import DeviceService, ServiceError
from e2ee_backend.http_app import create_server

_DETAIL_KEYS = ["job_id", "device_id", "state", "lease_id", "recoveries",
                "cancellation_id", "cancelled_at"]


class DetailMixin:
    def _build(self) -> None:
        self.service = DeviceService()
        self.service.store.add_device(Device("u", "alice", "ik"))
        self.service.store.add_device(Device(
            "u", "bob", "ik",
            prekeys=[SignedPreKey("pk1", "pubk1")]))
        self.service.store.add_device(Device("u", "carol", "ik"))
        self.sid1 = self.service.store.create_session(
            "alice", "bob", "pk1", "ek1").session_id

    def _post_message(self, message_id="m1", sequence=1) -> None:
        self.service.post_message({
            "session_id": self.sid1, "sender_device_id": "alice",
            "message_id": message_id, "sequence": sequence,
            "nonce": f"n{message_id}", "ciphertext": "ct"})

    def _queue(self, job_id, device="bob") -> None:
        body, status = self.service.inbox_job(
            {"device_id": device, "job_id": job_id, "op": "queue"})
        self.assertEqual(status, 201)
        return body

    def _op(self, job_id, op, device="bob", **extra):
        payload = {"device_id": device, "job_id": job_id, "op": op}
        payload.update(extra)
        return self.service.inbox_job(payload)


class DetailServiceTest(DetailMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def test_pending_job_shape(self) -> None:
        self._queue("J1")
        body = self.service.inbox_job_get("bob", "J1")
        self.assertEqual(list(body), _DETAIL_KEYS)
        self.assertEqual(body, {
            "job_id": "J1", "device_id": "bob", "state": "pending",
            "lease_id": None, "recoveries": [],
            "cancellation_id": None, "cancelled_at": None})

    def test_running_job_carries_dispatch_lease(self) -> None:
        self._post_message()
        self._queue("J1")
        self._op("J1", "dispatch")
        body = self.service.inbox_job_get("bob", "J1")
        self.assertEqual(body["state"], "running")
        self.assertEqual(body["lease_id"], "J1")
        self.assertEqual(body["recoveries"], [])

    def test_recoveries_in_commit_order_with_item_key_order(self) -> None:
        self._post_message()
        self._post_message("m2", 2)
        self._queue("J1")
        self._op("J1", "dispatch")
        self.service.inbox_release("bob", "J1")
        body, status = self._op("J1", "recover", recovery_id="R1")
        self.assertEqual(status, 201)
        self.service.inbox_release("bob", "R1")
        body, status = self._op("J1", "recover", recovery_id="R2")
        self.assertEqual(status, 201)
        body = self.service.inbox_job_get("bob", "J1")
        self.assertEqual(body["state"], "running")
        self.assertEqual(body["lease_id"], "R2")
        self.assertEqual(len(body["recoveries"]), 2)
        for item in body["recoveries"]:
            self.assertEqual(list(item), ["recovery_id", "lease_id"])
        self.assertEqual(body["recoveries"][0],
                         {"recovery_id": "R1", "lease_id": "R1"})
        self.assertEqual(body["recoveries"][1],
                         {"recovery_id": "R2", "lease_id": "R2"})

    def test_empty_recovery_freezes_null_lease(self) -> None:
        # A running job whose messages are all acked before the recovery
        # finds an empty selection: it ends succeeded and its recovery
        # record freezes lease_id null.
        self._post_message()
        self._queue("J3")
        self._op("J3", "dispatch")
        self.service.inbox_release("bob", "J3")
        self.service.sync_session_ack(self.sid1, {
            "device_id": "bob", "cursor": 1})
        body, status = self._op("J3", "recover", recovery_id="R9")
        self.assertEqual(status, 201)
        self.assertEqual(body["state"], "succeeded")
        body = self.service.inbox_job_get("bob", "J3")
        self.assertEqual(body["state"], "succeeded")
        self.assertIsNone(body["lease_id"])
        self.assertEqual(body["recoveries"],
                         [{"recovery_id": "R9", "lease_id": None}])

    def test_cancelled_job_fields(self) -> None:
        self._queue("J1")
        self._op("J1", "cancel", cancellation_id="X1")
        body = self.service.inbox_job_get("bob", "J1")
        self.assertEqual(body["state"], "cancelled")
        self.assertIsNone(body["lease_id"])
        self.assertEqual(body["cancellation_id"], "X1")
        self.assertIsInstance(body["cancelled_at"], str)
        self.assertTrue(body["cancelled_at"].endswith("+00:00"))
        # Six microsecond digits ahead of the UTC offset.
        fraction = body["cancelled_at"].split(".", 1)[1]
        self.assertEqual(fraction[:6].isdigit(), True)
        self.assertEqual(fraction[6:], "+00:00")

    def test_running_cancel_keeps_lease_id(self) -> None:
        self._post_message()
        self._queue("J1")
        self._op("J1", "dispatch")
        self._op("J1", "cancel", cancellation_id="X1")
        body = self.service.inbox_job_get("bob", "J1")
        self.assertEqual(body["state"], "cancelled")
        self.assertEqual(body["lease_id"], "J1")
        self.assertEqual(body["cancellation_id"], "X1")

    def test_unknown_device_is_404_device_id(self) -> None:
        with self.assertRaises(ServiceError) as caught:
            self.service.inbox_job_get("ghost", "J1")
        self.assertEqual(caught.exception.status_code, 404)
        self.assertEqual(caught.exception.field, "device_id")

    def test_unknown_device_checked_before_unknown_job(self) -> None:
        self._queue("J1")
        with self.assertRaises(ServiceError) as caught:
            self.service.inbox_job_get("ghost", "J1")
        self.assertEqual(caught.exception.status_code, 404)
        self.assertEqual(caught.exception.field, "device_id")

    def test_revoked_device_jobs_stay_readable(self) -> None:
        self._queue("J1")
        self.service.revoke_device("bob")
        body = self.service.inbox_job_get("bob", "J1")
        self.assertEqual(body["job_id"], "J1")
        self.assertEqual(body["state"], "pending")

    def test_unknown_job_is_404_job_id(self) -> None:
        with self.assertRaises(ServiceError) as caught:
            self.service.inbox_job_get("bob", "ghost")
        self.assertEqual(caught.exception.status_code, 404)
        self.assertEqual(caught.exception.field, "job_id")

    def test_cross_device_job_is_409_job_id(self) -> None:
        self._queue("J1", "carol")
        with self.assertRaises(ServiceError) as caught:
            self.service.inbox_job_get("bob", "J1")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "job_id")

    def test_query_is_read_only(self) -> None:
        self._post_message()
        self._queue("J1")
        self._op("J1", "dispatch")
        store = self.service.store
        with store._lock:
            before = {job_id: (job.state, job.lease_id,
                               list(job.recoveries),
                               job.cancellation_id, job.cancelled_at)
                      for job_id, job in store._redelivery_jobs.items()}
        first = self.service.inbox_job_get("bob", "J1")
        second = self.service.inbox_job_get("bob", "J1")
        self.assertEqual(first, second)
        with store._lock:
            after = {job_id: (job.state, job.lease_id,
                              list(job.recoveries),
                              job.cancellation_id, job.cancelled_at)
                     for job_id, job in store._redelivery_jobs.items()}
        self.assertEqual(before, after)


class DetailHTTPTest(DetailMixin, unittest.TestCase):
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

    def _queue_http(self, job_id, device="bob"):
        status, _, _ = self._request(
            "/v1/inbox-jobs",
            json.dumps({"device_id": device, "job_id": job_id,
                        "op": "queue"}),
            method="POST")
        self.assertEqual(status, 201)

    def test_get_pending_job_over_http(self) -> None:
        self._queue_http("J1")
        status, body, raw = self._request("/v1/devices/bob/inbox-jobs/J1")
        self.assertEqual(status, 200)
        self.assertEqual(list(body), _DETAIL_KEYS)
        self.assertEqual(body["job_id"], "J1")
        self.assertEqual(body["device_id"], "bob")
        self.assertEqual(body["state"], "pending")
        self.assertIsNone(body["lease_id"])
        self.assertEqual(body["recoveries"], [])
        self.assertIsNone(body["cancellation_id"])
        self.assertIsNone(body["cancelled_at"])
        for left, right in zip(_DETAIL_KEYS, _DETAIL_KEYS[1:]):
            self.assertLess(raw.index(f'"{left}"'), raw.index(f'"{right}"'))

    def test_recovery_chain_over_http(self) -> None:
        self._post_message()
        self._queue_http("J1")
        status, _, _ = self._request(
            "/v1/inbox-jobs",
            json.dumps({"device_id": "bob", "job_id": "J1",
                        "op": "dispatch"}),
            method="POST")
        self.assertEqual(status, 201)
        status, _, _ = self._request(
            "/v1/devices/bob/inbox/leases/J1/release", method="POST")
        self.assertEqual(status, 201)
        status, _, _ = self._request(
            "/v1/inbox-jobs",
            json.dumps({"device_id": "bob", "job_id": "J1",
                        "op": "recover", "recovery_id": "R1"}),
            method="POST")
        self.assertEqual(status, 201)
        status, body, _ = self._request("/v1/devices/bob/inbox-jobs/J1")
        self.assertEqual(status, 200)
        self.assertEqual(body["state"], "running")
        self.assertEqual(body["lease_id"], "R1")
        self.assertEqual(body["recoveries"],
                         [{"recovery_id": "R1", "lease_id": "R1"}])
        self.assertEqual(list(body["recoveries"][0]),
                         ["recovery_id", "lease_id"])

    def test_repeated_gets_are_byte_identical(self) -> None:
        self._queue_http("J1")
        _, _, first = self._request("/v1/devices/bob/inbox-jobs/J1")
        _, _, second = self._request("/v1/devices/bob/inbox-jobs/J1")
        self.assertEqual(first, second)

    def test_error_body_key_order(self) -> None:
        status, body, _ = self._request(
            "/v1/devices/bob/inbox-jobs/J1?bogus=1")
        self.assertEqual(status, 400)
        self.assertEqual(list(body), ["message", "field"])

    def test_query_parameter_is_400_query(self) -> None:
        for path in ("/v1/devices/bob/inbox-jobs/J1?x=1",
                     "/v1/devices/bob/inbox-jobs/J1?foo",
                     "/v1/devices/bob/inbox-jobs/J1?state=all"):
            with self.subTest(path=path):
                status, body, _ = self._request(path)
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], "query")

    def test_empty_query_string_is_allowed(self) -> None:
        self._queue_http("J1")
        status, _, _ = self._request("/v1/devices/bob/inbox-jobs/J1?")
        self.assertEqual(status, 200)

    def test_nonempty_body_is_400_request_body(self) -> None:
        status, body, _ = self._request(
            "/v1/devices/bob/inbox-jobs/J1", raw="{}")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "request_body")

    def test_bad_percent_escape_in_device_segment(self) -> None:
        for segment in ("%", "%2", "%zz", "a%2"):
            with self.subTest(segment=segment):
                status, body, _ = self._request(
                    f"/v1/devices/{segment}/inbox-jobs/J1")
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], "device_id")

    def test_bad_percent_escape_in_job_segment(self) -> None:
        for segment in ("%", "%2", "%zz", "a%2"):
            with self.subTest(segment=segment):
                status, body, _ = self._request(
                    f"/v1/devices/bob/inbox-jobs/{segment}")
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], "job_id")

    def test_invalid_utf8_escape_is_400(self) -> None:
        status, body, _ = self._request(
            "/v1/devices/bob/inbox-jobs/%FF")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "job_id")
        status, body, _ = self._request(
            "/v1/devices/%FF/inbox-jobs/J1")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "device_id")

    def test_percent_encoded_slash_is_part_of_the_id(self) -> None:
        self._queue_http("a/b")
        status, body, _ = self._request(
            "/v1/devices/bob/inbox-jobs/a%2Fb")
        self.assertEqual(status, 200)
        self.assertEqual(body["job_id"], "a/b")

    def test_percent_encoded_utf8_id(self) -> None:
        self._queue_http("é")
        status, body, _ = self._request(
            "/v1/devices/bob/inbox-jobs/%C3%A9")
        self.assertEqual(status, 200)
        self.assertEqual(body["job_id"], "é")

    def test_unknown_device_is_404(self) -> None:
        status, body, _ = self._request(
            "/v1/devices/ghost/inbox-jobs/J1")
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "device_id")

    def test_unknown_job_is_404(self) -> None:
        status, body, _ = self._request(
            "/v1/devices/bob/inbox-jobs/ghost")
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "job_id")

    def test_cross_device_job_is_409(self) -> None:
        self._queue_http("J1", "carol")
        status, body, _ = self._request("/v1/devices/bob/inbox-jobs/J1")
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "job_id")

    def test_revoked_device_job_stays_readable_over_http(self) -> None:
        self._queue_http("J1")
        status, _, _ = self._request("/v1/devices/bob/revoke",
                                     method="POST")
        self.assertEqual(status, 200)
        status, body, _ = self._request("/v1/devices/bob/inbox-jobs/J1")
        self.assertEqual(status, 200)
        self.assertEqual(body["job_id"], "J1")

    def test_sub_paths_are_not_the_detail_route(self) -> None:
        for path in ("/v1/devices/bob/inbox-jobs/",
                     "/v1/devices/bob/inbox-jobs/J1/extra",
                     "/v1/devices//inbox-jobs/J1"):
            with self.subTest(path=path):
                status, _, _ = self._request(path)
                self.assertEqual(status, 404)


if __name__ == "__main__":
    unittest.main()
