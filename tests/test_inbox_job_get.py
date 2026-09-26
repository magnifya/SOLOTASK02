"""Tests for the 1:1 inbox redelivery-job detail endpoint.

GET /v1/devices/{device_id}/inbox-jobs/{job_id} returns one job's detail
with its recovery chain. It takes no request body (non-empty ->
400/request_body) and no query parameters (any -> 400/query); a trailing
empty ``?`` is accepted. Both path identifiers must be non-empty single
segments, strictly percent-decoded as UTF-8 — a bad escape or an invalid
encoding is 400 with the offending segment (device_id decoded first, then
job_id), and an encoded slash (``%2F``) stays part of the identifier. An
unknown device is 404/device_id (a revoked device stays readable); an
unknown job is 404/job_id; a job owned by another device is 409/job_id.
The 200 body keys are job_id, device_id, state, lease_id, recoveries,
cancellation_id, cancelled_at; recoveries keep commit order with each
item recovery_id then lease_id (string or null). The query is purely
read-only (no write, no commit_seq change).
"""
import json
import threading
import unittest
from http.client import HTTPConnection

from e2ee_backend.models import Device, SignedPreKey
from e2ee_backend.service import DeviceService, ServiceError
from e2ee_backend.http_app import create_server


_PAST = "2000-01-01T00:00:00.000000+00:00"


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

    def _job(self, job_id, op="queue", device="bob", **extra):
        payload = {"device_id": device, "job_id": job_id, "op": op}
        payload.update(extra)
        return self.service.inbox_job(payload)

    def _queue(self, job_id, device="bob"):
        body, status = self._job(job_id, device=device)
        self.assertEqual(status, 201)
        return body

    def _dispatch(self, job_id):
        self._queue(job_id)
        body, status = self._job(job_id, "dispatch")
        self.assertEqual(status, 201)
        return body

    def _expire(self, lease_id) -> None:
        with self.service.store._lock:
            for state in self.service.store._delivery.values():
                for lease in state.leases:
                    if lease.lease_id == lease_id:
                        lease.leased_until = _PAST

    def _recover(self, job_id, recovery_id):
        return self._job(job_id, "recover", recovery_id=recovery_id)


class DetailServiceTest(DetailMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def test_pending_detail_shape_and_key_order(self) -> None:
        self._queue("J1")
        body = self.service.inbox_job_get("bob", "J1")
        self.assertEqual(list(body),
                         ["job_id", "device_id", "state", "lease_id",
                          "recoveries", "cancellation_id", "cancelled_at"])
        self.assertEqual(body["job_id"], "J1")
        self.assertEqual(body["device_id"], "bob")
        self.assertEqual(body["state"], "pending")
        self.assertIsNone(body["lease_id"])
        self.assertEqual(body["recoveries"], [])
        self.assertIsNone(body["cancellation_id"])
        self.assertIsNone(body["cancelled_at"])

    def test_running_detail_names_dispatch_lease(self) -> None:
        self._post_message()
        self._dispatch("J1")
        body = self.service.inbox_job_get("bob", "J1")
        self.assertEqual(body["state"], "running")
        self.assertEqual(body["lease_id"], "J1")

    def test_succeeded_empty_dispatch_keeps_null_lease(self) -> None:
        self._dispatch("J1")
        body = self.service.inbox_job_get("bob", "J1")
        self.assertEqual(body["state"], "succeeded")
        self.assertIsNone(body["lease_id"])

    def test_failed_detail_after_failed_completion(self) -> None:
        self._post_message()
        self._dispatch("J1")
        self.service.inbox_lease_complete(
            "bob", "J1", {"completion_id": "C1", "outcome": "failed"})
        body = self.service.inbox_job_get("bob", "J1")
        self.assertEqual(body["state"], "failed")
        self.assertEqual(body["lease_id"], "J1")

    def test_recoveries_listed_in_commit_order(self) -> None:
        self._post_message()
        self._dispatch("J1")
        # First recovery leases the one message under an ordinary lease.
        self._expire("J1")
        body, status = self._recover("J1", "R1")
        self.assertEqual(status, 201)
        # Acknowledge the message (without completing the lease) so a
        # later recovery finds nothing left to redeliver.
        self.service.ack_message(self.sid1, {
            "device_id": "bob", "message_id": "m1", "sequence": 1})
        self._expire("R1")
        body, status = self._recover("J1", "R2")
        self.assertEqual(status, 201)
        self.assertEqual(body["state"], "succeeded")

        detail = self.service.inbox_job_get("bob", "J1")
        self.assertEqual(detail["state"], "succeeded")
        self.assertIsNone(detail["lease_id"])
        recoveries = detail["recoveries"]
        self.assertEqual(len(recoveries), 2)
        self.assertEqual([list(item) for item in recoveries],
                         [["recovery_id", "lease_id"],
                          ["recovery_id", "lease_id"]])
        self.assertEqual(recoveries[0],
                         {"recovery_id": "R1", "lease_id": "R1"})
        self.assertEqual(recoveries[1],
                         {"recovery_id": "R2", "lease_id": None})

    def test_cancelled_pending_carries_cancellation_fields(self) -> None:
        self._queue("J1")
        body, status = self._job(
            "J1", "cancel", cancellation_id="X1")
        self.assertEqual(status, 201)
        detail = self.service.inbox_job_get("bob", "J1")
        self.assertEqual(detail["state"], "cancelled")
        self.assertIsNone(detail["lease_id"])
        self.assertEqual(detail["cancellation_id"], "X1")
        cancelled_at = detail["cancelled_at"]
        self.assertIsInstance(cancelled_at, str)
        self.assertTrue(cancelled_at.endswith("+00:00"))
        fractional = cancelled_at[:-len("+00:00")].split(".", 1)[1]
        self.assertEqual(len(fractional), 6)  # six microsecond digits

    def test_cancelled_running_keeps_current_lease_id(self) -> None:
        self._post_message()
        self._dispatch("J1")
        self._job("J1", "cancel", cancellation_id="X1")
        detail = self.service.inbox_job_get("bob", "J1")
        self.assertEqual(detail["state"], "cancelled")
        self.assertEqual(detail["lease_id"], "J1")
        self.assertEqual(detail["cancellation_id"], "X1")

    def test_unknown_device_is_404_device_id(self) -> None:
        with self.assertRaises(ServiceError) as caught:
            self.service.inbox_job_get("ghost", "J1")
        self.assertEqual((caught.exception.status_code,
                          caught.exception.field), (404, "device_id"))

    def test_unknown_job_is_404_job_id(self) -> None:
        with self.assertRaises(ServiceError) as caught:
            self.service.inbox_job_get("bob", "ghost")
        self.assertEqual((caught.exception.status_code,
                          caught.exception.field), (404, "job_id"))

    def test_other_devices_job_is_409_job_id(self) -> None:
        self._queue("J1")
        with self.assertRaises(ServiceError) as caught:
            self.service.inbox_job_get("carol", "J1")
        self.assertEqual((caught.exception.status_code,
                          caught.exception.field), (409, "job_id"))

    def test_revoked_device_jobs_stay_readable(self) -> None:
        self._queue("J1")
        self.service.revoke_device("bob")
        body = self.service.inbox_job_get("bob", "J1")
        self.assertEqual(body["state"], "pending")

    def test_identifier_with_slash_is_distinct(self) -> None:
        self._queue("a/b")
        body = self.service.inbox_job_get("bob", "a/b")
        self.assertEqual(body["job_id"], "a/b")

    def test_query_is_read_only(self) -> None:
        self._queue("J1")
        store = self.service.store
        with store._lock:
            before = {
                job_id: (job.state, job.lease_id,
                         [(r.recovery_id, r.lease_id) for r in job.recoveries],
                         job.cancellation_id, job.cancelled_at)
                for job_id, job in store._redelivery_jobs.items()}
        first = self.service.inbox_job_get("bob", "J1")
        second = self.service.inbox_job_get("bob", "J1")
        self.assertEqual(first, second)
        with store._lock:
            after = {
                job_id: (job.state, job.lease_id,
                         [(r.recovery_id, r.lease_id) for r in job.recoveries],
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

    def _queue_http(self, job_id):
        status, _, _ = self._request(
            "/v1/inbox-jobs",
            json.dumps({"device_id": "bob", "job_id": job_id,
                        "op": "queue"}),
            method="POST")
        self.assertEqual(status, 201)

    def test_pending_detail_200_with_key_order(self) -> None:
        self._queue_http("J1")
        status, body, raw = self._request(
            "/v1/devices/bob/inbox-jobs/J1")
        self.assertEqual(status, 200)
        self.assertEqual(list(body),
                         ["job_id", "device_id", "state", "lease_id",
                          "recoveries", "cancellation_id", "cancelled_at"])
        # Key order is the serialization order too.
        names = ['"job_id"', '"device_id"', '"state"', '"lease_id"',
                 '"recoveries"', '"cancellation_id"', '"cancelled_at"']
        positions = [raw.index(name) for name in names]
        self.assertEqual(positions, sorted(positions))

    def test_repeated_gets_are_byte_identical(self) -> None:
        self._queue_http("J1")
        _, _, first = self._request("/v1/devices/bob/inbox-jobs/J1")
        _, _, second = self._request("/v1/devices/bob/inbox-jobs/J1")
        self.assertEqual(first, second)

    def test_recoveries_chain_over_http(self) -> None:
        self._post_message()
        self._dispatch("J1")
        self._expire("J1")
        self._recover("J1", "R1")
        status, body, _ = self._request(
            "/v1/devices/bob/inbox-jobs/J1")
        self.assertEqual(status, 200)
        self.assertEqual(body["state"], "running")
        self.assertEqual(body["lease_id"], "R1")
        self.assertEqual(body["recoveries"],
                         [{"recovery_id": "R1", "lease_id": "R1"}])

    def test_cancelled_fields_over_http(self) -> None:
        self._queue_http("J1")
        status, _, _ = self._request(
            "/v1/inbox-jobs",
            json.dumps({"device_id": "bob", "job_id": "J1", "op": "cancel",
                        "cancellation_id": "X1"}),
            method="POST")
        self.assertEqual(status, 201)
        status, body, _ = self._request(
            "/v1/devices/bob/inbox-jobs/J1")
        self.assertEqual(status, 200)
        self.assertEqual(body["state"], "cancelled")
        self.assertIsNone(body["lease_id"])
        self.assertEqual(body["cancellation_id"], "X1")
        self.assertTrue(body["cancelled_at"].endswith("+00:00"))

    def test_errors_over_http(self) -> None:
        self._queue_http("J1")
        for path, field, code in (
                ("/v1/devices/ghost/inbox-jobs/J1", "device_id", 404),
                ("/v1/devices/bob/inbox-jobs/NOPE", "job_id", 404),
                ("/v1/devices/carol/inbox-jobs/J1", "job_id", 409)):
            with self.subTest(path=path):
                status, body, _ = self._request(path)
                self.assertEqual(status, code)
                self.assertEqual(body["field"], field)
                self.assertEqual(list(body), ["message", "field"])
                self.assertIsInstance(body["message"], str)

    def test_revoked_device_still_readable_over_http(self) -> None:
        self._queue_http("J1")
        self.service.revoke_device("bob")
        status, _, _ = self._request("/v1/devices/bob/inbox-jobs/J1")
        self.assertEqual(status, 200)

    def test_nonempty_body_is_400_request_body(self) -> None:
        self._queue_http("J1")
        status, body, _ = self._request(
            "/v1/devices/bob/inbox-jobs/J1", raw="{}")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "request_body")

    def test_any_query_parameter_is_400_query(self) -> None:
        self._queue_http("J1")
        for path in ("/v1/devices/bob/inbox-jobs/J1?x=1",
                     "/v1/devices/bob/inbox-jobs/J1?foo",
                     "/v1/devices/bob/inbox-jobs/J1?state=pending"):
            with self.subTest(path=path):
                status, body, _ = self._request(path)
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], "query")
        # A trailing empty query string carries no parameter.
        status, _, _ = self._request("/v1/devices/bob/inbox-jobs/J1?")
        self.assertEqual(status, 200)

    def test_encoded_slash_is_part_of_the_identifier(self) -> None:
        self._queue_http("a/b")
        # A raw slash splits into segments and does not match; the encoded
        # slash decodes to a slash inside the job id.
        status, body, _ = self._request(
            "/v1/devices/bob/inbox-jobs/a%2Fb")
        self.assertEqual(status, 200)
        self.assertEqual(body["job_id"], "a/b")
        status, body, _ = self._request(
            "/v1/devices/bob/inbox-jobs/a/b")
        self.assertEqual(status, 404)

    def test_encoded_slash_in_device_id(self) -> None:
        # Registering a device with a slash in its id is allowed through
        # the service; the HTTP path must reach it via %2F.
        self.service.store.add_device(Device("u", "d/v", "ik"))
        status, body, _ = self._request(
            "/v1/devices/d%2Fv/inbox-jobs/J1")
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "job_id")

    def test_bad_escape_and_encoding_400_by_segment(self) -> None:
        # A malformed device segment reports device_id ...
        for path in ("/v1/devices/bob%2/inbox-jobs/J1",
                     "/v1/devices/bob%zz/inbox-jobs/J1",
                     "/v1/devices/bob%FF/inbox-jobs/J1",
                     "/v1/devices/bob%e2%82/inbox-jobs/J1"):
            with self.subTest(path=path):
                status, body, _ = self._request(path)
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], "device_id")
        # ... and a well-formed device with a bad job segment reports
        # job_id (device is decoded first).
        for path in ("/v1/devices/bob/inbox-jobs/J1%2",
                     "/v1/devices/bob/inbox-jobs/J1%zz",
                     "/v1/devices/bob/inbox-jobs/J1%FF",
                     "/v1/devices/bob/inbox-jobs/J1%c0%80"):
            with self.subTest(path=path):
                status, body, _ = self._request(path)
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], "job_id")

    def test_segment_error_precedes_query_and_body_checks(self) -> None:
        status, body, _ = self._request(
            "/v1/devices/bob%FF/inbox-jobs/J1?x=1")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "device_id")

    def test_empty_segment_is_not_the_route(self) -> None:
        for path in ("/v1/devices//inbox-jobs/J1",
                     "/v1/devices/bob/inbox-jobs/"):
            with self.subTest(path=path):
                status, _, _ = self._request(path)
                self.assertEqual(status, 404)

    def test_unicode_identifier_round_trips_utf8(self) -> None:
        # "job-€" percent-encoded as its UTF-8 bytes e2 82 ac.
        self._queue_http("job-é")
        status, body, _ = self._request(
            "/v1/devices/bob/inbox-jobs/job-%C3%A9")
        self.assertEqual(status, 200)
        self.assertEqual(body["job_id"], "job-é")


if __name__ == "__main__":
    unittest.main()
