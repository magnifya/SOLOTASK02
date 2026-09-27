"""Tests for the single batch-cleanup audit lease lookup.

``GET /v1/event-gc-batch/leases/{lease_id}`` returns one committed
cleanup audit claim lease. The path identifier is a single non-empty
segment strictly percent-decoded as UTF-8 (an empty segment, a bad
escape or invalid UTF-8 is 400/lease_id; a deeper path or an
uncommitted id is 404/lease_id). The GET takes no query parameters
(any -> 400/query) and no request body (a non-empty one, however
framed, is 400/request_body), validated in that order.

The body keys are ``lease_id``, ``consumer_id``, ``expected``,
``next_after``, ``limit``, ``expires``, ``renewals``, ``terminal``,
``checkpoint``, ``effective_expires`` and ``state`` in that order:
the integers are non-negative, ``renewals`` keeps the commit order
with each item keyed ``renewal_id`` then ``expires``, ``terminal`` is
null/``confirm``/``release``, ``checkpoint`` is the consumer's current
checkpoint (0 when the consumer never advanced) and the timestamps are
UTC strings with six microsecond digits and ``+00:00``. The state is
decided at one query instant in the order released (terminal ==
release), confirmed (terminal == confirm or the checkpoint reached
next_after), expired (effective deadline at or before now), otherwise
active. The lookup is read-only: no persistence, no ``commit_seq``
advance, byte-identical while state is unchanged and stable across a
restart.
"""
import json
import os
import shutil
import tempfile
import threading
import unittest
from http.client import HTTPConnection

from e2ee_backend.models import Device
from e2ee_backend.service import DeviceService, ServiceError
from e2ee_backend.http_app import create_server
from e2ee_backend.persistence import attach_persistence

PATH = "/v1/event-gc-batch/leases"
PAST = "2020-01-01T00:00:00.000000+00:00"


class LeaseGetMixin:
    def _build(self) -> None:
        self.service = DeviceService()
        self.service.store.add_device(Device("u", "alice", "ik"))
        self.service.store.add_device(Device("u", "bob", "ik"))

    def _commit(self, request_id):
        # Each idempotent commit appends exactly one audit record.
        return self.service.event_gc_cleanup_expired_batch({
            "mode": "commit", "device_ids": ["bob"],
            "after": 0, "limit": 100, "request_id": request_id})

    def _claim(self, consumer, lease_id, expected=0, limit=1):
        return self.service.event_gc_batch_claim({
            "consumer_id": consumer, "lease_id": lease_id,
            "expected": expected, "limit": limit})

    def _op(self, consumer, lease_id, op, expected=0):
        return self.service.event_gc_batch_lease_op({
            "consumer_id": consumer, "lease_id": lease_id,
            "expected": expected, "op": op})

    def _renew(self, consumer, lease_id, renewal_id):
        return self.service.event_gc_batch_lease_renew({
            "consumer_id": consumer, "lease_id": lease_id,
            "renewal_id": renewal_id})

    def _advance(self, consumer, expected, after):
        return self.service.event_gc_batch_checkpoint({
            "consumer_id": consumer, "expected": expected,
            "after": after})

    def _get(self, lease_id):
        return self.service.event_gc_batch_lease_get(lease_id)

    def _seed(self):
        # Three audit records; five leases committed in the order
        # L1..L5 across three consumers:
        #   L1 (c1) released, L2 (c2) active with two renewals,
        #   L3 (c1) expired, L4 (c3) confirmed via explicit confirm,
        #   L5 (c1) confirmed implicitly once c1's checkpoint reaches
        #   its next_after (no terminal marker).
        self._commit("r1")
        self._commit("r2")
        self._commit("r3")
        self._claim("c1", "L1")
        self._op("c1", "L1", "release")
        self._claim("c2", "L2", limit=2)
        self._renew("c2", "L2", "renew-1")
        self._renew("c2", "L2", "renew-2")
        self._claim("c1", "L3", limit=100)
        self.service.store._cleanup_leases["L3"].expires = PAST
        self._claim("c3", "L4")
        self._op("c3", "L4", "confirm")
        self._claim("c1", "L5")
        # Moving c1's checkpoint to 1 implicitly acknowledges L5 (and
        # L1, whose release must still win the state classification).
        self._advance("c1", 0, 1)


class LeaseGetServiceTest(LeaseGetMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def test_key_order_and_frozen_values(self) -> None:
        self._seed()
        body = self._get("L2")
        self.assertEqual(list(body), [
            "lease_id", "consumer_id", "expected", "next_after",
            "limit", "expires", "renewals", "terminal", "checkpoint",
            "effective_expires", "state"])
        self.assertEqual(body["lease_id"], "L2")
        self.assertEqual(body["consumer_id"], "c2")
        self.assertEqual(body["expected"], 0)
        self.assertEqual(body["next_after"], 2)
        self.assertEqual(body["limit"], 2)
        self.assertIsNone(body["terminal"])
        self.assertEqual(body["checkpoint"], 0)  # c2 never advanced
        self.assertEqual(body["state"], "active")
        self.assertEqual([r["renewal_id"] for r in body["renewals"]],
                         ["renew-1", "renew-2"])
        for renewal in body["renewals"]:
            self.assertEqual(list(renewal), ["renewal_id", "expires"])
        lease = self.service.store._cleanup_leases["L2"]
        self.assertEqual(body["expires"], lease.expires)
        self.assertEqual(body["effective_expires"],
                         lease.renewals[-1].expires)
        for name in ("expires", "effective_expires"):
            text = body[name]
            self.assertTrue(text.endswith("+00:00"), text)
            fractional = text.split(".", 1)[1]
            self.assertEqual(len(fractional), 12)  # 6 digits + +00:00
            self.assertTrue(fractional[:6].isdigit())
            self.assertEqual(fractional[6:], "+00:00")

    def test_states_and_terminal_markers(self) -> None:
        self._seed()
        self.assertEqual(self._get("L1")["state"], "released")
        self.assertEqual(self._get("L1")["terminal"], "release")
        self.assertEqual(self._get("L2")["state"], "active")
        self.assertEqual(self._get("L3")["state"], "expired")
        self.assertEqual(self._get("L3")["expires"], PAST)
        self.assertEqual(self._get("L3")["effective_expires"], PAST)
        self.assertEqual(self._get("L4")["state"], "confirmed")
        self.assertEqual(self._get("L4")["terminal"], "confirm")
        # Implicit acknowledgement: no terminal marker, but the
        # consumer checkpoint reached the lease's next_after.
        l5 = self._get("L5")
        self.assertEqual(l5["state"], "confirmed")
        self.assertIsNone(l5["terminal"])
        self.assertEqual(l5["checkpoint"], 1)

    def test_checkpoint_tracks_consumer_progress(self) -> None:
        self._seed()
        # c1's checkpoint is 1; c3's confirm advanced its checkpoint
        # to the lease's next_after.
        self.assertEqual(self._get("L1")["checkpoint"], 1)
        self.assertEqual(self._get("L4")["checkpoint"], 1)
        self._advance("c2", 0, 2)
        self.assertEqual(self._get("L2")["checkpoint"], 2)
        # The checkpoint reaching next_after confirms implicitly.
        self.assertEqual(self._get("L2")["state"], "confirmed")

    def test_renewals_empty_for_unrenewed_lease(self) -> None:
        self._seed()
        self.assertEqual(self._get("L1")["renewals"], [])

    def test_unknown_lease_is_404(self) -> None:
        self._seed()
        for lease_id in ("nope", "L1 ", "l1"):
            with self.subTest(lease_id=lease_id):
                with self.assertRaises(ServiceError) as caught:
                    self._get(lease_id)
                self.assertEqual(caught.exception.status_code, 404)
                self.assertEqual(caught.exception.field, "lease_id")


class LeaseGetHTTPTest(LeaseGetMixin, unittest.TestCase):
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
        headers = {"Content-Type": "application/json"} if raw is not None \
            else {}
        conn.request(method, path, body=raw, headers=headers)
        response = conn.getresponse()
        data = response.read().decode("utf-8")
        conn.close()
        return response.status, (json.loads(data) if data else None), data

    def test_get_over_http_with_key_order(self) -> None:
        self._seed()
        status, body, raw = self._request(PATH + "/L2")
        self.assertEqual(status, 200)
        self.assertEqual(list(body), [
            "lease_id", "consumer_id", "expected", "next_after",
            "limit", "expires", "renewals", "terminal", "checkpoint",
            "effective_expires", "state"])
        for earlier, later in (
                ('"lease_id"', '"consumer_id"'),
                ('"consumer_id"', '"expected"'),
                ('"expected"', '"next_after"'),
                ('"next_after"', '"limit"'),
                ('"limit"', '"expires"'),
                ('"expires"', '"renewals"'),
                ('"renewals"', '"terminal"'),
                ('"terminal"', '"checkpoint"'),
                ('"checkpoint"', '"effective_expires"'),
                ('"effective_expires"', '"state"')):
            self.assertLess(raw.index(earlier), raw.index(later))
        # Inside each renewal the renewal_id key precedes its expires.
        renewal_id_at = raw.index('"renewal_id"')
        self.assertLess(renewal_id_at,
                        raw.index('"expires"', renewal_id_at))
        self.assertEqual(body["lease_id"], "L2")
        self.assertEqual(body["state"], "active")

    def test_unknown_lease_is_404(self) -> None:
        self._seed()
        status, body, _ = self._request(PATH + "/nope")
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "lease_id")
        self.assertEqual(list(body), ["message", "field"])

    def test_empty_id_is_400(self) -> None:
        status, body, _ = self._request(PATH + "/")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "lease_id")
        self.assertEqual(list(body), ["message", "field"])

    def test_deeper_path_is_404(self) -> None:
        self._seed()
        for path in (PATH + "/L1/extra", PATH + "//L1"):
            with self.subTest(path=path):
                status, body, _ = self._request(path)
                self.assertEqual(status, 404)
                self.assertEqual(body["field"], "lease_id")
                self.assertEqual(list(body), ["message", "field"])

    def test_strict_percent_decode(self) -> None:
        # A percent-encoded slash is part of the id and decodes to a/b.
        self._commit("r1")
        self._claim("c1", "a/b")
        status, body, _ = self._request(PATH + "/a%2Fb")
        self.assertEqual(status, 200)
        self.assertEqual(body["lease_id"], "a/b")
        for bad in ("/a%zz", "/a%2", "/a%", "/a%ff%ff"):
            with self.subTest(bad=bad):
                status, body, _ = self._request(PATH + bad)
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], "lease_id")
                self.assertEqual(list(body), ["message", "field"])

    def test_unicode_id_is_utf8_percent_decoded(self) -> None:
        self._commit("r1")
        self._claim("c1", "租约")
        status, body, _ = self._request(
            PATH + "/%E7%A7%9F%E7%BA%A6")
        self.assertEqual(status, 200)
        self.assertEqual(body["lease_id"], "租约")

    def test_query_rejected(self) -> None:
        self._seed()
        for query in ("?foo=1", "?foo", "?after=0", "?consumer_id=c1"):
            with self.subTest(query=query):
                status, body, _ = self._request(PATH + "/L1" + query)
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], "query")
                self.assertEqual(list(body), ["message", "field"])

    def test_trailing_question_mark_accepted(self) -> None:
        self._seed()
        status, body, _ = self._request(PATH + "/L1?")
        self.assertEqual(status, 200)
        self.assertEqual(body["lease_id"], "L1")

    def test_nonempty_body_rejected(self) -> None:
        self._seed()
        status, body, _ = self._request(PATH + "/L1", raw="{}")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "request_body")
        self.assertEqual(list(body), ["message", "field"])

    def test_query_checked_before_body(self) -> None:
        self._seed()
        status, body, _ = self._request(PATH + "/L1?foo=1", raw="{}")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "query")

    def _raw_request(self, request: bytes):
        import socket
        sock = socket.create_connection(("127.0.0.1", self.port), timeout=5)
        try:
            sock.sendall(request)
            chunks = []
            while True:
                data = sock.recv(4096)
                if not data:
                    break
                chunks.append(data)
        finally:
            sock.close()
        head, _, raw = b"".join(chunks).partition(b"\r\n\r\n")
        status = int(head.split(b" ", 2)[1])
        return status, (json.loads(raw.decode("utf-8")) if raw else None)

    def test_chunked_nonempty_body_rejected(self) -> None:
        # A legally framed (chunked) non-empty body is 400/request_body
        # just like a Content-Length framed one.
        self._seed()
        status, body = self._raw_request(
            b"GET " + (PATH + "/L1").encode() + b" HTTP/1.1\r\n"
            b"Host: 127.0.0.1\r\n"
            b"Transfer-Encoding: chunked\r\n"
            b"\r\n"
            b"2\r\n{}\r\n0\r\n\r\n")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "request_body")
        self.assertEqual(list(body), ["message", "field"])

    def test_post_is_404(self) -> None:
        self._seed()
        status, _, _ = self._request(PATH + "/L1", method="POST", raw="{}")
        self.assertEqual(status, 404)


class LeaseGetPersistenceTest(LeaseGetMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self._build()
        self.path = os.path.join(self.directory, "state.json")
        self.state_store = attach_persistence(self.service, self.path)

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def _document(self):
        with open(self.path, encoding="utf-8") as handle:
            return json.load(handle)

    def test_reads_write_nothing_and_advance_no_generation(self) -> None:
        self._seed()
        generation = self.state_store.commit_seq
        before = self._document()
        first = self._get("L2")
        first_raw = json.dumps(first, separators=(",", ":"),
                               ensure_ascii=False)
        # Repeated reads are byte-identical while nothing changes.
        second = self._get("L2")
        self.assertEqual(json.dumps(second, separators=(",", ":"),
                                    ensure_ascii=False), first_raw)
        self.assertEqual(self.state_store.commit_seq, generation)
        self.assertEqual(self._document(), before)

    def test_detail_stable_across_restart(self) -> None:
        self._seed()
        before = {lease_id: self._get(lease_id)
                  for lease_id in ("L1", "L2", "L3", "L4", "L5")}
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        for lease_id, expected in before.items():
            with self.subTest(lease_id=lease_id):
                self.assertEqual(
                    restarted.event_gc_batch_lease_get(lease_id), expected)

    def test_no_new_document_section(self) -> None:
        self._seed()
        keys_before = list(self._document())
        self._get("L1")
        self.assertEqual(list(self._document()), keys_before)


if __name__ == "__main__":
    unittest.main()
