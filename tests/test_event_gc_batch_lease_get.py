"""Tests for the single batch-cleanup audit lease query.

``GET /v1/event-gc-batch/leases/{lease_id}`` takes no request body (a
non-empty one, however framed, is 400/request_body) and no query
parameters (any is 400/query); validation order is request body then
query. The path id is a single non-empty segment, strictly
percent-decoded as UTF-8 (empty segment or a deeper path is
404/lease_id; a bad escape or invalid UTF-8 is 400/lease_id), and an
uncommitted lease id is 404/lease_id.

On success the body keys are ``lease_id``, ``consumer_id``,
``expected``, ``next_after``, ``limit``, ``expires``, ``renewals``,
``terminal``, ``checkpoint``, ``effective_expires`` and ``state`` in
that order; renewal items keep ``renewal_id`` then ``expires`` and the
state follows the same one-instant order as the paged view (released,
confirmed, expired, active). The lookup is read-only: no persistence,
no ``commit_seq`` advance, byte-identical while state is unchanged and
stable across a restart.
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

KEYS = ["lease_id", "consumer_id", "expected", "next_after", "limit",
        "expires", "renewals", "terminal", "checkpoint",
        "effective_expires", "state"]


class LeaseGetMixin:
    def _build(self) -> None:
        self.service = DeviceService()
        self.service.store.add_device(Device("u", "alice", "ik"))
        self.service.store.add_device(Device("u", "bob", "ik"))

    def _commit(self, request_id):
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
        # Five leases mirroring the paged-view seed:
        #   L1 (c1) released, L2 (c2) active with one renewal,
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
        self._claim("c1", "L3", limit=100)
        self.service.store._cleanup_leases["L3"].expires = PAST
        self._claim("c3", "L4")
        self._op("c3", "L4", "confirm")
        self._claim("c1", "L5")
        self._advance("c1", 0, 1)


class LeaseGetServiceTest(LeaseGetMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def test_unknown_lease_is_404(self) -> None:
        with self.assertRaises(ServiceError) as caught:
            self._get("nope")
        self.assertEqual(caught.exception.status_code, 404)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_empty_or_nonstring_lease_id_is_404(self) -> None:
        for lease_id in ("", 1, True, [], {}):
            with self.subTest(lease_id=lease_id):
                with self.assertRaises(ServiceError) as caught:
                    self._get(lease_id)
                self.assertEqual(caught.exception.status_code, 404)
                self.assertEqual(caught.exception.field, "lease_id")

    def test_active_lease_shape_and_key_order(self) -> None:
        self._commit("r1")
        self._claim("c1", "L1")
        body = self._get("L1")
        self.assertEqual(list(body), KEYS)
        self.assertEqual(body["lease_id"], "L1")
        self.assertEqual(body["consumer_id"], "c1")
        self.assertEqual(body["expected"], 0)
        self.assertEqual(body["next_after"], 1)
        self.assertEqual(body["limit"], 1)
        self.assertEqual(body["renewals"], [])
        self.assertIsNone(body["terminal"])
        self.assertEqual(body["checkpoint"], 0)
        self.assertEqual(body["effective_expires"], body["expires"])
        self.assertEqual(body["state"], "active")
        for name in ("expected", "next_after", "limit", "checkpoint"):
            self.assertIsInstance(body[name], int)
            self.assertIsNot(type(body[name]), bool)
            self.assertGreaterEqual(body[name], 0)

    def test_renewals_in_order_with_item_key_order(self) -> None:
        self._commit("r1")
        self._claim("c2", "L2", limit=2)
        first = self._renew("c2", "L2", "renew-1")
        second = self._renew("c2", "L2", "renew-2")
        body = self._get("L2")
        self.assertEqual(list(body), KEYS)
        self.assertEqual(len(body["renewals"]), 2)
        self.assertEqual(
            [list(item) for item in body["renewals"]],
            [["renewal_id", "expires"], ["renewal_id", "expires"]])
        self.assertEqual([item["renewal_id"] for item in body["renewals"]],
                         ["renew-1", "renew-2"])
        self.assertEqual([item["expires"] for item in body["renewals"]],
                         [first[0]["expires"], second[0]["expires"]])
        for item in body["renewals"]:
            self.assertIsInstance(item["renewal_id"], str)
            self.assertTrue(item["renewal_id"])
            self.assertIsInstance(item["expires"], str)
            self.assertTrue(item["expires"])
        self.assertEqual(body["effective_expires"],
                         second[0]["expires"])
        self.assertNotEqual(body["effective_expires"], body["expires"])
        self.assertEqual(body["state"], "active")

    def test_states(self) -> None:
        self._seed()
        self.assertEqual(self._get("L1")["state"], "released")
        self.assertEqual(self._get("L2")["state"], "active")
        self.assertEqual(self._get("L3")["state"], "expired")
        self.assertEqual(self._get("L4")["state"], "confirmed")
        self.assertEqual(self._get("L5")["state"], "confirmed")

    def test_terminals_and_checkpoints(self) -> None:
        self._seed()
        l1 = self._get("L1")
        self.assertEqual(l1["terminal"], "release")
        # A release moves no checkpoint: c1's record sits at 1 (advanced
        # implicitly for L5), which still serves L1's owner value.
        self.assertEqual(l1["checkpoint"], 1)
        l4 = self._get("L4")
        self.assertEqual(l4["terminal"], "confirm")
        self.assertEqual(l4["checkpoint"], 1)
        self.assertEqual(l4["next_after"], 1)
        l5 = self._get("L5")
        self.assertIsNone(l5["terminal"])
        self.assertEqual(l5["checkpoint"], 1)
        l3 = self._get("L3")
        self.assertEqual(l3["checkpoint"], 1)
        self.assertEqual(l3["effective_expires"], PAST)
        self.assertEqual(l3["expires"], PAST)
        l2 = self._get("L2")
        # c2 never advanced a checkpoint: the value reads as 0.
        self.assertEqual(l2["checkpoint"], 0)

    def test_timestamps_use_utc_format(self) -> None:
        self._commit("r1")
        self._claim("c1", "L1")
        body = self._get("L1")
        for name in ("expires", "effective_expires"):
            text = body[name]
            self.assertTrue(text.endswith("+00:00"), text)
            fractional = text.split(".", 1)[1]
            self.assertEqual(len(fractional), 12)
            self.assertTrue(fractional[:6].isdigit())
            self.assertEqual(fractional[6:], "+00:00")

    def test_released_wins_over_acknowledgement(self) -> None:
        self._seed()
        # c1's checkpoint (1) reaches L1's next_after too, but the
        # explicit release takes precedence in the state order.
        self.assertEqual(self._get("L1")["state"], "released")

    def test_implicit_confirm_has_no_terminal_marker(self) -> None:
        self._seed()
        l5 = self._get("L5")
        self.assertEqual(l5["state"], "confirmed")
        self.assertIsNone(l5["terminal"])
        self.assertGreaterEqual(l5["checkpoint"], l5["next_after"])

    def test_checkpoint_tracks_later_advance(self) -> None:
        self._commit("r1")
        self._commit("r2")
        self._claim("c1", "L1", limit=1)
        # Before any advance the checkpoint is 0 and the lease active.
        self.assertEqual(self._get("L1")["checkpoint"], 0)
        self._advance("c1", 0, 1)
        body = self._get("L1")
        self.assertEqual(body["checkpoint"], 1)
        self.assertEqual(body["state"], "confirmed")
        self.assertIsNone(body["terminal"])


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

    def _request(self, path=None, raw=None, method="GET"):
        if path is None:
            path = PATH + "/L1"
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        headers = {"Content-Type": "application/json"} if raw is not None \
            else {}
        conn.request(method, path, body=raw, headers=headers)
        response = conn.getresponse()
        data = response.read().decode("utf-8")
        conn.close()
        return response.status, (json.loads(data) if data else None), data

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

    def test_get_over_http_with_key_order(self) -> None:
        self._seed()
        status, body, raw = self._request(path=PATH + "/L2")
        self.assertEqual(status, 200)
        self.assertEqual(list(body), KEYS)
        for earlier, later in zip(KEYS, KEYS[1:]):
            self.assertLess(raw.index('"' + earlier + '"'),
                            raw.index('"' + later + '"'))
        renewal_pos = raw.index('"renewals"')
        self.assertLess(renewal_pos, raw.index('"renewal_id"'))
        self.assertLess(raw.index('"renewal_id"'),
                        raw.index('"terminal"'))

    def test_states_over_http(self) -> None:
        self._seed()
        for lease_id, state, terminal in (
                ("L1", "released", "release"),
                ("L2", "active", None),
                ("L3", "expired", None),
                ("L4", "confirmed", "confirm"),
                ("L5", "confirmed", None)):
            with self.subTest(lease_id=lease_id):
                status, body, _ = self._request(path=PATH + "/" + lease_id)
                self.assertEqual(status, 200)
                self.assertEqual(body["state"], state)
                self.assertEqual(body["terminal"], terminal)

    def test_unknown_lease_is_404(self) -> None:
        status, body, _ = self._request(path=PATH + "/nope")
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "lease_id")
        self.assertEqual(list(body), ["message", "field"])

    def test_empty_segment_is_400(self) -> None:
        status, body, _ = self._request(path=PATH + "/")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "lease_id")
        self.assertEqual(list(body), ["message", "field"])

    def test_deeper_path_is_404(self) -> None:
        status, body, _ = self._request(path=PATH + "/L1/extra")
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "lease_id")

    def test_trailing_slash_extra_segment_is_404(self) -> None:
        status, body, _ = self._request(path=PATH + "/L1/")
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "lease_id")

    def test_strict_percent_decode(self) -> None:
        self._commit("r1")
        self._claim("c1", "a/b")
        # A percent-encoded slash is part of the id and decodes to a/b.
        status, body, _ = self._request(path=PATH + "/a%2Fb")
        self.assertEqual(status, 200)
        self.assertEqual(body["lease_id"], "a/b")
        for bad in ("/a%zz", "/a%2", "/a%", "/a%ff%ff"):
            with self.subTest(bad=bad):
                status, body, _ = self._request(path=PATH + bad)
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], "lease_id")
                self.assertEqual(list(body), ["message", "field"])

    def test_unicode_id_is_utf8_percent_decoded(self) -> None:
        lease_id = "lease-é-☃"
        self._commit("r1")
        self._claim("c1", lease_id)
        encoded = "".join(
            f"%{byte:02X}" for byte in lease_id.encode("utf-8"))
        status, body, _ = self._request(path=PATH + "/" + encoded)
        self.assertEqual(status, 200)
        self.assertEqual(body["lease_id"], lease_id)

    def test_query_rejected(self) -> None:
        self._seed()
        for query in ("?after=0", "?limit=1", "?consumer_id=c1",
                      "?foo", "?x=1"):
            with self.subTest(query=query):
                status, body, _ = self._request(
                    path=PATH + "/L1" + query)
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], "query")
                self.assertEqual(list(body), ["message", "field"])
        # A bare trailing '?' carries no parameter and is accepted.
        status, _, _ = self._request(path=PATH + "/L1?")
        self.assertEqual(status, 200)

    def test_nonempty_body_rejected(self) -> None:
        self._seed()
        status, body, _ = self._request(path=PATH + "/L1", raw="{}")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "request_body")
        self.assertEqual(list(body), ["message", "field"])

    def test_chunked_nonempty_body_rejected(self) -> None:
        status, body = self._raw_request(
            b"GET " + PATH.encode() + b"/L1 HTTP/1.1\r\n"
            b"Host: 127.0.0.1\r\n"
            b"Transfer-Encoding: chunked\r\n"
            b"\r\n"
            b"2\r\n{}\r\n0\r\n\r\n")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "request_body")
        self.assertEqual(list(body), ["message", "field"])

    def test_chunked_empty_body_accepted(self) -> None:
        self._seed()
        status, body = self._raw_request(
            b"GET " + PATH.encode() + b"/L1 HTTP/1.1\r\n"
            b"Host: 127.0.0.1\r\n"
            b"Transfer-Encoding: chunked\r\n"
            b"\r\n"
            b"0\r\n\r\n")
        self.assertEqual(status, 200)
        self.assertEqual(body["lease_id"], "L1")

    def test_body_checked_before_query(self) -> None:
        status, body, _ = self._request(path=PATH + "/L1?foo=1",
                                        raw="{}")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "request_body")

    def test_query_on_unknown_lease_is_still_400(self) -> None:
        # Request validation precedes the lease lookup.
        status, body, _ = self._request(path=PATH + "/nope?foo=1")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "query")

    def test_nonempty_body_on_unknown_lease_is_still_400(self) -> None:
        status, body, _ = self._request(path=PATH + "/nope", raw="{}")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "request_body")

    def test_post_is_404(self) -> None:
        status, _, _ = self._request(path=PATH + "/L1", method="POST",
                                     raw="{}")
        self.assertEqual(status, 404)

    def test_post_on_collection_still_404(self) -> None:
        status, _, _ = self._request(path=PATH, method="POST", raw="{}")
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
        second = self._get("L2")
        self.assertEqual(json.dumps(second, separators=(",", ":"),
                                    ensure_ascii=False), first_raw)
        self.assertEqual(self.state_store.commit_seq, generation)
        self.assertEqual(self._document(), before)

    def test_lease_stable_across_restart(self) -> None:
        self._seed()
        before = self._get("L2")
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        after = restarted.event_gc_batch_lease_get("L2")
        self.assertEqual(after, before)
        self.assertEqual(after["state"], "active")
        self.assertEqual(
            restarted.event_gc_batch_lease_get("L1")["state"],
            "released")

    def test_no_new_document_section(self) -> None:
        self._seed()
        keys_before = list(self._document())
        self._get("L1")
        self._get("L2")
        self.assertEqual(list(self._document()), keys_before)


if __name__ == "__main__":
    unittest.main()
