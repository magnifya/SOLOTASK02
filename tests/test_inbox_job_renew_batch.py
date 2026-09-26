"""Tests for ``POST /v1/inbox-jobs/renew-batch``.

An atomic batch renewal of 1:1-inbox redelivery leases: the body carries a
``device_id`` and a non-empty ``items`` array of
``lease_id``/``renewal_id`` pairs (neither id repeats across items). Every
item is prechecked in array order with the single-lease renewal rules
(first error aborts the batch, its ``field`` prefixed to ``items[i].``);
only then does the batch renew in input order, each lease's own current
effective deadline extended by exactly 30 seconds, and commit once. A
batch whose items all replay their committed renewals answers 200 and
writes nothing; a partial replay conflicts 409 with the first replayed
item's ``items[i].renewal_id``.
"""
import json
import os
import shutil
import tempfile
import threading
import unittest
from datetime import datetime, timedelta, timezone
from http.client import HTTPConnection

from tests.test_device_inbox import InboxMixin
from e2ee_backend.persistence import (
    PersistenceUnavailable, attach_persistence)
from e2ee_backend.service import DeviceService, ServiceError
from e2ee_backend.http_app import create_server


_PAST = "2000-01-01T00:00:00.000000+00:00"


class _BatchMixin(InboxMixin):
    def _error(self, callable_):
        with self.assertRaises(ServiceError) as caught:
            callable_()
        return caught.exception

    def _job(self, device_id="bob", job_id="j1", op="queue", **extra):
        payload = {"device_id": device_id, "job_id": job_id, "op": op}
        payload.update(extra)
        return self.service.inbox_job(payload)

    def _dispatch(self, job_id="j1", device_id="bob"):
        self._job(device_id=device_id, job_id=job_id)
        return self._job(device_id=device_id, job_id=job_id, op="dispatch")

    def _claim(self, lease_id, device_id="bob", limit=100):
        return self.service.inbox_claim(
            device_id, {"lease_id": lease_id, "limit": limit})

    def _expire(self, lease_id: str) -> None:
        with self.service.store._lock:
            for state in self.service.store._delivery.values():
                for lease in state.leases:
                    if lease.lease_id == lease_id:
                        if lease.renewals:
                            lease.renewals[-1].leased_until = _PAST
                        else:
                            lease.leased_until = _PAST

    def _age_lease(self, lease_id: str, seconds_ago: float = 35.0) -> None:
        """Shift a lease's claim and every renewal into the past, in place.

        Unlike :meth:`_expire` this keeps every frozen deadline (and the
        +30s chain between them) intact, so a replayed renewal still finds
        its committed record and reports the shifted frozen value.
        """
        delta = timedelta(seconds=seconds_ago)

        def shift(stamp: str) -> str:
            return (datetime.fromisoformat(stamp) - delta).isoformat(
                timespec="microseconds")

        with self.service.store._lock:
            for state in self.service.store._delivery.values():
                for lease in state.leases:
                    if lease.lease_id == lease_id:
                        lease.leased_until = shift(lease.leased_until)
                        for renewal in lease.renewals:
                            renewal.leased_until = shift(
                                renewal.leased_until)

    def _deadline(self, lease_id: str, device_id="bob") -> str:
        return self.service.inbox_lease_get(device_id, lease_id)[
            "leased_until"]

    def _batch(self, device_id="bob", items=None, **extra):
        payload = {"device_id": device_id,
                   "items": items if items is not None else []}
        payload.update(extra)
        return self.service.inbox_job_renew_batch(payload)


def _plus_30s(value: str) -> str:
    parsed = datetime.fromisoformat(value)
    return (parsed + timedelta(seconds=30)).isoformat(
        timespec="microseconds")


class InboxJobRenewBatchServiceTest(_BatchMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def test_bad_body_and_device_id_errors(self) -> None:
        good = [{"lease_id": "L1", "renewal_id": "r1"}]
        for payload in (None, [], "x", 3):
            error = self._error(
                lambda payload=payload:
                self.service.inbox_job_renew_batch(payload))
            self.assertEqual((error.status_code, error.field),
                             (400, "request_body"), payload)
        for payload in (
                {"items": good},
                {"device_id": "", "items": good},
                {"device_id": 4, "items": good},
                {"device_id": None, "items": good}):
            error = self._error(
                lambda payload=payload:
                self.service.inbox_job_renew_batch(payload))
            self.assertEqual((error.status_code, error.field),
                             (400, "device_id"), payload)

    def test_items_shape_errors(self) -> None:
        for items in (None, {}, "x", 3, []):
            error = self._error(lambda items=items: self._batch(items=items))
            self.assertEqual((error.status_code, error.field),
                             (400, "items"), items)
        error = self._error(lambda: self._batch(items=["x"]))
        self.assertEqual((error.status_code, error.field), (400, "items[0]"))
        error = self._error(lambda: self._batch(items=[[]]))
        self.assertEqual((error.status_code, error.field), (400, "items[0]"))

    def test_extra_top_level_key_rejected_with_that_field(self) -> None:
        item = {"lease_id": "L1", "renewal_id": "r1"}
        error = self._error(lambda: self._batch(items=[item], op="done"))
        self.assertEqual((error.status_code, error.field), (400, "op"))
        error = self._error(lambda: self._batch(items=[item], bogus=1))
        self.assertEqual((error.status_code, error.field), (400, "bogus"))

    def test_item_field_errors_carry_the_index(self) -> None:
        cases = (
            ([{"renewal_id": "r1"}], "items[0].lease_id"),
            ([{"lease_id": "", "renewal_id": "r1"}],
             "items[0].lease_id"),
            ([{"lease_id": 4, "renewal_id": "r1"}],
             "items[0].lease_id"),
            ([{"lease_id": None, "renewal_id": "r1"}],
             "items[0].lease_id"),
            ([{"lease_id": "L1"}], "items[0].renewal_id"),
            ([{"lease_id": "L1", "renewal_id": ""}],
             "items[0].renewal_id"),
            ([{"lease_id": "L1", "renewal_id": 4}],
             "items[0].renewal_id"),
            ([{"lease_id": "L1", "renewal_id": None}],
             "items[0].renewal_id"),
            ([{"lease_id": "L1", "renewal_id": "r1"},
              {"renewal_id": "r2"}], "items[1].lease_id"),
        )
        for items, field in cases:
            error = self._error(lambda items=items: self._batch(items=items))
            self.assertEqual((error.status_code, error.field),
                             (400, field), items)

    def test_item_extra_keys_rejected_at_item_level(self) -> None:
        base = {"lease_id": "L1", "renewal_id": "r1"}
        cases = (
            [dict(base, extra=1)],
            [dict(base, completion_id="c1")],
            [dict(base), {"lease_id": "L2", "renewal_id": "r2",
                          "bogus": None}],
        )
        for items in cases:
            index = 0 if len(items) == 1 else 1
            error = self._error(lambda items=items: self._batch(items=items))
            self.assertEqual((error.status_code, error.field),
                             (400, f"items[{index}]"), items)

    def test_duplicate_ids_rejected_at_item_level(self) -> None:
        error = self._error(lambda: self._batch(items=[
            {"lease_id": "L1", "renewal_id": "r1"},
            {"lease_id": "L1", "renewal_id": "r2"}]))
        self.assertEqual((error.status_code, error.field), (400, "items[1]"))
        error = self._error(lambda: self._batch(items=[
            {"lease_id": "L1", "renewal_id": "r1"},
            {"lease_id": "L2", "renewal_id": "r1"}]))
        self.assertEqual((error.status_code, error.field), (400, "items[1]"))

    def test_device_gate_precedes_item_checks(self) -> None:
        self._claim("L1", limit=2)
        items = [{"lease_id": "L1", "renewal_id": "r1"}]
        error = self._error(lambda: self._batch(device_id="ghost",
                                                items=items))
        self.assertEqual((error.status_code, error.field),
                         (409, "device_id"))
        self.service.store.revoke_device("bob")
        error = self._error(lambda: self._batch(items=items))
        self.assertEqual((error.status_code, error.field),
                         (409, "device_id"))

    def test_first_batch_renews_one_lease_by_exactly_thirty_seconds(
            self) -> None:
        self._dispatch(job_id="j1")
        before = self._deadline("j1")
        body, status = self._batch(items=[
            {"lease_id": "j1", "renewal_id": "r1"}])
        self.assertEqual(status, 201)
        self.assertEqual(list(body), ["device_id", "results"])
        self.assertEqual(body["device_id"], "bob")
        self.assertEqual(len(body["results"]), 1)
        item = body["results"][0]
        self.assertEqual(list(item),
                         ["lease_id", "renewal_id", "leased_until"])
        self.assertEqual(item["lease_id"], "j1")
        self.assertEqual(item["renewal_id"], "r1")
        expected = _plus_30s(before)
        self.assertEqual(item["leased_until"], expected)
        self.assertTrue(item["leased_until"].endswith("+00:00"))
        # The lease view reports the same new effective deadline.
        self.assertEqual(self._deadline("j1"), expected)

    def test_batch_two_leases_each_extends_its_own_deadline(self) -> None:
        # L1 claims two messages, L2 the remaining three.
        self._claim("L1", limit=2)
        self._claim("L2", limit=100)
        before_l1 = self._deadline("L1")
        before_l2 = self._deadline("L2")
        body, status = self._batch(items=[
            {"lease_id": "L1", "renewal_id": "r1"},
            {"lease_id": "L2", "renewal_id": "r2"}])
        self.assertEqual(status, 201)
        self.assertEqual([r["lease_id"] for r in body["results"]],
                         ["L1", "L2"])
        self.assertEqual(body["results"][0]["leased_until"],
                         _plus_30s(before_l1))
        self.assertEqual(body["results"][1]["leased_until"],
                         _plus_30s(before_l2))

    def test_second_batch_chains_another_thirty_seconds(self) -> None:
        self._dispatch(job_id="j1")
        first, status = self._batch(items=[
            {"lease_id": "j1", "renewal_id": "r1"}])
        self.assertEqual(status, 201)
        second, status = self._batch(items=[
            {"lease_id": "j1", "renewal_id": "r2"}])
        self.assertEqual(status, 201)
        self.assertEqual(second["results"][0]["leased_until"],
                         _plus_30s(first["results"][0]["leased_until"]))

    def test_renewal_id_may_not_repeat_across_items_even_on_other_leases(
            self) -> None:
        # The single-lease entry scopes a renewal_id to one lease, but the
        # batch body requires renewal_id to be unique across every item.
        self._claim("L1", limit=2)
        self._claim("L2", limit=100)
        error = self._error(lambda: self._batch(items=[
            {"lease_id": "L1", "renewal_id": "shared"},
            {"lease_id": "L2", "renewal_id": "shared"}]))
        self.assertEqual((error.status_code, error.field),
                         (400, "items[1]"))

    def test_per_item_errors_use_indexed_fields(self) -> None:
        self._claim("L1", limit=2)
        # Unknown lease.
        error = self._error(lambda: self._batch(items=[
            {"lease_id": "ghost", "renewal_id": "r1"}]))
        self.assertEqual((error.status_code, error.field),
                         (404, "items[0].lease_id"))
        # A lease owned by another device (bob2's claim).
        self._claim("jb", device_id="bob2", limit=100)
        error = self._error(lambda: self._batch(items=[
            {"lease_id": "jb", "renewal_id": "rb"}]))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[0].lease_id"))
        # An expired lease conflicts at lease_id.
        self._expire("L1")
        error = self._error(lambda: self._batch(items=[
            {"lease_id": "L1", "renewal_id": "r1"}]))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[0].lease_id"))
        # A released lease conflicts at lease_id.
        self._dispatch(job_id="j1")
        self.service.inbox_release("bob", "j1")
        error = self._error(lambda: self._batch(items=[
            {"lease_id": "j1", "renewal_id": "r2"}]))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[0].lease_id"))
        # A completed lease conflicts at lease_id.
        self._dispatch(job_id="j2")
        self.service.inbox_lease_complete(
            "bob", "j2", {"completion_id": "c2", "outcome": "delivered"})
        error = self._error(lambda: self._batch(items=[
            {"lease_id": "j2", "renewal_id": "r3"}]))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[0].lease_id"))

    def test_first_error_in_array_order_wins_and_nothing_is_written(
            self) -> None:
        self._claim("L1", limit=2)
        self._claim("L2", limit=100)
        before_l1 = self._deadline("L1")
        # Item 0 fine, item 1 unknown: items[1] reported and L1 not
        # renewed.
        error = self._error(lambda: self._batch(items=[
            {"lease_id": "L1", "renewal_id": "r1"},
            {"lease_id": "ghost", "renewal_id": "r2"}]))
        self.assertEqual((error.status_code, error.field),
                         (404, "items[1].lease_id"))
        self.assertEqual(self._deadline("L1"), before_l1)
        # An earlier bad item wins over a later good one.
        error = self._error(lambda: self._batch(items=[
            {"lease_id": "ghost", "renewal_id": "r2"},
            {"lease_id": "L2", "renewal_id": "r3"}]))
        self.assertEqual((error.status_code, error.field),
                         (404, "items[0].lease_id"))
        self.assertEqual(
            self.service.inbox_lease_get("bob", "L2")["state"], "active")

    def test_all_replays_answer_200_with_frozen_values(self) -> None:
        self._claim("L1", limit=2)
        self._claim("L2", limit=100)
        items = [{"lease_id": "L1", "renewal_id": "r1"},
                 {"lease_id": "L2", "renewal_id": "r2"}]
        first, status = self._batch(items=items)
        self.assertEqual(status, 201)
        # The replays answer even once the leases have expired: the replay
        # decision precedes every lease-state check. Aging shifts the
        # frozen renewal deadlines too, so the frozen replay values are the
        # aged ones (the chain between them stays exactly +30s).
        aged = [
            (datetime.fromisoformat(item["leased_until"])
             - timedelta(seconds=65)).isoformat(timespec="microseconds")
            for item in first["results"]]
        self._age_lease("L1", seconds_ago=65)
        self._age_lease("L2", seconds_ago=65)
        self.assertEqual(
            self.service.inbox_lease_get("bob", "L1")["state"], "expired")
        replay, status = self._batch(items=items)
        self.assertEqual(status, 200)
        self.assertEqual([r["leased_until"] for r in replay["results"]],
                         aged)
        # A single-item all-replay answers 200 as well, with the same
        # frozen (aged) value.
        replay, status = self._batch(items=[items[1]])
        self.assertEqual(status, 200)
        self.assertEqual(replay["results"][0]["leased_until"], aged[1])

    def test_full_replay_still_requires_a_known_active_device(self) -> None:
        self._claim("L1", limit=2)
        items = [{"lease_id": "L1", "renewal_id": "r1"}]
        self._batch(items=items)
        self.service.store.revoke_device("bob")
        # The device gate runs first in the batch, ahead of the items.
        error = self._error(lambda: self._batch(items=items))
        self.assertEqual((error.status_code, error.field),
                         (409, "device_id"))

    def test_partial_replay_conflicts_with_first_replayed_item(self) -> None:
        self._claim("L1", limit=2)
        self._claim("L2", limit=100)
        self._batch(items=[{"lease_id": "L1", "renewal_id": "r1"}])
        before_l2 = self._deadline("L2")
        # The replay sits at index 0.
        error = self._error(lambda: self._batch(items=[
            {"lease_id": "L1", "renewal_id": "r1"},
            {"lease_id": "L2", "renewal_id": "r2"}]))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[0].renewal_id"))
        # ... and at index 1.
        error = self._error(lambda: self._batch(items=[
            {"lease_id": "L2", "renewal_id": "r2"},
            {"lease_id": "L1", "renewal_id": "r1"}]))
        self.assertEqual((error.status_code, error.field),
                         (409, "items[1].renewal_id"))
        # The fresh item was not written.
        self.assertEqual(self._deadline("L2"), before_l2)


class InboxJobRenewBatchPersistenceTest(_BatchMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self._build()
        self.path = os.path.join(self.directory, "state.json")
        self.state_store = attach_persistence(self.service, self.path)

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_batch_advances_one_generation_replays_and_failures_none(
            self) -> None:
        self._claim("L1", limit=2)
        self._claim("L2", limit=100)
        items = [{"lease_id": "L1", "renewal_id": "r1"},
                 {"lease_id": "L2", "renewal_id": "r2"}]
        before = self.state_store.commit_seq
        _, status = self._batch(items=items)
        self.assertEqual(status, 201)
        self.assertEqual(self.state_store.commit_seq, before + 1)
        _, status = self._batch(items=items)
        self.assertEqual(status, 200)
        self.assertEqual(self.state_store.commit_seq, before + 1)
        with self.assertRaises(ServiceError):
            self._batch(items=[{"lease_id": "ghost",
                                "renewal_id": "r9"}])
        self.assertEqual(self.state_store.commit_seq, before + 1)

    def test_failed_batch_writes_nothing(self) -> None:
        self._claim("L1", limit=2)
        self._claim("L2", limit=100)
        before = self.state_store.commit_seq
        before_l1 = self._deadline("L1")
        with self.assertRaises(ServiceError) as caught:
            self._batch(items=[
                {"lease_id": "L1", "renewal_id": "r1"},
                {"lease_id": "ghost", "renewal_id": "r2"}])
        self.assertEqual(caught.exception.status_code, 404)
        self.assertEqual(self.state_store.commit_seq, before)
        self.assertEqual(self._deadline("L1"), before_l1)

    def test_restart_restores_renewals_and_replays(self) -> None:
        self._dispatch(job_id="j1")
        first, status = self._batch(items=[
            {"lease_id": "j1", "renewal_id": "r1"}])
        self.assertEqual(status, 201)
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        lease = restarted.inbox_lease_get("bob", "j1")
        self.assertEqual(lease["leased_until"],
                         first["results"][0]["leased_until"])
        # The committed batch still replays as 200 with the frozen values.
        body, status = restarted.inbox_job_renew_batch(
            {"device_id": "bob", "items": [
                {"lease_id": "j1", "renewal_id": "r1"}]})
        self.assertEqual(status, 200)
        self.assertEqual(body["results"], first["results"])
        self.assertTrue(restarted.persistence_integrity()["consistent"])

    def test_save_failure_rolls_back_the_whole_batch(self) -> None:
        import e2ee_backend.persistence as persistence_mod

        real_fsync = persistence_mod.os.fsync
        calls = {"n": 0}

        def fail_first_directory_fsync(fd: int) -> None:  # noqa: ANN001
            if os.fstat(fd).st_mode & 0o170000 == 0o040000:
                calls["n"] += 1
                if calls["n"] == 1:
                    raise OSError("transient directory fsync failure")
            real_fsync(fd)

        self._claim("L1", limit=2)
        self._claim("L2", limit=100)
        before_l1 = self._deadline("L1")
        before_l2 = self._deadline("L2")
        generation = self.state_store.commit_seq
        persistence_mod.os.fsync = fail_first_directory_fsync
        try:
            with self.assertRaises(PersistenceUnavailable):
                self._batch(items=[
                    {"lease_id": "L1", "renewal_id": "r1"},
                    {"lease_id": "L2", "renewal_id": "r2"}])
        finally:
            persistence_mod.os.fsync = real_fsync
        self.assertEqual(self.state_store.commit_seq, generation)
        self.assertEqual(self._deadline("L1"), before_l1)
        self.assertEqual(self._deadline("L2"), before_l2)
        body, status = self._batch(items=[
            {"lease_id": "L1", "renewal_id": "r1"},
            {"lease_id": "L2", "renewal_id": "r2"}])
        self.assertEqual(status, 201)
        self.assertEqual(body["results"][0]["leased_until"],
                         _plus_30s(before_l1))
        self.assertEqual(self.state_store.commit_seq, generation + 1)


class InboxJobRenewBatchHTTPTest(_BatchMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()
        self.server, self.service = create_server(
            "127.0.0.1", 0, self.service)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        self._dispatch(job_id="j1")

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _request(self, payload, raw_body=None):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        body = raw_body if raw_body is not None else json.dumps(payload)
        conn.request("POST", "/v1/inbox-jobs/renew-batch", body=body,
                     headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        data = response.read().decode("utf-8")
        conn.close()
        return response.status, (json.loads(data) if data else None), data

    def test_renew_batch_over_http(self) -> None:
        status, body, raw = self._request(
            {"device_id": "bob", "items": [
                {"lease_id": "j1", "renewal_id": "r1"}]})
        self.assertEqual(status, 201)
        self.assertEqual(list(body), ["device_id", "results"])
        self.assertEqual(body["device_id"], "bob")
        item = body["results"][0]
        self.assertEqual(list(item),
                         ["lease_id", "renewal_id", "leased_until"])
        self.assertEqual(item["lease_id"], "j1")
        self.assertLess(raw.index('"device_id"'), raw.index('"results"'))
        self.assertLess(raw.index('"lease_id"'),
                        raw.index('"renewal_id"'))
        self.assertLess(raw.index('"renewal_id"'),
                        raw.index('"leased_until"'))
        # The full replay answers 200 byte-identically.
        status, body, raw2 = self._request(
            {"device_id": "bob", "items": [
                {"lease_id": "j1", "renewal_id": "r1"}]})
        self.assertEqual(status, 200)
        self.assertEqual(raw2, raw)

    def test_errors_over_http(self) -> None:
        status, body, _ = self._request(None, raw_body="{")
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "request_body")
        self.assertEqual(list(body), ["message", "field"])
        status, body, _ = self._request([1, 2])
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "request_body")
        status, body, _ = self._request({"device_id": "bob", "items": []})
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "items")
        status, body, _ = self._request(
            {"device_id": "bob", "items": [{"renewal_id": "r1"}]})
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "items[0].lease_id")
        status, body, _ = self._request(
            {"device_id": "bob", "items": [{"lease_id": "j1",
                                            "renewal_id": "r1",
                                            "bogus": 1}]})
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "items[0]")
        status, body, _ = self._request(
            {"device_id": "ghost", "items": [
                {"lease_id": "j1", "renewal_id": "r1"}]})
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "device_id")
        status, body, _ = self._request(
            {"device_id": "bob", "items": [
                {"lease_id": "ghost", "renewal_id": "r1"}]})
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "items[0].lease_id")
        # Partial replay: renew j1, release it (its exact replay still
        # answers after release), then dispatch j2 which now claims all
        # messages and stays active. The mixed batch conflicts at the
        # replay item's renewal_id.
        self._request({"device_id": "bob", "items": [
            {"lease_id": "j1", "renewal_id": "r1"}]})
        self.service.inbox_release("bob", "j1")
        self._dispatch(job_id="j2")
        status, body, _ = self._request(
            {"device_id": "bob", "items": [
                {"lease_id": "j1", "renewal_id": "r1"},
                {"lease_id": "j2", "renewal_id": "r2"}]})
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "items[0].renewal_id")
        self.assertEqual(list(body), ["message", "field"])
        # The failed batch wrote nothing: j2 was not renewed.
        lease = self.service.inbox_lease_get("bob", "j2")
        self.assertEqual(lease["state"], "active")


if __name__ == "__main__":
    unittest.main()
