"""Tests for the group-inbox lease renewal endpoint.

POST /v1/devices/{device_id}/group-inbox/leases/{lease_id}/renew extends
an occupied group-inbox lease by exactly 30 seconds per renewal under the
store lock shared with group claims, releases, acks, retries and device
revocation. ``renewal_id`` is idempotent only within one lease: a replay
returns the first response byte-identically with 200, even after expiry,
release or revocation; only a first renewal checks the device and the
lease liveness. Renewals persist on the per-device ``group_delivery``
records alongside the existing claim/release state, in version=1 files
with the same shape and validation as the 1:1-inbox renewals.
"""
import json
import os
import shutil
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from http.client import HTTPConnection

from e2ee_backend.persistence import (
    PersistenceUnavailable, StateFileError, attach_persistence)
from e2ee_backend.service import DeviceService, ServiceError
from e2ee_backend.http_app import create_server

from .test_group_inbox_claim import GroupLeaseMixin


def _fresh_service():
    mixin = GroupLeaseMixin()
    mixin._build()
    return mixin.service


class GroupLeaseRenewServiceTest(GroupLeaseMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._build()

    def test_first_renewal_extends_deadline_by_thirty_seconds(self) -> None:
        claim, status = self.service.group_inbox_claim(
            "d3", {"lease_id": "L1", "limit": 2})
        self.assertEqual(status, 201)
        body, status = self.service.group_inbox_lease_renew(
            "d3", "L1", {"renewal_id": "R1"})
        self.assertEqual(status, 201)
        self.assertEqual(list(body),
                         ["device_id", "lease_id", "renewal_id",
                          "leased_until"])
        self.assertEqual(body["device_id"], "d3")
        self.assertEqual(body["lease_id"], "L1")
        self.assertEqual(body["renewal_id"], "R1")
        self.assertRegex(body["leased_until"], r"\.\d{6}\+00:00$")
        extended = datetime.fromisoformat(body["leased_until"])
        self.assertEqual(
            extended - datetime.fromisoformat(claim["leased_until"]),
            timedelta(seconds=30))
        again, status = self.service.group_inbox_lease_renew(
            "d3", "L1", {"renewal_id": "R2"})
        self.assertEqual(status, 201)
        self.assertEqual(
            datetime.fromisoformat(again["leased_until"]) - extended,
            timedelta(seconds=30))

    def test_replay_is_frozen_after_expiry_release_revoke(self) -> None:
        self.service.group_inbox_claim("d3", {"lease_id": "L1", "limit": 2})
        first, status = self.service.group_inbox_lease_renew(
            "d3", "L1", {"renewal_id": "R1"})
        self.assertEqual(status, 201)
        self._expire_all_group_leases()
        replay, status = self.service.group_inbox_lease_renew(
            "d3", "L1", {"renewal_id": "R1"})
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)
        self.service.group_inbox_release("d3", "L1")
        replay, status = self.service.group_inbox_lease_renew(
            "d3", "L1", {"renewal_id": "R1"})
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)
        self.store.revoke_device("d3")
        replay, status = self.service.group_inbox_lease_renew(
            "d3", "L1", {"renewal_id": "R1"})
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)

    def test_renewal_id_scoped_to_one_lease(self) -> None:
        self.service.group_inbox_claim("d3", {"lease_id": "L1", "limit": 1})
        self.service.group_inbox_claim("d2", {"lease_id": "L2", "limit": 5})
        first, status = self.service.group_inbox_lease_renew(
            "d3", "L1", {"renewal_id": "SHARED"})
        self.assertEqual(status, 201)
        other, status = self.service.group_inbox_lease_renew(
            "d2", "L2", {"renewal_id": "SHARED"})
        self.assertEqual(status, 201)
        replay, status = self.service.group_inbox_lease_renew(
            "d3", "L1", {"renewal_id": "SHARED"})
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)

    def test_lease_resolution_precedes_device_state(self) -> None:
        self.service.group_inbox_claim("d3", {"lease_id": "L1", "limit": 1})
        with self.assertRaises(ServiceError) as caught:
            self.service.group_inbox_lease_renew(
                "ghost", "NOPE", {"renewal_id": "R1"})
        self.assertEqual(caught.exception.status_code, 404)
        self.assertEqual(caught.exception.field, "lease_id")
        with self.assertRaises(ServiceError) as caught:
            self.service.group_inbox_lease_renew(
                "ghost", "L1", {"renewal_id": "R1"})
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")
        self.service.inbox_claim("d2", {"lease_id": "I1", "limit": 1})
        with self.assertRaises(ServiceError) as caught:
            self.service.group_inbox_lease_renew(
                "d2", "I1", {"renewal_id": "R1"})
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_revoked_device_blocks_first_renewal_only(self) -> None:
        self.service.group_inbox_claim("d3", {"lease_id": "L1", "limit": 1})
        self.store.revoke_device("d3")
        with self.assertRaises(ServiceError) as caught:
            self.service.group_inbox_lease_renew(
                "d3", "L1", {"renewal_id": "FRESH"})
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "device_id")

    def test_released_lease_blocks_first_renewal_only(self) -> None:
        service = _fresh_service()
        service.group_inbox_claim("d3", {"lease_id": "L1", "limit": 1})
        first, _ = service.group_inbox_lease_renew(
            "d3", "L1", {"renewal_id": "R1"})
        service.group_inbox_release("d3", "L1")
        with self.assertRaises(ServiceError) as caught:
            service.group_inbox_lease_renew(
                "d3", "L1", {"renewal_id": "FRESH"})
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")
        replay, status = service.group_inbox_lease_renew(
            "d3", "L1", {"renewal_id": "R1"})
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)

    def test_expired_lease_blocks_first_renewal(self) -> None:
        service = _fresh_service()
        service.group_inbox_claim("d3", {"lease_id": "L1", "limit": 1})
        for state in service.store._group_delivery.values():
            for lease in state.leases:
                lease.leased_until = "2000-01-01T00:00:00.000000+00:00"
        with self.assertRaises(ServiceError) as caught:
            service.group_inbox_lease_renew(
                "d3", "L1", {"renewal_id": "R1"})
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.field, "lease_id")

    def test_payload_validation(self) -> None:
        self.service.group_inbox_claim("d3", {"lease_id": "L1", "limit": 1})
        cases = [
            ("{bad json", "request_body"),
            ([1, 2], "request_body"),
            ("text", "request_body"),
            ({}, "renewal_id"),
            ({"renewal_id": ""}, "renewal_id"),
            ({"renewal_id": 5}, "renewal_id"),
            ({"renewal_id": True}, "renewal_id"),
            ({"renewal_id": None}, "renewal_id"),
            ({"renewal_id": "R1", "extra": 1}, "extra"),
        ]
        for payload, field in cases:
            with self.assertRaises(ServiceError) as caught:
                self.service.group_inbox_lease_renew("d3", "L1", payload)
            self.assertEqual(caught.exception.status_code, 400, payload)
            self.assertEqual(caught.exception.field, field, payload)


class GroupLeaseRenewPersistenceTest(GroupLeaseMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self._build()
        self.path = os.path.join(self.directory, "state.json")
        self.state_store = attach_persistence(self.service, self.path)

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_renewal_commits_one_generation_replay_none(self) -> None:
        self.service.group_inbox_claim("d3", {"lease_id": "L1", "limit": 2})
        generation = self.state_store.commit_seq
        first, status = self.service.group_inbox_lease_renew(
            "d3", "L1", {"renewal_id": "R1"})
        self.assertEqual(status, 201)
        self.assertEqual(self.state_store.commit_seq, generation + 1)
        replay, status = self.service.group_inbox_lease_renew(
            "d3", "L1", {"renewal_id": "R1"})
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)
        self.assertEqual(self.state_store.commit_seq, generation + 1)
        with open(self.path, encoding="utf-8") as handle:
            document = json.load(handle)
        carrying = [record for record in document["group_delivery"]
                    if any(lease["lease_id"] == "L1"
                           for lease in record.get("leases", []))]
        self.assertEqual(len(carrying), 2)
        for record in carrying:
            lease = next(lease for lease in record["leases"]
                         if lease["lease_id"] == "L1")
            self.assertEqual(
                list(lease),
                ["lease_id", "limit", "leased_until", "released_at",
                 "renewals"])
            self.assertEqual(lease["renewals"], [{
                "renewal_id": "R1",
                "leased_until": first["leased_until"]}])

    def test_restart_restores_chain_and_replay(self) -> None:
        self.service.group_inbox_claim("d3", {"lease_id": "L1", "limit": 2})
        first, _ = self.service.group_inbox_lease_renew(
            "d3", "L1", {"renewal_id": "R1"})
        second, _ = self.service.group_inbox_lease_renew(
            "d3", "L1", {"renewal_id": "R2"})
        restarted = DeviceService()
        attach_persistence(restarted, self.path)
        replay, status = restarted.group_inbox_lease_renew(
            "d3", "L1", {"renewal_id": "R1"})
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)
        third, status = restarted.group_inbox_lease_renew(
            "d3", "L1", {"renewal_id": "R3"})
        self.assertEqual(status, 201)
        self.assertEqual(
            datetime.fromisoformat(third["leased_until"])
            - datetime.fromisoformat(second["leased_until"]),
            timedelta(seconds=30))

    def test_legacy_file_without_renewals_loads_empty(self) -> None:
        self.service.group_inbox_claim("d3", {"lease_id": "L1", "limit": 1})
        with open(self.path, encoding="utf-8") as handle:
            document = json.load(handle)
        for record in document["group_delivery"]:
            for lease in record.get("leases", []):
                lease.pop("renewals", None)
        document.pop("integrity_log_version", None)
        legacy_path = os.path.join(self.directory, "legacy.json")
        with open(legacy_path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        restarted = DeviceService()
        attach_persistence(restarted, legacy_path)  # must not raise
        _body, status = restarted.group_inbox_lease_renew(
            "d3", "L1", {"renewal_id": "R1"})
        self.assertEqual(status, 201)

    def _malformed(self, mutate) -> str:
        self.service.group_inbox_claim("d3", {"lease_id": "L1", "limit": 2})
        first, _ = self.service.group_inbox_lease_renew(
            "d3", "L1", {"renewal_id": "R1"})
        with open(self.path, encoding="utf-8") as handle:
            document = json.load(handle)
        mutate(document, first["leased_until"])
        document.pop("integrity_log_version", None)
        bad_path = os.path.join(self.directory, "bad.json")
        with open(bad_path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        return bad_path

    def _assert_refuses_startup(self, bad_path: str) -> None:
        with open(bad_path, "rb") as handle:
            original = handle.read()
        restarted = DeviceService()
        with self.assertRaises(StateFileError):
            attach_persistence(restarted, bad_path)
        with open(bad_path, "rb") as handle:
            self.assertEqual(handle.read(), original)

    def test_restore_rejects_malformed_renewals(self) -> None:
        def non_list(document, _until):
            document["group_delivery"][0]["leases"][0]["renewals"] = {}
        self._assert_refuses_startup(self._malformed(non_list))

        def empty_id(document, _until):
            document["group_delivery"][0]["leases"][0]["renewals"][0][
                "renewal_id"] = ""
        self._assert_refuses_startup(self._malformed(empty_id))

        def bad_timestamp(document, _until):
            document["group_delivery"][0]["leases"][0]["renewals"][0][
                "leased_until"] = "2030-01-01T00:00:00"
        self._assert_refuses_startup(self._malformed(bad_timestamp))

        def duplicate_id(document, until):
            lease = document["group_delivery"][0]["leases"][0]
            lease["renewals"].append({"renewal_id": "R1",
                                      "leased_until": _extend(until)})
        self._assert_refuses_startup(self._malformed(duplicate_id))

    def test_restore_rejects_wrong_increment_and_copy_mismatch(self) -> None:
        def wrong_increment(document, _until):
            # 31 seconds past the committed renewal deadline.
            lease = document["group_delivery"][0]["leases"][0]
            late = (datetime.fromisoformat(_until)
                    + timedelta(seconds=31)).isoformat(timespec="microseconds")
            lease["renewals"].append({"renewal_id": "R2",
                                      "leased_until": late})
        self._assert_refuses_startup(self._malformed(wrong_increment))

        def copy_mismatch(document, _until):
            records = [record for record in document["group_delivery"]
                       if record.get("leases")]
            records[0]["leases"][0]["renewals"][0]["renewal_id"] = "OTHER"
        self._assert_refuses_startup(self._malformed(copy_mismatch))

    def test_write_failure_rolls_back_renewal(self) -> None:
        import e2ee_backend.persistence as persistence_mod
        self.service.group_inbox_claim("d3", {"lease_id": "L1", "limit": 2})
        first, _ = self.service.group_inbox_lease_renew(
            "d3", "L1", {"renewal_id": "R1"})
        generation = self.state_store.commit_seq
        real_fsync = persistence_mod.os.fsync
        calls = {"n": 0}

        def fail_first_directory_fsync(fd):  # noqa: ANN001
            if os.fstat(fd).st_mode & 0o170000 == 0o040000:
                calls["n"] += 1
                if calls["n"] == 1:
                    raise OSError("transient directory fsync failure")
            real_fsync(fd)

        persistence_mod.os.fsync = fail_first_directory_fsync
        try:
            with self.assertRaises(PersistenceUnavailable):
                self.service.group_inbox_lease_renew(
                    "d3", "L1", {"renewal_id": "R2"})
        finally:
            persistence_mod.os.fsync = real_fsync
        self.assertEqual(self.state_store.commit_seq, generation)
        # R2 was rolled back: it applies fresh, and the R1 replay is frozen.
        _body, status = self.service.group_inbox_lease_renew(
            "d3", "L1", {"renewal_id": "R2"})
        self.assertEqual(status, 201)
        replay, status = self.service.group_inbox_lease_renew(
            "d3", "L1", {"renewal_id": "R1"})
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)

    def test_concurrent_identical_submissions_linearize(self) -> None:
        self.service.group_inbox_claim("d3", {"lease_id": "L1", "limit": 2})
        results = []

        def renew():
            results.append(self.service.group_inbox_lease_renew(
                "d3", "L1", {"renewal_id": "R1"}))

        with ThreadPoolExecutor(max_workers=12) as pool:
            list(pool.map(lambda _: renew(), range(12)))
        statuses = sorted(status for _body, status in results)
        self.assertEqual(statuses, [200] * 11 + [201])
        bodies = {json.dumps(body, sort_keys=True)
                  for body, _status in results}
        self.assertEqual(len(bodies), 1)
        with open(self.path, encoding="utf-8") as handle:
            document = json.load(handle)
        chains = [tuple((item["renewal_id"], item["leased_until"])
                        for item in lease["renewals"])
                  for record in document["group_delivery"]
                  for lease in record.get("leases", [])]
        self.assertTrue(chains)
        self.assertTrue(all(chain == chains[0] and len(chain) == 1
                            for chain in chains))


def _extend(until: str, seconds: int = 30) -> str:
    return (datetime.fromisoformat(until) + timedelta(seconds=seconds)) \
        .isoformat(timespec="microseconds")


class GroupLeaseRenewHTTPTest(GroupLeaseMixin, unittest.TestCase):
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

    def _request(self, target, raw):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("POST", target, body=raw,
                     headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        data = response.read().decode("utf-8")
        conn.close()
        return response.status, (json.loads(data) if data else None), data

    def _renew_path(self, device="d3", lease="L1"):
        return f"/v1/devices/{device}/group-inbox/leases/{lease}/renew"

    def test_renew_and_replay_over_http(self) -> None:
        self._request("/v1/devices/d3/group-inbox/claim",
                      json.dumps({"lease_id": "L1", "limit": 2}))
        status, body, raw = self._request(
            self._renew_path(), json.dumps({"renewal_id": "R1"}))
        self.assertEqual(status, 201)
        self.assertEqual(list(body),
                         ["device_id", "lease_id", "renewal_id",
                          "leased_until"])
        self.assertLess(raw.index('"device_id"'), raw.index('"lease_id"'))
        self.assertLess(raw.index('"lease_id"'),
                        raw.index('"renewal_id"'))
        self.assertLess(raw.index('"renewal_id"'),
                        raw.index('"leased_until"'))
        compact = raw.replace(" ", "")
        self.assertRegex(compact, r'"leased_until":"[^"]+\.\d{6}\+00:00"')
        status, replay, _ = self._request(
            self._renew_path(), json.dumps({"renewal_id": "R1"}))
        self.assertEqual(status, 200)
        self.assertEqual(replay, body)

    def test_body_validation_over_http(self) -> None:
        self._request("/v1/devices/d3/group-inbox/claim",
                      json.dumps({"lease_id": "L1", "limit": 2}))
        cases = [
            ("{bad", "request_body"),
            (json.dumps([1]), "request_body"),
            ("", "request_body"),
            (json.dumps({}), "renewal_id"),
            (json.dumps({"renewal_id": ""}), "renewal_id"),
            (json.dumps({"renewal_id": 9}), "renewal_id"),
            (json.dumps({"renewal_id": "R", "x": 1}), "x"),
        ]
        for raw, field in cases:
            status, body, _ = self._request(self._renew_path(), raw)
            self.assertEqual(status, 400, raw)
            self.assertEqual(body["field"], field, raw)

    def test_query_parameter_rejected(self) -> None:
        self._request("/v1/devices/d3/group-inbox/claim",
                      json.dumps({"lease_id": "L1", "limit": 2}))
        status, body, _ = self._request(
            self._renew_path() + "?foo=bar",
            json.dumps({"renewal_id": "R1"}))
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "query")
        # A trailing empty query string is not a parameter.
        status, _body, _ = self._request(
            self._renew_path() + "?", json.dumps({"renewal_id": "R1"}))
        self.assertEqual(status, 201)

    def test_path_segments_strictly_decoded(self) -> None:
        self._request("/v1/devices/d3/group-inbox/claim",
                      json.dumps({"lease_id": "L1", "limit": 2}))
        status, body, _ = self._request(
            self._renew_path(device="d%FF"),
            json.dumps({"renewal_id": "R1"}))
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "device_id")
        status, body, _ = self._request(
            self._renew_path(lease="L%FF"),
            json.dumps({"renewal_id": "R1"}))
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "lease_id")

    def test_encoded_slash_is_part_of_lease_id(self) -> None:
        status, _body, _ = self._request(
            "/v1/devices/d2/group-inbox/claim",
            json.dumps({"lease_id": "a/b", "limit": 1}))
        self.assertEqual(status, 201)
        status, body, _ = self._request(
            self._renew_path(device="d2", lease="a%2Fb"),
            json.dumps({"renewal_id": "R1"}))
        self.assertEqual(status, 201)
        self.assertEqual(body["lease_id"], "a/b")

    def test_unknown_and_cross_device_lease_over_http(self) -> None:
        status, body, _ = self._request(
            self._renew_path(lease="MISSING"),
            json.dumps({"renewal_id": "R1"}))
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "lease_id")
        self._request("/v1/devices/d3/group-inbox/claim",
                      json.dumps({"lease_id": "L1", "limit": 2}))
        status, body, _ = self._request(
            self._renew_path(device="d2"),
            json.dumps({"renewal_id": "R1"}))
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "lease_id")


if __name__ == "__main__":
    unittest.main()
