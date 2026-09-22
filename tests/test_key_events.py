"""Tests for the per-device hash-chained key audit log.

Covers the chain contents (service/store), the GET route (real loopback
socket), the CLI (real subprocess) and persistence/recovery:

* GET /v1/devices/{device_id}/key-events
* CLI ``key-events DEVICE_ID [--after N] [--limit N]``
* the optional ``key_events`` v1 state section
"""
import base64
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from http.client import HTTPConnection

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import x25519

from e2ee_backend.http_app import create_server
from e2ee_backend.key_events import compute_event_hash
from e2ee_backend.persistence import (
    PersistenceUnavailable,
    StateFileError,
    attach_persistence,
)
from e2ee_backend.service import DeviceService, ServiceError


def _raw_key_b64() -> str:
    key = x25519.X25519PrivateKey.generate().public_key()
    raw = key.public_bytes(serialization.Encoding.Raw,
                           serialization.PublicFormat.Raw)
    return base64.b64encode(raw).decode()


def _register_payload(device_id: str = "d1", user_id: str = "u1",
                      identity_key: str | None = None,
                      prekeys: list | None = None) -> dict:
    return {
        "user_id": user_id,
        "device_id": device_id,
        "identity_key": identity_key or _raw_key_b64(),
        "signed_prekeys": (prekeys if prekeys is not None
                           else [{"key_id": "k1", "public_key": _raw_key_b64()}]),
    }


def _independent_hash(event: dict) -> str:
    """Recompute an event hash the spec way, independently of the package."""
    body = {name: event[name] for name in (
        "device_id", "seq", "type", "payload", "prev_hash", "created_at")}
    canonical = json.dumps(
        body, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


class KeyEventChainServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = DeviceService()
        self.key_a = _raw_key_b64()
        self.key_b = _raw_key_b64()
        self.identity = _raw_key_b64()
        self.service.register(_register_payload(
            device_id="d1", identity_key=self.identity,
            prekeys=[{"key_id": "k1", "public_key": self.key_a},
                     {"key_id": "k2", "public_key": self.key_b}]))

    def _events(self, after: int = 0, limit: int = 100) -> list:
        return self.service.list_key_events("d1", after, limit)["events"]

    def test_registration_starts_the_chain(self) -> None:
        event = self._events()[0]
        self.assertEqual(set(event),
                         {"device_id", "seq", "type", "payload", "prev_hash",
                          "hash", "created_at"})
        self.assertEqual(event["device_id"], "d1")
        self.assertEqual(event["seq"], 1)
        self.assertEqual(event["type"], "registered")
        self.assertEqual(event["prev_hash"], "")
        self.assertEqual(event["payload"],
                         {"prekeys": [
                             {"key_id": "k1", "public_key": self.key_a},
                             {"key_id": "k2", "public_key": self.key_b}]})
        self.assertTrue(event["created_at"].endswith("+00:00"))
        self.assertEqual(event["hash"], _independent_hash(event))

    def test_rotation_appends_old_and_new_identity(self) -> None:
        new_identity = _raw_key_b64()
        self.service.rotate_identity_key("d1", {"identity_key": new_identity})
        event = self._events()[-1]
        self.assertEqual(event["type"], "identity_rotated")
        self.assertEqual(event["payload"], {
            "old_identity_key": self.identity,
            "new_identity_key": new_identity})
        self.assertEqual(event["hash"], _independent_hash(event))

    def test_same_key_rotation_appends_nothing(self) -> None:
        before = len(self._events())
        self.service.rotate_identity_key(
            "d1", {"identity_key": self.identity})
        self.assertEqual(len(self._events()), before)

    def test_prekey_added_event(self) -> None:
        key_c = _raw_key_b64()
        self.service.add_prekey("d1", {"key_id": "k3", "public_key": key_c})
        event = self._events()[-1]
        self.assertEqual(event["type"], "prekey_added")
        self.assertEqual(event["payload"],
                         {"key_id": "k3", "public_key": key_c})

    def test_idempotent_prekey_add_appends_nothing(self) -> None:
        before = len(self._events())
        self.service.add_prekey("d1", {"key_id": "k1",
                                       "public_key": self.key_a})
        self.assertEqual(len(self._events()), before)

    def test_prekey_revoked_event_and_idempotent_revoke(self) -> None:
        self.service.revoke_prekey("d1", "k1")
        events = self._events()
        self.assertEqual(events[-1]["type"], "prekey_revoked")
        self.assertEqual(events[-1]["payload"],
                         {"key_id": "k1", "public_key": self.key_a})
        count_after_first = len(events)
        self.service.revoke_prekey("d1", "k1")
        self.assertEqual(len(self._events()), count_after_first)

    def test_device_revoked_event_has_empty_payload_and_is_idempotent(self) -> None:
        self.service.revoke_device("d1")
        events = self._events()
        self.assertEqual(events[-1]["type"], "device_revoked")
        self.assertEqual(events[-1]["payload"], {})
        count_after_first = len(events)
        self.service.revoke_device("d1")
        self.assertEqual(len(self._events()), count_after_first)

    def test_failed_mutations_append_nothing(self) -> None:
        before = len(self._events())
        with self.assertRaises(ServiceError):
            self.service.rotate_identity_key(
                "d1", {"identity_key": "not-a-public-key"})
        with self.assertRaises(ServiceError):
            self.service.revoke_prekey("d1", "missing-key")
        with self.assertRaises(ServiceError):
            self.service.register(_register_payload(device_id="d1"))
        self.assertEqual(len(self._events()), before)

    def test_seq_starts_at_one_and_links_prev_hash(self) -> None:
        self.service.add_prekey("d1", {"key_id": "k3",
                                      "public_key": _raw_key_b64()})
        self.service.revoke_prekey("d1", "k1")
        self.service.rotate_identity_key("d1",
                                         {"identity_key": _raw_key_b64()})
        events = self._events()
        for index, event in enumerate(events):
            self.assertEqual(event["seq"], index + 1)
            if index == 0:
                self.assertEqual(event["prev_hash"], "")
            else:
                self.assertEqual(event["prev_hash"], events[index - 1]["hash"])
            self.assertEqual(event["hash"], _independent_hash(event))

    def test_chains_are_isolated_per_device(self) -> None:
        self.service.register(_register_payload(
            device_id="d2", identity_key=_raw_key_b64(),
            prekeys=[{"key_id": "p1", "public_key": _raw_key_b64()}]))
        for device_id in ("d1", "d2"):
            chain = self.service.list_key_events(device_id, 0, 100)["events"]
            self.assertEqual([e["seq"] for e in chain],
                             list(range(1, len(chain) + 1)))
            self.assertEqual(chain[0]["prev_hash"], "")
            self.assertTrue(all(e["device_id"] == device_id for e in chain))


class KeyEventPageServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = DeviceService()
        self.service.register(_register_payload(
            device_id="d1",
            prekeys=[{"key_id": "k1", "public_key": _raw_key_b64()}]))
        for index in range(4):
            self.service.add_prekey(
                "d1", {"key_id": f"k{index + 2}",
                       "public_key": _raw_key_b64()})

    def test_paging_after_and_limit(self) -> None:
        first = self.service.list_key_events("d1", 0, 2)
        self.assertEqual([e["seq"] for e in first["events"]], [1, 2])
        self.assertEqual(first["next_after"], 2)
        self.assertTrue(first["has_more"])

        second = self.service.list_key_events("d1", 2, 2)
        self.assertEqual([e["seq"] for e in second["events"]], [3, 4])
        self.assertEqual(second["next_after"], 4)
        self.assertTrue(second["has_more"])

        tail = self.service.list_key_events("d1", 4, 2)
        self.assertEqual([e["seq"] for e in tail["events"]], [5])
        self.assertEqual(tail["next_after"], 5)
        self.assertFalse(tail["has_more"])

    def test_empty_page_next_after_equals_after(self) -> None:
        page = self.service.list_key_events("d1", 99, 100)
        self.assertEqual(page["events"], [])
        self.assertEqual(page["next_after"], 99)
        self.assertFalse(page["has_more"])

    def test_unknown_device_is_404(self) -> None:
        with self.assertRaises(ServiceError) as ctx:
            self.service.list_key_events("ghost", 0, 100)
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.field, "device_id")

    def test_revoked_device_remains_readable(self) -> None:
        self.service.revoke_device("d1")
        page = self.service.list_key_events("d1", 0, 100)
        self.assertTrue(page["events"])
        self.assertEqual(page["events"][-1]["type"], "device_revoked")

    def test_bad_after_and_limit(self) -> None:
        for after, limit, field in (
                (-1, 100, "after"),
                (True, 100, "after"),
                (0, 0, "limit"),
                (0, 101, "limit"),
                ("1", 100, "after")):
            with self.subTest(after=after, limit=limit):
                with self.assertRaises(ServiceError) as ctx:
                    self.service.list_key_events("d1", after, limit)
                self.assertEqual(ctx.exception.field, field)


class KeyEventHTTPTest(unittest.TestCase):
    def setUp(self) -> None:
        self.server, self.service = create_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        self.service.register(_register_payload(device_id="d1"))
        self.service.add_prekey("d1", {"key_id": "k2",
                                       "public_key": _raw_key_b64()})

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _request(self, path: str):
        connection = HTTPConnection("127.0.0.1", self.port, timeout=5)
        connection.request("GET", path)
        response = connection.getresponse()
        data = response.read().decode("utf-8")
        connection.close()
        return response.status, json.loads(data)

    def test_default_page(self) -> None:
        status, body = self._request("/v1/devices/d1/key-events")
        self.assertEqual(status, 200)
        self.assertEqual([e["seq"] for e in body["events"]], [1, 2])
        self.assertEqual(body["next_after"], 2)
        self.assertFalse(body["has_more"])
        self.assertEqual(set(body["events"][0]),
                         {"device_id", "seq", "type", "payload", "prev_hash",
                          "hash", "created_at"})

    def test_after_and_limit_query_params(self) -> None:
        status, body = self._request("/v1/devices/d1/key-events?after=0&limit=1")
        self.assertEqual(status, 200)
        self.assertEqual([e["seq"] for e in body["events"]], [1])
        self.assertEqual(body["next_after"], 1)
        self.assertTrue(body["has_more"])

    def test_bad_after_and_limit_are_400_with_field(self) -> None:
        for query, field in (
                ("after=-1", "after"),
                ("after=x", "after"),
                ("after=1&after=2", "after"),
                ("limit=0", "limit"),
                ("limit=101", "limit"),
                ("limit=x", "limit")):
            with self.subTest(query=query):
                status, body = self._request(
                    f"/v1/devices/d1/key-events?{query}")
                self.assertEqual(status, 400)
                self.assertEqual(body["field"], field)

    def test_unknown_device_is_404(self) -> None:
        status, body = self._request("/v1/devices/ghost/key-events")
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "device_id")

    def test_revoked_device_chain_is_readable(self) -> None:
        self.service.revoke_device("d1")
        status, body = self._request("/v1/devices/d1/key-events")
        self.assertEqual(status, 200)
        self.assertEqual(body["events"][-1]["type"], "device_revoked")


class KeyEventCLITest(unittest.TestCase):
    def setUp(self) -> None:
        self.server, service = create_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        service.register(_register_payload(device_id="d1"))

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _run(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "e2ee_backend", "--base-url",
             self.base_url, *arguments],
            capture_output=True, text=True, timeout=15)

    def test_key_events_success(self) -> None:
        result = self._run("key-events", "d1", "--after", "0", "--limit", "10")
        self.assertEqual(result.returncode, 0, result.stderr)
        body = json.loads(result.stdout.strip())
        self.assertEqual([e["seq"] for e in body["events"]], [1])
        self.assertEqual(body["next_after"], 1)
        self.assertFalse(body["has_more"])

    def test_key_events_unknown_device_nonzero_stderr(self) -> None:
        result = self._run("key-events", "ghost")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(json.loads(result.stderr.strip())["field"],
                         "device_id")

    def test_key_events_bad_limit_nonzero(self) -> None:
        result = self._run("key-events", "d1", "--limit", "0")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(json.loads(result.stderr.strip())["field"], "limit")

    def test_key_events_server_unreachable(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "e2ee_backend", "--base-url",
             "http://127.0.0.1:1", "key-events", "d1"],
            capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(json.loads(result.stderr.strip())["field"], "server")


class KeyEventPersistenceTest(unittest.TestCase):
    def _build_history(self, path: str) -> DeviceService:
        service = DeviceService()
        attach_persistence(service, path)
        self.identity = _raw_key_b64()
        self.key_a = _raw_key_b64()
        service.register(_register_payload(
            device_id="d1", identity_key=self.identity,
            prekeys=[{"key_id": "k1", "public_key": self.key_a}]))
        self.new_identity = _raw_key_b64()
        service.rotate_identity_key(
            "d1", {"identity_key": self.new_identity})
        self.key_b = _raw_key_b64()
        service.add_prekey("d1", {"key_id": "k2", "public_key": self.key_b})
        service.revoke_prekey("d1", "k1")
        return service

    def test_chain_survives_restart(self) -> None:
        path = tempfile.mktemp(suffix=".json")
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        self._build_history(path)
        restarted = DeviceService()
        attach_persistence(restarted, path)
        page = restarted.list_key_events("d1", 0, 100)
        self.assertEqual(
            [(e["seq"], e["type"]) for e in page["events"]],
            [(1, "registered"), (2, "identity_rotated"),
             (3, "prekey_added"), (4, "prekey_revoked")])
        for event in page["events"]:
            self.assertEqual(event["hash"], _independent_hash(event))
        device = restarted.store.find_by_device_id("d1")
        self.assertEqual(device.identity_key, self.new_identity)
        self.assertEqual([pk.key_id for pk in device.prekeys], ["k1", "k2"])
        self.assertTrue(device.prekeys[0].revoked)
        self.assertFalse(device.prekeys[1].revoked)

    def test_legacy_doc_without_key_events_loads_empty(self) -> None:
        path = tempfile.mktemp(suffix=".json")
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({
                "version": 1,
                "devices": [{
                    "user_id": "u1", "device_id": "d1",
                    "identity_key": self.key_a if hasattr(self, "key_a")
                    else _raw_key_b64(),
                    "registered_at": "2026-01-01T00:00:00+00:00",
                    "revoked": False, "prekeys": []}],
                "sessions": [], "messages": {}, "delivery": [],
            }, handle)
        service = DeviceService()
        attach_persistence(service, path)  # must not raise
        self.assertEqual(service.list_key_events("d1", 0, 100)["events"], [])

    def test_legacy_doc_keeps_auditing_disabled_across_mutations(self) -> None:
        path = tempfile.mktemp(suffix=".json")
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({
                "version": 1,
                "devices": [{
                    "user_id": "u1", "device_id": "d1",
                    "identity_key": _raw_key_b64(),
                    "registered_at": "2026-01-01T00:00:00+00:00",
                    "revoked": False,
                    "prekeys": [{"key_id": "k1", "public_key": _raw_key_b64(),
                                 "revoked": False, "consumed": False}]}],
                "sessions": [], "messages": {}, "delivery": [],
            }, handle)
        service = DeviceService()
        attach_persistence(service, path)
        # A new key after upgrading from a section-less file appends no chain
        # (the registered head is missing); the file must stay reloadable.
        service.add_prekey("d1", {"key_id": "k2", "public_key": _raw_key_b64()})
        with open(path, "r", encoding="utf-8") as handle:
            document = json.load(handle)
        self.assertNotIn("key_events", document)
        restarted = DeviceService()
        attach_persistence(restarted, path)  # the written file reloads cleanly
        self.assertEqual(restarted.get_device("d1")["prekey_ids"],
                         ["k1", "k2"])
        self.assertEqual(
            restarted.list_key_events("d1", 0, 100)["events"], [])

    def test_tampered_payload_is_refused(self) -> None:
        path = tempfile.mktemp(suffix=".json")
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        self._build_history(path)
        with open(path, "r", encoding="utf-8") as handle:
            document = json.load(handle)
        document["key_events"][2]["payload"]["key_id"] = "k9"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        with self.assertRaises(StateFileError):
            attach_persistence(DeviceService(), path)

    def test_broken_prev_hash_is_refused(self) -> None:
        path = tempfile.mktemp(suffix=".json")
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        self._build_history(path)
        with open(path, "r", encoding="utf-8") as handle:
            document = json.load(handle)
        document["key_events"][1]["prev_hash"] = "deadbeef"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        with self.assertRaises(StateFileError):
            attach_persistence(DeviceService(), path)

    def test_seq_gap_is_refused(self) -> None:
        path = tempfile.mktemp(suffix=".json")
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        self._build_history(path)
        with open(path, "r", encoding="utf-8") as handle:
            document = json.load(handle)
        # Re-signing the remainder is impossible without the old hashes, so a
        # dropped event necessarily breaks both the seq chain and the hashes.
        document["key_events"].pop(2)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        with self.assertRaises(StateFileError):
            attach_persistence(DeviceService(), path)

    def test_missing_chain_for_a_device_is_refused(self) -> None:
        path = tempfile.mktemp(suffix=".json")
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        self._build_history(path)
        service = DeviceService()
        attach_persistence(service, path)
        service.register(_register_payload(
            device_id="d2", identity_key=_raw_key_b64(),
            prekeys=[{"key_id": "p1", "public_key": _raw_key_b64()}]))
        with open(path, "r", encoding="utf-8") as handle:
            document = json.load(handle)
        document["key_events"] = [
            event for event in document["key_events"]
            if event["device_id"] != "d2"]
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        with self.assertRaises(StateFileError):
            attach_persistence(DeviceService(), path)

    def test_replay_public_key_mismatch_is_refused(self) -> None:
        path = tempfile.mktemp(suffix=".json")
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        self._build_history(path)
        with open(path, "r", encoding="utf-8") as handle:
            document = json.load(handle)
        # Change a pre-key's stored public key: the registered event replays a
        # different key, so the device section contradicts the audit chain.
        document["devices"][0]["prekeys"][0]["public_key"] = _raw_key_b64()
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        with self.assertRaises(StateFileError):
            attach_persistence(DeviceService(), path)

    def test_rollback_keeps_chain_and_file_consistent(self) -> None:
        path = tempfile.mktemp(suffix=".json")
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        service = DeviceService()
        state_store = attach_persistence(service, path)
        service.register(_register_payload(
            device_id="d1", identity_key=_raw_key_b64(),
            prekeys=[{"key_id": "k1", "public_key": _raw_key_b64()}]))

        def raise_oserror(state):  # noqa: ANN001 - mimics JsonStateStore.save
            raise OSError("simulated disk failure")

        state_store.save = raise_oserror  # type: ignore[assignment]
        with self.assertRaises(PersistenceUnavailable):
            service.add_prekey("d1", {"key_id": "k2",
                                      "public_key": _raw_key_b64()})
        del state_store.save
        # The rolled-back append left neither a pre-key nor an audit event,
        # and the file re-loads with a consistent chain.
        restarted = DeviceService()
        attach_persistence(restarted, path)
        seqs = [e["seq"] for e in
                restarted.list_key_events("d1", 0, 100)["events"]]
        self.assertEqual(seqs, [1])
        self.assertEqual(restarted.get_device("d1")["prekey_ids"], ["k1"])


class CanonicalHashTest(unittest.TestCase):
    def test_hash_is_lowercase_hex_sha256_of_canonical_json(self) -> None:
        event = {
            "device_id": "d1", "seq": 1, "type": "registered",
            "payload": {"prekeys": []}, "prev_hash": "",
            "created_at": "2026-01-01T00:00:00+00:00"}
        digest = compute_event_hash(event)
        canonical = json.dumps(
            event, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False).encode("utf-8")
        self.assertEqual(digest, hashlib.sha256(canonical).hexdigest())
        self.assertRegex(digest, r"^[0-9a-f]{64}$")

    def test_hash_field_is_excluded(self) -> None:
        event = {"device_id": "d1", "seq": 1, "type": "registered",
                 "payload": {}, "prev_hash": "",
                 "hash": "ignored", "created_at": "2026-01-01T00:00:00+00:00"}
        without_hash = dict(event)
        del without_hash["hash"]
        self.assertEqual(compute_event_hash(event),
                         compute_event_hash(without_hash))


if __name__ == "__main__":
    unittest.main()
