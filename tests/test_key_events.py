"""Tests for the per-device key-audit event chain.

Covers the append-only chain written by registration, identity rotation,
pre-key add/revoke and device revocation (service, storage and HTTP layers),
the ``GET /v1/devices/{device_id}/key-events`` pagination contract, the
``key-events`` CLI command, and the restore-time validation: field/seq/hash
chain checks plus a full replay that must reproduce the devices section.
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
from e2ee_backend.persistence import StateFileError, attach_persistence
from e2ee_backend.service import DeviceService
from e2ee_backend.storage import key_event_hash


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


def _expected_hash(event: dict) -> str:
    """Recompute an event's chain hash from its public view."""
    document = {key: value for key, value in event.items() if key != "hash"}
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class KeyEventChainTest(unittest.TestCase):
    """Service/storage-level chain appends and idempotency rules."""

    def setUp(self) -> None:
        self.service = DeviceService()
        self.identity = _raw_key_b64()
        self.pk1 = _raw_key_b64()
        self.pk2 = _raw_key_b64()
        self.service.register(_register_payload(
            identity_key=self.identity,
            prekeys=[{"key_id": "k1", "public_key": self.pk1},
                     {"key_id": "k2", "public_key": self.pk2}]))

    def _events(self, device_id: str = "d1") -> list:
        page = self.service.store.key_events_page(device_id, 0, 100)
        assert page is not None
        return page[0]

    def test_register_appends_registered_event_with_ordered_prekeys(self) -> None:
        events = self._events()
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event["device_id"], "d1")
        self.assertEqual(event["seq"], 1)
        self.assertEqual(event["type"], "registered")
        self.assertEqual(event["prev_hash"], "")
        self.assertEqual(
            event["payload"],
            {"identity_key": self.identity,
             "signed_prekeys": [{"key_id": "k1", "public_key": self.pk1},
                                {"key_id": "k2", "public_key": self.pk2}]})
        self.assertEqual(event["hash"], _expected_hash(event))
        self.assertTrue(event["created_at"].endswith("+00:00"))

    def test_failed_and_conflicting_registrations_append_nothing(self) -> None:
        for payload in ({"user_id": "u1"},  # invalid
                        _register_payload(device_id="d1")):  # duplicate id
            try:
                self.service.register(payload)
            except Exception:
                pass
        self.assertEqual(len(self._events()), 1)

    def test_rotation_appends_only_on_change(self) -> None:
        new_key = _raw_key_b64()
        self.service.rotate_identity_key("d1", {"identity_key": self.identity})
        self.assertEqual(len(self._events()), 1)  # same key: no-op
        self.service.rotate_identity_key("d1", {"identity_key": new_key})
        events = self._events()
        self.assertEqual(len(events), 2)
        event = events[1]
        self.assertEqual(event["seq"], 2)
        self.assertEqual(event["type"], "identity_rotated")
        self.assertEqual(event["payload"],
                         {"old_identity_key": self.identity,
                          "new_identity_key": new_key})
        self.assertEqual(event["prev_hash"], events[0]["hash"])
        self.assertEqual(event["hash"], _expected_hash(event))

    def test_add_prekey_appends_only_on_create(self) -> None:
        new_key = _raw_key_b64()
        self.service.add_prekey("d1", {"key_id": "k3", "public_key": new_key})
        _, status = self.service.add_prekey(
            "d1", {"key_id": "k3", "public_key": new_key})
        self.assertEqual(status, 200)  # idempotent replay
        events = self._events()
        self.assertEqual(len(events), 2)
        self.assertEqual(events[1]["type"], "prekey_added")
        self.assertEqual(events[1]["payload"],
                         {"key_id": "k3", "public_key": new_key})

    def test_prekey_revoke_appends_once(self) -> None:
        self.service.revoke_prekey("d1", "k1")
        self.service.revoke_prekey("d1", "k1")  # idempotent: no event
        events = self._events()
        self.assertEqual(len(events), 2)
        self.assertEqual(events[1]["type"], "prekey_revoked")
        self.assertEqual(events[1]["payload"],
                         {"key_id": "k1", "public_key": self.pk1})

    def test_device_revoke_appends_single_empty_payload_event(self) -> None:
        self.service.revoke_device("d1")
        self.service.revoke_device("d1")  # idempotent: no event
        events = self._events()
        self.assertEqual(len(events), 2)
        self.assertEqual(events[1]["type"], "device_revoked")
        self.assertEqual(events[1]["payload"], {})

    def test_failed_operations_append_nothing(self) -> None:
        for call in (
                lambda: self.service.rotate_identity_key(
                    "d1", {"identity_key": "not-a-key"}),
                lambda: self.service.rotate_identity_key(
                    "ghost", {"identity_key": _raw_key_b64()}),
                lambda: self.service.add_prekey(
                    "d1", {"key_id": "k1", "public_key": _raw_key_b64()}),
                lambda: self.service.revoke_prekey("d1", "ghost-key"),
                lambda: self.service.revoke_device("ghost")):
            try:
                call()
            except Exception:
                pass
        self.assertEqual(len(self._events()), 1)

    def test_chains_are_isolated_per_device(self) -> None:
        self.service.register(_register_payload(device_id="d2"))
        self.service.revoke_device("d2")
        self.assertEqual([e["type"] for e in self._events("d2")],
                         ["registered", "device_revoked"])
        self.assertEqual([e["type"] for e in self._events("d1")],
                         ["registered"])

    def test_full_chain_links_and_hashes(self) -> None:
        self.service.rotate_identity_key("d1", {"identity_key": _raw_key_b64()})
        self.service.add_prekey("d1", {"key_id": "k3",
                                       "public_key": _raw_key_b64()})
        self.service.revoke_prekey("d1", "k1")
        self.service.revoke_device("d1")
        events = self._events()
        self.assertEqual([e["seq"] for e in events], [1, 2, 3, 4, 5])
        self.assertEqual([e["type"] for e in events],
                         ["registered", "identity_rotated", "prekey_added",
                          "prekey_revoked", "device_revoked"])
        self.assertEqual(events[0]["prev_hash"], "")
        for previous, event in zip(events, events[1:]):
            self.assertEqual(event["prev_hash"], previous["hash"])
        for event in events:
            self.assertEqual(event["hash"], _expected_hash(event))
            self.assertEqual(event["hash"],
                             key_event_hash(
                                 event["device_id"], event["seq"],
                                 event["type"], event["payload"],
                                 event["prev_hash"], event["created_at"]))

    def test_unicode_payload_hashes_as_utf8_unescaped(self) -> None:
        service = DeviceService()
        service.register(_register_payload(device_id="dévice-雪"))
        page = service.store.key_events_page("dévice-雪", 0, 100)
        assert page is not None
        event = page[0][0]
        self.assertEqual(event["hash"], _expected_hash(event))


class KeyEventsHTTPTest(unittest.TestCase):
    """GET /v1/devices/{device_id}/key-events over a real loopback socket."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.service = DeviceService()
        cls.service.register(_register_payload(
            prekeys=[{"key_id": f"k{i}", "public_key": _raw_key_b64()}
                     for i in range(1, 4)]))
        cls.service.rotate_identity_key("d1", {"identity_key": _raw_key_b64()})
        cls.service.add_prekey("d1", {"key_id": "k4",
                                      "public_key": _raw_key_b64()})
        cls.service.revoke_prekey("d1", "k1")
        cls.server, _ = create_server("127.0.0.1", 0, cls.service)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever,
                                      daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def _get(self, path: str) -> tuple:
        connection = HTTPConnection("127.0.0.1", self.port)
        connection.request("GET", path)
        response = connection.getresponse()
        body = json.loads(response.read().decode("utf-8"))
        connection.close()
        return response.status, body

    def test_default_page_returns_whole_chain(self) -> None:
        status, body = self._get("/v1/devices/d1/key-events")
        self.assertEqual(status, 200)
        self.assertEqual([e["seq"] for e in body["events"]], [1, 2, 3, 4])
        self.assertEqual(body["next_after"], 4)
        self.assertFalse(body["has_more"])

    def test_pagination_after_and_limit(self) -> None:
        status, body = self._get("/v1/devices/d1/key-events?after=1&limit=2")
        self.assertEqual(status, 200)
        self.assertEqual([e["seq"] for e in body["events"]], [2, 3])
        self.assertEqual(body["next_after"], 3)
        self.assertTrue(body["has_more"])
        status, body = self._get("/v1/devices/d1/key-events?after=3&limit=2")
        self.assertEqual([e["seq"] for e in body["events"]], [4])
        self.assertEqual(body["next_after"], 4)
        self.assertFalse(body["has_more"])

    def test_empty_page_keeps_next_after(self) -> None:
        status, body = self._get("/v1/devices/d1/key-events?after=4")
        self.assertEqual(status, 200)
        self.assertEqual(body["events"], [])
        self.assertEqual(body["next_after"], 4)
        self.assertFalse(body["has_more"])

    def test_after_beyond_chain_is_empty(self) -> None:
        status, body = self._get("/v1/devices/d1/key-events?after=99")
        self.assertEqual(status, 200)
        self.assertEqual(body["events"], [])
        self.assertEqual(body["next_after"], 99)
        self.assertFalse(body["has_more"])

    def test_invalid_params_are_400_with_field(self) -> None:
        for path, field in (
                ("/v1/devices/d1/key-events?after=-1", "after"),
                ("/v1/devices/d1/key-events?after=x", "after"),
                ("/v1/devices/d1/key-events?after=1&after=2", "after"),
                ("/v1/devices/d1/key-events?limit=0", "limit"),
                ("/v1/devices/d1/key-events?limit=101", "limit"),
                ("/v1/devices/d1/key-events?limit=x", "limit")):
            status, body = self._get(path)
            self.assertEqual(status, 400, path)
            self.assertEqual(body["field"], field, path)

    def test_unknown_device_is_404(self) -> None:
        status, body = self._get("/v1/devices/ghost/key-events")
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "device_id")

    def test_revoked_device_chain_stays_readable(self) -> None:
        self.service.register(_register_payload(device_id="d9"))
        self.service.revoke_device("d9")
        status, body = self._get("/v1/devices/d9/key-events")
        self.assertEqual(status, 200)
        self.assertEqual([e["type"] for e in body["events"]],
                         ["registered", "device_revoked"])

    def test_device_id_with_slash_subpath_is_404(self) -> None:
        status, body = self._get("/v1/devices/d1/key-events/extra")
        self.assertEqual(status, 404)


class KeyEventsCLITest(unittest.TestCase):
    """The ``key-events`` CLI command against a real server."""

    def setUp(self) -> None:
        self.server, self.service = create_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        self.service.register(_register_payload())
        self.service.revoke_prekey("d1", "k1")

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _run(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "e2ee_backend", "--base-url", self.base_url,
             *arguments],
            capture_output=True, text=True, timeout=15)

    def test_key_events_prints_single_line_json(self) -> None:
        result = self._run("key-events", "d1")
        self.assertEqual(result.returncode, 0)
        body = json.loads(result.stdout.strip())
        self.assertEqual([e["seq"] for e in body["events"]], [1, 2])
        self.assertEqual(body["next_after"], 2)
        self.assertFalse(body["has_more"])

    def test_key_events_after_and_limit_flags(self) -> None:
        result = self._run("key-events", "d1", "--after", "1", "--limit", "1")
        self.assertEqual(result.returncode, 0)
        body = json.loads(result.stdout.strip())
        self.assertEqual([e["seq"] for e in body["events"]], [2])
        self.assertFalse(body["has_more"])

    def test_key_events_unknown_device_exits_nonzero(self) -> None:
        result = self._run("key-events", "ghost")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout.strip(), "")
        body = json.loads(result.stderr.strip())
        self.assertEqual(body["field"], "device_id")

    def test_key_events_server_down_reports_field_server(self) -> None:
        dead = subprocess.run(
            [sys.executable, "-m", "e2ee_backend",
             "--base-url", "http://127.0.0.1:1", "key-events", "d1"],
            capture_output=True, text=True, timeout=15)
        self.assertEqual(dead.returncode, 1)
        body = json.loads(dead.stderr.strip())
        self.assertEqual(body["field"], "server")
        self.assertNotIn("Traceback", dead.stderr)


class KeyEventsPersistenceTest(unittest.TestCase):
    """Restart round-trip and restore-time chain validation."""

    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.path = os.path.join(self.directory, "state.json")
        self.service = DeviceService()
        attach_persistence(self.service, self.path)
        self.identity = _raw_key_b64()
        self.service.register(_register_payload(
            identity_key=self.identity,
            prekeys=[{"key_id": "k1", "public_key": _raw_key_b64()},
                     {"key_id": "k2", "public_key": _raw_key_b64()}]))
        self.new_identity = _raw_key_b64()
        self.service.rotate_identity_key(
            "d1", {"identity_key": self.new_identity})
        self.service.add_prekey("d1", {"key_id": "k3",
                                       "public_key": _raw_key_b64()})
        self.service.revoke_prekey("d1", "k1")

    def _document(self) -> dict:
        with open(self.path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _write_document(self, document: dict) -> None:
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)

    def _restart(self) -> DeviceService:
        service = DeviceService()
        attach_persistence(service, self.path)
        return service

    def test_restart_preserves_the_chain(self) -> None:
        service = self._restart()
        page = service.store.key_events_page("d1", 0, 100)
        assert page is not None
        events, next_after, has_more = page
        self.assertEqual([e["type"] for e in events],
                         ["registered", "identity_rotated", "prekey_added",
                          "prekey_revoked"])
        self.assertEqual(next_after, 4)
        self.assertFalse(has_more)
        for event in events:
            self.assertEqual(event["hash"], _expected_hash(event))

    def test_missing_section_loads_as_empty(self) -> None:
        document = self._document()
        del document["key_events"]
        # A legacy file predates the integrity-log marker/sidecar too.
        del document["integrity_log_version"]
        os.unlink(self.path + ".integrity")
        self._write_document(document)
        service = self._restart()
        page = service.store.key_events_page("d1", 0, 100)
        assert page is not None
        self.assertEqual(page[0], [])

    def _assert_refused(self, document: dict) -> None:
        """A contradictory document refuses startup; the file is untouched."""
        before = json.dumps(document, sort_keys=True)
        self._write_document(document)
        with self.assertRaises(StateFileError):
            self._restart()
        with open(self.path, "r", encoding="utf-8") as handle:
            self.assertEqual(json.dumps(json.load(handle), sort_keys=True),
                             before)

    def test_tampered_hash_is_refused(self) -> None:
        document = self._document()
        document["key_events"][1]["hash"] = "0" * 64
        self._assert_refused(document)

    def test_tampered_payload_is_refused(self) -> None:
        document = self._document()
        document["key_events"][0]["payload"]["identity_key"] = _raw_key_b64()
        self._assert_refused(document)

    def test_broken_prev_hash_link_is_refused(self) -> None:
        document = self._document()
        event = document["key_events"][2]
        event["prev_hash"] = "f" * 64
        event["hash"] = _expected_hash(event)
        self._assert_refused(document)

    def test_non_consecutive_seq_is_refused(self) -> None:
        document = self._document()
        event = document["key_events"][2]
        event["seq"] = 7
        event["hash"] = _expected_hash(event)
        self._assert_refused(document)

    def test_unknown_event_type_is_refused(self) -> None:
        document = self._document()
        event = document["key_events"][1]
        event["type"] = "compromised"
        event["hash"] = _expected_hash(event)
        self._assert_refused(document)

    def test_event_for_unknown_device_is_refused(self) -> None:
        document = self._document()
        event = document["key_events"][0]
        event["device_id"] = "ghost"
        event["hash"] = _expected_hash(event)
        self._assert_refused(document)

    def test_missing_chain_for_a_device_is_refused(self) -> None:
        document = self._document()
        document["devices"].append({
            "user_id": "u1", "device_id": "d2",
            "identity_key": _raw_key_b64(),
            "registered_at": "2026-01-01T00:00:00+00:00",
            "revoked": False, "prekeys": []})
        self._assert_refused(document)

    def test_identity_mismatch_with_devices_section_is_refused(self) -> None:
        document = self._document()
        # The chain replays to new_identity; claim the device never rotated.
        document["devices"][0]["identity_key"] = self.identity
        self._assert_refused(document)

    def test_prekey_order_mismatch_is_refused(self) -> None:
        document = self._document()
        prekeys = document["devices"][0]["prekeys"]
        prekeys[1], prekeys[2] = prekeys[2], prekeys[1]
        self._assert_refused(document)

    def test_revocation_mismatch_is_refused(self) -> None:
        document = self._document()
        # The chain revokes k1; the devices section must agree.
        k1 = next(pk for pk in document["devices"][0]["prekeys"]
                  if pk["key_id"] == "k1")
        k1["revoked"] = False
        self._assert_refused(document)

    def test_event_after_device_revocation_is_refused(self) -> None:
        document = self._document()
        tail = document["key_events"][-1]
        first = {
            "device_id": "d1", "seq": 5, "type": "device_revoked",
            "payload": {}, "prev_hash": tail["hash"],
            "created_at": tail["created_at"],
        }
        first["hash"] = _expected_hash(first)
        # A second device_revoked follows the revocation — a no-op the live
        # server never appends, so the chain is contradictory.
        second = {
            "device_id": "d1", "seq": 6, "type": "device_revoked",
            "payload": {}, "prev_hash": first["hash"],
            "created_at": tail["created_at"],
        }
        second["hash"] = _expected_hash(second)
        document["key_events"].extend([first, second])
        document["devices"][0]["revoked"] = True
        for pk in document["devices"][0]["prekeys"]:
            pk["revoked"] = True
        self._assert_refused(document)

    def test_device_revoked_roundtrip(self) -> None:
        self.service.revoke_device("d1")
        service = self._restart()
        page = service.store.key_events_page("d1", 0, 100)
        assert page is not None
        self.assertEqual(page[0][-1]["type"], "device_revoked")
        self.assertEqual(page[0][-1]["payload"], {})


class KeyEventsRollbackTest(unittest.TestCase):
    """A failed durable write rolls the chain back with the rest (503)."""

    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.path = os.path.join(self.directory, "state.json")
        self.service = DeviceService()
        self.state_store = attach_persistence(self.service, self.path)
        self.service.register(_register_payload())
        with open(self.path, "rb") as handle:
            self.good_bytes = handle.read()

    def _fail_writes(self) -> None:
        def raise_oserror(state):  # noqa: ANN001 - mimics JsonStateStore.save
            raise OSError("simulated disk failure")

        self.state_store.save = raise_oserror  # type: ignore[assignment]

    def test_failed_write_appends_no_event_and_answers_503(self) -> None:
        from e2ee_backend.persistence import PersistenceUnavailable

        self._fail_writes()
        with self.assertRaises(PersistenceUnavailable):
            self.service.revoke_prekey("d1", "k1")
        # The chain and the key both rolled back; the file is untouched.
        page = self.service.store.key_events_page("d1", 0, 100)
        assert page is not None
        self.assertEqual([e["type"] for e in page[0]], ["registered"])
        device = self.service.store.find_by_device_id("d1")
        assert device is not None
        self.assertFalse(device.prekeys[0].revoked)
        with open(self.path, "rb") as handle:
            self.assertEqual(handle.read(), self.good_bytes)

    def test_http_503_names_data_file(self) -> None:
        self._fail_writes()
        server, _ = create_server("127.0.0.1", 0, self.service)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            port = server.server_address[1]
            connection = HTTPConnection("127.0.0.1", port)
            connection.request("POST", "/v1/devices/d1/prekeys/k1/revoke")
            response = connection.getresponse()
            body = json.loads(response.read().decode("utf-8"))
            connection.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
        self.assertEqual(response.status, 503)
        self.assertEqual(body["field"], "data_file")


if __name__ == "__main__":
    unittest.main()
