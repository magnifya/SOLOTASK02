"""End-to-end tests for the HTTP API over a real loopback socket."""
import base64
import json
import threading
import unittest
from http.client import HTTPConnection
from urllib.parse import quote

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import x25519

from e2ee_backend.http_app import create_server


def _raw_key_b64() -> str:
    key = x25519.X25519PrivateKey.generate().public_key()
    raw = key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return base64.b64encode(raw).decode()


class HTTPApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.server, _ = create_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _request(self, method: str, path: str, body: object = None):
        connection = HTTPConnection("127.0.0.1", self.port, timeout=5)
        payload = json.dumps(body) if body is not None else None
        headers = {"Content-Type": "application/json"} if payload is not None else {}
        connection.request(method, path, body=payload, headers=headers)
        response = connection.getresponse()
        data = response.read().decode("utf-8")
        connection.close()
        return response.status, json.loads(data)

    def _payload(self, device_id: str = "d1", user_id: str = "u1") -> dict:
        return {
            "user_id": user_id,
            "device_id": device_id,
            "identity_key": _raw_key_b64(),
            "signed_prekeys": [
                {"key_id": "k1", "public_key": _raw_key_b64()},
                {"key_id": "k2", "public_key": _raw_key_b64()},
            ],
        }

    def test_register_success_is_201(self) -> None:
        status, body = self._request("POST", "/v1/devices", self._payload())
        self.assertEqual(status, 201)
        self.assertEqual(set(body), {"device_id", "registered_at"})
        self.assertEqual(body["device_id"], "d1")

    def test_duplicate_is_409(self) -> None:
        payload = self._payload()
        self.assertEqual(self._request("POST", "/v1/devices", payload)[0], 201)
        status, body = self._request("POST", "/v1/devices", payload)
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "device_id")

    def test_get_device_200_with_stable_order(self) -> None:
        self._request("POST", "/v1/devices", self._payload())
        status, first = self._request("GET", "/v1/devices/d1")
        self.assertEqual(status, 200)
        self.assertEqual(set(first),
                         {"identity_key", "prekey_ids", "registered_at"})
        self.assertEqual(first["prekey_ids"], ["k1", "k2"])
        _, second = self._request("GET", "/v1/devices/d1")
        self.assertEqual(first, second)

    def test_get_missing_device_is_404(self) -> None:
        status, body = self._request("GET", "/v1/devices/ghost")
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "device_id")

    def test_missing_field_is_400_and_names_it(self) -> None:
        payload = self._payload(device_id="d2")
        del payload["identity_key"]
        status, body = self._request("POST", "/v1/devices", payload)
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "identity_key")

    def test_bad_prekey_element_is_400_and_names_path(self) -> None:
        payload = self._payload(device_id="d3")
        payload["signed_prekeys"] = [{"key_id": "k1"}]
        status, body = self._request("POST", "/v1/devices", payload)
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "signed_prekeys[0].public_key")

    def test_body_not_json_is_400(self) -> None:
        connection = HTTPConnection("127.0.0.1", self.port, timeout=5)
        connection.request("POST", "/v1/devices", body=b"{not json",
                           headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        body = json.loads(response.read())
        connection.close()
        self.assertEqual(response.status, 400)
        self.assertEqual(body["field"], "request_body")

    def test_unknown_route_is_404(self) -> None:
        status, _ = self._request("GET", "/v1/nope")
        self.assertEqual(status, 404)

    def test_devices_with_same_user_are_independent(self) -> None:
        self._request("POST", "/v1/devices", self._payload("d1", "u1"))
        self._request("POST", "/v1/devices", self._payload("d2", "u1"))
        _, first = self._request("GET", "/v1/devices/d1")
        _, second = self._request("GET", "/v1/devices/d2")
        self.assertEqual(first["prekey_ids"], ["k1", "k2"])
        self.assertEqual(second["prekey_ids"], ["k1", "k2"])

    def test_url_encoded_device_id(self) -> None:
        payload = self._payload(device_id="dev x/y", user_id="u9")
        status, body = self._request("POST", "/v1/devices", payload)
        self.assertEqual(status, 201)
        status, body = self._request("GET", f"/v1/devices/{quote('dev x/y', safe='')}")
        self.assertEqual(status, 200)
        self.assertIn("prekey_ids", body)


class HTTPRevokeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.server, _ = create_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _request(self, method: str, path: str, body: object = None):
        connection = HTTPConnection("127.0.0.1", self.port, timeout=5)
        payload = json.dumps(body) if body is not None else None
        headers = {"Content-Type": "application/json"} if payload is not None else {}
        connection.request(method, path, body=payload, headers=headers)
        response = connection.getresponse()
        data = response.read().decode("utf-8")
        connection.close()
        return response.status, json.loads(data)

    @staticmethod
    def _payload(device_id: str = "d1", user_id: str = "u1") -> dict:
        return {
            "user_id": user_id,
            "device_id": device_id,
            "identity_key": _raw_key_b64(),
            "signed_prekeys": [
                {"key_id": "k1", "public_key": _raw_key_b64()},
                {"key_id": "k2", "public_key": _raw_key_b64()},
            ],
        }

    def _register(self, device_id: str = "d1", user_id: str = "u1") -> None:
        status, _ = self._request("POST", "/v1/devices",
                                  self._payload(device_id, user_id))
        self.assertEqual(status, 201)

    def test_revoke_device_200_and_idempotent(self) -> None:
        self._register()
        status, body = self._request("POST", "/v1/devices/d1/revoke")
        self.assertEqual(status, 200)
        self.assertEqual(body, {"device_id": "d1", "revoked": True})
        status, again = self._request("POST", "/v1/devices/d1/revoke")
        self.assertEqual(status, 200)
        self.assertEqual(again, body)

    def test_revoke_unknown_device_is_404(self) -> None:
        status, body = self._request("POST", "/v1/devices/ghost/revoke")
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "device_id")

    def test_revoked_device_get_lists_empty_keeps_rest(self) -> None:
        self._register()
        before_status, before = self._request("GET", "/v1/devices/d1")
        self.assertEqual(before_status, 200)
        self.assertEqual(before["prekey_ids"], ["k1", "k2"])
        self.assertEqual(self._request("POST", "/v1/devices/d1/revoke")[0], 200)
        status, after = self._request("GET", "/v1/devices/d1")
        self.assertEqual(status, 200)
        self.assertEqual(after["prekey_ids"], [])
        self.assertEqual(after["identity_key"], before["identity_key"])
        self.assertEqual(after["registered_at"], before["registered_at"])

    def test_revoke_prekey_200_and_idempotent(self) -> None:
        self._register()
        status, body = self._request(
            "POST", "/v1/devices/d1/prekeys/k1/revoke")
        self.assertEqual(status, 200)
        self.assertEqual(body, {"device_id": "d1", "key_id": "k1",
                                "revoked": True})
        status, again = self._request(
            "POST", "/v1/devices/d1/prekeys/k1/revoke")
        self.assertEqual(status, 200)
        self.assertEqual(again, body)

    def test_revoke_prekey_excludes_only_target(self) -> None:
        self._register("d1", "u1")
        self._register("d2", "u1")
        self.assertEqual(
            self._request("POST", "/v1/devices/d1/prekeys/k1/revoke")[0], 200)
        _, d1 = self._request("GET", "/v1/devices/d1")
        _, d2 = self._request("GET", "/v1/devices/d2")
        self.assertEqual(d1["prekey_ids"], ["k2"])
        self.assertEqual(d2["prekey_ids"], ["k1", "k2"])

    def test_revoke_prekey_unknown_device_404_field_device(self) -> None:
        status, body = self._request(
            "POST", "/v1/devices/ghost/prekeys/k1/revoke")
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "device_id")

    def test_revoke_prekey_unknown_key_404_field_key(self) -> None:
        self._register()
        status, body = self._request(
            "POST", "/v1/devices/d1/prekeys/nope/revoke")
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "key_id")
        _, d1 = self._request("GET", "/v1/devices/d1")
        self.assertEqual(d1["prekey_ids"], ["k1", "k2"])

    def test_malformed_revoke_routes_are_404(self) -> None:
        status, body = self._request("POST", "/v1/devices/d1/prekeys/revoke")
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "device_id")
        status, body = self._request(
            "POST", "/v1/devices/d1/prekeys/k1/extra/revoke")
        self.assertEqual(status, 404)

    def test_revoke_prekey_url_encoded_device_id(self) -> None:
        device_id = "dev x/y"
        self._register(device_id, "u9")
        encoded = quote(device_id, safe="")
        status, body = self._request(
            "POST", f"/v1/devices/{encoded}/prekeys/k1/revoke")
        self.assertEqual(status, 200)
        self.assertEqual(body["device_id"], device_id)


class HTTPSessionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.server, _ = create_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _request(self, method: str, path: str, body: object = None):
        connection = HTTPConnection("127.0.0.1", self.port, timeout=5)
        payload = json.dumps(body) if body is not None else None
        headers = {"Content-Type": "application/json"} if payload is not None else {}
        connection.request(method, path, body=payload, headers=headers)
        response = connection.getresponse()
        data = response.read().decode("utf-8")
        connection.close()
        return response.status, json.loads(data)

    def _register(self, device_id: str, prekey_ids=("pk",)) -> None:
        status, _ = self._request("POST", "/v1/devices", {
            "user_id": f"u-{device_id}", "device_id": device_id,
            "identity_key": _raw_key_b64(),
            "signed_prekeys": [{"key_id": key_id, "public_key": _raw_key_b64()}
                               for key_id in prekey_ids]})
        self.assertEqual(status, 201)

    def _session_payload(self, **overrides: object) -> dict:
        payload = {
            "initiator_device_id": "da",
            "recipient_device_id": "db",
            "prekey_id": "pk",
            "ephemeral_key": _raw_key_b64(),
        }
        payload.update(overrides)
        return payload

    def _create_session(self) -> dict:
        self._register("da")
        self._register("db")
        status, body = self._request("POST", "/v1/sessions",
                                     self._session_payload())
        self.assertEqual(status, 201, body)
        return body

    def test_create_session_201_with_eight_fields(self) -> None:
        body = self._create_session()
        self.assertEqual(set(body), {
            "session_id", "initiator_device_id", "recipient_device_id",
            "prekey_id", "ephemeral_key", "identity_key", "public_key",
            "created_at"})
        self.assertTrue(body["session_id"])
        self.assertTrue(body["created_at"].endswith("+00:00"))

    def test_echo_and_recipient_key_material(self) -> None:
        recp_identity = _raw_key_b64()
        recp_prekey = _raw_key_b64()
        status, _ = self._request("POST", "/v1/devices", {
            "user_id": "u-db", "device_id": "db",
            "identity_key": recp_identity,
            "signed_prekeys": [{"key_id": "pk", "public_key": recp_prekey}]})
        self.assertEqual(status, 201)
        self._register("da")
        ephemeral = _raw_key_b64()
        status, body = self._request(
            "POST", "/v1/sessions",
            self._session_payload(ephemeral_key=ephemeral))
        self.assertEqual(status, 201)
        self.assertEqual(body["initiator_device_id"], "da")
        self.assertEqual(body["recipient_device_id"], "db")
        self.assertEqual(body["prekey_id"], "pk")
        self.assertEqual(body["ephemeral_key"], ephemeral)
        self.assertEqual(body["identity_key"], recp_identity)
        self.assertEqual(body["public_key"], recp_prekey)

    def test_repeated_post_creates_distinct_sessions(self) -> None:
        body = self._create_session()
        status, second = self._request(
            "POST", "/v1/sessions",
            self._session_payload(ephemeral_key=_raw_key_b64()))
        self.assertEqual(status, 201)
        self.assertNotEqual(body["session_id"], second["session_id"])

    def test_get_session_200_matches_create(self) -> None:
        created = self._create_session()
        status, fetched = self._request(
            "GET", f"/v1/sessions/{created['session_id']}")
        self.assertEqual(status, 200)
        self.assertEqual(fetched, created)

    def test_get_unknown_session_is_404_field_session_id(self) -> None:
        status, body = self._request("GET", "/v1/sessions/nope")
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "session_id")

    def test_session_missing_field_is_400(self) -> None:
        self._register("da")
        self._register("db")
        for name in ("initiator_device_id", "recipient_device_id",
                     "prekey_id", "ephemeral_key"):
            payload = self._session_payload()
            del payload[name]
            status, body = self._request("POST", "/v1/sessions", payload)
            self.assertEqual(status, 400, name)
            self.assertEqual(body["field"], name, name)

    def test_session_bad_field_type_is_400(self) -> None:
        self._register("da")
        self._register("db")
        status, body = self._request(
            "POST", "/v1/sessions",
            self._session_payload(recipient_device_id=42))
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "recipient_device_id")

    def test_ephemeral_key_bad_encoding_is_400(self) -> None:
        self._register("da")
        self._register("db")
        status, body = self._request(
            "POST", "/v1/sessions",
            self._session_payload(ephemeral_key="garbage"))
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "ephemeral_key")

    def test_session_body_not_json_is_400(self) -> None:
        connection = HTTPConnection("127.0.0.1", self.port, timeout=5)
        connection.request("POST", "/v1/sessions", body=b"{bad",
                           headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        body = json.loads(response.read())
        connection.close()
        self.assertEqual(response.status, 400)
        self.assertEqual(body["field"], "request_body")

    def test_same_initiator_and_recipient_is_400(self) -> None:
        self._register("da")
        status, body = self._request(
            "POST", "/v1/sessions",
            self._session_payload(recipient_device_id="da"))
        self.assertEqual(status, 400)
        self.assertEqual(body["field"], "recipient_device_id")

    def test_unknown_devices_and_prekey_are_404(self) -> None:
        self._register("da")
        self._register("db")
        cases = (
            (self._session_payload(initiator_device_id="ghost"),
             "initiator_device_id"),
            (self._session_payload(recipient_device_id="ghost"),
             "recipient_device_id"),
            (self._session_payload(prekey_id="ghost"), "prekey_id"),
        )
        for payload, field in cases:
            status, body = self._request("POST", "/v1/sessions", payload)
            self.assertEqual(status, 404, field)
            self.assertEqual(body["field"], field, field)

    def test_revoked_parties_are_409(self) -> None:
        self._register("da")
        self._register("db")
        self._register("dc")

        self.assertEqual(self._request("POST", "/v1/devices/da/revoke")[0], 200)
        status, body = self._request("POST", "/v1/sessions",
                                     self._session_payload())
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "initiator_device_id")

        # Fresh initiator dc; now revoke the recipient db.
        self.assertEqual(self._request("POST", "/v1/devices/db/revoke")[0], 200)
        status, body = self._request(
            "POST", "/v1/sessions",
            self._session_payload(initiator_device_id="dc"))
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "recipient_device_id")

    def test_revoked_prekey_is_409(self) -> None:
        self._register("da")
        self._register("db")
        self.assertEqual(
            self._request("POST", "/v1/devices/db/prekeys/pk/revoke")[0], 200)
        status, body = self._request("POST", "/v1/sessions",
                                     self._session_payload())
        self.assertEqual(status, 409)
        self.assertEqual(body["field"], "prekey_id")

    def test_snapshot_survives_later_revocations(self) -> None:
        created = self._create_session()
        self.assertEqual(
            self._request("POST", "/v1/devices/db/prekeys/pk/revoke")[0], 200)
        self.assertEqual(self._request("POST", "/v1/devices/db/revoke")[0], 200)
        status, fetched = self._request(
            "GET", f"/v1/sessions/{created['session_id']}")
        self.assertEqual(status, 200)
        self.assertEqual(fetched, created)

    def test_session_id_with_slash_subpath_is_404(self) -> None:
        status, body = self._request("GET", "/v1/sessions/a/b")
        self.assertEqual(status, 404)
        self.assertEqual(body["field"], "session_id")


if __name__ == "__main__":
    unittest.main()
