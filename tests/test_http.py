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


if __name__ == "__main__":
    unittest.main()
