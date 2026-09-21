"""Tests for claim-based session creation (``POST /v1/sessions/from-claim``).

Covers the 400/404/409 validation contract, the frozen claim values in the
eight-field session view, the once-per-claim binding (repeat submissions are
409 and create nothing), shared-lock linearization with revocations, durable
persistence and restart recovery of the binding, old files missing the
``claim_sessions`` section, refusal to start on a malformed section, and the
503/field=data_file rollback when the durable write fails. The CLI
``create-session-from-claim`` subcommand is covered end to end.
"""
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from http.client import HTTPConnection

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import x25519

from e2ee_backend.persistence import (
    PersistenceUnavailable,
    StateFileError,
    attach_persistence,
)
from e2ee_backend.service import DeviceService, ServiceError
from e2ee_backend.http_app import create_server


def _raw_key_b64() -> str:
    key = x25519.X25519PrivateKey.generate().public_key()
    raw = key.public_bytes(serialization.Encoding.Raw,
                           serialization.PublicFormat.Raw)
    return base64.b64encode(raw).decode()


def _register_payload(device_id: str, key_ids=("k1", "k2")) -> dict:
    return {
        "user_id": "u-" + device_id,
        "device_id": device_id,
        "identity_key": _raw_key_b64(),
        "signed_prekeys": [{"key_id": kid, "public_key": _raw_key_b64()}
                           for kid in key_ids],
    }


_SESSION_FIELDS = {"session_id", "initiator_device_id", "recipient_device_id",
                   "prekey_id", "ephemeral_key", "identity_key", "public_key",
                   "created_at"}


class _ServiceBase(unittest.TestCase):
    """A service with a recipient (two pre-keys) and an initiator, one claim."""

    def setUp(self) -> None:
        self.service = DeviceService()
        self.recipient = _register_payload("recv")
        self.initiator = _register_payload("init", key_ids=())
        self.service.register(self.recipient)
        self.service.register(self.initiator)
        self.claim, status = self.service.claim_prekey(
            {"recipient_device_id": "recv", "claim_id": "c1"})
        self.assertEqual(status, 201)
        self.ephemeral = _raw_key_b64()

    def _create(self, claim_id="c1", initiator="init", ephemeral=None):
        return self.service.create_session_from_claim({
            "claim_id": claim_id,
            "initiator_device_id": initiator,
            "ephemeral_key": ephemeral if ephemeral is not None
            else self.ephemeral,
        })

    def _error(self, payload) -> ServiceError:
        with self.assertRaises(ServiceError) as ctx:
            self.service.create_session_from_claim(payload)
        return ctx.exception


class ClaimSessionValidationTest(_ServiceBase):
    def test_missing_and_empty_and_nonstring_fields_are_400(self) -> None:
        base = {"claim_id": "c1", "initiator_device_id": "init",
                "ephemeral_key": self.ephemeral}
        for field in ("claim_id", "initiator_device_id", "ephemeral_key"):
            missing = {k: v for k, v in base.items() if k != field}
            error = self._error(missing)
            self.assertEqual((error.status_code, error.field), (400, field))
            for bad in ("", 123, None, ["x"]):
                error = self._error({**base, field: bad})
                self.assertEqual((error.status_code, error.field),
                                 (400, field), (field, bad))

    def test_non_object_body_is_400_request_body(self) -> None:
        error = self._error(["not", "a", "dict"])
        self.assertEqual((error.status_code, error.field),
                         (400, "request_body"))

    def test_invalid_ephemeral_key_encoding_is_400(self) -> None:
        error = self._error({"claim_id": "c1", "initiator_device_id": "init",
                             "ephemeral_key": "not-a-key"})
        self.assertEqual((error.status_code, error.field),
                         (400, "ephemeral_key"))

    def test_unknown_claim_is_404_claim_id(self) -> None:
        error = self._error({"claim_id": "ghost",
                             "initiator_device_id": "init",
                             "ephemeral_key": self.ephemeral})
        self.assertEqual((error.status_code, error.field), (404, "claim_id"))

    def test_unknown_initiator_is_404(self) -> None:
        error = self._error({"claim_id": "c1",
                             "initiator_device_id": "ghost",
                             "ephemeral_key": self.ephemeral})
        self.assertEqual((error.status_code, error.field),
                         (404, "initiator_device_id"))

    def test_revoked_initiator_is_409(self) -> None:
        self.service.revoke_device("init")
        error = self._error({"claim_id": "c1",
                             "initiator_device_id": "init",
                             "ephemeral_key": self.ephemeral})
        self.assertEqual((error.status_code, error.field),
                         (409, "initiator_device_id"))

    def test_revoked_recipient_is_409(self) -> None:
        self.service.revoke_device("recv")
        error = self._error({"claim_id": "c1",
                             "initiator_device_id": "init",
                             "ephemeral_key": self.ephemeral})
        self.assertEqual((error.status_code, error.field),
                         (409, "recipient_device_id"))

    def test_revoked_prekey_is_409(self) -> None:
        self.service.revoke_prekey("recv", self.claim["key_id"])
        error = self._error({"claim_id": "c1",
                             "initiator_device_id": "init",
                             "ephemeral_key": self.ephemeral})
        self.assertEqual((error.status_code, error.field),
                         (409, "prekey_id"))

    def test_failed_attempt_does_not_bind_the_claim(self) -> None:
        # A 404 failure leaves the claim unbound: a later valid request wins.
        with self.assertRaises(ServiceError):
            self._create(initiator="ghost")
        body = self._create()
        self.assertEqual(set(body), _SESSION_FIELDS)


class ClaimSessionCreationTest(_ServiceBase):
    def test_success_returns_eight_fields_with_frozen_claim_values(self) -> None:
        body = self._create()
        self.assertEqual(set(body), _SESSION_FIELDS)
        self.assertEqual(body["initiator_device_id"], "init")
        self.assertEqual(body["recipient_device_id"], "recv")
        self.assertEqual(body["prekey_id"], self.claim["key_id"])
        self.assertEqual(body["identity_key"], self.claim["identity_key"])
        self.assertEqual(body["public_key"], self.claim["public_key"])
        self.assertEqual(body["ephemeral_key"], self.ephemeral)
        self.assertTrue(body["created_at"].endswith("+00:00"))
        # The snapshot is retrievable and identical via the existing GET.
        self.assertEqual(self.service.get_session(body["session_id"]), body)

    def test_claim_values_stay_frozen_after_identity_rotation(self) -> None:
        self.service.rotate_identity_key(
            "recv", {"identity_key": _raw_key_b64()})
        body = self._create()
        self.assertEqual(body["identity_key"], self.claim["identity_key"])
        self.assertNotEqual(body["identity_key"],
                            self.service.get_device("recv")["identity_key"])

    def test_repeat_submission_is_409_and_creates_nothing(self) -> None:
        first = self._create()
        before = dict(self.service.store._sessions)
        error = self._error({"claim_id": "c1",
                             "initiator_device_id": "init",
                             "ephemeral_key": _raw_key_b64()})
        self.assertEqual((error.status_code, error.field), (409, "claim_id"))
        self.assertEqual(dict(self.service.store._sessions), before)
        # The original session is untouched.
        self.assertEqual(self.service.get_session(first["session_id"]), first)

    def test_distinct_claims_create_distinct_sessions(self) -> None:
        self.service.claim_prekey(
            {"recipient_device_id": "recv", "claim_id": "c2"})
        first = self._create(claim_id="c1")
        second = self._create(claim_id="c2")
        self.assertNotEqual(first["session_id"], second["session_id"])
        self.assertNotEqual(first["prekey_id"], second["prekey_id"])


class ClaimSessionPersistenceTest(_ServiceBase):
    def setUp(self) -> None:
        super().setUp()
        self.directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.directory, True)
        self.path = os.path.join(self.directory, "state.json")

    def _attach(self, service: DeviceService):
        return attach_persistence(service, self.path)

    def test_binding_and_session_survive_restart(self) -> None:
        self._attach(self.service)
        body = self._create()

        restored = DeviceService()
        self._attach(restored)
        self.assertEqual(restored.get_session(body["session_id"]), body)
        with self.assertRaises(ServiceError) as ctx:
            restored.create_session_from_claim({
                "claim_id": "c1", "initiator_device_id": "init",
                "ephemeral_key": _raw_key_b64()})
        self.assertEqual((ctx.exception.status_code, ctx.exception.field),
                         (409, "claim_id"))

    def test_old_file_without_claim_sessions_section_loads_empty(self) -> None:
        self._attach(self.service)
        self._create()
        with open(self.path, "r", encoding="utf-8") as handle:
            document = json.load(handle)
        self.assertIn("claim_sessions", document)
        del document["claim_sessions"]
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)

        restored = DeviceService()
        self._attach(restored)  # must not refuse startup
        self.assertEqual(restored.store._claim_sessions, {})

    def test_malformed_claim_sessions_section_refuses_startup(self) -> None:
        self._attach(self.service)
        self._create()
        with open(self.path, "r", encoding="utf-8") as handle:
            document = json.load(handle)
        document["claim_sessions"] = [
            {"claim_id": "c1", "session_id": "no-such-session"}]
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)

        with self.assertRaises(StateFileError):
            self._attach(DeviceService())

    def test_durable_write_failure_rolls_back_binding_and_session(self) -> None:
        store = self._attach(self.service)
        real_save = store.save

        def failing_save(state):
            raise OSError("simulated disk failure")

        store.save = failing_save
        try:
            with self.assertRaises(PersistenceUnavailable):
                self._create()
        finally:
            store.save = real_save
        # Neither the binding nor the session survived the failed write...
        self.assertEqual(self.service.store._claim_sessions, {})
        self.assertEqual(self.service.store._sessions, {})
        # ...and the claim is still usable afterwards.
        body = self._create()
        self.assertEqual(set(body), _SESSION_FIELDS)


class ClaimSessionHTTPTest(unittest.TestCase):
    def setUp(self) -> None:
        self.server, self.service = create_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        self.addCleanup(self._stop)
        self.service.register(_register_payload("recv"))
        self.service.register(_register_payload("init", key_ids=()))
        self.service.claim_prekey(
            {"recipient_device_id": "recv", "claim_id": "c1"})

    def _stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _post(self, path: str, body) -> tuple:
        connection = HTTPConnection("127.0.0.1", self.port, timeout=5)
        data = body if isinstance(body, str) else json.dumps(body)
        connection.request("POST", path, body=data,
                           headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        connection.close()
        return response.status, payload

    def test_create_and_duplicate_over_http(self) -> None:
        status, body = self._post("/v1/sessions/from-claim", {
            "claim_id": "c1", "initiator_device_id": "init",
            "ephemeral_key": _raw_key_b64()})
        self.assertEqual(status, 201)
        self.assertEqual(set(body), _SESSION_FIELDS)
        self.assertEqual(body["recipient_device_id"], "recv")

        status, error = self._post("/v1/sessions/from-claim", {
            "claim_id": "c1", "initiator_device_id": "init",
            "ephemeral_key": _raw_key_b64()})
        self.assertEqual(status, 409)
        self.assertEqual(error["field"], "claim_id")

    def test_unknown_claim_and_bad_body_over_http(self) -> None:
        status, error = self._post("/v1/sessions/from-claim", {
            "claim_id": "ghost", "initiator_device_id": "init",
            "ephemeral_key": _raw_key_b64()})
        self.assertEqual(status, 404)
        self.assertEqual(error["field"], "claim_id")

        status, error = self._post("/v1/sessions/from-claim",
                                   {"claim_id": "c1"})
        self.assertEqual(status, 400)
        self.assertEqual(error["field"], "initiator_device_id")

        status, error = self._post("/v1/sessions/from-claim", "not json")
        self.assertEqual(status, 400)
        self.assertEqual(error["field"], "request_body")

    def test_original_sessions_endpoint_is_unchanged(self) -> None:
        # The claimed (consumed) key is still refused by POST /v1/sessions.
        status, error = self._post("/v1/sessions", {
            "initiator_device_id": "init", "recipient_device_id": "recv",
            "prekey_id": "k1", "ephemeral_key": _raw_key_b64()})
        self.assertEqual(status, 409)
        self.assertEqual(error["field"], "prekey_id")


class ClaimSessionCLITest(unittest.TestCase):
    def setUp(self) -> None:
        self.server, self.service = create_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        self.addCleanup(self._stop)
        self.service.register(_register_payload("recv"))
        self.service.register(_register_payload("init", key_ids=()))
        self.service.claim_prekey(
            {"recipient_device_id": "recv", "claim_id": "c1"})

    def _stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _run(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "e2ee_backend", "--base-url", self.base_url,
             *arguments],
            capture_output=True, text=True, timeout=15)

    def test_create_session_from_claim_single_line_json_and_exit_codes(self):
        result = self._run("create-session-from-claim",
                           "--claim-id", "c1",
                           "--initiator-device-id", "init",
                           "--ephemeral-key", _raw_key_b64())
        self.assertEqual(result.returncode, 0, result.stderr)
        line = result.stdout.strip()
        self.assertEqual(line.count("\n"), 0)
        body = json.loads(line)
        self.assertEqual(set(body), _SESSION_FIELDS)
        self.assertEqual(body["recipient_device_id"], "recv")
        self.assertEqual(body["prekey_id"], "k1")

        repeated = self._run("create-session-from-claim",
                             "--claim-id", "c1",
                             "--initiator-device-id", "init",
                             "--ephemeral-key", _raw_key_b64())
        self.assertEqual(repeated.returncode, 1)
        self.assertEqual(json.loads(repeated.stderr.strip())["field"],
                         "claim_id")


if __name__ == "__main__":
    unittest.main()
