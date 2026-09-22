"""HTTP layer: JSON API over the standard library's ``http.server``."""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Tuple
from urllib.parse import parse_qs, unquote, urlsplit

from .persistence import PersistenceUnavailable
from .service import DeviceService, ServiceError

_DEVICES_PATH = "/v1/devices"
_PREKEYS_PATH = "/v1/prekeys"
_SESSIONS_PATH = "/v1/sessions"
_SESSIONS_FROM_CLAIM_PATH = "/v1/sessions/from-claim"
_SESSIONS_FROM_BATCH_CLAIM_PATH = "/v1/sessions/from-batch-claim"
_MESSAGES_PATH = "/v1/messages"
_GROUPS_PATH = "/v1/groups"
_GROUP_SESSIONS_PATH = "/v1/group-sessions"

#: Sentinel meaning a 400 for a malformed body was already sent.
_BAD_REQUEST = object()


class DeviceHTTPHandler(BaseHTTPRequestHandler):
    """Routes for ``POST /v1/devices`` and ``GET /v1/devices/{device_id}``."""

    service: DeviceService  # injected via :func:`handler_for`

    def _send_data_file_unavailable(self) -> None:
        """Answer 503/field=data_file after a durable write was rolled back."""
        self._send_json(503, {
            "message": "state file is temporarily unavailable; the change "
                       "was not applied",
            "field": "data_file"})

    # -- routing ----------------------------------------------------------

    def do_POST(self) -> None:
        try:
            self._route_POST()
        except PersistenceUnavailable:
            self._send_data_file_unavailable()

    def _route_POST(self) -> None:
        path = urlsplit(self.path).path
        if path == _DEVICES_PATH:
            self._handle_register()
        elif path == _PREKEYS_PATH + "/claim-batch":
            self._handle_claim_prekey_batch()
        elif path == _PREKEYS_PATH + "/claim":
            self._handle_claim_prekey()
        elif path == _SESSIONS_FROM_BATCH_CLAIM_PATH:
            self._handle_create_sessions_from_batch_claim()
        elif path == _SESSIONS_FROM_CLAIM_PATH:
            self._handle_create_session_from_claim()
        elif path == _SESSIONS_PATH:
            self._handle_create_session()
        elif path == _MESSAGES_PATH:
            self._handle_post_message()
        elif path == _GROUPS_PATH:
            self._handle_create_group()
        elif path == _GROUP_SESSIONS_PATH:
            self._handle_create_group_session()
        elif path.startswith(_GROUP_SESSIONS_PATH + "/") \
                and path.endswith("/sync/checkpoint"):
            self._route_group_sync_checkpoint(path)
        elif path.startswith(_GROUP_SESSIONS_PATH + "/") \
                and path.endswith("/rotate"):
            self._route_rotate_group_session(path)
        elif path.startswith(_GROUPS_PATH + "/"):
            self._route_group_member(path)
        elif path.startswith(_DEVICES_PATH + "/") and path.endswith("/revoke"):
            self._route_revoke(path)
        elif path.startswith(_DEVICES_PATH + "/") \
                and path.endswith("/identity-key/rotate"):
            self._route_rotate_identity(path)
        elif path.startswith(_DEVICES_PATH + "/"):
            self._route_add_prekey(path)
        elif path.startswith(_MESSAGES_PATH + "/") and path.endswith("/acks"):
            self._route_acks(path)
        elif path.startswith(_MESSAGES_PATH + "/") and "/retry/" in path:
            self._route_retry(path)
        else:
            self._send_json(404, {"message": f"not found: {path}"})

    def _route_retry(self, path: str) -> None:
        suffix = path[len(_MESSAGES_PATH) + 1:]
        parts = suffix.split("/")
        if len(parts) == 3 and parts[1] == "retry" and parts[0] and parts[2]:
            self._handle_retry_message(unquote(parts[0]), unquote(parts[2]))
        else:
            self._send_json(404, {"message": "session not found",
                                  "field": "session_id"})

    def _route_acks(self, path: str) -> None:
        suffix = path[len(_MESSAGES_PATH) + 1:-len("/acks")]
        parts = suffix.split("/")
        if len(parts) == 1 and parts[0]:
            self._handle_ack_message(unquote(parts[0]))
        else:
            self._send_json(404, {"message": "session not found",
                                  "field": "session_id"})

    def _route_group_member(self, path: str) -> None:
        suffix = path[len(_GROUPS_PATH) + 1:]
        parts = suffix.split("/")
        if len(parts) == 2 and parts[0] and parts[1] == "members":
            self._handle_add_group_member(unquote(parts[0]))
        elif (len(parts) == 3 and parts[0] and parts[1] == "members"
                and parts[2] in ("add", "remove")):
            group_id = unquote(parts[0])
            if parts[2] == "add":
                self._handle_add_group_member(group_id)
            else:
                self._handle_remove_group_member(group_id)
        else:
            self._send_json(404, {"message": "group not found",
                                  "field": "group_id"})

    def _route_group_sync_checkpoint(self, path: str) -> None:
        suffix = path[len(_GROUP_SESSIONS_PATH) + 1:
                      -len("/sync/checkpoint")]
        if suffix and "/" not in suffix:
            self._handle_sync_checkpoint(unquote(suffix))
        else:
            self._send_json(404, {"message": "session not found",
                                  "field": "session_id"})

    def _route_rotate_group_session(self, path: str) -> None:
        suffix = path[len(_GROUP_SESSIONS_PATH) + 1:-len("/rotate")]
        if suffix and "/" not in suffix:
            self._handle_rotate_group_session(unquote(suffix))
        else:
            self._send_json(404, {"message": "session not found",
                                  "field": "session_id"})

    def _route_revoke(self, path: str) -> None:
        # Split on raw slashes only; a percent-encoded slash inside a segment
        # is part of the id, mirroring the GET route.
        suffix = path[len(_DEVICES_PATH) + 1:-len("/revoke")]
        parts = suffix.split("/")
        if len(parts) == 1:
            self._handle_revoke_device(unquote(parts[0]))
        elif len(parts) == 3 and parts[1] == "prekeys":
            self._handle_revoke_prekey(unquote(parts[0]), unquote(parts[2]))
        else:
            self._send_json(404, {"message": "device not found",
                                  "field": "device_id"})

    def _route_rotate_identity(self, path: str) -> None:
        suffix = path[len(_DEVICES_PATH) + 1:-len("/identity-key/rotate")]
        parts = suffix.split("/")
        if len(parts) == 1 and parts[0]:
            self._handle_rotate_identity(unquote(parts[0]))
        else:
            self._send_json(404, {"message": "device not found",
                                  "field": "device_id"})

    def _route_add_prekey(self, path: str) -> None:
        suffix = path[len(_DEVICES_PATH) + 1:]
        parts = suffix.split("/")
        if len(parts) == 2 and parts[0] and parts[1] == "prekeys":
            self._handle_add_prekey(unquote(parts[0]))
        else:
            self._send_json(404, {"message": "device not found",
                                  "field": "device_id"})

    def do_GET(self) -> None:
        try:
            self._route_GET()
        except PersistenceUnavailable:
            self._send_data_file_unavailable()

    def _route_GET(self) -> None:
        path = urlsplit(self.path).path
        if path.startswith(_GROUPS_PATH + "/"):
            suffix = path[len(_GROUPS_PATH) + 1:]
            if not suffix or "/" in suffix:
                self._send_json(404, {"message": "group not found",
                                      "field": "group_id"})
                return
            self._handle_show_group(unquote(suffix))
        elif path.startswith(_GROUP_SESSIONS_PATH + "/"):
            suffix = path[len(_GROUP_SESSIONS_PATH) + 1:]
            if suffix.endswith("/sync"):
                session_id = suffix[:-len("/sync")]
                if not session_id or "/" in session_id:
                    self._send_json(404, {"message": "session not found",
                                          "field": "session_id"})
                    return
                self._handle_sync_group_messages(unquote(session_id))
                return
            if not suffix or "/" in suffix:
                self._send_json(404, {"message": "session not found",
                                      "field": "session_id"})
                return
            self._handle_show_group_session(unquote(suffix))
        elif path.startswith(_DEVICES_PATH + "/"):
            suffix = path[len(_DEVICES_PATH) + 1:]
            # A real slash means a sub-path (…/devices/a/b); a percent-encoded
            # slash within a single segment is part of the device id.
            if not suffix or "/" in suffix:
                self._send_json(404, {"message": "device not found",
                                      "field": "device_id"})
                return
            self._handle_show(unquote(suffix))
        elif path.startswith(_SESSIONS_PATH + "/"):
            suffix = path[len(_SESSIONS_PATH) + 1:]
            if not suffix or "/" in suffix:
                self._send_json(404, {"message": "session not found",
                                      "field": "session_id"})
                return
            self._handle_show_session(unquote(suffix))
        elif path.startswith(_MESSAGES_PATH + "/"):
            suffix = path[len(_MESSAGES_PATH) + 1:]
            if "/" in suffix:
                parts = suffix.split("/")
                if len(parts) == 3 and parts[1] == "status" and parts[0] and parts[2]:
                    self._handle_message_status(
                        unquote(parts[0]), unquote(parts[2]))
                else:
                    self._send_json(404, {"message": "session not found",
                                          "field": "session_id"})
                return
            if not suffix:
                self._send_json(404, {"message": "session not found",
                                      "field": "session_id"})
                return
            self._handle_list_messages(unquote(suffix))
        else:
            self._send_json(404, {"message": f"not found: {path}"})

    # -- handlers ---------------------------------------------------------

    def _handle_register(self) -> None:
        payload = self._read_json_request()
        if payload is _BAD_REQUEST:
            return
        try:
            body = self.service.register(payload)
        except ServiceError as error:
            self._send_json(error.status_code, error.to_body())
            return
        self._send_json(201, body)

    def _handle_show(self, device_id: str) -> None:
        try:
            body = self.service.get_device(device_id)
        except ServiceError as error:
            self._send_json(error.status_code, error.to_body())
            return
        self._send_json(200, body)

    def _handle_revoke_device(self, device_id: str) -> None:
        try:
            body = self.service.revoke_device(device_id)
        except ServiceError as error:
            self._send_json(error.status_code, error.to_body())
            return
        self._send_json(200, body)

    def _handle_revoke_prekey(self, device_id: str, key_id: str) -> None:
        try:
            body = self.service.revoke_prekey(device_id, key_id)
        except ServiceError as error:
            self._send_json(error.status_code, error.to_body())
            return
        self._send_json(200, body)

    def _handle_rotate_identity(self, device_id: str) -> None:
        payload = self._read_json_request()
        if payload is _BAD_REQUEST:
            return
        try:
            body = self.service.rotate_identity_key(device_id, payload)
        except ServiceError as error:
            self._send_json(error.status_code, error.to_body())
            return
        self._send_json(200, body)

    def _handle_add_prekey(self, device_id: str) -> None:
        payload = self._read_json_request()
        if payload is _BAD_REQUEST:
            return
        try:
            body, status_code = self.service.add_prekey(device_id, payload)
        except ServiceError as error:
            self._send_json(error.status_code, error.to_body())
            return
        self._send_json(status_code, body)

    def _handle_claim_prekey(self) -> None:
        payload = self._read_json_request()
        if payload is _BAD_REQUEST:
            return
        try:
            body, status_code = self.service.claim_prekey(payload)
        except ServiceError as error:
            self._send_json(error.status_code, error.to_body())
            return
        self._send_json(status_code, body)

    def _handle_claim_prekey_batch(self) -> None:
        payload = self._read_json_request()
        if payload is _BAD_REQUEST:
            return
        try:
            body, status_code = self.service.claim_prekey_batch(payload)
        except ServiceError as error:
            self._send_json(error.status_code, error.to_body())
            return
        self._send_json(status_code, body)

    def _handle_create_session(self) -> None:
        payload = self._read_json_request()
        if payload is _BAD_REQUEST:
            return
        try:
            body = self.service.create_session(payload)
        except ServiceError as error:
            self._send_json(error.status_code, error.to_body())
            return
        self._send_json(201, body)

    def _handle_create_sessions_from_batch_claim(self) -> None:
        payload = self._read_json_request()
        if payload is _BAD_REQUEST:
            return
        try:
            body = self.service.create_sessions_from_batch_claim(payload)
        except ServiceError as error:
            self._send_json(error.status_code, error.to_body())
            return
        self._send_json(201, body)

    def _handle_create_session_from_claim(self) -> None:
        payload = self._read_json_request()
        if payload is _BAD_REQUEST:
            return
        try:
            body = self.service.create_session_from_claim(payload)
        except ServiceError as error:
            self._send_json(error.status_code, error.to_body())
            return
        self._send_json(201, body)

    def _handle_show_session(self, session_id: str) -> None:
        try:
            body = self.service.get_session(session_id)
        except ServiceError as error:
            self._send_json(error.status_code, error.to_body())
            return
        self._send_json(200, body)

    def _handle_create_group(self) -> None:
        payload = self._read_json_request()
        if payload is _BAD_REQUEST:
            return
        try:
            body = self.service.create_group(payload)
        except ServiceError as error:
            self._send_json(error.status_code, error.to_body())
            return
        self._send_json(201, body)

    def _handle_show_group(self, group_id: str) -> None:
        try:
            body = self.service.get_group(group_id)
        except ServiceError as error:
            self._send_json(error.status_code, error.to_body())
            return
        self._send_json(200, body)

    def _handle_add_group_member(self, group_id: str) -> None:
        payload = self._read_json_request()
        if payload is _BAD_REQUEST:
            return
        try:
            body, status_code = self.service.add_group_member(group_id, payload)
        except ServiceError as error:
            self._send_json(error.status_code, error.to_body())
            return
        self._send_json(status_code, body)

    def _handle_remove_group_member(self, group_id: str) -> None:
        payload = self._read_json_request()
        if payload is _BAD_REQUEST:
            return
        try:
            body = self.service.remove_group_member(group_id, payload)
        except ServiceError as error:
            self._send_json(error.status_code, error.to_body())
            return
        self._send_json(200, body)

    def _handle_create_group_session(self) -> None:
        payload = self._read_json_request()
        if payload is _BAD_REQUEST:
            return
        try:
            body = self.service.create_group_session(payload)
        except ServiceError as error:
            self._send_json(error.status_code, error.to_body())
            return
        self._send_json(201, body)

    def _handle_show_group_session(self, session_id: str) -> None:
        try:
            body = self.service.get_group_session(session_id)
        except ServiceError as error:
            self._send_json(error.status_code, error.to_body())
            return
        self._send_json(200, body)

    def _handle_rotate_group_session(self, session_id: str) -> None:
        payload = self._read_json_request()
        if payload is _BAD_REQUEST:
            return
        try:
            body, status_code = self.service.rotate_group_session(
                session_id, payload)
        except ServiceError as error:
            self._send_json(error.status_code, error.to_body())
            return
        self._send_json(status_code, body)

    def _handle_sync_group_messages(self, session_id: str) -> None:
        params = self._sync_query_params()
        if params is None:
            return  # a 400 response was already sent
        device_id, after, limit = params
        try:
            body = self.service.sync_group_messages(
                session_id, device_id, after, limit)
        except ServiceError as error:
            self._send_json(error.status_code, error.to_body())
            return
        self._send_json(200, body)

    def _handle_sync_checkpoint(self, session_id: str) -> None:
        payload = self._read_json_request()
        if payload is _BAD_REQUEST:
            return
        try:
            body, status_code = self.service.sync_group_checkpoint(
                session_id, payload)
        except ServiceError as error:
            self._send_json(error.status_code, error.to_body())
            return
        self._send_json(status_code, body)

    def _sync_query_params(self) -> Optional[Tuple[str, Optional[int], int]]:
        """Validate the sync GET params: single ``device_id``, optional
        non-negative ``after`` (omitted -> the stored cursor), and ``limit``
        in 1..100 (default 100). Returns ``None`` after sending a 400."""
        query = parse_qs(urlsplit(self.path).query)

        device_ids = query.get("device_id", [])
        if len(device_ids) != 1 or not device_ids[0]:
            self._send_json(400, {"message": "device_id is required",
                                  "field": "device_id"})
            return None

        after_values = query.get("after")
        if after_values is None:
            after: Optional[int] = None
        elif len(after_values) != 1:
            self._send_json(400, {"message": "after must appear at most once",
                                  "field": "after"})
            return None
        else:
            try:
                after = int(after_values[0])
            except ValueError:
                self._send_json(400, {"message": "after must be an integer",
                                      "field": "after"})
                return None
            if after < 0:
                self._send_json(400, {"message": "after is out of range",
                                      "field": "after"})
                return None

        limit = self._int_param(query, "limit", default=100,
                                minimum=1, maximum=100)
        if limit is None:
            return None
        return device_ids[0], after, limit

    def _handle_post_message(self) -> None:
        payload = self._read_json_request()
        if payload is _BAD_REQUEST:
            return
        try:
            body = self.service.post_message(payload)
        except ServiceError as error:
            self._send_json(error.status_code, error.to_body())
            return
        self._send_json(201, body)

    def _handle_list_messages(self, session_id: str) -> None:
        params = self._message_query_params()
        if params is None:
            return  # a 400 response was already sent
        device_id, after, limit = params
        try:
            body = self.service.list_messages(session_id, device_id,
                                              after, limit)
        except ServiceError as error:
            self._send_json(error.status_code, error.to_body())
            return
        self._send_json(200, body)

    def _handle_retry_message(self, session_id: str, message_id: str) -> None:
        payload = self._read_json_request()
        if payload is _BAD_REQUEST:
            return
        try:
            body, status_code = self.service.retry_message(
                session_id, message_id, payload)
        except ServiceError as error:
            self._send_json(error.status_code, error.to_body())
            return
        self._send_json(status_code, body)

    def _handle_ack_message(self, session_id: str) -> None:
        payload = self._read_json_request()
        if payload is _BAD_REQUEST:
            return
        try:
            body, status_code = self.service.ack_message(session_id, payload)
        except ServiceError as error:
            self._send_json(error.status_code, error.to_body())
            return
        self._send_json(status_code, body)

    def _handle_message_status(self, session_id: str, message_id: str) -> None:
        device_id = self._required_device_param()
        if device_id is None:
            return  # a 400 response was already sent
        try:
            body = self.service.message_status(
                session_id, message_id, device_id)
        except ServiceError as error:
            self._send_json(error.status_code, error.to_body())
            return
        self._send_json(200, body)

    def _required_device_param(self) -> Optional[str]:
        """Validate the single required ``device_id`` query parameter.

        Returns the value, or ``None`` after sending 400 when it is missing,
        empty, or supplied more than once.
        """
        query = parse_qs(urlsplit(self.path).query)
        values = query.get("device_id", [])
        if len(values) != 1 or not values[0]:
            self._send_json(400, {"message": "device_id is required",
                                  "field": "device_id"})
            return None
        return values[0]

    def _message_query_params(self) -> Optional[Tuple[str, int, int]]:
        """Validate ``device_id``/``after``/``limit`` query parameters.

        Returns ``(device_id, after, limit)``, or ``None`` after sending the
        400 response itself when a parameter is missing or malformed.
        """
        query = parse_qs(urlsplit(self.path).query)

        device_ids = query.get("device_id", [])
        if len(device_ids) != 1 or not device_ids[0]:
            self._send_json(400, {"message": "device_id is required",
                                  "field": "device_id"})
            return None

        after = self._int_param(query, "after", default=0, minimum=0)
        if after is None:
            return None
        limit = self._int_param(query, "limit", default=100,
                                minimum=1, maximum=100)
        if limit is None:
            return None
        return device_ids[0], after, limit

    def _int_param(self, query: Dict[str, list], name: str, default: int,
                   minimum: int, maximum: Optional[int] = None
                   ) -> Optional[int]:
        """Parse one integer query parameter; send a 400 and return ``None``
        when it is missing its single value, malformed, or out of range."""
        values = query.get(name)
        if not values:
            return default
        if len(values) != 1:
            self._send_json(400, {"message": f"{name} must appear at most once",
                                  "field": name})
            return None
        try:
            value = int(values[0])
        except ValueError:
            self._send_json(400, {"message": f"{name} must be an integer",
                                  "field": name})
            return None
        if value < minimum or (maximum is not None and value > maximum):
            self._send_json(400, {"message": f"{name} is out of range",
                                  "field": name})
            return None
        return value

    # -- plumbing ---------------------------------------------------------

    def _read_json_request(self) -> Any:
        """Read and decode a JSON request body.

        Returns the decoded value, or the :data:`_BAD_REQUEST` sentinel after
        sending the 400 response itself when the body cannot be read.
        """
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(400, {"message": "invalid Content-Length header",
                                  "field": "Content-Length"})
            return _BAD_REQUEST
        raw = self.rfile.read(length) if length > 0 else b""
        try:
            return json.loads(raw.decode("utf-8")) if raw else None
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(400, {"message": "request body must be valid JSON",
                                  "field": "request_body"})
            return _BAD_REQUEST

    def _send_json(self, status_code: int, body: Any) -> None:
        data = json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: Any) -> None:
        """Keep the server quiet; tests and CLIs speak JSON."""
        return


def handler_for(service: DeviceService) -> type:
    """Return a handler class bound to *service*."""

    class _BoundHandler(DeviceHTTPHandler):
        pass

    _BoundHandler.service = service
    return _BoundHandler


def create_server(host: str, port: int,
                  service: Optional[DeviceService] = None) -> Tuple[ThreadingHTTPServer, DeviceService]:
    """Build (but do not start) the HTTP server; return it with its service."""
    if service is None:
        service = DeviceService()
    server = ThreadingHTTPServer((host, port), handler_for(service))
    return server, service
