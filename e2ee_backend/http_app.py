"""HTTP layer: JSON API over the standard library's ``http.server``."""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional, Tuple
from urllib.parse import unquote, urlsplit

from .service import DeviceService, ServiceError

_DEVICES_PATH = "/v1/devices"


class DeviceHTTPHandler(BaseHTTPRequestHandler):
    """Routes for ``POST /v1/devices`` and ``GET /v1/devices/{device_id}``."""

    service: DeviceService  # injected via :func:`handler_for`

    # -- routing ----------------------------------------------------------

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        if path == _DEVICES_PATH:
            self._handle_register()
        else:
            self._send_json(404, {"message": f"not found: {path}"})

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path.startswith(_DEVICES_PATH + "/"):
            suffix = path[len(_DEVICES_PATH) + 1:]
            # A real slash means a sub-path (…/devices/a/b); a percent-encoded
            # slash within a single segment is part of the device id.
            if not suffix or "/" in suffix:
                self._send_json(404, {"message": "device not found",
                                      "field": "device_id"})
                return
            self._handle_show(unquote(suffix))
        else:
            self._send_json(404, {"message": f"not found: {path}"})

    # -- handlers ---------------------------------------------------------

    def _handle_register(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(400, {"message": "invalid Content-Length header",
                                  "field": "Content-Length"})
            return
        raw = self.rfile.read(length) if length > 0 else b""
        try:
            payload = json.loads(raw.decode("utf-8")) if raw else None
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(400, {"message": "request body must be valid JSON",
                                  "field": "request_body"})
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

    # -- plumbing ---------------------------------------------------------

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
