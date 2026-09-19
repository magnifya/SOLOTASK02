"""HTTP 服务：POST /v1/devices 注册设备，GET /v1/devices/{device_id} 查询。"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional, Tuple

from .store import Device, DeviceStore, SignedPreKey

DEVICE_PATH_RE = re.compile(r"^/v1/devices/([^/]+)$")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def validate_registration(payload: object) -> Tuple[Optional[dict], Optional[str]]:
    """校验注册请求体，返回 (规范化后的数据, 错误字段名)。

    校验通过时错误字段名为 None；失败时数据为 None，字段名指明出错的字段。
    """
    if not isinstance(payload, dict):
        return None, "body"
    for name in ("user_id", "device_id", "identity_key"):
        value = payload.get(name)
        if not isinstance(value, str) or not value:
            return None, name
    prekeys = payload.get("signed_prekeys")
    if not isinstance(prekeys, list):
        return None, "signed_prekeys"
    normalized_prekeys = []
    for index, item in enumerate(prekeys):
        if not isinstance(item, dict):
            return None, "signed_prekeys[%d]" % index
        for name in ("key_id", "public_key"):
            value = item.get(name)
            if not isinstance(value, str) or not value:
                return None, "signed_prekeys[%d].%s" % (index, name)
        normalized_prekeys.append({"key_id": item["key_id"], "public_key": item["public_key"]})
    return {
        "user_id": payload["user_id"],
        "device_id": payload["device_id"],
        "identity_key": payload["identity_key"],
        "signed_prekeys": normalized_prekeys,
    }, None


def make_handler(store: DeviceStore):
    """构造绑定指定存储的请求处理器类。"""

    class DeviceHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format, *args):  # noqa: A002 - 保持安静，测试时不刷屏
            pass

        def _send_json(self, status: int, body: dict) -> None:
            data = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler 约定
            if self.path != "/v1/devices":
                self._send_json(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = 0
            raw = self.rfile.read(length) if length > 0 else b""
            try:
                payload = json.loads(raw.decode("utf-8")) if raw else None
            except (ValueError, UnicodeDecodeError):
                self._send_json(400, {"error": "invalid JSON body", "field": "body"})
                return
            data, bad_field = validate_registration(payload)
            if data is None:
                self._send_json(
                    400,
                    {"error": "missing or invalid field: %s" % bad_field, "field": bad_field},
                )
                return
            device = Device(
                user_id=data["user_id"],
                device_id=data["device_id"],
                identity_key=data["identity_key"],
                registered_at=_utc_now_iso(),
                signed_prekeys=[
                    SignedPreKey(key_id=pk["key_id"], public_key=pk["public_key"])
                    for pk in data["signed_prekeys"]
                ],
            )
            if not store.add(device):
                self._send_json(
                    409,
                    {"error": "device already registered", "device_id": device.device_id},
                )
                return
            self._send_json(
                201,
                {"device_id": device.device_id, "registered_at": device.registered_at},
            )

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler 约定
            match = DEVICE_PATH_RE.match(self.path)
            if not match:
                self._send_json(404, {"error": "not found"})
                return
            device_id = match.group(1)
            device = store.get(device_id)
            if device is None:
                self._send_json(
                    404, {"error": "device not found", "device_id": device_id}
                )
                return
            self._send_json(
                200,
                {
                    "identity_key": device.identity_key,
                    "prekey_ids": store.active_prekey_ids(device_id),
                    "registered_at": device.registered_at,
                },
            )

    return DeviceHandler


def create_server(host: str, port: int, store: Optional[DeviceStore] = None) -> ThreadingHTTPServer:
    """创建 HTTP 服务实例。"""
    if store is None:
        store = DeviceStore()
    return ThreadingHTTPServer((host, port), make_handler(store))


def main() -> None:
    """命令行启动 HTTP 服务的入口。"""
    import argparse

    parser = argparse.ArgumentParser(description="E2E 设备注册 HTTP 服务")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    server = create_server(args.host, args.port)
    print("listening on http://%s:%d" % (args.host, args.port))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
