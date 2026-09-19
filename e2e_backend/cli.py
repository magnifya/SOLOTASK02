"""命令行入口：register 与 show 子命令，输出单行 JSON。"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import List, Optional

DEFAULT_SERVER = "http://127.0.0.1:8000"


def _server_url(args: argparse.Namespace) -> str:
    return (args.server or os.environ.get("E2E_SERVER") or DEFAULT_SERVER).rstrip("/")


def _request(method: str, url: str, payload: Optional[dict] = None) -> tuple:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8")
        try:
            return error.code, json.loads(body)
        except ValueError:
            return error.code, {"error": body}
    except urllib.error.URLError as error:
        return None, {"error": "cannot reach server: %s" % error.reason}


def _parse_prekey(text: str) -> dict:
    key_id, sep, public_key = text.partition(":")
    if not sep or not key_id or not public_key:
        raise argparse.ArgumentTypeError(
            "prekey 格式应为 key_id:public_key，收到 %r" % text
        )
    return {"key_id": key_id, "public_key": public_key}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="e2e-cli", description="E2E 设备注册命令行")
    parser.add_argument("--server", help="服务地址，默认环境变量 E2E_SERVER 或 %s" % DEFAULT_SERVER)
    subparsers = parser.add_subparsers(dest="command", required=True)

    register = subparsers.add_parser("register", help="注册设备（对应 POST /v1/devices）")
    register.add_argument("--user-id", required=True)
    register.add_argument("--device-id", required=True)
    register.add_argument("--identity-key", required=True)
    register.add_argument(
        "--prekey",
        action="append",
        type=_parse_prekey,
        default=[],
        metavar="KEY_ID:PUBLIC_KEY",
        help="预密钥，可重复指定",
    )

    show = subparsers.add_parser("show", help="查询设备（对应 GET /v1/devices/{device_id}）")
    show.add_argument("device_id")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """解析子命令并调用对应 HTTP 接口，打印单行 JSON。"""
    args = build_parser().parse_args(argv)
    base = _server_url(args)
    if args.command == "register":
        payload = {
            "user_id": args.user_id,
            "device_id": args.device_id,
            "identity_key": args.identity_key,
            "signed_prekeys": args.prekey,
        }
        status, body = _request("POST", base + "/v1/devices", payload)
    else:  # show
        status, body = _request("GET", base + "/v1/devices/" + args.device_id)
    print(json.dumps(body, ensure_ascii=False))
    return 0 if status in (200, 201) else 1


if __name__ == "__main__":
    sys.exit(main())
