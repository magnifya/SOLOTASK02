"""Command-line entry point: device and session API commands (plus ``serve``).

All API commands talk to an HTTP server and print the server's JSON response
on a single line, with exactly the field names used over HTTP.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Sequence, Tuple
from urllib import error as urllib_error
from urllib import request as urllib_request

from .crypto import CryptoError, decrypt_message, encrypt_message
from .http_app import create_server
from .service import DeviceService

_DEFAULT_BASE_URL = "http://127.0.0.1:8080"
_DEFAULT_TIMEOUT = 10.0


class ServerUnavailable(Exception):
    """Raised when the server cannot be reached (connection failure/timeout)."""


def _read_key_argument(value: str) -> str:
    """Return *value* directly, or read the file it points at with ``@path``."""
    if value.startswith("@"):
        with open(value[1:], "r", encoding="utf-8") as handle:
            return handle.read().strip()
    return value


def _parse_prekey(spec: str) -> Dict[str, str]:
    """Parse ``key_id:public_key``; a leading ``@`` loads a JSON mapping file."""
    if spec.startswith("@"):
        with open(spec[1:], "r", encoding="utf-8") as handle:
            element = json.load(handle)
        if (not isinstance(element, dict)
                or not isinstance(element.get("key_id"), str)
                or not isinstance(element.get("public_key"), str)):
            raise ValueError(f"prekey file must contain key_id and public_key: {spec}")
        return {"key_id": element["key_id"], "public_key": element["public_key"]}

    key_id, sep, public_key = spec.partition(":")
    # A PEM key contains colons? It does not, but contains the BEGIN preamble;
    # keep the first colon as the separator regardless.
    if not sep or not key_id or not public_key.strip():
        raise ValueError(
            f"prekey must have the form KEY_ID:PUBLIC_KEY (got: {spec!r})")
    return {"key_id": key_id, "public_key": public_key.strip()}


def _request_json(method: str, url: str, body: Any = None) -> Tuple[int, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib_request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib_request.urlopen(req, timeout=_DEFAULT_TIMEOUT) as response:
            status = response.status
            raw = response.read().decode("utf-8")
    except urllib_error.HTTPError as http_error:
        raw = http_error.read().decode("utf-8")
        try:
            return http_error.code, json.loads(raw)
        except json.JSONDecodeError:
            return http_error.code, {"message": raw}
    except (urllib_error.URLError, TimeoutError, OSError) as error:
        raise ServerUnavailable(str(error)) from None
    return status, json.loads(raw)


def _emit_server_error() -> int:
    """Print the single-line connection-error JSON on stderr; exit non-zero."""
    print(json.dumps({"message": "could not reach server", "field": "server"},
                     separators=(",", ":")),
          file=sys.stderr)
    return 1


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser with register/show subcommands."""
    parser = argparse.ArgumentParser(
        prog="e2ee-backend",
        description="End-to-end encrypted messaging backend CLI.")
    parser.add_argument(
        "--base-url", default=os.environ.get("E2EE_BASE_URL", _DEFAULT_BASE_URL),
        help="HTTP base URL of the server (default: %(default)s or $E2EE_BASE_URL).")
    sub = parser.add_subparsers(dest="command", required=True)

    p_register = sub.add_parser("register", help="register a device and publish pre-keys")
    p_register.add_argument("--user-id", required=True)
    p_register.add_argument("--device-id", required=True)
    p_register.add_argument(
        "--identity-key", required=True,
        help="identity public key (string), or @path to read it from a file")
    p_register.add_argument(
        "--prekey", action="append", default=[], metavar="KEY_ID:PUBLIC_KEY",
        help="signed pre-key; repeatable. Use @path for a JSON object file.")
    p_register.set_defaults(handler=cmd_register)

    p_show = sub.add_parser("show", help="show a device's public record")
    p_show.add_argument("device_id")
    p_show.set_defaults(handler=cmd_show)

    p_revoke_device = sub.add_parser("revoke-device", help="revoke a device")
    p_revoke_device.add_argument("--device-id", required=True)
    p_revoke_device.set_defaults(handler=cmd_revoke_device)

    p_revoke_prekey = sub.add_parser("revoke-prekey", help="revoke a signed pre-key")
    p_revoke_prekey.add_argument("--device-id", required=True)
    p_revoke_prekey.add_argument("--key-id", required=True)
    p_revoke_prekey.set_defaults(handler=cmd_revoke_prekey)

    p_create_session = sub.add_parser(
        "create-session", help="negotiate a session between two devices")
    p_create_session.add_argument("--initiator-device-id", required=True)
    p_create_session.add_argument("--recipient-device-id", required=True)
    p_create_session.add_argument("--prekey-id", required=True)
    p_create_session.add_argument(
        "--ephemeral-key", required=True,
        help="ephemeral public key (string), or @path to read it from a file")
    p_create_session.set_defaults(handler=cmd_create_session)

    p_show_session = sub.add_parser("show-session", help="show a session snapshot")
    p_show_session.add_argument("session_id")
    p_show_session.set_defaults(handler=cmd_show_session)

    p_send_message = sub.add_parser(
        "send-message", help="send an encrypted message envelope into a session")
    p_send_message.add_argument("--session-id", required=True)
    p_send_message.add_argument("--sender-device-id", required=True)
    p_send_message.add_argument("--message-id", required=True)
    p_send_message.add_argument("--sequence", required=True, type=int)
    p_send_message.add_argument("--nonce", required=True)
    p_send_message.add_argument("--ciphertext", required=True)
    p_send_message.set_defaults(handler=cmd_send_message)

    p_pull_messages = sub.add_parser(
        "pull-messages", help="pull a page of messages from a session")
    p_pull_messages.add_argument("session_id")
    p_pull_messages.add_argument("--device-id", required=True)
    p_pull_messages.add_argument("--after", type=int, default=0)
    p_pull_messages.add_argument("--limit", type=int, default=100)
    p_pull_messages.set_defaults(handler=cmd_pull_messages)

    p_encrypt = sub.add_parser(
        "encrypt-message",
        help="encrypt a plaintext locally with AES-256-GCM (no server needed)")
    p_encrypt.add_argument("--session-id", required=True)
    p_encrypt.add_argument(
        "--key", required=True,
        help="base64-encoded 32-byte AES key, or @path to read it from a file")
    p_encrypt.add_argument(
        "--plaintext", required=True,
        help="UTF-8 plaintext, or @path to read it from a file")
    p_encrypt.set_defaults(handler=cmd_encrypt_message)

    p_decrypt = sub.add_parser(
        "decrypt-message",
        help="decrypt an AES-256-GCM payload locally (no server needed)")
    p_decrypt.add_argument("--session-id", required=True)
    p_decrypt.add_argument(
        "--key", required=True,
        help="base64-encoded 32-byte AES key, or @path to read it from a file")
    p_decrypt.add_argument("--nonce", required=True,
                           help="base64-encoded 12-byte nonce")
    p_decrypt.add_argument("--ciphertext", required=True,
                           help="base64-encoded ciphertext with appended GCM tag")
    p_decrypt.set_defaults(handler=cmd_decrypt_message)

    p_serve = sub.add_parser("serve", help="run the HTTP server")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8080)
    p_serve.set_defaults(handler=cmd_serve)
    return parser


def cmd_register(args: argparse.Namespace) -> int:
    """Call POST /v1/devices and print the single-line JSON response."""
    try:
        prekeys: List[Dict[str, str]] = [_parse_prekey(spec) for spec in args.prekey]
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"message": str(error), "field": "signed_prekeys"}),
              file=sys.stderr)
        return 2

    payload = {
        "user_id": args.user_id,
        "device_id": args.device_id,
        "identity_key": _read_key_argument(args.identity_key),
        "signed_prekeys": prekeys,
    }
    try:
        status, response = _request_json("POST", f"{args.base_url}/v1/devices",
                                         body=payload)
    except ServerUnavailable:
        return _emit_server_error()
    stream = sys.stdout if status == 201 else sys.stderr
    print(json.dumps(response, separators=(",", ":"), ensure_ascii=False), file=stream)
    return 0 if status == 201 else 1


def cmd_show(args: argparse.Namespace) -> int:
    """Call GET /v1/devices/{device_id} and print the single-line JSON response."""
    from urllib.parse import quote

    url = f"{args.base_url}/v1/devices/{quote(args.device_id, safe='')}"
    try:
        status, response = _request_json("GET", url)
    except ServerUnavailable:
        return _emit_server_error()
    stream = sys.stdout if status == 200 else sys.stderr
    print(json.dumps(response, separators=(",", ":"), ensure_ascii=False), file=stream)
    return 0 if status == 200 else 1


def cmd_revoke_device(args: argparse.Namespace) -> int:
    """Call POST /v1/devices/{device_id}/revoke and print the JSON response."""
    from urllib.parse import quote

    url = (f"{args.base_url}/v1/devices/"
           f"{quote(args.device_id, safe='')}/revoke")
    try:
        status, response = _request_json("POST", url)
    except ServerUnavailable:
        return _emit_server_error()
    stream = sys.stdout if status == 200 else sys.stderr
    print(json.dumps(response, separators=(",", ":"), ensure_ascii=False), file=stream)
    return 0 if status == 200 else 1


def cmd_revoke_prekey(args: argparse.Namespace) -> int:
    """Call POST /v1/devices/{device_id}/prekeys/{key_id}/revoke."""
    from urllib.parse import quote

    url = (f"{args.base_url}/v1/devices/"
           f"{quote(args.device_id, safe='')}/prekeys/"
           f"{quote(args.key_id, safe='')}/revoke")
    try:
        status, response = _request_json("POST", url)
    except ServerUnavailable:
        return _emit_server_error()
    stream = sys.stdout if status == 200 else sys.stderr
    print(json.dumps(response, separators=(",", ":"), ensure_ascii=False), file=stream)
    return 0 if status == 200 else 1


def cmd_create_session(args: argparse.Namespace) -> int:
    """Call POST /v1/sessions and print the single-line JSON response."""
    payload = {
        "initiator_device_id": args.initiator_device_id,
        "recipient_device_id": args.recipient_device_id,
        "prekey_id": args.prekey_id,
        "ephemeral_key": _read_key_argument(args.ephemeral_key),
    }
    try:
        status, response = _request_json("POST", f"{args.base_url}/v1/sessions",
                                         body=payload)
    except ServerUnavailable:
        return _emit_server_error()
    stream = sys.stdout if status == 201 else sys.stderr
    print(json.dumps(response, separators=(",", ":"), ensure_ascii=False), file=stream)
    return 0 if status == 201 else 1


def cmd_show_session(args: argparse.Namespace) -> int:
    """Call GET /v1/sessions/{session_id} and print the single-line JSON."""
    from urllib.parse import quote

    url = f"{args.base_url}/v1/sessions/{quote(args.session_id, safe='')}"
    try:
        status, response = _request_json("GET", url)
    except ServerUnavailable:
        return _emit_server_error()
    stream = sys.stdout if status == 200 else sys.stderr
    print(json.dumps(response, separators=(",", ":"), ensure_ascii=False), file=stream)
    return 0 if status == 200 else 1


def cmd_send_message(args: argparse.Namespace) -> int:
    """Call POST /v1/messages and print the single-line JSON response."""
    payload = {
        "session_id": args.session_id,
        "sender_device_id": args.sender_device_id,
        "message_id": args.message_id,
        "sequence": args.sequence,
        "nonce": args.nonce,
        "ciphertext": args.ciphertext,
    }
    try:
        status, response = _request_json("POST", f"{args.base_url}/v1/messages",
                                         body=payload)
    except ServerUnavailable:
        return _emit_server_error()
    stream = sys.stdout if status == 201 else sys.stderr
    print(json.dumps(response, separators=(",", ":"), ensure_ascii=False), file=stream)
    return 0 if status == 201 else 1


def cmd_pull_messages(args: argparse.Namespace) -> int:
    """Call GET /v1/messages/{session_id} and print the single-line JSON."""
    from urllib.parse import quote, urlencode

    query = urlencode({"device_id": args.device_id,
                       "after": args.after, "limit": args.limit})
    url = (f"{args.base_url}/v1/messages/"
           f"{quote(args.session_id, safe='')}?{query}")
    try:
        status, response = _request_json("GET", url)
    except ServerUnavailable:
        return _emit_server_error()
    stream = sys.stdout if status == 200 else sys.stderr
    print(json.dumps(response, separators=(",", ":"), ensure_ascii=False), file=stream)
    return 0 if status == 200 else 1


def _emit_crypto_error(error: CryptoError) -> int:
    """Print a local crypto failure as single-line JSON on stderr; exit 2."""
    print(json.dumps({"message": error.message, "field": error.field},
                     separators=(",", ":")),
          file=sys.stderr)
    return 2


def cmd_encrypt_message(args: argparse.Namespace) -> int:
    """Encrypt locally and print ``session_id``/``nonce``/``ciphertext``."""
    try:
        result = encrypt_message(args.session_id,
                                 _read_key_argument(args.key),
                                 _read_key_argument(args.plaintext))
    except OSError as error:
        return _emit_crypto_error(CryptoError(str(error), "plaintext"))
    except CryptoError as error:
        return _emit_crypto_error(error)
    print(json.dumps(result, separators=(",", ":"), ensure_ascii=False))
    return 0


def cmd_decrypt_message(args: argparse.Namespace) -> int:
    """Decrypt locally and print ``session_id``/``plaintext``."""
    try:
        result = decrypt_message(args.session_id,
                                 _read_key_argument(args.key),
                                 args.nonce, args.ciphertext)
    except OSError as error:
        return _emit_crypto_error(CryptoError(str(error), "key"))
    except CryptoError as error:
        return _emit_crypto_error(error)
    print(json.dumps(result, separators=(",", ":"), ensure_ascii=False))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Run the HTTP server until interrupted."""
    server, _ = create_server(args.host, args.port, DeviceService())
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)
