"""Command-line entry point: device and session API commands (plus ``serve``).

All API commands talk to an HTTP server and print the server's JSON response
on a single line, with exactly the field names used over HTTP.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib import error as urllib_error
from urllib import request as urllib_request

from .crypto import (CryptoError, decrypt_message,
                     derive_session_key, derive_verified_session_key,
                     encrypt_message, is_nonempty_string,
                     verify_prekey_proof)
from .http_app import create_server
from .locking import StateFileLocked, acquire_state_file_lock
from .persistence import StateFileError, attach_persistence
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


def _read_text_argument(value: str, field: str) -> str:
    """Read a text option: the value itself, or the UTF-8 file at ``@path``.

    A missing, unreadable or non-UTF-8 file raises :class:`CryptoError`
    naming *field*; a successfully read file keeps the existing whitespace
    handling (the content is stripped).
    """
    if value.startswith("@"):
        try:
            with open(value[1:], "r", encoding="utf-8") as handle:
                return handle.read().strip()
        except (OSError, UnicodeDecodeError) as error:
            raise CryptoError(f"cannot read {field} file: {error}",
                              field) from None
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


def _parse_verified_prekey(spec: str) -> Dict[str, str]:
    """Parse ``key_id:public_key:signature``; ``@path`` loads a JSON mapping.

    The signature is the standard-base64 64-byte Ed25519 signature over the
    pre-key proof. Neither base64 nor PEM text contains a colon, so the
    first two colons delimit the three fields (a split with at most two
    separators, so trailing padding ``=`` survives).
    """
    if spec.startswith("@"):
        with open(spec[1:], "r", encoding="utf-8") as handle:
            element = json.load(handle)
        if (not isinstance(element, dict)
                or not isinstance(element.get("key_id"), str)
                or not isinstance(element.get("public_key"), str)
                or not isinstance(element.get("signature"), str)):
            raise ValueError(
                "verified prekey file must contain key_id, public_key and "
                f"signature: {spec}")
        return {"key_id": element["key_id"],
                "public_key": element["public_key"],
                "signature": element["signature"]}

    parts = spec.split(":", 2)
    if len(parts) != 3 or not parts[0] \
            or not parts[1].strip() or not parts[2].strip():
        raise ValueError(
            "verified prekey must have the form "
            f"KEY_ID:PUBLIC_KEY:SIGNATURE (got: {spec!r})")
    return {"key_id": parts[0], "public_key": parts[1].strip(),
            "signature": parts[2].strip()}


def _add_envelope_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the optional message-envelope metadata group to a crypto command.

    The three flags form an all-or-none group: omitting all of them keeps
    the legacy session-id-only AAD, while supplying any of them requires
    all three to be valid. ``--sequence`` is parsed leniently here (no
    ``type=int``) so an invalid value is reported through the same
    single-line JSON error contract (exit 2) as the other crypto failures
    instead of an argparse usage error.
    """
    parser.add_argument("--sender-device-id",
                        help="sender device id bound to the ciphertext")
    parser.add_argument("--message-id",
                        help="message id bound to the ciphertext")
    parser.add_argument("--sequence",
                        help="positive integer sequence bound to the "
                             "ciphertext")


def _parse_sequence_argument(value: Optional[str]) -> Any:
    """Convert ``--sequence`` text to an int; pass unparseable text through.

    The crypto layer validates the metadata group in a fixed field order
    (sender device id, message id, sequence), so an unparseable sequence is
    forwarded as-is and reported there as ``field=sequence`` after any
    sender/message problem.
    """
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return value


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

    p_register_verified = sub.add_parser(
        "register-verified",
        help="register a device and publish pre-keys with Ed25519 proofs")
    p_register_verified.add_argument("--user-id", required=True)
    p_register_verified.add_argument("--device-id", required=True)
    p_register_verified.add_argument(
        "--identity-key", required=True,
        help="Ed25519 identity public key (string), or @path for a file")
    p_register_verified.add_argument(
        "--prekey", action="append", default=[],
        metavar="KEY_ID:PUBLIC_KEY:SIGNATURE",
        help="signed pre-key with its Ed25519 proof; repeatable. Use @path "
             "for a JSON object file with key_id/public_key/signature.")
    p_register_verified.set_defaults(handler=cmd_register_verified)

    p_show = sub.add_parser("show", help="show a device's public record")
    p_show.add_argument("device_id")
    p_show.set_defaults(handler=cmd_show)

    p_key_events = sub.add_parser(
        "key-events", help="list a device's key-audit event chain")
    p_key_events.add_argument("device_id")
    p_key_events.add_argument(
        "--after", type=int, default=0,
        help="start after this sequence (default: 0, the whole chain)")
    p_key_events.add_argument("--limit", type=int, default=100)
    p_key_events.set_defaults(handler=cmd_key_events)

    p_revoke_device = sub.add_parser("revoke-device", help="revoke a device")
    p_revoke_device.add_argument("--device-id", required=True)
    p_revoke_device.set_defaults(handler=cmd_revoke_device)

    p_revoke_prekey = sub.add_parser("revoke-prekey", help="revoke a signed pre-key")
    p_revoke_prekey.add_argument("--device-id", required=True)
    p_revoke_prekey.add_argument("--key-id", required=True)
    p_revoke_prekey.set_defaults(handler=cmd_revoke_prekey)

    p_rotate_identity = sub.add_parser(
        "rotate-identity-key",
        help="rotate a device's identity public key")
    p_rotate_identity.add_argument("--device-id", required=True)
    p_rotate_identity.add_argument(
        "--identity-key", required=True,
        help="new identity public key (string), or @path to read it from a file")
    p_rotate_identity.set_defaults(handler=cmd_rotate_identity_key)

    p_rotate_identity_verified = sub.add_parser(
        "rotate-identity-key-verified",
        help="rotate a device's identity key with a signed authorization")
    p_rotate_identity_verified.add_argument("--device-id", required=True)
    p_rotate_identity_verified.add_argument(
        "--identity-key", required=True,
        help="new Ed25519 identity public key (string), or @path to read it "
             "from a file")
    p_rotate_identity_verified.add_argument(
        "--expected-version", required=True, type=_parse_sequence_argument,
        help="current identity_key_version (positive integer)")
    p_rotate_identity_verified.add_argument(
        "--signature", required=True,
        help="standard-base64 64-byte Ed25519 rotation authorization, or "
             "@path to read it from a file")
    p_rotate_identity_verified.set_defaults(
        handler=cmd_rotate_identity_key_verified)

    p_fingerprint = sub.add_parser(
        "fingerprint",
        help="show a device's identity fingerprint as seen by a verifier")
    p_fingerprint.add_argument("device_id")
    p_fingerprint.add_argument(
        "--verifier-device-id", required=True,
        help="device id of the peer performing the out-of-band comparison")
    p_fingerprint.set_defaults(handler=cmd_fingerprint)

    p_verify_identity = sub.add_parser(
        "verify-identity",
        help="explicitly confirm a peer device's identity fingerprint")
    p_verify_identity.add_argument("device_id")
    p_verify_identity.add_argument(
        "--verifier-device-id", required=True,
        help="device id of the verifying peer (must differ from device_id)")
    p_verify_identity.add_argument(
        "--verification-id", required=True,
        help="client-chosen id of this confirmation (globally unique)")
    p_verify_identity.add_argument(
        "--expected-fingerprint", required=True,
        help="64 lowercase hex characters compared against the current key")
    p_verify_identity.set_defaults(handler=cmd_verify_identity)

    p_add_prekey = sub.add_parser(
        "add-prekey", help="append a signed pre-key to a device")
    p_add_prekey.add_argument("--device-id", required=True)
    p_add_prekey.add_argument("--key-id", required=True)
    p_add_prekey.add_argument(
        "--public-key", required=True,
        help="pre-key public key (string), or @path to read it from a file")
    p_add_prekey.set_defaults(handler=cmd_add_prekey)

    p_add_prekey_verified = sub.add_parser(
        "add-prekey-verified",
        help="append an identity-authorized signed pre-key to a device")
    p_add_prekey_verified.add_argument("--device-id", required=True)
    p_add_prekey_verified.add_argument("--key-id", required=True)
    p_add_prekey_verified.add_argument(
        "--public-key", required=True,
        help="pre-key public key (string), or @path to read it from a file")
    p_add_prekey_verified.add_argument(
        "--signature", required=True,
        help="standard-base64 64-byte Ed25519 signature over the pre-key "
             "proof, or @path to read it from a file")
    p_add_prekey_verified.set_defaults(handler=cmd_add_prekey_verified)

    p_add_prekeys_verified = sub.add_parser(
        "add-prekeys-verified",
        help="append a batch of identity-authorized signed pre-keys")
    p_add_prekeys_verified.add_argument("--device-id", required=True)
    p_add_prekeys_verified.add_argument(
        "--prekey", action="append", default=[],
        metavar="KEY_ID:PUBLIC_KEY:SIGNATURE",
        help="signed pre-key with its Ed25519 proof; repeatable. Use @path "
             "for a JSON object file with key_id/public_key/signature.")
    p_add_prekeys_verified.set_defaults(handler=cmd_add_prekeys_verified)

    p_claim_prekey = sub.add_parser(
        "claim-prekey", help="claim one of a device's one-time pre-keys")
    p_claim_prekey.add_argument("--recipient-device-id", required=True)
    p_claim_prekey.add_argument("--claim-id", required=True)
    p_claim_prekey.set_defaults(handler=cmd_claim_prekey)

    p_claim_user_prekeys = sub.add_parser(
        "claim-user-prekeys",
        help="claim one pre-key from every active device of a user in one batch")
    p_claim_user_prekeys.add_argument("--user-id", required=True)
    p_claim_user_prekeys.add_argument("--claim-id", required=True)
    p_claim_user_prekeys.set_defaults(handler=cmd_claim_user_prekeys)

    p_create_session = sub.add_parser(
        "create-session", help="negotiate a session between two devices")
    p_create_session.add_argument("--initiator-device-id", required=True)
    p_create_session.add_argument("--recipient-device-id", required=True)
    p_create_session.add_argument("--prekey-id", required=True)
    p_create_session.add_argument(
        "--ephemeral-key", required=True,
        help="ephemeral public key (string), or @path to read it from a file")
    p_create_session.set_defaults(handler=cmd_create_session)

    p_create_session_from_claim = sub.add_parser(
        "create-session-from-claim",
        help="establish a session from a one-time pre-key claim")
    p_create_session_from_claim.add_argument("--claim-id", required=True)
    p_create_session_from_claim.add_argument("--initiator-device-id",
                                             required=True)
    p_create_session_from_claim.add_argument(
        "--ephemeral-key", required=True,
        help="ephemeral public key (string), or @path to read it from a file")
    p_create_session_from_claim.set_defaults(
        handler=cmd_create_session_from_claim)

    p_create_batch_sessions = sub.add_parser(
        "create-batch-sessions",
        help="establish one session per device of a batch pre-key claim")
    p_create_batch_sessions.add_argument("--claim-id", required=True)
    p_create_batch_sessions.add_argument("--initiator-device-id",
                                         required=True)
    p_create_batch_sessions.add_argument(
        "--ephemeral-key", action="append", default=[],
        metavar="DEVICE_ID:PUBLIC_KEY",
        help="ephemeral public key for one claimed device; repeat once per "
             "claimed device. The key may be @path to read it from a file.")
    p_create_batch_sessions.set_defaults(
        handler=cmd_create_batch_sessions)

    p_show_session = sub.add_parser("show-session", help="show a session snapshot")
    p_show_session.add_argument("session_id")
    p_show_session.set_defaults(handler=cmd_show_session)

    p_rotate_session = sub.add_parser(
        "rotate-session",
        help="rotate a one-to-one session into a fresh successor")
    p_rotate_session.add_argument("session_id")
    p_rotate_session.add_argument("--rotation-id", required=True)
    p_rotate_session.add_argument(
        "--actor-device-id", required=True,
        help="the predecessor session's initiator device")
    p_rotate_session.add_argument(
        "--prekey-id", required=True,
        help="a currently valid pre-key of the recipient")
    p_rotate_session.add_argument(
        "--ephemeral-key", required=True,
        help="new ephemeral public key (string), or @path to read from a file")
    p_rotate_session.set_defaults(handler=cmd_rotate_session)

    p_group_create = sub.add_parser("group-create", help="create a member group")
    p_group_create.add_argument("--group-id", required=True)
    p_group_create.add_argument("--creator-device-id", required=True)
    p_group_create.add_argument(
        "--member-device-id", action="append", default=[], metavar="DEVICE_ID",
        help="initial member device id (besides the creator); repeatable")
    p_group_create.set_defaults(handler=cmd_group_create)

    p_group_show = sub.add_parser("group-show", help="show a group snapshot")
    p_group_show.add_argument("group_id")
    p_group_show.set_defaults(handler=cmd_group_show)

    p_group_add = sub.add_parser(
        "group-add-member", help="add a device to a group (creator only)")
    p_group_add.add_argument("--group-id", required=True)
    p_group_add.add_argument("--actor-device-id", required=True)
    p_group_add.add_argument("--device-id", required=True)
    p_group_add.set_defaults(handler=cmd_group_add_member)

    p_group_remove = sub.add_parser(
        "group-remove-member", help="remove a device from a group (creator only)")
    p_group_remove.add_argument("--group-id", required=True)
    p_group_remove.add_argument("--actor-device-id", required=True)
    p_group_remove.add_argument("--device-id", required=True)
    p_group_remove.set_defaults(handler=cmd_group_remove_member)

    p_create_group_session = sub.add_parser(
        "create-group-session", help="freeze a group into a group session")
    p_create_group_session.add_argument("--group-id", required=True)
    p_create_group_session.add_argument("--initiator-device-id", required=True)
    p_create_group_session.add_argument(
        "--ephemeral-key", required=True,
        help="ephemeral public key (string), or @path to read it from a file")
    p_create_group_session.set_defaults(handler=cmd_create_group_session)

    p_show_group_session = sub.add_parser(
        "show-group-session", help="show a frozen group-session snapshot")
    p_show_group_session.add_argument("session_id")
    p_show_group_session.set_defaults(handler=cmd_show_group_session)

    p_rotate_group_session = sub.add_parser(
        "rotate-group-session",
        help="rotate a group session into a fresh frozen successor")
    p_rotate_group_session.add_argument("session_id")
    p_rotate_group_session.add_argument("--rotation-id", required=True)
    p_rotate_group_session.add_argument(
        "--actor-device-id", required=True)
    p_rotate_group_session.add_argument(
        "--ephemeral-key", required=True,
        help="ephemeral public key (string), or @path to read it from a file")
    p_rotate_group_session.add_argument(
        "--expected-revision", required=True, type=int)
    p_rotate_group_session.set_defaults(
        handler=cmd_rotate_group_session)

    p_sync_group_messages = sub.add_parser(
        "sync-group-messages",
        help="sync a page of a group session's messages for a device")
    p_sync_group_messages.add_argument("session_id")
    p_sync_group_messages.add_argument("--device-id", required=True)
    p_sync_group_messages.add_argument(
        "--after", type=int, default=None,
        help="start after this sequence (default: use the device's stored "
             "cursor and advance it)")
    p_sync_group_messages.add_argument("--limit", type=int, default=100)
    p_sync_group_messages.set_defaults(handler=cmd_sync_group_messages)

    p_sync_checkpoint = sub.add_parser(
        "sync-checkpoint",
        help="move a device's group-session sync cursor forward")
    p_sync_checkpoint.add_argument("session_id")
    p_sync_checkpoint.add_argument("--device-id", required=True)
    p_sync_checkpoint.add_argument("--cursor", required=True, type=int)
    p_sync_checkpoint.set_defaults(handler=cmd_sync_checkpoint)

    p_sync_session_messages = sub.add_parser(
        "sync-session-messages",
        help="sync a page of a 1:1 or group session's messages for a device")
    p_sync_session_messages.add_argument("session_id")
    p_sync_session_messages.add_argument("--device-id", required=True)
    p_sync_session_messages.add_argument(
        "--after", type=int, default=None,
        help="start after this sequence (default: use the device's stored "
             "cursor and advance it)")
    p_sync_session_messages.add_argument("--limit", type=int, default=100)
    p_sync_session_messages.set_defaults(handler=cmd_sync_session_messages)

    p_sync_session_checkpoint = sub.add_parser(
        "sync-session-checkpoint",
        help="move a device's 1:1/group-session sync cursor forward")
    p_sync_session_checkpoint.add_argument("session_id")
    p_sync_session_checkpoint.add_argument("--device-id", required=True)
    p_sync_session_checkpoint.add_argument("--cursor", required=True, type=int)
    p_sync_session_checkpoint.set_defaults(
        handler=cmd_sync_session_checkpoint)

    p_send_message = sub.add_parser(
        "send-message", help="send an encrypted message envelope into a session")
    p_send_message.add_argument("--session-id", required=True)
    p_send_message.add_argument("--sender-device-id", required=True)
    p_send_message.add_argument("--message-id", required=True)
    p_send_message.add_argument("--sequence", required=True, type=int)
    p_send_message.add_argument("--nonce", required=True)
    p_send_message.add_argument("--ciphertext", required=True)
    p_send_message.set_defaults(handler=cmd_send_message)

    p_submit_message = sub.add_parser(
        "submit-message",
        help="idempotently submit an encrypted message envelope")
    p_submit_message.add_argument("--request-id", required=True)
    p_submit_message.add_argument("--session-id", required=True)
    p_submit_message.add_argument("--sender-device-id", required=True)
    p_submit_message.add_argument("--message-id", required=True)
    p_submit_message.add_argument("--sequence", required=True, type=int)
    p_submit_message.add_argument("--nonce", required=True)
    p_submit_message.add_argument("--ciphertext", required=True)
    p_submit_message.set_defaults(handler=cmd_submit_message)

    p_pull_messages = sub.add_parser(
        "pull-messages", help="pull a page of messages from a session")
    p_pull_messages.add_argument("session_id")
    p_pull_messages.add_argument("--device-id", required=True)
    p_pull_messages.add_argument("--after", type=int, default=0)
    p_pull_messages.add_argument("--limit", type=int, default=100)
    p_pull_messages.set_defaults(handler=cmd_pull_messages)

    p_retry = sub.add_parser(
        "retry-message",
        help="record a reliable-delivery retry attempt for a message")
    p_retry.add_argument("session_id")
    p_retry.add_argument("message_id")
    p_retry.add_argument("--device-id", required=True)
    p_retry.add_argument("--attempt-id", required=True)
    p_retry.set_defaults(handler=cmd_retry_message)

    p_ack = sub.add_parser(
        "ack-message", help="acknowledge a message for a device")
    p_ack.add_argument("session_id")
    p_ack.add_argument("--device-id", required=True)
    p_ack.add_argument("--message-id", required=True)
    p_ack.add_argument("--sequence", required=True, type=int)
    p_ack.set_defaults(handler=cmd_ack_message)

    p_status = sub.add_parser(
        "message-status", help="show a message's delivery status")
    p_status.add_argument("session_id")
    p_status.add_argument("message_id")
    p_status.add_argument("--device-id", required=True)
    p_status.set_defaults(handler=cmd_message_status)

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
    _add_envelope_arguments(p_encrypt)
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
    _add_envelope_arguments(p_decrypt)
    p_decrypt.set_defaults(handler=cmd_decrypt_message)

    # The options are deliberately not marked required=True: a missing option
    # must surface through the same single-line JSON error contract (exit 2)
    # as the other derivation failures, not as an argparse usage error.
    p_derive = sub.add_parser(
        "derive-session-key",
        help="derive a one-to-one session key locally via X25519+HKDF "
             "(no server needed)")
    p_derive.add_argument("--session-id",
                          help="session id bound into the derived key")
    p_derive.add_argument(
        "--private-key",
        help="canonical standard-base64 raw 32-byte X25519 private key")
    p_derive.add_argument(
        "--peer-public-key",
        help="peer X25519 public key (any accepted public-key encoding)")
    p_derive.set_defaults(handler=cmd_derive_session_key)

    # The options are deliberately not marked required=True: a missing option
    # must surface through the same single-line JSON error contract (exit 2)
    # as the other verification failures, not as an argparse usage error.
    p_verify_prekey_proof = sub.add_parser(
        "verify-prekey-proof",
        help="verify a published signed pre-key proof offline against a "
             "trusted identity fingerprint (no server needed)")
    p_verify_prekey_proof.add_argument(
        "--proof",
        help="proof JSON object text, or @path to read it from a UTF-8 file")
    p_verify_prekey_proof.add_argument(
        "--user-id", help="expected user id, compared with the proof as-is")
    p_verify_prekey_proof.add_argument(
        "--device-id", help="expected device id, compared with the proof as-is")
    p_verify_prekey_proof.add_argument(
        "--key-id", help="expected pre-key id, compared with the proof as-is")
    p_verify_prekey_proof.add_argument(
        "--expected-fingerprint",
        help="64 lowercase hex characters of the trusted identity fingerprint")
    p_verify_prekey_proof.set_defaults(handler=cmd_verify_prekey_proof)

    # The options are deliberately not marked required=True: a missing option
    # must surface through the same single-line JSON error contract (exit 2)
    # as the other derivation failures, not as an argparse usage error.
    p_derive_verified = sub.add_parser(
        "derive-verified-session-key",
        help="derive a one-to-one session key locally from a session "
             "snapshot and a verified signed pre-key proof (no server needed)")
    p_derive_verified.add_argument(
        "--session",
        help="eight-field session snapshot JSON object text, or @path to "
             "read it from a UTF-8 file")
    p_derive_verified.add_argument(
        "--proof",
        help="signed pre-key proof JSON object text, or @path to read it "
             "from a UTF-8 file")
    p_derive_verified.add_argument(
        "--private-key",
        help="canonical standard-base64 raw 32-byte X25519 ephemeral private "
             "key whose public key matches the snapshot's ephemeral_key")
    p_derive_verified.add_argument(
        "--user-id", help="expected recipient user id, compared with the "
                          "proof as-is")
    p_derive_verified.add_argument(
        "--expected-fingerprint",
        help="64 lowercase hex characters of the trusted identity fingerprint")
    p_derive_verified.add_argument(
        "--role",
        help='optional role: "initiator" (the default; --private-key is the '
             'initiator\'s ephemeral private key) or "recipient" '
             '(--private-key is the recipient\'s pre-key private key); any '
             'other value fails with field "role"')
    p_derive_verified.set_defaults(handler=cmd_derive_verified_session_key)

    p_integrity_history_page = sub.add_parser(
        "integrity-history-page",
        help="page through the append-only persistence integrity history")
    p_integrity_history_page.add_argument(
        "--after", type=int, default=0,
        help="return entries with commit_seq strictly after this generation "
             "(default: 0)")
    p_integrity_history_page.add_argument(
        "--limit", type=int, default=100,
        help="maximum number of entries to return, 1..100 (default: 100)")
    p_integrity_history_page.set_defaults(
        handler=cmd_integrity_history_page)

    p_serve = sub.add_parser("serve", help="run the HTTP server")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8080)
    p_serve.add_argument(
        "--data-file", default=None,
        help="state file path (default: $E2EE_DATA_FILE; otherwise in-memory)")
    p_serve.set_defaults(handler=cmd_serve)
    return parser


def _build_serve_service(data_file: Optional[str]) -> DeviceService:
    """Build the service for ``serve``.

    Uses *data_file* when given, else ``$E2EE_DATA_FILE``, else a purely
    in-memory store. With a file, its contents are restored at startup and
    every subsequent change is persisted atomically; a corrupt or
    wrong-version file makes startup fail with :class:`StateFileError`.
    """
    path = _serve_data_file_path(data_file)
    service = DeviceService()
    if path:
        attach_persistence(service, path)
    return service


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


def cmd_register_verified(args: argparse.Namespace) -> int:
    """Call POST /v1/devices/verified and print the single-line JSON."""
    try:
        prekeys: List[Dict[str, str]] = [
            _parse_verified_prekey(spec) for spec in args.prekey]
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
        status, response = _request_json(
            "POST", f"{args.base_url}/v1/devices/verified", body=payload)
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


def cmd_key_events(args: argparse.Namespace) -> int:
    """Call GET /v1/devices/{device_id}/key-events and print the JSON."""
    from urllib.parse import quote, urlencode

    query = urlencode({"after": args.after, "limit": args.limit})
    url = (f"{args.base_url}/v1/devices/"
           f"{quote(args.device_id, safe='')}/key-events?{query}")
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


def cmd_rotate_identity_key(args: argparse.Namespace) -> int:
    """Call POST /v1/devices/{id}/identity-key/rotate and print the JSON."""
    from urllib.parse import quote

    url = (f"{args.base_url}/v1/devices/"
           f"{quote(args.device_id, safe='')}/identity-key/rotate")
    payload = {"identity_key": _read_key_argument(args.identity_key)}
    try:
        status, response = _request_json("POST", url, body=payload)
    except ServerUnavailable:
        return _emit_server_error()
    return _emit_api_response(status, response)


def cmd_rotate_identity_key_verified(args: argparse.Namespace) -> int:
    """Call POST /v1/devices/{id}/identity-key/rotate-verified; print the JSON.

    The signature is the standard-base64 64-byte Ed25519 authorization over
    the rotation message, verified server-side against the device's current
    identity key. Success (200) prints the single-line JSON on stdout and
    exits 0; any failure prints the server's single-line JSON with its
    ``field`` on stderr and exits non-zero.
    """
    from urllib.parse import quote

    url = (f"{args.base_url}/v1/devices/"
           f"{quote(args.device_id, safe='')}/identity-key/rotate-verified")
    payload = {"identity_key": _read_key_argument(args.identity_key),
               "expected_version": args.expected_version,
               "signature": _read_key_argument(args.signature)}
    try:
        status, response = _request_json("POST", url, body=payload)
    except ServerUnavailable:
        return _emit_server_error()
    return _emit_api_response(status, response)


def cmd_fingerprint(args: argparse.Namespace) -> int:
    """Call GET /v1/devices/{id}/identity-fingerprint and print the JSON."""
    from urllib.parse import quote, urlencode

    query = urlencode({"verifier_device_id": args.verifier_device_id})
    url = (f"{args.base_url}/v1/devices/"
           f"{quote(args.device_id, safe='')}/identity-fingerprint?{query}")
    try:
        status, response = _request_json("GET", url)
    except ServerUnavailable:
        return _emit_server_error()
    return _emit_api_response(status, response)


def cmd_verify_identity(args: argparse.Namespace) -> int:
    """Call POST /v1/devices/{id}/identity-verifications (201/200 success)."""
    from urllib.parse import quote

    url = (f"{args.base_url}/v1/devices/"
           f"{quote(args.device_id, safe='')}/identity-verifications")
    payload = {
        "verifier_device_id": args.verifier_device_id,
        "verification_id": args.verification_id,
        "expected_fingerprint": args.expected_fingerprint,
    }
    try:
        status, response = _request_json("POST", url, body=payload)
    except ServerUnavailable:
        return _emit_server_error()
    return _emit_api_response(status, response)


def cmd_add_prekey(args: argparse.Namespace) -> int:
    """Call POST /v1/devices/{id}/prekeys; 201 created or 200 idempotent."""
    from urllib.parse import quote

    url = (f"{args.base_url}/v1/devices/"
           f"{quote(args.device_id, safe='')}/prekeys")
    payload = {"key_id": args.key_id,
               "public_key": _read_key_argument(args.public_key)}
    try:
        status, response = _request_json("POST", url, body=payload)
    except ServerUnavailable:
        return _emit_server_error()
    return _emit_api_response(status, response)


def cmd_add_prekey_verified(args: argparse.Namespace) -> int:
    """Call POST /v1/devices/{id}/prekeys/verified; 201 created or 200 replay.

    The signature is the standard-base64 64-byte Ed25519 proof over the
    pre-key proof message, verified server-side against the device's current
    identity key. Either success status prints the single-line JSON on stdout
    and exits 0; any failure prints the server's single-line JSON with its
    ``field`` on stderr and exits non-zero.
    """
    from urllib.parse import quote

    url = (f"{args.base_url}/v1/devices/"
           f"{quote(args.device_id, safe='')}/prekeys/verified")
    payload = {"key_id": args.key_id,
               "public_key": _read_key_argument(args.public_key),
               "signature": _read_key_argument(args.signature)}
    try:
        status, response = _request_json("POST", url, body=payload)
    except ServerUnavailable:
        return _emit_server_error()
    return _emit_api_response(status, response)


def cmd_add_prekeys_verified(args: argparse.Namespace) -> int:
    """Call POST /v1/devices/{id}/prekeys/verified-batch; 201 or 200 replay.

    Each ``--prekey`` is ``KEY_ID:PUBLIC_KEY:SIGNATURE`` (or ``@path`` to a
    JSON object file), exactly as for ``register-verified``; the batch is
    published all-or-nothing. A 2xx response prints the single-line JSON on
    stdout and exits 0; any failure prints the server's single-line JSON
    with its ``field`` on stderr and exits non-zero. An unparseable
    ``--prekey`` is a local error (exit 2, field ``signed_prekeys``).
    """
    from urllib.parse import quote

    try:
        prekeys: List[Dict[str, str]] = [
            _parse_verified_prekey(spec) for spec in args.prekey]
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"message": str(error), "field": "signed_prekeys"}),
              file=sys.stderr)
        return 2

    url = (f"{args.base_url}/v1/devices/"
           f"{quote(args.device_id, safe='')}/prekeys/verified-batch")
    payload = {"signed_prekeys": prekeys}
    try:
        status, response = _request_json("POST", url, body=payload)
    except ServerUnavailable:
        return _emit_server_error()
    return _emit_api_response(status, response)


def cmd_claim_prekey(args: argparse.Namespace) -> int:
    """Call POST /v1/prekeys/claim; 201 created or 200 idempotent replay.

    Either success status prints the single-line claim JSON on stdout and
    exits 0; any failure prints the server's single-line JSON on stderr and
    exits non-zero.
    """
    payload = {"recipient_device_id": args.recipient_device_id,
               "claim_id": args.claim_id}
    try:
        status, response = _request_json(
            "POST", f"{args.base_url}/v1/prekeys/claim", body=payload)
    except ServerUnavailable:
        return _emit_server_error()
    return _emit_api_response(status, response)


def cmd_claim_user_prekeys(args: argparse.Namespace) -> int:
    """Call POST /v1/prekeys/claim-batch; 201 created or 200 idempotent replay.

    Either success status prints the single-line batch claim JSON on stdout
    and exits 0; any failure prints the server's single-line JSON on stderr
    and exits non-zero.
    """
    payload = {"user_id": args.user_id, "claim_id": args.claim_id}
    try:
        status, response = _request_json(
            "POST", f"{args.base_url}/v1/prekeys/claim-batch", body=payload)
    except ServerUnavailable:
        return _emit_server_error()
    return _emit_api_response(status, response)


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


def cmd_create_session_from_claim(args: argparse.Namespace) -> int:
    """Call POST /v1/sessions/from-claim; print the single-line JSON response."""
    payload = {
        "claim_id": args.claim_id,
        "initiator_device_id": args.initiator_device_id,
        "ephemeral_key": _read_key_argument(args.ephemeral_key),
    }
    try:
        status, response = _request_json(
            "POST", f"{args.base_url}/v1/sessions/from-claim", body=payload)
    except ServerUnavailable:
        return _emit_server_error()
    stream = sys.stdout if status == 201 else sys.stderr
    print(json.dumps(response, separators=(",", ":"), ensure_ascii=False), file=stream)
    return 0 if status == 201 else 1


def cmd_create_batch_sessions(args: argparse.Namespace) -> int:
    """Call POST /v1/sessions/from-batch-claim; print single-line JSON.

    Each ``--ephemeral-key`` has the form ``DEVICE_ID:PUBLIC_KEY`` (the key
    part may itself use ``@path`` to read from a file), repeated once per
    claimed device in any order; the server lists the resulting sessions in
    the claim's frozen order. Success prints the response on stdout and
    exits 0; any failure prints the server's single-line JSON on stderr and
    exits non-zero.
    """
    ephemeral_keys: List[Dict[str, str]] = []
    try:
        for spec in args.ephemeral_key:
            device_id, sep, public_spec = spec.partition(":")
            if not sep or not device_id or not public_spec.strip():
                raise ValueError(
                    "ephemeral key must have the form "
                    f"DEVICE_ID:PUBLIC_KEY (got: {spec!r})")
            ephemeral_keys.append({
                "device_id": device_id,
                "ephemeral_key": _read_key_argument(public_spec.strip()),
            })
    except (OSError, ValueError) as error:
        print(json.dumps({"message": str(error), "field": "ephemeral_keys"}),
              file=sys.stderr)
        return 2

    payload = {
        "claim_id": args.claim_id,
        "initiator_device_id": args.initiator_device_id,
        "ephemeral_keys": ephemeral_keys,
    }
    try:
        status, response = _request_json(
            "POST", f"{args.base_url}/v1/sessions/from-batch-claim",
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


def cmd_rotate_session(args: argparse.Namespace) -> int:
    """Call POST /v1/sessions/{sid}/rotate; 201 or 200 both succeed."""
    from urllib.parse import quote

    payload = {
        "rotation_id": args.rotation_id,
        "actor_device_id": args.actor_device_id,
        "prekey_id": args.prekey_id,
        "ephemeral_key": _read_key_argument(args.ephemeral_key),
    }
    url = (f"{args.base_url}/v1/sessions/"
           f"{quote(args.session_id, safe='')}/rotate")
    try:
        status, response = _request_json("POST", url, body=payload)
    except ServerUnavailable:
        return _emit_server_error()
    stream = sys.stdout if status in (200, 201) else sys.stderr
    print(json.dumps(response, separators=(",", ":"), ensure_ascii=False), file=stream)
    return 0 if status in (200, 201) else 1


def cmd_group_create(args: argparse.Namespace) -> int:
    """Call POST /v1/groups and print the single-line JSON response."""
    payload = {
        "group_id": args.group_id,
        "creator_device_id": args.creator_device_id,
        "member_device_ids": args.member_device_id,
    }
    try:
        status, response = _request_json("POST", f"{args.base_url}/v1/groups",
                                         body=payload)
    except ServerUnavailable:
        return _emit_server_error()
    stream = sys.stdout if status == 201 else sys.stderr
    print(json.dumps(response, separators=(",", ":"), ensure_ascii=False), file=stream)
    return 0 if status == 201 else 1


def cmd_group_show(args: argparse.Namespace) -> int:
    """Call GET /v1/groups/{group_id} and print the single-line JSON."""
    from urllib.parse import quote

    url = f"{args.base_url}/v1/groups/{quote(args.group_id, safe='')}"
    try:
        status, response = _request_json("GET", url)
    except ServerUnavailable:
        return _emit_server_error()
    stream = sys.stdout if status == 200 else sys.stderr
    print(json.dumps(response, separators=(",", ":"), ensure_ascii=False), file=stream)
    return 0 if status == 200 else 1


def cmd_group_add_member(args: argparse.Namespace) -> int:
    """Call POST /v1/groups/{group_id}/members; 201 created or 200 duplicate."""
    from urllib.parse import quote

    url = (f"{args.base_url}/v1/groups/"
           f"{quote(args.group_id, safe='')}/members")
    payload = {"actor_device_id": args.actor_device_id,
               "device_id": args.device_id}
    try:
        status, response = _request_json("POST", url, body=payload)
    except ServerUnavailable:
        return _emit_server_error()
    return _emit_api_response(status, response)


def cmd_group_remove_member(args: argparse.Namespace) -> int:
    """Call POST /v1/groups/{group_id}/members/remove; always 200 on success."""
    from urllib.parse import quote

    url = (f"{args.base_url}/v1/groups/"
           f"{quote(args.group_id, safe='')}/members/remove")
    payload = {"actor_device_id": args.actor_device_id,
               "device_id": args.device_id}
    try:
        status, response = _request_json("POST", url, body=payload)
    except ServerUnavailable:
        return _emit_server_error()
    return _emit_api_response(status, response)


def cmd_create_group_session(args: argparse.Namespace) -> int:
    """Call POST /v1/group-sessions and print the single-line JSON response."""
    payload = {
        "group_id": args.group_id,
        "initiator_device_id": args.initiator_device_id,
        "ephemeral_key": _read_key_argument(args.ephemeral_key),
    }
    try:
        status, response = _request_json(
            "POST", f"{args.base_url}/v1/group-sessions", body=payload)
    except ServerUnavailable:
        return _emit_server_error()
    stream = sys.stdout if status == 201 else sys.stderr
    print(json.dumps(response, separators=(",", ":"), ensure_ascii=False), file=stream)
    return 0 if status == 201 else 1


def cmd_show_group_session(args: argparse.Namespace) -> int:
    """Call GET /v1/group-sessions/{session_id} and print the single-line JSON."""
    from urllib.parse import quote

    url = (f"{args.base_url}/v1/group-sessions/"
           f"{quote(args.session_id, safe='')}")
    try:
        status, response = _request_json("GET", url)
    except ServerUnavailable:
        return _emit_server_error()
    stream = sys.stdout if status == 200 else sys.stderr
    print(json.dumps(response, separators=(",", ":"), ensure_ascii=False), file=stream)
    return 0 if status == 200 else 1


def cmd_rotate_group_session(args: argparse.Namespace) -> int:
    """Call POST /v1/group-sessions/{sid}/rotate; 201 or 200 both succeed."""
    from urllib.parse import quote

    payload = {
        "rotation_id": args.rotation_id,
        "actor_device_id": args.actor_device_id,
        "ephemeral_key": _read_key_argument(args.ephemeral_key),
        "expected_revision": args.expected_revision,
    }
    url = (f"{args.base_url}/v1/group-sessions/"
           f"{quote(args.session_id, safe='')}/rotate")
    try:
        status, response = _request_json("POST", url, body=payload)
    except ServerUnavailable:
        return _emit_server_error()
    stream = sys.stdout if status in (200, 201) else sys.stderr
    print(json.dumps(response, separators=(",", ":"), ensure_ascii=False), file=stream)
    return 0 if status in (200, 201) else 1


def cmd_sync_group_messages(args: argparse.Namespace) -> int:
    """Call GET /v1/group-sessions/{sid}/sync and print the single-line JSON.

    With ``--after`` omitted the query carries no ``after`` parameter, so the
    server starts from and advances the device's stored cursor; with
    ``--after N`` it queries from N without moving the stored cursor.
    """
    from urllib.parse import quote, urlencode

    params: Dict[str, Any] = {"device_id": args.device_id,
                             "limit": args.limit}
    if args.after is not None:
        params["after"] = args.after
    url = (f"{args.base_url}/v1/group-sessions/"
           f"{quote(args.session_id, safe='')}/sync?{urlencode(params)}")
    try:
        status, response = _request_json("GET", url)
    except ServerUnavailable:
        return _emit_server_error()
    stream = sys.stdout if status == 200 else sys.stderr
    print(json.dumps(response, separators=(",", ":"), ensure_ascii=False), file=stream)
    return 0 if status == 200 else 1


def cmd_sync_checkpoint(args: argparse.Namespace) -> int:
    """Call POST /v1/group-sessions/{sid}/sync/checkpoint; 201 or 200 succeed."""
    from urllib.parse import quote

    url = (f"{args.base_url}/v1/group-sessions/"
           f"{quote(args.session_id, safe='')}/sync/checkpoint")
    payload = {"device_id": args.device_id, "cursor": args.cursor}
    try:
        status, response = _request_json("POST", url, body=payload)
    except ServerUnavailable:
        return _emit_server_error()
    return _emit_api_response(status, response)


def cmd_sync_session_messages(args: argparse.Namespace) -> int:
    """Call GET /v1/sessions/{sid}/sync (1:1 or group) and print single-line JSON.

    With ``--after`` omitted the query carries no ``after`` parameter, so the
    server starts from and advances the device's stored cursor; with
    ``--after N`` it queries from N without moving the stored cursor.
    """
    from urllib.parse import quote, urlencode

    params: Dict[str, Any] = {"device_id": args.device_id,
                             "limit": args.limit}
    if args.after is not None:
        params["after"] = args.after
    url = (f"{args.base_url}/v1/sessions/"
           f"{quote(args.session_id, safe='')}/sync?{urlencode(params)}")
    try:
        status, response = _request_json("GET", url)
    except ServerUnavailable:
        return _emit_server_error()
    stream = sys.stdout if status == 200 else sys.stderr
    print(json.dumps(response, separators=(",", ":"), ensure_ascii=False), file=stream)
    return 0 if status == 200 else 1


def cmd_sync_session_checkpoint(args: argparse.Namespace) -> int:
    """Call POST /v1/sessions/{sid}/sync/checkpoint; 201 or 200 both succeed."""
    from urllib.parse import quote

    url = (f"{args.base_url}/v1/sessions/"
           f"{quote(args.session_id, safe='')}/sync/checkpoint")
    payload = {"device_id": args.device_id, "cursor": args.cursor}
    try:
        status, response = _request_json("POST", url, body=payload)
    except ServerUnavailable:
        return _emit_server_error()
    return _emit_api_response(status, response)


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


def cmd_submit_message(args: argparse.Namespace) -> int:
    """Call POST /v1/messages/submit; 201 committed or 200 idempotent replay.

    Either success status prints the single-line JSON on stdout and exits 0;
    any failure prints the server's single-line JSON on stderr and exits
    non-zero.
    """
    payload = {
        "request_id": args.request_id,
        "session_id": args.session_id,
        "sender_device_id": args.sender_device_id,
        "message_id": args.message_id,
        "sequence": args.sequence,
        "nonce": args.nonce,
        "ciphertext": args.ciphertext,
    }
    try:
        status, response = _request_json(
            "POST", f"{args.base_url}/v1/messages/submit", body=payload)
    except ServerUnavailable:
        return _emit_server_error()
    return _emit_api_response(status, response)


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


def cmd_retry_message(args: argparse.Namespace) -> int:
    """Call POST /v1/messages/{sid}/retry/{mid}; 201 or 200 both succeed."""
    from urllib.parse import quote

    url = (f"{args.base_url}/v1/messages/"
           f"{quote(args.session_id, safe='')}/retry/"
           f"{quote(args.message_id, safe='')}")
    payload = {"device_id": args.device_id, "attempt_id": args.attempt_id}
    try:
        status, response = _request_json("POST", url, body=payload)
    except ServerUnavailable:
        return _emit_server_error()
    return _emit_api_response(status, response)


def cmd_ack_message(args: argparse.Namespace) -> int:
    """Call POST /v1/messages/{sid}/acks; 201 or 200 both succeed."""
    from urllib.parse import quote

    url = (f"{args.base_url}/v1/messages/"
           f"{quote(args.session_id, safe='')}/acks")
    payload = {"device_id": args.device_id,
               "message_id": args.message_id,
               "sequence": args.sequence}
    try:
        status, response = _request_json("POST", url, body=payload)
    except ServerUnavailable:
        return _emit_server_error()
    return _emit_api_response(status, response)


def cmd_message_status(args: argparse.Namespace) -> int:
    """Call GET /v1/messages/{sid}/status/{mid}?device_id=…."""
    from urllib.parse import quote, urlencode

    query = urlencode({"device_id": args.device_id})
    url = (f"{args.base_url}/v1/messages/"
           f"{quote(args.session_id, safe='')}/status/"
           f"{quote(args.message_id, safe='')}?{query}")
    try:
        status, response = _request_json("GET", url)
    except ServerUnavailable:
        return _emit_server_error()
    return _emit_api_response(status, response)


def cmd_integrity_history_page(args: argparse.Namespace) -> int:
    """Call GET /v1/persistence/integrity/history/page and print one JSON line.

    Success (200) prints the server's compact page on stdout and exits 0;
    any API failure (400/409/503/…) prints the server's single-line JSON on
    stderr and exits non-zero; a connection failure prints one stderr JSON
    line with field=server and exits non-zero.
    """
    from urllib.parse import urlencode

    query = urlencode({"after": args.after, "limit": args.limit})
    url = (f"{args.base_url}/v1/persistence/integrity/history/page?{query}")
    try:
        status, response = _request_json("GET", url)
    except ServerUnavailable:
        return _emit_server_error()
    return _emit_api_response(status, response)


def _emit_api_response(status: int, response: Any) -> int:
    """Print a 2xx response on stdout (exit 0), otherwise stderr (exit 1)."""
    stream = sys.stdout if 200 <= status < 300 else sys.stderr
    print(json.dumps(response, separators=(",", ":"), ensure_ascii=False),
          file=stream)
    return 0 if 200 <= status < 300 else 1


def _emit_crypto_error(error: CryptoError) -> int:
    """Print a local crypto failure as single-line JSON on stderr; exit 2."""
    print(json.dumps({"message": error.message, "field": error.field},
                     separators=(",", ":")),
          file=sys.stderr)
    return 2


def cmd_encrypt_message(args: argparse.Namespace) -> int:
    """Encrypt locally and print ``session_id``/``nonce``/``ciphertext``.

    The key and plaintext may be given directly or as ``@path`` files; a
    file that is missing, unreadable or not valid UTF-8 fails with
    ``field=key`` or ``field=plaintext`` respectively (the key is read
    first, so it is reported first when both files fail). Any failure
    prints a single-line JSON with only ``message`` and ``field`` on
    stderr, leaves stdout empty and exits 2.
    """
    try:
        key = _read_text_argument(args.key, "key")
        plaintext = _read_text_argument(args.plaintext, "plaintext")
        result = encrypt_message(args.session_id, key, plaintext,
                                 sender_device_id=args.sender_device_id,
                                 message_id=args.message_id,
                                 sequence=_parse_sequence_argument(
                                     args.sequence))
    except CryptoError as error:
        return _emit_crypto_error(error)
    print(json.dumps(result, separators=(",", ":"), ensure_ascii=False))
    return 0


def cmd_decrypt_message(args: argparse.Namespace) -> int:
    """Decrypt locally and print ``session_id``/``plaintext``.

    The key may be given directly or as a ``@path`` file; a file that is
    missing, unreadable or not valid UTF-8 fails with ``field=key``. Any
    failure prints a single-line JSON with only ``message`` and ``field``
    on stderr, leaves stdout empty and exits 2.
    """
    try:
        key = _read_text_argument(args.key, "key")
        result = decrypt_message(args.session_id, key,
                                 args.nonce, args.ciphertext,
                                 sender_device_id=args.sender_device_id,
                                 message_id=args.message_id,
                                 sequence=_parse_sequence_argument(
                                     args.sequence))
    except CryptoError as error:
        return _emit_crypto_error(error)
    print(json.dumps(result, separators=(",", ":"), ensure_ascii=False))
    return 0


def cmd_derive_session_key(args: argparse.Namespace) -> int:
    """Derive the session key locally and print ``session_id``/``key``."""
    try:
        result = derive_session_key(args.session_id, args.private_key,
                                    args.peer_public_key)
    except CryptoError as error:
        return _emit_crypto_error(error)
    print(json.dumps(result, separators=(",", ":"), ensure_ascii=False))
    return 0


def _read_object_argument(value: Optional[str], field: str) -> Any:
    """Parse a JSON object option: object text, or ``@path`` for a UTF-8 file.

    A missing/empty option, unreadable file, UTF-8 decode failure, invalid
    JSON or a JSON value that is not an object all raise
    :class:`CryptoError` named *field*; the file's UTF-8 text is parsed as-is
    (no stripping).
    """
    if not is_nonempty_string(value):
        raise CryptoError(
            f"field must be a non-empty string: {field}", field)
    text = value
    if value.startswith("@"):
        try:
            with open(value[1:], "r", encoding="utf-8") as handle:
                text = handle.read()
        except (OSError, UnicodeDecodeError) as error:
            raise CryptoError(f"cannot read {field} file: {error}",
                              field) from None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as error:
        raise CryptoError(f"{field} must be valid JSON: {error}",
                          field) from None
    if not isinstance(obj, dict):
        raise CryptoError(f"{field} must be a JSON object", field)
    return obj


def _read_proof_argument(value: Optional[str]) -> Any:
    """Parse the ``--proof`` option as a JSON object text or ``@path`` file."""
    return _read_object_argument(value, "proof")


def cmd_verify_prekey_proof(args: argparse.Namespace) -> int:
    """Verify a signed pre-key proof offline and print it with its fingerprint.

    Purely local: no server is contacted and no backend state is read or
    written. Success prints the proof's six original fields plus
    ``fingerprint`` as single-line JSON on stdout and exits 0; any failure —
    including a missing option — prints a single-line JSON with only
    ``message`` and ``field`` on stderr, leaves stdout empty and exits 2.
    """
    try:
        proof = _read_proof_argument(args.proof)
        result = verify_prekey_proof(proof, args.user_id, args.device_id,
                                     args.key_id, args.expected_fingerprint)
    except CryptoError as error:
        return _emit_crypto_error(error)
    print(json.dumps(result, separators=(",", ":"), ensure_ascii=False))
    return 0


def cmd_derive_verified_session_key(args: argparse.Namespace) -> int:
    """Derive a verified session key locally; print ``session_id``/``key``.

    Purely local: no server is contacted and no backend state is read or
    written. The snapshot and proof are JSON object text or ``@path`` UTF-8
    files; ``--role`` selects the initiator (default) or recipient private
    key. Success prints a single-line JSON with only ``session_id`` and
    ``key`` on stdout and exits 0; any failure — including a missing option —
    prints a single-line JSON with only ``message`` and ``field`` on stderr,
    leaves stdout empty and exits 2.
    """
    try:
        session = _read_object_argument(args.session, "session")
        proof = _read_object_argument(args.proof, "proof")
        # An omitted --role stays omitted (the initiator default); an
        # explicitly given value — including an empty string — is validated
        # by the crypto entry point and reported as field "role".
        if args.role is None:
            result = derive_verified_session_key(
                session, proof, args.private_key, args.user_id,
                args.expected_fingerprint)
        else:
            result = derive_verified_session_key(
                session, proof, args.private_key, args.user_id,
                args.expected_fingerprint, role=args.role)
    except CryptoError as error:
        return _emit_crypto_error(error)
    print(json.dumps(result, separators=(",", ":"), ensure_ascii=False))
    return 0


def _serve_data_file_path(data_file: Optional[str]) -> Optional[str]:
    """Resolve the state-file path for ``serve``: flag, env var, or ``None``."""
    return data_file or os.environ.get("E2EE_DATA_FILE")


def cmd_serve(args: argparse.Namespace) -> int:
    """Run the HTTP server until interrupted."""
    # A configured state file gets a process-exclusive lock in its own
    # directory before any recovery, load, or write touches it. Another live
    # serve process on the same file refuses to start (single stderr JSON
    # line, field=data_file, exit 1) without altering the formal file. The
    # in-memory mode (no flag, no env var) is unchanged. The kernel releases
    # the lock when this process exits, even after SIGKILL.
    path = _serve_data_file_path(args.data_file)
    state_lock = None
    if path is not None:
        try:
            state_lock = acquire_state_file_lock(path)
        except StateFileLocked as error:
            print(json.dumps({"message": str(error), "field": "data_file"},
                             separators=(",", ":")),
                  file=sys.stderr)
            return 1
        except OSError as error:
            print(json.dumps(
                {"message": f"cannot lock state file {path}: {error}",
                 "field": "data_file"},
                separators=(",", ":")),
                  file=sys.stderr)
            return 1
    try:
        try:
            service = _build_serve_service(args.data_file)
        except StateFileError as error:
            # Corrupt or wrong-version state file: refuse to start cleanly.
            print(json.dumps({"message": str(error), "field": "data_file"},
                             separators=(",", ":")),
                  file=sys.stderr)
            return 1
        server, _ = create_server(args.host, args.port, service)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
        return 0
    finally:
        if state_lock is not None:
            state_lock.release()


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)
