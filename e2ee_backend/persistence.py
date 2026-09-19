"""Durable JSON-file persistence for the device store.

The whole server state is kept in one versioned JSON document. A missing file
is created on first use; a corrupted file or a document whose version is not
understood makes the server refuse to start (state is never silently
discarded). Every change is written atomically: serialize to a temporary file
in the same directory, ``fsync`` it, then ``os.replace`` it over the target,
so a crash never leaves a half-written state file.
"""
from __future__ import annotations

import json
import os
import tempfile
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .service import DeviceService

#: Persistence format version understood by this build.
STATE_VERSION = 1


class StateFileError(Exception):
    """The state file is missing fields, corrupt, or has an unknown version."""


class JsonStateStore:
    """Versioned JSON document loaded from and atomically saved to one file."""

    def __init__(self, path: str) -> None:
        self.path = path

    def load(self) -> Optional[Dict[str, Any]]:
        """Load the document, or ``None`` when the file does not exist yet.

        Raises :class:`StateFileError` if the file cannot be decoded as JSON,
        is not a JSON object, or carries a missing/unknown ``version``.
        """
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                raw = handle.read()
        except FileNotFoundError:
            return None
        except OSError as error:
            raise StateFileError(f"cannot read state file {self.path}: {error}")

        try:
            document = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise StateFileError(
                f"state file is not valid JSON: {self.path} ({error})")

        if not isinstance(document, dict):
            raise StateFileError("state document must be a JSON object")
        version = document.get("version")
        if not isinstance(version, int) or isinstance(version, bool):
            raise StateFileError("state document is missing a numeric 'version'")
        if version != STATE_VERSION:
            raise StateFileError(
                f"unsupported state file version: {version} "
                f"(this server supports {STATE_VERSION})")
        return document

    def save(self, state: Dict[str, Any]) -> None:
        """Atomically replace the file with *state* (version stamped).

        Writes a sibling temporary file, fsyncs it, then ``os.replace`` — the
        target is either the previous full document or the new full one,
        never a truncated mix.
        """
        directory = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(directory, exist_ok=True)
        document = {"version": STATE_VERSION, **state}
        tmp_handle = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=directory, delete=False,
            prefix=".state-", suffix=".tmp")
        tmp_path = tmp_handle.name
        try:
            json.dump(document, tmp_handle, separators=(",", ":"),
                      ensure_ascii=False)
            tmp_handle.flush()
            os.fsync(tmp_handle.fileno())
            tmp_handle.close()
            os.replace(tmp_path, self.path)
        except BaseException:
            tmp_handle.close()
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise


def attach_persistence(service: "DeviceService", path: str) -> JsonStateStore:
    """Load *path* into *service* and persist every subsequent change.

    A missing file is created immediately (with an empty version-1 document).
    A corrupt or wrong-version file raises :class:`StateFileError` before the
    server starts; the in-memory store is never touched in that case. After
    attachment, each committed mutation rewrites the file atomically.
    """
    state_store = JsonStateStore(path)
    document = state_store.load()
    if document is None:
        state_store.save(service.store.snapshot_state())
    else:
        try:
            service.store.restore_state(
                {key: value for key, value in document.items()
                 if key != "version"})
        except (ValueError, TypeError) as error:
            raise StateFileError(
                f"state file has a malformed payload: {error}") from None

    def persist() -> None:
        state_store.save(service.store.snapshot_state())

    service.store.on_change = persist
    return state_store

