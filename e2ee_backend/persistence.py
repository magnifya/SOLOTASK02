"""Durable JSON-file persistence for the device store.

The whole store is serialized as one ``version: 1`` JSON document. Every
mutation is written by replacing the data file atomically (write a sibling
temporary file, then ``os.replace``), so a crash mid-write can never leave a
torn file behind. A missing file starts empty and is created on first save;
a corrupt file or an unsupported version refuses to load.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict

from .storage import DeviceStore

SNAPSHOT_VERSION = 1


class PersistenceError(Exception):
    """The data file exists but cannot be used (corrupt or wrong version)."""


class FilePersister:
    """Writes store snapshots to one JSON file, atomically."""

    def __init__(self, path: str) -> None:
        self.path = path

    def save(self, snapshot: Dict[str, Any]) -> None:
        """Serialize *snapshot* and atomically replace the data file."""
        data = json.dumps(snapshot, ensure_ascii=False,
                          separators=(",", ":")).encode("utf-8")
        directory = os.path.dirname(os.path.abspath(self.path))
        temp_path = os.path.join(
            directory, f".{os.path.basename(self.path)}.{os.getpid()}.tmp")
        try:
            with open(temp_path, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, self.path)
        except BaseException:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            raise


def load_store(path: str) -> DeviceStore:
    """Load the store from *path*; a missing file yields an empty store.

    Raises :class:`PersistenceError` when the file exists but is not valid
    JSON, is not a version-1 snapshot, or fails schema reconstruction.
    """
    if not os.path.exists(path):
        return DeviceStore()
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PersistenceError(f"data file is corrupt: {path}: {error}") from None
    if not isinstance(data, dict) or data.get("version") != SNAPSHOT_VERSION:
        raise PersistenceError(
            f"data file has unsupported version: {path} "
            f"(expected version={SNAPSHOT_VERSION})")
    try:
        return DeviceStore.from_snapshot(data)
    except (KeyError, TypeError, ValueError) as error:
        raise PersistenceError(
            f"data file failed schema validation: {path}: {error}") from None
