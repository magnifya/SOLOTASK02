"""Thread-safe in-memory device storage.

Devices live for the lifetime of the server process. Only public keys and
identifiers are retained. Reads return pre-key ids in the exact insertion
order, so repeated requests list them identically.
"""
from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional, Tuple

from .models import Device


class DeviceStore:
    """In-memory store keyed by ``(user_id, device_id)``.

    Device ids are additionally indexed globally, because the public GET route
    addresses a device by ``device_id`` alone.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._devices: Dict[Tuple[str, str], Device] = {}
        self._device_index: Dict[str, Tuple[str, str]] = {}

    def add_device(self, device: Device) -> bool:
        """Insert a device.

        Return ``False`` (and store nothing) when the same ``device_id`` is
        already registered, whether under the same user or another one.
        """
        key = (device.user_id, device.device_id)
        with self._lock:
            if key in self._devices or device.device_id in self._device_index:
                return False
            self._devices[key] = device
            self._device_index[device.device_id] = key
            return True

    def get_device(self, user_id: str, device_id: str) -> Optional[Device]:
        """Return the device for the user, or ``None``."""
        with self._lock:
            return self._devices.get((user_id, device_id))

    def find_by_device_id(self, device_id: str) -> Optional[Device]:
        """Return the (globally unique) device with this id, or ``None``."""
        with self._lock:
            key = self._device_index.get(device_id)
            return self._devices.get(key) if key is not None else None

    def active_prekey_ids(self, device: Device) -> List[str]:
        """Return key ids of non-revoked pre-keys, in stable insertion order."""
        with self._lock:
            return [pk.key_id for pk in device.prekeys if not pk.revoked]

    def revoke_prekey(self, device: Device, key_id: str) -> bool:
        """Mark one of the device's pre-keys revoked. Return ``False`` if absent."""
        with self._lock:
            for prekey in device.prekeys:
                if prekey.key_id == key_id:
                    prekey.revoked = True
                    return True
            return False

    def revoke_device(self, device_id: str) -> Optional[Device]:
        """Mark the device (and its pre-keys) revoked.

        Idempotent: a previously revoked device stays revoked. Returns the
        device or ``None`` when the id is unknown.
        """
        with self._lock:
            key = self._device_index.get(device_id)
            device = self._devices.get(key) if key is not None else None
            if device is None:
                return None
            device.revoked = True
            for prekey in device.prekeys:
                prekey.revoked = True
            return device

    def revoke_prekey_by_id(self, device_id: str,
                            key_id: str) -> Tuple[Optional[Device], bool]:
        """Revoke one pre-key of a globally-addressed device.

        Returns ``(device, key_found)``: ``device`` is ``None`` for an unknown
        device; otherwise ``key_found`` says whether *key_id* existed. Revoking
        an already-revoked key is idempotent and reports it as found.
        """
        with self._lock:
            key = self._device_index.get(device_id)
            device = self._devices.get(key) if key is not None else None
            if device is None:
                return None, False
            for prekey in device.prekeys:
                if prekey.key_id == key_id:
                    prekey.revoked = True
                    return device, True
            return device, False

    def public_view(self, device_id: str) -> Optional[Dict[str, Any]]:
        """Atomically build the public snapshot of a device.

        The active pre-key id list is copied under the lock together with the
        rest of the record, so a concurrent revocation can never be observed
        half-applied.
        """
        with self._lock:
            key = self._device_index.get(device_id)
            device = self._devices.get(key) if key is not None else None
            if device is None:
                return None
            return {
                "identity_key": device.identity_key,
                "prekey_ids": [pk.key_id for pk in device.prekeys
                               if not pk.revoked],
                "registered_at": device.registered_at,
            }
