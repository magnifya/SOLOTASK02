"""设备与预密钥的内存存储。

只保存公开密钥与标识，不保存明文消息或私钥。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class SignedPreKey:
    """一条已签名的预密钥（仅公钥）。"""

    key_id: str
    public_key: str
    revoked: bool = False


@dataclass
class Device:
    """一台已注册设备。"""

    user_id: str
    device_id: str
    identity_key: str
    registered_at: str
    signed_prekeys: List[SignedPreKey] = field(default_factory=list)


class DeviceStore:
    """线程安全的设备存储，按 (user_id, device_id) 索引。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._devices: Dict[Tuple[str, str], Device] = {}

    def add(self, device: Device) -> bool:
        """注册设备；同一 user_id 下 device_id 已存在时返回 False。"""
        key = (device.user_id, device.device_id)
        with self._lock:
            if key in self._devices:
                return False
            self._devices[key] = device
            return True

    def get(self, device_id: str) -> Optional[Device]:
        """按 device_id 查询设备，不存在返回 None。"""
        with self._lock:
            for (uid, did), device in self._devices.items():
                if did == device_id:
                    return device
            return None

    def active_prekey_ids(self, device_id: str) -> Optional[List[str]]:
        """返回未被撤销的预密钥 key_id 列表，顺序稳定；设备不存在返回 None。"""
        device = self.get(device_id)
        if device is None:
            return None
        return [pk.key_id for pk in device.signed_prekeys if not pk.revoked]
