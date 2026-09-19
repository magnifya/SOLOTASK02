"""公钥辅助工具，基于 cryptography 库。

服务端只处理公开密钥：这里提供公钥指纹计算，用于日志与调试，
不涉及任何私钥或明文消息。
"""

from __future__ import annotations

import hashlib
from typing import Optional

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PublicKey


def load_public_key(pem_or_b64: str) -> Optional[object]:
    """尝试把字符串解析为公钥（PEM 或 base64 原始字节），失败返回 None。"""
    text = pem_or_b64.strip()
    if not text:
        return None
    if text.startswith("-----BEGIN"):
        try:
            return serialization.load_pem_public_key(text.encode("utf-8"))
        except Exception:
            return None
    import base64

    try:
        raw = base64.b64decode(text, validate=True)
    except Exception:
        return None
    for cls in (Ed25519PublicKey, X25519PublicKey):
        try:
            return cls.from_public_bytes(raw)
        except Exception:
            continue
    return None


def key_fingerprint(public_key: str) -> str:
    """计算公钥的稳定指纹（SHA-256 十六进制），无法解析时按原字符串哈希。"""
    key = load_public_key(public_key)
    if key is not None:
        raw = key.public_bytes(
            encoding=serialization.Encoding.Raw
            if isinstance(key, (Ed25519PublicKey, X25519PublicKey))
            else serialization.Encoding.DER,
            format=serialization.PublicFormat.Raw
            if isinstance(key, (Ed25519PublicKey, X25519PublicKey))
            else serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    else:
        raw = public_key.encode("utf-8")
    return hashlib.sha256(raw).hexdigest()
