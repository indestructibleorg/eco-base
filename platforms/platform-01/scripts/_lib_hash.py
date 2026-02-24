from __future__ import annotations

import hashlib
from typing import Iterable


def sha3_512_bytes(data: bytes) -> str:
    h = hashlib.sha3_512()
    h.update(data)
    return h.hexdigest()


def sha3_512_file(path: str) -> str:
    h = hashlib.sha3_512()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha3_512_join_hex(hex_digests: Iterable[str]) -> str:
    """
    Deterministic aggregation:
    - input: iterable of hex digests (strings)
    - output: sha3-512 of the concatenated bytes of all digests, in order
    """
    h = hashlib.sha3_512()
    for d in hex_digests:
        h.update(bytes.fromhex(d))
    return h.hexdigest()
