"""Deterministic feature-hashing embedder for offline tests and dry runs.

Not a semantic model: selected only by a profile with `kind: hashing`.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any

_TOKEN = re.compile(r"\w+", re.UNICODE)


class HashingEmbedder:
    def __init__(self, profile: dict[str, Any]) -> None:
        self.model_id = profile["model_id"]
        self.dim = profile["dim"]
        self.max_tokens = profile["max_tokens"]

    def _one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in _TOKEN.findall(text.casefold()):
            h = int.from_bytes(hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest(), "big")
            vec[h % self.dim] += 1.0 if (h >> 63) else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]
