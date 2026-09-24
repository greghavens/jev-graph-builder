"""Deterministic content hashing (P5). Every ID in the system is derived here."""

from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Iterable, Iterator
from typing import Any, TypeVar


T = TypeVar("T")


def canonical_json(value: Any) -> str:
    """Stable JSON encoding: sorted keys, no whitespace, UTF-8 preserved."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def sha256_hex(*parts: Any) -> str:
    """Hash an ordered sequence of parts. Non-string parts are canonical-JSON encoded."""
    h = hashlib.sha256()
    for part in parts:
        data = part if isinstance(part, str) else canonical_json(part)
        h.update(data.encode("utf-8"))
        h.update(b"\x1f")  # unit separator so ("ab","c") != ("a","bc")
    return h.hexdigest()


def short_key(*parts: Any, length: int = 12) -> str:
    """Opaque short identifier (question keys, fan-out candidate IDs). Never carries meaning."""
    return "k" + sha256_hex(*parts)[:length]


_HEX_SPACE = 16 ** 64


def unit_fraction(*parts: Any) -> float:
    """Deterministic value in [0, 1) from a hash: sampling, split assignment (P5)."""
    return int(sha256_hex(*parts), 16) / _HEX_SPACE


def pairs(items: Iterable[T]) -> Iterator[tuple[T, T]]:
    """Unordered pairs in input order (entity alignment, ontology overlap checks)."""
    return itertools.combinations(items, 2)
