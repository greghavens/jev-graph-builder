"""Token estimation before sending (§11.1). The approximation ratio is a
profile value, so a real tokenizer can replace it without code changes."""

from __future__ import annotations

import math
from typing import Any

from jev_graph_builder.ids import canonical_json


class TokenEstimator:
    def __init__(self, chars_per_token: float) -> None:
        self.chars_per_token = chars_per_token

    def text(self, s: str) -> int:
        return math.ceil(len(s) / self.chars_per_token)

    def value(self, v: Any) -> int:
        return self.text(v if isinstance(v, str) else canonical_json(v))

    def chars(self, tokens: int) -> int:
        """The most characters that fit in `tokens`."""
        return int(tokens * self.chars_per_token)

    def truncate(self, s: str, max_tokens: int, rule: str) -> str:
        """Deterministic truncation by the state template's declared rule (never silent: the rule is data)."""
        max_chars = self.chars(max_tokens)
        if len(s) <= max_chars:
            return s
        if rule == "head":
            return s[:max_chars]
        if rule == "tail":
            return s[-max_chars:]
        half = max_chars // 2
        return s[:half] + "\n…\n" + s[-half:]
