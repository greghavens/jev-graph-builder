"""Drift and consistency statistics (§8.8, §13.3).

Pure functions over pairs of stored and re-asked raw answers (the wire shape:
`{type, noul}` / `{type, choice, probabilities}` / `{type, score, probabilities}`).
Every threshold comes from `policies.audit.drift`.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any


def _unit(answer: dict[str, Any]) -> float | None:
    """Noul p, or a Score rescaled onto [0, 1] by its level range; Choice has no scalar."""
    if answer.get("type") == "noul":
        return float(answer["noul"])
    if answer.get("type") == "score":
        levels = sorted(int(k) for k in answer.get("probabilities") or {})
        if len(levels) < 2:
            return None
        return (float(answer["score"]) - levels[0]) / (levels[-1] - levels[0])
    return None


def _side(x: float, boundary: float) -> bool:
    return x >= boundary


@dataclass
class DriftStats:
    compared: int = 0
    flips: int = 0
    deltas: list[float] = field(default_factory=list)
    model_mismatch: list[str] = field(default_factory=list)

    @property
    def flip_rate(self) -> float:
        return self.flips / self.compared if self.compared else 0.0

    @property
    def std(self) -> float:
        return statistics.pstdev(self.deltas) if len(self.deltas) > 1 else 0.0

    def add(self, old: dict[str, Any], new: dict[str, Any], boundary: float) -> None:
        for key, a in old.items():
            b = new.get(key)
            if not isinstance(a, dict) or not isinstance(b, dict):
                continue
            self.compared += 1
            if a.get("type") == "choice":
                self.flips += int(a.get("choice") != b.get("choice"))
                continue
            x, y = _unit(a), _unit(b)
            if x is None or y is None:
                continue
            self.deltas.append(y - x)
            self.flips += int(_side(x, boundary) != _side(y, boundary))

    def breaches(self, policy: dict[str, Any]) -> list[str]:
        out = []
        if self.compared >= policy["min_compared"]:
            if self.std > policy["max_std"]:
                out.append(f"std {self.std:.4f} > {policy['max_std']}")
            if self.flip_rate > policy["max_flip_rate"]:
                out.append(f"flip rate {self.flip_rate:.4f} > {policy['max_flip_rate']}")
        if self.model_mismatch:
            out.append(f"model changed: {sorted(set(self.model_mismatch))}")
        return out

    def as_dict(self) -> dict[str, Any]:
        return {"compared": self.compared, "flips": self.flips, "flip_rate": self.flip_rate, "std": self.std,
                "model_mismatch": sorted(set(self.model_mismatch))}
