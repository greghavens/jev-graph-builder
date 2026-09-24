"""Question builder (§11.2, §11.3).

Renders a Registry question set into wire questions:
  * resolves `$from_ontology`, `$from_profiles`, `$from_input` criteria and `add` extras;
  * assigns opaque keys (hashes) — keys are never sent with meaning;
  * expands fan-out questions to one question per candidate;
  * splits over-large Choices hierarchically (parent, then child);
  * maps answers back to logical names and validates them with a
    dynamically built Pydantic model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

from jev_graph_builder.ids import sha256_hex, short_key
from jev_graph_builder.registry.loader import QuestionSet, Registry, RegistryError

_ITEM_PLACEHOLDER = re.compile(r"\{\{\s*item\s*\}\}")


class _Noul(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type: Literal["noul"]
    noul: float = Field(ge=0, le=1)


class _Choice(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type: Literal["choice"]
    choice: str
    probabilities: dict[str, float]
    confidence: float | None = None


class _Score(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type: Literal["score"]
    score: float
    probabilities: dict[int, float]
    confidence: float | None = None
    legend: dict[int, Any] | None = None


_ANSWER_MODELS = {"noul": _Noul, "choice": _Choice, "score": _Score}


@dataclass
class KeyInfo:
    logical: str
    item: str | None
    qtype: str
    levels: int | None = None


@dataclass
class Rendered:
    wire: dict[str, dict[str, Any]]
    keys: dict[str, KeyInfo]
    questions_hash: str
    hierarchical: dict[str, dict[str, list[str]]] = field(default_factory=dict)  # logical -> parent -> child options
    child_options: dict[str, dict[str, Any]] = field(default_factory=dict)       # logical -> full option map

    def response_model(self) -> type[BaseModel]:
        """Typed parsing of the `answers` object for exactly these keys."""
        fields = {k: (_ANSWER_MODELS[info.qtype], ...) for k, info in self.keys.items()}
        answers = create_model("Answers_" + self.questions_hash[:8], __config__=ConfigDict(extra="ignore"), **fields)  # type: ignore[call-overload]
        return create_model("Response_" + self.questions_hash[:8], model=(str, ...), answers=(answers, ...), usage=(dict, {}))


def normalize_answer(raw: BaseModel, info: KeyInfo) -> dict[str, Any]:
    """One stable shape per type for gating and audit."""
    if info.qtype == "noul":
        return {"type": "noul", "p": float(raw.noul)}  # type: ignore[attr-defined]
    if info.qtype == "choice":
        return {"type": "choice", "choice": raw.choice, "probabilities": dict(raw.probabilities), "confidence": raw.confidence}  # type: ignore[attr-defined]
    return {
        "type": "score",
        "score": float(raw.score),  # type: ignore[attr-defined]
        "probabilities": {int(k): v for k, v in raw.probabilities.items()},  # type: ignore[attr-defined]
        "confidence": raw.confidence,  # type: ignore[attr-defined]
        "levels": info.levels,
    }


class QuestionBuilder:
    def __init__(self, registry: Registry, limits: dict[str, Any]) -> None:
        self.reg = registry
        self.limits = limits

    def options(self, criteria: Any, dynamic: dict[str, dict[str, Any]] | None) -> dict[str, Any]:
        if isinstance(criteria, dict) and "$from_ontology" in criteria:
            opts: dict[str, Any] = {i["name"]: i["definition"] for i in self.reg.ontology(criteria["$from_ontology"])}
        elif isinstance(criteria, dict) and "$from_profiles" in criteria:
            opts = {k: v.get("description") for k, v in self.reg.profiles[criteria["$from_profiles"]].items() if v.get("routable", True)}
        elif isinstance(criteria, dict) and "$from_input" in criteria:
            key = criteria["$from_input"]
            if not dynamic or key not in dynamic:
                raise RegistryError(f"choice criteria needs dynamic options `{key}`")
            opts = dict(dynamic[key])
        else:
            return dict(criteria or {})
        opts.update(criteria.get("add") or {})
        return opts

    def _parents(self, criteria: Any, opts: dict[str, Any]) -> dict[str, list[str]] | None:
        if not (isinstance(criteria, dict) and "$from_ontology" in criteria):
            return None
        groups: dict[str, list[str]] = {}
        for item in self.reg.ontology(criteria["$from_ontology"]):
            if not item.get("parent"):
                return None
            groups.setdefault(item["parent"], []).append(item["name"])
        for extra in criteria.get("add") or {}:
            groups.setdefault(extra, []).append(extra)
        return groups

    def render(
        self,
        qs: QuestionSet,
        items: list[str] | None = None,
        dynamic: dict[str, dict[str, Any]] | None = None,
        only: set[str] | None = None,
        item_only: dict[str, set[str]] | None = None,
    ) -> Rendered:
        """`only` limits the logical questions for the whole call; `item_only` limits a fan-out
        question per item (items of different kinds are asked different questions in one call)."""
        wire: dict[str, dict[str, Any]] = {}
        keys: dict[str, KeyInfo] = {}
        hier: dict[str, dict[str, list[str]]] = {}
        child_opts: dict[str, dict[str, Any]] = {}
        over = qs.fanout["over"] if qs.fanout else None
        for logical, q in qs.questions.items():
            if only is not None and logical not in only:
                continue
            qtype = q["type"]
            body: dict[str, Any] = {"type": qtype, "instructions": q["instructions"]}
            levels = None
            if qtype == "choice":
                opts = self.options(q.get("criteria"), dynamic)
                if len(opts) > self.limits["choice_max_options"]:
                    groups = self._parents(q.get("criteria"), opts)
                    if groups is None:
                        raise RegistryError(f"{qs.ref}.{logical}: too many options and no hierarchy")
                    hier[logical] = groups
                    child_opts[logical] = opts
                    opts = {p: {"includes": sorted(children)} for p, children in groups.items()}
                body["criteria"] = opts
            elif qtype == "score":
                body["criteria"] = list(q["criteria"])
                levels = len(body["criteria"])
            elif q.get("criteria") is not None:
                body["criteria"] = q["criteria"]
            targets = items if (q.get("fanout") and qs.fanout) else [None]
            for item in targets or []:
                if item is not None and item_only is not None and logical not in item_only.get(item, ()):
                    continue
                b = dict(body)
                if item is not None:
                    b["instructions"] = _subst(b["instructions"], f"`{over}.{item}`")
                key = short_key(qs.ref, logical, item or "")
                wire[key] = b
                keys[key] = KeyInfo(logical=logical, item=item, qtype=qtype, levels=levels)
        return Rendered(wire=wire, keys=keys, questions_hash=sha256_hex(wire), hierarchical=hier, child_options=child_opts)

    def child_question(self, qs: QuestionSet, rendered: Rendered, key: str, parent: str) -> dict[str, Any]:
        """Second level of a hierarchical choice: the children of the chosen parent."""
        info = rendered.keys[key]
        children = rendered.hierarchical[info.logical][parent]
        opts = rendered.child_options[info.logical]
        body = dict(rendered.wire[key])
        body["criteria"] = {c: opts[c] for c in children}
        return body


def _subst(value: Any, ref: str) -> Any:
    if isinstance(value, str):
        return _ITEM_PLACEHOLDER.sub(ref, value)
    if isinstance(value, dict):
        return {k: _subst(v, ref) for k, v in value.items()}
    if isinstance(value, list):
        return [_subst(v, ref) for v in value]
    return value
