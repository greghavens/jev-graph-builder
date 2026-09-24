"""State builder (§11.4): minimal, named, JSON-structured state per question set.

Only the fields a template names are included. Strings are truncated by the
template's declared rule. Untrusted document text is wrapped in a field whose
name (Registry data) marks it as data. Numbers are refused: comparisons are
done in code and passed as booleans or categories.
"""

from __future__ import annotations

from typing import Any

from jev_graph_builder.jev.tokens import TokenEstimator
from jev_graph_builder.registry.loader import QuestionSet, Registry


class StateError(Exception):
    pass


_MISSING = object()


def _resolve(inputs: dict[str, Any], dotted: str) -> Any:
    node: Any = inputs
    for part in dotted.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return _MISSING
    return node


def _pick(value: Any, fields: list[str] | None) -> Any:
    if not fields:
        return value
    if isinstance(value, list):
        return [_pick(v, fields) for v in value]
    if isinstance(value, dict):
        return {f: value[f] for f in fields if f in value and value[f] not in (None, "", [])}
    raise StateError(f"cannot pick fields {fields} from {type(value).__name__}")


def _check_no_numbers(value: Any, path: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        raise StateError(f"state field `{path}` is numeric; compare in code and pass a boolean or category (§11.4)")
    if isinstance(value, dict):
        for k, v in value.items():
            _check_no_numbers(v, f"{path}.{k}")
    elif isinstance(value, list):
        for i, v in enumerate(value):
            _check_no_numbers(v, f"{path}[{i}]")
    else:
        raise StateError(f"state field `{path}` has unsupported type {type(value).__name__}")


class StateBuilder:
    def __init__(self, registry: Registry, estimator: TokenEstimator) -> None:
        self.reg = registry
        self.est = estimator
        self.data_field = registry.policy("jev.untrusted_field")
        self.default_rule = registry.policy("jev.default_truncation")

    def _truncate(self, value: Any, max_tokens: int | None, rule: str) -> Any:
        if max_tokens is None:
            return value
        if isinstance(value, str):
            return self.est.truncate(value, max_tokens, rule)
        if isinstance(value, dict):
            return {k: self._truncate(v, max_tokens, rule) for k, v in value.items()}
        if isinstance(value, list):
            return [self._truncate(v, max_tokens, rule) for v in value]
        return value

    def build(self, qs: QuestionSet, inputs: dict[str, Any], fanout_items: dict[str, Any] | None = None) -> dict[str, Any]:
        state: dict[str, Any] = {}
        for name, spec in qs.state_template.items():
            value = _resolve(inputs, spec.get("source", name))
            if value is _MISSING or value is None:
                if spec.get("optional"):
                    continue
                raise StateError(f"{qs.ref}: state input `{spec.get('source', name)}` missing")
            value = _pick(value, spec.get("fields"))
            max_tokens = self.reg.policy(spec["max_tokens_ref"]) if spec.get("max_tokens_ref") else None
            value = self._truncate(value, max_tokens, spec.get("truncate", self.default_rule))
            _check_no_numbers(value, name)
            state[name] = {self.data_field: value} if spec.get("untrusted") else value
        if qs.fanout:
            if fanout_items is None:
                raise StateError(f"{qs.ref}: fan-out question set needs items")
            fields = qs.fanout.get("item_fields")
            max_tokens = self.reg.policy(qs.fanout["item_max_tokens_ref"]) if qs.fanout.get("item_max_tokens_ref") else None
            rendered = {}
            for item_id, item in fanout_items.items():
                v = self._truncate(_pick(item, fields), max_tokens, self.default_rule)
                _check_no_numbers(v, f"{qs.fanout['over']}.{item_id}")
                rendered[item_id] = {self.data_field: v} if qs.fanout.get("untrusted") else v
            state[qs.fanout["over"]] = rendered
        return state
