"""JSON Schemas for Registry artifacts. Structural only (allowed by P1): these
describe the *shape* of jev-graph-builder's own records, never their content."""

from __future__ import annotations

_STR = {"type": "string"}
_STR_LIST = {"type": "array", "items": _STR}

CORPUS = {
    "type": "object",
    "required": ["name", "scope", "doc_sources", "uses"],
    "properties": {
        "name": _STR,
        "scope": _STR,
        "languages": _STR_LIST,
        "doc_sources": _STR_LIST,
        "uses": _STR_LIST,
        "training_templates": _STR_LIST,
        "split_policy": _STR,
    },
}

ONTOLOGY_ITEM = {
    "type": "object",
    "required": ["name", "definition"],
    "properties": {
        "name": _STR,
        "definition": _STR,
        "parent": {"type": ["string", "null"]},
        "direction": {"enum": ["directed", "symmetric"]},
        "direction_semantics": _STR,
        "allowed_src": _STR_LIST,
        "allowed_dst": _STR_LIST,
        "examples": {"type": "array"},
        "evidence": {"type": "array"},
    },
}

ONTOLOGY_FILE = {"type": "object", "required": ["items"], "properties": {"items": {"type": "array", "items": ONTOLOGY_ITEM}}}

STRUCTURAL_LABELS = {"type": "object", "additionalProperties": _STR}

_THRESHOLD = {"type": "number", "minimum": 0, "exclusiveMaximum": 1}

_CONDITION: dict = {
    "type": "object",
    "properties": {
        "all": {"type": "array", "items": {"$ref": "#/$defs/cond"}},
        "any": {"type": "array", "items": {"$ref": "#/$defs/cond"}},
        "not": {"$ref": "#/$defs/cond"},
        "q": _STR,
        "is": {"enum": ["yes", "no"]},
        "eq": _STR,
        "in": _STR_LIST,
        "not_in": _STR_LIST,
        "level_in": {"type": "array", "items": {"type": "integer", "minimum": 0}},
        "if_asked": {"type": "boolean"},
        "threshold": _THRESHOLD,  # a Noul leaf's own threshold
    },
    "additionalProperties": False,
}

QUESTION = {
    "type": "object",
    "required": ["type", "instructions"],
    "properties": {
        "type": {"enum": ["noul", "choice", "score"]},
        "instructions": {},
        "criteria": {},
        "fanout": {"type": "boolean"},
    },
    "additionalProperties": False,
}

STATE_FIELD = {
    "type": "object",
    "properties": {
        "source": _STR,
        "fields": _STR_LIST,
        "max_tokens_ref": _STR,
        "truncate": {"enum": ["head", "tail", "head_tail"]},
        "untrusted": {"type": "boolean"},
        "optional": {"type": "boolean"},
    },
    "additionalProperties": False,
}

QUESTION_SET = {
    "type": "object",
    "required": ["name", "version", "jev_model", "state_template", "questions"],
    "$defs": {"cond": _CONDITION},
    "properties": {
        "name": _STR,
        "version": {"type": "integer", "minimum": 1},
        "jev_model": _STR,
        "purpose": _STR,
        "state_template": {"type": "object", "additionalProperties": STATE_FIELD},
        "questions": {"type": "object", "minProperties": 1, "additionalProperties": QUESTION},
        "fanout": {
            "type": "object",
            "required": ["over", "max_ref"],
            "properties": {"over": _STR, "max_ref": _STR, "item_fields": _STR_LIST, "item_max_tokens_ref": _STR, "untrusted": {"type": "boolean"}},
            "additionalProperties": False,
        },
        "gating": {
            "type": "object",
            "required": ["accept_when"],
            # which of Jev's answers mean accept; `threshold` overrides policies.gating.threshold for this set
            "properties": {"accept_when": {"$ref": "#/$defs/cond"}, "threshold": _THRESHOLD},
            "additionalProperties": False,
        },
    },
    "additionalProperties": False,
}

PROMPT_FRONTMATTER = {
    "type": "object",
    "required": ["name", "version", "purpose", "inputs", "output_schema"],
    "properties": {
        "name": _STR,
        "version": {"type": "integer"},
        "purpose": _STR,
        "inputs": {"type": "array"},
        "output_schema": {"type": ["string", "null"]},
        "output_mode": {"enum": ["structured", "file"]},
        "record_schema": {"type": ["string", "null"]},
        "harness_notes": {"type": ["object", "string", "null"]},
        "verified_by": {"type": ["string", "array", "null"]},
        "enters": {"enum": ["graph", "training", "registry", "none"]},
        "plugins": {"type": "array", "items": _STR},
    },
}

TRAINING_TEMPLATE = {
    "type": "object",
    "required": ["name", "kind", "question_sets"],
    "additionalProperties": False,
    "properties": {
        "name": _STR,
        "description": _STR,
        "kind": {"enum": ["qa_single", "instruction", "qa_multihop", "graph_reasoning", "triples",
                          "retrieval_pairs", "rerank_soft_labels"]},
        "prompt": {"type": ["string", "null"]},
        "fields": {"type": "object", "additionalProperties": _STR},
        "question_sets": {"type": "object", "additionalProperties": _STR},
    },
}

PROFILES = {
    "type": "object",
    "required": ["jev", "harness", "embedding"],
    "properties": {
        "jev": {"type": "object", "additionalProperties": {"type": "object", "required": ["kind"]}},
        "harness": {"type": "object", "additionalProperties": {"type": "object", "required": ["harness", "autonomy"]}},
        "embedding": {"type": "object", "additionalProperties": {"type": "object", "required": ["kind", "model_id", "dim", "max_tokens"]}},
        "postgres": {"type": "object", "additionalProperties": {"type": "object",
                                                                  "required": ["image", "container", "volume", "data_dir", "env", "user", "database"]}},
    },
}

POLICIES = {"type": "object", "required": ["question_sets", "jev"]}
