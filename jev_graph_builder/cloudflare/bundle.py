"""What the Cloudflare Worker needs from the Registry, rendered by this package (`bundle.json`).

The Worker never renders questions itself: the question sets the query path asks are rendered here with
the same `QuestionBuilder` the build uses, with their keys, state templates (token limits resolved) and
gating. The Worker builds the state, calls System One, gates the answers and runs the query flow; the
policies and relation definitions it needs come with them.
"""

from __future__ import annotations

from typing import Any

from jev_graph_builder.jev.gating import Bar
from jev_graph_builder.jev.questions import QuestionBuilder
from jev_graph_builder.pipeline.common import Context
from jev_graph_builder.pipeline.s6_links import relation_definitions
from jev_graph_builder.query.graphrag import QS_EXPAND, QS_ROUTE
from jev_graph_builder.query.mcp_server import instructions_for
from jev_graph_builder.query.search import QS_PASSAGE

# The rag_passage question asked when a candidate has no peers to contradict.
NO_PEERS = "no_peers"
SYSTEM_ONE_PATH = "/v1/systemone"  # TypeSafe's System One endpoint, under the profile's base URL


def _question_set(ctx: Context, builder: QuestionBuilder, bar: Bar, name: str, only: set[str] | None = None) -> dict[str, Any]:
    qs = ctx.reg.question_set(name)
    rendered = builder.render(qs, only=only)
    template = {}
    for field, spec in qs.state_template.items():
        template[field] = {
            "source": spec.get("source", field), "fields": spec.get("fields"), "optional": bool(spec.get("optional")),
            "untrusted": bool(spec.get("untrusted")), "truncate": spec.get("truncate", ctx.reg.policy("jev.default_truncation")),
            "max_tokens": ctx.reg.policy(spec["max_tokens_ref"]) if spec.get("max_tokens_ref") else None,
        }
    return {
        "ref": qs.ref, "questions_hash": rendered.questions_hash, "wire": rendered.wire,
        "keys": {k: {"logical": i.logical, "type": i.qtype, "levels": i.levels} for k, i in rendered.keys.items()},
        "state_template": template, "accept_when": (qs.gating or {}).get("accept_when"),
        "threshold": bar.for_qs(qs), "leaf_thresholds": bar.for_leaves(qs),
    }


def system_one(profile: dict[str, Any]) -> dict[str, Any]:
    """How the Worker calls System One: the build's own provider profile (any endpoint and key)."""
    if profile["kind"] == "typesafe":
        out = {"url": profile["endpoint"].rstrip("/") + SYSTEM_ONE_PATH, "model": profile["model"],
               "field_map": {"model": "model", "state": "state", "questions": "questions", "answers": "answers"},
               "response_root": None, "auth_header": "Authorization", "auth_scheme": "Bearer"}
    else:
        out = {"url": profile["endpoint"], "model": profile["model"], "field_map": profile["field_map"],
               "response_root": profile.get("response_root"), "auth_header": profile.get("auth_header", "Authorization"),
               "auth_scheme": profile.get("auth_scheme", "Bearer")}
    return {**out, "max_questions": profile.get("max_questions"), "context_tokens": profile["context_tokens"],
            "key_env": profile["key_env"]}


def build(ctx: Context, serving_embedding: dict[str, Any]) -> dict[str, Any]:
    reg = ctx.reg
    profile = ctx.jev.provider.profile
    builder = QuestionBuilder(reg, profile["limits"])
    bar = ctx.jev.bar(0)
    passage = reg.question_set(QS_PASSAGE)
    search = reg.policy("search")
    return {
        "registry_version": ctx.registry_version,
        "corpus_id": ctx.corpus_id,
        "server": {"name": ctx.settings.mcp_name, "instructions": instructions_for(reg)},
        "chars_per_token": profile["chars_per_token"],
        "untrusted_field": reg.policy("jev.untrusted_field"),
        "system_one": system_one(profile),
        "retry": reg.policy("jev.retry"),
        "question_sets": {
            QS_ROUTE: _question_set(ctx, builder, bar, QS_ROUTE),
            QS_PASSAGE: _question_set(ctx, builder, bar, QS_PASSAGE),
            f"{QS_PASSAGE}.{NO_PEERS}": _question_set(ctx, builder, bar, QS_PASSAGE,
                                                      set(passage.questions) - {search["contradiction_question"]}),
            QS_EXPAND: _question_set(ctx, builder, bar, QS_EXPAND),
        },
        "policies": {"query": reg.policy("query"), "retrieval_plans": reg.policy("retrieval_plans"), "search": search,
                     "concurrency": reg.policy("run.concurrency.search")},
        "relations": {name: {"name": r["name"], "definition": r["definition"]} for name, r in relation_definitions(ctx).items()},
        "corpus_scope": reg.corpus["scope"],
        "embedding": {"workers_ai_model": serving_embedding["workers_ai_model"],
                      "pooling": serving_embedding.get("workers_ai_pooling"), "dim": serving_embedding["dim"]},
    }
