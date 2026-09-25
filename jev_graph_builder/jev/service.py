"""JevService: one entry point for asking a Registry question set (§11).

`ask()` builds the state (§11.4), renders the questions (§11.2), packs them into
as few calls as the context window allows (§11.3), serves repeated calls from
the cache (§11.7), resolves hierarchical choices with a second call, and gates
the answers per subject / per fan-out item (§11.5): if Jev says yes it is yes, if
Jev says no it is no.

`jev_calls` rows are written as soon as a call returns, in their own
transaction, so that a crash never repeats a paid call (the cache is the
resume mechanism for Jev). `decisions` rows are returned to the caller and
written with `write_decision()` inside the item's transaction (§14.2).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from jev_graph_builder import log
from jev_graph_builder.ids import sha256_hex, short_key
from jev_graph_builder.jev.client import JevCreditsExhausted, JevProvider, JevRequestError, RawResult
from jev_graph_builder.jev.gating import Bar, evaluate
from jev_graph_builder.jev.questions import QuestionBuilder, Rendered, normalize_answer
from jev_graph_builder.jev.state import StateBuilder
from jev_graph_builder.registry.loader import QuestionSet, Registry
from jev_graph_builder.store import repo
from jev_graph_builder.store.db import Database

_CHILD_SUFFIX = "child"


class JevAnswerError(Exception):
    """Jev returned answers that do not match the requested keys or types."""


@dataclass
class Decision:
    decision_id: str
    subject_kind: str
    subject_id: str
    question_set: str
    item: str | None
    outcome: str
    answers: dict[str, dict[str, Any]]
    call_ids: list[str]
    provider: str
    run_id: str | None
    threshold: float  # the Noul threshold in force for this decision's question set (after back-off)
    repeat: int       # times this question already came back to Jev for the same subject
    leaf_thresholds: dict[str, float] = field(default_factory=dict)  # Noul leaves with their own threshold

    def threshold_for(self, question: str) -> float:
        """The threshold Jev's answer to `question` was gated at."""
        return self.leaf_thresholds.get(question, self.threshold)

    def row(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "subject_kind": self.subject_kind,
            "subject_id": self.subject_id,
            "question_set": self.question_set,
            "call_ids": self.call_ids,
            "outcome": self.outcome,
            "run_id": self.run_id,
            "answers": {"provider": self.provider, "item": self.item, "answers": self.answers,
                        "threshold": self.threshold, "repeat": self.repeat,
                        **({"leaf_thresholds": self.leaf_thresholds} if self.leaf_thresholds else {})},
        }


@dataclass
class AskResult:
    qs: QuestionSet
    decisions: list[Decision]
    call_ids: list[str] = field(default_factory=list)
    cache_hits: int = 0
    dynamic: dict[str, dict[str, Any]] | None = None

    @property
    def single(self) -> Decision:
        if len(self.decisions) != 1:
            raise ValueError(f"{self.qs.ref} produced {len(self.decisions)} decisions, expected one")
        return self.decisions[0]

    def by_item(self) -> dict[str | None, Decision]:
        return {d.item: d for d in self.decisions}


async def write_decision(conn: Any, decision: Decision) -> None:
    await repo.upsert(conn, "decisions", decision.row(), key=("decision_id",))


async def write_decisions(conn: Any, result: AskResult) -> None:
    for d in result.decisions:
        await write_decision(conn, d)


class JevService:
    def __init__(
        self,
        registry: Registry,
        provider: JevProvider,
        db: Database | None,
        run_id: str | None = None,
    ) -> None:
        self.reg = registry
        self.provider = provider
        self.db = db
        self.run_id = run_id
        self.profile = provider.profile
        self.states = StateBuilder(registry, provider.estimator)
        self.questions = QuestionBuilder(registry, self.profile["limits"])
        self.store_state = registry.policy("jev.store_state")
        self._in_flight: dict[str, asyncio.Future[tuple[RawResult, str, str, bool]]] = {}
        self._credits = asyncio.Event()  # cleared while the account has no credits
        self._credits.set()

    # ------------------------------------------------------------------ public

    async def ask(
        self,
        qs_name: str,
        inputs: dict[str, Any],
        subject_kind: str,
        subject_id: str,
        *,
        fanout_items: dict[str, Any] | None = None,
        dynamic: dict[str, dict[str, Any]] | None = None,
        language: str | None = None,
        bypass_cache: bool = False,
        qs_version: int | None = None,
        only: set[str] | None = None,
        item_only: dict[str, set[str]] | None = None,
        repeat: int = 0,
    ) -> AskResult:
        """`repeat`: how many times this question already came back to Jev for this subject; each
        repeat raises the Noul threshold (§11.5 back-off)."""
        qs = self.reg.question_set(qs_name, qs_version)
        if qs.fanout and fanout_items is not None:
            cap = self.reg.policy(qs.fanout["max_ref"])
            if len(fanout_items) > cap:
                raise JevRequestError(f"{qs.ref}: {len(fanout_items)} fan-out items exceed policy max {cap}")
        state = self.states.build(qs, inputs, fanout_items)
        return await self.ask_state(qs, state, subject_kind, subject_id, dynamic=dynamic,
                                    language=language, bypass_cache=bypass_cache, only=only, item_only=item_only,
                                    repeat=repeat)

    async def ask_state(
        self,
        qs: QuestionSet,
        state: dict[str, Any],
        subject_kind: str,
        subject_id: str,
        *,
        dynamic: dict[str, dict[str, Any]] | None = None,
        language: str | None = None,
        bypass_cache: bool = False,
        only: set[str] | None = None,
        item_only: dict[str, set[str]] | None = None,
        repeat: int = 0,
    ) -> AskResult:
        """Ask over an already-built state (gold-set evaluation replays stored states, §13)."""
        items = list(state[qs.fanout["over"]]) if qs.fanout else None
        rendered = self.questions.render(qs, items=items, dynamic=dynamic, only=only, item_only=item_only)
        state_hash = sha256_hex(state)

        result = AskResult(qs=qs, decisions=[], dynamic=dynamic)
        answers, provider = await self._answer(qs, state, state_hash, rendered.wire, rendered, bypass_cache, result)
        if rendered.hierarchical:
            answers = await self._resolve_children(qs, state, state_hash, rendered, answers, bypass_cache, result)

        bar = self.bar(repeat)
        per_item = self._split_answers(rendered, answers, items)
        for item, logical_answers in per_item.items():
            gate = evaluate(qs, logical_answers, bar)
            result.decisions.append(
                Decision(
                    decision_id=sha256_hex(subject_kind, subject_id, qs.ref, state_hash, rendered.questions_hash, item or "",
                                           *([repeat] if repeat else [])),
                    subject_kind=subject_kind,
                    subject_id=subject_id,
                    question_set=qs.ref,
                    item=item,
                    outcome=gate.outcome,
                    answers=logical_answers,
                    call_ids=list(result.call_ids),
                    provider=provider,
                    run_id=self.run_id,
                    threshold=bar.for_qs(qs),
                    repeat=repeat,
                    leaf_thresholds=bar.for_leaves(qs),
                )
            )
        return result

    def bar(self, repeat: int) -> Bar:
        """§11.5: the Noul threshold and its back-off, from policy."""
        return Bar(self.reg.policy("gating.threshold"), self.reg.policy("gating.backoff"), repeat)

    async def replay(self, call: dict[str, Any]) -> RawResult:
        """Re-ask a stored call verbatim, bypassing the cache (drift sampling, §8.8).

        `call` is a `jev_calls` row with `state`, `questions` and `keys` stored."""
        qs = self.reg.question_set(call["question_set"], call["qs_version"])
        raw, _, _, _ = await self._call(qs, call["state"], call["state_hash"], call["questions"], call["keys"], True, call.get("dynamic"))
        return raw

    # ---------------------------------------------------------------- internals

    def _split_answers(
        self, rendered: Rendered, answers: dict[str, dict], items: list[str] | None
    ) -> dict[str | None, dict[str, dict]]:
        shared = {rendered.keys[k].logical: a for k, a in answers.items() if rendered.keys[k].item is None}
        if not items:
            return {None: shared}
        out: dict[str | None, dict[str, dict]] = {item: dict(shared) for item in items}
        for k, a in answers.items():
            info = rendered.keys[k]
            if info.item is not None:
                out[info.item][info.logical] = a
        return out

    def _batches(self, state: Any, wire: dict[str, dict]) -> list[dict[str, dict]]:
        """Pack questions sharing one state into as few calls as the context window allows (§11.1, §11.3)."""
        est = self.provider.estimator
        window = self.profile["context_tokens"]
        longest = self.profile["longest_question_tokens"]
        state_tokens = est.value(state)
        room = window - state_tokens
        if room <= 0:
            raise JevRequestError(f"state alone needs {state_tokens} tokens, over the {window}-token context window; tighten the state template")
        batches: list[dict[str, dict]] = []
        current: dict[str, dict] = {}
        used = 0
        for key, q in wire.items():
            size = est.value({key: q})
            if size > longest or size > room:
                raise JevRequestError(f"question {key} needs {size} tokens, over the per-question limit")
            if current and used + size > room:
                batches.append(current)
                current, used = {}, 0
            current[key] = q
            used += size
        if current:
            batches.append(current)
        return batches

    async def _answer(
        self,
        qs: QuestionSet,
        state: Any,
        state_hash: str,
        wire: dict[str, dict],
        rendered: Rendered,
        bypass_cache: bool,
        result: AskResult,
    ) -> tuple[dict[str, dict], str]:
        answers: dict[str, dict] = {}
        for batch in self._batches(state, wire):
            keys = {k: [rendered.keys[k].logical, rendered.keys[k].item] for k in batch}
            raw, call_id, _, cached = await self._call(qs, state, state_hash, batch, keys, bypass_cache, result.dynamic)
            result.call_ids.append(call_id)
            result.cache_hits += int(cached)
            answers.update(self._parse(rendered, batch, raw.answers))
        return answers, self.provider.name

    async def _resolve_children(
        self,
        qs: QuestionSet,
        state: Any,
        state_hash: str,
        rendered: Rendered,
        answers: dict[str, dict],
        bypass_cache: bool,
        result: AskResult,
    ) -> dict[str, dict]:
        """Second level of a hierarchical choice: ask among the chosen parent's children."""
        child_wire: dict[str, dict] = {}
        child_of: dict[str, str] = {}
        for key, info in rendered.keys.items():
            if info.logical not in rendered.hierarchical:
                continue
            parent = answers[key]["choice"]
            children = rendered.hierarchical[info.logical][parent]
            if len(children) == 1:
                answers[key] = {**answers[key], "choice": children[0], "parent": parent}
                continue
            ckey = short_key(key, _CHILD_SUFFIX, parent)
            child_wire[ckey] = self.questions.child_question(qs, rendered, key, parent)
            child_of[ckey] = key
        if not child_wire:
            return answers
        child_rendered = Rendered(
            wire=child_wire,
            keys={ck: rendered.keys[pk] for ck, pk in child_of.items()},
            questions_hash=sha256_hex(child_wire),
        )
        child_answers, _ = await self._answer(qs, state, state_hash, child_wire, child_rendered, bypass_cache, result)
        for ckey, pkey in child_of.items():
            child = child_answers[ckey]
            answers[pkey] = {**child, "parent": answers[pkey]["choice"]}
        return answers

    def _parse(self, rendered: Rendered, batch: dict[str, dict], raw_answers: dict[str, Any]) -> dict[str, dict]:
        sub = Rendered(wire=batch, keys={k: rendered.keys[k] for k in batch}, questions_hash=rendered.questions_hash)
        model = sub.response_model()
        try:
            parsed = model.model_validate({"model": "", "answers": raw_answers})
        except ValidationError as exc:
            raise JevAnswerError(f"answers do not match the requested questions: {exc.errors()[:3]}") from exc
        out: dict[str, dict] = {}
        for key in batch:
            info = rendered.keys[key]
            answer = normalize_answer(getattr(parsed.answers, key), info)
            if info.qtype == "choice" and answer["choice"] not in batch[key]["criteria"]:
                raise JevAnswerError(f"choice `{answer['choice']}` is not one of the offered options")
            out[key] = answer
        return out

    def _cache_key(self, qs: QuestionSet, state_hash: str, batch: dict[str, dict]) -> str:
        return sha256_hex(self.provider.model, qs.ref, state_hash, sha256_hex(batch))

    async def _call(
        self, qs: QuestionSet, state: Any, state_hash: str, batch: dict[str, dict], keys: dict[str, list], bypass_cache: bool,
        dynamic: dict[str, dict[str, Any]] | None = None,
    ) -> tuple[RawResult, str, str, bool]:
        cache_key = self._cache_key(qs, state_hash, batch)
        if bypass_cache:
            return await self._cached_or_paid(qs, state, state_hash, batch, keys, True, dynamic, cache_key)
        # An identical call already in flight serves this one: both would miss the cache and pay twice.
        # The shared call outlives a cancelled caller (shield); its failure reaches every caller and
        # leaves nothing behind, so the next identical ask tries again.
        flight = self._in_flight.get(cache_key)
        if flight is not None:
            raw, call_id, provider, _ = await asyncio.shield(flight)
            log.get().debug("jev_call_shared", call_id=call_id, question_set=qs.ref)
            return raw, call_id, provider, True
        flight = asyncio.ensure_future(
            self._cached_or_paid(qs, state, state_hash, batch, keys, False, dynamic, cache_key))
        self._in_flight[cache_key] = flight
        flight.add_done_callback(lambda f: self._landed(cache_key, f))
        return await asyncio.shield(flight)

    def _landed(self, cache_key: str, flight: asyncio.Future) -> None:
        del self._in_flight[cache_key]
        if not flight.cancelled():
            flight.exception()  # retrieved here too: every caller may have been cancelled

    async def _reuse(
        self, qs: QuestionSet, state_hash: str, batch: dict[str, dict], keys: dict[str, list],
    ) -> tuple[RawResult, str, str, bool] | None:
        """A stored call of this question set (any version) on the same state, by the same model, that asked
        every question in `batch` word for word: its answers are this call's. So a version that only removes
        questions re-asks nothing; a reworded question, a changed option list or a new question is asked."""
        rows = await self.db.fetch(
            "SELECT call_id, questions, keys, answers, jev_model, provider FROM jev_calls "
            "WHERE question_set = %s AND state_hash = %s AND provider = %s AND jev_model = %s AND NOT drift "
            "ORDER BY created_at",
            (qs.name, state_hash, self.provider.name, self.provider.model),
        )
        for row in rows:
            stored = {tuple(v): k for k, v in (row["keys"] or {}).items()}
            answers: dict[str, Any] = {}
            for key, question in batch.items():
                old = stored.get(tuple(keys[key]))
                if old is None or row["questions"].get(old) != question or old not in row["answers"]:
                    break
                answers[key] = row["answers"][old]
            else:
                log.get().info("jev_call_reused", question_set=qs.ref, call_id=row["call_id"], questions=len(batch))
                return RawResult(answers, row["jev_model"], {}, None, 0, row["provider"]), row["call_id"], row["provider"], True
        return None

    async def _call_when_credited(self, state: Any, batch: dict[str, dict]) -> RawResult:
        """Out of credits, calls queue instead of failing: the call that hit the outage retries on an
        interval while every other call waits for it, and all of them go ahead once credits are back."""
        while True:
            await self._credits.wait()
            try:
                return await self.provider.call(state, batch)
            except JevCreditsExhausted as exc:
                if not self._credits.is_set():
                    continue  # another call is already waiting for the credits
                self._credits.clear()
                wait_s = self.reg.policy("jev.credits_wait_s")
                log.get().warning("jev_credits_exhausted_waiting", error=str(exc), retry_every_s=wait_s)
                try:
                    while True:
                        await asyncio.sleep(wait_s)
                        try:
                            raw = await self.provider.call(state, batch)
                        except JevCreditsExhausted:
                            continue
                        log.get().warning("jev_credits_restored")
                        return raw
                finally:
                    self._credits.set()

    async def _cached_or_paid(
        self, qs: QuestionSet, state: Any, state_hash: str, batch: dict[str, dict], keys: dict[str, list], bypass_cache: bool,
        dynamic: dict[str, dict[str, Any]] | None, cache_key: str,
    ) -> tuple[RawResult, str, str, bool]:
        if not bypass_cache and self.db is not None:
            row = await self.db.fetchone(
                "SELECT call_id, answers, usage, jev_model, provider FROM jev_calls "
                "WHERE cache_key = %s AND provider = %s AND NOT drift ORDER BY created_at LIMIT 1",
                (cache_key, self.provider.name),
            )
            if row is not None:
                raw = RawResult(row["answers"], row["jev_model"], row["usage"] or {}, None, 0, row["provider"])
                return raw, row["call_id"], row["provider"], True
            reused = await self._reuse(qs, state_hash, batch, keys)
            if reused is not None:
                return reused

        tokens = self.provider.estimator.value({"state": state, "questions": batch})
        await self.provider.limiter.acquire(tokens)
        raw = await self._call_when_credited(state, batch)
        # Drift samples get their own call row so they never overwrite (or serve as) the cache.
        call_id = sha256_hex(cache_key, "drift", self.run_id or "") if bypass_cache else cache_key
        if self.db is not None:
            async with self.db.tx() as conn:
                await repo.upsert(
                    conn,
                    "jev_calls",
                    {
                        "call_id": call_id,
                        "request_id": raw.request_id,
                        "question_set": qs.name,
                        "qs_version": qs.version,
                        "jev_model": raw.model,
                        "state_hash": state_hash,
                        "state": state if self.store_state else None,
                        "questions_hash": sha256_hex(batch),
                        "questions": batch,
                        "keys": keys,
                        "dynamic": dynamic,
                        "answers": raw.answers,
                        "usage": raw.usage,
                        "latency_ms": raw.latency_ms,
                        "run_id": self.run_id,
                        "provider": raw.provider,
                        "cache_key": cache_key,
                        "drift": bypass_cache,
                    },
                    key=("call_id",),
                )
        log.get().info(
            "jev_call", call_id=call_id, question_set=qs.ref, model=raw.model, provider=raw.provider,
            latency_ms=raw.latency_ms, usage=raw.usage,
        )
        return raw, call_id, raw.provider, False
