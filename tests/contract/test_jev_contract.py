"""Contract tests: Jev providers against recorded responses (§20).

`system_one_mixed.json` is a real `jev-1.13.0` exchange (request and response),
recorded once and replayed through `httpx2.MockTransport`.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import httpx2
import pytest

from jev_graph_builder.jev.client import (GatewayProvider, JevCreditsExhausted, JevRequestError, JevUnavailable, RawResult,
                                         TypeSafeProvider)
from jev_graph_builder.jev.service import JevService
from jev_graph_builder.jev.questions import KeyInfo, Rendered, normalize_answer
from jev_graph_builder.registry.loader import Registry

ROOT = Path(__file__).resolve().parents[2]
CASSETTE = json.loads((ROOT / "tests/fixtures/cassettes/system_one_mixed.json").read_text())


@pytest.fixture(scope="module")
def reg() -> Registry:
    return Registry(ROOT / "registry")


def _profile(reg: Registry) -> dict:
    return dict(reg.profile("jev", "typesafe"))


def _replay(status: int = 200, body: dict | None = None, seen: list | None = None) -> httpx2.MockTransport:
    def handler(request: httpx2.Request) -> httpx2.Response:
        if seen is not None:
            seen.append(json.loads(request.content))
        return httpx2.Response(status, json=body if body is not None else CASSETTE["response"], headers=CASSETTE["headers"])

    return httpx2.MockTransport(handler)


async def test_typesafe_provider_replays_cassette(reg: Registry) -> None:
    seen: list = []
    prov = TypeSafeProvider(_profile(reg), reg.policies, transport=_replay(seen=seen))
    raw = await prov.call(CASSETTE["request"]["state"], CASSETTE["request"]["questions"])
    await prov.aclose()
    assert seen[0]["questions"] == CASSETTE["request"]["questions"]
    assert seen[0]["model"] == reg.profile("jev", "typesafe")["model"]
    assert raw.model == "jev-1.13.0"
    assert raw.usage["input_tokens"] == CASSETTE["response"]["usage"]["input_tokens"]
    assert set(raw.answers) == {"k1", "k2", "k3"}


def test_recorded_answers_parse_and_normalize() -> None:
    keys = {"k1": KeyInfo("a", None, "noul"), "k2": KeyInfo("b", None, "choice"), "k3": KeyInfo("c", None, "score", 3)}
    rendered = Rendered(wire=CASSETTE["request"]["questions"], keys=keys, questions_hash="0" * 16)
    parsed = rendered.response_model().model_validate(CASSETTE["response"])
    noul = normalize_answer(parsed.answers.k1, keys["k1"])
    choice = normalize_answer(parsed.answers.k2, keys["k2"])
    score = normalize_answer(parsed.answers.k3, keys["k3"])
    assert 0 <= noul["p"] <= 1
    assert choice["choice"] in CASSETTE["request"]["questions"]["k2"]["criteria"]
    assert set(score["probabilities"]) == {0, 1, 2}  # levels are 0-based on the wire
    assert score["levels"] == 3


def test_missing_answer_key_is_rejected() -> None:
    keys = {"k1": KeyInfo("a", None, "noul"), "k9": KeyInfo("z", None, "noul")}
    rendered = Rendered(wire={}, keys=keys, questions_hash="1" * 16)
    with pytest.raises(Exception):
        rendered.response_model().model_validate(CASSETTE["response"])


async def test_throttle_adapts_limiter_and_raises_unavailable(reg: Registry) -> None:
    policies = json.loads(json.dumps(reg.policies))
    policies["jev"]["retry"]["max_retries"] = 0
    prov = TypeSafeProvider(_profile(reg), policies, transport=_replay(429, {"error": {"message": "slow down"}}))
    before = prov.limiter._scale
    with pytest.raises(JevUnavailable):
        await prov.call({"x": {"t": "a"}}, {"k": {"type": "noul", "instructions": "Is `x.t` set?"}})
    assert prov.limiter._scale < before
    await prov.aclose()


async def test_no_credits_is_a_run_stop_not_an_item_failure(reg: Registry) -> None:
    """402: every call would fail until credits are added, so the provider raises the run-stop error."""
    prov = TypeSafeProvider(_profile(reg), reg.policies, transport=_replay(402, {"error": {"message": "no credits"}}))
    with pytest.raises(JevCreditsExhausted):
        await prov.call({"x": {"t": "a"}}, {"k": {"type": "noul", "instructions": "Is `x.t` set?"}})
    await prov.aclose()


async def test_validation_error_is_not_retried_as_outage(reg: Registry) -> None:
    body = {"error": {"message": "questions.k3.score.criteria: Input should be a valid list"}}
    prov = TypeSafeProvider(_profile(reg), reg.policies, transport=_replay(422, body))
    with pytest.raises(JevRequestError):
        await prov.call({"x": {"t": "a"}}, {"k": {"type": "noul", "instructions": "Is `x.t` set?"}})
    await prov.aclose()


async def test_gateway_field_map(reg: Registry) -> None:
    profile = {
        **_profile(reg),
        "name": "gw",
        "endpoint": "https://gateway.example/v2/judge",
        "model": "vendor/jev-1.13",
        "field_map": {"model": "engine", "state": "context", "questions": "items", "answers": "results"},
    }
    seen: list = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"results": CASSETTE["response"]["answers"], "model": "vendor/jev-1.13"})

    prov = GatewayProvider(profile, reg.policies, transport=httpx.MockTransport(handler))
    raw = await prov.call({"x": {"t": "a"}}, CASSETTE["request"]["questions"])
    await prov.aclose()
    assert set(seen[0]) == {"engine", "context", "items"}
    assert raw.answers == CASSETTE["response"]["answers"]


async def test_identical_calls_in_flight_pay_once(reg: Registry) -> None:
    """Identical requests asked concurrently (e.g. the same link between two copies of a document) share one paid call."""
    prov = TypeSafeProvider(_profile(reg), reg.policies, transport=_replay())
    paid, release = [], asyncio.Event()

    async def call(state, questions):  # noqa: ANN001, ANN202
        paid.append(questions)
        await release.wait()
        return RawResult(CASSETTE["response"]["answers"], "jev-1.13.0", {"input_tokens": 10}, None, 0, prov.name)

    prov.call = call
    jev = JevService(reg, prov, db=None)
    qs, state, batch = reg.question_set("relation_overlap"), {"s": 1}, CASSETTE["request"]["questions"]
    same = [jev._call(qs, state, "h", batch, {}, False) for _ in range(3)]
    other = jev._call(qs, state, "h2", batch, {}, False)
    tasks = [asyncio.ensure_future(c) for c in [*same, other]]
    await asyncio.sleep(0)
    release.set()
    results = await asyncio.gather(*tasks)
    await prov.aclose()
    assert len(paid) == 2                                   # one per distinct request
    assert [cached for *_, cached in results] == [False, True, True, False]
    assert len({call_id for _, call_id, _, _ in results[:3]}) == 1
    assert jev._in_flight == {}


async def test_shared_call_failure_reaches_every_caller_and_is_retried(reg: Registry) -> None:
    prov = TypeSafeProvider(_profile(reg), reg.policies, transport=_replay())
    attempts, release = [], asyncio.Event()

    async def call(state, questions):  # noqa: ANN001, ANN202
        attempts.append(1)
        await release.wait()
        if len(attempts) == 1:
            raise JevUnavailable("outage")
        return RawResult(CASSETTE["response"]["answers"], "jev-1.13.0", {"input_tokens": 10}, None, 0, prov.name)

    prov.call = call
    jev = JevService(reg, prov, db=None)
    qs, batch = reg.question_set("relation_overlap"), CASSETTE["request"]["questions"]
    tasks = [asyncio.ensure_future(jev._call(qs, {"s": 1}, "h", batch, {}, False)) for _ in range(2)]
    await asyncio.sleep(0)
    release.set()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    assert len(attempts) == 1 and all(isinstance(r, JevUnavailable) for r in results)
    *_, cached = await jev._call(qs, {"s": 1}, "h", batch, {}, False)
    await prov.aclose()
    assert len(attempts) == 2 and not cached


async def test_cancelled_caller_does_not_cancel_the_shared_call(reg: Registry) -> None:
    prov = TypeSafeProvider(_profile(reg), reg.policies, transport=_replay())
    release = asyncio.Event()

    async def call(state, questions):  # noqa: ANN001, ANN202
        await release.wait()
        return RawResult(CASSETTE["response"]["answers"], "jev-1.13.0", {"input_tokens": 10}, None, 0, prov.name)

    prov.call = call
    jev = JevService(reg, prov, db=None)
    qs, batch = reg.question_set("relation_overlap"), CASSETTE["request"]["questions"]
    first = asyncio.ensure_future(jev._call(qs, {"s": 1}, "h", batch, {}, False))
    second = asyncio.ensure_future(jev._call(qs, {"s": 1}, "h", batch, {}, False))
    await asyncio.sleep(0)
    first.cancel()
    release.set()
    raw, *_, cached = await second
    await prov.aclose()
    assert first.cancelled() and cached and raw.answers == CASSETTE["response"]["answers"]
