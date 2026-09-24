"""Jev providers (§11.1, §11.7).

* `TypeSafeProvider` wraps `typesafe-sdk`'s `AsyncTypeSafeClient` (native API or
  an SDK-compatible gateway via `endpoint`).
* `GatewayProvider` speaks to third-party gateways whose field names differ; the
  translation lives in the provider profile (`field_map`).

Every decision comes from Jev. There is no LLM stand-in: when Jev is
unavailable, calls fail with `JevUnavailable` and the ledger leaves the item
pending for the next run (§14.2).

All endpoints, model IDs, limits and retry settings come from profiles
and policies; nothing is left at library defaults.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from jev_graph_builder.config import secret
from jev_graph_builder.jev.limiter import DualLimiter, LimiterSettings
from jev_graph_builder.jev.tokens import TokenEstimator

THROTTLE_STATUSES = frozenset({429, 529})
CREDITS_EXHAUSTED_STATUS = 402


class JevUnavailable(Exception):
    """The provider could not answer (network, 5xx, 529 after retries)."""


class JevRequestError(Exception):
    """The request itself is wrong (401/422): never retried, never failed over."""


class JevCreditsExhausted(Exception):
    """The Jev account has no credits (402): every call fails until credits are added, so the
    build stops and leaves unfinished items pending instead of failing them one by one."""


@dataclass
class RawResult:
    answers: dict[str, Any]
    model: str
    usage: dict[str, Any]
    request_id: str | None
    latency_ms: int
    provider: str


class JevProvider(Protocol):
    name: str
    kind: str
    model: str
    profile: dict[str, Any]
    limiter: DualLimiter
    estimator: TokenEstimator

    async def call(self, state: Any, questions: dict[str, dict]) -> RawResult: ...

    async def aclose(self) -> None: ...


def _limiter(profile: dict, policies: dict) -> DualLimiter:
    rate = profile["rate"]
    adapt = policies["jev"]["limiter"]
    return DualLimiter(
        LimiterSettings(
            requests_per_minute=rate["requests_per_minute"],
            tokens_per_second=rate["tokens_per_second"],
            backoff_factor=adapt["backoff_factor"],
            floor_fraction=adapt["floor_fraction"],
            recover_after_s=adapt["recover_after_s"],
            recover_factor=adapt["recover_factor"],
        )
    )


def _throttle_observer(inner: Any, on_throttle: Any) -> Any:
    """Wrap the SDK transport so every attempt's 429/529 reaches the adaptive limiter.

    `RetryPolicy.predicate` cannot do this: the SDK consults it only when the
    built-in status list has already declined, and 429/529 are on that list.
    """
    import httpx2

    class _Observer(httpx2.AsyncBaseTransport):
        def __init__(self) -> None:
            self.inner = inner or httpx2.AsyncHTTPTransport()

        async def handle_async_request(self, request: Any) -> Any:
            response = await self.inner.handle_async_request(request)
            if response.status_code in THROTTLE_STATUSES:
                on_throttle()
            return response

        async def aclose(self) -> None:
            await self.inner.aclose()

    return _Observer()


def _retry_policy(policies: dict):
    from typesafe_sdk import RetryPolicy

    r = policies["jev"]["retry"]
    return RetryPolicy(
        max_retries=r["max_retries"],
        backoff_initial=r["backoff_initial_s"],
        backoff_max=r["backoff_max_s"],
        backoff_jitter=r["backoff_jitter"],
        http_statuses=set(r["http_statuses"]),
        respect_retry_after=True,
        timeout=r["total_timeout_s"],
    )


class TypeSafeProvider:
    kind = "typesafe"

    def __init__(self, profile: dict, policies: dict, transport: Any = None) -> None:
        from typesafe_sdk import AsyncTypeSafeClient

        self.profile = profile
        self.name = profile["name"]
        self.model = profile["model"]
        self.limiter = _limiter(profile, policies)
        self.estimator = TokenEstimator(profile["chars_per_token"])
        key = secret(profile["key_env"])
        if key is None and transport is None:
            raise JevRequestError(f"missing API key in ${profile['key_env']}")

        self._client = AsyncTypeSafeClient(
            api_key=key or "test-key",
            model=self.model,
            base_url=profile.get("endpoint"),
            timeout=profile["timeout_s"],
            retry=_retry_policy(policies),
            transport=_throttle_observer(transport, self.limiter.throttled),
        )

    async def call(self, state: Any, questions: dict[str, dict]) -> RawResult:
        from typesafe_sdk import TypeSafeAPIConnectionError, TypeSafeAPIError

        t0 = time.perf_counter()
        try:
            resp = await self._client.system_one(state=state, questions=questions, model=self.model)
        except TypeSafeAPIError as exc:
            if exc.status == CREDITS_EXHAUSTED_STATUS:
                raise JevCreditsExhausted(str(exc)) from exc
            if exc.status in THROTTLE_STATUSES or exc.status >= 500:
                raise JevUnavailable(str(exc)) from exc
            raise JevRequestError(str(exc)) from exc
        except TypeSafeAPIConnectionError as exc:
            raise JevUnavailable(str(exc)) from exc
        body = resp.model_dump(mode="json")
        return RawResult(
            answers=body["answers"],
            model=body["model"],
            usage=body.get("usage") or {},
            request_id=resp.request_id,
            latency_ms=int((time.perf_counter() - t0) * 1000),
            provider=self.name,
        )

    async def aclose(self) -> None:
        await self._client.aclose()


class GatewayProvider:
    """Third-party gateway with its own model IDs and field names (R-011)."""

    kind = "gateway"

    def __init__(self, profile: dict, policies: dict, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.profile = profile
        self.name = profile["name"]
        self.model = profile["model"]
        self.limiter = _limiter(profile, policies)
        self.estimator = TokenEstimator(profile["chars_per_token"])
        self.fields = profile["field_map"]
        key = secret(profile["key_env"])
        headers = {profile.get("auth_header", "Authorization"): f"{profile.get('auth_scheme', 'Bearer')} {key}"} if key else {}
        self._http = httpx.AsyncClient(timeout=profile["timeout_s"], headers=headers, transport=transport)
        self._retries = policies["jev"]["retry"]

    async def call(self, state: Any, questions: dict[str, dict]) -> RawResult:
        f = self.fields
        body = {f["model"]: self.model, f["state"]: state, f["questions"]: questions}
        t0 = time.perf_counter()
        attempt = 0
        while True:
            try:
                resp = await self._http.post(self.profile["endpoint"], json=body)
            except httpx.HTTPError as exc:
                raise JevUnavailable(str(exc)) from exc
            if resp.status_code in THROTTLE_STATUSES:
                self.limiter.throttled()
            if resp.status_code in set(self._retries["http_statuses"]) and attempt < self._retries["max_retries"]:
                attempt += 1
                import asyncio

                await asyncio.sleep(min(self._retries["backoff_max_s"], self._retries["backoff_initial_s"] * (2 ** (attempt - 1))))
                continue
            break
        if resp.status_code == CREDITS_EXHAUSTED_STATUS:
            raise JevCreditsExhausted(f"gateway {resp.status_code}")
        if resp.status_code >= 500 or resp.status_code in THROTTLE_STATUSES:
            raise JevUnavailable(f"gateway {resp.status_code}")
        if resp.status_code >= 400:
            raise JevRequestError(f"gateway {resp.status_code}: {resp.text[:500]}")
        data = resp.json()
        return RawResult(
            answers=data[f["answers"]],
            model=data.get(f.get("response_model", "model"), self.model),
            usage=data.get(f.get("usage", "usage")) or {},
            request_id=resp.headers.get(self.profile.get("request_id_header", "x-typesafe-request-id")),
            latency_ms=int((time.perf_counter() - t0) * 1000),
            provider=self.name,
        )

    async def aclose(self) -> None:
        await self._http.aclose()


def build_provider(profile: dict, policies: dict, transport: Any = None) -> JevProvider:
    kind = profile["kind"]
    if kind == "typesafe":
        return TypeSafeProvider(profile, policies, transport=transport)
    if kind == "gateway":
        return GatewayProvider(profile, policies, transport=transport)
    raise JevRequestError(f"unknown Jev provider kind `{kind}`")
