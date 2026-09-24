"""Global async limiter on both requests/min and tokens/s (R-110).

Two token buckets gate every Jev call. On 429/529 both rates shrink by a
policy factor (never below a policy floor) and recover gradually after a quiet
period. All numbers come from the provider profile and policies.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass

SECONDS_PER_MINUTE = 60.0


@dataclass
class LimiterSettings:
    requests_per_minute: float
    tokens_per_second: float
    backoff_factor: float          # multiply rates by this on 429/529
    floor_fraction: float          # never go below this fraction of the profile rate
    recover_after_s: float         # quiet period before rates step back up
    recover_factor: float          # multiply the reduced rate by this per recovery step (> 1)


class _Bucket:
    def __init__(self, rate_per_s: float, capacity: float, clock: Callable[[], float]) -> None:
        self.rate = rate_per_s
        self.capacity = capacity
        self.level = capacity
        self._clock = clock
        self._t = clock()

    def _refill(self) -> None:
        now = self._clock()
        self.level = min(self.capacity, self.level + (now - self._t) * self.rate)
        self._t = now

    def wait_time(self, amount: float) -> float:
        self._refill()
        amount = min(amount, self.capacity)
        return 0.0 if self.level >= amount else (amount - self.level) / self.rate

    def take(self, amount: float) -> None:
        self._refill()
        self.level -= min(amount, self.capacity)


class DualLimiter:
    def __init__(self, s: LimiterSettings, clock: Callable[[], float] = time.monotonic, sleep=asyncio.sleep) -> None:
        self.s = s
        self._clock = clock
        self._sleep = sleep
        self._scale = 1.0
        self._last_throttle = -float("inf")
        self._lock = asyncio.Lock()
        rps = s.requests_per_minute / SECONDS_PER_MINUTE
        self._req = _Bucket(rps, max(1.0, rps), clock)
        self._tok = _Bucket(s.tokens_per_second, s.tokens_per_second, clock)

    @property
    def scale(self) -> float:
        return self._scale

    def _apply_scale(self) -> None:
        self._req.rate = self.s.requests_per_minute / SECONDS_PER_MINUTE * self._scale
        self._tok.rate = self.s.tokens_per_second * self._scale

    def throttled(self) -> None:
        """Called on 429/529: adapt downward."""
        self._scale = max(self.s.floor_fraction, self._scale * self.s.backoff_factor)
        self._last_throttle = self._clock()
        self._apply_scale()

    def _maybe_recover(self) -> None:
        if self._scale < 1.0 and self._clock() - self._last_throttle >= self.s.recover_after_s:
            self._scale = min(1.0, self._scale * self.s.recover_factor)
            self._last_throttle = self._clock()
            self._apply_scale()

    async def acquire(self, tokens: int) -> None:
        async with self._lock:
            self._maybe_recover()
            while True:
                wait = max(self._req.wait_time(1), self._tok.wait_time(tokens))
                if wait <= 0:
                    break
                await self._sleep(wait)
            self._req.take(1)
            self._tok.take(tokens)
