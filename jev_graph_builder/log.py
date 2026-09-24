"""Structured JSON logging keyed by run_id / item_id / decision_id / harness_run_id (§16)."""

from __future__ import annotations

import logging
import sys

import structlog


def configure(level: str = "INFO", json_output: bool = True) -> None:
    logging.basicConfig(stream=sys.stderr, level=getattr(logging, level.upper(), logging.INFO), format="%(message)s")
    # The HTTP clients log every request at INFO; the `jev_call` event already records each call.
    for noisy in ("httpx", "httpx2", "httpcore", "typesafe_sdk"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]
    processors.append(structlog.processors.JSONRenderer() if json_output else structlog.dev.ConsoleRenderer())
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level.upper(), logging.INFO)),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=False,
    )


def get(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)


bind = structlog.contextvars.bind_contextvars
unbind = structlog.contextvars.unbind_contextvars
