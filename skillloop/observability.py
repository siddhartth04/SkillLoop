"""Structured logging + metrics for production.

Off by default (a library shouldn't hijack logging). Turn on with SKILLLOOP_LOG=json|text, or call
configure_logging(). Every core operation emits one structured event with a request id so a single
learn()/recall() can be traced end to end, plus skill version ids on promotion/demotion.
"""
from __future__ import annotations

import contextvars
import json
import logging
import os
import time
import uuid
from contextlib import contextmanager

_request_id: contextvars.ContextVar[str] = contextvars.ContextVar("skillloop_request_id", default="")
logger = logging.getLogger("skillloop")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": round(record.created, 3),
            "level": record.levelname,
            "event": record.getMessage(),
            "request_id": _request_id.get() or None,
        }
        for k, v in getattr(record, "fields", {}).items():
            payload[k] = v
        return json.dumps(payload, default=str)


def configure_logging(mode: str | None = None, level: str = "INFO") -> None:
    mode = mode or os.getenv("SKILLLOOP_LOG", "")
    if not mode:
        return
    h = logging.StreamHandler()
    h.setFormatter(JsonFormatter() if mode == "json"
                   else logging.Formatter("%(asctime)s skillloop %(levelname)s %(message)s"))
    logger.handlers[:] = [h]
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False


def log(event: str, level: str = "INFO", **fields) -> None:
    if not logger.handlers:
        return
    rec = logger.makeRecord(logger.name, getattr(logging, level, 20), __file__, 0, event, (), None)
    rec.fields = fields
    logger.handle(rec)


@contextmanager
def request(kind: str, **fields):
    """Scope one logical operation. Emits <kind>.start / .ok / .error with a shared request id and latency."""
    rid = uuid.uuid4().hex[:12]
    token = _request_id.set(rid)
    t0 = time.time()
    log(f"{kind}.start", **fields)
    try:
        yield rid
    except Exception as e:
        log(f"{kind}.error", level="ERROR", error=f"{type(e).__name__}: {e}",
            latency_ms=int((time.time() - t0) * 1000))
        raise
    else:
        log(f"{kind}.ok", latency_ms=int((time.time() - t0) * 1000))
    finally:
        _request_id.reset(token)


def current_request_id() -> str:
    return _request_id.get()
