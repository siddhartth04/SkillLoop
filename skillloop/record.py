"""Trace recording. The two-call contract is only two calls if the second one is easy.

Before this, `learn(trace)` meant hand-assembling steps, tool calls, results and an outcome signal. That is
the real integration cost of SkillLoop, and it is what stopped the contract from being two calls in practice.

    from skillloop import SkillLoop

    loop = SkillLoop()
    with loop.session("fix the failing test") as s:
        for skill in s.recall():                 # skill text to put in your prompt
            ...
        out = subprocess.run(cmd, capture_output=True)
        s.tool("bash", {"cmd": cmd}, result=out.stdout, error=out.stderr if out.returncode else None)
        s.say("running the suite again")
        s.ok()                                   # or s.fail("still red") - or omit and let the judge decide

On exit the session builds the Trace and calls learn() for you. An uncaught exception is recorded as a
failure signal, which is usually the most valuable trace of all.
"""
from __future__ import annotations

import functools
import json
import time
from typing import Any, Callable

from .schema import Signal, Step, ToolCall, Trace


class Session:
    """Records what an agent did, then hands it to learn() on exit."""

    def __init__(self, loop, task: str, agent: str = "", metadata: dict | None = None,
                 auto_learn: bool = True, error_hints: bool = True,
                 on_hint: Callable[[list[dict[str, Any]]], Any] | None = None):
        self._loop = loop
        self.task = task
        self.agent = agent
        self.metadata = metadata or {}
        self.auto_learn = auto_learn
        self.steps: list[Step] = [Step("user", task)]
        self.skills_used: list[str] = []
        self.helpful: list[str] = []
        self.harmful: list[str] = []
        self._outcome: str | None = None
        self._detail = ""
        self._confidence = 0.9
        self.trace: Trace | None = None
        self.result: dict | None = None
        self.started = time.time()
        # error-time recall: when a tool call fails, skills whose recorded symptoms match the error are queued
        # here for the agent's NEXT step. Mistakes show up mid-task, not in the task description.
        self.error_hints = error_hints
        self.on_hint = on_hint
        self._hints: list[dict[str, Any]] = []
        self.hint_log: list[dict[str, Any]] = []

    # ---------------- recall ----------------
    def recall(self, task: str | None = None, limit: int = 3) -> list[dict[str, Any]]:
        """Skills to put in the agent's prompt. Records which ones were shown, so later feedback is attributed."""
        hits = self._loop.recall(task or self.task, limit=limit)
        for h in hits:
            if h["name"] != "_principles" and h["name"] not in self.skills_used:
                self.skills_used.append(h["name"])
        return hits

    def skills_text(self, task: str | None = None, limit: int = 3) -> str:
        """Same as recall(), already joined into prompt-ready text."""
        return "\n\n".join(h["skill_md"] for h in self.recall(task, limit))

    # ---------------- recording ----------------
    def say(self, content: str) -> "Session":
        self.steps.append(Step("assistant", str(content)))
        return self

    def tool(self, name: str, args: dict | None = None, result: Any = None, error: Any = None,
             thought: str = "") -> "Session":
        """Record one tool call. Pass `error` when it failed - failures are the most useful traces.

        A failed call also triggers error-time recall: matching skills are queued for the agent's next step
        (read them with hints() / hints_text(), or receive them via on_hint)."""
        self.steps.append(Step("assistant", thought, [ToolCall(name, args or {},
                                                               result=None if result is None else str(result),
                                                               error=None if error is None else str(error))]))
        if error is not None and str(error).strip() and self.error_hints:
            self._recall_on_error(str(error))
        return self

    def _recall_on_error(self, error: str) -> None:
        try:
            shown = set(self.skills_used) | {h["name"] for h in self._hints}
            hits = self._loop.recall_for_error(error, task=self.task, exclude=tuple(shown))
        except Exception:
            return            # recall must never break the caller's task
        if not hits:
            return
        for h in hits:
            self.skills_used.append(h["name"])         # attribute the outcome to the skill that was shown
        self._hints.extend(hits)
        self.hint_log.append({"step": len(self.steps) - 1, "skills": [h["name"] for h in hits]})
        if self.on_hint:
            try:
                self.on_hint(hits)
            except Exception:
                pass

    def hints(self) -> list[dict[str, Any]]:
        """Skills surfaced by errors since the last call. Put them in the agent's context before its next step."""
        out, self._hints = self._hints, []
        return out

    def hints_text(self) -> str:
        """hints(), joined into prompt-ready text ('' when there is nothing new)."""
        hs = self.hints()
        if not hs:
            return ""
        return ("A past failure matching this error was turned into a verified procedure. Follow it:\n\n"
                + "\n\n".join(h["skill_md"] for h in hs))

    def observe(self, content: str) -> "Session":
        self.steps.append(Step("tool", str(content)))
        return self

    def used(self, *bullet_ids: str) -> "Session":
        """Bullets that helped. Drives per-bullet credit assignment."""
        self.helpful.extend(bullet_ids)
        return self

    def misled(self, *bullet_ids: str) -> "Session":
        """Bullets that were wrong or sent the agent the wrong way."""
        self.harmful.extend(bullet_ids)
        return self

    # ---------------- outcome ----------------
    def ok(self, detail: str = "", confidence: float = 0.9) -> "Session":
        self._outcome, self._detail, self._confidence = "success", detail, confidence
        return self

    def fail(self, detail: str = "", confidence: float = 0.9) -> "Session":
        self._outcome, self._detail, self._confidence = "failure", detail, confidence
        return self

    def partial(self, detail: str = "", confidence: float = 0.8) -> "Session":
        self._outcome, self._detail, self._confidence = "partial", detail, confidence
        return self

    def verified_by(self, check, detail: str = "") -> "Session":
        """Outcome from an independent check - far stronger evidence than the agent's own opinion.

        Accepts BOTH conventions and does not confuse them:
          - a bool: True = passed, False = failed        (e.g. s.verified_by(actual == expected))
          - an int exit code: 0 = passed, nonzero = failed (e.g. s.verified_by(proc.returncode))
        This distinction matters because in Python `False == 0`, so a naive `== 0` test would record the
        exact OPPOSITE for a boolean, teaching the library from tasks that actually succeeded and hiding the
        ones that failed.
        """
        if isinstance(check, bool):
            passed = check
        else:
            passed = (check == 0)
        return self.ok(detail or "checker passed") if passed else self.fail(detail or "checker failed")

    # ---------------- lifecycle ----------------
    def build(self) -> Trace:
        signals = []
        if self._outcome:
            signals.append(Signal("session", self._outcome, self._confidence, self._detail[:300]))
        meta = dict(self.metadata)
        if self.hint_log:
            meta["error_hints"] = self.hint_log
        self.trace = Trace(task=self.task, agent=self.agent, steps=self.steps, signals=signals,
                           skills_used=list(self.skills_used), metadata=meta,
                           bullets_helpful=list(self.helpful), bullets_harmful=list(self.harmful))
        return self.trace

    def submit(self) -> dict:
        self.result = self._loop.learn(self.build())
        return self.result

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None and self._outcome is None:
            # a crash IS the outcome, and the exception is the most informative part of the trace
            self.tool("exception", {"type": exc_type.__name__}, error=f"{exc_type.__name__}: {exc}")
            self.fail(f"uncaught {exc_type.__name__}: {str(exc)[:200]}")
        if self.auto_learn:
            try:
                self.submit()
            except Exception:
                pass          # recording must never break the caller's task
        return False          # never swallow the caller's exception


def record_tool(session: Session, name: str | None = None) -> Callable:
    """Decorator: a tool function records itself, including exceptions.

        @record_tool(s)
        def run_sql(query): ...
    """
    def deco(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*a, **kw):
            args = {"args": [_short(x) for x in a], **{k: _short(v) for k, v in kw.items()}}
            try:
                out = fn(*a, **kw)
            except Exception as e:
                session.tool(name or fn.__name__, args, error=f"{type(e).__name__}: {e}")
                raise
            session.tool(name or fn.__name__, args, result=_short(out))
            return out
        return wrapper
    return deco


def _short(v: Any, limit: int = 500) -> Any:
    if isinstance(v, (str, int, float, bool)) or v is None:
        s = str(v)
    else:
        try:
            s = json.dumps(v, default=str)
        except Exception:
            s = str(v)
    return s if len(s) <= limit else s[:limit] + "…"