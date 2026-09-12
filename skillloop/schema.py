"""Canonical trajectory schema + normalizers.

Every agent logs differently. We accept the common shapes and normalize them
once into `Trace`. Everything downstream (reflection, synthesis, evals) only
ever sees `Trace`.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Literal

Outcome = Literal["success", "failure", "partial", "unknown"]


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any] = field(default_factory=dict)
    result: str | None = None
    error: str | None = None
    duration_ms: int | None = None


@dataclass
class Step:
    role: str                         # "user" | "assistant" | "tool" | "system"
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass
class Signal:
    """One piece of evidence about the outcome. Several may be attached."""
    source: str                       # "user" | "test" | "tool_errors" | "self" | "judge" | "eval"
    outcome: Outcome
    confidence: float = 0.5           # 0..1
    detail: str = ""


@dataclass
class Trace:
    task: str
    steps: list[Step]
    signals: list[Signal] = field(default_factory=list)
    agent: str = "unknown"            # which agent produced it
    skills_used: list[str] = field(default_factory=list)
    bullets_helpful: list[str] = field(default_factory=list)   # bullet ids the agent says helped
    bullets_harmful: list[str] = field(default_factory=list)   # ...and ones that misled it
    eval_id: str | None = None        # set when this run is a host-run eval
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    ts: float = field(default_factory=time.time)

    # ---- derived ----
    @property
    def outcome(self) -> Outcome:
        """Weighted vote over signals. Tool-error heuristics are added if no signal."""
        sigs = list(self.signals) or [self.heuristic_signal()]
        score = {"success": 0.0, "failure": 0.0, "partial": 0.0}
        for s in sigs:
            if s.outcome in score:
                score[s.outcome] += s.confidence
        if not any(score.values()):
            return "unknown"
        return max(score, key=score.get)  # type: ignore[return-value]

    @property
    def confidence(self) -> float:
        sigs = list(self.signals) or [self.heuristic_signal()]
        return min(1.0, sum(s.confidence for s in sigs if s.outcome == self.outcome))

    def heuristic_signal(self) -> Signal:
        errs = sum(1 for st in self.steps for tc in st.tool_calls if tc.error)
        calls = sum(len(st.tool_calls) for st in self.steps)
        if calls == 0:
            return Signal("tool_errors", "unknown", 0.0, "no tool calls")
        ratio = errs / calls
        if ratio > 0.4:
            return Signal("tool_errors", "failure", 0.4, f"{errs}/{calls} tool calls errored")
        if errs == 0:
            return Signal("tool_errors", "success", 0.25, f"0/{calls} tool calls errored")
        return Signal("tool_errors", "partial", 0.3, f"{errs}/{calls} tool calls errored")

    @property
    def n_tool_calls(self) -> int:
        return sum(len(st.tool_calls) for st in self.steps)

    @property
    def errors(self) -> list[ToolCall]:
        return [tc for st in self.steps for tc in st.tool_calls if tc.error]

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> "Trace":
        return from_dict(json.loads(s))

    def compact(self, max_chars: int = 12000) -> str:
        """Human/LLM readable rendering, truncated from the middle."""
        lines = [f"TASK: {self.task}", f"AGENT: {self.agent}", f"SKILLS USED: {self.skills_used or 'none'}", ""]
        for i, st in enumerate(self.steps):
            if st.content:
                lines.append(f"[{i}] {st.role.upper()}: {st.content[:800]}")
            for tc in st.tool_calls:
                a = json.dumps(tc.args, ensure_ascii=False)[:400]
                lines.append(f"[{i}] TOOL {tc.name}({a})")
                if tc.error:
                    lines.append(f"      ERROR: {tc.error[:600]}")
                elif tc.result is not None:
                    lines.append(f"      -> {str(tc.result)[:400]}")
        lines.append("")
        lines.append("SIGNALS: " + "; ".join(f"{s.source}={s.outcome}({s.confidence:.2f}) {s.detail}" for s in (self.signals or [self.heuristic_signal()])))
        lines.append(f"OUTCOME: {self.outcome} (conf {self.confidence:.2f})")
        text = "\n".join(lines)
        if len(text) > max_chars:
            half = max_chars // 2
            text = text[:half] + "\n...[truncated]...\n" + text[-half:]
        return text


# --------------------------------------------------------------------------
# Normalizers
# --------------------------------------------------------------------------

def from_dict(d: dict[str, Any]) -> Trace:
    """Canonical dict -> Trace."""
    steps = [
        Step(role=s.get("role", "assistant"), content=s.get("content", "") or "",
             tool_calls=[ToolCall(**tc) for tc in s.get("tool_calls", [])])
        for s in d.get("steps", [])
    ]
    signals = [Signal(**s) for s in d.get("signals", [])]
    return Trace(
        task=d["task"], steps=steps, signals=signals,
        agent=d.get("agent", "unknown"), skills_used=d.get("skills_used", []),
        eval_id=d.get("eval_id"), metadata=d.get("metadata", {}),
        id=d.get("id", uuid.uuid4().hex[:12]), ts=d.get("ts", time.time()),
    )


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                out.append(block.get("text", ""))
            elif isinstance(block, str):
                out.append(block)
        return "\n".join(out)
    return "" if content is None else str(content)


def from_openai_messages(task: str, messages: list[dict], **kw) -> Trace:
    """OpenAI chat-completions style: assistant.tool_calls + role=tool results."""
    steps: list[Step] = []
    pending: dict[str, ToolCall] = {}
    for m in messages:
        role = m.get("role", "assistant")
        if role == "assistant":
            st = Step(role="assistant", content=_content_to_text(m.get("content")))
            for tc in m.get("tool_calls", []) or []:
                fn = tc.get("function", {})
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except Exception:
                    args = {"raw": fn.get("arguments")}
                call = ToolCall(name=fn.get("name", "?"), args=args)
                pending[tc.get("id", str(len(pending)))] = call
                st.tool_calls.append(call)
            steps.append(st)
        elif role == "tool":
            call = pending.get(m.get("tool_call_id", ""))
            text = _content_to_text(m.get("content"))
            if call is not None:
                if _looks_like_error(text) or m.get("is_error"):
                    call.error = text
                else:
                    call.result = text
            else:
                steps.append(Step(role="tool", content=text))
        else:
            steps.append(Step(role=role, content=_content_to_text(m.get("content"))))
    return Trace(task=task, steps=steps, **kw)


def from_anthropic_messages(task: str, messages: list[dict], **kw) -> Trace:
    """Anthropic messages style: assistant tool_use blocks + user tool_result blocks."""
    steps: list[Step] = []
    pending: dict[str, ToolCall] = {}
    for m in messages:
        role = m.get("role", "assistant")
        content = m.get("content")
        if role == "assistant":
            st = Step(role="assistant", content=_content_to_text(content))
            if isinstance(content, list):
                for b in content:
                    if isinstance(b, dict) and b.get("type") == "tool_use":
                        call = ToolCall(name=b.get("name", "?"), args=b.get("input", {}) or {})
                        pending[b.get("id", str(len(pending)))] = call
                        st.tool_calls.append(call)
            steps.append(st)
        elif role == "user" and isinstance(content, list) and any(
            isinstance(b, dict) and b.get("type") == "tool_result" for b in content
        ):
            for b in content:
                if not (isinstance(b, dict) and b.get("type") == "tool_result"):
                    continue
                call = pending.get(b.get("tool_use_id", ""))
                text = _content_to_text(b.get("content"))
                if call is None:
                    continue
                if b.get("is_error") or _looks_like_error(text):
                    call.error = text
                else:
                    call.result = text
        else:
            steps.append(Step(role=role, content=_content_to_text(content)))
    return Trace(task=task, steps=steps, **kw)


_ERR_MARKERS = ("error", "exception", "traceback", "failed", "not found", "permission denied", "no such file")


def _looks_like_error(text: str) -> bool:
    head = (text or "")[:200].lower()
    return any(k in head for k in _ERR_MARKERS)


def normalize(payload: dict[str, Any]) -> Trace:
    """Auto-detect format. Accepts:
    - canonical: {"task", "steps", ...}
    - {"task", "format": "openai"|"anthropic", "messages": [...], ...}
    """
    if not isinstance(payload, dict):
        raise TypeError(f"trace must be a dict or Trace, got {type(payload).__name__}")
    if "task" not in payload:
        raise ValueError("trace needs a 'task' field describing what the agent was asked to do")
    if "steps" not in payload and "messages" not in payload:
        raise ValueError("trace needs either 'steps' (canonical) or 'messages' (openai/anthropic format)")
    fmt = payload.get("format")
    common = {k: payload[k] for k in ("agent", "skills_used", "eval_id", "metadata",
                                      "bullets_helpful", "bullets_harmful") if k in payload}
    if "signals" in payload:
        common["signals"] = [Signal(**s) for s in payload["signals"]]
    if fmt == "openai":
        return from_openai_messages(payload["task"], payload["messages"], **common)
    if fmt == "anthropic":
        return from_anthropic_messages(payload["task"], payload["messages"], **common)
    if "messages" in payload and "steps" not in payload:
        # guess
        msgs = payload["messages"]
        if any(m.get("role") == "tool" for m in msgs):
            return from_openai_messages(payload["task"], msgs, **common)
        return from_anthropic_messages(payload["task"], msgs, **common)
    return from_dict(payload)
