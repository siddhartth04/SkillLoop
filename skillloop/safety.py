"""Safety: traces contain tool output from the internet, so the skill library is a persistent
prompt-injection sink unless we treat it as one.

Three layers, cheapest first:
  1. redact()        - strip secrets from tool args/results BEFORE anything is stored.
  2. cap()           - size limits so one giant trace can't blow up storage or prompts.
  3. scan_injection()- regex heuristics on skill text; findings block promotion until a judge clears it.
Plus an LLM injection judge (gate.py) and an optional human-approval mode (core.py).
"""
from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

from .schema import Trace

# ---------------------------------------------------------------- 1. secrets
_SECRET_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("openai_key", re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}")),
    ("anthropic_key", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{16,}")),
    ("groq_key", re.compile(r"\bgsk_[A-Za-z0-9]{20,}")),
    ("github_token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr|github_pat)_[A-Za-z0-9_]{20,}")),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}")),
    ("stripe_key", re.compile(r"\b(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{16,}")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}")),
    ("private_key_block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S)),
    ("bearer_header", re.compile(r"(?i)(authorization\s*[:=]\s*bearer\s+)[A-Za-z0-9_\-\.=]{8,}")),
    ("url_userinfo", re.compile(r"(?i)(://[^/\s:@]+:)[^@/\s]{3,}(@)")),
    ("kv_secret", re.compile(
        r"(?i)\b((?:api[_\-]?key|apikey|secret|token|password|passwd|pwd|access[_\-]?key|private[_\-]?key|"
        r"client[_\-]?secret|auth)\s*[:=]\s*['\"]?)([^\s'\",;&]{6,})")),
]


def redact_text(text: str) -> tuple[str, int]:
    """Replace secrets with [REDACTED:<kind>]. Returns (text, n_replacements)."""
    if not text:
        return text, 0
    n = 0
    for kind, pat in _SECRET_PATTERNS:
        def _sub(m: re.Match, kind=kind) -> str:
            nonlocal n
            n += 1
            g = m.groups()
            if kind == "bearer_header":
                return f"{g[0]}[REDACTED:{kind}]"
            if kind == "url_userinfo":
                return f"{g[0]}[REDACTED:{kind}]{g[1]}"
            if kind == "kv_secret":
                return f"{g[0]}[REDACTED:{kind}]"
            return f"[REDACTED:{kind}]"
        text = pat.sub(_sub, text)
    return text, n


def _redact_any(v: Any) -> tuple[Any, int]:
    if isinstance(v, str):
        return redact_text(v)
    if isinstance(v, dict):
        n = 0
        out = {}
        for k, x in v.items():
            x2, m = _redact_any(x)
            out[k] = x2
            n += m
        return out, n
    if isinstance(v, list):
        n = 0
        out = []
        for x in v:
            x2, m = _redact_any(x)
            out.append(x2)
            n += m
        return out, n
    return v, 0


# ---------------------------------------------------------------- 2. size caps
class Caps:
    max_steps = 400
    max_content_chars = 4000       # per step content
    max_tool_result_chars = 3000   # per tool result / error
    max_args_chars = 2000          # per tool call args (json)
    max_total_chars = 200_000      # whole trace json


def sanitize_trace(t: Trace, caps: Caps | None = None) -> tuple[Trace, dict[str, int]]:
    """Redact secrets + apply size caps. Returns a new Trace and stats."""
    caps = caps or Caps()
    stats = {"redacted": 0, "truncated": 0, "steps_dropped": 0}
    steps = list(t.steps)
    if len(steps) > caps.max_steps:
        stats["steps_dropped"] = len(steps) - caps.max_steps
        half = caps.max_steps // 2
        steps = steps[:half] + steps[-half:]
    new_steps = []
    for st in steps:
        content, n = redact_text(st.content or "")
        stats["redacted"] += n
        if len(content) > caps.max_content_chars:
            content = content[: caps.max_content_chars] + "…[truncated]"
            stats["truncated"] += 1
        calls = []
        for tc in st.tool_calls:
            args, n = _redact_any(tc.args or {})
            stats["redacted"] += n
            import json as _j
            if len(_j.dumps(args, ensure_ascii=False)) > caps.max_args_chars:
                args = {"_truncated": _j.dumps(args, ensure_ascii=False)[: caps.max_args_chars]}
                stats["truncated"] += 1
            res, err = tc.result, tc.error
            if res is not None:
                res, n = redact_text(str(res))
                stats["redacted"] += n
                if len(res) > caps.max_tool_result_chars:
                    res = res[: caps.max_tool_result_chars] + "…[truncated]"
                    stats["truncated"] += 1
            if err is not None:
                err, n = redact_text(str(err))
                stats["redacted"] += n
                if len(err) > caps.max_tool_result_chars:
                    err = err[: caps.max_tool_result_chars] + "…[truncated]"
                    stats["truncated"] += 1
            calls.append(replace(tc, args=args, result=res, error=err))
        new_steps.append(replace(st, content=content, tool_calls=calls))
    task, n = redact_text(t.task)
    stats["redacted"] += n
    out = replace(t, task=task, steps=new_steps)

    # HARD invariant: the serialized trace must end up <= max_total_chars. The previous single-shot "keep the
    # first/last quarter" pass could still leave a trace far over the cap (a 2.4M-char input came out at 670k)
    # because a quarter of a huge trace is still huge. Reduce repeatedly until the invariant actually holds.
    def _too_big(tr):
        return len(tr.to_json()) > caps.max_total_chars

    guard = 0
    while _too_big(out) and out.steps and guard < 64:
        guard += 1
        steps_now = out.steps
        if len(steps_now) > 2:
            # halve the step count, keeping head and tail (where the failure usually is)
            keep = max(2, len(steps_now) // 2)
            h = keep // 2
            kept = steps_now[:h] + steps_now[-(keep - h):]
            stats["steps_dropped"] += len(steps_now) - len(kept)
            out = replace(out, steps=kept)
        else:
            # down to a couple of steps and still over: the content itself is oversized, so shrink it
            budget = max(200, caps.max_total_chars // max(1, len(steps_now)) // 2)
            shrunk = []
            for st in steps_now:
                c = st.content[:budget] + "…[truncated]" if len(st.content) > budget else st.content
                calls = []
                for tc in st.tool_calls:
                    res = tc.result[:budget] + "…[truncated]" if tc.result and len(tc.result) > budget else tc.result
                    err = tc.error[:budget] + "…[truncated]" if tc.error and len(tc.error) > budget else tc.error
                    import json as _j
                    a = tc.args
                    if len(_j.dumps(a, ensure_ascii=False)) > budget:
                        a = {"_truncated": _j.dumps(a, ensure_ascii=False)[:budget]}
                    calls.append(replace(tc, content=None) if False else replace(tc, args=a, result=res, error=err))
                shrunk.append(replace(st, content=c, tool_calls=calls))
                stats["truncated"] += 1
            out = replace(out, steps=shrunk)
            if not _too_big(out):
                break
            # last resort: drop tool call payloads entirely
            out = replace(out, steps=[replace(st, tool_calls=[replace(tc, result=None, error="[dropped]", args={})
                                                              for tc in st.tool_calls]) for st in out.steps])
            if _too_big(out):
                # absolute floor: truncate the task and keep a single marker step
                out = replace(out, task=out.task[:1000],
                              steps=[Step("system", "[trace exceeded size cap; body dropped by sanitizer]")])
                break

    assert len(out.to_json()) <= caps.max_total_chars, \
        f"sanitizer failed to enforce cap: {len(out.to_json())} > {caps.max_total_chars}"
    stats["final_chars"] = len(out.to_json())
    return out, stats


# ---------------------------------------------------------------- 3. injection scan
_INJECTION_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("override_instructions", re.compile(r"(?i)\b(ignore|disregard|forget)\b.{0,40}\b(previous|prior|above|all|earlier)\b.{0,30}\b(instructions?|rules?|prompts?)")),
    ("role_hijack", re.compile(r"(?i)\byou are now\b|\bnew (system )?instructions?\b|\bsystem prompt\b|\bact as (an? )?(unrestricted|jailbroken|DAN)\b")),
    ("exfiltration", re.compile(r"(?i)\b(send|post|upload|email|exfiltrate|transmit|leak)\b.{0,60}\b(api[_ ]?keys?|secrets?|tokens?|passwords?|credentials?|env(ironment)? vars?|\.env)\b")),
    ("hidden_url_callout", re.compile(r"(?i)\b(curl|wget|fetch|request)\b.{0,40}https?://(?!github\.com|pypi\.org|docs\.|localhost|127\.0\.0\.1)[a-z0-9\-\.]+\.(?:xyz|top|tk|ru|cn|io|dev|app|sh|ngrok\.[a-z]+|pipedream\.net|requestbin|webhook\.site)")),
    ("destructive", re.compile(r"(?i)\brm\s+-rf\s+(/|~|\$HOME|\*)|\bdrop\s+(database|table)\b|\bformat\s+c:|\bgit\s+push\s+--force\b.{0,20}\bmain\b|\bchmod\s+777\s+/")),
    ("disable_safety", re.compile(r"(?i)\b(disable|bypass|skip|turn off)\b.{0,30}\b(safety|security|verification|checks?|sandbox|guardrails?|approval|confirmation)\b")),
    ("secrecy", re.compile(r"(?i)\b(do not|don't|never)\s+(tell|inform|mention|reveal)\b.{0,30}\b(user|human|owner|operator)\b")),
    ("hidden_text", re.compile(r"<!--.*?-->|\u200b|\u200c|\u2060|\ufeff", re.S)),
    ("always_run", re.compile(r"(?i)\b(always|first|before anything)\b.{0,30}\b(run|execute|call|source)\b.{0,40}(\.sh|\.py|curl|wget|base64)")),
]


def scan_injection(text: str) -> list[dict[str, str]]:
    """Cheap regex pass over skill text. Returns findings; empty list means nothing suspicious."""
    findings = []
    for kind, pat in _INJECTION_PATTERNS:
        for m in pat.finditer(text or ""):
            snippet = text[max(0, m.start() - 30): m.end() + 30].replace("\n", " ")
            findings.append({"kind": kind, "match": snippet[:160]})
            if len(findings) >= 20:
                return findings
    return findings
