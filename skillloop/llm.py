"""Minimal LLM client. Anthropic native or any OpenAI-compatible endpoint (Groq, OpenRouter, Ollama, vLLM).

Config via env:
  SKILLLOOP_PROVIDER = anthropic | openai | fake     (default: auto-detect from keys)
  SKILLLOOP_MODEL    = default model id for every role
  SKILLLOOP_MODEL_REFLECT / _SYNTH / _JUDGE / _OUTCOME / _INJECT = per-role override
  SKILLLOOP_EMBED_MODEL = embedding model on the OpenAI-compatible /embeddings endpoint (optional)
  ANTHROPIC_API_KEY / OPENAI_API_KEY
  OPENAI_API_KEYS    = comma-separated pool; on 429 the client rotates to the next key before backing off.
                       (Note: Groq rate limits are per organization; keys from the same org share one quota.)
  OPENAI_BASE_URL    = e.g. https://api.groq.com/openai/v1

Roles: reflect (post-mortem), synth (write skill), judge (gate reviewer), outcome (did the task succeed),
inject (prompt-injection check). Cheap roles can use a small fast model; reflect/synth want the strongest.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any

ROLES = ("reflect", "synth", "judge", "outcome", "inject")

# --- manual provider -------------------------------------------------------
# Runs the learning pipeline with NO API key: each call is written to a queue file, and answers are read back
# from a companion file. Anything can fill them in - a human, a notebook, or a stronger model in another
# session. Useful when the provider you have is rate-limited, or when you want a person to review what the
# pipeline is being asked before it writes to the library.
MANUAL_DIR = "SKILLLOOP_MANUAL_DIR"


# Minimum keys each role's answer must contain. A wrong-shaped answer used to surface much later as
# AttributeError('NoneType' object has no attribute 'get'), with no indication of which file was at fault.
_MANUAL_REQUIRED = {
    "reflect": ("root_cause", "rule", "generalizable", "confidence"),
    "synth": ("bullets", "name"),
    "judge": (),          # several distinct judge prompts; checked structurally below
    "outcome": ("outcome",),
    "inject": ("safe",),
}


def _validate_manual(text: str, role: str | None) -> str | None:
    """Return a human-readable problem with a manual answer file, or None if it is usable."""
    try:
        data = json.loads(text)
    except Exception as e:
        return f"not valid JSON ({e})"
    if data is None:
        return "file contains null - write the answer object, not null"
    if not isinstance(data, dict):
        return f"expected a JSON object, got {type(data).__name__}"
    missing = [k for k in _MANUAL_REQUIRED.get(role or "", ()) if k not in data]
    if missing:
        return f"role '{role}' answer is missing required key(s): {', '.join(missing)}"
    if role == "synth" and not data.get("operations") and not data.get("bullets"):
        return "synth answer needs either 'bullets' (create) or 'operations' (patch)"
    if role == "judge" and not any(k in data for k in ("support", "overall", "assertions", "safe")):
        return "judge answer should contain one of: support (skeptic), overall/assertions (gate), safe (security)"
    return None


class BadManualAnswer(ValueError):
    """A manual answer file exists but cannot be used."""


class PendingManualCall(RuntimeError):
    """Raised by the manual provider when a request has been queued but not yet answered."""

    def __init__(self, key: str, role: str, path: str):
        super().__init__(f"manual call {key} ({role}) awaiting an answer: {path}")
        self.key, self.role, self.path = key, role, path


class LLM:
    def __init__(self, provider: str | None = None, model: str | None = None, models: dict[str, str] | None = None):
        provider = provider or os.getenv("SKILLLOOP_PROVIDER") or (
            "anthropic" if os.getenv("ANTHROPIC_API_KEY") else "openai"
        )
        self.provider = provider
        self.calls: list[dict[str, Any]] = []      # per-call telemetry (role, model, ms, ok)
        if provider == "anthropic":
            try:
                import anthropic
            except ImportError as e:
                raise ImportError("SkillLoop needs the anthropic SDK for this provider: pip install 'skillloop[anthropic]'") from e
            self.client = anthropic.Anthropic()
            default = model or os.getenv("SKILLLOOP_MODEL", "claude-sonnet-4-5")
        elif provider == "openai":
            try:
                import openai
            except ImportError as e:
                raise ImportError("SkillLoop needs the openai SDK for this provider: pip install 'skillloop[openai]'") from e
            pool = [k.strip() for k in os.getenv("OPENAI_API_KEYS", "").split(",") if k.strip()]
            single = os.getenv("OPENAI_API_KEY")
            if single and single not in pool:
                pool.insert(0, single)
            self._keys = pool or [None]          # None -> let the SDK resolve (env / default)
            # Several clients (agent + learner, parallel workers) sharing one key pool would all hammer key #0
            # and only spread out after each hits its rate limit. Starting at a random offset avoids that, but
            # it makes the starting key unpredictable, so it is opt-in: set SKILLLOOP_SPREAD_KEYS=1 when you
            # actually run several clients against one pool. The default start is deterministic (key 0).
            if os.getenv("SKILLLOOP_SPREAD_KEYS", "").lower() in ("1", "true", "yes"):
                import random as _random
                self._key_idx = _random.randrange(len(self._keys))
            else:
                self._key_idx = 0
            self._key_cooldown: dict[int, float] = {}   # key index -> unix time it becomes usable again
            self._base_url = os.getenv("OPENAI_BASE_URL") or None
            self.client = openai.OpenAI(api_key=self._keys[self._key_idx], base_url=self._base_url, timeout=120.0,
                                        max_retries=1)
            default = model or os.getenv("SKILLLOOP_MODEL", "gpt-4.1-mini")
        elif provider == "manual":
            import pathlib
            self.client = None
            self.manual_dir = pathlib.Path(os.getenv(MANUAL_DIR, ".skillloop-manual"))
            self.manual_dir.mkdir(parents=True, exist_ok=True)
            default = "manual"
        elif provider == "fake":
            self.client = None
            default = "fake"
        else:
            raise ValueError(f"unknown provider {provider}")
        self.model = default
        self.models = {r: (models or {}).get(r) or os.getenv(f"SKILLLOOP_MODEL_{r.upper()}") or default for r in ROLES}
        self.embed_model = os.getenv("SKILLLOOP_EMBED_MODEL")

    def model_for(self, role: str | None) -> str:
        return self.models.get(role or "", self.model)

    max_retries = 6   # on 429 / 5xx, exponential backoff honoring Retry-After when present

    def complete(self, system: str, user: str, max_tokens: int = 2000, temperature: float = 0.2,
                 role: str | None = None) -> str:
        for attempt in range(self.max_retries + 1):
            try:
                return self._complete(system, user, max_tokens, temperature, role)
            except Exception as e:
                wait, exact = _retry_after(e)
                if wait is None or attempt == self.max_retries:
                    raise
                if self._rotate_key(cooldown=wait):
                    continue                       # another key is available: retry immediately
                # when the provider states the exact wait (Groq does: "try again in 1.77s"), honour it and add
                # only a little jitter. Inflating an exact wait exponentially just burns wall-clock time.
                delay = wait + 0.25 if exact else min(60.0, max(wait, 1.0) * (1.5 ** attempt))
                time.sleep(min(60.0, delay))
        raise RuntimeError("unreachable")

    def _rotate_key(self, cooldown: float) -> bool:
        """Mark the current key as cooling down and switch to the next usable one. False if none available."""
        if self.provider != "openai" or len(getattr(self, "_keys", [])) < 2:
            return False
        import openai
        now = time.time()
        self._key_cooldown[self._key_idx] = now + max(cooldown, 1.0)
        for step in range(1, len(self._keys)):
            j = (self._key_idx + step) % len(self._keys)
            if self._key_cooldown.get(j, 0) <= now:
                self._key_idx = j
                self.client = openai.OpenAI(api_key=self._keys[j], base_url=self._base_url, timeout=120.0, max_retries=1)
                self.calls.append({"role": "_rotate", "model": f"key#{j}", "ms": 0, "ok": True})
                return True
        return False

    @property
    def active_key_index(self) -> int:
        return getattr(self, "_key_idx", 0)

    def _complete(self, system: str, user: str, max_tokens: int, temperature: float, role: str | None) -> str:
        model = self.model_for(role)
        t0 = time.time()
        ok = True
        try:
            if self.provider == "anthropic":
                r = self.client.messages.create(
                    model=model, max_tokens=max_tokens, temperature=temperature,
                    system=system, messages=[{"role": "user", "content": user}],
                )
                return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
            if self.provider == "openai":
                r = self.client.chat.completions.create(
                    model=model, max_tokens=max_tokens, temperature=temperature,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                )
                return r.choices[0].message.content or ""
            if self.provider == "manual":
                return self._manual(system, user, role)
            if self.provider == "fake":
                return self._fake(system, user)
            raise RuntimeError(self.provider)
        except Exception:
            ok = False
            raise
        finally:
            self.calls.append({"role": role, "model": model, "ms": int((time.time() - t0) * 1000), "ok": ok})

    def json(self, system: str, user: str, max_tokens: int = 2000, role: str | None = None,
             retries: int = 1) -> dict[str, Any]:
        sys_ = system + "\n\nRespond with a single JSON object and nothing else. No markdown fences, no prose."
        last: Exception | None = None
        for attempt in range(retries + 1):
            text = self.complete(sys_, user, max_tokens, role=role, temperature=0.2 if attempt == 0 else 0.0)
            try:
                return extract_json(text)
            except ValueError as e:
                last = e
        raise last  # type: ignore[misc]

    # ---- embeddings (optional; OpenAI-compatible only) ----
    def embed(self, texts: list[str]) -> list[list[float]] | None:
        if self.provider != "openai" or not self.embed_model or not texts:
            return None
        try:
            r = self.client.embeddings.create(model=self.embed_model, input=texts)
            return [d.embedding for d in r.data]
        except Exception:
            return None

    def _manual(self, system: str, user: str, role: str | None) -> str:
        """Write the request out; read the answer back. Raises PendingManualCall if not answered yet, so
        `process()` can be run repeatedly and will pick up whatever has been filled in."""
        import hashlib
        key = hashlib.sha1(f"{role}|{system[:200]}|{user}".encode()).hexdigest()[:12]
        ans = self.manual_dir / f"{key}.answer.json"
        if ans.exists():
            text = ans.read_text(encoding="utf-8")
            problem = _validate_manual(text, role)
            if problem:
                # fail loudly at the file, not four steps later inside the pipeline with an AttributeError
                raise BadManualAnswer(f"{ans}: {problem}")
            return text
        req = self.manual_dir / f"{key}.request.json"
        if not req.exists():
            req.write_text(json.dumps({"id": key, "role": role, "system": system, "user": user}, indent=1),
                           encoding="utf-8")
        raise PendingManualCall(key, role or "?", str(req))

    # a deterministic stand-in so tests run without keys
    _fake_handler = None

    def _fake(self, system: str, user: str) -> str:
        if LLM._fake_handler:
            return LLM._fake_handler(system, user)
        return "{}"


def _retry_after(e: Exception) -> tuple[float | None, bool]:
    """(seconds to wait, provider_stated_it_exactly). (None, False) if the error is not retryable."""
    status = getattr(e, "status_code", None)
    name = type(e).__name__
    if status not in (408, 409, 429, 500, 502, 503, 529) and "RateLimit" not in name and "Overloaded" not in name \
            and "APIConnection" not in name and "Timeout" not in name:
        return None, False
    msg = str(e)
    m = re.search(r"try again in (?:([0-9.]+)m)?([0-9.]+)(ms|s)", msg)
    if m:
        mins = float(m.group(1)) if m.group(1) else 0.0
        v = float(m.group(2))
        secs = v / 1000 if m.group(3) == "ms" else v
        return mins * 60 + secs, True
    hdrs = getattr(getattr(e, "response", None), "headers", None) or {}
    try:
        return float(hdrs["retry-after"]), True
    except Exception:
        return 2.0, False


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    # strip <think>...</think> blocks some open models emit
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    raise ValueError(f"model did not return JSON: {text[:300]}")
