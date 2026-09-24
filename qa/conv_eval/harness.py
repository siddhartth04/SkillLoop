"""Execution and agent episodes for the conventions eval.

Guards against every harness artifact seen in earlier attempts:
  - markdown fences are stripped before execution (and several blocks are handled)
  - output is never truncated silently: an unclosed fence is a HARNESS error, not an agent failure
  - rate limits / 5xx are retried by skillloop.llm.LLM; a call that still fails is a HARNESS error
  - HARNESS errors are excluded from scoring and reported separately, never counted as agent failures
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field

from .generator import Suite, Task

SYSTEM = ("You are a Python programmer. Reply with a single ```python code block and nothing else. "
          "The code runs as a script; put the final answer in a variable named `result` when the task asks "
          "for a value.")
WRONG = "Your code ran, but the result is incorrect."
MAX_TOKENS = 3000


class HarnessError(Exception):
    """Something that is not the agent's fault. Never scored."""


def extract_code(text: str) -> str:
    if text is None:
        raise HarnessError("empty completion")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    if text.count("```") % 2 == 1:
        raise HarnessError("unclosed code fence (truncated output?)")
    blocks = re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```", text, flags=re.S)
    code = "\n".join(b.strip() for b in blocks) if blocks else text
    code = code.strip()
    if not code:
        raise HarnessError("no code in completion")
    return code


RUNNER = r'''
import json, sys, traceback, importlib
sys.path.insert(0, sys.argv[1])
code = open(sys.argv[2], encoding="utf-8").read()
check = open(sys.argv[3], encoding="utf-8").read(); pkg = sys.argv[4]
mod = importlib.import_module(pkg)
ns = {"__name__": "__main__", pkg: mod}
try:
    exec(compile(code, "<agent>", "exec"), ns)
except BaseException:
    tb = traceback.format_exc().splitlines()
    print(json.dumps({"ok": False, "stage": "code", "error": "\n".join(tb[-6:])})); sys.exit(0)
ns[pkg] = mod
try:
    ok = bool(eval(check, ns))
except BaseException as e:
    print(json.dumps({"ok": False, "stage": "check", "error": f"{type(e).__name__}: {e}"})); sys.exit(0)
print(json.dumps({"ok": ok, "stage": "check", "error": "" if ok else "check failed"}))
'''


def execute(suite: Suite, task: Task, code: str, timeout: float = 10.0) -> dict:
    d = tempfile.mkdtemp(prefix="conv-")
    with open(os.path.join(d, f"{suite.pkg}.py"), "w", encoding="utf-8") as f:
        f.write(suite.library)
    for name, body in (("code.txt", code), ("check.txt", task.check), ("runner.py", RUNNER)):
        with open(os.path.join(d, name), "w", encoding="utf-8") as f:
            f.write(body)
    try:
        p = subprocess.run([sys.executable, os.path.join(d, "runner.py"), d, os.path.join(d, "code.txt"),
                            os.path.join(d, "check.txt"), suite.pkg], capture_output=True, text=True,
                           timeout=timeout, cwd=d, encoding="utf-8", errors="replace",
                           env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    except subprocess.TimeoutExpired:
        return {"ok": False, "stage": "code", "error": "TimeoutError: code ran longer than 10s"}
    try:
        return json.loads(p.stdout.strip().splitlines()[-1])
    except Exception:
        raise HarnessError(f"runner produced no verdict: {p.stderr[-300:]}")


def agent_feedback(verdict: dict) -> str:
    """What the AGENT sees after a failed attempt: the traceback if it crashed, else only that it was wrong.
    The hidden check is never shown to the agent at test time."""
    return verdict["error"] if verdict["stage"] == "code" else WRONG


def learning_message(task: Task, verdict: dict) -> str:
    """What a LEARNER sees about a training attempt: like a failing unit test, it includes the assertion.
    SkillLoop and the simple-memory baseline receive exactly the same messages."""
    if verdict["ok"]:
        return "tests passed"
    if verdict["stage"] == "code":
        return verdict["error"]
    return f"AssertionError: test failed: assert {task.check}"


@dataclass
class Episode:
    task_id: str
    condition: str
    attempts: list[dict] = field(default_factory=list)   # {"code","verdict"}
    harness_error: str = ""
    hints_shown: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return bool(self.attempts) and self.attempts[-1]["verdict"]["ok"]

    @property
    def first_try(self) -> bool:
        return bool(self.attempts) and self.attempts[0]["verdict"]["ok"]


def prompt_for(suite: Suite, task: Task, context: str, feedback: str = "", prev_code: str = "") -> str:
    parts = [suite.api_doc]
    if context.strip():
        parts.append(context.strip())
    parts.append(f"Task: {task.prompt}")
    if prev_code:
        parts.append(f"Your previous attempt:\n```python\n{prev_code}\n```\nIt failed with:\n{feedback}\n"
                     "Write a corrected version.")
    return "\n\n".join(parts)


def run_episode(agent, suite: Suite, task: Task, condition: str, context: str = "",
                on_fail=None, max_attempts: int = 2) -> Episode:
    """One task, up to `max_attempts` tries. `on_fail(code, feedback) -> extra context` lets a condition add
    something after a failure (SkillLoop's error-time hints). Identical for every condition otherwise."""
    ep = Episode(task.id, condition)
    code, feedback, extra = "", "", ""
    for _ in range(max_attempts):
        user = prompt_for(suite, task, context + ("\n\n" + extra if extra else ""), feedback, code)
        try:
            raw = agent.complete(SYSTEM, user, max_tokens=MAX_TOKENS, temperature=0.0, role="agent")
            code = extract_code(raw)
            verdict = execute(suite, task, code)
        except HarnessError as e:
            ep.harness_error = str(e)
            return ep
        except Exception as e:                           # provider failed even after retries
            ep.harness_error = f"agent call failed: {e!r}"[:300]
            return ep
        ep.attempts.append({"code": code, "verdict": verdict})
        if verdict["ok"]:
            break
        feedback = agent_feedback(verdict)
        if on_fail:
            extra = on_fail(code, feedback) or ""
    return ep
