"""E5 agent harness: NATIVE tool-calling.

Why this exists (CALIBRATION.md, 2026-09-11 "E5 attempted: INVALID"): the old harness asked the model to
encode actions as JSON inside message content. On multi-line Python files the model emitted native tool calls
instead, the provider rejected them, recovery mangled the payload, and 8/8 "failures" were harness errors.

This harness uses the provider's tool-calling API and reads `tool_calls` off the response, so writing a
multi-line file is a first-class action with no escaping layer. Both A/B arms use this exact code path; the
ONLY difference between arms is whether a skill block is appended to the user message.

Honesty rules:
- The agent's own "done" never counts. An independent checker decides pass/fail after the agent stops.
- Any API/harness problem is recorded as harness_error=True and must NEVER be counted as a task failure.
  Callers re-run or drop such runs; they do not score them.
- Every run's full transcript (messages, tool calls, tool results, checker output) is saved to disk so it can
  be read before any number is trusted.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from skillloop import Signal, Step, ToolCall, Trace  # noqa: E402

AGENT_MODEL = os.getenv("E5_AGENT_MODEL", "openai/gpt-4o-mini")
TEMPERATURE = 0.7          # fixed before any results were seen
MAX_TURNS = 10
SCRATCH = Path(os.getenv("E5_SCRATCH", "/home/claude/qa/e5_scratch"))
# Grading is serialized: this box has 1 CPU, and EvalPlus time limits scale from reference runtimes, so
# concurrent grading could turn correct code into TIMEOUT failures (a fake signal). Agents still run in parallel.
CHECK_LOCK = threading.Lock()

SYSTEM = """You are a software engineering agent working in a Linux sandbox with Python 3.
Solve the task using the tools. write_file creates or overwrites a file with exact contents. run executes a
shell command in the working directory and returns its real output. Test your code before finishing.
When the task is complete, call finish. Relative paths only."""

TOOLS = [
    {"type": "function", "function": {
        "name": "write_file", "description": "Create or overwrite a file with the given full contents.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "run", "description": "Run a shell command in the working directory. Returns exit code and output.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "finish", "description": "Declare the task complete.",
        "parameters": {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}}},
]


def sh(cmd: str, cwd: Path, timeout: int = 30) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, shell=True, cwd=str(cwd), capture_output=True, text=True, timeout=timeout)
        out = (p.stdout or "") + (("\n" + p.stderr) if p.stderr else "")
        return p.returncode, out.strip()[:2000]
    except subprocess.TimeoutExpired:
        return 124, "TIMEOUT"


def write_file(workdir: Path, path: str, content: str) -> tuple[bool, str]:
    target = (workdir / path).resolve()
    root = workdir.resolve()
    if root != target and root not in target.parents:
        return False, "error: path escapes the working directory"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return True, f"wrote {path} ({len(content)} bytes)"


def execute(workdir: Path, name: str, args: dict) -> tuple[bool, str]:
    """The single tool executor. verify_checkers drives this same function with reference solutions."""
    if name == "write_file":
        return write_file(workdir, str(args.get("path", "")), str(args.get("content", "")))
    if name == "run":
        code, out = sh(str(args.get("command", "")), workdir)
        return code == 0, f"(exit {code})\n{out}"
    if name == "finish":
        return True, "ok"
    return False, f"error: unknown tool {name}"


def setup(task: dict, tag: str) -> Path:
    wd = SCRATCH / f"{task['id']}__{tag}"
    if wd.exists():
        shutil.rmtree(wd)
    wd.mkdir(parents=True)
    return wd


def client():
    from openai import OpenAI
    return OpenAI(base_url=os.environ["OPENAI_BASE_URL"], api_key=os.environ["OPENAI_API_KEY"],
                  timeout=90, max_retries=0)


def _call(cl, messages, attempts: int = 5):
    last = None
    for i in range(attempts):
        try:
            return cl.chat.completions.create(model=AGENT_MODEL, messages=messages, tools=TOOLS,
                                              tool_choice="auto", temperature=TEMPERATURE, max_tokens=1500)
        except Exception as e:  # 429 / 402 in-flight / 5xx / network: back off, never score as a failure
            last = e
            # 402 in-flight-budget: the account has no credit headroom for parallel calls. Waiting for other
            # requests to settle is the documented remedy, so back off harder than for a transient 5xx.
            slow = "402" in str(e) or "in_flight" in str(e) or "429" in str(e)
            time.sleep(min(90, (8 if slow else 2) ** min(i, 2) + 1))
    raise last


def run_task(task: dict, tag: str, skills_text: str = "", cl=None, log_dir: Path | None = None) -> dict:
    """Run one task. Returns {passed, harness_error, trace, transcript_path, ...}."""
    cl = cl or client()
    wd = setup(task, tag)
    user = f"TASK: {task['prompt']}"
    if skills_text:
        user += f"\n\nRELEVANT SKILLS FROM PAST EXPERIENCE (follow them):\n{skills_text}"
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    steps = [Step("user", task["prompt"])]
    harness_error, finished, n_tool_calls, usage = None, False, 0, {"prompt": 0, "completion": 0}

    for _turn in range(MAX_TURNS):
        try:
            r = _call(cl, messages)
        except Exception as e:
            harness_error = f"api: {type(e).__name__}: {str(e)[:300]}"
            break
        if not getattr(r, "choices", None):
            harness_error = f"api: empty choices: {str(r)[:300]}"
            break
        if r.usage:
            usage["prompt"] += r.usage.prompt_tokens or 0
            usage["completion"] += r.usage.completion_tokens or 0
        msg = r.choices[0].message
        calls = msg.tool_calls or []
        am = {"role": "assistant", "content": msg.content or ""}
        if calls:
            am["tool_calls"] = [{"id": c.id, "type": "function",
                                 "function": {"name": c.function.name, "arguments": c.function.arguments}}
                                for c in calls]
        messages.append(am)
        if not calls:                      # plain text reply = agent stopped
            steps.append(Step("assistant", (msg.content or "")[:400]))
            break
        tcs = []
        for c in calls:
            n_tool_calls += 1
            try:
                args = json.loads(c.function.arguments or "{}")
            except json.JSONDecodeError as e:
                ok, out = False, f"error: arguments were not valid JSON: {e}"
                args = {"_raw": (c.function.arguments or "")[:500]}
            else:
                ok, out = execute(wd, c.function.name, args)
            if c.function.name == "finish":
                finished = True
            logged = {k: (v if k != "content" else v[:1500]) for k, v in args.items()}
            tcs.append(ToolCall(c.function.name, logged, result=out if ok else None, error=None if ok else out))
            messages.append({"role": "tool", "tool_call_id": c.id, "content": out[:1500]})
        steps.append(Step("assistant", (msg.content or "")[:200], tcs))
        if finished:
            break

    ct = task.get("checker_timeout", 30)
    with CHECK_LOCK:
        ccode, cout = sh(task["checker"], wd, timeout=ct)
        rechecked = False
        if ccode in (3, 124):             # TIMEOUT: re-check once, serialized, before believing it
            ccode, cout = sh(task["checker"], wd, timeout=ct)
            rechecked = True
    status = "pass" if ccode == 0 else ("timeout" if ccode in (3, 124) else "fail")
    passed = (ccode == 0) and harness_error is None
    steps.append(Step("system", f"[independent checker] {'PASS' if ccode == 0 else 'FAIL'}: {cout[:300]}"))
    trace = Trace(task=task["prompt"], agent="e5-native-agent", steps=steps,
                  signals=[Signal("checker", "success" if ccode == 0 else "failure", 1.0, cout[:200])])
    files = {p.name: p.read_text()[:3000] for p in wd.iterdir() if p.is_file() and p.suffix == ".py"}
    rec = {"task": task["id"], "tag": tag, "passed": passed, "checker_exit": ccode, "checker_out": cout[:600],
           "checker_status": status, "rechecked": rechecked,
           "harness_error": harness_error, "finished": finished, "tool_calls": n_tool_calls, "usage": usage,
           "skills_injected": bool(skills_text), "files": files, "messages": messages}
    tp = None
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        tp = log_dir / f"{task['id']}__{tag}.json"
        tp.write_text(json.dumps(rec, indent=1, default=str))
    return {**rec, "trace": trace, "transcript_path": str(tp) if tp else None}
