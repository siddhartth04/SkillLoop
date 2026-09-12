"""QA harness: a REAL agent doing REAL tasks in a sandbox, with SkillLoop attached.

Design choices that keep this honest:
- The agent is a separate LLM (Groq) that emits shell commands. It is NOT me deciding what it does.
- Commands really execute in a scratch dir. Errors are real errors.
- Success is decided by an independent CHECKER command per task, run after the agent stops.
  The agent's own "I'm done" claim never counts as success.
- Phase A runs with SkillLoop recall DISABLED (baseline). Phase B runs variant tasks with recall ENABLED.
- Retrieval is measured separately from outcome: did a skill fire at all, on a differently-worded task?
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/claude/skillloop")
from skillloop import SkillLoop, Signal, Step, ToolCall, Trace  # noqa: E402
from skillloop.llm import LLM  # noqa: E402

SCRATCH = Path("/home/claude/qa/scratch")
AGENT_SYSTEM = """You are a command-line engineering agent working in a Linux sandbox. You solve the task by
issuing one shell command at a time and reading its real output.

You have NO tools and NO function-calling available. Never emit a tool call or function call of any kind.
Output plain JSON text only.

Reply with ONLY a JSON object, no prose, no fences. Either run a command:
{"thought": "<one short sentence>", "command": "<shell command to run>", "done": false}
or write a file directly (USE THIS for any multi-line file - never a heredoc):
{"thought": "...", "write_file": "path/to/file.py", "content": "<full file contents>", "done": false}
When the task is complete, reply {"thought": "...", "command": "", "done": true, "answer": "<what you did>"}

Rules: one action per turn. You are in a scratch directory; relative paths are fine. Do not use interactive
editors (vim/nano). For creating or rewriting a file, ALWAYS use write_file - heredocs get mangled.
Keep commands short. Max 10 turns."""


def sh(cmd: str, cwd: Path, timeout: int = 60) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, shell=True, cwd=str(cwd), capture_output=True, text=True, timeout=timeout)
        out = (p.stdout or "") + (("\n" + p.stderr) if p.stderr else "")
        return p.returncode, out.strip()[:2000]
    except subprocess.TimeoutExpired:
        return 124, "TIMEOUT"


def _recover(e: Exception) -> dict | None:
    """Some Groq models wrap the requested JSON inside a native tool call and the API 400s.
    The payload we asked for is still in `failed_generation` - dig it out."""
    blob = None
    body = getattr(e, "body", None)
    if isinstance(body, dict):
        blob = (body.get("error") or {}).get("failed_generation")
    if blob is None:
        try:
            blob = ((e.response.json().get("error") or {}).get("failed_generation"))
        except Exception:
            blob = None
    if not blob:
        return None
    d = None
    for attempt in (blob, blob.encode().decode("unicode_escape", errors="ignore")):
        try:
            d = json.loads(attempt)
            break
        except Exception:
            m2 = re.search(r"\{.*\}", attempt, re.S)
            if m2:
                try:
                    d = json.loads(m2.group(0))
                    break
                except Exception:
                    continue
    if d is None:
        return None
    if not isinstance(d, dict):
        return None
    if "arguments" in d and isinstance(d["arguments"], dict):
        d = d["arguments"]
    if "cmd" in d and "command" not in d:
        c = d["cmd"]
        # container.exec shape: {"cmd": ["bash", "-lc", "<the actual command>"]}
        d["command"] = c[-1] if isinstance(c, list) and c else str(c)
    if "content" in d and "write_file" not in d and "path" in d:
        d["write_file"] = d["path"]
    return d if ("command" in d or d.get("done")) else None


def run_task(llm: LLM, task: dict, workdir: Path, skills_text: str = "", max_turns: int = 10) -> Trace:
    """Run one task with a real agent. Returns a real Trace of what happened."""
    steps = [Step("user", task["prompt"])]
    convo = [f"TASK: {task['prompt']}"]
    if skills_text:
        convo.append(f"\nRELEVANT SKILLS FROM PAST EXPERIENCE (follow them):\n{skills_text}")
    done = False
    for turn in range(max_turns):
        try:
            raw = llm.json(AGENT_SYSTEM, "\n".join(convo)[-12000:], max_tokens=500, role="reflect")
        except Exception as e:
            raw = _recover(e)
            if raw is None:
                steps.append(Step("assistant", f"[agent error: {e!r}]"))
                break
        thought, cmd, done = str(raw.get("thought", ""))[:200], str(raw.get("command", "")).strip(), bool(raw.get("done"))
        wf = raw.get("write_file")
        if wf and not done:
            target = (workdir / str(wf)).resolve()
            try:
                if workdir.resolve() not in target.parents and target != workdir.resolve():
                    raise ValueError("path escapes the working directory")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(str(raw.get("content", "")))
                res, err = f"wrote {wf} ({len(str(raw.get('content','')))} bytes)", None
            except Exception as e:
                res, err = None, f"{type(e).__name__}: {e}"
            steps.append(Step("assistant", thought, [ToolCall("write_file", {"path": str(wf)}, result=res, error=err)]))
            convo.append(f"$ write_file {wf}\n{res or err}")
            continue
        if done or not cmd:
            steps.append(Step("assistant", (thought + " " + str(raw.get("answer", "")))[:400]))
            done = True
            break
        code, out = sh(cmd, workdir)
        tc = ToolCall("bash", {"cmd": cmd}, result=(out if code == 0 else None), error=(out if code != 0 else None))
        steps.append(Step("assistant", thought, [tc]))
        convo.append(f"$ {cmd}\n(exit {code}) {out[:1200]}")
    # INDEPENDENT verification - the agent does not get a vote
    ccode, cout = sh(task["checker"], workdir)
    passed = ccode == 0
    steps.append(Step("system", f"[independent checker] {'PASS' if passed else 'FAIL'}: {cout[:300]}"))
    return Trace(task=task["prompt"], agent="qa-agent", steps=steps,
                 signals=[Signal("checker", "success" if passed else "failure", 1.0, cout[:200])])


def setup(task: dict) -> Path:
    wd = SCRATCH / task["id"]
    if wd.exists():
        shutil.rmtree(wd)
    wd.mkdir(parents=True)
    for f, content in task.get("files", {}).items():
        p = wd / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    if task.get("setup"):
        sh(task["setup"], wd)
    return wd
