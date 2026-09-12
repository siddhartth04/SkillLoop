"""MCP server. Run: `skillloop mcp` (stdio) — add to any MCP-capable agent.

Tools:
  skill_recall(task)                -> skills to read before doing the task
  skill_learn(trace)                -> submit what happened (canonical / openai / anthropic message format)
  skill_pending_evals()             -> tasks the host should re-run to prove a candidate skill
  skill_process()                   -> run the learning loop now (normally a background worker does this)
  skill_inspect(name?)              -> library state / one skill's history + evals
  skill_rollback(name, version)     -> revert a skill
  skill_set_status(name, status)    -> manual override
"""
from __future__ import annotations

import json
import os
from typing import Any

try:                                    # mcp >= 2.x
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:                     # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server

from .core import SkillLoop

mcp = _Server("skillloop")
_loop: SkillLoop | None = None


def loop() -> SkillLoop:
    global _loop
    if _loop is None:
        _loop = SkillLoop(auto_process=os.getenv("SKILLLOOP_AUTOPROCESS", "1") == "1")
    return _loop


@mcp.tool()
def skill_recall(task: str, limit: int = 3) -> str:
    """Call this BEFORE starting any non-trivial task. Returns previously learned skills (procedures and
    pitfalls) relevant to the task. Read them and follow them. Record the names you used so you can
    pass them as skills_used in skill_learn afterwards."""
    return json.dumps(loop().recall(task, limit=limit), indent=2)


@mcp.tool()
def skill_learn(task: str, messages: list[dict[str, Any]] | None = None, steps: list[dict[str, Any]] | None = None,
                format: str = "auto", outcome: str | None = None, outcome_confidence: float = 0.8,
                outcome_detail: str = "", skills_used: list[str] | None = None, eval_id: str | None = None,
                agent: str = "mcp-host") -> str:
    """Call this AFTER finishing a task (success OR failure). Pass the conversation as `messages`
    (OpenAI or Anthropic format; auto-detected) or canonical `steps`. Set `outcome` to
    success|failure|partial if you know it. Include `skills_used` (names from skill_recall).
    If you were running a task from skill_pending_evals, pass its eval_id."""
    payload: dict[str, Any] = {"task": task, "agent": agent, "skills_used": skills_used or [], "eval_id": eval_id}
    if messages is not None:
        payload["messages"] = messages
        if format != "auto":
            payload["format"] = format
    if steps is not None:
        payload["steps"] = steps
    if outcome:
        payload["signals"] = [{"source": "self", "outcome": outcome, "confidence": outcome_confidence,
                               "detail": outcome_detail}]
    return json.dumps(loop().learn(payload), indent=2)


@mcp.tool()
def skill_pending_evals(limit: int = 5) -> str:
    """Tasks that should be re-run to verify a candidate skill. Run one with skill_recall, then report
    the result with skill_learn(..., eval_id=<id>). Passing promotes the skill to active."""
    return json.dumps(loop().pending_evals(limit), indent=2)


@mcp.tool()
def skill_process() -> str:
    """Run the learning loop over queued traces now (reflect -> write/patch skill -> gate -> curate)."""
    return json.dumps(loop().process(), indent=2)


@mcp.tool()
def skill_inspect(name: str | None = None) -> str:
    """Library overview, or full history + evals for one skill."""
    return json.dumps(loop().inspect(name), indent=2, default=str)


@mcp.tool()
def skill_rollback(name: str, version: int) -> str:
    """Revert a skill to an earlier version (see skill_inspect for versions)."""
    return json.dumps(loop().rollback(name, version), indent=2)


@mcp.tool()
def skill_set_status(name: str, status: str) -> str:
    """Manual override: status = active | candidate | quarantine | retired."""
    return json.dumps(loop().set_status(name, status), indent=2)


@mcp.tool()
def skill_approve(name: str) -> str:
    """Human approval for a skill (required for activation when SKILLLOOP_REQUIRE_APPROVAL=1)."""
    return json.dumps(loop().approve(name), indent=2)


@mcp.tool()
def skill_metrics() -> str:
    """Success rate over episodes, library size, regression rate."""
    return json.dumps(loop().metrics(), indent=2)


def main():
    mcp.run()


if __name__ == "__main__":
    main()
