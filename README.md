<div align="center">

<img src="assets/skillloop-banner.png" alt="SkillLoop" width="720"/>

<p align="center">
  <a href="#quickstart"><img src="https://img.shields.io/badge/tests-66%20passing-18181b?style=for-the-badge" alt="tests"/></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-6366f1?style=for-the-badge" alt="python"/>
  <img src="https://img.shields.io/badge/license-MIT-8b5cf6?style=for-the-badge" alt="license"/>
  <a href="CALIBRATION.md"><img src="https://img.shields.io/badge/evidence-pre--registered-a855f7?style=for-the-badge" alt="evidence"/></a>
</p>

<p align="center"><b>Your agent keeps making the same mistake every session. SkillLoop makes it stop.</b></p>

<p align="center">
  <a href="#the-idea">Idea</a> ·
  <a href="#quickstart">Quickstart</a> ·
  <a href="#status--honest-scope">Evidence</a> ·
  <a href="#use-it-with-any-agent">Integrations</a> ·
  <a href="CALIBRATION.md">Calibration</a> ·
  <a href="DEPLOYMENT.md">Deploy</a>
</p>

</div>

---

## Why

LLM agents have amnesia. Every session starts from zero. An agent that failed a task yesterday, figured out the fix, and succeeded — makes the **same mistake tomorrow**. On its 100th task it's no better than its 1st.

Most "agent memory" tools remember *facts about the user* or raw chat history. SkillLoop remembers **how to do the work** — learned from what went wrong, and **verified before it's trusted**.

## The idea

> **A skill is a hypothesis until it passes a test.**

```
agent fails a task
      │
      ▼  reflect      what DECISION went wrong?  (a skeptic must quote the evidence, or it's discarded)
      ▼  synthesize   write a SKILL.md           (Preconditions · Procedure · Verification · Failure modes · Scope)
      ▼  gate         no concrete verification?  → rejected. quarantined until it passes. rollback-able.
      ▼  recall       next similar task          → the skill is in the prompt, before the agent starts
```

## Quickstart

```bash
pip install "skillloop[openai]"     # or [anthropic]; core is stdlib-only
```

```python
from skillloop import SkillLoop

loop = SkillLoop()

with loop.session("migrate the postgres schema") as s:
    prompt += s.skills_text()               # inject what past failures taught
    ...                                      # your agent does its work
    s.tool("bash", {"cmd": cmd}, result=out, error=err)
    s.verified_by(check.returncode)          # an independent check — not the agent's opinion
```

That's the whole integration. The session records the trajectory and learns from it on exit — including an uncaught exception, which is often the most valuable trace of all.

## Status & honest scope

This project's defining trait is that **it does not overclaim.**

### ✅ Demonstrated — pre-registered, independently checked

On tasks a baseline agent **reliably fails**, attaching learned skills raised success dramatically — replicated across two independent model families, statistics fixed *before* results were seen:

| Agent | Without skills | With skills | Sign-test *p* |
|:--|:--:|:--:|:--:|
| gpt-oss-20b | 1 / 9 · 11% | **9 / 9 · 100%** | **0.0078** |
| qwen3.8-27b | 3 / 9 · 33% | **9 / 9 · 100%** | **0.0312** |

Two independent skills — one lucky skill isn't carrying the result. The agent followed the skill's procedure *verbatim*. Regenerate: `python qa/report_ab.py evidence/ab.json`.

### ⚠️ Caveats that travel with those numbers

- **Selection effect by design** — measured only where the baseline fails ≥60%. Shows a skill *can* help, not how often one applies.
- **The margin shrinks as agents get stronger** (11% → 33% control). Report per-model, never pooled.
- **Single domain** — shell/text. A HumanEval+ coding evaluation is *built and in progress*, not yet measured. **No cross-domain claim is made.**

### 🔬 What we're proudest of

SkillLoop **caught its own false positives five times** — impossible checkers, a hallucinating agent model, a GIL-masked concurrency test, rate-limit errors miscounted as failures, and a wrong-interpreter checker. Each would have manufactured a fake result. All in [`CALIBRATION.md`](CALIBRATION.md), including the experiments that *failed*. The rigor is the point.

## Use it with any agent

| Path | How | For |
|:--|:--|:--|
| **Python** | `with loop.session(task) as s:` | embedding an agent in Python |
| **MCP** | `skillloop mcp` | Claude Code, Cursor, any MCP host *(protocol-verified)* |
| **HTTP** | `skillloop http` → `POST /recall`, `/learn` | any language, any stack |
| **Docker** | `docker run -p 7331:7331 -v data:/data skillloop` | production — health, backups, graceful shutdown |

Works with **any** provider: Anthropic and OpenAI natively, plus anything OpenAI-compatible via `OPENAI_BASE_URL` — Groq, OpenRouter, **Ollama (fully local, zero data leaves your machine)**, vLLM. A `manual` provider runs the learning pipeline with no API key at all. Skills export as [agentskills.io](https://agentskills.io) `SKILL.md`.

## When it helps

Best when your agent **(a)** does repeatable tasks, **(b)** can tell if it succeeded — a test, exit code, checker — and **(c)** keeps failing the same things. Its sweet spot is *project-specific* knowledge no model has from pretraining. It does **not** make the model smarter at novel problems, and it needs a success signal to learn from.

## Under the hood

- **Retrieval** — doc2query + phrase-level matching + real abstention. Held-out: 5/5 correct, **0 false fires**, holds to 200 skills.
- **Skills are playbooks** — itemized bullets with helpful/harmful counters; patches are deterministic *delta ops*, never a rewrite (avoids context-collapse — [ACE, ICLR 2026](https://arxiv.org/abs/2510.04618)).
- **Security** — secret redaction, size caps, injection scanning + LLM judge, quarantine-by-default. `pytest tests/test_redteam.py`.
- **Production** — thread-safe SQLite, structured JSON logs with request IDs, `/health`, graceful shutdown, migrations, online backup. See [`DEPLOYMENT.md`](DEPLOYMENT.md).

## Research lineage

Fills the **procedural-memory** gap most agent-memory tools leave open. Closest work: [Agent Workflow Memory](https://arxiv.org/abs/2409.07429) (ICML 2025), [Agentic Context Engineering](https://arxiv.org/abs/2510.04618) (ICLR 2026). The distinguishing move: **learn from failure**, **gate on a test**.

## License

MIT — bring your own agent.

<div align="center"><br/><sub><b>A skill is a hypothesis until it passes a test.</b></sub></div>
