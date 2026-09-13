<div align="center">

<img src="assets/skillloop-banner.png" alt="SkillLoop" width="720"/>

<p align="center">
  <img src="https://img.shields.io/badge/tests-72%20passing-18181b?style=for-the-badge" alt="tests"/>
  <img src="https://img.shields.io/badge/python-3.10%2B-6366f1?style=for-the-badge" alt="python"/>
  <img src="https://img.shields.io/badge/license-MIT-8b5cf6?style=for-the-badge" alt="license"/>
  <a href="CALIBRATION.md"><img src="https://img.shields.io/badge/evidence-pre--registered-a855f7?style=for-the-badge" alt="evidence"/></a>
</p>

<p align="center"><b>Your agent keeps making the same mistake every session. SkillLoop makes it stop.</b></p>

<p align="center">
  <a href="#why">Why</a> ·
  <a href="#architecture">Architecture</a> ·
  <a href="#how-it-works-step-by-step">How it works</a> ·
  <a href="#quickstart">Quickstart</a> ·
  <a href="#status--honest-scope">Evidence</a> ·
  <a href="CALIBRATION.md">Calibration</a>
</p>

</div>

---

## Why

LLM agents have amnesia. Every session starts from zero. An agent that failed a task yesterday, figured out the fix, and succeeded — makes the **same mistake tomorrow**. On its 100th task it's no better than its 1st.

Most "agent memory" tools remember *facts about the user* ("prefers TypeScript") or raw chat history. SkillLoop remembers **how to do the work** — a *procedure*, learned from what went wrong, and **verified by a test before the agent is ever allowed to trust it**.

## Architecture

<div align="center">
<img src="assets/architecture.svg" alt="SkillLoop architecture: a loop — Task, recall, Act, learn, Reflect, Gate — feeding a Skill Library that recall reads from and the gate writes to on pass" width="100%"/>
</div>

Everything is two calls to your agent — **`recall(task)`** before a task, **`learn(trace)`** after. The rest runs in the background: a failure becomes a root cause, a root cause becomes a `SKILL.md`, and that skill is **gated by a test** before it can ever be retrieved. Skills that stop helping are demoted. Nothing is trusted until it passes.

## How it works, step by step

Follow the loop in the diagram above. Here is exactly what happens on each pass:

**1 · Task.** Your agent is about to do something — fix a test, edit a config, deploy a service.

**2 · `recall(task)`.** *Before* the agent acts, SkillLoop looks in its library for skills relevant to this task and returns them as text you drop into the prompt. This is **hybrid retrieval**: embeddings find semantically-related skills even when the wording differs, a lexical layer adds precision, an **intent gate** refuses to fire on non-tasks ("write a poem", "what is X"), and if nothing is genuinely relevant it **abstains** and returns nothing. Fast, local, no LLM call.

**3 · Act.** The agent does the task, now primed with any hard-won lessons from past failures.

**4 · `learn(trace)`.** *After* the task, you hand SkillLoop the trajectory — the commands run, what came back, and whether it worked (a test result, an exit code, a checker). This is fire-and-forget; the agent moves on. Secrets are redacted and the trace is size-capped before anything is stored.

**5 · Reflect.** In the background, if the task *failed*, a model examines the trajectory and names the **decision** that went wrong — not the error message, the decision. A second *skeptic* pass must quote the evidence for that root cause, or the lesson is thrown away as a guess.

**6 · Synthesize.** The lesson becomes a `SKILL.md` in a fixed shape: **preconditions · procedure · verification · failure modes · scope**. The *verification* section is mandatory — a skill that can't say how to prove it worked is rejected.

**7 · Gate.** The new skill is treated as a hypothesis. It's scanned for injected/malicious content, then **tested**: does an agent following it actually satisfy the assertions? Only if it passes does it enter the library as a **candidate**. If not, it's quarantined.

**8 · Into the Skill Library — and back around.** A candidate that proves itself in real use is promoted to **active**. One that starts failing is **demoted** back to quarantine. Every version is rollback-able. Next time a similar task comes up, `recall` surfaces the skill — and the loop tightens.

> **A skill is a hypothesis until it passes a test.**

## Quickstart

```bash
# not on PyPI yet - install from source:
pip install "skillloop[openai] @ git+https://github.com/siddhartth04/SkillLoop"
# or clone and: pip install -e ".[openai]"     # [anthropic] also available; core is stdlib-only
```

```python
from skillloop import SkillLoop

loop = SkillLoop()

with loop.session("migrate the postgres schema") as s:
    prompt += s.skills_text()               # step 2: inject what past failures taught
    ...                                      # step 3: your agent does its work
    s.tool("bash", {"cmd": cmd}, result=out, error=err)
    s.verified_by(check.returncode)          # step 4: an independent check — not the agent's opinion
```

That's the whole integration. The `session` records the trajectory and submits it on exit — including an uncaught exception, which is often the most valuable trace of all. Already have trajectories in OpenAI/Anthropic format? `loop.learn({"task": ..., "format": "openai", "messages": [...]})` takes them directly.

## Status & honest scope

This project's defining trait is that **it does not overclaim.**

### 🟡 Promising pilot — pre-registered, honest about power

On **3 necessity-screened tasks** (each run 3×), where a baseline agent reliably fails, attaching learned skills flipped every task to success — replicated in direction across two model families. Statistics were fixed before results were seen.

| Agent | Tasks (control → treat) | Task-level sign test |
|:--|:--:|:--:|
| gpt-oss-20b | 3/3 tasks flipped fail → pass | p = 0.25 *(underpowered)* |
| qwen3.8-27b | 3/3 tasks flipped fail → pass | p = 0.25 *(underpowered)* |

**Read this honestly.** The independent unit is the *task*, and there are only three. Three tasks cannot reach p < 0.05 even with a perfect split — the sign test bottoms out at p = 0.25. So this is a **promising 3-task pilot with a clean directional signal**, *not* a significant result yet. (A per-trial test gives p = 0.0078, but repeated runs of the same task aren't independent, so that number is anti-conservative and we don't report it.) Reproduce and see for yourself: `python qa/report_ab.py evidence/ab.json`.

**To earn a real claim, the project needs more distinct tasks** — the HumanEval+ run (in progress) is the path there.

### ⚠️ Caveats that travel with those numbers

- **Selection effect by design** — measured only where the baseline fails ≥60%. Shows a skill *can* help, not how often one applies.
- **The margin shrinks as agents get stronger** (11% → 33% control). Report per-model, never pooled.
- **Single domain** — shell/text. A HumanEval+ coding evaluation is *built and in progress*, not yet measured. **No cross-domain claim is made.**
- **Retrieval on paraphrase is a known frontier** — lexical-only scored F1 0.47 on a blind adversarial probe set; the hybrid embedding path (below) is the fix, and needs an embedding model configured to reach its ceiling.

### 🔬 What we're proudest of

SkillLoop **caught its own false positives five times** during development — impossible task checkers, a hallucinating agent model, a GIL-masked concurrency test, rate-limit errors miscounted as failures, and a wrong-interpreter checker. Each would have manufactured a fake result. All documented in [`CALIBRATION.md`](CALIBRATION.md), including the experiments that *failed*. The rigor is the point.

## Use it with any agent

| Path | How | For |
|:--|:--|:--|
| **Python** | `with loop.session(task) as s:` | embedding an agent in Python |
| **MCP** | `skillloop mcp` | Claude Code, Cursor, any MCP host *(protocol-verified)* |
| **HTTP** | `skillloop http` → `POST /recall`, `/learn` | any language, any stack |
| **Docker** | `docker run -p 7331:7331 -v data:/data skillloop` | production — health, backups, graceful shutdown |

Works with **any** provider: Anthropic and OpenAI natively, plus anything OpenAI-compatible via `OPENAI_BASE_URL` — Groq, OpenRouter, **Ollama (fully local, zero data leaves your machine)**, vLLM. Set `SKILLLOOP_EMBED_MODEL` to turn on hybrid retrieval. A `manual` provider runs the learning pipeline with no API key at all. Skills export as [agentskills.io](https://agentskills.io) `SKILL.md`.

## When it helps

Best when your agent **(a)** does repeatable tasks, **(b)** can tell if it succeeded — a test, exit code, checker — and **(c)** keeps failing the same things. Its sweet spot is *project-specific* knowledge no model has from pretraining: this repo's quirks, this build's silent failure, this API's odd error. It does **not** make the model smarter at novel problems, and it needs a success signal to learn from.

## Under the hood

- **Retrieval** — hybrid: embeddings surface candidates independently of wording, lexical + anchor rules add precision, an intent gate blocks non-task queries, and abstention is a real answer. Held-out (author probes): precision 1.0, 0 false-fires; scales to 200+ skills at single-digit ms.
- **Skills are playbooks** — itemized bullets with helpful/harmful counters; patches are deterministic *delta ops*, never a full rewrite (avoids context-collapse — [ACE, ICLR 2026](https://arxiv.org/abs/2510.04618)).
- **Security** — secret redaction, size caps, injection scanning + LLM judge, quarantine-by-default. `pytest tests/test_redteam.py` (20 cases).
- **Production** — thread-safe SQLite (per-thread connections, WAL), structured JSON logs with request IDs, `/health`, graceful shutdown, migrations, online backup. See [`DEPLOYMENT.md`](DEPLOYMENT.md).

## Research lineage

Fills the **procedural-memory** gap most agent-memory tools leave open (they do facts or chat history). Closest work: [Agent Workflow Memory](https://arxiv.org/abs/2409.07429) (ICML 2025), [Agentic Context Engineering](https://arxiv.org/abs/2510.04618) (ICLR 2026). The distinguishing move: **learn from failure**, **gate on a test**.

## Documentation

- [`CALIBRATION.md`](CALIBRATION.md) — every experiment, including the failures and the self-caught bugs *(the evidence page)*
- [`DEPLOYMENT.md`](DEPLOYMENT.md) — Docker, config, health, backup, security posture
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — how to contribute, and the honesty ground rules

## License

MIT — bring your own agent.

<div align="center"><br/><sub><b>A skill is a hypothesis until it passes a test.</b></sub></div>