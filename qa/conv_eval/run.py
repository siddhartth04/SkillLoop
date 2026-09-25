"""Conventions eval runner. Protocol: qa/conv_eval/PROTOCOL.md (fixed before any model run).

  python -m qa.conv_eval.run --selfcheck            # no model, no key: validates generator + harness + stats
  python -m qa.conv_eval.run --pilot                # ~60 agent calls: null + oracle gates, prints every failure
  python -m qa.conv_eval.run --full --seeds 1 2 3   # the eval (stops early if a gate fails)

Model: whatever skillloop.llm.LLM is configured for (ANTHROPIC_API_KEY, or OPENAI_API_KEY + OPENAI_BASE_URL for
Groq / OpenRouter / Ollama / vLLM, with SKILLLOOP_MODEL). The agent and SkillLoop's learner use the same model
unless CONV_AGENT_MODEL is set.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

from skillloop import SkillLoop
from skillloop.llm import LLM

from .generator import POOL, Suite, Task, generate
from .harness import (Episode, HarnessError, agent_feedback, execute, extract_code, learning_message,
                      run_episode)

OUT = Path(__file__).resolve().parent / "results"
MEMORY_CHARS = 8000

# --------------------------------------------------------------------------- statistics
def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - m), min(1.0, c + m))


def sign_test(wins: int, losses: int) -> float:
    """Exact two-sided binomial test on discordant pairs (McNemar exact)."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def paired(a: dict[str, Episode], b: dict[str, Episode], ids: list[str]) -> dict:
    ids = [i for i in ids if i in a and i in b and not a[i].harness_error and not b[i].harness_error]
    w = sum(a[i].success and not b[i].success for i in ids)
    l = sum(b[i].success and not a[i].success for i in ids)
    return {"n": len(ids), "wins": w, "losses": l, "p": round(sign_test(w, l), 5)}


def rate(eps: dict[str, Episode], ids: list[str], first=False) -> dict:
    ok = [e for i, e in eps.items() if i in ids and not e.harness_error]
    k = sum((e.first_try if first else e.success) for e in ok)
    lo, hi = wilson(k, len(ok))
    ctx = [getattr(e, "context_chars", 0) for e in ok]
    return {"k": k, "n": len(ok), "rate": round(k / len(ok), 3) if ok else None,
            "ci95": [round(lo, 2), round(hi, 2)],
            # Cost, alongside accuracy. A method that matches another on a fraction of the context is better
            # even at equal accuracy, and this is the axis agent-memory benchmarks routinely omit.
            "mean_context_chars": round(sum(ctx) / len(ctx)) if ctx else 0,
            "harness_errors": sum(1 for i, e in eps.items() if i in ids and e.harness_error)}


# --------------------------------------------------------------------------- conditions
def oracle_context(suite: Suite) -> str:
    return "Notes about this library:\n" + "\n".join(f"- {suite.hints[q]}" for q in suite.quirks)


def train_phase(agent, suite: Suite) -> list[tuple[Task, Episode]]:
    """The agent works the TRAIN tasks with no help. Both learners see exactly these episodes."""
    return [(t, run_episode(agent, suite, t, "train")) for t in suite.split("train")]


def memory_context(records: list[tuple[Task, Episode]]) -> str:
    """Simple-memory baseline (Reflexion-style raw notes): every training attempt, verbatim, most recent last."""
    lines = []
    for t, ep in records:
        for att in ep.attempts:
            lines.append(f"Task: {t.prompt}\nCode:\n{att['code']}\nOutcome: {learning_message(t, att['verdict'])}")
    text = "\n\n".join(lines)
    return "Notes from your previous work with this library:\n\n" + text[-MEMORY_CHARS:]


def skillloop_learn(loop: SkillLoop, records: list[tuple[Task, Episode]]) -> list:
    """The REAL SkillLoop pipeline: Session -> learn() -> process() (reflect, synthesize, gate) with a real model."""
    import warnings
    warnings.filterwarnings("ignore", message="SkillLoop: .* traces are queued")   # we process right below
    for t, ep in records:
        if ep.harness_error or not ep.attempts:
            continue
        with loop.session(t.prompt, agent="conv-eval", error_hints=False) as s:
            for att in ep.attempts:
                ok = att["verdict"]["ok"]
                s.tool("python", {"code": att["code"]}, result="tests passed" if ok else None,
                       error=None if ok else learning_message(t, att["verdict"]))
            s.verified_by(ep.success)
    report = []
    for _ in range(3):                       # process() takes 20 traces per call
        r = loop.process()
        report.extend(r)
        if not r or all("curate" in x or "principles" in x for x in r):
            break
    return report


def run_promotion(agent, loop: SkillLoop, suite: Suite, records: list[tuple[Task, Episode]]) -> list[dict]:
    """Close the gate: re-run each new skill's own task with the skill in context, graded by the hidden check.

    Without this the eval stops at `candidate` and never exercises the project's central claim, that a skill
    is a hypothesis until an independent check agrees. The re-run is a real agent episode and the verdict
    comes from the task's checker, never from the model's opinion of its own work.
    """
    by_prompt = {t.prompt: t for t, _ in records}

    def runner(task_prompt: str, skill_md: str):
        task = by_prompt.get(task_prompt)
        if task is None:                      # the eval only knows how to grade its own training tasks
            return {"task": task_prompt, "outcome": "failure",
                    "steps": [{"role": "tool", "content": "no checker for this task"}]}
        ep = run_episode(agent, suite, task, "promotion", skill_md, max_attempts=1)
        s = loop.session(task_prompt, agent="conv-eval-promotion", auto_learn=False)
        if ep.harness_error or not ep.attempts:
            s.tool("python", {}, error=ep.harness_error or "no attempt")
            return s.verified_by(1)
        att = ep.attempts[-1]
        s.tool("python", {"code": att["code"]},
               result="tests passed" if att["verdict"]["ok"] else None,
               error=None if att["verdict"]["ok"] else learning_message(task, att["verdict"]))
        return s.verified_by(bool(att["verdict"]["ok"]))      # the hidden check decides, not the agent

    return loop.run_pending_evals(runner, limit=25)


def skillloop_episode(agent, loop: SkillLoop, suite: Suite, t: Task, condition: str = "skillloop",
                      extra_context: str = "") -> Episode:
    """One test episode with the skill library: recall at task start, error-time hints after a failure.

    `extra_context` is prepended to the recalled skills. The `combined` condition uses it to give the agent
    the raw past attempts as well, which is the hypothesis seed 2 pointed at: the two methods won on
    different conventions (skills 3/3 on unit errors where memory scored 0/3, memory 3/3 on return-shape
    errors where skills scored 1/3), so giving the agent both should beat either alone.
    """
    s = loop.session(t.prompt, agent="conv-eval", auto_learn=False)
    ctx = s.skills_text()
    if extra_context:
        ctx = extra_context.strip() + "\n\n" + ctx if ctx else extra_context.strip()

    def on_fail(code, feedback):
        s.tool("python", {"code": code}, error=feedback)
        return s.hints_text()

    ep = run_episode(agent, suite, t, condition, ctx, on_fail=on_fail)
    ep.hints_shown = list(s.skills_used)
    return ep


# --------------------------------------------------------------------------- runs
def make_models():
    learner = LLM()
    agent_model = os.getenv("CONV_AGENT_MODEL")
    agent = LLM(model=agent_model) if agent_model else LLM()
    return agent, learner


def run_conditions(agent, suite, ids, names, context=None) -> dict[str, dict[str, Episode]]:
    out = {}
    tasks = [t for t in suite.tasks if t.id in ids]
    for name in names:
        ctx = (context or {}).get(name, "")
        out[name] = {t.id: run_episode(agent, suite, t, name, ctx) for t in tasks}
    return out


def gates(ctrl, ctrl2, oracle, trap_ids) -> dict:
    c, o = rate(ctrl, trap_ids), rate(oracle, trap_ids)
    common = [i for i in trap_ids if not ctrl[i].harness_error and not ctrl2[i].harness_error]
    agree = sum(ctrl[i].success == ctrl2[i].success for i in common) / len(common) if common else 0.0
    herr = sum(bool(e.harness_error) for d in (ctrl, ctrl2, oracle) for e in d.values())
    total = sum(len(d) for d in (ctrl, ctrl2, oracle))
    g = {
        "null_agreement": round(agree, 3), "null_ok": agree >= 0.85,
        "control_rate": c["rate"], "oracle_rate": o["rate"],
        "headroom_ok": c["rate"] is not None and o["rate"] is not None
                       and c["rate"] <= 0.70 and (o["rate"] - c["rate"]) >= 0.30,
        "harness_error_rate": round(herr / total, 3) if total else 0.0,
    }
    g["harness_ok"] = g["harness_error_rate"] <= 0.05
    g["pass"] = g["null_ok"] and g["headroom_ok"] and g["harness_ok"]
    return g


def print_failures(eps: dict[str, Episode], label: str):
    for e in eps.values():
        if e.harness_error:
            print(f"  [{label}] HARNESS {e.task_id}: {e.harness_error}")
        elif not e.success:
            last = e.attempts[-1]
            print(f"  [{label}] FAIL {e.task_id}: {last['verdict']['error'][:140]!r}\n      code: "
                  + last["code"].replace("\n", " | ")[:160])


def pilot(seed: int = 101):
    agent, _ = make_models()
    suite = generate(seed)
    trap_ids = [t.id for t in suite.split("test") if t.trap][::3] + [t.id for t in suite.split("test") if t.trap][1::3]
    print(f"pilot seed={seed} pkg={suite.pkg} quirks={suite.quirks} tasks={len(trap_ids)}")
    r = run_conditions(agent, suite, trap_ids, ["control", "control_repeat", "oracle"],
                       {"oracle": oracle_context(suite)})
    g = gates(r["control"], r["control_repeat"], r["oracle"], trap_ids)
    print(json.dumps(g, indent=2))
    print("\nEvery failure, for the artifact audit (is each one the AGENT's fault?):")
    for k in r:
        print_failures(r[k], k)
    return g


def full(seeds: list[int]):
    agent, learner = make_models()
    OUT.mkdir(exist_ok=True)
    report = {"seeds": seeds, "started": time.strftime("%Y-%m-%d %H:%M"), "per_seed": {}}
    allc: dict[str, dict[str, Episode]] = {k: {} for k in ("control", "control_repeat", "oracle", "memory",
                                                           "skillloop", "combined")}
    for seed in seeds:
        suite = generate(seed)
        test = suite.split("test")
        trap_ids = [t.id for t in test if t.trap]
        print(f"\n== seed {seed}: pkg={suite.pkg} quirks={suite.quirks}")
        r = run_conditions(agent, suite, [t.id for t in test], ["control", "control_repeat", "oracle"],
                           {"oracle": oracle_context(suite)})
        g = gates(r["control"], r["control_repeat"], r["oracle"], trap_ids)
        print("gates:", json.dumps(g))
        seed_rep = {"gates": g, "quirks": suite.quirks}
        if not g["pass"]:
            print("GATE FAILED for this seed: no valid comparison is possible; SkillLoop is NOT run on it.")
            report["per_seed"][seed] = seed_rep
            continue
        records = train_phase(agent, suite)
        r["memory"] = {t.id: run_episode(agent, suite, t, "memory", memory_context(records)) for t in test}
        loop = SkillLoop(home=tempfile.mkdtemp(prefix=f"conv-sl-{seed}-"), llm=learner)
        learn_report = skillloop_learn(loop, records)
        promo = run_promotion(agent, loop, suite, records)
        print("promotion re-runs:", [(r.get("skill"), r.get("outcome")) for r in promo])
        skills = {s.name: s.status for s in loop.store.all_skills()}
        print("skills learned:", skills)
        r["skillloop"] = {t.id: skillloop_episode(agent, loop, suite, t) for t in test}
        # COMBINED: skills AND raw past attempts. Seed 2 showed the two methods solve different
        # conventions, so this tests whether they add rather than overlap.
        mem_ctx = memory_context(records)
        r["combined"] = {t.id: skillloop_episode(agent, loop, suite, t, "combined", mem_ctx) for t in test}
        seed_rep.update({"skills": skills, "promotion": promo, "train_success": sum(e.success for _, e in records),
                         "learn_report": learn_report})
        for k in allc:
            allc[k].update(r[k])
        report["per_seed"][seed] = seed_rep

    trap = [i for i in allc["control"] if "-notrap" not in i and i in allc["skillloop"]]
    notrap = [i for i in allc["control"] if "-notrap" in i and i in allc["skillloop"]]
    if not trap:
        report["verdict"] = "NO VALID SEED: every seed failed a gate. No claim about SkillLoop can be made."
    else:
        report["primary"] = {k: rate(allc[k], trap) for k in allc}
        report["first_try"] = {k: rate(allc[k], trap, first=True) for k in allc}
        report["harm_notrap"] = {k: rate(allc[k], notrap) for k in allc}
        report["paired"] = {"skillloop_vs_control": paired(allc["skillloop"], allc["control"], trap),
                            "skillloop_vs_memory": paired(allc["skillloop"], allc["memory"], trap),
                            "memory_vs_control": paired(allc["memory"], allc["control"], trap),
                            "combined_vs_memory": paired(allc["combined"], allc["memory"], trap),
                            "combined_vs_skillloop": paired(allc["combined"], allc["skillloop"], trap)}
        report["verdict"] = verdict(report)
    report["episodes"] = {k: {i: vars(e) for i, e in v.items()} for k, v in allc.items()}
    path = OUT / f"full-{int(time.time())}.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print_report(report)
    print(f"\nfull record: {path}")
    return report


def verdict(r) -> str:
    pc, pm = r["paired"]["skillloop_vs_control"], r["paired"]["skillloop_vs_memory"]
    h = r["harm_notrap"]
    harm = (h["skillloop"]["rate"] or 0) < (h["control"]["rate"] or 0) - 0.15
    if pc["p"] < 0.05 and pc["wins"] > pc["losses"]:
        v = "SkillLoop beats control (p<0.05)"
        v += "; and beats simple memory (p<0.05)." if pm["p"] < 0.05 and pm["wins"] > pm["losses"] else \
             "; but NOT significantly better than simple memory, so its extra machinery is not yet justified."
    else:
        v = "No significant improvement over control."
    return v + (" WARNING: skills hurt tasks where the convention was not active." if harm else "")


def print_report(r):
    if "primary" not in r:
        print("\n" + r["verdict"]); return
    print("\nPrimary: success within 2 attempts on held-out TRAP tasks")
    for k, v in r["primary"].items():
        print(f"  {k:<15} {v['k']}/{v['n']} = {v['rate']}  CI95 {v['ci95']}  harness errors: {v['harness_errors']}")
    print("Context cost (mean chars of condition context per task):")
    for k, v in r["primary"].items():
        n = v.get("mean_context_chars", 0)
        print(f"  {k:<15} {n:>7,}")
    print("First-try success:", {k: v["rate"] for k, v in r["first_try"].items()})
    print("No-trap tasks (harm check):", {k: v["rate"] for k, v in r["harm_notrap"].items()})
    print("Paired:", json.dumps(r["paired"]))
    print("VERDICT:", r["verdict"])


# --------------------------------------------------------------------------- self-check (no model)
class ScriptedAgent:
    """A deterministic stand-in used ONLY to validate plumbing. It writes the naive program unless the exact
    oracle hint for the task's convention is in its prompt. It proves the harness can score, gate and compare;
    it says nothing about SkillLoop."""
    def __init__(self, suite: Suite):
        self.suite = suite
        self.by_prompt = {t.prompt: t for t in suite.tasks}

    def complete(self, system, user, **kw):
        t = next(t for p, t in self.by_prompt.items() if f"Task: {p}" in user)
        knows = t.trap and self.suite.hints[t.quirk] in user
        return "Sure:\n```python\n" + (t.quirk_code if knows else t.naive) + "\n```"


def selfcheck(n_seeds: int = 20) -> bool:
    ok = True
    print("1. generator + checkers across", n_seeds, "seeds")
    for seed in range(1, n_seeds + 1):
        s = generate(seed)
        assert len(s.split("train")) == 18 and len(s.split("test")) == 21, seed
        assert len({t.prompt for t in s.tasks}) == len(s.tasks), f"duplicate prompts seed {seed}"
        for t in s.tasks:
            ref = execute(s, t, t.reference)
            if not ref["ok"]:
                ok = False; print("  REFERENCE FAILS", t.id, ref)
            naive = execute(s, t, t.naive)
            if t.trap and naive["ok"]:
                ok = False; print("  NO HEADROOM (naive passes a trap task)", t.id)
            if not t.trap and not naive["ok"]:
                ok = False; print("  NO-TRAP task fails for naive code", t.id, naive)
    print("   ", "ok" if ok else "FAILED")

    print("2. code extraction")
    cases = {"```python\nx=1\n```": "x=1", "text\n```\ny=2\n```\nmore": "y=2", "z=3": "z=3",
             "<think>a</think>```py\nq=1\n```": "q=1", "```python\na=1\n```\n```python\nb=2\n```": "a=1\nb=2"}
    for raw, want in cases.items():
        got = extract_code(raw)
        if got != want:
            ok = False; print("  extract", repr(raw), "->", repr(got))
    try:
        extract_code("```python\nx = 1")
        ok = False; print("  truncated output was not flagged")
    except HarnessError:
        pass
    print("   ", "ok" if ok else "FAILED")

    print("3. full pipeline with a scripted agent (plumbing only, NOT a result)")
    s = generate(7)
    agent = ScriptedAgent(s)
    test = s.split("test")
    trap = [t.id for t in test if t.trap]
    r = run_conditions(agent, s, [t.id for t in test], ["control", "control_repeat", "oracle"],
                       {"oracle": oracle_context(s)})
    g = gates(r["control"], r["control_repeat"], r["oracle"], trap)
    exp = g["control_rate"] == 0.0 and g["oracle_rate"] == 1.0 and g["null_agreement"] == 1.0 and g["pass"]
    print("    gates:", g, "ok" if exp else "UNEXPECTED")
    ok &= exp
    records = train_phase(agent, s)
    mem = memory_context(records)
    assert "AssertionError: test failed" in mem or "Error" in mem
    loop = SkillLoop(home=tempfile.mkdtemp(), llm=LLM(provider="fake"))
    skillloop_learn(loop, records)
    traces = loop.store.db.execute("SELECT COUNT(*) FROM traces").fetchone()[0]
    sl = {t.id: skillloop_episode(agent, loop, s, t) for t in test}
    plumbing = traces == len(records) and len(sl) == len(test)
    print(f"    SkillLoop wiring: {traces} traces reached learn(), {len(sl)} test episodes ran:",
          "ok" if plumbing else "FAILED")
    ok &= plumbing
    p = paired(r["oracle"], r["control"], trap)
    stats = p["wins"] == len(trap) and p["losses"] == 0 and p["p"] < 0.001
    print(f"    stats: oracle vs control {p}", "ok" if stats else "UNEXPECTED")
    ok &= stats
    print("\nSELF-CHECK:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="*", default=[1, 2, 3])
    a = ap.parse_args()
    if a.selfcheck:
        sys.exit(0 if selfcheck() else 1)
    if a.pilot:
        sys.exit(0 if pilot()["pass"] else 2)
    if a.full:
        full(a.seeds)
    else:
        ap.print_help()
