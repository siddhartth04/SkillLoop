"""Resumable version of `run --full`: identical protocol, but every episode is saved the moment it finishes.

  python -m qa.conv_eval.resume SEED [--state DIR]

Re-running the same command after an interruption skips everything already done: cached episodes are reused,
and SkillLoop's own database keeps its learning progress (processed traces stay processed).
Writes qa/conv_eval/results/full-seed{SEED}.json in the same format as run.py, so pool.py combines them.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from skillloop import SkillLoop

from .generator import generate
from .harness import Episode, run_episode
from .run import (OUT, gates, make_models, memory_context, oracle_context, paired, print_report, rate,
                  run_promotion, skillloop_episode, skillloop_learn, verdict)

STATE = Path("/home/claude/conv_state")


class Cache:
    def __init__(self, path: Path):
        self.path = path
        self.data: dict[str, Episode] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    d = json.loads(line)
                    self.data[d["task_id"]] = Episode(**d)

    def run(self, tasks, fn):
        for t in tasks:
            if t.id in self.data:
                continue
            ep = fn(t)
            if ep.harness_error.startswith("agent call failed"):
                # provider outage / exhausted quota: do NOT cache, so a resume retries it
                print(f"  provider failure on {t.id}, will retry on resume: {ep.harness_error[:120]}", flush=True)
                continue
            self.data[t.id] = ep
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(vars(ep), default=str) + "\n")
        return self.data


def main(seed: int, state: Path = STATE):
    d = state / f"seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    agent, learner = make_models()
    suite = generate(seed)
    test = suite.split("test")
    trap = [t.id for t in test if t.trap]
    print(f"== seed {seed}: pkg={suite.pkg} quirks={suite.quirks}", flush=True)

    r = {}
    for cond, ctx in (("control", ""), ("control_repeat", ""), ("oracle", oracle_context(suite))):
        r[cond] = Cache(d / f"{cond}.jsonl").run(test, lambda t, c=cond, x=ctx: run_episode(agent, suite, t, c, x))
        print(f"  {cond}: {len(r[cond])}/{len(test)} episodes", flush=True)
    if any(len(r[c]) < len(test) for c in r):
        print("INCOMPLETE (provider failures); re-run the same command to resume."); return None
    g = gates(r["control"], r["control_repeat"], r["oracle"], trap)
    print("gates:", json.dumps(g), flush=True)
    rep = {"seeds": [seed], "per_seed": {seed: {"gates": g, "quirks": suite.quirks}}}
    if g["pass"]:
        train = Cache(d / "train.jsonl").run(suite.split("train"), lambda t: run_episode(agent, suite, t, "train"))
        if len(train) < len(suite.split("train")):
            print("INCOMPLETE training; re-run to resume."); return None
        records = [(t, train[t.id]) for t in suite.split("train")]
        r["memory"] = Cache(d / "memory.jsonl").run(
            test, lambda t: run_episode(agent, suite, t, "memory", memory_context(records)))
        loop = SkillLoop(home=str(d / "skillloop"), llm=learner)
        n_traces = loop.store.db.execute("SELECT COUNT(*) FROM traces").fetchone()[0]
        if n_traces == 0:
            skillloop_learn(loop, records)          # submits traces + processes
            # close the gate: re-run each new skill's own task, graded by the hidden check, so the eval
            # actually exercises promotion instead of leaving every skill at `candidate`
            promo = run_promotion(agent, loop, suite, records)
            print("promotion re-runs:", [(r.get("skill"), r.get("outcome")) for r in promo])
        while loop.store.pending_traces(1):         # resume: finish any unprocessed traces
            loop.process()
        pending = len(loop.store.pending_traces(999))
        if pending:
            # Scoring now would measure a library that is missing most of its skills, and the report would
            # look exactly like a run where the model learned little. That is how a quota limit turns into a
            # false negative, so refuse to produce a number instead.
            print(f"INCOMPLETE: {pending} trace(s) still unprocessed (quota or provider limit). "
                  f"Re-run this command once the quota resets; everything already done is cached.")
            return None
        skills = {s.name: s.status for s in loop.store.all_skills()}
        print("skills learned:", skills, flush=True)
        r["skillloop"] = Cache(d / "skillloop.jsonl").run(test, lambda t: skillloop_episode(agent, loop, suite, t))
        # COMBINED: the skill library AND the raw past attempts, which seed 2 suggested are complementary
        r["combined"] = Cache(d / "combined.jsonl").run(
            test, lambda t: skillloop_episode(agent, loop, suite, t, "combined", memory_context(records)))
        if any(len(r[c]) < len(test) for c in ("memory", "skillloop", "combined")):
            print("INCOMPLETE test phase; re-run to resume."); return None
        rep["per_seed"][seed].update({"skills": skills, "train_success": sum(e.success for e in train.values())})
        notrap = [t.id for t in test if not t.trap]
        rep["primary"] = {k: rate(v, trap) for k, v in r.items()}
        rep["first_try"] = {k: rate(v, trap, first=True) for k, v in r.items()}
        rep["harm_notrap"] = {k: rate(v, notrap) for k, v in r.items()}
        rep["paired"] = {"skillloop_vs_control": paired(r["skillloop"], r["control"], trap),
                         "skillloop_vs_memory": paired(r["skillloop"], r["memory"], trap),
                         "memory_vs_control": paired(r["memory"], r["control"], trap),
                         "combined_vs_memory": paired(r["combined"], r["memory"], trap),
                         "combined_vs_skillloop": paired(r["combined"], r["skillloop"], trap)}
        rep["verdict"] = verdict(rep)
    else:
        rep["verdict"] = "gate failed; seed excluded"
    rep["episodes"] = {k: {i: vars(e) for i, e in v.items()} for k, v in r.items()} if g["pass"] else {}
    OUT.mkdir(exist_ok=True)
    (OUT / f"full-seed{seed}.json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    # Copy the per-episode caches next to the report. They live in the state directory while the run is
    # resumable, but analyze.py and anyone auditing the result need them committed alongside it - without
    # them the training-feedback split silently reports "no training episodes" and the most informative
    # part of the analysis disappears.
    ep_out = OUT / f"seed{seed}_episodes"
    ep_out.mkdir(exist_ok=True)
    for src in sorted(d.glob("*.jsonl")):
        (ep_out / src.name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"episodes archived to {ep_out}")
    print_report(rep)
    print("DONE", flush=True)
    return rep


if __name__ == "__main__":
    main(int(sys.argv[1]))
