"""Run the QA cycle. Resumable: state kept in /home/claude/qa/state.json

  python run_qa.py phaseA [n]     - run baseline tasks (no skills)
  python run_qa.py learn          - reflect/synthesize/gate over the collected traces
  python run_qa.py phaseB [n]     - run variant tasks WITH recall enabled
  python run_qa.py control [n]    - run the same variant tasks WITHOUT recall (control arm)
  python run_qa.py report         - print results
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/claude/qa")
sys.path.insert(0, "/home/claude/skillloop")

from harness import run_task, setup  # noqa: E402
from tasks import HARD_A, HARD_B, PHASE_A, PHASE_B  # noqa: E402

from skillloop import SkillLoop  # noqa: E402
from skillloop.gate import Policy  # noqa: E402
from skillloop.llm import LLM  # noqa: E402

STATE = Path("/home/claude/qa/state.json")
HOME = "/home/claude/qa/library"


def state() -> dict:
    d = json.loads(STATE.read_text()) if STATE.exists() else {}
    for k in ("A", "B", "C", "HA", "HB", "HC", "recall"):
        d.setdefault(k, {})
    return d


def save(s: dict) -> None:
    STATE.write_text(json.dumps(s, indent=1, default=str))


def loop() -> SkillLoop:
    p = Policy()
    p.min_host_successes_for_active = 1
    return SkillLoop(home=HOME, llm=LLM(), policy=p)


def agent_llm() -> LLM:
    """The acting agent: a small fast model. Deliberately not the strongest - a weak agent makes the
    presence or absence of a skill visible."""
    return LLM(model="openai/gpt-oss-20b")


def phase(which: str, n: int, use_recall: bool, key: str) -> None:
    s = state()
    lp = loop()
    ag = agent_llm()
    tasks = {"A": PHASE_A, "B": PHASE_B, "HA": HARD_A, "HB": HARD_B}[which]
    ran = 0
    for t in tasks:
        if t["id"] in s[key] or ran >= n:
            continue
        wd = setup(t)
        skills_text, names = "", []
        if use_recall:
            hits = lp.recall(t["prompt"], limit=2)
            names = [h["name"] for h in hits]
            skills_text = "\n\n".join(h["skill_md"] for h in hits)
            s["recall"][t["id"]] = names
        tr = run_task(ag, t, wd, skills_text=skills_text)
        passed = tr.outcome == "success"
        s[key][t["id"]] = {"passed": passed, "trace_id": tr.id, "turns": len(tr.steps),
                           "recalled": names, "detail": tr.signals[-1].detail[:120]}
        if key != "C":                       # control arm is never learned from
            lp.learn(tr)
        save(s)
        print(f"[{key}] {t['id']:14s} {'PASS' if passed else 'FAIL'} turns={len(tr.steps):2d} recalled={names}", flush=True)
        ran += 1
    print(f"done: {len(s[key])}/{len(tasks)} in arm {key}")


def report() -> None:
    s = state()
    lp = loop()
    def rate(d):
        v = [x["passed"] for x in d.values()]
        return f"{sum(v)}/{len(v)}" if v else "-"
    print("\n=== QA RESULTS ===")
    print(f"Phase A (baseline, no skills):      {rate(s['A'])}")
    print(f"Phase C (variants, control, no skills): {rate(s['C'])}")
    print(f"Phase B (variants, WITH recall):    {rate(s['B'])}")
    print(f"HARD baseline:                      {rate(s['HA'])}")
    print(f"HARD variants, control (no skills): {rate(s['HC'])}")
    print(f"HARD variants, WITH recall:         {rate(s['HB'])}")
    print("\nper-task:")
    for tid in sorted(set(list(s["B"]) + list(s["C"]) + list(s["HB"]) + list(s["HC"]))):
        b = s["B"].get(tid) or s["HB"].get(tid)
        c = s["C"].get(tid) or s["HC"].get(tid)
        print(f"  {tid:14s} control={'PASS' if c and c['passed'] else 'FAIL' if c else '-':4s} "
              f"withskill={'PASS' if b and b['passed'] else 'FAIL' if b else '-':4s} recalled={b['recalled'] if b else []}")
    fired = [v for v in s["recall"].values() if v]
    print(f"\nretrieval: a skill fired on {len(fired)}/{len(s['recall'])} variant tasks")
    print("\nlibrary:")
    for sk in lp.store.all_skills():
        print(f"  {sk.name:34s} {sk.status:11s} v{sk.version} recalls={sk.recalls} uses={sk.uses} "
              f"prov={sk.provisional} evidence={sk.evidence_count}")
    print("\nmetrics:", json.dumps(lp.metrics(), indent=1))


if __name__ == "__main__":
    cmd = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 99
    if cmd == "phaseA":
        phase("A", n, False, "A")
    elif cmd == "phaseB":
        phase("B", n, True, "B")
    elif cmd == "control":
        phase("B", n, False, "C")
    elif cmd == "hardA":
        phase("HA", n, False, "HA")
    elif cmd == "hardControl":
        phase("HB", n, False, "HC")
    elif cmd == "hardB":
        phase("HB", n, True, "HB")
    elif cmd == "learn":
        rep = loop().process()
        print(json.dumps(rep, indent=1, default=str)[:4000])
    elif cmd == "report":
        report()
