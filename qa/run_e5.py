"""E5: second domain (Python coding) - screen -> learn -> control vs treatment. Resumable.

  python qa/run_e5.py verify   - qa/verify_checkers.py (must pass before anything else)
  python qa/run_e5.py screen   - 3 baseline runs per task, no skills
  python qa/run_e5.py learn    - feed screen traces to SkillLoop, reflect + gate
  python qa/run_e5.py ab       - eligible tasks only: 3 control + 3 treatment runs, interleaved C,T,C,T,...
  python qa/run_e5.py export   - write e5_ab.json + e5_skills.json for qa/report_ab.py

Protocol (fixed before results were seen):
  - agent openai/gpt-4o-mini, native tool calling, temperature 0.7, max 10 turns (qa/agent_native.py)
  - eligible  <=> baseline passes <= 1 of 3 screening runs
  - the skill library is FROZEN after `learn`: neither A/B arm is learned from
  - harness/API errors are re-run (max 2 retries) and never scored as failures
  - treatment = recall(task prompt, limit=2) injected; control = identical harness, no injection
Env: OPENAI_BASE_URL, OPENAI_API_KEY. SkillLoop's own LLM (reflect/synth/judge) uses SKILLLOOP_MODEL.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import agent_native as A  # noqa: E402
from tasks4 import DOMAIN2  # noqa: E402

from skillloop import SkillLoop  # noqa: E402
from skillloop.gate import Policy  # noqa: E402
from skillloop.llm import LLM  # noqa: E402

OUT = Path(os.getenv("E5_OUT", "/home/claude/qa/e5"))
STATE = OUT / "state.json"
TRACES = OUT / "traces"
HOME = str(OUT / "library")
REPEATS = 3
ELIGIBLE_MAX_PASSES = 1


def load() -> dict:
    s = json.loads(STATE.read_text()) if STATE.exists() else {}
    for k in ("screen", "control", "treat", "recall", "errors"):
        s.setdefault(k, {})
    s.setdefault("protocol", {"agent_model": A.AGENT_MODEL, "temperature": A.TEMPERATURE,
                              "max_turns": A.MAX_TURNS, "repeats": REPEATS,
                              "eligible": f"baseline passes <= {ELIGIBLE_MAX_PASSES}/{REPEATS}",
                              "learner_model": os.getenv("SKILLLOOP_MODEL")})
    return s


def save(s: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(s, indent=1, default=str))


def loop() -> SkillLoop:
    p = Policy()
    p.min_host_successes_for_active = 1
    return SkillLoop(home=HOME, llm=LLM(), policy=p)


def one(task, tag, s, skills_text="", cl=None):
    """Run with retries on harness error. Returns the result dict, or None if it never ran cleanly."""
    for attempt in range(3):
        r = A.run_task(task, f"{tag}_a{attempt}" if attempt else tag, skills_text, cl=cl, log_dir=TRACES)
        if r["harness_error"] is None:
            return r
        s["errors"][f"{task['id']}|{tag}|a{attempt}"] = r["harness_error"]
        save(s)
        print(f"   harness error (not scored): {r['harness_error'][:120]}", flush=True)
    return None


def screen():
    s, cl = load(), A.client()
    for t in DOMAIN2:
        runs = s["screen"].setdefault(t["id"], [])
        while len(runs) < REPEATS:
            i = len(runs)
            r = one(t, f"screen{i}", s, cl=cl)
            if r is None:
                sys.exit("persistent harness errors - stopping; do not score")
            runs.append({"passed": r["passed"], "trace": r["transcript_path"], "tool_calls": r["tool_calls"],
                         "finished": r["finished"], "checker_out": r["checker_out"][:200]})
            Path(r["transcript_path"]).with_suffix(".trace.json").write_text(r["trace"].to_json())
            save(s)
            print(f"[screen] {t['id']:28s} run{i} {'PASS' if r['passed'] else 'FAIL'} "
                  f"calls={r['tool_calls']} checker={r['checker_out'][:80]!r}", flush=True)
    elig = eligible(s)
    print("\nscreen:", {k: f"{sum(x['passed'] for x in v)}/{len(v)}" for k, v in s["screen"].items()})
    print("eligible:", elig or "NONE")


def eligible(s) -> list[str]:
    return [k for k, v in s["screen"].items()
            if len(v) == REPEATS and sum(x["passed"] for x in v) <= ELIGIBLE_MAX_PASSES]


def learn():
    """Rebuild traces from saved transcripts and feed every screening trace to SkillLoop, then reflect/gate."""
    from skillloop import Trace
    s, lp = load(), loop()
    fed = 0
    for tid, runs in s["screen"].items():
        for run in runs:
            tr = Trace.from_json(Path(run["trace"]).with_suffix(".trace.json").read_text())
            lp.learn(tr)
            fed += 1
    print(f"fed {fed} screening traces")
    rep = lp.process(max_traces=50)
    print(json.dumps(rep, indent=1, default=str)[:6000])
    lib = [{"name": k.name, "status": k.status, "description": k.description} for k in lp.store.all_skills()]
    s["library_after_learn"] = lib
    save(s)
    print("\nlibrary:", json.dumps(lib, indent=1))


def ab():
    s, cl, lp = load(), A.client(), loop()
    elig = eligible(s)
    if not elig:
        sys.exit("no eligible tasks - E5 cannot be measured with this agent")
    for tid in elig:
        t = next(x for x in DOMAIN2 if x["id"] == tid)
        hits = lp.recall(t["prompt"], limit=2)                  # library is frozen: same hits every run
        names = [h["name"] for h in hits]
        skills_text = "\n\n".join(h.get("skill_md") or "" for h in hits if h.get("skill_md"))
        s["recall"][tid] = names
        save(s)
        print(f"[recall] {tid}: {names}", flush=True)
        c, tr = s["control"].setdefault(tid, []), s["treat"].setdefault(tid, [])
        for i in range(REPEATS):                                # interleaved C,T,C,T,...
            for arm, bucket, txt in (("control", c, ""), ("treat", tr, skills_text)):
                if len(bucket) > i:
                    continue
                r = one(t, f"{arm}{i}", s, skills_text=txt, cl=cl)
                if r is None:
                    sys.exit("persistent harness errors - stopping; do not score")
                bucket.append({"passed": r["passed"], "trace": r["transcript_path"],
                               "skills_injected": r["skills_injected"], "checker_out": r["checker_out"][:200]})
                save(s)
                print(f"[{arm:7s}] {tid:28s} run{i} {'PASS' if r['passed'] else 'FAIL'} "
                      f"calls={r['tool_calls']} checker={r['checker_out'][:80]!r}", flush=True)
    export()


def export():
    s = load()
    ab_, skills = {}, {}
    for tid in s["control"]:
        c = [x["passed"] for x in s["control"][tid]]
        t = [x["passed"] for x in s["treat"].get(tid, [])]
        n = min(len(c), len(t))
        ab_[f"{tid}|control"], ab_[f"{tid}|treat"] = c[:n], t[:n]
        rec = s["recall"].get(tid) or []
        skills[tid] = rec[0] if rec else "(no skill recalled)"
    (OUT / "e5_ab.json").write_text(json.dumps(ab_))
    (OUT / "e5_skills.json").write_text(json.dumps(skills, indent=1))
    print("wrote", OUT / "e5_ab.json", "and", OUT / "e5_skills.json")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    if cmd == "verify":
        import verify_checkers
        sys.exit(verify_checkers.main())
    {"screen": screen, "learn": learn, "ab": ab, "export": export}.get(cmd, lambda: print(__doc__))()
