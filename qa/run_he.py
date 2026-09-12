"""E5-HE: SkillLoop on HumanEval+ (EvalPlus), the industry-standard hardened coding benchmark. Resumable.

  python qa/run_he.py screen    - 1 baseline run on all problems; failures get 2 more runs (eligibility)
  python qa/run_he.py learn     - SkillLoop learns from LEARN-split screen traces ONLY, then library frozen
  python qa/run_he.py ab        - eligible TEST problems: control / placebo / treatment x 3, interleaved
  python qa/run_he.py export    - he_ab.json (control vs treat), he_placebo.json (placebo vs treat), he_skills.json
  python qa/run_he.py status

PRE-REGISTERED PROTOCOL (fixed before any agent ran on these problems):
  dataset   HumanEval+ v0.1.10 via evalplus 0.3.1; checker = EvalPlus check_correctness, base AND plus must pass
  excluded  HumanEval/32 - EvalPlus 0.3.1 find_zero oracle bug: the reference solution itself fails (0/888)
  split     random.Random(20260911) shuffle of the remaining 163 ids; first 81 = LEARN, rest (82) = TEST
  agent     openai/gpt-4o-mini, native tool calling, temperature 0.7, max 10 turns (qa/agent_native.py)
  eligible  baseline passes <= 1 of 3 screening runs
  held out  SkillLoop never sees a TEST-split trace. Skills must TRANSFER to unseen problems.
  frozen    no learning during the A/B; recall() once per problem, same text for every treatment repeat
  ITT       every eligible TEST problem is in the A/B even if recall abstains (then treatment = bare prompt)
  arms      control = bare prompt; placebo = fixed generic checklist (below); treatment = recall() output
  primary   control vs treatment, qa/report_ab.py sign test, p<0.05 and >=2 distinct skills
  secondary placebo vs treatment, same statistic
  errors    harness/API errors re-run (max 2 retries), never scored; checker TIMEOUT re-checked serially
"""
from __future__ import annotations

import json
import os
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import agent_native as A  # noqa: E402

from skillloop import SkillLoop, Trace  # noqa: E402
from skillloop.gate import Policy  # noqa: E402
from skillloop.llm import LLM  # noqa: E402

SEED = 20260911
EXCLUDE = {"HumanEval/32": "EvalPlus 0.3.1 find_zero oracle bug - reference solution fails"}
REPEATS, ELIGIBLE_MAX_PASSES = 3, 1
WORKERS = int(os.getenv("HE_WORKERS", "8"))
OUT = Path(os.getenv("HE_OUT", "/home/claude/qa/he"))
STATE, TRACES, HOME = OUT / "state.json", OUT / "traces", str(OUT / "library")
LOCK = threading.Lock()

PLACEBO = """---
name: careful-python-function-checklist
description: Generic checklist for implementing a Python function from a docstring.
---
# Careful Python function checklist
1. Re-read the docstring and every example; restate the exact input and output contract before coding.
2. Keep the exact function name and signature. Include every import and helper the function needs.
3. List edge cases before writing code: empty input, a single element, zero, negative numbers, duplicates,
   very large values, and inputs at the boundaries mentioned in the docstring.
4. Match the return type precisely (int vs float, list vs tuple, str vs None) and the required ordering.
5. Do not mutate the arguments unless the docstring asks for it.
6. Prefer a simple, clearly correct algorithm over a clever one; avoid needless quadratic work on big inputs.
7. Run every docstring example, then your own edge cases, and compare the output to what you expect.
8. If any check fails, fix the root cause and re-run all checks before finishing.
"""


def problems() -> dict:
    import he_check
    P, _ = he_check.data()
    return P


def split() -> tuple[list[str], list[str]]:
    ids = sorted((k for k in problems() if k not in EXCLUDE), key=lambda k: int(k.split("/")[1]))
    random.Random(SEED).shuffle(ids)
    return ids[:81], ids[81:]


def task(tid: str) -> dict:
    p = problems()[tid]
    prompt = ("Implement the Python function below in a file named solution.py. solution.py must be "
              "self-contained: include the complete function with the exact same name and signature, plus any "
              "imports and helper functions shown. Hidden tests will import solution.py and call "
              f"{p['entry_point']}.\n\n```python\n{p['prompt']}```")
    return {"id": "he%03d" % int(tid.split("/")[1]), "task_id": tid, "prompt": prompt,
            "checker": f"{sys.executable} {HERE / 'he_check.py'} {tid} solution.py", "checker_timeout": 180}


def load() -> dict:
    s = json.loads(STATE.read_text()) if STATE.exists() else {}
    for k in ("screen", "control", "placebo", "treat", "recall", "errors"):
        s.setdefault(k, {})
    return s


def save(s: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(s, indent=1, default=str))
    tmp.replace(STATE)


def loop() -> SkillLoop:
    p = Policy()
    p.min_host_successes_for_active = 1
    return SkillLoop(home=HOME, llm=LLM(), policy=p)


def one(tid: str, tag: str, skills_text: str = "") -> dict:
    t, cl = task(tid), A.client()
    errs = []
    for attempt in range(3):
        r = A.run_task(t, f"{tag}_a{attempt}" if attempt else tag, skills_text, cl=cl, log_dir=TRACES)
        if r["harness_error"] is None:
            Path(r["transcript_path"]).with_suffix(".trace.json").write_text(r["trace"].to_json())
            return {"tid": tid, "tag": tag, "passed": r["passed"], "status": r["checker_status"],
                    "rechecked": r["rechecked"], "trace": r["transcript_path"], "tool_calls": r["tool_calls"],
                    "finished": r["finished"], "checker_out": r["checker_out"][:300], "errors": errs}
        errs.append(r["harness_error"])
    return {"tid": tid, "tag": tag, "harness_failed": True, "errors": errs}


def run_jobs(jobs, record):
    """jobs: list of (tid, tag, skills_text). record(result) is called under LOCK."""
    done = 0
    with ThreadPoolExecutor(WORKERS) as ex:
        futs = [ex.submit(one, *j) for j in jobs]
        for f in as_completed(futs):
            r = f.result()
            with LOCK:
                record(r)
                done += 1
            flag = "HARNESS-FAIL" if r.get("harness_failed") else ("PASS" if r["passed"] else r["status"].upper())
            print(f"[{done}/{len(jobs)}] {r['tid']:14s} {r['tag']:10s} {flag:12s} "
                  f"{(r.get('checker_out') or '').splitlines()[0][:60] if r.get('checker_out') else ''}", flush=True)


def screen():
    s = load()
    learn_ids, test_ids = split()
    s["split"] = {"learn": learn_ids, "test": test_ids, "seed": SEED, "excluded": EXCLUDE}
    save(s)

    def rec(r):
        if r.get("harness_failed"):
            s["errors"][f"{r['tid']}|{r['tag']}"] = r["errors"]
        else:
            s["screen"].setdefault(r["tid"], []).append(r)
        save(s)

    all_ids = learn_ids + test_ids
    run_jobs([(t, "screen0", "") for t in all_ids if not s["screen"].get(t)], rec)
    fails = [t for t in all_ids if s["screen"].get(t) and not s["screen"][t][0]["passed"]]
    print(f"\npass-1: {sum(1 for t in all_ids if s['screen'].get(t) and s['screen'][t][0]['passed'])}"
          f"/{len(all_ids)}; re-screening {len(fails)} failures x2")
    run_jobs([(t, f"screen{i}", "") for t in fails for i in range(len(s["screen"][t]), REPEATS)], rec)
    status()


def eligible(s, ids) -> list[str]:
    return [t for t in ids if len(s["screen"].get(t, [])) == REPEATS
            and sum(r["passed"] for r in s["screen"][t]) <= ELIGIBLE_MAX_PASSES]


def learn():
    s, lp = load(), loop()
    learn_ids = set(s["split"]["learn"])
    fed = 0
    for tid, runs in s["screen"].items():
        if tid not in learn_ids:
            continue                                     # HELD OUT: TEST traces never reach SkillLoop
        for r in runs:
            lp.learn(Trace.from_json(Path(r["trace"]).with_suffix(".trace.json").read_text()))
            fed += 1
    print(f"fed {fed} LEARN-split traces (TEST split withheld)")
    reports = []
    while lp.store.pending_traces(1):
        reports += lp.process(max_traces=20)
    errs = [x for x in reports if "error" in x]
    lib = [{"name": k.name, "status": k.status, "description": k.description} for k in lp.store.all_skills()]
    s["library_after_learn"] = lib
    s["learn_errors"] = errs
    s["principles"] = lp.store.principles_text()
    save(s)
    print(f"process errors: {len(errs)}", errs[:3])
    print("library:", json.dumps(lib, indent=1))
    print("principles:", (s["principles"] or "(none)")[:1500])


def ab():
    s, lp = load(), loop()
    elig = eligible(s, s["split"]["test"])
    if not elig:
        sys.exit("no eligible TEST problems")
    texts = {}
    for tid in elig:
        hits = lp.recall(task(tid)["prompt"], limit=2)
        names = [h["name"] for h in hits]
        texts[tid] = "\n\n".join(h["skill_md"] for h in hits if h.get("skill_md"))
        s["recall"][tid] = {"names": names, "chars": len(texts[tid])}
        print(f"[recall] {tid}: {names or '(abstained)'}")
    save(s)
    arms = {"control": lambda t: "", "placebo": lambda t: PLACEBO, "treat": lambda t: texts[t]}
    jobs = [(t, f"{arm}{i}", arms[arm](t)) for i in range(REPEATS) for t in elig for arm in arms
            if len(s[arm].get(t, [])) <= i]

    def rec(r):
        if r.get("harness_failed"):
            s["errors"][f"{r['tid']}|{r['tag']}"] = r["errors"]
        else:
            s[r["tag"].rstrip("0123456789")].setdefault(r["tid"], []).append(r)
        save(s)
    run_jobs(jobs, rec)
    export()


def export():
    s = load()
    ab_, plc, skills = {}, {}, {}
    for tid in s["treat"]:
        c = [r["passed"] for r in sorted(s["control"].get(tid, []), key=lambda r: r["tag"])]
        p = [r["passed"] for r in sorted(s["placebo"].get(tid, []), key=lambda r: r["tag"])]
        t = [r["passed"] for r in sorted(s["treat"][tid], key=lambda r: r["tag"])]
        n = min(len(c), len(t))
        ab_[f"{tid}|control"], ab_[f"{tid}|treat"] = c[:n], t[:n]
        m = min(len(p), len(t))
        plc[f"{tid}|control"], plc[f"{tid}|treat"] = p[:m], t[:m]
        names = [x for x in s["recall"].get(tid, {}).get("names", []) if x != "_principles"]
        has_pr = "_principles" in s["recall"].get(tid, {}).get("names", [])
        skills[tid] = names[0] if names else ("_principles" if has_pr else "(abstained)")
    for name, obj in (("he_ab.json", ab_), ("he_placebo.json", plc), ("he_skills.json", skills)):
        (OUT / name).write_text(json.dumps(obj, indent=1))
    print("wrote he_ab.json, he_placebo.json, he_skills.json in", OUT)


def status():
    s = load()
    if "split" not in s:
        return print("not started")
    for name, ids in (("LEARN", s["split"]["learn"]), ("TEST", s["split"]["test"])):
        ran = [t for t in ids if s["screen"].get(t)]
        p1 = sum(s["screen"][t][0]["passed"] for t in ran)
        print(f"{name}: pass-1 {p1}/{len(ran)} = {p1 / max(1, len(ran)):.1%}   eligible: {eligible(s, ids)}")
    tos = [(t, r["tag"]) for t, rs in s["screen"].items() for r in rs if r["status"] == "timeout"]
    print("screen timeouts:", tos or "none", "  harness errors:", len(s["errors"]))


if __name__ == "__main__":
    {"screen": screen, "learn": learn, "ab": ab, "export": export, "status": status}.get(
        sys.argv[1] if len(sys.argv) > 1 else "", lambda: print(__doc__))()
