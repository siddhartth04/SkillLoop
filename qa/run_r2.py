"""R2: held-out retrieval eval for SkillLoop's PURPOSE - the agent must get the lesson back when it matters.

  python qa/run_r2.py              # N=0 and N=500 distractors
  python qa/run_r2.py --json out.json

Probe set (qa/fixtures/r2_heldout.json) was written BEFORE any retrieval change, against the R1 library:
  Q_question_task  real tasks phrased as questions          -> must fire the target skill
  S_error_output   raw error text an agent sees mid-task     -> must fire the target via error-time recall
  T_trap           knowledge / creation / creative requests  -> must abstain

S probes use loop.recall_for_error(error, task="") when it exists, else plain recall(error) (the baseline).
Deterministic, no API key.
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from skillloop import SkillLoop
from skillloop.llm import LLM
from bench_scale import make_distractors
from run_r1 import build_library, wilson

PROBES = HERE / "fixtures" / "r2_heldout.json"
SIZES = (0, 500)


def _names(hits):
    return [h["name"] for h in hits if h["name"] != "_principles"]


def evaluate(loop: SkillLoop, probes: dict) -> dict:
    per_class, records, lat = {}, [], []
    has_err = hasattr(loop, "recall_for_error")
    for cls, items in probes.items():
        if cls.startswith("_"):
            continue
        correct = 0
        for query, target in items:
            t0 = time.perf_counter()
            if cls.startswith("S_") and has_err:
                hits = _names(loop.recall_for_error(query, task=""))
            else:
                hits = _names(loop.recall(query, limit=3))
            lat.append((time.perf_counter() - t0) * 1000)
            top = hits[0] if hits else None
            ok = (top == target) if target else (top is None)
            correct += ok
            records.append({"class": cls, "query": query, "expected": target, "got": top, "ok": ok})
        lo, hi = wilson(correct, len(items))
        per_class[cls] = {"correct": correct, "n": len(items), "ci95": [round(lo, 2), round(hi, 2)]}
    pos = [r for r in records if r["expected"]]
    neg = [r for r in records if not r["expected"]]
    tp = sum(r["ok"] for r in pos)
    fp = sum(1 for r in pos if r["got"] and not r["ok"]) + sum(1 for r in neg if r["got"])
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / len(pos) if pos else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"per_class": per_class, "records": records,
            "overall": {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3),
                        "median_ms": round(sorted(lat)[len(lat) // 2], 2)}}


def run(sizes=SIZES, json_out=None):
    probes = json.loads(PROBES.read_text())
    results = {}
    for size in sizes:
        lp = SkillLoop(home=tempfile.mkdtemp(prefix="r2-"), llm=LLM(provider="fake"))
        build_library(lp)
        for d in make_distractors(size):
            lp.store.save_skill(d, triggers=list(d.queries), snapshot=False)
        results[size] = evaluate(lp, probes)
    classes = [c for c in probes if not c.startswith("_")]
    hdr = f"{'class':<18}" + "".join(f"{'N=' + str(s):>18}" for s in sizes)
    print(hdr); print("-" * len(hdr))
    for c in classes:
        print(f"{c:<18}" + "".join(
            f"{results[s]['per_class'][c]['correct']}/{results[s]['per_class'][c]['n']} "
            f"[{results[s]['per_class'][c]['ci95'][0]:.2f}-{results[s]['per_class'][c]['ci95'][1]:.2f}]".rjust(18)
            for s in sizes))
    print("-" * len(hdr))
    for m in ("precision", "recall", "f1", "median_ms"):
        print(f"{m:<18}" + "".join(f"{results[s]['overall'][m]:>18}" for s in sizes))
    print(f"\nFAILURES at N={sizes[0]}:")
    for r in results[sizes[0]]["records"]:
        if not r["ok"]:
            q = r["query"].replace("\n", " ")[:60]
            print(f"  [{r['class']}] {q:<60} expected={r['expected']} got={r['got']}")
    if json_out:
        Path(json_out).write_text(json.dumps(results, indent=2) + "\n")
    return results


if __name__ == "__main__":
    a = sys.argv[1:]
    out = None
    if "--json" in a:
        i = a.index("--json"); out = a[i + 1]; a = a[:i] + a[i + 2:]
    run(tuple(int(x) for x in a) if a else SIZES, out)
