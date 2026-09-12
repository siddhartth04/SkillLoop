"""R1: adversarial retrieval eval. Protocol pre-registered in qa/R1_PROTOCOL.md.

  python qa/run_r1.py                    # all library sizes
  python qa/run_r1.py --json out.json

Deterministic: retrieval makes no model call, so this needs no API key and is exactly reproducible.
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from skillloop import SkillLoop
from skillloop.llm import LLM
from skillloop.store import Skill
from skillloop.playbook import new_bullet
from bench_scale import make_distractors

LIB = HERE / "fixtures" / "r1_library.json"
PROBES = HERE / "fixtures" / "r1_probes.json"
SIZES = (0, 100, 500, 2000)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval. With n=14 a point estimate on its own is not reportable."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - m), min(1.0, c + m))


def build_library(loop: SkillLoop) -> int:
    for d in json.loads(LIB.read_text()):
        sk = Skill(
            name=d["name"], description=d["description"], body="", status="active",
            bullets=[new_bullet("preconditions", f"Confirm this is a {d['applies_when'][0]} situation."),
                     new_bullet("procedure", f"Apply the documented fix for {d['name'].replace('-', ' ')}."),
                     new_bullet("verification", "Read the resulting state back and assert the intended property."),
                     new_bullet("failure_modes", f"Symptom: {(d['symptoms'] or ['n/a'])[0]}."),
                     new_bullet("scope_limits", f"Not for: {', '.join(d['not_for'])}.")],
            facets={"applies_when": d["applies_when"], "symptoms": d["symptoms"], "not_for": d["not_for"]},
            queries=d["queries"], security={"cleared": True}, evidence_count=3)
        loop.store.save_skill(sk, triggers=d["queries"], snapshot=False)
    return len(json.loads(LIB.read_text()))


def evaluate(loop: SkillLoop, probes: dict) -> dict:
    per_class, records = {}, []
    lat = []
    for cls, items in probes.items():
        correct = 0
        for query, target in items:
            t0 = time.perf_counter()
            hits = [h["name"] for h in loop.recall(query, limit=3) if h["name"] != "_principles"]
            lat.append((time.perf_counter() - t0) * 1000)
            top = hits[0] if hits else None
            ok = (top == target) if target else (top is None)
            correct += ok
            records.append({"class": cls, "query": query, "expected": target, "got": top,
                            "all": hits[:3], "ok": ok})
        n = len(items)
        lo, hi = wilson(correct, n)
        per_class[cls] = {"correct": correct, "n": n, "rate": round(correct / n, 3),
                          "ci95": [round(lo, 3), round(hi, 3)]}

    pos = [r for r in records if r["expected"]]
    neg = [r for r in records if not r["expected"]]
    tp = sum(1 for r in pos if r["ok"])
    fn_abstain = sum(1 for r in pos if r["got"] is None)
    fp_wrong = sum(1 for r in pos if r["got"] and not r["ok"])
    fp_neg = sum(1 for r in neg if r["got"])
    precision = tp / (tp + fp_wrong + fp_neg) if (tp + fp_wrong + fp_neg) else 0.0
    recall = tp / len(pos) if pos else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"per_class": per_class,
            "overall": {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3),
                        "true_positive": tp, "wrong_skill": fp_wrong,
                        "missed_abstained": fn_abstain, "false_fire_on_negative": fp_neg,
                        "median_ms": round(sorted(lat)[len(lat) // 2], 2)},
            "records": records}


def run(sizes=SIZES, json_out=None):
    probes = json.loads(PROBES.read_text())
    results = {}
    for size in sizes:
        lp = SkillLoop(home=tempfile.mkdtemp(prefix="r1-"), llm=LLM(provider="fake"))
        n_real = build_library(lp)
        for d in make_distractors(size):
            lp.store.save_skill(d, triggers=list(d.queries), snapshot=False)
        results[size] = evaluate(lp, probes)
        results[size]["library"] = n_real + size

    hdr = f"{'class':<18}" + "".join(f"{'N=' + str(s):>16}" for s in sizes)
    print(hdr); print("-" * len(hdr))
    for cls in probes:
        row = f"{cls:<18}"
        for s in sizes:
            c = results[s]["per_class"][cls]
            row += f"{c['correct']}/{c['n']} [{c['ci95'][0]:.2f}-{c['ci95'][1]:.2f}]".rjust(16)
        print(row)
    print("-" * len(hdr))
    for m in ("precision", "recall", "f1", "median_ms"):
        print(f"{m:<18}" + "".join(f"{results[s]['overall'][m]:>16}" for s in sizes))

    print("\nFAILURES at N=0:")
    for r in results[sizes[0]]["records"]:
        if not r["ok"]:
            print(f"  [{r['class']}] {r['query'][:58]:<58} expected={r['expected']} got={r['got']}")

    if json_out:
        Path(json_out).write_text(json.dumps(results, indent=2) + "\n")
        print(f"\nwrote {json_out}")
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    out = None
    if "--json" in args:
        i = args.index("--json"); out = args[i + 1]; args = args[:i] + args[i + 2:]
    sys.exit(run(tuple(int(a) for a in args) if args else SIZES, out))
