"""Why a run came out the way it did, not just what the score was.

    python -m qa.conv_eval.analyze                      # every committed result file
    python -m qa.conv_eval.analyze results/full-seed2.json

Seed 2 produced a headline ("SkillLoop does not beat simple memory") that was true but not the whole story.
Splitting the same tasks by how much the TRAINING feedback actually revealed changes the picture, and points
at the mechanism instead of the score:

  * When a training attempt crashed, the learner saw a traceback naming the problem. Raw past attempts are
    then hard to beat: the fixed code is right there to copy.
  * When a training attempt merely failed its check, the learner saw "check failed" and nothing else. The
    correct value was never shown to it. Copying past attempts cannot help, because every recorded attempt is
    wrong. A skill can still help, but only if it encodes a way to FIND the value rather than asserting one.

That is the split this script measures, and it is what the UNKNOWN VALUES rule in the synthesis prompt is
meant to act on. Underpowered on one seed; run more seeds before treating it as established.
"""
from __future__ import annotations

import collections
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sign_test(w: int, l: int) -> float:
    n = w + l
    if not n:
        return 1.0
    k = min(w, l)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def informative_conventions(seed_dir: Path) -> collections.Counter:
    """How many training attempts per convention produced a real traceback rather than a bare check failure."""
    info: collections.Counter = collections.Counter()
    train = seed_dir / "train.jsonl"
    if not train.exists():
        return info
    for line in train.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        t = json.loads(line)
        q = t["task_id"].split("-")[1]
        for a in t["attempts"]:
            if a["verdict"]["stage"] == "code":      # a traceback: the agent was told what broke
                info[q] += 1
    return info


def solved(eps: dict, cond: str, tid: str) -> bool:
    e = eps[cond][tid]
    return bool(e["attempts"]) and e["attempts"][-1]["verdict"]["ok"]


def analyse(path: Path) -> None:
    d = json.loads(path.read_text(encoding="utf-8"))
    eps = d.get("episodes") or {}
    if not eps:
        print(f"{path.name}: no episodes (seed excluded by a gate)")
        return
    seed = list(d.get("per_seed", {}))[0] if d.get("per_seed") else "?"
    ep_dir = HERE / "results" / f"seed{seed}_episodes"
    info = informative_conventions(ep_dir)
    conds = [c for c in ("control", "memory", "skillloop", "combined") if c in eps]
    trap = [t for t in eps["control"] if "-notrap" not in t]

    print(f"\n=== {path.name} (seed {seed}) ===")
    if not info:
        print("  no training episodes recorded; cannot split by feedback quality")
    else:
        rich = [t for t in trap if info[t.split("-")[1]] > 0]
        poor = [t for t in trap if info[t.split("-")[1]] == 0]
        print(f"\n  Split by what the TRAINING feedback revealed:")
        print(f"  {'':<34}" + "".join(f"{c:>12}" for c in conds))
        for label, group in (("traceback shown  (n=%d)" % len(rich), rich),
                             ("only 'check failed' (n=%d)" % len(poor), poor)):
            row = "".join(f"{str(sum(solved(eps,c,t) for t in group)) + chr(47) + str(len(group)):>12}" for c in conds)
            print(f"  {label:<34}{row}")
        if poor and "memory" in conds and "skillloop" in conds:
            w = sum(1 for t in poor if solved(eps,"skillloop",t) and not solved(eps,"memory",t))
            l = sum(1 for t in poor if solved(eps,"memory",t) and not solved(eps,"skillloop",t))
            print(f"\n  Where the answer was never shown: skillloop vs memory "
                  f"{w}-{l}, p = {sign_test(w,l):.4f}")
            print("  (This is the case skills exist for: past attempts are all wrong, so copying cannot help.)")

    print(f"\n  By convention:")
    hdr = f"  {'convention':<12}{'tracebacks':>11}" + "".join(f"{c:>12}" for c in conds)
    print(hdr + "\n  " + "-" * (len(hdr) - 2))
    agg = {c: collections.defaultdict(lambda: [0, 0]) for c in conds}
    for c in conds:
        for t in trap:
            q = t.split("-")[1]
            agg[c][q][0] += solved(eps, c, t); agg[c][q][1] += 1
    for q in sorted(agg[conds[0]]):
        cells = "".join(f"{str(agg[c][q][0]) + chr(47) + str(agg[c][q][1]):>12}" for c in conds)
        print(f"  {q:<12}{info[q]:>11}" + cells)

    if "combined" in conds:
        print(f"\n  Combined condition (skills + raw attempts):")
        for other in ("memory", "skillloop"):
            if other in conds:
                w = sum(1 for t in trap if solved(eps,"combined",t) and not solved(eps,other,t))
                l = sum(1 for t in trap if solved(eps,other,t) and not solved(eps,"combined",t))
                print(f"    vs {other:<10} {w}-{l}, p = {sign_test(w,l):.4f}")


def main() -> int:
    args = sys.argv[1:]
    paths = [Path(a) for a in args] if args else sorted((HERE / "results").glob("full-*.json"))
    if not paths:
        print("no result files found"); return 1
    for p in paths:
        analyse(p if p.is_absolute() or p.exists() else HERE / p)
    print("\nOne seed is not a result. Run more with: python -m qa.conv_eval.resume <SEED>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
