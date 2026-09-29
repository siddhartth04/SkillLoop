"""Is an eval actually running, and how far has it got?

    python -m qa.conv_eval.status            # every seed it can find
    python -m qa.conv_eval.status 5          # one seed

Answers three questions without trusting anyone's summary:
  1. Is a process alive and writing? (the database mtime, not a claim)
  2. How far has each phase got? (episode counts against what the phase needs)
  3. Did it finish, and with what verdict? (read from the committed report)

A run that has stopped mid-phase says so, and says which phase, so "it is running" is never a guess.
"""
from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path

STATE = Path("/home/claude/conv_state")
RESULTS = Path(__file__).resolve().parent / "results"
NEEDED = {"control": 21, "control_repeat": 21, "oracle": 21, "train": 18,
          "memory": 21, "skillloop": 21, "combined": 21}
ORDER = ["control", "control_repeat", "oracle", "train", "memory", "skillloop", "combined"]


def _count(p: Path) -> int:
    if not p.exists():
        return 0
    return sum(1 for line in p.read_text(encoding="utf-8").splitlines() if line.strip())


def seed_status(seed: int) -> None:
    d = STATE / f"seed{seed}"
    report = RESULTS / f"full-seed{seed}.json"
    print(f"\n=== seed {seed} ===")

    if report.exists():
        r = json.loads(report.read_text(encoding="utf-8"))
        v = r.get("verdict", "(no verdict)")
        print(f"  FINISHED - report written {time.strftime('%Y-%m-%d %H:%M', time.localtime(report.stat().st_mtime))}")
        pri = r.get("primary") or {}
        for k in ("control", "memory", "skillloop", "combined", "oracle"):
            if k in pri:
                print(f"    {k:<14} {pri[k]['k']:>2}/{pri[k]['n']}")
        print(f"  verdict: {v}")
        return

    if not d.exists():
        print("  not started")
        return

    print("  phases:")
    for name in ORDER:
        n = _count(d / f"{name}.jsonl")
        need = NEEDED[name]
        mark = "done" if n >= need else ("...." if n else "    ")
        print(f"    {mark} {name:<16} {n:>2}/{need}")

    db = d / "skillloop" / "skillloop.db"
    if db.exists():
        age = time.time() - db.stat().st_mtime
        c = sqlite3.connect(str(db))
        proc = c.execute("SELECT COUNT(*) FROM traces WHERE processed=1").fetchone()[0]
        tot = c.execute("SELECT COUNT(*) FROM traces").fetchone()[0]
        sk = c.execute("SELECT COUNT(*) FROM skills").fetchone()[0]
        print(f"  learning: {proc}/{tot} traces processed, {sk} skills")
        state = "ACTIVE (written in the last 3 min)" if age < 180 else f"IDLE for {age/60:.0f} min"
        print(f"  database: {state}")
        if age >= 180:
            print("  -> nothing is writing. Either it finished this phase, or the run stopped.")
            print("     Re-run `python -m qa.conv_eval.resume %d`; cached episodes are reused." % seed)
    else:
        print("  learning has not started")


def main() -> int:
    args = [int(a) for a in sys.argv[1:] if a.isdigit()]
    if not args:
        seeds = sorted({int(p.name.replace("seed", "")) for p in STATE.glob("seed*") if p.is_dir()}
                       | {int(p.stem.split("seed")[1]) for p in RESULTS.glob("full-seed*.json")})
        args = seeds or [4]
    for s in args:
        seed_status(s)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
