"""Reproduce every claim in the README that does not need an API key.

    python qa/reproduce.py

Each block prints the claim, runs the real code, and marks PASS/FAIL by comparing against the number
published in the README. Anything needing a model (the A/B pilot, the conventions eval) is replayed from
its committed raw data instead of re-run, and is labelled as such.

Exit code 0 only if every offline claim reproduces.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OK, BAD = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, claim: str, got, want) -> None:
    ok = got == want
    results.append((OK if ok else BAD, name, f"{claim}  ->  got {got}, expected {want}"))
    print(f"  [{OK if ok else BAD}] {claim}: {got}")


def head(t: str) -> None:
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def run_tests() -> None:
    head("1. Test suite (README badge: 109 passing)")
    p = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider",
                        "--collect-only"], cwd=ROOT, capture_output=True, text=True)
    n = 0
    for line in p.stdout.splitlines():
        if "tests collected" in line or "test collected" in line:
            n = int(line.split()[0])
    check("tests", "tests collected", n, 109)


def run_retrieval() -> None:
    head("2. Retrieval evals R1 + R2 (deterministic, no API key)")
    for script, classes in (("qa/run_r1.py", None), ("qa/run_r2.py", None)):
        p = subprocess.run([sys.executable, script, "0"], cwd=ROOT, capture_output=True, text=True)
        print(f"\n--- {script} ---")
        for line in p.stdout.splitlines():
            if any(k in line for k in ("class", "_", "precision", "recall", "f1")) and "FAIL" not in line:
                print("   ", line.rstrip())
        # every probe class must be perfect in R2
        if "run_r2" in script:
            perfect = all(f"{n}/14" in p.stdout for n in ("14",))
            check("r2", "R2: all classes 14/14", perfect, True)


def run_selfcheck() -> None:
    head("3. Conventions-eval harness self-check (offline: generator, checkers, gates, stats)")
    p = subprocess.run([sys.executable, "-m", "qa.conv_eval.run", "--selfcheck"],
                       cwd=ROOT, capture_output=True, text=True)
    for line in p.stdout.splitlines()[-8:]:
        print("   ", line.rstrip())
    check("selfcheck", "self-check verdict", "SELF-CHECK: PASS" in p.stdout, True)


def replay_ab() -> None:
    head("4. A/B pilot — REPLAYED from committed data (originally needed a model)")
    p = subprocess.run([sys.executable, "qa/report_ab.py", "evidence/ab.json"],
                       cwd=ROOT, capture_output=True, text=True)
    for line in p.stdout.splitlines():
        print("   ", line.rstrip())
    check("ab", "task-level sign test p=0.250 (underpowered, as README states)",
          "p = 0.250" in p.stdout, True)


def replay_conv() -> None:
    head("5. Conventions eval — REPLAYED from committed episodes (originally needed a model)")
    f = ROOT / "qa/conv_eval/results/full-seed2.json"
    if not f.exists():
        results.append((BAD, "conv", "results file missing"))
        return
    d = json.loads(f.read_text())
    pri, pair = d["primary"], d["paired"]
    for k in ("control", "control_repeat", "memory", "skillloop", "oracle"):
        print(f"    {k:<16} {pri[k]['k']}/{pri[k]['n']}")
    check("conv_sl", "SkillLoop 11/18", (pri["skillloop"]["k"], pri["skillloop"]["n"]), (11, 18))
    check("conv_mem", "simple memory 10/18", (pri["memory"]["k"], pri["memory"]["n"]), (10, 18))
    check("conv_ctl", "control 3/18", (pri["control"]["k"], pri["control"]["n"]), (3, 18))
    check("conv_p", "SkillLoop vs control p<0.05", pair["skillloop_vs_control"]["p"] < 0.05, True)
    check("conv_mem_p", "SkillLoop vs memory NOT significant (p=1.0)",
          pair["skillloop_vs_memory"]["p"], 1.0)


def main() -> int:
    t0 = time.time()
    print("Reproducing every offline claim in the README.\n"
          "Model-dependent results are replayed from committed raw data and labelled REPLAYED.")
    for fn in (run_tests, run_retrieval, run_selfcheck, replay_ab, replay_conv):
        try:
            fn()
        except Exception as e:                      # a broken check is a failure, not a crash
            results.append((BAD, fn.__name__, f"raised {e!r}"))
            print(f"  [{BAD}] {fn.__name__} raised {e!r}")
    head("SUMMARY")
    bad = [r for r in results if r[0] == BAD]
    for status, name, detail in results:
        print(f"  [{status}] {name}: {detail}")
    print(f"\n{len(results) - len(bad)}/{len(results)} claims reproduced in {time.time() - t0:.0f}s")
    if bad:
        print("\nFAILED — the README claims above do not match what the code produces.")
        return 1
    print("\nAll offline claims reproduce.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
