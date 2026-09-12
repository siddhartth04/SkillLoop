"""Independent checker for HumanEval+ tasks: EvalPlus's own check_correctness (base + plus tests).

  python3 qa/he_check.py <task_id> <solution.py>
Exit 0 iff base AND plus both PASS. Prints status and counts; on failure, the first failing input (truncated) so
the learner can reflect on it. Exit 3 = TIMEOUT (a possible instrument artifact on a loaded 1-CPU box - callers
re-check serially instead of scoring it).
"""
import sys
from pathlib import Path

from evalplus.data import get_human_eval_plus, get_human_eval_plus_hash
from evalplus.evaluate import check_correctness, get_groundtruth

PASS, TIMEOUT = "pass", "timeout"
_P = _GT = None


def data():
    global _P, _GT
    if _P is None:
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            _P = get_human_eval_plus()
            _GT = get_groundtruth(_P, get_human_eval_plus_hash(), [])
    return _P, _GT


def check(task_id: str, code: str) -> tuple[int, str]:
    P, GT = data()
    prob = P[task_id]
    r = check_correctness("humaneval", 0, prob, code, GT[task_id], fast_check=False)
    lines, code_out = [], 0
    for part in ("base", "plus"):
        stat, details = r[part]
        n = len(prob[f"{part}_input"])
        ok = sum(bool(x) for x in details)
        lines.append(f"{part}: {stat} ({ok}/{n} passed)")
        if stat != PASS:
            code_out = 3 if (stat == TIMEOUT and code_out == 0) else (code_out or 1)
            if stat != TIMEOUT:
                idx = next((i for i, x in enumerate(details) if not x), len(details))
                if idx < n:
                    lines.append(f"  first failing {part} input #{idx}: {repr(prob[f'{part}_input'][idx])[:200]}")
    return code_out, "\n".join(lines)


if __name__ == "__main__":
    tid, sol = sys.argv[1], Path(sys.argv[2])
    if not sol.exists():
        print(f"{sol} does not exist")
        sys.exit(1)
    rc, msg = check(tid, sol.read_text())
    print(msg)
    sys.exit(rc)
