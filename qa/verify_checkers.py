"""Assert every E5 task is winnable AND the trap is real, through the exact harness executor.

For each task:
  1. empty workdir            -> checker must FAIL
  2. naive buggy solution     -> checker must FAIL   (the trap is detectable)
  3. reference solution       -> checker must PASS   (the task is winnable) via execute('run', heredoc)
  4. reference via write_file -> checker must PASS   (the native write path works)
  5. END-TO-END: run_task() driven by a scripted fake client that emits native tool_calls with the reference
     file as JSON arguments (multi-line, quotes) -> must PASS. This is the exact path that broke last time.
Exit non-zero on any violation.
"""
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parent))
import agent_native as A  # noqa: E402
from tasks4 import DOMAIN2, SOLUTIONS_D2  # noqa: E402

NAIVE = {
    "p1_mutable_default": ("cart.py", "def add_item(item, basket=[]):\n    basket.append(item)\n    return basket\n"),
    "p2_float_equality": ("money.py", "def is_equal(a, b):\n    return a == b\n"),
    "p3_timezone": ("when.py", "from datetime import datetime\ndef to_utc_hour(s):\n    return datetime.fromisoformat(s).hour\n"),
    "p4_mutate_while_iterating": ("clean.py", "def drop_negatives(nums):\n    for n in nums:\n        if n < 0:\n            nums.remove(n)\n    return nums\n"),
    "p5_greedy_regex": ("tags.py", "import re\ndef first_tag(text):\n    return re.search(r'<.*>', text).group(0)\n"),
    "p6_rounding": ("split.py", "def split_bill(total_cents, people):\n    return [round(total_cents / people)] * people\n"),
}


def heredoc(cmd):
    m = re.match(r"cat > (\S+) <<'EOF'\n(.*)\nEOF$", cmd, re.S)
    return m.group(1), m.group(2) + "\n"


class ScriptedClient:
    """Fake OpenAI client: turn 1 write_file(reference), turn 2 run(checker-ish), turn 3 finish."""
    def __init__(self, path, content):
        self.script = [("write_file", {"path": path, "content": content}),
                       ("run", {"command": f"python3 -c \"import {path[:-3]}\" && echo imported"}),
                       ("finish", {"summary": "done"})]
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kw):
        name, args = self.script.pop(0)
        tc = NS(id=f"c{len(self.script)}", function=NS(name=name, arguments=json.dumps(args)))
        return NS(choices=[NS(message=NS(content="", tool_calls=[tc]))], usage=None)


def main():
    bad = 0
    for t in DOMAIN2:
        tid, row = t["id"], []
        wd = A.setup(t, "verify_empty"); row.append(("empty fails", A.sh(t["checker"], wd)[0] != 0))
        wd = A.setup(t, "verify_naive"); A.write_file(wd, *NAIVE[tid])
        row.append(("naive fails", A.sh(t["checker"], wd)[0] != 0))
        wd = A.setup(t, "verify_ref_run"); A.execute(wd, "run", {"command": SOLUTIONS_D2[tid]})
        row.append(("ref(run) passes", A.sh(t["checker"], wd)[0] == 0))
        path, content = heredoc(SOLUTIONS_D2[tid])
        wd = A.setup(t, "verify_ref_wf"); A.execute(wd, "write_file", {"path": path, "content": content})
        row.append(("ref(write_file) passes", A.sh(t["checker"], wd)[0] == 0))
        res = A.run_task(t, "verify_e2e", cl=ScriptedClient(path, content))
        row.append(("e2e native passes", res["passed"] and res["harness_error"] is None and res["finished"]))
        ok = all(v for _, v in row)
        bad += not ok
        print(f"{tid:28s} {'OK ' if ok else 'BAD'}  " + "  ".join(f"{k}={'y' if v else 'N'}" for k, v in row))
    print("ALL TASKS WINNABLE, TRAPS REAL, NATIVE PATH WORKS" if not bad else f"{bad} TASK(S) FAILED VERIFICATION")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
