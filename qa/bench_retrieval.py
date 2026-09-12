"""Offline retrieval benchmark. No LLM calls - runs in milliseconds so retrieval can be iterated on cheaply.

Ground truth per query: which skill SHOULD fire (good), and the abstain case where nothing should.
Measures precision@1, whether the right skill is in the returned set, and false-fire rate (the thing that
actually hurt us in QA: a skill firing on a task it has nothing to do with).
"""
from __future__ import annotations

import sys

sys.path.insert(0, "/home/claude/skillloop")
from skillloop import SkillLoop  # noqa: E402
from skillloop.llm import LLM  # noqa: E402

# (query, acceptable skill names or empty set = must abstain)
QUERIES: list[tuple[str, set[str]]] = [
    ("Update settings.json so retries becomes 5. Keep the file parseable.", {"json-edit-verify"}),
    ("In config.json, change the port from 8000 to 8080", {"json-edit-verify"}),
    ("Set up the 'humanize' library in this environment", {"handle-externally-managed-env"}),
    ("Install the python package tabulate", {"handle-externally-managed-env"}),
    ("test_tax.py is red. Make it green.", {"percentage-calculation"}),
    ("the VAT calculation returns the wrong amount", {"percentage-calculation"}),
    ("Append the comment '// checked' as the last line of each .js file here", {"verify-bulk-edit"}),
    ("Add the line '# reviewed' to the top of every .py file in this project", {"verify-bulk-edit"}),
    ("Create avg.py that prints the average of scores.txt, then put its output in avg.txt",
     {"verify-file-creation"}),
    ("Load cities.csv in python and write the city of the last row into last.txt",
     {"when-writing-a-python-one-liner-with-python-c-av", "when-writing-python-one-liner",
      "python-one-liner-csv", "csv-last-row-extraction"}),
    # --- must abstain: nothing in the library is about these ---
    ("What is the capital of France?", set()),
    ("Summarise this markdown document for me", set()),
    ("deploy.sh exits 0 but the artifact was never uploaded", set()),
    ("Change the cache size to 512 in server.ini, leaving the other settings alone", set()),
    ("Refactor the React component to use hooks", set()),
]


def run(home: str = "/home/claude/qa/library", limit: int = 2, verbose: bool = True) -> dict:
    lp = SkillLoop(home=home, llm=LLM(provider="fake"))
    # a skill in quarantine is deliberately not retrievable; scoring a query against it measures the gate,
    # not retrieval, so those queries are reported separately rather than counted as retrieval misses.
    eligible = {s.name for s in lp.store.all_skills() if s.status in ("active", "candidate")}
    p1 = fired_ok = abstain_ok = false_fire = 0
    n_should_fire = sum(1 for _, g in QUERIES if g)
    n_abstain = len(QUERIES) - n_should_fire
    rows = []
    skipped = []
    for q, good in QUERIES:
        if good and not (good & eligible):
            skipped.append(q[:40])
            n_should_fire -= 1
            continue
        hits = [h for h in lp.recall(q, limit=limit) if h["name"] != "_principles"]
        names = [h["name"] for h in hits]
        top = names[0] if names else None
        if good:
            if top in good:
                p1 += 1
            if set(names) & good:
                fired_ok += 1
            if top and top not in good:
                false_fire += 1
        else:
            if not names:
                abstain_ok += 1
            else:
                false_fire += 1
        rows.append((q[:52], good or {"(abstain)"}, names))
        if verbose:
            mark = "ok " if ((good and top in good) or (not good and not names)) else "MISS"
            print(f"  [{mark}] {q[:50]:52s} -> {names}")
    res = {
        "precision_at_1": f"{p1}/{n_should_fire}",
        "right_skill_in_set": f"{fired_ok}/{n_should_fire}",
        "abstained_correctly": f"{abstain_ok}/{n_abstain}",
        "false_fires": false_fire,
        "score": round((p1 + abstain_ok) / (n_should_fire + n_abstain), 3),
        "skipped_target_quarantined": skipped,
    }
    if verbose:
        print("  " + str(res))
    return res


if __name__ == "__main__":
    run(home=sys.argv[1] if len(sys.argv) > 1 else "/home/claude/qa/library")
