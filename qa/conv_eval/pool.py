"""Pool per-seed result files into the pre-registered analysis (seeds that failed a gate contribute nothing)."""
import glob, json, sys
from .run import paired, rate, verdict, print_report
from .harness import Episode

def load(paths):
    allc = {}
    for p in paths:
        r = json.load(open(p))
        for cond, eps in r.get("episodes", {}).items():
            for i, e in eps.items():
                ep = Episode(e["task_id"], e["condition"], e["attempts"], e["harness_error"], e.get("hints_shown", []))
                allc.setdefault(cond, {})[i] = ep
    return allc

if __name__ == "__main__":
    paths = sys.argv[1:] or sorted(glob.glob("qa/conv_eval/results/full-*.json"))
    allc = load(paths)
    trap = [i for i in allc.get("skillloop", {}) if "-notrap" not in i]
    notrap = [i for i in allc.get("skillloop", {}) if "-notrap" in i]
    r = {}
    if trap:
        r["primary"] = {k: rate(v, trap) for k, v in allc.items()}
        r["first_try"] = {k: rate(v, trap, first=True) for k, v in allc.items()}
        r["harm_notrap"] = {k: rate(v, notrap) for k, v in allc.items()}
        r["paired"] = {"skillloop_vs_control": paired(allc["skillloop"], allc["control"], trap),
                       "skillloop_vs_memory": paired(allc["skillloop"], allc["memory"], trap),
                       "memory_vs_control": paired(allc["memory"], allc["control"], trap)}
        r["verdict"] = verdict(r)
    else:
        r["verdict"] = "no valid seed"
    print(f"pooled files: {paths}")
    print_report(r)
