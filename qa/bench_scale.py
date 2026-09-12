"""S1: does retrieval hold at library scale?

Every retrieval number so far came from a library of 2-6 skills. IDF, the abstention threshold and breadth
normalisation all depend on corpus statistics, so they can behave very differently at 100+. This builds a
synthetic library of realistic distractor skills around the real ones and re-measures precision and
abstention as the library grows.

Reproducible from a clean checkout: the real skills are seeded from qa/fixtures/real_skills.json and the
library is built in a temp dir, so no machine-local path is required.

  python qa/bench_scale.py                 # default sizes
  python qa/bench_scale.py 0 50 200 500    # explicit sizes
  python qa/bench_scale.py --json out.json # machine-readable, for evidence/
"""
import json
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from skillloop import SkillLoop
from skillloop.llm import LLM
from skillloop.store import Skill
from skillloop.playbook import new_bullet

FIXTURES = HERE / "fixtures" / "real_skills.json"


def load_real_skills():
    """The real, gated skills from the E3/E4 runs, committed as a fixture so this bench is reproducible."""
    out = []
    for d in json.loads(FIXTURES.read_text()):
        out.append(Skill(
            name=d["name"], description=d["description"], body="", status="active",
            bullets=[new_bullet(sec, txt) for sec, txt in d["bullets"]],
            facets=d["facets"], queries=d["queries"],
            security={"cleared": True}, evidence_count=2))
    return out


DOMAINS = [
    ("docker-{}", ["docker build", "container image", "dockerfile"], ["build the image", "container won't start"]),
    ("git-{}", ["git rebase", "merge conflict", "branch"], ["rebase my branch", "resolve a merge conflict"]),
    ("npm-{}", ["npm install", "node modules", "package.json"], ["install node deps", "npm fails to install"]),
    ("postgres-{}", ["postgres query", "sql migration", "index"], ["run a migration", "slow query"]),
    ("k8s-{}", ["kubectl deploy", "pod crashloop", "helm chart"], ["deploy to the cluster", "pod keeps restarting"]),
    ("aws-{}", ["s3 bucket", "iam policy", "lambda"], ["upload to s3", "fix an iam permission"]),
    ("pytest-{}", ["pytest fixture", "flaky test", "test collection"], ["fix a flaky test", "tests won't collect"]),
    ("nginx-{}", ["nginx config", "reverse proxy", "tls cert"], ["reload nginx", "set up a proxy"]),
    ("redis-{}", ["redis cache", "key expiry", "eviction"], ["cache keeps missing", "set a ttl"]),
    ("webpack-{}", ["webpack bundle", "source map", "tree shaking"], ["bundle is huge", "source maps missing"]),
]


def make_distractors(n):
    out = []
    for i in range(n):
        pat, applies, queries = DOMAINS[i % len(DOMAINS)]
        name = pat.format(i)
        out.append(Skill(
            name=name, description=f"Handle {applies[0]} tasks reliably.", body="", status="candidate",
            bullets=[new_bullet("procedure", f"Run the standard {applies[0]} workflow."),
                     new_bullet("verification", f"`{applies[0].split()[0]} --version` exits 0.")],
            facets={"applies_when": applies, "symptoms": [], "not_for": []},
            queries=[f"{q} ({i})" for q in queries]))
    return out


PROBES = [
    ("Write the uppercase version of the text in name.txt into upper.txt", "unicode-case-conversion"),
    ("make city.txt lowercase", "unicode-case-conversion"),
    ("In conf.yaml set replicas to 4. The file must stay valid YAML.", "structured-config-edit"),
    ("Change the timeout to 90 under the server section of settings.yaml", "structured-config-edit"),
    ("What is the capital of France?", None),
    ("Refactor the React component to use hooks", None),
    ("summarise this markdown document", None),
]


def run(sizes=(0, 20, 50, 100, 200, 500), json_out=None):
    tmp = tempfile.mkdtemp(prefix="skillloop-scale-")
    lp = SkillLoop(home=tmp, llm=LLM(provider="fake"))
    real = load_real_skills()
    for s in real:
        lp.store.save_skill(s, triggers=list(s.queries), snapshot=False)

    rows = []
    added = 0
    print(f"{'library':>8} {'hit@1':>8} {'abstain':>8} {'falsefire':>10} {'ms/query':>9}")
    for size in sorted(sizes):
        while added < size:
            for d in make_distractors(size)[added:size]:
                lp.store.save_skill(d, triggers=list(d.queries), snapshot=False)
                added += 1
        hit = abst = false_fire = wrong_skill = 0
        t0 = time.time()
        for q, want in PROBES:
            names = [h["name"] for h in lp.recall(q, limit=2) if h["name"] != "_principles"]
            top = names[0] if names else None
            if want:
                if top == want:
                    hit += 1
                elif top:
                    wrong_skill += 1
            else:
                if names:
                    false_fire += 1
                else:
                    abst += 1
        ms = (time.time() - t0) * 1000 / len(PROBES)
        n_real = sum(1 for _, w in PROBES if w)
        n_abs = len(PROBES) - n_real
        lib = added + len(real)
        print(f"{lib:>8} {hit}/{n_real:<6} {abst}/{n_abs:<6} {false_fire:>10} {ms:>9.2f}")
        rows.append({"library": lib, "distractors": added, "hit_at_1": hit, "n_positive": n_real,
                     "abstained": abst, "n_negative": n_abs, "false_fires": false_fire,
                     "wrong_skill": wrong_skill, "ms_per_query": round(ms, 3)})

    if json_out:
        Path(json_out).write_text(json.dumps(
            {"bench": "S1-retrieval-at-scale", "probes": len(PROBES), "real_skills": [s.name for s in real],
             "rows": rows}, indent=2) + "\n")
        print(f"\nwrote {json_out}")

    degraded = [r for r in rows if r["hit_at_1"] < rows[0]["hit_at_1"] or r["false_fires"] > rows[0]["false_fires"]]
    if degraded:
        print(f"\nDEGRADED at library sizes: {[r['library'] for r in degraded]}")
        return 1
    print("\nNo degradation in hit@1 or false fires across the tested range.")
    return 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:]]
    out = None
    if "--json" in args:
        i = args.index("--json")
        out = args[i + 1]
        args = args[:i] + args[i + 2:]
    sizes = tuple(int(a) for a in args) if args else (0, 20, 50, 100, 200, 500)
    sys.exit(run(sizes, out))
