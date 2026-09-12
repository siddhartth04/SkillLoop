"""S1: does retrieval hold at library scale?

Every retrieval number so far came from a library of 2-6 skills. IDF, the abstention threshold and breadth
normalisation all depend on corpus statistics, so they can behave very differently at 100+. This builds a
synthetic library of realistic distractor skills around the real ones and re-measures precision and
abstention as the library grows.
"""
import sys, time
sys.path.insert(0, "/home/claude/skillloop")
from skillloop import SkillLoop
from skillloop.llm import LLM
from skillloop.store import Skill
from skillloop.playbook import new_bullet

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


def run(sizes=(0, 20, 50, 100, 200)):
    lp = SkillLoop(home="/home/claude/qa/lib_scale", llm=LLM(provider="fake"))
    real = [s for s in SkillLoop(home="/home/claude/qa/lib_claude", llm=LLM(provider="fake")).store.all_skills()]
    for s in real:
        s.status = "candidate"
        lp.store.save_skill(s, triggers=[], snapshot=False)
    added = 0
    print(f"{'library':>8} {'hit@1':>7} {'abstain':>8} {'falsefire':>10} {'ms/query':>9}")
    for size in sizes:
        while added < size:
            for s in make_distractors(size - added)[: size - added]:
                lp.store.save_skill(s, triggers=[], snapshot=False)
                added += 1
        hit = abst = false = 0
        t0 = time.time()
        for q, want in PROBES:
            names = [h["name"] for h in lp.recall(q, limit=2) if h["name"] != "_principles"]
            top = names[0] if names else None
            if want:
                hit += 1 if top == want else 0
                false += 1 if (top and top != want) else 0
            else:
                abst += 1 if not names else 0
                false += 1 if names else 0
        ms = (time.time() - t0) * 1000 / len(PROBES)
        n_real = sum(1 for _, w in PROBES if w)
        n_abs = len(PROBES) - n_real
        print(f"{added + len(real):>8} {hit}/{n_real:<5} {abst}/{n_abs:<6} {false:>10} {ms:>9.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(run())
