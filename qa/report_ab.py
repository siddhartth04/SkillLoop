"""Paired A/B report over the necessity-screened tasks. Statistics fixed before results were seen."""
import json, math, sys

SKILL = {"n7_upper": "unicode-case-conversion", "n8_lower": "unicode-case-conversion",
         "n9_yaml": "structured-config-edit", "n10_silent": "verify-side-effect-not-exit-code"}


def main(path="/home/claude/qa/ab.json", skills_path=None):
    """skills_path (optional): JSON {task_id: skill_name} of the skill ACTUALLY recalled per task. Entries
    starting with "(" (e.g. "(abstained)") or "_principles" are not independent skills and are not counted."""
    r = json.load(open(path))
    if skills_path:
        SKILL.update(json.load(open(skills_path)))
    tasks = sorted({k.split("|")[0] for k in r})
    wins = losses = C = T = n = 0
    skills = set()
    print(f'{"task":12s} {"skill":34s} control  treat')
    for t in tasks:
        c, tr = r.get(t + "|control", []), r.get(t + "|treat", [])
        if not c or not tr:
            print(f'{t:12s} {SKILL.get(t,"?"):34s} {sum(c)}/{len(c) or 0}      (arm incomplete)')
            continue
        print(f'{t:12s} {SKILL.get(t,"?"):34s} {sum(c)}/{len(c)}      {sum(tr)}/{len(tr)}')
        sk = SKILL.get(t, t)
        if not (sk.startswith("(") or sk == "_principles"):
            skills.add(sk)
        C += sum(c); T += sum(tr); n += min(len(c), len(tr))
        for a, b in zip(c, tr):
            if a != b:
                wins += 1 if b else 0
                losses += 1 if a else 0
    nd = wins + losses
    p = 1.0 if nd == 0 else min(1.0, 2 * sum(math.comb(nd, i) for i in range(0, min(wins, losses) + 1)) / 2 ** nd)
    print(f"\ncontrol {C}/{n} = {C/n:.2f}    treat {T}/{n} = {T/n:.2f}    delta +{(T-C)/n:.2f}")
    print(f"discordant pairs {nd} (treat {wins}, control {losses})   sign-test p = {p:.4f}")
    print(f"distinct skills: {len(skills)} -> {sorted(skills)}")
    ok = p < 0.05 and wins > losses and len(skills) >= 2
    print("VERDICT:", "treatment better (p<0.05, >=2 independent skills)" if ok
          else "insufficient evidence - do not claim an effect")
    print("\nScope: measured ONLY on tasks the baseline agent fails >=60% of the time without the skill.")
    print("This is a selection effect BY DESIGN - it tests whether a skill can help, not how often one applies.")
    print("It is not a claim about all tasks, and the margin narrows as the agent gets stronger")
    print("(gpt-oss-20b control 11% -> qwen3.8-27b control 33%). Report per-model, never as one figure.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
