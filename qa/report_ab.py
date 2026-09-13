"""Paired A/B report over necessity-screened tasks. Statistics fixed before results were seen.

CORRECT UNIT OF ANALYSIS: the TASK, not the individual trial. Three repeated runs of the same task with the
same skill are NOT independent observations - they share the task, the skill, and the agent's tendencies on
that task. Treating each trial as an independent pair inflates n and understates the p-value. So we:
  1. reduce each task's repeated trials to a per-task outcome (majority success rate per arm),
  2. run the sign test over TASKS that flip direction (control-fail -> treat-succeed),
  3. report the honest p, which with only a handful of tasks will be modest even for a perfect split.

A per-TRIAL p is also printed, clearly labelled as anti-conservative, for transparency only.
"""
import json, math, os, sys

# Fallback map used only if no --skills file is given. Prefer recording the recalled skill in the data.
SKILL = {"n7_upper": "unicode-case-conversion", "n8_lower": "unicode-case-conversion",
         "n9_yaml": "structured-config-edit", "n10_silent": "verify-side-effect-not-exit-code"}

DEFAULT = os.path.join(os.path.dirname(__file__), "..", "evidence", "ab.json")


def _sign_p(wins, losses):
    nd = wins + losses
    if nd == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(nd, i) for i in range(0, min(wins, losses) + 1)) / 2 ** nd)


def main(path=DEFAULT, skills_path=None):
    r = json.load(open(path))
    if skills_path:
        SKILL.update(json.load(open(skills_path)))
    tasks = sorted({k.split("|")[0] for k in r})

    rows = []                      # (task, skill, control_rate, treat_rate, n_c, n_t)
    incomplete = []
    for t in tasks:
        c, tr = r.get(t + "|control", []), r.get(t + "|treat", [])
        if not c or not tr:
            incomplete.append((t, len(c), len(tr)))
            continue
        rows.append((t, SKILL.get(t, t), sum(c) / len(c), sum(tr) / len(tr), len(c), len(tr)))

    print(f'{"task":12s} {"skill":34s} control   treat')
    for t, sk, cr, trr, nc, nt in rows:
        print(f'{t:12s} {sk:34s} {int(cr*nc)}/{nc}      {int(trr*nt)}/{nt}')
    for t, nc, nt in incomplete:
        print(f'{t:12s} {"(arm incomplete)":34s} control={nc} treat={nt}  -- EXCLUDED')

    # ---- TASK-LEVEL sign test (the honest one) ----
    task_wins = task_losses = 0
    for _, _, cr, trr, _, _ in rows:
        # a task "flips" if treatment's success rate beats control's (and vice-versa)
        if trr > cr:
            task_wins += 1
        elif cr > trr:
            task_losses += 1
    p_task = _sign_p(task_wins, task_losses)

    # ---- per-TRIAL sign test (anti-conservative; shown for transparency only) ----
    trial_wins = trial_losses = 0
    for t, _, _, _, _, _ in rows:
        c, tr = r[t + "|control"], r[t + "|treat"]
        for a, b in zip(c, tr):
            if a != b:
                trial_wins += 1 if b else 0
                trial_losses += 1 if a else 0
    p_trial = _sign_p(trial_wins, trial_losses)

    skills = {sk for _, sk, _, _, _, _ in rows if not (sk.startswith("(") or sk == "_principles")}
    n_tasks = len(rows)

    print()
    print(f"TASKS analysed: {n_tasks}   (independent unit of analysis)")
    print(f"  task-level sign test: {task_wins} tasks favour treatment, {task_losses} favour control"
          f"  ->  p = {p_task:.3f}   [THE NUMBER TO REPORT]")
    print(f"  per-trial sign test:  {trial_wins} vs {trial_losses}  ->  p = {p_trial:.4f}"
          f"   [anti-conservative: trials within a task are NOT independent — do not report this]")
    print(f"  distinct skills exercised: {len(skills)} -> {sorted(skills)}")

    # honest verdict: require task-level significance
    significant = p_task < 0.05 and task_wins > task_losses
    if significant:
        verdict = f"treatment better at the task level (p={p_task:.3f})"
    elif task_wins > task_losses and n_tasks < 8:
        verdict = (f"DIRECTION favours treatment ({task_wins}/{n_tasks} tasks) but UNDERPOWERED: "
                   f"n={n_tasks} tasks cannot reach p<0.05 (a perfect split bottoms out near p={p_task:.2f}). "
                   f"Run more distinct tasks to earn a claim.")
    else:
        verdict = "insufficient evidence - do not claim an effect"
    print(f"\nVERDICT: {verdict}")

    print("\nScope: measured ONLY on tasks the baseline agent fails >=60% of the time without the skill "
          "(selection effect by design). Not a claim about all tasks; the margin narrows as the agent "
          "gets stronger (gpt-oss-20b control 11% -> qwen3.8-27b control 33%).")
    return 0 if significant else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))