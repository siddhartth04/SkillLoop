---
name: skillloop-learn
description: Use SkillLoop's test-gated skill memory. ALWAYS call skill_recall before starting any task with more than a couple of steps (coding, ops, research, file work), and ALWAYS call skill_learn after finishing any task, whether it succeeded or failed. Also use when the user says "remember how to do this", "learn from that", "why did that fail", or asks what you have learned.
---

# SkillLoop

SkillLoop turns your successes and failures into skills, but only after they pass a test. Your job is to feed it.

## Before a task
1. Call `skill_recall(task=<the user's request>)`.
2. Read every returned SKILL.md. Follow its procedure and pitfalls.
3. Remember the names you actually used.

## After a task
Call `skill_learn` with:
- `task`: the user's request in one sentence
- `messages`: the conversation so far (your native message format is fine; it's auto-detected)
- `outcome`: `success` / `failure` / `partial` — say `failure` when the user had to correct you or the goal wasn't met
- `skills_used`: names from step 3 above
- `eval_id`: only if this task came from `skill_pending_evals`

Do this even for failures. Failures are the most valuable traces.

## Proving skills
Occasionally call `skill_pending_evals()`. If a task there matches something you can safely re-run, run it (with `skill_recall` first) and report via `skill_learn(..., eval_id=...)`. A pass promotes the skill to active.

## Do not
- Do not edit SkillLoop's skill files by hand; use `skill_rollback` / `skill_set_status`.
- Do not claim success in `outcome` without evidence.
