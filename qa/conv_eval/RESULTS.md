# Conventions eval: results

Model: `openai/gpt-oss-120b` on Groq (agent and SkillLoop learner), temperature 0, 2 attempts per task.
Protocol: `PROTOCOL.md`, fixed before any model run. Raw data: `results/`.

## Gates

| Seed | Null agreement | Control | Oracle | Harness errors | Included? |
|---|---|---|---|---|---|
| 101 (pilot) | 100% | 33% | 100% | 0 | pilot only |
| 1 | 83% | 50% | 100% | 0 | **No** (null < 85%) |
| 2 | 89% | 17% | 100% | 0 | **Yes** |
| 3 | 82% (14/17) | 50% | not run | 0 | **No** (null < 85%, cannot recover) |

Seed 3 passed its null gate (89%) in an earlier run that was lost when the process was stopped; the saved rerun
failed it. The gate sits near this model's own run-to-run noise.

## Primary result (seed 2, 18 held-out trap tasks)

| Condition | Solved (2 attempts) | First try | vs control (paired) |
|---|---|---|---|
| Control | 3/18 = 17% | 6% | |
| Control, repeated | 5/18 = 28% | 6% | |
| Simple memory | 10/18 = 56% | 44% | 7 wins / 0 losses, p = 0.016 |
| **SkillLoop** | **11/18 = 61%** | 22% | **8 wins / 0 losses, p = 0.008** |
| Oracle | 18/18 = 100% | 100% | |

SkillLoop vs simple memory: 4 wins / 3 losses, p = 1.0.
No-trap tasks (harm check): SkillLoop 3/3, memory 3/3, control 3/3.

**Verdict (pre-registered wording):** SkillLoop beats control (p<0.05), but NOT significantly better than simple
memory, so its extra machinery is not yet justified.

## By convention (seed 2)

| Convention | Control | Memory | SkillLoop |
|---|---|---|---|
| cents | 1/3 | 3/3 | 3/3 |
| commit | 0/3 | 1/3 | 2/3 |
| inplace | 2/3 | 3/3 | 2/3 |
| keys | 0/3 | 0/3 | 0/3 |
| tuple | 0/3 | 3/3 | 1/3 |
| units | 0/3 | 0/3 | 3/3 |

## Why it came out this way

The headline ("not better than simple memory") is true but hides the mechanism. Splitting the same 18 tasks
by **what the training feedback actually revealed** separates them cleanly:

| Training feedback | n | control | memory | SkillLoop |
|---|---|---|---|---|
| A traceback naming the problem | 9 | 3/9 | **9/9** | 6/9 |
| Only "check failed" | 9 | 0/9 | 1/9 | **5/9** |

Where a training attempt crashed, the learner saw a traceback and the fix is in the recorded code, so raw past
attempts are very hard to beat — memory scored 9/9. Where an attempt merely failed its check, the learner was
told "check failed" and nothing more: **the correct value was never shown to it.** Every recorded attempt is
wrong, so copying them cannot help, and memory collapses to 1/9. That is the case a learned skill exists for,
and there SkillLoop wins 4–0 head-to-head (p = 0.125, n = 9 — directional, not significant).

Two skills from this run show the whole mechanism:

- **units (3/3, memory 0/3)** — the traces never revealed that timeouts are milliseconds, so the skill wrote a
  procedure to *find out*: `probe = new_job('sync', 1)`, read `probe._eff()`, compare, then convert. Correct
  for every timeout value, not just the ones seen.
- **keys (0/3)** — the traces never revealed that option keys are `UPPER-KEBAB-CASE`, and the skill **guessed**:
  "use `'retries'`". The real key is `'MAX-RETRIES'`. The agent followed the skill and lost all three attempts,
  even though the documented `read_config()` returns `{'MAX-RETRIES': 0, 'LOG-LEVEL': 'info'}` and would have
  answered it immediately.

Same pipeline, opposite outcomes: one encoded a way to discover the unknown, the other asserted it. The
synthesis prompt forbade inventing commands and paths but said nothing about inventing **the unknown value at
the centre of the lesson**, which is exactly the thing the evidence usually fails to establish. It now carries
an UNKNOWN VALUES rule: when the evidence does not fix the value, write the procedure that discovers it.

**That change is untested against a real model** — it was written after this run and needs a fresh seed to
confirm. Reproduce this analysis on any committed result with:

```bash
python -m qa.conv_eval.analyze
```

## Observations

- 7 of SkillLoop's 11 successes came on the second attempt: error-time recall does much of the work. On the first
  attempt, raw past examples (memory) are stronger than skills (44% vs 22%).
- The two methods succeed on different conventions, so combining them is the obvious next experiment.
- SkillLoop learned two config-key skills yet solved 0/3 config-key tasks: likely wrong skill content. See
  `results/seed2_learned_skills/`.
- Earlier runs showed near-duplicate skills and one verification step referring to an attribute that does not exist.
- SkillLoop's learning is token-expensive; free-tier daily token caps (200k/org/day) were the main operational limit.

## Limits

One valid seed (n = 18), one model, one domain (Python library conventions). The convention pool was written by
someone who knows SkillLoop's design; the simple-memory baseline receives identical information, which is why
beating it, not control, is the claim that matters.
