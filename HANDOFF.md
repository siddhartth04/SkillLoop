# SkillLoop — session handoff (2026-09-12)

Paste this into a new session along with `skillloop-v0_4.zip` and `humaneval-plus-screen-evidence.zip`.

**One-line status:** SkillLoop is demonstrated on shell/text tasks across two model families at p<0.05.
Cross-domain transfer has been attempted three times and measured zero times. The HumanEval+ rig that would
settle it is built, validated, and resumable — it is blocked only on API credit.

---

## 1. Results that hold

Domain 1 (shell/text). Paired control-vs-treatment, independent checkers, statistics fixed before results
were seen. Regenerate any row with `python qa/report_ab.py evidence/<file>`.

| Agent | Control | Treatment | Delta | Discordant pairs | Sign-test p | Verdict |
|---|---|---|---|---|---|---|
| gpt-oss-20b | 1/9 (11%) | 9/9 (100%) | +0.89 | 8 (8–0) | **0.0078** | clears p<0.05 |
| qwen3.8-27b | 3/9 (33%) | 9/9 (100%) | +0.67 | 6 (6–0) | **0.0312** | clears p<0.05 |

- Both rest on **2 distinct skills** (`unicode-case-conversion`, `structured-config-edit`), so one lucky skill
  is not carrying the result.
- Replication across two independent model families is the strongest single fact in the project.
- Raw data: `evidence/ab.json`, `evidence/ab_qwen.json`, screening in `evidence/necessity.json`.
- Test suite: **66/66 pass.**

### Two caveats that must travel with these numbers
1. **Selection effect by design.** Measured only on tasks the baseline fails >=60% of the time. This shows a
   skill *can* help, not how often one applies. Pass-rate over a representative workload is unknown.
2. **The margin shrinks as agents get stronger** (control 11% -> 33% from gpt-oss-20b to qwen3.8-27b).
   Report per-model, never pooled into one figure.

---

## 2. Results that were retracted

**The original domain-2 screen was an instrument artifact.** It reported p3_timezone 0/3, p5_greedy_regex 1/3,
p6_rounding 0/3 — three apparently eligible tasks. All 8 of those "failures" were harness errors: the agent
emitted native tool calls, the provider rejected them, and the recovery path mangled the payload. On a working
native-tool-calling harness those same tasks pass 3/3.

The old `HANDOFF.md` claim that float-equality / timezone / mutation tasks were eligible was **wrong** and has
been corrected. It also referenced `qa/run_e5.py`, `qa/agent_native.py`, `qa/necessity.py`,
`qa/verify_checkers.py` as existing when they were not in the zip; they exist now.

**Rule this establishes: a 0/3 is a smell, not a find. Read traces before trusting any number.**

---

## 3. Negative but informative

**E5 on six toy traps: 18/18 pass, 0 harness errors, 0 eligible tasks.** Verified beforehand that an empty dir
fails every checker, the classic buggy version fails (trap really bites), and the reference passes. The traps
are simply too easy for gpt-4o-mini — nothing failed, so nothing to learn from and no A/B. Cost $0.008.

**HumanEval+ baseline pass-1: 130/163 = 79.8%** (gpt-4o-mini). Matches published figures (~79–85%) —
independent evidence the harness measures the real benchmark. All 32 failures audited from traces: every one
wrote `solution.py`; 31 are genuine wrong answers (e.g. `make_palindrome` fails `'xyx'`); 1 is a plus-set
timeout needing a serial re-check.

**Dataset bug found: HumanEval/32 is excluded.** EvalPlus 0.3.1's `find_zero` special oracle fails its own
reference solution (0/888). Undetected, it would have read as a permanently-hard problem. 163 problems remain.

---

## 4. Never measured

**Cross-domain transfer.** Three attempts, zero measurements:
1. Harness bug -> invalid (retracted above).
2. Toy traps too easy -> no eligible tasks.
3. HumanEval+ -> **out of credits mid-screen**; 0 of 32 candidates have the 3 runs eligibility requires.

No eligible set, no A/B, no p-value. **Do not make any cross-domain claim.**

---

## 5. Current blocker

OpenRouter account exhausted: `total_credits: 0`, `total_usage: $0.196`. Error is explicit —
*"You requested up to 1500 tokens, but can only afford 1194."* Small calls still return 200, so
**`openrouter.ai` reachability is fine**; this is purely billing.

**Do NOT work around this by lowering `max_tokens`.** It changes the protocol mid-experiment and truncated
completions would manufacture exactly the fake failures that invalidated attempt #1.

### Two ways forward

**(a) Add ~$0.20 OpenRouter credit** (~150 runs at observed ~$0.001/run; $1 for headroom). Resumes from the
162 completed screening runs. No code changes.

**(b) Switch to Groq free tier** (`api.groq.com` confirmed reachable, on the allowlist). Arguably better:
- A weaker agent (`openai/gpt-oss-20b`) fails more problems -> more eligible -> **more statistical power**.
  This matters: at 79.8% baseline only ~30 problems fail and half land in TEST, possibly too few to reach
  p<0.05 even if the effect is real.
- Same agent as the domain-1 headline result -> **"same agent, new domain"** is a far cleaner transfer claim.
- Free, so no mid-run credit cliff.
- **Cost:** the 162 gpt-4o-mini screening runs become unusable (a screen measures *that* agent's failures;
  mixing agents across arms is a confound). Fresh screen ~230 runs. Keep the gpt-4o-mini data as a separate,
  honestly-labelled baseline measurement.
- **Verify first:** that gpt-oss-20b does native tool calling correctly through this harness (one live run +
  `qa/verify_checkers.py`) before launching ~230 runs.

---

## 6. How to resume

```bash
export OPENAI_BASE_URL=https://openrouter.ai/api/v1   # or https://api.groq.com/openai/v1
export OPENAI_API_KEY=<key>
export SKILLLOOP_PROVIDER=openai SKILLLOOP_MODEL=openai/gpt-4o-mini
export HE_SCRATCH=/home/claude/qa/he_scratch HE_WORKERS=4

python qa/verify_checkers.py          # instrument sanity, offline
python qa/run_he.py screen            # resumable; finishes 32 candidates x2
python qa/run_he.py learn             # LEARN split only; library then frozen
python qa/run_he.py ab                # control / placebo / treatment x3, interleaved
python qa/report_ab.py <HE_OUT>/he_ab.json <HE_OUT>/he_skills.json
python qa/report_ab.py <HE_OUT>/he_placebo.json <HE_OUT>/he_skills.json   # secondary
```
If switching agent: set `E5_AGENT_MODEL`, point `HE_OUT` at a **fresh** directory, re-screen from scratch.

Long runs die when a foreground poll hits the tool timeout — launch detached:
`setsid nohup python3 qa/run_he.py screen >> log 2>&1 < /dev/null &`, then poll in <300s increments.

---

## 7. Experimental design (pre-registered, do not change silently)

- **Dataset:** HumanEval+ (EvalPlus 0.3.1, HumanEvalPlus v0.1.10), ~80x more tests/problem than HumanEval.
  SWE-bench Verified would be stronger but is infeasible here (per-repo Docker images, `ghcr.io` not on the
  allowlist, 1 CPU).
- **Checker:** EvalPlus's own `check_correctness`; pass <=> base AND plus both PASS. Validated across all 164:
  canonical solution passes all, `return None` stub fails all.
- **Split:** seeded (SEED=20260911) 81 LEARN / 82 TEST. SkillLoop sees LEARN traces **only** — skills must
  TRANSFER to unseen problems, not memorize the problem they're graded on.
- **Three arms:** control (bare prompt) / **placebo** (fixed generic "careful Python" checklist) / treatment
  (recall output). The placebo exists so a win can't be explained by "more text in the prompt."
- **Intention-to-treat:** eligible TEST problems stay in even when recall abstains.
- **Eligible:** baseline passes <=1 of 3 screening runs.
- **Frozen library:** no learning during the A/B; recall once per problem, same text every repeat.
- **Errors:** harness/API errors re-run (max 2 retries), **never scored**; checker TIMEOUT re-checked serially
  (1 CPU + EvalPlus time limits scaled from reference runtimes = concurrent grading can fake a failure).

---

## 8. Statistical caveat for the finish line

At 79.8% baseline the eligible TEST set may be too small for the sign test to reach p<0.05 even if SkillLoop
genuinely helps. Minimum achievable p is `2/2^n` for n discordant pairs, so **n>=6 pairs are required**.
If the eligible set comes in small, the fix is **adding MBPP+** (378 problems, same checker, ~one line) —
not stretching a thin result.

---

## 9. How to describe this project honestly

> SkillLoop turns real agent failures into reusable skills. On tasks a baseline agent reliably fails,
> attaching learned skills raised success from 11% to 100% (gpt-oss-20b, p=0.0078) and 33% to 100%
> (qwen3.8-27b, p=0.0312), across 2 independent skills with pre-registered statistics and independent
> verification. Validated so far on shell/text tasks; the coding-domain evaluation on HumanEval+ is built
> and in progress.

Do **not** say "it works" unqualified — that implies generality the evidence doesn't support.

The strongest asset here is not the p-value: **this project caught its own false positives twice** (the 0/3
harness artifact, and the EvalPlus reference-solution bug). Lead with that rigor; the modest scope then reads
as discipline rather than weakness.

---

## 10. Files

- `skillloop-v0_4.zip` — full code. New: `qa/agent_native.py` (native tool calling), `qa/he_check.py`
  (EvalPlus checker), `qa/run_he.py` (HumanEval+ runner), `qa/verify_checkers.py`, `qa/run_e5.py`.
  `qa/report_ab.py` gained an optional skill-map arg (domain-1 output byte-identical — verified).
  `CALIBRATION.md` and `HANDOFF.md` updated with everything above.
- `humaneval-plus-screen-evidence.zip` — `state.json`, all 383 raw traces, full screen log.
- **Security:** no API key appears in either zip (checked). The OpenRouter key was pasted in plaintext in chat
  and **should be rotated.**
