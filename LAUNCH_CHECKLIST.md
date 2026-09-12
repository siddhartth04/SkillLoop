# Launch checklist

Exit criteria for a public launch that claims SkillLoop improves agent success. Each item states what
"done" means as a **measurement**, not an opinion. Nothing here is checked off on vibes.

Status legend: `[ ]` not started · `[~]` in progress · `[x]` done · `[!]` blocked

---

## P0 — evidence for the headline claim (blocks launch)

- [x] **E1. One skill with a demonstrated causal effect.**
  Done: n7_upper 0/3 → 3/3, n8_lower 1/3 → 3/3. Control 1/6, treatment 6/6. 5 discordant pairs, sign test
  p = 0.0625 (the floor for 5 pairs). Mechanism traced: the agent ran the skill's procedure verbatim.

- [~] **E2. Two more independent skills A/B tested.**
  `structured-config-edit`: DONE — n9_yaml control 0/3 → treatment 3/3.
  `verify-side-effect-not-exit-code`: control 2/3 (borderline on the >=60% necessity threshold), treatment
  arm not run. Either find a harder variant of that task or drop it from the claim.

- [x] **E3. p < 0.05 across independent skills.**
  Control 1/9 (11%) → treatment 9/9 (100%). 8 discordant pairs, all favouring treatment.
  **Sign test p = 0.0078**, 2 distinct skills. `qa/report_ab.py` prints the verdict and exits non-zero if the
  bar is not met, so this cannot be fudged later.

- [x] **E4. Replication on a second agent model.** DONE — and it held.
  `qwen/qwen3.8-27b` (different family), SAME skills, no re-teaching: control 3/9 (33%) → treatment 9/9 (100%),
  6 discordant pairs all favouring treatment, **p = 0.0312** standing alone.
  The predicted shrinkage is visible and worth stating: qwen's control passed 3/9 where gpt-oss-20b passed
  1/9, so a stronger agent does already know some of this. The margin narrows; it does not vanish.
  Combined across both models: 14 discordant pairs, 14-0, p = 0.00012.

- [!] **E5. A second task domain — BLOCKED on the harness, not the product.**
  Six Python coding tasks built and verified winnable (`qa/tasks4.py`). Screening looked clean (3 tasks at
  0/3 or 1/3) but inspection showed **8 of 8 failing traces were harness errors, 0 were real failures**:
  writing a multi-line file makes the agent emit a native tool call that Groq rejects, so nothing executed.
  Added a `write_file` action and hardened tool-call recovery; still insufficient.
  Unblocks when: the harness uses the provider's native tool-calling API instead of JSON-in-content.
  Until then no cross-domain claim may be made. Domain-1 results are unaffected.

## P1 — correctness at scale (blocks launch)

- [x] **S1. Retrieval at 100+ skills.** `qa/bench_scale.py` grows the real library with realistic distractor
  skills (docker, git, npm, postgres, k8s, aws, pytest, nginx, redis, webpack) and re-probes.
  | library | hit@1 | abstain | false fires | ms/query |
  |---|---|---|---|---|
  | 2 | 4/4 | 3/3 | 0 | 0.9 |
  | 52 | 4/4 | 3/3 | 0 | 3.5 |
  | 202 | 4/4 | 3/3 | 0 | 8.6 |
  No precision loss to 200 skills. Latency grows linearly (a full scan) — fine at this size, and the point
  where an index is needed is now measured rather than guessed.

- [x] **S2. Concurrency.** FIXED after an external reviewer found a shared-connection bug my own test missed
  (it passed only because the GIL serialised writes in one process). Now one SQLite connection PER THREAD
  (`threading.local`, WAL, busy_timeout, synchronous=NORMAL). Regression test uses a real ThreadPoolExecutor:
  8x25 -> 200/200 stored, 0 errors. `test_concurrent_writes_thread_local_connections`.

- [x] **S3. Trace cap is a hard invariant.** External reviewer: a 2.4M-char trace sanitised to ~670k, over
  the 200k cap. Now reduced in a loop with a closing assertion; verified on three pathological shapes.
  `test_sanitizer_enforces_hard_cap`.

## P4 — production hardening (external reviewer punch list)

- [x] **PR1. QA portability.** Full suite passes on a fresh clone under `env -i` (empty environment),
  no developer-specific paths (all were in `qa/`, excluded from the wheel). 46 passed, 1 skipped.
- [x] **PR2. Observability.** `skillloop/observability.py`: `SKILLLOOP_LOG=json|text`, per-operation
  `request_id`, latency, `skill.promote`/`demote` with version ids, `trace.sanitized` stats. `/metrics`.
- [x] **PR3. Deployment.** Dockerfile (non-root, volume, HEALTHCHECK), env-var config, `/health` + `/version`,
  SIGTERM/SIGINT graceful shutdown, schema versioning + online backup/restore (`skillloop/migrate.py`),
  documented resource limits. See `DEPLOYMENT.md`.
- [x] **PR4. Security red team.** `tests/test_redteam.py`: 19 cases — instruction override, env exfiltration,
  remote code fetch, secrecy, destructive commands, role hijack, hidden HTML, secret redaction, size-cap
  flooding, and a full-loop poisoned trace that must not reach an active skill. All pass.
- [!] **PR5. Real-LLM 300-task benchmark (Baseline vs Memory vs SkillLoop).** THE remaining scientific
  question. Needs an OpenRouter model and budget; the harness (`qa/necessity.py`, three arms) is ready.
  Blocks the "demonstrably improves real agents" claim, not the experimental-product launch.

## P2 — adoption (should not block, but hurts without)

- [x] **A1. Recorder.** `loop.session(...)` — 7 tests.
- [~] **A2. Claude Code adapter.** Verified at PROTOCOL level, not against a real Claude Code install
  (no access to one from the dev environment — stated plainly rather than implied).
  `qa/test_claude_code_hook.py` drives the hook with realistic `UserPromptSubmit` and `Stop` payloads and a
  JSONL transcript: transcript parsed, tool errors preserved, re-entry guard honoured, exit 0 throughout.
  Bug found and fixed: the hook dispatched on `argv` only, so a `settings.json` missing the `prompt`/`stop`
  argument made it silently do nothing and exit 0. It now derives the mode from `hook_event_name`, warns on
  a mismatch, and reports transcript problems on stderr instead of failing silently.
  Re-verified against the current hooks reference (code.claude.com/docs/en/hooks, Sep 2026). Confirmed:
  common envelope `session_id / transcript_path / cwd / hook_event_name`; `prompt` on UserPromptSubmit;
  `stop_hook_active` + `last_assistant_message` on Stop/SubagentStop. Three further fixes from the docs:
    - **Prompt-injection defences.** UserPromptSubmit stdout is added to context, and text framed as
      out-of-band system commands makes Claude Code surface it to the user instead of using it. The old
      wrapper (`<skillloop_recall> ... Follow them`) was exactly that shape. Now phrased as project
      information ("Notes from earlier work in this project..."), with a test asserting no banned framing.
    - `last_assistant_message` used as a fallback when the transcript tail is empty.
    - Documented that `--continue` / `--resume` replays saved hook text rather than re-running the hook,
      so recalled skills can be stale in a resumed session.
  Added `skillloop_hook.py doctor` — validates connectivity and settings.json registration in one command.
  REMAINING (yours, ~10 min): run it in a real Claude Code session and confirm a trace lands.
- [x] **A3. Quickstart runs end to end on a clean machine.** `qa/test_quickstart.py`, executed against the
  packaged artifact in a fresh venv: session records a real failure -> learning produces a gated skill ->
  recall returns it with a Verification section -> exported as SKILL.md -> per-bullet feedback recorded.

## P3 — release hygiene

- [x] **H1. Packaging.** stdlib-only core, optional extras, `qa/`+`tests/` excluded, version 0.3.0.
- [x] **H2. Honest README.** Status banner states the negative results plainly.
- [x] **H3. Calibration log.** Every run recorded, including the ones that failed.
- [ ] **H4. CONTRIBUTING + issue templates.**
- [ ] **H5. Tag v0.3.0, publish `CALIBRATION.md` as the evidence page.**

---

## Known risks, stated up front

1. **The effect may be model-specific** (E4). Most likely failure mode.
2. **One skill is carrying the entire result.** Until E2/E3 land, we have one lesson shown twice.
3. **Necessity screening is a selection effect by design.** It is the right design for measuring whether a
   skill *can* help, but the honest framing of any headline number is "on tasks the agent fails without the
   skill", never "on all tasks".
4. **Free-tier API throttling** has cost more hours than the engineering. Budget is the schedule risk.
