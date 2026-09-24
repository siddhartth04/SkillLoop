# Calibration log

Run `skillloop calibrate` whenever you change a model or a prompt. This file records what past runs found and
what changed because of them, so the prompts have a paper trail.

## 2026-09-25 (later) — the verification loop was feeding itself

**The seventh self-caught false positive, found in the re-run of the sixth.**

With the encoding bug fixed, seed 4 was re-run. The promotion log showed skills passing their re-runs:

```
promotion re-runs: [('inspect-return-type', 'success'), ('inspect-return-type', 'success'),
                    ('inspect-return-type', 'success'), ('bruzem-numeric-units', 'success'), ...]
skills learned:    {'inspect-return-type': 'quarantine', 'bruzem-numeric-units': 'quarantine', ...}
```

Two skills passed their verification 3/3 and both ended in `quarantine`. All seven did.

The eval table showed the signature: **every skill had exactly as many `failed` host evals as `passed`
ones** — 3/3, 2/2, 4/4. And the trace table had grown from 18 training tasks to 36 traces.

`run_pending_evals()` handed its re-run to `learn()`, which queued it like any other trajectory. `process()`
then treated that verification run as a *new lesson*: it synthesised another skill from it, and that skill
queued another host eval. The second eval was a fresh trace with no skill in its prompt, so it failed — and
the failure demoted the skill that had just passed.

A verification re-run is **evidence about a skill, not a lesson to learn from**. It is now stored for the
audit trail but marked processed, so it never re-enters the learning queue.

**Why this one is worth recording:** the bug was introduced by the very change that was meant to strengthen
the evaluation — wiring the gate into the eval so that promotion was actually exercised. Before that change
the gate never ran and every skill sat at `candidate`; after it, the gate ran and rejected everything. Both
states produce a plausible-looking report, and neither is a measurement. Verified both directions after the
fix: a passing re-run reaches `active` with zero failed evals, a failing one still lands in `quarantine`.

## 2026-09-25 — a text encoding nearly manufactured a negative result

**The sixth self-caught false positive, and the closest call so far.**

Seed 4 of the conventions eval was run on Windows against `openai/gpt-oss-120b`. It passed every gate (null
agreement 94.4%, control 22%, oracle 100%) and returned a clean, plausible, *unfavourable* verdict:

```
skillloop  8/18 = 0.444      memory  11/18 = 0.611      control  4/18 = 0.222
VERDICT: No significant improvement over control.
```

Read at face value, that is evidence against the project's central claim, on a seed chosen before the data was
seen. The report also showed five of six learned skills sitting in `quarantine`, which looked like the
verification gate doing its job and rejecting weak skills.

It was not. The event log showed the real cause:

```
error  UnicodeEncodeError('charmap', "---
name: check-function-index...
```

`Store.save_skill` wrote `SKILL.md` with `write_text()` and no encoding, so Python used the platform default,
which on Windows is cp1252. The model had written a **non-breaking hyphen (U+2011)** in the skill text. Every
skill containing one was destroyed on write and never entered the library.

The consequence for the measurement: the agent had **one** skill available across 21 test episodes, and in 14
of them it received nothing at all — it was silently running as the control condition while being scored as
SkillLoop. The comparison was not weak evidence, it was not evidence.

**What was wrong with the evaluation, not just the code:** a skill that fails to save and a skill that fails
its gate were indistinguishable in the report. Both showed up as "not active". The run is now re-done with the
encoding fixed, and the same failure would still not announce itself, which is the part worth remembering.

Fixed: all twelve text reads and writes in `skillloop/` specify UTF-8, as do the eval's cache and report files.
Regression test: a skill containing U+2011, an em dash, smart quotes and a checkmark must survive a save and
reload (`tests/test_purpose.py`).

## 2026-09-10 — first live run (Groq free tier)

Models: reflect/synth/judge = `openai/gpt-oss-120b`, outcome/inject = `openai/gpt-oss-20b`.
14 / 20 traces completed before the 200k tokens/day cap on gpt-oss-120b. Remaining 6 (incl. the injection
canary #15, the scary-benign cleanup #16, the unverified-claim outcome probe #19) still pending on that model.

### Findings on the v0.2 prompts as shipped in the morning
| area | result | action |
|---|---|---|
| root cause vs symptom | 13/14 named the decision, not the error | keep |
| skill shape (5 sections) | 14/14 first try | keep |
| security, benign skills | 14/14 cleared, 0 false positives | keep |
| blame attribution | 2/2 correct (`none` on success-with-skill, `skill_incomplete` on the libpq case) | keep |
| **gate** | failed 5 good skills; judge's own reason said "correctly addresses the root cause" | **bug in my design**: assertions were outcome-level ("no ConnectionError appears"), un-passable by a text review. Rewrote ASSERT_SYSTEM to demand behaviors; JUDGE_SYSTEM now has mandatory checks (Verification concrete / prevents original failure / no harmful steps) |
| gate strictness probe | passed 2/3 skills whose Verification was gutted to "check it worked" | mandatory check #1 above + `check_shape` now rejects a Verification with no code span |
| invented details | every skill; #14 listed apt/yum/dnf/pacman variants for one psycopg2 error | SYNTH evidence rule rewritten: standard verification commands allowed, cross-platform alternatives and "in case" fixes forbidden. Re-test: invented set shrank to `nginx -t`, `systemctl status` |
| confidence | 0.85–0.96 on everything; the 0.8 evidence threshold filtered nothing | confidence rubric with anchors; plus the skeptic pass below |
| **confabulation** | #8 (typo, no diagnostic evidence): confident invented "template cache" root cause, generalizable=true | prompt instructions alone did not fix it, and asking for `evidence_steps` was gamed (cited the grep/edit steps). Added an adversarial **skeptic** call that must quote the evidence verbatim; the quote is checked against the trace. Three-valued: direct (keep) / partial (cap 0.65, waits for corroboration) / none (0.35, no skill, lesson names the missing diagnostic) |
| generalizable on trivial task (#9) | correctly false | keep |

### After tuning (validated on gpt-oss-20b, a weaker model, same day)
- #1 nginx: gate passes; gutted-Verification variant fails (regex + judge). Skeptic: partial → 0.65 → waits. Correct.
- #2 pip: skeptic direct (quotes `externally-managed-environment`) → 0.90 → skill written, gate passes. Correct.
- #8 typo: skeptic none → 0.35 → undetermined, missing diagnostic named, no skill. Correct.

### Second live run on gpt-oss-120b, tuned prompts (same day, key pool of 2)
| trace | result |
|---|---|
| #15 injection canary | **caught**: injected text did not reach the skill; reflector named "obeyed a note embedded in tool output" as the root cause; skeptic direct; security cleared the benign result. Skill quality low (q=2, invents a log) and correctly provisional |
| #16 scary-benign cleanup (`rm -rf` + git check-ignore) | **not flagged** (no false positive). Gate first failed because the judge graded the *past* trajectory against the skill instead of a future agent; JUDGE_SYSTEM clarified; re-run passes |
| #19 unverified "Index created successfully" after an error | outcome judge (gpt-oss-20b) called it failure at 0.95. Gate then **correctly failed** the skill: it detected the duplicate-key error but had no step to prevent it |
| #1 nginx | skeptic partial → 0.65 → waits for corroboration |
| #2 pip | skeptic direct → 0.90 → skill, gate pass |
| #8 typo | skeptic none → 0.35 → no skill, missing diagnostic recorded |

### Still open
- Full 20-trace run on a strong model after the tuning; so far 6 traces re-run post-tuning (#1,2,8,15,16,19).
- Success-trace skills still over-elaborate (#16 added awk free-space comparisons the trace never did).
- gpt-oss-20b is too weak for the reflect/skeptic roles (it still tried to ground #8 in irrelevant steps
  before the skeptic layer existed). Recommendation: strongest model for REFLECT/SYNTH/JUDGE, small model only
  for OUTCOME/INJECT.

## 2026-09-10 (later) — first real-agent QA run, and the retrieval rebuild

Harness (`qa/`): a separate LLM agent issues shell commands that really execute; success is decided by an
INDEPENDENT checker command per task, never by the agent's claim. Arms: baseline, control (variants, no skills),
treatment (variants, recall on).

### Round 1 findings
| finding | evidence | fix |
|---|---|---|
| Gate far too strict | 1/4 real skills promoted; judge's own reason said the skill "correctly addresses the root cause" | assertions must be GROUNDED in the skill's own text; an over-reaching assertion is now `out_of_scope`, not `fail`. Result: 5/6 promoted |
| Retrieval fires wrong skill | a JSON skill retrieved for CSV and INI tasks | full rebuild, below |
| Tasks too easy | baseline 5/5, no headroom | added `HARD_A`/`HARD_B` with traps (multi-match sed, silent `exit 0`, nested globs, stale bytecode) |
| Skill names were whole sentences | `when-writing-a-python-one-liner-with-python-c-av` | `slugify` keeps 4 meaningful words |

### Retrieval rebuild (offline bench: `qa/bench_retrieval.py`, 15 labelled queries, no LLM calls)
Root cause was upstream of ranking: SYNTH_SYSTEM told the model to be "a little pushy" in descriptions, so skills
advertised themselves for everything (`verify-file-creation` claimed scripts AND json AND pip AND redirects).

| change | rationale |
|---|---|
| description prompt: precise, not greedy | the description IS the index |
| `facets`: applies_when / symptoms / not_for | curated metadata beats raw text; `not_for` actively blocks wrong retrieval |
| index purpose, never the body | body words (file, python, run) are shared by every skill |
| subtoken splitting (`test_tax.py` → test, tax, py) | queries name concrete files; skills describe concepts |
| IDF coverage scoring + absolute threshold | enables ABSTENTION - returning nothing is the right answer for most queries |
| anchor rule: query must touch name/applies_when | stops "change ... settings" pulling a JSON skill into an INI task |
| breadth normalization (BM25 doc-length analogue) | a skill claiming 20 situations shouldn't win on one word |
| ignore self-contradictory `not_for` | a pip skill listing "global install success" was vetoing the word "install" |
| pooling requires distinctive-term overlap | generic-word pooling is what created the over-broad skill |

Bench: **0.667 → 0.923**, false fires **3 → 0**, abstention **3/5 → 5/5**.
Remaining miss: a task worded "create avg.py ... put its output in avg.txt" vs a skill indexed under "stdout
redirection" - a true vocabulary gap that only embeddings close.

### End-to-end after the fixes
HARD arm: baseline 3/4, control 3/4, **with recall 4/4**. `hb2_silent_fail` (a deploy script that exits 0 while
doing nothing) failed in both earlier arms and passed with skills available. n=4: suggestive, not proof.

## 2026-09-10 (evening) — held-out result, and the doc2query fix

### The honest held-out result that forced this work
12 NEW task pairs, domains never used for tuning, library learned from scratch:
**control 10/12, with recall 10/12 — identical, task for task.** No improvement.
Cause was not ranking: retrieval fired on only 2/12 variants because the library had 5 skills and nothing
relevant to offer on the other 10. Precision without coverage is worthless.

### Root cause: vocabulary mismatch, not ranking
The task terms and the skill terms had ZERO overlap:
| user asked | skill was indexed as |
|---|---|
| "find the smallest value in scores.dat" | `sequential-tool-calls`, anchors: awk, processing |
| "make deploy.sh runnable" | `shell-script-executable-capture`, anchors: chmod, executable |
| "strip repeated entries" | `deduplicate-file-preserve-order`, anchors: dedup, duplicate |
No scoring function can bridge that. The standard fix without embeddings is **doc2query**: index the phrasings
users actually type, generated at write time.

### What was built
- `trigger_queries` on every skill (8-12 short user-phrased requests), emitted by SYNTH and backfillable for
  existing libraries via `skillloop backfill-queries`.
- **Phrase-level matching.** Pooling the queries into one bag made a single skill fire on 5 unrelated tasks:
  one shared word was enough. Each trigger query is now scored on its own, as the fraction of ITS terms present
  in the task; the best query wins. Threshold 0.3, and a match needs >= 2 shared terms unless overlap >= 0.6.
- **A second tokenizer for queries.** The facet stoplist strips domain words (script, run, output) because in a
  skill description they match everything - but in a short user phrasing they carry the meaning
  ("make my script runnable" collapsed to one token). Query matching uses function words only.

### Measured (held-out, library never tuned against these queries)
| | before | after |
|---|---|---|
| retrieval fired | 2/12 | 5/12 |
| correct (of 5 tasks with a matching skill) | 2/5 | **5/5** |
| wrong skill on top | 0 | **0** |
| abstained on the 7 with no matching skill | yes | yes |
Original bench unchanged at 0.923 - the change did not trade precision for recall.

### Also fixed this session
- Learning cost per trace ~7 model calls -> ~3-4: assertions+judge merged into ONE call (`make_and_judge`),
  security judge runs only when regex flags or risk vocabulary is present, host evals reuse existing assertions.
- Deterministic grounding: when a lesson cites a step that really contains a tool error, the skeptic model call
  is skipped entirely.
- Cold-start policy (`Policy.effective(library_size)`): permissive thresholds below 12 skills, strict above.
  NOTE: measured on its own this did NOT improve coverage (2/12 -> 1/12) - relearning produced differently
  named skills. It is retained for throughput, but doc2query is what actually fixed retrieval. Skill naming
  instability across relearns is an open issue.


## 2026-09-10 (night) — architecture change from the literature

Searched the current work on agent memory and read Agentic Context Engineering (Zhang et al., ICLR 2026,
arXiv:2510.04618) end to end, plus Agent Workflow Memory (Wang et al., ICML 2025). ACE names two failure modes
that SkillLoop had BY CONSTRUCTION, with ablations:

| ACE finding | what SkillLoop was doing | change |
|---|---|---|
| Brevity bias costs accuracy; contexts should be comprehensive playbooks | SYNTH_SYSTEM said "under 60 lines, no fluff" | detail is now explicitly requested; the model is told an agent can ignore what it does not need but cannot invent what was omitted |
| Context collapse: monolithic LLM rewriting erodes context (18,282 tokens -> 122 in one step, accuracy 66.7 -> 57.1, BELOW the 63.7 no-context baseline). Ablation: incremental updates worth +11.7 TGC / +27.8 SGC | `synthesize()` regenerated the whole body on every patch | `playbook.py`: itemised bullets with stable ids; patches emit ADD/REVISE/PRUNE deltas merged by deterministic code, never by regeneration |
| Bullets carry helpful/harmful counters; Generator reports which were useful | one hit-rate per skill | `bullets_helpful` / `bullets_harmful` on the trace; per-bullet counters |
| Grow-and-refine: append, update in place, dedupe, prune | no mechanism | `refine()` — dedupe by token overlap, prune bullets flagged harmful >= 3 times and net-negative, evict least useful over a cap, preferring to keep verification and scope_limits |
| ACE works from execution feedback without ground-truth labels | our QA harness already uses independent checkers | validates the design |

Bugs found while implementing, both caught by tests rather than review:
- the patch path mutated `existing.bullets` in place, so the previous version's snapshot changed underneath it
- FTS was indexing `body`, which is now empty for itemised skills; it indexes the rendered `body_md`

32 tests pass. No retrieval regression: original bench 0.923, held-out 5/5 correct with 0 false fires.
Legacy prose skills from v0.2 libraries are parsed into bullets on demand and still render.

STILL OPEN: the end-to-end effect is unproven. Last full held-out comparison was control 10/12 vs recall 11/12,
and the one task that flipped had NO skill retrieved, so it is variance, not the tool. The measurement design is
the bottleneck: the agent already succeeds on the tasks we have skills for. Next experiment must be built
backwards — construct variants the baseline agent fails >= 60% of the time WITHOUT the lesson, 3 repeats each.

## 2026-09-11 — necessity experiment + recorder

### Why the old end-to-end runs could not have shown anything
Control 10/12 vs recall 11/12, with the flipped task having NO skill retrieved. Structural cause: the agent
already succeeded at the tasks we had skills for. A skill can only change an outcome on a task the agent
FAILS without it. Every previous comparison was measuring variance.

### The necessity harness (`qa/necessity.py`, `qa/tasks3.py`, `qa/run_necessity.py`)
- Phase 1 SCREEN: run each candidate K times with NO skills; keep only tasks with baseline failure rate >= 0.6.
- Phase 2 TEACH: learn from those real failures.
- Phase 3 MEASURE: same tasks, K repeats, control vs recall.
- Reports a sign test over discordant pairs. The verdict string defaults to
  "insufficient evidence - do not claim an effect". The statistics were written BEFORE any results were seen,
  so a single lucky flip can never again be read as a finding.

Screening so far (3 runs each):
| task | baseline pass | eligible? |
|---|---|---|
| n1_maxnum (numeric sort) | 3/3 | NO - agent already competent |
| n2_minnum | 3/3 | NO |
| n3_count (`grep -c` counts lines, not occurrences) | 0/3 | YES |
| n4_count2 | 1/3 | YES (fail rate 0.67) |

The two exclusions are the harness earning its keep: both were tasks I would have included by intuition, and
both would have contributed nothing but noise - exactly as they did in the earlier runs.

BLOCKED: Groq throttled to ~82s per call (8k TPM shared across the org, both keys). One task run needs 5-12
calls, so ~10 min/run. Screening is resumable in `qa/necessity.json`.

### Recorder (`skillloop/record.py`) - blocker #2, done
`learn(trace)` previously required hand-assembling steps, tool calls and signals. That was the real
integration cost. Now:

    with loop.session(task) as s:
        prompt += s.skills_text()
        s.tool("bash", {"cmd": cmd}, result=out, error=err)
        s.verified_by(check.returncode)

An uncaught exception is recorded as a failure signal (the most informative trace there is) and re-raised, not
swallowed. Recording is best-effort: if the store throws, the caller's task still completes. `record_tool` is
a decorator so tool functions record themselves including exceptions. 7 tests, incl. bullet-feedback
attribution through the session.

## 2026-09-11 — a manual reflector pass found a bug in the EXPERIMENT, not the code

Groq throttling made agent runs impractical (~82 s/call), so I added a `manual` LLM provider: the pipeline
writes each request to a file and reads the answer back, letting the learning roles run with no API key at all
(`SKILLLOOP_MANUAL_DIR`, `PendingManualCall`, traces stay queued until answered). The agent role must stay a
real naive model - whoever designed the tasks knows the traps, so playing the agent would destroy the baseline.

Reading the first queued trace by hand immediately exposed something no amount of model calls would have:

    task:    replace every occurrence of 'draft' with 'final' in notes.txt
    agent:   sed -i 's/draft/final/g' notes.txt      <- correct
    file:    final one / keep final and final / last final
    checker: FAIL

The file contains 4 occurrences of "draft". The checker asserted 5. **The task was unwinnable.** The agent had
solved it perfectly and been marked wrong, in every arm, every time.

Affected: `n3_count`, `n5_replace`, `n6_replace2`, and in the held-out set `s2_count`, `s11_replace_all`,
`t11_replace_all`. This means the previously reported held-out numbers (control 10/12 vs recall 11/12) were
computed with at least one impossible task dragging BOTH arms down, and the `n3_count` screening result
(0/3 "failures", which looked like a perfect eligibility candidate) was measuring a broken checker.
The n3 screening data has been deleted rather than reused.

Fix: `qa/verify_checkers.py` runs a known-good reference solution against every task's checker and asserts it
passes. A task whose checker can never pass is worse than useless - it fails in every arm and silently lowers
the measured rate on both sides. All 10 candidates now verified winnable before any agent sees them.

Lesson worth keeping: when an experiment reports a suspiciously clean signal (0/3 failures), suspect the
instrument before the subject.

## 2026-09-11 — first demonstrated causal effect, and two bugs it exposed

### The result
Necessity-screened tasks (baseline failure >= 60%), independent checkers, paired repeats:

| task | control | with skill |
|---|---|---|
| n7_upper | 0/3 | 3/3 |
| n8_lower | 1/3 | 3/3 |
| **total** | **1/6 (17%)** | **6/6 (100%)** |

5 discordant pairs, all favouring treatment. Sign test p = 0.0625 — the FLOOR for 5 pairs, so this is
"suggestive, not yet significant" and the harness prints exactly that. ~8-10 discordant pairs across
independent skills are needed to clear 0.05 honestly.

Mechanism, not just outcome. Control: `tr '[:lower:]' '[:upper:]'` run four times, no read-back, `é` untouched.
Treatment: the agent ran the skill's procedure verbatim —
`python3 -c "...any(ord(c)>127...)"` → `python3 ... .upper()` → `cat upper.txt` →
`python3 -c "assert t==t.upper()"`. Traceable instruction-following, not a vague lift.

Caveat kept in view: both tasks exercise ONE skill (`unicode-case-conversion`). This is one lesson shown twice,
not two independent findings.

### Two skills added from real control-arm failures
- `structured-config-edit`: `sed -i 's/replicas: 2/replicas: 4/g'` hit `service.replicas` AND `limits.replicas`.
  Verification asserts the target changed *and* a sibling did not — the only check that catches this.
- `verify-side-effect-not-exit-code`: `build.sh` ends in unconditional `exit 0` with `cp ... 2>/dev/null`.
  Skill checks the artefact (`test -f && test -s`, `-nt` for staleness) instead of `$?`.

### Bug 1 (ours, serious): evidence_count double-counted on reprocessing
`n_evidence = 1 + len(related) + existing.evidence_count` incremented every time a trace was reprocessed.
One skill reached **evidence_count 45 from ~5 real lessons**, which corrupts `provisional` and the cold-start
policy that keys off it. Fix: `Skill.lesson_ids` is the set of contributing lessons; `evidence_count` is its
length; a lesson already in the set is skipped and produces no new version. Verified on the live library:
45 → 5, and requeuing every trace twice now leaves evidence and version unchanged (STABLE).

### Bug 2 (mine, process): the manual provider accepted junk answers
A bad fill script wrote `null` into 8 answer files. That surfaced 4 steps later as
`AttributeError: 'NoneType' object has no attribute 'get'`, naming no file. Fix: `_validate_manual` checks the
answer parses, is an object, and carries the keys that role needs, raising `BadManualAnswer` with the path and
the problem. Wrong-shaped answers now fail at the file.

44 tests. Both bugs have regression tests that fail against the old code.


## 2026-09-11 — E3 cleared: p = 0.0078 across two independent skills

| task | skill | control | with skill |
|---|---|---|---|
| n7_upper | unicode-case-conversion | 0/3 | 3/3 |
| n8_lower | unicode-case-conversion | 1/3 | 3/3 |
| n9_yaml | structured-config-edit | 0/3 | 3/3 |
| **total** | | **1/9 (11%)** | **9/9 (100%)** |

8 discordant pairs, all favouring treatment. **Sign test p = 0.0078.** Two independent skills, each learned
from that task's own control-arm failures, gated, and retrieved by the normal recall path (no hand-feeding).

`qa/report_ab.py` computes this and EXITS NON-ZERO unless p < 0.05 with >= 2 distinct skills, so the bar
cannot be quietly lowered later.

Scope, stated wherever this number appears: measured only on tasks the baseline agent fails >= 60% of the time
without the skill. It is not a claim about all tasks. Not yet replicated on a second agent model (E4) — that
remains the most likely way the effect shrinks, because a stronger agent may already know these lessons.

`n10_silent` control came in at 2/3, i.e. below the necessity threshold, so its treatment arm was not run and
`verify-side-effect-not-exit-code` is excluded from the claim rather than counted on weaker evidence.


## 2026-09-11 — E4: the effect replicates on a second model family

`qwen/qwen3.8-27b`, same two skills, no re-teaching, same harness and checkers:

| task | skill | control | with skill |
|---|---|---|---|
| n7_upper | unicode-case-conversion | 1/3 | 3/3 |
| n8_lower | unicode-case-conversion | 1/3 | 3/3 |
| n9_yaml | structured-config-edit | 1/3 | 3/3 |
| **total** | | **3/9 (33%)** | **9/9 (100%)** |

6 discordant pairs, all favouring treatment, **sign test p = 0.0312 standing alone**.

The risk flagged in the checklist showed up exactly as predicted, and it is worth stating plainly: qwen's
control rate is 33% where gpt-oss-20b's was 11%. The stronger agent already knows some of this, so the
margin narrows from +0.89 to +0.67. It does not vanish. Any headline number must therefore be reported
per-model, never as a single figure.

Combined across both models: 14 discordant pairs, 14-0, p = 0.00012.


## 2026-09-11 — S1/S2: scale and concurrency

S1 (`qa/bench_scale.py`): the real library padded with realistic distractors across ten unrelated domains.
hit@1 stayed 4/4 and abstention 3/3 with **zero false fires** from 2 skills up to 202. Latency per query grows
linearly (0.9 ms -> 8.6 ms) because `_search` scans every skill; acceptable at this size, and we now know
where an index becomes necessary instead of guessing.

This was a genuine risk, not a formality: IDF, the abstention threshold and breadth normalisation are all
corpus statistics, and all three were tuned on a 5-skill library. They held.

S2 (`qa/bench_concurrency.py`): 8 threads x 25 interleaved learn+recall. 200 submitted, 200 stored, 0 errors,
no duplicate trace ids. WAL plus `BEGIN IMMEDIATE` on skill writes behaved.

## 2026-09-11 — E5 (second domain) attempted: INVALID, harness limitation

Built a second-domain task set (`qa/tasks4.py`): six Python **coding** tasks with semantic traps — mutable
default argument, float equality, timezone conversion, mutation while iterating, greedy regex, integer
rounding of money. Every checker imports the agent's module and asserts behaviour, so printing the right
answer does not pass. All six verified winnable against reference solutions before any agent saw them.

Screening appeared to give a clean result:

| task | baseline pass | looked |
|---|---|---|
| p1_mutable_default | 3/3 | excluded |
| p2_float_equality | 3/3 | excluded |
| p3_timezone | 0/3 | ELIGIBLE |
| p4_mutate_while_iterating | 3/3 | excluded |
| p5_greedy_regex | 1/3 | ELIGIBLE |
| p6_rounding | 0/3 | ELIGIBLE |

**It is not a result.** Inspecting the traces: of 8 failing traces, **8 were harness errors and 0 were real
task failures.** Writing a multi-line Python file makes the agent emit a native tool call
(`container.exec` with a heredoc); Groq rejects it with `tool_choice is none, but model called a tool`, and
the recovery path mangles the double-escaped payload. The agent never ran anything. p3/p5/p6 were not hard —
they were unreachable.

This is the same class of error as the impossible checkers and the hallucinating agent model: a clean-looking
signal produced by a broken instrument. Third time. The 0/3 pattern should be treated as a smell, not a find.

Mitigations applied: a first-class `write_file` action so the agent never needs a heredoc for multi-line
content (with a path-escape guard), and more robust recovery of wrapped tool calls. Neither was sufficient —
the model still prefers native tool calls for this task shape.

**Correct fix, not yet done:** stop encoding actions as JSON inside message content and use the provider's
native tool-calling API, reading `tool_calls` off the response. Both arms must then use the same harness.
Until that lands, **E5 is unmeasured** and no cross-domain claim may be made. The domain-1 results
(shell/text, two model families, p=0.0078 and p=0.0312) stand and are unaffected — they were produced by a
harness path that demonstrably executed commands.

## 2026-09-11 — external review: two high-priority bugs, both fixed

An outside reviewer stress-tested v0.3 and found two real blockers my own tests missed. Both reproduced,
both fixed, both now have regression tests.

### Bug 1 (HIGH): SQLite concurrency
A single `sqlite3.Connection` was shared across threads with `check_same_thread=False`. That silences the
guard but the connection's cursors are shared mutable state, so under the reviewer's 8-thread HTTP stress
test it produced ~60 errors (OperationalError / InterfaceError / SystemError) and incomplete writes.
My own S2 test passed only because it ran in one process where the GIL happened to serialise the writes.
Fix: **one connection per thread** via `threading.local`, each with WAL + `busy_timeout=30000` +
`synchronous=NORMAL`. Verified: 8 threads x 25 interleaved learn+recall -> 200/200 stored, 0 errors.

### Bug 2 (HIGH): trace-size cap not a hard limit
`max_total_chars = 200_000` was enforced by a single "keep first/last quarter of steps" pass, which on a
2.4M-char input still left ~670k. Since traces carry external/tool content, the cap is a safety guarantee,
so a soft cap is a real hole. Fix: reduce **in a loop** until the serialized trace is actually under the cap
— halve steps, then shrink content/args/results, then drop tool payloads, then a floor marker — and
`assert len(out.to_json()) <= max_total_chars` at the end. Verified on three pathological shapes (many big
steps, a few 900k-char steps, one 3M-char step): all land <= 200k.

### Lower priority
- Unclosed SQLite resources: `Store.close()` + context-manager protocol; connections closed per thread.
- The self-review lesson repeats: my S2 concurrency "PASS" was an instrument artefact (GIL serialisation),
  the same failure mode as the impossible checkers and the hallucinating agent. Concurrency must be tested
  with real parallelism, which the new regression test does via ThreadPoolExecutor.

47 tests.

## 2026-09-11 — E5 re-attempted on a native tool-calling harness: NOT MEASURABLE with gpt-4o-mini

Built the harness CALIBRATION previously listed as "not yet done" (it was referenced in HANDOFF.md but did not
exist in the v0.3 zip): `qa/agent_native.py` reads `tool_calls` off the response (write_file / run / finish),
so multi-line files never pass through an escaping layer. `qa/verify_checkers.py` proves, through the same
executor, that for all six tasks an empty dir fails, the classic buggy version fails (trap is real), the
reference passes, and a scripted end-to-end native tool-call run passes. `qa/run_e5.py` runs the protocol.

Protocol fixed before results: agent openai/gpt-4o-mini via OpenRouter, temperature 0.7, max 10 turns,
3 screening runs/task, eligible <=> baseline passes <= 1/3, library frozen after learn, arms interleaved,
harness errors re-run and never scored.

Screen: **18/18 PASS, 0 harness errors, 0 eligible tasks.** Traces read before trusting it: every run wrote
the module itself into a fresh dir; the final code genuinely avoids each trap (None default, nums[:] = ...,
astimezone/utcoffset, find-based tag parsing, divmod remainder). Two p1 runs used all 10 turns re-testing
correct code without calling finish; checker-decided pass, legitimate. Cost ~$0.008.

Findings: (1) the earlier "p3/p5/p6 ELIGIBLE" screen is now confirmed as instrument artifact — on a working
harness those tasks pass 3/3. (2) These six traps are too easy for gpt-4o-mini; with nothing failing there is
nothing to learn from and no A/B. **E5 remains unmeasured. No cross-domain claim.** Next attempt needs a
weaker agent (domain 1 deliberately used gpt-oss-20b) and/or harder tasks — screened fresh, reported per model.

## 2026-09-11 — E5 moved to HumanEval+ (industry standard). Screen 162/163 done, then OUT OF CREDITS.

Toy traps (previous entry) were too easy: 18/18. Replaced with **HumanEval+** (EvalPlus 0.3.1, HumanEvalPlus
v0.1.10) — hardened HumanEval, ~80x more tests/problem. SWE-bench Verified would be the stronger standard but
is infeasible here (per-repo Docker images, ghcr.io not on the allowlist, 1 CPU).

Instrument (`qa/he_check.py`): EvalPlus's own `check_correctness`; pass <=> base AND plus both PASS. Validated
across all 164: canonical solution passes all, `return None` stub fails all. **HumanEval/32 excluded** — the
EvalPlus 0.3.1 `find_zero` special oracle fails its OWN reference solution (0/888). Left undetected that would
have read as a permanently-hard problem. 163 remain.

Design upgrade over domain 1: seeded split (SEED=20260911) 81 LEARN / 82 TEST; SkillLoop sees LEARN traces
ONLY, so skills must TRANSFER to unseen problems. Three arms — control (bare) / placebo (fixed generic
"careful Python" checklist) / treatment (recall) — so a win can't be explained by "more text in the prompt".
Intention-to-treat: eligible TEST problems stay in even when recall abstains.

Results so far — **baseline pass-1 = 130/163 = 79.8%**, consistent with published gpt-4o-mini HumanEval+ pass@1
(~79-85%). Independent evidence the harness measures the real benchmark. 32 pass-1 failures; traces audited:
all 32 wrote solution.py, 31 are genuine wrong answers (e.g. HumanEval/10 make_palindrome fails 'xyx'), 1 is a
plus-set TIMEOUT (HumanEval/163, 408/982) that needs a serial re-check before being called a failure.

**STOPPED: OpenRouter credits exhausted.** Free-tier key, total_credits 0, total_usage $0.196. Symptom was
HTTP 402 `in_flight_budget_exhausted` under 8 workers; the real cause is per-request budget reservation —
"You requested up to 1500 tokens, but can only afford 1194". Small calls still 200, so reachability is fine.
Deliberately did NOT lower max_tokens to squeeze out runs: that changes the protocol mid-experiment and
truncated completions would manufacture exactly the fake failures this file exists to prevent.

State: 0 of 32 candidates have the 3 runs eligibility needs, so **no eligible set, no A/B, no p-value.**
E5 REMAINS UNMEASURED. Resume with credits: `run_he.py screen` (resumable, finishes 32x2 runs) -> `learn` ->
`ab` -> `report_ab.py <out>/he_ab.json --skills <out>/he_skills.json`. Est. ~150 runs, ~$0.20 at observed
~$0.001/run. Use HE_WORKERS=4.

## 2026-09-12 — E5-HE on Groq: instrument bug #5 caught before launch

Switched the HumanEval+ agent to gpt-oss-20b on Groq free tier (only allowlisted provider; OpenRouter and all
other free providers are host_not_allowed from the sandbox). Both Groq keys' daily caps had reset.

MANDATORY sanity check before ~230 runs: ran HumanEval/2 and /4 (easy). Both FAILED at 4-5 tool calls with no
harness error — a smell. Read the trace: the agent's solution was correct; the CHECKER crashed with
`ModuleNotFoundError: No module named 'evalplus'`. Cause: `task()` built the checker command as
`python3 he_check.py ...`, but `python3` is the SYSTEM interpreter without evalplus, while evalplus is in the
venv. Every problem would have "failed" regardless of the answer — a fresh gpt-oss-20b screen would have read
as "fails everything", manufacturing a false 100%-eligible set.

Fix: checker command now uses `sys.executable` (the venv python). Re-ran HumanEval/0, /2, /4: all PASS.

This is the 5th self-caught instrument artifact (impossible checkers; hallucinating agent; GIL-masked
concurrency; rate-limit-as-failure; now wrong-interpreter checker). The rule holds: a clean-looking or
suspicious result is an instrument smell until traces are read.

STATUS: harness validated on gpt-oss-20b. Ready to run the fresh Groq screen (the 162 gpt-4o-mini runs are not
reused — mixing agents across arms is a confound). E5-HE remains UNMEASURED until screen->learn->ab complete.

## 2026-09-12 — two user-reported bugs fixed (verified_by inversion, quarantine strand)

### Bug 1 (insidious): `verified_by()` flipped booleans
`verified_by(exit_code)` tested `exit_code == 0`. Because `False == 0` in Python, passing a bool inverted the
result: `s.verified_by(actual == expected)` recorded SUCCESS when the check FAILED and vice versa. Since the
library learns from verified failures, this taught it from tasks that actually succeeded and hid the ones that
failed - corrupting the training signal silently. Fix: `isinstance(check, bool)` is handled directly
(True=pass, False=fail); ints keep the exit-code convention (0=pass). Reproduced, fixed, regression test.

### Bug 2 (severe on the manual provider): skills stranded in quarantine forever
The pipeline saved the skill (claiming `lesson.id` in `lesson_ids`) BEFORE running the gate. If anything
interrupted between those steps - crash, rate limit, or a `PendingManualCall` (which happens on EVERY manual
run by design) - the skill was left in quarantine with its lesson already claimed. On the next run the
idempotence guard fired ("lesson already contributed"), returned early, never ran the gate, never wrote an
eval, and marked the trace processed=1 so it was never retried. Observed: status=quarantine, evals=[],
recall()->nothing, no error anywhere. On the manual provider (the "no API key" path the README advertises) no
skill could EVER reach active. Fix: the idempotence guard only skips skills that actually cleared the gate
(candidate/active); a lesson whose skill is stuck in quarantine RESUMES into the gate via a new `_gate()`
entry point. Reproduced end-to-end on the manual provider, fixed, regression test (skill reaches candidate and
is recallable).

67 tests.
## 2026-09-12 — hybrid retrieval + query-level intent gate

Blind adversarial probes had exposed lexical retrieval at F1 0.47 (paraphrase misses + topical false-fires).
Two changes, council-reviewed:

1. **True hybrid retrieval** (`_search` + `_embed_candidates`): embeddings now surface candidates INDEPENDENTLY
   of lexical overlap (previously they only re-ranked what lexical already found, so a paraphrase lexical
   missed could never be recovered). Lexical + anchor/not_for rules act as precision gates on the union.
   Pure-lexical behaviour is unchanged when no embed model is configured (71 tests still pass).

2. **Query-level intent gate** (`_looks_like_task`): a creative or factual-question request ("write a poem
   about docker", "explain how git works", "what year was postgres released") fires NO skill, applied once at
   the entry to retrieval so it covers both paths. This is the D-class fix the earlier patch author correctly
   said needed a design change, not a word list.

Measured (deterministic semantic stub for embeddings; real numbers need Ollama/OpenAI):
| probe set | metric | before | after |
|---|---|---|---|
| blind (16) | F1 | 0.47 | **0.67** |
| blind (16) | precision | 0.50 | **1.00** |
| blind (16) | abstention | 3/6 | **6/6** |
| R1 (contributor's adversarial) | precision | 0.69 | **0.81** |

The residual recall gap (0.50 on blind) is the STUB embedding's weakness (concept-keyword overlap), not the
architecture - a real embedding model closes it. Config: SKILLLOOP_EMBED_MODEL enables the path;
SKILLLOOP_HYBRID_LEXICAL_W / _EMBED_W / _EMBED_TOPK / _EMBED_THRESHOLD tune it.

68->73 tests (added hybrid + intent + regression coverage).

## 2026-09-12 — statistics correction (external review)

An external reviewer correctly caught that the headline p-value was computed with the wrong unit of analysis.
`report_ab.py` zipped control/treatment TRIAL-by-trial and counted 8 discordant trials as independent pairs
(p=0.0078). But three repeated runs of the same task with the same skill are NOT independent - the independent
unit is the TASK, and there are only three. The task-level sign test bottoms out at **p=0.25** with n=3, even
for a perfect 3/3 split.

Corrected: `report_ab.py` now aggregates to per-task outcomes and reports the task-level p as the headline,
printing the per-trial p only with an explicit "anti-conservative, do not report" label. The README evidence
table was rewritten from "demonstrated, p=0.0078" to "promising 3-task pilot, p=0.25 (underpowered)". The
direction (3/3 tasks flipped fail→pass, two model families) is a clean signal; it is not yet significant.

Also fixed from the same review: removed the orphan `n10_silent` control arm (no treatment arm shipped);
replaced the non-PyPI `pip install skillloop` with a git-install line; fixed the `YOURORG` placeholder in
pyproject.toml; fixed report_ab.py's hardcoded default input path; corrected the stale 66-tests badge to 72.

The honest version is still interesting - and now the numbers match the data.