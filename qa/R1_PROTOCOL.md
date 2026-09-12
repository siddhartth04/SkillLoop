# R1 — Adversarial retrieval eval (pre-registered)

Registered **before any probe was run**. Nothing below was changed after seeing a result.

## What is being tested

`SkillLoop.recall()` only. This is the component the README makes a hard quantitative claim about
("Held-out: 5/5 correct, **0 false fires**, holds to 200 skills"). Retrieval is deterministic — IDF-weighted
coverage plus phrase matching, no model call — so it can be evaluated exactly and reproducibly with no API key.

This is **not** the HumanEval+ A/B. It measures whether the right skill is retrieved, not whether the skill
helps an agent. A perfect score here says nothing about end-to-end lift.

## Library

14 skills across 7 domains, written to be realistic (full facets, 8-12 trigger queries, bullets in every
section). Crucially the library contains **sibling skills**: pairs in the same domain separated only by the
operation or the file format. A retriever that matches on domain vocabulary alone will confuse them.

Plus N synthetic distractors from `qa/bench_scale.py`, at N ∈ {0, 100, 500, 2000}.

## Probe classes (the eval is adversarial by construction)

| class | n | what it tests | correct behaviour |
|---|---|---|---|
| **A. direct** | 14 | the obvious phrasing | retrieve the target skill at rank 1 |
| **B. paraphrase** | 14 | user vocabulary, none of the skill's technical terms | retrieve the target at rank 1 |
| **C. sibling** | 14 | same domain as a skill, but the *other* sibling is correct | retrieve the CORRECT sibling at rank 1 |
| **D. lexical trap** | 14 | shares content words with a skill, different intent | ABSTAIN |
| **E. pure negative** | 14 | unrelated to anything in the library | ABSTAIN |

70 probes. Classes C and D exist specifically to produce false fires. If the retriever has no real
discrimination, C degrades to ~50% and D to ~0% abstention.

## Metrics (fixed before running)

- **hit@1** on A/B/C — the target skill is the top non-principle result.
- **abstention** on D/E — no skill returned at all.
- **false fire rate** — any skill returned on D/E, or the wrong skill returned on A/B/C.
- **precision / recall / F1**, treating "returns the correct skill" as the positive class.
- **Wilson 95% CI** on each rate — with n=14 per class, point estimates alone are not reportable.
- **median latency per query**.

## Success criteria (fixed before running)

The README's claim generalises if:
1. hit@1 on A ≥ 0.90, **and**
2. hit@1 on B ≥ 0.70 (paraphrase is the realistic case), **and**
3. abstention on E ≥ 0.90, **and**
4. abstention on D ≥ 0.70, **and**
5. no metric degrades by more than 0.10 between library size 0 and 2000.

Anything less is reported as a partial or failed result. Criteria are not revised after seeing numbers.

## Known limitation, stated up front

The skills and the probes were written by the same author (me), in that order, in one sitting. Probes were
written from the skill *descriptions* without running any of them, and no probe was edited after seeing a
result — but this is not a blind or independent probe set, and a co-authored eval flatters the system.
Treat this as a stress test that can *falsify* the retrieval claim, not as independent confirmation of it.

---

## Result (main @ aaacfa3, unmodified)

| class | n | result | 95% CI | criterion | verdict |
|---|---|---|---|---|---|
| A direct | 14 | 14/14 | 0.79–1.00 | ≥0.90 | pass |
| B paraphrase | 14 | 7/14 | 0.27–0.73 | ≥0.70 | **fail** |
| C sibling | 14 | 14/14 | 0.79–1.00 | — | pass |
| D lexical trap | 14 | 1/14 | 0.01–0.32 | ≥0.70 | **fail** |
| E pure negative | 14 | 12/14 | 0.60–0.96 | ≥0.90 | **fail** |

precision 0.66 · recall 0.83 · F1 0.74 · stable to 2000 skills

Three of five pre-registered criteria failed. Criteria were not revised after seeing the numbers.

### What passed, and it matters

**Sibling disambiguation is 14/14.** Separating `yaml-single-key-edit` from `yaml-merge-two-files`, or
`pytest-collection-error` from `pytest-flaky-order-dependence`, is the hardest thing in this eval, and the
anchor rule handles it cleanly. This is a real strength and it is not an artifact of easy probes.

### What failed, and why

**E (2 false fires) is a bug, fixed separately.** `applies_when` phrases are tokenised word-by-word and every
token becomes an anchor at weight 3.0, so a phrase donates its function words as anchors. `"script lies about
success"` donated the anchor `about`, and *"write a haiku about the sea"* retrieved an exit-code skill;
`"explain analyze"` donated `explain`, and *"explain the plot of Hamlet"* retrieved a Postgres skill. Anchors
are the mechanism that prevents false fires, so a preposition satisfying the anchor rule defeats it. See the
`fix/function-word-anchors` branch — it takes E to 14/14 and leaves A and C unchanged.

**D (13 false fires) is not a bug — it is the design boundary.** The retriever models *topic*, not *intent*.
*"write a Dockerfile for a node application"* fires the docker-cache skill on `dockerfile`, a legitimately
relevant term. Nothing in the query says "my build is broken", and `not_for` is itself topical
(`"container wont start"`), so it blocks nothing here. In practice this means fix-skills are injected into
ordinary in-domain work. Fixing it needs a design change — an intent signal separating "help me build X" from
"X is broken" — not a word list, so no fix is proposed here.

**B at 7/14** is the other soft spot, and it is the common case: users describing a symptom in their own
words, without the skill's vocabulary, get the right skill half the time.

### Honest scope of this eval

This does not measure end-to-end lift. A perfect score here would say nothing about whether a retrieved skill
helps an agent; that is what `qa/run_he.py` is for.
