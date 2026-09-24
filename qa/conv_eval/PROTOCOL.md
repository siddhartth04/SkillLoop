# Conventions eval: pre-registered protocol

Fixed before any model run. Nothing below changes after results are seen.

## Question
Does SkillLoop, running its real pipeline, make an agent stop repeating a mistake on NEW tasks that hit the same
trap, better than (a) no memory and (b) simple memory (raw past attempts in the prompt)?

## Environment
`generator.py`: a library that exists nowhere, with 6 of 9 generic API conventions active per seed (1-based
positions, shifted range bounds, millisecond timeouts, UPPER-KEBAB config keys, commits needing confirm=True,
int cents, (error, value) returns, descending sort, in-place dedupe). Names, data and active conventions come from
the seed. Docs are truthful but omit the conventions. Per seed: 18 train tasks, 18 trap test tasks, 3 no-trap tests.
Known limitation: the convention pool was written by the SkillLoop maintainers' assistant, who knows how SkillLoop
works. Mitigation: the simple-memory baseline receives exactly the same training information, so any bias in the
environment favours both learners equally; SkillLoop must beat it, not just control.

## Conditions (same model, temperature 0, 2 attempts per task, identical prompts except the context block)
- control: API doc only. Run twice (null check).
- oracle: API doc + a perfect one-line note per active convention (ceiling).
- memory: API doc + every training attempt verbatim with its test outcome (last 8,000 chars).
- skillloop: real SkillLoop. Training episodes go through Session -> learn() -> process() with the same model.
  At test time: recall() at task start, error-time hints on a failed attempt. Library frozen during test.
Training episodes are run once and shared by memory and skillloop. At test time the agent sees a traceback if its
code crashed, otherwise only "the result is incorrect"; the hidden check is never shown to the agent.

## Gates (per seed; a seed failing any gate is excluded and SkillLoop is not run on it)
1. Null: control and its repeat agree on >= 85% of trap test tasks.
2. Headroom: control <= 70% and oracle - control >= 30 points on trap test tasks.
3. Harness: <= 5% of episodes end in a harness error (unclosed fence, empty output, provider failure after retries).
Harness errors are never scored as agent failures.

## Metrics
- Primary: success within 2 attempts on held-out trap test tasks, pooled over valid seeds.
- Secondary: first-attempt success; no-trap success (harm: skills misapplied where no convention is active).
- Test: exact two-sided sign test on discordant pairs (McNemar exact), alpha 0.05.
  Comparisons: skillloop vs control (primary), skillloop vs memory, memory vs control.

## Plan
`--selfcheck` (no model) -> `--pilot` (seed 101, 12 tasks, gates only; read every failure) -> `--full --seeds 1 2 3`
(54 paired trap tasks if all seeds pass). At most ~1,100 model calls for the full run.

## Verdict wording (fixed)
- SkillLoop beats control AND memory (p<0.05): "SkillLoop helps, and its machinery beats simple memory."
- Beats control only: "SkillLoop helps, but not significantly more than simple memory."
- Otherwise: "No significant improvement over control."
Plus a warning if no-trap success drops more than 15 points versus control.
