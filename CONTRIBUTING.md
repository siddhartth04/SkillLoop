# Contributing to SkillLoop

Thanks for your interest. SkillLoop is an experimental research tool, and its defining trait is
**measurement honesty** — so contributions that strengthen the evidence or catch a false positive are
especially welcome.

## Ground rules

- **Never claim a result you haven't measured.** If you add an experiment, add the raw data and a way to
  regenerate the number (see `qa/report_ab.py`). A suspiciously clean result (e.g. `0/3`) is an *instrument
  smell* — read the traces before trusting it. This project has caught its own false positives five times;
  keep that bar.
- **Statistics are pre-registered.** Don't change a test's success criterion after seeing results.
- **Scope claims precisely.** "On tasks the agent fails without the skill" is not "improves agents."

## Dev setup

```bash
git clone <your-fork>
cd skillloop
pip install -e ".[dev,openai]"
pytest -q                     # 65 pass, 1 skipped (optional dep)
python demo/run_demo.py --fake
```

## Before opening a PR

- `pytest -q` is green.
- New behavior has a test. Security-relevant changes have a case in `tests/test_redteam.py`.
- No API keys, tokens, or personal data in code, tests, or fixtures.
- Run `python demo/run_demo.py --fake` to confirm the happy path still works.

## Good first contributions

- **A new provider adapter** or integration (Codex, a framework hook).
- **Reproducing the domain-1 result** on your own model — independent replication is the most valuable thing
  you can add.
- **The open cross-domain experiment** — `qa/run_he.py` is a resumable HumanEval+ rig; see `HANDOFF.md`.
- **Retrieval or security hardening**, with a test that fails against `main`.

## Reporting issues

Use the templates. For a suspected false positive or a measurement bug, include the raw trace — that's the
single most useful thing you can attach.

By contributing you agree your work is licensed under the MIT License.
