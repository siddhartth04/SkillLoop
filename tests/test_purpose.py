"""The purpose, tested end to end: an agent that made a mistake must not make it again.

A simulated agent with one deterministic bad habit: it imports a freshly installed package without checking the
install worked. When a skill telling it to verify is in its context, it follows it and succeeds. Everything else
(learning, gating, recall at task start, recall at error time, promotion) is the real SkillLoop pipeline.
"""
import tempfile
import warnings

import pytest

from skillloop import SkillLoop
from skillloop.llm import LLM
from skillloop.store import Skill
from skillloop.playbook import new_bullet

from tests.test_loop import fake_handler

INSTALL_ERR = "error: externally-managed-environment\n× This environment is externally managed"
IMPORT_ERR = ("Traceback (most recent call last):\n  File \"/srv/app/report.py\", line 2, in <module>\n"
              "    import pandas\nModuleNotFoundError: No module named 'pandas'")


@pytest.fixture
def loop():
    LLM._fake_handler = staticmethod(fake_handler)
    lp = SkillLoop(home=tempfile.mkdtemp(prefix="purpose-"), llm=LLM(provider="fake"))
    yield lp
    LLM._fake_handler = None


def run_agent(loop, task, extra_context=""):
    """One episode. Returns (succeeded, session). The agent verifies only if a skill in context says so."""
    with loop.session(task, agent="sim") as s:
        context = s.skills_text() + extra_context
        knows = "verify-pip-install" in context
        s.tool("bash", {"cmd": "pip install pandas"}, error=None if knows else INSTALL_ERR)
        if not knows:
            # habit: charge ahead and import anyway -> the error surfaces mid-task
            s.tool("python", {"code": "import pandas"}, error=IMPORT_ERR)
            hint = s.hints_text()
            if "verify-pip-install" in hint:          # error-time recall rescued the task
                s.tool("bash", {"cmd": "python -m venv .venv && .venv/bin/pip install pandas"}, result="ok")
                s.tool("python", {"code": "import pandas"}, result="ok")
                s.verified_by(0)
                return True, s
            s.verified_by(1)
            return False, s
        s.tool("python", {"code": "import pandas"}, result="ok")
        s.verified_by(0)
        return True, s


def learned(loop):
    return {s.name: s.status for s in loop.store.all_skills()}


def test_mistake_is_learned_then_not_repeated_under_new_phrasings(loop):
    ok, _ = run_agent(loop, "install pandas and build the weekly report")
    assert not ok
    loop.process()
    assert "verify-pip-install" in learned(loop), "a failure must produce a skill"

    # the SAME mistake, phrased the ways people actually phrase work - including as questions
    for task in ["pip install pandas for the report script",
                 "how do I get pandas installed with pip for the report",
                 "why does import pandas fail after pip install",
                 "how do I get the pandas package working, the import keeps failing"]:
        ok, s = run_agent(loop, task)
        assert ok, f"mistake repeated on: {task!r}"
        assert "verify-pip-install" in s.skills_used


def test_error_time_recall_rescues_a_task_whose_description_gives_no_hint(loop):
    run_agent(loop, "install pandas and build the weekly report")
    loop.process()
    # nothing in this task mentions pip or packages; only the error does
    ok, s = run_agent(loop, "generate the quarterly revenue summary")
    assert ok, "the lesson must arrive when the error appears, not only at task start"
    assert s.trace.metadata.get("error_hints"), "the rescue is recorded in the trace for measurement"


def test_repeat_mistake_rate_drops(loop):
    tasks = ["build the churn dashboard", "install pandas and plot signups", "refresh the finance report",
             "how do I run the pandas cleanup job", "generate the inventory export",
             "set up the notebook for the cohort analysis"]
    before = sum(not run_agent(SkillLoop(home=tempfile.mkdtemp(), llm=LLM(provider="fake")), t)[0] for t in tasks)
    run_agent(loop, "install pandas and build the weekly report")
    loop.process()
    after = sum(not run_agent(loop, t)[0] for t in tasks)
    assert before == len(tasks) and after == 0, (before, after)


def test_verification_rerun_promotes_skill_automatically(loop):
    run_agent(loop, "install pandas and build the weekly report")
    loop.process()
    assert learned(loop)["verify-pip-install"] == "candidate"
    assert loop.pending_evals(), "a host eval must be queued for the new skill"

    def runner(task, skill_md):
        s = loop.session(task, auto_learn=False)
        run_ok = "verify-pip-install" in skill_md
        s.tool("bash", {"cmd": "pip install pandas"}, result="ok")
        s.verified_by(run_ok)
        return s

    res = loop.run_pending_evals(runner)
    assert res and res[0]["skill"] == "verify-pip-install"
    assert learned(loop)["verify-pip-install"] == "active"
    assert not loop.pending_evals()


def test_failed_or_crashing_rerun_does_not_promote(loop):
    run_agent(loop, "install pandas and build the weekly report")
    loop.process()

    def crashing_runner(task, skill_md):
        raise RuntimeError("agent crashed")

    loop.run_pending_evals(crashing_runner)
    assert learned(loop)["verify-pip-install"] != "active"


def test_no_skill_fires_on_knowledge_or_creation_requests(loop):
    run_agent(loop, "install pandas and build the weekly report")
    loop.process()
    for q in ["what is pip", "how does pip resolve dependency versions", "write a poem about python packages",
              "write a new setup.py for the package", "how do I create a new python package"]:
        assert not [h for h in loop.recall(q) if h["name"] != "_principles"], q


def test_warns_once_when_nothing_processes_the_queue(loop):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        for i in range(4):
            run_agent(loop, f"install pandas and build report {i}")
    msgs = [str(x.message) for x in w if "nothing is learning" in str(x.message)]
    assert len(msgs) == 1


# ---------------------------------------------------------------- intent classifier, as a unit
@pytest.mark.parametrize("text,intent", [
    ("How do I migrate the postgres schema", "task"),
    ("What is failing in the deploy script, fix it", "task"),
    ("Explain and fix the nginx config error", "task"),
    ("why does my container exit right after it starts", "task"),
    ("migrate the postgres schema", "task"),
    ("explain analyze shows a seq scan on users", "task"),
    ("write a pytest fixture that returns a db session", "create"),     # 'fixture' is not 'fix'
    ("how do I create a new branch in git", "create"),
    ("design a postgres schema for a blog", "create"),
    ("what is the default max_connections in postgres", "knowledge"),
    ("how does pip resolve dependency versions", "knowledge"),
    ("explain how git rebase works", "knowledge"),
    ("write a haiku about docker", "creative"),
])
def test_intent(loop, text, intent):
    assert loop._intent(text) == intent


# ---------------------------------------------------------------- error-time recall, as a unit
def _skill(name, symptoms, not_for=(), applies=("x",)):
    return Skill(name=name, description=name.replace("-", " "), body="", status="active",
                 bullets=[new_bullet("procedure", "do the thing"), new_bullet("verification", "exit code 0")],
                 facets={"applies_when": list(applies), "symptoms": list(symptoms), "not_for": list(not_for)},
                 queries=[], security={"cleared": True}, evidence_count=3)


def test_verbatim_symptom_beats_single_word_exclusion(loop):
    loop.store.save_skill(_skill("git-rebase-conflict", ["CONFLICT (content)"], not_for=["merge conflicts on a merge"],
                                 applies=["rebase conflict"]), snapshot=False)
    err = "CONFLICT (content): Merge conflict in a.py\nerror: could not apply 3f2a9c1... x\nhint: Resolve all conflicts"
    assert [h["name"] for h in loop.recall_for_error(err)] == ["git-rebase-conflict"]


def test_error_recall_excludes_skills_already_shown(loop):
    loop.store.save_skill(_skill("pip-import-mismatch", ["ModuleNotFoundError", "No module named"]), snapshot=False)
    assert loop.recall_for_error(IMPORT_ERR)
    assert loop.recall_for_error(IMPORT_ERR, exclude=["pip-import-mismatch"]) == []


def test_error_signature_strips_noise():
    sig = SkillLoop._error_signature("step 1\n/very/long/path/to/app/main.py ok\ncontainer 4b2c9a1f8e3d\n"
                                     "FATAL: sorry, too many clients already")
    assert "FATAL" in sig and "4b2c9a1f8e3d" not in sig and "/very/long" not in sig


# ---------------------------------------------------------------- embedding path: paraphrase rescue
def test_strong_embedding_match_fires_without_word_overlap(loop):
    sk = _skill("pytest-order-dependence", ["passes alone fails in suite"], applies=["test order dependence"])
    sk.embedding = [1.0, 0.0]
    loop.store.save_skill(sk, snapshot=False)
    llm = loop.llm
    llm.embed_model = "stub"
    para = "everything is green one at a time and red all together"
    llm.embed = lambda texts: [[0.95, 0.31] if t == para else [0.0, 1.0] for t in texts]
    assert [h["name"] for h in loop.recall(para) if h["name"] != "_principles"] == ["pytest-order-dependence"]
    loop.embed_strong = 0.99       # below the strong bar, the anchor rule applies again -> abstain
    assert not [h for h in loop.recall(para) if h["name"] != "_principles"]


def test_unverified_candidates_are_labelled_in_the_prompt(loop):
    """A candidate has passed only a model's judgement. It must not read like a proven procedure.

    This is the failure mode the conventions eval exposed: a skill can state a confident specific that is
    simply wrong, and the agent follows it and burns an attempt. Labelling does not make the skill correct,
    but it lets the agent weigh it against the documentation in front of it.
    """
    run_agent(loop, "install pandas and build the weekly report")
    loop.process()
    q = "how do I get the pandas package working, the import keeps failing"

    assert learned(loop)["verify-pip-install"] == "candidate"
    text = loop.session(q, auto_learn=False).skills_text()
    assert "verify-pip-install" in text
    assert "UNVERIFIED" in text, "an unverified candidate must say so in the prompt"

    def runner(task, skill_md):
        s = loop.session(task, auto_learn=False)
        s.tool("bash", {"cmd": "follow the skill"}, result="ok")
        return s.verified_by(0)

    loop.run_pending_evals(runner)
    assert learned(loop)["verify-pip-install"] == "active"
    promoted = loop.session(q, auto_learn=False).skills_text()
    assert "verify-pip-install" in promoted
    assert "UNVERIFIED" not in promoted, "a verified skill must not carry the warning"


def test_gate_refuses_to_promote_a_skill_that_fails_its_rerun(loop):
    """The central claim: a skill is a hypothesis until an independent check agrees."""
    run_agent(loop, "install pandas and build the weekly report")
    loop.process()
    assert learned(loop)["verify-pip-install"] == "candidate"

    def failing_runner(task, skill_md):
        s = loop.session(task, auto_learn=False)
        s.tool("bash", {"cmd": "follow the skill"}, error="still broken")
        return s.verified_by(1)          # independent check FAILS

    loop.run_pending_evals(failing_runner)
    assert learned(loop)["verify-pip-install"] != "active", "a failing re-run must never reach active"


def test_a_skill_containing_non_ascii_survives_a_save_and_reload(loop):
    """Models write en dashes, smart quotes and non-breaking hyphens. Skills must not be lost to them.

    Seed 4 of the conventions eval lost five of six learned skills to UnicodeEncodeError: SKILL.md was
    written with the platform default encoding, which on Windows is cp1252, and a single U+2011 in the
    model's output destroyed the skill. The run looked like the verification gate rejecting them, which is
    the kind of mistake that quietly invalidates an evaluation.
    """
    tricky = "non‑breaking — dash, “smart quotes”, check ✓, café"
    sk = _skill("unicode-safe-skill", ["a symptom with — in it"])
    sk.description = tricky
    loop.store.save_skill(sk, snapshot=True)

    back = loop.store.get_skill("unicode-safe-skill")
    assert back is not None, "a skill with non-ASCII text must survive the round trip"
    assert back.description == tricky
    assert (loop.store.skills_dir / "unicode-safe-skill" / "SKILL.md").exists()


def test_a_trace_that_errors_warns_instead_of_silently_producing_no_skill(loop):
    """A failed trace must be distinguishable from a trace that legitimately taught nothing.

    In the seed-4 conventions run, five of six skills were destroyed by an encoding error while being saved.
    process() logged each one and carried on, so the library simply came out nearly empty - which looks
    exactly like a run where the model found nothing worth learning. The eval then scored that library and
    produced a clean, plausible, WRONG verdict. Failing traces now say so out loud.
    """
    import skillloop.core as core

    run_agent(loop, "install pandas and build the weekly report")
    original = core.SkillLoop._process_one

    def boom(self, t):
        raise UnicodeEncodeError("charmap", "x", 0, 1, "simulated encoding failure")

    core.SkillLoop._process_one = boom
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            loop.process()
    finally:
        core.SkillLoop._process_one = original

    msgs = [str(w.message) for w in caught if "produced no skill" in str(w.message)]
    assert msgs, "a trace that raises must warn, not fail silently"
    assert "UnicodeEncodeError" in msgs[0], "the warning must name the error kind"


def test_a_verification_rerun_is_not_itself_learned_from(loop):
    """A host eval is evidence ABOUT a skill, not a new lesson.

    When run_pending_evals() fed its re-run back through learn(), the re-run became training data: it
    produced a second skill, which queued a second host eval, which ran WITHOUT the skill in its prompt and
    failed. Every skill therefore ended with exactly as many failed evals as passed ones and was demoted to
    quarantine regardless of how well it had actually performed. Seed 4 of the conventions eval showed all
    seven skills quarantined while their re-runs were passing 3/3.
    """
    run_agent(loop, "install pandas and build the weekly report")
    loop.process()
    n_before = loop.store.db.execute("SELECT COUNT(*) FROM traces").fetchone()[0]

    def runner(task, skill_md):
        s = loop.session(task, auto_learn=False)
        s.tool("bash", {"cmd": "follow the skill"}, result="ok")
        return s.verified_by(0)

    loop.run_pending_evals(runner)
    while loop.store.pending_traces(1):          # a worker or a resume would drain this
        loop.process()

    assert learned(loop)["verify-pip-install"] == "active", "a passing re-run must reach active and stay"
    failed = loop.store.db.execute("SELECT COUNT(*) FROM evals WHERE status='failed'").fetchone()[0]
    assert failed == 0, "no host eval should have failed; a second self-inflicted eval is the bug"
    n_after = loop.store.db.execute("SELECT COUNT(*) FROM traces").fetchone()[0]
    assert n_after > n_before, "the re-run is still recorded, for the audit trail"
    assert not loop.store.pending_traces(1), "but it must not sit in the learning queue"


def test_daily_quota_is_distinguished_from_a_burst_rate_limit():
    """A daily cap and a per-minute cap are both HTTP 429 and must not be treated the same.

    A burst limit clears in seconds and retrying is correct. A daily limit clears at the provider's reset,
    so the retry budget is spent in silence and the run looks like a hung process: no output, no error, and
    every key still answering a trivial probe. A seed-4 run lost 13 minutes to exactly this before the cause
    (Groq's 200k tokens-per-day, shared across every key in one organization) was visible at all.
    """
    from skillloop.llm import _is_daily_quota

    daily = ("Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01km` service tier "
             "`on_demand` on tokens per day (TPD): Limit 200000, Used 199716, Requested 613.")
    burst = "Rate limit reached on tokens per minute (TPM): Limit 8000, Used 7900. Please try again in 1.77s"
    assert _is_daily_quota(daily)
    assert not _is_daily_quota(burst), "a burst limit must still be retried"


def test_process_stops_and_says_so_when_the_quota_runs_out(loop):
    """Every remaining trace would fail identically, so stop, keep them queued, and explain."""
    import skillloop.core as core
    from skillloop.llm import QuotaExhausted

    for i in range(4):
        run_agent(loop, f"install pandas and build report {i}")
    original = core.SkillLoop._process_one

    def out_of_quota(self, t):
        raise QuotaExhausted("daily quota exhausted")

    core.SkillLoop._process_one = out_of_quota
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            report = loop.process()
    finally:
        core.SkillLoop._process_one = original

    assert any("quota_exhausted" in str(r) for r in report)
    assert loop.store.pending_traces(99), "traces must stay queued for a later run, not be dropped"
    assert [w for w in caught if "stopped processing" in str(w.message)], "the stop must be announced"
