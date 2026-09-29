"""The conventions eval harness must be valid before it is allowed to judge SkillLoop."""
from qa.conv_eval.generator import generate
from qa.conv_eval.harness import HarnessError, execute, extract_code
from qa.conv_eval.run import gates, oracle_context, run_conditions, ScriptedAgent
import pytest


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_every_trap_has_headroom_and_every_reference_passes(seed):
    s = generate(seed)
    for t in s.tasks:
        assert execute(s, t, t.reference)["ok"], t.id
        assert execute(s, t, t.naive)["ok"] is (not t.trap), t.id


def test_truncated_output_is_a_harness_error_not_an_agent_failure():
    with pytest.raises(HarnessError):
        extract_code("```python\nresult = (")


def test_gates_pass_only_with_real_headroom():
    s = generate(7)
    trap = [t.id for t in s.split("test") if t.trap]
    r = run_conditions(ScriptedAgent(s), s, trap, ["control", "control_repeat", "oracle"],
                       {"oracle": oracle_context(s)})
    g = gates(r["control"], r["control_repeat"], r["oracle"], trap)
    assert g["pass"] and g["control_rate"] == 0.0 and g["oracle_rate"] == 1.0


def test_analysis_splits_by_what_the_training_feedback_revealed():
    """The committed seed-2 run must still show the split the analysis is built on.

    A task whose training attempt only ever printed "check failed" never showed the learner the correct
    value, so raw past attempts cannot supply it. That is the case a learned skill has to earn its keep on,
    and it is the split the UNKNOWN VALUES rule in the synthesis prompt is aimed at.
    """
    from qa.conv_eval.analyze import informative_conventions
    from pathlib import Path
    import qa.conv_eval.analyze as A

    info = informative_conventions(Path(A.HERE) / "results" / "seed2_episodes")
    assert info, "seed-2 training episodes must be committed for the analysis to work"
    # keys and units never produced a traceback; cents and inplace did
    assert info["keys"] == 0 and info["units"] == 0
    assert info["cents"] > 0 and info["inplace"] > 0


def test_sign_test_matches_known_values():
    from qa.conv_eval.analyze import sign_test
    assert sign_test(0, 0) == 1.0
    assert sign_test(4, 0) == 0.125
    assert round(sign_test(8, 0), 5) == 0.00781
