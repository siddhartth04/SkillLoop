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
