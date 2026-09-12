"""A2: exercise the Claude Code hook with realistic payloads. Cannot launch Claude Code here, but this
tests everything the hook itself does: stdin contract, transcript JSONL parsing, stdout contract."""
import json, os, subprocess, sys, tempfile

HOOK = "/home/claude/skillloop/adapters/claude_code/skillloop_hook.py"
home = tempfile.mkdtemp()
env = dict(os.environ, SKILLLOOP_HOME=home, SKILLLOOP_PROVIDER="fake",
           PYTHONPATH="/home/claude/skillloop")

# a realistic Claude Code transcript: JSONL, one message per line
tdir = tempfile.mkdtemp()
tpath = os.path.join(tdir, "transcript.jsonl")
with open(tpath, "w") as f:
    for m in [
        {"type": "user", "message": {"role": "user", "content": "install requests and fetch a page"}},
        {"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "text", "text": "Installing."},
            {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "pip install requests"}}]}},
        {"type": "user", "message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "is_error": True,
             "content": "error: externally-managed-environment"}]}},
        {"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "text", "text": "It failed."}]}},
    ]:
        f.write(json.dumps(m) + "\n")

def run(payload):
    p = subprocess.run([sys.executable, HOOK], input=json.dumps(payload), capture_output=True,
                       text=True, env=env, timeout=60)
    return p.returncode, p.stdout.strip(), p.stderr.strip()[:300]

print("--- UserPromptSubmit (empty library) ---")
rc, out, err = run({"hook_event_name": "UserPromptSubmit", "prompt": "install a python package",
                    "session_id": "s1", "transcript_path": tpath, "cwd": "/tmp"})
print("exit", rc, "| stdout:", (out[:120] or "(empty)"), "| stderr:", err or "(none)")
assert rc == 0, f"hook must exit 0 or it blocks the user's prompt: {err}"

print("--- Stop (submits the transcript) ---")
rc, out, err = run({"hook_event_name": "Stop", "session_id": "s1", "transcript_path": tpath,
                    "stop_hook_active": False, "cwd": "/tmp"})
print("exit", rc, "| stderr:", err or "(none)")
assert rc == 0, err

sys.path.insert(0, "/home/claude/skillloop")
from skillloop.store import Store
st = Store(home)
traces = st.pending_traces(5)
print(f"traces captured: {len(traces)}")
if traces:
    t = traces[0]
    print("  task:", t.task[:60])
    print("  tool calls:", t.n_tool_calls, "| outcome:", t.outcome)
    errs = [tc.error for s in t.steps for tc in s.tool_calls if tc.error]
    print("  errors parsed:", errs[:1])

print("--- injected text must read as project info, not out-of-band commands ---")
# seed a real skill so recall has something to inject
sys.path.insert(0, "/home/claude/skillloop")
from skillloop import SkillLoop
from skillloop.llm import LLM
from skillloop.store import Skill
from skillloop.playbook import new_bullet
_lp = SkillLoop(home=home, llm=LLM(provider="fake"))
_lp.store.save_skill(Skill(name="verify-pip-install", description="Install and verify python packages.",
    body="", status="candidate",
    bullets=[new_bullet("procedure", "Run `pip install <pkg>` and check the exit code."),
             new_bullet("verification", "`python -c 'import <pkg>'` exits 0.")],
    facets={"applies_when": ["pip install", "python package"], "symptoms": [], "not_for": []},
    queries=["install a python package and import it", "pip install fails"]), triggers=["pip install"], snapshot=False)
rc, out, err = run({"hook_event_name": "UserPromptSubmit", "prompt": "install a python package and import it",
                    "session_id": "s1", "transcript_path": tpath, "cwd": "/tmp"})
banned = ["<skillloop", "Follow them", "You must", "SYSTEM:", "IMPORTANT:"]
hits = [b for b in banned if b.lower() in out.lower()]
print("injected chars:", len(out), "| injection-defence triggers:", hits or "none")
assert not hits, f"phrasing likely to trip prompt-injection defences: {hits}"

print("--- last_assistant_message fallback (empty transcript) ---")
empty = os.path.join(tdir, "empty.jsonl"); open(empty, "w").close()
rc, out, err = run({"hook_event_name": "Stop", "session_id": "s2", "transcript_path": empty,
                    "stop_hook_active": False, "last_assistant_message": "I finished the task."})
print("exit", rc, "| stderr:", (err or "(none)")[:80])

print("--- doctor ---")
p2 = subprocess.run([sys.executable, HOOK, "doctor"], capture_output=True, text=True, env=env, timeout=60)
print(p2.stdout.strip()[-120:])

print("--- Stop re-entry guard (stop_hook_active=True must not loop) ---")
n_before = len(Store(home).pending_traces(20))
rc, out, err = run({"hook_event_name": "Stop", "session_id": "s1", "transcript_path": tpath,
                    "stop_hook_active": True, "cwd": "/tmp"})
n_after = len(Store(home).pending_traces(20))
print("exit", rc, f"| traces {n_before} -> {n_after}")
assert n_after == n_before, "re-entry guard failed: stop_hook_active=True still submitted a trace"

ok = traces and traces[0].n_tool_calls >= 1 and any(tc.error for s in traces[0].steps for tc in s.tool_calls)
print("\nRESULT:", "PASS - transcript parsed, tool errors preserved" if ok else "FAIL")
