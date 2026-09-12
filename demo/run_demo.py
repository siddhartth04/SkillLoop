"""Demo: an agent fails a task, SkillLoop learns, the next run has a gated skill to follow.

    export ANTHROPIC_API_KEY=...   (or OPENAI_API_KEY [+ OPENAI_BASE_URL])
    python demo/run_demo.py            # real reflector model
    python demo/run_demo.py --fake     # offline, canned model answers

Uses a fresh temp SKILLLOOP_HOME so it never touches your real library.
"""
import json
import sys
import tempfile

sys.path.insert(0, ".")
from skillloop import SkillLoop, Signal, Step, ToolCall, Trace
from skillloop.llm import LLM

FAKE = "--fake" in sys.argv

# A realistic failure: agent edits a config, restarts a service, never checks the restart worked,
# then declares victory. User comes back angry.
failing = Trace(
    task="change the nginx upstream port from 8000 to 8080 and reload nginx",
    agent="demo-agent",
    steps=[
        Step("user", "change the nginx upstream port from 8000 to 8080 and reload nginx"),
        Step("assistant", "Editing the config.", [
            ToolCall("bash", {"cmd": "sed -i 's/8000/8080/' /etc/nginx/conf.d/app.conf"}, result=""),
        ]),
        Step("assistant", "Reloading.", [
            ToolCall("bash", {"cmd": "systemctl reload nginx"},
                     error="Job for nginx.service failed. See 'systemctl status nginx.service' and 'journalctl -xeu nginx.service'."),
        ]),
        Step("assistant", "Done! nginx is now pointing at port 8080."),
        Step("user", "no it isn't, the site is down now"),
    ],
    signals=[Signal("user", "failure", 0.95, "site is down after change")],
)


def main():
    home = tempfile.mkdtemp(prefix="skillloop-demo-")
    if FAKE:
        from tests.test_loop import fake_handler
        LLM._fake_handler = staticmethod(fake_handler)
        loop = SkillLoop(home=home, llm=LLM(provider="fake"))
    else:
        loop = SkillLoop(home=home)

    print("== 1. recall before learning ==")
    print(loop.recall("update nginx config and reload") or "  (nothing known)")

    print("\n== 2. submit the failing trace ==")
    print(loop.learn(failing))

    print("\n== 3. process: reflect -> synthesize -> judge gate ==")
    for r in loop.process():
        print(json.dumps(r, indent=2))

    print("\n== 4. what the library looks like now ==")
    for s in loop.store.all_skills():
        print(f"- {s.name} [{s.status}] v{s.version}")
        print("  " + (loop.store.skills_dir / s.name / "SKILL.md").read_text().replace("\n", "\n  ")
              if s.status in ("active", "candidate") else "  (quarantined; see quarantine/)")

    print("\n== 5. recall for the next similar task ==")
    for h in loop.recall("edit nginx conf and reload the service"):
        print(f"  -> {h['name']} ({h['status']})")

    pend = loop.pending_evals()
    if pend:
        print(f"\n== 6. host-run eval queued: {pend[0]['id']} ==")
        print("  assertions:", *pend[0]["assertions"], sep="\n   - ")
        print("  (Run that task in your agent with the skill, then learn(trace, eval_id=...) to promote it to active.)")
    print(f"\nlibrary home: {home}")


if __name__ == "__main__":
    main()
