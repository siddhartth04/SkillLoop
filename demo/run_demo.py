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
        # The offline model gives canned answers about pip installs, so the offline demo uses a pip failure:
        # the story it tells (fail -> learn -> recalled next time) has to be true for the answers it gets.
        from tests.test_loop import fake_handler, failing_trace
        LLM._fake_handler = staticmethod(fake_handler)
        loop = SkillLoop(home=home, llm=LLM(provider="fake"))
        trace = failing_trace()
        next_task = "how do I get the requests package working, the import keeps failing"
        unrelated_task = "generate the monthly usage export"
        mid_task_error = "Traceback (most recent call last):\n  File \"export.py\", line 1\nModuleNotFoundError: No module named 'requests'"
    else:
        loop = SkillLoop(home=home)
        trace = failing
        next_task = "how do I point nginx at the new upstream port and reload it"
        unrelated_task = "roll out the new API release"
        mid_task_error = "Job for nginx.service failed. See 'systemctl status nginx.service'"

    print("== 1. recall before learning ==")
    print([h["name"] for h in loop.recall(next_task)] or "  (nothing known)")

    print("\n== 2. the agent fails; submit the trace ==")
    print(loop.learn(trace))

    print("\n== 3. process: reflect -> synthesize -> judge gate ==")
    for r in loop.process():
        print(json.dumps(r, indent=2))

    print("\n== 4. what the library looks like now ==")
    for s in loop.store.all_skills():
        print(f"- {s.name} [{s.status}] v{s.version}")
        print("  " + (loop.store.skills_dir / s.name / "SKILL.md").read_text().replace("\n", "\n  ")
              if s.status in ("active", "candidate") else "  (quarantined; see quarantine/)")

    print(f"\n== 5. next time, phrased differently: {next_task!r} ==")
    hits = [h for h in loop.recall(next_task) if h["name"] != "_principles"]
    for h in hits:
        print(f"  -> {h['name']} ({h['status']})")
    if not hits:
        print("  (nothing recalled)")

    print(f"\n== 6. a task that gives no hint ({unrelated_task!r}) hits the error mid-task ==")
    with loop.session(unrelated_task, auto_learn=False) as s:
        s.tool("bash", {"cmd": "run"}, error=mid_task_error)
        hint = s.hints()
        for h in hint:
            print(f"  -> error-time recall: {h['name']} ({h['status']}) is handed to the agent before its next step")
        if not hint:
            print("  (nothing recalled)")

    print("\n== 7. verification re-run, automated ==")
    def runner(task, skill_md):
        # your agent goes here: run `task` with `skill_md` in its prompt, judge the outcome with a real check
        s = loop.session(task, auto_learn=False)
        s.tool("bash", {"cmd": "follow the skill"}, result="ok")
        return s.verified_by(0)
    for r in loop.run_pending_evals(runner):
        print(f"  eval {r['eval_id']} for {r['skill']}: {r['outcome']}")
    for s in loop.store.all_skills():
        print(f"  {s.name} is now [{s.status}]")
    print(f"\nlibrary home: {home}")


if __name__ == "__main__":
    main()
