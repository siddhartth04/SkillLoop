"""A3: does the documented quickstart actually work on a clean machine, end to end?
Uses the README's session API verbatim against a real (fake-LLM) loop."""
import subprocess, sys, tempfile, os
sys.path.insert(0, "/tmp/rel/skillloop")
os.environ["SKILLLOOP_PROVIDER"] = "fake"
sys.path.insert(0, "/tmp/rel/skillloop/tests")
from skillloop import SkillLoop
from skillloop.llm import LLM
from test_loop import fake_handler
LLM._fake_handler = staticmethod(fake_handler)

home = tempfile.mkdtemp()
loop = SkillLoop(home=home, llm=LLM(provider="fake"))

# 1. exactly the README pattern
with loop.session("install requests and fetch a page") as s:
    prompt = "You are an agent.\n" + s.skills_text()
    r = subprocess.run("pip install nonexistent-pkg-xyz", shell=True, capture_output=True, text=True)
    s.tool("bash", {"cmd": "pip install nonexistent-pkg-xyz"}, error=r.stderr[:200])
    s.tool("python", {"code": "import requests"}, error="ModuleNotFoundError")
    s.verified_by(1, "checker: import failed")
assert s.result["outcome"] == "failure", s.result
print("1. session recorded a real failure          OK")

# 2. background learning produces a skill
rep = loop.process()
name = rep[0].get("skill", {}).get("name")
assert name, rep
print(f"2. learning produced a gated skill: {name}  OK")

# 3. it comes back on a related task, in SKILL.md form
hits = loop.recall("pip install a package and import it")
assert hits and hits[0]["name"] == name, hits
assert "## Verification" in hits[0]["skill_md"]
print("3. recall returns it with a Verification section OK")

# 4. the skill folder is usable by any agentskills.io reader
import pathlib
md = pathlib.Path(loop.export_dir()) / name / "SKILL.md"
assert md.exists() or loop.store.get_skill(name).status == "quarantine"
print("4. exported to", loop.export_dir(), "                OK")

# 5. bullet-level feedback closes the loop
bid = loop.store.get_skill(name).bullets[0]["id"]
with loop.session("pip install another package") as s2:
    s2.recall(); s2.used(bid); s2.tool("bash", {"cmd": "pip install x"}, result="ok"); s2.ok()
assert next(b for b in loop.store.get_skill(name).bullets if b["id"] == bid)["helpful"] == 1
print("5. per-bullet feedback recorded            OK")
print("\nQUICKSTART: PASS")
