"""Embed SkillLoop directly in your own agent loop (no server)."""
from skillloop import SkillLoop, Signal

loop = SkillLoop(auto_process=True)   # background worker reflects/gates every 15s

def run_agent(task: str):
    skills = loop.recall(task)                       # 1. before: fast, local
    system_extra = "\n\n".join(s["skill_md"] for s in skills)
    messages = my_agent(task, extra_context=system_extra)   # your agent, your format
    ok = my_tests_passed()                           # whatever outcome signal you have
    loop.learn({                                     # 2. after: fire-and-forget
        "task": task,
        "format": "openai",                          # or "anthropic", or canonical "steps"
        "messages": messages,
        "skills_used": [s["name"] for s in skills],
        "signals": [{"source": "test", "outcome": "success" if ok else "failure", "confidence": 0.9}],
        "agent": "my-agent",
    })
