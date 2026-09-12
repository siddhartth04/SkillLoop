"""S2: concurrency. SQLite WAL is configured but was never tested with parallel agents."""
import sys, threading, time, tempfile
sys.path.insert(0, "/home/claude/skillloop")
sys.path.insert(0, "/home/claude/skillloop/tests")
from skillloop import SkillLoop, Signal, Step, ToolCall, Trace
from skillloop.llm import LLM

home = tempfile.mkdtemp()
loop = SkillLoop(home=home, llm=LLM(provider="fake"))
errors, learned = [], []

def trace(i):
    return Trace(task=f"task {i}", agent="conc",
                 steps=[Step("assistant", "", [ToolCall("bash", {"cmd": f"echo {i}"}, error="boom")])],
                 signals=[Signal("test", "failure", 0.9)])

def worker(n, wid):
    for i in range(n):
        try:
            r = loop.learn(trace(f"{wid}-{i}"))
            learned.append(r["trace_id"])
            loop.recall("echo something to a file")
        except Exception as e:
            errors.append(repr(e))

threads = [threading.Thread(target=worker, args=(25, w)) for w in range(8)]
t0 = time.time()
[t.start() for t in threads]; [t.join() for t in threads]
dur = time.time() - t0
stored = loop.store.stats()["traces"]
print(f"8 threads x 25 learn+recall in {dur:.1f}s")
print(f"submitted {len(learned)}  stored {stored}  errors {len(errors)}")
if errors:
    print("first errors:", errors[:3])
print("RESULT:", "PASS - no corruption, no lost traces" if (not errors and stored == 200 and len(set(learned)) == 200)
      else "FAIL")
