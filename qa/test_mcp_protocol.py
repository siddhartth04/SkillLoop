import json, subprocess, sys, tempfile, os
env = dict(os.environ, SKILLLOOP_HOME=tempfile.mkdtemp(), SKILLLOOP_PROVIDER="fake")
p = subprocess.Popen([sys.executable, "-m", "skillloop.cli", "mcp"], stdin=subprocess.PIPE,
                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, cwd="/home/claude/skillloop")

def send(obj):
    p.stdin.write(json.dumps(obj) + "\n"); p.stdin.flush()

def read():
    line = p.stdout.readline()
    return json.loads(line) if line.strip() else None

send({"jsonrpc":"2.0","id":1,"method":"initialize","params":{
     "protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"qa","version":"1"}}})
r = read()
print("initialize ->", r["result"]["serverInfo"] if r and "result" in r else r)
send({"jsonrpc":"2.0","method":"notifications/initialized"})
send({"jsonrpc":"2.0","id":2,"method":"tools/list"})
r = read()
tools = [t["name"] for t in r["result"]["tools"]]
print("tools/list ->", tools)
send({"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"skill_recall","arguments":{"task":"pip install a package"}}})
r = read()
print("skill_recall ->", str(r["result"]["content"][0]["text"])[:80])
p.terminate()
required = {"skill_recall","skill_learn"}
print("RESULT:", "PASS" if required <= set(tools) else f"FAIL missing {required - set(tools)}")
