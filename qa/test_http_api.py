import json, os, subprocess, sys, tempfile, time, urllib.request
env = dict(os.environ, SKILLLOOP_HOME=tempfile.mkdtemp(), SKILLLOOP_PROVIDER="fake")
p = subprocess.Popen([sys.executable,"-m","skillloop.cli","http","--port","8899"],
                     env=env, cwd="/home/claude/skillloop", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(2)
def call(path, body=None):
    req = urllib.request.Request(f"http://127.0.0.1:8899{path}",
        data=json.dumps(body).encode() if body else None,
        headers={"Content-Type":"application/json"}, method="POST" if body else "GET")
    return json.load(urllib.request.urlopen(req, timeout=10))
try:
    print("GET /health ->", call("/health"))
    print("GET  /recall ->", call("/recall?task=pip+install+something"))
    print("POST /recall ->", call("/recall", {"task":"pip install something and import it"}))
    r = call("/learn", {"task":"install requests","steps":[{"role":"assistant","content":"","tool_calls":[{"name":"bash","args":{"cmd":"pip install requests"},"error":"externally-managed-environment"}]}],"signals":[{"source":"test","outcome":"failure","confidence":0.9}]})
    print("POST /learn ->", {k:r[k] for k in ("outcome","queued") if k in r})
    print("GET /metrics ->", list(call("/metrics").keys())[:4])
    print("RESULT: PASS")
finally:
    p.terminate()
