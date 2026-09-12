"""HTTP door. stdlib only, no framework. Run: `skillloop http --port 7331`

  GET  /recall?task=...&limit=3
  POST /learn            body: trace payload (canonical / openai / anthropic)
  GET  /pending_evals
  POST /process
  GET  /inspect[?name=]
  POST /rollback         {"name","version"}
  POST /status           {"name","status"}
"""
from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .core import SkillLoop

_loop: SkillLoop | None = None


def loop() -> SkillLoop:
    global _loop
    if _loop is None:
        _loop = SkillLoop(auto_process=os.getenv("SKILLLOOP_AUTOPROCESS", "1") == "1")
    return _loop


class H(BaseHTTPRequestHandler):
    def _send(self, obj, code=200):
        data = json.dumps(obj, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path == "/recall":
                return self._send(loop().recall(q.get("task", ""), int(q.get("limit", 3))))
            if u.path == "/pending_evals":
                return self._send(loop().pending_evals(int(q.get("limit", 5))))
            if u.path == "/inspect":
                return self._send(loop().inspect(q.get("name")))
            if u.path == "/metrics":
                return self._send(loop().metrics())
            if u.path == "/health":
                # liveness + readiness: confirm the DB answers, report version and library size
                try:
                    st = loop().store.stats()
                    return self._send({"ok": True, "version": _version(), "skills": st["skills"],
                                       "traces": st["traces"]})
                except Exception as e:
                    return self._send({"ok": False, "error": repr(e)}, 503)
            if u.path == "/version":
                return self._send({"version": _version()})
            self._send({"error": "not found"}, 404)
        except Exception as e:
            self._send({"error": repr(e)}, 500)

    def do_POST(self):
        u = urlparse(self.path)
        try:
            b = self._body()
            # recall is also available as POST: task descriptions are long and URL-encoding them is awkward
            if u.path == "/recall":
                return self._send(loop().recall(b.get("task", ""), int(b.get("limit", 3))))
            if u.path == "/learn":
                return self._send(loop().learn(b))
            if u.path == "/process":
                return self._send(loop().process())
            if u.path == "/rollback":
                return self._send(loop().rollback(b["name"], int(b["version"])))
            if u.path == "/status":
                return self._send(loop().set_status(b["name"], b["status"]))
            if u.path == "/approve":
                return self._send(loop().approve(b["name"]))
            self._send({"error": "not found"}, 404)
        except Exception as e:
            self._send({"error": repr(e)}, 500)

    def log_message(self, *a):  # quiet
        pass


def _version():
    try:
        from . import __version__
        return __version__
    except Exception:
        return "unknown"


def main(port: int = 7331):
    import signal
    srv = ThreadingHTTPServer(("0.0.0.0", port), H)
    srv.daemon_threads = True

    def _shutdown(signum, frame):
        print("skillloop http: draining and shutting down")
        # stop accepting, let in-flight requests finish (daemon threads + WAL make this safe), close DB
        import threading
        threading.Thread(target=srv.shutdown, daemon=True).start()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _shutdown)
        except ValueError:
            pass  # not on the main thread (e.g. under a test)
    print(f"skillloop http on :{port}  (home={loop().store.home}, version={_version()})")
    try:
        srv.serve_forever()
    finally:
        try:
            loop().store.close()
        except Exception:
            pass
        print("skillloop http: stopped")


if __name__ == "__main__":
    main()
