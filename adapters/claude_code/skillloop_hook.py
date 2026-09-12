#!/usr/bin/env python3
"""Claude Code adapter. Hooks force the loop — the model does not have to remember to call anything.

  UserPromptSubmit -> `skillloop_hook.py prompt`  : recall skills for the prompt, print them (added to context)
  Stop             -> `skillloop_hook.py stop`    : read the session transcript, submit it as a trace

Install: copy the hooks block from settings.example.json into ~/.claude/settings.json (or .claude/settings.json
in a repo) and make sure `skillloop` is importable by the python3 on PATH (pip install skillloop).

Talks to the local library directly (no server needed). Set SKILLLOOP_HTTP=http://host:7331 to use a server instead.
Input fields verified against the Claude Code hooks reference (code.claude.com/docs/en/hooks), Sep 2026:
  common envelope : session_id, transcript_path, cwd, hook_event_name
  UserPromptSubmit: + prompt   (stdout from this hook is ADDED TO CONTEXT)
  Stop/SubagentStop: + stop_hook_active, last_assistant_message   (Stop may omit cwd)
  transcript      : JSONL, each line carrying `message: {role, content}` in Anthropic format
Exit codes: 0 = success; 2 = BLOCKING (stderr fed back to Claude); anything else = non-blocking error shown
to the user. This hook always exits 0 - a memory system must never block the user's turn.

Known limitation: on `--continue` / `--resume`, Claude Code replays saved hook text rather than re-running
the hook, so recalled skills from an earlier turn can be stale in a resumed session.

Run `python3 skillloop_hook.py doctor` to validate an installation.
"""
import json
import os
import sys
import urllib.request

_ARGV_MODE = sys.argv[1] if len(sys.argv) > 1 else None

# Hook event names -> our two modes. Deriving the mode from the payload means a settings.json that forgets
# the argument still does the right thing instead of silently exiting 0, which is the worst way for a hook
# to fail: the user sees no error and no skills, and has nothing to debug.
_EVENT_MODE = {"UserPromptSubmit": "prompt", "Stop": "stop", "SubagentStop": "stop"}


def resolve_mode(payload):
    event = payload.get("hook_event_name")
    if event in _EVENT_MODE:
        mode = _EVENT_MODE[event]
        if _ARGV_MODE and _ARGV_MODE != mode:
            print(f"[skillloop hook: settings.json passes '{_ARGV_MODE}' but the event is '{event}'; "
                  f"using '{mode}']", file=sys.stderr)
        return mode
    if _ARGV_MODE in ("prompt", "stop"):
        return _ARGV_MODE
    print(f"[skillloop hook: cannot tell which hook this is - no known hook_event_name "
          f"({event!r}) and no 'prompt'/'stop' argument]", file=sys.stderr)
    return None
HTTP = os.getenv("SKILLLOOP_HTTP")


def call(kind, **kw):
    if HTTP:
        if kind == "recall":
            q = urllib.request.quote(kw["task"])
            return json.load(urllib.request.urlopen(f"{HTTP}/recall?task={q}&limit={kw.get('limit', 3)}", timeout=3))
        req = urllib.request.Request(f"{HTTP}/learn", data=json.dumps(kw["payload"]).encode(),
                                     headers={"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req, timeout=5))
    from skillloop import SkillLoop
    loop = SkillLoop()
    return loop.recall(kw["task"], kw.get("limit", 3)) if kind == "recall" else loop.learn(kw["payload"])


def read_transcript(path):
    """Claude Code transcripts are JSONL; each line with a `message` is an Anthropic-format message."""
    msgs, first_prompt = [], None
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            m = rec.get("message")
            if not isinstance(m, dict) or "role" not in m:
                continue
            msgs.append({"role": m["role"], "content": m.get("content")})
            if first_prompt is None and m["role"] == "user":
                c = m.get("content")
                text = c if isinstance(c, str) else " ".join(b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text")
                if text.strip():
                    first_prompt = text.strip()
    return first_prompt or "unknown task", msgs


def doctor():
    """Validate an installation without needing a live Claude Code session."""
    ok = True
    print("skillloop hook doctor")
    try:
        hits = call("recall", task="a representative task description for checking connectivity")
        print(f"  [ok]   reached skillloop, recall returned {len(hits)} skill(s)")
    except Exception as e:
        ok = False
        print(f"  [FAIL] cannot reach skillloop: {e}")
        print("         start it with `skillloop http` or set SKILLLOOP_URL / SKILLLOOP_HOME")
    settings = os.path.expanduser("~/.claude/settings.json")
    if os.path.exists(settings):
        try:
            cfg = json.load(open(settings)).get("hooks", {})
            for ev in ("UserPromptSubmit", "Stop"):
                found = json.dumps(cfg.get(ev, [])).find("skillloop_hook") >= 0
                print(f"  [{'ok' if found else 'WARN'}] {ev} hook {'registered' if found else 'NOT registered'} in {settings}")
                ok = ok and found
        except Exception as e:
            print(f"  [WARN] could not parse {settings}: {e}")
    else:
        print(f"  [WARN] {settings} not found - register the hooks from settings.example.json")
    print("RESULT:", "ready" if ok else "not ready - fix the items above")
    return 0 if ok else 1


def main():
    if _ARGV_MODE == "doctor":
        sys.exit(doctor())
    try:
        inp = json.load(sys.stdin)
    except Exception:
        inp = {}
    mode = resolve_mode(inp)
    if mode is None:
        return
    if mode == "prompt":
        task = inp.get("prompt", "")
        if len(task) < 20:
            return
        hits = call("recall", task=task)
        if not hits:
            return
        # UserPromptSubmit stdout is added to the model's context. Claude Code's prompt-injection defences
        # fire on text framed as out-of-band system commands ("Follow these instructions"), which makes it
        # surface the block to the user instead of using it. Phrase it as PROJECT INFORMATION instead.
        print("Notes from earlier work in this project. These were written after previous attempts at "
              "similar tasks failed, and record what worked:\n")
        for h in hits:
            prov = h.get("provenance", {})
            print(f"### {h['name']}  (learned from {prov.get('traces', '?')} past runs, "
                  f"used {h['uses']}x, success rate {h['hit_rate']})\n{h['skill_md']}\n")
    elif mode == "stop":
        if inp.get("stop_hook_active"):
            return                      # re-entry guard: we are running because of our own Stop hook
        path = inp.get("transcript_path")
        if not path:
            print("[skillloop hook: Stop payload had no transcript_path]", file=sys.stderr)
            return
        if not os.path.exists(path):
            print(f"[skillloop hook: transcript not found at {path}]", file=sys.stderr)
            return
        task, msgs = read_transcript(path)
        # Stop/SubagentStop also carry the final assistant message; use it when the transcript tail is empty
        last = inp.get("last_assistant_message")
        if last and not msgs:
            msgs = [{"role": "assistant", "content": last}]
        if not msgs:
            print(f"[skillloop hook: no messages parsed from {path} - transcript format may have changed]",
                  file=sys.stderr)
            return
        # skills the model mentioned by name count as used
        text = json.dumps(msgs)
        used = [h["name"] for h in call("recall", task=task, limit=10) if h["name"] in text]
        payload = {"task": task, "format": "anthropic", "messages": msgs[-400:], "agent": "claude-code",
                   "skills_used": used, "metadata": {"session_id": inp.get("session_id")}}
        call("learn", payload=payload)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # never break the host agent
        print(f"[skillloop hook error: {e}]", file=sys.stderr)
