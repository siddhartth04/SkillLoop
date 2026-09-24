"""CLI. `skillloop <cmd>`"""
from __future__ import annotations

import argparse
import json
import sys

from .core import SkillLoop


def main(argv=None):
    p = argparse.ArgumentParser(prog="skillloop", description="Test-gated skill learning for any agent.")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("mcp", help="run MCP server (stdio)")
    h = sub.add_parser("http", help="run HTTP server"); h.add_argument("--port", type=int, default=7331)
    r = sub.add_parser("recall"); r.add_argument("task"); r.add_argument("--limit", type=int, default=3)
    l = sub.add_parser("learn", help="submit a trace JSON file (or - for stdin)"); l.add_argument("file")
    sub.add_parser("process", help="run the learning loop over queued traces")
    i = sub.add_parser("inspect"); i.add_argument("name", nargs="?")
    rb = sub.add_parser("rollback"); rb.add_argument("name"); rb.add_argument("version", type=int)
    st = sub.add_parser("status"); st.add_argument("name"); st.add_argument("status")
    sub.add_parser("evals", help="pending host-run evals")
    sub.add_parser("path", help="print the skills folder to point agents at")
    ap = sub.add_parser("approve", help="human-approve a skill (needed when SKILLLOOP_REQUIRE_APPROVAL=1)"); ap.add_argument("name")
    sub.add_parser("backfill-queries", help="doc2query: generate user-phrased trigger queries for existing skills")
    sub.add_parser("backfill-facets", help="add retrieval facets to skills created before facets existed")
    sub.add_parser("metrics", help="success rate over episodes, library size, regression rate")
    sub.add_parser("principles", help="print distilled cross-cutting principles")
    sub.add_parser("calibrate", help="run the synthetic corpus through the real model and report prompt quality (args: --limit N --out FILE)")
    a, rest = p.parse_known_args(argv)

    if a.cmd == "mcp":
        from .server_mcp import main as m; return m()
    if a.cmd == "http":
        from .server_http import main as m; return m(a.port)
    if a.cmd == "calibrate":
        from .calibrate import main as m; return m(rest)

    loop = SkillLoop()
    out = None
    if a.cmd == "recall":
        out = loop.recall(a.task, a.limit)
    elif a.cmd == "learn":
        raw = sys.stdin.read() if a.file == "-" else open(a.file, encoding="utf-8").read()
        out = loop.learn(json.loads(raw))
    elif a.cmd == "process":
        out = loop.process()
    elif a.cmd == "inspect":
        out = loop.inspect(a.name)
    elif a.cmd == "rollback":
        out = loop.rollback(a.name, a.version)
    elif a.cmd == "status":
        out = loop.set_status(a.name, a.status)
    elif a.cmd == "evals":
        out = loop.pending_evals()
    elif a.cmd == "path":
        print(loop.export_dir()); return
    elif a.cmd == "approve":
        out = loop.approve(a.name)
    elif a.cmd == "backfill-queries":
        from .learner import backfill_queries
        out = {"updated": backfill_queries(loop.llm, loop.store)}
    elif a.cmd == "backfill-facets":
        from .learner import backfill_facets
        out = {"updated": backfill_facets(loop.llm, loop.store)}
    elif a.cmd == "metrics":
        out = loop.metrics()
    elif a.cmd == "principles":
        print(loop.store.principles_text() or "(none yet)"); return
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
