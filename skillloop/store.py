"""Persistence. SQLite for state, plain folders for skills (agentskills.io format).

Layout under SKILLLOOP_HOME (default ~/.skillloop):
  skillloop.db
  skills/<name>/SKILL.md          active + candidate skills (readable by any agent)
  quarantine/<name>/SKILL.md      failed gate / demoted
  history/<name>/v<N>.md          every version, for rollback
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from .schema import Trace, from_dict

SkillStatus = str  # "quarantine" | "candidate" | "active" | "retired"


@dataclass
class Lesson:
    id: str
    trace_id: str
    outcome: str
    root_cause: str
    rule: str                 # the generalizable "when X, do Y"
    triggers: list[str]       # phrases/contexts where this applies
    generalizable: bool
    blame: str                # "none" | "skill_wrong" | "skill_incomplete" | "skill_misapplied" | "no_skill"
    skill_ref: str | None
    confidence: float
    ts: float = field(default_factory=time.time)
    verification: str = ""            # v0.2: how success could have been proven
    principle: str | None = None      # v0.2: broader rule this instantiates, if any
    suggested_name: str | None = None
    evidence_steps: list[int] = field(default_factory=list)   # v0.2: trace step indices that demonstrate the cause
    skeptic: dict | None = None                                 # v0.2: adversarial grounding check result


@dataclass
class Skill:
    name: str
    description: str
    body: str
    status: SkillStatus = "quarantine"
    version: int = 1
    provenance: list[str] = field(default_factory=list)   # trace ids
    uses: int = 0
    successes: int = 0
    failures: int = 0
    last_used: float = 0.0
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)
    # v0.2
    recalls: int = 0                  # times returned by recall() (trigger precision = uses / recalls)
    provisional: bool = False         # written from a single incident; treat with suspicion
    evidence_count: int = 1           # distinct lessons backing this skill
    security: dict[str, Any] = field(default_factory=dict)   # {"regex": [...], "judge": {...}, "cleared": bool}
    approved: bool = False            # human approval (only matters when SKILLLOOP_REQUIRE_APPROVAL=1)
    embedding: list[float] | None = None
    facets: dict[str, list[str]] = field(default_factory=dict)  # applies_when / symptoms / not_for
    queries: list[str] = field(default_factory=list)            # doc2query: how users actually ask for this
    bullets: list[dict] = field(default_factory=list)           # itemized body; see playbook.py
    lesson_ids: list[str] = field(default_factory=list)         # which lessons actually contributed

    @property
    def hit_rate(self) -> float:
        return self.successes / self.uses if self.uses else 0.0

    @property
    def trigger_precision(self) -> float:
        return self.uses / self.recalls if self.recalls else 0.0

    @property
    def body_md(self) -> str:
        from .playbook import render
        return render(self.bullets) if self.bullets else self.body

    def to_skill_md(self) -> str:
        desc = self.description
        if self.provisional and "provisional" not in desc.lower():
            desc = desc.rstrip(".") + ". (Provisional: based on a single incident; verify before relying on it.)"
        return f"---\nname: {self.name}\ndescription: {yaml_str(desc)}\n---\n\n{self.body_md.strip()}\n"


@dataclass
class Eval:
    id: str
    skill_name: str
    task: str
    kind: str                 # "judge" | "host"
    assertions: list[str]
    source_trace: str
    status: str = "pending"   # pending | passed | failed
    result: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)


@dataclass
class Principle:
    id: str
    rule: str
    evidence_count: int
    lesson_ids: list[str] = field(default_factory=list)
    active: bool = True
    ts: float = field(default_factory=time.time)


def _skill_from_json(js: str) -> Skill:
    d = json.loads(js)
    known = {f for f in Skill.__dataclass_fields__}
    return Skill(**{k: v for k, v in d.items() if k in known})


def yaml_str(s: str) -> str:
    s = s.replace("\n", " ").strip()
    return json.dumps(s) if any(c in s for c in ':#"\'{}[]') else s


_SLUG_STOP = {"when", "a", "an", "the", "and", "or", "of", "to", "in", "with", "for", "you", "your", "is", "are",
              "be", "that", "this", "it", "its", "if", "then", "before", "after", "because", "should", "must",
              "always", "never", "make", "sure", "use", "using"}


def slugify(s: str) -> str:
    """Short, readable skill name. Rules often arrive as a whole sentence ("when writing a python one-liner
    with python -c avoid...") - keep the first few meaningful words, not the first 48 characters."""
    words = [w for w in re.sub(r"[^a-z0-9]+", " ", s.lower()).split() if w]
    kept = [w for w in words if w not in _SLUG_STOP] or words
    out = "-".join(kept[:4])
    return out[:40].strip("-") or "skill"


class Store:
    def __init__(self, home: str | Path | None = None):
        self.home = Path(home or os.getenv("SKILLLOOP_HOME") or Path.home() / ".skillloop")
        self.skills_dir = self.home / "skills"
        self.quarantine_dir = self.home / "quarantine"
        self.history_dir = self.home / "history"
        for d in (self.home, self.skills_dir, self.quarantine_dir, self.history_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.principles_dir = self.home / "principles"
        self.principles_dir.mkdir(exist_ok=True)
        # One SQLite connection per thread. A single shared connection with check_same_thread=False is not
        # safe under real concurrency (the HTTP server handles requests on multiple threads): its cursors are
        # shared mutable state, which surfaces as OperationalError / InterfaceError / SystemError under load.
        self._db_path = str(self.home / "skillloop.db")
        self._local = threading.local()
        # Bumped by this Store's own skill writes; see library_version.
        self._library_version = 0
        self._init()
        from .migrate import migrate
        migrate(self.db)

    @property
    def library_version(self) -> tuple[int, int]:
        """Cache key for anything derived from the whole skill library.

        Two components, because one is not enough:
          - a local counter, bumped by this Store's own writes;
          - PRAGMA data_version, which SQLite increments when ANOTHER connection commits. Without it a cache
            in the HTTP server would keep serving profiles from before a `skillloop` CLI write to the same DB.
        """
        try:
            dv = self.db.execute("PRAGMA data_version").fetchone()[0]
        except sqlite3.OperationalError:
            dv = 0
        return (self._library_version, int(dv))

    @property
    def db(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._db_path, timeout=30.0)
            conn.row_factory = sqlite3.Row
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA busy_timeout=30000")
                conn.execute("PRAGMA synchronous=NORMAL")
            except sqlite3.OperationalError:
                pass
            self._local.conn = conn
        return conn

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _init(self):
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS traces (id TEXT PRIMARY KEY, ts REAL, task TEXT, agent TEXT, outcome TEXT,
            confidence REAL, processed INTEGER DEFAULT 0, json TEXT);
        CREATE TABLE IF NOT EXISTS lessons (id TEXT PRIMARY KEY, trace_id TEXT, ts REAL, json TEXT);
        CREATE TABLE IF NOT EXISTS skills (name TEXT PRIMARY KEY, json TEXT);
        CREATE TABLE IF NOT EXISTS evals (id TEXT PRIMARY KEY, skill_name TEXT, status TEXT, json TEXT);
        CREATE TABLE IF NOT EXISTS events (ts REAL, kind TEXT, subject TEXT, detail TEXT);
        CREATE VIRTUAL TABLE IF NOT EXISTS skill_fts USING fts5(name, description, body, triggers);
        CREATE TABLE IF NOT EXISTS principles (id TEXT PRIMARY KEY, ts REAL, json TEXT);
        CREATE TABLE IF NOT EXISTS metrics (ts REAL, key TEXT, value REAL);
        """)
        self.db.commit()

    # ---------------- traces ----------------
    def add_trace(self, t: Trace) -> None:
        self.db.execute("INSERT OR REPLACE INTO traces VALUES (?,?,?,?,?,?,0,?)",
                        (t.id, t.ts, t.task, t.agent, t.outcome, t.confidence, t.to_json()))
        self.db.commit()

    def pending_traces(self, limit: int = 50) -> list[Trace]:
        rows = self.db.execute("SELECT json FROM traces WHERE processed=0 ORDER BY ts LIMIT ?", (limit,)).fetchall()
        return [Trace.from_json(r["json"]) for r in rows]

    def mark_processed(self, trace_id: str) -> None:
        self.db.execute("UPDATE traces SET processed=1 WHERE id=?", (trace_id,))
        self.db.commit()

    def get_trace(self, trace_id: str) -> Trace | None:
        r = self.db.execute("SELECT json FROM traces WHERE id=?", (trace_id,)).fetchone()
        return Trace.from_json(r["json"]) if r else None

    def recent_traces(self, n: int = 20) -> list[Trace]:
        rows = self.db.execute("SELECT json FROM traces ORDER BY ts DESC LIMIT ?", (n,)).fetchall()
        return [Trace.from_json(r["json"]) for r in rows]

    # ---------------- lessons ----------------
    def add_lesson(self, l: Lesson) -> None:
        self.db.execute("INSERT OR REPLACE INTO lessons VALUES (?,?,?,?)", (l.id, l.trace_id, l.ts, json.dumps(asdict(l))))
        self.db.commit()

    def lesson_for_trace(self, trace_id: str) -> "Lesson | None":
        for l in self.lessons(500):
            if l.trace_id == trace_id:
                return l
        return None

    def lessons(self, limit: int = 200) -> list[Lesson]:
        rows = self.db.execute("SELECT json FROM lessons ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
        known = set(Lesson.__dataclass_fields__)
        return [Lesson(**{k: v for k, v in json.loads(r["json"]).items() if k in known}) for r in rows]

    # ---------------- skills ----------------
    def get_skill(self, name: str) -> Skill | None:
        r = self.db.execute("SELECT json FROM skills WHERE name=?", (name,)).fetchone()
        return _skill_from_json(r["json"]) if r else None

    def all_skills(self, status: str | None = None) -> list[Skill]:
        rows = self.db.execute("SELECT json FROM skills").fetchall()
        out = [_skill_from_json(r["json"]) for r in rows]
        return [s for s in out if status is None or s.status == status]

    def save_skill(self, s: Skill, triggers: list[str] | None = None, snapshot: bool = True) -> None:
        s.updated = time.time()
        self._library_version += 1
        if triggers is None:
            # keep the existing trigger column
            r = self.db.execute("SELECT triggers FROM skill_fts WHERE name=?", (s.name,)).fetchone()
            trig = r["triggers"] if r else ""
        else:
            trig = " ".join(triggers)
        self.db.execute("BEGIN IMMEDIATE")
        try:
            self.db.execute("INSERT OR REPLACE INTO skills VALUES (?,?)", (s.name, json.dumps(asdict(s))))
            self.db.execute("DELETE FROM skill_fts WHERE name=?", (s.name,))
            self.db.execute("INSERT INTO skill_fts VALUES (?,?,?,?)", (s.name, s.description, s.body_md, trig))
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise
        self._write_files(s, snapshot=snapshot)

    def triggers_for(self, name: str) -> str:
        r = self.db.execute("SELECT triggers FROM skill_fts WHERE name=?", (name,)).fetchone()
        return r["triggers"] if r else ""

    def all_triggers(self) -> dict[str, str]:
        """Every skill's triggers in ONE query. Retrieval builds a profile per skill on every recall(), and
        doing that with a per-skill SELECT costs O(library) round-trips per query."""
        return {r["name"]: r["triggers"] for r in
                self.db.execute("SELECT name, triggers FROM skill_fts").fetchall()}

    def bump_recalls(self, names: list[str]) -> None:
        for n in names:
            s = self.get_skill(n)
            if s:
                s.recalls += 1
                self.db.execute("UPDATE skills SET json=? WHERE name=?", (json.dumps(asdict(s)), n))
        self.db.commit()

    def _write_files(self, s: Skill, snapshot: bool) -> None:
        live = self.skills_dir / s.name
        quar = self.quarantine_dir / s.name
        target = live if s.status in ("active", "candidate") else quar
        other = quar if target is live else live
        if other.exists():
            shutil.rmtree(other)
        target.mkdir(parents=True, exist_ok=True)
        (target / "SKILL.md").write_text(s.to_skill_md())
        meta = asdict(s); meta.pop("embedding", None)
        (target / "META.json").write_text(json.dumps(meta, indent=2))
        if snapshot:
            h = self.history_dir / s.name
            h.mkdir(exist_ok=True)
            (h / f"v{s.version}.md").write_text(s.to_skill_md())

    def skill_versions(self, name: str) -> list[int]:
        h = self.history_dir / name
        if not h.exists():
            return []
        return sorted(int(p.stem[1:]) for p in h.glob("v*.md"))

    def rollback(self, name: str, version: int) -> Skill:
        s = self.get_skill(name)
        if not s:
            raise KeyError(name)
        text = (self.history_dir / name / f"v{version}.md").read_text()
        fm, body = parse_skill_md(text)
        from .playbook import parse_body
        s.description = fm.get("description", s.description)
        # restore the itemised bullets of that version, preserving their ids so counters stay meaningful
        s.bullets, s.body = parse_body(body), ""
        s.version += 1
        s.status = "active"
        self.save_skill(s)
        self.log("rollback", name, f"to v{version} as v{s.version}")
        return s

    def search_skills(self, query: str, statuses: tuple[str, ...] = ("active",), limit: int = 5) -> list[tuple[Skill, float]]:
        q = fts_query(query)
        if not q:
            return []
        try:
            rows = self.db.execute(
                "SELECT name, bm25(skill_fts, 5.0, 3.0, 1.0, 4.0) AS score FROM skill_fts WHERE skill_fts MATCH ? ORDER BY score LIMIT ?",
                (q, limit * 3)).fetchall()
        except sqlite3.OperationalError:
            return []
        out = []
        for r in rows:
            s = self.get_skill(r["name"])
            if s and s.status in statuses:
                out.append((s, -float(r["score"])))
        return out[:limit]

    # ---------------- evals ----------------
    def save_eval(self, e: Eval) -> None:
        self.db.execute("INSERT OR REPLACE INTO evals VALUES (?,?,?,?)", (e.id, e.skill_name, e.status, json.dumps(asdict(e))))
        self.db.commit()

    def get_eval(self, eid: str) -> Eval | None:
        r = self.db.execute("SELECT json FROM evals WHERE id=?", (eid,)).fetchone()
        return Eval(**json.loads(r["json"])) if r else None

    def evals_for(self, skill_name: str) -> list[Eval]:
        rows = self.db.execute("SELECT json FROM evals WHERE skill_name=? ORDER BY json", (skill_name,)).fetchall()
        return [Eval(**json.loads(r["json"])) for r in rows]

    def pending_host_evals(self, limit: int = 10) -> list[Eval]:
        rows = self.db.execute("SELECT json FROM evals WHERE status='pending' LIMIT ?", (limit,)).fetchall()
        return [e for e in (Eval(**json.loads(r["json"])) for r in rows) if e.kind == "host"]

    # ---------------- audit ----------------
    def log(self, kind: str, subject: str, detail: str = "") -> None:
        self.db.execute("INSERT INTO events VALUES (?,?,?,?)", (time.time(), kind, subject, detail))
        self.db.commit()

    def events(self, n: int = 50) -> list[dict]:
        rows = self.db.execute("SELECT * FROM events ORDER BY ts DESC LIMIT ?", (n,)).fetchall()
        return [dict(r) for r in rows]

    # ---------------- principles ----------------
    def add_principle(self, p: "Principle") -> None:
        self.db.execute("INSERT OR REPLACE INTO principles VALUES (?,?,?)", (p.id, p.ts, json.dumps(asdict(p))))
        self.db.commit()
        self.write_principles_md()

    def principles(self) -> list["Principle"]:
        rows = self.db.execute("SELECT json FROM principles ORDER BY ts").fetchall()
        return [Principle(**json.loads(r["json"])) for r in rows]

    def write_principles_md(self) -> Path:
        ps = [p for p in self.principles() if p.active]
        lines = ["# Principles", "", "Cross-cutting rules distilled from many lessons. Apply on every task.", ""]
        for p in ps:
            lines.append(f"- **{p.rule}** _(evidence: {p.evidence_count} lessons)_")
        path = self.principles_dir / "PRINCIPLES.md"
        path.write_text("\n".join(lines) + "\n")
        return path

    def principles_text(self) -> str:
        ps = [p for p in self.principles() if p.active]
        return "\n".join(f"- {p.rule}" for p in ps)

    # ---------------- metrics ----------------
    def metric(self, key: str, value: float) -> None:
        self.db.execute("INSERT INTO metrics VALUES (?,?,?)", (time.time(), key, float(value)))
        self.db.commit()

    def metrics(self, key: str | None = None, n: int = 500) -> list[dict]:
        q = "SELECT ts, key, value FROM metrics" + (" WHERE key=?" if key else "") + " ORDER BY ts DESC LIMIT ?"
        rows = self.db.execute(q, ((key, n) if key else (n,))).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> dict[str, Any]:
        c = lambda q, *a: self.db.execute(q, a).fetchone()[0]
        return {
            "traces": c("SELECT COUNT(*) FROM traces"),
            "traces_unprocessed": c("SELECT COUNT(*) FROM traces WHERE processed=0"),
            "lessons": c("SELECT COUNT(*) FROM lessons"),
            "skills": {st: c("SELECT COUNT(*) FROM skills WHERE json LIKE ?", f'%"status": "{st}"%')
                       for st in ("active", "candidate", "quarantine", "retired")},
            "evals": {st: c("SELECT COUNT(*) FROM evals WHERE status=?", st) for st in ("pending", "passed", "failed")},
            "principles": c("SELECT COUNT(*) FROM principles"),
            "home": str(self.home),
        }


def fts_query(q: str) -> str:
    words = re.findall(r"[a-zA-Z0-9_]{3,}", q.lower())
    stop = {"the", "and", "for", "with", "this", "that", "from", "into", "how", "can", "you", "please", "using"}
    words = [w for w in words if w not in stop][:20]
    return " OR ".join(f'"{w}"' for w in words)


def parse_skill_md(text: str) -> tuple[dict[str, str], str]:
    m = re.match(r"^---\n(.*?)\n---\n?(.*)$", text, flags=re.S)
    if not m:
        return {}, text
    fm: dict[str, str] = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            v = v.strip()
            if v.startswith('"') and v.endswith('"'):
                v = json.loads(v)
            fm[k.strip()] = v
    return fm, m.group(2).strip()
