"""SkillLoop core. Two calls for the agent, the rest is ours.

    loop = SkillLoop()
    skills = loop.recall("migrate the postgres schema")    # before the task, fast, no LLM
    loop.learn(trace)                                       # after the task, queued
    loop.process()                                          # background: reflect -> synth -> gate -> curate
"""
from __future__ import annotations

import math
import os
import threading
import time
from dataclasses import asdict
from typing import Any

from . import gate
from . import observability as obs
from .learner import distill_principles, find_related, judge_outcome, reflect, synthesize
from .playbook import parse_body, record_feedback, refine
from .llm import LLM
from .safety import sanitize_trace
from .schema import Trace, normalize
from .store import Skill, Store


class SkillLoop:
    def __init__(self, home: str | None = None, llm: LLM | None = None, policy: gate.Policy | None = None,
                 min_tool_calls_to_learn: int = 2, auto_process: bool = False):
        obs.configure_logging()
        self.store = Store(home)
        self._llm = llm
        self.policy = policy or gate.Policy.from_env()
        self.min_tool_calls = min_tool_calls_to_learn
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._lessons_since_distill = 0
        self.recall_threshold = float(os.getenv("SKILLLOOP_RECALL_THRESHOLD", "0.08"))
        self.recall_margin = float(os.getenv("SKILLLOOP_RECALL_MARGIN", "0.55"))
        self.embed_threshold = float(os.getenv("SKILLLOOP_EMBED_THRESHOLD", "0.35"))
        self.query_threshold = float(os.getenv("SKILLLOOP_QUERY_THRESHOLD", "0.3"))
        self.duplicate_threshold = float(os.getenv("SKILLLOOP_DUPLICATE_THRESHOLD", "0.75"))
        if auto_process:
            self.start_worker()

    @property
    def llm(self) -> LLM:
        if self._llm is None:
            self._llm = LLM()
        return self._llm

    # ------------------------------------------------------------------ hot path
    def recall(self, task: str, limit: int = 3, include_candidates: bool = True,
               include_principles: bool = True) -> list[dict[str, Any]]:
        """Fast, local, no LLM. Returns skills the agent should read before this task, each with provenance.
        If principles exist, the first entry is the PRINCIPLES block (name="_principles")."""
        statuses = ("active", "candidate") if include_candidates else ("active",)
        hits = self._search(task, statuses, limit)
        out = []
        if include_principles:
            ptxt = self.store.principles_text()
            if ptxt:
                out.append({"name": "_principles", "status": "principles", "description":
                            "Cross-cutting rules distilled from many past lessons. Apply on every task.",
                            "skill_md": ptxt, "path": str(self.store.principles_dir / "PRINCIPLES.md")})
        for s, score in hits:
            out.append({
                "name": s.name, "status": s.status, "version": s.version, "score": round(score, 2),
                "hit_rate": round(s.hit_rate, 2), "uses": s.uses, "provisional": s.provisional,
                "provenance": {"traces": len(s.provenance), "evidence": s.evidence_count,
                               "recalls": s.recalls, "trigger_precision": round(s.trigger_precision, 2),
                               "security_cleared": bool(s.security.get("cleared")),
                               "approved": s.approved},
                "description": s.description, "skill_md": s.to_skill_md(),
                "bullet_ids": [b["id"] for b in s.bullets],
                "path": str(self.store.skills_dir / s.name / "SKILL.md"),
            })
        self.store.bump_recalls([s.name for s, _ in hits])
        obs.log("recall", task=task[:80], hits=[h["name"] for h in out if h.get("name") != "_principles"])
        return out

    # ------------------------------------------------------------------ retrieval
    _STOP = {
        "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "from", "into", "that", "this",
        "then", "it", "its", "is", "are", "be", "you", "your", "use", "using", "used", "make", "made", "get",
        "put", "set", "run", "runs", "running", "create", "creates", "created", "creating", "write", "writes",
        "writing", "written", "read", "reads", "reading", "file", "files", "output", "input", "value", "values",
        "data", "line", "lines", "content", "contents", "result", "results", "task", "agent", "step", "steps",
        "command", "commands", "check", "checks", "verify", "verifies", "ensure", "before", "after", "when",
        "should", "must", "not", "no", "if", "any", "all", "new", "one", "two", "number", "numbers", "name",
        "names", "text", "code", "directory", "folder", "path", "need", "needs", "want", "wants", "please",
        "user", "here", "each", "every", "other", "others", "leaving", "alone", "keep", "kept", "also", "does",
        "but", "was", "were", "never", "always", "only", "just", "still", "than", "both", "some", "can", "will",
        "has", "have", "had", "did", "do", "done", "how", "why", "what", "which", "who", "there", "their",
        "them", "they", "our", "out", "via", "per", "may", "might", "would", "could", "such", "same", "more",
        "most", "less", "least", "very", "much", "many", "few", "now", "yet", "so", "up", "down", "over",
        "under", "again", "once", "first", "last", "next", "prev", "previous", "current", "actually", "really",
        "script", "scripts", "program", "programs", "python", "bash", "shell", "sh",
        # Function words and generic instruction verbs. These matter most for FACET text: applies_when
        # phrases are tokenised word-by-word and every token becomes an ANCHOR at weight 3.0, so a
        # multi-word phrase donates its prepositions as anchors. Anchors are what prevent false fires, so a
        # function word satisfying the anchor rule defeats the mechanism. Measured (qa/run_r1.py): the
        # phrase "script lies about success" made "about" an anchor and "write a haiku about the sea"
        # retrieved an exit-code skill; "explain analyze" made "explain" an anchor and "explain the plot of
        # Hamlet" retrieved a Postgres skill.
        "about", "between", "against", "during", "without", "within", "across", "among", "upon", "toward",
        "explain", "describe", "difference", "differences", "instead", "rather", "whether", "regardless",
    }

    @classmethod
    def _terms(cls, text: str) -> set[str]:
        """Distinctive content words, with identifiers split into subtokens.

        `test_tax.py` -> {test_tax.py, test, tax, py}. Without this, a query naming a concrete file never
        matches a skill that talks about the concept ("tax calculation"), which is most real queries.
        """
        import re as _re
        out: set[str] = set()
        for raw in _re.findall(r"[A-Za-z0-9_.\-]{2,}", (text or "").lower()):
            parts = [raw] + [p for p in _re.split(r"[_.\-]+", raw) if p]
            for p in parts:
                if len(p) >= 3 and p not in cls._STOP and not p.isdigit():
                    out.add(p)
        return out

    def _profiles(self, statuses: tuple[str, ...]) -> list[tuple[Skill, dict[str, float], set[str]]]:
        """What each skill is FOR - name, description, triggers, facets. Deliberately NOT the body: matching
        on body text is what made a JSON skill fire on a CSV task (both mention 'file' and 'python').

        Cached against Store.library_version: profiles depend only on the library, not the query, so an
        unchanged library is not re-read and re-tokenised on every call. Rebuilding this per query was 81% of
        recall() time at a 500-skill library (one SELECT per skill, plus tokenisation of every facet).
        """
        key = (tuple(sorted(statuses)), self.store.library_version)
        cached = getattr(self, "_profiles_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        trig = self.store.all_triggers()          # one query, not one per skill
        out = []
        for sk in self.store.all_skills():
            if sk.status not in statuses:
                continue
            weighted: dict[str, float] = {}
            anchors: set[str] = set()
            fac = sk.facets or {}
            fields = [
                (sk.name.replace("-", " "), 3.0),
                (" ".join(fac.get("applies_when", [])), 3.0),
                (trig.get(sk.name, ""), 2.0),
                (" ".join(fac.get("symptoms", [])), 2.0),
                (sk.description, 1.0),
            ]
            for text, w in fields:
                for t in self._terms(text):
                    weighted[t] = max(weighted.get(t, 0.0), w)
                    if w >= 3.0:
                        anchors.add(t)
            out.append((sk, weighted, anchors))
        self._profiles_cache = (key, out)
        self._idf_cache = None
        return out

    # Function words only. The facet stoplist strips domain vocabulary (script, run, output) because in a
    # skill DESCRIPTION those words match everything. In a short user phrasing they carry the meaning:
    # "make my script runnable" collapses to one token under the facet stoplist and stops matching anything.
    _QUERY_STOP = {"the", "a", "an", "to", "in", "of", "my", "me", "i", "do", "does", "is", "it", "its", "how",
                   "what", "why", "when", "and", "or", "for", "this", "that", "with", "can", "should", "would",
                   "please", "need", "want", "on", "at", "be", "am", "are", "was", "were", "have", "has", "get",
                   "there", "their", "from", "into", "some", "any", "all", "so", "if", "but", "not", "no"}

    @classmethod
    def _query_terms(cls, text: str) -> set[str]:
        import re as _re
        out: set[str] = set()
        for raw in _re.findall(r"[A-Za-z0-9_.\-]{2,}", (text or "").lower()):
            for p in [raw] + [x for x in _re.split(r"[_.\-]+", raw) if x]:
                if len(p) >= 2 and p not in cls._QUERY_STOP and not p.isdigit():
                    out.add(p)
        return out

    def _best_query_match(self, qt: set[str], sk: Skill) -> float:
        """doc2query matching, done at PHRASE level.

        Pooling every trigger query into one bag makes a skill with 10 queries match almost anything, because
        a single shared word is enough. Instead each trigger query is compared on its own and we take the best:
        the score is the fraction of THAT query's terms present in the task. "how many times does X appear"
        only scores when most of it is there, so it stops firing on "make deploy.sh runnable".
        """
        best = 0.0
        for q in sk.queries:
            terms = self._query_terms(q)
            if len(terms) < 2:
                continue
            shared = qt & terms
            # one word in common is a coincidence, not a match ("project timeline" vs "add a dependency to the
            # project"). Demand two shared terms unless the overlap is overwhelming.
            if len(shared) < 2 and len(shared) / len(terms) < 0.6:
                continue
            best = max(best, len(shared) / len(terms))
        return best

    def _idf(self, profiles) -> dict[str, float]:
        """IDF over the profile corpus. Cached alongside _profiles: it is a pure function of the library."""
        cached = getattr(self, "_idf_cache", None)
        if cached is not None and cached[0] is profiles:
            return cached[1]
        import math
        n = len(profiles) or 1
        df: dict[str, int] = {}
        for _, prof, _a in profiles:
            for t in prof:
                df[t] = df.get(t, 0) + 1
        out = {t: math.log(1 + n / c) for t, c in df.items()}
        self._idf_cache = (profiles, out)
        return out

    def _search(self, task: str, statuses: tuple[str, ...], limit: int) -> list[tuple[Skill, float]]:
        """IDF-weighted coverage over each skill's purpose, with abstention.

        Score = (weighted IDF mass of query terms this skill covers) / (total IDF mass of the query).
        A skill is only returned if it clears an absolute threshold AND matches at least one term that is
        rare across the library - otherwise every skill wins on shared generic vocabulary.
        """
        qt = self._terms(task)
        if not qt:
            return []
        profiles = self._profiles(statuses)
        if not profiles:
            return []
        idf = self._idf(profiles)
        default_idf = max(idf.values(), default=1.0)
        q_mass = sum(idf.get(t, default_idf) for t in qt) or 1.0
        scored: list[tuple[Skill, float, int]] = []
        for sk, prof, anchors in profiles:
            hits = qt & set(prof)
            qsim = self._best_query_match(self._query_terms(task), sk)
            if not hits and qsim < self.query_threshold:
                continue
            # ANCHOR RULE: the query must touch what the skill IS (its name or applies_when facets), not just
            # words that happen to appear in its description. This is what stops "change ... settings" pulling
            # a JSON skill into an INI task.
            # a strong phrase-level query match is itself an anchor: the user said what the skill is for,
            # in their own words. Otherwise the query must touch the skill's name/applies_when.
            if qsim < self.query_threshold and anchors and not (qt & anchors):
                self.store.log("recall_filtered", sk.name, f"no anchor overlap :: {task[:50]}")
                continue
            # a not_for term that is also one of the skill's own anchors is self-contradictory (e.g. a pip
            # skill listing "global install success" excludes the word "install"); ignore those.
            neg = self._terms(" ".join((sk.facets or {}).get("not_for", []))) - anchors
            if neg & qt:
                self.store.log("recall_filtered", sk.name, f"excluded by not_for {sorted(neg & qt)[:3]}")
                continue
            mass = sum(idf.get(t, default_idf) * prof[t] for t in hits)
            # breadth normalization: a skill advertising itself for twenty situations should not beat a focused
            # one on a single shared word. Same idea as document-length normalization in BM25 - kept gentle so
            # it only breaks ties, never outweighs actually matching more of the query.
            breadth = max(1.0, len(anchors) / 4.0) ** 0.25
            facet_score = mass / (q_mass * 3.0 * breadth) if hits else 0.0
            scored.append((sk, max(facet_score, qsim), len(hits) + (2 if qsim >= self.query_threshold else 0)))
        # rank by how many distinct query terms the skill covers; breadth-adjusted mass separates equals
        scored.sort(key=lambda x: (-x[2], -x[1]))
        scored = [(sk, sc) for sk, sc, _n in scored]
        keep = [(sk, sc) for sk, sc in scored if sc >= self.recall_threshold]
        if keep and len(keep) > 1:
            # drop anything far weaker than the best match; a trailing weak hit is noise, not a second opinion
            best = keep[0][1]
            keep = [(sk, sc) for sk, sc in keep if sc >= best * self.recall_margin]
        return self._blend_embeddings(task, keep, statuses)[:limit]

    def _blend_embeddings(self, task, keep, statuses):
        emb_model = getattr(self._llm, "embed_model", None) if self._llm is not None else None
        if not emb_model:
            return keep
        try:
            qv = self.llm.embed([task])
        except Exception:
            qv = None
        if not qv:
            return keep
        q = qv[0]
        merged: dict[str, tuple[Skill, float]] = {sk.name: (sk, sc) for sk, sc in keep}
        for sk in self.store.all_skills():
            if sk.status not in statuses or not sk.embedding:
                continue
            cos = _cosine(q, sk.embedding)
            if cos < self.embed_threshold:
                continue
            base = merged.get(sk.name, (sk, 0.0))[1]
            merged[sk.name] = (sk, base + 0.5 * cos)
        return sorted(merged.values(), key=lambda x: -x[1])

    def session(self, task: str, agent: str = "", metadata: dict | None = None, auto_learn: bool = True):
        """Record a task and submit it on exit. See record.py - this is the easy path to learn()."""
        from .record import Session
        return Session(self, task, agent=agent, metadata=metadata, auto_learn=auto_learn)

    def pending_evals(self, limit: int = 5) -> list[dict[str, Any]]:
        """Host-run evals waiting for an agent to execute. Run the task, then learn(trace, eval_id=...)."""
        return [asdict(e) for e in self.store.pending_host_evals(limit)]

    def learn(self, trace: Trace | dict[str, Any]) -> dict[str, Any]:
        """Queue a trajectory. Returns immediately. Secrets are redacted and sizes capped before storage."""
        t = trace if isinstance(trace, Trace) else normalize(trace)
        with obs.request("learn", task=t.task[:80] if hasattr(t, "task") else None):
            t, san = sanitize_trace(t)
            self.store.add_trace(t)
            obs.log("trace.sanitized", trace_id=t.id, redacted=san.get("redacted"),
                    steps_dropped=san.get("steps_dropped"), final_chars=san.get("final_chars"))
            return self._learn_body(t, san)

    def _learn_body(self, t, san):
        # per-bullet credit assignment: the agent tells us which bullets actually helped
        if t.bullets_helpful or t.bullets_harmful:
            for name in t.skills_used:
                sk = self.store.get_skill(name)
                if not sk:
                    continue
                if not sk.bullets:
                    sk.bullets = parse_body(sk.body)
                record_feedback(sk.bullets, t.bullets_helpful, t.bullets_harmful)
                sk.bullets, _ = refine(sk.bullets)
                self.store.save_skill(sk, snapshot=False)
        # usage stats + promotion are cheap and synchronous
        notes = gate.apply_use(self.store, t, self.policy)
        if t.eval_id:
            ev = self.store.get_eval(t.eval_id)
            if ev:
                passed = gate.record_host_eval(self.store, ev, t, llm=self._llm)
                s = self.store.get_skill(ev.skill_name)
                if s and passed and s.status != "active":
                    ok, why = gate.can_activate(s, self.policy)
                    if ok:
                        s.status = "active"
                        self.store.log("promote", s.name, f"host eval {ev.id} passed")
                        notes.append(f"{s.name}: -> active (host eval passed)")
                    else:
                        s.status = "candidate"
                        notes.append(f"{s.name}: host eval passed but stays candidate ({why})")
                    self.store.save_skill(s, snapshot=False)
                elif s and not passed:
                    s.status = "quarantine"
                    self.store.save_skill(s, snapshot=False)
                    self.store.log("demote", s.name, f"host eval {ev.id} failed")
                    notes.append(f"{s.name}: -> quarantine (host eval failed)")
        self.store.log("learn", t.id, f"{t.outcome} ({t.confidence:.2f}) {t.task[:80]}")
        self.store.metric("episode_success", 1.0 if t.outcome == "success" else 0.0)
        obs.log("learn.queued", trace_id=t.id, outcome=t.outcome, request_id=obs.current_request_id())
        return {"trace_id": t.id, "outcome": t.outcome, "confidence": t.confidence, "queued": True,
                "notes": notes, "sanitized": san, "request_id": obs.current_request_id()}

    # ------------------------------------------------------------------ background
    def process(self, max_traces: int = 20) -> list[dict[str, Any]]:
        """Run the learning loop over queued traces. Safe to call repeatedly."""
        with self._lock:
            report = []
            from .llm import PendingManualCall
            for t in self.store.pending_traces(max_traces):
                try:
                    report.append(self._process_one(t))
                except PendingManualCall as p:
                    # leave the trace queued; re-running process() resumes once the answer is filled in
                    report.append({"trace_id": t.id, "awaiting_manual": p.key, "role": p.role, "path": p.path})
                    continue
                except Exception as e:  # never let one bad trace stall the queue
                    self.store.log("error", t.id, repr(e))
                    report.append({"trace_id": t.id, "error": repr(e)})
                self.store.mark_processed(t.id)
            if self._lessons_since_distill >= self.policy.principle_every_n_lessons:
                try:
                    new = distill_principles(self.llm, self.store)
                    if new:
                        report.append({"principles": [p.rule for p in new]})
                except Exception as e:
                    self.store.log("error", "principles", repr(e))
                self._lessons_since_distill = 0
            notes = gate.curate(self.store, self.policy)
            if notes:
                report.append({"curate": notes})
            self._record_library_metrics()
            return report

    def _library_size(self) -> int:
        return sum(1 for s in self.store.all_skills() if s.status in ("active", "candidate"))

    def _process_one(self, t: Trace) -> dict[str, Any]:
        policy = self.policy.effective(self._library_size())
        r: dict[str, Any] = {"trace_id": t.id, "task": t.task[:80], "outcome": t.outcome,
                             "cold_start": getattr(policy, "cold", False)}
        if t.eval_id:
            r["skipped"] = "eval run (handled in learn)"
            return r
        if t.n_tool_calls < self.min_tool_calls and t.outcome != "failure":
            r["skipped"] = "trivial success"
            return r
        if t.outcome == "unknown" or t.confidence < 0.5:
            # the host didn't tell us how it went; ask a judge before deciding anything
            sig = judge_outcome(self.llm, t)
            t.signals.append(sig)
            self.store.add_trace(t)
            self.store.db.execute("UPDATE traces SET processed=1 WHERE id=?", (t.id,))
            r["judged_outcome"] = {"outcome": sig.outcome, "confidence": sig.confidence, "detail": sig.detail}
            r["outcome"] = t.outcome
            if t.outcome == "unknown":
                r["skipped"] = "judge could not determine outcome"
                return r
            gate.apply_use(self.store, t, self.policy)

        # idempotent: if this trace was already reflected on (e.g. a previous run paused at a later stage),
        # reuse the lesson rather than paying for another reflection and duplicating it in the store
        lesson = self.store.lesson_for_trace(t.id)
        if lesson is None:
            lesson = reflect(self.llm, t, policy=policy)
            self.store.add_lesson(lesson)
        r["lesson"] = {"rule": lesson.rule, "root_cause": lesson.root_cause, "blame": lesson.blame,
                       "generalizable": lesson.generalizable, "confidence": lesson.confidence,
                       "verification": lesson.verification, "principle": lesson.principle}
        if not lesson.generalizable or lesson.confidence < policy.min_lesson_confidence:
            r["skipped"] = "lesson not generalizable / low confidence"
            return r
        self._lessons_since_distill += 1

        # decide create vs patch
        existing: Skill | None = None
        if lesson.blame in ("skill_wrong", "skill_incomplete", "skill_misapplied") and lesson.skill_ref:
            existing = self.store.get_skill(lesson.skill_ref)
        if existing is None:
            existing = self._find_duplicate(lesson)

        # idempotence: a lesson that already contributed to a skill must not be applied twice - BUT only skip
        # when that skill actually cleared the gate. A skill stranded in quarantine (a crash, a rate limit, or
        # a PendingManualCall between save and gate) must be RESUMED into the gate, not skipped forever, or the
        # learned skill is lost with no error and can never be recalled (quarantine is not a live status).
        if existing and lesson.id in existing.lesson_ids and existing.status in ("candidate", "active"):
            r["skipped"] = f"lesson {lesson.id} already contributed to {existing.name} (v{existing.version})"
            return r

        # RESUME: this lesson already built a skill, but it stalled in quarantine before the gate (crash / rate
        # limit / manual-provider pause). Re-run the gate on the existing skill instead of rebuilding it.
        if existing and lesson.id in existing.lesson_ids and existing.status == "quarantine":
            r["resumed"] = f"{existing.name} v{existing.version} was stranded in quarantine; re-running the gate"
            return self._gate(existing, lesson, t, r)

        related = find_related(self.store, lesson)
        if existing:
            related = [l for l in related if l.id not in existing.lesson_ids]

        # evidence threshold: don't write a whole skill from one ambiguous trace
        if existing is None and not related and lesson.confidence < policy.single_trace_confidence:
            r["skipped"] = (f"single lesson, confidence {lesson.confidence:.2f} < "
                            f"{policy.single_trace_confidence}; waiting for corroborating lessons")
            return r

        skill, change = synthesize(self.llm, self.store, lesson, related, existing)
        r["_change"] = change
        return self._gate(skill, lesson, t, r)

    def _gate(self, skill, lesson, t, r):
        """Shape -> security -> judge. Saving the skill (which claims the lesson id) happens INSIDE here,
        right before each gate step, and the caller resumes here if a step was interrupted. A skill only
        leaves quarantine when the judge actually passes."""
        change = r.pop("_change", "resumed")
        shape_problems = gate.shape_ok(skill)
        if shape_problems:
            r["rejected"] = f"skill shape incomplete: {shape_problems}"
            self.store.save_skill(skill, triggers=lesson.triggers)
            self.store.log("reject", skill.name, r["rejected"])
            return r
        skill.security = gate.security_check(self.llm, skill)   # cheap path when no risk markers
        if emb := self._maybe_embed(skill):
            skill.embedding = emb
        # NOTE: this save claims lesson.id in lesson_ids. If make_and_judge below is interrupted (crash / rate
        # limit / PendingManualCall), the skill stays in quarantine and the RESUME path re-enters _gate here.
        self.store.save_skill(skill, triggers=lesson.triggers)
        self.store.log("synthesize", skill.name, f"v{skill.version}: {change}")
        r["skill"] = {"name": skill.name, "version": skill.version, "change": change,
                      "provisional": skill.provisional, "evidence": skill.evidence_count,
                      "security_cleared": skill.security.get("cleared")}
        if not skill.security.get("cleared"):
            self.store.log("reject", skill.name, f"security: {skill.security}")
            r["rejected"] = "security check failed; skill stays in quarantine"
            r["security"] = skill.security
            return r
        # gate: assertions + verdict in a single model call (may raise PendingManualCall -> resumed later)
        ev, passed = gate.make_and_judge(self.llm, self.store, skill, t)
        r["judge_eval"] = {"id": ev.id, "passed": passed, "assertions": ev.assertions,
                           "reason": ev.result.get("reason", "")}
        if passed:
            skill.status = "candidate"
            self.store.save_skill(skill, triggers=lesson.triggers, snapshot=False)
            self.store.log("promote", skill.name, "candidate (judge eval passed)")
            hev = gate.queue_host_eval(self.store, skill, t, ev.assertions)
            r["host_eval_queued"] = hev.id
        else:
            self.store.log("reject", skill.name, f"judge eval failed: {ev.result.get('reason', '')}")
        return r

    def _find_duplicate(self, lesson) -> Skill | None:
        """Find an existing skill covering the SAME situation, so a second lesson patches it instead of
        creating a near-duplicate under a different name.

        Relearning the same traces used to yield differently-named skills each time (`yaml-edit` one run,
        `yaml-key-edit` the next), which silently fragmented the library and broke retrieval. Matching on the
        lesson's own words against every skill's queries and facets is stable regardless of naming.
        """
        probe = " ".join(lesson.triggers) + " " + (lesson.rule or "")
        qt = self._query_terms(probe)
        ft = self._terms(probe)
        best, best_score = None, 0.0
        for sk in self.store.all_skills():
            anchors = self._terms(f"{sk.name.replace('-', ' ')} {' '.join(sk.facets.get('applies_when', []))}")
            # A skill only counts as the SAME situation if the lesson touches what that skill is about.
            # Without this, one broad skill ("write output to a file") swallows every later lesson: in a real
            # run a single skill absorbed 20 unrelated lessons and the whole library collapsed into it.
            if not anchors or not (ft & anchors):
                continue
            score = max(self._best_query_match(qt, sk), len(ft & anchors) / len(anchors))
            if score > best_score:
                best, best_score = sk, score
        return best if best_score >= self.duplicate_threshold else None

    def _maybe_embed(self, skill: Skill) -> list[float] | None:
        try:
            if self._llm is not None and getattr(self._llm, "embed_model", None):
                v = self.llm.embed([f"{skill.name}\n{skill.description}\n{skill.body[:2000]}"])
                return v[0] if v else None
        except Exception:
            pass
        return None

    def _record_library_metrics(self) -> None:
        st = self.store.stats()
        self.store.metric("library_active", st["skills"]["active"])
        self.store.metric("library_candidate", st["skills"]["candidate"])
        self.store.metric("library_quarantine", st["skills"]["quarantine"])

    def start_worker(self, interval_s: float = 15.0) -> None:
        if self._worker:
            return

        def loop():
            while True:
                try:
                    self.process()
                except Exception as e:
                    self.store.log("error", "worker", repr(e))
                time.sleep(interval_s)

        self._worker = threading.Thread(target=loop, daemon=True)
        self._worker.start()

    # ------------------------------------------------------------------ developer
    def inspect(self, name: str | None = None) -> dict[str, Any]:
        if name:
            s = self.store.get_skill(name)
            if not s:
                return {"error": f"no skill {name}"}
            d = asdict(s); d.pop("embedding", None)
            return {"skill": d, "versions": self.store.skill_versions(name),
                    "evals": [asdict(e) for e in self.store.evals_for(name)],
                    "hit_rate": s.hit_rate, "trigger_precision": s.trigger_precision}
        return {"stats": self.store.stats(),
                "skills": [{"name": s.name, "status": s.status, "v": s.version, "uses": s.uses,
                            "recalls": s.recalls, "hit_rate": round(s.hit_rate, 2),
                            "precision": round(s.trigger_precision, 2), "provisional": s.provisional,
                            "cleared": bool(s.security.get("cleared")), "approved": s.approved}
                           for s in self.store.all_skills()],
                "principles": [p.rule for p in self.store.principles() if p.active],
                "recent_events": self.store.events(20)}

    def metrics(self) -> dict[str, Any]:
        """Success rate over episodes (rolling), library size over time, regression rate."""
        ep = [m["value"] for m in self.store.metrics("episode_success", 200)]
        ep.reverse()
        def rate(xs): return round(sum(xs) / len(xs), 3) if xs else None
        demotes = sum(1 for e in self.store.events(500) if e["kind"] == "demote")
        promotes = sum(1 for e in self.store.events(500) if e["kind"] == "promote")
        return {
            "episodes": len(ep),
            "success_rate_all": rate(ep),
            "success_rate_first_half": rate(ep[: len(ep) // 2]) if len(ep) >= 4 else None,
            "success_rate_second_half": rate(ep[len(ep) // 2:]) if len(ep) >= 4 else None,
            "success_rate_last_20": rate(ep[-20:]),
            "library": self.store.stats()["skills"],
            "regression_rate": round(demotes / promotes, 3) if promotes else None,
            "promotions": promotes, "demotions": demotes,
        }

    def rollback(self, name: str, version: int) -> dict[str, Any]:
        d = asdict(self.store.rollback(name, version)); d.pop("embedding", None); return d

    def set_status(self, name: str, status: str) -> dict[str, Any]:
        s = self.store.get_skill(name)
        if not s:
            return {"error": f"no skill {name}"}
        s.status = status
        self.store.save_skill(s, snapshot=False)
        self.store.log("manual", name, f"status -> {status}")
        d = asdict(s); d.pop("embedding", None); return d

    def approve(self, name: str) -> dict[str, Any]:
        """Human approval. Also clears security if a human has read it. Promotes candidate -> active."""
        s = self.store.get_skill(name)
        if not s:
            return {"error": f"no skill {name}"}
        s.approved = True
        s.security = {**s.security, "cleared": True, "human_cleared": True}
        if s.status == "candidate" and s.successes >= self.policy.min_host_successes_for_active:
            s.status = "active"
        self.store.save_skill(s, snapshot=False)
        self.store.log("approve", name, f"human approved; status={s.status}")
        d = asdict(s); d.pop("embedding", None); return d

    def export_dir(self) -> str:
        """Folder any agentskills.io-compatible agent can point at."""
        return str(self.store.skills_dir)


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0