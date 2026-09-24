"""SkillLoop core. Two calls for the agent, the rest is ours.

    loop = SkillLoop()
    skills = loop.recall("migrate the postgres schema")    # before the task, fast, no LLM
    loop.learn(trace)                                       # after the task, queued
    loop.process()                                          # background: reflect -> synth -> gate -> curate
"""
from __future__ import annotations

import math
import re
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
from .retrieval import RetrievalMixin
from .store import Skill, Store


class SkillLoop(RetrievalMixin):
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
        self.embed_threshold = float(os.getenv("SKILLLOOP_EMBED_THRESHOLD", "0.45"))
        self.embed_topk = int(os.getenv("SKILLLOOP_EMBED_TOPK", "10"))
        self.hybrid_lexical_w = float(os.getenv("SKILLLOOP_HYBRID_LEXICAL_W", "0.5"))
        self.hybrid_embed_w = float(os.getenv("SKILLLOOP_HYBRID_EMBED_W", "0.5"))
        self.query_threshold = float(os.getenv("SKILLLOOP_QUERY_THRESHOLD", "0.3"))
        self.duplicate_threshold = float(os.getenv("SKILLLOOP_DUPLICATE_THRESHOLD", "0.75"))
        # cosine at which an embedding match fires without lexical anchor overlap (paraphrase rescue)
        self.embed_strong = float(os.getenv("SKILLLOOP_EMBED_STRONG", "0.62"))
        self._warned_unprocessed = False
        if auto_process or os.getenv("SKILLLOOP_AUTO_PROCESS", "").lower() in ("1", "true", "yes"):
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
        out = self._format_hits(hits, include_principles)
        obs.log("recall", task=task[:80], hits=[h["name"] for h in out if h.get("name") != "_principles"])
        return out

    def recall_for_error(self, error: str, task: str = "", limit: int = 3, include_candidates: bool = True,
                         exclude: list[str] | tuple[str, ...] = ()) -> list[dict[str, Any]]:
        """Skills relevant to an error the agent is looking at RIGHT NOW. Fast, local, no LLM.

        Mistakes rarely announce themselves in the task description; they show up mid-task as an error.
        This matches the error's signature against each skill's recorded symptoms (the literal error text of
        the failure the skill was learned from), plus the task for context. Principles are not repeated here;
        they were shown at the start. `exclude` drops skills the agent already has in its prompt.
        """
        sig = self._error_signature(error)
        if not sig:
            return []
        statuses = ("active", "candidate") if include_candidates else ("active",)
        hits = self._search((sig + "\n" + (task or "")).strip(), statuses, limit + len(exclude), mode="error")
        hits = [(s, sc) for s, sc in hits if s.name not in set(exclude)][:limit]
        out = self._format_hits(hits, include_principles=False)
        obs.log("recall_error", error=sig[:80], hits=[h["name"] for h in out])
        return out

    def _format_hits(self, hits, include_principles: bool) -> list[dict[str, Any]]:
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
        return out

    def session(self, task: str, agent: str = "", metadata: dict | None = None, auto_learn: bool = True,
                error_hints: bool = True, on_hint=None):
        """Record a task and submit it on exit. See record.py - this is the easy path to learn().
        error_hints: a failed tool call surfaces matching skills for the next step (Session.hints_text())."""
        from .record import Session
        return Session(self, task, agent=agent, metadata=metadata, auto_learn=auto_learn,
                       error_hints=error_hints, on_hint=on_hint)

    def run_pending_evals(self, runner, limit: int = 5) -> list[dict[str, Any]]:
        """Close the verification loop automatically.

        A new skill stays a *candidate* until the agent actually re-runs the task that produced it, with the
        skill in context, and succeeds. Without this, that re-run depends on someone doing it by hand, so skills
        rarely reach *active*. `runner(task, skill_md)` must run your agent on `task` with `skill_md` in its
        prompt and return the resulting Trace (or dict, or a Session built with auto_learn=False). Its outcome
        should come from an independent check (Session.verified_by), not the agent's own opinion.
        """
        from .record import Session
        out = []
        for ev in self.store.pending_host_evals(limit):
            sk = self.store.get_skill(ev.skill_name)
            if not sk:
                continue
            try:
                res = runner(ev.task, sk.to_skill_md())
                if isinstance(res, Session):
                    res = res.trace or res.build()
                t = res if isinstance(res, Trace) else normalize(res)
            except Exception as e:           # a crashing run is a failed eval, and must be recorded as one
                t = normalize({"task": ev.task, "outcome": "failure",
                               "steps": [{"role": "tool", "content": f"runner raised {e!r}"}]})
            t.eval_id = ev.id
            if ev.skill_name not in t.skills_used:
                t.skills_used.append(ev.skill_name)
            r = self.learn(t)
            r["eval_id"], r["skill"] = ev.id, ev.skill_name
            out.append(r)
        return out

    def pending_evals(self, limit: int = 5) -> list[dict[str, Any]]:
        """Host-run evals waiting for an agent to execute. Run the task, then learn(trace, eval_id=...)."""
        return [asdict(e) for e in self.store.pending_host_evals(limit)]

    def learn(self, trace: Trace | dict[str, Any]) -> dict[str, Any]:
        """Queue a trajectory. Returns immediately. Secrets are redacted and sizes capped before storage."""
        t = trace if isinstance(trace, Trace) else normalize(trace)
        with obs.request("learn", task=t.task[:80] if hasattr(t, "task") else None):
            t, san = sanitize_trace(t)
            self.store.add_trace(t)
            if t.eval_id:
                # A verification re-run is EVIDENCE ABOUT a skill, not a new lesson to learn from. Leaving it
                # in the queue made the loop feed itself: the re-run became training data, produced another
                # skill, queued another host eval, and that second eval ran without the skill in its prompt
                # and failed - so every skill ended with equal passed/failed evals and was demoted to
                # quarantine no matter how well it actually performed (see CALIBRATION.md, 2026-09-25).
                # It is still stored, so the audit trail is complete; it is simply not re-learned from.
                self.store.mark_processed(t.id)
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
        self._warn_if_not_learning()
        return {"trace_id": t.id, "outcome": t.outcome, "confidence": t.confidence, "queued": True,
                "notes": notes, "sanitized": san, "request_id": obs.current_request_id()}

    def _warn_if_not_learning(self, backlog: int = 3) -> None:
        """learn() only QUEUES. If nothing ever calls process(), no skill is ever written and the loop never
        closes - silently. Say so, once, as soon as a backlog forms with no worker running."""
        if self._worker or self._warned_unprocessed:
            return
        try:
            pending = len(self.store.pending_traces(backlog))
        except Exception:
            return
        if pending >= backlog:
            self._warned_unprocessed = True
            import warnings
            msg = (f"SkillLoop: {pending}+ traces are queued but nothing is learning from them. Call "
                   f"loop.process() after tasks, use SkillLoop(auto_process=True), or set SKILLLOOP_AUTO_PROCESS=1.")
            warnings.warn(msg, RuntimeWarning, stacklevel=3)
            obs.log("learn.backlog_unprocessed", pending=pending)

    # ------------------------------------------------------------------ background
    def process(self, max_traces: int = 20) -> list[dict[str, Any]]:
        """Run the learning loop over queued traces. Safe to call repeatedly."""
        with self._lock:
            report = []
            errors: list[str] = []
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
                    errors.append(repr(e))
                self.store.mark_processed(t.id)
            if errors:
                # A trace that raises produces no skill, and until now that was indistinguishable from a
                # trace that legitimately taught nothing. An eval could therefore score a library that had
                # silently lost every skill (see CALIBRATION.md, 2026-09-25) and report it as a result.
                import warnings as _w
                kinds = sorted({e.split("(")[0] for e in errors})
                _w.warn(f"SkillLoop: {len(errors)} trace(s) failed during process() and produced no skill "
                        f"({', '.join(kinds)}). These are errors, not lessons; the library is incomplete.",
                        RuntimeWarning, stacklevel=2)
                obs.log("process.errors", count=len(errors), kinds=kinds)
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
        #
        # "quarantine" has TWO causes and only one of them should resume. A skill demoted by apply_use() for
        # a collapsed hit rate is also in quarantine with its lesson still claimed, and re-gating it would
        # send it back to candidate - the judge only reads the skill text, which has not changed, so it
        # passes again and the record of five real-world failures is erased. Demotion is what retires bad
        # skills, so undoing it silently would defeat the lifecycle policy.
        #
        # The two cases are distinguishable in the store: a skill stranded BEFORE the gate has no eval,
        # while a demoted one has a passed judge eval from when it was promoted.
        if (existing and lesson.id in existing.lesson_ids and existing.status == "quarantine"
                and not self.store.evals_for(existing.name)):
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


