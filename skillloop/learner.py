"""Reflect on a trace -> Lesson. Turn lessons -> Skill create / patch. Distill lessons -> Principles.

Design rules baked in:
- Root cause, not symptom. The reflector must name the decision that went wrong.
- Blame attribution when a skill was in play: wrong / incomplete / misapplied / none.
- A lesson is only turned into a skill if it is marked generalizable.
- Skills are agentskills.io SKILL.md with a FORCED shape: Preconditions / Procedure / Verification /
  Failure modes / Scope limits. Verification is mandatory: a skill without a way to prove it worked is rejected.
- Principles: cross-cutting rules distilled from many lessons, injected on every recall.
"""
from __future__ import annotations

import copy
import re
import time
import uuid

from .llm import LLM
from .schema import Trace
from .playbook import apply_delta, missing_sections, new_bullet, parse_body, refine
from .store import Lesson, Principle, Skill, Store, slugify

REFLECT_SYSTEM = """You are the post-mortem analyst for an autonomous agent. You will be shown one task trajectory.
Your job: extract the single most useful lesson, stated as a rule the agent could follow next time.

Be ruthless and specific:
- root_cause must name the DECISION or ASSUMPTION that caused the result, not the error message. Ask "what did the
  agent believe that was false, or what did it skip that a senior engineer would never skip?"
- rule must be an actionable "When <situation>, do <action> (because <reason>)". No platitudes like "be careful".
- If the task succeeded, the rule is the reusable procedure that made it work (only if it was non-trivial: 3+ tool
  calls or a non-obvious trick). Trivial successes: generalizable=false.
- generalizable=false if this only applies to this exact file/repo/one-off input.
- If a skill was used (see SKILLS USED), assign blame:
    "skill_wrong"      - the skill's instructions caused the failure
    "skill_incomplete" - the skill was right but missing a case that mattered here
    "skill_misapplied" - the skill was fine but should not have triggered / was used incorrectly
    "none"             - skill used and the outcome was fine or unrelated to it
  If no skill was used: blame="no_skill".
- triggers: 3-6 short phrases a user might say, or contexts, where this rule should fire.
- verification: how the agent could have PROVEN the task worked (a command to run, state to read back, a test).
- principle: if this lesson is an instance of a broader discipline rule that applies across many task types
  (e.g. "never declare success without reading state back"), state it in one sentence; else null.
- EVIDENCE DISCIPLINE: only name a root cause the trajectory actually shows. evidence_steps must list the step
  indices (the [N] numbers) whose tool output or error DEMONSTRATES the mechanism you name. A user saying "it
  didn't work" is a symptom, not evidence of a mechanism. If no tool output shows WHY (e.g. the user says "it
  still shows Welcom" and nothing in the trajectory reveals whether it is a cache, a wrong file, or a build step),
  set root_cause to "undetermined: <the diagnostic the agent should have run>", evidence_steps=[],
  generalizable=false, confidence <= 0.4. The rule in that case is the diagnostic to run, not a guessed fix.
  Never invent a plausible mechanism (caching, permissions, DNS...) that no step demonstrates.
- confidence (calibrate it, this number gates whether a skill is written):
    0.9+  the trajectory shows the full causal chain: the skipped check AND the error it caused AND the fix working
    0.7-0.85 the failing decision is visible and the rule is standard practice, but the fix was not demonstrated
    0.5-0.65 the decision is a reasonable inference; another cause is possible
    <=0.4 undetermined / the trajectory is too short to tell
  Most single failing traces without a demonstrated fix belong in 0.7-0.85, not above.

JSON schema:
{"root_cause": str, "evidence_steps": [int], "rule": str, "triggers": [str], "generalizable": bool,
 "blame": "none|skill_wrong|skill_incomplete|skill_misapplied|no_skill", "skill_ref": str|null,
 "confidence": float, "suggested_skill_name": str, "verification": str, "principle": str|null}"""

PATCH_SYSTEM = """You are the CURATOR of an existing agent skill. New evidence has arrived. Emit ONLY the delta:
the specific bullets to add, revise, or prune. Do NOT rewrite the skill and do NOT restate bullets that are
already correct - rewriting whole documents causes accumulated knowledge to be silently dropped.

Each existing bullet is shown with its id. Rules:
- ADD a bullet only for knowledge that is genuinely absent. Be specific and keep each bullet self-contained.
- REVISE (with the bullet's id) when an existing bullet is wrong or incomplete; give the full corrected text.
- PRUNE (with the id) only when a bullet is actively misleading. Never prune a failure mode that earlier
  evidence established.
- Sections: preconditions | procedure | verification | failure_modes | scope_limits
- Prefer several precise bullets over one vague one. Detail is valuable here; do not compress.

JSON: {"operations": [{"type": "ADD|REVISE|PRUNE", "section": str, "content": str, "id": str|null}],
       "change_summary": str,
       "facets": {"applies_when": [str], "symptoms": [str], "not_for": [str]},
       "trigger_queries": [str]}
(facets and trigger_queries: only NEW ones to merge in; omit or leave empty if nothing new.)"""


SYNTH_SYSTEM = """You write skills for autonomous agents in agentskills.io SKILL.md format. The skill must make the agent
behave like a senior engineer on this kind of task: check preconditions, follow a procedure, PROVE it worked, know
the failure modes, and know when NOT to use the skill.

Rules:
- description: one or two sentences: what the skill does and the SITUATION it applies to. Be PRECISE, not
  greedy - this text is what retrieval matches on. Name the specific artifact, tool, or error involved
  ("a JSON config file", "pip install failing", "a percentage/tax calculation"). Do NOT list adjacent
  situations you did not learn about, and do NOT pad with generic verbs like "writes a script", "creates a
  file", "runs a command" - every task does those, so they make the skill fire on everything.
- facets: the retrieval index. Keep each entry to 1-4 words.
    applies_when - 3-6 phrases naming the SITUATION: the artifact/format ("json config", "ini file"), the tool
                   ("pip", "jq", "pytest"), the operation ("bulk edit", "schema migration"). These are what a
                   user's request will actually contain.
    symptoms     - distinctive error text or observable failures ("externally-managed-environment",
                   "ModuleNotFoundError", "exit 0 but no output"). Empty list if none.
    not_for      - adjacent situations this skill must NOT be pulled into. If the skill is about JSON, say
                   "ini", "yaml", "toml", "csv". This actively blocks wrong retrieval, so be specific.
- trigger_queries: 8-12 SHORT requests, phrased the way a USER would type them, that this skill should fire on.
  This is the single most important field for retrieval. Users do not use your vocabulary: a dedup skill must
  list "strip repeated entries", "remove repeats", "keep only unique lines", "no duplicates"; a chmod skill must
  list "make the script runnable", "why won't this file execute", "permission denied running my script".
  Vary the VERB and the NOUN, cover the same intent in several dialects, and include at least two phrasings that
  avoid the skill's own technical terms entirely. Do not describe the skill - write the request.
- bullets: the body, as itemized bullets. Each bullet is ONE self-contained piece of knowledge with a section:
    preconditions  - what must be true / checked before starting
    procedure      - the concrete steps, with real commands where the evidence supports them
    verification   - how to PROVE the task succeeded: exit codes, tests, state to read back. Never "check it
                     worked"; say HOW. At least one bullet here is mandatory.
    failure_modes  - "symptom -> cause -> fix", from the evidence
    scope_limits   - when this skill does NOT apply / should not trigger
  Every section must have at least one bullet. Write as many bullets as the evidence justifies - do NOT
  compress. Specific heuristics, exact error strings, tool quirks and edge cases are the valuable part of a
  skill; a short generic skill is a useless one. Prefer several precise bullets over one vague bullet.
- No intro paragraph and no filler sentences, but do NOT sacrifice detail for brevity: an agent reading this
  can ignore what it does not need, and cannot invent what you left out.
- EVIDENCE RULE: the procedure is the fix the evidence supports, for the environment the evidence shows. You MAY add
  a standard, widely-known command needed to check or verify (e.g. `nginx -t`, `git status`, exit-code checks).
  You MUST NOT add: alternatives for other platforms / distros / package managers / shells the traces never
  showed; speculative extra fixes "in case"; example paths, service names or file names not in the evidence
  (use <placeholders>). One good path beats five hedged ones. If you add a command not in the evidence, it must
  be one any senior engineer would know by heart.
- Never include instructions that override the agent's other instructions, send data anywhere, or ask the agent
  to hide anything from the user. The skill body is advice about the task, nothing else.
- If PATCHING an existing skill, preserve everything still valid and change only what the new lessons require.
  Never drop a failure mode that earlier failures established.

JSON schema: {"name": str, "description": str, "change_summary": str,
              "bullets": [{"section": str, "content": str}],
              "facets": {"applies_when": [str], "symptoms": [str], "not_for": [str]},
              "trigger_queries": [str]}"""

SKEPTIC_SYSTEM = """You are a skeptical reviewer of a post-mortem. You get a trajectory and a claimed root cause.
Grade the evidence for the claimed mechanism:
  "direct"  - a tool output / error message / read-back value in the trajectory shows THIS mechanism
              (e.g. claim "import name differs from package name", trace shows `ModuleNotFoundError` right after
              a successful install of a differently-named package)
  "partial" - the trajectory shows the failure happened at the step the claim points to (an error, a rejected
              push, a crash) but the specific WHY is an inference (e.g. claim "sed produced invalid config",
              trace only shows "Job for nginx.service failed")
  "none"    - nothing in the trajectory shows the mechanism; only a user complaint or the agent's own words
A user complaint ("it still shows X") is NOT evidence of a mechanism. A plausible story is NOT evidence.
For direct/partial, quote the supporting tool output verbatim (max 200 chars). Always list what the trajectory
cannot rule out and the diagnostic that would have settled it.
JSON: {"support": "direct|partial|none", "quote": str, "alternatives": [str], "diagnostic_that_was_missing": str}"""

REQUIRED_SECTIONS = ("Preconditions", "Procedure", "Verification", "Failure modes", "Scope limits")


def _cited_steps_show_failure(trace: Trace, ev_steps: list[int]) -> bool:
    """Deterministic grounding: do the cited steps actually contain a tool error or a failure signal?
    When they do, the lesson is anchored in something real and the model-based skeptic is not needed."""
    # compact() labels steps by their position, so ev_steps are indices into trace.steps
    for i in ev_steps:
        if 0 <= i < len(trace.steps) and any(tc.error for tc in trace.steps[i].tool_calls):
            return True
    return False


def reflect(llm: LLM, trace: Trace, policy=None) -> Lesson:
    raw = llm.json(REFLECT_SYSTEM, trace.compact(), role="reflect")
    conf = float(raw.get("confidence", 0.5) or 0.0)
    ev_steps = [int(x) for x in raw.get("evidence_steps", []) if str(x).lstrip("-").isdigit()]
    rc = str(raw.get("root_cause", "")).strip()
    # deterministic guard: a cause with no demonstrating step is a guess, whatever the model's confidence says
    if trace.outcome != "success" and not ev_steps and not rc.lower().startswith("undetermined"):
        conf = min(conf, 0.35)
        rc = "unsupported (no evidence step cited): " + rc
    # adversarial pass: a second model call must quote the evidence, or the lesson is a guess
    skeptic = None
    grounded_locally = bool(ev_steps) and _cited_steps_show_failure(trace, ev_steps)
    if grounded_locally:
        # the cited steps contain a real tool error: treat as partial evidence without spending a model call
        skeptic = {"support": "partial", "effective": "partial", "source": "deterministic",
                   "quote": "(cited step contains a tool error)"}
        conf = min(conf, 0.75)
    elif trace.outcome != "success" and conf >= 0.5 and not rc.lower().startswith(("undetermined", "unsupported")):
        try:
            try:
                skeptic = llm.json(SKEPTIC_SYSTEM, f"--- TRAJECTORY ---\n{trace.compact(6000)}\n\n--- CLAIMED ROOT CAUSE ---\n{rc}",
                                   max_tokens=800, role="judge")
            except Exception as e:
                if type(e).__name__ == "PendingManualCall":
                    raise                     # the manual provider must be allowed to pause the pipeline here
                raise
            quote = str(skeptic.get("quote") or "")
            support = str(skeptic.get("support") or ("direct" if skeptic.get("supported") else "none")).lower()
            quoted = len(quote) >= 8 and _quote_in(quote, trace)
            diag = str(skeptic.get("diagnostic_that_was_missing") or "").strip()
            if support == "direct" and quoted:
                pass
            elif support in ("direct", "partial") and quoted:
                # failure is shown, mechanism is inferred: keep the lesson, but it must wait for corroboration
                conf = min(conf, 0.65)
                skeptic["effective"] = "partial"
            else:
                conf = min(conf, 0.35)
                skeptic["effective"] = "none"
                rc = "undetermined (skeptic found no direct evidence): " + rc + (f" | missing diagnostic: {diag}" if diag else "")
        except Exception:
            pass
    return Lesson(
        id=uuid.uuid4().hex[:10],
        trace_id=trace.id,
        outcome=trace.outcome,
        root_cause=rc,
        rule=str(raw.get("rule", "")).strip(),
        triggers=[str(t) for t in raw.get("triggers", [])][:8],
        generalizable=bool(raw.get("generalizable", False)),
        blame=str(raw.get("blame", "no_skill")),
        skill_ref=raw.get("skill_ref") or (trace.skills_used[0] if trace.skills_used else None),
        confidence=conf,
        evidence_steps=ev_steps,
        skeptic=skeptic,
        verification=str(raw.get("verification") or "").strip(),
        principle=(str(raw["principle"]).strip() if raw.get("principle") else None),
        suggested_name=str(raw.get("suggested_skill_name") or "").strip() or None,
    )


_VAGUE_VERIFY = re.compile(r"^\W*(check|confirm|ensure|verify|make sure) (it|that it|everything|the (change|task)) (works?|worked|succeeded)\W*$", re.I)


def _quote_in(quote: str, trace: Trace) -> bool:
    """Is the skeptic's quote actually present in the trajectory (tool output/error/content)? Loose match."""
    import re as _re
    norm = lambda x: _re.sub(r"[\s`'\"“”‘’]+", " ", (x or "")).strip().lower()
    q = norm(quote)
    hay = norm(trace.compact(20000))
    if not q:
        return False
    # accept if a substantial window of the quote appears verbatim (models trim/elide long lines)
    for win in (60, 40, 25):
        if len(q) >= win and any(q[i:i + win] in hay for i in range(0, len(q) - win + 1, 10)):
            return True
    return q in hay


def check_shape(body: str) -> list[str]:
    """Return the list of missing/empty/vague required sections. Verification must be concrete: at least one
    code span (a command, a value to read back) and not just 'check it worked'."""
    missing = []
    for sec in REQUIRED_SECTIONS:
        m = re.search(rf"^##\s+{re.escape(sec)}\s*$(.*?)(?=^##\s|\Z)", body, flags=re.S | re.M | re.I)
        if not m or len(m.group(1).strip()) < 10:
            missing.append(sec)
            continue
        if sec == "Verification":
            txt = m.group(1).strip()
            if "`" not in txt or _VAGUE_VERIFY.match(txt.strip("-* ")):
                missing.append("Verification (must name a concrete command/observable, in backticks)")
    return missing


def synthesize(llm: LLM, store: Store, lesson: Lesson, related: list[Lesson], existing: Skill | None,
               suggested_name: str | None = None) -> tuple[Skill, str]:
    """Create a skill from evidence, or PATCH one with incremental delta operations.

    Patching never regenerates the body. See playbook.py for why (context collapse).
    """
    evidence = "\n".join(
        f"- [{l.outcome}] cause: {l.root_cause} | rule: {l.rule}"
        + (f" | verify: {l.verification}" if l.verification else "")
        for l in [lesson, *related]
    )
    if existing:
        # deep-copy: apply_delta mutates, and the caller's Skill (and its snapshot) must not change underneath
        bullets = copy.deepcopy(existing.bullets) if existing.bullets else parse_body(existing.body)
        listing = "\n".join(f"[{b['id']}] ({b['section']}) {b['content']}" for b in bullets) or "(empty)"
        user = (f"--- EXISTING SKILL: {existing.name} ---\n{existing.description}\n\n--- EXISTING BULLETS ---\n"
                f"{listing}\n\n--- NEW EVIDENCE ---\n{evidence}\n"
                f"--- BLAME on the existing skill: {lesson.blame} ---")
        raw = llm.json(PATCH_SYSTEM, user, max_tokens=2000, role="synth")
        ops = [o for o in raw.get("operations", []) if isinstance(o, dict)]
        bullets, stats = apply_delta(bullets, ops)
        bullets, rstats = refine(bullets)
        change = f"{raw.get('change_summary', 'patched')} [{stats['added']}+ {stats['revised']}~ {stats['pruned']}- {stats['deduped']}dup]"
        name, description = existing.name, existing.description
    else:
        raw = llm.json(SYNTH_SYSTEM, f"CREATE a new skill from this evidence.\n\n--- EVIDENCE ---\n{evidence}\n\n"
                       f"--- SUGGESTED NAME --- {suggested_name or lesson.suggested_name or slugify(lesson.rule)}\n"
                       f"--- TRIGGER PHRASES --- {', '.join(lesson.triggers)}",
                       max_tokens=3000, role="synth")
        bullets = [new_bullet(str(b.get("section", "procedure")), str(b.get("content", "")))
                   for b in raw.get("bullets", []) if str(b.get("content", "")).strip()]
        missing = missing_sections(bullets)
        if missing:
            raw2 = llm.json(SYNTH_SYSTEM, f"CREATE a new skill from this evidence.\n\n--- EVIDENCE ---\n{evidence}\n\n"
                            f"--- REJECTED: your last draft had no bullets in: {', '.join(missing)}. "
                            f"Every section needs at least one bullet. ---", max_tokens=3000, role="synth")
            b2 = [new_bullet(str(b.get("section", "procedure")), str(b.get("content", "")))
                  for b in raw2.get("bullets", []) if str(b.get("content", "")).strip()]
            if len(missing_sections(b2)) < len(missing):
                bullets, raw = b2, raw2
        anchor = " ".join((raw.get("facets") or {}).get("applies_when", [])[:2])
        name = slugify(raw.get("name") or anchor or suggested_name or lesson.suggested_name or lesson.rule)
        description = str(raw.get("description", "")).strip() or lesson.rule
        change = str(raw.get("change_summary", "created"))

    fac = raw.get("facets") or {}
    facets = {k: [str(x)[:40] for x in (fac.get(k) or [])][:8] for k in ("applies_when", "symptoms", "not_for")}
    _anchor = {w for ph in facets["applies_when"] for w in ph.lower().split()} | set(name.lower().replace("-", " ").split())
    facets["not_for"] = [n for n in facets["not_for"] if not (set(n.lower().split()) & _anchor)]
    queries = [str(q)[:120] for q in (raw.get("trigger_queries") or [])][:14]
    if existing:
        for k in facets:
            facets[k] = sorted(set(facets[k]) | set(existing.facets.get(k, [])))[:8]
        queries = sorted(set(queries) | set(existing.queries))[:20]

    prov = sorted(set((existing.provenance if existing else []) + [l.trace_id for l in [lesson, *related]]))
    # evidence is the SET of lessons that contributed, not a counter. Incrementing a counter double-counted
    # every time a trace was reprocessed: one skill reached evidence_count 45 from about five real lessons,
    # which in turn corrupted the `provisional` flag and the cold-start policy that keys off it.
    lesson_ids = sorted(set((existing.lesson_ids if existing else []) + [l.id for l in [lesson, *related]]))
    n_evidence = len(lesson_ids)
    skill = Skill(
        name=name, description=description, body="", bullets=bullets, lesson_ids=lesson_ids,
        facets=facets, queries=queries,
        status="quarantine",                       # always re-gate after a change
        version=(existing.version + 1) if existing else 1,
        provenance=prov,
        uses=existing.uses if existing else 0,
        successes=existing.successes if existing else 0,
        failures=existing.failures if existing else 0,
        recalls=existing.recalls if existing else 0,
        created=existing.created if existing else time.time(),
        evidence_count=n_evidence,
        provisional=n_evidence < 2,
    )
    return skill, change


_POOL_GENERIC = {"file", "files", "script", "python", "output", "write", "create", "run", "command", "data",
                 "value", "values", "check", "verify", "read", "line", "lines", "text", "code", "result",
                 "content", "contents", "save", "saved", "into", "onto", "with", "from", "that", "this",
                 "txt", "log", "csv", "json", "yaml", "yml", "sh", "md", "dat", "ini", "conf",
                 "task", "agent", "shell", "bash", "before", "after", "using", "should", "must", "when",
                 "then", "correct", "correctly", "properly", "ensure", "ensures", "make", "makes", "sure"}


def find_related(store: Store, lesson: Lesson, limit: int = 4) -> list[Lesson]:
    """Other lessons about the SAME situation. Pooling is what lets a skill generalize past one incident, but
    pooling on generic words is how a skill ends up claiming pip installs and heredocs and JSON at once - it
    then gets retrieved for everything. Require overlap on distinctive terms only."""
    def words(l: Lesson) -> set[str]:
        ws = {w.lower().strip(".,:;()") for t in l.triggers for w in t.split()}
        ws |= {w.lower().strip(".,:;()") for w in (l.rule or "").split()}
        return {w for w in ws if len(w) > 3 and w not in _POOL_GENERIC}

    mine = words(lesson)
    out = []
    for l in store.lessons(300):
        if l.id == lesson.id or not l.generalizable:
            continue
        shared = mine & words(l)
        if len(shared) >= 3:          # 2 was loose enough to pool unrelated tasks through shared filler
            out.append(l)
        if len(out) >= limit:
            break
    return out


OUTCOME_SYSTEM = """You judge whether an agent completed its task. Read the trajectory and decide.
success = the user's stated goal was clearly achieved AND there is evidence in the trajectory (tool output, test
result, state read back) — not just the agent saying so. failure = it was not, or the agent gave up / hallucinated
completion. partial = some but not all. Look at tool errors, retries, and whether the final message actually
delivers the goal. Be skeptical of the agent claiming success without evidence.
JSON: {"outcome": "success|failure|partial", "confidence": float, "detail": str}"""


def judge_outcome(llm: LLM, trace: Trace):
    from .schema import Signal
    raw = llm.json(OUTCOME_SYSTEM, trace.compact(8000), role="outcome")
    return Signal("judge", str(raw.get("outcome", "unknown")), float(raw.get("confidence", 0.5) or 0.0),
                  str(raw.get("detail", ""))[:300])


# ---------------------------------------------------------------- principles
PRINCIPLE_SYSTEM = """You distill cross-cutting engineering principles from a list of lessons an agent learned.
A principle is a discipline rule that applies across MANY kinds of tasks (not one tool, not one repo), e.g.
"Never report success without reading the resulting state back" or "After any install, import/run the thing
before building on it". Merge near-duplicates. Only emit a principle if >= 3 lessons support it.
Existing principles are given; do not re-emit them unless you are refining the wording (then mark replaces).
JSON: {"principles": [{"rule": str, "lesson_ids": [str], "replaces": str|null}]}"""


def distill_principles(llm: LLM, store: Store, min_support: int = 3) -> list[Principle]:
    lessons = [l for l in store.lessons(400) if l.generalizable]
    if len(lessons) < min_support:
        return []
    existing = store.principles()
    ex_txt = "\n".join(f"- [{p.id}] {p.rule}" for p in existing if p.active) or "(none)"
    ls_txt = "\n".join(
        f"- [{l.id}] {l.rule}" + (f" (principle hint: {l.principle})" if l.principle else "") for l in lessons
    )
    raw = llm.json(PRINCIPLE_SYSTEM, f"--- EXISTING PRINCIPLES ---\n{ex_txt}\n\n--- LESSONS ---\n{ls_txt}",
                   max_tokens=2000, role="synth")
    out = []
    for p in raw.get("principles", []):
        ids = [str(i) for i in p.get("lesson_ids", [])]
        if len(ids) < min_support or not p.get("rule"):
            continue
        if p.get("replaces"):
            for e in existing:
                if e.id == p["replaces"]:
                    e.active = False
                    store.add_principle(e)
        pr = Principle(id=uuid.uuid4().hex[:8], rule=str(p["rule"]).strip(), evidence_count=len(ids), lesson_ids=ids)
        store.add_principle(pr)
        store.log("principle", pr.id, pr.rule[:120])
        out.append(pr)
    return out


QUERY_SYSTEM = """You write the retrieval index for an existing agent skill. Emit 8-12 SHORT requests, phrased the
way a USER would type them, that should cause this skill to be retrieved.

Users do not use the skill's vocabulary. A dedup skill must list "strip repeated entries", "remove repeats",
"keep only unique lines", "no duplicates". A chmod skill must list "make the script runnable", "permission denied
when running my script". A numeric-sort skill must list "find the largest number", "what's the smallest value".
Vary the VERB and the NOUN. At least three phrasings must avoid the skill's own technical terms entirely.
Write requests, not descriptions. No numbering.
JSON: {"queries": [str]}"""


def backfill_queries(llm: LLM, store: Store, only_missing: bool = True) -> list[str]:
    """doc2query for skills written before trigger_queries existed. This is what closes the vocabulary gap
    between how a skill describes itself and how a user asks for it."""
    done = []
    for sk in store.all_skills():
        if only_missing and sk.queries:
            continue
        try:
            raw = llm.json(QUERY_SYSTEM, f"--- SKILL ---\n{sk.to_skill_md()}", max_tokens=700, role="judge")
        except Exception:
            continue
        sk.queries = [str(q)[:120] for q in (raw.get("queries") or [])][:14]
        store.save_skill(sk, snapshot=False)
        done.append(f"{sk.name}({len(sk.queries)})")
    return done


FACET_SYSTEM = """You write the retrieval index for an existing agent skill. Read the skill and emit facets that
decide WHEN it should be pulled into a task. Keep each entry 1-4 words, lowercase.
  applies_when - 3-6 phrases naming the situation: artifact/format ("json config", "ini file"), tool ("pip",
                 "jq", "pytest"), operation ("bulk edit"). These are words a user's actual request will contain.
  symptoms     - distinctive error text or observable failures. Empty list if none.
  not_for      - adjacent situations this skill must NOT be pulled into. Be specific: if it is about JSON, list
                 "ini", "yaml", "toml", "csv". This actively blocks wrong retrieval.
Do not include generic words (file, script, run, output, create, verify) - they make the skill match everything.
JSON: {"applies_when": [str], "symptoms": [str], "not_for": [str]}"""


def backfill_facets(llm: LLM, store: Store, only_missing: bool = True) -> list[str]:
    """Add retrieval facets to skills written before facets existed."""
    done = []
    for sk in store.all_skills():
        if only_missing and sk.facets:
            continue
        try:
            raw = llm.json(FACET_SYSTEM, f"--- SKILL ---\n{sk.to_skill_md()}", max_tokens=600, role="judge")
        except Exception:
            continue
        sk.facets = {k: [str(x)[:40].lower() for x in (raw.get(k) or [])][:8]
                     for k in ("applies_when", "symptoms", "not_for")}
        store.save_skill(sk, snapshot=False)
        done.append(sk.name)
    return done
