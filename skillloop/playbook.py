"""Itemized skill bodies with incremental delta updates.

Motivated by Agentic Context Engineering (Zhang et al., ICLR 2026, arXiv:2510.04618), which identifies two
failure modes that SkillLoop v0.2 had by construction:

  1. BREVITY BIAS - v0.2 told the synthesizer "under 60 lines, no fluff". ACE shows that compressing away
     domain heuristics, tool quirks and failure modes costs real accuracy; contexts should be comprehensive
     playbooks and the model can pick what is relevant at inference time.
  2. CONTEXT COLLAPSE - v0.2 patched a skill by asking a model to REWRITE the whole body. ACE measures this
     directly: monolithic rewriting collapsed an 18k-token context to 122 tokens in a single step, dropping
     accuracy below the no-context baseline. Their ablation puts incremental updates at +11.7 TGC / +27.8 SGC.

So a skill body here is a list of itemized bullets, each with a stable id and helpful/harmful counters. A patch
produces DELTA OPERATIONS (add / revise / prune) that are merged by deterministic code, never by regenerating
the document. Counters come from the agent telling us which bullets actually helped, giving per-bullet credit
assignment instead of one hit-rate for the whole skill.
"""
from __future__ import annotations

import re
import time
import uuid
from typing import Any

SECTIONS = ("preconditions", "procedure", "verification", "failure_modes", "scope_limits")
SECTION_TITLES = {
    "preconditions": "Preconditions",
    "procedure": "Procedure",
    "verification": "Verification",
    "failure_modes": "Failure modes",
    "scope_limits": "Scope limits",
}


def new_bullet(section: str, content: str, **kw) -> dict[str, Any]:
    return {"id": f"b-{uuid.uuid4().hex[:6]}", "section": section if section in SECTIONS else "procedure",
            "content": content.strip(), "helpful": 0, "harmful": 0, "ts": time.time(), **kw}


def render(bullets: list[dict], show_ids: bool = True) -> str:
    """Markdown body grouped by section. Ids are shown so the agent can report which bullets it used."""
    out = []
    for sec in SECTIONS:
        items = [b for b in bullets if b.get("section") == sec]
        if not items:
            continue
        out.append(f"## {SECTION_TITLES[sec]}")
        for b in items:
            tag = f"[{b['id']}] " if show_ids else ""
            out.append(f"- {tag}{b['content']}")
        out.append("")
    return "\n".join(out).strip()


def parse_body(body: str) -> list[dict]:
    """Convert a legacy prose body into bullets, so v0.2 libraries keep working."""
    bullets: list[dict] = []
    section = "procedure"
    for line in (body or "").splitlines():
        line = line.rstrip()
        m = re.match(r"^##\s+(.*)$", line)
        if m:
            title = m.group(1).strip().lower()
            section = next((k for k, v in SECTION_TITLES.items() if v.lower() == title), "procedure")
            continue
        item = re.match(r"^\s*(?:[-*]|\d+\.)\s+(.*)$", line)
        if item and item.group(1).strip():
            text = item.group(1).strip()
            m2 = re.match(r"^\[(b-[0-9a-f]{6})\]\s*(.*)$", text)
            if m2:
                bullets.append({"id": m2.group(1), "section": section, "content": m2.group(2).strip(),
                                "helpful": 0, "harmful": 0, "ts": time.time()})
            else:
                bullets.append(new_bullet(section, text))
    return bullets


def _norm(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]{3,}", (s or "").lower())}


def similarity(a: str, b: str) -> float:
    ta, tb = _norm(a), _norm(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def apply_delta(bullets: list[dict], ops: list[dict], dedup_threshold: float = 0.85) -> tuple[list[dict], dict]:
    """Merge delta operations deterministically. No model call happens here - that is the point.

    ops: [{"type": "ADD"|"REVISE"|"PRUNE", "section": str, "content": str, "id": str}]
    Returns (bullets, stats).
    """
    stats = {"added": 0, "revised": 0, "pruned": 0, "deduped": 0}
    by_id = {b["id"]: b for b in bullets}
    for op in ops or []:
        kind = str(op.get("type", "ADD")).upper()
        if kind == "PRUNE" and op.get("id") in by_id:
            bullets = [b for b in bullets if b["id"] != op["id"]]
            by_id.pop(op["id"], None)
            stats["pruned"] += 1
            continue
        content = str(op.get("content", "")).strip()
        if not content:
            continue
        if kind == "REVISE" and op.get("id") in by_id:
            tgt = by_id[op["id"]]
            tgt["content"] = content
            tgt["ts"] = time.time()
            stats["revised"] += 1
            continue
        section = op.get("section", "procedure")
        dup = next((b for b in bullets if b.get("section") == section
                    and similarity(b["content"], content) >= dedup_threshold), None)
        if dup:                       # grow-and-refine: same knowledge, keep the existing id and counters
            stats["deduped"] += 1
            continue
        b = new_bullet(section, content)
        bullets.append(b)
        by_id[b["id"]] = b
        stats["added"] += 1
    return bullets, stats


def record_feedback(bullets: list[dict], helpful: list[str], harmful: list[str]) -> None:
    """Per-bullet credit assignment from the agent's own report."""
    idx = {b["id"]: b for b in bullets}
    for bid in helpful or []:
        if bid in idx:
            idx[bid]["helpful"] = idx[bid].get("helpful", 0) + 1
    for bid in harmful or []:
        if bid in idx:
            idx[bid]["harmful"] = idx[bid].get("harmful", 0) + 1


def refine(bullets: list[dict], max_bullets: int = 60, min_harmful: int = 3) -> tuple[list[dict], dict]:
    """Grow-and-refine, lazily: drop bullets the agent repeatedly flagged as harmful, de-duplicate, and cap
    size by evicting the least useful. Deliberately NOT a rewrite - that is what causes collapse."""
    stats = {"pruned_harmful": 0, "deduped": 0, "evicted": 0}
    kept: list[dict] = []
    for b in bullets:
        if b.get("harmful", 0) >= min_harmful and b.get("harmful", 0) > b.get("helpful", 0):
            stats["pruned_harmful"] += 1
            continue
        dup = next((k for k in kept if k.get("section") == b.get("section")
                    and similarity(k["content"], b["content"]) >= 0.85), None)
        if dup:
            dup["helpful"] = dup.get("helpful", 0) + b.get("helpful", 0)
            dup["harmful"] = dup.get("harmful", 0) + b.get("harmful", 0)
            stats["deduped"] += 1
            continue
        kept.append(b)
    if len(kept) > max_bullets:
        # keep verification and scope limits preferentially; evict lowest-value procedure detail
        def value(b):
            core = 1 if b.get("section") in ("verification", "scope_limits") else 0
            return (core, b.get("helpful", 0) - b.get("harmful", 0), b.get("ts", 0))
        kept.sort(key=value, reverse=True)
        stats["evicted"] = len(kept) - max_bullets
        kept = kept[:max_bullets]
        kept.sort(key=lambda b: (SECTIONS.index(b.get("section", "procedure")), b.get("ts", 0)))
    return kept, stats


def missing_sections(bullets: list[dict]) -> list[str]:
    have = {b.get("section") for b in bullets}
    return [SECTION_TITLES[s] for s in SECTIONS if s not in have]
