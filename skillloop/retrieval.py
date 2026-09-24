"""Retrieval: choosing which skills come back for a task, an error, or neither.

This is the hot path. It runs on every `recall()` and every failed tool call, so it is local,
deterministic and makes no model call; the only optional network hop is the embedding lookup in
`_embed_candidates`, and the lexical path stands on its own without it.

The engine is split out of `core.py` because it is the part a reviewer most often wants to read on
its own, and because its behaviour is pinned by two standalone evals (`qa/run_r1.py`, `qa/run_r2.py`)
that exercise it without the rest of the pipeline. It is a mixin rather than a separate object so
that the public surface stays exactly as it was: `SkillLoop.recall(...)`, not
`SkillLoop.retrieval.recall(...)`.

Three ideas carry most of the precision, and each exists because a probe class failed without it:

* **Anchor rule** - the query must touch what a skill *is* (its name or `applies_when`), not merely a
  word in its description. This is what stops "change ... settings" pulling a JSON skill into an INI
  task (R1 class D).
* **Intent gate** - a knowledge question ("what is the default max_connections") or a creative request
  ("write a poem about docker") is not a task any procedural skill should answer, and a creation
  request may only be answered by an authoring-scoped skill (R2 class T).
* **Symptom matching** - when the agent is staring at an error, the literal error text recorded on a
  skill is the strongest evidence available, so a verbatim symptom match outranks the usual lexical
  scoring and can only be vetoed by a verbatim `not_for` match (R2 class S).

Abstention is a first-class answer throughout: returning nothing is correct far more often than
returning a plausible-but-wrong skill, because a wrong skill costs the agent a whole attempt.
"""
from __future__ import annotations

import math
import os
import re
from typing import Any

from .store import Skill


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0

class RetrievalMixin:
    """The retrieval half of :class:`skillloop.core.SkillLoop`.

    Mixed into ``SkillLoop``; it relies on ``self.store``, ``self.llm`` and the retrieval tuning
    attributes set in ``SkillLoop.__init__`` (``query_threshold``, ``embed_strong``, ``embed_topk``,
    ``hybrid_lexical_w``, ``hybrid_embed_w``). It is not meant to be instantiated on its own.
    """

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

    def _best_symptom_match(self, text: str, terms: set[str], sk: Skill) -> float:
        """Phrase-level match of the text against the skill's recorded symptoms. 1.0 when a symptom appears
        verbatim (case-insensitive); else the fraction of that symptom's terms present, when at least two are
        shared. Same discipline as _best_query_match: one shared word is a coincidence."""
        low = (text or "").lower()
        best = 0.0
        for sym in (sk.facets or {}).get("symptoms", []):
            sym_l = (sym or "").strip().lower()
            if len(sym_l) >= 6 and sym_l in low:
                return 1.0
            st = self._query_terms(sym_l)
            if len(st) < 2:
                continue
            shared = terms & st
            if len(shared) < 2:
                continue
            best = max(best, len(shared) / len(st))
        return best

    _ERR_LINE = re.compile(r"error|fail|fatal|conflict|exception|traceback|denied|not found|no such|exited"
                           r"|refused|timeout|timed out|cannot|can't|unable|invalid|warning|killed|abort"
                           r"|unchanged|did not|didn't|nothing|no tests|seq scan|too many", re.I)

    @classmethod
    def _error_signature(cls, error: str, max_chars: int = 900) -> str:
        """The informative part of raw error output. Tool errors are long and noisy (stack frames, paths,
        hashes, timings); the signal is in the lines that say what went wrong, and usually in the LAST such
        line (Python tracebacks end with the exception). Paths and hex ids are reduced to their last
        component / removed so they don't pose as rare, highly-weighted terms."""
        if not error:
            return ""
        text = str(error)
        text = re.sub(r"\b[0-9a-f]{7,40}\b", " ", text)                  # commit / container ids
        text = re.sub(r"(?:/[\w.\-]+)+/([\w.\-]+)", r"\1", text)         # /a/b/c.py -> c.py
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if not lines:
            return ""
        keep = [l for l in lines if cls._ERR_LINE.search(l) or cls._ERR_ID.search(l)]
        if lines[-1] not in keep:
            keep.append(lines[-1])
        sig = "\n".join(keep[-8:]) if keep else "\n".join(lines[-4:])
        return sig[-max_chars:]

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

    def _search(self, task: str, statuses: tuple[str, ...], limit: int,
                mode: str = "task") -> list[tuple[Skill, float]]:
        """IDF-weighted coverage over each skill's purpose, with abstention.

        Score = (weighted IDF mass of query terms this skill covers) / (total IDF mass of the query).
        A skill is only returned if it clears an absolute threshold AND matches at least one term that is
        rare across the library - otherwise every skill wins on shared generic vocabulary.
        """
        qt = self._terms(task)
        if not qt:
            return []
        # INTENT GATE (query-level): a creative or factual-question request is not a task any procedural
        # skill should answer. Applied once, before any retrieval path, so it covers lexical AND embedding.
        # Error mode skips it: an error message IS a task by definition.
        intent = "task" if mode == "error" else self._intent(task)
        if intent in ("knowledge", "creative"):
            self.store.log("recall_filtered", "*", f"{intent} query, no skill fires :: {task[:60]}")
            return []
        profiles = self._profiles(statuses)
        if intent == "create":
            # "write a Dockerfile" is not "my docker build is slow": only authoring skills may answer it
            profiles = [p for p in profiles if self._authoring_scoped(p[0])]
            if not profiles:
                self.store.log("recall_filtered", "*", f"create request, no authoring skill :: {task[:60]}")
                return []
        if not profiles:
            return []
        idf = self._idf(profiles)
        default_idf = max(idf.values(), default=1.0)
        q_mass = sum(idf.get(t, default_idf) for t in qt) or 1.0
        scored: list[tuple[Skill, float, int]] = []
        err_terms = self._query_terms(task) if mode == "error" else set()
        problem = mode == "error" or bool(self._ERR_ID.search(task) or self._PROBLEM.search(task))
        for sk, prof, anchors in profiles:
            hits = qt & set(prof)
            qsim = self._best_query_match(self._query_terms(task), sk)
            # ERROR MODE: a skill's symptoms are literal error signatures captured from a real failure. When the
            # agent is looking at an error, a symptom match is the strongest evidence there is, so symptom
            # terms count as anchors and a phrase-level symptom match scores like a trigger-query match.
            ssim = 0.0
            if mode == "error":
                ssim = self._best_symptom_match(task, err_terms, sk)
                anchors = anchors | self._terms(" ".join((sk.facets or {}).get("symptoms", [])))
                qsim = max(qsim, ssim)
            if not hits and qsim < self.query_threshold:
                continue
            # ONE shared word is a coincidence, not a match - the rule _best_query_match already applies to
            # trigger phrases, now applied to facets too. "how do I measure test coverage" shares only "test"
            # with a flaky-test skill. Exempt: short queries (1-2 terms, nothing else to share) and identifier
            # or error-code terms (ModuleNotFoundError, test_orders.py), which are specific on their own, and
            # an anchor word (what the skill is FOR) when the user is reporting a problem: "the package import
            # keeps failing" is about the skill's subject AND someone is stuck; "how do I measure test coverage"
            # only mentions the subject.
            if len(hits) == 1 and qsim < self.query_threshold and len(qt) >= 3:
                only = next(iter(hits))
                specific = self._ERR_ID.search(only) or re.search(r"[_.\d]", only) or only in err_terms
                if not specific and not (only in anchors and problem):
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
            if self._excluded_by_not_for(sk, qt, anchors, strict_phrase=(ssim >= 0.6),
                                         verbatim_text=(task.lower() if ssim >= 1.0 else None)):
                continue
            mass = sum(idf.get(t, default_idf) * prof[t] for t in hits)
            # breadth normalization: a skill advertising itself for twenty situations should not beat a focused
            # one on a single shared word. Same idea as document-length normalization in BM25 - kept gentle so
            # it only breaks ties, never outweighs actually matching more of the query.
            breadth = max(1.0, len(anchors) / 4.0) ** 0.25
            facet_score = mass / (q_mass * 3.0 * breadth) if hits else 0.0
            rank = len(hits) + (2 if qsim >= self.query_threshold else 0) + (4 if ssim >= 0.6 else 0)
            scored.append((sk, max(facet_score, qsim), rank))
        # rank by how many distinct query terms the skill covers; breadth-adjusted mass separates equals
        scored.sort(key=lambda x: (-x[2], -x[1]))
        scored = [(sk, sc) for sk, sc, _n in scored]
        keep = [(sk, sc) for sk, sc in scored if sc >= self.recall_threshold]
        if keep and len(keep) > 1:
            # drop anything far weaker than the best match; a trailing weak hit is noise, not a second opinion
            best = keep[0][1]
            keep = [(sk, sc) for sk, sc in keep if sc >= best * self.recall_margin]
        lexical = keep  # (skill, lexical_score) that passed the lexical gates above

        # ---- HYBRID: let embeddings surface candidates lexical missed (paraphrase robustness) ----
        emb_hits = self._embed_candidates(task, statuses)   # [(skill, cos)] above embed_threshold, or []
        if not emb_hits:
            return lexical[:limit]

        # union by name; fuse scores. A skill strong on EITHER signal is a candidate; the precision gates
        # (anchor / not_for / intent) below decide whether it actually fires.
        lex = {sk.name: (sk, sc) for sk, sc in lexical}
        fused: dict[str, tuple[Skill, float, float]] = {}
        for name, (sk, lsc) in lex.items():
            fused[name] = (sk, lsc, 0.0)
        for sk, cos in emb_hits:
            l = fused.get(sk.name, (sk, 0.0, 0.0))
            fused[sk.name] = (sk, l[1], cos)

        out = []
        for name, (sk, lsc, cos) in fused.items():
            # embedding-only candidate (lexical missed it): apply the SAME precision gates lexical uses, so a
            # semantically-near but wrong-intent skill is still filtered. Reuses anchors + not_for.
            if lsc == 0.0:
                if intent == "create" and not self._authoring_scoped(sk):
                    continue
                if not self._passes_precision_gate(task, sk, cos=cos, mode=mode):
                    continue
            # weighted fusion: reciprocal-rank-style blend, embeddings weighted for recall, lexical for precision
            score = self.hybrid_lexical_w * min(1.0, lsc) + self.hybrid_embed_w * cos
            out.append((sk, score))
        out.sort(key=lambda x: -x[1])
        # relative-margin prune, same discipline as lexical
        if out and len(out) > 1:
            best = out[0][1]
            out = [(sk, sc) for sk, sc in out if sc >= best * self.recall_margin]
        return out[:limit]

    def _embed_candidates(self, task, statuses):
        """Skills whose embedding is semantically near the query, independent of lexical overlap.
        Returns [] when no embedding model is configured (pure-lexical fallback, unchanged behavior)."""
        emb_model = getattr(self._llm, "embed_model", None) if self._llm is not None else None
        if not emb_model:
            return []
        try:
            qv = self.llm.embed([task])
        except Exception:
            qv = None
        if not qv:
            return []
        q = qv[0]
        out = []
        for sk in self.store.all_skills():
            if sk.status not in statuses or not sk.embedding:
                continue
            cos = _cosine(q, sk.embedding)
            if cos >= self.embed_threshold:
                out.append((sk, cos))
        out.sort(key=lambda x: -x[1])
        return out[: self.embed_topk]

    # ------------------------------------------------------------------ intent
    # Procedural skills answer TASKS: something is broken, or something must be done. They must not answer
    # knowledge questions ("what is the default max_connections"), creative requests ("poem about docker") or,
    # unless the skill is about authoring, requests to write something new ("write a Dockerfile").
    #
    # The previous filter rejected every sentence starting with a question word. Agents are routinely handed
    # tasks phrased as questions ("how do I migrate the schema", "why does my container exit"), so the most
    # natural phrasings of real work never received their skills (held-out eval: 2/14). Intent is now decided
    # by what the sentence ASKS FOR, not how it starts:
    #   - an error identifier or problem word  -> task (someone is stuck; this is exactly what skills are for)
    #   - "how do/can I ...", "what should I do" -> task (asking for a procedure)
    #   - "why does/is/do ..."                  -> task (diagnosing observed behaviour)
    #   - other questions                       -> knowledge (asking for a fact)
    #   - "write/create/design a new X"         -> create (only authoring-scoped skills may fire)
    _CREATIVE = re.compile(r"\b(poem|haiku|song|limerick|story|essay|joke)\b|\blogo\b|to a (five|5).year.old"
                           r"|^\s*translate\b", re.I)
    _ERR_ID = re.compile(r"\b[A-Z][A-Za-z]*(Error|Exception)\b|\bFATAL\b|\bCONFLICT\b|\bTraceback\b"
                         r"|\bE[A-Z]{3,}\b|\bexit(ed)? (code|status) ?[1-9]")
    _PROBLEM = re.compile(
        r"\b(errors?|fail\w*|broken|breaks?|crash\w*|exceptions?|conflicts?|stuck|hang(s|ing)?|freez\w*|"
        r"slow\w*|wrong|missing|lost|refus\w*|won'?t|can'?t|cannot|doesn'?t|does not|isn'?t|not working|"
        r"no longer|stopped|timed? ?out|timeout|denied|unchanged|too many|flaky|fix(es|ed|ing)?|repair\w*|debug\w*|"
        r"troubleshoot\w*|recover\w*|restor\w*|undo|by accident|accidentally|even though)\b", re.I)
    _PROCEDURAL = re.compile(
        r"^\s*(how (do|can|should|would|could|to) (i|we)?|how to\b|what (do|should|can) (i|we) do"
        r"|what('?s| is) the (best |right |proper )?way to|is there a way to|help me)", re.I)
    _WHY = re.compile(r"^\s*why\b", re.I)
    _QUESTION = re.compile(r"^\s*(what|who|when|where|which|how|is|are|does|do|can|should|could|would)\b"
                           r"|^\s*(recommend|suggest)\b|\bopinion\b|\?\s*$", re.I)
    _EXPLAIN = re.compile(r"^\s*explain (how|why|what|the|me|to|this|that|a|an|in|when|where|it)\b", re.I)
    _CREATE = re.compile(r"^\s*(please )?(write|create|design|generate|scaffold|draft|compose|bootstrap"
                         r"|build (me )?(a|an)\b|start (a|an) new)\b", re.I)
    _PROC_CREATE = re.compile(r"^\s*(how (do|can|should|would|could) (i|we)|how to|what('?s| is) the (best |right )?"
                              r"way to|help me)\s+(write|create|design|generate|scaffold|draft|set up a new|start a new"
                              r"|make a new|add a new)\b", re.I)
    _AUTHORING = re.compile(r"\b(writ\w*|creat\w*|new|scaffold\w*|author\w*|generat\w*|template|scratch"
                            r"|bootstrap\w*|design\w*)\b", re.I)

    def _intent(self, task: str) -> str:
        """'task' | 'create' | 'knowledge' | 'creative'. Deterministic, no model call."""
        t = task or ""
        if self._CREATIVE.search(t):
            return "creative"
        if self._ERR_ID.search(t) or self._PROBLEM.search(t):
            return "task"
        if self._EXPLAIN.search(t):
            return "knowledge"
        if self._CREATE.search(t) or self._PROC_CREATE.search(t):
            return "create"
        if self._PROCEDURAL.search(t) or self._WHY.search(t):
            return "task"
        if self._QUESTION.search(t):
            return "knowledge"
        return "task"          # imperative ("migrate the postgres schema")

    def _looks_like_task(self, task: str) -> bool:
        """Backwards-compatible: True unless the request is a knowledge question or creative."""
        return self._intent(task) in ("task", "create")

    def _authoring_scoped(self, sk) -> bool:
        """A skill about WRITING something (name / applies_when say so) may fire on a creation request.
        A repair lesson ("docker build reinstalls every time") must not fire on "write a Dockerfile"."""
        fac = sk.facets or {}
        return bool(self._AUTHORING.search(sk.name.replace("-", " ") + " " + " ".join(fac.get("applies_when", []))))

    def _passes_precision_gate(self, task, sk, cos: float = 0.0, mode: str = "task") -> bool:
        """The precision discipline applied to embedding-only candidates: the query must touch what the skill
        IS (anchor overlap) OR match a trigger query, and must not hit a not_for term. Without this, a
        semantically-adjacent skill would false-fire on topically-related but wrong-intent queries."""
        # an embedding-only candidate answering a non-task (question/creative) request is a false fire
        if mode != "error" and not self._looks_like_task(task):
            self.store.log("recall_filtered", sk.name, f"embed candidate, non-task query :: {task[:50]}")
            return False
        qt = self._terms(task)
        _, _, anchors = self._profile_of(sk)
        qsim = self._best_query_match(self._query_terms(task), sk)
        # A paraphrase shares no words with the skill BY DEFINITION, so demanding anchor overlap made the
        # embedding path unable to rescue exactly the queries it exists for. A strong semantic match is
        # accepted on its own; weaker ones still need an anchor. not_for is still enforced below.
        strong = cos >= self.embed_strong
        if not strong and qsim < self.query_threshold and anchors and not (qt & anchors):
            self.store.log("recall_filtered", sk.name, f"embed candidate, no anchor :: {task[:50]}")
            return False
        if self._excluded_by_not_for(sk, qt, anchors):
            return False
        return True

    def _excluded_by_not_for(self, sk, qt: set[str], anchors: set[str], strict_phrase: bool = False,
                             verbatim_text: str | None = None) -> bool:
        """not_for says what a skill is NOT for. Normally one shared word excludes (cheap, precise on user
        phrasing). When the text already matches one of the skill's recorded symptoms verbatim, that is direct
        evidence FOR the skill, and a single word must not veto it: a rebase error literally says "Merge
        conflict", which would otherwise hit not_for "merge conflicts on a merge". Then a not_for entry only
        excludes if most of its phrase is present."""
        not_for = (sk.facets or {}).get("not_for", [])
        if not not_for:
            return False
        if verbatim_text is not None:
            # a recorded symptom appears verbatim: literal evidence can only be vetoed by literal evidence
            for phrase in not_for:
                if phrase and phrase.lower() in verbatim_text:
                    self.store.log("recall_filtered", sk.name, f"excluded by verbatim not_for '{phrase}'")
                    return True
            return False
        if not strict_phrase:
            neg = self._terms(" ".join(not_for)) - anchors
            if neg & qt:
                self.store.log("recall_filtered", sk.name, f"excluded by not_for {sorted(neg & qt)[:3]}")
                return True
            return False
        for phrase in not_for:
            pt = self._terms(phrase) - anchors
            if pt and len(pt & qt) / len(pt) >= 0.6 and len(pt & qt) >= 2:
                self.store.log("recall_filtered", sk.name, f"excluded by not_for phrase '{phrase}'")
                return True
        return False

    def _profile_of(self, sk):
        """Single-skill profile (weighted terms + anchors), reusing the same logic as _profiles."""
        for s2, prof, anch in self._profiles((sk.status,)):
            if s2.name == sk.name:
                return s2, prof, anch
        return sk, {}, set()
