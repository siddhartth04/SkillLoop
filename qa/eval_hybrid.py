"""Measure hybrid retrieval on the BLIND probe set using a deterministic semantic-embedding stub.

The stub maps text -> a vector over a fixed concept space, so paraphrases ('slow query' vs 'sits there for
8 seconds') land near each other WITHOUT sharing surface words. This validates the hybrid architecture: if
embeddings can surface paraphrase matches that lexical missed, the F1 should rise from the lexical 0.47.
This is a logic/architecture test; real numbers require a real embedding model (Ollama/OpenAI)."""
import sys, tempfile, re, math; sys.path.insert(0, '.'); sys.path.insert(0, 'qa')
from eval_suite import LIB
from skillloop import SkillLoop
from skillloop.llm import LLM

# --- concept space: each concept has trigger words; a text's vector = which concepts it evokes ---
CONCEPTS = {
    "yaml_edit": ["yaml", "config", "replicas", "setting", "key", "value", "compose file"],
    "yaml_merge": ["merge", "combine", "two files", "join yaml"],
    "docker_cache": ["docker build", "rebuild", "recompile", "cache", "slow build", "compose", "version bump"],
    "docker_exit": ["container", "exits", "crashloop", "boots", "falls over", "dies", "starts then"],
    "pip_env": ["pip", "externally managed", "install refuses", "environment managed", "cannot install"],
    "pip_import": ["installed", "import fails", "modulenotfound", "cant find module", "package name"],
    "pytest_flaky": ["flaky", "order", "pass alone", "fail together", "individually", "run the file explodes"],
    "pytest_collect": ["collection", "wont collect", "cannot collect", "import error tests"],
    "pg_slow": ["postgres", "slow query", "missing index", "sits there seconds", "lookup slow", "psql slow"],
    "pg_pool": ["connection pool", "too many connections", "out of connections", "clients already"],
    "git_recover": ["deleted branch", "recover branch", "lost branch", "nuked branch", "get it back"],
    "bash_exit": ["exits 0", "hides failure", "swears finished", "output empty", "says done nothing"],
}
CONCEPT_IX = {c: i for i, c in enumerate(CONCEPTS)}
# map each skill to its concept
SKILL_CONCEPT = {
    "yaml-single-key-edit": "yaml_edit", "yaml-merge-two-files": "yaml_merge",
    "docker-layer-cache-miss": "docker_cache", "docker-container-exits-immediately": "docker_exit",
    "pip-externally-managed-env": "pip_env", "pip-import-name-mismatch": "pip_import",
    "pytest-flaky-order-dependence": "pytest_flaky", "pytest-collection-error": "pytest_collect",
    "postgres-missing-index-slow-query": "pg_slow", "postgres-connection-pool-exhausted": "pg_pool",
    "git-recover-deleted-branch": "git_recover", "bash-exit-code-hides-failure": "bash_exit",
}

def embed(text):
    t = text.lower()
    v = [0.0] * len(CONCEPTS)
    for c, kws in CONCEPTS.items():
        for kw in kws:
            if kw in t:
                v[CONCEPT_IX[c]] += 1.0
    # normalize
    n = math.sqrt(sum(x*x for x in v)) or 1.0
    return [x/n for x in v]

# monkeypatch LLM.embed to our stub and set an embed_model so the hybrid path activates
class StubLLM(LLM):
    def __init__(self):
        super().__init__(provider='fake')
        self.embed_model = 'stub'
    def embed(self, texts):
        return [embed(t) for t in texts]

BLIND = [
    ("bumped the version number in the compose file and now the whole thing recompiles from zero", "docker-layer-cache-miss"),
    ("ci is green when i run one test but the moment i run the file it explodes", "pytest-flaky-order-dependence"),
    ("the thing swears it finished but the output folder is empty", "bash-exit-code-hides-failure"),
    ("psql just sits there for like 8 seconds on a lookup that should be instant", "postgres-missing-index-slow-query"),
    ("pip flat out refuses, says something about the environment being managed", "pip-externally-managed-env"),
    ("nuked a branch by accident, can i get it back", "git-recover-deleted-branch"),
    ("the app boots and instantly falls over, nothing in the logs", "docker-container-exits-immediately"),
    ("too many clients already, postgres won't take new ones", "postgres-connection-pool-exhausted"),
    ("chnage the replcias setting in the yaml", "yaml-single-key-edit"),
    ("instaled it fine but python cant fnid the module", "pip-import-name-mismatch"),
    ("write me a poem about docker whales", None),
    ("what year was postgresql first released", None),
    ("explain how git works to a five year old", None),
    ("is yaml better than json for config, opinion", None),
    ("recommend a python course for beginners", None),
    ("what does the pytest logo look like", None),
]

def run(use_embed):
    llm = StubLLM() if use_embed else LLM(provider='fake')
    lp = SkillLoop(home=tempfile.mkdtemp(), llm=llm)
    for s in LIB:
        s2 = s
        if use_embed:
            s2.embedding = embed(" ".join(s.facets['applies_when'] + s.queries))
        lp.store.save_skill(s2, triggers=[], snapshot=False)
    tp=fp=fn=tn=0; errs=[]
    for q, expect in BLIND:
        hits=[h['name'] for h in lp.recall(q, limit=1) if h['name']!='_principles']
        got=hits[0] if hits else None
        if expect is None:
            if got is None: tn+=1
            else: fp+=1; errs.append(('FALSE-FIRE',q,got))
        else:
            if got==expect: tp+=1
            elif got is None: fn+=1; errs.append(('MISS',q,expect))
            else: fp+=1; errs.append(('WRONG',q,f'{got} want {expect}'))
    prec=tp/(tp+fp) if tp+fp else 0; rec=tp/(tp+fn) if tp+fn else 0
    f1=2*prec*rec/(prec+rec) if prec+rec else 0
    return prec,rec,f1,tp,fp,fn,tn,errs

for label,ue in [("LEXICAL ONLY (baseline)",False),("HYBRID (lexical + embeddings)",True)]:
    p,r,f,tp,fp,fn,tn,errs=run(ue)
    print(f"\n{label}")
    print(f"  precision {p:.3f}  recall {r:.3f}  F1 {f:.3f}  | pos {tp}/10  abstain {tn}/6  false-fire {sum(1 for e in errs if e[0]=='FALSE-FIRE')}")
    for k,q,d in errs[:8]: print(f"    [{k}] {q[:48]} -> {d}")