"""Cross-PROCESS determinism (external review): same-process repeats are
not evidence — hash-seed-dependent set iteration reorders equal scores
and even changes fallback seed choice. This runs the same retrievals in
fresh interpreters under different PYTHONHASHSEED values and demands
byte-identical ranked output, in both kNN mode and KG mode, with
engineered exact ties and a forced fallback-seed path.
"""
import os
import subprocess
import sys
import textwrap

import pytest

SCRIPT = textwrap.dedent("""
    import json, numpy as np, networkx as nx
    from hubmesh import Planner, Document
    from hubmesh.adapters import InMemoryStore
    from hubmesh.kg import EntityKG

    def vec(seed, dim=16):
        rng = np.random.default_rng(seed); v = rng.normal(size=dim)
        return (v / np.linalg.norm(v)).astype(np.float32)

    # ---- kNN mode: three docs share one vector -> exact cosine ties
    shared = vec(7)
    docs = [Document(id=i, text="tie " + i, vector=shared) for i in ("m", "z", "a")]
    docs += [Document(id=str(k), text=f"d{k}", vector=vec(k)) for k in range(10, 22)]
    planner = Planner(store=InMemoryStore(docs, k=4))
    r = planner.retrieve(shared, top_k=8, budget_tokens=100000)
    out = {"knn": [s.doc.id for s in r.sources]}

    # ---- KG mode: hand-built KG, exact ties, plus a no-entity query that
    # forces the fallback path through top-3 docs' entity SETS
    class Ent:
        def __init__(self, t): self.text, self.label_ = t, "PERSON"
    class FakeDoc:
        def __init__(self, ents): self.ents = ents
    class FakeNLP:
        def __call__(self, text):
            toks = [w for w in text.lower().split() if w.startswith("alice")]
            return FakeDoc([Ent("alice")] if toks else [])

    kdocs = [Document(id=f"t{i}", text="tied", vector=shared) for i in range(3)]
    kdocs += [Document(id=f"h{i}", text=f"hub {i}", vector=vec(100 + i)) for i in range(4)]
    G = nx.Graph()
    d2e, e2d, c2n, n2l = {}, {}, {}, {}
    for d in kdocs:
        G.add_node("doc:" + d.id, kind="doc"); d2e[d.id] = set()
    def link(doc_id, ent):
        node = "ent:" + ent
        G.add_node(node, kind="entity", canonical=ent, label=ent)
        G.add_edge("doc:" + doc_id, node, kind="mentions")
        d2e[doc_id].add(node); e2d.setdefault(node, set()).add(doc_id)
        c2n[ent] = node; n2l[node] = ent
    for d in kdocs[:3]:                      # symmetric: identical structure
        link(d.id, "alice")
    ents = [f"e{j}" for j in range(9)]
    for i, d in enumerate(kdocs[3:]):        # hubs carry >4 entities each so
        for e in ents[i:i + 6]:              # fallback seeds[:4] depend on order
            link(d.id, e)
    kg = EntityKG(graph=G, doc_to_entities=d2e, entity_to_docs=e2d,
                  entity_canonical_to_node=c2n, entity_node_to_label=n2l,
                  alias_to_node=dict(c2n))
    kp = Planner(store=InMemoryStore(kdocs, k=3), kg=kg, nlp=FakeNLP())
    r1 = kp.retrieve("who is alice", query_vec=shared, top_k=7, budget_tokens=100000)
    r2 = kp.retrieve("zzz nothing", query_vec=vec(100), top_k=7, budget_tokens=100000)
    out["kg_entity"] = [s.doc.id for s in r1.sources]
    out["kg_fallback"] = [s.doc.id for s in r2.sources]
    out["kg_fallback_seeds"] = r2.debug.get("ppr_seeds") if r2.debug else None
    print(json.dumps(out))
""")


def run_with_hashseed(seed: int, script: str = SCRIPT) -> str:
    env = {**os.environ, "PYTHONHASHSEED": str(seed)}
    proc = subprocess.run([sys.executable, "-c", script], env=env,
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr[-2000:]
    return proc.stdout.strip().splitlines()[-1]


@pytest.mark.parametrize("seeds", [(1, 2), (2, 3), (1, 3)])
def test_ranked_output_identical_across_hash_seeds(seeds):
    a, b = (run_with_hashseed(s) for s in seeds)
    assert a == b, f"PYTHONHASHSEED {seeds[0]} vs {seeds[1]}:\n{a}\n{b}"


# ---- 2026-09-14 review: two paths the first pass did not cover ----------
# (a) capped BFS expansion in kNN mode — which nodes survive the cap
#     depended on set iteration order (hash seeds 1-3 gave a,b,d; 4-5
#     gave a,b,c): candidate MEMBERSHIP changed before any ranking;
# (b) the explicit SubstringLinker sorted equal-length forms by set
#     order, so "lee" mapped to "bob lee" or "ann lee" per process.
SCRIPT_BOUNDARY = textwrap.dedent("""
    import json
    from hubmesh.graph import build_induced_subgraph
    from hubmesh.entity_linker import SubstringLinker

    class Store:
        NB = {'a': ['c'], 'b': ['d'], 'c': ['a', 'g'], 'd': ['b', 'h'],
              'e': ['f'], 'f': ['e'], 'g': ['c'], 'h': ['d']}
        def neighbors(self, n, k): return self.NB[n]

    out = {}
    out["cap3_h1"] = sorted(build_induced_subgraph(Store(), ['a', 'b'], hops=1, cap=3).nodes)
    out["cap5_h2"] = sorted(build_induced_subgraph(Store(), ['b', 'a'], hops=2, cap=5).nodes)
    out["cap_below_seeds"] = sorted(build_induced_subgraph(Store(), ['b', 'a', 'e'], hops=1, cap=2).nodes)
    out["edges_cap5_h2"] = sorted(map(sorted, build_induced_subgraph(Store(), ['b', 'a'], hops=2, cap=5).edges))
    out["linker"] = SubstringLinker().link(["ann lee", "bob lee", "lee"])["lee"]
    out["linker_set_input"] = SubstringLinker().link({"ann lee", "bob lee", "al lee", "lee"})["lee"]
    print(json.dumps(out, sort_keys=True))
""")


@pytest.mark.parametrize("seeds", [(1, 4), (2, 5), (3, 4), (1, 5)])
def test_capped_expansion_and_linker_identical_across_hash_seeds(seeds):
    a, b = (run_with_hashseed(s, SCRIPT_BOUNDARY) for s in seeds)
    assert a == b, f"PYTHONHASHSEED {seeds[0]} vs {seeds[1]}:\n{a}\n{b}"


def test_capped_expansion_tie_rule_is_documented_order():
    """The rule, not just the invariance: frontier nodes are visited by
    id, so with seeds a,b and cap 3 the survivor is a's neighbour."""
    import json
    out = json.loads(run_with_hashseed(1, SCRIPT_BOUNDARY))
    assert out["cap3_h1"] == ["a", "b", "c"]
    assert out["cap_below_seeds"] == ["a", "b", "e"]     # seeds always kept
    assert out["linker"] == "ann lee"                     # lexical tie-break
