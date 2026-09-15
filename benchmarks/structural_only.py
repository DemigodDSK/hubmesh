"""Structural-only ablation arm: SAME graph, SAME seeds, SAME readout,
SAME post-selection as hubmesh — cosine removed.

`hippo_style` (entity-only subgraph, entity-mean readout, different
fallback) is a *different pipeline*, so `hubmesh − hippo_style` mixes
the scoring contribution with substrate differences (external review).
This arm isolates scoring alone: it ranks documents purely by hubmesh's
own pooled-PPR mass on doc nodes, using the Planner's exact seed
selection (NER → alias index, top-3-doc-entity fallback) AND the
Planner's exact post-selection (same packer, same token budget, same
near-duplicate filter, same document cap). Any gap between this arm and
hubmesh is attributable to the score, nothing else.

Seedless queries (no entity match even after the fallback) carry no
structural signal. They are handled the way `composite_score` handles a
constant component — every document gets the neutral score, ties break
by document id — and then go through the same packer, so the arm never
substitutes a different empty-result policy for the Planner's
(external review, 2026-09-14). Such queries are counted in
`stats["seedless"]` and reported next to the result; on the recorded
HotpotQA/MuSiQue runs that count is zero.
"""
from __future__ import annotations
import numpy as np

from hubmesh.kg import EntityKG, extract_query_entities
from hubmesh.packing import pack, estimate_tokens
from hubmesh.ppr import PPRSolver
from hubmesh.types import ScoredDocument


def structural_only_retrieve(
    kg: EntityKG,
    solver: PPRSolver,
    nlp,
    query_text: str,
    query_vec: np.ndarray,
    store,
    top_k: int,
    alpha: float = 0.15,
    budget_tokens: int = 10_000,
    redundancy_lambda: float = 0.3,
    stats: dict | None = None,
) -> list[str]:
    mentions = extract_query_entities(query_text, nlp=nlp)
    seeds = list(dict.fromkeys(kg.query_entity_nodes(mentions)))
    fallback = False
    if not seeds:   # production fallback, verbatim
        fallback = True
        for d, _ in store.search(query_vec, top_k=3):
            ents = kg.doc_to_entities.get(d, set())
            seeds.extend(e for e in sorted(ents) if e in kg.graph)
        seeds = list(dict.fromkeys(seeds))[:8]
    if stats is not None:
        stats["queries"] = stats.get("queries", 0) + 1
        stats["fallback"] = stats.get("fallback", 0) + int(fallback)
        stats["seedless"] = stats.get("seedless", 0) + int(not seeds)
        stats.setdefault("n_seeds", []).append(len(seeds))

    # No seeds -> no diffusion -> every document scores 0.0 here and 0.5
    # after min-max (the neutral value composite_score assigns a constant
    # component); the id tie rule and the packer below still apply.
    mass = solver.solve(seeds, alpha=alpha) if seeds else {}
    docs = sorted(((n[4:], float(mass.get(n, 0.0)))
                   for n in kg.graph.nodes if n.startswith("doc:")),
                  key=lambda kv: (-kv[1], kv[0]))
    if not docs:
        return []
    # Same scale the Planner's composite has ([0, 1] after min-max) so
    # the packer's near-duplicate rule behaves identically.
    vals = np.array([s for _, s in docs], dtype=float)
    lo, hi = float(vals.min()), float(vals.max())
    norm = (vals - lo) / (hi - lo) if hi - lo > 1e-9 else np.full_like(vals, 0.5)

    cand = [(d, float(norm[i])) for i, (d, _) in enumerate(docs[:top_k * 5])]
    fetched = {doc.id: doc for doc in store.get_many([d for d, _ in cand])}
    scored = [ScoredDocument(doc=fetched[d], similarity=0.0,
                             ppr_score=float(mass.get(f"doc:{d}", 0.0)),
                             composite_score=s, rank=r)
              for r, (d, s) in enumerate(cand) if d in fetched]
    _, picked = pack(scored, budget_tokens=budget_tokens,
                     redundancy_lambda=redundancy_lambda,
                     vec_of=store.vector_of, max_docs=top_k,
                     count_tokens=estimate_tokens)
    return [p.doc.id for p in picked[:top_k]]
