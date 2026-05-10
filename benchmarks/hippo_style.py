"""HippoRAG-style retrieval baseline.

Uses the SAME entity-linked KG as hubmesh — but ranks documents using the
HippoRAG paper's formula instead of hubmesh's multi-component score:

  • PPR runs over the entity-only subgraph (entity nodes only, no doc nodes)
  • For each candidate document, score = mean of PPR scores of the
    entities mentioned in the document

This is the ablation that isolates hubmesh's *scoring* contribution from
its *KG-as-substrate* contribution. If hubmesh beats hippo_style, the
multi-component pattern adds value. If they tie, the win is purely from
having a KG at all (and HippoRAG would already get it for free).
"""
from __future__ import annotations
import numpy as np
import networkx as nx

from hubmesh.kg import EntityKG, extract_query_entities
from hubmesh.ppr import personalized_pagerank


def hippo_style_retrieve(
    kg: EntityKG,
    nlp,
    query_text: str,
    query_vec: np.ndarray,
    store,
    top_k: int,
    alpha: float = 0.15,
) -> list[str]:
    """HippoRAG-style document ranking.

    1. Extract entities from query, match to KG entity nodes
    2. PPR over entity-only subgraph from those seeds
    3. Score each doc = mean PPR over its entities
    4. Return top_k doc_ids
    """
    # Entity subgraph (no document nodes)
    entity_nodes = [n for n in kg.graph.nodes if n.startswith("ent:")]
    G_ent = kg.graph.subgraph(entity_nodes).copy()

    # Seed entities from query
    q_mentions = extract_query_entities(query_text, nlp=nlp)
    seeds = kg.query_entity_nodes(q_mentions)
    seeds = [s for s in seeds if s in G_ent]

    if not seeds:
        # Fallback: HippoRAG's own fallback is also dense retrieval (DPR/Contriever).
        # Return cosine top-k as a graceful degradation.
        return [doc_id for doc_id, _ in store.search(query_vec, top_k=top_k)]

    ppr = personalized_pagerank(G_ent, seeds, alpha=alpha)

    # Score each doc by mean PPR of entities it contains
    doc_scores: dict[str, float] = {}
    for doc_id, ent_set in kg.doc_to_entities.items():
        if not ent_set:
            continue
        scores = [ppr.get(e, 0.0) for e in ent_set]
        doc_scores[doc_id] = float(np.mean(scores))

    # Tie-breaker: cosine similarity, so docs with no entity coverage still get
    # placed somewhere reasonable (matches HippoRAG's hybrid mode in spirit).
    if doc_scores:
        max_ppr = max(doc_scores.values()) or 1.0
    else:
        max_ppr = 1.0
    qn = query_vec / max(float(np.linalg.norm(query_vec)), 1e-12)

    # For docs that don't appear in doc_scores at all (no extracted entities),
    # they currently get nothing. Pull cosine top-k as a fallback to fill.
    ordered = sorted(doc_scores.items(), key=lambda kv: -kv[1])
    out = [doc_id for doc_id, _ in ordered[:top_k]]
    if len(out) < top_k:
        existing = set(out)
        for doc_id, _ in store.search(query_vec, top_k=top_k * 3):
            if doc_id not in existing:
                out.append(doc_id)
                existing.add(doc_id)
                if len(out) >= top_k:
                    break
    return out[:top_k]
