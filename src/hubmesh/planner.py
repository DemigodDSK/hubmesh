"""The main Planner — entry point for hubmesh."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np

from .adapters.base import VectorStore
from .types import Document, ScoredDocument, RetrievalResult, ReasoningPath
from .graph import build_induced_subgraph, detect_communities, best_community_for_query
from .ppr import personalized_pagerank
from .scoring import (
    ScoringWeights, compute_relevance, compute_coherence, composite_score,
)
from .packing import pack


@dataclass
class PlannerConfig:
    seed_top_k: int = 10           # first-pass ANN candidate pool
    n_seeds_for_ppr: int = 5       # how many to feed as PPR teleport sources
    subgraph_hops: int = 2         # induced subgraph radius
    subgraph_cap: int = 2000       # hard cap on subgraph size (latency invariant)
    ppr_alpha: float = 0.15        # teleport probability
    redundancy_lambda: float = 0.3 # MMR diversity vs. score tradeoff
    weights: ScoringWeights = None # set in __post_init__

    # Anchoring is robust for single-community retrieval but actively harmful
    # for multi-hop QA (where the answer spans multiple communities). Default
    # off; turn on for KG-style topical retrieval.
    use_community_anchor: bool = False
    use_coherence: bool = False    # ditto — rewards staying in one community

    # Integration mode. 'sum' (default) is forgiving — a high relevance
    # compensates for low PPR. 'geom' enforces "must be strong on every
    # component" (the original NNSI flavour, useful when all components
    # positively correlate with the answer).
    integration: str = "sum"

    def __post_init__(self):
        if self.weights is None:
            # Heavily relevance-biased default — cosine sim is highly
            # informative for QA. Structural acts as a tail-recovery boost
            # for multi-hop paragraphs whose cosine is low. Coherence off
            # by default (off-topic for multi-hop).
            self.weights = ScoringWeights(relevance=3.0, structural=1.0, coherence=0.0)


def _vec_of_factory(store: VectorStore) -> Callable[[str], np.ndarray]:
    """The store protocol doesn't expose vectors directly; concrete stores
    do. For the in-memory adapter we have `vector_of`; for others we'd cache
    embeddings on first lookup. Falls back to a NotImplementedError otherwise."""
    if hasattr(store, "vector_of"):
        return store.vector_of  # type: ignore[attr-defined]
    raise NotImplementedError(
        "This adapter doesn't expose raw vectors. Wrap it or extend the "
        "VectorStore protocol with `vector_of(doc_id) -> np.ndarray`."
    )


class Planner:
    """Centrality-aware GraphRAG retrieval planner.

    Pipeline:
      1. First-pass ANN  -> seed candidate doc_ids
      2. Build induced 2-hop subgraph
      3. Detect communities, anchor to the one closest to query
      4. Personalized PageRank from seeds restricted to anchored community
      5. Score every node by  relevance × structural × coherence (geom mean)
      6. Pack into context budget with redundancy control
    """

    def __init__(
        self,
        store: VectorStore,
        embed: Callable[[str], np.ndarray] | None = None,
        config: PlannerConfig | None = None,
    ):
        self.store = store
        self.embed = embed
        self.config = config or PlannerConfig()
        self._vec_of = _vec_of_factory(store)

    def retrieve(
        self,
        query: str | np.ndarray,
        top_k: int = 10,
        budget_tokens: int = 4000,
    ) -> RetrievalResult:
        # 0. embed if needed
        if isinstance(query, np.ndarray):
            qvec = query
            qtext = ""
        else:
            if self.embed is None:
                raise ValueError("Pass embed=callable to Planner or supply a query vector")
            qvec = self.embed(query)
            qtext = query

        # 1. first-pass ANN
        seeds_with_sim = self.store.search(qvec, top_k=self.config.seed_top_k)
        seed_ids = [s for s, _ in seeds_with_sim]
        seed_sim = dict(seeds_with_sim)

        # 2. induced subgraph
        G = build_induced_subgraph(
            self.store, seed_ids,
            hops=self.config.subgraph_hops,
            cap=self.config.subgraph_cap,
        )
        if G.number_of_nodes() == 0:
            return RetrievalResult(query=qtext, context="", sources=[], reasoning=[])

        # 3. (optional) community anchoring — useful for single-topic
        #    retrieval, harmful for multi-hop. Disabled by default.
        if self.config.use_community_anchor or self.config.use_coherence:
            communities = detect_communities(G)
            anchor = best_community_for_query(qvec, G, communities, self._vec_of)
        else:
            communities = {}
            anchor = set(G.nodes)   # "no anchor" = whole subgraph

        # 4. PPR with seed teleport. Use first-pass ANN seeds directly — for
        #    multi-hop the right move is to let diffusion reach far hops, not
        #    confine teleport to a single community.
        ppr_seeds = [s for s in seed_ids[: self.config.n_seeds_for_ppr] if s in G]
        ppr_scores = personalized_pagerank(
            G, ppr_seeds, alpha=self.config.ppr_alpha,
        )

        # 5. multi-component scoring (coherence is dropped when off — the
        #    composite collapses to relevance × structural geometric mean).
        relevance = compute_relevance(G, qvec, self._vec_of)
        if self.config.use_coherence and communities:
            coherence = compute_coherence(G, communities, anchor)
        else:
            coherence = {n: 1.0 for n in G.nodes}   # neutral element
        composite = composite_score(
            relevance=relevance,
            structural=ppr_scores,
            coherence=coherence,
            weights=self.config.weights,
            integration=self.config.integration,
        )

        # Pull docs and rank
        ordered = sorted(composite.items(), key=lambda kv: -kv[1])
        scored: list[ScoredDocument] = []
        for rank, (nid, score) in enumerate(ordered):
            try:
                doc = self.store.get(nid)
            except KeyError:
                continue
            scored.append(ScoredDocument(
                doc=doc,
                similarity=float(seed_sim.get(nid, relevance.get(nid, 0.0))),
                ppr_score=float(ppr_scores.get(nid, 0.0)),
                composite_score=float(score),
                rank=rank,
            ))

        # 6. pack into budget
        context, picked = pack(
            scored[:top_k * 5],            # consider top 5×k for packing
            budget_tokens=budget_tokens,
            redundancy_lambda=self.config.redundancy_lambda,
            vec_of=self._vec_of,
        )

        return RetrievalResult(
            query=qtext,
            context=context,
            sources=picked[:top_k],
            reasoning=[],   # paths come in a later iteration
            debug={
                "subgraph_nodes": G.number_of_nodes(),
                "subgraph_edges": G.number_of_edges(),
                "communities": len(set(communities.values())) if communities else 0,
                "anchor_size": len(anchor),
                "ppr_seeds": ppr_seeds,
            },
        )
