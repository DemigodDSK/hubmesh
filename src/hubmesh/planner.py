"""The main Planner — entry point for hubmesh."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np

from .adapters.base import VectorStore
from .types import Document, ScoredDocument, RetrievalResult, ReasoningPath
from .graph import build_induced_subgraph, detect_communities, best_community_for_query
from .ppr import personalized_pagerank, PPRSolver
from .scoring import (
    ScoringWeights, compute_relevance, compute_coherence, composite_score,
)
from .packing import pack
from .kg import EntityKG, extract_query_entities
from .paths import build_reasoning_paths


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

    Two modes:

      kg_mode (when an EntityKG is provided): the production / multi-hop path.
        1. Extract entities from query
        2. Match to KG entity nodes (teleport seeds)
        3. PPR over the bipartite KG
        4. Doc score = relevance × structural (PPR at doc node)
        5. Pack with budget+redundancy

      knn_mode (default): the prototyping / non-KG path.
        1. First-pass ANN → seed doc_ids
        2. Induced subgraph on the kNN proximity graph
        3. (optional) community anchoring
        4. PPR from seeds → score by relevance × structural × coherence
        5. Pack
    """

    def __init__(
        self,
        store: VectorStore,
        embed: Callable[[str], np.ndarray] | None = None,
        config: PlannerConfig | None = None,
        kg: EntityKG | None = None,
        nlp=None,
    ):
        self.store = store
        self.embed = embed
        self.config = config or PlannerConfig()
        self._vec_of = _vec_of_factory(store)
        self.kg = kg
        self._nlp = nlp   # spaCy pipeline for query-side NER (loaded lazily)
        # Pre-compute the sparse PPR transition matrix once. Per-query PPR
        # then becomes ~10-20× faster than rebuilding the sparse matrix on
        # every call (which nx.pagerank does).
        self._ppr_solver: PPRSolver | None = (
            PPRSolver(kg.graph, weight_attr="weight") if kg is not None else None
        )

    def retrieve(
        self,
        query: str | np.ndarray,
        top_k: int = 10,
        budget_tokens: int = 4000,
        query_vec: np.ndarray | None = None,
    ) -> RetrievalResult:
        """Retrieve top_k documents.

        Pass `query` as a string (preferred — required for KG mode's NER).
        For batched benchmarking where embeddings are precomputed, pass the
        text as `query` and the precomputed vector as `query_vec` to skip
        re-embedding.
        """
        # 0. resolve text + vector
        if isinstance(query, np.ndarray):
            qvec = query
            qtext = ""
        else:
            qtext = query
            if query_vec is not None:
                qvec = query_vec
            else:
                if self.embed is None:
                    raise ValueError("Pass embed= to Planner, supply query_vec, "
                                     "or pass a vector as `query`")
                qvec = self.embed(query)

        # Route to KG-mode if a knowledge graph is attached, else kNN mode.
        if self.kg is not None and qtext:
            return self._retrieve_kg(qtext, qvec, top_k, budget_tokens)

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
                "mode": "knn",
                "subgraph_nodes": G.number_of_nodes(),
                "subgraph_edges": G.number_of_edges(),
                "communities": len(set(communities.values())) if communities else 0,
                "anchor_size": len(anchor),
                "ppr_seeds": ppr_seeds,
            },
        )

    # ------------------------------------------------------------------
    # KG-mode retrieval — the multi-hop path (HippoRAG-style + multi-component)
    # ------------------------------------------------------------------

    def _retrieve_kg(
        self, qtext: str, qvec: np.ndarray,
        top_k: int, budget_tokens: int,
    ) -> RetrievalResult:
        """Retrieval that uses the entity-linked KG as the routing graph
        instead of a kNN proximity graph."""
        kg = self.kg
        if self._nlp is None:
            import spacy
            self._nlp = spacy.load("en_core_web_sm")

        # 1. extract query entities, match to KG nodes
        q_mentions = extract_query_entities(qtext, nlp=self._nlp)
        ppr_seeds = kg.query_entity_nodes(q_mentions)

        # If query has no extractable entities OR none match the KG, fall back
        # to entities mentioned by the top cosine-similar passages. Using
        # entity nodes (rather than passage nodes) as PPR seeds keeps the
        # diffusion meaningful: it propagates through the entity graph as
        # a real entity match would.
        if not ppr_seeds:
            seed_docs = [doc_id for doc_id, _ in self.store.search(qvec, top_k=3)]
            for d in seed_docs:
                ents = kg.doc_to_entities.get(d, set())
                ppr_seeds.extend([e for e in ents if e in kg.graph])
            ppr_seeds = list(dict.fromkeys(ppr_seeds))[:8]   # dedup, cap

        # 2. PPR over the bipartite KG (uses precomputed sparse solver)
        if ppr_seeds and self._ppr_solver is not None:
            ppr_scores = self._ppr_solver.solve(
                ppr_seeds, alpha=self.config.ppr_alpha,
            )
        else:
            ppr_scores = {}

        # 3. Score each candidate document. Vectorise cosine over the whole
        #    doc set in one matmul rather than per-doc Python loop (~3×
        #    speed-up on the doc-scoring stage).
        doc_ids = [n[4:] for n in kg.graph.nodes if n.startswith("doc:")]
        if not doc_ids:
            doc_relevance, doc_structural = {}, {}
        else:
            mat_rows = []
            valid_ids: list[str] = []
            for doc_id in doc_ids:
                try:
                    mat_rows.append(self._vec_of(doc_id))
                    valid_ids.append(doc_id)
                except (KeyError, AttributeError):
                    continue
            mat = np.stack(mat_rows).astype(np.float32)
            mat_norms = np.linalg.norm(mat, axis=1, keepdims=True)
            mat_norms[mat_norms == 0] = 1.0
            mat_unit = mat / mat_norms
            qn = qvec / max(float(np.linalg.norm(qvec)), 1e-12)
            sims = (mat_unit @ qn).astype(np.float64)
            doc_relevance = {valid_ids[i]: float(sims[i])
                             for i in range(len(valid_ids))}
            doc_structural = {
                valid_ids[i]: float(ppr_scores.get(f"doc:{valid_ids[i]}", 0.0))
                for i in range(len(valid_ids))
            }

        # Multi-component scoring on docs only. Coherence is irrelevant in KG
        # mode (the graph already encodes topical structure via entities).
        composite = composite_score(
            relevance=doc_relevance,
            structural=doc_structural,
            coherence={d: 1.0 for d in doc_relevance},   # neutral
            weights=self.config.weights,
            integration=self.config.integration,
        )

        ordered = sorted(composite.items(), key=lambda kv: -kv[1])
        scored: list[ScoredDocument] = []
        for rank, (doc_id, score) in enumerate(ordered):
            try:
                doc = self.store.get(doc_id)
            except KeyError:
                continue
            scored.append(ScoredDocument(
                doc=doc,
                similarity=doc_relevance[doc_id],
                ppr_score=doc_structural[doc_id],
                composite_score=float(score),
                rank=rank,
            ))

        context, picked = pack(
            scored[:top_k * 5],
            budget_tokens=budget_tokens,
            redundancy_lambda=self.config.redundancy_lambda,
            vec_of=self._vec_of,
        )

        # Build reasoning paths from query entities → retrieved docs.
        # Useful for explainability: shows the multi-hop chain that
        # surfaced each document.
        picked_ids = [p.doc.id for p in picked[:top_k]]
        reasoning = build_reasoning_paths(
            kg.graph, ppr_seeds, picked_ids, ppr_scores,
            max_paths=min(5, top_k),
            max_hops=4,
        )

        return RetrievalResult(
            query=qtext,
            context=context,
            sources=picked[:top_k],
            reasoning=reasoning,
            debug={
                "mode": "kg",
                "query_mentions": q_mentions,
                "ppr_seeds": ppr_seeds[:10],
                "kg_nodes": kg.graph.number_of_nodes(),
                "kg_edges": kg.graph.number_of_edges(),
            },
        )
