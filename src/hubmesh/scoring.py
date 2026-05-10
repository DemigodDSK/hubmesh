"""Multi-component scoring — the novel contribution.

The pattern carries forward from the NNSI / SDN thesis: instead of one metric,
combine multiple metrics chosen by the *role* they play. Components are
designed for the GraphRAG-retrieval problem, not borrowed verbatim from NNSI:

  R (Relevance)          — cosine similarity of doc to query
  S (Structural fit)     — Personalized PageRank score on induced subgraph
                            (high = doc is reachable from query seeds)
  C (Community coherence)— fraction of doc's neighbours in the query's
                            community (positively correlates with answer
                            membership; the SDN-direct C term anti-correlated
                            and broke our earlier formula)

We use a logarithmic geometric mean for integration:

    score(i) = exp((w_R log(ε+R_i) + w_S log(ε+S_i) + w_C log(ε+C_i))
                   / (w_R + w_S + w_C))

Geometric-mean integration is more robust than raw multiplication: a near-zero
component drags the score down (preserving the "must be good at every role"
property) without zeroing it out catastrophically when one component is
genuinely small.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import networkx as nx

EPS = 1e-9


@dataclass
class ScoringWeights:
    """All weights default to equal — tune per workload."""
    relevance: float = 1.0
    structural: float = 1.0
    coherence: float = 1.0


@dataclass
class ScoringConfig:
    """Combine weights with the integration mode.

    integration:
      'geom' — geometric mean (multi-criteria, all-must-be-strong; original
               NNSI flavour but more brittle when components have low
               correlation with the answer).
      'sum'  — weighted sum (forgiving — a strong component can compensate
               for a weak one; better when one component is highly
               informative, like cosine similarity for QA queries).
    """
    weights: ScoringWeights = None
    integration: str = "sum"   # "geom" | "sum"

    def __post_init__(self):
        if self.weights is None:
            self.weights = ScoringWeights()


def _minmax_dict(d: dict[str, float]) -> dict[str, float]:
    if not d:
        return {}
    vs = np.fromiter(d.values(), dtype=float)
    lo, hi = vs.min(), vs.max()
    if hi - lo < EPS:
        return {k: 0.5 for k in d}
    return {k: (v - lo) / (hi - lo) for k, v in d.items()}


def compute_relevance(
    G: nx.Graph, query_vec: np.ndarray, vec_of: callable,
) -> dict[str, float]:
    """R component — cosine similarity of each subgraph node to the query."""
    qn = query_vec / max(float(np.linalg.norm(query_vec)), 1e-12)
    out: dict[str, float] = {}
    for n in G.nodes:
        v = vec_of(n)
        vn = v / max(float(np.linalg.norm(v)), 1e-12)
        out[n] = float(qn @ vn)
    return _minmax_dict(out)


def compute_coherence(
    G: nx.Graph, communities: dict[str, int], anchor_community: set[str],
) -> dict[str, float]:
    """C component — fraction of neighbours that belong to the query's
    anchor community. High C = doc is structurally inside the right cluster."""
    out: dict[str, float] = {}
    for n in G.nodes:
        nbrs = list(G.neighbors(n))
        if not nbrs:
            out[n] = 0.0
            continue
        in_anchor = sum(1 for nb in nbrs if nb in anchor_community)
        out[n] = in_anchor / len(nbrs)
    return out  # already in [0,1]


def composite_score(
    relevance: dict[str, float],
    structural: dict[str, float],
    coherence: dict[str, float],
    weights: ScoringWeights = ScoringWeights(),
    integration: str = "sum",
) -> dict[str, float]:
    """Combine the three components into a per-node score.

    All three inputs are min-max normalised to [0, 1] before integration so
    components with different natural scales (PPR sums to 1; cosine in
    [-1, 1]; coherence in [0, 1]) are comparable.

    integration:
      'geom' — log-geometric mean. Strong "all must be high" property.
               Use when all components positively correlate with the answer.
      'sum'  — weighted arithmetic sum. Forgiving — a high component
               compensates for a low one. Use when one component (e.g.
               relevance) is highly informative on its own.
    """
    R = _minmax_dict(relevance)
    S = _minmax_dict(structural)
    C = _minmax_dict(coherence)
    nodes = set(R) | set(S) | set(C)
    out: dict[str, float] = {}
    w_total = weights.relevance + weights.structural + weights.coherence
    if integration == "geom":
        for n in nodes:
            log_geo = (
                weights.relevance  * np.log(EPS + R.get(n, 0.0)) +
                weights.structural * np.log(EPS + S.get(n, 0.0)) +
                weights.coherence  * np.log(EPS + C.get(n, 0.0))
            ) / w_total
            out[n] = float(np.exp(log_geo))
    elif integration == "sum":
        for n in nodes:
            out[n] = float(
                (weights.relevance  * R.get(n, 0.0) +
                 weights.structural * S.get(n, 0.0) +
                 weights.coherence  * C.get(n, 0.0)) / w_total
            )
    else:
        raise ValueError(f"unknown integration mode: {integration!r}")
    return out
