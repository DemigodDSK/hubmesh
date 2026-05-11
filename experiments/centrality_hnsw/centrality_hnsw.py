"""Centrality-biased HNSW — the core hypothesis from the design docs.

Standard HNSW assigns each node's maximum layer by

    ℓ = floor(-ln(U) · m_L),   U ~ Uniform(0,1),  m_L = 1/ln(M)

This is random. The CR-HNG design doc proposes replacing it with

    Pr(ℓ ≥ ℓ_target) = exp(-ℓ_target / (m_0 + m_1 · I_i))

where I_i is the node's *importance* (a centrality measure). High-I nodes
end up in upper layers more often → become "highway" hubs that searches
traverse early, ideally reducing hops and improving recall at a given
ef_search.

The naive question is what to use as I_i during construction — the node
has no neighbours until it's inserted. We sidestep this with a **two-pass
build**:

  Pass 1: build vanilla HNSW (random levels).
  Compute degree centrality on the layer-0 graph.
  Pass 2: re-build with each node's level drawn using its centrality
          from Pass 1.

This costs 2× the construction time but lets us test the algorithmic
hypothesis cleanly. A production version would compute centrality
incrementally with streaming updates.
"""
from __future__ import annotations
import math
import numpy as np

from mini_hnsw import MiniHNSW, _default_level


def compute_degree_centrality_layer0(index: MiniHNSW) -> dict[int, float]:
    """Layer-0 degree, min-max normalised to [0, 1]."""
    g0 = index._graph[0] if index._graph else {}
    if not g0:
        return {}
    degs = {node: len(nbrs) for node, nbrs in g0.items()}
    lo, hi = min(degs.values()), max(degs.values())
    if hi == lo:
        return {n: 0.5 for n in degs}
    return {n: (d - lo) / (hi - lo) for n, d in degs.items()}


def compute_proximity_to_centroid(X: np.ndarray) -> dict[int, float]:
    """Geometric centrality: 1 - distance-to-centroid, normalised.
    Nodes near the data centroid are 'central' in vector space.
    Doesn't require building the graph first — single-pass usable."""
    centroid = X.mean(axis=0)
    dists = np.linalg.norm(X - centroid, axis=1)
    inv = -dists                    # closer to centroid → higher value
    lo, hi = inv.min(), inv.max()
    if hi == lo:
        return {int(i): 0.5 for i in range(len(X))}
    return {int(i): float((inv[i] - lo) / (hi - lo)) for i in range(len(X))}


def make_centrality_level_fn(centrality: dict[int, float],
                              m1: float = 1.0):
    """Build a level_fn closure that uses `centrality[node_id]` as I_i.

    Formula: ℓ = floor(-ln(U) · (m_L + m1 · I_norm))
    where m_L is the standard HNSW constant and m1 controls how much
    centrality stretches the level distribution.
    """
    def level_fn(*, node_id: int, rng: np.random.Generator,
                 ml: float, index: MiniHNSW) -> int:
        u = float(rng.random())
        if u <= 0:
            u = 1e-12
        I_i = float(centrality.get(node_id, 0.0))
        scale = ml + m1 * I_i * ml   # multiplicative boost
        return int(-math.log(u) * scale)
    return level_fn


class CentralityHNSW(MiniHNSW):
    """A two-pass MiniHNSW variant. Use `build_from_array(X, m1=...)` to
    perform the full two-pass procedure end-to-end."""

    def build_from_array(self, X: np.ndarray, m1: float = 1.0,
                         centrality_measure: str = "degree") -> None:
        """Build with centrality-biased levels.

        centrality_measure:
            "degree"    — layer-0 degree from a vanilla first-pass build.
                          Two-pass; 2× construction cost.
            "centroid"  — proximity to the data centroid. Single-pass;
                          no extra cost beyond level scaling.

        After this returns the index is the biased version.
        """
        X = np.asarray(X, dtype=np.float32)
        if centrality_measure == "degree":
            # Pass 1: vanilla, then compute degree on layer 0
            self._reset()
            self.level_fn = _default_level
            super().add_items(X)
            cent = compute_degree_centrality_layer0(self)
            self._reset()
        elif centrality_measure == "centroid":
            cent = compute_proximity_to_centroid(X)
            self._reset()
        else:
            raise ValueError(f"unsupported: {centrality_measure}")
        # Build (or rebuild) with biased levels
        self.level_fn = make_centrality_level_fn(cent, m1=m1)
        super().add_items(X)

    def _reset(self) -> None:
        """Wipe internal state so we can rebuild."""
        self._data = []
        self._levels = []
        self._graph = []
        self._entry_point = None
        self._top_layer = -1
        # Reset RNG to keep run reproducibility independent of pass-1 calls
        self._rng = np.random.default_rng(self.seed)
