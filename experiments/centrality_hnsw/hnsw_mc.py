"""HNSW-MC — multi-component neighbour selection at construction time.

The first centrality-HNSW attempt (`centrality_hnsw.py`) modified level
assignment. That didn't work, because HNSW's quality comes from neighbour
selection (Algorithm 4), not the random hierarchy. This variant attacks
the right step.

For each candidate neighbour c, score:

    Score(c, q, S) = P(c, q)^w_P
                   · (D(c, S) + ε)^w_D
                   · (H(c) + ε)^w_H
                   · (G(c) + ε)^w_G

  P  proximity to the inserting node            — like Algorithm 4
  D  diversity from already-selected            — like Algorithm 4
  H  hub utility (online degree, log-normalised) — NEW
  G  centroid proximity                          — NEW

Components P and D recover (a multiplicative analogue of) HNSW's
heuristic. H and G are the additional components real vector DB
literature treats as separately important (NSG's medoid, IVF centroids,
PageRank-style hub awareness).

Greedy selection with rescoring: at each step pick the highest-scoring
unselected candidate, then update D for the rest.
"""
from __future__ import annotations
import math
import numpy as np

from mini_hnsw import MiniHNSW


def _select_neighbors_with_rejection(query, candidates, M, data, dist_fn):
    """Algorithm 4 rejection rule. `candidates` is already in the desired
    consideration order (may be biased away from pure distance order);
    we still use RAW distances for the rejection check.

    Accept c if R is empty OR c is closer to query than to any selected.
    """
    selected: list[int] = []
    for d, c in candidates:
        if len(selected) >= M:
            break
        if not selected:
            selected.append(c)
            continue
        c_vec = data[c]
        # Reject c if any selected is closer to it than query is
        keep = True
        for s in selected:
            d_to_s = float(dist_fn(c_vec, data[s]))
            if d_to_s < d:
                keep = False
                break
        if keep:
            selected.append(c)
    # If we didn't fill to M (rejected too many), top up with the
    # remaining closest unselected — preserves connectivity.
    if len(selected) < M:
        for d, c in candidates:
            if c not in selected:
                selected.append(c)
                if len(selected) >= M:
                    break
    return selected


class HNSWMultiComponent(MiniHNSW):
    """Multi-component neighbour selection variant of MiniHNSW.

    Configure with `set_weights(P=…, D=…, H=…, G=…)`. Defaults give a
    sensible starting point matching the spirit of Algorithm 4 plus the
    two new components."""

    def __post_init__(self):
        super().__post_init__()
        # Online degree of each node — incremented every time it's added
        # to a neighbour list. Used for the H component.
        self._online_degree: dict[int, int] = {}
        # Data centroid — set when add_items is called.
        self._centroid: np.ndarray | None = None
        self._weights: dict[str, float] = {
            "P": 1.0,
            "D": 1.0,
            "H": 0.5,
            "G": 0.5,
        }
        self._eps = 1e-3

    def set_weights(self, **kw):
        for k, v in kw.items():
            if k not in self._weights:
                raise ValueError(f"unknown weight: {k}")
            self._weights[k] = float(v)
        return self

    def add_items(self, X: np.ndarray, ids: np.ndarray | None = None) -> None:
        X = np.asarray(X, dtype=np.float32)
        # Precompute the data centroid once for the whole insert batch.
        self._centroid = X.mean(axis=0).astype(np.float32)
        super().add_items(X, ids)

    def _select_neighbors(
        self, vec: np.ndarray,
        candidates: list[tuple[float, int]],
        max_M: int, *, layer: int, node_id: int,
    ) -> list[int]:
        """Modified Algorithm 4: keep the geometric rejection rule (uses
        raw distances) but bias the candidate-ordering with H and G.

        Adjusted distance for ordering only:
            d'(c, q) = d(c, q) · (1 − α_H · H_norm(c) − α_G · G_norm(c))

        H and G effectively *promote* hubby / centrally-positioned nodes
        to be considered earlier — without breaking the diversity
        rejection that makes Algorithm 4 work. Weights bounded to keep
        d' positive."""
        if len(candidates) <= max_M:
            chosen = [c[1] for c in sorted(candidates)]
            self._bump_degrees(chosen)
            return chosen

        wH = self._weights["H"]
        wG = self._weights["G"]
        # wP and wD are inherent in Algorithm 4; not used as exponents here.
        # We bound the combined bias to keep adjusted distances positive.
        alpha_H = min(0.49, wH)
        alpha_G = min(0.49, wG)

        cand_ids = [c[1] for c in candidates]
        cand_dists = np.array([c[0] for c in candidates], dtype=np.float64)
        cand_vecs = np.stack([self._data[c] for c in cand_ids]).astype(np.float32)

        # H: online degree, log-normalised
        if self._online_degree:
            max_deg = max(self._online_degree.values())
        else:
            max_deg = 0
        log_max = math.log(1 + max_deg) if max_deg > 0 else 1.0
        H_vals = np.array([
            math.log(1 + self._online_degree.get(c, 0)) / log_max
            for c in cand_ids
        ])

        # G: centroid proximity, normalised to [0, 1] across candidates
        if self._centroid is not None:
            G_dists = np.linalg.norm(cand_vecs - self._centroid, axis=1)
            # invert and normalise: closer-to-centroid → higher G
            G_raw = -G_dists
            lo, hi = G_raw.min(), G_raw.max()
            if hi > lo:
                G_vals = (G_raw - lo) / (hi - lo)
            else:
                G_vals = np.full_like(G_raw, 0.5)
        else:
            G_vals = np.full(len(cand_ids), 0.5)

        # Adjusted distance for ORDERING (always positive by construction)
        d_adj = cand_dists * (1.0 - alpha_H * H_vals - alpha_G * G_vals)

        # Sort candidates by ADJUSTED distance, but remember the raw too
        order = np.argsort(d_adj)
        ordered_cands = [
            (float(cand_dists[i]), int(cand_ids[i])) for i in order
        ]

        # Run Algorithm 4's rejection rule using RAW distance — that's
        # what gives HNSW its geometric correctness. Only the ORDER
        # changed via the H/G bias.
        selected = _select_neighbors_with_rejection(
            vec, ordered_cands, max_M, self._data, self._dist,
        )
        self._bump_degrees(selected)
        return selected

    def _bump_degrees(self, chosen: list[int]) -> None:
        """Track online degree for the H component."""
        for n in chosen:
            self._online_degree[n] = self._online_degree.get(n, 0) + 1
