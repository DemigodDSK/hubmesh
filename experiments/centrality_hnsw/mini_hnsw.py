"""Minimal NumPy HNSW — readable, modifiable reference.

This is a faithful implementation of the algorithm from Malkov & Yashunin
(2018) — enough to validate against hnswlib's recall, and structured so
that the level-assignment strategy is a single swappable function.

Not fast — pure Python loops + NumPy vector ops, no C extension. Targets
10k–100k vector experiments where the algorithmic comparison matters more
than throughput.

API parity with hnswlib for benchmark plug-and-play:
  index = MiniHNSW(dim, space='l2', M=16, ef_construction=200)
  index.add_items(X)            # batched add; deterministic given seed
  index.set_ef(50)
  labels, dists = index.knn_query(Q, k=10)
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Callable
import numpy as np


def _dist_l2_sq(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Squared L2 distance — argmin-equivalent to L2 and faster."""
    diff = a - b
    return np.einsum("...i,...i->...", diff, diff)


def _dist_ip(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Negative inner product (so smaller = better, matching L2 semantics)."""
    return -np.einsum("...i,...i->...", a, b)


_DISTANCE_FNS: dict[str, Callable] = {
    "l2": _dist_l2_sq,
    "ip": _dist_ip,
}


@dataclass
class MiniHNSW:
    dim: int
    space: str = "l2"
    M: int = 16                       # max neighbours per node at layer > 0
    M0: int | None = None             # at layer 0 (default 2*M)
    ef_construction: int = 200
    seed: int = 0
    # Level-assignment function: takes (node_id, rng, ml) → int level
    # Default is HNSW's standard exponential decay.
    level_fn: Callable[..., int] | None = None

    # Internals — initialised lazily
    _data: list[np.ndarray] = field(default_factory=list)
    _levels: list[int] = field(default_factory=list)
    # graph[level] is a dict of node_id -> list of neighbor ids
    _graph: list[dict[int, list[int]]] = field(default_factory=list)
    _entry_point: int | None = None
    _top_layer: int = -1
    _ef: int = 50
    _rng: np.random.Generator | None = None
    _ml: float = 1.0

    def __post_init__(self):
        if self.M0 is None:
            self.M0 = 2 * self.M
        self._rng = np.random.default_rng(self.seed)
        self._ml = 1.0 / math.log(self.M) if self.M > 1 else 1.0
        if self.level_fn is None:
            self.level_fn = _default_level

    # ---------- public API ----------

    def add_items(self, X: np.ndarray, ids: np.ndarray | None = None) -> None:
        X = np.asarray(X, dtype=np.float32)
        n = X.shape[0]
        if ids is not None:
            assert len(ids) == n
        else:
            ids = np.arange(len(self._data), len(self._data) + n)
        for i, vec in enumerate(X):
            self._add_one(int(ids[i]), vec)

    def set_ef(self, ef: int) -> None:
        self._ef = max(1, int(ef))

    def knn_query(self, queries: np.ndarray, k: int):
        queries = np.asarray(queries, dtype=np.float32)
        labels = np.empty((len(queries), k), dtype=np.int64)
        dists = np.empty((len(queries), k), dtype=np.float32)
        for i, q in enumerate(queries):
            results = self._search_layer0(q, k, self._ef)
            for j in range(k):
                if j < len(results):
                    labels[i, j] = results[j][1]
                    dists[i, j] = results[j][0]
                else:
                    labels[i, j] = -1
                    dists[i, j] = np.inf
        return labels, dists

    # ---------- internals ----------

    def _dist(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(_DISTANCE_FNS[self.space](a, b))

    def _add_one(self, node_id: int, vec: np.ndarray) -> None:
        # Reserve the slot
        if node_id != len(self._data):
            # Allow only sequential ids in this minimal impl
            raise ValueError(f"expected id {len(self._data)}, got {node_id}")
        self._data.append(vec.astype(np.float32))
        # Assign level
        level = self.level_fn(node_id=node_id, rng=self._rng, ml=self._ml,
                              index=self)
        self._levels.append(level)
        # Make sure graph has enough layers
        while len(self._graph) <= level:
            self._graph.append({})
        for lvl in range(level + 1):
            self._graph[lvl][node_id] = []

        # First node?
        if self._entry_point is None:
            self._entry_point = node_id
            self._top_layer = level
            return

        # Phase 1: greedy from top layer down to (level + 1) — single
        # nearest neighbour at each layer
        cur_dist = self._dist(vec, self._data[self._entry_point])
        cur_node = self._entry_point
        for lc in range(self._top_layer, level, -1):
            cur_node, cur_dist = self._greedy_step(vec, cur_node, cur_dist, lc)

        # Phase 2: from level downward, beam search + connect
        # Start the beam from the single nearest we found above
        candidates: list[tuple[float, int]] = [(cur_dist, cur_node)]
        for lc in range(min(level, self._top_layer), -1, -1):
            # Beam search at this layer
            results = self._search_layer_with_seeds(
                vec, candidates, ef=self.ef_construction, layer=lc,
            )
            # Pick the best M (or M0) neighbours — subclasses override
            # `_select_neighbors` to swap in a different selection rule
            # (the multi-component variant uses this hook).
            max_M = self.M0 if lc == 0 else self.M
            neighbours = self._select_neighbors(
                vec, results, max_M, layer=lc, node_id=node_id,
            )
            # Connect
            self._graph[lc][node_id] = neighbours.copy()
            for nb in neighbours:
                self._graph[lc].setdefault(nb, []).append(node_id)
                # Cap neighbour's connections back if exceeded
                if len(self._graph[lc][nb]) > max_M:
                    # Reselect that node's neighbours by heuristic
                    nb_vec = self._data[nb]
                    nb_candidates = [
                        (self._dist(nb_vec, self._data[x]), x)
                        for x in self._graph[lc][nb]
                    ]
                    nb_candidates.sort()
                    self._graph[lc][nb] = _select_neighbors_heuristic(
                        nb_vec, nb_candidates, max_M,
                        self._data, self._dist
                    )
            # Carry over the beam to the next (lower) layer
            candidates = results

        # Update entry point if this node was promoted higher
        if level > self._top_layer:
            self._top_layer = level
            self._entry_point = node_id

    def _greedy_step(self, q: np.ndarray, cur: int, cur_dist: float,
                     layer: int) -> tuple[int, float]:
        """Greedy descent at one layer — move to the closest neighbour until
        no neighbour is closer."""
        improved = True
        while improved:
            improved = False
            for nb in self._graph[layer].get(cur, ()):
                d = self._dist(q, self._data[nb])
                if d < cur_dist:
                    cur_dist = d
                    cur = nb
                    improved = True
        return cur, cur_dist

    def _search_layer_with_seeds(
        self, q: np.ndarray, seeds: list[tuple[float, int]],
        ef: int, layer: int,
    ) -> list[tuple[float, int]]:
        """Beam search at `layer` starting from `seeds`. Returns a list of
        (dist, node_id) sorted by distance ascending, length ≤ ef."""
        visited: set[int] = set()
        # candidates: nodes to expand (min-heap-like)
        # results:    closest seen (we cap at ef)
        candidates = list(seeds)
        results = list(seeds)
        for _, n in seeds:
            visited.add(n)
        candidates.sort()
        results.sort()
        while candidates:
            c_dist, c = candidates[0]
            # Stop when the closest candidate is worse than the worst result
            if len(results) >= ef and c_dist > results[-1][0]:
                break
            candidates.pop(0)
            for nb in self._graph[layer].get(c, ()):
                if nb in visited:
                    continue
                visited.add(nb)
                d = self._dist(q, self._data[nb])
                if len(results) < ef or d < results[-1][0]:
                    candidates.append((d, nb))
                    results.append((d, nb))
                    candidates.sort()
                    results.sort()
                    if len(results) > ef:
                        results = results[:ef]
        return results

    def _select_neighbors(
        self, vec: np.ndarray,
        candidates: list[tuple[float, int]],
        max_M: int, *, layer: int, node_id: int,
    ) -> list[int]:
        """Default neighbour selection — HNSW Algorithm 4 (Malkov-Yashunin).
        Subclasses override this to implement different selection criteria
        (e.g. multi-component scoring)."""
        return _select_neighbors_heuristic(
            vec, candidates, max_M, self._data, self._dist
        )

    def _search_layer0(self, q: np.ndarray, k: int, ef: int):
        """Full search: greedy down to layer 1, beam at layer 0."""
        if self._entry_point is None:
            return []
        cur = self._entry_point
        cur_dist = self._dist(q, self._data[cur])
        for lc in range(self._top_layer, 0, -1):
            cur, cur_dist = self._greedy_step(q, cur, cur_dist, lc)
        results = self._search_layer_with_seeds(
            q, [(cur_dist, cur)], ef=max(ef, k), layer=0,
        )
        return results[:k]


def _default_level(*, node_id: int, rng: np.random.Generator,
                   ml: float, index: MiniHNSW) -> int:
    """Standard HNSW level assignment: -ln(uniform) * mL, floored."""
    u = float(rng.random())
    # Guard against u==0
    if u <= 0:
        u = 1e-12
    return int(-math.log(u) * ml)


def _select_neighbors_heuristic(
    query: np.ndarray,
    candidates: list[tuple[float, int]],
    M: int,
    data: list[np.ndarray],
    dist_fn: Callable,
) -> list[int]:
    """Diversified neighbour selection from Malkov & Yashunin (Algorithm 4).
    Among the candidates, pick `M` that are mutually distant — improves
    long-range connectivity over plain top-M."""
    if len(candidates) <= M:
        return [c[1] for c in sorted(candidates)]
    selected: list[int] = []
    # candidates already sorted ascending by distance to query
    candidates = sorted(candidates)
    for d, n in candidates:
        if len(selected) >= M:
            break
        # Accept if n is closer to query than to any already-selected
        # neighbour (heuristic preserves diversity / long edges)
        good = True
        n_vec = data[n]
        for s in selected:
            d_to_s = float(dist_fn(n_vec, data[s]))
            if d_to_s < d:
                good = False
                break
        if good:
            selected.append(n)
    return selected
