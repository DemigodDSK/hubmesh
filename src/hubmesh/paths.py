"""Reasoning path extraction over the bipartite KG.

When hubmesh retrieves a document via PPR diffusion from query-entity
seeds, the implicit "reason" is a path through the KG:

    query_entity_A  →  passage_X (mentions A and B)  →  query_entity_B  →  ...
                       ↘
                         retrieved_doc (mentions B and other entities)

We materialise these paths so users get *why* each document was returned
— useful for explainability, debugging, and downstream re-ranking by an
LLM.

For each retrieved document, we find the shortest path to *any* query
seed in the KG. Paths are scored by a combination of length (shorter is
better) and end-to-end PPR mass.
"""
from __future__ import annotations
from heapq import heappush, heappop
import networkx as nx

from .types import ReasoningPath


def shortest_paths_from_seeds(
    G: nx.Graph,
    seeds: list[str],
    target_doc_nodes: list[str],
    max_hops: int = 3,
) -> dict[str, list[str]]:
    """For each target doc node, return the shortest path (as node ids)
    from the closest seed to it. Paths longer than `max_hops` are dropped
    (we only care about plausibly explanatory chains)."""
    if not seeds or not target_doc_nodes:
        return {}
    valid_seeds = [s for s in seeds if s in G]
    if not valid_seeds:
        return {}
    targets = set(target_doc_nodes) & set(G.nodes)
    if not targets:
        return {}

    # Multi-source BFS from all seeds. For each visited node we remember
    # which seed it came from and one predecessor → reconstructs a path.
    came_from: dict[str, tuple[str, str]] = {}   # node -> (predecessor, seed)
    seed_of: dict[str, str] = {s: s for s in valid_seeds}
    visited: set[str] = set(valid_seeds)
    frontier: list[tuple[int, str]] = [(0, s) for s in valid_seeds]
    found: dict[str, str] = {}                    # target -> reached-via-seed

    while frontier:
        depth, node = heappop(frontier)
        if node in targets:
            found[node] = seed_of[node]
            if len(found) == len(targets):
                break
        if depth >= max_hops:
            continue
        for nb in G.neighbors(node):
            if nb in visited:
                continue
            visited.add(nb)
            came_from[nb] = (node, seed_of[node])
            seed_of[nb] = seed_of[node]
            heappush(frontier, (depth + 1, nb))

    # Reconstruct paths
    paths: dict[str, list[str]] = {}
    for tgt, seed in found.items():
        path = [tgt]
        cur = tgt
        while cur != seed:
            prev, _ = came_from[cur]
            path.append(prev)
            cur = prev
        path.reverse()
        paths[tgt] = path
    return paths


def build_reasoning_paths(
    G: nx.Graph,
    seeds: list[str],
    retrieved_doc_ids: list[str],
    ppr_scores: dict[str, float],
    max_paths: int = 5,
    max_hops: int = 3,
) -> list[ReasoningPath]:
    """Materialise a list of ReasoningPath for the top retrieved documents.

    Path score = (1/(1+len)) * end_node_ppr — favours short paths to
    high-PPR endpoints. Returns at most `max_paths` paths.
    """
    target_nodes = [f"doc:{d}" for d in retrieved_doc_ids if f"doc:{d}" in G]
    if not target_nodes:
        return []
    raw = shortest_paths_from_seeds(G, seeds, target_nodes, max_hops=max_hops)
    out: list[ReasoningPath] = []
    for tgt, path in raw.items():
        if len(path) < 2:
            continue
        edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        end_ppr = float(ppr_scores.get(tgt, 0.0))
        score = end_ppr / (1.0 + (len(path) - 1))
        out.append(ReasoningPath(node_ids=path, edges=edges, score=score))
    out.sort(key=lambda p: -p.score)
    return out[:max_paths]
