"""Graph utilities: induced-subgraph construction + Louvain community anchoring."""
from __future__ import annotations
import networkx as nx
import numpy as np

from .adapters.base import VectorStore


def build_induced_subgraph(
    store: VectorStore,
    seed_ids: list[str],
    hops: int = 2,
    cap: int = 2000,
) -> nx.Graph:
    """BFS expand `hops` hops out from each seed, returning the induced subgraph
    of the proximity graph. Capped at `cap` nodes to keep query-time cost
    bounded — the most important real-time invariant of hubmesh.
    """
    nodes: set[str] = set(seed_ids)
    frontier: set[str] = set(seed_ids)
    for _ in range(hops):
        if len(nodes) >= cap:
            break
        new_frontier: set[str] = set()
        for nid in frontier:
            for nb in store.neighbors(nid, k=16):
                if nb not in nodes:
                    new_frontier.add(nb)
                    if len(nodes) + len(new_frontier) >= cap:
                        break
            if len(nodes) + len(new_frontier) >= cap:
                break
        nodes.update(new_frontier)
        frontier = new_frontier
        if not frontier:
            break

    G = nx.Graph()
    G.add_nodes_from(nodes)
    for nid in nodes:
        for nb in store.neighbors(nid, k=16):
            if nb in nodes:
                G.add_edge(nid, nb)
    return G


def detect_communities(G: nx.Graph, seed: int = 0) -> dict[str, int]:
    """Run Louvain. Return node_id → community_id."""
    if G.number_of_nodes() == 0:
        return {}
    communities = nx.community.louvain_communities(G, seed=seed)
    out: dict[str, int] = {}
    for ci, comm in enumerate(communities):
        for n in comm:
            out[n] = ci
    return out


def best_community_for_query(
    query_vec: np.ndarray,
    G: nx.Graph,
    communities: dict[str, int],
    vec_of: callable,
) -> set[str]:
    """Pick the community whose centroid is closest to the query — robust
    even when individual top-similar nodes scatter across communities under
    feature overlap (the failure mode we observed in the SDN→GraphRAG
    transfer experiments)."""
    if not communities:
        return set(G.nodes)
    by_comm: dict[int, list[str]] = {}
    for n, c in communities.items():
        by_comm.setdefault(c, []).append(n)
    q = query_vec / max(float(np.linalg.norm(query_vec)), 1e-12)
    best_c = None
    best_d = float("inf")
    for c, members in by_comm.items():
        centroid = np.mean([vec_of(m) for m in members], axis=0)
        cn = centroid / max(float(np.linalg.norm(centroid)), 1e-12)
        d = float(1.0 - q @ cn)  # cosine distance
        if d < best_d:
            best_d = d
            best_c = c
    return set(by_comm[best_c]) if best_c is not None else set(G.nodes)
