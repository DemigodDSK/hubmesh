"""Personalized PageRank over the induced subgraph."""
from __future__ import annotations
import networkx as nx
import numpy as np


def personalized_pagerank(
    G: nx.Graph,
    seeds: list[str],
    alpha: float = 0.15,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> dict[str, float]:
    """Standard PPR with teleport distribution concentrated on the seed set.

    `alpha` here follows the HippoRAG / NetworkX convention as the *teleport*
    probability (so 1-alpha is the propagation probability). Lower alpha →
    diffusion spreads further from seeds.
    """
    if G.number_of_nodes() == 0:
        return {}
    if not seeds:
        # Uniform restart distribution if no seeds — degrades to vanilla PageRank
        return nx.pagerank(G, alpha=1.0 - alpha, max_iter=max_iter, tol=tol)
    valid_seeds = [s for s in seeds if s in G]
    if not valid_seeds:
        return nx.pagerank(G, alpha=1.0 - alpha, max_iter=max_iter, tol=tol)
    personalization = {n: 0.0 for n in G.nodes}
    for s in valid_seeds:
        personalization[s] = 1.0 / len(valid_seeds)
    return nx.pagerank(
        G,
        alpha=1.0 - alpha,            # nx.pagerank's `alpha` is propagation prob
        personalization=personalization,
        max_iter=max_iter,
        tol=tol,
    )
