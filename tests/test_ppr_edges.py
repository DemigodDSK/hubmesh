"""PPRSolver edge cases (external review): isolated/dangling nodes must
not leak probability mass, duplicate seeds must not shrink the restart
mass, and parameters are validated."""
import networkx as nx
import numpy as np
import pytest

from hubmesh.ppr import PPRSolver


def total(d):
    return float(sum(d.values()))


def test_single_isolated_node_keeps_unit_mass():
    G = nx.Graph(); G.add_node("doc:a")
    p = PPRSolver(G).solve(["doc:a"])
    assert abs(total(p) - 1.0) < 1e-9          # was 0.15 (mass leaked)


def test_dangling_mass_is_redistributed_not_lost():
    G = nx.Graph()
    G.add_edge("doc:a", "ent:x"); G.add_node("doc:iso")   # iso is dangling
    solver = PPRSolver(G)
    p = solver.solve(["doc:iso"])
    assert abs(total(p) - 1.0) < 1e-9
    p2 = solver.solve(["ent:x"])
    assert abs(total(p2) - 1.0) < 1e-9


def test_duplicate_seeds_do_not_reduce_restart_mass():
    G = nx.Graph(); G.add_edge("doc:a", "ent:x"); G.add_edge("doc:b", "ent:x")
    solver = PPRSolver(G)
    once = solver.solve(["ent:x"])
    dup = solver.solve(["ent:x", "ent:x"])
    assert abs(total(dup) - 1.0) < 1e-9        # was 0.5
    for n in once:
        assert abs(once[n] - dup[n]) < 1e-9    # identical distribution


def test_solve_multi_columns_each_sum_to_one():
    G = nx.Graph(); G.add_edge("doc:a", "ent:x"); G.add_edge("doc:b", "ent:y")
    G.add_node("doc:iso")
    res = PPRSolver(G).solve_multi([["ent:x"], ["ent:y", "ent:y"], ["doc:iso"]])
    for d in res:
        assert abs(total(d) - 1.0) < 1e-9


@pytest.mark.parametrize("alpha", [0.0, 1.5, -0.1, float("nan")])
def test_invalid_alpha_rejected(alpha):
    G = nx.Graph(); G.add_edge("doc:a", "ent:x")
    with pytest.raises(ValueError):
        PPRSolver(G).solve(["ent:x"], alpha=alpha)


def test_non_finite_or_negative_edge_weight_rejected():
    for w in (-1.0, float("inf")):
        G = nx.Graph(); G.add_edge("doc:a", "ent:x", weight=w)
        with pytest.raises(ValueError):
            PPRSolver(G)


def test_normal_graph_unchanged_where_no_dangling_nodes():
    """Redistribution must be a no-op when nothing dangles: results on an
    ordinary connected graph stay within solver tolerance of the
    reference power iteration."""
    G = nx.karate_club_graph()
    G = nx.relabel_nodes(G, {n: f"ent:{n}" for n in G.nodes})
    p = PPRSolver(G).solve(["ent:0"], tol=1e-12, max_iter=500)
    ref = nx.pagerank(G, alpha=0.85, personalization={"ent:0": 1.0},
                      tol=1e-12, max_iter=500)
    diff = max(abs(p[n] - ref[n]) for n in G.nodes)
    assert diff < 1e-6
