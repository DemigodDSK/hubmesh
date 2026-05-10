"""Basic smoke tests. Run with `pytest`."""
import numpy as np
import pytest

from hubmesh import Planner, Document
from hubmesh.adapters import InMemoryStore


def make_corpus(n_per_topic=15, topics=5, dim=32, seed=0):
    rng = np.random.default_rng(seed)
    centroids = rng.normal(0, 5.0, size=(topics, dim))
    docs = []
    for t in range(topics):
        for k in range(n_per_topic):
            v = centroids[t] + rng.normal(0, 0.6, size=dim)
            v = v / max(float(np.linalg.norm(v)), 1e-12)
            docs.append(Document(
                id=f"t{t}_d{k}",
                text=f"Doc topic={t} idx={k}",
                vector=v.astype(np.float32),
                metadata={"topic": t},
            ))
    return docs, centroids


def test_inmemory_store_basic():
    docs, _ = make_corpus()
    store = InMemoryStore(docs, k=5)
    assert store.dim == 32
    assert len(store.all_ids()) == len(docs)
    nbrs = store.neighbors(docs[0].id, k=3)
    assert len(nbrs) == 3
    assert docs[0].id not in nbrs  # no self-loops


def test_planner_returns_topical_results():
    docs, centroids = make_corpus()
    store = InMemoryStore(docs, k=8)
    planner = Planner(store=store)
    qvec = (centroids[2] + np.random.default_rng(1).normal(0, 0.4, size=32))
    qvec = (qvec / np.linalg.norm(qvec)).astype(np.float32)

    result = planner.retrieve(query=qvec, top_k=5, budget_tokens=1000)
    assert result.context, "should produce non-empty context"
    assert len(result.sources) > 0
    on_topic = sum(1 for s in result.sources if s.doc.metadata["topic"] == 2)
    # At minimum, majority of returned docs should be from the queried topic
    assert on_topic >= len(result.sources) // 2 + 1


def test_planner_handles_empty_subgraph_gracefully():
    docs, _ = make_corpus(n_per_topic=2, topics=1)
    store = InMemoryStore(docs, k=1)
    planner = Planner(store=store)
    # Query orthogonal to corpus — still shouldn't crash
    qvec = np.zeros(32, dtype=np.float32)
    qvec[0] = 1.0
    result = planner.retrieve(query=qvec, top_k=2, budget_tokens=500)
    assert result.sources is not None  # may be empty list, mustn't crash


def test_composite_score_finite_and_orders_correctly():
    """All-strong > one-zero > all-mediocre.

    After internal min-max, an "all 0.5" node becomes a uniform-zero node
    (because 0.5 is the min on every axis here). It therefore ranks below
    a node that's max on two axes and zero on one — which matches the
    semantic intent: 'broadly useful' beats 'uniformly mediocre'.
    """
    from hubmesh.scoring import composite_score, ScoringWeights
    R = {"strong": 1.0, "weak_R": 0.0, "middle": 0.5}
    S = {"strong": 1.0, "weak_R": 1.0, "middle": 0.5}
    C = {"strong": 1.0, "weak_R": 1.0, "middle": 0.5}
    out = composite_score(R, S, C, ScoringWeights())
    assert all(np.isfinite(v) for v in out.values())
    assert out["strong"] > out["weak_R"], f"strong should beat weak_R: {out}"
    assert out["weak_R"] > out["middle"], f"weak_R should beat middle: {out}"


def test_composite_score_no_nan_on_uniform_input():
    """When all values are identical the minmax produces 0.5 floor; output
    should be finite and equal across nodes."""
    from hubmesh.scoring import composite_score, ScoringWeights
    R = {"a": 0.7, "b": 0.7}
    S = {"a": 0.001, "b": 0.001}
    C = {"a": 0.0, "b": 0.0}
    out = composite_score(R, S, C, ScoringWeights())
    assert all(np.isfinite(v) for v in out.values())
    assert abs(out["a"] - out["b"]) < 1e-9
