"""Qdrant adapter tests using qdrant's in-memory mode (no infra needed)."""
import numpy as np
import pytest

qdrant_client = pytest.importorskip("qdrant_client")

from hubmesh import Planner, Document
from hubmesh.adapters import QdrantStore


def make_corpus(n_per_topic=15, topics=4, dim=32, seed=0):
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


def test_qdrant_store_basic():
    docs, _ = make_corpus()
    store = QdrantStore.from_documents(docs)   # defaults to :memory:
    assert store.dim == 32
    assert len(store.all_ids()) == len(docs)
    # round-trip
    fetched = store.get("t0_d0")
    assert fetched.id == "t0_d0"
    assert "topic" in fetched.metadata
    assert fetched.metadata["topic"] == 0
    # neighbors should not include self
    nbrs = store.neighbors("t0_d0", k=3)
    assert len(nbrs) == 3
    assert "t0_d0" not in nbrs


def test_qdrant_get_many_is_batched_ordered_and_strict():
    docs, _ = make_corpus()
    store = QdrantStore.from_documents(docs)
    store._vec_cache.clear()                      # behave like a fresh process
    want = ["t3_d2", "t0_d0", "t1_d7", "t2_d14"]
    got = store.get_many(want)
    assert [d.id for d in got] == want            # order follows the request
    assert got[0].text == "Doc topic=3 idx=2" and got[0].metadata == {"topic": 3}
    assert all(d.vector is not None and d.vector.shape == (32,) for d in got)
    assert set(want) <= set(store._vec_cache)     # fetched vectors are cached
    assert store.get_many([]) == []
    with pytest.raises(KeyError):                  # same contract as get()
        store.get_many(["t0_d0", "does-not-exist"])
    # one round trip per 256 ids: a batch larger than that still returns everything, in order
    all_ids = [d.id for d in docs]
    assert [d.id for d in store.get_many(all_ids, batch_size=7)] == all_ids


def test_qdrant_store_search_and_planner():
    docs, centroids = make_corpus()
    store = QdrantStore.from_documents(docs)
    planner = Planner(store=store)

    qvec = (centroids[1] + np.random.default_rng(42).normal(0, 0.4, size=32))
    qvec = (qvec / np.linalg.norm(qvec)).astype(np.float32)

    result = planner.retrieve(query=qvec, top_k=5, budget_tokens=1000)
    assert result.context
    assert len(result.sources) > 0
    # Majority of returned docs should be from the queried topic
    on_topic = sum(1 for s in result.sources if s.doc.metadata["topic"] == 1)
    assert on_topic >= len(result.sources) // 2 + 1
