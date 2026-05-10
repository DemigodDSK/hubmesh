"""Chroma adapter tests using its ephemeral (in-memory) mode."""
import numpy as np
import pytest

chromadb = pytest.importorskip("chromadb")

from hubmesh import Planner, Document
from hubmesh.adapters import ChromaStore


def make_corpus(n_per_topic=12, topics=4, dim=32, seed=0):
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


def test_chroma_store_basic():
    docs, _ = make_corpus()
    store = ChromaStore.from_documents(docs)
    assert store.dim == 32
    assert len(store.all_ids()) == len(docs)
    fetched = store.get("t0_d0")
    assert fetched.id == "t0_d0"
    assert fetched.metadata["topic"] == 0
    nbrs = store.neighbors("t0_d0", k=3)
    assert len(nbrs) == 3
    assert "t0_d0" not in nbrs


def test_chroma_store_search_and_planner():
    docs, centroids = make_corpus()
    store = ChromaStore.from_documents(docs)
    planner = Planner(store=store)
    qvec = (centroids[2] + np.random.default_rng(7).normal(0, 0.4, size=32))
    qvec = (qvec / np.linalg.norm(qvec)).astype(np.float32)
    result = planner.retrieve(query=qvec, top_k=5, budget_tokens=1000)
    on_topic = sum(1 for s in result.sources if s.doc.metadata["topic"] == 2)
    assert on_topic >= len(result.sources) // 2 + 1
