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


def test_chunk_by_sentences_round_trip():
    from hubmesh import chunk_by_sentences
    text = ("Sentence one. Sentence two. Sentence three is a bit longer. "
            "Sentence four. Sentence five! Sentence six? Sentence seven.")
    chunks = chunk_by_sentences("doc1", text, target_tokens=10,
                                overlap_sentences=1)
    assert len(chunks) >= 2
    assert all(c.metadata["source_id"] == "doc1" for c in chunks)
    # chunks should cover most of the text (with overlap, total length > original)
    total_chars = sum(len(c.text) for c in chunks)
    assert total_chars >= len(text)
    # ids are unique and well-formed
    assert len(set(c.id for c in chunks)) == len(chunks)
    assert chunks[0].id.startswith("doc1#chunk")


def test_kg_llm_with_mock_llm():
    """build_entity_kg_llm should accept any callable LLM and produce a
    valid EntityKG. We use a mock LLM that returns canned triples."""
    from hubmesh.kg_llm import build_entity_kg_llm
    docs = [
        type("D", (), {"id": "d1", "text": "Alice founded Acme.", "metadata": {}})(),
        type("D", (), {"id": "d2", "text": "Acme is based in Boston.", "metadata": {}})(),
    ]
    canned = {
        "Alice founded Acme.": '{"triples": [["Alice", "founded", "Acme"]]}',
        "Acme is based in Boston.": '{"triples": [["Acme", "based in", "Boston"]]}',
    }

    def mock_llm(prompt):
        for k, v in canned.items():
            if k in prompt:
                return v
        return '{"triples": []}'

    kg = build_entity_kg_llm(docs, llm=mock_llm)
    nodes = list(kg.graph.nodes)
    assert "ent:alice" in nodes
    assert "ent:acme" in nodes
    assert "ent:boston" in nodes
    assert "doc:d1" in nodes
    assert "doc:d2" in nodes
    # Entity-entity edges from triples
    assert kg.graph.has_edge("ent:alice", "ent:acme")
    assert kg.graph.has_edge("ent:acme", "ent:boston")
    # Predicate stored on the edge
    preds = kg.graph["ent:alice"]["ent:acme"].get("predicates", [])
    assert "founded" in preds


def test_chunk_by_chars_handles_short_text():
    from hubmesh import chunk_by_chars
    chunks = chunk_by_chars("d", "short", chunk_chars=100)
    assert len(chunks) == 1
    assert chunks[0].text == "short"
    assert chunks[0].metadata["chunk_idx"] == 0


def test_reasoning_paths_built_from_seeds_to_docs():
    """build_reasoning_paths should return non-empty paths from seed
    nodes to retrieved docs in a small bipartite KG."""
    import networkx as nx
    from hubmesh.paths import build_reasoning_paths
    G = nx.Graph()
    G.add_node("ent:alice", kind="entity")
    G.add_node("ent:bob",   kind="entity")
    G.add_node("doc:p1",    kind="doc")
    G.add_node("doc:p2",    kind="doc")
    G.add_node("doc:p3",    kind="doc")
    G.add_edge("doc:p1", "ent:alice")
    G.add_edge("doc:p2", "ent:alice")
    G.add_edge("doc:p2", "ent:bob")
    G.add_edge("doc:p3", "ent:bob")
    ppr = {"doc:p1": 0.3, "doc:p2": 0.5, "doc:p3": 0.2}
    paths = build_reasoning_paths(G, seeds=["ent:alice"],
                                  retrieved_doc_ids=["p2", "p3"],
                                  ppr_scores=ppr, max_paths=5, max_hops=4)
    # Both p2 (1 hop via alice) and p3 (3 hops via alice→p2→bob→p3) reachable
    assert len(paths) >= 1
    # First should be the higher-PPR shorter path
    assert paths[0].node_ids[0] == "ent:alice"
    assert paths[0].score > 0
