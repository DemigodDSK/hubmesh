"""Smoke test: end-to-end retrieval on a tiny synthetic corpus.

Generates 10 topical clusters of fake docs, runs a query, and prints the
ranked context. Verifies the whole pipeline wires together without error
and produces sensible scores.
"""
import numpy as np
from hubmesh import Planner
from hubmesh.adapters import InMemoryStore
from hubmesh.types import Document


def make_corpus(n_per_topic=20, topics=10, dim=64, seed=0):
    rng = np.random.default_rng(seed)
    centroids = rng.normal(0, 5.0, size=(topics, dim))
    docs = []
    for t in range(topics):
        for k in range(n_per_topic):
            v = centroids[t] + rng.normal(0, 0.8, size=dim)
            v = v / max(float(np.linalg.norm(v)), 1e-12)
            docs.append(Document(
                id=f"t{t}_d{k}",
                text=f"Document about topic {t}, instance {k}. "
                     f"Synthetic corpus filler text used for smoke testing hubmesh.",
                vector=v.astype(np.float32),
                metadata={"topic": t},
            ))
    return docs, centroids


def main():
    docs, centroids = make_corpus()
    store = InMemoryStore(docs, k=8)
    planner = Planner(store=store)

    # query for topic 3
    query_vec = centroids[3] + np.random.default_rng(99).normal(0, 0.5, size=64)
    query_vec = query_vec / max(float(np.linalg.norm(query_vec)), 1e-12)

    result = planner.retrieve(
        query=query_vec.astype(np.float32),
        top_k=10,
        budget_tokens=2000,
    )

    print("=" * 70)
    print(f"hubmesh end-to-end demo")
    print("=" * 70)
    print(f"Corpus: {len(docs)} docs, {len(centroids)} topics")
    print(f"Query: synthetic topic 3 centroid + noise")
    print(f"Subgraph: {result.debug['subgraph_nodes']} nodes / "
          f"{result.debug['subgraph_edges']} edges, "
          f"{result.debug['communities']} communities, "
          f"anchor size {result.debug['anchor_size']}")
    print(f"PPR seeds: {result.debug['ppr_seeds']}")
    print()
    print("Top retrievals (with topic id in metadata — should be mostly topic 3):")
    print(f"{'rank':>4}  {'doc_id':<10}  {'topic':>5}  {'sim':>7}  "
          f"{'ppr':>7}  {'composite':>10}")
    for r, src in enumerate(result.sources, 1):
        print(f"{r:>4}  {src.doc.id:<10}  {src.doc.metadata['topic']:>5}  "
              f"{src.similarity:>7.3f}  {src.ppr_score:>7.4f}  "
              f"{src.composite_score:>10.4f}")
    in_topic = sum(1 for s in result.sources if s.doc.metadata["topic"] == 3)
    print()
    print(f"On-topic hits: {in_topic}/{len(result.sources)}")


if __name__ == "__main__":
    main()
