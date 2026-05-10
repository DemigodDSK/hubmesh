# hubmesh

**Centrality-aware GraphRAG retrieval planner. Drop-in layer over any vector DB.**

`hubmesh` is a Python library that improves multi-hop RAG quality on top of an existing
vector database. You don't replace your infrastructure — you add a smart planner between
your vector DB and your LLM.

## What problem this solves

Naive vector retrieval ("embed query, get top-k by cosine similarity") fails on multi-hop
questions like *"Where was the founder of the company that acquired Slack born?"* The
correct answer requires retrieving entities along a reasoning path, not the single most
similar item.

GraphRAG and HippoRAG showed that running a small Personalized PageRank over a knowledge
graph at query time can substantially improve multi-hop retrieval. `hubmesh` extends
that line with two contributions:

1. **Multi-component seed selection.** Instead of picking PPR seeds by raw query
   similarity (which picks wrong-community seeds at high feature overlap), seeds are
   chosen by a multi-component score combining query relevance, structural fit, and
   coverage diversity.
2. **Budget-aware context packing.** Once relevant entities are scored, pack them into
   the LLM's context window with explicit coverage and redundancy control rather than
   just truncating top-k.

The multi-component scoring pattern is adapted from the NNSI framework
([Naidu & Modarresi, iComp 2025](https://example.invalid)) for SDN topology
optimization, repurposed here for retrieval planning.

## Quickstart

```python
from hubmesh import Planner
from hubmesh.adapters import InMemoryStore

# bring your own embeddings (callable taking text -> np.ndarray)
embed = ...
documents = [...]   # list of strings or {id, text, vector, metadata}

store = InMemoryStore.from_documents(documents, embed=embed)
planner = Planner(store=store, embed=embed)

result = planner.retrieve(
    query="Where was the founder of the company that bought Slack born?",
    top_k=10,
    budget_tokens=4000,
)
print(result.context)       # packed context string
print(result.sources)       # ranked source documents with scores
print(result.reasoning)     # paths through the graph (when multi-hop)
```

## Design

```
query → first-pass ANN  → induced subgraph → multi-component scoring
                              ↓                        ↓
                       community anchoring → Personalized PageRank
                              ↓                        ↓
                              └─────→ ranking → budget-aware packing → context
```

Each layer is independently testable and replaceable. Adapters wrap your existing vector
DB so you don't have to migrate.

## Benchmarks

**HotpotQA dev, paragraph retrieval, N=500 questions, supporting-fact recall:**

| Strategy | recall@2 | recall@5 | recall@10 |
|---|---|---|---|
| naive cosine top-k | 57.8% | 74.0% | 81.9% |
| **hubmesh** (cosine + PPR multi-component) | **57.2%** | **76.9%** | **85.6%** |
| HippoRAG-style (PPR-only over same KG) | 35.0% | 49.6% | 56.5% |

- vs naive cosine: **+2.9 / +3.7 pts** at recall@5/10 (the multi-hop regime)
- vs PPR-only on the same KG: **+27.3 / +29.1 pts** at recall@5/10

The PPR-only ablation isolates the multi-component scoring contribution: combining
cosine with graph-structural PPR via the multiplicative recipe is what makes both
signals usable together. Either alone loses substantial recall.

Latency: ~535ms/query in KG mode (spaCy NER + global PPR; both optimizable).

Reproduce: `python benchmarks/run_hotpotqa.py --n 500 --kg`

## Status

Pre-alpha. Core algorithms implemented and validated; in-memory adapter only;
production adapters (Pinecone/Qdrant/Weaviate/pgvector) and harder multi-hop
benchmarks (MuSiQue, 2WikiMultiHopQA) on the roadmap.

## Acknowledgements

The multi-component scoring pattern is adapted from the **Network Node Significance
Index (NNSI)** framework introduced in
[Naidu & Modarresi, "A Framework for Improving Network Topology Based on Graph
Theory in Software-Defined Networking", iComp 2025](#) — repurposed here from
SDN topology optimization to retrieval planning.

## License

MIT
