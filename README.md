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
([Naidu Dsk, iComp 2025](https://example.invalid)) for SDN topology
optimization, repurposed here for retrieval planning.

## Quickstart

### In-memory (testing, small corpora)

```python
from hubmesh import Planner
from hubmesh.adapters import InMemoryStore

embed = ...   # callable: text -> np.ndarray
docs = [...]  # list of Document or strings or dicts

store = InMemoryStore.from_documents(docs, embed=embed)
planner = Planner(store=store, embed=embed)
result = planner.retrieve(query="...", top_k=10, budget_tokens=4000)
```

### Qdrant adapter (production)

```python
from hubmesh import Planner
from hubmesh.adapters import QdrantStore

# Local: Qdrant in-memory or on-disk
store = QdrantStore.from_documents(docs)                          # in-memory
store = QdrantStore.from_documents(docs, path="./qdrant_data")    # on-disk
# Remote
store = QdrantStore.from_documents(docs, url="http://localhost:6333")

planner = Planner(store=store, embed=embed)
result = planner.retrieve(query="...", top_k=10)
```

### Multi-hop / KG mode

```python
from hubmesh.kg import build_entity_kg
import spacy

nlp = spacy.load("en_core_web_sm")
kg = build_entity_kg(docs, nlp=nlp)

planner = Planner(store=store, kg=kg, nlp=nlp)
result = planner.retrieve(query="Where was the founder of the company that bought Slack born?",
                          top_k=10, budget_tokens=4000)
```

## Installation

```bash
pip install hubmesh                   # core
pip install "hubmesh[qdrant]"         # with Qdrant adapter
pip install "hubmesh[kg]"             # with entity-linked KG support (spaCy)
pip install "hubmesh[all]"            # everything
python -m spacy download en_core_web_sm   # required for KG mode
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

**Headline:** on multi-hop QA, hubmesh's KG mode beats both naive cosine
retrieval and a HippoRAG-style PPR-only ablation that uses the same KG.
The win grows with hop count — exactly the regime where graph-structural
retrieval should help most.

| Benchmark | Setting | recall@10 vs naive |
|---|---|---:|
| HotpotQA dev, N=500 | KG mode | **+3.7 pts** |
| MuSiQue dev, N=300, 2-hop | KG mode | **+1.7 pts** |
| MuSiQue dev, N=300, 3-hop | KG mode | **+1.9 pts** |
| MuSiQue dev, N=300, 4-hop | KG mode | **+2.8 pts** |

vs PPR-only ablation on the same KG: **+29.1 pts** on HotpotQA — the
multi-component scoring is doing the work, not just "having a graph."

Latency: **~22 ms** mean / 26 ms p95 per query on a 7K-node KG (after PPR
matrix caching).

See [BENCHMARKS.md](BENCHMARKS.md) for the full methodology, ablations,
per-hop breakdown, and notes on what this proves and doesn't.

Reproduce:
```bash
python benchmarks/run_hotpotqa.py --n 500 --kg
python benchmarks/run_musique.py  --n 300 --kg
python benchmarks/profile_query.py        # latency profile
```

## Status

Pre-alpha. Core algorithms implemented and validated; in-memory adapter only;
production adapters (Pinecone/Qdrant/Weaviate/pgvector) and harder multi-hop
benchmarks (MuSiQue, 2WikiMultiHopQA) on the roadmap.

## Acknowledgements

The multi-component scoring pattern is adapted from the **Network Node Significance
Index (NNSI)** framework introduced in
[Naidu Dsk, "A Framework for Improving Network Topology Based on Graph
Theory in Software-Defined Networking", iComp 2025](#) — repurposed here from
SDN topology optimization to retrieval planning.

## License

MIT
