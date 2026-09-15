# hubmesh

[![tests](https://github.com/DemigodDSK/hubmesh/actions/workflows/test.yml/badge.svg)](https://github.com/DemigodDSK/hubmesh/actions/workflows/test.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://github.com/DemigodDSK/hubmesh)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/DemigodDSK/hubmesh/blob/main/LICENSE)
[![Release](https://img.shields.io/github/v/release/DemigodDSK/hubmesh?include_prereleases)](https://github.com/DemigodDSK/hubmesh/releases)

<!-- mcp-name: io.github.DemigodDSK/hubmesh -->

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

1. **Entity-anchored seeding, multi-component ranking.** In KG mode the PPR
   seeds are the question's own entities resolved against the corpus graph
   (alias index; falls back to the entities of the top cosine matches when
   the question names none); in kNN mode they are the ANN top-k. The
   multi-component score — cosine relevance, pooled PPR mass, and
   multi-anchor convergence, min-max normalized and fused 3:1:1 — is
   applied to the *document ranking*, not to seed choice.
2. **Budget-aware context packing.** Once relevant entities are scored, pack them into
   the LLM's context window with explicit coverage and redundancy control rather than
   just truncating top-k.

The multi-component scoring pattern is adapted from the NNSI framework
([Naidu et al., CCIS 2934, Springer, 2026](https://doi.org/10.1007/978-3-032-22190-2_1))
for SDN topology optimization, repurposed here for retrieval planning.

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

store = QdrantStore.from_documents(docs)                          # in-memory
store = QdrantStore.from_documents(docs, path="./qdrant_data")    # on-disk
store = QdrantStore.from_documents(docs, url="http://localhost:6333")  # remote

planner = Planner(store=store, embed=embed)
result = planner.retrieve(query="...", top_k=10)
```

### Chroma adapter

```python
from hubmesh.adapters import ChromaStore

store = ChromaStore.from_documents(docs)                          # ephemeral
store = ChromaStore.from_documents(docs, persist_directory="./chroma_data")
store = ChromaStore.from_documents(docs, host="localhost", port=8000)
```

### Multi-hop / KG mode

```python
from hubmesh.kg import build_entity_kg
import spacy

nlp = spacy.load("en_core_web_sm")
kg = build_entity_kg(docs, nlp=nlp)

planner = Planner(store=store, kg=kg, nlp=nlp, embed=embed)   # embed= needed for text queries
result = planner.retrieve(query="Where was the founder of the company that bought Slack born?",
                          top_k=10, budget_tokens=4000)

# RetrievalResult includes reasoning paths showing why each doc was returned
for path in result.reasoning:
    print(f"  score={path.score:.3f}  {' → '.join(path.node_ids)}")
```

### LLM-extracted KG (richer than spaCy)

```python
from hubmesh.kg_llm import build_entity_kg_llm
from hubmesh.entity_linker import EmbeddingLinker, make_st_embedder

def llm(prompt):  # provider-agnostic — bring your own
    return your_llm_call(prompt)

kg = build_entity_kg_llm(docs, llm=llm, cache_path="kg_cache.json")

# optional: cross-document entity dedup — same Linker protocol as the spaCy path
kg = build_entity_kg_llm(docs, llm=llm, cache_path="kg_cache.json",
                         linker=EmbeddingLinker(embed=make_st_embedder()),
                         llm_identity="gpt-5-mini")   # namespaces the cache

planner = Planner(store=store, kg=kg, nlp=nlp, embed=embed)
```

### Better entity linking

```python
from hubmesh.kg import build_entity_kg
from hubmesh.entity_linker import EmbeddingLinker, make_st_embedder

# Cluster surface variations: "United States" / "U.S." / "USA" → one entity
linker = EmbeddingLinker(embed=make_st_embedder(), threshold=0.82)
kg = build_entity_kg(docs, linker=linker)
```

### Iterative multi-hop: let your agent drive

```python
r1 = planner.retrieve(query=question, top_k=5)

# your agent reads r1, spots the bridge entity, then aims hop 2 at it:
r2 = planner.retrieve(
    query=question, top_k=5,
    seed_entities=["Nimbus Analytics"],           # merged with the query's own seeds
    exclude_docs=[s.doc.id for s in r1.sources],  # don't re-retrieve consumed docs
)
```

Seed mentions resolve through the alias index, so free-text entity names
work. The query path stays deterministic and LLM-free — the planning
intelligence lives in the caller.

### MCP server: plug hubmesh into any agent

```bash
pip install "hubmesh[mcp]"
python -m spacy download en_core_web_sm
```

```json
{"mcpServers": {"hubmesh": {"command": "hubmesh-mcp"}}}
```

Exposes the planner as deterministic operator tools over stdio —
`index_corpus`, `retrieve` (seed-steerable, as above), `resolve_entities`,
`entity_neighbors`, `path_between`, `get_document`, `graph_stats`,
`list_corpora`. Your agent is the solver: it decomposes the question,
reads each hop, and aims the next one; the server answers in
milliseconds with zero LLM calls. Corpora persist as plain JSON/NPZ
under `~/.hubmesh/corpora`.

The server warms up models and persisted corpora in the background at
launch (~5-10s on first run), so tool calls stay fast from the start —
relevant for strict-timeout connector clients (Perplexity, etc.).

For web-based connector clients, serve SSE natively — no gateway
process needed:

```bash
export HUBMESH_API_KEY="$(openssl rand -hex 24)"   # any strong secret
hubmesh-mcp --transport sse --port 8000 --allow-tunnel
ngrok http 8000     # paste https://<your-url>/sse into the connector
```

Tunneled serving **requires** the API key (the server refuses to start
without one) and defaults to **read-only** — pass `--allow-writes` to
keep `index_corpus` enabled. Clients must send
`Authorization: Bearer <key>`. If your connector client cannot set
headers, the tunnel edge must **authenticate callers itself** (ngrok
OAuth / IP-restriction traffic policy, Cloudflare Access, …) *before*
it adds the upstream header — injecting the header for anonymous
traffic hands every caller full read access (read-only protects corpora
from replacement, not from disclosure; `get_document` returns full
text). A client that can neither send the header nor sit behind an
authenticating edge is unsupported for private corpora.

Tunnel field notes (from a live Perplexity integration): **ngrok works**
(free tier included); **cloudflared quick tunnels buffer SSE bodies**
and hang tool calls; **supergateway is unnecessary** here and crashes
on reconnect. `--allow-tunnel` accepts the tunnel's forwarded Host
header — without it, proxied requests get 421 Misdirected Request.

Full field report — setup, error decoder, a 9/9 test battery run
through Perplexity, and two findings about reasoning-model behaviour —
in [docs/perplexity.md](docs/perplexity.md).

### Chunking long documents

```python
from hubmesh import chunk_by_sentences, chunk_documents

chunks = chunk_documents(
    [{"id": "doc1", "text": long_text}, ...],
    strategy="sentences", target_tokens=200,
)
# Then embed chunks and index normally
```

## Installation

```bash
pip install hubmesh                   # core
pip install "hubmesh[qdrant]"         # Qdrant adapter
pip install "hubmesh[chroma]"         # Chroma adapter
pip install "hubmesh[kg]"             # entity-linked KG (spaCy)
pip install "hubmesh[linker]"         # embedding-based entity linker
pip install "hubmesh[all]"            # everything
python -m spacy download en_core_web_sm   # required for KG mode
```

## Design

KG mode — the benchmarked, production path:

```
query ─► spaCy NER ─► alias index ─► entity seeds ─► Personalized PageRank over the corpus KG
  │                   (fallback: entities of the top-3 cosine documents)              │
  └───────────► cosine similarity against every document ─────────────────────────────┤
                                                                                      ▼
            3·minmax(cosine) + 1·minmax(pooled PPR) + 1·minmax(per-anchor geomean)   [weighted sum]
                                                                                      ▼
                          budget-aware packing ─► context + sources + reasoning paths
```

kNN mode (no KG; prototyping): first-pass ANN → capped induced proximity
subgraph → PPR from the ANN seeds → the same scoring and packing.
Community anchoring exists for single-topic retrieval and is off by
default.

Each layer is independently testable and replaceable. Adapters wrap your
existing vector DB so you don't have to migrate — note that KG mode
scores every document (vectors are gathered once per store version and
cached) and uses the store's ANN index only for the seed fallback.

## Benchmarks

Supporting-fact paragraph recall over pooled distractor corpora. Every
row is a separate experiment: **document representation and embedding
model change the absolute numbers materially**, so rows are never
compared across representations. Protocol, ablations and limits are in
[BENCHMARKS.md](BENCHMARKS.md).

**Full HotpotQA dev (7,405 questions, 66,581 pooled paragraphs),
hubmesh vs naive cosine, v0.4 defaults:**

| representation · embedding | naive @10 | hubmesh @10 | Δ @10 | Δ @5 | Δ @2 |
|---|---:|---:|---:|---:|---:|
| body only · MiniLM-L6 | 69.3% | 75.2% | **+5.90** | +4.21 | −0.75 |
| title+body · MiniLM-L6 | 70.0% | 77.3% | **+7.24** | +5.88 | +0.25 |
| title+body · **bge-m3** | 83.5% | 84.8% | +1.38 | **−1.41** | **−9.09** |

Read both directions. With a small embedding the graph layer adds 5–7
points of depth recall; with a strong one the depth gain shrinks to
+1.4 and the defaults **hurt the top ranks** (−9.1 at recall@2). The
convergence term trades top-rank precision for depth: for top-2/top-5
workloads on strong embeddings use `use_convergence=False` or plain
cosine, and evaluate on your own workload before turning the graph
layer on everywhere.

**Full MuSiQue-Ans dev (2,417 questions, MiniLM, body only), hubmesh vs
naive, recall@10 with paired 95% CIs:** **+3.67** [+3.00, +4.39] overall
(+2.2 at @2, +3.3 at @5); by hop count +2.9 / +4.1 / **+5.3**
(n = 1,252 / 760 / 405). The gain grows with hop count, and on MuSiQue
hubmesh beats naive at recall@2 as well.

**What the scoring adds (HotpotQA N=500, body only, recall@10):** on the
*same* graph, seeds, fallback and packer, cosine-fused scoring reaches
0.871 against 0.676 for the pure structural (PPR-only) signal —
**+19.5 pts** [+16.3, +22.7] (MuSiQue N=300: +16.3). Earlier versions
quoted +29.8 against a HippoRAG-style ranker; that comparison also
changed the pipeline and is no longer cited as scoring attribution.

**Convergence term (default on):** +0.9 pts @10 over convergence-off on
full MuSiQue dev and +1.1 on HotpotQA N=500. A single-solve log-pooled
signal in the same slot matches it in aggregate; the geomean keeps ~1 pt
at three and four hops. Multi-seed queries cost ~1.5–2× (still zero LLM
tokens, deterministic).

Latency: **~22 ms** mean / 26 ms p95 per query on a 7K-node KG (after PPR
matrix caching). ~3 s/query was measured at the 66K-paragraph full-dev
scale with convergence on, before the per-query vector re-gather was
removed; that scale has not been re-measured since.

Reproduce (each run writes a JSON with per-query records and a manifest
carrying the commit, dirty flag and source/harness content hashes):
```bash
python benchmarks/run_hotpotqa.py --n 500 --kg --out hotpot.json
python benchmarks/run_musique.py  --n 300 --kg --out musique.json
python benchmarks/run_ablation_coherence.py --dataset musique --n 2417
python benchmarks/profile_query.py        # latency profile
```

## Status

Pre-alpha (v0.4.1). Core algorithms implemented and validated; adapters for
in-memory, Qdrant, and Chroma; entity-linked KG with both spaCy NER and
LLM-based extraction (both linker-aware); alias-indexed entity resolution;
NNSI-KG scoring (multi-source convergence default-on, hub-discounted PPR
opt-in); agent-driven iterative multi-hop via `seed_entities` /
`exclude_docs`; MCP operator server (`hubmesh-mcp`, native SSE) with
JSON/NPZ corpus persistence; document chunking; reasoning-path
explanation; PPR-cache latency optimisation. Pinecone / pgvector / Weaviate adapters
and additional multi-hop benchmarks are tracked as
[good first issues](https://github.com/DemigodDSK/hubmesh/issues).

## Acknowledgements

The multi-component scoring pattern is adapted from the **Network Node Significance
Index (NNSI)** framework introduced in:

> D. S. K. Naidu et al., "A Framework for Improving Network Topology
> Based on Graph Theory in Software-Defined Networking," in *Internet Computing,
> Internet of Things, Artificial Intelligence, and Applications*, Communications
> in Computer and Information Science, vol. 2934, H. R. Arabnia, L. Deligiannidis,
> K. Ferens, F. Ghareh Mohammadi, F. Shenavarmasouleh, and S. Amirian, Eds.
> Cham: Springer, 2026, pp. 3–18.
> doi: [10.1007/978-3-032-22190-2_1](https://doi.org/10.1007/978-3-032-22190-2_1)

```bibtex
@inproceedings{naidu2026nnsi,
  author    = {Naidu, Datta Sai Krishna and others},
  title     = {A Framework for Improving Network Topology Based on Graph Theory
               in Software-Defined Networking},
  booktitle = {Internet Computing, Internet of Things, Artificial Intelligence,
               and Applications},
  series    = {Communications in Computer and Information Science},
  volume    = {2934},
  editor    = {Arabnia, Hamid R. and Deligiannidis, Leonidas and Ferens, Ken and
               Ghareh Mohammadi, Farid and Shenavarmasouleh, Farzan and
               Amirian, Soheyla},
  pages     = {3--18},
  publisher = {Springer},
  address   = {Cham},
  year      = {2026},
  doi       = {10.1007/978-3-032-22190-2_1},
  isbn      = {978-3-032-22189-6}
}
```

NNSI is repurposed here from SDN topology optimization to retrieval planning; the
application to retrieval over an entity-linked KG is new to this work.

## License

MIT
