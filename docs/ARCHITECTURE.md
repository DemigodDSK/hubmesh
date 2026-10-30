# hubmesh architecture

Version 0.4.2, commit `52604f9`, written 2026-10-30. Paths are relative to the
repository root; line counts are for orientation and will drift.

hubmesh is a retrieval planner that sits beside an existing vector database.
The database stays the system of record for documents and embeddings. hubmesh
adds an entity graph built once at index time, and at query time ranks
documents by a weighted combination of cosine relevance and two graph signals
computed with Personalized PageRank. It makes no LLM calls at query time, its
output order is deterministic across processes, and it talks to any store
through an eight-method adapter contract. One design decision explains most of
what follows: the planner owns the ranking and scores every document itself,
rather than re-ranking the database's own top-k. That is where its recall gain
on multi-hop questions comes from, and it is also the source of its scaling
limit at large corpora (section 8).

## 1. System at a glance

```
INDEX TIME (once per corpus, or on rebuild)

  source docs ──► chunking.py ──► your embedder ──► VectorStore adapter ──► vector DB
                      │                                   (InMemory / Qdrant / Chroma)
                      └──► kg.py  spaCy NER ──► canonicalise ──► EntityKG (bipartite)
                           or kg_llm.py (LLM triples)   │
                                                        ├──► ppr.PPRSolver  (transition matrix, cached)
                                                        └──► corpus.CorpusManager (JSON/NPZ generations)

QUERY TIME (Planner._retrieve_kg, zero LLM calls)

  question ──► NER ──► alias index ──► entity seeds ──┬──► pooled PPR ─────────────► S  (structural)
      │                 (fallback: entities of the      └──► per-seed PPR, geomean ──► C  (convergence)
      │                  store's top-3 documents)
      └──► query vector ──► cosine against every document (cached matrix) ──► R  (relevance)

  score = (3·minmax(R) + 1·minmax(S) + 1·minmax(C)) / 5     [weighted sum, default weights]
      ──► sort (score desc, id asc) ──► fetch top 5k bodies in one batch ──► pack to token budget
      ──► RetrievalResult(context, sources, reasoning paths, debug)
```

Module map (`src/hubmesh/`):

| module | lines | role |
|---|---:|---|
| `planner.py` | 465 | `PlannerConfig`, `Planner`, both retrieval modes, the per-store vector matrix cache |
| `kg.py` | 273 | `EntityKG`, `build_entity_kg` (spaCy NER), alias index, `extract_query_entities` |
| `kg_llm.py` | 362 | `build_entity_kg_llm`: triples from any `(prompt) -> str` callable, content-hash cache |
| `entity_linker.py` | 144 | `Linker` protocol, `SubstringLinker`, `EmbeddingLinker` |
| `ppr.py` | 188 | `personalized_pagerank` (networkx wrapper, kNN mode) and `PPRSolver` (sparse, cached) |
| `scoring.py` | 146 | min-max normalisation, `composite_score` with `sum` and `geom` integration |
| `packing.py` | 94 | greedy budgeted packing with a near-duplicate filter |
| `paths.py` | 107 | shortest seed-to-document paths for explanation |
| `graph.py` | 94 | induced kNN subgraph and Louvain anchoring (kNN mode only) |
| `chunking.py` | 174 | sentence and character chunkers that emit `Document`s |
| `corpus.py` | 564 | `CorpusManager`: named corpora on disk, generations, planner cache |
| `mcp_server.py` | 389 | `hubmesh-mcp`: eight deterministic tools over stdio or SSE, auth policy |
| `adapters/base.py` | 49 | the `VectorStore` protocol |
| `adapters/inmemory.py` | 148 | reference store, brute-force cosine, kNN graph |
| `adapters/qdrant.py` | 262 | Qdrant collection as a store |
| `adapters/chroma.py` | 189 | Chroma collection as a store |
| `types.py` | 42 | `Document`, `ScoredDocument`, `ReasoningPath`, `RetrievalResult` |

Runtime dependencies are numpy, networkx and scipy. spaCy (`hubmesh[kg]`),
sentence-transformers (`hubmesh[linker]`), qdrant-client and chromadb are
optional extras, imported lazily where they are used.

## 2. Data model

`Document(id, text, vector, metadata)` is the unit of retrieval. Chunkers emit
documents with `vector=None`; stores embed them on ingest when given an
`embed` callable.

`EntityKG` is an undirected networkx graph with two node kinds and two edge
kinds:

- `doc:<doc_id>` nodes, one per document.
- `ent:<canonical>` nodes, one per canonicalised entity.
- `mentions` edges between a document and each entity it names (weight 1).
- `co_occurs` edges between two entities named in the same document, with
  `weight` counting how many documents name both.

Beside the graph it keeps `doc_to_entities`, `entity_to_docs`, the
canonical-to-node map, display labels (the longest surface form seen), and
`alias_to_node`, an index from every canonicalised surface form to its node.
Node insertion order is sorted, so the PageRank matrix has the same row order
in every process.

`RetrievalResult` carries the packed `context` string, the ranked `sources`
(each a `ScoredDocument` with cosine, PageRank mass, composite score and
rank), `reasoning` paths, and a `debug` dict with the mode, the query
mentions, the seeds used and the graph size.

## 3. Index time

**Chunking** (`chunking.py`). `chunk_by_sentences` splits on sentence
boundaries toward a soft token target; `chunk_by_chars` uses a fixed window
with overlap (defaults 800 and 100 characters). Chunk ids are
`<source_id>#chunk<i>` and the source id is kept in metadata.

**Embedding.** hubmesh never chooses the embedder. `Planner` takes an
`embed` callable for queries; stores take one for ingest; `CorpusManager`
defaults to a lazily loaded sentence-transformers model named by
`HUBMESH_EMBED_MODEL` and records its identity in `meta.json` so a corpus
cannot be queried with a different model of the same dimension.

**Graph construction** (`kg.py`). `build_entity_kg` runs spaCy over every
document and keeps entities with eleven labels (PERSON, ORG, GPE, LOC, NORP,
FAC, EVENT, WORK_OF_ART, PRODUCT, LAW, LANGUAGE). Canonicalisation lowercases,
strips punctuation and collapses whitespace. Within one document, a shorter
mention that is a word-bounded substring of a longer one is absorbed by it
("Derrickson" into "scott derrickson"). A `linker` replaces that heuristic
with a corpus-wide mapping: `EmbeddingLinker` clusters mentions by cosine
similarity over a threshold and names each cluster by its longest member.
`kg_llm.build_entity_kg_llm` is the alternative builder: it asks any LLM
callable for subject-predicate-object triples per passage, caches them by a
namespace of model identity, prompt hash and passage hash, and returns the
same `EntityKG` shape, so the rest of the pipeline does not know which
builder ran. The LLM cost is paid at index time only.

Measured on this project's laptop (Apple M1, 8 GB, while a Docker VM held
4 GB): the HotpotQA full dev corpus of 66,581 paragraphs produced 323,973
nodes and 2,948,424 edges in 1,710 s, about 25 ms per paragraph.

**PageRank matrix** (`ppr.PPRSolver`). At `Planner` construction the graph
becomes a CSR matrix: edge weights, optionally divided by `log(e+deg)^γ` of
entity endpoints (`hub_discount`, default 0, so off), row-normalised, and
stored transposed. Degree-zero nodes are recorded as dangling. This is the
one-time cost that makes per-query PageRank a sequence of sparse
matrix-vector products.

**Document vector matrix** (`Planner._kg_doc_matrix`). On the first query
against a store version, the planner pulls the vector of every document node
through `vector_of`, unit-normalises them and caches the matrix keyed by the
store's `mutation_counter`. Against a remote store this is the cold-attach
cost: one round trip per document unless the adapter prefilled its cache
during ingest. On Qdrant 1.20 over localhost it measured 2.3 ms per vector,
150 s for 66,581 documents.

## 4. Query time, KG mode

`Planner.retrieve(query, top_k=10, budget_tokens=4000, query_vec=None,
seed_entities=None, exclude_docs=None)` routes to `_retrieve_kg` when a graph
is attached and the query is text. The steps, in order:

1. **Seeds.** `extract_query_entities` runs NER on the question;
   `EntityKG.query_entity_nodes` resolves each mention through the alias
   index, then exact canonical match, then substring match (longest canonical
   wins, ties by node id). Caller-supplied `seed_entities` are resolved the
   same way and merged in front of the question's own seeds.
2. **Fallback.** If no seed resolves, the planner asks the store for its top-3
   documents by cosine and uses their entities as seeds, capped at eight.
   This is the only place the store's own search touches the ranking.
3. **Pooled PageRank.** `PPRSolver.solve(seeds, alpha=0.15)` with the restart
   vector uniform over the distinct seeds; power iteration, at most 50 steps,
   stopping when the L1 change drops below 1e-6. Dangling mass is returned
   through the restart vector. The mass at each `doc:` node is the structural
   score S.
4. **Relevance.** One matrix product of the cached unit matrix with the unit
   query vector gives cosine R for every document. Documents in
   `exclude_docs` are dropped here.
5. **Convergence** (`use_convergence=True`, needs at least two seeds).
   `solve_multi` runs one batched power iteration with the first four seeds as
   separate restart columns. C for a document is the geometric mean of its
   mass from each seed: a document must be reachable from every anchor, not
   flooded from one. With one seed, C is neutral.
6. **Composite.** `composite_score` min-max normalises R, S and C over all
   scored documents and combines them. Default weights are 3, 1, 1 with
   `integration="sum"`; `"geom"` is the log-geometric mean for workloads
   where every component must be strong. Sorting is by score descending,
   then id ascending, so ties never depend on hash seeds.
7. **Bodies and packing.** The top `5 × top_k` ids are fetched in one
   `get_many` call (falling back to per-id `get` if an id is stale).
   `packing.pack` walks them best-first, measures the complete serialised
   context against the token budget with the configured counter (chars/4 by
   default, or a tokenizer callable), skips a candidate whose cosine to an
   already picked document exceeds 0.95 when its redundancy-adjusted score
   goes negative (`redundancy_lambda=0.3`), and stops at `top_k` documents.
   The context is built from exactly the picked list.
8. **Explanation.** `paths.build_reasoning_paths` finds the shortest path in
   the graph from any seed to each picked document, at most four hops, scores
   paths by length and end-to-end mass, and returns up to five.

The `debug` dict records the mode, mentions, up to ten seeds, injected seeds,
the number of excluded documents and the graph size, which is what the MCP
server surfaces as `seeds_used`.

**Iterative multi-hop.** An agent reads one result, notices the bridge entity,
and calls again with it in `seed_entities` and the consumed ids in
`exclude_docs`. The planner stays stateless; the loop lives in the caller.

**kNN mode** is the path without a graph: the store's top-`seed_top_k`
documents become seeds, `graph.build_induced_subgraph` expands two hops over
the store's `neighbors` with a 2,000-node cap and deterministic visit order,
`personalized_pagerank` runs over that subgraph, and the same scoring and
packing follow. Louvain community anchoring and the coherence term exist for
single-topic retrieval and are off by default because they hurt multi-hop
questions. All published numbers are from KG mode.

## 5. The store contract and the adapters

`adapters/base.py` defines `VectorStore` as a runtime-checkable protocol:

| member | planner use |
|---|---|
| `search(query_vec, top_k)` | kNN-mode seeds; KG-mode fallback seeds |
| `get(doc_id)` | per-id fallback when a batch contains a stale id |
| `get_many(doc_ids)` | bodies of the top 5k candidates, one call |
| `neighbors(doc_id, k)` | kNN-mode subgraph expansion only |
| `all_ids()` | corpus-wide operations (CorpusManager build, benchmarks) |
| `vector_of(doc_id)` | the document matrix and the packer's duplicate check; required |
| `dim` | embedding dimension |
| `mutation_counter` | optional; a write increments it and invalidates the planner's matrix cache |

The planner refuses construction if an adapter lacks `vector_of`.

**InMemoryStore** holds a float32 matrix and answers `search` by brute-force
cosine. It precomputes a k-nearest-neighbour graph eagerly up to 5,000
documents and lazily above that; KG mode never uses it.

**QdrantStore** wraps a collection and creates it with cosine distance if
missing. String ids become stable 63-bit integers through a blake2b hash; the
original id, the text and `meta_`-prefixed metadata travel in the payload.
`search` is `query_points`, `get_many` retrieves in batches of 256 ids in the
caller's order and raises `KeyError` for a missing id, `neighbors` is derived
by searching with the document's own vector, `all_ids` scrolls 1,024 points
at a time, and vectors are cached on first lookup. `upsert` increments
`mutation_counter`. Embedded (`:memory:`), on-disk and remote clients all
work; only the remote server path has been measured (section 8).

**ChromaStore** is the same shape over a Chroma collection in ephemeral,
persistent or HTTP mode; Chroma accepts string ids natively so no hashing is
needed.

A new backend (Pinecone, pgvector, Weaviate) is one class implementing the
eight members plus the counter, roughly 200 lines by the two existing
examples, and `tests/test_qdrant_adapter.py` shows the behaviours to test.

## 6. Persistence and serving

**CorpusManager** (`corpus.py`) stores a named corpus as plain files, never as
serialised Python objects, so a corpus directory can be inspected, diffed and
shared, and loading one cannot execute code. Under `<root>/<name>/` each
build writes an immutable generation directory `gen-<hex8>/` containing
`meta.json` (format version 2, counts, embedder identity and dimension),
`docs.jsonl`, `vectors.npz` and `kg.json`, then atomically replaces a
`CURRENT` pointer file. Writers serialise on a per-corpus `flock`; readers
resolve the pointer once and stay inside that generation, and a superseded
generation survives for `retention_seconds` (900 by default) so in-flight
readers finish. On load the manager rejects a newer format, refuses an
embedder identity that differs from the stored one, and checks the stored
dimension against the array. Planners are cached per corpus, per
`PlannerConfig` and per generation, so a rebuild published by another
process is picked up on the next call. The default root is
`~/.hubmesh/corpora` (`HUBMESH_CORPORA_ROOT` overrides).

**hubmesh-mcp** (`mcp_server.py`) exposes the manager as MCP tools built on
FastMCP: `list_corpora`, `index_corpus`, `retrieve`, `resolve_entities`,
`entity_neighbors`, `path_between`, `get_document` and `graph_stats`. The
division of labour is deliberate: the calling agent decomposes the question
and decides what to look for next; the server only retrieves, resolves and
traverses, with no LLM call of its own. `retrieve` returns 280-character
snippets with cosine, graph score and composite score per source, the seeds
used and up to five reasoning paths; full text comes from `get_document`.
Tool parameters use empty-list defaults rather than optional unions because
at least one connector refused to compile union-typed schemas.

Transport is stdio by default or HTTP+SSE through uvicorn. `resolve_security`
encodes the exposure rules: stdio needs no key; any bind beyond loopback, or
`--allow-tunnel`, requires `--api-key` or `HUBMESH_API_KEY` and the server
exits otherwise; tunnel mode is read-only unless `--allow-writes` is explicit;
a key given on loopback is still enforced. `BearerAuthASGI` checks
`Authorization: Bearer <key>` with a constant-time comparison on every HTTP
request. `index_corpus` refuses in read-only mode. On start the manager warms
the embedder, spaCy and every persisted corpus on a background thread, because
connector clients drop tool calls that take longer than a few seconds on the
first request.

## 7. Determinism and provenance

Equal scores are ordered by document id, sets are sorted before they become
graph insertion order, restart vectors or subgraph frontiers, and alias
collisions are resolved in sorted order, so two processes with different
hash seeds produce identical rankings. `tests/test_determinism.py` covers
this. The in-memory store and a Qdrant server returned identical hubmesh
top-10 lists for every question on MuSiQue (300) and 2Wiki (1,000), and on
HotpotQA N=500, with recall equal to the third decimal.

Every benchmark result file carries a manifest from `benchmarks/manifest.py`:
git commit, dirty flag and dirty paths (or an explicit `git_error` when git
is unusable), SHA-256 of the source tree, the harness and the protocol file,
model revisions, and run settings. The complementarity experiment was
pre-registered in `benchmarks/PROTOCOL_complementarity.md` with two dated
amendments recorded before results were examined, and
`benchmarks/examine_complementarity.py` is the single planned examination.

## 8. Performance characteristics

Measured on an Apple M1 laptop with 8 GB, MiniLM-L6 embeddings (384
dimensions), default configuration, one query at a time. The Qdrant column
is a Qdrant 1.20 server in Docker on the same host, same embeddings and
graph.

| corpus | graph | hubmesh median, in-memory | hubmesh median, over Qdrant | naive top-10, over Qdrant |
|---|---|---:|---:|---:|
| MuSiQue, 3,936 paragraphs | 22.6k nodes, 144k edges | 126 ms | 141 ms | 6 ms |
| 2Wiki, 6,250 paragraphs | 35.8k nodes, 320k edges | 135 ms | 150 ms | 7 ms |
| HotpotQA pilot, 4,943 paragraphs | 32k nodes, 250k edges | 202 ms | 217 ms | 7 ms |
| HotpotQA full dev, 66,581 paragraphs | 324k nodes, 2.95M edges | 2.5 s (p95 4.0 s) | 2.6 s | 20 ms |

The database adds 15 to 20 ms per question. The rest is the graph side,
and it grows with the graph: at 66k paragraphs each question runs up to 50
power iterations for the pooled solve and up to 50 for the batched
convergence solve over a matrix with 5.9 million nonzeros, then materialises
five whole-graph Python dicts of 324k entries. With the damping at 0.15 the
change per iteration shrinks by 0.85, so the 1e-6 tolerance is not reached
within the 50-step cap and the cap always binds. The full-dev figure was
taken with 65 MB of free memory and 2.6 GB compressed, so it is an upper
bound for this hardware, but the shape of the cost is structural, not
environmental.

Memory per corpus is the document matrix (66,581 × 384 float32, about
100 MB) plus the transition matrix and the networkx graph.

Practical reading: interactive-grade below roughly ten thousand paragraphs;
suited to agentic multi-hop and batch work above that; not suited to a
search box at 66k paragraphs without the changes in section 10.

Cold attach to an existing remote collection is one round trip per
document (section 3), so a 66k-document collection takes about 2.5 minutes
before the first query against a local server, longer over a network.

## 9. What the benchmarks established

Supporting-paragraph recall over pooled distractor corpora, hubmesh against
top-k cosine with the same embedder; full tables and protocol in
`BENCHMARKS.md`.

- HotpotQA full dev, MiniLM, body: 75.2% against 69.3% at recall@10, +4.2 at
  @5, −0.75 at @2. Title+body: +7.24 at @10. With bge-m3 the gain at @10 is
  +1.38 and the defaults lose 9.1 points at @2.
- MuSiQue full dev, MiniLM: +3.67 at @10 [+3.00, +4.39], growing with hop
  count (+2.9, +4.1, +5.3 for 2, 3, 4 hops).
- Scoring attribution on the same graph and packer: cosine-fused scoring
  reaches 0.871 recall@10 against 0.676 for the structural signal alone.
- The convergence term is worth about +0.9 at @10 on full MuSiQue and costs
  one extra batched solve.
- Pre-registered and negative: adding hubmesh's top-20 to a hybrid top-30
  before a `bge-reranker-v2-m3` cross-encoder at the same 50-pair budget
  changed complete-evidence@5 by +0.30 [−0.10, +0.80] on 2Wiki, under a
  rule that required +3.0. hubmesh is a cheap partial substitute for a
  reranker on multi-hop recall, not an addition to one.

The graph layer pays when the embedder is small, the questions are
multi-hop and there is no reranker. With a strong embedder or a reranker in
the pipeline it adds little and can hurt the top ranks, which is why
`use_convergence=False` is the documented setting for top-2 workloads.

## 10. Known limits and open directions

These are observations and proposals, not commitments.

- **Global PageRank is the scaling limit.** Two routes exist. Personalized
  PageRank is linear in its restart vector, so per-entity vectors can be
  cached and summed at query time, which also gives the per-seed vectors the
  convergence term needs; the dangling-mass term in `solve` feeds leaked mass
  back through the restart vector and would need to redistribute uniformly
  for that linearity to be exact. Alternatively, forward-push local PageRank
  costs in proportion to the tolerance rather than the graph size, but it is
  an approximation and changes rankings, and the min-max normalisation
  population would shrink from every document to a candidate set, which
  rescales the three terms. Either change needs a parity-and-recall check
  against the current pipeline; `benchmarks/run_qdrant_parity.py` already
  compares two pipelines question by question.
- **The seed fallback inherits the store's search accuracy.** On a Qdrant
  collection above the HNSW indexing threshold (10,000 points on the 1.20
  server used here) `search` is approximate. In the full-dev run hubmesh's
  top-10 differed between the in-memory store and Qdrant on 3 of the first
  7,000 questions; the case examined traced to the fallback receiving a
  different approximate top-3, and its gold documents kept ranks 1 and 2 on
  both stores. Over the first 1,400 questions the naive baseline's top-10
  differed on 77 and its recall@10 was 0.7 points lower through Qdrant. An
  exact-search flag on that one fallback call is the cheap fix.
- **Cold attach is per-document.** Batching the vector pull in the adapters
  is the next adapter change.
- **The graph is rebuilt, not updated.** `CorpusManager.build` replaces a
  corpus; there is no incremental NER on upsert. Storing each document's
  entity list in the store's payload would let a fresh process load the
  adjacency without re-running NER and would make incremental updates
  natural.
- **NER quality bounds everything.** `en_core_web_sm` is English-only and
  misses entities (titles of works especially); the fallback exists for that
  reason. The embedding linker and the LLM builder improve canonicalisation
  at index-time cost.
- **No head-to-head with other GraphRAG systems.** A LightRAG protocol
  agreed with its maintainers exists in `benchmarks/lightrag_compare/` and
  has not been run.

## 11. Extension points

- **Store**: implement the eight members and `mutation_counter`
  (section 5).
- **Entity linker**: implement `Linker.link(mentions) -> {raw: canonical}`
  and pass it as `linker=` to either graph builder.
- **Graph builder**: return an `EntityKG` with the node and edge conventions
  of section 2; the planner, paths and server work unchanged.
- **Scoring**: `PlannerConfig.weights`, `integration`, `use_convergence`,
  `hub_discount`, `ppr_alpha`; `redundancy_lambda` and `token_counter` for
  packing.
- **Agent loop**: `seed_entities` and `exclude_docs` on `retrieve`, or the
  MCP tools, for caller-driven multi-hop.

## 12. Where to look

| concern | code | tests |
|---|---|---|
| ranking and modes | `planner.py`, `scoring.py`, `ppr.py` | `test_planner.py`, `test_ppr_edges.py`, `test_determinism.py` |
| graph construction | `kg.py`, `kg_llm.py`, `entity_linker.py` | `test_correctness_batch.py`, `test_review_sep14.py` |
| agent-driven hops | `planner.py` (`seed_entities`, `exclude_docs`) | `test_seed_injection.py` |
| stores | `adapters/` | `test_qdrant_adapter.py`, `test_chroma_adapter.py` |
| persistence | `corpus.py` | `test_corpus.py`, `test_batch4_maintenance.py` |
| server and auth | `mcp_server.py` | `test_mcp_server.py`, `test_mcp_auth.py` |
| evaluation | `benchmarks/run_*.py`, `manifest.py`, `*_loader.py` | `test_complementarity_protocol.py`, `test_review_sep14_eval.py` |

Fifteen test files; the suite collected 211 tests at this commit. Review
history, including the external reviews that shaped the persistence,
determinism and evaluation contracts, is in `docs/REVIEW_BATCHES.md`.
