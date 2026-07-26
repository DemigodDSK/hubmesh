# Changelog

All notable changes to **hubmesh** are documented here. The format is loosely
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project follows [SemVer](https://semver.org/) starting from 0.1.0.

## [0.3.0] — 2026-07-26

### Added
- **hubmesh-mcp.** `hubmesh.mcp_server` + `hubmesh-mcp` console script
  (`pip install "hubmesh[mcp]"`): hubmesh as eight deterministic MCP
  operator tools — `index_corpus`, `retrieve` (with the v0.2
  `seed_entities` / `exclude_docs` door), `resolve_entities`,
  `entity_neighbors`, `path_between`, `get_document`, `graph_stats`,
  `list_corpora`. The calling agent plays the solver — decomposing
  multi-hop questions and steering each hop — while the server makes
  zero LLM calls: every tool is numpy/networkx work, token-free and
  ~milliseconds after indexing. Responses are token-lean (snippets +
  ids; full text via `get_document`).
- **CorpusManager** (`hubmesh.corpus`, exported at top level): named
  corpora persisted to plain JSON/NPZ under `~/.hubmesh/corpora`
  (`HUBMESH_CORPORA_ROOT` overrides) — no binary object serialisation,
  so corpus directories are inspectable and loading one can never
  execute code. Planners are cached per corpus with the precomputed
  PPR transition matrix.

### Tests
- 29 passing (4 new: KG dict round-trip, build/persist/reload/retrieve
  with the seed door, corpus listing, missing-corpus error). Live MCP
  stdio smoke-verified: a client walked a 3-hop chain over the
  protocol, answer doc ranked first.

## [0.2.0] — 2026-07-25

### Added
- **Iterative multi-hop door.** `Planner.retrieve` accepts `seed_entities`
  (extra PPR teleport seeds, *merged* with the query's own NER seeds) and
  `exclude_docs` (drop already-consumed documents). Any agent or outer
  loop can now run plan-driven retrieval — hop N steered by entities read
  in hop N-1 — while the query path stays deterministic and LLM-free.
  `seed_entities` is KG-mode only (raises `ValueError` otherwise);
  `exclude_docs` works in both modes. On the synthetic 3-hop chain in
  `examples/agent_hop_demo.py`, the answer doc moves from rank 5
  (single-shot) to rank 1 (seed-injected hops) with no change to query
  text — run the script to reproduce.
- **Alias index.** `EntityKG.alias_to_node` retains every canonicalised
  surface form seen at build time → its graph node. `query_entity_nodes`
  checks it first: O(1), never returns a node absent from the graph, and
  resolves absorbed short forms ("Derrickson" → "scott derrickson").
  Exact canonical names are bound before display-derived aliases, so an
  alias can never shadow a different entity's exact name. KGs persisted
  before 0.2 restore with an empty index (via a `__setstate__` shim) and
  behave exactly as before.
- **Linker support in LLM extraction.** `build_entity_kg_llm(..., linker=)`
  routes triple mentions through the same `Linker` protocol as the spaCy
  path — closing the gap where LLM-extracted entities got weaker
  cross-document dedup than heuristic ones.

### Changed
- Query-side entity resolution for KGs **rebuilt** on 0.2.0 can differ
  from 0.1.x wherever the alias index now answers directly (absorbed
  short forms, display variants). Confirmed as a recall improvement by
  re-run: HotpotQA N=500 recall@10 +3.7→**+4.0** vs naive and
  +29.1→**+29.8** vs PPR-only (recall@2 −0.4, disclosed); MuSiQue
  2/3/4-hop +1.7/+1.9/+2.8 → **+3.0/+2.6/+3.4**. Full-dev N=7405
  numbers remain v0.1.1 (not re-run).

### Tests
- 25 passing (12 new: seed merge semantics + dedup, exclusion in both
  modes, kNN-mode guard, linker routing, alias resolution, alias-shadow
  parity, phantom-node guard, pre-0.2 state restore). All new tests run
  without spaCy models or an LLM.

## [0.1.1] — 2026-05-10

### Fixed
- **Project URLs** in PyPI metadata pointed at a non-existent GitHub
  user (`dattasaikrishnanaidu`); corrected to the actual repo at
  [DemigodDSK/hubmesh](https://github.com/DemigodDSK/hubmesh).
- Added `Repository` and `Changelog` URL entries so the PyPI sidebar
  shows links to source and history.

No code changes — purely metadata. `pip install hubmesh==0.1.0` is
identical to 0.1.1 at runtime; upgrade only if you want clickable links
on PyPI to resolve correctly.

## [0.1.0] — 2026-05-10

The first feature-complete pre-alpha. Multi-component scoring, KG mode,
adapters for Qdrant and Chroma, reasoning-path explanation, latency
optimisations, and document chunking are all in place.

### Added
- **Reasoning paths.** `RetrievalResult.reasoning` is now populated with
  multi-hop traces from query-entity seeds to retrieved documents.
  Useful for explainability and downstream LLM re-ranking. New module
  `hubmesh.paths`.
- **Embedding-based entity linker.** `hubmesh.entity_linker.EmbeddingLinker`
  clusters mentions by sentence-transformer cosine similarity, replacing
  the fragile substring-collapse heuristic. Plug into `build_entity_kg`
  via `linker=`.
- **Chroma adapter.** `hubmesh.adapters.ChromaStore` — ephemeral,
  persistent, and HTTP modes.
- **LLM-based KG construction.** `hubmesh.kg_llm.build_entity_kg_llm`
  extracts (subject, predicate, object) triples via any callable LLM
  (provider-agnostic). Cached by passage hash so re-runs are free.
- **Document chunking.** `chunk_by_sentences`, `chunk_by_chars`, and
  `chunk_documents` for splitting long source documents before indexing.
- **PPRSolver.** `hubmesh.ppr.PPRSolver` precomputes the sparse
  transition matrix once at Planner init for ~10× faster per-query PPR.
- **Vectorised document scoring** in KG mode — replaces the per-doc
  Python loop with a single dense matmul.
- **Qdrant adapter** in 0.0.1 (now production-tested).

### Performance
- Per-query latency on a 7K-node KG: **100.7 ms → 22.3 ms mean** (4.5×
  speedup), p95 175 ms → 26 ms.

### Benchmarks
- HotpotQA dev N=500, recall@10: hubmesh **+3.7 pts** vs naive cosine,
  **+29.1 pts** vs HippoRAG-style PPR-only ablation.
- MuSiQue dev N=300, recall@10: **+1.7 / +1.9 / +2.8 pts** at 2/3/4-hop.
  Win grows with hop count.

### Tests
- 13 passing across 3 adapters, scoring, paths, chunking, LLM-KG (mocked).

## [0.0.1] — 2026-05-10

Initial release.

- VectorStore protocol + InMemoryStore reference adapter
- Induced-subgraph builder + Louvain anchoring (kNN-graph mode)
- Personalized PageRank
- Multi-component scoring (R × S × C, geometric mean and weighted sum)
- Budget-aware MMR context packer
- HotpotQA paragraph-retrieval benchmark
- Qdrant adapter
- HippoRAG-style ablation in `benchmarks/hippo_style.py`
