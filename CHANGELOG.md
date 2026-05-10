# Changelog

All notable changes to **hubmesh** are documented here. The format is loosely
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project follows [SemVer](https://semver.org/) starting from 0.1.0.

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
