# External review — response dossier (2026-09)

This is the single entry point for reviewing the four batches of changes
made in response to the September external folder review (five rounds on
batch 1; batches 2–4 delivered together). Each batch lists: the findings
it answers, the files it touches, the regression tests that guard it,
and what remains *outside* the verified guarantee. Nothing here claims
more than its tests demonstrate.

| Batch | Scope | Status |
|---|---|---|
| 1 | Correctness & safe persistence | **committed `715e295`**, cleared by the reviewer after 5 rounds |
| 2 | Service readiness (MCP auth, writes, CI) | in working tree |
| 3 | Scientific credibility (benchmarks + research) | in working tree |
| 4 | Determinism, caching, performance, maintenance | in working tree |

Suite: see the "Verification" section at the end for the final counts.

---

## Batch 1 — correctness & safe persistence (`715e295`)

Findings: corpus path traversal; context/sources mismatch; chunk→embed
dead-end; chunk-argument hang; no embedding fingerprint; non-atomic
rebuild; then (rounds 2–4) identity lost on default lazy init,
two-rename non-transaction, reader/generation mixing, retention measured
from creation instead of retirement.

Changes: `src/hubmesh/corpus.py` (identifier-only names + resolved
containment; generation dirs behind an atomically replaced `CURRENT`
pointer; per-corpus `flock` writer lock around stage→publish→prune;
`.retired` marker starts the reader grace window, conservative when
missing; legacy flat files never deleted; embedding `{identity, dim}`
recorded and enforced), `planner.py`/`packing.py` (`pack(max_docs=)`),
`adapters/inmemory.py` (embed `Document(vector=None)`), `chunking.py`
(argument validation, unknown strategy rejected).

Tests: `tests/test_correctness_batch.py` — 45 tests, each reviewer
reproduction encoded (traversal variants, publish-step crash, aged live
corpus first rebuild, missing/unparseable retirement marker, writer
overlap with threads, default-model identity path).

Verified contract / limits: POSIX writer coordination (no locking on
Windows); readers protected for `retention_seconds` from retirement,
not beyond; power-loss durability (fsync ordering) not guaranteed.

---

## Batch 2 — service readiness

Findings: SSE transport exposed read/write tools with no authentication;
DNS-rebinding protection disabled for tunnels; README documented public
tunneling of the unauthenticated service; MCP tests skipped in CI;
no transport-level tests.

Changes:
- `src/hubmesh/mcp_server.py`: `resolve_security()` — non-loopback bind
  or `--allow-tunnel` **refuses to start without** `--api-key` /
  `HUBMESH_API_KEY`; tunnel mode defaults to read-only (`--allow-writes`
  to opt out); `--read-only` flag; `BearerAuthASGI` middleware
  (constant-time compare, 401 + `WWW-Authenticate`, lifespan passthrough)
  wraps `mcp.sse_app()` via `build_sse_app()`; `index_corpus` refuses in
  read-only mode; rebinding protection is disabled only behind mandatory
  auth; stdio unchanged (local trust boundary).
- `README.md`, `docs/perplexity.md`: recipes generate a key, state the
  read-only default, give the edge-header-injection fallback for
  header-less connector UIs.
- `.github/workflows/test.yml`: installs `[dev,mcp]` so MCP tests run.
- `.github/workflows/publish.yml`: verifies tag == `pyproject` ==
  `__init__` version and runs the suite before building.
- Operator note: the founder's local `~/.hubmesh/run/start.sh` now
  generates/persists a key (`~/.hubmesh/run/api_key`, 0600).

Tests: `tests/test_mcp_auth.py` (policy matrix incl. every refusal
case; middleware over real ASGI semantics; real HTTP 401 on `/sse`
through starlette's TestClient; read-only refusal),
`tests/test_batch4_maintenance.py::TestMCPExtras` (tool schemas contain
no `anyOf`/null unions — the Perplexity regression; concurrent
`index_corpus` calls serialize through the writer lock).

Limits: no rate limiting or key rotation; per-corpus authorization is
not modelled (one key = full read access); a connector that cannot send
headers needs the tunnel edge to inject them.

---

## Batch 3 — scientific credibility

Findings: Hippo-style arm doesn't isolate scoring; MuSiQue identity is
title-based (collisions); LightRAG failures dropped from averages;
appended reruns double-count; no reproducibility manifests; contradictory
tables in BENCHMARKS.md; stronger-embedding regressions not shown;
research: test-set selection in the L1 path, gold-injected candidates,
overclaims.

Changes:
- `benchmarks/structural_only.py` (new): same graph, same Planner seed
  selection (incl. fallback), same doc-node readout, cosine removed —
  the attributable scoring ablation; wired into `run_hotpotqa.py` and
  `run_musique.py` as `structural_only`.
- `benchmarks/manifest.py` (new): commit, dirty flag, versions, platform,
  models, dataset slice, representation, planner config; both runners
  gained `--out` JSON with manifest + per-query recalls, and
  `--title-body`.
- `benchmarks/musique_loader.py`: `paragraph_identity="title"|"title+text"`
  (default preserves the published protocol) and an `audit=` dict.
  **Audit, N=300 seed 0: 284 titles carry multiple passages, 637 passages
  collapsed, 140 gold passages hidden behind a shared title; exact ids
  grow the pool 3,936 → 4,573.**
- `benchmarks/lightrag_compare/score_eval.py`: per-(pass,qid) dedupe
  (last wins), completion rate, `completed_only` / `strict`
  (failures = 0) / `shared` (intersection) metrics, JSON output with
  manifest.
- `BENCHMARKS.md`: consolidated current table with representation and
  embedding labelled per row, including the bge-m3 regressions
  (−9.09 @2, −1.41 @5); older tables marked historical; identity audit
  and ablation-arm semantics documented.
- Research (`research/RESEARCH.md` + scripts): L1-path λ selection moved
  to train (contaminated artifact preserved); result relabelled
  "corrected exploratory analysis"; theory downgraded to conjecture;
  `dump_components.py` records candidate recall **without** injection —
  see `research/track1_learned_fusion/results_candidate_recall.json`
  (numbers in the Verification section).

Limits / still open: no completed competitor lane exists (LightRAG pilot
blocked on the Moonshot account balance; KAG lane parked by decision) —
no head-to-head claim is made anywhere; MuSiQue numbers with exact
paragraph ids are produced by `--paragraph-identity title+text` and
reported in run JSON, not yet re-published in the README.

---

## Batch 4 — determinism, caching, performance, maintenance

Findings: set-iteration tie order (cross-process nondeterminism);
fallback seeds from sets; PPR dangling-mass leak, duplicate-seed mass
loss, no validation; planner cache ignores config; adapter neighbor
caches never invalidated and never grow; KG mode fetches every vector
and every document per query; packing budget ignores serialization
overhead; `VectorStore` protocol omits `vector_of`; LLM-KG failures
silently empty, cache ignores model/prompt, `max_workers` unused;
README/CONTRIBUTING/version drift.

Changes:
- Determinism: `scoring.py` sorted node iteration; explicit
  `(-score, id)` tie-breaks in `planner.py` (both modes) and
  `packing.py`; fallback seeds iterate sorted entity sets; `kg.py` builds
  the graph from sorted canonical sets and breaks substring-match ties
  by node id.
- `ppr.py`: dangling mass redistributed to the teleport vector (total
  mass 1.0 on isolated nodes), duplicate seeds deduplicated, `alpha` and
  edge weights validated.
- Caching: `CorpusManager.planner()` keyed by (name, config); store+KG
  loaded once per name; rebuild invalidates all configs. Chroma/Qdrant
  clear neighbor caches on upsert and expose `mutation_counter`; all
  adapters recompute when a request exceeds the cached neighbor list.
- Performance: KG-mode unit-normalized doc matrix cached per store
  version; document bodies fetched only for the 5×k packable candidates,
  in one `get_many` call (both modes).
- `packing.py`: budget counts the serialized context (prefix +
  separators) with the same counter; `PlannerConfig.token_counter`
  accepts a tokenizer-backed callable; docstring now describes the real
  policy (greedy score-first + near-duplicate filter).
- `adapters/base.py`: `vector_of` declared in the protocol;
  `mutation_counter` documented.
- `kg_llm.py`: extraction failures counted (not cached, retried next
  build) and surfaced in `kg.extraction_stats` with a warning; cache
  format 2 namespaced by `llm_identity` + prompt-template hash (legacy
  caches honoured only when unverifiable-by-necessity, with a warning);
  `max_workers` honoured via a thread pool.
- Docs/metadata: README status 0.4.1, opening description matches the
  code (entity-anchored seeding, multi-component *ranking*), examples
  pass `embed=`; CONTRIBUTING lists shipped adapters; CHANGELOG
  `[Unreleased]` per batch.

Tests: `tests/test_batch4_maintenance.py` (planner cache by config;
neighbor growth/invalidation incl. Chroma; protocol contract; budget
never exceeded under its own counter; custom counter plumbed; tie-break
by id; single batched fetch; LLM-KG failure accounting, namespaced
cache, worker parity), `tests/test_determinism.py` (fresh interpreters
under three `PYTHONHASHSEED`s must print identical rankings in kNN mode,
KG mode with exact ties, and the forced fallback-seed path),
`tests/test_ppr_edges.py` (unit mass on isolated nodes, dangling
redistribution, duplicate seeds, validation, reference agreement with
`networkx.pagerank` on a connected graph).

Limits / still open: KG mode still scores the whole corpus per query
(candidate-set scoring + subgraph PPR is the v0.5 scaling item);
p50/p95/p99 at named corpus sizes not re-measured here; LLM-KG relation
edges remain undirected without full subject/object/document provenance;
entity matching stays heuristic.

---

## Batch 5 — second external review of the current tree (2026-09-14)

The reviewer ran the 122-test suite (green) and then reproduced ten
failures with targeted probes (`review_repros_sep14.py`). Every fix
below ships with that reproduction as a test plus at least one boundary
case beyond it (`tests/test_review_sep14.py`, `tests/test_determinism.py`);
the review's process finding was that earlier fixes matched the
reproduction and the surrounding claim was then generalized.

Findings and changes:
1. **P1 — tunnel guidance recommended injecting the bearer token at an
   unauthenticated edge.** `docs/perplexity.md` and `README.md` now
   require the edge to authenticate callers before adding the upstream
   header and declare header-less connectors that cannot do so
   unsupported for private corpora; tunnel mode prints the caveat at
   startup (`SecurityPolicy.notice`, `TUNNEL_AUTH_NOTICE`).
   Tests: policy carries the notice only in tunnel mode; both docs
   contain the requirement.
2. **Same-manager cache race / cross-process staleness.**
   `CorpusManager.planner()` keys the cache by generation, re-reads
   `CURRENT` on every call, loads outside the lock, and installs a
   loaded generation only if it is still current at install time
   (`_peek_gen`, `_load_gen`, `_load_current`, `_drop_name`, `_cache_lock`).
   Tests: the reviewer's mid-load rebuild interleaving; a rebuild by a
   second manager on the same root; two rebuilds during one load;
   unchanged corpus keeps its cached planner.
3. **Partial adapter writes left caches valid.** Chroma and Qdrant
   invalidate neighbor caches and bump `mutation_counter` before the
   first batch and again in `finally`; Qdrant populates its vector
   cache only after the backend confirms a batch; a failed batch's ids
   are evicted. Tests: failure at batch 1, 2 and 3 of 3 for both
   adapters. (Batch-4 test relaxed from `== v0 + 1` to `> v0`.)
4. **Token budget false under the default counter.** `pack()` measures
   the complete serialized context a candidate would produce and
   returns exactly the string it measured. Tests: the 4×"abc"/budget-7
   reproduction; every budget 0–12; exact fit; a ceiling-rounding
   counter; the returned string is one of the measured strings.
5. **Determinism gaps.** `graph.py` capped expansion visits the frontier
   in id order with an insertion-ordered candidate set; `SubstringLinker`
   ties break lexically. Tests: subprocess runs under hash seeds 1–5 at
   the cap boundary (hops 1 and 2, cap below seed count) and on the
   linker with list and set inputs; the tie rule itself is asserted.
6. **Self-loop weight doubled.** `PPRSolver` adds the reverse entry only
   for `u != v`. Tests: weighted and unweighted self-loop parity with
   networkx; isolated self-loop node; an LLM self-relation triple.
7. **Malformed LLM replies cached as valid empty.** `_parse_triples`
   returns `None` for unparseable replies (including JSON objects
   without a `triples` key); those are counted, warned, and NOT cached;
   a parsed empty list is cached. Tests: the NOT-JSON-then-valid
   reproduction; valid empty is served from cache; four non-result
   replies are retried; partially malformed lists keep valid triples.
8. **`structural_only` did not isolate scoring.** The arm now runs
   through the same packer, budget, near-duplicate filter and document
   cap as hubmesh; a seedless query (no entity match even after the
   fallback) gets the neutral structural score, the document-id tie
   rule and the same packer instead of an empty result, so the arm no
   longer substitutes a different policy for the planner's; seedless
   and fallback queries are counted (`stats=`); both runners log
   per-query seed counts and write `structural_only_stats`. On every
   recorded run the seedless count is zero, so the recorded outputs
   are unaffected by the seedless change (that branch never executed;
   the harness fingerprint in those files predates it). Tests: the
   reviewer's document-only probe now returns the same document as
   hubmesh and is counted; id tie rule; packing budget enforced
   identically. Re-measured numbers: see Verification.
9. **LightRAG scorer ignored never-written queries.** `score_eval.py`
   scores each pass against its intended roster (questions file for
   main; `roster_<scope>_cold.json`, now written by `run_queries.py`,
   or `--cold-sample N`, else observed ids flagged as possibly
   overstated); reports `n_missing`, `n_unexpected`, and strict recall
   with missing and failed queries as zero. Tests: the 3-question /
   1-record reproduction (1/3, not 1/1); unexpected ids reported and
   unscored; cold roster file honoured; observed fallback flagged.
10. **README causal claim.** The `+29.8 vs PPR-only` line is replaced by
    the packing-constant attribution number; the bge-m3 row and the
    full-dev MuSiQue per-hop numbers move next to the headline; the
    Design diagram now draws the KG path.
- **Manifest:** whole-tree `git_dirty` plus `git_dirty_paths`,
  `src_sha256` over `src/hubmesh/**/*.py`, `benchmarks_sha256` over
  `benchmarks/**/*.py` (the imported loaders and ablation helpers a
  runner's own hash misses), `harness` and `harness_sha256`; runners
  record `embed_device` and `embed_batch_size`. Tests: hashes present
  and stable; tree hash tracks content and is order-independent. The
  every cited result file was re-recorded on 2026-09-15 from a clean
  checkout at `c724c24` (`git_dirty=false`, CPU embedding), so all of
  them carry `benchmarks_sha256` and the embed fields; the interim
  `_b5` files are superseded and removed.
- **Audit corrections:** `docs/COUNCIL_AUDIT.md` no longer claims
  "sorted iteration everywhere / fixed and proven"; the "+19.6"
  attribution is replaced by the re-measured value.

Limits: the tunnel caveat is guidance plus a startup notice, not an
enforcement mechanism (hubmesh cannot see whether an edge authenticated
the caller). Cache freshness costs one small file read per `planner()`
call. Remote `get_many` is still a per-id loop (not in this batch).
The environment note from the review stands: SciPy 1.15.2's `_propack`
extension does not load on macOS 27; the suite and re-runs here used an
isolated venv with SciPy 1.13.1 over the system packages.

## Verification

- **Test suite: 122 passed** (`pytest tests/`, Python 3.10, mcp/chroma/
  qdrant extras installed) — 45 batch-1, 17 batch-2 auth, 18 batch-4
  maintenance, 3 cross-process determinism (subprocess), 7 PPR edge
  cases, plus the pre-existing suite; two warnings are designed
  behavior (legacy-corpus fingerprint notice; LLM-KG failure report).
- **Candidate recall without gold injection** (research dumps rerun
  with tracking): HotpotQA 99.2% / 99.7% (MiniLM / bge-m3), MuSiQue
  93.5% / 94.0%. Learned-fusion results are bounded by these.
- **Numeric impact of the PPR change**: a provable no-op for
  entity-seeded walks (isolated doc nodes cannot receive mass; seeds
  were already deduplicated in the Planner); confirmed empirically by
  the pilot reruns below.
- **Pilot reruns with the new arms/manifests** (`benchmarks/results/`,
  manifests record commit `715e295` + dirty working tree, body-only,
  MiniLM, KG mode, seed 0). Recall@2/@5/@10:

  | Run | naive | hubmesh | structural_only | hippo_style |
  |---|---|---|---|---|
  | HotpotQA N=500 | .578/.740/**.819** | .570/.764/**.871** | .360/.520/.675 | .346/.494/.561 |
  | MuSiQue N=300, `title` id | .371/.489/**.567** | .394/.518/**.616** | .293/.373/.453 | .254/.315/.355 |
  | MuSiQue N=300, `title+text` id | .359/.490/**.571** | .395/.519/**.618** | .287/.392/.470 | .238/.325/.378 |

  Readings: (a) **PPR/determinism changes are numerically a no-op** —
  naive reproduces the historical N=500 numbers exactly and hubmesh
  lands at .871 vs the previously published .870 @10 (one tie flip);
  (b) the honest **scoring-attribution gap** (fusion over the same
  graph's pure structure) is **+19.6 pts @10** on HotpotQA and
  +16.3 on MuSiQue — smaller than the +31 previously implied by the
  `hippo_style` comparison, which mixed in pipeline differences
  (`structural_only` itself beats `hippo_style` by 11 pts);
  (c) **exact paragraph identity does not inflate the MuSiQue
  advantage**: hubmesh − naive @10 is +4.9 under `title` and +4.7 under
  `title+text`, per hop +5.6/+3.2/+6.1 vs +5.0/+3.9/+5.6.
- Two research dumps (bge-m3) ran after the determinism edits; the PPR
  change is a no-op for them (see above) and graph-order changes affect
  only floating-point summation order — disclosed.

### Batch 5 verification (2026-09-14, evening)

- Suite: **178 passed** — 122 prior plus 56 new (`tests/test_review_sep14.py`
  51, `tests/test_determinism.py` +5) — in an isolated venv with SciPy
  1.13.1 over the system packages. The system SciPy 1.15.2 `_propack`
  extension does not load on macOS 27 (the reviewer hit the same); the
  venv recipe is `python3 -m venv --system-site-packages venv &&
  venv/bin/pip install scipy==1.13.1`.
- Re-measured with fallback AND packing held constant
  (`benchmarks/results/hotpotqa_n500_kg_body.json`,
  `musique_n300_kg_title.json`; first measured on the dirty tree at
  `715e295`, then re-recorded from a clean checkout at `c724c24` with
  identical numbers — see the clean-checkout entry below): HotpotQA N=500 @10 naive 0.819 /
  structural_only **0.676** (was 0.675) / hubmesh 0.871 / hippo 0.561 →
  scoring attribution **+19.5** [+16.3, +22.7]; MuSiQue N=300
  structural_only 0.453 (unchanged) → **+16.3** [+11.9, +20.7]. Seedless
  queries: 0/500 and 0/300 (the fallback fired on 24 and 44 queries,
  every one ending with ≥1 seed). Holding packing constant changed one
  HotpotQA query's recorded recall and none on MuSiQue.
- For naive, hubmesh and hippo_style every recorded per-query recall
  value is identical to the batch-4 run on both datasets (ranked lists
  and numerical scores were not compared — the result files do not
  store them). That is consistent with the packing rewrite (a
  10,000-token budget never binds at 10 documents) and the self-loop
  fix (spaCy KGs have no self-loops) leaving these runs' outputs
  unchanged, and is what the files can establish.
- Both runners gained `--embed-device` / `--embed-batch-size` (defaults
  unchanged) after the first re-run was killed by memory pressure on
  this 8 GB machine during Metal-backed embedding; the recorded runs
  used `--embed-device cpu --embed-batch-size 16`.

### Clean-checkout verification (2026-09-15)

All six result files the documentation cites were re-run from a git
worktree at `c724c24` with outputs written outside the worktree, so their
manifests record `git_dirty=false`, `src_sha256`, `benchmarks_sha256`,
the harness hash, and `embed_device=cpu` / `embed_batch_size=16`:
`hotpotqa_n500_kg_body.json`, `musique_n300_kg_title.json`,
`musique_n300_kg_titletext.json`, `ablation_coherence_hotpotqa_n500.json`,
`ablation_coherence_musique_n300.json`, `ablation_coherence_musique_n2417.json`.
Every cited number reproduced to the printed precision (scoring
attribution +19.5 / +16.3; coherence contrasts B−A +0.30, A−C +1.10,
A−E +0.60 on HotpotQA; +0.28 / +2.14 / +2.17 on MuSiQue N=300; +0.03 /
+0.87 / +1.55 on full MuSiQue dev). One uncited value moved by a single
tie: `structural_only` recall@10 on the MuSiQue title+text run,
0.4697 → 0.4706, consistent with CPU-versus-Metal floating-point
differences in the embeddings. These files replace the earlier
dirty-tree and `_b5` versions.

## What was NOT done (deliberately)

- No competitor lane was run (LightRAG pilot awaits the Moonshot
  balance; KAG lane parked): no head-to-head claim exists anywhere.
- No new research features (the learned head stays an evaluated
  option, not a product change).
- Windows writer locking, power-loss durability, per-corpus
  authorization, rate limiting: outside scope, documented as limits.
