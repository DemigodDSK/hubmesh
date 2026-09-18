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

### Step-2 protocol: review response and Amendment 2 (2026-09-15)

An external review of the complementarity runner arrived after stage R
(candidate generation) had completed for all six configurations and
before any reranking result existed. Each claim was checked against the
code and the stage files; everything below is recorded in
`benchmarks/PROTOCOL_complementarity.md` as Amendment 2, before any
result was examined, and lives in commit `bc289d0`.

| review claim | finding | change |
|---|---|---|
| reference arms keep five results while reporting @10 | confirmed: A1/A2/A3/A5 stored 5 ids, `final = cand[:10]`, so ce@10 = ce@5 for those arms | stage R stores a 10-prefix; old stage files extended from the exact A7/A4/A8 prefixes plus a deterministic BM25 recomputation, every stored 5-prefix verified (`extend_reference_arms`) |
| unresolved gold can be dropped | confirmed as a code path: gold was filtered to the pool before scoring; a 2Wiki supporting title with no context paragraph was also dropped inside the loader. Checked against the data: in all six stage files every stored gold list equals the full gold list (0 of 2,481 sampled 2Wiki supporting facts unresolved), so no number changes | metrics use the full gold list, `n_gold_unresolved` recorded per question; the loader keeps `<title>::unresolved` ids (`refresh_gold`) |
| rerank timings come from A4 alone | confirmed | each reranked arm's own list is scored from scratch and timed on the first 100 queries (primary configuration) / 50 (secondary); composed latency for reranked arms is over those queries with the count reported; ranking scores come from one untimed pass over the union of candidates, cached per (question, paragraph) with provenance |
| resumed stage files lack identity validation | confirmed: `--stage rerank` loaded the file unchecked | `validate_stage`: dataset / n / seed / embedder where present, question order against the loaded sample, every candidate in the pool, exactly 50 distinct for A4/A6/A7, ≤ 50 for A8, prefix relations A1 ⊂ A7, A3 ⊂ A4, A5 ⊂ A8; new stage files carry qid and pool hashes, the protocol hash and the commit |
| A8's smaller candidate count is not equal-budget | verified per query: A4, A6, A7 have exactly 50 distinct candidates in every query of every file; A8 has fewer than 50 in 493 of 1,000 primary-configuration queries (minimum 16; 4 queries below 20) | A8 labelled a variable-count, token-budgeted arm in every result file; per-arm candidate counts reported; construction unchanged |
| the memory explanation is plausible, not proven | agreed; the runner now samples allocated and driver Metal memory through every run and stores the samples in the manifest, interpreted only as "grows / does not grow" | — |
| "5 pairs/s is the physical ceiling" is unsupported | withdrawn; throughput is reported as measured on this machine | — |
| validate the optimized reranker before using its scores | done, below | `benchmarks/validate_rerank.py`; every stage K run repeats a one-question comparison and aborts above 0.05 logits |

The stage R files were kept, not recomputed: each passes the identity
checks above, and question order, candidates and gold match a fresh load
of the sample.

**Reranker validation** (`benchmarks/results/rerank_validation_2wiki_mps_fp16.json`,
run from the worktree at `bc289d0` on the real 2Wiki candidates). The
fixed-shape path uses the CrossEncoder's own tokenizer and weights, the
same pair truncation (`longest_first`) and 512 limit, right padding with
an attention mask, and the classification logit at position 0; it
differs from the library only in padding every pair to the smallest of
128 / 256 / 384 / 512 tokens and scoring constant batches of 8. Ranking
uses the raw logit.

| check | result |
|---|---|
| A. library path (`CrossEncoder.predict`, identity activation, dynamic padding) vs fixed-shape, both fp16 on Metal, 20 queries | max abs logit difference 0.0156, mean 0.0014; top-5 order identical for A4, A6, A7 and A8 in 20/20 queries; top-10 order identical in 18/20 (A4) and 19/20 (A6, A7, A8); the full ~70-candidate order identical in 6/20. This is sampled top-5 agreement, not a proof of exact ranking equivalence |
| B. endurance, fixed-shape only, 150 queries / 11,119 pairs | 5.26 pairs/s; Metal driver memory 6.77 GB at query 0 → 6.86 GB at query 149 (allocated tensors flat at 1.06 GB, RSS 1.9 GB). The level was set by the 20 library-path queries that preceded it; a fixed-shape-only 40-query test earlier the same day ran at 2.78 → 2.98 GB. Under the library path, driver memory had grown ~0.6 GB per query and the run failed near query 25 |
| C. fp32 CPU library reference vs fixed-shape fp16 Metal, 5 queries | max abs logit difference 0.0151, mean 0.0026; top-5 identical for every arm in 5/5; CPU path 3.2 pairs/s |

Stage K was then launched over the six kept stage files from `bc289d0`,
primary configuration first, each configuration on Metal fp16 with a
whole-configuration CPU fp32 redo on failure (device and precision are
recorded in every manifest and score cache, so results are never mixed
without saying so). Results are examined once, after all six complete.

### Step-2 protocol: run 1 lost, second review, run 2 (2026-09-16/17)

**Run 1.** All six configurations finished stage K on 2026-09-16 (Metal
fp16, no fallback). Two things then went wrong, both recorded in
`benchmarks/results/complementarity_run1_2026-09-16_transcribed.md`:
the machine's Xcode `git` began refusing to run mid-day, so the last two
result files carried no commit (their content hashes matched the other
four); and the session's scratch directory, where every output lived,
was purged when the session ended, before the copy into the repository
had been made. The four configurations with complete manifests had been
examined once, at 14:11 EDT, before the loss; the two without a commit
never were. The registered outcome as examined in run 1 — 2Wiki,
bge-m3, A6 − A4 complete-evidence@5 = **+0.30 [−0.10, +0.80], n = 1000,
"no useful gain"**; guard slice saturated at 1.000 for both arms;
latency ratio 1.054 — is on record in that file, together with every
number the examination printed. No per-question record of run 1 exists.

**Second review (2026-09-17)** and what changed, in the commit that adds
this section:

| review point | response |
|---|---|
| the resume checker writes qid/pool/protocol hashes but does not verify them; wrong hashes and an incompatible embedding sequence length were accepted | `validate_stage` now recomputes the question-id and pool hashes from the loaded sample and rejects a mismatch; embedding settings given on the command line (`--embed-dtype`, `--embed-batch-size`, `--embed-max-seq-len`) must equal the stage file's; the protocol hash is compared and any change between stage R and stage K is recorded in the manifest (an amendment is legitimate, silence is not); validation now runs before stage K for fresh stage files too |
| the score cache lacks fingerprints of the question/document text and a pinned model revision | cache provenance now includes SHA-256 fingerprints of the exact questions and pool texts, the reranker's Hugging Face snapshot revision and the actual parameter dtype; a cache with any key outside the current sample is refused; the revision is recorded in every manifest |
| "identical top-5/top-10" overstated the validation | corrected above and in the commit: top-5 identical 20/20; top-10 18/20 (A4), 19/20 (A6–A8); full order 6/20 — sampled agreement, not exact equivalence |
| lead with the registered result; exploratory findings are hypotheses | the run-1 record and the run-2 write-up lead with A6 − A4 ce@5, its interval, the guard and the latency rule; candidate-list completeness and per-type contrasts are labelled post-hoc |
| recovering the two commit-less runs from cached scores needs verified cache identity and disclosure | moot: those runs and the caches were lost. In run 2 the caches are fingerprinted as above and `rerank_cache.hits` in each manifest says how many pair scores were reused |
| a manifest must never silently record a missing commit | `build_manifest` records `git_error` and sets `git_dirty` to null (unknown) when git is unusable; the runner refuses to start in that state |

**Run 2** re-runs the whole experiment — stage R and stage K for all six
configurations — from a clean worktree at the commit that adds this
section, in a durable directory, gzipping each result file into
`benchmarks/results/` the moment it is written. Its examination is the
second look at the primary outcome (run 1's is above), and it is the run
whose files are committed and reported.

## What was NOT done (deliberately)

- No competitor lane was run (LightRAG pilot awaits the Moonshot
  balance; KAG lane parked): no head-to-head claim exists anywhere.
- No new research features (the learned head stays an evaluated
  option, not a product change).
- Windows writer locking, power-loss durability, per-corpus
  authorization, rate limiting: outside scope, documented as limits.
