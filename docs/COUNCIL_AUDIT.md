# Audit of the external "Model Council" review and the convergence ablation

Date: 2026-09-14. Inputs audited: the Council synthesis, the three
individual assessments (Claude Opus 5, Astra, Gemini 3.1 Pro), the
"HubMesh vs Aethel" comparison, and the "convergence ablation report".
All six were produced against commit `715e295` (batch 1 of the
correctness work only). The working tree here carries batches 2–4
(uncommitted since 2026-09-10, see `docs/REVIEW_BATCHES.md`), so a large
share of the council's code findings were already fixed before the
review was written. This document separates what is right, what is
stale, what is wrong, and what was re-measured locally.

Nothing in the product was changed while producing this audit. One new
benchmark script was added (`benchmarks/run_ablation_coherence.py`) and
its results were written to `benchmarks/results/`.

---

## 1. Verified against the tree and the web

| Claim | Verdict | Evidence |
|---|---|---|
| BGE-M3 full HotpotQA: +1.38 @10, −1.41 @5, **−9.09 @2**, 3.03 s/q vs 33 ms | **Exact** | `benchmarks/lightrag_compare/results_titlebody_hotpotqa_full_BAAI_bge-m3.json`: naive .6656/.7836/.8346, hubmesh .5747/.7695/.8484; 22,471 s / 7,405 q |
| MiniLM title+body full: +7.24 @10, +0.25 @2 | **Exact** | `…_full_all-MiniLM-L6-v2.json` |
| MuSiQue bge-m3 pilot: +4.03 @10, 4-hop +5.00 @10 (n=45) | **Exact** | `…_musique_pilot_BAAI_bge-m3.json` |
| "The README does not report the bge-m3 run" | **Half-true** | README: correct, it still leads with body-only MiniLM +5.90 and never mentions bge-m3. But `BENCHMARKS.md` (uncommitted batch 3) carries the three-representation table with −9.09 @2 in bold and a "read both directions" paragraph, and the full grid has been public on HKUDS/LightRAG#3571 since 2026-08-08. "Not in the README" is right; "hidden" is not. |
| Aethel, arXiv:2607.24826 | **Exists** | Krish Sapru, submitted 20 Jul 2026, cs.IR + cs.MA. Repo `ksapru/aethel-clean` is now reframed as "Off-the-Shelf Dense Retrieval Degrades Faster Than BM25 at Scale"; BCT renamed Alias-Expanded Seeding; its own ablation: "+0.005 / +0.010 HR@5 — one and two questions out of 200"; ranks on PPR alone; baselines BM25, four dense encoders, Hybrid-RRF. Matches the pasted description. |
| Seed-Anchored Budget-Bounded Graph Rendering, arXiv:2609.02011 | **Exists** | Manoharan & Sehgal, 2 Sep 2026, eess.SY + cs.AI + cs.IR. Power-grid CIM vertical; 0.450→0.970 accuracy on a 100-item set under an 8,000-character budget. Overlap is the "deterministic, parameter-free, seed-anchored, budget-bounded" framing, not bipartite PPR or score-level fusion. |
| EdgeMem, arXiv:2609.05553 | **Exists** | Cui et al. (PolyU), 3 Sep 2026, cs.AI + cs.MA. Agent conversational memory; shares the words "LLM-free", "multi-anchor", "evidence-preserving", not the problem. |

---

## 2. Council findings already fixed in the uncommitted tree (batches 2–4)

The reviewers could not see these; they are in `docs/REVIEW_BATCHES.md`
and are ready for the founder's commit decision.

| Council finding | Status | Where |
|---|---|---|
| KG mode re-gathers every doc vector, re-stacks and re-normalises the full matrix on every query (Opus measured 0.53 s of identical per-query work at 66K) | **Fixed in code, unmeasured at 66K** | `Planner._kg_doc_matrix()` caches the unit-normalised matrix per store `mutation_counter` (`planner.py:146-166`). The gather still happens once per store version. The 3.03 s/q figure in `BENCHMARKS.md` predates this and must be re-measured before any latency claim changes. |
| Fetches every ranked document before cutting to the packer's pool | **Fixed** | `_fetch_docs(ordered[:top_k*5])`, one `get_many` call (`planner.py:411-415`) |
| `vector_of` required but absent from the `VectorStore` protocol | **Fixed** | `adapters/base.py:36` |
| "Zero LLM ≠ deterministic": seed fallback iterates sets, composite iterates node-ID sets, alias ties by iteration order | **Fixed on the tested paths; two more paths found and fixed in batch 5** | *Correction (2026-09-14, second review):* the earlier wording here, "sorted iteration everywhere, fixed and proven", was false. The reviewer reproduced hash-seed-dependent candidate membership in the capped kNN expansion (`graph.py`) and hash-seed-dependent ties in the explicit `SubstringLinker`. Both are fixed in batch 5 with subprocess tests at the cap boundary and on the linker (`tests/test_determinism.py`). Determinism is established only for the paths those tests cover. |
| Isolated seeded node returns PPR mass 0.15 instead of 1.0 (no dangling-mass handling) | **Fixed** | dangling mass redistributed to the teleport vector (`ppr.py`); `tests/test_ppr_edges.py` incl. networkx parity. Convergence-tolerance reporting (whether `max_iter` was hit) is still not exposed — minor, open. |
| LLM KG builder: `max_workers` unused, failures silently become empty triples, cache key ignores model/prompt | **Fixed** | `ThreadPoolExecutor`, failures warned and not cached, `kg.extraction_stats`, cache format 2 namespaced by `llm_identity` + template hash |
| Public-tunnel recipe with no auth; `--allow-tunnel` disables DNS-rebinding protection; `index_corpus` can replace a corpus | **Fixed** | `resolve_security()`: non-loopback or tunnel requires a key or refuses to start; tunnel implies read-only unless `--allow-writes`; `BearerAuthASGI`; `docs/perplexity.md` rewritten; 17 tests |
| Packer's chars/4 estimate ignores citation labels and separators | **Fixed** | per-piece cost counts `[n] ` and `\n\n`; `PlannerConfig.token_counter` for exact model tokens |
| No per-query records → no paired significance tests | **Fixed** | `--out` JSON with per-query records and a manifest (commit, dirty flag, versions, config) on both runners |
| "+29.8 vs HippoRAG-style" is not a scoring ablation | **Measured, re-measured in batch 5** | `benchmarks/structural_only.py` (same graph, seeds, fallback, doc readout; cosine removed) — after the second review it also runs through the same packer: hubmesh .871 vs structural_only .676 → honest scoring attribution **+19.5 @10** [+16.3, +22.7] (MuSiQue N=300 +16.3); seedless queries 0/500 and 0/300. Stated in `BENCHMARKS.md`; README updated in batch 5. |
| Title dedup can credit the wrong passage | **Audited on MuSiQue** | 284 colliding titles, 140 gold passages hidden; `--paragraph-identity title+text` re-run: +4.9 → +4.7 @10, so the collision does not inflate the advantage. HotpotQA titles are unique article titles and were not audited. |
| `CorpusManager` cache keyed by name ignores config | **Fixed** | keyed on `(name, config)` |
| Run-to-run ±0.2 pt nondeterminism in the fallback path (ablation report's own caveat) | **Fixed** | same determinism work as above; the report ran on `715e295` |

---

## 3. Confirmed and still open

These are real, not addressed by any batch, and are the founder's calls.

1. **No lexical or reranker baseline.** No BM25, no hybrid RRF, no
   cross-encoder over ANN top-50, and no complementarity arm
   `rerank(ANN ∪ hubmesh)`. All three reviewers and Aethel's own
   negative result converge here. This is the single largest scientific
   gap; every published delta is "vs dense-only".
2. **No incremental indexing.** `build()` replaces a generation;
   there is no `add_documents`/`remove_documents`, and co-occurrence
   edge weights do not record which documents contributed.
3. **MCP `retrieve` returns 280-character snippets and never the packed
   context** (`mcp_server.py:104-117`). The library's packer output is
   not reachable over MCP.
4. **Qdrant adapter:** `get_many` is a per-id loop (`qdrant.py:148`) and
   `vector_of` is a per-id retrieve with a local cache. With the new
   matrix cache the first KG-mode query against a remote collection
   still issues N round-trips, then none until the store changes.
5. **Another process's `CorpusManager` keeps serving its cached planner
   after a rebuild elsewhere**; generation pointers are resolved at load
   and never re-checked.
6. **README drift** (batch 3 fixed the seeding description but not
   these): the Design diagram (README lines 222–228) still draws the
   kNN path (ANN → induced subgraph → community anchoring), not the
   benchmarked KG path; line 254 still carries "+29.8 pts vs PPR-only …
   the multi-component scoring is doing the work" (superseded by
   +19.6); the headline table is body-only MiniLM with no embedder grid
   and no bge-m3 row.
7. **CHANGELOG 0.4.0 wording:** "single-seed queries are unaffected (the
   component is neutral there)" is a property of the `len(seeds) >= 2`
   gate, phrased as a measurement. Fair critique.
8. **LightRAG lane** has a locked protocol and no competitor numbers
   (blocked on the Moonshot balance since 2026-08-12);
   `<pending maintainer pointer>` is at
   `benchmarks/lightrag_compare/README.md:33` (Opus placed it in
   `run_queries.py`; wrong file, right point).
9. **recall@k is a proxy; no answer-level EM/F1; MuSiQue 4-hop n=45 is
   underpowered.** Known and disclosed, still true.
10. **NNSI/ICOMP'25 provenance.** Opus argues it prices the artifact by
    the venue. This is the founder's paper and the founder's call; the
    engineering evidence does not depend on it either way.

*Status after batch 5 (same day, second external review): items 6 and 7
are closed (README embedder grid, attribution, KG-path diagram,
bounded convergence wording; CHANGELOG note). Items 1–5 and 8–10 remain
open as stated. Batch 5 itself is documented in
`docs/REVIEW_BATCHES.md`.*

---

## 4. Where the council is wrong, stale, or overstated

- **Opus's mechanism story for convergence was wrong, and the ablation
  report itself says so.** Opus's synthetic probe found the geomean
  produces a graded signal (57% of docs > 0.1 after min-max) and
  concluded the v0.4.0 gain was "a log-compression artifact". On the
  real corpora the report measured the opposite: the geomean is
  near-binary (1.0% of docs > 0.1), Spearman 0.979 against pooled PPR.
  `exp(mean(log))` undoes the log. The council synthesis still repeats
  the "log-compression artifact" line; it is superseded by the report.
- **The ablation report's headline "AND-ness contributes nothing" is
  stronger than its own design supports.** Personalized PageRank is
  linear in the restart vector, so `mean_j p_j` over the first four
  seeds is exactly the pooled PPR whenever a query has ≤4 seeds (mean
  seeds: 2.1 on HotpotQA, 2.6 on MuSiQue). Arm D ("log-OR over the same
  anchors") is therefore arithmetically identical to arm B on nearly
  every query, and "D ≈ A" is the same measurement as "B ≈ A", not an
  independent one. Meanwhile the report's own arm E (raw pooled PPR,
  ≡ weights 3:2:0) loses 5.1 pts @2 to arm A on HotpotQA. That is
  AND-pooling beating OR-pooling by five points *when the shape is held
  raw* — evidence that pooling does matter, in the one comparison where
  it is isolated. The defensible claim is narrower: *a single-solve
  log-pooled signal matches the shipped geomean within CI on every
  metric, at one PPR solve instead of 1+m.* Whether AND-ness matters
  once the shape is graded is the test the report did not run; §5
  runs it.
- **Hub discount "half-cancelled" needs one correction.** The algebra is
  right: `w_ij / (log(e+deg_i)^γ · log(e+deg_j)^γ)` followed by row
  normalisation cancels the *source* factor, so a hub's outflow is
  untouched; the *destination* factor survives, so inflow to hubs is
  damped from every source. Opus's own probe shows that surviving half
  working (−42.6% mass on the top-20 hubs). The recommendation to add
  "explicit inbound-only damping" describes what the code already does.
  The neutral ablation result is explained by the doc-level readout
  being cosine-dominated (10/10 identical top-10), not by cancellation.
  Astra's phrasing ("reweights toward less-penalised destinations") is
  the accurate one.
- **Priority threat is real for vocabulary, modest for mechanism.**
  Aethel's alias expansion is the same family as the v0.2.0 alias index
  but opposite polarity (query-time fan-out vs index-time collapse),
  Aethel does not fuse cosine, has no packing, no hub discount, no
  multi-anchor term, and its own ablation now rates the alias component
  at one or two questions out of 200. Seed-Anchored overlaps on the
  budget-packing *framing* in a different vertical. EdgeMem overlaps on
  three adjectives. The synthesis line "two preprints independently
  published your two contributions" overstates it; "two preprints
  occupy adjacent vocabulary and one shares a seeding mechanism" is
  accurate. The clock is still real: none cites hubmesh, and a dated
  write-up is the only thing that fixes that.
- **Gemini's assessment is one evidence layer behind** (README-only);
  its "sound and clever" verdict on convergence and its 80–250 ms
  LightRAG latency figure (third-party blog) should not be relied on.
- **eps "inert because min-max absorbs the additive floor"** — the
  reasoning is loose (`minmax(log(eps+p)) = log(1+p/eps)/log(1+p_max/eps)`
  changes shape with eps, not just offset) but the report's empirical
  sweep (τ ≈ 1.000 over 1e-9…1e-15) is what matters, and it holds.

---

## 5. Local reproduction of the coherence-slot ablation, with the missing arm

### 5.1 Setup

Script: `benchmarks/run_ablation_coherence.py` (new, untracked). Results:
`benchmarks/results/ablation_coherence_hotpotqa_n500.json` and
`…_musique_n300.json`, each with a manifest and per-query recall vectors.
Run on the current tree (batches 2–4 applied), MiniLM-L6-v2, α = 0.15,
weights 3:1:1 with `integration="sum"`, `top_k=10`, budget 10,000,
eps = 1e-12, same loaders and metric as `run_hotpotqa.py` /
`run_musique.py`. HotpotQA dev N=500 seed 0 → 4,943 paragraphs, KG 32,079
nodes / 249,804 edges. MuSiQue-Ans dev N=300 seed 0 → 3,936 paragraphs,
KG 22,627 / 143,897 (title identity, as published).

Every arm shares one pooled `solve` and one `solve_multi` per query; only
the vector handed to the coherence slot differs:

| arm | coherence value | pooling | shape | solves |
|---|---|---|---|---|
| A | `exp(mean_j log(eps+p_j))`, seeds[:4] — shipped | AND | raw (near-binary after min-max) | 1+m |
| **G** | `mean_j log(eps+p_j)`, seeds[:4] — **new** | AND | log (graded) | 1+m |
| B | `log(eps + pooled)` | OR | log | 1 |
| D | `log(eps + mean_j p_j)`, seeds[:4] | OR | log | 1+m (≡ B when ≤4 seeds, by linearity) |
| E | `pooled` raw (≡ weights 3:2:0) | OR | raw | 1 |
| C | off (weight 0) | — | — | 1 |

A/G/D are active only with ≥2 seeds (the shipped gate): 316/500
HotpotQA, 159/300 MuSiQue. B/E are active on every seeded query. Paired
bootstrap, 10,000 resamples, reported both unconditionally and on the
active subset.

### 5.2 The external report reproduces

| | HotpotQA @2/@5/@10 (report → here) | MuSiQue @2/@5/@10 (report → here) |
|---|---|---|
| A | .567/.761/.869 → .570/.764/.871 | .392/.518/.616 → .394/.518/.616 |
| B | .566/.771/.873 → .567/.770/.874 | .403/.526/.615 → .401/.525/.619 |
| C | .574/.766/.859 → .575/.767/.860 | .394/.508/.594 → .393/.509/.594 |
| D | .571/.768/.868 → .572/.766/.870 | .395/.511/.609 → .393/.510/.609 |
| E | .516/.737/.863 → .516/.740/.865 | .377/.512/.594 → .376/.513/.594 |
| naive | .578/.740/.819 → **identical** | .371/.489/.567 → **identical** |

Active-query counts (316 / 159), mean seeds (2.12 / 2.56), the 94%-of-docs
shape statistic for the log arm (0.9396 / median 0.9488) and the key
intervals all match: B−A @10 HotpotQA +0.40 [−0.80, +1.60] there vs
+0.30 [−0.90, +1.50] here; MuSiQue active −1.52 [−4.04, +0.94] vs
−1.26 [−3.41, +0.89]; A−C MuSiQue active +3.98 [+1.47, +6.55] vs
+4.04 [+1.62, +6.60]. The report's harness is sound, and the batch-4
PPR/determinism changes are numerically a no-op here as well. Solve-stage
cost on this machine: `solve_multi` 0.122 s vs pooled 0.028 s (HotpotQA),
0.076 s vs 0.017 s (MuSiQue) — 4.4–4.6× on the solve alone.

### 5.3 The 2×2 the report did not run

Recall, unconditional (active-only @10 in the last column):

| HotpotQA N=500 | @2 | @5 | @10 | active @10 |
|---|---:|---:|---:|---:|
| A geomean (shipped) | .570 | .764 | .871 | .889 |
| G log-AND | .578 | .766 | .868 | .884 |
| B log-OR, 1 solve | .567 | .770 | **.874** | .884 |
| D log-OR, same anchors | .572 | .766 | .870 | .888 |
| E raw-OR (≡ 3:2:0) | .516 | .740 | .865 | .880 |
| C off | .575 | .767 | .860 | .872 |
| naive | .578 | .740 | .819 | .828 |

| MuSiQue N=300 | @2 | @5 | @10 | active @10 |
|---|---:|---:|---:|---:|
| A geomean (shipped) | .394 | .518 | .616 | **.663** |
| G log-AND | .393 | .515 | .614 | .660 |
| B log-OR, 1 solve | .401 | .525 | **.619** | .650 |
| D log-OR, same anchors | .393 | .510 | .609 | .650 |
| E raw-OR (≡ 3:2:0) | .376 | .513 | .594 | .613 |
| C off | .393 | .509 | .594 | .623 |
| naive | .371 | .489 | .567 | .594 |

The four contrasts that isolate one factor each (active queries, points,
95% CI):

| contrast | isolates | HotpotQA @2 / @5 / @10 | MuSiQue @2 / @5 / @10 |
|---|---|---|---|
| G − A | shape, AND held | +1.3 [−0.8, +3.3] / +0.3 [−1.4, +2.2] / −0.5 [−1.6, +0.6] | −0.1 [−2.1, +1.8] / −0.5 [−3.3, +2.2] / −0.3 [−2.2, +1.6] |
| B − C vs E − C | shape, OR held | log: −0.3 / +0.3 / +1.3 ; raw: **−4.4** / **−2.1** / +0.8 | log: +0.2 / −0.2 / **+2.8** ; raw: −1.8 / −0.3 / −0.9 |
| A − E | AND-ness, raw held | **+3.6 [+1.9, +5.5]** / **+1.6 [+0.2, +3.2]** / **+1.0 [+0.2, +1.9]** | **+1.8 [+0.1, +3.8]** / +2.0 [−0.2, +4.3] / **+5.0 [+2.8, +7.3]** |
| G − B | AND-ness, log held | +0.8 [+0.2, +1.6] / −0.5 [−1.4, +0.5] / **0.0 [−0.5, +0.5]** | −0.2 [−0.6, 0.0] / +1.4 [+0.1, +2.8] / +1.0 [−0.4, +2.6] |

Reading, in order of confidence:

1. **Under AND pooling, shape is irrelevant.** G ≡ A within a point at
   every k on both datasets, although A's coherence vector is
   near-binary (0.8% / 1.7% of docs above 0.1 after min-max) and G's is
   graded (94% / 87%).
2. **Under OR pooling, shape is decisive.** Raw pooled PPR in the slot
   (E, which is exactly doubling the structural weight) costs 4.4 pts @2
   on HotpotQA and gains nothing @10 on MuSiQue; the log of the same
   vector (B) is +1.3 / +2.8 @10 with the @2 loss gone.
3. **Under raw shape, AND-ness is decisive.** A beats E at every k on
   both datasets with CIs excluding zero (up to +5.0 @10 on MuSiQue
   active queries). This is the comparison the report's own arm E
   contained but did not frame.
4. **Under log shape, AND-ness is worth at most ~1 point and the sign
   is not stable** across datasets or k. No G−B contrast clears zero on
   both datasets.

So the geometric mean and the log-compression are two *substitutable*
ways of keeping the coherence slot from re-injecting the seed-adjacent
cliff that raw pooled PPR produces. Either alone delivers the whole
aggregate effect (+1.1–1.7 @10 HotpotQA, +2.1–4.0 @10 MuSiQue over
off); both together (G) add nothing. The report's "AND-ness contributes
nothing" is true only in the presence of the log. On the shipped arm's
own terms (A vs E) the multi-anchor pooling *is* the mechanism that makes
it work; the report's real result is that a one-solve transform gets
there too at the aggregate level.

Two side findings. `D ≡ B` on ≤4-seed queries as predicted (top-10
overlap with A 0.902 vs 0.899). And on the single-seed queries where the
shipped gate turns convergence off (184 / 141 queries), a log-pooled term
from the solve that already exists adds +1.6 / +2.0 @10 over off, at
−1.6 @2 on HotpotQA and +1.4 @2 on MuSiQue.

### 5.4 Per hop: where they are *not* interchangeable

MuSiQue recall@10 by hop count (N=300):

| hop (n) | A | G | B | D | E | C | naive |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2 (151) | .689 | .689 | **.699** | .685 | .675 | .659 | .632 |
| 3 (104) | .601 | .612 | **.620** | .599 | .572 | .595 | .569 |
| 4 (45) | **.406** | .370 | .346 | .376 | .372 | .378 | .344 |

Paired bootstrap @10: 2-hop A−B −1.0 [−3.6, +1.3]; 3-hop A−B −1.9
[−5.1, +1.3]; **4-hop A−B +5.9 [+2.8, +9.4], A−G +3.5 [+1.1, +6.3],
A−C +2.8 [+0.6, +5.0], B−C −3.2 [−6.5, 0.0], B−naive +0.2 [−4.8, +5.0]**
(and @5: A−B +4.3 [+1.1, +7.8]).

At four hops the shipped near-binary geomean is the only arm that beats
convergence-off, the single-solve log arm is no better than naive
cosine, and A beats G — so at depth the *shape* of the geomean matters,
not only its pooling. A plausible mechanism (hypothesis, not shown): a
four-hop chain has four gold paragraphs and three or four anchors; the
documents reachable from every anchor are few and deep; a near-binary
lift drops them into the top-10, whereas a graded signal spreads its
weight over ~90% of the corpus and is averaged away by the 3× cosine
term.

Caveats, stated plainly: n=45 (23 active); 63 intervals were inspected
across the per-hop table, so one or two chance exclusions are expected;
the A>B, A>G, A>C, B≈naive pattern is internally consistent and in the
direction the design predicted, which is why it is reported as a live
hypothesis rather than dismissed — but it is not confirmed.
**Confirmatory run launched 2026-09-14: full MuSiQue-Ans dev (2,417
questions, ~400 four-hop) on the same script; result to be appended
below.**

### 5.5 Full MuSiQue dev (confirmatory, N = 2,417)

Completed 2026-09-14 15:50, same script, same settings.
`benchmarks/results/ablation_coherence_musique_n2417.json` (manifest:
commit `715e295`, dirty). Pool 17,629 paragraphs (title identity), KG
88,394 nodes / 590,347 edges (317 s build). 1,239 active queries, 296
fallback, mean seeds 2.4. Solve stage per query: `solve_multi` 0.333 s
vs pooled 0.077 s (4.3×). At this n the CIs on paired differences are
about ±0.5 pt.

| arm | @2 | @5 | @10 | active @10 |
|---|---:|---:|---:|---:|
| A geomean (shipped) | .336 | .444 | .507 | .513 |
| G log-AND | .339 | .445 | .505 | .511 |
| B log-OR, 1 solve | .333 | .441 | .507 | .508 |
| D log-OR, same anchors | .337 | .442 | .505 | .509 |
| E raw-OR (≡ 3:2:0) | .304 | .421 | .491 | .490 |
| C off | .337 | .441 | .498 | .496 |
| naive | .314 | .411 | .470 | .459 |

Aggregate contrasts (unconditional, pts, 95% CI, @2 / @5 / @10):

- **B − A: −0.27 [−0.82, +0.27] / −0.29 [−0.83, +0.24] / +0.03 [−0.53,
  +0.60].** Interchangeable in aggregate, confirmed at eight times the
  sample.
- G − A: +0.36 [+0.01, +0.70] / +0.11 [−0.28, +0.50] / −0.13 [−0.51,
  +0.24]. Shape irrelevant under AND pooling, confirmed.
- **A − E: +3.14 [+2.55, +3.73] / +2.27 [+1.71, +2.83] / +1.55 [+1.03,
  +2.07].** AND-ness under raw shape, confirmed.
- G − B: +0.62 [+0.17, +1.09] / +0.40 [−0.02, +0.84] / −0.17 [−0.64,
  +0.30]. AND-ness under log shape: at most 0.6 pt, at @2 only.
- **A − C @10: +0.87 [+0.46, +1.27]**; B − C +0.90 [+0.33, +1.46];
  G − C +0.73 [+0.37, +1.11]. The shipped convergence gain over
  convergence-off on full MuSiQue dev is **+0.9 @10, not the +2.2 the
  0.4.0 CHANGELOG quotes from N=300** (this run's own N=300 gave +2.14;
  the small sample overstated it 2.5×).

Per hop, recall@10 and paired CIs:

| hop (n) | A | G | B | C | naive | A − B | A − C | B − C | G − B |
|---|---:|---:|---:|---:|---:|---|---|---|---|
| 2 (1,252) | .578 | .578 | **.588** | .572 | .549 | −1.00 [−1.88, −0.16] | +0.64 [+0.16, +1.12] | +1.64 [+0.80, +2.52] | −1.04 [−1.76, −0.32] |
| 3 (760) | **.491** | .491 | .480 | .480 | .450 | +1.05 [+0.04, +2.06] | +1.03 [+0.13, +1.93] | −0.02 [−0.94, +0.88] | +1.03 [+0.26, +1.82] |
| 4 (405) | **.317** | .311 | .308 | .305 | .264 | +0.91 [−0.02, +1.83] | +1.26 [+0.39, +2.14] | +0.35 [−0.64, +1.36] | +0.27 [−0.47, +1.01] |

Active-only A − B @10: 2-hop −0.29 [−1.63, +0.96]; 3-hop +1.10 [−0.18,
+2.35]; 4-hop +1.24 [+0.03, +2.44].

**Verdict on the N=45 hypothesis: direction confirmed, magnitude
inflated six-fold.** At n=405 the geomean's four-hop edge over the
log-pooled arm is +0.9 @10 with the CI touching zero (+1.2 on active
queries, just clearing it), not +5.9. And the log arm is *not* "no
better than naive" at four hops on full dev: B − naive is +4.38 [+2.76,
+5.99] there. What does hold, with CIs clear of zero, is a
hop-dependent crossover in which every effect is about one point: the
log-pooled arm is the better default at two hops (+1.0 over A, driven by
the single-seed queries the gate excludes; on active two-hop queries the
two tie); the geomean is the better arm at three hops (+1.05 over B,
where the log arm gains nothing over off and G − B = +1.03 says the
AND-pooling is what does it); and the geomean keeps a ~1-point edge at
four hops.

Two further things this run settles:

- **The hybrid "A when ≥2 seeds, log-pooled otherwise" is a wash, not a
  free win**: vs A it is +0.31 @10 [−0.11, +0.73] for −0.49 @2 [−0.90,
  −0.09]. On the 1,178 single-seed queries the log term costs 1.00 @2
  [−1.85, −0.18] for +0.64 @10 [−0.19, +1.51]. Withdrawn as an option.
- **This is the first full-MuSiQue-dev measurement of hubmesh.** A vs
  naive: +2.22 @2 [+1.40, +3.06], +3.32 @5 [+2.53, +4.12], **+3.67 @10
  [+3.00, +4.39]**; per hop @10 **+2.88 / +4.12 / +5.29** (n = 1,252 /
  760 / 405, every CI clear of zero). The "gain grows with hop count"
  pattern the council called unstable at N=300 is monotonic at full dev
  for the shipped arm, and not for the log arm (+3.87 / +3.07 / +4.38).
  The README's N=300 per-hop figures (+6.0 / +3.2 / +5.0) were noisy and
  should be replaced by these. Unlike HotpotQA, hubmesh beats naive at
  recall@2 on MuSiQue. Of the +3.67 @10 over naive, +2.81 [+2.26, +3.38]
  is cosine + pooled-PPR fusion (C − naive) and +0.87 is the coherence
  slot.

---

## 6. What this means for the decisions on the table

1. **Scoring default: keep A; expose the one-solve log-pooled form as a
   documented cheap mode; do not ship the hybrid.** Full dev (§5.5) says
   the two are interchangeable in aggregate (B − A = +0.03 @10 at
   n=2,417), that the geomean is about one point better at three and
   four hops and about one point worse at two hops, and that the hybrid
   trades −0.5 @2 for +0.3 @10. The product's claim is deep composition,
   so A stays. B is the right answer for latency budgets or top-k ≤ 5,
   at 4.3× less solve time, and should become a one-line config option
   (e.g. `convergence="log_pooled"`) whose docstring states the per-hop
   trade. The bge-m3 question (item 3) could still overturn this and is
   the next run.
2. **Rewrite the convergence claim wherever it appears** (CHANGELOG
   0.4.0, README, `PlannerConfig` docstring, the preprint): "geometric
   mean of per-anchor PPR mass" stays accurate for what A computes; the
   justification "reachable from every anchor beats flooded from one" is
   supported by A vs E on every sample; add "a log-compressed
   single-solve pooled signal reaches the same aggregate recall at one
   PPR solve; the geomean's advantage is ~1 pt at three and four hops";
   replace "+2.2 pts MuSiQue" with "+0.9 @10 on full MuSiQue dev (+2.2
   was N=300)"; replace the README's N=300 per-hop row with the full-dev
   +2.9 / +4.1 / +5.3; and replace "single-seed queries are unaffected
   (neutral)" with the gate statement plus the measured single-seed
   effect (a log term there is +0.6 @10 for −1.0 @2).
3. **The bge-m3 question is still unmeasured.** Everything in §5 is
   MiniLM. The product problem the reviewers weigh most is the −9.09 @2
   on bge-m3, and arm E's @2 collapse is the same kind of failure
   (structural weight overwhelming cosine at the top). Running {A, B, G,
   C} on the two bge-m3 pilots is the run that says whether the slot's
   shape is a lever on that regression. The new script needs
   `--embed-model` plus the `--batch-size 8 --max-seq-len 1024` handling
   from `run_hubmesh_titlebody.py` (~10 lines) and ~30 min on MPS;
   founder-gated because of the OOM history and the 99%-full disk.
4. **Baselines before the preprint.** BM25, hybrid RRF, a cross-encoder
   over ANN top-50, and the complementarity arm `rerank(ANN ∪ hubmesh)`.
   I agree with the council that this is the must-win and that the
   complementarity arm is the go/no-go for the "retrieval quality"
   framing; the assurance framing (connectivity, provenance, explicit
   absence) is where to land if it fails. One day of compute on the
   pilots.
5. **README, gated:** embedder grid as the headline with the bge-m3 row;
   +29.8 → +19.5 (`structural_only`, packing held constant); Design diagram redrawn for the KG
   path; a non-goals section; the ICOMP line is the founder's call.
6. **Priority:** cite Aethel, Seed-Anchored and EdgeMem with the delta
   wording from the comparison document's §6.4, with one edit — where
   it lists multi-source convergence as "uncontested", add that its
   measured value is bounded by §5. A dated write-up with the embedder
   grid, the `structural_only` attribution and this ablation is the
   only thing that fixes the citation gap; the ablation makes that
   write-up more credible, not less.
7. **LightRAG lane:** post the pilot or close it publicly; blocked on
   the Moonshot balance either way.

Not done here, deliberately: no scoring default changed, no README or
CHANGELOG text changed, nothing committed, nothing posted, no bge-m3
run. Files added: `benchmarks/run_ablation_coherence.py`,
`benchmarks/results/ablation_coherence_hotpotqa_n500.json`,
`benchmarks/results/ablation_coherence_musique_n300.json`,
`benchmarks/results/ablation_coherence_musique_n2417.json`, this
document.
