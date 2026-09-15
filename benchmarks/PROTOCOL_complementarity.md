# Pre-registered protocol: does hubmesh add evidence a strong retriever misses?

Frozen before any run. The commit that introduces this file is the
protocol's identity; every result file produced under it records that
commit, this file's SHA-256, and the content hashes of the code that ran.
Nothing below is tuned: every knob is fixed here a priori, and the
results are examined once, after all arms complete.

## Question

Given a credible retrieval stack (hybrid BM25 + dense, reranked), does
adding hubmesh's graph-expanded candidates — at the **same** reranking
budget — increase the fraction of questions for which **all** supporting
paragraphs are in the final top-5?

## Data

- **Primary (held out):** 2WikiMultihopQA dev (corrected release
  `data_ids_april7`, official Alab-NII distribution), **1,000 questions**,
  seed-0 permutation, pooled distractor contexts (10 paragraphs per
  question), **exact paragraph identity** = title + text hash. Gold = the
  distinct supporting-fact paragraphs. Held out from hubmesh development:
  no hubmesh code or result before this protocol references 2Wiki (the
  only mention is a CONTRIBUTING wish-list line); a dataset cache with a
  similar name existed on the development machine in August 2026 and was
  deleted in a disk cleanup, origin not recorded. That history is
  disclosed rather than claimed away.
- **Secondary, already seen:** HotpotQA dev N=500 seed 0 (title
  identity); MuSiQue-Ans dev N=300 seed 0 (title + text identity).
- Document representation for every lane: `"{title}\n\n{text}"`.

## Systems (all knobs fixed)

- Dense: `BAAI/bge-m3` (primary; batch 8, max sequence 1024, normalized)
  and `all-MiniLM-L6-v2` (secondary). Query embedding timed per query.
- BM25: `rank_bm25.BM25Okapi`, tokens = lowercase `\w+`, library defaults.
- Hybrid: reciprocal-rank fusion, constant 60, over the top-200 of the
  dense and BM25 rankings.
- hubmesh: KG mode, released defaults (3:1:1 weights, `integration="sum"`,
  convergence on), spaCy `en_core_web_sm` KG, one `retrieve(top_k=50,
  budget_tokens=10_000)` call per query; the top-5 and top-20 prefixes of
  that call are the top-5 / top-20 arms (greedy packing makes prefixes
  identical to shorter calls; the timing is therefore an upper bound for
  the top-5 arm).
- Reranker: `BAAI/bge-reranker-v2-m3` cross-encoder, max length 512,
  batch 16, input (question, document text); ties in score break by
  candidate order.

## Arms (reranked arms all rerank exactly 50 candidates)

| arm | candidates to reranker | role |
|---|---|---|
| A1 dense top-5 | none | reference |
| A2 BM25 top-5 | none | reference |
| A3 hybrid top-5 | none | reference |
| **A4** hybrid top-50 | 50 | **equal-budget control** |
| A5 hubmesh top-5 | none | reference |
| **A6** hybrid top-30 ∪ hubmesh top-20 | 50 | **treatment** |
| A7 dense top-30 ∪ hubmesh top-20 | 50 | secondary (dense-only base) |
| A8 hubmesh top-50 | 50 | secondary (graph as sole candidate source) |

Union rule for A6/A7: the base list's first 30, then hubmesh's first 20
in hubmesh order skipping ids already present, then backfill from the
base list's rank 31 onward until exactly 50 distinct ids (fewer only if
the corpus is smaller). The candidate count is recorded per query.

## Metrics

- **Primary: complete-evidence@5** — 1 if every distinct gold paragraph
  id is among the 5 distinct returned ids, else 0. A gold paragraph
  absent from the pool counts as failure; the denominator is every
  sampled question. This is a retrieval metric and says nothing about
  answer correctness.
- Secondary: complete-evidence@10, fractional recall@5 and @10.
- Sample-size rationale: the primary metric is binary per question; with
  roughly 10–15% discordant pairs at n = 1,000 the paired-bootstrap
  interval half-width is about 2 points, so the 3-point threshold below
  is resolvable.

## Primary contrast and decision rule

**A6 − A4, complete-evidence@5, 2Wiki, bge-m3.** Paired bootstrap over
questions, 10,000 resamples, numpy seed 0, percentile 95% interval.

- **Pass (screening):** observed difference ≥ +3.0 points **and** the
  interval's lower bound > 0. This is evidence of improvement; it does
  not establish that the true effect is at least 3 points.
- **No useful gain:** the interval's upper bound < +3.0, whether or not
  it excludes zero.
- **Inconclusive:** anything else. Predefined handling: no default
  changes and no new arms; report the three diagnostics below and stop.

Diagnostics computed for every run, examined only if inconclusive:
(1) fraction of questions with ≥2 hubmesh seeds; (2) mean overlap
|hybrid top-30 ∩ hubmesh top-20|; (3) fraction of gold paragraphs whose
document node has no extracted entities.

## Regression guard

- **Slice: 2Wiki questions whose `type` annotation is `comparison`** —
  a question-type slice, reported as "comparison-type questions"; it is
  not claimed to represent simple questions in general. Size reported
  from the sample, not chosen.
- Rule: observed A6 − A4 on the slice ≥ −1.0 point (point estimate;
  interval reported). Guards against large regressions only.
- Secondary slice: questions with fewer than two hubmesh seeds.

## Latency (feasible measurement, labelled as such)

The two models cannot both be resident on the 8 GB development machine,
so latency is measured in **stages** with one model resident at a time:

- Stage R (embedder resident): per query — query embedding, dense
  search, BM25 scoring, RRF, hubmesh retrieval — each timed separately.
- Stage K (reranker resident): per query — reranking one 50-candidate
  list (A4's), timed; every 50-candidate arm reranks the same number of
  pairs, so this measurement is applied to A4, A6, A7 and A8 alike.

**Composed latency** for an arm = the sum, **per query**, of the stage
times that arm uses (e.g. A6 = embed + dense + BM25 + RRF + hubmesh +
rerank); the median and p95 are taken over the per-query sums, never by
adding stage percentiles. Composed latency is an **estimate** of a
deployment that keeps both models resident: it excludes model switching,
cold start, process boundaries and any request overhead, all of which a
real deployment pays. Absolute milliseconds are reported alongside the
ratio. Rule: p95(A6 composed) ≤ 1.5 × p95(A4 composed). Hardware, device
and batch sizes are recorded in the manifest.

## Provenance

Per-question records for every arm (candidate ids, final ids, metric
values, stage timings, seed count, overlap). Manifest with the commit,
whole-tree dirty flag, `src_sha256`, `benchmarks_sha256`, harness hash,
this file's SHA-256, embedding/reranker models, device and batch sizes.
Results are examined once, after all arms complete.

## Decision table (engineering and product kept separate)

| result | engineering next | product next |
|---|---|---|
| pass at acceptable latency | bounded expansion API; MCP returns packed context | validate with partners on representative tasks |
| gain confined to one slice or embedder | make that mode explicit, default off elsewhere | pursue only if a partner has that workload |
| inconclusive | run the three diagnostics, no new arms | keep interviewing, pitch no numbers |
| no useful gain | diagnose a specific suspected limitation or reconsider the proposition; a narrower application only with separate evidence | proposition weak; partner tasks decide |
| partners ask for evaluation tooling unprompted | package the harness | test as its own hypothesis with its own buyer |

Run: `python benchmarks/run_complementarity.py --dataset 2wiki --n 1000 --embed-model BAAI/bge-m3 --out benchmarks/results/complementarity_2wiki_n1000_bge-m3.json`
