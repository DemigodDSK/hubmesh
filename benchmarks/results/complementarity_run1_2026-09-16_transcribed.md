# Complementarity protocol — run 1 (2026-09-15/16): record of a lost run

**Status: the result files of this run no longer exist.** This document is
transcribed from the run log and from the single examination pass; it is
not a result file and carries no per-question records. It is kept so that
what was seen is on record before the experiment was re-run.

## What happened

- Stage R (candidate generation) for all six configurations ran on
  2026-09-15 from a worktree at `1e2297f` (Amendment 1), outputs in a
  session-scoped scratch directory.
- Stage K (reranking) ran from a clean worktree at `bc289d0`
  (Amendment 2), 2026-09-15 21:49 → 2026-09-16 14:08 EDT, all six on
  Metal fp16, no fallback. In-run parity gate (library dynamic-padding
  path vs fixed-shape, same weights, one question): max |Δlogit|
  0.0078 in five configurations, 0.0117 in HotpotQA/MiniLM; same top-5
  and top-10 in all six. Timing-pass vs scoring-pass logits differed by
  0.0 for every overlapping pair in every configuration. Metal driver
  memory ≤ 2.13 GB throughout.
- Between 11:44 and 13:12 EDT on 2026-09-16 the machine's `/usr/bin/git`
  (Xcode's) began refusing to run until an Xcode licence was accepted
  (Command Line Tools 27.0 had been installed on 09-14). The manifests of
  the last two configurations (HotpotQA/MiniLM, MuSiQue/MiniLM) therefore
  recorded `git_commit: null`; their content hashes (`src_sha256
  7f30b326…`, `benchmarks_sha256 11cc2dd4…`, `harness_sha256 8bd86259…`,
  `protocol_sha256 620f9e19…`) were identical to the other four files'.
- 2026-09-16 14:11 EDT: the four configurations with complete manifests
  were examined once (below). The two without a commit were **not**
  examined (the examination script stopped at their header line) and were
  set aside for a re-run at the same commit with the Command Line Tools
  git, launched 14:50.
- 2026-09-16 ~15:23 EDT: the session's scratch directory was purged when
  the session ended, taking the six result files, the six stage files,
  the score caches, the re-run in progress and the logs. Copies had not
  yet been written into the repository (the copy step was interrupted).
  Consequence: no per-question records exist for run 1; the numbers below
  are the only record, and the whole experiment was re-run (run 2) from a
  durable directory. Run 2 is the run whose files are committed; run 1 is
  reported here for comparison only.

## Provenance of the four examined files (all identical except as noted)

commit `bc289d0` (clean tree) · src `7f30b326` · benchmarks `11cc2dd4` ·
harness `8bd86259` · protocol `620f9e19` · reranker
`BAAI/bge-reranker-v2-m3` on mps, parameters `torch.float16`, buckets
128/256/384/512, batch 8, raw logits · stage files: qid order matched the
sample, every candidate in the pool, A4/A6/A7 exactly 50 distinct, prefix
relations held, gold lists = full gold lists (0 unresolved) · reference
arms extended to 10 (stored 5-prefixes verified) · timing queries 100
(primary) / 50 (others).

| configuration | embed | pairs scored | cache hits | parity max | driver GB (last) |
|---|---|---:|---:|---:|---:|
| 2Wiki n=1000 bge-m3 | mps/fp16 | 94,230 | 0 | 0.0078 | 1.96 |
| 2Wiki n=1000 MiniLM | mps/fp32 | 52,744 | 31,536 | 0.0078 | 1.93 |
| HotpotQA n=500 bge-m3 | mps/fp16 | 46,240 | 0 | 0.0078 | 2.11 |
| MuSiQue n=300 bge-m3 | mps/fp16 | 31,902 | 0 | 0.0078 | 2.02 |

## Registered outcome (primary configuration: 2Wiki, N=1000, bge-m3)

- **A6 − A4, complete-evidence@5: +0.30 points, 95% paired-bootstrap
  interval [−0.10, +0.80], n = 1000 → "no useful gain"** (upper bound
  below +3.0). Discordant questions: A6 better on 4, A4 better on 1.
- Regression guard, comparison-type questions (n = 236): A6 − A4 =
  +0.00 [+0.00, +0.00]; both arms score 1.000 on every question of the
  slice, so the guard is met trivially (slice saturated).
- Latency rule: p95(A6 composed) / p95(A4 composed) = 1.054 ≤ 1.5, met;
  over the 100 timed questions, rerank medians A4 16,545 ms, A6 16,593 ms,
  A7 18,028 ms, A8 16,848 ms for one 50-candidate list on this machine
  (fp16, Metal, fixed-shape batching); retrieval-stage medians embed 36.5,
  dense 1.1, BM25 11.0, RRF 0.2, hubmesh 143.2 ms.

| arm | ce@5 | ce@10 | r@5 | r@10 |
|---|---:|---:|---:|---:|
| A1 dense top-5 | 0.426 | 0.485 | 0.717 | 0.752 |
| A2 BM25 top-5 | 0.313 | 0.399 | 0.643 | 0.698 |
| A3 hybrid top-5 | 0.386 | 0.461 | 0.703 | 0.745 |
| A4 hybrid 50 → rerank (control) | 0.400 | 0.466 | 0.714 | 0.749 |
| A5 hubmesh top-5 | 0.455 | 0.529 | 0.741 | 0.785 |
| A6 hybrid30 ∪ hub20 → rerank (treatment) | 0.403 | 0.470 | 0.717 | 0.753 |
| A7 dense30 ∪ hub20 → rerank | 0.399 | 0.457 | 0.714 | 0.746 |
| A8 hubmesh ≤50 → rerank (variable count) | 0.402 | 0.464 | 0.715 | 0.748 |

Pre-listed contrasts, ce@5 / ce@10 / r@5 / r@10 (points, 95% interval):
A7−A4 −0.10 [−1.00, +0.80] / −0.90 [−2.20, +0.30] / +0.03 [−0.42, +0.47] / −0.30 [−0.95, +0.35];
A8−A4 +0.20 [−0.70, +1.10] / −0.20 [−1.60, +1.10] / +0.10 [−0.40, +0.60] / −0.03 [−0.72, +0.68];
A6−A8 +0.10 [−0.80, +1.00] / +0.60 [−0.60, +1.90] / +0.20 [−0.28, +0.68] / +0.42 [−0.20, +1.07];
A4−A3 +1.40 [−0.60, +3.40] / +0.50 [−1.50, +2.50] / +1.12 [+0.05, +2.20] / +0.35 [−0.65, +1.38];
A5−A1 +2.90 [+1.40, +4.40] / +4.40 [+2.80, +6.00] / +2.42 [+1.60, +3.27] / +3.33 [+2.50, +4.15];
A5−A3 +6.90 [+4.90, +9.00] / +6.80 [+4.80, +8.90] / +3.85 [+2.75, +4.97] / +4.03 [+2.98, +5.08];
A6−A5 −5.20 [−7.40, −3.10] / −5.90 [−8.10, −3.70] / −2.42 [−3.58, −1.32] / −3.27 [−4.40, −2.15];
A3−A1 −4.00 [−5.70, −2.40] / −2.40 [−4.10, −0.80] / −1.43 [−2.35, −0.52] / −0.70 [−1.60, +0.20].
Slice "fewer than two hubmesh seeds" (n = 526): A6 − A4 +0.00; A4 0.316, A6 0.316, A5 0.378, A3 0.300.
Diagnostics: questions with ≥ 2 seeds 47.4%; mean |hybrid30 ∩ hub20| 12.81; gold paragraphs without entities 0.0%; unresolved gold 0; A8 candidates min 16, mean 46.22, below 50 in 493 questions.

## Secondary configurations (examined at the same time)

| configuration | A6−A4 ce@5 | guard | latency ratio | A1 | A3 | A4 | A5 | A6 | A7 | A8 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2Wiki MiniLM (n=1000) | +1.00 [+0.40, +1.70] | +0.00 (n=236, saturated) | 1.023 | 0.288 | 0.331 | 0.402 | 0.375 | 0.412 | 0.392 | 0.405 |
| HotpotQA bge-m3 (n=500) | +0.40 [+0.00, +1.00] | +0.00 (n=100, saturated) | 1.079 | 0.770 | 0.668 | 0.854 | 0.738 | 0.858 | 0.844 | 0.854 |
| MuSiQue bge-m3 (n=300) | +0.33 [−1.00, +2.00] | no comparison slice | 1.013 | 0.220 | 0.157 | 0.290 | 0.273 | 0.293 | 0.287 | 0.290 |

(ce@5; A2 = 0.313 / 0.462 / 0.093.) Other pre-listed contrasts, ce@5:
2Wiki MiniLM — A7−A4 −1.00 [−2.50, +0.50]; A8−A4 +0.30 [−1.20, +1.80]; A6−A8 +0.70 [−0.70, +2.20]; A4−A3 +7.10 [+4.80, +9.40]; A5−A1 +8.70 [+6.70, +10.70]; A5−A3 +4.40 [+2.10, +6.60]; A6−A5 +3.70 [+1.30, +6.20]; A3−A1 +4.30 [+2.30, +6.30]; slice < 2 seeds (n=526) A6−A4 +0.76 [+0.19, +1.52]; diagnostics ≥2 seeds 47.4%, overlap 13.79, A8 min 26 / below 50 in 28.
HotpotQA bge-m3 — A7−A4 −1.00 [−2.40, +0.20]; A8−A4 +0.00 [−1.20, +1.20]; A6−A8 +0.40 [−0.60, +1.60]; A4−A3 +18.60 [+14.60, +22.60]; A5−A1 −3.20 [−6.60, +0.00]; A5−A3 +7.00 [+3.00, +10.80]; A6−A5 +12.00 [+8.00, +16.00]; A3−A1 −10.20 [−13.80, −6.60]; slice < 2 seeds (n=185) +0.54 [+0.00, +1.62]; diagnostics ≥2 seeds 63.0%, overlap 14.24, gold without entities 0.7%, A8 min 43 / below 50 in 15.
MuSiQue bge-m3 — A7−A4 −0.33 [−2.67, +2.00]; A8−A4 +0.00 [−2.33, +2.33]; A6−A8 +0.33 [−1.33, +2.00]; A4−A3 +13.33 [+9.00, +18.00]; A5−A1 +5.33 [+1.33, +9.33]; A5−A3 +11.67 [+7.33, +16.00]; A6−A5 +2.00 [−3.00, +6.67]; A3−A1 −6.33 [−10.33, −2.33]; by hops A6−A4 ce@5: 2-hop (n=151) +0.00, 3-hop (n=104) +0.96 [+0.00, +2.88], 4-hop (n=45) +0.00 with every arm at 0.000; diagnostics ≥2 seeds 52.0%, overlap 13.36, A8 min 42 / below 50 in 21.

HotpotQA/MiniLM and MuSiQue/MiniLM: completed (13:12 and 14:08 EDT) but
never examined; lost with the rest.

## Post-hoc diagnosis on the primary file (exploratory; not a registered analysis)

Computed after the registered outcome was read, as the decision table's
"no useful gain → diagnose a specific suspected limitation" step:

- A4 already had complete evidence at 5 for 400 of 1,000 questions.
- hubmesh's top-20 held a gold paragraph absent from the hybrid top-30 in
  106 questions (109 paragraphs); 77 of those 109 were absent from the
  entire hybrid top-50, i.e. evidence the control never sees.
- The cross-encoder's rank for those 109 paragraphs within A6's 50: 16 in
  ranks 1–5, 15 in 6–10, 30 in 11–20, 48 beyond 20. A6 gained complete
  evidence at 5 on 4 of the 106 questions and lost it on none; A4 already
  had it on 6.
- Complete evidence contained anywhere in the candidate list (before
  reranking): A4 0.572, A6 0.612, A7 0.624, A8 0.645; mean gold recall of
  the lists 0.806 / 0.832 / 0.837 / 0.848.
- By question type, ce@5 (A4 / A6 / A5): compositional (n=407) 0.270 /
  0.278 / 0.381; bridge_comparison (n=238) 0.017 / 0.017 / 0.021;
  comparison (n=236) 1.000 / 1.000 / 0.987; inference (n=119) 0.420 /
  0.420 / 0.521.

Reading: the candidate lists that include hubmesh's paragraphs contain
complete evidence for more questions than the control's list, but the
cross-encoder ranks most of the added gold outside its top-10, so the
registered metric barely moves. "The reranker discards the added
evidence" is the interpretation this suggests; it is a hypothesis for a
separate, pre-registered test, not a conclusion of this experiment.
