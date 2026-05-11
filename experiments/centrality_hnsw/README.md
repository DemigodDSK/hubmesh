# Centrality-aware HNSW: two attempts, one narrow win

This dir contains two empirical tests of ideas from the CR-HNG / NNSI-HNSW
design docs. **Attempt 1** (level assignment bias) failed cleanly across
seeds. **Attempt 2** (neighbour-selection ordering bias with the same
multi-component pattern) shows a small but reproducible win in a narrow
sweet spot.

## Attempt 2: multi-component biased neighbour ordering (`hnsw_mc.py`)

**Mechanism.** Keep HNSW Algorithm 4's geometric rejection rule intact
(uses raw distances). Only bias the *order* in which candidates are
considered, using an adjusted distance:

    d'(c, q) = d(c, q) · (1 − α_H · H_norm(c) − α_G · G_norm(c))

where H is log-normalised online degree and G is centroid proximity.
This nudges hubby/central candidates to be considered earlier; the
rejection rule still protects geometric correctness.

**Result on MNIST 784-d, 3 seeds, wH=0.1, wG=0.2:**

| n | M | mean Δ recall@5 | per-seed consistency |
|---|---|---:|---|
| 3000 | 4 | +0.20 pts | all positive |
| **3000** | **8** | **+0.62 pts** | **all positive, std 0.0002** |
| 3000 | 16 | +0.07 pts | mixed |
| 5000 | 4 | −0.17 pts | mostly negative |
| 5000 | 8 | +0.28 pts | mostly positive |

**Sweet spot at n=3000, M=8: +0.62 pts** at recall@5, +0.62 at recall@8,
+0.35 at recall@12. Every seed positive at low ef. Build time 1.2× vanilla
(vs 2× for the two-pass approach in Attempt 1).

**Limits:** narrow window. No effect at HNSW's typical M=16. Doesn't scale
to larger n cleanly with the same weights. Pushing the H/G weights higher
(0.2/0.3) destroys the effect — Algorithm 4's geometry is fragile to
strong bias.

**Reproduce the win:**
```bash
python run_mc.py --n 3000 --M 8 --seeds 0 1 2 \
    --wP 0 --wD 0 --wH 0.1 --wG 0.2 --ef-search 5 8 12 20
```

---

## Attempt 1: centrality-biased level assignment (`centrality_hnsw.py`)

This experiment tests one of the central hypotheses from the
"Centrality-Routed Vector Index Design" / "NNSI-HNSW" design docs:

> Replace HNSW's random level assignment
>
>     ℓ = floor(-ln(U) · m_L),   m_L = 1/ln(M)
>
> with a centrality-biased version
>
>     ℓ = floor(-ln(U) · (m_L + m_1 · I_i))
>
> where I_i is the node's centrality. High-I nodes are promoted to upper
> layers more often, becoming "highway hubs" that searches traverse
> early — intended to improve recall at a fixed search budget.

**Empirical answer on MNIST 784-d: it doesn't help.** Across multiple
seeds, two centrality measures, several `M`/`m_1` settings, and ef
sweeps, centrality-biased promotion either matches vanilla HNSW or
loses to it, with run-to-run variance dwarfing any systematic effect.

## What we built

- **`mini_hnsw.py`** — a faithful HNSW implementation in pure
  Python+NumPy (~330 lines). Sanity-checked against `hnswlib` on a
  small clustered dataset: matches recall exactly at every `ef_search`
  setting (Δ ≤ 0.05% at ef=10, exact match at ef ≥ 25). This validates
  that our implementation is correct and our test of the hypothesis
  is sound.

- **`centrality_hnsw.py`** — same algorithm with a swappable
  `level_fn`. Two centrality measures provided:
  - `degree` — two-pass build: first pass uses vanilla random levels,
    then layer-0 degree centrality is computed and used to bias the
    second-pass level assignment.
  - `centroid` — single-pass: proximity to the data centroid in
    vector space, used directly as `I_i`.

- **`bench.py`** — recall-vs-QPS evaluation harness, agnostic to the
  index. Brute-force ground truth, configurable k and ef sweep.

- **`run_bench.py`** / **`run_mnist.py`** — drivers. The latter loads
  cached MNIST embeddings and runs head-to-head vanilla vs biased.

- **`sanity_check.py`** — verifies our minimal HNSW matches hnswlib.

## Results

### Setup

- **Data:** MNIST 784-d, L2-normalised, sampled at n ∈ {3000, 5000}.
- **Index:** M ∈ {4, 8, 16}, ef_construction=100–200.
- **Bias strength:** m_1 ∈ {1.0, 2.0, 3.0, 5.0}.
- **Centrality:** degree (post-hoc, two-pass) and centroid-proximity.
- **Evaluation:** 200 queries; brute-force ground-truth recall@10;
  ef_search sweep ∈ {3, 5, 8, 12, 20, 40, 80}.

### Headline (centroid centrality, M=4, m_1=3, 3 seeds, recall@10)

| ef | vanilla (mean) | centrality (mean) | Δ (pts) |
|---:|---:|---:|---:|
| 3 | 0.876 | 0.874 | **−0.20** |
| 5 | 0.876 | 0.874 | **−0.20** |
| 8 | 0.876 | 0.874 | **−0.20** |
| 12 | 0.902 | 0.903 | +0.02 |

**Per-seed variance (±1.35 pts) is roughly 7× larger than the mean
effect.** Any single seed showing a "win" is not reproducible.

### Same pattern, different centrality

With **degree** centrality (two-pass build) at M=4, m_1=5: centrality
loses by 0.5–0.8 pts at every ef ≤ 20. The two-pass cost is wasted.

At M=8, M=16 (denser graphs), both vanilla and centrality saturate at
~99% recall by ef=20; no headroom to differentiate.

## Why this likely doesn't help

Three explanations consistent with what we observed:

1. **HNSW's heuristic neighbour selection already does centrality-aware
   diversification.** Algorithm 4 in Malkov & Yashunin (2018) explicitly
   diversifies neighbours by preserving "long-range" edges. The result
   is that high-betweenness nodes already end up well-connected without
   any explicit centrality bias on level assignment.

2. **Random promotion is a feature, not a bug.** The exponential decay
   `Pr(ℓ ≥ k) = M^{-k}` produces a hierarchy where each layer has
   `~1/M` the nodes of the layer below. Centrality biasing
   concentrates the upper layers (more nodes at L1+, level distribution
   shifts up — we observed mean level 0.07 → 0.16 in our runs). Denser
   upper layers mean *more* neighbours to check during the greedy
   descent, slowing search rather than speeding it.

3. **Post-hoc centrality is a stale signal.** Layer-0 degree from a
   vanilla build reflects the vanilla hierarchy, not the optimal
   topology for the biased one. The two-pass approach assumes the
   centrality you'd want is the centrality you get from a different
   hierarchy — a circularity that may have no fixed point.

## What this *doesn't* refute

The broader CR-HNG / NNSI-HNSW vision has several independent components.
This experiment only tests one of them:

- ✓ Tested + refuted: centrality-biased **level assignment**.
- ✗ Not tested: **multi-entry beam search** routed by community.
- ✗ Not tested: **explicit bridge-node promotion** to a separate overlay
  layer.
- ✗ Not tested: **filter-aware native routing**.
- ✗ Not tested: **streaming maintenance** under updates.

A future iteration could test any of these in isolation; this one
isolates the specific claim that centrality-biased promotion improves
recall on standard ANN benchmarks. It does not.

## Reproduce

```bash
# Sanity-check that our HNSW matches hnswlib
python sanity_check.py

# Head-to-head on synthetic clustered Gaussian
python run_bench.py --n 5000 --dim 128 --M 8 --m1 3.0 \
    --ef-search 3 5 8 12 20

# Head-to-head on real MNIST (requires the cached embeddings from
# hubmesh's run_real.py)
python run_mnist.py --n 5000 --M 4 --m1 3.0 --centrality centroid \
    --ef-search 3 5 8 12 --seed 0
python run_mnist.py --n 5000 --M 4 --m1 3.0 --centrality centroid \
    --ef-search 3 5 8 12 --seed 1
python run_mnist.py --n 5000 --M 4 --m1 3.0 --centrality centroid \
    --ef-search 3 5 8 12 --seed 2
```

## Honest framing of this experiment

This is a clean negative result. The design docs proposed score-based
promotion as a likely improvement; we tested the proposal carefully
(faithful HNSW implementation, validated against hnswlib, multiple
centralities, multiple seeds) and the proposal didn't hold.

That's useful to know. It saves anyone building on these docs from
spending engineering effort on a path that doesn't pay off. The win we
actually got with `hubmesh` (the retrieval-planner layer, +4.92 pts
recall@10 on full HotpotQA) came from a different mechanism — combining
cosine and PPR over an entity-linked KG at *query time*, not at *index
construction time*.

The intellectual lineage from the NNSI thesis (multi-component
multiplicative scoring) generalised to the retrieval layer, but not
to the index layer.
