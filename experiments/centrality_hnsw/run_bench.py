"""Head-to-head: vanilla HNSW vs centrality-biased HNSW on the same dataset.

This is the test of the design-doc hypothesis: does replacing HNSW's
random level assignment with a centrality-biased one improve the recall
vs QPS frontier?

Methodology:
  • Same data, same M, same ef_construction.
  • Three indices built: hnswlib reference, vanilla MiniHNSW, biased
    CentralityHNSW.
  • Same query set; sweep ef_search; report recall@k + QPS at each.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import argparse
import numpy as np
import hnswlib

from mini_hnsw import MiniHNSW
from centrality_hnsw import CentralityHNSW
from bench import brute_force_ground_truth, recall_at_k


def make_data(n: int, dim: int, n_clusters: int | None = None, seed: int = 0):
    rng = np.random.default_rng(seed)
    if n_clusters is None:
        n_clusters = max(5, n // 100)
    centroids = rng.normal(0, 5.0, size=(n_clusters, dim))
    labels = rng.integers(0, n_clusters, size=n)
    X = centroids[labels] + rng.normal(0, 1.0, size=(n, dim))
    queries = X[rng.choice(n, size=min(200, n // 10), replace=False)]
    queries = queries + rng.normal(0, 0.3, size=queries.shape)
    return X.astype(np.float32), queries.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--ef-construction", type=int, default=200)
    ap.add_argument("--m1", type=float, default=1.0,
                    help="Centrality-bias strength (higher = bigger effect)")
    ap.add_argument("--ef-search", type=int, nargs="+",
                    default=[10, 25, 50, 100])
    args = ap.parse_args()

    print("=" * 80)
    print(f"Centrality-biased HNSW vs vanilla — n={args.n}, dim={args.dim}, "
          f"M={args.M}, efC={args.ef_construction}, m1={args.m1}")
    print("=" * 80)

    print("\n[1/4] Generating data + ground truth...")
    X, Q = make_data(args.n, args.dim)
    truth = brute_force_ground_truth(X, Q, k=args.k)

    # ---- hnswlib (reference) ----
    print(f"[2/4] Building hnswlib reference...")
    ref = hnswlib.Index(space="l2", dim=args.dim)
    ref.init_index(max_elements=args.n,
                   ef_construction=args.ef_construction, M=args.M)
    t0 = time.perf_counter()
    ref.add_items(X, np.arange(args.n))
    ref_build = time.perf_counter() - t0
    print(f"      built in {ref_build:.2f}s")

    # ---- vanilla MiniHNSW ----
    print(f"[3/4] Building vanilla MiniHNSW...")
    vanilla = MiniHNSW(dim=args.dim, M=args.M,
                       ef_construction=args.ef_construction, seed=0)
    t0 = time.perf_counter()
    vanilla.add_items(X)
    vanilla_build = time.perf_counter() - t0
    print(f"      built in {vanilla_build:.2f}s")

    # ---- biased CentralityHNSW (two-pass) ----
    print(f"[4/4] Building CentralityHNSW (two-pass)...")
    biased = CentralityHNSW(dim=args.dim, M=args.M,
                            ef_construction=args.ef_construction, seed=0)
    t0 = time.perf_counter()
    biased.build_from_array(X, m1=args.m1)
    biased_build = time.perf_counter() - t0
    print(f"      built in {biased_build:.2f}s (2× — two-pass)")

    # Inspect level distributions
    def level_summary(name, idx):
        if hasattr(idx, "_levels"):
            lv = np.array(idx._levels)
            print(f"      {name:<14} levels: max={lv.max()} "
                  f"mean={lv.mean():.2f} "
                  f"distribution={dict(zip(*np.unique(lv, return_counts=True)))}")
    print()
    print("Level distributions:")
    level_summary("vanilla", vanilla)
    level_summary("centrality", biased)

    # ---- benchmarks ----
    print(f"\nRecall@{args.k} vs ef_search:")
    fmt = "{:<14}  {:>5}  {:>11}  {:>11}"
    print(fmt.format("index", "ef", f"recall@{args.k}", "qps"))
    print("-" * 50)
    rows: list[tuple] = []
    for ef in args.ef_search:
        # hnswlib
        ref.set_ef(ef)
        t0 = time.perf_counter()
        rl, _ = ref.knn_query(Q, k=args.k)
        r_qps = len(Q) / (time.perf_counter() - t0)
        r_rec = recall_at_k(rl, truth, args.k)
        rows.append(("hnswlib", ef, r_rec, r_qps))

        vanilla.set_ef(ef)
        t0 = time.perf_counter()
        vl, _ = vanilla.knn_query(Q, k=args.k)
        v_qps = len(Q) / (time.perf_counter() - t0)
        v_rec = recall_at_k(vl, truth, args.k)
        rows.append(("vanilla", ef, v_rec, v_qps))

        biased.set_ef(ef)
        t0 = time.perf_counter()
        bl, _ = biased.knn_query(Q, k=args.k)
        b_qps = len(Q) / (time.perf_counter() - t0)
        b_rec = recall_at_k(bl, truth, args.k)
        rows.append(("centrality", ef, b_rec, b_qps))

    for name, ef, rec, qps in rows:
        print(fmt.format(name, ef, f"{rec:.4f}", f"{qps:.0f}"))

    # Summarize Δ at each ef
    print(f"\nΔ centrality − vanilla recall@{args.k}:")
    by_ef: dict[int, dict] = {}
    for name, ef, rec, qps in rows:
        by_ef.setdefault(ef, {})[name] = (rec, qps)
    for ef in args.ef_search:
        v_rec = by_ef[ef]["vanilla"][0]
        b_rec = by_ef[ef]["centrality"][0]
        d = b_rec - v_rec
        print(f"  ef={ef:>5}: vanilla={v_rec:.4f}  centrality={b_rec:.4f}  "
              f"Δ={d:+.4f}")


if __name__ == "__main__":
    main()
