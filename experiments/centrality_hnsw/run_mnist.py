"""Same head-to-head but on real MNIST embeddings (cached by hubmesh
benchmarks). MNIST 784-d is a realistic ANN setting — proper distribution,
genuine local structure, no perfect clusters."""
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


def load_mnist(cache_path: Path) -> np.ndarray:
    """Load the MNIST cache produced by hubmesh's run_real.py."""
    cache = cache_path / "mnist_norm.npy"
    if not cache.exists():
        raise FileNotFoundError(
            f"MNIST cache not found at {cache}. Run hubmesh's real-data "
            "benchmark first to generate it.")
    return np.load(cache)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--n-queries", type=int, default=200)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--M", type=int, default=8)
    ap.add_argument("--ef-construction", type=int, default=100)
    ap.add_argument("--m1", type=float, default=3.0)
    ap.add_argument("--centrality", choices=["degree", "centroid"],
                    default="degree")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ef-search", type=int, nargs="+",
                    default=[5, 10, 20, 40, 80])
    args = ap.parse_args()

    print("=" * 80)
    print(f"MNIST 784-d: vanilla vs centrality-biased HNSW")
    print(f"  n={args.n}, M={args.M}, efC={args.ef_construction}, m1={args.m1}")
    print("=" * 80)

    # MNIST was cached by the earlier nnsi-network-optimization runs.
    cache_path = Path("/Users/dattasaikrishnanaidu/Documents/claude/"
                      "nnsi-network-optimization/experiments/vector_db/data")
    X_all = load_mnist(cache_path)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(X_all))
    X = X_all[perm[:args.n]]
    Q = X_all[perm[args.n: args.n + args.n_queries]]
    print(f"  base={X.shape}  queries={Q.shape}")

    print("\nComputing brute-force ground truth...")
    t0 = time.perf_counter()
    truth = brute_force_ground_truth(X, Q, k=args.k, space="l2")
    print(f"  done in {time.perf_counter()-t0:.1f}s")

    # ---- builds ----
    print(f"\nBuilding hnswlib...")
    ref = hnswlib.Index(space="l2", dim=X.shape[1])
    ref.init_index(max_elements=len(X),
                   ef_construction=args.ef_construction, M=args.M)
    t0 = time.perf_counter()
    ref.add_items(X, np.arange(len(X)))
    print(f"  hnswlib built in {time.perf_counter()-t0:.1f}s")

    print(f"Building vanilla MiniHNSW (this is slow — pure Python)...")
    vanilla = MiniHNSW(dim=X.shape[1], M=args.M,
                       ef_construction=args.ef_construction, seed=args.seed)
    t0 = time.perf_counter()
    vanilla.add_items(X)
    print(f"  vanilla built in {time.perf_counter()-t0:.1f}s")

    print(f"Building CentralityHNSW (two-pass)...")
    biased = CentralityHNSW(dim=X.shape[1], M=args.M,
                            ef_construction=args.ef_construction,
                            seed=args.seed)
    t0 = time.perf_counter()
    biased.build_from_array(X, m1=args.m1,
                            centrality_measure=args.centrality)
    print(f"  centrality built in {time.perf_counter()-t0:.1f}s")

    # ---- diagnostic: level distribution ----
    print()
    for name, idx in [("vanilla", vanilla), ("centrality", biased)]:
        lv = np.array(idx._levels)
        counts = dict(zip(*np.unique(lv, return_counts=True)))
        counts_str = ", ".join(f"L{k}: {v}" for k, v in sorted(counts.items()))
        print(f"  {name:<12} max_level={lv.max()} mean={lv.mean():.3f}  "
              f"({counts_str})")

    # ---- bench ----
    print(f"\nRecall@{args.k} vs ef_search:")
    fmt = "{:<14}  {:>5}  {:>11}  {:>11}"
    print(fmt.format("index", "ef", f"recall@{args.k}", "qps"))
    print("-" * 50)
    summary = {ef: {} for ef in args.ef_search}
    for ef in args.ef_search:
        for name, idx in [("hnswlib", ref), ("vanilla", vanilla),
                          ("centrality", biased)]:
            idx.set_ef(ef)
            t0 = time.perf_counter()
            labels, _ = idx.knn_query(Q, k=args.k)
            qps = len(Q) / (time.perf_counter() - t0)
            rec = recall_at_k(labels, truth, args.k)
            summary[ef][name] = (rec, qps)
            print(fmt.format(name, ef, f"{rec:.4f}", f"{qps:.0f}"))

    print(f"\nΔ centrality − vanilla recall@{args.k}:")
    for ef in args.ef_search:
        v_rec = summary[ef]["vanilla"][0]
        b_rec = summary[ef]["centrality"][0]
        print(f"  ef={ef:>4}: vanilla={v_rec:.4f}  "
              f"centrality={b_rec:.4f}  Δ={b_rec - v_rec:+.4f}")


if __name__ == "__main__":
    main()
