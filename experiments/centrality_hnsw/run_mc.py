"""HNSW-MC head-to-head benchmark, multi-seed, multi-config.

The previous experiment (level-assignment bias) failed across seeds. This
one tests the multi-component pattern at the *neighbour selection* step
— where HNSW's quality actually lives.
"""
from __future__ import annotations
import sys, time, argparse
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import hnswlib

from mini_hnsw import MiniHNSW
from hnsw_mc import HNSWMultiComponent
from bench import brute_force_ground_truth, recall_at_k


def load_mnist() -> np.ndarray:
    p = Path("/Users/dattasaikrishnanaidu/Documents/claude/"
             "nnsi-network-optimization/experiments/vector_db/data/mnist_norm.npy")
    if not p.exists():
        raise FileNotFoundError(f"MNIST cache not found at {p}")
    return np.load(p)


def run_one_seed(seed, n, M, ef_construction, weights, ef_search_list,
                 dim_force=None):
    X_all = load_mnist()
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X_all))
    X = X_all[perm[:n]]
    Q = X_all[perm[n: n + 200]]
    if dim_force is not None and dim_force < X.shape[1]:
        # PCA-ish dim reduction not needed; just truncate. (Quick way to
        # change the difficulty without re-embedding.)
        X = X[:, :dim_force]
        Q = Q[:, :dim_force]
    truth = brute_force_ground_truth(X, Q, k=10, space="l2")

    # Vanilla
    v = MiniHNSW(dim=X.shape[1], M=M, ef_construction=ef_construction, seed=seed)
    t0 = time.perf_counter(); v.add_items(X); v_build = time.perf_counter() - t0

    # MC
    mc = HNSWMultiComponent(dim=X.shape[1], M=M,
                            ef_construction=ef_construction, seed=seed)
    mc.set_weights(**weights)
    t0 = time.perf_counter(); mc.add_items(X); mc_build = time.perf_counter() - t0

    out = []
    for ef in ef_search_list:
        v.set_ef(ef); mc.set_ef(ef)
        rl_v, _ = v.knn_query(Q, k=10)
        rl_m, _ = mc.knn_query(Q, k=10)
        r_v = recall_at_k(rl_v, truth, 10)
        r_m = recall_at_k(rl_m, truth, 10)
        out.append((ef, r_v, r_m, r_m - r_v))
    return out, v_build, mc_build


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--M", type=int, default=4)
    ap.add_argument("--ef-construction", type=int, default=100)
    ap.add_argument("--ef-search", type=int, nargs="+", default=[3, 5, 8, 12, 20])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--wP", type=float, default=1.0)
    ap.add_argument("--wD", type=float, default=1.0)
    ap.add_argument("--wH", type=float, default=0.5)
    ap.add_argument("--wG", type=float, default=0.5)
    args = ap.parse_args()

    weights = dict(P=args.wP, D=args.wD, H=args.wH, G=args.wG)
    print("=" * 80)
    print(f"HNSW-MC vs vanilla MiniHNSW — MNIST 784-d")
    print(f"  n={args.n}, M={args.M}, efC={args.ef_construction}, "
          f"weights={weights}")
    print(f"  seeds={args.seeds}, ef_search={args.ef_search}")
    print("=" * 80)

    # Collect deltas per ef across seeds
    deltas: dict[int, list[float]] = {ef: [] for ef in args.ef_search}
    raw_rows = []

    for seed in args.seeds:
        print(f"\n--- seed={seed} ---")
        out, vb, mb = run_one_seed(
            seed, args.n, args.M, args.ef_construction,
            weights, args.ef_search,
        )
        print(f"  build: vanilla={vb:.1f}s  mc={mb:.1f}s  (mc/vanilla={mb/vb:.1f}x)")
        print(f"  {'ef':>5}  {'vanilla':>10}  {'mc':>10}  {'Δ':>9}")
        for ef, rv, rm, d in out:
            print(f"  {ef:>5}  {rv:>10.4f}  {rm:>10.4f}  {d:+>9.4f}")
            deltas[ef].append(d)
            raw_rows.append((seed, ef, rv, rm, d))

    print()
    print("=" * 80)
    print(f"AGGREGATE across {len(args.seeds)} seeds")
    print("=" * 80)
    print(f"  {'ef':>5}  {'mean Δ':>10}  {'std':>10}  {'all seeds':>40}")
    print("  " + "-" * 70)
    for ef in args.ef_search:
        ds = np.array(deltas[ef])
        per_seed = "  ".join(f"{d:+.4f}" for d in ds)
        print(f"  {ef:>5}  {ds.mean():+10.4f}  {ds.std():>10.4f}  {per_seed:>40}")

    # Verdict
    print()
    means = np.array([np.mean(deltas[ef]) for ef in args.ef_search])
    stds  = np.array([np.std(deltas[ef]) for ef in args.ef_search])
    if (means > 0).sum() >= len(means) * 0.7 and means.mean() > 0.005:
        print(f"  VERDICT: mc beats vanilla on most ef settings, "
              f"mean Δ={means.mean()*100:+.2f} pts.")
    elif means.mean() > stds.mean():
        print(f"  VERDICT: mc beats vanilla on average (mean Δ={means.mean()*100:+.2f} pts) "
              f"with moderate variance ({stds.mean()*100:.2f} pts).")
    else:
        print(f"  VERDICT: not a clear win. Mean Δ={means.mean()*100:+.2f} pts, "
              f"std={stds.mean()*100:.2f} pts. "
              f"Variance is comparable to the effect — coin-flip.")


if __name__ == "__main__":
    main()
