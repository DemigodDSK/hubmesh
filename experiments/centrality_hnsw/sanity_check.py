"""Sanity-check: does our mini_hnsw match hnswlib's recall on a small dataset?

If our impl is correct, both should produce similar recall vs ef curves on
the same data (mini will be much slower because pure Python).
"""
import sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import hnswlib

from mini_hnsw import MiniHNSW
from bench import brute_force_ground_truth, recall_at_k


def make_data(n: int, dim: int, seed: int = 0):
    """Clustered Gaussian — realistic but reproducible."""
    rng = np.random.default_rng(seed)
    n_clusters = max(5, n // 100)
    centroids = rng.normal(0, 5.0, size=(n_clusters, dim))
    labels = rng.integers(0, n_clusters, size=n)
    X = centroids[labels] + rng.normal(0, 1.0, size=(n, dim))
    queries = X[rng.choice(n, size=min(200, n // 10), replace=False)]
    queries = queries + rng.normal(0, 0.3, size=queries.shape)
    return X.astype(np.float32), queries.astype(np.float32)


def main():
    n, dim, k = 2000, 64, 10
    print(f"Dataset: clustered Gaussian n={n} dim={dim}")

    X, Q = make_data(n, dim)
    print(f"Computing ground truth (brute force, n={n} × q={len(Q)})...")
    t0 = time.perf_counter()
    truth = brute_force_ground_truth(X, Q, k=k, space="l2")
    print(f"  done in {time.perf_counter()-t0:.2f}s")

    M, efC = 16, 200

    # ---- hnswlib reference ----
    print(f"\nBuilding hnswlib (M={M}, ef_construction={efC})...")
    ref = hnswlib.Index(space="l2", dim=dim)
    ref.init_index(max_elements=n, ef_construction=efC, M=M)
    t0 = time.perf_counter()
    ref.add_items(X, np.arange(n))
    ref_build = time.perf_counter() - t0
    print(f"  built in {ref_build:.2f}s")

    # ---- mini HNSW ----
    print(f"\nBuilding mini_hnsw (M={M}, ef_construction={efC})...")
    mini = MiniHNSW(dim=dim, space="l2", M=M, ef_construction=efC, seed=0)
    t0 = time.perf_counter()
    mini.add_items(X)
    mini_build = time.perf_counter() - t0
    print(f"  built in {mini_build:.2f}s (slower, that's expected — pure Python)")

    # ---- compare recall at several ef ----
    print(f"\nRecall@{k} vs ef_search:")
    print(f"  {'ef':>5}  {'hnswlib':>10}  {'mini':>10}  {'Δ':>8}")
    for ef in [10, 25, 50, 100, 200]:
        ref.set_ef(ef)
        rlabels, _ = ref.knn_query(Q, k=k)
        r_recall = recall_at_k(rlabels, truth, k)

        mini.set_ef(ef)
        mlabels, _ = mini.knn_query(Q, k=k)
        m_recall = recall_at_k(mlabels, truth, k)

        delta = m_recall - r_recall
        print(f"  {ef:>5}  {r_recall:>10.4f}  {m_recall:>10.4f}  {delta:>+8.4f}")

    print("\nDone. If mini's recall is within ~2 pts of hnswlib, our impl is sound.")


if __name__ == "__main__":
    main()
