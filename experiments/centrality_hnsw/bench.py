"""Recall-vs-QPS evaluation harness for HNSW variants.

Standard ANN-benchmark methodology:
  • Build the index once.
  • For each `ef_search` value, query all test points; measure recall@k and QPS.
  • Plot/log a recall vs throughput curve. Higher curves are better.

This module is data-agnostic — pass any (base, queries, ground_truth) and
any index that exposes `add_items(X)` + `knn_query(Q, k, ef=...)`.
"""
from __future__ import annotations
import time
from dataclasses import dataclass
import numpy as np


@dataclass
class BenchResult:
    name: str
    ef_search: int
    recall_at_k: float
    qps: float
    build_time_s: float
    avg_distances_per_query: float | None = None


def brute_force_ground_truth(base: np.ndarray, queries: np.ndarray, k: int,
                             space: str = "l2") -> np.ndarray:
    """Compute exact k-NN ground truth. Slow but only run once per dataset."""
    if space == "l2":
        # ||q - x||² = ||q||² + ||x||² - 2 q·x → drop constants for ranking
        # but keep the full distance for clarity.
        d = np.linalg.norm(base[None] - queries[:, None], axis=2)
    elif space == "ip":
        d = -(queries @ base.T)              # negate so argsort gives top-IP
    elif space == "cosine":
        bn = base / np.linalg.norm(base, axis=1, keepdims=True).clip(1e-12)
        qn = queries / np.linalg.norm(queries, axis=1, keepdims=True).clip(1e-12)
        d = -(qn @ bn.T)
    else:
        raise ValueError(f"unsupported space: {space}")
    return np.argsort(d, axis=1)[:, :k]


def recall_at_k(retrieved: np.ndarray, truth: np.ndarray, k: int) -> float:
    """Mean recall@k over a batch of queries.
    retrieved, truth: (n_queries, k_or_more) int arrays of vector ids.
    """
    n = retrieved.shape[0]
    hits = 0
    for i in range(n):
        ret = set(int(x) for x in retrieved[i, :k])
        tru = set(int(x) for x in truth[i, :k])
        hits += len(ret & tru)
    return hits / (n * k)


def bench_one(
    name: str,
    index,
    queries: np.ndarray,
    truth: np.ndarray,
    k: int,
    ef_search_values: list[int],
    build_time_s: float,
) -> list[BenchResult]:
    """Run the queries at each ef_search; collect recall+QPS."""
    out = []
    for ef in ef_search_values:
        index.set_ef(ef) if hasattr(index, "set_ef") else None
        t0 = time.perf_counter()
        if hasattr(index, "knn_query"):
            labels, _ = index.knn_query(queries, k=k)
        else:
            labels = index.search(queries, k=k, ef=ef)
        elapsed = time.perf_counter() - t0
        r = recall_at_k(labels, truth, k)
        qps = len(queries) / elapsed
        out.append(BenchResult(
            name=name, ef_search=ef, recall_at_k=r, qps=qps,
            build_time_s=build_time_s,
        ))
    return out


def print_results(results: list[BenchResult], k: int) -> None:
    """Pretty-print a recall/QPS table grouped by index name."""
    by_name: dict[str, list[BenchResult]] = {}
    for r in results:
        by_name.setdefault(r.name, []).append(r)
    fmt = "{:<28} {:>5} {:>11} {:>11} {:>11}"
    print(fmt.format("index", "ef", f"recall@{k}", "qps", "build_s"))
    print("-" * 70)
    for name, rs in by_name.items():
        for r in rs:
            print(fmt.format(name, r.ef_search,
                             f"{r.recall_at_k:.4f}",
                             f"{r.qps:.0f}",
                             f"{r.build_time_s:.2f}"))
        print()
