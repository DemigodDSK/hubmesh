"""Coherence-slot ablation: is the v0.4.0 convergence gain multi-anchor
AND-ness, or the *shape* of the signal placed in the coherence slot?

Every arm shares byte-identical corpus, embeddings, KG, query seeds, PPR
solves, weights (3:1:1, integration="sum"), top_k, budget and packer.
The ONLY thing that changes is the vector handed to `composite_score`
as `coherence` (which min-max normalises it before the weighted sum):

    arm  coherence value                          pooling  shape   solves
    A    exp(mean_j log(eps+p_j))   [shipped]     AND      raw     1+m
    G    mean_j log(eps+p_j)                      AND      log     1+m
    B    log(eps + pooled_ppr)                    OR       log     1
    D    log(eps + mean_j p_j)  (first 4 seeds)   OR       log     1+m
    E    pooled_ppr   (== weights 3:2:0)          OR       raw     1
    C    off (coherence weight 0)                 -        -       1

p_j is the single-seed PPR from anchor j (first 4 seeds, as shipped);
pooled_ppr is the PPR with uniform restart over ALL seeds — the solve
the planner performs anyway. PPR is linear in the restart vector, so on
queries with <=4 seeds D == B exactly; D is reported to make that
visible, not as an independent test.

Which contrast isolates what:
    A vs G   shape, AND-pooling held fixed
    E vs B   shape, OR-pooling held fixed
    G vs B   AND-ness, shape held graded (log)      <- the real test
    A vs E   AND-ness, shape held raw
    A vs C   the shipped gain

Gating: A, G, D exist only when >=2 seeds (the shipped gate; single-seed
queries are neutral == C). B and E use the pooled solve and are active on
every query with seeds. Both unconditional and "active-only" subsets
are reported, with paired bootstrap CIs.

Run:  python benchmarks/run_ablation_coherence.py --dataset hotpotqa --n 500
      python benchmarks/run_ablation_coherence.py --dataset musique  --n 300
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

from hubmesh import Document                                  # noqa: E402
from hubmesh.types import ScoredDocument                      # noqa: E402
from hubmesh.scoring import ScoringWeights, composite_score   # noqa: E402
from hubmesh.packing import pack, estimate_tokens             # noqa: E402
from hubmesh.adapters import InMemoryStore                    # noqa: E402
from hubmesh.kg import build_entity_kg, extract_query_entities  # noqa: E402
from hubmesh.ppr import PPRSolver                             # noqa: E402
from manifest import build_manifest                           # noqa: E402

ARMS = ["A", "G", "B", "D", "E", "C"]
ARM_DESC = {
    "A": "exp(mean_j log(eps+p_j)) over seeds[:4]  [shipped geomean; AND/raw; 1+m solves]",
    "G": "mean_j log(eps+p_j) over seeds[:4]       [AND/log; 1+m solves]",
    "B": "log(eps + pooled_ppr)                    [OR/log; 1 solve]",
    "D": "log(eps + mean_j p_j) over seeds[:4]     [OR/log; == B when <=4 seeds]",
    "E": "pooled_ppr raw                           [OR/raw; == weights 3:2:0]",
    "C": "coherence off (weight 0)                 [1 solve]",
}
KS = [2, 5, 10]
EPS = 1e-12
TOP_K = 10
BUDGET = 10_000
ALPHA = 0.15


def embed_texts(texts, model_name="all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    return model.encode(texts, batch_size=64, show_progress_bar=True,
                        normalize_embeddings=True,
                        convert_to_numpy=True).astype(np.float32)


def recall_at_k(retrieved, gold, k):
    top = set(retrieved[:k])
    return sum(1 for g in gold if g in top) / len(gold)


def minmax(v: np.ndarray) -> np.ndarray:
    lo, hi = float(v.min()), float(v.max())
    if hi - lo < 1e-9:
        return np.full_like(v, 0.5, dtype=float)
    return (v - lo) / (hi - lo)


def paired_boot(a, b, n_boot, rng):
    """mean(a-b) in points with a 95% percentile-bootstrap CI."""
    d = (np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) * 100.0
    n = len(d)
    if n == 0:
        return None
    idx = rng.integers(0, n, size=(n_boot, n))
    means = d[idx].mean(axis=1)
    return {"mean": round(float(d.mean()), 2),
            "lo": round(float(np.percentile(means, 2.5)), 2),
            "hi": round(float(np.percentile(means, 97.5)), 2),
            "n": int(n)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["hotpotqa", "musique"], required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--boot", type=int, default=10_000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print(f"[1/3] Loading {args.dataset} n={args.n} seed={args.seed}...")
    if args.dataset == "hotpotqa":
        from hotpotqa_loader import load_hotpotqa, retrievable_gold
        examples, pool = load_hotpotqa(n_questions=args.n, seed=args.seed)
        hop_of = {}
    else:
        from musique_loader import load_musique, retrievable_gold
        examples, pool = load_musique(n_questions=args.n, seed=args.seed)
        hop_of = {ex.qid: ex.n_hops for ex in examples}
    pool_titles = list(pool.keys())
    pool_set = set(pool_titles)
    print(f"      {len(pool_titles)} paragraphs, {len(examples)} questions")

    print("[2/3] Embedding + KG (shared across all arms)...")
    para_vecs = embed_texts([pool[t] for t in pool_titles])
    query_vecs = embed_texts([ex.question for ex in examples])
    docs = [Document(id=t, text=pool[t], vector=para_vecs[i],
                     metadata={"title": t})
            for i, t in enumerate(pool_titles)]
    store = InMemoryStore(docs, k=8)
    import spacy
    nlp = spacy.load("en_core_web_sm")
    t0 = time.perf_counter()
    kg = build_entity_kg(docs, nlp=nlp)
    kg_s = time.perf_counter() - t0
    print(f"      KG {kg.graph.number_of_nodes()} nodes / "
          f"{kg.graph.number_of_edges()} edges in {kg_s:.0f}s")
    solver = PPRSolver(kg.graph, weight_attr="weight", hub_discount=0.0)

    # Unit-normalised doc matrix, once (mirrors Planner._kg_doc_matrix).
    doc_ids = sorted(n[4:] for n in kg.graph.nodes if n.startswith("doc:"))
    mat = np.stack([store.vector_of(d) for d in doc_ids]).astype(np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat = mat / norms
    doc_keys = [f"doc:{d}" for d in doc_ids]

    W_ON = ScoringWeights(relevance=3.0, structural=1.0, coherence=1.0)
    W_OFF = ScoringWeights(relevance=3.0, structural=1.0, coherence=0.0)

    print("[3/3] Evaluating arms (one solve set per query, shared)...")
    per_query = []
    shape_stats = {a: [] for a in ARMS if a != "C"}
    overlap_vs_A = {a: [] for a in ARMS if a != "A"}
    t_solve_all, t_multi_all = [], []

    for ex_idx, ex in enumerate(tqdm(examples, desc="queries")):
        gold = retrievable_gold(ex, pool_set)
        if not gold:
            continue
        qvec = query_vecs[ex_idx]

        # --- seeds: mirror Planner._retrieve_kg exactly ---
        q_mentions = extract_query_entities(ex.question, nlp=nlp)
        seeds = list(dict.fromkeys(kg.query_entity_nodes(q_mentions)))
        fallback = False
        if not seeds:
            fallback = True
            seed_docs = [d for d, _ in store.search(qvec, top_k=3)]
            for d in seed_docs:
                ents = kg.doc_to_entities.get(d, set())
                seeds.extend([e for e in sorted(ents) if e in kg.graph])
            seeds = list(dict.fromkeys(seeds))[:8]
        active = len(seeds) >= 2

        # --- solves (shared by every arm) ---
        t0 = time.perf_counter()
        pooled = solver.solve(seeds, alpha=ALPHA) if seeds else {}
        t_solve = time.perf_counter() - t0
        t_multi = 0.0
        per_seed = []
        if active:
            t0 = time.perf_counter()
            per_seed = solver.solve_multi([[s] for s in seeds[:4]], alpha=ALPHA)
            t_multi = time.perf_counter() - t0
        t_solve_all.append(t_solve)
        t_multi_all.append(t_multi)

        # --- shared R and S ---
        qn = qvec / max(float(np.linalg.norm(qvec)), 1e-12)
        sims = (mat @ qn).astype(np.float64)
        rel = {doc_ids[i]: float(sims[i]) for i in range(len(doc_ids))}
        pooled_arr = np.fromiter((pooled.get(k, 0.0) for k in doc_keys),
                                 dtype=float, count=len(doc_keys))
        struct = {doc_ids[i]: float(pooled_arr[i]) for i in range(len(doc_ids))}

        # --- coherence vectors per arm ---
        coh_vec: dict[str, np.ndarray | None] = {}
        if active:
            P = np.stack([np.fromiter((ps.get(k, 0.0) for k in doc_keys),
                                      dtype=float, count=len(doc_keys))
                          for ps in per_seed])              # m x N
            logP = np.log(EPS + P)
            coh_vec["A"] = np.exp(logP.mean(axis=0))
            coh_vec["G"] = logP.mean(axis=0)
            coh_vec["D"] = np.log(EPS + P.mean(axis=0))
        else:
            coh_vec["A"] = coh_vec["G"] = coh_vec["D"] = None   # neutral
        if seeds:
            coh_vec["B"] = np.log(EPS + pooled_arr)
            coh_vec["E"] = pooled_arr.copy()
        else:
            coh_vec["B"] = coh_vec["E"] = None
        coh_vec["C"] = None

        # --- naive cosine ---
        naive_ids = [doc_ids[i] for i in np.argsort(-sims, kind="stable")[:TOP_K]]

        rec = {"qid": ex.qid, "n_seeds": len(seeds), "fallback": fallback,
               "active": active, "hop": hop_of.get(ex.qid),
               "t_solve": round(t_solve, 4), "t_multi": round(t_multi, 4),
               "naive": {f"@{k}": recall_at_k(naive_ids, gold, k) for k in KS}}
        top10 = {}
        for arm in ARMS:
            v = coh_vec[arm]
            if v is None:
                coh = {d: 1.0 for d in doc_ids}
                w = W_OFF if arm == "C" else W_ON   # neutral const -> no effect
            else:
                coh = {doc_ids[i]: float(v[i]) for i in range(len(doc_ids))}
                w = W_ON
                mm = minmax(v)
                shape_stats[arm].append(float((mm > 0.1).mean()))
            composite = composite_score(relevance=rel, structural=struct,
                                        coherence=coh, weights=w,
                                        integration="sum")
            ordered = sorted(composite.items(), key=lambda kv: (-kv[1], kv[0]))
            cand = ordered[:TOP_K * 5]
            fetched = {d.id: d for d in store.get_many([d for d, _ in cand])}
            scored = [ScoredDocument(doc=fetched[d], similarity=rel[d],
                                     ppr_score=struct[d],
                                     composite_score=float(s), rank=r)
                      for r, (d, s) in enumerate(cand) if d in fetched]
            _, picked = pack(scored, budget_tokens=BUDGET,
                             redundancy_lambda=0.3, vec_of=store.vector_of,
                             max_docs=TOP_K, count_tokens=estimate_tokens)
            got = [p.doc.id for p in picked[:TOP_K]]
            top10[arm] = set(got)
            rec[arm] = {f"@{k}": recall_at_k(got, gold, k) for k in KS}
        for arm in overlap_vs_A:
            overlap_vs_A[arm].append(len(top10[arm] & top10["A"]) / TOP_K)
        per_query.append(rec)

    # ------------------------------------------------------------------
    # aggregate
    n = len(per_query)
    act = [r for r in per_query if r["active"]]
    rng = np.random.default_rng(args.seed)

    def agg(rows, arm):
        return {f"@{k}": round(float(np.mean([r[arm][f"@{k}"] for r in rows])), 4)
                for k in KS}

    def contrast(rows, x, y):
        return {f"@{k}": paired_boot([r[x][f"@{k}"] for r in rows],
                                     [r[y][f"@{k}"] for r in rows],
                                     args.boot, rng) for k in KS}

    contrasts = [("B", "A"), ("G", "A"), ("D", "A"), ("E", "A"),
                 ("G", "B"), ("A", "E"),
                 ("A", "C"), ("B", "C"), ("G", "C"), ("E", "C"),
                 ("naive", "A"), ("naive", "C")]
    results = {
        "manifest": build_manifest(harness=__file__,
                                   dataset=args.dataset, n=args.n,
                                   seed=args.seed, embed_model="all-MiniLM-L6-v2",
                                   embed_device="auto", embed_batch_size=64,
                                   alpha=ALPHA, weights="3:1:1 sum",
                                   top_k=TOP_K, budget_tokens=BUDGET, eps=EPS),
        "arms": ARM_DESC,
        "corpus": {"paragraphs": len(pool_titles),
                   "kg_nodes": kg.graph.number_of_nodes(),
                   "kg_edges": kg.graph.number_of_edges(),
                   "kg_build_s": round(kg_s, 1)},
        "n": n, "n_active": len(act),
        "mean_seeds": round(float(np.mean([r["n_seeds"] for r in per_query])), 2),
        "n_fallback": sum(1 for r in per_query if r["fallback"]),
        "recall_all": {**{a: agg(per_query, a) for a in ARMS},
                       "naive": agg(per_query, "naive")},
        "recall_active": {**{a: agg(act, a) for a in ARMS},
                          "naive": agg(act, "naive")},
        "contrasts_all": {f"{x}-{y}": contrast(per_query, x, y)
                          for x, y in contrasts},
        "contrasts_active": {f"{x}-{y}": contrast(act, x, y)
                             for x, y in contrasts},
        "shape_frac_gt_0.1_after_minmax": {
            a: {"mean": round(float(np.mean(v)), 4),
                "median": round(float(np.median(v)), 4)}
            for a, v in shape_stats.items() if v},
        "top10_overlap_vs_A_active": {
            a: round(float(np.mean([o for o, r in zip(v, per_query) if r["active"]])), 3)
            for a, v in overlap_vs_A.items()},
        "timing_s_per_query": {
            "pooled_solve_mean": round(float(np.mean(t_solve_all)), 4),
            "solve_multi_mean_active": round(float(np.mean(
                [t for t, r in zip(t_multi_all, per_query) if r["active"]])), 4)
            if act else None},
        "per_query": per_query,
    }
    if hop_of:
        by_hop = {}
        for h in sorted({r["hop"] for r in per_query if r["hop"]}):
            rows = [r for r in per_query if r["hop"] == h]
            by_hop[f"{h}-hop (n={len(rows)})"] = {
                **{a: agg(rows, a)["@10"] for a in ARMS},
                "naive": agg(rows, "naive")["@10"]}
        results["recall10_by_hop"] = by_hop

    out = Path(args.out) if args.out else (
        HERE / "results" / f"ablation_coherence_{args.dataset}_n{args.n}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))

    print(f"\n=== {args.dataset} n={n} (active={len(act)}, mean seeds "
          f"{results['mean_seeds']}, fallback {results['n_fallback']}) ===")
    print(f"{'arm':<6}{'@2':>8}{'@5':>8}{'@10':>8}   | active-only @10")
    for a in ARMS + ["naive"]:
        r = results["recall_all"][a]
        ra = results["recall_active"][a]
        print(f"{a:<6}{r['@2']:>8.3f}{r['@5']:>8.3f}{r['@10']:>8.3f}   | {ra['@10']:.3f}")
    print("\npaired bootstrap (pts, 95% CI) — unconditional | active-only:")
    for key in results["contrasts_all"]:
        u = results["contrasts_all"][key]["@10"]
        a = results["contrasts_active"][key]["@10"]
        print(f"  {key:<8} @10  {u['mean']:+.2f} [{u['lo']:+.2f},{u['hi']:+.2f}]"
              f"  |  {a['mean']:+.2f} [{a['lo']:+.2f},{a['hi']:+.2f}]")
        u2 = results["contrasts_all"][key]["@2"]
        print(f"  {'':<8} @2   {u2['mean']:+.2f} [{u2['lo']:+.2f},{u2['hi']:+.2f}]")
    print("\nshape (frac docs >0.1 after min-max):",
          results["shape_frac_gt_0.1_after_minmax"])
    print("top-10 overlap vs A (active):", results["top10_overlap_vs_A_active"])
    print("timing:", results["timing_s_per_query"])
    if hop_of:
        print("recall@10 by hop:", json.dumps(results["recall10_by_hop"], indent=1))
    print(f"\nwritten -> {out}")


if __name__ == "__main__":
    main()
