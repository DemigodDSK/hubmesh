"""MuSiQue paragraph-retrieval benchmark.

Runs the same 3-way comparison (naive cosine / hubmesh-multi-component /
HippoRAG-style PPR-only) on MuSiQue. MuSiQue is harder than HotpotQA
because it includes 3- and 4-hop questions and 20 distractors per
question (vs HotpotQA's 10).

If hubmesh's win over naive holds (or grows) on MuSiQue, the
multi-hop hypothesis is robust.
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path
import sys
import numpy as np
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

from hubmesh import Planner, Document
from hubmesh.adapters import InMemoryStore

from musique_loader import load_musique, retrievable_gold
from hippo_style import hippo_style_retrieve
from run_hotpotqa import (
    embed_texts, naive_topk_retrieve, hubmesh_retrieve, recall_at_k,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="number of questions")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--knn", type=int, default=8)
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    ap.add_argument("--kg", action="store_true")
    args = ap.parse_args()

    print("=" * 84)
    print(f"MuSiQue paragraph-retrieval benchmark — n={args.n} questions")
    print("=" * 84)

    print("[1/4] Loading MuSiQue dev...")
    examples, pool = load_musique(n_questions=args.n, seed=args.seed)
    pool_titles = list(pool.keys())
    pool_texts = [pool[t] for t in pool_titles]

    hops_dist: dict[int, int] = {}
    for ex in examples:
        hops_dist[ex.n_hops] = hops_dist.get(ex.n_hops, 0) + 1
    print(f"      pooled corpus: {len(pool_titles)} unique paragraphs")
    print(f"      hop distribution: {sorted(hops_dist.items())}")
    n_with_gold = sum(1 for ex in examples
                      if retrievable_gold(ex, set(pool_titles)))
    print(f"      questions with retrievable gold: "
          f"{n_with_gold}/{len(examples)}")

    print(f"[2/4] Embedding...")
    para_vecs = embed_texts(pool_texts, model_name=args.model)
    query_vecs = embed_texts([ex.question for ex in examples],
                             model_name=args.model)

    print("[3/4] Building store + Planner...")
    docs = [
        Document(id=t, text=pool[t], vector=para_vecs[i],
                 metadata={"title": t})
        for i, t in enumerate(pool_titles)
    ]
    store = InMemoryStore(docs, k=args.knn)

    kg = None
    if args.kg:
        from hubmesh.kg import build_entity_kg
        import spacy
        print("      building entity-linked KG...")
        nlp = spacy.load("en_core_web_sm")
        t0 = time.perf_counter()
        kg = build_entity_kg(docs, nlp=nlp)
        print(f"        KG built in {time.perf_counter()-t0:.1f}s "
              f"({kg.graph.number_of_nodes()} nodes, "
              f"{kg.graph.number_of_edges()} edges)")
        planner = Planner(store=store, kg=kg, nlp=nlp)
    else:
        planner = Planner(store=store)

    print("[4/4] Evaluating...")
    ks = [2, 5, 10]
    has_kg = kg is not None
    strategies = ["naive_topk", "hubmesh"]
    if has_kg:
        strategies.append("hippo_style")
    # Track results per strategy AND per hop count, so we can break out
    # 2-hop vs 3-/4-hop performance (the actual point of MuSiQue).
    results_overall = {s: {k: [] for k in ks} for s in strategies}
    results_by_hop  = {h: {s: {k: [] for k in ks} for s in strategies}
                       for h in [2, 3, 4]}
    timings = {s: 0.0 for s in strategies}
    nlp_q = planner._nlp if has_kg else None

    for ex_idx, ex in enumerate(tqdm(examples, desc="queries")):
        gold = retrievable_gold(ex, set(pool_titles))
        if not gold:
            continue
        qvec = query_vecs[ex_idx]

        t0 = time.perf_counter()
        naive = naive_topk_retrieve(store, qvec, max(ks))
        timings["naive_topk"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        hub = hubmesh_retrieve(planner, ex.question, qvec, max(ks))
        timings["hubmesh"] += time.perf_counter() - t0

        retrieved = {"naive_topk": naive, "hubmesh": hub}

        if has_kg:
            if nlp_q is None:
                import spacy
                nlp_q = spacy.load("en_core_web_sm"); planner._nlp = nlp_q
            t0 = time.perf_counter()
            hippo = hippo_style_retrieve(kg, nlp_q, ex.question, qvec, store,
                                          max(ks))
            timings["hippo_style"] += time.perf_counter() - t0
            retrieved["hippo_style"] = hippo

        for s in strategies:
            for k in ks:
                r = recall_at_k(retrieved[s], gold, k)
                results_overall[s][k].append(r)
                if ex.n_hops in results_by_hop:
                    results_by_hop[ex.n_hops][s][k].append(r)

    print()
    print("=" * 84)
    print("OVERALL")
    print("=" * 84)
    fmt = "{:<20} {:>10} {:>10} {:>10} {:>14}"
    print(fmt.format("strategy", "recall@2", "recall@5", "recall@10",
                     "total_time_s"))
    print("-" * 84)
    for name in strategies:
        row = [name]
        for k in ks:
            arr = np.array(results_overall[name][k], dtype=float)
            row.append(f"{arr.mean():.3f}")
        row.append(f"{timings[name]:.1f}")
        print(fmt.format(*row))

    for hop in [2, 3, 4]:
        n_in_bucket = len(results_by_hop[hop]["naive_topk"][2])
        if n_in_bucket == 0:
            continue
        print()
        print(f"=== {hop}-HOP only (n={n_in_bucket}) ===")
        print(fmt.format("strategy", "recall@2", "recall@5", "recall@10", ""))
        print("-" * 84)
        for name in strategies:
            row = [name]
            for k in ks:
                arr = np.array(results_by_hop[hop][name][k], dtype=float)
                row.append(f"{arr.mean():.3f}")
            row.append("")
            print(fmt.format(*row))


if __name__ == "__main__":
    main()
