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
    ap.add_argument("--title-body", action="store_true",
                    help="index '{title}\\n\\n{text}' (LightRAG-comparison "
                         "representation; changes absolute numbers)")
    ap.add_argument("--paragraph-identity", choices=["title", "title+text"],
                    default="title",
                    help="'title' = published protocol (passages sharing a "
                         "title collapse); 'title+text' = exact paragraph ids")
    ap.add_argument("--out", default=None,
                    help="write results JSON (manifest + per-query recalls)")
    ap.add_argument("--embed-batch-size", type=int, default=64,
                    help="sentence-transformers batch size (lower on small-RAM machines)")
    ap.add_argument("--embed-device", default=None,
                    help="torch device for embedding, e.g. cpu (default: auto)")
    args = ap.parse_args()
    from structural_only import structural_only_retrieve
    from manifest import build_manifest
    import json
    from pathlib import Path

    print("=" * 84)
    print(f"MuSiQue paragraph-retrieval benchmark — n={args.n} questions")
    print("=" * 84)

    print("[1/4] Loading MuSiQue dev...")
    identity_audit: dict = {}
    examples, pool = load_musique(n_questions=args.n, seed=args.seed,
                                  paragraph_identity=args.paragraph_identity,
                                  audit=identity_audit)
    print(f"      paragraph identity audit: {identity_audit}")
    pool_titles = list(pool.keys())
    pool_texts = [f"{t.split('::')[0]}\n\n{pool[t]}" if args.title_body
                  else pool[t] for t in pool_titles]

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
    para_vecs = embed_texts(pool_texts, batch_size=args.embed_batch_size, model_name=args.model,
                             device=args.embed_device)
    query_vecs = embed_texts([ex.question for ex in examples], batch_size=args.embed_batch_size,
                             model_name=args.model, device=args.embed_device)

    print("[3/4] Building store + Planner...")
    docs = [
        Document(id=t, text=pool_texts[i], vector=para_vecs[i],
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
        strategies.append("structural_only")   # same-graph scoring ablation
        strategies.append("hippo_style")
    per_query: list[dict] = []
    # Track results per strategy AND per hop count, so we can break out
    # 2-hop vs 3-/4-hop performance (the actual point of MuSiQue).
    results_overall = {s: {k: [] for k in ks} for s in strategies}
    results_by_hop  = {h: {s: {k: [] for k in ks} for s in strategies}
                       for h in [2, 3, 4]}
    timings = {s: 0.0 for s in strategies}
    struct_stats: dict = {}      # seed counts / fallback / seedless per query
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
            struct = structural_only_retrieve(
                kg, planner._ppr_solver, nlp_q, ex.question, qvec, store,
                max(ks), alpha=planner.config.ppr_alpha,
                budget_tokens=10_000,                       # same as hubmesh_retrieve
                redundancy_lambda=planner.config.redundancy_lambda,
                stats=struct_stats)
            timings["structural_only"] += time.perf_counter() - t0
            retrieved["structural_only"] = struct
            t0 = time.perf_counter()
            hippo = hippo_style_retrieve(kg, nlp_q, ex.question, qvec, store,
                                          max(ks))
            timings["hippo_style"] += time.perf_counter() - t0
            retrieved["hippo_style"] = hippo

        rec = {"qid": ex.qid, "n_hops": ex.n_hops, "n_gold": len(gold)}
        if has_kg:
            rec["structural_only_seeds"] = struct_stats["n_seeds"][-1]
        for s in strategies:
            rec[s] = {}
            for k in ks:
                r = recall_at_k(retrieved[s], gold, k)
                rec[s][k] = r
                results_overall[s][k].append(r)
                if ex.n_hops in results_by_hop:
                    results_by_hop[ex.n_hops][s][k].append(r)
        per_query.append(rec)

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

    if args.out:
        from dataclasses import asdict
        out = {
            "manifest": build_manifest(
                harness=__file__,
                embed_device=args.embed_device or "auto",
                embed_batch_size=args.embed_batch_size,
                dataset="musique", split="validation",
                n_questions=args.n, seed=args.seed,
                n_pooled_paragraphs=len(pool_titles),
                paragraph_identity=args.paragraph_identity,
                identity_audit=identity_audit,
                representation="title+body" if args.title_body else "body",
                embed_model=args.model, kg_mode=has_kg,
                planner_config=asdict(planner.config),
                strategies=strategies),
            "summary": {s: {f"recall@{k}": round(float(np.mean(results_overall[s][k])), 4)
                            for k in ks} for s in strategies},
            "structural_only_stats": {k: v for k, v in struct_stats.items()
                                      if k != "n_seeds"},
            "by_hop": {h: {s: {f"recall@{k}": round(float(np.mean(results_by_hop[h][s][k])), 4)
                               for k in ks} for s in strategies}
                       for h in [2, 3, 4] if results_by_hop[h]["naive_topk"][2]},
            "total_time_s": {s: round(timings[s], 1) for s in strategies},
            "per_query": per_query,
        }
        Path(args.out).write_text(json.dumps(out, indent=1))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
