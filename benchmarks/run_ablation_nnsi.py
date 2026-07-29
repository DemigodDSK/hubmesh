"""NNSI-KG formula ablation: baseline vs hub-discount vs convergence.

Builds the corpus, embeddings, and entity KG ONCE, then evaluates four
scoring configurations over identical ground:

    baseline    — v0.3.2 shipped formula (3·R + 1·S)
    hub         — + hub-discounted PPR transition matrix (γ=1)
    conv        — + per-seed convergence component (3·R + 1·S + 1·C)
    hub+conv    — both

Retrieval recall@{2,5,10}; per-hop breakdown on MuSiQue. New components
ship only if this table says they earn it.

Run:  python benchmarks/run_ablation_nnsi.py --dataset hotpotqa --n 500
      python benchmarks/run_ablation_nnsi.py --dataset musique  --n 300
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

from hubmesh import Planner, Document
from hubmesh.planner import PlannerConfig
from hubmesh.scoring import ScoringWeights
from hubmesh.adapters import InMemoryStore
from hubmesh.kg import build_entity_kg


ARMS = {
    "baseline": dict(hub_discount=0.0, use_convergence=False),
    "hub": dict(hub_discount=1.0, use_convergence=False),
    "conv": dict(hub_discount=0.0, use_convergence=True),
    "hub+conv": dict(hub_discount=1.0, use_convergence=True),
}


def embed_texts(texts, model_name="all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    return model.encode(texts, batch_size=64, show_progress_bar=True,
                        normalize_embeddings=True,
                        convert_to_numpy=True).astype(np.float32)


def recall_at_k(retrieved, gold, k):
    top = set(retrieved[:k])
    return sum(1 for g in gold if g in top) / len(gold)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["hotpotqa", "musique"],
                    required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"[1/3] Loading {args.dataset} n={args.n} seed={args.seed}...")
    if args.dataset == "hotpotqa":
        from hotpotqa_loader import load_hotpotqa, retrievable_gold
        examples, pool = load_hotpotqa(n_questions=args.n, seed=args.seed)
        hop_of = {ex.qid: None for ex in examples}
    else:
        from musique_loader import load_musique, retrievable_gold
        examples, pool = load_musique(n_questions=args.n, seed=args.seed)
        hop_of = {ex.qid: getattr(ex, "n_hops", None) for ex in examples}
    pool_titles = list(pool.keys())
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
    print(f"      KG {kg.graph.number_of_nodes()} nodes / "
          f"{kg.graph.number_of_edges()} edges in "
          f"{time.perf_counter()-t0:.0f}s")

    print("[3/3] Evaluating arms...")
    ks = [2, 5, 10]
    results = {}
    for arm, flags in ARMS.items():
        cfg = PlannerConfig(
            weights=ScoringWeights(
                relevance=3.0, structural=1.0,
                coherence=1.0 if flags["use_convergence"] else 0.0),
            **flags)
        t0 = time.perf_counter()
        planner = Planner(store=store, kg=kg, nlp=nlp, config=cfg)
        build_s = time.perf_counter() - t0
        per_q = {k: [] for k in ks}
        per_hop = {}
        t0 = time.perf_counter()
        for ex_idx, ex in enumerate(tqdm(examples, desc=arm)):
            gold = retrievable_gold(ex, set(pool_titles))
            if not gold:
                continue
            res = planner.retrieve(query=ex.question,
                                   query_vec=query_vecs[ex_idx],
                                   top_k=10, budget_tokens=10_000)
            got = [s.doc.id for s in res.sources]
            for k in ks:
                per_q[k].append(recall_at_k(got, gold, k))
            h = hop_of.get(ex.qid)
            if h:
                per_hop.setdefault(h, []).append(
                    recall_at_k(got, gold, 10))
        results[arm] = {
            "recall": {f"@{k}": round(float(np.mean(per_q[k])), 4)
                       for k in ks},
            "per_hop_r10": {str(h): round(float(np.mean(v)), 4)
                            for h, v in sorted(per_hop.items())},
            "n": len(per_q[ks[0]]),
            "matrix_build_s": round(build_s, 1),
            "query_time_s": round(time.perf_counter() - t0, 1),
        }
        print(f"      {arm}: {results[arm]['recall']}"
              + (f"  per-hop@10 {results[arm]['per_hop_r10']}"
                 if per_hop else ""))

    out = HERE / f"ablation_nnsi_{args.dataset}_n{args.n}.json"
    out.write_text(json.dumps(results, indent=2))
    print("\n=== SUMMARY (recall@10 delta vs baseline) ===")
    base = results["baseline"]["recall"]["@10"]
    for arm, r in results.items():
        d = (r["recall"]["@10"] - base) * 100
        print(f"  {arm:<10} @10={r['recall']['@10']:.4f}  ({d:+.2f} pts)")
    print(f"\nwritten → {out}")


if __name__ == "__main__":
    main()
