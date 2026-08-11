"""Re-measure the hubmesh and naive-cosine columns on the title+body
representation ("{title}\n\n{text}"), as committed in HKUDS/LightRAG#3571:
every lane sees identical information (LightRAG's markdown corpus embeds the
title as a heading, so ours must carry it too).

Zero LLM calls. Writes a results JSON next to this script.

    python run_hubmesh_titlebody.py --scope pilot
    python run_hubmesh_titlebody.py --scope pilot --embed-model BAAI/bge-m3 \
        --batch-size 8 --max-seq-len 1024
    python run_hubmesh_titlebody.py --dataset musique --scope pilot \
        --embed-model BAAI/bge-m3 --batch-size 8 --max-seq-len 1024
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))          # benchmarks/ (loaders)
sys.path.insert(0, str(HERE.parent.parent / "src"))  # hubmesh package

from hubmesh import Planner, Document                        # noqa: E402
from hubmesh.planner import PlannerConfig                    # noqa: E402
from hubmesh.adapters import InMemoryStore                   # noqa: E402

# pilot sizes match the repo's published N=500 / N=300 benchmark slices;
# "full" oversizes n so the loader's permutation[:n] returns the whole split.
SCOPES = {
    "hotpotqa": {"pilot": 500, "full": 7405},
    "musique": {"pilot": 300, "full": 999_999},
}


def load_examples(dataset: str, scope: str, seed: int):
    if dataset == "hotpotqa":
        from hotpotqa_loader import load_hotpotqa, retrievable_gold
        ex, pool = load_hotpotqa(n_questions=SCOPES[dataset][scope], seed=seed)
    else:
        from musique_loader import load_musique, retrievable_gold
        ex, pool = load_musique(n_questions=SCOPES[dataset][scope], seed=seed)
    return ex, pool, retrievable_gold


def embed(texts, model_name, batch_size=64, max_seq_length=None):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model_name)
    if max_seq_length:
        # bge-m3 defaults to 8192; a 1,300-word doc padded into a batch of 64
        # blows MPS memory. p99 of this corpus is 254 words, so 1024 is lossless
        # for all but the 3 outliers.
        m.max_seq_length = max_seq_length
    return m.encode(texts, batch_size=batch_size, show_progress_bar=True,
                    normalize_embeddings=True,
                    convert_to_numpy=True).astype(np.float32)


def recall_at_k(retrieved, gold, k):
    top = set(retrieved[:k])
    return sum(1 for g in gold if g in top) / len(gold)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=SCOPES, default="hotpotqa")
    ap.add_argument("--scope", choices=["pilot", "full"], default="pilot")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--embed-model", default="all-MiniLM-L6-v2",
                    help="also run with BAAI/bge-m3 for the naive-on-bge-m3 "
                         "baseline column")
    ap.add_argument("--batch-size", type=int, default=64,
                    help="use 8 or less for large models on MPS")
    ap.add_argument("--max-seq-len", type=int, default=None,
                    help="cap tokenized length (e.g. 1024 for bge-m3 on MPS)")
    ap.add_argument("--knn", type=int, default=8)
    ap.add_argument("--no-convergence", action="store_true",
                    help="ablation: use_convergence=False (README-prescribed "
                         "for top-2 workloads; isolates the @2 dip)")
    args = ap.parse_args()
    ks = [2, 5, 10]

    examples, pool, retrievable_gold = load_examples(
        args.dataset, args.scope, args.seed)
    titles = list(pool.keys())
    texts = [f"{t}\n\n{pool[t]}" for t in titles]          # title+body
    print(f"{args.dataset}: {len(examples)} questions, {len(titles)} "
          f"paragraphs (title+body representation), model={args.embed_model}")

    para_vecs = embed(texts, args.embed_model, args.batch_size, args.max_seq_len)
    query_vecs = embed([ex.question for ex in examples], args.embed_model,
                       args.batch_size, args.max_seq_len)

    docs = [Document(id=t, text=texts[i], vector=para_vecs[i],
                     metadata={"title": t}) for i, t in enumerate(titles)]
    store = InMemoryStore(docs, k=args.knn)

    print("building entity KG (spaCy NER over title+body corpus)...")
    from hubmesh.kg import build_entity_kg
    import spacy
    nlp = spacy.load("en_core_web_sm")
    kg = build_entity_kg(docs, nlp=nlp)
    cfg = PlannerConfig(use_convergence=not args.no_convergence)
    planner = Planner(store=store, kg=kg, nlp=nlp, config=cfg)

    # per-question records so MuSiQue can be aggregated by hop count
    records = []
    timings = {"naive": 0.0, "hubmesh": 0.0}
    pool_set = set(titles)
    for i, ex in enumerate(examples):
        gold = retrievable_gold(ex, pool_set)
        if not gold:
            continue
        t0 = time.perf_counter()
        naive = [d for d, _ in store.search(query_vecs[i], top_k=max(ks))]
        timings["naive"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        r = planner.retrieve(query=ex.question, query_vec=query_vecs[i],
                             top_k=max(ks), budget_tokens=10_000)
        hub = [s.doc.id for s in r.sources]
        timings["hubmesh"] += time.perf_counter() - t0
        records.append({
            "n_hops": getattr(ex, "n_hops", None),
            "naive": {k: recall_at_k(naive, gold, k) for k in ks},
            "hubmesh": {k: recall_at_k(hub, gold, k) for k in ks},
        })
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(examples)}")

    def aggregate(rows):
        return {s: {f"@{k}": round(float(np.mean([r[s][k] for r in rows])), 4)
                    for k in ks} for s in ("naive", "hubmesh")}

    out = {
        "dataset": args.dataset, "scope": args.scope, "seed": args.seed,
        "representation": "title+body",
        "embed_model": args.embed_model, "embed_max_seq_len": args.max_seq_len,
        "use_convergence": not args.no_convergence,
        "n_questions": len(records),
        "recall": aggregate(records),
        "total_query_time_s": {s: round(timings[s], 1) for s in timings},
    }
    if args.dataset == "musique":
        by_hop = defaultdict(list)
        for r in records:
            by_hop[r["n_hops"]].append(r)
        out["recall_by_hop"] = {
            f"{h}-hop (n={len(rows)})": aggregate(rows)
            for h, rows in sorted(by_hop.items())
        }

    suffix = "_noconv" if args.no_convergence else ""
    out_path = HERE / (f"results_titlebody_{args.dataset}_{args.scope}_"
                       f"{args.embed_model.replace('/', '_')}{suffix}.json")
    out_path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
