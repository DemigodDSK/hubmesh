"""HotpotQA paragraph-retrieval benchmark.

Compares:
  • naive_topk        — vector top-k by cosine similarity (no graph at all)
  • hubmesh           — full Planner pipeline (subgraph + community anchor +
                          PPR + multi-component scoring + budget packing)

Metric:
  recall@k of supporting-fact paragraphs at k ∈ {2, 5, 10}

Subset size is configurable. Start small (N=100), validate, then scale.
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path
import numpy as np
from tqdm import tqdm

# Make the local package importable when running this file directly.
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from hubmesh import Planner, Document
from hubmesh.adapters import InMemoryStore

from hotpotqa_loader import load_hotpotqa, retrievable_gold
from hippo_style import hippo_style_retrieve


def embed_texts(texts: list[str], batch_size: int = 64, model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    return model.encode(
        texts, batch_size=batch_size, show_progress_bar=True,
        normalize_embeddings=True, convert_to_numpy=True,
    ).astype(np.float32)


def naive_topk_retrieve(store: InMemoryStore, query_vec: np.ndarray, k: int) -> list[str]:
    """Single-shot top-k cosine — the strawman baseline."""
    return [doc_id for doc_id, _ in store.search(query_vec, top_k=k)]


def hubmesh_retrieve(planner: Planner, query_text: str,
                     query_vec: np.ndarray, k: int) -> list[str]:
    """Full hubmesh pipeline. Passes text + vec so KG-mode NER works."""
    result = planner.retrieve(
        query=query_text, query_vec=query_vec.astype(np.float32),
        top_k=k, budget_tokens=10_000,
    )
    return [s.doc.id for s in result.sources]


def recall_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    if not gold:
        return float("nan")
    top = set(retrieved[:k])
    hit = sum(1 for g in gold if g in top)
    return hit / len(gold)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="number of questions")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--knn", type=int, default=8, help="k for proximity graph")
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    ap.add_argument("--kg", action="store_true",
                    help="Build entity-linked KG and use KG-mode retrieval "
                         "(HippoRAG-style; this is the real test)")
    ap.add_argument("--no-hippo", action="store_true",
                    help="Skip the hippo_style ablation (saves ~50% time on "
                         "large N when you only care about hubmesh vs naive).")
    args = ap.parse_args()

    print("=" * 84)
    print(f"HotpotQA paragraph-retrieval benchmark — n={args.n} questions")
    print("=" * 84)

    print("[1/4] Loading HotpotQA distractor dev...")
    examples, pool = load_hotpotqa(n_questions=args.n, seed=args.seed)
    pool_titles = list(pool.keys())
    pool_texts = [pool[t] for t in pool_titles]
    print(f"      pooled corpus: {len(pool_titles)} unique paragraphs")
    n_with_gold = sum(1 for ex in examples
                      if retrievable_gold(ex, set(pool_titles)))
    print(f"      questions with at least one retrievable gold: "
          f"{n_with_gold}/{len(examples)}")

    print(f"[2/4] Embedding corpus + queries with {args.model}...")
    t0 = time.perf_counter()
    para_vecs = embed_texts(pool_texts, model_name=args.model)
    query_vecs = embed_texts([ex.question for ex in examples],
                             model_name=args.model)
    print(f"      embedded {len(pool_texts)} paragraphs + {len(examples)} "
          f"questions in {time.perf_counter()-t0:.1f}s")

    print("[3/4] Building InMemoryStore + Planner...")
    docs = [
        Document(id=t, text=pool[t], vector=para_vecs[i],
                 metadata={"title": t})
        for i, t in enumerate(pool_titles)
    ]
    store = InMemoryStore(docs, k=args.knn)

    kg = None
    if args.kg:
        print("      building entity-linked KG (spaCy NER over corpus)...")
        from hubmesh.kg import build_entity_kg
        import spacy
        nlp = spacy.load("en_core_web_sm")
        t0 = time.perf_counter()
        kg = build_entity_kg(docs, nlp=nlp)
        ent_nodes = sum(1 for n in kg.graph.nodes if n.startswith("ent:"))
        doc_nodes = sum(1 for n in kg.graph.nodes if n.startswith("doc:"))
        co_edges = sum(1 for _, _, d in kg.graph.edges(data=True)
                       if d.get("kind") == "co_occurs")
        print(f"        KG: {doc_nodes} doc nodes, {ent_nodes} entity nodes, "
              f"{kg.graph.number_of_edges()} edges "
              f"({co_edges} entity-entity co-occurrence) "
              f"in {time.perf_counter()-t0:.1f}s")
        planner = Planner(store=store, kg=kg, nlp=nlp)
        print(f"      mode: KG-mode retrieval enabled")
    else:
        planner = Planner(store=store)
        print(f"      mode: kNN graph k={args.knn}")

    print("[4/4] Evaluating retrieval strategies...")
    ks = [2, 5, 10]
    has_kg = kg is not None
    strategies = ["naive_topk", "hubmesh"]
    if has_kg and not args.no_hippo:
        strategies.append("hippo_style")
    print(f"      strategies: {strategies}")
    results = {s: {k: [] for k in ks} for s in strategies}
    timings = {s: 0.0 for s in strategies}

    nlp_for_query = planner._nlp if has_kg else None

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

        for k in ks:
            results["naive_topk"][k].append(recall_at_k(naive, gold, k))
            results["hubmesh"][k].append(recall_at_k(hub, gold, k))

        if has_kg and "hippo_style" in strategies:
            # Lazy-load nlp here in case planner hadn't yet
            if nlp_for_query is None:
                import spacy
                nlp_for_query = spacy.load("en_core_web_sm")
                planner._nlp = nlp_for_query
            t0 = time.perf_counter()
            hippo = hippo_style_retrieve(
                kg, nlp_for_query, ex.question, qvec, store, max(ks),
            )
            timings["hippo_style"] += time.perf_counter() - t0
            for k in ks:
                results["hippo_style"][k].append(recall_at_k(hippo, gold, k))

    print()
    print("=" * 84)
    print(f"RESULTS — supporting-fact recall (mean over "
          f"{len(results['naive_topk'][2])} questions with retrievable gold)")
    print("=" * 84)
    fmt = "{:<20} {:>10} {:>10} {:>10} {:>14}"
    print(fmt.format("strategy", "recall@2", "recall@5", "recall@10",
                     "total_time_s"))
    print("-" * 84)
    for name in strategies:
        row = [name]
        for k in ks:
            arr = np.array(results[name][k], dtype=float)
            row.append(f"{arr.mean():.3f}")
        row.append(f"{timings[name]:.1f}")
        print(fmt.format(*row))

    print()
    print("Δ vs baselines (hubmesh - X):")
    for baseline in (s for s in strategies if s != "hubmesh"):
        for k in ks:
            d = (np.mean(results["hubmesh"][k]) -
                 np.mean(results[baseline][k])) * 100
            print(f"  recall@{k:<2} hubmesh − {baseline:<14} = {d:+.2f} pts")
        print()


if __name__ == "__main__":
    main()
