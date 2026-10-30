"""hubmesh over a REAL Qdrant server vs the in-memory store — parity and
latency, same corpus, embeddings and knowledge graph.

Every published hubmesh number was measured on `InMemoryStore`. The
Qdrant adapter had only ever run in Qdrant's embedded `:memory:` mode on
synthetic data. This script answers the operational questions a Qdrant
user would ask, with numbers:

  1. Is the ranking identical?  KG-mode hubmesh pulls every stored vector
     into its own matrix and ranks itself, so the top-k lists over Qdrant
     and over the in-memory store should match exactly; any difference is
     a bug in the adapter path.
  2. What does the first query cost?  The planner's vector matrix is built
     from `vector_of` — against a server that is one round trip per
     document (as shipped). Measured as `warmup_s`.
  3. What does a query cost?  Per-query latency for hubmesh over each
     store, and for naive top-k over each store (Qdrant's HNSW is
     approximate; the in-memory search is exact).
  4. Do writes invalidate correctly?  An upsert must bump the store's
     `mutation_counter` and make the planner rebuild its matrix.

Usage (Qdrant running locally, e.g. `docker run -p 6333:6333 qdrant/qdrant`):
  python benchmarks/run_qdrant_parity.py --n 500 --embed-device mps \
      --qdrant-url http://localhost:6333 --out benchmarks/results/qdrant_parity_hotpotqa_n500.json
"""
from __future__ import annotations
import argparse
import json
import platform
import sys
import time
from importlib.metadata import version as pkg_version
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

from hubmesh import Planner, Document  # noqa: E402
from hubmesh.adapters import InMemoryStore  # noqa: E402
from hubmesh.adapters.qdrant import QdrantStore, _stable_id  # noqa: E402
from hubmesh.kg import build_entity_kg  # noqa: E402
from hotpotqa_loader import load_hotpotqa, retrievable_gold  # noqa: E402
from manifest import build_manifest  # noqa: E402

KS = (2, 5, 10)


def embed_texts(texts, model_name, device, batch_size=64):
    """Identical to benchmarks/run_hotpotqa.py: normalised float32."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device=device)
    return model.encode(texts, batch_size=batch_size, show_progress_bar=False,
                        normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)


def recall_at_k(retrieved, gold, k):
    if not gold:
        return float("nan")
    top = set(retrieved[:k])
    return sum(1 for g in gold if g in top) / len(gold)


def stats(xs):
    a = np.asarray(xs, dtype=float) * 1000.0
    return {"mean_ms": round(float(a.mean()), 2), "median_ms": round(float(np.median(a)), 2),
            "p95_ms": round(float(np.percentile(a, 95)), 2), "max_ms": round(float(a.max()), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--embed-model", default="all-MiniLM-L6-v2")
    ap.add_argument("--embed-device", default=None)
    ap.add_argument("--qdrant-url", default="http://localhost:6333")
    ap.add_argument("--collection", default="hubmesh_parity")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from qdrant_client import QdrantClient
    import requests
    server = requests.get(args.qdrant_url, timeout=10).json()
    print(f"[0/6] Qdrant server: {server}")

    print(f"[1/6] Loading HotpotQA n={args.n} seed={args.seed} (body-only, as hotpotqa_n500_kg_body.json)")
    examples, pool = load_hotpotqa(n_questions=args.n, seed=args.seed)
    titles = list(pool.keys())
    texts = [pool[t] for t in titles]
    print(f"      pool {len(titles)} paragraphs, {len(examples)} questions")

    print(f"[2/6] Embedding with {args.embed_model} on {args.embed_device or 'auto'}")
    t0 = time.perf_counter()
    para_vecs = embed_texts(texts, args.embed_model, args.embed_device)
    q_vecs = embed_texts([ex.question for ex in examples], args.embed_model, args.embed_device)
    embed_s = time.perf_counter() - t0
    docs = [Document(id=t, text=texts[i], vector=para_vecs[i], metadata={"title": t})
            for i, t in enumerate(titles)]

    print("[3/6] Building the entity KG (shared by both stores)")
    import spacy
    nlp = spacy.load("en_core_web_sm")
    t0 = time.perf_counter()
    kg = build_entity_kg(docs, nlp=nlp)
    kg_s = time.perf_counter() - t0
    print(f"      KG {kg.graph.number_of_nodes()} nodes / {kg.graph.number_of_edges()} edges in {kg_s:.1f}s")

    print("[4/6] Stores: in-memory, and a fresh Qdrant collection on the server")
    mem = InMemoryStore(docs, k=8)
    client = QdrantClient(url=args.qdrant_url, timeout=120)
    if args.collection in {c.name for c in client.get_collections().collections}:
        client.delete_collection(args.collection)
    t0 = time.perf_counter()
    qd = QdrantStore.from_documents(docs, client=client, collection=args.collection)
    upsert_s = time.perf_counter() - t0
    count = client.count(args.collection, exact=True).count
    print(f"      upserted {len(docs)} docs in {upsert_s:.1f}s; server count {count}")
    assert count == len(docs), f"Qdrant holds {count} points, expected {len(docs)} (id collision?)"

    planner_mem = Planner(store=mem, kg=kg, nlp=nlp)
    planner_qd = Planner(store=qd, kg=kg, nlp=nlp)

    # warm-up: the planner's document-vector matrix (one vector_of per doc)
    t0 = time.perf_counter()
    ids_mem, mat_mem = planner_mem._kg_doc_matrix()
    warm_mem_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    ids_qd, mat_qd = planner_qd._kg_doc_matrix()
    warm_qd_s = time.perf_counter() - t0
    assert ids_mem == ids_qd, "document id sets differ between stores"
    mat_max_diff = float(np.max(np.abs(mat_mem - mat_qd)))
    print(f"      warm-up (adapter that did the ingest; its vector cache is already populated): "
          f"in-memory {warm_mem_s * 1000:.0f} ms, Qdrant {warm_qd_s:.2f}s; max |Δvector| {mat_max_diff:.2e}")

    # The deployment case: a fresh process attaches to an EXISTING collection,
    # so nothing is cached and every vector has to come from the server.
    cold = QdrantStore(client, collection=args.collection, dim=int(para_vecs.shape[1]))
    t0 = time.perf_counter()
    cold_ids = cold.all_ids()
    cold_ids_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    for d in cold_ids:
        cold.vector_of(d)
    cold_vectors_s = time.perf_counter() - t0
    probe = cold_ids[:50]
    t0 = time.perf_counter()
    cold.get_many(probe)
    fetch50_batched_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    for d in probe:
        cold.get(d)
    fetch50_per_id_s = time.perf_counter() - t0
    print(f"      cold attach: all_ids {cold_ids_s:.2f}s, {len(cold_ids)} vector_of round trips {cold_vectors_s:.1f}s; "
          f"50 bodies: get_many {fetch50_batched_s * 1000:.0f} ms vs per-id {fetch50_per_id_s * 1000:.0f} ms")

    print(f"[5/6] {len(examples)} queries on each store")
    pool_set = set(titles)
    recs = []
    lat = {"hub_mem": [], "hub_qd": [], "naive_mem": [], "naive_qd": []}
    identical = identical_naive = 0
    for i, ex in enumerate(examples):
        gold = retrievable_gold(ex, pool_set)
        qv = q_vecs[i]
        t0 = time.perf_counter()
        r_mem = planner_mem.retrieve(query=ex.question, query_vec=qv, top_k=args.top_k, budget_tokens=10_000)
        lat["hub_mem"].append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        r_qd = planner_qd.retrieve(query=ex.question, query_vec=qv, top_k=args.top_k, budget_tokens=10_000)
        lat["hub_qd"].append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        n_mem = [d for d, _ in mem.search(qv, top_k=args.top_k)]
        lat["naive_mem"].append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        n_qd = [d for d, _ in qd.search(qv, top_k=args.top_k)]
        lat["naive_qd"].append(time.perf_counter() - t0)
        h_mem = [s.doc.id for s in r_mem.sources]
        h_qd = [s.doc.id for s in r_qd.sources]
        identical += h_mem == h_qd
        identical_naive += n_mem == n_qd
        recs.append({"qid": ex.qid, "gold": gold, "hub_mem": h_mem, "hub_qd": h_qd,
                     "naive_mem": n_mem, "naive_qd": n_qd,
                     "lat_ms": {k: round(v[-1] * 1000, 2) for k, v in lat.items()},
                     "same_hub": h_mem == h_qd, "same_naive": n_mem == n_qd})
        if i % 100 == 0:
            print(f"      {i}/{len(examples)}  hub mem {lat['hub_mem'][-1] * 1000:.0f} ms  "
                  f"hub qdrant {lat['hub_qd'][-1] * 1000:.0f} ms  identical so far {identical}/{i + 1}", flush=True)

    def recall(key):
        return {f"recall@{k}": round(float(np.nanmean([recall_at_k(r[key], r["gold"], k) for r in recs])), 4)
                for k in KS}

    print("[6/6] Write path: upsert one document, expect the counter to move and the matrix to rebuild")
    counter_before = qd.mutation_counter
    extra = Document(id="__parity_probe__", text="probe document for write invalidation",
                     vector=np.asarray(para_vecs[0], dtype=np.float32), metadata={"title": "probe"})
    qd.upsert([extra])
    counter_after = qd.mutation_counter
    cache_version_before = planner_qd._kg_mat_cache[0]
    t0 = time.perf_counter()
    planner_qd._kg_doc_matrix()
    rebuild_s = time.perf_counter() - t0
    cache_version_after = planner_qd._kg_mat_cache[0]
    client.delete(args.collection, points_selector=[_stable_id(extra.id)])

    out = {
        "manifest": build_manifest(
            harness=__file__, dataset="hotpotqa", n=args.n, seed=args.seed, representation="body",
            embed_model=args.embed_model, embed_device=args.embed_device or "auto",
            qdrant_server=server, qdrant_client=pkg_version("qdrant-client"), qdrant_url=args.qdrant_url,
            collection=args.collection, hardware=f"{platform.machine()} {platform.platform()}"),
        "corpus": {"paragraphs": len(titles), "questions": len(examples),
                   "kg_nodes": kg.graph.number_of_nodes(), "kg_edges": kg.graph.number_of_edges(),
                   "embed_s": round(embed_s, 1), "kg_build_s": round(kg_s, 1)},
        "qdrant_ingest": {"upsert_s": round(upsert_s, 2), "points": count},
        "warm_up": {"in_memory_s": round(warm_mem_s, 4),
                    "qdrant_after_ingest_s": round(warm_qd_s, 3),
                    "vectors": len(ids_qd), "matrix_max_abs_diff": mat_max_diff,
                    "note": "the adapter that ingested already holds every vector in its cache; see cold_attach "
                            "for a fresh process attaching to the existing collection"},
        "cold_attach": {"all_ids_s": round(cold_ids_s, 2), "vector_of_all_s": round(cold_vectors_s, 2),
                        "vectors": len(cold_ids), "per_vector_ms": round(cold_vectors_s / max(1, len(cold_ids)) * 1000, 2),
                        "fetch_50_bodies_get_many_ms": round(fetch50_batched_s * 1000, 1),
                        "fetch_50_bodies_per_id_ms": round(fetch50_per_id_s * 1000, 1),
                        "note": "vector_of is one retrieve round trip per document (as shipped); "
                                "get_many is batched since this change"},
        "parity": {"hubmesh_identical_top_k": identical, "naive_identical_top_k": identical_naive,
                   "queries": len(recs), "top_k": args.top_k},
        "recall": {k: recall(k) for k in ("hub_mem", "hub_qd", "naive_mem", "naive_qd")},
        "latency": {k: stats(v) for k, v in lat.items()},
        "write_invalidation": {"mutation_counter_before": counter_before, "mutation_counter_after": counter_after,
                               "planner_cache_version_before": cache_version_before,
                               "planner_cache_version_after": cache_version_after,
                               "matrix_rebuild_s": round(rebuild_s, 2),
                               "ok": counter_after > counter_before and cache_version_after == counter_after},
        "per_query": recs,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("per_query", "manifest")}, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
