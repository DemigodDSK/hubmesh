"""hubmesh over a REAL Qdrant server vs the in-memory store — parity and
latency, same corpus, embeddings and knowledge graph. Any of the three
benchmark datasets, any sample size up to the full dev split.

Every published hubmesh number was measured on `InMemoryStore`. This
script answers the operational questions a Qdrant user would ask, with
numbers, per question:

  1. Is the ranking identical?  KG-mode hubmesh pulls every stored vector
     into its own matrix and ranks itself, so the top-k lists over Qdrant
     and over the in-memory store should match exactly; any difference is
     a bug in the adapter path.
  2. What does ingest cost, and what does a fresh process pay to attach to
     an existing collection (`cold_attach`: one vector_of round trip per
     document, plus the batched body fetch)?
  3. What does a query cost?  Per-query latency for hubmesh over each
     store, and for naive top-k over each store (Qdrant's HNSW is
     approximate; the in-memory search is exact).
  4. Do writes invalidate correctly?  An upsert must bump the store's
     `mutation_counter` and make the planner rebuild its matrix.

Long runs checkpoint every `--checkpoint` queries to `<out>.progress.jsonl`
and `<out>.meta.json`; `--resume` continues from them (embeddings are
cached in `--embed-cache`, the Qdrant collection is reused when its point
count matches). Output is gzipped when `--out` ends in `.gz`.

Usage (Qdrant running locally, e.g. `docker run -p 6333:6333 qdrant/qdrant`):
  python benchmarks/run_qdrant_parity.py --dataset hotpotqa --n 500 --embed-device cpu \
      --qdrant-url http://localhost:6333 --out benchmarks/results/qdrant_parity_hotpotqa_n500.json
"""
from __future__ import annotations
import argparse
import gzip
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
from manifest import build_manifest  # noqa: E402

KS = (2, 5, 10)


def load_dataset(dataset: str, n: int, seed: int, representation: str):
    """ids, texts (id -> indexed text), questions [(qid, text)], gold_of."""
    if dataset == "hotpotqa":
        from hotpotqa_loader import load_hotpotqa, retrievable_gold
        examples, pool = load_hotpotqa(n_questions=n, seed=seed)
        ids = list(pool.keys())
        titles = {i: i for i in ids}
    elif dataset == "musique":
        from musique_loader import load_musique, retrievable_gold
        examples, pool = load_musique(n_questions=n, seed=seed, paragraph_identity="title")
        ids = list(pool.keys())
        titles = {i: i for i in ids}
    elif dataset == "2wiki":
        from wiki2_loader import load_wiki2, retrievable_gold
        examples, pool, titles = load_wiki2(n_questions=n, seed=seed)
        ids = list(pool.keys())
    else:
        raise ValueError(dataset)
    pool_set = set(ids)
    gold_of = {ex.qid: retrievable_gold(ex, pool_set) for ex in examples}
    texts = {i: (f"{titles[i]}\n\n{pool[i]}" if representation == "title+body" else pool[i]) for i in ids}
    return ids, texts, [(ex.qid, ex.question) for ex in examples], gold_of


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
    if not xs:
        return {}
    a = np.asarray(xs, dtype=float) * 1000.0
    return {"mean_ms": round(float(a.mean()), 2), "median_ms": round(float(np.median(a)), 2),
            "p95_ms": round(float(np.percentile(a, 95)), 2), "max_ms": round(float(a.max()), 2)}


def write_json(path: Path, obj) -> None:
    data = json.dumps(obj, indent=1)
    if str(path).endswith(".gz"):
        with gzip.open(path, "wt") as f:
            f.write(data)
    else:
        path.write_text(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["hotpotqa", "musique", "2wiki"], default="hotpotqa")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--representation", choices=["body", "title+body"], default="body")
    ap.add_argument("--embed-model", default="all-MiniLM-L6-v2")
    ap.add_argument("--embed-device", default=None)
    ap.add_argument("--embed-cache", default=None, help="npz file to save/load paragraph and question embeddings")
    ap.add_argument("--qdrant-url", default="http://localhost:6333")
    ap.add_argument("--collection", default=None, help="default qdrant_parity_<dataset>_n<n>")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--checkpoint", type=int, default=100, help="queries between progress flushes")
    ap.add_argument("--resume", action="store_true", help="continue from <out>.progress.jsonl / .meta.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    collection = args.collection or f"qdrant_parity_{args.dataset}_n{args.n}"
    out_path = Path(args.out)
    base = str(out_path)[:-3] if str(out_path).endswith(".gz") else str(out_path)
    progress_path = Path(base + ".progress.jsonl")
    meta_path = Path(base + ".meta.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from qdrant_client import QdrantClient
    import requests
    server = requests.get(args.qdrant_url, timeout=10).json()
    print(f"[0/6] Qdrant server: {server}", flush=True)

    print(f"[1/6] Loading {args.dataset} n={args.n} seed={args.seed} ({args.representation})", flush=True)
    t0 = time.perf_counter()
    ids, texts, questions, gold_of = load_dataset(args.dataset, args.n, args.seed, args.representation)
    load_s = time.perf_counter() - t0
    print(f"      pool {len(ids)} paragraphs, {len(questions)} questions ({load_s:.0f}s)", flush=True)

    print(f"[2/6] Embedding with {args.embed_model} on {args.embed_device or 'auto'}", flush=True)
    cache = Path(args.embed_cache) if args.embed_cache else None
    para_vecs = q_vecs = None
    if cache and cache.exists():
        z = np.load(cache)
        if list(z["ids"]) == ids and list(z["qids"]) == [q for q, _ in questions]:
            para_vecs, q_vecs, embed_s = z["para_vecs"], z["q_vecs"], float(z["embed_s"])
            print(f"      embeddings loaded from {cache}", flush=True)
    if para_vecs is None:
        t0 = time.perf_counter()
        para_vecs = embed_texts([texts[i] for i in ids], args.embed_model, args.embed_device)
        q_vecs = embed_texts([q for _, q in questions], args.embed_model, args.embed_device)
        embed_s = time.perf_counter() - t0
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache, ids=np.array(ids), qids=np.array([q for q, _ in questions]),
                     para_vecs=para_vecs, q_vecs=q_vecs, embed_s=np.float64(embed_s))
        print(f"      embedded in {embed_s:.0f}s", flush=True)
    docs = [Document(id=i, text=texts[i], vector=para_vecs[k], metadata={}) for k, i in enumerate(ids)]

    print("[3/6] Building the entity KG (shared by both stores)", flush=True)
    import spacy
    nlp = spacy.load("en_core_web_sm")
    t0 = time.perf_counter()
    kg = build_entity_kg(docs, nlp=nlp)
    kg_s = time.perf_counter() - t0
    print(f"      KG {kg.graph.number_of_nodes()} nodes / {kg.graph.number_of_edges()} edges in {kg_s:.0f}s", flush=True)

    print("[4/6] Stores: in-memory, and the Qdrant collection on the server", flush=True)
    mem = InMemoryStore(docs, k=8)
    client = QdrantClient(url=args.qdrant_url, timeout=300)
    existing = {c.name for c in client.get_collections().collections}
    meta = json.loads(meta_path.read_text()) if (args.resume and meta_path.exists()) else {}
    reused = False
    if args.resume and collection in existing and client.count(collection, exact=True).count == len(docs):
        qd = QdrantStore(client, collection=collection, dim=int(para_vecs.shape[1]))
        upsert_s = meta.get("qdrant_ingest", {}).get("upsert_s")
        reused = True
        print(f"      reusing collection {collection} ({len(docs)} points)", flush=True)
    else:
        if collection in existing:
            client.delete_collection(collection)
        t0 = time.perf_counter()
        qd = QdrantStore.from_documents(docs, client=client, collection=collection)
        upsert_s = round(time.perf_counter() - t0, 2)
    count = client.count(collection, exact=True).count
    assert count == len(docs), f"Qdrant holds {count} points, expected {len(docs)} (id collision?)"
    print(f"      collection {collection}: {count} points; ingest {upsert_s}s", flush=True)

    planner_mem = Planner(store=mem, kg=kg, nlp=nlp)
    planner_qd = Planner(store=qd, kg=kg, nlp=nlp)

    if reused and "warm_up" in meta and "cold_attach" in meta:
        warm_up, cold_attach = meta["warm_up"], meta["cold_attach"]
        planner_mem._kg_doc_matrix()
        planner_qd._kg_doc_matrix()
        print("      warm-up and cold-attach figures kept from the first pass", flush=True)
    else:
        t0 = time.perf_counter()
        ids_mem, mat_mem = planner_mem._kg_doc_matrix()
        warm_mem_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        ids_qd, mat_qd = planner_qd._kg_doc_matrix()
        warm_qd_s = time.perf_counter() - t0
        assert ids_mem == ids_qd, "document id sets differ between stores"
        mat_max_diff = float(np.max(np.abs(mat_mem - mat_qd)))
        warm_up = {"in_memory_s": round(warm_mem_s, 4), "qdrant_s": round(warm_qd_s, 3),
                   "qdrant_vector_cache_prefilled_by_ingest": not reused,
                   "vectors": len(ids_qd), "matrix_max_abs_diff": mat_max_diff}
        print(f"      warm-up: in-memory {warm_mem_s * 1000:.0f} ms, Qdrant {warm_qd_s:.2f}s "
              f"(cache prefilled by ingest: {not reused}); max |Δvector| {mat_max_diff:.2e}", flush=True)
        # the deployment case: a fresh process attaches to an EXISTING collection
        cold = QdrantStore(client, collection=collection, dim=int(para_vecs.shape[1]))
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
        cold_attach = {"all_ids_s": round(cold_ids_s, 2), "vector_of_all_s": round(cold_vectors_s, 2),
                       "vectors": len(cold_ids),
                       "per_vector_ms": round(cold_vectors_s / max(1, len(cold_ids)) * 1000, 2),
                       "fetch_50_bodies_get_many_ms": round(fetch50_batched_s * 1000, 1),
                       "fetch_50_bodies_per_id_ms": round(fetch50_per_id_s * 1000, 1),
                       "note": "vector_of is one retrieve round trip per document; get_many is batched"}
        print(f"      cold attach: all_ids {cold_ids_s:.2f}s, {len(cold_ids)} vector_of round trips "
              f"{cold_vectors_s:.1f}s; 50 bodies: get_many {fetch50_batched_s * 1000:.0f} ms vs per-id "
              f"{fetch50_per_id_s * 1000:.0f} ms", flush=True)
        del cold
    meta = {"dataset": args.dataset, "n": args.n, "seed": args.seed, "representation": args.representation,
            "corpus": {"paragraphs": len(ids), "questions": len(questions),
                       "kg_nodes": kg.graph.number_of_nodes(), "kg_edges": kg.graph.number_of_edges(),
                       "load_s": round(load_s, 1), "embed_s": round(float(embed_s), 1), "kg_build_s": round(kg_s, 1)},
            "qdrant_ingest": {"upsert_s": upsert_s, "points": count, "collection": collection, "reused": reused},
            "warm_up": warm_up, "cold_attach": cold_attach}
    meta_path.write_text(json.dumps(meta, indent=1))

    # ---- queries, with checkpointing -------------------------------------------
    done: dict[str, dict] = {}
    if args.resume and progress_path.exists():
        for line in progress_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["qid"]] = r
        print(f"      resuming: {len(done)} queries already recorded", flush=True)
    todo = [(qi, qid, q) for qi, (qid, q) in enumerate(questions) if qid not in done]
    print(f"[5/6] {len(todo)} queries to run on each store ({len(questions)} total)", flush=True)
    t_start = time.perf_counter()
    buf: list[str] = []
    for j, (qi, qid, q) in enumerate(todo):
        gold = gold_of[qid]
        qv = q_vecs[qi]
        t0 = time.perf_counter()
        r_mem = planner_mem.retrieve(query=q, query_vec=qv, top_k=args.top_k, budget_tokens=10_000)
        l_hm = time.perf_counter() - t0
        t0 = time.perf_counter()
        r_qd = planner_qd.retrieve(query=q, query_vec=qv, top_k=args.top_k, budget_tokens=10_000)
        l_hq = time.perf_counter() - t0
        t0 = time.perf_counter()
        n_mem = [d for d, _ in mem.search(qv, top_k=args.top_k)]
        l_nm = time.perf_counter() - t0
        t0 = time.perf_counter()
        n_qd = [d for d, _ in qd.search(qv, top_k=args.top_k)]
        l_nq = time.perf_counter() - t0
        h_mem = [s.doc.id for s in r_mem.sources]
        h_qd = [s.doc.id for s in r_qd.sources]
        rec = {"qid": qid, "gold": gold, "hub_mem": h_mem, "hub_qd": h_qd, "naive_mem": n_mem, "naive_qd": n_qd,
               "lat_ms": {"hub_mem": round(l_hm * 1000, 2), "hub_qd": round(l_hq * 1000, 2),
                          "naive_mem": round(l_nm * 1000, 2), "naive_qd": round(l_nq * 1000, 2)},
               "same_hub": h_mem == h_qd, "same_naive": n_mem == n_qd}
        done[qid] = rec
        buf.append(json.dumps(rec))
        if (j + 1) % args.checkpoint == 0 or j + 1 == len(todo):
            with open(progress_path, "a") as f:
                f.write("\n".join(buf) + "\n")
            buf = []
            elapsed = time.perf_counter() - t_start
            rate = (j + 1) / elapsed
            eta_min = (len(todo) - j - 1) / max(rate, 1e-9) / 60
            ident = sum(1 for r in done.values() if r["same_hub"])
            print(f"      {len(done)}/{len(questions)} | hub mem {l_hm * 1000:.0f} ms, hub qdrant {l_hq * 1000:.0f} ms "
                  f"| identical {ident}/{len(done)} | {rate * 60:.1f} q/min, ETA {eta_min:.0f} min", flush=True)

    recs = [done[qid] for qid, _ in questions if qid in done]

    def recall(key):
        return {f"recall@{k}": round(float(np.nanmean([recall_at_k(r[key], r["gold"], k) for r in recs])), 4)
                for k in KS}

    print("[6/6] Write path: upsert one document, expect the counter to move and the matrix to rebuild", flush=True)
    counter_before = qd.mutation_counter
    extra = Document(id="__parity_probe__", text="probe document for write invalidation",
                     vector=np.asarray(para_vecs[0], dtype=np.float32), metadata={})
    qd.upsert([extra])
    counter_after = qd.mutation_counter
    cache_version_before = planner_qd._kg_mat_cache[0]
    t0 = time.perf_counter()
    planner_qd._kg_doc_matrix()
    rebuild_s = time.perf_counter() - t0
    cache_version_after = planner_qd._kg_mat_cache[0]
    client.delete(collection, points_selector=[_stable_id(extra.id)])

    out = {
        "manifest": build_manifest(
            harness=__file__, dataset=args.dataset, n=args.n, seed=args.seed, representation=args.representation,
            embed_model=args.embed_model, embed_device=args.embed_device or "auto",
            qdrant_server=server, qdrant_client=pkg_version("qdrant-client"), qdrant_url=args.qdrant_url,
            collection=collection, resumed=bool(args.resume), hardware=f"{platform.machine()} {platform.platform()}"),
        **{k: meta[k] for k in ("corpus", "qdrant_ingest", "warm_up", "cold_attach")},
        "parity": {"hubmesh_identical_top_k": sum(1 for r in recs if r["same_hub"]),
                   "naive_identical_top_k": sum(1 for r in recs if r["same_naive"]),
                   "queries": len(recs), "top_k": args.top_k},
        "recall": {k: recall(k) for k in ("hub_mem", "hub_qd", "naive_mem", "naive_qd")},
        "latency": {k: stats([r["lat_ms"][k] / 1000 for r in recs]) for k in ("hub_mem", "hub_qd", "naive_mem", "naive_qd")},
        "write_invalidation": {"mutation_counter_before": counter_before, "mutation_counter_after": counter_after,
                               "planner_cache_version_before": cache_version_before,
                               "planner_cache_version_after": cache_version_after,
                               "matrix_rebuild_s": round(rebuild_s, 2),
                               "ok": counter_after > counter_before and cache_version_after == counter_after},
        "per_query": recs,
    }
    write_json(out_path, out)
    print(json.dumps({k: v for k, v in out.items() if k not in ("per_query", "manifest")}, indent=1), flush=True)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
