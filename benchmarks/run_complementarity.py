"""Complementarity experiment — the pre-registered protocol in
PROTOCOL_complementarity.md. Read that file first; every knob here is
fixed there.

Arms: dense / BM25 / hybrid-RRF references; hybrid top-50 -> rerank
(equal-budget control, A4); hybrid top-30 ∪ hubmesh top-20 -> rerank
(treatment, A6); dense-based union (A7); hubmesh top-50 -> rerank (A8);
hubmesh top-5 (A5). Primary metric: complete-evidence@5. Primary
contrast: A6 − A4 with a paired bootstrap and the decision rule from the
protocol.

Two models cannot be resident together on an 8 GB machine, so the run is
staged: stage R (embedder + graph resident) produces candidates and
per-query retrieval timings; stage K (reranker resident) scores the
candidate lists. `--stage all` runs both in one process, unloading the
embedder before loading the reranker; `--stage retrieve` / `--stage
rerank` allow resuming from the stage file.

Latency is composed PER QUERY from stage times before percentiles are
taken, and is labelled an estimate in the output.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import platform
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

from manifest import build_manifest  # noqa: E402

RRF_K = 60
TOPN_FUSE = 200
BASE_N, HUB_N, RERANK_N = 30, 20, 50
KS = (5, 10)
ARMS = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"]
ARM_DESC = {
    "A1": "dense top-5", "A2": "BM25 top-5", "A3": "hybrid RRF top-5",
    "A4": "hybrid top-50 -> rerank -> top-5 (equal-budget control)",
    "A5": "hubmesh top-5",
    "A6": "hybrid top-30 ∪ hubmesh top-20 (backfilled to 50) -> rerank -> top-5 (treatment)",
    "A7": "dense top-30 ∪ hubmesh top-20 (backfilled to 50) -> rerank -> top-5",
    "A8": "hubmesh top-50 -> rerank -> top-5",
}
RERANKED = {"A4", "A6", "A7", "A8"}
STAGES_USED = {           # which retrieval stage times each arm pays
    "A1": ("embed", "dense"), "A2": ("bm25",), "A3": ("embed", "dense", "bm25", "rrf"),
    "A4": ("embed", "dense", "bm25", "rrf", "rerank"),
    "A5": ("embed", "hubmesh"),
    "A6": ("embed", "dense", "bm25", "rrf", "hubmesh", "rerank"),
    "A7": ("embed", "dense", "hubmesh", "rerank"),
    "A8": ("embed", "hubmesh", "rerank"),
}
_TOKEN_RE = re.compile(r"\w+")


# ---- pure functions (unit-tested in tests/test_complementarity_protocol.py) --

def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def rrf_fuse(rankings: list[list[str]], k: int = RRF_K) -> list[str]:
    """Reciprocal-rank fusion of ranked id lists; ties broken by id."""
    score: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            score[doc] = score.get(doc, 0.0) + 1.0 / (k + rank + 1)
    return [d for d, _ in sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))]


def union_backfill(base: list[str], extra: list[str], base_n: int = BASE_N,
                   extra_n: int = HUB_N, total: int = RERANK_N) -> list[str]:
    """base[:base_n], then extra[:extra_n] in extra's order skipping
    duplicates, then base[base_n:] until `total` distinct ids."""
    out: list[str] = []
    seen: set[str] = set()
    for d in list(base[:base_n]) + list(extra[:extra_n]) + list(base[base_n:]):
        if d in seen:
            continue
        out.append(d)
        seen.add(d)
        if len(out) >= total:
            break
    return out


def complete_evidence(retrieved: list[str], gold: list[str], k: int) -> float:
    """1.0 iff every distinct gold id is among the top-k distinct retrieved ids."""
    if not gold:
        return 0.0
    top = set(retrieved[:k])
    return 1.0 if all(g in top for g in set(gold)) else 0.0


def recall_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    if not gold:
        return 0.0
    top = set(retrieved[:k])
    gold_set = set(gold)
    return sum(1 for g in gold_set if g in top) / len(gold_set)


def paired_bootstrap(a: list[float], b: list[float], n_boot: int = 10_000,
                     seed: int = 0) -> dict | None:
    d = (np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) * 100.0
    if len(d) == 0:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return {"mean": round(float(d.mean()), 2),
            "lo": round(float(np.percentile(means, 2.5)), 2),
            "hi": round(float(np.percentile(means, 97.5)), 2), "n": int(len(d))}


def decide(contrast: dict | None, threshold: float = 3.0) -> str:
    """Protocol decision rule on the primary contrast (points)."""
    if contrast is None:
        return "inconclusive"
    if contrast["mean"] >= threshold and contrast["lo"] > 0:
        return "pass"
    if contrast["hi"] < threshold:
        return "no useful gain"
    return "inconclusive"


def compose_latency(stage_times: list[dict], stages: tuple[str, ...]) -> dict:
    """Per-query sums of the named stage times, THEN median / p95."""
    sums = np.array([sum(float(st.get(s, 0.0)) for s in stages) for st in stage_times])
    if len(sums) == 0:
        return {"median_ms": None, "p95_ms": None}
    return {"median_ms": round(float(np.median(sums)) * 1000, 2),
            "p95_ms": round(float(np.percentile(sums, 95)) * 1000, 2)}


def protocol_sha256(path: Path = HERE / "PROTOCOL_complementarity.md") -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


# ---- data ------------------------------------------------------------------

def load_data(dataset: str, n: int, seed: int):
    """Return (examples, pool_ids, texts, titles, gold_of, qtype_of, hop_of)."""
    if dataset == "2wiki":
        from wiki2_loader import load_wiki2, retrievable_gold
        examples, pool, titles = load_wiki2(n_questions=n, seed=seed)
        ids = list(pool.keys())
        gold_of = {ex.qid: retrievable_gold(ex, set(ids)) for ex in examples}
        qtype_of = {ex.qid: ex.qtype for ex in examples}
        hop_of = {}
    elif dataset == "hotpotqa":
        from hotpotqa_loader import load_hotpotqa, retrievable_gold
        examples, pool = load_hotpotqa(n_questions=n, seed=seed)
        ids = list(pool.keys())
        titles = {t: t for t in ids}
        gold_of = {ex.qid: retrievable_gold(ex, set(ids)) for ex in examples}
        qtype_of = {ex.qid: ex.qtype for ex in examples}
        hop_of = {}
    elif dataset == "musique":
        from musique_loader import load_musique, retrievable_gold
        examples, pool = load_musique(n_questions=n, seed=seed,
                                      paragraph_identity="title+text")
        ids = list(pool.keys())
        titles = {t: t.split("::")[0] for t in ids}
        gold_of = {ex.qid: retrievable_gold(ex, set(ids)) for ex in examples}
        qtype_of = {ex.qid: f"{ex.n_hops}-hop" for ex in examples}
        hop_of = {ex.qid: ex.n_hops for ex in examples}
    else:
        raise ValueError(dataset)
    texts = {i: f"{titles[i]}\n\n{pool[i]}" for i in ids}
    return examples, ids, texts, titles, gold_of, qtype_of, hop_of


# ---- stage R: embedder + graph resident ------------------------------------

def stage_retrieve(args, examples, ids, texts, gold_of):
    from sentence_transformers import SentenceTransformer
    from rank_bm25 import BM25Okapi
    from hubmesh import Planner, Document
    from hubmesh.adapters import InMemoryStore
    from hubmesh.kg import build_entity_kg
    import spacy

    model = SentenceTransformer(args.embed_model, device=args.embed_device)
    if args.embed_max_seq_len:
        model.max_seq_length = args.embed_max_seq_len
    t0 = time.perf_counter()
    para_vecs = model.encode([texts[i] for i in ids], batch_size=args.embed_batch_size,
                             show_progress_bar=True, normalize_embeddings=True,
                             convert_to_numpy=True).astype(np.float32)
    embed_pool_s = time.perf_counter() - t0

    docs = [Document(id=i, text=texts[i], vector=para_vecs[j]) for j, i in enumerate(ids)]
    store = InMemoryStore(docs, k=8)
    nlp = spacy.load("en_core_web_sm")
    t0 = time.perf_counter()
    kg = build_entity_kg(docs, nlp=nlp)
    kg_build_s = time.perf_counter() - t0
    planner = Planner(store=store, kg=kg, nlp=nlp)
    bm25 = BM25Okapi([tokenize(texts[i]) for i in ids])
    id_arr = np.array(ids)

    records = []
    for ex in examples:
        gold = gold_of[ex.qid]
        st: dict[str, float] = {}
        t0 = time.perf_counter()
        qvec = model.encode([ex.question], normalize_embeddings=True,
                            convert_to_numpy=True)[0].astype(np.float32)
        st["embed"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        dense = [d for d, _ in store.search(qvec, top_k=TOPN_FUSE)]
        st["dense"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        scores = bm25.get_scores(tokenize(ex.question))
        order = np.argsort(-scores, kind="stable")[:TOPN_FUSE]
        lex = [str(id_arr[j]) for j in order]
        st["bm25"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        hybrid = rrf_fuse([dense, lex])
        st["rrf"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        res = planner.retrieve(query=ex.question, query_vec=qvec, top_k=RERANK_N,
                               budget_tokens=10_000)
        st["hubmesh"] = time.perf_counter() - t0
        hub = [s.doc.id for s in res.sources]
        n_seeds = len(res.debug.get("ppr_seeds", [])) if res.debug else 0
        records.append({
            "qid": ex.qid, "gold": gold, "n_gold": len(gold),
            "n_seeds": n_seeds,
            "overlap_hybrid30_hub20": len(set(hybrid[:BASE_N]) & set(hub[:HUB_N])),
            "gold_docs_without_entities": sum(
                1 for g in gold if not kg.doc_to_entities.get(g)),
            "stage_times": st,
            "cand": {
                "A1": dense[:5], "A2": lex[:5], "A3": hybrid[:5],
                "A4": hybrid[:RERANK_N], "A5": hub[:5],
                "A6": union_backfill(hybrid, hub),
                "A7": union_backfill(dense, hub),
                "A8": hub[:RERANK_N],
            },
        })
    stage = {
        "embed_model": args.embed_model, "embed_device": args.embed_device or "auto",
        "embed_batch_size": args.embed_batch_size,
        "embed_max_seq_len": args.embed_max_seq_len,
        "pool": len(ids), "embed_pool_s": round(embed_pool_s, 1),
        "kg_nodes": kg.graph.number_of_nodes(), "kg_edges": kg.graph.number_of_edges(),
        "kg_build_s": round(kg_build_s, 1), "records": records,
    }
    del model, planner, store, kg, bm25
    gc.collect()
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass
    return stage


# ---- stage K: reranker resident --------------------------------------------

def stage_rerank(args, stage, texts, questions):
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder(args.rerank_model, max_length=512, device=args.rerank_device)
    for rec in stage["records"]:
        q = questions[rec["qid"]]
        # score the union of the reranked arms' candidates once (a pair's
        # score does not depend on which list it came from) ...
        union = list(dict.fromkeys(d for a in RERANKED for d in rec["cand"][a]))
        t0 = time.perf_counter()
        # ... but time ONE 50-candidate list (A4's) as the per-query rerank
        # cost every 50-candidate arm pays (protocol: stage K).
        a4 = rec["cand"]["A4"]
        s4 = ce.predict([(q, texts[d]) for d in a4], batch_size=args.rerank_batch_size,
                        show_progress_bar=False)
        rec["stage_times"]["rerank"] = time.perf_counter() - t0
        rest = [d for d in union if d not in set(a4)]
        s_rest = (ce.predict([(q, texts[d]) for d in rest], batch_size=args.rerank_batch_size,
                             show_progress_bar=False) if rest else [])
        score = {d: float(s) for d, s in zip(a4, s4)}
        score.update({d: float(s) for d, s in zip(rest, s_rest)})
        rec["final"] = {}
        for a in ARMS:
            cand = rec["cand"][a]
            if a in RERANKED:
                # stable sort by score desc; ties keep candidate order
                idx = sorted(range(len(cand)), key=lambda i: (-score[cand[i]], i))
                rec["final"][a] = [cand[i] for i in idx[:10]]
            else:
                rec["final"][a] = cand[:10]
    stage["rerank_model"] = args.rerank_model
    stage["rerank_device"] = args.rerank_device or "auto"
    stage["rerank_batch_size"] = args.rerank_batch_size
    del ce
    gc.collect()
    return stage


# ---- analysis ----------------------------------------------------------------

def analyse(args, stage, qtype_of, hop_of):
    recs = stage["records"]
    for r in recs:
        r["metrics"] = {a: {f"ce@{k}": complete_evidence(r["final"][a], r["gold"], k)
                            for k in KS} | {f"r@{k}": recall_at_k(r["final"][a], r["gold"], k)
                                             for k in KS}
                        for a in ARMS}
        r["qtype"] = qtype_of.get(r["qid"])
        if hop_of:
            r["hop"] = hop_of.get(r["qid"])

    def agg(rows, arm, key):
        return round(float(np.mean([r["metrics"][arm][key] for r in rows])), 4) if rows else None

    def contrast(rows, x, y, key):
        return paired_bootstrap([r["metrics"][x][key] for r in rows],
                                [r["metrics"][y][key] for r in rows],
                                n_boot=args.boot, seed=args.seed)

    metric_keys = [f"ce@{k}" for k in KS] + [f"r@{k}" for k in KS]
    contrasts = [("A6", "A4"), ("A7", "A4"), ("A8", "A4"), ("A6", "A8"),
                 ("A4", "A3"), ("A5", "A1"), ("A5", "A3"), ("A6", "A5"), ("A3", "A1")]
    slices = {"all": recs}
    comp = [r for r in recs if r.get("qtype") == "comparison"]
    if comp:
        slices["comparison-type questions"] = comp
    few = [r for r in recs if r["n_seeds"] < 2]
    if few:
        slices["fewer than two hubmesh seeds"] = few
    if hop_of:
        for h in sorted({r["hop"] for r in recs if r.get("hop")}):
            slices[f"{h}-hop"] = [r for r in recs if r.get("hop") == h]

    primary = contrast(recs, "A6", "A4", "ce@5")
    is_primary_config = (args.dataset == "2wiki" and "bge-m3" in args.embed_model)
    guard = contrast(comp, "A6", "A4", "ce@5") if comp else None

    out = {
        "manifest": build_manifest(
            harness=__file__, protocol="benchmarks/PROTOCOL_complementarity.md",
            protocol_sha256=protocol_sha256(), dataset=args.dataset, n=args.n,
            seed=args.seed, representation="title+body",
            embed_model=stage["embed_model"], embed_device=stage["embed_device"],
            embed_batch_size=stage["embed_batch_size"],
            embed_max_seq_len=stage.get("embed_max_seq_len"),
            rerank_model=stage.get("rerank_model"), rerank_device=stage.get("rerank_device"),
            rerank_batch_size=stage.get("rerank_batch_size"),
            hardware=f"{platform.machine()} {platform.platform()}",
            rrf_k=RRF_K, fuse_topn=TOPN_FUSE, base_n=BASE_N, hub_n=HUB_N,
            rerank_n=RERANK_N, boot=args.boot),
        "arms": ARM_DESC,
        "corpus": {k: stage[k] for k in ("pool", "kg_nodes", "kg_edges", "kg_build_s",
                                         "embed_pool_s")},
        "n": len(recs),
        "primary": {
            "contrast": "A6 - A4, complete-evidence@5",
            "is_primary_configuration": is_primary_config,
            "result": primary,
            "decision": decide(primary) if is_primary_config else
                        f"(secondary configuration) {decide(primary)}",
            "regression_guard": {
                "slice": "comparison-type questions (question-type slice; not a claim about simple questions)",
                "n": len(comp), "A6-A4 ce@5": guard,
                "rule": "point estimate >= -1.0",
                "met": (guard["mean"] >= -1.0) if guard else None,
            },
        },
        "diagnostics": {
            "frac_queries_with_ge2_seeds": round(float(np.mean([r["n_seeds"] >= 2 for r in recs])), 4),
            "mean_overlap_hybrid30_hub20": round(float(np.mean([r["overlap_hybrid30_hub20"] for r in recs])), 2),
            "frac_gold_docs_without_entities": round(
                float(sum(r["gold_docs_without_entities"] for r in recs)
                      / max(1, sum(r["n_gold"] for r in recs))), 4),
            "mean_candidates_A6": round(float(np.mean([len(r["cand"]["A6"]) for r in recs])), 2),
        },
        "results": {name: {a: {k: agg(rows, a, k) for k in metric_keys} for a in ARMS}
                    for name, rows in slices.items()},
        "contrasts": {name: {f"{x}-{y}": {k: contrast(rows, x, y, k) for k in metric_keys}
                             for x, y in contrasts}
                      for name, rows in slices.items()},
        "latency_composed_estimate": {
            "note": ("per-query sums of measured stage times, then median/p95; "
                     "excludes model switching, cold start and process/request "
                     "overhead — an estimate of a both-models-resident deployment"),
            "stages_per_arm": STAGES_USED,
            "stage_medians_ms": {s: round(float(np.median([r["stage_times"].get(s, 0.0)
                                                            for r in recs])) * 1000, 2)
                                 for s in ("embed", "dense", "bm25", "rrf", "hubmesh", "rerank")},
            "arms": {a: compose_latency([r["stage_times"] for r in recs], STAGES_USED[a])
                     for a in ARMS},
        },
        "per_query": recs,
    }
    a6 = out["latency_composed_estimate"]["arms"]["A6"]["p95_ms"]
    a4 = out["latency_composed_estimate"]["arms"]["A4"]["p95_ms"]
    out["latency_composed_estimate"]["p95_ratio_A6_over_A4"] = (
        round(a6 / a4, 3) if a6 and a4 else None)
    out["latency_composed_estimate"]["rule_met_le_1.5"] = (
        (a6 / a4 <= 1.5) if a6 and a4 else None)
    return out


# ---- main ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["2wiki", "hotpotqa", "musique"], required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--embed-model", default="BAAI/bge-m3")
    ap.add_argument("--embed-device", default=None)
    ap.add_argument("--embed-batch-size", type=int, default=8)
    ap.add_argument("--embed-max-seq-len", type=int, default=1024)
    ap.add_argument("--rerank-model", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--rerank-device", default=None)
    ap.add_argument("--rerank-batch-size", type=int, default=16)
    ap.add_argument("--boot", type=int, default=10_000)
    ap.add_argument("--stage", choices=["all", "retrieve", "rerank"], default="all")
    ap.add_argument("--stage-file", default=None,
                    help="intermediate candidates/timings (default: <out>.stage.json)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out_path = Path(args.out)
    stage_path = Path(args.stage_file) if args.stage_file else out_path.with_suffix(".stage.json")

    print(f"[1/4] Loading {args.dataset} n={args.n} seed={args.seed}")
    examples, ids, texts, titles, gold_of, qtype_of, hop_of = load_data(args.dataset, args.n, args.seed)
    print(f"      pool {len(ids)} paragraphs, {len(examples)} questions, "
          f"{sum(1 for ex in examples if gold_of[ex.qid])} with retrievable gold")
    questions = {ex.qid: ex.question for ex in examples}

    if args.stage in ("all", "retrieve"):
        print(f"[2/4] Stage R: {args.embed_model} + KG + BM25 + hubmesh")
        stage = stage_retrieve(args, examples, ids, texts, gold_of)
        stage_path.parent.mkdir(parents=True, exist_ok=True)
        stage_path.write_text(json.dumps(stage))
        print(f"      stage file -> {stage_path}")
        if args.stage == "retrieve":
            return
    else:
        stage = json.loads(stage_path.read_text())

    print(f"[3/4] Stage K: {args.rerank_model}")
    stage = stage_rerank(args, stage, texts, questions)
    stage_path.write_text(json.dumps(stage))

    print("[4/4] Analysis (examined once)")
    out = analyse(args, stage, qtype_of, hop_of)
    out_path.write_text(json.dumps(out, indent=1))
    p = out["primary"]
    print(f"primary A6-A4 ce@5: {p['result']} -> {p['decision']}")
    print(f"guard {p['regression_guard']['slice']}: n={p['regression_guard']['n']} "
          f"{p['regression_guard']['A6-A4 ce@5']} met={p['regression_guard']['met']}")
    for a in ARMS:
        r = out["results"]["all"][a]
        print(f"  {a:<3} ce@5 {r['ce@5']:.3f}  ce@10 {r['ce@10']:.3f}  r@5 {r['r@5']:.3f}  r@10 {r['r@10']:.3f}"
              f"  | composed p95 {out['latency_composed_estimate']['arms'][a]['p95_ms']} ms")
    print("diagnostics:", out["diagnostics"])
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
