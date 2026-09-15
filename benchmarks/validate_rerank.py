"""Validation of the fixed-shape reranker path (protocol Amendment 2)
against the library path, plus an endurance run, on real stage-R
candidates. Computes NO metric — only score differences, rank
agreement, throughput and memory.

  A. parity on Metal: the same CrossEncoder weights and tokenizer scored
     (i) through CrossEncoder.predict (dynamic padding, identity
     activation) and (ii) through FixedShapeReranker; per query: max and
     mean |logit difference|, full-order agreement, top-5 / top-10
     agreement for each reranked arm.
  B. endurance on Metal: fixed-shape scoring of the union of candidates
     for E queries; allocated / driver Metal memory and RSS sampled every
     query; throughput reported as measured (an estimate for this
     machine, not a hardware limit).
  C. fp32 CPU reference (small): the library path in fp32 on CPU for the
     first C queries vs. the fixed-shape fp16 Metal scores from A — the
     size of the precision/device effect on logits and rankings.

Usage:
  python benchmarks/validate_rerank.py --dataset 2wiki --n 1000 --stage-file <stage.json> \
      --out benchmarks/results/rerank_validation_2wiki_mps_fp16.json
"""
from __future__ import annotations
import argparse
import gc
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run_complementarity import (  # noqa: E402
    FixedShapeReranker, RERANKED, RERANK_BUCKETS, RERANK_MAX_LENGTH, _mps_empty_cache,
    _mps_memory, load_data, validate_stage,
)
from manifest import build_manifest  # noqa: E402


def rss_gb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30, 3)


def order(scores: list[float]) -> list[int]:
    return sorted(range(len(scores)), key=lambda i: (-scores[i], i))


def arm_agreement(rec: dict, docs: list[str], a: list[float], b: list[float]) -> dict:
    """Per reranked arm: does ranking the arm's candidates by a vs. by b
    give the same top-5 / top-10 (order-sensitive)?"""
    pos = {d: i for i, d in enumerate(docs)}
    out = {}
    for arm in sorted(RERANKED):
        cand = rec["cand"][arm]
        oa = sorted(range(len(cand)), key=lambda i: (-a[pos[cand[i]]], i))
        ob = sorted(range(len(cand)), key=lambda i: (-b[pos[cand[i]]], i))
        out[arm] = {"top5": oa[:5] == ob[:5], "top10": oa[:10] == ob[:10],
                    "top5_set": set(oa[:5]) == set(ob[:5])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["2wiki", "hotpotqa", "musique"])
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stage-file", required=True)
    ap.add_argument("--embed-model", default="BAAI/bge-m3")
    ap.add_argument("--rerank-model", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--parity-queries", type=int, default=20)
    ap.add_argument("--endurance-queries", type=int, default=150)
    ap.add_argument("--cpu-queries", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from sentence_transformers import CrossEncoder

    examples, ids, texts, _titles, gold_of, _q, _h = load_data(args.dataset, args.n, args.seed)
    questions = {ex.qid: ex.question for ex in examples}
    stage = json.loads(Path(args.stage_file).read_text())
    validation = validate_stage(stage, {"dataset": args.dataset, "n": args.n, "seed": args.seed,
                                        "embed_model": args.embed_model},
                                [ex.qid for ex in examples], ids)
    recs = stage["records"]
    print(f"stage file validated: {validation}", flush=True)

    def union_pairs(rec):
        docs = list(dict.fromkeys(d for a in sorted(RERANKED) for d in rec["cand"][a]))
        return docs, [(questions[rec["qid"]], texts[d]) for d in docs]

    ce_kwargs = {"model_kwargs": {"torch_dtype": torch.float16}} if args.dtype == "fp16" else {}
    ce = CrossEncoder(args.rerank_model, max_length=RERANK_MAX_LENGTH, device=args.device, **ce_kwargs)
    fx = FixedShapeReranker(ce, device=args.device, buckets=RERANK_BUCKETS, batch=args.batch)
    report = {"manifest": build_manifest(harness=__file__, dataset=args.dataset, n=args.n, seed=args.seed,
                                         stage_file=args.stage_file, stage_validation=validation,
                                         rerank_model=args.rerank_model, device=args.device,
                                         dtype_requested=args.dtype, dtype_actual=fx.dtype(),
                                         buckets=list(RERANK_BUCKETS), batch=args.batch,
                                         max_length=RERANK_MAX_LENGTH,
                                         hardware=f"{platform.machine()} {platform.platform()}",
                                         mps_high_watermark_ratio=os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO"),
                                         mps_low_watermark_ratio=os.environ.get("PYTORCH_MPS_LOW_WATERMARK_RATIO"))}
    print(f"model on {args.device} as {fx.dtype()} | {_mps_memory()} rss {rss_gb()} GB", flush=True)
    report["warm_up"] = fx.warm_up()

    # ---- A. parity: library path vs fixed-shape, same weights, same device
    P = min(args.parity_queries, len(recs))
    per_q = []
    fixed_scores: dict[str, tuple[list[str], list[float]]] = {}
    t0 = time.perf_counter()
    for qi in range(P):
        rec = recs[qi]
        docs, pairs = union_pairs(rec)
        lib = [float(x) for x in np.asarray(ce.predict(pairs, batch_size=args.batch,
                                                        activation_fn=torch.nn.Identity(),
                                                        show_progress_bar=False)).reshape(-1)]
        f = fx.score(pairs)
        fixed_scores[rec["qid"]] = (docs, f)
        diffs = [abs(a - b) for a, b in zip(lib, f)]
        per_q.append({"qid": rec["qid"], "n_pairs": len(pairs), "max_abs_diff": round(max(diffs), 6),
                      "mean_abs_diff": round(float(np.mean(diffs)), 6),
                      "same_full_order": order(lib) == order(f),
                      "arms": arm_agreement(rec, docs, lib, f),
                      "logit_range": [round(min(lib), 3), round(max(lib), 3)]})
        _mps_empty_cache()
        print(f"  parity q{qi}: {per_q[-1]} | {_mps_memory()}", flush=True)
    report["A_parity_metal"] = {
        "library_path": f"CrossEncoder.predict(activation_fn=Identity, batch_size={args.batch}) on {args.device} {fx.dtype()}",
        "queries": P, "seconds": round(time.perf_counter() - t0, 1),
        "max_abs_diff": max(q["max_abs_diff"] for q in per_q),
        "mean_abs_diff": round(float(np.mean([q["mean_abs_diff"] for q in per_q])), 6),
        "queries_same_full_order": sum(q["same_full_order"] for q in per_q),
        "arm_top5_agreement": {a: sum(q["arms"][a]["top5"] for q in per_q) for a in sorted(RERANKED)},
        "arm_top10_agreement": {a: sum(q["arms"][a]["top10"] for q in per_q) for a in sorted(RERANKED)},
        "per_query": per_q,
    }
    print(f"A: {json.dumps({k: v for k, v in report['A_parity_metal'].items() if k != 'per_query'})}", flush=True)
    _mps_empty_cache()

    # ---- B. endurance: fixed-shape only, union per query, memory every query
    E = min(args.endurance_queries, len(recs))
    samples = []
    t0 = time.perf_counter()
    n_pairs = 0
    for qi in range(E):
        rec = recs[qi]
        docs, pairs = union_pairs(rec)
        tq = time.perf_counter()
        f = fx.score(pairs)
        dt = time.perf_counter() - tq
        n_pairs += len(pairs)
        if rec["qid"] not in fixed_scores:
            fixed_scores[rec["qid"]] = (docs, f)
        _mps_empty_cache()
        samples.append({"q": qi, "pairs": len(pairs), "s": round(dt, 2), **_mps_memory(), "rss_gb": rss_gb()})
        if qi % 10 == 0 or qi == E - 1:
            print(f"  endurance q{qi}: {len(pairs)} pairs in {dt:.1f}s | "
                  f"{n_pairs / (time.perf_counter() - t0):.2f} pairs/s cumulative | {samples[-1]}", flush=True)
    drv = [s.get("driver_gb") for s in samples if "driver_gb" in s]
    report["B_endurance_metal"] = {
        "queries": E, "pairs": n_pairs, "seconds": round(time.perf_counter() - t0, 1),
        "pairs_per_s_measured": round(n_pairs / max(1e-9, time.perf_counter() - t0), 3),
        "driver_gb_first": drv[0] if drv else None, "driver_gb_max": max(drv) if drv else None,
        "driver_gb_last": drv[-1] if drv else None,
        "driver_gb_by_50": {str(i): drv[i] for i in range(0, len(drv), 50)} if drv else None,
        "allocated_gb_max": max(s.get("allocated_gb", 0) for s in samples),
        "rss_gb_max": max(s["rss_gb"] for s in samples),
        "batches": fx.n_batches, "padded_rows": fx.n_padded_rows, "pairs_scored_total": fx.n_pairs,
        "samples": samples,
    }
    print(f"B: {json.dumps({k: v for k, v in report['B_endurance_metal'].items() if k != 'samples'})}", flush=True)

    # ---- C. fp32 CPU reference on a few queries (Metal model released first)
    del ce, fx
    gc.collect()
    _mps_empty_cache()
    C = min(args.cpu_queries, len(recs))
    if C:
        ce32 = CrossEncoder(args.rerank_model, max_length=RERANK_MAX_LENGTH, device="cpu")
        dtype32 = str(next(ce32[0].auto_model.parameters()).dtype)
        rows = []
        t0 = time.perf_counter()
        for qi in range(C):
            rec = recs[qi]
            docs, f = fixed_scores[rec["qid"]]
            pairs = [(questions[rec["qid"]], texts[d]) for d in docs]
            ref = [float(x) for x in np.asarray(ce32.predict(pairs, batch_size=args.batch,
                                                              activation_fn=torch.nn.Identity(),
                                                              show_progress_bar=False)).reshape(-1)]
            diffs = [abs(a - b) for a, b in zip(ref, f)]
            rows.append({"qid": rec["qid"], "n_pairs": len(pairs), "max_abs_diff": round(max(diffs), 6),
                         "mean_abs_diff": round(float(np.mean(diffs)), 6),
                         "same_full_order": order(ref) == order(f),
                         "arms": arm_agreement(rec, docs, ref, f)})
            print(f"  cpu fp32 q{qi}: {rows[-1]}", flush=True)
        report["C_cpu_fp32_reference"] = {
            "reference": f"CrossEncoder.predict(activation_fn=Identity) on cpu {dtype32}",
            "compared_to": f"fixed-shape on {args.device} {report['manifest']['dtype_actual']}",
            "queries": C, "seconds": round(time.perf_counter() - t0, 1),
            "pairs_per_s_cpu": round(sum(r["n_pairs"] for r in rows) / max(1e-9, time.perf_counter() - t0), 3),
            "max_abs_diff": max(r["max_abs_diff"] for r in rows),
            "mean_abs_diff": round(float(np.mean([r["mean_abs_diff"] for r in rows])), 6),
            "queries_same_full_order": sum(r["same_full_order"] for r in rows),
            "arm_top5_agreement": {a: sum(r["arms"][a]["top5"] for r in rows) for a in sorted(RERANKED)},
            "arm_top5_set_agreement": {a: sum(r["arms"][a]["top5_set"] for r in rows) for a in sorted(RERANKED)},
            "per_query": rows,
        }
        print(f"C: {json.dumps({k: v for k, v in report['C_cpu_fp32_reference'].items() if k != 'per_query'})}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
