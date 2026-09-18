"""The single, pre-planned examination of the complementarity result files
(protocol: results are examined once, after all arms complete). Prints
everything the write-up needs in one pass — registered outcome first —
so nothing has to be re-read selectively afterwards. Reads .json or
.json.gz. Usage:

  python benchmarks/examine_complementarity.py benchmarks/results
"""
from __future__ import annotations
import gzip
import json
import sys
from pathlib import Path

R = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent / "results")
CONFIGS = [("2wiki", 1000, "bge-m3"), ("2wiki", 1000, "minilm"), ("hotpotqa", 500, "bge-m3"),
           ("musique", 300, "bge-m3"), ("hotpotqa", 500, "minilm"), ("musique", 300, "minilm")]
ARMS = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"]


def load(stem: str) -> dict | None:
    for p in (R / f"{stem}.json", R / f"{stem}.json.gz"):
        if p.exists():
            with (gzip.open(p, "rt") if p.suffix == ".gz" else open(p)) as f:
                return json.load(f)
    return None


def ci(c) -> str:
    return "—" if c is None else f"{c['mean']:+.2f} [{c['lo']:+.2f}, {c['hi']:+.2f}] n={c['n']}"


for ds, n, tag in CONFIGS:
    o = load(f"complementarity_{ds}_n{n}_{tag}")
    if o is None:
        print(f"\n##### {ds} n={n} {tag}: MISSING")
        continue
    m = o["manifest"]
    val = m.get("stage_validation") or {}
    print(f"\n##### {ds} n={n} {tag}  (commit {(m.get('git_commit') or 'NONE')[:7]} dirty={m.get('git_dirty')} "
          f"git_error={m.get('git_error')} embed {m['embed_device']}/{m['embed_dtype']} "
          f"rerank {m['rerank_device']}/{m['rerank_dtype_actual']} rev={str(m.get('rerank_model_revision'))[:8]} "
          f"timing_q={m['rerank_timing_queries']} parity_max={m['rerank_smoke_parity']['max_abs_diff']} "
          f"comp_diff_max={m['rerank_batch_composition_abs_diff']['max']} cache hits={m['rerank_cache']['hits']} "
          f"pairs={m['rerank_pairs_scored']} mem_driver_last={m['rerank_memory_samples'][-1].get('driver_gb')} "
          f"hashes={val.get('checked', {}).get('qids_sha256')}/{val.get('checked', {}).get('pool_sha256')} "
          f"protocol_changed={val.get('checked', {}).get('protocol_changed_since_stage_r')})")
    pr = o["primary"]
    print(f"PRIMARY A6-A4 ce@5: {ci(pr['result'])} -> {pr['decision']}  (primary configuration: {pr['is_primary_configuration']})")
    g = pr["regression_guard"]
    print(f"GUARD {g['slice']}: n={g['n']} A6-A4 ce@5 {ci(g['A6-A4 ce@5'])} rule {g['rule']} met={g['met']}")
    lat = o["latency_composed_estimate"]
    print(f"LATENCY p95 ratio A6/A4 {lat['p95_ratio_A6_over_A4']} rule<=1.5 met={lat['rule_met_le_1.5']} | "
          f"stage medians ms {lat['stage_medians_ms']} | rerank median ms by arm {lat['rerank_median_ms_by_arm']}")
    print("ARM        ce@5    ce@10   r@5     r@10   | composed median / p95 ms (n)")
    for a in ARMS:
        r = o["results"]["all"][a]
        la = lat["arms"][a]
        print(f"{a:<4} {r['ce@5']:7.4f} {r['ce@10']:7.4f} {r['r@5']:7.4f} {r['r@10']:7.4f}  | "
              f"{la['median_ms']} / {la['p95_ms']} ({la['n_queries']})")
    print("CONTRASTS (all questions), ce@5 / ce@10 / r@5 / r@10:")
    for name, c in o["contrasts"]["all"].items():
        print(f"  {name:<6} {ci(c['ce@5'])} | {ci(c['ce@10'])} | {ci(c['r@5'])} | {ci(c['r@10'])}")
    for sl, rows in o["results"].items():
        if sl == "all":
            continue
        c = o["contrasts"][sl]["A6-A4"]["ce@5"]
        print(f"SLICE {sl}: n={c['n'] if c else 0} A6-A4 ce@5 {ci(c)} | A4 ce@5 {rows['A4']['ce@5']} A6 ce@5 {rows['A6']['ce@5']} "
              f"A5 ce@5 {rows['A5']['ce@5']} A3 ce@5 {rows['A3']['ce@5']}")
    d = o["diagnostics"]
    print(f"DIAGNOSTICS: ge2 seeds {d['frac_queries_with_ge2_seeds']} | overlap {d['mean_overlap_hybrid30_hub20']} | "
          f"gold w/o entities {d['frac_gold_docs_without_entities']} | unresolved gold {d['gold_ids_unresolved']} | "
          f"candidate counts {d['candidate_counts']}")
    recs = o["per_query"]
    added_gold = kept5 = 0
    for r in recs:
        hyb30 = set(r["cand"]["A4"][:30])
        hub20 = r["cand"]["A8"][:20]
        new_gold = [g for g in r["gold"] if g not in hyb30 and g in hub20]
        if new_gold:
            added_gold += 1
            if all(g in r["final"]["A6"][:5] for g in new_gold):
                kept5 += 1
    wins = sum(1 for r in recs if r["metrics"]["A6"]["ce@5"] > r["metrics"]["A4"]["ce@5"])
    losses = sum(1 for r in recs if r["metrics"]["A6"]["ce@5"] < r["metrics"]["A4"]["ce@5"])
    print(f"MECHANISM (pre-planned): queries where hubmesh top-20 held a gold paragraph absent from hybrid top-30: "
          f"{added_gold}/{len(recs)}; reranker kept all such gold in A6 top-5: {kept5}; "
          f"A6-A4 ce@5 discordant: {wins} wins / {losses} losses")
