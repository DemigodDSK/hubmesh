"""Unit tests for the pure functions of the complementarity protocol
runner (benchmarks/run_complementarity.py) — fusion, backfill, the
complete-evidence metric, the decision rule, per-query latency
composition — and the 2Wiki loader's identity/gold rules on a synthetic
dev file. No models, no downloads.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))

from run_complementarity import (  # noqa: E402
    ARMS, REF_N, arm_stage_times, complete_evidence, compose_latency, decide,
    extend_reference_arms, paired_bootstrap, plan_fixed_shape_batches, protocol_sha256,
    recall_at_k, refresh_gold, rrf_fuse, tokenize, union_backfill, validate_stage,
)


class TestFusionAndBackfill:
    def test_rrf_prefers_docs_high_in_both_lists(self):
        fused = rrf_fuse([["a", "b", "c"], ["b", "a", "d"]])
        assert fused[:2] == ["a", "b"] or fused[:2] == ["b", "a"]
        assert set(fused) == {"a", "b", "c", "d"}

    def test_rrf_tie_breaks_by_id(self):
        assert rrf_fuse([["b"], ["a"]]) == ["a", "b"]

    def test_union_backfill_exact_count_and_order(self):
        base = [f"h{i}" for i in range(60)]
        extra = ["h2", "x1", "h40", "x2"] + [f"y{i}" for i in range(30)]
        out = union_backfill(base, extra)
        assert len(out) == 50 and len(set(out)) == 50
        assert out[:30] == base[:30]                       # base first
        assert out[30:33] == ["x1", "h40", "x2"]           # extra in its order, "h2" skipped
        assert out[33:49] == [f"y{i}" for i in range(16)]  # 20 extra minus the duplicate
        assert out[49] == "h30"                            # backfilled from base[30:], h40 skipped

    def test_union_backfill_small_corpus_returns_fewer(self):
        assert union_backfill(["a", "b"], ["b", "c"]) == ["a", "b", "c"]


class TestMetrics:
    def test_complete_evidence_requires_every_gold(self):
        assert complete_evidence(["g1", "x", "g2"], ["g1", "g2"], 5) == 1.0
        assert complete_evidence(["g1", "x", "y", "z", "w", "g2"], ["g1", "g2"], 5) == 0.0
        assert complete_evidence(["g1"], ["g1", "g2"], 5) == 0.0
        assert complete_evidence([], ["g1"], 5) == 0.0

    def test_complete_evidence_uses_distinct_ids(self):
        assert complete_evidence(["g1", "g1", "g2"], ["g1", "g2", "g2"], 3) == 1.0

    def test_missing_gold_counts_as_failure_not_removed(self):
        # gold absent from any retrievable list can never be satisfied
        assert complete_evidence(["a", "b"], ["a", "missing"], 5) == 0.0
        assert recall_at_k(["a", "b"], ["a", "missing"], 5) == 0.5

    def test_tokenizer(self):
        assert tokenize("Hello, World! It's 2026.") == ["hello", "world", "it", "s", "2026"]


class TestDecisionRule:
    def test_pass_requires_threshold_and_ci_above_zero(self):
        assert decide({"mean": 3.4, "lo": 1.1, "hi": 5.7}) == "pass"
        assert decide({"mean": 3.4, "lo": -0.2, "hi": 7.0}) == "inconclusive"
        assert decide({"mean": 2.9, "lo": 1.0, "hi": 4.8}) == "inconclusive"

    def test_no_useful_gain_when_upper_bound_below_threshold(self):
        assert decide({"mean": 1.2, "lo": 0.3, "hi": 2.1}) == "no useful gain"   # real but small
        assert decide({"mean": -0.5, "lo": -2.0, "hi": 1.0}) == "no useful gain"

    def test_none_is_inconclusive(self):
        assert decide(None) == "inconclusive"

    def test_paired_bootstrap_interval_contains_mean(self):
        a = [1.0] * 60 + [0.0] * 40
        b = [1.0] * 50 + [0.0] * 50
        c = paired_bootstrap(a, b, n_boot=2000, seed=0)
        assert c["mean"] == pytest.approx(10.0) and c["lo"] <= 10.0 <= c["hi"] and c["n"] == 100


class TestLatencyComposition:
    def test_per_query_sums_before_percentiles(self):
        # two stages whose p95s are on DIFFERENT queries: adding stage p95s
        # would overstate; per-query sums must not.
        times = [{"a": 0.100, "b": 0.001}, {"a": 0.001, "b": 0.100}] + \
                [{"a": 0.001, "b": 0.001}] * 98
        both = compose_latency(times, ("a", "b"))
        assert both["p95_ms"] < 200 - 1e-6                # not 100 + 100
        assert both["p95_ms"] >= compose_latency(times, ("a",))["p95_ms"]

    def test_missing_stage_counts_zero(self):
        assert compose_latency([{"a": 0.010}], ("a", "rerank"))["median_ms"] == pytest.approx(10.0)

    def test_protocol_file_present_and_hashed(self):
        h = protocol_sha256()
        assert h and len(h) == 64


class TestWiki2Loader:
    def test_exact_identity_and_gold(self, tmp_path):
        from wiki2_loader import load_wiki2, retrievable_gold
        rows = []
        for i in range(3):
            ctx = [[f"T{i}a", ["Sentence one.", "Sentence two."]],
                   ["Shared", [f"distinct text {i}"]],
                   ["Shared", ["same text everywhere"]]]
            rows.append({"_id": f"q{i}", "question": f"Q{i}?", "answer": "x",
                         "type": "bridge-comparison" if i else "comparison",
                         "context": ctx,
                         "supporting_facts": [[f"T{i}a", 0], ["Shared", 0]]})
        dev = tmp_path / "dev.json"
        dev.write_text(json.dumps(rows))
        examples, pool, titles = load_wiki2(n_questions=3, seed=0, dev_path=dev)
        assert len(examples) == 3
        # "same text everywhere" pools to ONE id; the distinct texts stay separate
        same = [pid for pid, t in pool.items() if t == "same text everywhere"]
        assert len(same) == 1 and titles[same[0]] == "Shared"
        assert sum(1 for t in titles.values() if t == "Shared") == 4
        for ex in examples:
            assert len(ex.gold_ids) == 2 and all(g.split("::")[0] in ("Shared", ex.gold_ids[0].split("::")[0])
                                               for g in ex.gold_ids)
            assert retrievable_gold(ex, set(pool)) == ex.gold_ids
        types = sorted(ex.qtype for ex in examples)
        assert types == ["bridge_comparison", "bridge_comparison", "comparison"]

    def test_supporting_title_maps_to_first_paragraph_with_that_title(self, tmp_path):
        from wiki2_loader import load_wiki2
        rows = [{"_id": "q", "question": "?", "answer": "", "type": "inference",
                 "context": [["A", ["a1"]], ["A", ["a2"]], ["B", ["b"]]],
                 "supporting_facts": [["A", 0], ["B", 0]]}]
        dev = tmp_path / "dev.json"
        dev.write_text(json.dumps(rows))
        examples, pool, titles = load_wiki2(n_questions=1, seed=0, dev_path=dev)
        gold = examples[0].gold_ids
        assert [titles[g] for g in gold] == ["A", "B"]
        assert pool[gold[0]] == "a1"

    def test_unresolved_supporting_title_is_kept_as_failure(self, tmp_path):
        # Amendment 2: a supporting fact whose title has no context paragraph
        # must not silently disappear from the gold list.
        from wiki2_loader import load_wiki2, retrievable_gold
        rows = [{"_id": "q", "question": "?", "answer": "", "type": "inference",
                 "context": [["A", ["a1"]], ["B", ["b"]]],
                 "supporting_facts": [["A", 0], ["Missing", 0]]}]
        dev = tmp_path / "dev.json"
        dev.write_text(json.dumps(rows))
        examples, pool, _ = load_wiki2(n_questions=1, seed=0, dev_path=dev)
        gold = examples[0].gold_ids
        assert len(gold) == 2 and gold[1] == "Missing::unresolved" and gold[1] not in pool
        assert retrievable_gold(examples[0], set(pool)) == gold[:1]
        assert complete_evidence(list(pool), gold, 10) == 0.0      # can never be satisfied


# ---- Amendment 2 ------------------------------------------------------------

def _stage(n=3, ref=5):
    """Synthetic stage file: pool p0..p199; per query the lists are
    rotations so every arm relation the validator checks holds."""
    pool = [f"p{i}" for i in range(200)]
    recs = []
    for q in range(n):
        dense = pool[q:] + pool[:q]
        lex = pool[::-1]
        hybrid = rrf_fuse([dense[:200], lex[:200]])
        hub = pool[50 + q:50 + q + 40]                              # 40 candidates: budget bound
        recs.append({"qid": f"q{q}", "gold": [dense[0], dense[1]], "n_gold": 2, "n_seeds": 2,
                     "overlap_hybrid30_hub20": 0, "gold_docs_without_entities": 0,
                     "stage_times": {"embed": 0.01, "dense": 0.001, "bm25": 0.002, "rrf": 0.0005,
                                     "hubmesh": 0.1},
                     "cand": {"A1": dense[:ref], "A2": lex[:ref], "A3": hybrid[:ref],
                              "A4": hybrid[:50], "A5": hub[:ref],
                              "A6": union_backfill(hybrid, hub), "A7": union_backfill(dense, hub),
                              "A8": hub[:50]}})
    return {"embed_model": "m", "records": recs}, pool


class TestFixedShapePlan:
    def test_every_pair_once_constant_batches_and_bucket_fits(self):
        lengths = [20, 130, 512, 256, 257, 5, 384, 300, 300, 300, 300, 300, 300, 300, 300, 300]
        plan = plan_fixed_shape_batches(lengths, (128, 256, 384, 512), batch=8)
        real = [i for b, idx, n in plan for i in idx[:n]]
        assert sorted(real) == list(range(len(lengths)))              # each index exactly once
        for bucket, idx, n in plan:
            assert len(idx) == 8 and 1 <= n <= 8
            assert all(lengths[i] <= bucket for i in idx)              # never truncated by the bucket
            assert idx[n:] == [idx[n - 1]] * (8 - n)                   # padding = duplicate of the last real row
        # 9 pairs in (256, 384] -> two batches of that shape; 5-token pair in the 128 bucket
        assert sum(1 for b, _, _ in plan if b == 384) == 2
        assert any(b == 128 and 0 in idx for b, idx, _ in plan)

    def test_lengths_beyond_the_last_bucket_are_an_error(self):
        with pytest.raises(ValueError):
            plan_fixed_shape_batches([10, 513], (128, 512), batch=4)
        with pytest.raises(ValueError):
            plan_fixed_shape_batches([10], (512, 128), batch=4)


class TestStageValidation:
    def test_valid_stage_passes_and_reports(self):
        stage, pool = _stage()
        rep = validate_stage(stage, {"dataset": "x", "n": 3, "seed": 0, "embed_model": "m"},
                             ["q0", "q1", "q2"], pool)
        assert rep["A4_A6_A7_exactly_50_distinct"] and rep["A8_queries_below_50"] == 3
        assert rep["checked"]["embed_model"] == "m" and "absent" in rep["checked"]["dataset"]

    def test_identity_mismatches_are_rejected(self):
        stage, pool = _stage()
        exp = {"dataset": "x", "n": 3, "seed": 0, "embed_model": "m"}
        with pytest.raises(ValueError, match="qid list"):
            validate_stage(stage, exp, ["q1", "q0", "q2"], pool)
        with pytest.raises(ValueError, match="embed_model"):
            validate_stage(stage, {**exp, "embed_model": "other"}, ["q0", "q1", "q2"], pool)
        stage["dataset"], stage["n"] = "x", 4
        with pytest.raises(ValueError, match="n: stage file 4"):
            validate_stage(stage, exp, ["q0", "q1", "q2"], pool)

    def test_candidate_defects_are_rejected(self):
        stage, pool = _stage()
        exp = {"dataset": "x", "n": 3, "seed": 0, "embed_model": "m"}
        bad = json.loads(json.dumps(stage))
        bad["records"][0]["cand"]["A4"] = bad["records"][0]["cand"]["A4"][:49]
        with pytest.raises(ValueError, match="A4: 1 queries"):
            validate_stage(bad, exp, ["q0", "q1", "q2"], pool)
        bad = json.loads(json.dumps(stage))
        bad["records"][1]["cand"]["A6"][3] = "not-in-pool"
        with pytest.raises(ValueError, match="not in the pool"):
            validate_stage(bad, exp, ["q0", "q1", "q2"], pool)
        bad = json.loads(json.dumps(stage))
        bad["records"][2]["cand"]["A1"] = bad["records"][2]["cand"]["A1"][::-1]
        with pytest.raises(ValueError, match="A1 is not a prefix of A7"):
            validate_stage(bad, exp, ["q0", "q1", "q2"], pool)


class TestReferenceArmExtension:
    def test_extends_from_prefixes_and_bm25_with_prefix_checks(self):
        stage, pool = _stage(ref=5)
        recs = stage["records"]
        bm25 = {r["qid"]: pool[::-1] for r in recs}                    # the lex ranking used above
        n = extend_reference_arms(recs, bm25)
        assert n == 4 * len(recs)
        for r in recs:
            for a in ("A1", "A2", "A3", "A5"):
                assert len(r["cand"][a]) == REF_N
            assert r["cand"]["A1"] == r["cand"]["A7"][:REF_N]
            assert r["cand"]["A3"] == r["cand"]["A4"][:REF_N]
            assert r["cand"]["A5"] == r["cand"]["A8"][:REF_N]
            assert r["cand"]["A2"] == pool[::-1][:REF_N]
        assert extend_reference_arms(recs, bm25) == 0                  # idempotent

    def test_prefix_mismatch_rejects_the_file(self):
        stage, pool = _stage(ref=5)
        recs = stage["records"]
        with pytest.raises(ValueError, match="BM25"):
            extend_reference_arms(recs, {r["qid"]: pool for r in recs})   # wrong ranking
        stage, pool = _stage(ref=5)
        stage["records"][0]["cand"]["A5"][0] = "p199"
        with pytest.raises(ValueError, match="A5 is not a prefix of A8"):
            extend_reference_arms(stage["records"], {r["qid"]: pool[::-1] for r in stage["records"]})


class TestGoldRefresh:
    def test_full_gold_replaces_retrievable_and_counts_unresolved(self):
        stage, pool = _stage()
        recs = stage["records"]
        gold_of = {r["qid"]: list(r["gold"]) for r in recs}
        gold_of["q1"] = gold_of["q1"] + ["ghost::unresolved"]
        rep = refresh_gold(recs, gold_of, pool)
        assert rep == {"gold_ids_unresolved": 1, "queries_with_unresolved_gold": 1}
        r1 = recs[1]
        assert r1["n_gold"] == 3 and r1["n_gold_unresolved"] == 1
        assert complete_evidence(r1["cand"]["A4"], r1["gold"], 50) == 0.0      # failure, not dropped
        assert recall_at_k(r1["cand"]["A4"], r1["gold"], 50) == pytest.approx(2 / 3)
        assert refresh_gold(recs, gold_of, pool) == rep                          # accepts the full list too

    def test_inconsistent_stored_gold_rejects_the_file(self):
        stage, pool = _stage()
        gold_of = {r["qid"]: ["p5", "p6"] for r in stage["records"]}
        with pytest.raises(ValueError, match="neither the full gold"):
            refresh_gold(stage["records"], gold_of, pool)


class TestPerArmLatency:
    def test_reranked_arms_use_their_own_time_or_are_excluded(self):
        rec = {"stage_times": {"embed": 0.01, "dense": 0.001, "bm25": 0.002, "rrf": 0.0005, "hubmesh": 0.1,
                               "rerank_by_arm": {"A4": 2.0, "A6": 2.5, "A7": 2.4, "A8": 1.6}}}
        assert arm_stage_times(rec, "A4")["rerank"] == 2.0
        assert arm_stage_times(rec, "A8")["rerank"] == 1.6
        assert arm_stage_times(rec, "A1")["embed"] == 0.01 and "rerank" not in arm_stage_times(rec, "A1")
        untimed = {"stage_times": {"embed": 0.01, "hubmesh": 0.1}}
        assert arm_stage_times(untimed, "A6") is None                    # outside the timing subsample
        assert arm_stage_times(untimed, "A5") is not None
        assert set(ARMS) >= {"A4", "A6", "A7", "A8"}
