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
    complete_evidence, compose_latency, decide, paired_bootstrap, recall_at_k,
    rrf_fuse, tokenize, union_backfill, protocol_sha256,
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
