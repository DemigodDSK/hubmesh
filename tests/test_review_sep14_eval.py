"""Regression tests for the 2026-09-14 external review (batch 5) —
benchmark-side findings (8–10): the structural-only ablation arm, the
LightRAG scorer's roster handling, and manifest identity. Library-side
findings are in `tests/test_review_sep14.py`.
"""
from __future__ import annotations
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hubmesh import Document
from hubmesh.adapters.inmemory import InMemoryStore
from hubmesh.kg_llm import build_entity_kg_llm
from hubmesh.planner import Planner
from hubmesh.ppr import PPRSolver

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))

TRIPLE = '{"triples": [["Alice", "knows", "Bob"]]}'


class NoEntities:
    def __call__(self, text):
        return SimpleNamespace(ents=[])


class AliceOnly:
    def __call__(self, text):
        return SimpleNamespace(ents=[SimpleNamespace(text="Alice", label_="PERSON")])


def vec2(x, y):
    return np.array([x, y], dtype=np.float32)


# ---- 8. structural-only ablation policy -------------------------------------

class TestStructuralOnlyPolicy:
    def test_seedless_query_gets_neutral_score_same_packer_and_is_counted(self):
        """Reviewer's probe: a document-only KG, no recognised entities.
        hubmesh returns the document (pure cosine); the structural arm
        must not substitute an empty-result policy — it applies the
        neutral score, the id tie rule and the same packer, and counts
        the query as seedless."""
        from structural_only import structural_only_retrieve
        doc = Document("plain", "ordinary text", vec2(1., 0.))
        store = InMemoryStore([doc])
        kg = build_entity_kg_llm([doc], llm=lambda _: '{"triples": []}')
        planner = Planner(store, embed=lambda _: vec2(1., 0.), kg=kg, nlp=NoEntities())
        assert [s.doc.id for s in planner.retrieve("query", top_k=1).sources] == ["plain"]
        stats: dict = {}
        got = structural_only_retrieve(kg, PPRSolver(kg.graph), NoEntities(),
                                       "query", vec2(1., 0.), store, 1, stats=stats)
        assert got == ["plain"]
        assert stats["seedless"] == 1 and stats["fallback"] == 1 and stats["n_seeds"] == [0]

    def test_seedless_tie_rule_is_document_id_order(self):
        from structural_only import structural_only_retrieve
        docs = [Document("zeta", "z", vec2(1., 0.)), Document("alpha", "a", vec2(0., 1.)),
                Document("mid", "m", vec2(1., 1.) / np.sqrt(2))]
        store = InMemoryStore(docs)
        kg = build_entity_kg_llm(docs, llm=lambda _: '{"triples": []}')
        got = structural_only_retrieve(kg, PPRSolver(kg.graph), NoEntities(),
                                       "query", vec2(1., 0.), store, 2)
        assert got == ["alpha", "mid"]

    def test_post_selection_goes_through_the_same_packer(self):
        from structural_only import structural_only_retrieve
        docs = [Document("d1", "Alice knows Bob", vec2(1., 0.)),
                Document("d2", "Alice knows Bob", vec2(0., 1.)),
                Document("d3", "Alice knows Bob", vec2(1., 1.) / np.sqrt(2))]
        store = InMemoryStore(docs)
        kg = build_entity_kg_llm(docs, llm=lambda _: TRIPLE)
        solver = PPRSolver(kg.graph)
        stats: dict = {}
        wide = structural_only_retrieve(kg, solver, AliceOnly(), "Alice", vec2(1., 0.),
                                        store, 3, budget_tokens=10_000, stats=stats)
        tight = structural_only_retrieve(kg, solver, AliceOnly(), "Alice", vec2(1., 0.),
                                         store, 3, budget_tokens=5, stats=stats)
        assert len(wide) == 3 and len(tight) == 1          # budget enforced, like the planner
        assert stats["seedless"] == 0 and stats["fallback"] == 0
        assert stats["n_seeds"] == [1, 1]


# ---- 9. LightRAG scorer: interrupted runs ----------------------------------

def _load_scorer(root: Path):
    spec = importlib.util.spec_from_file_location(
        "score_eval", ROOT / "benchmarks" / "lightrag_compare" / "score_eval.py")
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    scorer.HERE = root
    scorer.build_manifest = lambda **kw: kw
    return scorer


def _write_eval_fixture(root: Path, raw_rows: list[dict], n_questions: int = 3):
    (root / "manifest_pilot.json").write_text(
        json.dumps([{"title": "doc", "filename": "doc.txt"}]))
    (root / "questions_pilot.json").write_text(
        json.dumps([{"qid": str(i), "gold_titles": ["doc"]} for i in range(n_questions)]))
    (root / "raw_lightrag_pilot.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in raw_rows))


HIT = {"response": {"chunks": [{"doc_id": "doc"}]}}


class TestScoreEvalRoster:
    def test_reviewer_repro_missing_queries_count(self, tmp_path, monkeypatch):
        _write_eval_fixture(tmp_path, [{"pass": "main", "qid": "0", **HIT}])
        scorer = _load_scorer(tmp_path)
        monkeypatch.setattr(sys, "argv", ["score_eval"])
        scorer.main()
        main = json.loads((tmp_path / "scored_pilot.json").read_text())["passes"]["main"]
        assert (main["n_total"], main["n_completed"], main["n_missing"]) == (3, 1, 2)
        assert main["completion_rate"] == pytest.approx(1 / 3, abs=1e-4)
        assert main["strict_missing_and_failures_as_zero"]["recall@10"] == pytest.approx(1 / 3, abs=1e-4)
        assert main["completed_only"]["recall@10"] == 1.0

    def test_unexpected_ids_are_reported_not_scored(self, tmp_path, monkeypatch):
        _write_eval_fixture(tmp_path, [{"pass": "main", "qid": "0", **HIT},
                                       {"pass": "main", "qid": "zzz", **HIT}])
        scorer = _load_scorer(tmp_path)
        monkeypatch.setattr(sys, "argv", ["score_eval"])
        scorer.main()
        main = json.loads((tmp_path / "scored_pilot.json").read_text())["passes"]["main"]
        assert main["n_unexpected"] == 1 and main["unexpected_qids"] == ["zzz"]
        assert main["n_completed"] == 1

    def test_cold_roster_file_is_honoured(self, tmp_path, monkeypatch):
        _write_eval_fixture(tmp_path, [{"pass": "main", "qid": str(i), **HIT} for i in range(3)]
                            + [{"pass": "cold2", "qid": "0", **HIT}])
        (tmp_path / "roster_pilot_cold.json").write_text(json.dumps(["0", "1"]))
        scorer = _load_scorer(tmp_path)
        monkeypatch.setattr(sys, "argv", ["score_eval"])
        scorer.main()
        passes = json.loads((tmp_path / "scored_pilot.json").read_text())["passes"]
        assert passes["cold2"]["roster_source"] == "roster_pilot_cold.json"
        assert (passes["cold2"]["n_total"], passes["cold2"]["n_missing"]) == (2, 1)
        assert passes["main"]["n_missing"] == 0

    def test_observed_fallback_is_flagged(self, tmp_path, monkeypatch):
        _write_eval_fixture(tmp_path, [{"pass": "main", "qid": "0", **HIT},
                                       {"pass": "cold2", "qid": "0", **HIT}])
        scorer = _load_scorer(tmp_path)
        monkeypatch.setattr(sys, "argv", ["score_eval"])
        scorer.main()
        cold = json.loads((tmp_path / "scored_pilot.json").read_text())["passes"]["cold2"]
        assert "overstated" in cold["roster_source"]


# ---- manifest identity -------------------------------------------------------

class TestManifestIdentity:
    def test_hashes_present_and_stable(self):
        from manifest import build_manifest
        a = build_manifest(harness=__file__, embed_device="cpu", embed_batch_size=16)
        b = build_manifest(harness=__file__, embed_device="cpu", embed_batch_size=16)
        for key in ("src_sha256", "benchmarks_sha256"):
            assert len(a[key]) == 64 and a[key] == b[key]
        assert a["harness_sha256"] and a["harness"].endswith("test_review_sep14_eval.py")
        assert isinstance(a["git_dirty_paths"], list)
        assert (a["embed_device"], a["embed_batch_size"]) == ("cpu", 16)

    def test_tree_hash_tracks_content(self, tmp_path):
        from manifest import _tree_sha256
        f1, f2 = tmp_path / "a.py", tmp_path / "b.py"
        f1.write_text("x = 1"); f2.write_text("y = 2")
        h0 = _tree_sha256([f1, f2], tmp_path)
        assert _tree_sha256([f2, f1], tmp_path) == h0          # order-independent
        f2.write_text("y = 3")
        assert _tree_sha256([f1, f2], tmp_path) != h0
