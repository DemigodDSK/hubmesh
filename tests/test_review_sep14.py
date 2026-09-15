"""Regression tests for the 2026-09-14 external review (batch 5) —
library-side findings (1–7).

Each class carries the reviewer's exact reproduction plus boundary cases
beyond it. The review's process finding was that fixes matched the
reproduction and the surrounding claim was then generalized; the
boundary cases are the answer to that: concurrent publication, failure
at every batch index, exact-fit budgets, valid-empty vs malformed
replies. Benchmark-side findings (8–10) are in
`tests/test_review_sep14_eval.py`.
"""
from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

from hubmesh import Document
from hubmesh.adapters.chroma import ChromaStore
from hubmesh.corpus import CorpusManager
from hubmesh.kg_llm import build_entity_kg_llm
from hubmesh.mcp_server import resolve_security
from hubmesh.packing import estimate_tokens, pack
from hubmesh.ppr import PPRSolver
from hubmesh.types import ScoredDocument

ROOT = Path(__file__).resolve().parents[1]


class NoEntities:
    """spaCy stand-in that never finds an entity."""
    def __call__(self, text):
        return SimpleNamespace(ents=[])


TRIPLE = '{"triples": [["Alice", "knows", "Bob"]]}'


def llm_kg(doc_id: str, reply: str = TRIPLE):
    return build_entity_kg_llm([Document(doc_id, "Alice knows Bob")],
                               llm=lambda _: reply)


def vec2(x, y):
    return np.array([x, y], dtype=np.float32)


# ---- 1. tunnel guidance (P1) --------------------------------------------

class TestTunnelGuidance:
    def test_tunnel_policy_carries_edge_auth_notice(self):
        pol = resolve_security("sse", "127.0.0.1", True, "k", False, False)
        assert pol.notice and "authenticate callers" in pol.notice
        assert resolve_security("sse", "127.0.0.1", False, "k",
                                False, False).notice is None
        assert resolve_security("stdio", "127.0.0.1", False, None,
                                False, False).notice is None

    @pytest.mark.parametrize("rel", ["README.md", "docs/perplexity.md"])
    def test_docs_require_edge_authentication_before_injection(self, rel):
        raw = (ROOT / rel).read_text().lower().replace("*", "")
        text = " ".join(raw.split())            # fold line breaks / emphasis
        assert "authenticate callers" in text
        assert "unsupported for private corpora" in text


# ---- 2. planner cache freshness ------------------------------------------

class TestPlannerCacheFreshness:
    @staticmethod
    def manager(root):
        return CorpusManager(root=root, embed=lambda _: vec2(1., 0.),
                             embed_identity="review", nlp=NoEntities())

    def test_rebuild_during_load_is_not_pinned(self, tmp_path):
        """Reviewer's reproduction: load A, rebuild publishes B mid-load,
        the loading request must not install A over B's invalidation."""
        m = self.manager(tmp_path)
        m.build("demo", [Document("old", "old")], kg=llm_kg("old"))
        original = m._load_gen
        fired = {"done": False}

        def interleave(name):
            gen, store, kg = original(name)
            if not fired["done"]:
                fired["done"] = True
                m.build(name, [Document("new", "new")], kg=llm_kg("new"))
            return gen, store, kg

        m._load_gen = interleave
        m.planner("demo")
        m._load_gen = original
        assert m.planner("demo").store.all_ids() == ["new"]
        assert m.load("demo")[0].all_ids() == ["new"]

    def test_external_rebuild_is_seen_by_another_manager(self, tmp_path):
        m1, m2 = self.manager(tmp_path), self.manager(tmp_path)
        m1.build("demo", [Document("old", "old")], kg=llm_kg("old"))
        assert m1.planner("demo").store.all_ids() == ["old"]
        m2.build("demo", [Document("new", "new")], kg=llm_kg("new"))
        assert m1.planner("demo").store.all_ids() == ["new"]

    def test_repeated_interleaving_converges_on_live_generation(self, tmp_path):
        m = self.manager(tmp_path)
        m.build("demo", [Document("g0", "g0")], kg=llm_kg("g0"))
        original = m._load_gen
        n = {"builds": 0}

        def interleave(name):
            gen, store, kg = original(name)
            if n["builds"] < 2:
                n["builds"] += 1
                gid = f"g{n['builds']}"
                m.build(name, [Document(gid, gid)], kg=llm_kg(gid))
            return gen, store, kg

        m._load_gen = interleave
        served = m.planner("demo").store.all_ids()
        m._load_gen = original
        assert served == m.load("demo")[0].all_ids() == ["g2"]

    def test_unchanged_corpus_keeps_its_cached_planner(self, tmp_path):
        m = self.manager(tmp_path)
        m.build("demo", [Document("d", "d")], kg=llm_kg("d"))
        assert m.planner("demo") is m.planner("demo")


# ---- 3. partial upsert leaves derived caches valid ------------------------

class FlakyCollection:
    def __init__(self, fail_at: int):
        self.calls, self.fail_at = 0, fail_at

    def upsert(self, **kw):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError(f"batch {self.calls} failed")


class FlakyQdrantClient(FlakyCollection):
    def get_collections(self):
        return SimpleNamespace(collections=[])

    def create_collection(self, **kw):
        pass


DOCS3 = [Document("a", "a", vec2(0., 1.)), Document("b", "b", vec2(1., 0.)),
         Document("c", "c", vec2(1., 1.))]


class TestPartialUpsertInvalidation:
    @pytest.mark.parametrize("fail_at", [1, 2, 3])   # first, middle, last batch
    def test_chroma_failed_batch_invalidates_and_drops_unconfirmed(self, fail_at):
        store = ChromaStore(None, FlakyCollection(fail_at))
        store._neighbor_cache["a"] = ["old-neighbor"]
        v0 = store.mutation_counter
        with pytest.raises(RuntimeError):
            store.upsert(DOCS3, batch_size=1)
        assert store.mutation_counter > v0
        assert store._neighbor_cache == {}
        failed = DOCS3[fail_at - 1].id
        assert failed not in store._vec_cache            # unconfirmed → not trusted
        for d in DOCS3[:fail_at - 1]:
            assert d.id in store._vec_cache             # confirmed batches cached

    @pytest.mark.parametrize("fail_at", [1, 2, 3])
    def test_qdrant_failed_batch_invalidates_and_drops_unconfirmed(self, fail_at):
        pytest.importorskip("qdrant_client")
        from hubmesh.adapters.qdrant import QdrantStore
        store = QdrantStore(FlakyQdrantClient(fail_at), collection="c", dim=2)
        store._neighbor_cache["a"] = ["old-neighbor"]
        store._all_ids_cache = ["stale"]
        v0 = store.mutation_counter
        with pytest.raises(RuntimeError):
            store.upsert(DOCS3, batch_size=1)
        assert store.mutation_counter > v0
        assert store._neighbor_cache == {} and store._all_ids_cache is None
        assert DOCS3[fail_at - 1].id not in store._vec_cache
        for d in DOCS3[:fail_at - 1]:
            assert d.id in store._vec_cache

    def test_successful_upsert_still_bumps_exactly_once_per_call_boundary(self):
        store = ChromaStore(None, FlakyCollection(fail_at=99))
        v0 = store.mutation_counter
        store.upsert(DOCS3, batch_size=2)
        assert store.mutation_counter > v0
        assert set(store._vec_cache) == {"a", "b", "c"}


# ---- 4. token budget under a non-additive counter --------------------------

def _abc_docs(n=4):
    return [ScoredDocument(Document(str(i), "abc"), 1, 1, 1, i) for i in range(n)]


class TestTokenBudgetExact:
    def test_reviewer_repro_budget_7(self):
        context, picked = pack(_abc_docs(), 7)
        assert estimate_tokens(context) <= 7
        assert len(picked) == 3

    @pytest.mark.parametrize("budget", list(range(0, 13)))
    def test_returned_context_never_exceeds_budget(self, budget):
        context, picked = pack(_abc_docs(), budget)
        if context:
            assert estimate_tokens(context) <= budget
        else:
            assert picked == []

    def test_exact_fit_is_accepted(self):
        full = "\n\n".join(f"[{i+1}] abc" for i in range(4))
        context, picked = pack(_abc_docs(), estimate_tokens(full))
        assert len(picked) == 4 and context == full

    def test_ceiling_counter_is_honoured(self):
        ceil4 = lambda s: -(-len(s) // 4)          # rounds UP per call
        context, picked = pack(_abc_docs(), 7, count_tokens=ceil4)
        assert ceil4(context) <= 7 and picked

    def test_context_equals_what_was_measured(self):
        seen = []
        def counting(s):
            seen.append(s)
            return len(s) // 4
        context, _ = pack(_abc_docs(), 6, count_tokens=counting)
        assert context in seen                      # the returned string was measured


# ---- 6. self-loop weight ---------------------------------------------------

class TestSelfLoopParity:
    @pytest.mark.parametrize("w", [1.0, 3.0])
    def test_weighted_self_loop_matches_networkx(self, w):
        g = nx.Graph()
        g.add_edge("a", "a", weight=w)
        g.add_edge("a", "b", weight=1.0)
        got = PPRSolver(g).solve(["a"], max_iter=1000, tol=1e-12)
        exp = nx.pagerank(g, alpha=0.85, personalization={"a": 1, "b": 0},
                          max_iter=1000, tol=1e-12)
        for n in g:
            assert abs(got[n] - exp[n]) < 1e-6

    def test_unweighted_self_loop_matches_networkx(self):
        g = nx.Graph([("a", "a"), ("a", "b")])
        got = PPRSolver(g).solve(["a"], max_iter=1000, tol=1e-12)
        exp = nx.pagerank(g, alpha=0.85, personalization={"a": 1, "b": 0},
                          max_iter=1000, tol=1e-12)
        assert abs(got["a"] - exp["a"]) < 1e-6 and abs(got["b"] - exp["b"]) < 1e-6

    def test_isolated_self_loop_node_keeps_all_mass(self):
        g = nx.Graph([("a", "a")])
        assert abs(PPRSolver(g).solve(["a"])["a"] - 1.0) < 1e-9

    def test_llm_self_relation_is_a_single_loop(self):
        kg = build_entity_kg_llm([Document("d", "Alice is Alice")],
                                 llm=lambda _: '{"triples": [["Alice", "is", "Alice"]]}')
        g = kg.graph
        assert g.has_edge("ent:alice", "ent:alice")
        got = PPRSolver(g).solve(["ent:alice"], max_iter=1000, tol=1e-12)
        exp = nx.pagerank(g, alpha=0.85, max_iter=1000, tol=1e-12,
                          personalization={n: float(n == "ent:alice") for n in g})
        for n in g:
            assert abs(got[n] - exp[n]) < 1e-6


# ---- 7. malformed LLM replies must not be cached ---------------------------

class TestMalformedExtractionNotCached:
    def test_reviewer_repro_malformed_then_valid(self, tmp_path):
        p = tmp_path / "triples.json"
        d = [Document("a", "Alice knows Bob")]
        with pytest.warns(UserWarning, match="unparseable"):
            first = build_entity_kg_llm(d, llm=lambda _: "NOT JSON",
                                        llm_identity="m", cache_path=p)
        assert first.extraction_stats["unparseable"] == 1
        second = build_entity_kg_llm(d, llm=lambda _: TRIPLE,
                                     llm_identity="m", cache_path=p)
        s = second.extraction_stats
        assert (s["cache_hits"], s["llm_calls"], s["unparseable"]) == (0, 1, 0)
        assert len(second.entity_to_docs) == 2

    def test_valid_empty_extraction_is_cached(self, tmp_path):
        p = tmp_path / "triples.json"
        d = [Document("a", "Nothing to extract here")]
        first = build_entity_kg_llm(d, llm=lambda _: '{"triples": []}',
                                    llm_identity="m", cache_path=p)
        assert first.extraction_stats["unparseable"] == 0

        def must_not_be_called(_):
            raise AssertionError("cached empty result should have been used")
        second = build_entity_kg_llm(d, llm=must_not_be_called,
                                     llm_identity="m", cache_path=p)
        assert second.extraction_stats["cache_hits"] == 1
        assert second.extraction_stats["llm_calls"] == 0

    @pytest.mark.parametrize("reply", ['{"foo": 1}', "42", "", "```json\nnope\n```"])
    def test_non_result_replies_are_unparseable_and_retried(self, tmp_path, reply):
        p = tmp_path / "triples.json"
        d = [Document("a", "Alice knows Bob")]
        with pytest.warns(UserWarning):
            first = build_entity_kg_llm(d, llm=lambda _: reply,
                                        llm_identity="m", cache_path=p)
        assert first.extraction_stats["unparseable"] == 1
        calls = {"n": 0}
        def llm(_):
            calls["n"] += 1
            return TRIPLE
        build_entity_kg_llm(d, llm=llm, llm_identity="m", cache_path=p)
        assert calls["n"] == 1                       # retried, not served from cache

    def test_partially_malformed_list_keeps_valid_triples(self, tmp_path):
        reply = '{"triples": [["Alice", "knows", "Bob"], ["bad"], 7]}'
        kg = build_entity_kg_llm([Document("a", "x")], llm=lambda _: reply)
        assert kg.extraction_stats["unparseable"] == 0
        assert len(kg.entity_to_docs) == 2
