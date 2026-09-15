"""Batch-4 regression tests (external review): cache invalidation,
token-budget accounting, protocol contract, batched fetches, LLM-KG
extraction robustness, and MCP schema/write-policy extras.
"""
import json
import threading

import numpy as np
import networkx as nx
import pytest

from hubmesh import Document, Planner
from hubmesh.adapters import InMemoryStore
from hubmesh.adapters.base import VectorStore
from hubmesh.corpus import CorpusManager
from hubmesh.kg import EntityKG
from hubmesh.packing import pack, estimate_tokens
from hubmesh.planner import PlannerConfig
from hubmesh.types import ScoredDocument


def stub_embed(text: str) -> np.ndarray:
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    v = rng.normal(size=16).astype(np.float32)
    return v / np.linalg.norm(v)


def minimal_kg(doc_ids):
    g = nx.Graph()
    for d in doc_ids:
        g.add_node(f"doc:{d}", kind="doc")
    return EntityKG(graph=g, doc_to_entities={d: set() for d in doc_ids},
                    entity_to_docs={}, entity_canonical_to_node={},
                    entity_node_to_label={}, alias_to_node={})


# ---- corpus planner cache keyed by config ---------------------------------

class TestPlannerCacheByConfig:
    def test_config_change_returns_different_planner(self, tmp_path):
        """Round-1 repro: convergence-off then convergence-on returned the
        first planner unchanged."""
        mgr = CorpusManager(root=tmp_path, embed=stub_embed,
                            embed_identity="m")
        mgr.build("c", ["alpha text", "beta text"], kg=minimal_kg(["0", "1"]))
        p_off = mgr.planner("c", PlannerConfig(use_convergence=False))
        p_on = mgr.planner("c", PlannerConfig(use_convergence=True))
        assert p_off is not p_on
        assert p_off.config.use_convergence is False
        assert p_on.config.use_convergence is True
        # same config -> same cached instance; store/kg shared across configs
        assert mgr.planner("c", PlannerConfig(use_convergence=True)) is p_on
        assert p_off.store is p_on.store

    def test_rebuild_invalidates_all_configs(self, tmp_path):
        mgr = CorpusManager(root=tmp_path, embed=stub_embed,
                            embed_identity="m")
        mgr.build("c", ["one"], kg=minimal_kg(["0"]))
        p1 = mgr.planner("c", PlannerConfig(use_convergence=False))
        mgr.build("c", ["two"], kg=minimal_kg(["0"]))
        p2 = mgr.planner("c", PlannerConfig(use_convergence=False))
        assert p1 is not p2
        assert p2.store.get("0").text == "two"


# ---- adapter neighbor caches ---------------------------------------------

class TestNeighborCaches:
    def make_store(self, n=20, k=3):
        docs = [Document(id=str(i), text=f"d{i}", vector=stub_embed(f"v{i}"))
                for i in range(n)]
        return InMemoryStore(docs, k=k)

    def test_inmemory_larger_k_recomputes_instead_of_truncating(self):
        store = self.make_store(n=20, k=3)
        assert len(store.neighbors("0", 3)) == 3     # eager graph at k=3
        assert len(store.neighbors("0", 8)) == 8     # must grow, not cap
        assert len(store.neighbors("0", 19)) == 19   # bounded by n-1
        assert len(store.neighbors("0", 50)) == 19

    def test_inmemory_declares_mutation_counter(self):
        store = self.make_store()
        assert store.mutation_counter == 0

    def test_chroma_upsert_invalidates_neighbors_and_bumps_version(self):
        chromadb = pytest.importorskip("chromadb")
        from hubmesh.adapters import ChromaStore
        docs = [Document(id=str(i), text=f"d{i}", vector=stub_embed(f"v{i}"))
                for i in range(6)]
        # own collection: the ephemeral client is process-wide and the
        # default "hubmesh" collection is used (at another dim) by the
        # adapter's own tests
        store = ChromaStore.from_documents(docs, collection_name="b4_nbrs")
        try:
            before = store.neighbors("0", 2)
            v0 = store.mutation_counter
            # insert a doc identical to doc 0's vector -> it must appear as
            # the nearest neighbor after upsert, which requires invalidation
            store.upsert([Document(id="twin", text="twin",
                                   vector=stub_embed("v0"))])
            # batch 5: the version is bumped before AND after the write
            # (partial-write invalidation), so assert growth, not +1
            assert store.mutation_counter > v0
            after = store.neighbors("0", 2)
            assert "twin" in after and after != before
        finally:
            store._client.delete_collection("b4_nbrs")


# ---- protocol contract ---------------------------------------------------

class TestProtocol:
    def test_vector_of_is_part_of_protocol(self):
        assert "vector_of" in VectorStore.__protocol_attrs__ \
            if hasattr(VectorStore, "__protocol_attrs__") \
            else hasattr(VectorStore, "vector_of")

    def test_store_without_vector_of_fails_at_planner_construction(self):
        class NoVectors:
            def search(self, q, top_k): return []
            def get(self, i): raise KeyError(i)
            def get_many(self, ids): return []
            def neighbors(self, i, k): return []
            def all_ids(self): return []
            dim = 16
        with pytest.raises(NotImplementedError, match="vector_of"):
            Planner(store=NoVectors())


# ---- packing token accounting ---------------------------------------------

class TestPackingBudget:
    def make_scored(self, texts):
        return [ScoredDocument(doc=Document(id=str(i), text=t,
                                            vector=stub_embed(t)),
                               similarity=0.5, ppr_score=0.1,
                               composite_score=1.0 - i * 0.01, rank=i)
                for i, t in enumerate(texts)]

    def test_context_never_exceeds_budget_under_its_own_counter(self):
        """Round-1 repro: a 4-token body budget produced a 5-token context
        because numbering/separators were uncounted."""
        scored = self.make_scored(["a" * 16, "b" * 16, "c" * 16])
        budget = estimate_tokens("a" * 16)            # exactly one body
        context, picked = pack(scored, budget_tokens=budget)
        assert estimate_tokens(context) <= budget
        assert len(picked) <= 1
        context, picked = pack(scored, budget_tokens=100)
        assert estimate_tokens(context) <= 100

    def test_custom_token_counter_is_honored(self):
        scored = self.make_scored(["x" * 40, "y" * 40])
        counter = lambda s: len(s)                    # 1 token per char
        context, picked = pack(scored, budget_tokens=50, count_tokens=counter)
        assert len(context) <= 50 and len(picked) == 1

    def test_equal_scores_break_ties_by_doc_id(self):
        docs = [ScoredDocument(doc=Document(id=i, text="t", vector=stub_embed(i)),
                               similarity=0, ppr_score=0, composite_score=0.5,
                               rank=0) for i in ("b", "c", "a")]
        _, picked = pack(docs, budget_tokens=1000)
        assert [p.doc.id for p in picked] == ["a", "b", "c"]

    def test_planner_config_token_counter_plumbed(self):
        docs = [Document(id=str(i), text="word " * 50, vector=stub_embed(str(i)))
                for i in range(5)]
        calls = []

        def counter(s):
            calls.append(s); return len(s)
        planner = Planner(store=InMemoryStore(docs, k=2),
                          config=PlannerConfig(token_counter=counter))
        planner.retrieve(stub_embed("q"), top_k=3, budget_tokens=10_000)
        assert calls, "custom counter was never invoked"


# ---- batched fetch ---------------------------------------------------------

class TestBatchedFetch:
    def test_planner_fetches_candidates_in_one_get_many(self):
        docs = [Document(id=str(i), text=f"doc {i}", vector=stub_embed(str(i)))
                for i in range(30)]
        store = InMemoryStore(docs, k=4)
        calls = {"get_many": 0, "get": 0}
        orig_many, orig_get = store.get_many, store.get

        def counted_many(ids):
            calls["get_many"] += 1; return orig_many(ids)

        def counted_get(i):
            calls["get"] += 1; return orig_get(i)
        store.get_many, store.get = counted_many, counted_get
        planner = Planner(store=store)
        planner.retrieve(stub_embed("q"), top_k=3, budget_tokens=10_000)
        assert calls["get_many"] == 1
        assert calls["get"] <= 1   # only the fallback path calls get()


# ---- LLM-KG extraction robustness ------------------------------------------

class TestLLMKG:
    def docs(self):
        return [Document(id="a", text="Alice founded Acme in Paris."),
                Document(id="b", text="Bob leads Acme."),
                Document(id="c", text="")]

    def test_failures_are_counted_reported_and_not_cached(self, tmp_path):
        from hubmesh.kg_llm import build_entity_kg_llm
        seen = []

        def flaky(prompt):
            seen.append(prompt)
            if "Bob" in prompt:
                raise RuntimeError("provider down")
            return '{"triples": [["Alice", "founded", "Acme"]]}'
        cache = tmp_path / "c.json"
        with pytest.warns(UserWarning, match="calls failed"):
            kg = build_entity_kg_llm(self.docs(), llm=flaky, cache_path=cache,
                                     llm_identity="test-llm")
        s = kg.extraction_stats
        assert s["failed_calls"] == 1 and s["llm_calls"] == 2
        assert s["empty_docs"] == 1
        # the failed doc was NOT cached -> a rerun retries it
        entries = json.loads(cache.read_text())["entries"]
        assert len(entries) == 1
        kg2 = build_entity_kg_llm(self.docs(), llm=flaky, cache_path=cache,
                                  llm_identity="test-llm")
        assert kg2.extraction_stats["cache_hits"] == 1
        assert kg2.extraction_stats["llm_calls"] == 1     # only Bob retried

    def test_cache_is_namespaced_by_identity(self, tmp_path):
        from hubmesh.kg_llm import build_entity_kg_llm
        cache = tmp_path / "c.json"
        llm_a = lambda p: '{"triples": [["Alice", "founded", "Acme"]]}'
        llm_b = lambda p: '{"triples": [["Alice", "left", "Acme"]]}'
        d = self.docs()[:1]
        build_entity_kg_llm(d, llm=llm_a, cache_path=cache, llm_identity="A")
        kg_b = build_entity_kg_llm(d, llm=llm_b, cache_path=cache,
                                   llm_identity="B")
        assert kg_b.extraction_stats["cache_hits"] == 0   # no cross-model hit
        preds = [dat.get("predicates") for _, _, dat in kg_b.graph.edges(data=True)
                 if dat.get("predicates")]
        assert any("left" in p for p in preds)
        assert len(json.loads(cache.read_text())["entries"]) == 2

    def test_max_workers_preserves_results(self, tmp_path):
        from hubmesh.kg_llm import build_entity_kg_llm
        llm = lambda p: '{"triples": [["X", "rel", "Y"]]}'
        d = [Document(id=str(i), text=f"passage {i}") for i in range(8)]
        kg1 = build_entity_kg_llm(d, llm=llm, max_workers=1)
        kg4 = build_entity_kg_llm(d, llm=llm, max_workers=4)
        assert kg4.extraction_stats["llm_calls"] == 8
        assert set(kg1.graph.nodes) == set(kg4.graph.nodes)


# ---- MCP extras: schema unions + concurrent indexing -----------------------

class TestMCPExtras:
    def test_tool_schemas_have_no_nullable_unions(self):
        """Strict connector executors (observed: Perplexity) reject
        anyOf/null parameter schemas; the founder's fix must not regress."""
        pytest.importorskip("mcp")
        from hubmesh import mcp_server
        import asyncio
        tools = asyncio.run(mcp_server.mcp.list_tools())
        for t in tools:
            for name, schema in t.inputSchema.get("properties", {}).items():
                assert "anyOf" not in schema, f"{t.name}.{name} has a union"

    def test_concurrent_index_calls_serialize(self, tmp_path, monkeypatch):
        pytest.importorskip("mcp")
        from hubmesh import mcp_server
        mgr = CorpusManager(root=tmp_path, embed=stub_embed,
                            embed_identity="m", retention_seconds=0.0)
        monkeypatch.setattr(mcp_server, "_manager", mgr)
        monkeypatch.setattr(mcp_server, "_read_only", False)
        fn = getattr(mcp_server.index_corpus, "fn", mcp_server.index_corpus)
        kg = minimal_kg(["0"])
        monkeypatch.setattr(mgr, "build",
                            lambda name, docs, kg=None, _b=mgr.build:
                            _b(name, docs, kg=minimal_kg(["0"])))
        errors = []

        def worker(txt):
            try:
                fn("corp", [{"id": "0", "text": txt}])
            except Exception as e:      # pragma: no cover
                errors.append(e)
        ts = [threading.Thread(target=worker, args=(f"t{i}",)) for i in range(4)]
        [t.start() for t in ts]; [t.join() for t in ts]
        assert errors == []
        assert set(mgr.list()) == {"corp"}
        store, _ = mgr.load("corp")
        assert store.get("0").text.startswith("t")
