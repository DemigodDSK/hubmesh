"""MCP server tool-function tests. Skipped entirely when the mcp SDK
isn't installed (CI installs only [dev]); the tools are plain functions
after FastMCP registration, so they're testable without a transport."""
import zlib
import numpy as np
import pytest

pytest.importorskip("mcp")

import hubmesh.mcp_server as srv
from hubmesh.corpus import CorpusManager
from hubmesh.kg_llm import build_entity_kg_llm


def _fake_embed(text: str) -> np.ndarray:
    v = np.zeros(32, dtype=np.float32)
    for tok in text.lower().split():
        v[zlib.crc32(tok.encode()) % 32] += 1.0
    n = np.linalg.norm(v)
    return v / n if n else v


class FakeNLP:
    def __call__(self, text):
        return type("D", (), {"ents": []})()


def _stub_doc(doc_id, text):
    return type("D", (), {"id": doc_id, "text": text, "metadata": {}})()


_TEXTS = {
    "acq": "Nimbus acquired Voxel.",
    "founder": "Elena founded Nimbus.",
    "bio": "Elena studied at Tromso.",
}
_TRIPLES = {
    "Nimbus acquired Voxel.": '{"triples": [["Nimbus", "acquired", "Voxel"]]}',
    "Elena founded Nimbus.": '{"triples": [["Elena", "founded", "Nimbus"]]}',
    "Elena studied at Tromso.": '{"triples": [["Elena", "studied at", "Tromso"]]}',
}


def _mock_llm(prompt):
    for passage, response in _TRIPLES.items():
        if passage in prompt:
            return response
    return '{"triples": []}'


@pytest.fixture()
def corpus(tmp_path, monkeypatch):
    mgr = CorpusManager(root=tmp_path, embed=_fake_embed, nlp=FakeNLP())
    kg = build_entity_kg_llm(
        [_stub_doc(k, v) for k, v in _TEXTS.items()], llm=_mock_llm)
    mgr.build("demo", [{"id": k, "text": v} for k, v in _TEXTS.items()],
              kg=kg)
    monkeypatch.setattr(srv, "_manager", mgr)
    return "demo"


def test_path_between_returns_k_annotated_paths(corpus):
    out = srv.path_between(corpus, "Voxel", "Tromso", k_paths=3)
    assert out["connected"] is True
    assert 1 <= len(out["paths"]) <= 3
    first = out["paths"][0]
    assert first["hops"] == len(first["nodes"]) - 1
    assert "via_documents" in first
    # paths come shortest-first
    hops = [p["hops"] for p in out["paths"]]
    assert hops == sorted(hops)
    # endpoints resolve to the right entities
    assert first["nodes"][0]["node"] == "ent:voxel"
    assert first["nodes"][-1]["node"] == "ent:tromso"


def test_path_between_disconnected_and_missing(corpus, monkeypatch):
    out = srv.path_between(corpus, "Voxel", "NoSuchEntity")
    assert "error" in out


def test_retrieve_tool_passes_seed_and_exclude(corpus):
    out = srv.retrieve(corpus, "where did she study", top_k=3,
                       seed_entities=["Elena"], exclude_docs=["founder"])
    ids = [s["id"] for s in out["sources"]]
    assert "founder" not in ids
    assert "ent:elena" in out["seeds_used"]


def test_graph_stats_shape(corpus):
    out = srv.graph_stats(corpus)
    assert out["documents"] == 3
    assert out["entities"] > 0
    assert isinstance(out["top_hub_entities"], list)
