"""CorpusManager persistence round-trip. No spaCy models, no
sentence-transformers, no MCP SDK: fake embedder + mock-LLM KG."""
import zlib
import numpy as np
import pytest

from hubmesh.corpus import CorpusManager, kg_to_dict, kg_from_dict
from hubmesh.kg_llm import build_entity_kg_llm


def _fake_embed(text: str) -> np.ndarray:
    """Deterministic hashed bag-of-words (process-independent)."""
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


def _build_kg():
    return build_entity_kg_llm(
        [_stub_doc(k, v) for k, v in _TEXTS.items()], llm=_mock_llm)


def test_kg_dict_roundtrip_preserves_everything():
    kg = _build_kg()
    kg2 = kg_from_dict(kg_to_dict(kg))
    assert set(kg2.graph.nodes) == set(kg.graph.nodes)
    assert set(kg2.graph.edges) == set(kg.graph.edges)
    # edge attributes (weights, predicates) survive
    e = ("ent:elena", "ent:nimbus")
    assert kg2.graph[e[0]][e[1]] == kg.graph[e[0]][e[1]]
    assert kg2.doc_to_entities == kg.doc_to_entities
    assert kg2.entity_to_docs == kg.entity_to_docs
    assert kg2.alias_to_node == kg.alias_to_node
    assert kg2.entity_node_to_label == kg.entity_node_to_label


def test_build_persist_reload_retrieve(tmp_path):
    docs = [{"id": k, "text": v} for k, v in _TEXTS.items()]
    mgr = CorpusManager(root=tmp_path, embed=_fake_embed, nlp=FakeNLP())
    meta = mgr.build("demo", docs, kg=_build_kg())
    assert meta["n_docs"] == 3
    assert meta["kg_nodes"] > 0

    # a FRESH manager (new process equivalent) reopens from disk alone
    mgr2 = CorpusManager(root=tmp_path, embed=_fake_embed, nlp=FakeNLP())
    store, kg = mgr2.load("demo")
    assert sorted(store.all_ids()) == sorted(_TEXTS)
    assert kg.alias_to_node   # alias index survived the round trip

    # the v0.2 door works through a reloaded corpus
    res = mgr2.planner("demo").retrieve(
        query="which university", top_k=3,
        seed_entities=["Elena"], exclude_docs=["founder"])
    ids = [s.doc.id for s in res.sources]
    assert "founder" not in ids
    assert res.debug["injected_seeds"] == ["ent:elena"]


def test_list_shows_meta(tmp_path):
    mgr = CorpusManager(root=tmp_path, embed=_fake_embed, nlp=FakeNLP())
    mgr.build("demo", [{"id": k, "text": v} for k, v in _TEXTS.items()],
              kg=_build_kg())
    listing = mgr.list()
    assert "demo" in listing
    assert listing["demo"]["n_docs"] == 3


def test_missing_corpus_raises(tmp_path):
    mgr = CorpusManager(root=tmp_path, embed=_fake_embed)
    with pytest.raises(FileNotFoundError, match="no corpus"):
        mgr.load("nope")
