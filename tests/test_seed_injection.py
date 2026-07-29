"""Tests for the v0.2 iterative-retrieval door (seed_entities /
exclude_docs, merge semantics), the kg_llm linker routing, and the
alias index. All spaCy-free: KGs come from the mock-LLM builder or a
fake NLP stub, so CI needs no model download."""
import numpy as np
import pytest

from hubmesh import Planner, Document
from hubmesh.adapters import InMemoryStore
from hubmesh.kg import EntityKG, build_entity_kg, canonicalize
from hubmesh.kg_llm import build_entity_kg_llm


class FakeNLP:
    """Stands in for spaCy: returns a doc whose .ents yields fixed
    mentions regardless of input text (query-side use)."""
    def __init__(self, mentions=()):
        self._mentions = list(mentions)

    def __call__(self, text):
        ents = [type("E", (), {"text": m, "label_": "ORG"})()
                for m in self._mentions]
        return type("D", (), {"ents": ents})()


class FakeCorpusNLP:
    """Stands in for spaCy at build time: maps doc text → mentions."""
    def __init__(self, mapping):
        self._mapping = mapping

    def __call__(self, text):
        ents = [type("E", (), {"text": m, "label_": "PERSON"})()
                for m in self._mapping.get(text, [])]
        return type("D", (), {"ents": ents})()


def _stub_doc(doc_id, text):
    return type("D", (), {"id": doc_id, "text": text, "metadata": {}})()


# ---------------------------------------------------------------------
# Fixture corpus: a 3-doc chain plus a graph-disconnected distractor.
#   Voxel —(acq)— Nimbus —(founder)— Elena —(bio)— Tromso
# Vectors are hand-set so the distractor has the best cosine to the
# query while the bio doc is only second — structural signal must do
# the discriminating work.
# ---------------------------------------------------------------------

_TEXTS = {
    "acq": "Nimbus acquired Voxel.",
    "founder": "Elena founded Nimbus.",
    "bio": "Elena studied at Tromso.",
    "distract": "Harvard hosts a founders club.",
}
_TRIPLES = {
    "Nimbus acquired Voxel.": '{"triples": [["Nimbus", "acquired", "Voxel"]]}',
    "Elena founded Nimbus.": '{"triples": [["Elena", "founded", "Nimbus"]]}',
    "Elena studied at Tromso.": '{"triples": [["Elena", "studied at", "Tromso"]]}',
    "Harvard hosts a founders club.":
        '{"triples": [["Harvard", "hosts", "Founders Club"]]}',
}
_VECS = {
    "acq": [0.2, 0.9798, 0.0, 0.0],
    "founder": [0.2, 0.0, 0.0, 0.9798],
    "bio": [0.5, 0.0, 0.866, 0.0],
    "distract": [0.95, 0.31225, 0.0, 0.0],
}
_QVEC = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def _mock_llm(prompt):
    for passage, response in _TRIPLES.items():
        if passage in prompt:
            return response
    return '{"triples": []}'


def _chain_setup(nlp=None):
    docs = [Document(id=k, text=v,
                     vector=np.array(_VECS[k], dtype=np.float32))
            for k, v in _TEXTS.items()]
    store = InMemoryStore(docs, k=3)
    kg = build_entity_kg_llm(docs, llm=_mock_llm)
    planner = Planner(store=store, kg=kg, nlp=nlp or FakeNLP())
    return planner


def test_injected_seeds_steer_ppr():
    planner = _chain_setup()
    res = planner.retrieve(query="which university", query_vec=_QVEC,
                           top_k=4, seed_entities=["Elena"])
    assert res.debug["injected_seeds"] == ["ent:elena"]
    assert "ent:elena" in res.debug["ppr_seeds"]
    ppr = {s.doc.id: s.ppr_score for s in res.sources}
    # bio and founder are 1 hop from the seed — they must carry PPR mass
    assert ppr.get("bio", 0.0) > 0.0
    # distract sits in a disconnected component — no mass can reach it
    assert ppr.get("distract", 1.0) < 1e-9


def test_merge_semantics_keep_ner_seeds():
    planner = _chain_setup(nlp=FakeNLP(["Harvard"]))
    res = planner.retrieve(query="anything", query_vec=_QVEC,
                           top_k=4, seed_entities=["Elena"])
    seeds = res.debug["ppr_seeds"]
    # injected seeds lead, the query's own NER seeds survive the merge
    assert seeds[0] == "ent:elena"
    assert "ent:harvard" in seeds


def test_exclude_docs_kg_mode():
    planner = _chain_setup()
    res = planner.retrieve(query="anything", query_vec=_QVEC, top_k=4,
                           seed_entities=["Elena"],
                           exclude_docs=["distract", "founder"])
    ids = [s.doc.id for s in res.sources]
    assert "distract" not in ids
    assert "founder" not in ids
    assert res.debug["excluded_docs"] == 2
    # with the cosine-best distractor gone, the seeded doc wins
    assert ids[0] == "bio"


def test_seed_entities_requires_kg_mode():
    docs = [Document(id=k, text=v,
                     vector=np.array(_VECS[k], dtype=np.float32))
            for k, v in _TEXTS.items()]
    store = InMemoryStore(docs, k=3)
    planner = Planner(store=store)   # no KG → kNN mode
    with pytest.raises(ValueError, match="KG mode"):
        planner.retrieve(query=_QVEC, top_k=2, seed_entities=["Elena"])


def test_exclude_docs_knn_mode():
    docs = [Document(id=k, text=v,
                     vector=np.array(_VECS[k], dtype=np.float32))
            for k, v in _TEXTS.items()]
    store = InMemoryStore(docs, k=3)
    planner = Planner(store=store)
    first = planner.retrieve(query=_QVEC, top_k=4)
    top = first.sources[0].doc.id
    again = planner.retrieve(query=_QVEC, top_k=4, exclude_docs=[top])
    assert top not in [s.doc.id for s in again.sources]


# ---------------------------------------------------------------------
# kg_llm linker routing + alias index
# ---------------------------------------------------------------------

class MergeUSALinker:
    """Toy Linker: unifies USA / United States, canonicalizes the rest."""
    def link(self, mentions):
        out = {}
        for m in mentions:
            c = canonicalize(m)
            out[m] = "united states" if c in ("usa", "united states") else c
        return out


_USA_TRIPLES = {
    "Alice moved to USA.": '{"triples": [["Alice", "moved to", "USA"]]}',
    "Bob lives in United States.":
        '{"triples": [["Bob", "lives in", "United States"]]}',
}


def _usa_llm(prompt):
    for passage, response in _USA_TRIPLES.items():
        if passage in prompt:
            return response
    return '{"triples": []}'


def test_kg_llm_routes_through_linker():
    docs = [_stub_doc("d1", "Alice moved to USA."),
            _stub_doc("d2", "Bob lives in United States.")]
    kg = build_entity_kg_llm(docs, llm=_usa_llm, linker=MergeUSALinker())
    assert "ent:united states" in kg.graph
    assert "ent:usa" not in kg.graph          # merged, not duplicated
    assert kg.entity_to_docs["ent:united states"] == {"d1", "d2"}
    # both surface forms resolve through the alias index
    assert kg.alias_to_node["usa"] == "ent:united states"
    assert kg.query_entity_nodes(["USA"]) == ["ent:united states"]


def test_kg_llm_without_linker_keeps_old_behaviour():
    docs = [_stub_doc("d1", "Alice moved to USA."),
            _stub_doc("d2", "Bob lives in United States.")]
    kg = build_entity_kg_llm(docs, llm=_usa_llm)
    # the historical gap, unchanged when no linker is passed
    assert "ent:usa" in kg.graph
    assert "ent:united states" in kg.graph
    assert kg.alias_to_node["usa"] == "ent:usa"


def test_spacy_path_alias_resolves_absorbed_form():
    text = "Scott Derrickson was mentioned, then Derrickson again."
    docs = [_stub_doc("d1", text)]
    nlp = FakeCorpusNLP({text: ["Scott Derrickson", "Derrickson"]})
    kg = build_entity_kg(docs, nlp=nlp)
    # substring collapse absorbed the short form into the long one...
    assert "ent:scott derrickson" in kg.graph
    assert "ent:derrickson" not in kg.graph
    # ...and the alias index still routes the short form to the survivor
    assert kg.alias_to_node["derrickson"] == "ent:scott derrickson"
    assert kg.query_entity_nodes(["Derrickson"]) == ["ent:scott derrickson"]


def test_alias_short_form_first_never_phantom():
    """Reversed mention order: the absorbed short form is seen BEFORE the
    long form. Pins the 'graph-backed only' invariant — without the
    nid-in-G guard, 'derrickson' would bind to phantom 'ent:derrickson'
    and PPR would silently degrade to uniform restart."""
    text = "Derrickson was mentioned, then Scott Derrickson."
    docs = [_stub_doc("d1", text)]
    nlp = FakeCorpusNLP({text: ["Derrickson", "Scott Derrickson"]})
    kg = build_entity_kg(docs, nlp=nlp)
    assert "ent:derrickson" not in kg.graph
    assert kg.alias_to_node["derrickson"] == "ent:scott derrickson"
    # every alias target must be a real graph node — no phantom seeds
    assert all(n in kg.graph for n in kg.alias_to_node.values())


def test_alias_never_shadows_exact_canonical():
    """A display alias of one entity must not claim a DIFFERENT entity's
    exact canonical name (v0.1 parity: exact match wins)."""
    t1 = "Apple Inc. was mentioned, then Apple."
    t2 = "Just Apple here."
    docs = [_stub_doc("d1", t1), _stub_doc("d2", t2)]
    nlp = FakeCorpusNLP({t1: ["Apple Inc.", "Apple"], t2: ["Apple"]})
    kg = build_entity_kg(docs, nlp=nlp)
    # both entities exist: d1's absorbed pair and d2's standalone
    assert "ent:apple inc" in kg.graph
    assert "ent:apple" in kg.graph
    # the exact canonical must win over apple-inc's display-derived alias
    assert kg.query_entity_nodes(["Apple"]) == ["ent:apple"]


def test_pre_02_state_gets_empty_alias_index():
    """KGs persisted before v0.2 restore without alias_to_node (attribute
    restore skips __init__). The __setstate__ shim must backfill an empty
    index so queries fall through to the old lookup paths."""
    docs = [_stub_doc("d1", "Alice moved to USA.")]
    kg = build_entity_kg_llm(docs, llm=_usa_llm)
    old_state = {k: v for k, v in kg.__dict__.items()
                 if k != "alias_to_node"}
    old = EntityKG.__new__(EntityKG)
    old.__setstate__(old_state)
    assert old.alias_to_node == {}
    assert old.query_entity_nodes(["USA"]) == ["ent:usa"]


def test_solve_multi_matches_single_solves():
    """Batched multi-seed PPR must agree with one-at-a-time solves."""
    from hubmesh.ppr import PPRSolver
    planner = _chain_setup()
    solver = planner._ppr_solver
    seeds = ["ent:elena", "ent:nimbus"]
    batched = solver.solve_multi([[s] for s in seeds])
    for s, dist in zip(seeds, batched):
        single = solver.solve([s])
        for node in single:
            assert abs(single[node] - dist[node]) < 1e-6


def test_hub_discount_shifts_mass_off_hub_routes():
    """With γ>0, diffusion into a high-degree entity is damped relative
    to low-degree neighbours (row normalisation cancels uniform
    factors, so it's the RELATIVE within-row weights that shift)."""
    from hubmesh.ppr import PPRSolver
    planner = _chain_setup()
    G = planner.kg.graph
    plain = PPRSolver(G).solve(["ent:elena"])
    damped = PPRSolver(G, hub_discount=1.0).solve(["ent:elena"])
    # ent:nimbus is elena's highest-degree entity neighbour — the hub
    # route. Its share of the diffusion must shrink under the discount.
    assert damped["ent:nimbus"] < plain["ent:nimbus"]
    # distributions still sum to ~1 (stochasticity preserved)
    assert abs(sum(damped.values()) - 1.0) < 1e-6


def test_convergence_component_prefers_multi_anchor_docs():
    """With two seeds, a doc reachable from both must beat a doc
    reachable from only one on the convergence component."""
    from hubmesh.planner import PlannerConfig
    from hubmesh.scoring import ScoringWeights
    docs = [Document(id=k, text=v,
                     vector=np.array(_VECS[k], dtype=np.float32))
            for k, v in _TEXTS.items()]
    store = InMemoryStore(docs, k=3)
    kg = build_entity_kg_llm(docs, llm=_mock_llm)
    cfg = PlannerConfig(use_convergence=True,
                        weights=ScoringWeights(relevance=0.0, structural=0.0,
                                               coherence=1.0))
    planner = Planner(store=store, kg=kg, nlp=FakeNLP(), config=cfg)
    # seeds: elena + voxel. 'founder' (elena+nimbus doc) and 'acq'
    # (nimbus+voxel doc) sit between the anchors; 'distract' touches
    # neither and must land at the bottom on pure convergence.
    res = planner.retrieve(query="anything", query_vec=_QVEC, top_k=4,
                           seed_entities=["Elena", "Voxel"])
    order = [s.doc.id for s in res.sources]
    assert order[-1] == "distract"


def test_merge_dedups_overlapping_seeds():
    """Re-injecting an entity the query already mentions (the natural
    iterative-retrieval pattern) must not duplicate it in the teleport
    set — duplication would double its restart mass."""
    planner = _chain_setup(nlp=FakeNLP(["Elena"]))
    res = planner.retrieve(query="anything", query_vec=_QVEC,
                           top_k=4, seed_entities=["Elena"])
    assert res.debug["ppr_seeds"].count("ent:elena") == 1
    assert res.debug["injected_seeds"] == ["ent:elena"]
