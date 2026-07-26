"""Agent-driven iterative multi-hop: the v0.2 seed-injection door.

A 3-hop chain is hidden in a corpus with lexical distractors:

    Voxel Systems --acquired by--> Nimbus Analytics
        --founded by--> Elena Marsh --studied at--> Univ. of Tromso

Question: "Which university did the founder of the company that
acquired Voxel Systems study at?"  (answer doc: 'bio_doc')

Single-shot retrieval fails here by design: PPR mass decays over the
five bipartite hops from the question's entity to the answer, and
cosine similarity prefers the distractor that lexically mirrors the
question. An iterative caller — an LLM agent, or the hardcoded loop
below standing in for one — reads each hop, names the bridge entity it
found, and aims the next hop at it via `seed_entities`, masking
consumed docs via `exclude_docs`. The query path stays LLM-free; the
planning intelligence lives in the caller.

Run: python examples/agent_hop_demo.py
Requires: pip install "hubmesh[kg,linker]" + en_core_web_sm
"""
from __future__ import annotations

from hubmesh import Planner
from hubmesh.adapters import InMemoryStore
from hubmesh.kg import build_entity_kg
from hubmesh.entity_linker import make_st_embedder


DOCS = [
    {"id": "acq_doc", "text": (
        "Nimbus Analytics completed its acquisition of Voxel Systems in 2020 "
        "for 1.2 billion dollars. Voxel Systems, based in Denver, builds "
        "real-time 3D graphics engines for simulation software.")},
    {"id": "founder_doc", "text": (
        "Nimbus Analytics is a data infrastructure company headquartered in "
        "Oslo. The company was founded in 2011 by Elena Marsh, who previously "
        "led engineering teams at Google.")},
    {"id": "bio_doc", "text": (
        "Elena Marsh is a Norwegian-American engineer. She studied computer "
        "science at the University of Tromso and later completed a doctorate "
        "at ETH Zurich before moving into industry.")},
    {"id": "distractor_pixel", "text": (
        "TechVentura acquired Pixel Labs in 2019. The founder of Pixel Labs, "
        "John Reed, studied economics at Harvard University before starting "
        "the company that was later acquired.")},
    {"id": "distractor_stanford", "text": (
        "Many startup founders study at top universities. Stanford University "
        "has produced numerous company founders whose companies were acquired "
        "by larger firms after they finished their studies.")},
    {"id": "distractor_michigan", "text": (
        "The University of Michigan announced a research center funded by "
        "companies founded by alumni. Several of these companies were "
        "acquired in recent years by public acquirers.")},
    {"id": "noise_ocean", "text": (
        "The Pacific Ocean is the largest body of water on Earth, covering "
        "about 63 million square miles between Asia and the Americas.")},
    {"id": "noise_baking", "text": (
        "Sourdough bread requires a fermented starter of flour and water. "
        "The dough is folded, proofed overnight, and baked in a dutch oven.")},
]

QUESTION = ("Which university did the founder of the company that acquired "
            "Voxel Systems study at?")

ANSWER_DOC = "bio_doc"

# What an agent would read out of each hop's top document. Hardcoded here
# so the demo runs offline; in production an LLM plays this role.
AGENT_BRIDGE_ENTITIES = ["Nimbus Analytics", "Elena Marsh"]


def rank_of(result, doc_id):
    ids = [s.doc.id for s in result.sources]
    return ids.index(doc_id) + 1 if doc_id in ids else None


def main():
    batched = make_st_embedder("all-MiniLM-L6-v2")

    def embed_one(text: str):
        return batched([text])[0]

    store = InMemoryStore.from_documents(DOCS, embed=embed_one, k=4)
    import spacy
    nlp = spacy.load("en_core_web_sm")
    kg = build_entity_kg(store.get_many(store.all_ids()), nlp=nlp)
    planner = Planner(store=store, kg=kg, nlp=nlp, embed=embed_one)

    print(f"Q: {QUESTION}\n")

    # -- Single-shot baseline
    single = planner.retrieve(QUESTION, top_k=len(DOCS),
                              budget_tokens=100_000)
    print(f"single-shot:  answer rank = {rank_of(single, ANSWER_DOC)} "
          f"of {len(single.sources)}")

    # -- Iterative loop: hop N seeded by what the 'agent' read in hop N-1
    consumed: list[str] = []
    result = planner.retrieve(QUESTION, top_k=3)
    for bridge in AGENT_BRIDGE_ENTITIES:
        consumed.append(result.sources[0].doc.id)
        result = planner.retrieve(
            QUESTION, top_k=len(DOCS), budget_tokens=100_000,
            seed_entities=[bridge],       # merged with the query's own seeds
            exclude_docs=consumed,        # don't re-retrieve consumed docs
        )
        print(f"hop via {bridge!r}:  top = {result.sources[0].doc.id}  "
              f"(seeds: {result.debug['ppr_seeds']})")

    print(f"\niterative:    answer rank = {rank_of(result, ANSWER_DOC)} "
          f"of {len(result.sources)}")


if __name__ == "__main__":
    main()
