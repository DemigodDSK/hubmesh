"""End-to-end demo: real text → chunking → embedding → entity-linked KG → multi-hop retrieval.

What this exercises:
  1. Document chunking          (hubmesh.chunk_documents)
  2. Sentence-transformer embeddings  (sentence_transformers)
  3. InMemoryStore vector index (hubmesh.adapters.InMemoryStore)
  4. Entity-linked KG construction with spaCy NER (hubmesh.kg.build_entity_kg)
  5. Embedding-based entity linking (hubmesh.entity_linker.EmbeddingLinker)
  6. KG-mode retrieval with reasoning paths (hubmesh.Planner)

The corpus is a small set of Wikipedia-style paragraphs about connected
people, companies, and acquisitions — designed so the only way to answer
a multi-hop question is to traverse the entity graph.

Run: python examples/multi_hop_demo.py
"""
from __future__ import annotations
import time

from hubmesh import Planner, chunk_documents
from hubmesh.adapters import InMemoryStore
from hubmesh.kg import build_entity_kg
from hubmesh.entity_linker import EmbeddingLinker, make_st_embedder


# A small corpus where multi-hop reasoning genuinely matters.
# To answer "Where was the founder of Salesforce educated?", a retriever
# must connect:    Salesforce → Marc Benioff → University of Southern California
RAW_DOCS = [
    {
        "id": "salesforce_co",
        "text": (
            "Salesforce is an American cloud-based software company headquartered "
            "in San Francisco, California. It was founded in 1999. Salesforce "
            "provides customer relationship management software and applications "
            "focused on sales, customer service, marketing automation, and analytics. "
            "The company is one of the largest enterprise software companies in the world."
        ),
    },
    {
        "id": "marc_benioff",
        "text": (
            "Marc Benioff is an American billionaire internet entrepreneur. He is "
            "the founder, chairman and chief executive officer of Salesforce, an "
            "enterprise cloud computing company. Benioff was born in San Francisco "
            "and earned a Bachelor of Science in business administration from the "
            "University of Southern California. He worked at Oracle Corporation "
            "before founding Salesforce."
        ),
    },
    {
        "id": "usc_overview",
        "text": (
            "The University of Southern California (USC) is a private research "
            "university in Los Angeles, California, founded in 1880. USC is the "
            "oldest private research university in California and is known for its "
            "schools of business, cinematic arts, and engineering. Notable alumni "
            "include Marc Benioff, George Lucas, and astronaut Neil Armstrong."
        ),
    },
    {
        "id": "slack_acq",
        "text": (
            "In 2021 Salesforce completed its acquisition of Slack Technologies for "
            "approximately 27.7 billion dollars. Slack was founded in 2009 by Stewart "
            "Butterfield and is headquartered in San Francisco. After the acquisition, "
            "Slack continued to operate under its existing brand within the Salesforce "
            "portfolio."
        ),
    },
    {
        "id": "stewart_butterfield",
        "text": (
            "Stewart Butterfield is a Canadian businessman best known as the co-founder "
            "of Flickr and Slack. He was born in Lund, British Columbia. Butterfield "
            "studied philosophy at the University of Victoria and later earned a "
            "Master of Philosophy from the University of Cambridge."
        ),
    },
    {
        "id": "oracle_corp",
        "text": (
            "Oracle Corporation is an American multinational computer technology "
            "company headquartered in Austin, Texas. It was founded in 1977 by "
            "Larry Ellison, Bob Miner, and Ed Oates as Software Development Laboratories. "
            "Oracle is the third-largest software company in the world by revenue "
            "and market capitalization."
        ),
    },
    {
        "id": "noise_paragraph",
        "text": (
            "The Pacific Ocean is the largest body of water on Earth, covering about "
            "63 million square miles. It extends from the Arctic Ocean in the north to "
            "the Southern Ocean in the south, and from Asia and Australia in the west "
            "to the Americas in the east."
        ),
    },
]


def main():
    print("=" * 72)
    print("hubmesh end-to-end multi-hop demo")
    print("=" * 72)

    # -- 1. Chunk source documents into retrieval units
    print("\n[1/5] Chunking source documents...")
    chunks = chunk_documents(RAW_DOCS, strategy="sentences",
                             target_tokens=80, overlap_sentences=1)
    print(f"      {len(RAW_DOCS)} source docs → {len(chunks)} chunks")

    # -- 2. Embed
    print("\n[2/5] Embedding chunks (all-MiniLM-L6-v2)...")
    embed = make_st_embedder("all-MiniLM-L6-v2")
    t0 = time.perf_counter()
    vecs = embed([c.text for c in chunks])
    for c, v in zip(chunks, vecs):
        c.vector = v
    print(f"      embedded {len(chunks)} chunks in {time.perf_counter()-t0:.2f}s")

    # -- 3. Index in the in-memory adapter
    store = InMemoryStore(chunks, k=6)
    print(f"\n[3/5] Indexed in InMemoryStore (dim={store.dim})")

    # -- 4. Build the entity-linked KG with embedding-based linking
    print("\n[4/5] Building entity-linked KG (spaCy NER + EmbeddingLinker)...")
    import spacy
    nlp = spacy.load("en_core_web_sm")
    linker = EmbeddingLinker(embed=embed, threshold=0.82)
    t0 = time.perf_counter()
    kg = build_entity_kg(chunks, nlp=nlp, linker=linker)
    n_ent = sum(1 for n in kg.graph.nodes if n.startswith("ent:"))
    n_doc = sum(1 for n in kg.graph.nodes if n.startswith("doc:"))
    print(f"      {n_doc} doc nodes + {n_ent} entity nodes, "
          f"{kg.graph.number_of_edges()} edges, "
          f"{time.perf_counter()-t0:.2f}s")

    # -- 5. Run multi-hop queries
    # The planner's embedder takes a string and returns a single np.ndarray;
    # our `embed` callable is batched (takes list[str] and returns matrix).
    # Wrap it so the Planner can call it with one query at a time.
    def embed_one(text: str):
        return embed([text])[0]

    planner = Planner(store=store, kg=kg, nlp=nlp, embed=embed_one)

    queries = [
        "Where did the founder of Salesforce go to university?",
        "What university did the founder of the company that bought Slack attend?",
        "Where is the headquarters of the company Marc Benioff founded?",
    ]

    print("\n[5/5] Multi-hop retrieval:\n")
    for q in queries:
        result = planner.retrieve(query=q, top_k=3, budget_tokens=1500)
        print(f"  Q: {q}")
        print(f"     query_mentions: {result.debug.get('query_mentions', [])}")
        print(f"     ppr_seeds:      {result.debug.get('ppr_seeds', [])[:5]}")
        print("     retrieved:")
        for s in result.sources:
            sid = s.doc.id
            preview = s.doc.text[:90].replace("\n", " ")
            print(f"       [{s.composite_score:.3f}] {sid:<25} {preview}…")
        if result.reasoning:
            print("     reasoning paths:")
            for p in result.reasoning[:3]:
                arrow = " → ".join(p.node_ids)
                print(f"       (score={p.score:.4f})  {arrow}")
        print()


if __name__ == "__main__":
    main()
