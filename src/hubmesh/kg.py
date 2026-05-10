"""Entity-linked knowledge graph construction.

Builds a bipartite KG over a document corpus:

  • document nodes  (id="doc:<doc_id>")
  • entity nodes    (id="ent:<canonical_form>")
  • doc—entity edges    (entity is mentioned in document)
  • entity—entity edges (entities co-occur in the same document, weighted)

This is the substrate hubmesh's Planner uses for query-time PPR. The graph
is *not* a similarity graph (the previous kNN approach); it encodes which
documents talk about which entities and which entities are mentioned
together. Multi-hop retrieval works over this graph because PPR can
reach distant documents through shared entities even when their
embeddings are dissimilar.

Entity linking uses a simple canonicalization (lowercase, punctuation
stripping, plus collapse of substring mentions like "Derrickson" →
"scott derrickson" if present). This is good enough for an MVP. A
production version would swap in a learned linker (e.g. BLINK).
"""
from __future__ import annotations
import re
from collections import defaultdict
from dataclasses import dataclass, field
import networkx as nx

_PUNCT_RE = re.compile(r"[^\w\s]")


# spaCy entity labels we keep — most informative for retrieval.
DEFAULT_ENT_LABELS = {
    "PERSON", "ORG", "GPE", "LOC", "NORP", "FAC",
    "EVENT", "WORK_OF_ART", "PRODUCT", "LAW", "LANGUAGE",
}


def canonicalize(mention: str) -> str:
    """Crude canonicalization: lowercase, strip punctuation, collapse spaces.
    Production should use a real entity linker."""
    s = _PUNCT_RE.sub(" ", mention.lower()).strip()
    return re.sub(r"\s+", " ", s)


@dataclass
class EntityKG:
    graph: nx.Graph                                  # bipartite KG
    doc_to_entities: dict[str, set[str]]             # doc_id → entity node ids
    entity_to_docs: dict[str, set[str]]              # entity node id → doc_ids
    entity_canonical_to_node: dict[str, str]         # canonical → node id
    entity_node_to_label: dict[str, str]             # node id → display label

    def query_entity_nodes(
        self, mentions: list[str], allow_substring: bool = True,
    ) -> list[str]:
        """Match raw mentions to KG entity nodes. Tries exact canonicalised
        match first; if `allow_substring`, falls back to substring match."""
        out = []
        for m in mentions:
            c = canonicalize(m)
            if not c:
                continue
            if c in self.entity_canonical_to_node:
                out.append(self.entity_canonical_to_node[c])
                continue
            if allow_substring:
                hits = [
                    nid for canon, nid in self.entity_canonical_to_node.items()
                    if c in canon or canon in c
                ]
                if hits:
                    # Prefer the longer (more specific) canonical form
                    hits.sort(key=lambda nid: -len(self.entity_node_to_label[nid]))
                    out.append(hits[0])
        # dedupe but preserve order
        seen = set()
        deduped = []
        for n in out:
            if n not in seen:
                seen.add(n)
                deduped.append(n)
        return deduped


def _entity_node(canonical: str) -> str:
    return f"ent:{canonical}"


def _doc_node(doc_id: str) -> str:
    return f"doc:{doc_id}"


def build_entity_kg(
    documents: list,                                 # list of Document
    nlp=None,
    labels: set[str] = None,
    min_entity_length: int = 2,
    linker=None,
) -> EntityKG:
    """Run NER over each document, build the bipartite KG.

    `nlp` is a spaCy pipeline; if None, loads `en_core_web_sm`.

    `linker` is an optional `entity_linker.Linker` for canonicalising
    mentions globally across the corpus. When None (default), the
    substring-collapse heuristic is applied per-document. Pass
    `EmbeddingLinker(...)` for cross-document embedding-based clustering
    — substantially better recall on entities with surface variation
    ("United States" / "U.S." / "USA").
    """
    if nlp is None:
        import spacy
        nlp = spacy.load("en_core_web_sm")
    if labels is None:
        labels = DEFAULT_ENT_LABELS

    # Pass 1: extract per-doc mentions (raw)
    raw_per_doc: dict[str, list[str]] = {}
    all_mentions: set[str] = set()
    for doc in documents:
        if not doc.text.strip():
            raw_per_doc[doc.id] = []
            continue
        spacy_doc = nlp(doc.text)
        ms: list[str] = []
        for ent in spacy_doc.ents:
            if ent.label_ not in labels:
                continue
            text = ent.text.strip()
            if len(text) < min_entity_length:
                continue
            ms.append(text)
            all_mentions.add(text)
        raw_per_doc[doc.id] = ms

    # Pass 2: canonicalise.
    canonical_to_displays: dict[str, set[str]] = defaultdict(set)
    if linker is not None:
        # Cross-document linking — produces a stable raw→canonical map.
        link_map = linker.link(all_mentions)   # raw → canonical
        canonical_per_doc: dict[str, set[str]] = {}
        for doc_id, mentions in raw_per_doc.items():
            canon_set = set()
            for m in mentions:
                c = link_map.get(m, canonicalize(m))
                if c:
                    canon_set.add(c)
                    canonical_to_displays[c].add(m)
            canonical_per_doc[doc_id] = canon_set
    else:
        # Default: per-doc substring collapse (the original behaviour).
        canonical_per_doc = {}
        for doc_id, mentions in raw_per_doc.items():
            canon_set = set()
            canon_list = [canonicalize(m) for m in mentions]
            for c, m in zip(canon_list, mentions):
                if c:
                    canon_set.add(c)
                    canonical_to_displays[c].add(m)
            sorted_canons = sorted(canon_set, key=len, reverse=True)
            keep: set[str] = set()
            for c in sorted_canons:
                absorbed = False
                for longer in keep:
                    if c != longer and (f" {c} " in f" {longer} "
                                        or longer.startswith(c + " ")
                                        or longer.endswith(" " + c)):
                        absorbed = True
                        canonical_to_displays[longer] |= canonical_to_displays.get(c, set())
                        break
                if not absorbed:
                    keep.add(c)
            canonical_per_doc[doc_id] = keep

    # Build canonical → node id map (stable: longest display form as label)
    entity_canonical_to_node: dict[str, str] = {}
    entity_node_to_label: dict[str, str] = {}
    for c, displays in canonical_to_displays.items():
        nid = _entity_node(c)
        entity_canonical_to_node[c] = nid
        # display label = longest mention seen
        entity_node_to_label[nid] = max(displays, key=len) if displays else c

    # Build the graph
    G = nx.Graph()
    doc_to_entities: dict[str, set[str]] = {}
    entity_to_docs: dict[str, set[str]] = defaultdict(set)
    for doc_id, canon_set in canonical_per_doc.items():
        d_node = _doc_node(doc_id)
        G.add_node(d_node, kind="doc", doc_id=doc_id)
        ent_nodes: set[str] = set()
        for c in canon_set:
            if c not in entity_canonical_to_node:
                continue   # filtered above
            e_node = entity_canonical_to_node[c]
            G.add_node(e_node, kind="entity", canonical=c,
                       label=entity_node_to_label[e_node])
            G.add_edge(d_node, e_node, kind="mentions")
            ent_nodes.add(e_node)
            entity_to_docs[e_node].add(doc_id)
        doc_to_entities[doc_id] = ent_nodes
        # entity—entity co-occurrence within this doc
        ent_list = sorted(ent_nodes)
        for i in range(len(ent_list)):
            for j in range(i + 1, len(ent_list)):
                a, b = ent_list[i], ent_list[j]
                if G.has_edge(a, b):
                    G[a][b]["weight"] = G[a][b].get("weight", 1) + 1
                else:
                    G.add_edge(a, b, kind="co_occurs", weight=1)

    return EntityKG(
        graph=G,
        doc_to_entities=doc_to_entities,
        entity_to_docs=dict(entity_to_docs),
        entity_canonical_to_node=entity_canonical_to_node,
        entity_node_to_label=entity_node_to_label,
    )


def extract_query_entities(query: str, nlp=None,
                           labels: set[str] = None) -> list[str]:
    """Run NER on a question, return raw mention strings."""
    if nlp is None:
        import spacy
        nlp = spacy.load("en_core_web_sm")
    if labels is None:
        labels = DEFAULT_ENT_LABELS
    spacy_doc = nlp(query)
    return [ent.text.strip() for ent in spacy_doc.ents if ent.label_ in labels]
