"""LLM-based knowledge graph construction (alternative to spaCy NER).

Real HippoRAG and other production GraphRAG systems use an LLM to extract
(subject, predicate, object) triples from each passage, then build a KG
from those triples. The KG is much richer than what spaCy NER produces,
because the LLM understands relations and resolves entities semantically.

This module makes the LLM call provider-agnostic — pass any callable
`(prompt: str) -> str` and it works with OpenAI, Anthropic, Ollama,
together.ai, vLLM, or anything else. We don't depend on any specific
SDK.

Usage::

    from hubmesh.kg_llm import build_entity_kg_llm
    import openai
    client = openai.OpenAI()

    def llm(prompt: str) -> str:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        return r.choices[0].message.content

    kg = build_entity_kg_llm(documents, llm=llm, cache_path="kg_cache.json")

The cache is keyed by document content hash; re-runs over an unchanged
corpus are free.
"""
from __future__ import annotations
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Callable
import networkx as nx

from .kg import EntityKG, canonicalize, _entity_node, _doc_node


DEFAULT_TRIPLE_PROMPT = """\
You are an information-extraction assistant. Extract factual triples from
the passage below as a JSON array. Each triple must be:

  [SUBJECT, PREDICATE, OBJECT]

Where SUBJECT and OBJECT are concrete named entities (people, places,
organisations, works, events) — not pronouns or generic noun phrases.
PREDICATE is a short verb phrase capturing the relation.

Rules:
  - Only extract relations *explicitly stated* in the passage.
  - Use the most specific surface form for each entity (e.g.
    "Bill Clinton" not "Clinton" if both appear).
  - Output ONLY a JSON object: {{"triples": [[s, p, o], ...]}}.
  - Empty array if no clean triples.

Passage:
\"\"\"
{passage}
\"\"\"
"""


def _hash(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=12).hexdigest()


def _parse_triples(raw: str) -> list[tuple[str, str, str]]:
    """Lenient JSON parsing — handle stray prose around the JSON."""
    raw = raw.strip()
    # Try direct parse first
    candidates = [raw]
    # Common: code-fenced JSON
    if "```" in raw:
        between = raw.split("```")
        for block in between:
            block = block.strip()
            if block.startswith("json"):
                block = block[4:].strip()
            if block.startswith("{") or block.startswith("["):
                candidates.append(block)
    # Try parsing each candidate
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        # Accept either {"triples": [[s,p,o],...]} or just [[s,p,o],...]
        if isinstance(obj, dict):
            obj = obj.get("triples", [])
        if isinstance(obj, list):
            out = []
            for t in obj:
                if (isinstance(t, list) and len(t) == 3
                        and all(isinstance(x, str) and x.strip() for x in t)):
                    out.append((t[0].strip(), t[1].strip(), t[2].strip()))
            return out
    return []


def build_entity_kg_llm(
    documents: list,
    llm: Callable[[str], str],
    prompt_template: str | None = None,
    cache_path: str | Path | None = None,
    max_workers: int = 1,
    progress: bool = False,
) -> EntityKG:
    """Build the KG by extracting (subject, predicate, object) triples
    from each document via an LLM.

    Returns an `EntityKG` with the same shape as `kg.build_entity_kg`,
    so the rest of the pipeline (Planner, retrieval, paths) works
    unchanged.

    `cache_path` (optional) JSON file caching extracted triples by
    passage hash — strongly recommended for large corpora.
    """
    template = prompt_template or DEFAULT_TRIPLE_PROMPT
    cache: dict[str, list] = {}
    cache_p = Path(cache_path) if cache_path else None
    if cache_p and cache_p.exists():
        try:
            cache = json.loads(cache_p.read_text())
        except json.JSONDecodeError:
            cache = {}

    # Step 1: extract triples per doc (with caching)
    triples_per_doc: dict[str, list[tuple[str, str, str]]] = {}
    iterator = enumerate(documents)
    if progress:
        try:
            from tqdm import tqdm
            iterator = enumerate(tqdm(documents, desc="LLM KG extract"))
        except ImportError:
            pass

    for _, doc in iterator:
        if not doc.text.strip():
            triples_per_doc[doc.id] = []
            continue
        key = _hash(doc.text)
        if key in cache:
            triples_per_doc[doc.id] = [tuple(t) for t in cache[key]]
            continue
        prompt = template.format(passage=doc.text)
        try:
            raw = llm(prompt)
        except Exception:
            triples_per_doc[doc.id] = []
            continue
        triples = _parse_triples(raw)
        triples_per_doc[doc.id] = triples
        cache[key] = [list(t) for t in triples]

    # Persist cache
    if cache_p is not None:
        cache_p.parent.mkdir(parents=True, exist_ok=True)
        cache_p.write_text(json.dumps(cache, indent=2))

    # Step 2: canonicalise entity mentions across all triples
    all_mentions: list[str] = []
    for triples in triples_per_doc.values():
        for s, _, o in triples:
            all_mentions.append(s)
            all_mentions.append(o)

    canonical_to_displays: dict[str, set[str]] = defaultdict(set)
    raw_to_canon: dict[str, str] = {}
    for m in set(all_mentions):
        c = canonicalize(m)
        if c:
            raw_to_canon[m] = c
            canonical_to_displays[c].add(m)

    entity_canonical_to_node: dict[str, str] = {
        c: _entity_node(c) for c in canonical_to_displays
    }
    entity_node_to_label: dict[str, str] = {
        nid: max(canonical_to_displays[c], key=len)
        for c, nid in entity_canonical_to_node.items()
    }

    # Step 3: build the bipartite graph + entity-entity edges from
    # explicit predicates (stored on edge as `predicate`)
    G = nx.Graph()
    doc_to_entities: dict[str, set[str]] = {}
    entity_to_docs: dict[str, set[str]] = defaultdict(set)
    for doc_id, triples in triples_per_doc.items():
        d_node = _doc_node(doc_id)
        G.add_node(d_node, kind="doc", doc_id=doc_id)
        ent_nodes: set[str] = set()
        for s, p, o in triples:
            sc = raw_to_canon.get(s)
            oc = raw_to_canon.get(o)
            if not sc or not oc:
                continue
            s_node = entity_canonical_to_node[sc]
            o_node = entity_canonical_to_node[oc]
            G.add_node(s_node, kind="entity", canonical=sc,
                       label=entity_node_to_label[s_node])
            G.add_node(o_node, kind="entity", canonical=oc,
                       label=entity_node_to_label[o_node])
            G.add_edge(d_node, s_node, kind="mentions")
            G.add_edge(d_node, o_node, kind="mentions")
            ent_nodes.add(s_node)
            ent_nodes.add(o_node)
            entity_to_docs[s_node].add(doc_id)
            entity_to_docs[o_node].add(doc_id)
            # Entity-entity edge: weighted by occurrence; predicate stored
            if G.has_edge(s_node, o_node):
                G[s_node][o_node]["weight"] = G[s_node][o_node].get("weight", 1) + 1
                preds = G[s_node][o_node].setdefault("predicates", [])
                if p not in preds:
                    preds.append(p)
            else:
                G.add_edge(s_node, o_node, kind="relates",
                           weight=1, predicates=[p])
        doc_to_entities[doc_id] = ent_nodes

    return EntityKG(
        graph=G,
        doc_to_entities=doc_to_entities,
        entity_to_docs=dict(entity_to_docs),
        entity_canonical_to_node=entity_canonical_to_node,
        entity_node_to_label=entity_node_to_label,
    )
