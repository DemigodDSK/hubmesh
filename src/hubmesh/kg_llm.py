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
import warnings
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


def _parse_triples(raw: str) -> list[tuple[str, str, str]] | None:
    """Lenient JSON parsing — handle stray prose around the JSON.

    Returns a (possibly empty) list when the reply parses to a triple
    list, and **None when nothing parseable was found**: a valid empty
    extraction and a malformed reply are different outcomes, and only
    the former may be cached (external review, 2026-09-14: `NOT JSON`
    was cached as "no relations" and never retried)."""
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
        # Accept either {"triples": [[s,p,o],...]} or just [[s,p,o],...].
        # A JSON object WITHOUT the key is not an extraction result.
        if isinstance(obj, dict):
            if "triples" not in obj:
                continue
            obj = obj["triples"]
        if isinstance(obj, list):
            out = []
            for t in obj:
                if (isinstance(t, list) and len(t) == 3
                        and all(isinstance(x, str) and x.strip() for x in t)):
                    out.append((t[0].strip(), t[1].strip(), t[2].strip()))
            return out
    return None


def build_entity_kg_llm(
    documents: list,
    llm: Callable[[str], str],
    prompt_template: str | None = None,
    cache_path: str | Path | None = None,
    max_workers: int = 1,
    progress: bool = False,
    linker=None,
    llm_identity: str | None = None,
) -> EntityKG:
    """Build the KG by extracting (subject, predicate, object) triples
    from each document via an LLM.

    Returns an `EntityKG` with the same shape as `kg.build_entity_kg`,
    so the rest of the pipeline (Planner, retrieval, paths) works
    unchanged.

    `cache_path` (optional) JSON file caching extracted triples by
    passage hash — strongly recommended for large corpora.

    `linker` (optional) — same contract as `build_entity_kg`'s linker=
    argument: an `entity_linker.Linker` that canonicalises mentions
    across the whole corpus. Without it, triple arguments get only the
    bare `canonicalize()` treatment, so "USA" and "United States" stay
    separate entities even when they co-refer.
    """
    template = prompt_template or DEFAULT_TRIPLE_PROMPT
    # Cache entries are namespaced by (llm identity, prompt-template hash,
    # passage hash): a cache filled by one model/prompt can never serve
    # hits to another, and one file can hold several namespaces without
    # ever discarding expensive extractions. Legacy flat caches (passage
    # hash only, no namespace) are honoured only when no identity is
    # declared and the default template is in use, with a warning.
    template_hash = _hash(template)
    namespace = f"{llm_identity or ''}|{template_hash}|"
    entries: dict[str, list] = {}
    legacy_entries: dict[str, list] = {}
    cache_p = Path(cache_path) if cache_path else None
    if cache_p and cache_p.exists():
        try:
            loaded = json.loads(cache_p.read_text())
        except json.JSONDecodeError:
            loaded = {}
        if isinstance(loaded, dict) and loaded.get("_format") == 2:
            entries = dict(loaded.get("entries", {}))
        elif isinstance(loaded, dict):
            if llm_identity is None and prompt_template is None:
                legacy_entries = loaded
                warnings.warn(
                    f"{cache_p}: legacy triple cache without model/prompt "
                    "identity — reused because no llm_identity is declared; "
                    "pass llm_identity= so future caches are verifiable",
                    stacklevel=2)
            else:
                warnings.warn(
                    f"{cache_p}: legacy triple cache ignored (it carries no "
                    "identity and llm_identity/prompt_template are set); "
                    "entries will be re-extracted under a namespaced key",
                    stacklevel=2)

    def key_for(text: str) -> str:
        return namespace + _hash(text)

    # Step 1: extract triples per doc (cached; failures COUNTED, not
    # silently swallowed as "no triples")
    stats = {"docs": len(documents), "empty_docs": 0, "cache_hits": 0,
             "llm_calls": 0, "failed_calls": 0, "unparseable": 0}
    triples_per_doc: dict[str, list[tuple[str, str, str]]] = {}
    pending: list = []
    for doc in documents:
        if not doc.text.strip():
            triples_per_doc[doc.id] = []
            stats["empty_docs"] += 1
            continue
        k2 = key_for(doc.text)
        k1 = _hash(doc.text)
        if k2 in entries:
            triples_per_doc[doc.id] = [tuple(t) for t in entries[k2]]
            stats["cache_hits"] += 1
        elif k1 in legacy_entries:
            triples_per_doc[doc.id] = [tuple(t) for t in legacy_entries[k1]]
            entries[k2] = legacy_entries[k1]      # migrate into namespace
            stats["cache_hits"] += 1
        else:
            pending.append(doc)

    def extract(doc):
        prompt = template.format(passage=doc.text)
        try:
            raw = llm(prompt)
        except Exception as e:      # noqa: BLE001 — provider errors vary
            return doc, None, e
        return doc, _parse_triples(raw), None

    results = []
    if pending:
        if max_workers > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                it = pool.map(extract, pending)
                if progress:
                    try:
                        from tqdm import tqdm
                        it = tqdm(it, total=len(pending), desc="LLM KG extract")
                    except ImportError:
                        pass
                results = list(it)
        else:
            it = pending
            if progress:
                try:
                    from tqdm import tqdm
                    it = tqdm(pending, desc="LLM KG extract")
                except ImportError:
                    pass
            results = [extract(d) for d in it]

    for doc, triples, err in results:      # pool.map preserves doc order
        stats["llm_calls"] += 1
        if err is not None:
            stats["failed_calls"] += 1
            triples_per_doc[doc.id] = []
            continue                        # NOT cached: retry next build
        if triples is None:
            stats["unparseable"] += 1       # malformed reply, NOT a result
            triples_per_doc[doc.id] = []
            continue                        # NOT cached: retry next build
        # A parsed reply — including a legitimately empty triple list —
        # is a result and is cached.
        triples_per_doc[doc.id] = triples
        entries[key_for(doc.text)] = [list(t) for t in triples]

    if stats["failed_calls"] or stats["unparseable"]:
        warnings.warn(
            f"LLM KG extraction: {stats['failed_calls']}/{stats['llm_calls']} "
            f"calls failed, {stats['unparseable']}/{stats['llm_calls']} "
            "replies unparseable — those documents have NO triples in this "
            "KG and were NOT cached (rerun to retry). Inspect "
            "kg.extraction_stats.",
            stacklevel=2)

    # Persist cache (format 2, namespaced)
    if cache_p is not None:
        cache_p.parent.mkdir(parents=True, exist_ok=True)
        cache_p.write_text(json.dumps(
            {"_format": 2, "entries": entries,
             "last_writer": {"llm_identity": llm_identity,
                             "template_hash": template_hash}},
            indent=2))

    # Step 2: canonicalise entity mentions across all triples
    all_mentions: list[str] = []
    for triples in triples_per_doc.values():
        for s, _, o in triples:
            all_mentions.append(s)
            all_mentions.append(o)

    canonical_to_displays: dict[str, set[str]] = defaultdict(set)
    raw_to_canon: dict[str, str] = {}
    if linker is not None:
        # Cross-document linking — same path as build_entity_kg's linker=
        # argument. Closes the gap where LLM-extracted mentions bypassed
        # the Linker protocol and got weaker dedup than the spaCy path.
        link_map = linker.link(set(all_mentions))
        for m in set(all_mentions):
            c = link_map.get(m, canonicalize(m))
            if c:
                raw_to_canon[m] = c
                canonical_to_displays[c].add(m)
    else:
        for m in set(all_mentions):
            c = canonicalize(m)
            if c:
                raw_to_canon[m] = c
                canonical_to_displays[c].add(m)

    entity_canonical_to_node: dict[str, str] = {
        c: _entity_node(c) for c in canonical_to_displays
    }
    entity_node_to_label: dict[str, str] = {
        nid: max(sorted(canonical_to_displays[c]), key=len)   # stable ties
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

    # Alias index — mirrors build_entity_kg: every surface form seen in
    # the triples, canonicalised, → its graph node (graph-backed only).
    # Two passes so a display alias can never shadow a different entity's
    # exact canonical name; sorted iteration for cross-process determinism.
    alias_to_node: dict[str, str] = {}
    for c, nid in entity_canonical_to_node.items():
        if nid in G:
            alias_to_node[c] = nid
    for c, displays in sorted(canonical_to_displays.items()):
        nid = entity_canonical_to_node.get(c)
        if nid is None or nid not in G:
            continue
        for m in sorted(displays):
            mc = canonicalize(m)
            if mc:
                alias_to_node.setdefault(mc, nid)

    kg = EntityKG(
        graph=G,
        doc_to_entities=doc_to_entities,
        entity_to_docs=dict(entity_to_docs),
        entity_canonical_to_node=entity_canonical_to_node,
        entity_node_to_label=entity_node_to_label,
        alias_to_node=alias_to_node,
    )
    kg.extraction_stats = stats   # explicit failure policy: visible, not silent
    return kg
