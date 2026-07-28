"""hubmesh-mcp — hubmesh as deterministic MCP operator tools.

The KAG-style planner/operator split, inverted: the *calling agent* is
the solver (decompose the question, read each hop, decide what to look
for next); hubmesh is the operator (retrieve, resolve, traverse). Every
tool here is deterministic numpy/networkx work — the server makes zero
LLM calls, so per-hop retrieval is ~milliseconds and token-free. The
planning intelligence lives entirely in the caller.

Run:  hubmesh-mcp            (stdio transport; requires `hubmesh[mcp]`)

Claude Desktop / Claude Code config:

    {"mcpServers": {"hubmesh": {"command": "hubmesh-mcp"}}}

Corpora persist under ~/.hubmesh/corpora (HUBMESH_CORPORA_ROOT
overrides). Responses are token-lean: `retrieve` returns snippets +
ids; fetch full text with `get_document`.
"""
from __future__ import annotations
import os

from mcp.server.fastmcp import FastMCP

from .corpus import CorpusManager

mcp = FastMCP(
    "hubmesh",
    instructions=(
        "Deterministic graph-retrieval operators over named document "
        "corpora. For multi-hop questions, iterate: retrieve, read the "
        "snippets, then retrieve again passing the bridge entity you "
        "discovered as seed_entities and the doc ids you already "
        "consumed as exclude_docs."
    ),
)

_manager: CorpusManager | None = None


def _mgr() -> CorpusManager:
    global _manager
    if _manager is None:
        root = os.environ.get("HUBMESH_CORPORA_ROOT")
        _manager = CorpusManager(root=root) if root else CorpusManager()
    return _manager


_SNIPPET_CHARS = 280


@mcp.tool()
def list_corpora() -> dict:
    """List indexed corpora with their stats (doc / node / edge counts).
    Start here to see what is queryable."""
    return _mgr().list()


@mcp.tool()
def index_corpus(name: str, documents: list[dict]) -> dict:
    """Index documents into a named corpus: embeds them, builds the
    entity knowledge graph (spaCy NER), and persists everything to
    disk. `documents` is a list of {"id": str, "text": str}. Re-using
    an existing name replaces that corpus. Indexing cost is paid once —
    queries afterwards are ~milliseconds."""
    return _mgr().build(name, documents)


@mcp.tool()
def retrieve(
    corpus: str,
    query: str,
    top_k: int = 8,
    seed_entities: list[str] | None = None,
    exclude_docs: list[str] | None = None,
) -> dict:
    """Graph-aware retrieval: entity seeds + Personalized PageRank over
    the corpus knowledge graph, fused with cosine similarity.

    For MULTI-HOP questions, iterate: retrieve once, read the top
    snippets, then call again passing the bridge entity you discovered
    as `seed_entities` (aims the graph diffusion at it; merged with the
    query's own entities) and the doc ids you already consumed as
    `exclude_docs` (so the next hop explores new ground). Returns
    snippets — use get_document for full text."""
    planner = _mgr().planner(corpus)
    res = planner.retrieve(query=query, top_k=top_k,
                           seed_entities=seed_entities,
                           exclude_docs=exclude_docs)
    return {
        "sources": [{
            "id": s.doc.id,
            "snippet": s.doc.text[:_SNIPPET_CHARS],
            "score": round(s.composite_score, 4),
            "cosine": round(s.similarity, 4),
            "graph_score": round(s.ppr_score, 6),
        } for s in res.sources],
        "seeds_used": res.debug.get("ppr_seeds", []),
        "reasoning_paths": [
            {"nodes": p.node_ids, "score": round(p.score, 4)}
            for p in res.reasoning[:5]
        ],
    }


@mcp.tool()
def resolve_entities(corpus: str, mentions: list[str]) -> list[dict]:
    """Resolve free-text entity names to knowledge-graph nodes (alias
    index + fuzzy fallback). Use before seeding retrieve() to confirm
    an entity exists and see its canonical label."""
    _, kg = _load(corpus)
    out = []
    for m in mentions:
        nodes = kg.query_entity_nodes([m])
        out.append({
            "mention": m,
            "node": nodes[0] if nodes else None,
            "label": kg.entity_node_to_label.get(nodes[0]) if nodes else None,
        })
    return out


@mcp.tool()
def entity_neighbors(corpus: str, entity: str, limit: int = 20) -> dict:
    """Explore around an entity: which documents mention it and which
    entities it connects to (with co-occurrence weight / predicates).
    Useful for choosing the next hop when retrieve() alone is ambiguous."""
    _, kg = _load(corpus)
    nodes = kg.query_entity_nodes([entity])
    if not nodes:
        return {"error": f"no entity matching {entity!r} in {corpus!r}"}
    node = nodes[0]
    ents, docs = [], []
    for nbr in kg.graph.neighbors(node):
        data = kg.graph[node][nbr]
        if nbr.startswith("doc:"):
            docs.append(nbr[4:])
        else:
            ents.append({
                "node": nbr,
                "label": kg.entity_node_to_label.get(nbr, nbr),
                "weight": data.get("weight", 1),
                **({"predicates": data["predicates"]}
                   if "predicates" in data else {}),
            })
    ents.sort(key=lambda e: -e["weight"])
    return {"entity": node,
            "label": kg.entity_node_to_label.get(node, node),
            "connected_entities": ents[:limit],
            "documents": docs[:limit]}


@mcp.tool()
def path_between(corpus: str, entity_a: str, entity_b: str) -> dict:
    """Shortest path between two entities through the entity-document
    graph — shows the document chain that connects them. Empty result
    means no connection exists in this corpus."""
    import networkx as nx
    _, kg = _load(corpus)
    a = kg.query_entity_nodes([entity_a])
    b = kg.query_entity_nodes([entity_b])
    if not a or not b:
        missing = entity_a if not a else entity_b
        return {"error": f"no entity matching {missing!r} in {corpus!r}"}
    try:
        path = nx.shortest_path(kg.graph, a[0], b[0])
    except nx.NetworkXNoPath:
        return {"path": [], "connected": False}
    return {"connected": True, "path": [
        {"node": n,
         "label": (kg.entity_node_to_label.get(n, n)
                   if n.startswith("ent:") else n[4:])}
        for n in path
    ]}


@mcp.tool()
def get_document(corpus: str, doc_id: str) -> dict:
    """Full text of one document (retrieve() returns snippets only)."""
    store, _ = _load(corpus)
    doc = store.get(doc_id)
    return {"id": doc.id, "text": doc.text, "metadata": doc.metadata or {}}


@mcp.tool()
def graph_stats(corpus: str) -> dict:
    """Corpus overview: sizes plus the highest-degree entities — the
    hubs PPR diffusion flows through."""
    _, kg = _load(corpus)
    ent_degrees = sorted(
        ((n, kg.graph.degree(n)) for n in kg.graph.nodes
         if n.startswith("ent:")),
        key=lambda kv: -kv[1])
    return {
        "documents": sum(1 for n in kg.graph.nodes if n.startswith("doc:")),
        "entities": len(ent_degrees),
        "edges": kg.graph.number_of_edges(),
        "top_hub_entities": [
            {"label": kg.entity_node_to_label.get(n, n), "degree": d}
            for n, d in ent_degrees[:10]
        ],
    }


def _load(corpus: str):
    """Store+KG via the planner cache so repeat tool calls stay warm."""
    planner = _mgr().planner(corpus)
    return planner.store, planner.kg


def main():
    import threading
    # Warm up off the serving thread: the first tool call must not pay
    # the ~5-10s model cold start — connector clients (e.g. Perplexity)
    # drop SSE tool calls in exactly that window.
    threading.Thread(target=lambda: _mgr().warmup(), daemon=True).start()
    mcp.run()


if __name__ == "__main__":
    main()
