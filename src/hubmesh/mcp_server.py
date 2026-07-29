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
    seed_entities: list[str] = [],
    exclude_docs: list[str] = [],
) -> dict:
    """Graph-aware retrieval: entity seeds + Personalized PageRank over
    the corpus knowledge graph, fused with cosine similarity.

    For MULTI-HOP questions, iterate: retrieve once, read the top
    snippets, then call again passing the bridge entity you discovered
    as `seed_entities` (aims the graph diffusion at it; merged with the
    query's own entities) and the doc ids you already consumed as
    `exclude_docs` (so the next hop explores new ground). Returns
    snippets — use get_document for full text."""
    # Empty-list defaults (not None) keep the advertised JSON schema free
    # of anyOf/null unions — strict connector executors (observed with
    # Perplexity) refuse to compile union-typed tool parameters and fail
    # with "Error during tool execution" without ever calling the server.
    planner = _mgr().planner(corpus)
    res = planner.retrieve(query=query, top_k=top_k,
                           seed_entities=seed_entities or None,
                           exclude_docs=exclude_docs or None)
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
def path_between(corpus: str, entity_a: str, entity_b: str,
                 k_paths: int = 3) -> dict:
    """Connection paths between two entities through the entity-document
    graph — up to `k_paths` paths, shortest first. INTERPRETATION
    CAUTION: the shortest path can ride an incidental co-mention (two
    names in the same sentence); longer paths with more `via_documents`
    often reflect the more meaningful chain. Report the nature of the
    connection, not just its existence."""
    from itertools import islice
    import networkx as nx
    _, kg = _load(corpus)
    a = kg.query_entity_nodes([entity_a])
    b = kg.query_entity_nodes([entity_b])
    if not a or not b:
        missing = entity_a if not a else entity_b
        return {"error": f"no entity matching {missing!r} in {corpus!r}"}
    k = max(1, min(k_paths, 10))
    try:
        # Keep DISTINCT routes, not detour-variants of one shortcut: a
        # candidate whose intermediates contain all of an already-kept
        # path's intermediates is the same bridge with extra stops.
        found: list = []
        for path in islice(nx.shortest_simple_paths(kg.graph, a[0], b[0]),
                           50):
            mids = set(path[1:-1])
            if any(set(p[1:-1]) and set(p[1:-1]) <= mids for p in found):
                continue
            found.append(path)
            if len(found) >= k:
                break
    except nx.NetworkXNoPath:
        return {"paths": [], "connected": False}
    return {"connected": True, "paths": [{
        "nodes": [{"node": n,
                   "label": (kg.entity_node_to_label.get(n, n)
                             if n.startswith("ent:") else n[4:])}
                  for n in path],
        "hops": len(path) - 1,
        "via_documents": sum(1 for n in path if n.startswith("doc:")),
    } for path in found]}


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
    import argparse
    import threading
    ap = argparse.ArgumentParser(
        prog="hubmesh-mcp",
        description="hubmesh MCP operator server. stdio by default; "
                    "--transport sse serves HTTP+SSE natively (no "
                    "gateway process needed).")
    ap.add_argument("--transport", choices=["stdio", "sse"],
                    default="stdio")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--allow-tunnel", action="store_true",
                    help="accept forwarded Host headers (disables "
                         "DNS-rebinding protection) — required behind "
                         "ngrok-style tunnels, which otherwise get 421 "
                         "Misdirected Request")
    args = ap.parse_args()

    if args.transport == "sse":
        mcp.settings.host = args.host
        mcp.settings.port = args.port
    if args.allow_tunnel:
        from mcp.server.transport_security import TransportSecuritySettings
        mcp.settings.transport_security = TransportSecuritySettings(
            enable_dns_rebinding_protection=False)

    # Warm up off the serving thread: the first tool call must not pay
    # the ~5-10s model cold start — connector clients (e.g. Perplexity)
    # drop SSE tool calls in exactly that window.
    threading.Thread(target=lambda: _mgr().warmup(), daemon=True).start()
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
